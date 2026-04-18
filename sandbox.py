"""Execute untrusted LLM-generated Python in an isolated subprocess.

Each call forks a fresh Python interpreter that:
  * compiles the code with RestrictedPython (blocks open(), unsafe imports,
    attribute access to dunders, mutation of modules/classes, etc.);
  * runs with CPU / address-space / file-size rlimits so runaways can't
    starve the host;
  * runs in an empty tempdir (cwd) — any stray relative file writes land in
    throwaway space, and RLIMIT_FSIZE=0 blocks them outright anyway;
  * is killed by wall-clock timeout from the parent if it overruns.

The prompt/question is passed in as the PROMPT global variable (a plain
string); there is no prompt file. stdout is captured as a string and
returned. The function NEVER raises — failures are reported in the `error`
field of the returned dict.
"""

from __future__ import annotations

import asyncio
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

RLIMIT_CPU_SECONDS = 1
RLIMIT_MEM_BYTES = 512 * 1024 * 1024
RLIMIT_FILE_BYTES = 0  # disk writes forbidden; pipes to parent are exempt

# Child-side worker. Reads {"code", "prompt", "record"} JSON from stdin,
# runs `code` under RestrictedPython with PROMPT=prompt and RECORD=record,
# prints a one-line JSON result.
_WORKER = r"""
import io, json, sys, traceback

def _emit(d):
    sys.__stdout__.write(json.dumps(d) + "\n")
    sys.__stdout__.flush()

try:
    from RestrictedPython import (
        compile_restricted, safe_builtins, limited_builtins, utility_builtins,
    )
    from RestrictedPython.Guards import (
        guarded_iter_unpack_sequence, guarded_unpack_sequence,
        safer_getattr, full_write_guard,
    )
    from RestrictedPython.PrintCollector import PrintCollector
except Exception as e:
    _emit({"stdout": "", "stderr": "", "error": "RestrictedPython missing: %s" % e})
    sys.exit(0)

SAFE_MODULES = {
    "re", "math", "itertools", "collections", "string", "json",
    "functools", "operator", "heapq", "bisect", "random",
    "fractions", "decimal", "statistics",
}

# Compound-assignment handler: RestrictedPython rewrites `x op= y` into
# `x = _inplacevar_("op=", x, y)`; without this, any `+=`/`^=`/... raises.
_INPLACE_OPS = {
    "+=":  lambda a, b: a + b,   "-=":  lambda a, b: a - b,
    "*=":  lambda a, b: a * b,   "/=":  lambda a, b: a / b,
    "//=": lambda a, b: a // b,  "%=":  lambda a, b: a % b,
    "**=": lambda a, b: a ** b,  "@=":  lambda a, b: a.__matmul__(b),
    "<<=": lambda a, b: a << b,  ">>=": lambda a, b: a >> b,
    "&=":  lambda a, b: a & b,   "^=":  lambda a, b: a ^ b,
    "|=":  lambda a, b: a | b,
}
def _inplacevar_(op, x, y):
    return _INPLACE_OPS[op](x, y)

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root in SAFE_MODULES:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError("module %r not available in sandbox" % name)

# RestrictedPython's safe_/limited_ dicts miss a handful of staples; whitelist
# them explicitly so normal Python code runs.
EXTRA_BUILTINS = {
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "sorted": sorted, "reversed": reversed,
    "sum": sum, "min": min, "max": max, "all": all, "any": any,
    "len": len, "range": range, "iter": iter, "next": next,
    "abs": abs, "round": round, "divmod": divmod, "pow": pow,
    "int": int, "str": str, "float": float, "bool": bool, "bytes": bytes,
    "bin": bin, "hex": hex, "oct": oct, "format": format,
    "list": list, "tuple": tuple, "set": set, "dict": dict, "frozenset": frozenset,
    "type": type, "isinstance": isinstance, "issubclass": issubclass,
    "hash": hash, "ord": ord, "chr": chr, "repr": repr, "slice": slice,
    "print": print,
}

try:
    payload = json.loads(sys.stdin.read())
    code = payload["code"]
    prompt = payload.get("prompt", "")
    record = payload.get("record")
except Exception:
    _emit({"stdout": "", "stderr": "", "error": "bad input payload"})
    sys.exit(0)

# Pre-parse each sample so models can use the integer / bitstring directly
# without re-implementing hex parsing. Keeps originals intact.
if record and isinstance(record.get("samples"), list):
    nbits = int(record.get("n", 0)) + int(record.get("p", 0))
    new_samples = []
    for s in record["samples"]:
        s2 = dict(s)
        xv = s.get("x")
        try:
            s2["x_int"] = int(xv, 16) if isinstance(xv, str) else int(xv)
            if nbits > 0:
                s2["bits"] = format(s2["x_int"], "0%db" % nbits)  # x_0 is MSB
        except Exception:
            pass
        new_samples.append(s2)
    record = dict(record)
    record["samples"] = new_samples

out = {"stdout": "", "stderr": "", "error": None}

try:
    byte_code = compile_restricted(code, filename="<llm>", mode="exec")
except SyntaxError as e:
    out["error"] = "compile: %s" % e
    _emit(out); sys.exit(0)

builtins_d = dict(safe_builtins)
builtins_d.update(limited_builtins)
builtins_d.update(utility_builtins)
builtins_d.update(EXTRA_BUILTINS)
builtins_d["__import__"] = _safe_import

glb = {
    "__builtins__": builtins_d,
    "__name__": "__restricted__",
    "__metaclass__": type,
    "_getattr_": safer_getattr,
    "_getitem_": lambda obj, k: obj[k],
    "_getiter_": iter,
    "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
    "_unpack_sequence_": guarded_unpack_sequence,
    "_inplacevar_": _inplacevar_,
    "_write_": full_write_guard,
    "_print_": PrintCollector,
    "PROMPT": prompt,
    "RECORD": record,
}

def _collect_print():
    pc = glb.get("_print")
    if pc is None:
        return ""
    try:
        return pc() if callable(pc) else str(pc)
    except Exception:
        try:
            return str(pc)
        except Exception:
            return ""

try:
    exec(byte_code, glb)
    out["stdout"] = _collect_print()
except SystemExit:
    out["stdout"] = _collect_print()
except BaseException:
    out["stdout"] = _collect_print()
    out["error"] = traceback.format_exc(limit=5)

_emit(out)
"""


def _preexec():
    # Hard caps; wall-clock timeout in the parent covers sub-second cases.
    resource.setrlimit(resource.RLIMIT_CPU, (RLIMIT_CPU_SECONDS, RLIMIT_CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (RLIMIT_MEM_BYTES, RLIMIT_MEM_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (RLIMIT_FILE_BYTES, RLIMIT_FILE_BYTES))
    try:
        os.setsid()  # new session — lets the parent reap with SIGKILL cleanly
    except OSError:
        pass


def run_sandboxed(code: str, prompt: str, timeout_ms: int = 100,
                  record: Optional[dict] = None) -> dict:
    """Run `code` with the question string exposed as the global `PROMPT` and
    the parsed experiment dict exposed as `RECORD` (or None if not provided).

    Never raises. Always returns a dict with these keys:
        stdout       — str: whatever the code printed (via RestrictedPython)
        stderr       — str: subprocess stderr (worker crashes / rlimit hits)
        error        — Optional[str]: non-None iff something went wrong
        returncode   — Optional[int]: subprocess exit code
        elapsed_ms   — float
    """
    timeout_s = max(timeout_ms / 1000.0, 0.02)
    start = time.time()
    result = {"stdout": "", "stderr": "", "error": None,
              "returncode": None, "elapsed_ms": 0.0}
    payload = json.dumps({"code": code, "prompt": prompt, "record": record})
    try:
        with tempfile.TemporaryDirectory(prefix="llm_sandbox_") as d:
            proc = subprocess.run(
                [sys.executable, "-c", _WORKER],
                input=payload,
                capture_output=True, text=True,
                timeout=timeout_s,
                cwd=d,
                preexec_fn=_preexec,
            )
            result["returncode"] = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            # Worker ends with one JSON line. Parse the last non-empty line.
            last = ""
            for line in stdout.splitlines()[::-1]:
                if line.strip():
                    last = line
                    break
            if last:
                try:
                    inner = json.loads(last)
                    result["stdout"] = inner.get("stdout", "") or ""
                    result["error"] = inner.get("error")
                    result["stderr"] = stderr
                except json.JSONDecodeError:
                    result["stdout"] = stdout
                    result["stderr"] = stderr
                    if proc.returncode != 0:
                        result["error"] = f"worker exited {proc.returncode}"
            else:
                result["stdout"] = stdout
                result["stderr"] = stderr
                if proc.returncode != 0:
                    result["error"] = f"worker exited {proc.returncode}"
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        err = e.stderr or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        result["stdout"] = out
        result["stderr"] = err
        result["error"] = f"timeout ({timeout_ms}ms)"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    result["elapsed_ms"] = (time.time() - start) * 1000
    return result


def default_workers() -> int:
    """Pick a reasonable pool size from SLURM hints, falling back to cpu_count."""
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(var)
        if v and v.isdigit() and int(v) > 0:
            return int(v)
    return os.cpu_count() or 8


class SandboxPool:
    """Thread-pool dispatcher so many sandbox subprocesses run concurrently.

    subprocess.run releases the GIL while the child executes, so N threads
    give N real parallel sandboxes — no IPC or pickling overhead.
    """

    def __init__(self, workers: Optional[int] = None):
        self.workers = workers or default_workers()
        self._pool = ThreadPoolExecutor(max_workers=self.workers,
                                        thread_name_prefix="sandbox")

    async def submit(self, code: str, prompt: str, timeout_ms: int = 100,
                     record: Optional[dict] = None) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._pool, run_sandboxed, code, prompt, timeout_ms, record,
        )

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
