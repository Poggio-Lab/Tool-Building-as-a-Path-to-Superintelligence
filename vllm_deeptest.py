"""vLLM Local Inference for Monomial Prediction (Client-Server Mode).

Starts a local vLLM server and runs inference using the OpenAI API.
Saves responses incrementally to JSONL files.
"""

import argparse, asyncio, fcntl, json, os, random, re, socket, subprocess, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from tqdm.asyncio import tqdm

REPO_ROOT = Path(__file__).resolve().parent
STORAGE_DIR = REPO_ROOT / "storage"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompts import (
    build_prompt, build_prompt_tools, estimate_tokens, estimate_output_tokens,
    extract_answer_from_response, extract_code_from_response, format_sandbox_feedback,
    strip_thinking, validate_prediction,
)
from sandbox import SandboxPool, default_workers

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

MODELS = [
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
]
LLM_DEFAULT_DATASET = "experiments_llm_adversarial.json"


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def visible_gpus() -> List[str]:
    """Return the list of physical GPU IDs this process can use.

    Prefers CUDA_VISIBLE_DEVICES (SLURM-set); falls back to nvidia-smi; else [0].
    """
    cv = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cv:
        return [x.strip() for x in cv.split(",") if x.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=5, check=True)
        n = sum(1 for ln in out.stdout.splitlines() if ln.startswith("GPU "))
        return [str(i) for i in range(max(n, 1))]
    except Exception:
        return ["0"]


def wait_for_server(url: str, timeout: int = 600) -> bool:
    import requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(f"{url}/health").status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def abs_path(s) -> Path:
    p = Path(s)
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_input_path(input_arg: Optional[str]) -> Path:
    """Find the dataset JSON. Checks abs path, repo root, then storage/."""
    p = Path(input_arg or LLM_DEFAULT_DATASET)
    if p.is_absolute() and p.exists():
        return p
    for candidate in (REPO_ROOT / p, STORAGE_DIR / p.name):
        if candidate.exists():
            return candidate
    return REPO_ROOT / p  # fallback for the error-path branch


def describe_input_search(input_arg: Optional[str]) -> str:
    """Diagnostic text listing every path tried for the input JSON."""
    p = Path(input_arg or LLM_DEFAULT_DATASET)
    tried = [p, REPO_ROOT / p, STORAGE_DIR / p.name]
    lines = [f"  tried: {t}  exists={t.exists()}" for t in tried]
    if STORAGE_DIR.exists():
        lines.append(f"  storage/ contents: "
                     f"{sorted(s.name for s in STORAGE_DIR.iterdir())[:10]}")
    else:
        lines.append(f"  storage/ does not exist ({STORAGE_DIR})")
    return "\n".join(lines)


def append_jsonl(output_file: Path, record: dict):
    with open(output_file, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(record) + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_completed(paths: List[Path]) -> set:
    done = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                try:
                    idx = json.loads(line).get("record_idx")
                    if idx is not None:
                        done.add(idx)
                except json.JSONDecodeError:
                    pass
    return done


def resolve_resume_paths(arg: Optional[str]) -> List[Path]:
    return [abs_path(s.strip()) for s in (arg or "").split(",") if s.strip()]


def launch_replica(args, gpu_group: List[str], idx: int, log_base: Optional[str]) -> dict:
    """Spawn one vLLM server bound to `gpu_group` on a free port.

    Returns {proc, url, log, gpus}. Child inherits env with CUDA_VISIBLE_DEVICES
    restricted to this group's physical IDs — isolates it from sibling replicas.
    """
    port = find_free_port()
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model, "--host", host, "--port", str(port),
        "--tensor-parallel-size", str(len(gpu_group)),
        "--max-model-len", str(args.max_model_len),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--dtype", args.dtype, "--no-enable-log-requests",
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_group)

    log_handle = None
    stdout = stderr = subprocess.DEVNULL
    if log_base:
        log_path = abs_path(f"{log_base}.{idx}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "a", buffering=1)
        stdout = stderr = log_handle

    proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
    print(f"[replica {idx}] GPUs={gpu_group} port={port} pid={proc.pid}")
    return {"proc": proc, "url": url, "log": log_handle, "gpus": gpu_group, "idx": idx}


def shard_tasks(tasks: list, world_size: int, rank: int) -> list:
    if world_size <= 1:
        return tasks
    base, rem = divmod(len(tasks), world_size)
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return tasks[start:end]


@dataclass
class Task:
    record_idx: int
    record: dict
    prompt: str         # Sent to the LLM.
    question: str = ""  # Exposed to tool-mode code as the `PROMPT` global.


async def process_task(client, task: Task, model: str, max_tokens: int, temperature: float,
                       stop_sequences: List[str], pbar, output_file: Path, file_lock: asyncio.Lock,
                       tools_allowed: bool, tool_revise: bool, max_revisions: int,
                       sandbox_pool: Optional[SandboxPool], sandbox_timeout_ms: int):
    """One dataset record → one or more LLM turns.

    Normal:            one turn, parse boxed answer from the response.
    tools, no revise:  one turn, extract code, run sandbox, parse stdout.
    tools + revise:    loop — after each sandbox run, feed stdout/stderr back
                       as a user message; stop when the model replies without
                       a code block (final answer) or we hit max_revisions.
    """
    start = time.time()
    messages = [{"role": "user", "content": task.prompt}]
    codes: List[str] = []
    sandboxes: List[dict] = []
    response_text = None
    input_tokens = output_tokens = 0
    error = None
    reasoning = ""

    while True:
        try:
            resp = await client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature, stop=stop_sequences,
                timeout=3600.0,
            )
        except Exception as e:
            error = str(e)
            break
        response_text = resp.choices[0].message.content or ""
        if resp.usage:
            input_tokens += resp.usage.prompt_tokens
            output_tokens += resp.usage.completion_tokens
        # Capture the first turn's thinking (mirrors prior behavior).
        if not reasoning:
            m = re.search(r"<think(?:ing)?>(.*?)</think(?:ing)?>", response_text,
                          re.DOTALL | re.IGNORECASE)
            if m:
                reasoning = m.group(1).strip()

        code = extract_code_from_response(response_text) if tools_allowed else None

        if code and sandbox_pool is not None:
            # Keep history compact: strip thinking from the assistant turn.
            messages.append({"role": "assistant", "content": strip_thinking(response_text)})
            sb = await sandbox_pool.submit(code, task.question, sandbox_timeout_ms,
                                           record=task.record)
            codes.append(code)
            sandboxes.append(sb)

            if not tool_revise or len(codes) > max_revisions:
                break
            messages.append({"role": "user",
                             "content": format_sandbox_feedback(sb, revise=True)})
            continue

        # No code block (or tools off) → this response holds the final answer.
        break

    parsed_answer, validation = None, None
    if response_text is not None:
        # Priority:
        #   1) If the model's final turn has no code block, it IS the answer.
        #   2) Else parse the last sandbox stdout.
        #   3) Fallback: parse the response text (salvages cases where the
        #      model pasted `\boxed{...}` *inside* a code block, which
        #      RestrictedPython then failed to compile as Python).
        final_has_code = extract_code_from_response(response_text) is not None
        if tools_allowed and not final_has_code:
            parsed_answer = extract_answer_from_response(response_text)
        elif tools_allowed and sandboxes:
            parsed_answer = extract_answer_from_response(sandboxes[-1].get("stdout") or "")
            if parsed_answer is None:
                parsed_answer = extract_answer_from_response(response_text)
        elif not tools_allowed:
            parsed_answer = extract_answer_from_response(response_text)
        if parsed_answer is not None:
            validation = validate_prediction(parsed_answer, task.record)

    result = {
        "record_idx": task.record_idx,
        "p": task.record["p"], "g": task.record["g"],
        "d_max": task.record["d_max"], "target": task.record["target"],
        "prompt": task.prompt, "response": response_text,
        "reasoning": reasoning, "parsed_answer": parsed_answer,
        "is_correct": validation["is_correct"] if validation else False,
        "is_valid_format": validation["is_valid_format"] if validation else False,
        "model": model, "tools_allowed": tools_allowed, "tool_revise": tool_revise,
        "codes": codes, "sandboxes": sandboxes, "num_turns": len(codes) + 1,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "elapsed_seconds": time.time() - start,
        "success": error is None and response_text is not None, "error": error,
    }
    async with file_lock:
        append_jsonl(output_file, result)
    pbar.update(1)
    return result


async def run_client(base_urls: List[str], model: str, tasks: List[Task], output_file: Path,
                     max_tokens: int, temperature: float, stop_sequences: List[str],
                     clear_output: bool, tools_allowed: bool, tool_revise: bool,
                     max_revisions: int, sandbox_workers: int, sandbox_timeout_ms: int,
                     per_replica_concurrency: int = 64):
    """Distribute tasks round-robin across N vLLM replicas.

    Each replica gets its own AsyncOpenAI client and its own semaphore, so a
    slow task on one GPU doesn't starve the others. Global concurrency =
    N * per_replica_concurrency.
    """
    if clear_output:
        open(output_file, "w").close()
    pool = SandboxPool(workers=sandbox_workers) if tools_allowed else None
    async with AsyncExitStack() as stack:
        clients = [
            await stack.enter_async_context(
                AsyncOpenAI(base_url=f"{u}/v1", api_key="EMPTY")
            ) for u in base_urls
        ]
        sems = [asyncio.Semaphore(per_replica_concurrency) for _ in clients]
        file_lock = asyncio.Lock()
        try:
            with tqdm(total=len(tasks), desc="Processing") as pbar:
                async def run(i, t):
                    ci = i % len(clients)
                    async with sems[ci]:
                        return await process_task(clients[ci], t, model, max_tokens, temperature,
                                                  stop_sequences, pbar, output_file, file_lock,
                                                  tools_allowed, tool_revise, max_revisions,
                                                  pool, sandbox_timeout_ms)
                return await asyncio.gather(*[run(i, t) for i, t in enumerate(tasks)])
        finally:
            if pool is not None:
                pool.close()


def main():
    p = argparse.ArgumentParser(description="Run LLM inference locally using vLLM Server + OpenAI Client")
    p.add_argument("--input"); p.add_argument("--model", default=MODELS[0])
    p.add_argument("--output-dir", default="all_results"); p.add_argument("--output-file")
    p.add_argument("--max-records", type=int); p.add_argument("--max-g", type=int)
    p.add_argument("--g-values", type=str, default=None,
                   help="Comma-separated whitelist of g values to keep (e.g. '31,63').")
    p.add_argument("--max-dp-per-g", type=int)
    p.add_argument("--max-tokens", type=int); p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--data-parallel-size", type=int)
    p.add_argument("--dp-world-size", type=int, default=1)
    p.add_argument("--dp-rank", type=int, default=0)
    p.add_argument("--max-model-len", type=int, default=262144)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--yes", "-y", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-from", help="Comma-separated JSONL paths to resume from")
    p.add_argument("--server-log")
    p.add_argument("--stop", default="<|im_end|>,<|endoftext|>,]}")
    p.add_argument("--shuffle", action="store_true")
    tools_grp = p.add_mutually_exclusive_group()
    tools_grp.add_argument("--tools", dest="tools", action="store_true", help="Allow tool use")
    tools_grp.add_argument("--no-tools", dest="tools", action="store_false", help="Disallow tool use")
    p.set_defaults(tools=False)
    p.add_argument("--tool-revise", action="store_true",
                   help="Feed sandbox output back; LLM iterates until it replies without code or hits --max-revisions")
    p.add_argument("--max-revisions", type=int, default=10,
                   help="Safety cap on revision rounds (ignored without --tool-revise)")
    p.add_argument("--sandbox-workers", type=int, default=None,
                   help="Concurrent sandbox subprocesses (default: SLURM_CPUS_PER_TASK or cpu_count)")
    p.add_argument("--sandbox-timeout-ms", type=int, default=500,
                   help="Wall-clock cap on each sandbox execution (default 500 ms)")
    args = p.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)

    if AsyncOpenAI is None:
        print("Error: 'openai' package is required. Install with: uv pip install openai")
        return 1
    if args.dp_world_size < 1 or not (0 <= args.dp_rank < args.dp_world_size):
        print("ERROR: invalid --dp-world-size / --dp-rank")
        return 1
    if args.tool_revise and not args.tools:
        print("ERROR: --tool-revise requires --tools")
        return 1

    if not args.dry_run:
        delay = random.uniform(0, 10)
        print(f"Waiting {delay:.2f}s to stagger startup...")
        time.sleep(delay)

    input_path = resolve_input_path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print(describe_input_search(args.input))
        return 1

    print(f"Loading experiments from {input_path}...")
    with open(input_path) as f:
        records = json.load(f)

    if args.max_g is not None:
        records = [r for r in records if r.get("g", 0) <= args.max_g]
    if args.g_values:
        allowed = {int(x) for x in args.g_values.split(",") if x.strip()}
        before = len(records)
        records = [r for r in records if r.get("g") in allowed]
        print(f"Filtered by g in {sorted(allowed)}: {before} -> {len(records)} records")
    if args.max_dp_per_g is not None:
        random.seed(42)
        by_g = defaultdict(list)
        for r in records:
            by_g[r.get("g", 0)].append(r)
        records = []
        for g in sorted(by_g):
            grp = by_g[g]
            records.extend(random.sample(grp, args.max_dp_per_g) if len(grp) > args.max_dp_per_g else grp)
        records.sort(key=lambda x: x.get("record_idx", 0))
    if args.max_records:
        records = records[:args.max_records]

    max_tokens = args.max_tokens or args.max_new_tokens or 16384

    est_in = sum(estimate_tokens(build_prompt(r)) for r in records)
    print(f"Records: {len(records)}")
    print(f"Est Input Tokens: {est_in:,}")

    if args.dry_run:
        return 0
    if not args.yes and input("Proceed? [y/N] ").lower() not in ("y", "yes"):
        return 0

    if args.output_file:
        output_file = abs_path(args.output_file)
    else:
        output_dir = abs_path(args.output_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = args.model.replace("/", "_").replace(":", "_")
        tools_tag = ("toolsrevise" if args.tool_revise else "tools") if args.tools else "notools"
        output_file = output_dir / f"{Path(input_path).stem}_{model_safe}_{tools_tag}_{ts}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Data-parallel: one vLLM replica per GPU group of size --tensor-parallel-size.
    gpu_ids = visible_gpus()
    tp = args.tensor_parallel_size
    n_replicas = max(1, len(gpu_ids) // tp)
    if tp * n_replicas != len(gpu_ids):
        print(f"WARN: {len(gpu_ids)} visible GPUs / TP={tp} = {n_replicas} replicas "
              f"(ignoring {len(gpu_ids) - tp * n_replicas} leftover GPU(s))")
    gpu_groups = [gpu_ids[i * tp:(i + 1) * tp] for i in range(n_replicas)]
    print(f"Launching {n_replicas} vLLM replica(s) (TP={tp} each) over GPUs {gpu_ids}")

    servers = [launch_replica(args, grp, i, args.server_log)
               for i, grp in enumerate(gpu_groups)]

    try:
        # Wait for all replicas to become ready in parallel.
        print(f"Waiting for {n_replicas} server(s) to be ready...")
        with ThreadPoolExecutor(max_workers=n_replicas) as ex:
            ready = list(ex.map(lambda s: wait_for_server(s["url"], 1200), servers))
        if not all(ready):
            for s, ok in zip(servers, ready):
                if not ok:
                    print(f"Replica {s['idx']} on GPUs {s['gpus']} failed to start.")
            return 1
        print(f"All {n_replicas} server(s) ready. Starting inference...")
        base_urls = [s["url"] for s in servers]

        is_instruct = "Instruct" in args.model and "Thinking" not in args.model
        tasks = []
        for i, r in enumerate(records):
            if args.tools:
                question, prompt = build_prompt_tools(r, no_think=is_instruct,
                                                      revise=args.tool_revise)
            else:
                question = prompt = build_prompt(r, no_think=is_instruct)
            tasks.append(Task(i, r, prompt, question))

        if args.shuffle:
            random.seed(42)
            random.shuffle(tasks)
            print(f"Shuffled {len(tasks)} tasks for randomized processing order")

        if args.resume:
            resume_paths = resolve_resume_paths(args.resume_from)
            if output_file not in resume_paths:
                resume_paths.append(output_file)
            completed = load_completed(resume_paths)
            if completed:
                before = len(tasks)
                tasks = [t for t in tasks if t.record_idx not in completed]
                print(f"Resume enabled: skipping {before - len(tasks)} completed tasks")

        tasks = shard_tasks(tasks, args.dp_world_size, args.dp_rank)
        print(f"DP shard: rank {args.dp_rank}/{args.dp_world_size} -> {len(tasks)} tasks")

        stop_sequences = [s.strip() for s in args.stop.split(",") if s.strip()]
        # ]} would truncate code blocks (e.g. `d = {"k": [1,2]}`); drop it in tools mode.
        if args.tools:
            stop_sequences = [s for s in stop_sequences if s != "]}"]

        sandbox_workers = args.sandbox_workers or default_workers()
        if args.tools:
            print(f"Sandbox pool: {sandbox_workers} workers"
                  f"{' (revise up to ' + str(args.max_revisions) + ' rounds)' if args.tool_revise else ''}")
        results = asyncio.run(run_client(
            base_urls, args.model, tasks, output_file, max_tokens,
            args.temperature, stop_sequences,
            clear_output=not args.resume and args.dp_rank == 0,
            tools_allowed=args.tools,
            tool_revise=args.tool_revise,
            max_revisions=args.max_revisions,
            sandbox_workers=sandbox_workers,
            sandbox_timeout_ms=args.sandbox_timeout_ms,
        ))

        print(f"Done. Success: {sum(1 for r in results if r['success'])}/{len(results)}")
        print(f"Saved to {output_file}")
    finally:
        print(f"Terminating {len(servers)} server(s)...")
        for s in servers:
            s["proc"].terminate()
        for s in servers:
            try:
                s["proc"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                s["proc"].kill()
            if s["log"]:
                s["log"].close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
