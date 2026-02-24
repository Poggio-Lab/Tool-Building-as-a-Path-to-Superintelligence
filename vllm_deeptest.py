"""
vLLM Local Inference for Monomial Prediction (Client-Server Mode).

Starts a local vLLM server and runs inference using the OpenAI API.
Saves responses incrementally to JSONL files.

Usage:
    uv run expirements/vllm_deeptest.py
    uv run expirements/vllm_deeptest.py --input experiments_llm_diagnostic.json
    uv run expirements/vllm_deeptest.py --model Qwen/Qwen3-4B-Instruct-2507
"""

import argparse
import asyncio
import fcntl
import json
import os
import random
import re
import socket
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from tqdm.asyncio import tqdm

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
STORAGE_DIR = REPO_ROOT / "storage"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompts import (
    build_prompt,
    estimate_tokens,
    estimate_output_tokens,
    extract_answer_from_response,
    validate_prediction,
)

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

# ============================================================
# Configuration
# ============================================================

MODELS = [
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
]

QWEN_4B_PREFIX = "Qwen/Qwen3-4B"
LLM_DEFAULT_DATASET = "experiments_llm_adversarial.json"


# ============================================================
# Helpers
# ============================================================

def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def wait_for_server(url: str, timeout: int = 600):
    """Wait for the vLLM server to be ready."""
    import requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{url}/health")
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def resolve_input_path(input_arg: Optional[str], model: str) -> Path:
    candidate = input_arg or LLM_DEFAULT_DATASET
    input_path = Path(candidate)
    if input_path.is_absolute() and input_path.exists():
        return input_path
    repo_path = REPO_ROOT / input_path
    if repo_path.exists():
        return repo_path
    storage_path = STORAGE_DIR / input_path.name
    if storage_path.exists():
        return storage_path
    return repo_path


def estimate_token_summary(records: list) -> dict:
    total_input_tokens = 0
    total_output_tokens = 0
    for record in records:
        prompt = build_prompt(record)
        total_input_tokens += estimate_tokens(prompt)
        total_output_tokens += estimate_output_tokens(record)
    return {
        "num_records": len(records),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }

def append_jsonl(output_file: Path, record: dict):
    line = json.dumps(record)
    with open(output_file, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_completed_record_idxs(output_file: Path) -> set:
    completed = set()
    if not output_file.exists():
        return completed
    with open(output_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_idx = record.get("record_idx")
            if record_idx is not None:
                completed.add(record_idx)
    return completed


def load_completed_record_idxs_from_paths(paths: List[Path]) -> set:
    completed = set()
    for path in paths:
        completed |= load_completed_record_idxs(path)
    return completed


def resolve_resume_paths(resume_from_arg: Optional[str]) -> List[Path]:
    paths = []
    if not resume_from_arg:
        return paths
    for raw in resume_from_arg.split(","):
        raw = raw.strip()
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        paths.append(path)
    return paths


def shard_tasks(tasks: List["Task"], world_size: int, rank: int) -> List["Task"]:
    if world_size <= 1:
        return tasks
    total = len(tasks)
    base = total // world_size
    remainder = total % world_size
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return tasks[start:end]

# ============================================================
# Client Logic
# ============================================================

@dataclass
class Task:
    record_idx: int
    record: dict
    prompt: str


async def process_task(
    client: AsyncOpenAI,
    task: Task,
    model: str,
    max_tokens: int,
    temperature: float,
    stop_sequences: List[str],
    pbar: tqdm,
    output_file: Path,
    file_lock: asyncio.Lock,
):
    start_time = time.time()
    response_text = None
    input_tokens = 0
    output_tokens = 0
    success = False
    error = None

    try:
        # Use a very long timeout to prevent timeouts on long prompts
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": task.prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_sequences,
            timeout=3600.0,  # 1 hour timeout
        )
        
        response_text = response.choices[0].message.content
        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
        success = True

    except Exception as e:
        error = str(e)
        success = False

    elapsed = time.time() - start_time

    # Parse and validate
    parsed_answer = None
    validation = None
    reasoning = ""
    
    if success and response_text:
        match = re.search(r"<thinking>(.*?)</thinking>", response_text, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            match = re.search(r"<thought>(.*?)</thought>", response_text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            reasoning = match.group(1).strip()
            
        parsed_answer = extract_answer_from_response(response_text)
        if parsed_answer is not None:
            validation = validate_prediction(parsed_answer, task.record)

    result = {
        "record_idx": task.record_idx,
        "p": task.record["p"],
        "g": task.record["g"],
        "d_max": task.record["d_max"],
        "target": task.record["target"],
        "prompt": task.prompt,
        "response": response_text,
        "reasoning": reasoning,
        "parsed_answer": parsed_answer,
        "is_correct": validation["is_correct"] if validation else False,
        "is_valid_format": validation["is_valid_format"] if validation else False,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_seconds": elapsed,
        "success": success,
        "error": error,
    }

    # Write to file
    async with file_lock:
        append_jsonl(output_file, result)
    
    pbar.update(1)
    return result


async def run_client(
    base_url: str,
    api_key: str,
    model: str,
    tasks: List[Task],
    output_file: Path,
    max_tokens: int,
    temperature: float,
    stop_sequences: List[str],
    clear_output: bool,
):
    if clear_output:
        with open(output_file, "w") as f:
            pass

    # Use async context manager to ensure client is properly closed
    # This prevents connection pool hangs after the first batch completes
    async with AsyncOpenAI(base_url=f"{base_url}/v1", api_key=api_key) as client:
        file_lock = asyncio.Lock()

        # Create tasks
        # Limit concurrency to avoid overloading the server and HTTP connection pool.
        # Lower values prevent connection pool exhaustion issues.
        sem = asyncio.Semaphore(64)

        async def sem_task(t):
            async with sem:
                return await process_task(
                    client, t, model, max_tokens, temperature, 
                    stop_sequences, pbar, output_file, file_lock
                )

        results = []
        with tqdm(total=len(tasks), desc="Processing") as pbar:
            coros = [sem_task(t) for t in tasks]
            results = await asyncio.gather(*coros)

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM inference locally using vLLM Server + OpenAI Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--model", type=str, default=MODELS[0])
    parser.add_argument("--output-dir", type=str, default="llm_results")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", "-y", action="store_true")
    parser.add_argument("--max-g", type=int, default=None)
    parser.add_argument("--max-dp-per-g", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=None) # Ignored in server mode usually, or maps to TP
    parser.add_argument("--dp-world-size", type=int, default=1)
    parser.add_argument("--dp-rank", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None, help="Comma-separated JSONL paths to resume from")
    parser.add_argument("--server-log", type=str, default=None)
    parser.add_argument("--stop", type=str, default="<|im_end|>,<|endoftext|>,]}")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle task order for faster coverage across all g values")
    
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    if AsyncOpenAI is None:
        print("Error: 'openai' package is required. Install with: uv pip install openai")
        return 1

    # Add a random sleep to prevent race conditions when starting multiple instances
    # concurrently (e.g. port selection, cache initialization)
    if not args.dry_run:
        startup_delay = random.uniform(0, 10)
        print(f"Waiting {startup_delay:.2f}s to stagger startup...")
        time.sleep(startup_delay)

    if args.dp_world_size < 1:
        print("ERROR: --dp-world-size must be >= 1")
        return 1
    if args.dp_rank < 0 or args.dp_rank >= args.dp_world_size:
        print("ERROR: --dp-rank must be in [0, dp-world-size)")
        return 1

    # Resolve input
    input_path = resolve_input_path(args.input, args.model)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    # Load records
    print(f"Loading experiments from {input_path}...")
    with open(input_path, "r") as f:
        records = json.load(f)
    
    # Filter records
    if args.max_g is not None:
        records = [r for r in records if r.get("g", 0) <= args.max_g]
    
    if args.max_dp_per_g is not None:
        random.seed(42)
        by_g = defaultdict(list)
        for r in records:
            by_g[r.get("g", 0)].append(r)
        new_records = []
        for g_val in sorted(by_g.keys()):
            g_records = by_g[g_val]
            if len(g_records) > args.max_dp_per_g:
                g_records = random.sample(g_records, args.max_dp_per_g)
            new_records.extend(g_records)
        records = new_records
        records.sort(key=lambda x: x.get("record_idx", 0))

    if args.max_records:
        records = records[:args.max_records]

    if args.max_tokens is None and args.max_new_tokens is not None:
        args.max_tokens = args.max_new_tokens
    effective_max_tokens = args.max_tokens if args.max_tokens is not None else 16384

    # Estimate
    estimate = estimate_token_summary(records)
    print(f"Records: {len(records)}")
    print(f"Est Input Tokens: {estimate['total_input_tokens']:,}")
    
    if args.dry_run:
        return 0

    if not args.yes:
        if input("Proceed? [y/N] ").lower() not in ("y", "yes"):
            return 0

    # Prepare output
    if args.output_file:
        output_file = Path(args.output_file)
        if not output_file.is_absolute():
            output_file = REPO_ROOT / output_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = args.model.replace("/", "_").replace(":", "_")
        input_base = Path(input_path).stem
        output_file = output_dir / f"{input_base}_{model_safe}_{timestamp}.jsonl"

    # Start Server
    port = find_free_port()
    host = "127.0.0.1"
    base_url = f"http://{host}:{port}"
    
    server_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--max-model-len", str(args.max_model_len),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--dtype", args.dtype,
        "--disable-log-requests",
    ]
    if args.trust_remote_code:
        server_cmd.append("--trust-remote-code")

    print(f"Starting vLLM server on port {port}...")
    print(f"Command: {' '.join(server_cmd)}")
    
    server_log_handle = None
    server_stdout = subprocess.DEVNULL
    server_stderr = subprocess.DEVNULL
    if args.server_log:
        server_log_path = Path(args.server_log)
        if not server_log_path.is_absolute():
            server_log_path = REPO_ROOT / server_log_path
        server_log_path.parent.mkdir(parents=True, exist_ok=True)
        server_log_handle = open(server_log_path, "a", buffering=1)
        server_stdout = server_log_handle
        server_stderr = server_log_handle

    # Pass current environment (including CUDA_VISIBLE_DEVICES)
    # IMPORTANT: Use DEVNULL to prevent pipe buffer deadlock.
    # If stdout/stderr use PIPE but are never read, the buffer fills up
    # and the server process blocks on write, causing a hang.
    server_process = subprocess.Popen(
        server_cmd,
        stdout=server_stdout,
        stderr=server_stderr,
        env=os.environ.copy()
    )

    try:
        # Wait for server ready
        print("Waiting for server to be ready...")
        
        # We can stream stdout/stderr to see startup progress
        # But for now, just poll health check
        if not wait_for_server(base_url, timeout=1200): # 20 min startup timeout
            print("Server failed to start. Check if another process is using the GPU or if there's an OOM issue.")
            return 1
            
        print("Server ready. Starting inference...")

        # Prepare tasks
        is_instruct_model = "Instruct" in args.model and "Thinking" not in args.model
        tasks = [Task(i, r, build_prompt(r, no_think=is_instruct_model)) for i, r in enumerate(records)]
        
        # Shuffle tasks for faster coverage across all g values
        if args.shuffle:
            random.seed(42)  # Fixed seed for repeatability
            random.shuffle(tasks)
            print(f"Shuffled {len(tasks)} tasks for randomized processing order")

        if args.resume:
            resume_paths = resolve_resume_paths(args.resume_from)
            if output_file not in resume_paths:
                resume_paths.append(output_file)
            completed = load_completed_record_idxs_from_paths(resume_paths)
            if completed:
                before = len(tasks)
                tasks = [t for t in tasks if t.record_idx not in completed]
                skipped = before - len(tasks)
                if skipped:
                    print(f"Resume enabled: skipping {skipped} completed tasks")

        # Shard after filtering so remaining work splits evenly
        tasks = shard_tasks(tasks, args.dp_world_size, args.dp_rank)
        print(f"DP shard: rank {args.dp_rank}/{args.dp_world_size} -> {len(tasks)} tasks")
        
        stop_sequences = [s.strip() for s in args.stop.split(",") if s.strip()]

        # Run Client
        results = asyncio.run(run_client(
            base_url=base_url,
            api_key="EMPTY",
            model=args.model,
            tasks=tasks,
            output_file=output_file,
            max_tokens=effective_max_tokens,
            temperature=args.temperature,
            stop_sequences=stop_sequences,
            clear_output=not args.resume and args.dp_rank == 0,
        ))

        # Final stats
        successes = sum(1 for r in results if r["success"])
        print(f"Done. Success: {successes}/{len(results)}")
        print(f"Saved to {output_file}")

    finally:
        print("Terminating server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()
        if server_log_handle:
            server_log_handle.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())
