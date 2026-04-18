"""
prompts.py - LLM prompt generation and response parsing for monomial prediction.

This module:
1. Loads experiments from experiments.json
2. Constructs LLM prompts to predict missing monomials from examples and prefix
3. Parses LLM responses (handling reasoning tokens like <thinking>)
4. Validates if predictions are correct suffix monomials
5. Estimates token counts
"""

import csv
import json
import re
import itertools
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
STORAGE_DIR = REPO_ROOT / "storage"


def resolve_experiment_path(filepath: str) -> Path:
    path = Path(filepath)
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    storage_path = STORAGE_DIR / path.name
    if storage_path.exists():
        return storage_path
    return repo_path


# ============================================================
# Data Loading and Parsing
# ============================================================

def load_experiments(filepath: str) -> dict:
    """Load experiments from JSON file."""
    path = resolve_experiment_path(filepath)
    with open(path, 'r') as f:
        return json.load(f)


def parse_hex(h: str, n: int, p: int) -> tuple[list[int], list[int]]:
    """
    Parse hex string to (selector bits a, payload bits v).
    
    Args:
        h: Hex string like "0x200003f"
        n: Number of selector bits
        p: Number of payload bits
    
    Returns:
        (a, v) where a is selector bits and v is payload bits
    """
    v = int(h, 16)
    bits = [(v >> i) & 1 for i in range(n + p)][::-1]
    return bits[:n], bits[n:]


def monomial_to_str(indices: list[int]) -> str:
    """Convert monomial indices to human-readable string like 'x0*x3*x5'."""
    if not indices:
        return "1"  # Empty product is 1
    return "*".join(f"x{i}" for i in sorted(indices))


def monomial_to_set_notation(indices: list[int]) -> str:
    """Convert monomial indices to set notation like '{0, 3, 5}'."""
    return "{" + ", ".join(str(i) for i in sorted(indices)) + "}"


def bits_to_str(bits: list[int]) -> str:
    """Convert bit list to string like '101101'."""
    return "".join(str(b) for b in bits)


def bits_to_hex(bits: list[int]) -> str:
    """Convert bit list to hex string like '0x3f'."""
    value = 0
    for b in bits:
        value = (value << 1) | b
    return hex(value)


# ============================================================
# Prompt Construction
# ============================================================

def build_prompt(record: dict, include_reasoning_instruction: bool = False, no_think: bool = False) -> str:
    """
    Build a prompt for the LLM to predict the unknown monomial term.
    
    Args:
        record: Experiment record containing circuit parameters and samples
        include_reasoning_instruction: Whether to include reasoning instructions
        no_think: Whether to suppress thinking tokens
    
    Returns:
        Formatted prompt string
    """
    # Extract parameters from record
    n = record["n"]  # number of selector bits
    p = record["p"]  # number of payload bits
    g = record["g"]  # prefix length (number of known terms)
    d = record["d_max"] - 1  # degree of monomials (3 for d_max=4)
    prefix = record["prefix"]  # list of g known monomials (payload indices)
    samples = record["samples"]  # list of {"x": hex, "y": 0|1}
    
    total_bits = n + p
    
    # The known selector bit index is g (this is one term of the monomial)
    known_term_index = g
    
    # Build known terms description (shifted to absolute indices)
    # Each known term is: x_i AND (product of payload bits)
    # where i is 0, 1, ..., g-1 and payload bits are shifted by n
    known_terms_list = []
    for i in range(g):
        term_indices = [i] + [idx + n for idx in sorted(prefix[i])]
        term_str = " \\land ".join(f"x_{{{idx}}}" for idx in term_indices)
        known_terms_list.append(f"* $M_{{{i+1}}} = {term_str}$")
    
    if known_terms_list:
        known_terms_section = "**Known Monomials:**\n" + "\n".join(known_terms_list)
    else:
        known_terms_section = "**Known Monomials:** *(None)*"
    
    # Build example rows from samples - show all bits together as x_0, x_1, ..., x_{n+p-1}
    example_rows = []
    for row_idx, sample in enumerate(samples):
        # Parse hex to get all bits
        a, v = parse_hex(sample["x"], n, p)
        all_bits = a + v  # combine selector and payload
        y = sample["y"]
        
        # Format all bits as a single string
        bits_str = " ".join(str(b) for b in all_bits)
        
        example_rows.append(f"| {row_idx + 1} | `{bits_str}` | {y} |")
    
    example_rows_string = "\n".join(example_rows)
    
    # Total degree of the unknown monomial (including the known selector bit)
    total_degree = d + 1
    remaining_degree = d  # number of terms to find
    
    prompt = f"""Your task is to find the hidden monomial in this GF(2) polynomial.

**System Definition:**
You have {total_bits} binary variables ($x_0, x_1, \\dots, x_{{{total_bits-1}}}$).
The output $y$ is the **XOR sum** of monomials:
$$y = M_1 \\oplus M_2 \\oplus \\dots \\oplus M_{{{g+1}}}$$

{known_terms_section}

**Unknown Monomial ($M_{{{g+1}}}$):**
* This is a monomial of degree **{total_degree}**.
* One of the terms is **$x_{{{known_term_index}}}$** (this is given).
* You must find the other **{remaining_degree}** terms (indices in range $[{n}, {total_bits-1}]$).

**Observations:**
| Row | $x_0 \\; x_1 \\; \\dots \\; x_{{{total_bits-1}}}$ | $y$ |
|:---:|:---:|:---:|
{example_rows_string}

**Question:**
Find the {remaining_degree} indices (from {n} to {total_bits-1}) that, together with $x_{{{known_term_index}}}$, form the unknown monomial $M_{{{g+1}}}$.
"""

    if include_reasoning_instruction:
        prompt += f"""
**Instructions:**
1. For rows where $x_{{{known_term_index}}}=1$, the unknown monomial may contribute to $y$.
2. Account for the known monomials to isolate the unknown monomial's contribution.
3. Find the {remaining_degree} indices whose AND (with $x_{{{known_term_index}}}$) best explains the data.
"""

    prompt += f"""
**ANSWER:** Provide the {remaining_degree} indices as a list in a box, e.g., `\\boxed{[{n}, {n+2}, {n+4}]}`
"""

    return prompt
def build_prompt_compact(record: dict) -> str:
    """
    Build a more compact prompt for efficiency.
    Just calls the main build_prompt function.
    """
    return build_prompt(record, include_reasoning_instruction=False, no_think=True)


TOOLS_INSTRUCTION = """
**TOOL MODE — you MUST answer by writing Python code. No code = no credit.**

Wrap your full solution in ONE ```python ... ``` fenced block. Do not answer
in prose only. Code runs in a RestrictedPython sandbox (fresh subprocess, hard
CPU/memory limits).

**CRITICAL: The data is already loaded for you. DO NOT redefine it.**
Two globals are already set inside the sandbox — do NOT copy the prose table
into your code, do NOT assign to `RECORD = {...}`, do NOT rebuild `samples`:
  * `RECORD` — a dict with the parsed experiment:
      * `RECORD["n"]` (int) — selector-bit count (indices `0..n-1`)
      * `RECORD["p"]` (int) — payload-bit count (indices `n..n+p-1`)
      * `RECORD["g"]` (int) — number of known prefix monomials; the unknown
        monomial contains selector bit `x_g`
      * `RECORD["d_max"]` (int) — full degree; you must find `d_max - 1`
        payload indices in `[n, n+p-1]`
      * `RECORD["prefix"]` — list of lists of payload indices (offset 0..p-1;
        add `n` to get absolute indices)
      * `RECORD["samples"]` — list of dicts. Each sample has:
          * `"x"` (str) — the raw hex string like `"0x5fc..."`
          * `"x_int"` (int) — **already parsed to an integer**, use this directly
          * `"bits"` (str) — `"010110…"`, length `n+p`, with `x_0` as the
            leftmost char (bit `i` is `int(sample["bits"][i])`)
          * `"y"` (int) — 0 or 1
  * `PROMPT` — the full question text if you want to re-read it. Prefer
    `RECORD`; don't parse `PROMPT`.

**Technical limits:**
* Python 3 syntax — f-strings fine. Time budget ~500 ms wall-clock.
* No filesystem, network, subprocess, ctypes, eval/exec.
* Allowed imports ONLY: `re`, `math`, `itertools`, `collections`, `string`,
  `json`, `functools`, `operator`, `heapq`, `bisect`, `random`, `fractions`,
  `decimal`, `statistics`.

**Output — print ONE line as your final answer:**
```
answer = sorted(my_indices)     # absolute bit indices, in [n, n+p-1]
print("\\\\boxed{" + str(answer) + "}")
```

**Skeleton to start from (copy-paste, then fill in the logic):**
```
from itertools import combinations
n, p, g = RECORD["n"], RECORD["p"], RECORD["g"]
d = RECORD["d_max"] - 1
samples = [(int(s["x"], 16), s["y"]) for s in RECORD["samples"]]
# ... your reasoning over `samples` and `RECORD["prefix"]` here ...
answer = sorted(best_indices)
print("\\\\boxed{" + str(answer) + "}")
```
"""

TOOLS_REVISE_SUFFIX = """
* **Iteration:** After your code runs you will see its stdout/stderr/error. You
  may then either write another ```python ... ``` block to try again, OR give
  your final answer in plain text as `\\boxed{[i1, i2, ...]}` with no code
  block. Only respond with the boxed answer once you are confident; any turn
  that contains a code block will be re-executed.
"""


def build_prompt_tools(record: dict, no_think: bool = False,
                       revise: bool = False) -> tuple[str, str]:
    """Return (question_text, llm_prompt).

    question_text is what's exposed to the LLM's sandboxed code as the global
    variable `PROMPT`. llm_prompt is what gets sent to the model — question +
    tool-mode instructions (plus the revise suffix if enabled).
    """
    base = build_prompt(record, no_think=no_think)
    instr = TOOLS_INSTRUCTION
    if revise:
        instr += TOOLS_REVISE_SUFFIX
    return base, base + instr


def _trim(s: str, limit: int = 4000) -> str:
    if s is None:
        return ""
    return s if len(s) <= limit else s[:limit] + f"\n...[truncated {len(s) - limit} chars]"


def format_sandbox_feedback(sb: dict, revise: bool) -> str:
    """Render a sandbox result as a short user message for the next LLM turn."""
    parts = ["Execution result:"]
    if sb.get("error"):
        parts.append(f"error: {sb['error']}")
    if sb.get("stdout"):
        parts.append(f"stdout:\n{_trim(sb['stdout'])}")
    if sb.get("stderr"):
        parts.append(f"stderr:\n{_trim(sb['stderr'])}")
    parts.append(f"returncode={sb.get('returncode')}, "
                 f"elapsed={sb.get('elapsed_ms', 0):.1f}ms")
    if revise:
        parts.append(
            "\nRespond with either:\n"
            "  (a) another ```python ... ``` block to iterate, or\n"
            "  (b) your final answer in plain text as `\\boxed{[...]}` (no code)."
        )
    return "\n".join(parts)


def strip_thinking(text: Optional[str]) -> str:
    """Remove <think>/<thinking>/<thought> blocks so prior-turn history stays compact."""
    if not text:
        return ""
    t = re.sub(r'<think(?:ing)?>.*?</think(?:ing)?>', '', text, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r'<thought>.*?</thought>', '', t, flags=re.DOTALL | re.IGNORECASE)
    return t.strip()


def extract_code_from_response(response: Optional[str]) -> Optional[str]:
    """Pull a Python code block from an LLM response (with or without a closing fence)."""
    if not response:
        return None
    r = re.sub(r'<think(?:ing)?>.*?</think(?:ing)?>', '', response, flags=re.DOTALL | re.IGNORECASE)
    r = re.sub(r'<thought>.*?</thought>', '', r, flags=re.DOTALL | re.IGNORECASE)
    m = re.search(r'```(?:python|py)?\s*\n(.*?)```', r, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'```(?:python|py)?\s*\n(.*)$', r, re.DOTALL)
    return m.group(1).strip() if m else None


# ============================================================
# Response Parsing
# ============================================================

def extract_answer_from_response(response: str) -> Optional[list[int]]:
    """
    Extract the predicted monomial indices from an LLM response.
    
    Handles:
    - Reasoning tokens (text before ANSWER:)
    - <thinking>...</thinking> blocks
    - Various answer formats
    - LaTeX-style \boxed{...} answers
    
    Args:
        response: Raw LLM response text
    
    Returns:
        List of indices if found, None if parsing fails
    """
    # Remove <thinking> blocks if present
    response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)
    response = re.sub(r'<thought>.*?</thought>', '', response, flags=re.DOTALL)
    
    # Look for ANSWER: pattern (case insensitive)
    answer_match = re.search(r'ANSWER:\s*(.+)', response, re.IGNORECASE)
    
    if answer_match:
        answer_text = answer_match.group(1).strip()
        parsed = parse_monomial_answer(answer_text)
        if parsed is not None:
            return parsed
    
    # Fallback 1: Look for \boxed{...} patterns (common in math-trained models)
    # Find all occurrences and take the last one (models often repeat)
    boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', response)
    if boxed_matches:
        # Take the last match (most recent answer)
        answer_text = boxed_matches[-1].strip()
        parsed = parse_monomial_answer(answer_text)
        if parsed is not None:
            return parsed

    boxed_prefix_match = re.search(r'\\boxed\s*\{(.+)$', response)
    if boxed_prefix_match:
        answer_text = boxed_prefix_match.group(1).strip()
        parsed = parse_monomial_answer(answer_text)
        if parsed is not None:
            return parsed
    
    # Fallback 2: Look for last line or last set-like pattern
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    if lines:
        answer_text = lines[-1]
        parsed = parse_monomial_answer(answer_text)
        if parsed is not None:
            return parsed
    
    return None


def parse_monomial_answer(text: str) -> Optional[list[int]]:
    """
    Parse a monomial answer from various formats.
    
    Supported formats:
    - {0, 3, 5} or {0,3,5}
    - [0, 3, 5] or [0,3,5]
    - (0, 3, 5)
    - x0*x3*x5 or x0 * x3 * x5
    - 0, 3, 5
    
    Args:
        text: The answer text to parse
    
    Returns:
        Sorted list of indices, or None if parsing fails
    """
    text = text.strip().strip("`")
    # Handle answers that still include the \boxed{ prefix (possibly truncated)
    text = re.sub(r'^\\boxed\s*\{', '', text).strip()
    
    def safe_int(s: str) -> Optional[int]:
        """Convert string to int, returning None for unreasonable values."""
        s = s.strip()
        # Reject strings too long to be valid bit indices (max ~3 digits expected)
        if len(s) > 4:
            return None
        try:
            return int(s)
        except ValueError:
            return None
    
    def parse_int_list(parts: list[str]) -> Optional[list[int]]:
        """Parse a list of strings to ints, returning None if any fail."""
        indices = []
        for p in parts:
            val = safe_int(p)
            if val is None:
                return None
            indices.append(val)
        return sorted(indices) if indices else None
    
    # Try set/list notation: {0, 3, 5} or [0, 3, 5] or (0, 3, 5)
    # Also tolerate missing closing bracket/brace (common if a stop token strips it).
    set_match = re.match(r'[\[{(]\s*(\d+(?:\s*,\s*\d+)*)\s*[\]})]', text)
    if not set_match:
        set_match = re.match(r'[\[{(]\s*(\d+(?:\s*,\s*\d+)*)\s*$', text)
    if set_match:
        indices_str = set_match.group(1)
        result = parse_int_list(indices_str.split(','))
        if result:
            return result
    
    # Try variable notation: x0*x3*x5
    var_match = re.findall(r'x(\d+)', text)
    if var_match:
        result = parse_int_list(var_match)
        if result:
            return result
    
    # Try comma-separated numbers: 0, 3, 5
    comma_match = re.match(r'(\d+(?:\s*,\s*\d+)*)', text)
    if comma_match:
        indices_str = comma_match.group(1)
        result = parse_int_list(indices_str.split(','))
        if result:
            return result
    
    # Try space-separated numbers: 0 3 5
    space_match = re.findall(r'\d+', text)
    if space_match:
        result = parse_int_list(space_match)
        if result:
            return result
    
    return None


# ============================================================
# Validation
# ============================================================

def validate_prediction(
    predicted: list[int],
    record: dict,
    check_exact: bool = True
) -> dict:
    """
    Validate if the prediction is a correct suffix monomial.
    
    The LLM answers with absolute bit indices in range [n, n+p-1]. The target is
    stored as payload indices (0 to p-1), so we shift the target by n for comparison.
    
    Args:
        predicted: List of predicted indices (absolute bit indices in [n, n+p-1])
        record: The experiment record
        check_exact: If True, check exact match; if False, check if valid monomial
    
    Returns:
        Dictionary with validation results
    """
    n = record["n"]
    p = record["p"]
    total_bits = n + p
    d = record["d_max"] - 1
    # Shift target indices by n to get absolute bit indices
    target = sorted([i + n for i in record["target"]])
    
    result = {
        "predicted": predicted,
        "target": target,
        "is_valid_format": False,
        "is_correct": False,
        "errors": []
    }
    
    if predicted is None:
        result["errors"].append("Failed to parse prediction")
        return result
    
    # Check format validity
    if len(predicted) != d:
        result["errors"].append(f"Wrong degree: expected {d}, got {len(predicted)}")
    elif len(set(predicted)) != len(predicted):
        result["errors"].append("Duplicate indices")
    elif any(i < n or i >= total_bits for i in predicted):
        result["errors"].append(f"Indices out of range [{n}, {total_bits-1}]")
    else:
        result["is_valid_format"] = True
    
    # Check correctness
    if check_exact and result["is_valid_format"]:
        result["is_correct"] = (sorted(predicted) == target)
    
    return result


def all_valid_monomials(n: int, p: int, d: int) -> list[list[int]]:
    """
    Generate all valid degree-d monomials over n+p bits.
    
    Args:
        n: Number of selector bits (for offset)
        p: Number of payload bits
        d: Degree of monomial
    
    Returns:
        List of valid monomial index lists (using absolute bit indices)
    """
    total_bits = n + p
    return [list(combo) for combo in itertools.combinations(range(total_bits), d)]


# ============================================================
# Token Estimation
# ============================================================

def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """
    Estimate token count for text.
    
    Uses a simple heuristic based on character count.
    For more accurate estimates, use tiktoken or model-specific tokenizers.
    
    Args:
        text: Input text
        chars_per_token: Average characters per token (4.0 is typical for English)
    
    Returns:
        Estimated token count
    """
    return max(1, int(len(text) / chars_per_token))


def estimate_output_tokens(record: dict, with_reasoning: bool = True) -> int:
    """
    Estimate expected output tokens for a response.
    
    Args:
        record: Experiment record
        with_reasoning: Whether response includes reasoning
    
    Returns:
        Estimated output token count
    """
    d = record["d_max"] - 1
    
    # Base answer: "ANSWER: {0, 3, 5}" ~ 20 tokens
    base_tokens = 20
    
    if with_reasoning:
        # Reasoning varies, estimate ~200-500 tokens for thinking
        # Longer for harder problems (more prefix terms, larger p)
        reasoning_tokens = 200 + record["g"] * 30 + record["p"] * 5
        return base_tokens + reasoning_tokens
    
    return base_tokens


# ============================================================
# Batch Processing
# ============================================================

def generate_all_prompts(
    records: list,
    compact: bool = False,
    max_records: Optional[int] = None
) -> list[dict]:
    """
    Generate prompts for all experiments.
    
    Args:
        records: List of experiment records
        compact: Use compact prompt format
        max_records: Maximum number of records to process (None for all)
    
    Returns:
        List of dicts with 'prompt', 'record', 'input_tokens', 'output_tokens_estimate'
    """
    if max_records:
        records = records[:max_records]
    
    prompt_fn = build_prompt_compact if compact else build_prompt
    
    results = []
    for record in records:
        prompt = prompt_fn(record)
        results.append({
            "prompt": prompt,
            "record": record,
            "input_tokens": estimate_tokens(prompt),
            "output_tokens_estimate": estimate_output_tokens(record, with_reasoning=True)
        })
    
    return results


def print_token_summary(prompts: list[dict], output_reasoning_tokens: int = 300):
    """
    Print summary of token estimates.
    
    Args:
        prompts: List of prompt dicts from generate_all_prompts
        output_reasoning_tokens: Expected output tokens including reasoning
    """
    total_input = sum(p["input_tokens"] for p in prompts)
    total_output = sum(p["output_tokens_estimate"] for p in prompts)
    num_questions = len(prompts)
    
    avg_input = total_input / num_questions if num_questions > 0 else 0
    avg_output = total_output / num_questions if num_questions > 0 else 0
    
    print("=" * 60)
    print("TOKEN ESTIMATE SUMMARY")
    print("=" * 60)
    print(f"Number of questions: {num_questions:,}")
    print()
    print("Input Tokens:")
    print(f"  Average per question: {avg_input:,.0f}")
    print(f"  Total: {total_input:,}")
    print()
    print("Output Tokens (estimated with reasoning):")
    print(f"  Average per question: {avg_output:,.0f}")
    print(f"  Total: {total_output:,}")
    print()
    print("Combined:")
    print(f"  Total tokens: {total_input + total_output:,}")
    print("=" * 60)


def save_example_prompts(
    records: list,
    mode_name: str,
    output_file: str = "example_prompts.txt",
    num_examples: int = 6,
    compact: bool = False
):
    """
    Save example prompts from the dataset.
    
    For small datasets (<=10 records), shows all examples.
    For larger datasets, shows one example per unique g value.
    
    Args:
        records: List of experiment records
        mode_name: Name of the mode (for labeling)
        output_file: Path to output file
        num_examples: Number of examples to save (ignored - behavior based on dataset size)
        compact: Use compact prompt format
    """
    prompt_fn = build_prompt_compact if compact else build_prompt
    
    lines = []
    lines.append("=" * 80)
    lines.append("EXAMPLE PROMPTS FOR MONOMIAL PREDICTION")
    lines.append(f"MODE: {mode_name.upper()}")
    lines.append("=" * 80)
    lines.append("")
    
    if not records:
        lines.append("(No records found)")
        with open(output_file, 'w') as f:
            f.write("\n".join(lines))
        return
    
    # If dataset is small (<=10 records), show all examples
    # Otherwise, show one example per unique g value
    if len(records) <= 10:
        selected = records
    else:
        # Find all unique g values and select one example for each
        g_values = sorted(set(r["g"] for r in records))
        selected = []
        
        for g_val in g_values:
            # Find first record with this g value
            for r in records:
                if r["g"] == g_val:
                    selected.append(r)
                    break
    
    # Write selected examples
    for i, record in enumerate(selected):
        n = record["n"]
        # Shift target indices by n to show absolute bit indices
        shifted_target = sorted([idx + n for idx in record["target"]])
        lines.append("-" * 80)
        lines.append(f"EXAMPLE {i+1}: g={record['g']}, n={n}, p={record['p']}, d_max={record['d_max']}")
        lines.append(f"Target answer: {shifted_target}")
        lines.append("-" * 80)
        lines.append("")
        lines.append(prompt_fn(record))
        lines.append("")
        lines.append("")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"Saved {len(selected)} example prompts to {output_file}")


def save_g_prompts(
    records: list,
    g_value: int,
    output_file: str,
    source_name: str,
    compact: bool = False,
    num_examples: int = 10
):
    """
    Save prompts for a specific g value (first N examples).
    
    Args:
        records: List of experiment records
        g_value: g value to filter
        output_file: Output filename
        source_name: Name of source dataset
        compact: Use compact prompt format
        num_examples: Number of examples to save
    """
    prompt_fn = build_prompt_compact if compact else build_prompt
    selected = [r for r in records if r["g"] == g_value][:num_examples]
    
    lines = []
    lines.append("=" * 80)
    lines.append("LLM EXAMPLE PROMPTS FOR MONOMIAL PREDICTION")
    lines.append(f"Source dataset: {source_name}")
    lines.append(f"g value: {g_value}")
    lines.append(f"Number of examples: {len(selected)}")
    lines.append("=" * 80)
    lines.append("")
    
    if not selected:
        lines.append("(No records found)")
        with open(output_file, 'w') as f:
            f.write("\n".join(lines))
        return
    
    for i, record in enumerate(selected):
        n = record["n"]
        p = record["p"]
        d_max = record["d_max"]
        shifted_target = sorted([idx + n for idx in record["target"]])
        lines.append("-" * 80)
        lines.append(f"EXAMPLE {i+1} (g={g_value}): n={n}, p={p}, d_max={d_max}")
        lines.append(f"Target answer: {shifted_target}")
        lines.append("-" * 80)
        lines.append("")
        lines.append(prompt_fn(record))
        lines.append("")
        lines.append("")
    
    with open(output_file, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"Saved {len(selected)} example prompts to {output_file}")


def save_g_prompts_csv(
    groups: list[tuple[int, list]],
    output_file: str,
    compact: bool = False,
    num_examples: int = 10
):
    """
    Save prompts for multiple g values into a single CSV.
    
    Each example is three rows: g, target, prompt.
    Order follows the groups list.
    
    Args:
        groups: List of tuples (g_value, records)
        output_file: Output CSV filename
        compact: Use compact prompt format
        num_examples: Number of examples per g value
    """
    prompt_fn = build_prompt_compact if compact else build_prompt
    safety_prefix = "DO NOT RUN ANY CODE. DO NOT USE TOOL CALLS."
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for g_value, records in groups:
            selected = [r for r in records if r["g"] == g_value][:num_examples]
            for record in selected:
                n = record["n"]
                shifted_target = sorted([idx + n for idx in record["target"]])
                writer.writerow(["g", g_value])
                writer.writerow(["target", shifted_target])
                prompt = prompt_fn(record)
                writer.writerow(["prompt", prompt])
                writer.writerow(["prompt", f"{safety_prefix} {prompt}"])
    
    print(f"Saved g-specific prompts to {output_file}")


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Generate LLM prompts for monomial prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Default behavior (no args):
  Generates llm_examples.txt with prompts for first 10 problems in llm dataset
  for p=12, d=4 (d_max=5), g=[31, 63, 123].

Examples:
  uv run prompts.py
  uv run prompts.py --input experiments_llm_diagnostic.json --mode diagnostic
  uv run prompts.py --show-example --compact
  uv run prompts.py --test-parse "ANSWER: [20, 23, 25]"
        """
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Path to experiments JSON (default: None, uses llm dataset)")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["diagnostic", "adversarial", "both"],
                        help="Which experiment mode to process (default: None, uses llm dataset)")
    parser.add_argument("--compact", action="store_true",
                        help="Use compact prompt format")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Maximum records to process (default: all)")
    parser.add_argument("--show-example", action="store_true",
                        help="Show an example prompt")
    parser.add_argument("--test-parse", type=str, default=None,
                        help="Test parsing a response string")
    parser.add_argument("--example-file", type=str, default=None,
                        help="Output file for example prompts (default: llm_examples.txt when no args)")
    parser.add_argument("--num-examples", type=int, default=6,
                        help="Number of examples per mode to save (default: 6)")
    args = parser.parse_args()
    
    # Test response parsing if requested
    if args.test_parse:
        print("\nTesting response parsing:")
        result = extract_answer_from_response(args.test_parse)
        print(f"  Input: {args.test_parse}")
        print(f"  Parsed: {result}")
        return
    
    # Check if running with no arguments (default behavior)
    # If input is None and no other args were explicitly provided, use default llm behavior
    no_args_mode = (args.input is None and args.mode is None and 
                    args.example_file is None and not args.show_example and 
                    args.max_records is None and not args.compact)
    
    if no_args_mode:
        # Default behavior: generate llm_examples.txt for specific parameters
        print("Running in default mode: generating llm_examples.txt")
        print("=" * 60)
        
        # Load llm adversarial dataset
        input_file = "experiments_llm_adversarial.json"
        print(f"Loading experiments from {input_file}...")
        records = load_experiments(input_file)
        print(f"  Loaded {len(records)} records")
        
        # Filter for p=12, d_max=5 (d=4), g in [31, 63, 123]
        # Note: d=4 means d_max=5, but if that doesn't exist, we'll try d_max=4 (d=3)
        p = 12
        d_max_target = 5  # d = d_max - 1 = 4
        g_values = [31, 63, 123]
        
        # Try d_max=5 first, fall back to d_max=4 if not found
        filtered_records = []
        d_max_used = None
        
        for d_max in [d_max_target, 4]:
            matching_all = []
            for g_val in g_values:
                matching = [r for r in records 
                           if r["p"] == p and r["d_max"] == d_max and r["g"] == g_val]
                # Take first 10 problems for this g value
                matching_all.extend(matching[:10])
                if matching:
                    print(f"  Found {len(matching)} records for p={p}, d_max={d_max}, g={g_val} (taking first 10)")
            
            if matching_all:
                filtered_records = matching_all
                d_max_used = d_max
                break
        
        if not filtered_records:
            print(f"\nWARNING: No records found matching p={p}, d_max={d_max_target} or d_max=4, g in {g_values}")
            print("Available parameters in dataset:")
            p12_records = [r for r in records if r["p"] == 12]
            if p12_records:
                d_max_vals = sorted(set(r["d_max"] for r in p12_records))
                g_vals = sorted(set(r["g"] for r in p12_records))
                print(f"  d_max values: {d_max_vals}")
                print(f"  g values: {g_vals}")
            return
        
        # Filter again with the d_max that worked, grouping by g
        filtered_by_g = {}
        for g_val in g_values:
            matching = [r for r in filtered_records 
                       if r["p"] == p and r["d_max"] == d_max_used and r["g"] == g_val]
            if matching:
                filtered_by_g[g_val] = matching[:10]
        
        if not filtered_by_g:
            print(f"\nWARNING: No records found after filtering")
            return
        
        # Generate prompts
        prompt_fn = build_prompt_compact if args.compact else build_prompt
        
        lines = []
        lines.append("=" * 80)
        lines.append("LLM EXAMPLE PROMPTS FOR MONOMIAL PREDICTION")
        d_actual = d_max_used - 1
        lines.append(f"Parameters: p={p}, d_max={d_max_used} (d={d_actual})")
        available_g = sorted(filtered_by_g.keys())
        lines.append(f"g values: {available_g} (requested: {g_values})")
        if set(available_g) != set(g_values):
            missing = set(g_values) - set(available_g)
            lines.append(f"Note: g values {sorted(missing)} not found in dataset")
        lines.append("=" * 80)
        lines.append("")
        
        # Group by g value
        total_examples = 0
        for g_val in g_values:
            if g_val not in filtered_by_g:
                continue
            g_records = filtered_by_g[g_val]
            total_examples += len(g_records)
            
            lines.append("-" * 80)
            lines.append(f"PARAMETER SET: p={p}, d_max={d_max_used}, g={g_val}")
            lines.append(f"Number of examples: {len(g_records)}")
            lines.append("-" * 80)
            lines.append("")
            
            for i, record in enumerate(g_records):
                n = record["n"]
                shifted_target = sorted([idx + n for idx in record["target"]])
                lines.append("-" * 80)
                lines.append(f"EXAMPLE {i+1} (g={g_val}): n={n}, p={p}, d_max={d_max_used}")
                lines.append(f"Target answer: {shifted_target}")
                lines.append("-" * 80)
                lines.append("")
                lines.append(prompt_fn(record))
                lines.append("")
                lines.append("")
        
        # Write to file
        output_file = "llm_examples.txt"
        with open(output_file, 'w') as f:
            f.write("\n".join(lines))
        
        print(f"\nSaved {total_examples} example prompts to {output_file}")

        # Also generate g-specific outputs for large adversarial set
        print("\nGenerating g-specific prompt CSV...")
        llm_large_file = "storage/experiments_llm_large_adversarial.json"
        print(f"Loading experiments from {llm_large_file}...")
        llm_large_records = load_experiments(llm_large_file)
        print(f"  Loaded {len(llm_large_records)} records")
        
        g_values_csv = [31, 63, 127]
        datasets = [
            ("llm", records),
            ("llm_large", llm_large_records),
        ]
        groups = []
        for g_val in g_values_csv:
            selected_records = None
            selected_source = None
            for source_name, dataset in datasets:
                if any(r["g"] == g_val for r in dataset):
                    selected_records = dataset
                    selected_source = source_name
                    break
            if selected_records is None:
                print(f"  WARNING: No records found for g={g_val} in any dataset")
                continue
            print(f"  Using source '{selected_source}' for g={g_val}")
            groups.append((g_val, selected_records))
        
        save_g_prompts_csv(
            groups=groups,
            output_file="all_results/llm_g_prompts.csv",
            compact=args.compact,
            num_examples=10
        )
        return
    
    # Original behavior when arguments are provided
    input_file = args.input or "experiments_llm_diagnostic.json"
    
    # Load experiments
    print(f"Loading experiments from {input_file}...")
    records = load_experiments(input_file)
    print(f"  Loaded {len(records)} records")
    
    # Determine mode from filename if not explicit
    mode_name = args.mode
    if mode_name is None:
        if "diagnostic" in input_file:
            mode_name = "diagnostic"
        elif "adversarial" in input_file:
            mode_name = "adversarial"
        else:
            mode_name = "unknown"
    
    # Save example prompts to file
    example_file = args.example_file or "example_prompts.txt"
    save_example_prompts(
        records,
        mode_name=mode_name,
        output_file=example_file,
        num_examples=args.num_examples,
        compact=args.compact
    )
    
    print(f"\n{'='*60}")
    print(f"Processing {mode_name.upper()} mode")
    print(f"{'='*60}")
    
    # Generate prompts
    prompts = generate_all_prompts(
        records, 
        compact=args.compact,
        max_records=args.max_records
    )
    
    # Show example if requested
    if args.show_example and prompts:
        print("\n" + "-"*60)
        print("EXAMPLE PROMPT:")
        print("-"*60)
        print(prompts[0]["prompt"])
        print("-"*60)
        
        # Show expected answer (shifted to absolute bit indices)
        record = prompts[0]["record"]
        n = record["n"]
        shifted_target = sorted([idx + n for idx in record["target"]])
        print(f"\nExpected answer: {shifted_target}")
    
    # Print token summary
    print()
    print_token_summary(prompts)


if __name__ == "__main__":
    main()
