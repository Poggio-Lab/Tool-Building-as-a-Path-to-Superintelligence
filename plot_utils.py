from __future__ import annotations

import csv
import itertools
import json
import math
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex


# ---------- paths ----------
REPO_ROOT = Path(__file__).resolve().parent
PLOTS_DIR = REPO_ROOT / "plots"
LLM_RESULTS_DIR = REPO_ROOT / "llm_results"
STORAGE_DIR = REPO_ROOT / "storage"
LLM_PROMPTS_FILE = REPO_ROOT / "llm_g_prompts2.xlsx"


def resolve_output_path(path_str: str | Path, base_dir: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p)


# ---------- combinatorics ----------
def nCr(n: int, r: int) -> int:
    return 0 if (r < 0 or r > n) else math.comb(n, r)


# ---------- colors (11 distinct categorical colors) ----------
def _palette_11() -> list[str]:
    """
    Deterministic categorical palette with 11 distinct colors.
    Uses Matplotlib's qualitative 'tab20' and takes the first 11 entries.
    """
    cmap = plt.get_cmap("tab20")
    cols = getattr(cmap, "colors", None)
    if cols is None:
        return [to_hex(cmap(i / 10.0)) for i in range(11)]
    return [to_hex(c) for c in cols[:11]]


_COLOR11 = _palette_11()
_ESTIMATOR_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def get_estimator_color(estimator: str) -> str:
    return _COLOR11[_ESTIMATOR_IDX.get(estimator.upper(), 0)]


def get_llm_plot_color(model_name: str) -> str | None:
    s = str(model_name).lower()

    # small LLMs: 4 slots (4..7)
    if "30b" in s and "thinking" in s:
        return _COLOR11[4]
    if "30b" in s and "instruct" in s:
        return _COLOR11[5]
    if "4b" in s and "thinking" in s:
        return _COLOR11[6]
    if "4b" in s and "instruct" in s:
        return _COLOR11[7]

    # frontier: 3 slots (8..10)
    if "chatgpt" in s or re.search(r"\bgpt\b", s):
        return _COLOR11[8]
    if "opus" in s:
        return _COLOR11[9]
    if "gemini" in s:
        return _COLOR11[10]
    return None


def short_model_name(model_name: str, max_len: int = 20) -> str:
    name = model_name.split("/")[-1] if "/" in model_name else model_name
    if ":" in name:
        name = name.split(":", 1)[0]
    return name if len(name) <= max_len else (name[: max_len - 3] + "...")


# ---------- parsing / boxed fallback ----------
# Updated regex to handle \boxed{...} (nested braces) and \boxed[...] (brackets)
_BOXED_RE = re.compile(r"\\boxed\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|\[((?:[^\[\]]|\[[^\[\]]*\])*)\])")
_BOXED_PREFIX_RE = re.compile(r"\\boxed\s*(?:\{|\[)(.+)$")


def _parse_indices_from_text(text: str) -> list[int] | None:
    t = text.strip().strip("`")
    # Simple robust parsing: extract all numbers
    nums = re.findall(r"\d+", t)
    if nums:
        try:
            xs = [int(n) for n in nums]
            return sorted(xs) if xs else None
        except ValueError:
            return None
    return None


def parse_boxed_answer(text: str) -> list[list[int]]:
    out: list[list[int]] = []
    for m in _BOXED_RE.finditer(text or ""):
        # Group 1 is {}, Group 2 is []
        content = m.group(1) if m.group(1) is not None else m.group(2)
        parsed = _parse_indices_from_text(content.strip())
        if parsed is not None:
            out.append(parsed)
    pm = _BOXED_PREFIX_RE.search(text or "")
    if pm:
        parsed = _parse_indices_from_text(pm.group(1).strip())
        if parsed is not None and parsed not in out:
            out.append(parsed)
    return out


def _extract_n_from_prompt(prompt: str) -> int | None:
    if not prompt:
        return None
    for pat in (
        r"\(from\s+(\d+)\s+to\s+\d+\)",
        r"indices in range \$\[(\d+),",
        r"\[(\d+),\s*\d+\]",
    ):
        m = re.search(pat, prompt)
        if m:
            return int(m.group(1))
    return None


def recompute_is_correct_with_boxed_fallback(record: dict) -> bool:
    if record.get("is_correct", False):
        return True
    resp = record.get("response", "") or ""
    tgt = record.get("target", [])
    if not resp or not tgt:
        return False

    n = record.get("n")
    if n is None:
        n = _extract_n_from_prompt(record.get("prompt", "") or "")
    if n is None:
        return False

    try:
        target_abs = sorted([int(i) + int(n) for i in tgt])
    except Exception:
        return False

    for ans in parse_boxed_answer(resp):
        if sorted(ans) == target_abs:
            return True
    return False


# ---------- LLM JSONL ----------
def load_llm_results(
    llm_results_dir: str | Path = LLM_RESULTS_DIR,
    filter_adversarial: bool = True,
) -> dict[str, list[dict]]:
    results_dir = resolve_output_path(llm_results_dir, REPO_ROOT)
    if not results_dir.exists():
        return {}

    out: dict[str, list[dict]] = {}
    for f in results_dir.glob("*.jsonl"):
        if filter_adversarial and "adversarial" not in f.name:
            continue

        records: list[dict] = []
        with f.open("r") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            continue

        model = records[0].get("model")
        if model is None:
            parts = f.stem.split("_")
            model_parts, capture = [], False
            for part in parts:
                if part in ("adversarial", "diagnostic"):
                    capture = True
                    continue
                if capture:
                    if part.isdigit() and len(part) >= 8:
                        break
                    model_parts.append(part)
            model = "/".join(model_parts) if len(model_parts) >= 2 else "_".join(model_parts)

        out.setdefault(str(model), []).extend(records)

    return out


# ---------- aggregation ----------
def aggregate_results(results: list[dict], group_keys: list[str], with_std: bool = False) -> dict:
    agg: dict[tuple, dict[str, list[float]]] = {}
    for r in results:
        key = tuple(r[k] for k in group_keys)
        d = agg.setdefault(key, {"A": [], "B": [], "C": [], "D": []})
        for alg in ("A", "B", "C", "D"):
            d[alg].append(float(r[f"gamma_{alg}"]))

    out: dict = {}
    for key, d in agg.items():
        out[key] = {}
        for alg, vals in d.items():
            arr = np.asarray(vals, dtype=float)
            if with_std:
                out[key][alg] = {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}
            else:
                out[key][alg] = float(arr.mean())
    return out


# ---------- Jeffreys interval (Beta(1/2, 1/2)) ----------
def jeffreys_interval(k: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Jeffreys equal-tailed interval for a binomial proportion.
    This is the posterior credible interval under the Jeffreys prior Beta(1/2, 1/2).
    """
    if n <= 0:
        return 0.0, 1.0
    try:
        from scipy.stats import beta
        a = float(k) + 0.5
        b = float(n - k) + 0.5
        return float(beta.ppf(alpha / 2, a, b)), float(beta.ppf(1 - alpha / 2, a, b))
    except Exception:
        # Wilson fallback
        z = 1.959963984540054
        p = float(k) / float(n)
        denom = 1 + (z * z) / n
        center = (p + (z * z) / (2 * n)) / denom
        half = (z / denom) * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))
        return max(0.0, center - half), min(1.0, center + half)


# ---------- chance-centered Beta model (Bayesian shrinkage) ----------
def random_prior_d_minus_1(p: int, g: int, d: int) -> float:
    denom = nCr(p, d - 1)
    return 0.0 if denom == 0 else 1.0 / denom


def beta_prior_from_chance(p0: float, kappa: float = 1.0, eps: float = 1e-12) -> tuple[float, float]:
    p0 = float(np.clip(p0, eps, 1.0 - eps))
    a0 = max(eps, kappa * p0)
    b0 = max(eps, kappa * (1.0 - p0))
    return a0, b0


def compute_bayesian_credible_stats(
    k: float,
    n: int,
    p: int,
    g: int,
    d_max: int,
    alpha: float = 0.05,
    kappa: float = 1.0,
    chance_fn: Callable[[int, int, int], float] = random_prior_d_minus_1,
) -> dict:
    """
    Conjugate Beta-Binomial model with a chance-centered prior.

    Let p0 be a baseline success probability (e.g., random guessing).
    Use a Beta prior Beta(a0, b0) with mean p0 and concentration kappa:
      a0 = kappa * p0
      b0 = kappa * (1 - p0)

    Given k successes in n trials, the posterior is Beta(a0 + k, b0 + n - k).
    The posterior mean is (a0 + k) / (a0 + b0 + n).
    We report an equal-tailed (1 - alpha) posterior credible interval.
    """
    d = d_max - 1
    p0 = float(chance_fn(p, g, d))
    a0, b0 = beta_prior_from_chance(p0, kappa=kappa)
    a_post = a0 + float(k)
    b_post = b0 + float(n - k)

    try:
        from scipy.stats import beta
        lo = float(beta.ppf(alpha / 2, a_post, b_post))
        hi = float(beta.ppf(1 - alpha / 2, a_post, b_post))
    except Exception:
        mean = float(a_post / (a_post + b_post))
        var = (a_post * b_post) / (((a_post + b_post) ** 2) * (a_post + b_post + 1))
        z = 1.959963984540054
        half = z * math.sqrt(max(var, 0.0))
        lo, hi = mean - half, mean + half

    return {
        "k": float(k),
        "n": int(n),
        "p0": float(p0),
        "prior_a0": float(a0),
        "prior_b0": float(b0),
        "corrected_mean": float(a_post / (a_post + b_post)),
        "cred_lo": float(max(1e-12, min(1.0 - 1e-12, lo))),
        "cred_hi": float(max(1e-12, min(1.0 - 1e-12, hi))),
    }


# ---------- LLM gamma ----------
def compute_llm_gamma_stats_by_g(llm_records: list[dict], alpha: float = 0.05) -> dict[int, dict]:
    by_g: dict[int, dict[str, int]] = {}
    for r in llm_records:
        g = int(r.get("g", 0))
        ok = recompute_is_correct_with_boxed_fallback(r)
        c = by_g.setdefault(g, {"correct": 0, "total": 0})
        c["total"] += 1
        c["correct"] += int(ok)

    stats: dict[int, dict] = {}
    for g, c in by_g.items():
        k, n = float(c["correct"]), int(c["total"])
        mean = float(k / n) if n else 0.0
        lo, hi = jeffreys_interval(k, n, alpha=alpha)
        stats[g] = {"mean": mean, "lo": lo, "hi": hi, "n": n}
    return stats


def compute_llm_gamma_by_g(llm_records: list[dict]) -> dict[int, float]:
    stats = compute_llm_gamma_stats_by_g(llm_records, alpha=0.05)
    return {g: float(v["mean"]) for g, v in stats.items()}


def get_min_gamma_for_llm(llm_records: list[dict]) -> float:
    m = compute_llm_gamma_by_g(llm_records)
    return float(min(m.values())) if m else 0.0


def get_overall_gamma_for_llm(llm_records: list[dict]) -> float:
    if not llm_records:
        return 0.0
    correct = sum(1 for r in llm_records if recompute_is_correct_with_boxed_fallback(r))
    return float(correct / len(llm_records))


# ---------- spreadsheet helpers ----------
def _parse_target_cell(target_value) -> list[int]:
    if isinstance(target_value, (list, tuple, np.ndarray)):
        try:
            return sorted(int(x) for x in target_value)
        except Exception:
            return []
    nums = re.findall(r"\d+", str(target_value))
    try:
        return sorted(int(n) for n in nums)
    except Exception:
        return []


def _clean_spreadsheet_response(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v)
    marker = "No file chosenNo file chosen"
    return s.split(marker)[0] if marker in s else s


def _parse_last_3_numbers(response_str: str) -> list[int]:
    if not isinstance(response_str, str):
        return []
    try:
        thought = re.search(r"Thought for\s+(?:(?:\d+m\s*)?\d+s|\d+m)", response_str, flags=re.I)
        if thought:
            after = response_str[thought.end():]
            nums = re.findall(r"\d+", after)
            if len(nums) >= 3:
                return sorted(int(n) for n in nums[:3])
        nums = re.findall(r"\d+", response_str)
        if len(nums) < 3:
            return []
        return sorted(int(n) for n in nums[-3:])
    except Exception:
        return []


# ---------- prompt/response monomial alignment ----------
_MONOMIAL_BLOCK_RE = re.compile(
    r"M[_\s]*\{?(\d+)\}?\s*[:=]\s*(.*?)(?=\n\s*(?:\*?\s*\$?M[_\s]*\{?\d+\}?\s*[:=]|\Z)|\n\s*\n)",
    flags=re.IGNORECASE | re.DOTALL,
)
_MONOMIAL_VAR_RE = re.compile(r"x[_\s]*\{?(\d+)\}?", flags=re.IGNORECASE)


def _extract_monomial_map(text: str) -> dict[int, tuple[int, ...]]:
    if not text:
        return {}
    monomials: dict[int, tuple[int, ...]] = {}
    for m in _MONOMIAL_BLOCK_RE.finditer(text):
        try:
            idx = int(m.group(1))
        except Exception:
            continue
        rhs = m.group(2)
        vars_ = sorted({int(v) for v in _MONOMIAL_VAR_RE.findall(rhs)})
        if vars_:
            monomials[idx] = tuple(vars_)
    return monomials


def _monomial_signature(text: str) -> tuple[tuple[int, tuple[int, ...]], ...] | None:
    monomials = _extract_monomial_map(text)
    if not monomials:
        return None
    return tuple((k, monomials[k]) for k in sorted(monomials))


def _monomial_key(text: str, marker: str) -> tuple[tuple[tuple[int, tuple[int, ...]], ...] | None, bool]:
    # Robust check for marker: ignore whitespace and check for key phrases
    t = (text or "").replace("\n", " ").replace("\r", "")
    t = " ".join(t.split())
    has_marker = ("DO NOT RUN ANY CODE" in t) or ("DO NOT USE TOOL CALLS" in t) or (marker in (text or ""))
    return _monomial_signature(text), has_marker


def _format_monomial_summary(monomials: dict[int, tuple[int, ...]]) -> str:
    if not monomials:
        return "no monomials"
    idx = 1 if 1 in monomials else sorted(monomials)[0]
    vars_ = monomials[idx]
    return f"M{idx}=" + " & ".join(f"x{v}" for v in vars_)


def _monomials_match(prompt_map: dict[int, tuple[int, ...]], resp_map: dict[int, tuple[int, ...]]) -> bool:
    for idx, vars_ in prompt_map.items():
        if resp_map.get(idx) != vars_:
            return False
    return True


def verify_and_reorder_llm_responses(
    df: pd.DataFrame,
    columns: tuple[str, ...] = ("ChatGPT 5.2 Thinking", "Gemini"),
    prompt_col: str = "Prompt",
    keep_unmatched: bool = True,
) -> pd.DataFrame:
    if prompt_col not in df.columns:
        return df
    df = df.reset_index(drop=True)
    no_tools_marker = "DO NOT RUN ANY CODE. DO NOT USE TOOL CALLS."

    prompt_monomials_by_row: dict[int, dict[int, tuple[int, ...]]] = {}
    sig_to_row: dict[tuple[tuple[tuple[int, tuple[int, ...]], ...] | None, bool], int] = {}
    prompt_entries: list[dict] = []

    for idx, prompt in df[prompt_col].items():
        prompt_text = str(prompt)
        sig_key = _monomial_key(prompt_text, no_tools_marker)
        mon = _extract_monomial_map(str(prompt))
        prompt_monomials_by_row[int(idx)] = mon
        prompt_entries.append({
            "row_idx": int(idx),
            "is_no_tools": sig_key[1],
            "monomials": mon,
        })
        if sig_key[0] is not None and sig_key not in sig_to_row:
            sig_to_row[sig_key] = int(idx)

    for col in columns:
        if col not in df.columns:
            continue

        out_of_order = 0
        matched_rows: set[int] = set()
        new_col = [None] * len(df)

        for i, resp in df[col].items():
            if resp is None or (isinstance(resp, float) and np.isnan(resp)):
                continue
            resp_text = str(resp)
            resp_key = _monomial_key(resp_text, no_tools_marker)
            resp_map = _extract_monomial_map(resp_text)
            if resp_key[0] is None and not resp_map:
                continue
            target_idx = sig_to_row.get(resp_key)

            if target_idx is None:
                # Allow response to include extra monomials (e.g., unknown M_g)
                for entry in prompt_entries:
                    if entry["is_no_tools"] != resp_key[1]:
                        continue
                    if _monomials_match(entry["monomials"], resp_map):
                        target_idx = entry["row_idx"]
                        break
            if target_idx is None:
                continue
            if target_idx != i:
                out_of_order += 1
            matched_rows.add(target_idx)
            if new_col[target_idx] is None:
                new_col[target_idx] = resp

        print(f"{col}: {out_of_order} responses out of order; reordering.")

        missing = [sig for sig, row_idx in sig_to_row.items() if row_idx not in matched_rows]
        if missing:
            missing_info = []
            for sig in missing:
                row_idx = sig_to_row.get(sig)
                mon = prompt_monomials_by_row.get(row_idx, {})
                missing_info.append(f"row {row_idx}: {_format_monomial_summary(mon)}")
            print(f"{col}: missing {len(missing)} responses:")
            print("  " + ", ".join(missing_info))

        if keep_unmatched:
            # Fill in any gaps with original responses (unmatched/empty)
            for i, resp in df[col].items():
                if new_col[i] is None:
                    new_col[i] = resp

        df[col] = new_col

    return df


def save_cleaned_llm_prompts_file(
    input_path: str | Path = LLM_PROMPTS_FILE,
    output_path: str | Path | None = None,
    columns: tuple[str, ...] = ("ChatGPT 5.2 Thinking", "Gemini"),
) -> Path | None:
    prompts_path = resolve_output_path(input_path, REPO_ROOT)
    if not prompts_path.exists():
        return None
    df = pd.read_excel(prompts_path,sheet_name="main_old")
    df = verify_and_reorder_llm_responses(df, columns=columns, keep_unmatched=False)
    outp = Path(output_path) if output_path is not None else prompts_path.with_name("llm_g_prompts2_cleaned.xlsx")
    df.to_excel(outp, index=False)
    print(f"Saved cleaned prompts to {outp}")
    return outp


# ---------- Estimator D: k-known prefix model + LLM fitting ----------
def gamma_D_with_k(record: dict, k_known: int) -> float:
    """
    Estimator D generalized: treat the first k_known prefix monomials as known.
    Returns posterior mass on the true target under the model used in simulate.py.
    """
    p = int(record["p"])
    d = int(record["d_max"]) - 1
    g = int(record["g"])
    n = int(record["n"])

    def parse_hex(h: str):
        v = int(h, 16)
        bits = [(v >> i) & 1 for i in range(n + p)][::-1]
        return bits[:n], bits[n:]

    parsed = []
    for s in record["samples"]:
        a, v = parse_hex(s["x"])
        parsed.append((a, v, int(s["y"])))

    k_known = max(0, min(g, int(k_known)))
    known_prefix = record["prefix"][:k_known]
    candidates = list(itertools.combinations(range(p), d))
    weights = np.ones(len(candidates), dtype=float)

    def monomial(v, S):
        for i in S:
            if v[i] == 0:
                return 0
        return 1

    def prefix_eval(a, v, terms):
        y = 0
        for j, S in enumerate(terms):
            if a[j]:
                y ^= monomial(v, S)
        return y

    def prob_random_monomial_is_one(v):
        s = sum(v)
        denom = nCr(p, d)
        return 0.0 if denom == 0 else nCr(s, d) / denom

    def xor_parity_prob_one(qs):
        prod = 1.0
        for q in qs:
            prod *= (1.0 - 2.0 * q)
        return 0.5 * (1.0 - prod)

    for a, v, y in parsed:
        known = prefix_eval(a, v, known_prefix)
        q = prob_random_monomial_is_one(v)
        qs = [q if a[j] == 1 else 0.0 for j in range(k_known, g)]
        p1 = xor_parity_prob_one(qs) if qs else 0.0
        p0 = 1.0 - p1

        for idx, S in enumerate(candidates):
            base = known ^ monomial(v, S)
            need = y ^ base
            weights[idx] *= (p1 if need == 1 else p0)

    Z = float(weights.sum())
    if Z <= 0:
        return 0.0
    target_idx = candidates.index(tuple(record["target"]))
    return float(weights[target_idx] / Z)


def fit_llm_to_estimator_d(
    llm_results_dir: str | Path = LLM_RESULTS_DIR,
    experiments_file: str | Path = STORAGE_DIR / "experiments_llm_adversarial.json",
    output_csv: str | Path = PLOTS_DIR / "llm_estimator_d_fit.csv",
) -> dict:
    """
    Fit two 1-parameter capacity models for effective known-prefix length k:

      proportional: k = u * g
      constant:     k = v

    We compute predictions by looking up mean gamma_D(g, round(k)).
    Model comparison: AIC from binomial log-likelihood.
    Evidence ratio uses exp((AIC_alt - AIC_best)/2) with exponent clamped
    to avoid overflow.
    """
    llm = load_llm_results(llm_results_dir, filter_adversarial=True)
    if not llm:
        print("Warning: no LLM results; skipping fit")
        return {}

    exp_path = resolve_output_path(experiments_file, STORAGE_DIR)
    experiments = json.loads(exp_path.read_text())

    rows: list[dict] = []
    results: dict = {}

    def safe_exp(x: float) -> float:
        # exp(709) ~ 8e307 is near float max; clamp above to avoid OverflowError
        if x > 700:
            return float("inf")
        if x < -700:
            return 0.0
        return float(math.exp(x))

    for model_name, records in sorted(llm.items()):
        by_g = {}
        idx_by_g = {}
        for r in records:
            g = int(r.get("g", 0))
            ok = recompute_is_correct_with_boxed_fallback(r)
            c = by_g.setdefault(g, [0, 0])  # [correct,total]
            c[1] += 1
            c[0] += int(ok)
            ridx = r.get("record_idx")
            if isinstance(ridx, int) and 0 <= ridx < len(experiments):
                idx_by_g.setdefault(g, []).append(ridx)

        g_vals = sorted(by_g)
        obs = {g: (by_g[g][0] / by_g[g][1]) for g in g_vals if by_g[g][1] > 0}
        counts = {g: by_g[g][1] for g in g_vals}

        if not obs:
            continue

        p_val = int(records[0].get("p", 12))
        d_max = int(records[0].get("d_max", 4))
        d = d_max - 1
        rand = 1.0 / nCr(p_val, d)

        # precompute gamma_D grid
        grid = {}
        for g in g_vals:
            ids = idx_by_g.get(g, [])[:50]
            if not ids:
                continue
            for k in range(g + 1):
                vals = [gamma_D_with_k(experiments[i], k) for i in ids]
                grid[(g, k)] = float(np.mean(vals)) if vals else rand

        def pred_gamma(g: int, k_eff: float) -> float:
            k = int(round(max(0, min(g, k_eff))))
            return float(grid.get((g, k), rand))

        def loglik(k_func: Callable[[int], float]) -> float:
            ll = 0.0
            for g in g_vals:
                if g not in obs:
                    continue
                n = int(counts[g])
                k_obs = int(round(obs[g] * n))
                p = min(1 - 1e-10, max(1e-10, pred_gamma(g, k_func(g))))
                ll += k_obs * math.log(p) + (n - k_obs) * math.log(1 - p)
            return ll

        # grid search u in [0,1]
        best_u, best_ll_u = 0.5, -float("inf")
        for u in np.linspace(0.0, 1.0, 101):
            ll = loglik(lambda gg, uu=float(u): uu * gg)
            if ll > best_ll_u:
                best_ll_u, best_u = ll, float(u)

        # grid search v in [0, max_g]
        max_g = max(g_vals)
        best_v, best_ll_v = 0.0, -float("inf")
        for v in range(max_g + 1):
            ll = loglik(lambda gg, vv=float(v): vv)
            if ll > best_ll_v:
                best_ll_v, best_v = ll, float(v)

        aic_u = 2 - 2 * best_ll_u
        aic_v = 2 - 2 * best_ll_v
        delta = aic_v - aic_u  # positive favors proportional
        ratio = safe_exp(delta / 2.0)

        if delta > 2:
            winner = "proportional"
        elif delta < -2:
            winner = "constant"
        else:
            winner = "inconclusive"

        results[model_name] = {"u": best_u, "v": best_v, "aic_u": aic_u, "aic_v": aic_v, "delta_aic": delta, "evidence_ratio": ratio, "winner": winner}

        rows.append({
            "model": short_model_name(model_name, 40),
            "u (k=u*g)": f"{best_u:.4f}",
            "v (k=v)": f"{best_v:.1f}",
            "aic_proportional": f"{aic_u:.2f}",
            "aic_constant": f"{aic_v:.2f}",
            "delta_AIC (const - prop)": f"{delta:.2f}",
            "evidence_ratio exp(delta/2)": ("inf" if math.isinf(ratio) else f"{ratio:.3g}"),
            "better_model": winner,
        })

    outp = resolve_output_path(output_csv, PLOTS_DIR)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with outp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {outp}")

    return results
