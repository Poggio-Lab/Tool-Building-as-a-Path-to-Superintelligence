"""
Simulate estimators A, B, C, D on experiment datasets.

Runs the Bayesian estimators on the full datasets and saves gamma results.

Usage:
    # Run on all full datasets:
    uv run simulate.py
    
    # Run on specific files:
    uv run simulate.py --input experiments_full_diagnostic.json --output simulation_full_diagnostic.json
"""

import json
import math
import itertools
import argparse
from pathlib import Path
import numpy as np
from numba import njit

REPO_ROOT = Path(__file__).resolve().parent
STORAGE_DIR = REPO_ROOT / "storage"
LLM_RESULTS_DIR = REPO_ROOT / "all_results"


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return base_dir / path


# ============================================================
# Utilities
# ============================================================

def nCr(n, r):
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


def parse_hex(h, n, p):
    v = int(h, 16)
    bits = [(v >> i) & 1 for i in range(n + p)][::-1]
    return bits[:n], bits[n:]


def all_sets(p, d):
    return list(itertools.combinations(range(p), d))


def monomial_eval(v, S):
    """Returns 1 iff all v[i] == 1 for i in S."""
    for i in S:
        if v[i] == 0:
            return 0
    return 1


@njit
def monomial_eval_numba(v, S):
    """Returns 1 iff all v[i] == 1 for i in S. Numba-optimized version."""
    for i in S:
        if v[i] == 0:
            return 0
    return 1


def prefix_eval(a, v, prefix_terms):
    """Evaluate prefix terms contribution."""
    y = 0
    for j, S in enumerate(prefix_terms):
        if a[j] == 0:
            continue
        y ^= monomial_eval(v, S)
    return y


@njit
def prefix_eval_numba(a, v, prefix_terms_flat, prefix_terms_lengths, prefix_terms_offsets):
    """Evaluate prefix terms contribution. Numba-optimized version."""
    y = 0
    for j in range(len(prefix_terms_lengths)):
        if a[j] == 0:
            continue
        offset = prefix_terms_offsets[j]
        length = prefix_terms_lengths[j]
        S = prefix_terms_flat[offset:offset + length]
        y ^= monomial_eval_numba(v, S)
    return y


@njit
def xor_parity_prob_one_numba(qs):
    """
    If Z_i ~ Bernoulli(q_i) independent, then XOR_i Z_i has:
      P(XOR=1) = 0.5 * (1 - Π_i (1 - 2 q_i))
    Numba-optimized version.
    """
    prod = 1.0
    for q in qs:
        prod *= (1.0 - 2.0 * q)
    return 0.5 * (1.0 - prod)


def xor_parity_prob_one(qs):
    """
    If Z_i ~ Bernoulli(q_i) independent, then XOR_i Z_i has:
      P(XOR=1) = 0.5 * (1 - Π_i (1 - 2 q_i))
    """
    prod = 1.0
    for q in qs:
        prod *= (1.0 - 2.0 * q)
    return 0.5 * (1.0 - prod)


@njit
def prob_random_payload_monomial_is_one_numba(v, p, d, ncr_cache):
    """
    A uniformly random d-subset S ⊆ [p] gives monomial ∏_{i∈S} v_i.
    This equals 1 iff S is entirely inside the set of ones.
    If s = #ones in v, then P(monomial=1) = C(s,d)/C(p,d).
    Numba-optimized version with precomputed nCr values.
    """
    s = 0
    for i in range(len(v)):
        s += v[i]
    
    # Access precomputed nCr values
    denom = ncr_cache[p, d]
    if denom == 0:
        return 0.0
    
    numer = ncr_cache[s, d] if s < len(ncr_cache) and d < len(ncr_cache[0]) else 0
    return numer / denom


def prob_random_payload_monomial_is_one(v, p, d):
    """
    A uniformly random d-subset S ⊆ [p] gives monomial ∏_{i∈S} v_i.
    This equals 1 iff S is entirely inside the set of ones.
    If s = #ones in v, then P(monomial=1) = C(s,d)/C(p,d).
    """
    s = sum(v)
    denom = nCr(p, d)
    if denom == 0:
        return 0.0
    return nCr(s, d) / denom


def get_parsed(record):
    """Parse samples from hex format."""
    n, p = record["n"], record["p"]
    parsed = []
    for s in record["samples"]:
        a, v = parse_hex(s["x"], n, p)
        parsed.append((a, v, s["y"]))
    return parsed


# ============================================================
# Helper functions for numba optimization
# ============================================================

def flatten_prefix_terms(prefix_terms):
    """Flatten list of tuples into arrays for numba compatibility."""
    if not prefix_terms:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    
    flat = []
    lengths = []
    offsets = []
    offset = 0
    
    for term in prefix_terms:
        flat.extend(term)
        lengths.append(len(term))
        offsets.append(offset)
        offset += len(term)
    
    return (np.array(flat, dtype=np.int64), 
            np.array(lengths, dtype=np.int64), 
            np.array(offsets, dtype=np.int64))


def create_ncr_cache(max_n):
    """Precompute nCr values for fast lookup."""
    cache = np.zeros((max_n + 1, max_n + 1), dtype=np.float64)
    for n in range(max_n + 1):
        for r in range(n + 1):
            cache[n, r] = nCr(n, r)
    return cache


@njit
def gamma_A_numba(parsed_a, parsed_v, parsed_y, candidates_flat, candidates_lengths, 
                   candidates_offsets, prefix_terms_flat, prefix_terms_lengths, 
                   prefix_terms_offsets, target_idx):
    """
    Estimator A: Full prefix + data (numba-optimized).
    """
    n_candidates = len(candidates_lengths)
    weights = np.ones(n_candidates, dtype=np.float64)
    
    for i in range(len(parsed_y)):
        a = parsed_a[i]
        v = parsed_v[i]
        y = parsed_y[i]
        
        # Compute residual
        r = y ^ prefix_eval_numba(a, v, prefix_terms_flat, prefix_terms_lengths, prefix_terms_offsets)
        
        # Update weights for each candidate
        for cand_idx in range(n_candidates):
            offset = candidates_offsets[cand_idx]
            length = candidates_lengths[cand_idx]
            S = candidates_flat[offset:offset + length]
            t = monomial_eval_numba(v, S)
            
            if r != t:
                weights[cand_idx] = 0.0
    
    Z = np.sum(weights)
    if Z == 0:
        return 0.0
    return weights[target_idx] / Z


@njit
def gamma_B_numba(parsed_a, parsed_v, parsed_y, candidates_flat, candidates_lengths, 
                   candidates_offsets, target_idx, p, d, g, ncr_cache):
    """
    Estimator B: Data only, no prefix knowledge (numba-optimized).
    """
    n_candidates = len(candidates_lengths)
    weights = np.ones(n_candidates, dtype=np.float64)
    
    for i in range(len(parsed_y)):
        a = parsed_a[i]
        v = parsed_v[i]
        y = parsed_y[i]
        
        q = prob_random_payload_monomial_is_one_numba(v, p, d, ncr_cache)
        
        # Build qs array
        qs = np.zeros(g, dtype=np.float64)
        for j in range(g):
            qs[j] = q if a[j] == 1 else 0.0
        
        p_noise_1 = xor_parity_prob_one_numba(qs)
        p_noise_0 = 1.0 - p_noise_1
        
        # Update weights for each candidate
        for cand_idx in range(n_candidates):
            offset = candidates_offsets[cand_idx]
            length = candidates_lengths[cand_idx]
            S = candidates_flat[offset:offset + length]
            t = monomial_eval_numba(v, S)
            base = t
            need = y ^ base
            like = p_noise_1 if need == 1 else p_noise_0
            weights[cand_idx] *= like
    
    Z = np.sum(weights)
    if Z == 0:
        return 0.0
    return weights[target_idx] / Z


@njit
def gamma_D_numba(parsed_a, parsed_v, parsed_y, candidates_flat, candidates_lengths, 
                   candidates_offsets, known_prefix_flat, known_prefix_lengths, 
                   known_prefix_offsets, target_idx, p, d, g, k, ncr_cache):
    """
    Estimator D: Partial prefix only (numba-optimized).
    """
    n_candidates = len(candidates_lengths)
    weights = np.ones(n_candidates, dtype=np.float64)
    
    for i in range(len(parsed_y)):
        a = parsed_a[i]
        v = parsed_v[i]
        y = parsed_y[i]
        
        known = prefix_eval_numba(a, v, known_prefix_flat, known_prefix_lengths, known_prefix_offsets)
        
        q = prob_random_payload_monomial_is_one_numba(v, p, d, ncr_cache)
        
        # Build qs array for unknown terms
        n_unknown = g - k
        qs = np.zeros(n_unknown, dtype=np.float64)
        for j in range(n_unknown):
            qs[j] = q if a[k + j] == 1 else 0.0
        
        p_noise_1 = xor_parity_prob_one_numba(qs)
        p_noise_0 = 1.0 - p_noise_1
        
        # Update weights for each candidate
        for cand_idx in range(n_candidates):
            offset = candidates_offsets[cand_idx]
            length = candidates_lengths[cand_idx]
            S = candidates_flat[offset:offset + length]
            t = monomial_eval_numba(v, S)
            base = known ^ t
            need = y ^ base
            like = p_noise_1 if need == 1 else p_noise_0
            weights[cand_idx] *= like
    
    Z = np.sum(weights)
    if Z == 0:
        return 0.0
    return weights[target_idx] / Z


# ============================================================
# Estimators returning gamma (posterior mass on truth)
# ============================================================

def gamma_A(record):
    """
    Estimator A: Full prefix + data.
    
    Has access to all prefix terms, so can compute exact residuals.
    Deterministic residual reveals target term exactly, so gamma=1
    when sufficient diagnostic payloads are available.
    """
    p = record["p"]
    d = record["d_max"] - 1
    g = record["g"]
    prefix = record["prefix"]
    parsed = get_parsed(record)

    candidates = all_sets(p, d)
    
    # Prepare data for numba
    parsed_a = np.array([a for a, v, y in parsed], dtype=np.int32)
    parsed_v = np.array([v for a, v, y in parsed], dtype=np.int32)
    parsed_y = np.array([y for a, v, y in parsed], dtype=np.int32)
    
    # Flatten candidates
    candidates_flat, candidates_lengths, candidates_offsets = flatten_prefix_terms(candidates)
    
    # Flatten prefix terms
    prefix_flat, prefix_lengths, prefix_offsets = flatten_prefix_terms(prefix)
    
    # Find target index
    target_tuple = tuple(record["target"])
    target_idx = candidates.index(target_tuple)
    
    # Call numba-optimized version
    return gamma_A_numba(parsed_a, parsed_v, parsed_y, candidates_flat, candidates_lengths,
                         candidates_offsets, prefix_flat, prefix_lengths, prefix_offsets, target_idx)


def gamma_B(record):
    """
    Estimator B: Data only, no prefix knowledge.
    
    Marginalizes prefix contribution as unknown random ANF terms.
    Each prefix term contributes Bernoulli noise based on payload Hamming weight.
    """
    p = record["p"]
    d = record["d_max"] - 1
    g = record["g"]
    parsed = get_parsed(record)

    candidates = all_sets(p, d)
    
    # Prepare data for numba
    parsed_a = np.array([a for a, v, y in parsed], dtype=np.int32)
    parsed_v = np.array([v for a, v, y in parsed], dtype=np.int32)
    parsed_y = np.array([y for a, v, y in parsed], dtype=np.int32)
    
    # Flatten candidates
    candidates_flat, candidates_lengths, candidates_offsets = flatten_prefix_terms(candidates)
    
    # Find target index
    target_tuple = tuple(record["target"])
    target_idx = candidates.index(target_tuple)
    
    # Create nCr cache
    max_n = max(p, g) + 1
    ncr_cache = create_ncr_cache(max_n)
    
    # Call numba-optimized version
    return gamma_B_numba(parsed_a, parsed_v, parsed_y, candidates_flat, candidates_lengths,
                         candidates_offsets, target_idx, p, d, g, ncr_cache)


def gamma_C(record):
    """
    Estimator C: Prefix only, no data.
    
    Must guess among C(p,d) candidates uniformly.
    This is the random baseline.
    """
    p = record["p"]
    d = record["d_max"] - 1
    denom = nCr(p, d)
    return 1.0 / denom if denom > 0 else 0.0


def gamma_D(record):
    """
    Estimator D: Partial prefix only.
    
    Has access to first half of prefix terms (k = g//2).
    Remaining prefix terms are marginalized as noise.
    """
    p = record["p"]
    d = record["d_max"] - 1
    g = record["g"]
    parsed = get_parsed(record)

    k = g // 2
    known_prefix = record["prefix"][:k]

    candidates = all_sets(p, d)
    
    # Prepare data for numba
    parsed_a = np.array([a for a, v, y in parsed], dtype=np.int32)
    parsed_v = np.array([v for a, v, y in parsed], dtype=np.int32)
    parsed_y = np.array([y for a, v, y in parsed], dtype=np.int32)
    
    # Flatten candidates
    candidates_flat, candidates_lengths, candidates_offsets = flatten_prefix_terms(candidates)
    
    # Flatten known prefix terms
    known_prefix_flat, known_prefix_lengths, known_prefix_offsets = flatten_prefix_terms(known_prefix)
    
    # Find target index
    target_tuple = tuple(record["target"])
    target_idx = candidates.index(target_tuple)
    
    # Create nCr cache
    max_n = max(p, g) + 1
    ncr_cache = create_ncr_cache(max_n)
    
    # Call numba-optimized version
    return gamma_D_numba(parsed_a, parsed_v, parsed_y, candidates_flat, candidates_lengths,
                         candidates_offsets, known_prefix_flat, known_prefix_lengths, 
                         known_prefix_offsets, target_idx, p, d, g, k, ncr_cache)


# ============================================================
# Simulation Functions
# ============================================================

def calculate_gammas(records, verbose=True):
    """Calculate gamma for all estimators on loaded records.
    
    Args:
        records: List of experiment records
        verbose: Print progress updates
        
    Returns:
        List of result dicts with gamma values for each estimator
    """
    results = []
    total = len(records)
    
    for i, r in enumerate(records):
        if verbose and i % 100 == 0:
            print(f"  Processing record {i}/{total} ({100*i/total:.1f}%)")
        
        # gamma_C is analytically 1/C(p,d) - no need to compute from data
        p, d = r["p"], r["d_max"] - 1
        gamma_c = 1.0 / nCr(p, d) if nCr(p, d) > 0 else 0.0
        
        result = {
            "p": r["p"],
            "g": r["g"],
            "d_max": r["d_max"],
            "gamma_A": gamma_A(r),
            "gamma_B": gamma_B(r),
            "gamma_C": gamma_c,  # Analytical value
            "gamma_D": gamma_D(r),
        }
        results.append(result)
    
    if verbose:
        print(f"  Completed {total} records")
    
    return results


def simulate_single_file(input_file, output_file):
    """Run estimators on a single experiment file.
    
    Args:
        input_file: Path to experiments JSON (list of records)
        output_file: Path to save results JSON
    """
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        records = json.load(f)
    
    print(f"  Loaded {len(records)} records")
    print(f"Running estimators A, B, C, D...")
    
    results = calculate_gammas(records)
    
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f)
    
    print(f"  Done!")
    return results


def simulate_all_full_datasets(input_dir=STORAGE_DIR, output_dir=LLM_RESULTS_DIR):
    """Run estimators on all full datasets.

    Expects:
        - experiments_full_diagnostic.json
        - experiments_full_adversarial.json
        - experiments_simulation_diagnostic.json
        - experiments_simulation_adversarial.json

    Creates (output names prefixed with 'simulation_'):
        - simulation_full_diagnostic.json
        - simulation_full_adversarial.json
        - simulation_simulation_diagnostic.json
        - simulation_simulation_adversarial.json
    """
    input_dir = resolve_path(str(input_dir), REPO_ROOT)
    output_dir = resolve_path(str(output_dir), REPO_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("experiments_full_diagnostic.json",        "simulation_full_diagnostic.json"),
        ("experiments_full_adversarial.json",       "simulation_full_adversarial.json"),
        ("experiments_simulation_adversarial.json", "simulation_simulation_adversarial.json"),
        ("experiments_simulation_diagnostic.json",  "simulation_simulation_diagnostic.json"),
    ]
    
    for input_name, output_name in datasets:
        input_path = input_dir / input_name
        output_path = output_dir / output_name
        
        if not input_path.exists():
            print(f"WARNING: {input_path} not found, skipping...")
            continue
        
        print("=" * 60)
        print(f"Processing: {input_name}")
        print("=" * 60)
        
        simulate_single_file(str(input_path), str(output_path))
        print()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Results saved:")
    for _, output_name in datasets:
        output_path = output_dir / output_name
        if output_path.exists():
            print(f"  - {output_name}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run A/B/C/D estimators on experiment datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on all full datasets (default):
  uv run simulate.py
  
  # Run on a specific file:
  uv run simulate.py --input experiments_full_diagnostic.json --output simulation_full_diagnostic.json
  
  # Specify input/output directories:
  uv run simulate.py --input-dir ./data --output-dir ./results
        """
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Input experiments JSON file (single file mode)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output results JSON file (single file mode)")
    parser.add_argument("--input-dir", type=str, default=str(STORAGE_DIR),
                        help="Directory containing experiment files (batch mode)")
    parser.add_argument("--output-dir", type=str, default=str(LLM_RESULTS_DIR),
                        help="Directory to save result files (batch mode)")
    parser.add_argument("--all", action="store_true",
                        help="Process all full datasets (default behavior)")
    
    args = parser.parse_args()
    
    # Single file mode
    if args.input and args.output:
        input_path = resolve_path(args.input, STORAGE_DIR)
        output_path = resolve_path(args.output, LLM_RESULTS_DIR)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        simulate_single_file(str(input_path), str(output_path))
    else:
        # Batch mode - process all full datasets
        simulate_all_full_datasets(
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )
