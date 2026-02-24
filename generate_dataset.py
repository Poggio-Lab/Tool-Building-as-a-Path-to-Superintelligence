"""
Generate experiment datasets for gamma estimation.

Creates JSON files with experiment records containing:
- Circuit parameters (n, p, d_max, g)
- Prefix terms and target term
- Input/output samples

Usage:
    # Generate all 4 datasets (full/llm × diagnostic/adversarial):
    uv run generate_dataset.py --generate-both
    
    # Generate with specific preset:
    uv run generate_dataset.py --generate --preset llm
"""

import json
import random
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
STORAGE_DIR = REPO_ROOT / "storage"


# ============================================================
# Utilities
# ============================================================

def nCr(n, r):
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


def int_to_hex(bits):
    v = 0
    for b in bits:
        v = (v << 1) | b
    return hex(v)


def monomial_eval(v, S):
    """Returns 1 iff all v[i] == 1 for i in S."""
    for i in S:
        if v[i] == 0:
            return 0
    return 1


# ============================================================
# Circuit & Data Generation
# ============================================================

class BooleanCircuit:
    def __init__(self, n, p, d_max):
        self.n = n
        self.p = p
        self.d = d_max - 1
        self.total_vars = n + p

        if self.d > p:
            raise ValueError("Need p >= d_max-1 to enforce exact degree.")

        self.terms = [sorted(random.sample(range(p), self.d)) for _ in range(n)]

    def eval(self, a, v):
        y = 0
        for j, S in enumerate(self.terms):
            if a[j] == 0:
                continue
            y ^= monomial_eval(v, S)
        return y


def generate_samples(c, g, N=64):
    """
    Diagnostic sampling scheme:
      - disable tail: a_{g+2..n} = 0
      - enable next term: a_{g+1} = 1
      - one-time-pad prefix: a_1..a_g random
      - include diagnostic payloads: v(0)=1^p and v(i)=1^p with i-th flipped to 0
    """
    samples = []

    diagnostic = [[1] * c.p] + [
        [0 if i == j else 1 for i in range(c.p)] for j in range(c.p)
    ]

    for k in range(N):
        a_prefix = [random.randint(0, 1) for _ in range(g)]
        a_target = [1]
        a_tail = [0] * (c.n - g - 1)
        a = a_prefix + a_target + a_tail

        v = diagnostic[k] if k < len(diagnostic) else [random.randint(0, 1) for _ in range(c.p)]
        y = c.eval(a, v)

        samples.append({"x": int_to_hex(a + v), "y": y})

    return samples


def find_adversarial_hamming_weight(p, d):
    """Find Hamming weight s such that C(s,d)/C(p,d) ≈ 0.5."""
    denom = nCr(p, d)
    if denom == 0:
        return p // 2
    
    best_s = p // 2
    best_diff = float('inf')
    for s in range(d, p + 1):
        prob = nCr(s, d) / denom
        diff = abs(prob - 0.5)
        if diff < best_diff:
            best_diff = diff
            best_s = s
    return best_s


def generate_adversarial_samples(c, g, N=64):
    """
    Adversarial sampling scheme:
      - disable tail: a_{g+2..n} = 0
      - enable next term: a_{g+1} = 1
      - one-time-pad prefix: a_1..a_g random
      - payloads have Hamming weight chosen so P(monomial=1) ≈ 0.5
    """
    samples = []
    
    adv_weight = find_adversarial_hamming_weight(c.p, c.d)
    
    for k in range(N):
        a_prefix = [random.randint(0, 1) for _ in range(g)]
        a_target = [1]
        a_tail = [0] * (c.n - g - 1)
        a = a_prefix + a_target + a_tail
        
        ones_positions = random.sample(range(c.p), adv_weight)
        v = [1 if i in ones_positions else 0 for i in range(c.p)]
        
        y = c.eval(a, v)
        samples.append({"x": int_to_hex(a + v), "y": y})
    
    return samples


# ============================================================
# Preset Configurations
# ============================================================

PRESETS = {
    "full": {
        "p_values": [4, 8, 10, 12, 16, 32] ,  # 9 values: 6,8,10,12,14,16,18,20,32
        "g_values": [0, 1, 3, 7, 15, 31],  # 7 values: exponential sampling 2^n - 1
        "trials": 100,
        "samples": 32,
        "description": "Full dense grid for estimator evaluation (includes p=32 for LLM comparison)"
    },

    "simulation": {
        "p_values": [12] ,  # 9 values: 6,8,10,12,14,16,18,20,32
        "g_values": [0, 1, 3, 7, 15, 31, 63, 127],  # 7 values: exponential sampling 2^n - 1
        "trials": 2000,
        "samples": 32,
        "description": "Full dense grid for estimator evaluation (includes p=32 for LLM comparison)"
    },
    # "llm": {
    #     "p_values": [12],                    # Fixed p=32 for LLM evaluation
    #     "g_values": [0, 1, 3, 7, 15, 31],  # 7 values: exponential sampling 2^n - 1
    #     "trials": 500,
    #     "samples": 32,
    #     "description": "Fixed p=32, exponential g values for LLM evaluation"
    # },
    "llm_large": {
        "p_values": [12],                # 2 values
        "g_values": [63, 127],              # 3 values
        "trials": 10,
        "samples": 32,
        "description": "Tiny grid for quick LLM testing (~60 experiments)"
    },
    "medium": {
        "p_values": [6, 10, 14, 18],         # 4 values
        "g_values": [0, 3, 6, 9, 12, 15],    # 6 values
        "trials": 30,
        "samples": 64,
        "description": "Balanced grid for moderate evaluation"
    }
}


# ============================================================
# Dataset Generation Functions
# ============================================================

def generate_single_dataset(output_file, adversarial, trials=100, N=64, seed=0,
                             p_values=None, g_values=None):
    """Generate a single dataset (diagnostic OR adversarial) and save to JSON.
    
    Args:
        output_file: Path to save JSON
        adversarial: If True, generate adversarial samples; if False, diagnostic
        trials: Number of trials per (p, g) configuration
        N: Number of samples per trial
        seed: Random seed
        p_values: List of p values to test
        g_values: List of g values to test
    """
    random.seed(seed)
    
    if p_values is None:
        p_values = PRESETS["medium"]["p_values"]
    if g_values is None:
        g_values = PRESETS["medium"]["g_values"]
    
    mode = "adversarial" if adversarial else "diagnostic"
    sample_fn = generate_adversarial_samples if adversarial else generate_samples
    
    total_experiments = len(p_values) * len(g_values) * trials
    print(f"  Mode: {mode}")
    print("  n: varies by g (n = g + 1)")
    print(f"  Grid: {len(p_values)} p-values × {len(g_values)} g-values × {trials} trials")
    print(f"  Total experiments: {total_experiments}")
    
    records = []
    for p in p_values:
        for g in g_values:
            n = g + 1
            for _ in range(trials):
                c = BooleanCircuit(n=n, p=p, d_max=4)
                records.append({
                    "exp": "grid",
                    "n": n, "p": p, "d_max": 4, "g": g,
                    "prefix": c.terms[:g],
                    "target": c.terms[g],
                    "samples": sample_fn(c, g, N=N),
                })
    
    with open(output_file, 'w') as f:
        json.dump(records, f)
    
    print(f"  Saved {len(records)} records to {output_file}")
    return records


def generate_all_datasets(seed=0, output_dir=STORAGE_DIR):
    """Generate all 4 dataset files: full/llm × diagnostic/adversarial.
    
    Creates:
        - experiments_full_diagnostic.json
        - experiments_full_adversarial.json
        - experiments_llm_diagnostic.json
        - experiments_llm_adversarial.json
    """
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    files_created = []
    
    for preset_name in ["full", "llm", "simulation"]:
        cfg = PRESETS[preset_name]
        
        for adversarial in [False, True]:
            mode = "adversarial" if adversarial else "diagnostic"
            filename = output_dir / f"experiments_{preset_name}_{mode}.json"
            
            print("=" * 60)
            print(f"Generating: {filename.name}")
            print(f"  Preset: {preset_name} ({cfg['description']})")
            print("=" * 60)
            
            generate_single_dataset(
                str(filename),
                adversarial=adversarial,
                trials=cfg["trials"],
                N=cfg["samples"],
                seed=seed,
                p_values=cfg["p_values"],
                g_values=cfg["g_values"]
            )
            files_created.append(filename)
            print()
    
    # Print summary
    print("=" * 60)
    print("SUMMARY - All 4 datasets generated:")
    print("=" * 60)
    
    for preset_name in ["full", "llm", "simulation"]:
        cfg = PRESETS[preset_name]
        count_per_mode = len(cfg["p_values"]) * len(cfg["g_values"]) * cfg["trials"]
        print(f"\n{preset_name.upper()} ({cfg['description']}):")
        print(f"  p: {cfg['p_values']}")
        print(f"  g: {cfg['g_values']}")
        print(f"  {cfg['trials']} trials × {cfg['samples']} samples each")
        print(f"  experiments_{preset_name}_diagnostic.json  ({count_per_mode:,} experiments)")
        print(f"  experiments_{preset_name}_adversarial.json ({count_per_mode:,} experiments)")
    
    full_cfg = PRESETS["full"]
    llm_cfg = PRESETS["llm"]
    full_count = len(full_cfg["p_values"]) * len(full_cfg["g_values"]) * full_cfg["trials"]
    llm_count = len(llm_cfg["p_values"]) * len(llm_cfg["g_values"]) * llm_cfg["trials"]
    
    print(f"\nReduction factor (full vs llm): {full_count / llm_count:.1f}x per mode")
    
    return files_created


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate experiment datasets for gamma estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets available (use --preset):
  full      Dense grid (8p × 7g × 100 trials) - for estimators
  llm       Sparse grid (1p × 7g × 20 trials) - for LLM evaluation  
  llm_tiny  Minimal grid (2p × 3g × 5 trials) - for quick LLM testing
  medium    Balanced (4p × 6g × 30 trials) - default

Examples:
  # Generate all 4 datasets (full/llm × diagnostic/adversarial):
  uv run generate_dataset.py --generate-both
  
  # This creates:
  #   experiments_full_diagnostic.json   (5,600 experiments)
  #   experiments_full_adversarial.json  (5,600 experiments)
  #   experiments_llm_diagnostic.json    (140 experiments)
  #   experiments_llm_adversarial.json   (140 experiments)
  
  # Generate single dataset with preset:
  uv run generate_dataset.py --generate --preset llm --mode diagnostic
  
  # Custom configuration:
  uv run generate_dataset.py --generate --p-values "8,12" --g-values "0,5,10" --trials 20
        """
    )
    parser.add_argument("--generate", action="store_true",
                        help="Generate single dataset")
    parser.add_argument("--generate-both", action="store_true",
                        help="Generate all 4 datasets: full/llm × diagnostic/adversarial")
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()), default="medium",
                        help="Use a preset configuration (default: medium)")
    parser.add_argument("--mode", type=str, choices=["diagnostic", "adversarial"], default="adversarial",
                        help="Sampling mode (default: diagnostic)")
    parser.add_argument("--trials", type=int, default=None,
                        help="Number of trials per configuration (overrides preset)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of samples per trial (overrides preset)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: storage/experiments_{preset}_{mode}.json)")
    parser.add_argument("--p-values", type=str, default=None,
                        help="Comma-separated p values (overrides preset)")
    parser.add_argument("--g-values", type=str, default=None,
                        help="Comma-separated g values (overrides preset)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--list-presets", action="store_true",
                        help="Show available presets and exit")
    
    args = parser.parse_args()
    
    # List presets and exit
    if args.list_presets:
        print("Available presets:")
        print("-" * 60)
        for name, cfg in PRESETS.items():
            count = len(cfg["p_values"]) * len(cfg["g_values"]) * cfg["trials"]
            print(f"\n{name}:")
            print(f"  {cfg['description']}")
            print(f"  p: {cfg['p_values']}")
            print(f"  g: {cfg['g_values']}")
            print(f"  trials: {cfg['trials']}, samples: {cfg['samples']}")
            print(f"  Experiments per mode: {count:,}")
        exit(0)
    
    # Generate all 4 datasets
    if args.generate_both:
        generate_all_datasets(seed=args.seed)
        exit(0)
    
    # Generate single dataset
    if args.generate:
        preset = PRESETS[args.preset]
        
        # Get values from preset
        p_values = preset["p_values"]
        g_values = preset["g_values"]
        trials = preset["trials"]
        samples = preset["samples"]
        
        # Override with command line args
        if args.p_values:
            p_values = [int(x) for x in args.p_values.split(",")]
        if args.g_values:
            g_values = [int(x) for x in args.g_values.split(",")]
        if args.trials is not None:
            trials = args.trials
        if args.samples is not None:
            samples = args.samples
        
        # Output file
        output = args.output or f"experiments_{args.preset}_{args.mode}.json"
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = STORAGE_DIR / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        adversarial = (args.mode == "adversarial")
        
        print("=" * 60)
        print(f"Generating dataset: {output_path}")
        print(f"  Preset: {args.preset}")
        print("=" * 60)
        
        generate_single_dataset(
            str(output_path),
            adversarial=adversarial,
            trials=trials,
            N=samples,
            seed=args.seed,
            p_values=p_values,
            g_values=g_values
        )
        exit(0)
    
    # Default: generate all 4 datasets
    print("No action specified, defaulting to --generate-both")
    print()
    generate_all_datasets(seed=args.seed)
