from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as ticker

import scienceplots  # noqa: F401

from plot_utils import (
    REPO_ROOT, PLOTS_DIR, STORAGE_DIR, LLM_RESULTS_DIR, LLM_PROMPTS_FILE,
    resolve_output_path, nCr,
    short_model_name,
    aggregate_results,
    load_llm_results,
    compute_llm_gamma_stats_by_g, compute_llm_gamma_by_g,
    get_min_gamma_for_llm, get_overall_gamma_for_llm,
    recompute_is_correct_with_boxed_fallback, parse_boxed_answer,
    is_correct_after_first_tool,
    _parse_target_cell, _clean_spreadsheet_response, _parse_last_3_numbers,
    verify_and_reorder_llm_responses,
    save_cleaned_llm_prompts_file,
    jeffreys_interval,
    compute_bayesian_credible_stats,
    fit_llm_to_estimator_d,
)

# ==========================================
# 1. VISUALIZATION CONFIGURATION
# ==========================================

plt.style.use(["science", "no-latex"])
# Ensure math mode parser works correctly with no-latex

# The "Vibrant 11" - High Saturation, Distinct, No Muddy Colors
PALETTE_11 = [
    "#1F77B4",  # 1. Steel Blue (Standard)
    "#FF7F0E",  # 2. Safety Orange (Standard)
    "#2CA02C",  # 3. Forest Green (Standard)
    "#D62728",  # 4. Brick Red (Standard)
    "#9467BD",  # 5. Royal Purple (Standard)
    "#1ABC9C",  # 6. Turquoise (Replaces Brown - Clean)
    "#E377C2",  # 7. Raspberry Pink (Standard)
    "#F1C40F",  # 8. Vivid Gold (Replaces Gray - Bright)
    "#4B0082",  # 9. Deep Indigo (High Contrast Dark)
    "#17BECF",  # 10. Cyan (Standard)
    "#008080",  # 11. Rich Teal (Distinct from Green/Turquoise)
]

class ColorManager:
    """
    Ensures every model/estimator gets a unique, consistent color 
    from the PALETTE_11 list across all plots.
    """
    def __init__(self):
        self.color_map = {}
        # Pre-assign Estimators to the first 4 distinct primary colors (keep original ABC colors)
        self.color_map["A"] = PALETTE_11[0] # Blue
        self.color_map["B"] = PALETTE_11[1] # Orange
        self.color_map["C"] = PALETTE_11[3] # Red
        self.color_map["D"] = PALETTE_11[2] # Green
        
        # Pre-assign LLM model colors
        # Small LLMs: Use B, C, D colors for 4B Thinking, 4B Instruct, 30B Thinking
        self.color_map["4b thinking"] = PALETTE_11[1] # Orange (B)
        self.color_map["4b instruct"] = PALETTE_11[3] # Red (C)
        self.color_map["30b thinking"] = PALETTE_11[2] # Green (D)
        self.color_map["30b instruct"] = PALETTE_11[4] # Royal Purple (light purple)
        
        # Frontier LLMs
        self.color_map["chatgpt"] = PALETTE_11[9] # Cyan
        self.color_map["opus"] = PALETTE_11[7] # Vivid Gold (Yellow)
        self.color_map["gemini"] = PALETTE_11[6] # Raspberry Pink
        
        self.palette_index = 5 # Start assigning new models from color 6 (skip 4,6,7,10 which are used)

    def get_color(self, name: str) -> str:
        # Check for estimators first (before lowercasing)
        name_upper = name.strip().upper()
        if name_upper in ["A", "B", "C", "D"]:
            return self.color_map[name_upper]
        
        key = name.strip().lower()
        
        # Check for frontier LLM models first
        if "chatgpt" in key or ("gpt" in key and "gptq" not in key):
            return self.color_map["chatgpt"]
        if "opus" in key:
            return self.color_map["opus"]
        if "gemini" in key:
            return self.color_map["gemini"]
        
        # Check for small LLM model patterns
        if "30b" in key and "thinking" in key:
            return self.color_map["30b thinking"]
        if "30b" in key and "instruct" in key:
            return self.color_map["30b instruct"]
        if "4b" in key and "thinking" in key:
            return self.color_map["4b thinking"]
        if "4b" in key and "instruct" in key:
            return self.color_map["4b instruct"]
        
        # Return existing assignment if present
        if key in self.color_map:
            return self.color_map[key]
        
        # Assign next available color from palette
        # Skip indices that are already assigned: 0,1,2,3,4 (ABC + 30B Instruct), 6 (Gemini), 7 (Opus), 9 (ChatGPT)
        used_indices = {0, 1, 2, 3, 4, 6, 7, 9}
        while self.palette_index < len(PALETTE_11) and self.palette_index in used_indices:
            self.palette_index += 1
        
        if self.palette_index < len(PALETTE_11):
            c = PALETTE_11[self.palette_index]
            self.palette_index += 1
        else:
            # Cycle through available colors if we exceed 11
            available = [i for i in range(len(PALETTE_11)) if i not in used_indices]
            if available:
                idx = (self.palette_index - len(PALETTE_11)) % len(available)
                c = PALETTE_11[available[idx]]
            else:
                # Fallback if all colors are used
                c = PALETTE_11[self.palette_index % len(PALETTE_11)]
            self.palette_index += 1
            
        self.color_map[key] = c
        return c

# Global instance
COLORS = ColorManager()


def abbreviate_model_name(name: str) -> str:
    """Shorten model identifiers: 'Qwen/Qwen3-4B-Thinking-2507' -> '4B-Thinking'."""
    s = name.split("/")[-1] if "/" in name else name
    s = re.sub(r"Qwen3-", "", s, flags=re.IGNORECASE)
    s = re.sub(r"-?2507", "", s)
    return s


def _save(fig, out: str | Path, dpi: int = 300):
    outp = resolve_output_path(out, PLOTS_DIR)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outp}")

def _safe_set_text(obj, method_name: str, text: str):
    """Safely set text on matplotlib objects, handling ParseException with no-latex mode."""
    try:
        getattr(obj, method_name)(text)
    except Exception as e:
        if "ParseException" in str(type(e).__name__) or "Parse" in str(e):
            # Fallback: try without math mode
            text_plain = text.replace("$", "").replace("\\gamma", "gamma").replace("\\log", "log")
            getattr(obj, method_name)(text_plain)
        else:
            raise

def _axis_break(ax):
    d = 0.015
    kw = dict(transform=ax.transAxes, color="k", clip_on=False, lw=0.8)
    ax.plot((-d, +d), (0.01 - d, 0.01 + d), **kw)
    ax.plot((-d, +d), (0.03 - d, 0.03 + d), **kw)

def _set_log_xy(ax, xs, y_min, y_max):
    ax.set_xscale("log")
    ax.set_yscale("log")
    xs = [x for x in xs if x > 0]
    if xs:
        ax.set_xticks(xs)
        # Use FuncFormatter to avoid math mode parsing issues
        def format_x(x, pos):
            if x < 1:
                return f"{x:.3f}"
            return f"{int(x)}"
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(format_x))
    # Use FuncFormatter to avoid math mode parsing issues
    def format_y(x, pos):
        return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_y))
    ax.set_ylim(bottom=y_min, top=y_max)


# -------- plot 1a/1b: gamma vs g for estimators --------
def plot_gamma_vs_g_single(results: list[dict], p_fixed: int, title: str, out: str):
    fig, ax = plt.subplots(figsize=(4, 3))
    filtered = [r for r in results if int(r["p"]) == int(p_fixed)]
    if not filtered:
        ax.set_title(title)
        _save(fig, out)
        return {}

    d_max = int(filtered[0].get("d_max", 4))
    d = d_max - 1
    ref = 1.0 / nCr(p_fixed, d)
    y_min_clip = ref * 0.3

    xs = sorted({int(r["g"]) for r in filtered if int(r["g"]) > 0})
    by = {(g, a): [] for g in xs for a in "ABCD"}
    for r in filtered:
        g = int(r["g"])
        if g <= 0: continue
        for a in "ABCD":
            by[(g, a)].append(float(r[f"gamma_{a}"]))

    max_y = y_min_clip
    for a in "ABCD":
        means, lows, highs = [], [], []
        for g in xs:
            vals = np.asarray(by[(g, a)], dtype=float)
            if vals.size == 0:
                means.append(y_min_clip); lows.append(y_min_clip); highs.append(y_min_clip)
                continue
            m = float(vals.mean())
            if a == "C":
                lo = hi = m
            else:
                k = float(vals.sum())
                n = int(vals.size)
                lo, hi = jeffreys_interval(k, n, alpha=0.05)
            means.append(max(m, y_min_clip))
            lows.append(max(lo, y_min_clip))
            highs.append(max(hi, y_min_clip))

        c = COLORS.get_color(a)
        line, = ax.plot(xs, means, marker="o", markersize=3, label=f"Estimator {a}", color=c)
        ax.fill_between(xs, lows, highs, alpha=0.15, color=c)
        max_y = max(max_y, float(np.max(highs)))

    # Use Black for Random line (Sharp contrast)
    ax.axhline(ref, color="k", linestyle="--", label="Random", linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Probability Correct γ")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", frameon=True)
    _set_log_xy(ax, xs, y_min_clip * 0.8, max_y * 2.0)
    _axis_break(ax)
    _save(fig, out)

    return aggregate_results(filtered, ["g"], with_std=False)


def plot_gamma_vs_g(results_diag: list[dict], results_adv: list[dict], p_fixed: int = 12):
    agg_diag = plot_gamma_vs_g_single(
        results_diag, p_fixed,
        f"Without Adversarial (p={p_fixed}, d_max=4)",
        "plot1a_gamma_vs_g_diagnostic.pdf",
    )
    d_max = int(results_adv[0].get("d_max", 4)) if results_adv else 4
    agg_adv = plot_gamma_vs_g_single(
        results_adv, p_fixed,
        f"Bayesian Estimator Gamma Scaling (p={p_fixed}, d={d_max})",
        "plot1b_gamma_vs_g_adversarial.pdf",
    )
    return agg_diag, agg_adv


# -------- heatmaps (Viridis for maximum color) --------
def plot_heatmap_single(results: list[dict], estimator: str, title: str, out: str):
    agg = aggregate_results(results, ["g", "p"], with_std=False)
    g_vals = sorted({int(k[0]) for k in agg})
    p_vals = sorted({int(k[1]) for k in agg})
    if not g_vals or not p_vals:
        return

    mat = np.full((len(p_vals), len(g_vals)), np.nan, dtype=float)
    g_i = {g: j for j, g in enumerate(g_vals)}
    p_i = {p: i for i, p in enumerate(p_vals)}
    for (g, p), v in agg.items():
        mat[p_i[int(p)], g_i[int(g)]] = float(v[estimator])

    mat_log = np.log10(np.maximum(mat, 1e-12))
    
    fig, ax = plt.subplots(figsize=(4, 3))
    # Viridis provides the most professional "Rainbow" effect
    im = ax.imshow(mat_log, aspect="auto", origin="lower", cmap="viridis") 
    
    ax.set_xticks(range(0, len(g_vals), 2))
    ax.set_xticklabels([g_vals[i] for i in range(0, len(g_vals), 2)])
    ax.set_yticks(range(len(p_vals)))
    ax.set_yticklabels(p_vals)
    ax.set_xlabel("Prefix Length (g)")
    ax.set_ylabel("Number of Inputs (p)")
    ax.set_title(title)
    
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Use FuncFormatter for scientific notation
    def format_cb(x, pos):
        return f"{10**x:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    # Note: The heatmap is log10 data, so the colorbar ticks are log values. 
    # If we want the label to show the actual value, we need to format 10^x.
    # However, the label says "log10(γ)". So maybe just keep it as is?
    # The user said "y axis for all plots is scientific notation". 
    # For heatmaps, the Y axis is "Number of Inputs (p)". That is an integer. 
    # The colorbar is the value. 
    # Let's assume user meant the value axis (y-axis on line plots, colorbar on heatmaps?).
    # But "y axis for all plots" usually refers to the vertical axis.
    # For heatmaps, Y axis is p (integer). X axis is g (integer). 
    # So scientific notation probably doesn't apply to heatmap axes. 
    # I will leave heatmap axes alone unless "y axis" implies the colorbar values?
    # The user said "y axis for all plots".
    # I will assume this applies to 1D plots (line/bar).
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    cb.set_label("log10(γ)")
    _save(fig, out)


def plot_heatmap_combined(results: list[dict], title_prefix: str, out: str):
    """
    Create a single figure with 4 subplots (A, B, C, D) sharing the same color scale.
    Each subplot is square with a vertical colorbar on the right.
    """
    # First, compute all matrices and find global min/max for shared scale
    matrices = {}
    g_vals_all = set()
    p_vals_all = set()
    
    for e in "ABCD":
        agg = aggregate_results(results, ["g", "p"], with_std=False)
        g_vals = sorted({int(k[0]) for k in agg})
        p_vals = sorted({int(k[1]) for k in agg})
        if not g_vals or not p_vals:
            continue
        
        g_vals_all.update(g_vals)
        p_vals_all.update(p_vals)
        
        mat = np.full((len(p_vals), len(g_vals)), np.nan, dtype=float)
        g_i = {g: j for j, g in enumerate(g_vals)}
        p_i = {p: i for i, p in enumerate(p_vals)}
        for (g, p), v in agg.items():
            mat[p_i[int(p)], g_i[int(g)]] = float(v[e])
        
        matrices[e] = {
            "mat": mat,
            "g_vals": g_vals,
            "p_vals": p_vals,
            "g_i": g_i,
            "p_i": p_i,
        }
    
    if not matrices:
        return
    
    # Find global min/max for shared scale
    all_values = []
    for e in "ABCD":
        if e in matrices:
            mat = matrices[e]["mat"]
            valid = mat[~np.isnan(mat)]
            if len(valid) > 0:
                all_values.extend(valid)
    
    if not all_values:
        return
    
    global_min = np.log10(np.maximum(np.min(all_values), 1e-12))
    global_max = np.log10(np.maximum(np.max(all_values), 1e-12))
    
    # Create figure: 4 inches wide x 3 inches tall
    fig = plt.figure(figsize=(4, 3))
    
    # GridSpec: 2x2 for subplots + 1 column for colorbar
    # Width ratios: subplots get equal space, colorbar is thin
    # Added more top padding for title
    gs = fig.add_gridspec(2, 3, 
                          width_ratios=[1, 1, 0.08],
                          left=0.10, right=0.88, 
                          bottom=0.12, top=0.84,
                          wspace=0.15, hspace=0.30)
    
    axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]
    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
    
    im = None  # Store reference to last image for colorbar
    for idx, e in enumerate("ABCD"):
        ax = axes_flat[idx]
        
        if e not in matrices:
            ax.axis("off")
            continue
        
        data = matrices[e]
        mat = data["mat"]
        g_vals = data["g_vals"]
        p_vals = data["p_vals"]
        
        # Convert to log scale
        mat_log = np.log10(np.maximum(mat, 1e-12))
        
        # Use aspect="equal" for square cells, then set_box_aspect(1) for square subplot
        im = ax.imshow(mat_log, aspect="auto", origin="lower", cmap="viridis", 
                       vmin=global_min, vmax=global_max)
        ax.set_box_aspect(1)  # Force square subplot
        
        # Set ticks
        ax.set_xticks(range(0, len(g_vals), 3))
        ax.set_xticklabels([g_vals[i] for i in range(0, len(g_vals), 3)], fontsize=6)
        ax.set_yticks(range(len(p_vals)))
        ax.set_yticklabels(p_vals, fontsize=6)
        
        # Only show tick labels on outer edges
        if idx < 2:  # Top row - hide x tick labels
            ax.tick_params(labelbottom=False)
        if idx % 2 == 1:  # Right column - hide y tick labels
            ax.tick_params(labelleft=False)
        
        # Labels only on outer edges
        if idx >= 2:  # Bottom row
            ax.set_xlabel("Depth g", fontsize=8)
        if idx % 2 == 0:  # Left column
            ax.set_ylabel("p", fontsize=8)
        
        ax.set_title(r"Estimator $\gamma_" + e + r"$", fontsize=8, pad=3)
    
    # Add vertical colorbar on the right
    if im is not None:
        cbar_ax = fig.add_axes([0.90, 0.12, 0.025, 0.72])
        cb = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
        cb.set_label("γ", fontsize=8, labelpad=4)
        
        # Use FuncFormatter for colorbar
        def format_cb(x, pos):
            return f"{10**x:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_cb))
        cb.ax.tick_params(labelsize=6)
    
    fig.suptitle(title_prefix, y=0.94, fontsize=10)
    
    outp = resolve_output_path(out, PLOTS_DIR)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=300)
    plt.close(fig)
    print(f"Saved {outp}")


def plot_all_heatmaps(results_diag: list[dict], results_adv: list[dict]):
    plot_heatmap_combined(results_diag, r"Estimator $\gamma_g$ across $p$ — Diagnostic Data", "plot2_heatmap_diagnostic.pdf")
    plot_heatmap_combined(results_adv, r"Estimator $\gamma_g$ across $p$ — Adversarial Data", "plot3_heatmap_adversarial.pdf")


# -------- plot 1c: adversarial estimator A + LLMs --------
def plot_gamma_vs_g_adversarial_llm_only(results_adv: list[dict], llm_results_dir: str | Path = LLM_RESULTS_DIR, p_fixed: int | None = None):
    llm = load_llm_results(llm_results_dir, filter_adversarial=True)
    if not llm: return

    # Load no-tools records (files contain 'notools' in the name, skipped by load_llm_results)
    notools_by_model: dict[str, list[dict]] = {}
    notools_dir = resolve_output_path(llm_results_dir, REPO_ROOT)
    for f in sorted(notools_dir.glob("*notools*.jsonl")):
        with f.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                mdl = r.get("model") or ""
                notools_by_model.setdefault(mdl, []).append(r)

    if p_fixed is None:
        models_per_p: dict[int, set[str]] = {}
        for name, recs in llm.items():
            for r in recs:
                models_per_p.setdefault(int(r.get("p", 0)), set()).add(name)
        p_fixed = max(models_per_p.items(), key=lambda kv: len(kv[1]))[0]

    filtered = [r for r in results_adv if int(r.get("p", -1)) == int(p_fixed)]
    d_max = int(filtered[0].get("d_max", 4)) if filtered else int(next(iter(llm.values()))[0].get("d_max", 4))
    d = d_max - 1
    ref = 1.0 / nCr(int(p_fixed), d)
    y_min_clip = ref * 0.3

    small_llm_order = [
        ("4B Thinking", lambda s: "4b" in s and "thinking" in s),
        ("4B Instruct", lambda s: "4b" in s and "instruct" in s),
        ("30B Thinking", lambda s: "30b" in s and "thinking" in s),
        ("30B Instruct", lambda s: "30b" in s and "instruct" in s),
    ]

    def _stats_by_g(records: list[dict], correctness_fn) -> dict[int, dict]:
        by_g: dict[int, dict[str, int]] = {}
        for r in records:
            g = int(r.get("g", 0))
            c = by_g.setdefault(g, {"k": 0, "n": 0})
            c["n"] += 1
            c["k"] += int(bool(correctness_fn(r)))
        out = {}
        for g, c in by_g.items():
            k, n = float(c["k"]), int(c["n"])
            mean = k / n if n else 0.0
            lo, hi = jeffreys_interval(k, n, alpha=0.05)
            out[g] = {"mean": mean, "lo": lo, "hi": hi, "n": n}
        return out

    G_MAX = 31

    # Pre-resolve matched model names for both conditions
    matched: list[tuple[str, str, list[dict], list[dict]]] = []  # (label_name, color, multi_recs, nt_recs)
    for _label, pred in small_llm_order:
        name = next((n for n in sorted(llm.keys()) if pred(str(n).lower())), None)
        if name is None:
            continue
        recs_p = [r for r in llm.get(name, []) if int(r.get("p", -1)) == int(p_fixed)]
        if not recs_p:
            continue
        # Aggregate all notools records whose `model` field matches the predicate — some files
        # have multiple model-ID variants (e.g. 'Qwen/...' and '/tmp/.../Qwen_...') interleaved.
        recs_nt: list[dict] = []
        for nm, nrecs in notools_by_model.items():
            if pred(str(nm).lower()):
                recs_nt.extend(r for r in nrecs if int(r.get("p", -1)) == int(p_fixed))
        matched.append((abbreviate_model_name(name), COLORS.get_color(name), recs_p, recs_nt))

    def _draw_on_axis(ax, recs_picker, linestyle: str, marker: str, title_suffix: str, exclude_labels: set[str] | None = None):
        max_y = y_min_clip
        all_g: set[int] = set()
        legend_entries: list[tuple[str, str]] = []
        exclude_labels = exclude_labels or set()

        for label_name, color, recs_p, recs_nt in matched:
            if label_name in exclude_labels:
                continue
            recs = recs_picker(recs_p, recs_nt)
            if not recs:
                continue
            stats = _stats_by_g(recs, recompute_is_correct_with_boxed_fallback)
            xs = sorted(g for g in stats if 0 < g <= G_MAX)
            if not xs:
                continue
            m = np.maximum([stats[g]["mean"] for g in xs], y_min_clip)
            lo = np.maximum([stats[g]["lo"] for g in xs], y_min_clip)
            hi = np.maximum([stats[g]["hi"] for g in xs], y_min_clip)
            ax.plot(xs, m, marker=marker, markersize=3, linewidth=1, alpha=0.9,
                    linestyle=linestyle, color=color)
            ax.fill_between(xs, lo, hi, alpha=0.1, color=color)
            max_y = max(max_y, float(np.max(hi)))
            all_g.update(xs)
            legend_entries.append((label_name, color))

        ax.axhline(ref, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Depth g")
        ax.set_title(title_suffix)
        ax.grid(True, alpha=0.3)
        return legend_entries, sorted(all_g), max_y

    # Combined 2-subplot figure: No Tool Use (left) + Multi Tool Use (right), shared y axis.
    fig, (ax_notool, ax_multi) = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

    le_n, g_n, my_n = _draw_on_axis(
        ax_notool,
        recs_picker=lambda recs_p, recs_nt: recs_nt,
        linestyle=":",
        marker="v",
        title_suffix=f"Qwen3-2507 — No Tool Use (p={p_fixed}, d={d_max})",
        exclude_labels={"4B-Instruct"},
    )
    le_m, g_m, my_m = _draw_on_axis(
        ax_multi,
        recs_picker=lambda recs_p, recs_nt: recs_p,
        linestyle="-",
        marker="^",
        title_suffix=f"Qwen3-2507 — Multi Tool Use (p={p_fixed}, d={d_max})",
    )

    ax_notool.set_ylabel("Probability Correct γ")
    max_y = max(my_m, my_n)
    for ax, xs in [(ax_notool, g_n), (ax_multi, g_m)]:
        _set_log_xy(ax, xs, y_min_clip * 0.8, max_y * 2.0)
        _axis_break(ax)

    legend_entries = le_m if le_m else le_n
    handles = []
    labels = []
    for name, c in legend_entries:
        handles.append(plt.Line2D([0], [0], color=c, linewidth=2))
        labels.append(name)
    handles.append(plt.Line2D([0], [0], color="gray", linestyle="-", linewidth=1.5))
    labels.append("Multi Tool Use")
    handles.append(plt.Line2D([0], [0], color="gray", linestyle=":", linewidth=1.5))
    labels.append("No Tool Use")
    handles.append(plt.Line2D([0], [0], color="k", linestyle="--", linewidth=0.8))
    labels.append("Random Guess")
    fig.legend(handles, labels, fontsize=7, loc="lower center",
               bbox_to_anchor=(0.5, -0.05), ncol=len(labels),
               frameon=False, handletextpad=0.5, columnspacing=1.2)

    _save(fig, "plot1c_gamma_vs_g_adversarial_llm_notool.pdf")


# -------- bar: A/B/C/D at fixed g,p --------
def plot_gamma_bar_abcd(results_diag: list[dict], results_adv: list[dict], g_fixed: int = 31, p_fixed: int = 12):
    fd = [r for r in results_diag if int(r["p"]) == p_fixed and int(r["g"]) == g_fixed]
    fa = [r for r in results_adv if int(r["p"]) == p_fixed and int(r["g"]) == g_fixed]
    if not fd and not fa: return

    d_max = int((fd[0] if fd else fa[0]).get("d_max", 4))
    d = d_max - 1
    ref = 1.0 / nCr(p_fixed, d)
    y_min_clip = ref * 0.3

    def stats_for(rows, alg):
        if not rows: return 0.0, 0.0, 0.0
        vals = np.asarray([float(r[f"gamma_{alg}"]) for r in rows], dtype=float)
        m = float(vals.mean())
        if alg == "C": return m, m, m
        k = float(vals.sum()); n = int(vals.size)
        lo, hi = jeffreys_interval(k, n, alpha=0.05)
        return m, lo, hi

    ests = list("ABCD")
    diag = {e: stats_for(fd, e) for e in ests}
    adv = {e: stats_for(fa, e) for e in ests}

    fig, ax = plt.subplots(figsize=(4, 3))
    x = np.arange(len(ests))
    width = 0.35

    def clip(m, lo, hi):
        return max(m, y_min_clip), max(lo, y_min_clip), max(hi, y_min_clip)

    for i, e in enumerate(ests):
        c = COLORS.get_color(e)

        # Diagnostic: White with colored edge + hatch (High contrast, no ugly gray)
        m, lo, hi = clip(*diag[e])
        yerr = np.array([[max(0, m - lo)], [max(0, hi - m)]])
        ax.bar(x[i] - width / 2, m, width, color='white', edgecolor=c, yerr=yerr, capsize=3,
               hatch="///", linewidth=1.2)

        # Adversarial: Full Vibrant Color
        m, lo, hi = clip(*adv[e])
        yerr = np.array([[max(0, m - lo)], [max(0, hi - m)]])
        ax.bar(x[i] + width / 2, m, width, color=c, yerr=yerr, capsize=3,
               linewidth=1.2)

    ax.axhline(ref, color="k", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ests)
    ax.set_yscale("log")
    ax.set_ylim(bottom=y_min_clip * 0.8, top=2.0)
    # Use FuncFormatter for scientific notation
    def format_y(x, pos):
        return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_y))
    ax.set_ylabel("Probability Correct Step $\\gamma_g$")
    ax.set_xlabel("Estimator", labelpad=10)
    ax.set_title(f"Simulation $\\gamma_g$ (p={p_fixed}, g={g_fixed})")
    ax.grid(True, alpha=0.3, axis="y")
    
    legend_elements = [
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Without Adversarial"),
        Patch(facecolor="gray", edgecolor="none", label="With Adversarial"),
        plt.Line2D([0], [0], color="k", linestyle="--", label="Random Guess"),
    ]
    ax.legend(handles=legend_elements, fontsize=6, loc="best")
    _save(fig, "plot_gamma_bar_abcd.pdf")


# -------- Hardcoded LLM gamma data --------
def get_llm_gamma_data(excel_path: str | Path = LLM_PROMPTS_FILE):
    """
    Load LLM gamma values from Excel summary page (J3:L8).
    Data format: {model: {tool_use: {g: gamma_value}}}
    """
    try:
        # Load the specific range J3:L8 from 'summary' sheet
        # J is 10th col (idx 9), L is 12th col (idx 11) -> 9:12 (exclusive 12)
        # Row 3 is idx 2, Row 8 is idx 7 -> 2:8 (exclusive 8)
        df = pd.read_excel(excel_path, sheet_name="summary", header=None)
        data_block = df.iloc[2:8, 9:12].values
        
        # Labels corresponding to the rows in the Excel block
        row_map = [
            ("ChatGPT 5.2 Thinking", "Multi Tools"),
            ("ChatGPT 5.2 Thinking", "No Tools"),
            ("Opus", "Multi Tools"),
            ("Opus", "No Tools"),
            ("Gemini", "Multi Tools"),
            ("Gemini", "No Tools"),
        ]
        
        # Columns correspond to g values
        g_values = [31, 63, 127]
        
        result = {}
        for r_idx, (model, tool_status) in enumerate(row_map):
            if model not in result:
                result[model] = {}
            if tool_status not in result[model]:
                result[model][tool_status] = {}
                
            for c_idx, g in enumerate(g_values):
                val = data_block[r_idx, c_idx]
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = 0.0
                result[model][tool_status][g] = val
                
        print(f"Loaded LLM gamma data from {excel_path}")
        return result
        
    except Exception as e:
        print(f"Error loading LLM gamma data from {excel_path}: {e}")
        print("Falling back to hardcoded values.")
        return {
            "ChatGPT 5.2 Thinking": {
                "Multi Tools": {31: 1.0, 63: 1.0, 127: 0.9},
                "No Tools": {31: 0.9, 63: 0.5, 127: 0.0}
            },
            "Opus": {
                "Multi Tools": {31: 1.0, 63: 1.0, 127: 1.0},
                "No Tools": {31: 0.3, 63: 0.1, 127: 0.1}
            },
            "Gemini": {
                "Multi Tools": {31: 0.3, 63: 0.2, 127: 0.3},
                "No Tools": {31: 0.4, 63: 0.0, 127: 0.0}
            }
        }


# -------- bar: LLMs at fixed g --------
def _render_grouped_bars(
    *,
    groups: list[dict],   # each: {"label": str, "color": str, "variants": [{"hatch": str, "m": float, "lo": float, "hi": float}, ...]}
    hatch_legend: list[tuple[str, str]],
    ref_val: float,
    title: str,
    filename: str,
    y_min: float = 1e-3,
    y_max: float = 1.0,
    figsize: tuple[float, float] = (4, 3),
):
    """Render groups of bars; each group may have a different number of variants."""
    if not groups:
        return

    fig, ax = plt.subplots(figsize=figsize)
    bar_w = 0.5
    bar_gap_within = 0.05
    gap_between_groups = 0.6

    group_centers: list[float] = []
    cursor = 0.0
    for g in groups:
        nv = len(g["variants"])
        width = nv * bar_w + (nv - 1) * bar_gap_within
        group_centers.append(cursor + width / 2)
        for vi, v in enumerate(g["variants"]):
            xpos = cursor + vi * (bar_w + bar_gap_within) + bar_w / 2
            m = max(v["m"], y_min * 0.5)
            yerr = np.asarray([[max(0.0, v["m"] - v["lo"])], [max(0.0, v["hi"] - v["m"])]])
            bar = ax.bar(xpos, m, width=bar_w, color=g["color"], yerr=yerr, capsize=3, error_kw={"linewidth": 0.8})[0]
            h = v["hatch"]
            if h:
                bar.set_hatch(h)
                bar.set_edgecolor("white")
                bar.set_linewidth(1.5)
            else:
                bar.set_edgecolor(g["color"])
        cursor += width + gap_between_groups

    ax.set_xticks(group_centers)
    ax.set_xticklabels([g["label"] for g in groups], rotation=30, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(bottom=y_min, top=y_max)
    def format_y(x, pos):
        return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_y))
    random_line = ax.axhline(ref_val, color="k", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Smallest Probability Correct γ")
    ax.set_xlabel("Model")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    handles = []
    labels = []
    for h, desc in hatch_legend:
        if h:
            handles.append(Patch(facecolor="lightgray", edgecolor="white", hatch=h, linewidth=1.5))
        else:
            handles.append(Patch(facecolor="lightgray", edgecolor="lightgray"))
        labels.append(desc)
    handles.append(random_line)
    labels.append("Random Guess")
    ax.legend(handles=handles, labels=labels, fontsize=6, loc="upper right", frameon=True)

    _save(fig, filename)


def plot_gamma_bar_llm(g_fixed: int = 31, p_fixed: int = 12, llm_prompts_file: str | Path = LLM_PROMPTS_FILE, llm_results_dir: str | Path = LLM_RESULTS_DIR):
    # Use hardcoded data instead of parsing
    llm_data = get_llm_gamma_data()

    frontier = ["ChatGPT 5.2 Thinking", "Opus", "Gemini"]
    frontier_display = {"ChatGPT 5.2 Thinking": "GPT 5.2", "Opus": "Opus 4.5", "Gemini": "Gemini 3"}
    counts = {m: {"Multi Tools": {"k": 0, "n": 0}, "No Tools": {"k": 0, "n": 0}} for m in frontier}

    for model in frontier:
        if model not in llm_data:
            continue
        for tool_type in ["Multi Tools", "No Tools"]:
            if tool_type not in llm_data[model]:
                continue
            if g_fixed not in llm_data[model][tool_type]:
                continue
            gamma_val = llm_data[model][tool_type][g_fixed]
            n = 10
            k = int(round(gamma_val * n))
            counts[model][tool_type]["k"] = k
            counts[model][tool_type]["n"] = n

    small = load_llm_results(llm_results_dir, filter_adversarial=True)

    d_max = 4
    for recs in small.values():
        for r in recs:
            if int(r.get("g", -1)) == g_fixed and int(r.get("p", -1)) == p_fixed:
                d_max = int(r.get("d_max", 4))
                break
        else:
            continue
        break
    d = d_max - 1
    ref_val = 1.0 / nCr(p_fixed, d)

    def _stats(k: int, n: int):
        s = compute_bayesian_credible_stats(k, n, p_fixed, g_fixed, d_max, alpha=0.05)
        return s["corrected_mean"], s["cred_lo"], s["cred_hi"]

    # Load small-LLM no-tools records (separate experimental run, files contain 'notools' in name)
    notools_by_model: dict[str, list[dict]] = {}
    notools_dir = resolve_output_path(llm_results_dir, REPO_ROOT)
    for f in sorted(notools_dir.glob("*notools*.jsonl")):
        with f.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                mdl = r.get("model") or ""
                notools_by_model.setdefault(mdl, []).append(r)

    groups: list[dict] = []

    # Frontier groups: No Tool Use (xx), Multi Tool Use (solid)
    for model in frontier:
        means_variants = []
        for pt, hatch in [("No Tools", "xx"), ("Multi Tools", "")]:
            k, n = counts[model][pt]["k"], counts[model][pt]["n"]
            m, lo, hi = _stats(k, n) if n else (0.0, 0.0, 0.0)
            means_variants.append({"hatch": hatch, "m": m, "lo": lo, "hi": hi})
        groups.append({
            "label": frontier_display.get(model, model),
            "color": COLORS.get_color(model),
            "variants": means_variants,
        })

    # Small Qwen groups: No Tool Use (xx), Single Tool Use (//), Multi Tool Use (solid)
    small_order = [
        ("4B-Thinking", lambda s: "4b" in s and "thinking" in s),
        ("4B-Instruct", lambda s: "4b" in s and "instruct" in s),
        ("30B-Thinking", lambda s: "30b" in s and "thinking" in s),
        ("30B-Instruct", lambda s: "30b" in s and "instruct" in s),
    ]
    for label, pred in small_order:
        name = next((n for n in sorted(small.keys()) if pred(str(n).lower())), None)
        if name is None:
            continue
        recs = small.get(name, [])
        rr = [r for r in recs if int(r.get("g", -1)) == g_fixed and int(r.get("p", -1)) == p_fixed]
        n_rr = len(rr)

        # No Tool Use: aggregate ALL notools records whose model matches the predicate
        # (files may mix 'Qwen/...' and '/tmp/.../Qwen_...' model IDs)
        rr_zero: list[dict] = []
        for nm, nrecs in notools_by_model.items():
            if pred(str(nm).lower()):
                rr_zero.extend(r for r in nrecs
                               if int(r.get("g", -1)) == g_fixed and int(r.get("p", -1)) == p_fixed)
        k0 = sum(1 for r in rr_zero if recompute_is_correct_with_boxed_fallback(r))
        m0, lo0, hi0 = _stats(k0, len(rr_zero))

        # Single Tool Use
        k1 = sum(1 for r in rr if is_correct_after_first_tool(r))
        m1, lo1, hi1 = _stats(k1, n_rr)

        # Multi Tool Use
        k10 = sum(1 for r in rr if recompute_is_correct_with_boxed_fallback(r))
        m10, lo10, hi10 = _stats(k10, n_rr)

        groups.append({
            "label": label,
            "color": COLORS.get_color(name),
            "variants": [
                {"hatch": "xx", "m": m0, "lo": lo0, "hi": hi0},
                {"hatch": "//", "m": m1, "lo": lo1, "hi": hi1},
                {"hatch": "",   "m": m10, "lo": lo10, "hi": hi10},
            ],
        })

    _render_grouped_bars(
        groups=groups,
        hatch_legend=[("xx", "No Tool Use"), ("//", "Single Tool Use"), ("", "Multi Tool Use")],
        ref_val=ref_val,
        title=f"LLM Performance (p={p_fixed}, d={d_max}, g={g_fixed})",
        filename="plot_gamma_bar_llm.pdf",
        figsize=(6, 3),
        y_min=1e-3,
    )


# -------- frontier-only bars at multiple g --------
def plot_gamma_bar_llm_1d(g_values=(63, 127), p_fixed: int = 12, out: str = "plot_gamma_bar_large_llm.pdf", llm_prompts_file: str | Path = LLM_PROMPTS_FILE, llm_results_dir: str | Path = LLM_RESULTS_DIR):
    # Use hardcoded data instead of parsing
    llm_data = get_llm_gamma_data()
    d_max = 4
    d = d_max - 1
    ref_val = 1.0 / nCr(p_fixed, d)
    y_min_clip = ref_val * 0.3

    variants = [
        ("ChatGPT (T.)", "ChatGPT 5.2 Thinking", "Multi Tools"),
        ("ChatGPT (N.T.)", "ChatGPT 5.2 Thinking", "No Tools"),
        ("Opus (T.)", "Opus", "Multi Tools"),
        ("Opus (N.T.)", "Opus", "No Tools"),
        ("Gemini (T.)", "Gemini", "Multi Tools"),
        ("Gemini (N.T.)", "Gemini", "No Tools"),
    ]

    stats: dict[str, dict] = {}
    for label, model, pt in variants:
        stats[label] = {"color": COLORS.get_color(model), "hatch": ("" if pt == "Multi Tools" else "xx"), "by_g": {}}
        for g in g_values:
            if model not in llm_data or pt not in llm_data[model] or g not in llm_data[model][pt]:
                continue
            gamma_val = llm_data[model][pt][g]
            # Assume n=10 samples per condition (arbitrary but needed for Bayesian stats)
            n = 10
            k = int(round(gamma_val * n))
            if n:
                s = compute_bayesian_credible_stats(k, n, p_fixed, g, d_max, alpha=0.05)
                stats[label]["by_g"][g] = (s["corrected_mean"], s["cred_lo"], s["cred_hi"])

    # --- small LLMs: 1 Tool vs 10 Tools from actual records ---
    small = load_llm_results(llm_results_dir, filter_adversarial=True)
    small_order = [
        ("4B-Thinking", lambda s: "4b" in s and "thinking" in s),
        ("4B-Instruct", lambda s: "4b" in s and "instruct" in s),
        ("30B-Thinking", lambda s: "30b" in s and "thinking" in s),
        ("30B-Instruct", lambda s: "30b" in s and "instruct" in s),
    ]
    small_variant_order: list[str] = []
    for short, pred in small_order:
        name = next((n for n in sorted(small.keys()) if pred(str(n).lower())), None)
        if name is None:
            continue
        recs = small.get(name, [])
        color = COLORS.get_color(name)
        for cond_label, hatch, correctness_fn in (
            ("Single Tool Use", "//", is_correct_after_first_tool),
            ("Multi Tool Use", "", recompute_is_correct_with_boxed_fallback),
        ):
            key = f"{short} ({cond_label})"
            stats[key] = {"color": color, "hatch": hatch, "by_g": {}}
            for g in g_values:
                rr = [r for r in recs if int(r.get("g", -1)) == int(g) and int(r.get("p", -1)) == int(p_fixed)]
                n = len(rr)
                if not n:
                    continue
                k = sum(1 for r in rr if correctness_fn(r))
                s = compute_bayesian_credible_stats(k, n, p_fixed, g, d_max, alpha=0.05)
                stats[key]["by_g"][g] = (s["corrected_mean"], s["cred_lo"], s["cred_hi"])
            if stats[key]["by_g"]:
                small_variant_order.append(key)

    frontier_order = [v[0] for v in variants if stats.get(v[0], {}).get("by_g")]

    def _draw_on_ax(
        ax,
        variant_order: list[str],
        title: str,
        group_labels: list[str] | None = None,
        hatch_legend: list[tuple[str, str]] | None = None,
        show_ylabel: bool = True,
        show_legend: bool = True,
    ):
        if not variant_order:
            ax.axis("off")
            return None
        n_variants = len(variant_order)
        x = np.arange(len(g_values))
        width = 0.8 / n_variants

        group_size = (n_variants // len(group_labels)) if group_labels else 1
        group_variant_idxs: list[list[int]] = [[] for _ in range(len(group_labels))] if group_labels else []

        for i, label in enumerate(variant_order):
            color = stats[label]["color"]
            hatch = stats[label]["hatch"]
            means, el, eh = [], [], []
            for g in g_values:
                if g in stats[label]["by_g"]:
                    m, lo, hi = stats[label]["by_g"][g]
                    means.append(max(m, y_min_clip))
                    el.append(max(0.0, m - lo))
                    eh.append(max(0.0, hi - m))
                else:
                    means.append(y_min_clip); el.append(0.0); eh.append(0.0)
            off = (i - n_variants / 2 + 0.5) * width
            xpos = x + off
            yerr = np.asarray([el, eh], dtype=float)
            bars = ax.bar(xpos, means, width=width, label=label, color=color, yerr=yerr, capsize=3, error_kw={"linewidth": 0.8})
            for b in bars:
                if hatch:
                    b.set_hatch(hatch)
                    b.set_edgecolor("white")
                    b.set_linewidth(1.5)
                else:
                    b.set_edgecolor(color)
            if group_labels:
                group_variant_idxs[i // group_size].append(i)
            else:
                for b, lbl in zip(bars, [label] * len(bars)):
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        -0.08,
                        lbl,
                        rotation=35,
                        ha="right",
                        va="top",
                        transform=ax.get_xaxis_transform(),
                        fontsize=10,
                    )

        if group_labels:
            for gi, glabel in enumerate(group_labels):
                idxs = group_variant_idxs[gi]
                if not idxs:
                    continue
                mid_off = float(np.mean([(i - n_variants / 2 + 0.5) * width for i in idxs]))
                for j in range(len(g_values)):
                    ax.text(
                        x[j] + mid_off, -0.08, glabel,
                        rotation=35, ha="right", va="top",
                        transform=ax.get_xaxis_transform(), fontsize=8,
                    )

        random_line = ax.axhline(ref_val, color="k", linestyle="--", label="Random Guess", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["" for _ in g_values])
        ax.set_yscale("log")
        ax.set_ylim(bottom=y_min_clip * 0.8, top=2.0)
        def format_y(x, pos):
            return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_y))
        if show_ylabel:
            ax.set_ylabel("Bayesian Inference for $\\gamma_g$")
        ax.set_title(title)

        ax_top = ax.secondary_xaxis("top")
        ax_top.set_xticks(x)
        ax_top.set_xticklabels([f"g={g}" for g in g_values])
        ax_top.set_xlabel("Depth g")

        handles = [random_line]
        labels = ["Random Guess"]
        if hatch_legend:
            for h, desc in hatch_legend:
                if h:
                    handles.append(Patch(facecolor="lightgray", edgecolor="white", hatch=h, linewidth=1.5))
                else:
                    handles.append(Patch(facecolor="lightgray", edgecolor="lightgray"))
                labels.append(desc)
        if show_legend:
            ax.legend(handles=handles, labels=labels, fontsize=6, loc="upper right", frameon=True)
        ax.grid(True, alpha=0.3, axis="y")
        return random_line

    def _render(
        variant_order: list[str],
        title: str,
        filename: str,
        group_labels: list[str] | None = None,
        hatch_legend: list[tuple[str, str]] | None = None,
    ):
        if not variant_order:
            return
        fig, ax = plt.subplots(figsize=(4, 3))
        _draw_on_ax(ax, variant_order, title, group_labels, hatch_legend, show_ylabel=True)
        _save(fig, filename)

    frontier_models = list(dict.fromkeys(v[1] for v in variants if v[0] in frontier_order))
    frontier_display_map = {"ChatGPT 5.2 Thinking": "GPT 5.2", "Opus": "Opus 4.5", "Gemini": "Gemini 3"}
    frontier_display_labels = [frontier_display_map.get(m, m) for m in frontier_models]

    # Combined: Qwen series (left) + Frontier LLMs (right), sharing y axis.
    small_out = str(out)
    if small_out.endswith(".pdf"):
        small_out = small_out[:-4] + "_small.pdf"
    else:
        small_out = small_out + "_small"

    small_group_labels = [k for k in [v[0] for v in small_order] if any(key.startswith(f"{k} (") for key in small_variant_order)]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    _draw_on_ax(
        ax_left,
        small_variant_order,
        "Qwen3-2507 Series",
        group_labels=small_group_labels,
        hatch_legend=[("", "Multi Tool Use"), ("//", "Single Tool Use")],
        show_ylabel=True,
        show_legend=False,
    )
    rl = _draw_on_ax(
        ax_right,
        frontier_order,
        "Frontier LLMs",
        group_labels=frontier_display_labels,
        hatch_legend=[("", "Multi Tool Use"), ("xx", "No Tool Use")],
        show_ylabel=False,
        show_legend=False,
    )

    leg_handles = [
        Patch(facecolor="lightgray", edgecolor="lightgray"),
        Patch(facecolor="lightgray", edgecolor="white", hatch="//", linewidth=1.5),
        Patch(facecolor="lightgray", edgecolor="white", hatch="xx", linewidth=1.5),
        plt.Line2D([0], [0], color="k", linestyle="--", linewidth=0.8),
    ]
    leg_labels = ["Multi Tool Use", "Single Tool Use", "No Tool Use", "Random Guess"]
    fig.legend(leg_handles, leg_labels, fontsize=7, loc="lower center",
               bbox_to_anchor=(0.5, -0.05), ncol=len(leg_labels),
               frameon=False, handletextpad=0.5, columnspacing=1.2)
    _save(fig, small_out)


# -------- export: tool-usage success-rate table (CSV) --------
def export_tool_usage_table(
    output_csv: str | Path = "tool_usage_table.csv",
    p_fixed: int = 12,
    g_values: tuple[int, ...] = (63, 127),
    llm_results_dir: str | Path = LLM_RESULTS_DIR,
):
    """CSV with success rates for No / Single / Multi tool use across small + frontier LLMs."""
    import csv

    # --- small LLMs: tool-use (multi + single) ---
    small = load_llm_results(llm_results_dir, filter_adversarial=True)

    # --- small LLMs: no-tools ---
    notools_by_model: dict[str, list[dict]] = {}
    notools_dir = resolve_output_path(llm_results_dir, REPO_ROOT)
    for f in sorted(notools_dir.glob("*notools*.jsonl")):
        with f.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                notools_by_model.setdefault(r.get("model") or "", []).append(r)

    small_order = [
        ("Qwen3-4B-Thinking",       lambda s: "4b" in s and "thinking" in s),
        ("Qwen3-4B-Instruct",       lambda s: "4b" in s and "instruct" in s),
        ("Qwen3-30B-A3B-Thinking",  lambda s: "30b" in s and "thinking" in s),
        ("Qwen3-30B-A3B-Instruct",  lambda s: "30b" in s and "instruct" in s),
    ]

    # --- frontier hardcoded data (only Multi-Tool "With Tools" and "No Tools") ---
    llm_data = get_llm_gamma_data()
    frontier_order = [
        ("ChatGPT-5.2-Thinking", "ChatGPT 5.2 Thinking"),
        ("Opus",                 "Opus"),
        ("Gemini",               "Gemini"),
    ]

    def _rate(k: int, n: int) -> str:
        return f"{k/n:.4f}" if n else ""

    rows: list[dict] = []

    # Small LLMs rows
    for label, pred in small_order:
        name = next((n for n in sorted(small.keys()) if pred(str(n).lower())), None)
        recs = small.get(name, []) if name else []
        for g in g_values:
            rr = [r for r in recs if int(r.get("g", -1)) == int(g) and int(r.get("p", -1)) == int(p_fixed)]
            n_rr = len(rr)
            k_single = sum(1 for r in rr if is_correct_after_first_tool(r))
            k_multi  = sum(1 for r in rr if recompute_is_correct_with_boxed_fallback(r))

            rr_nt: list[dict] = []
            for nm, nrecs in notools_by_model.items():
                if pred(str(nm).lower()):
                    rr_nt.extend(r for r in nrecs
                                 if int(r.get("g", -1)) == int(g) and int(r.get("p", -1)) == int(p_fixed))
            n_nt = len(rr_nt)
            k_nt = sum(1 for r in rr_nt if recompute_is_correct_with_boxed_fallback(r))

            rows.append({
                "model_class": "small",
                "model":       label,
                "g":           g,
                "p":           p_fixed,
                "no_tool_rate":     _rate(k_nt, n_nt),
                "no_tool_n":        n_nt,
                "single_tool_rate": _rate(k_single, n_rr),
                "single_tool_n":    n_rr,
                "multi_tool_rate":  _rate(k_multi, n_rr),
                "multi_tool_n":     n_rr,
            })

    # Frontier rows (no single-tool data for frontier)
    for label, data_key in frontier_order:
        if data_key not in llm_data:
            continue
        for g in g_values:
            nt = llm_data[data_key].get("No Tools", {}).get(g)
            mt = (llm_data[data_key].get("Multi Tools") or llm_data[data_key].get("With Tools") or {}).get(g)
            rows.append({
                "model_class": "frontier",
                "model":       label,
                "g":           g,
                "p":           p_fixed,
                "no_tool_rate":     f"{nt:.4f}" if nt is not None else "",
                "no_tool_n":        "",
                "single_tool_rate": "",
                "single_tool_n":    "",
                "multi_tool_rate":  f"{mt:.4f}" if mt is not None else "",
                "multi_tool_n":     "",
            })

    out_path = resolve_output_path(output_csv, PLOTS_DIR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_class", "model", "g", "p",
        "no_tool_rate", "no_tool_n",
        "single_tool_rate", "single_tool_n",
        "multi_tool_rate", "multi_tool_n",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_path} ({len(rows)} rows)")
    return out_path


# -------- main with parallel plotting --------
def main(
    diag_file: str | Path = LLM_RESULTS_DIR / "simulation_full_diagnostic.json",
    adv_file: str | Path = LLM_RESULTS_DIR / "simulation_full_adversarial.json",
    sim_diag_file: str | Path = LLM_RESULTS_DIR / "simulation_simulation_diagnostic.json",
    sim_adv_file: str | Path = LLM_RESULTS_DIR / "simulation_simulation_adversarial.json",
    p_fixed: int = 12,
    llm_results_dir: str | Path = LLM_RESULTS_DIR,
    workers: int = 8,
):
    diag_path = resolve_output_path(diag_file, LLM_RESULTS_DIR)
    adv_path = resolve_output_path(adv_file, LLM_RESULTS_DIR)
    sim_diag_path = resolve_output_path(sim_diag_file, LLM_RESULTS_DIR)
    sim_adv_path = resolve_output_path(sim_adv_file, LLM_RESULTS_DIR)

    results_diag = json.loads(diag_path.read_text())
    results_adv = json.loads(adv_path.read_text())
    sim_diag = json.loads(sim_diag_path.read_text())
    sim_adv = json.loads(sim_adv_path.read_text())

    agg_diag, agg_adv = plot_gamma_vs_g(sim_diag, sim_adv, p_fixed=p_fixed)

    tasks = [
        ("heatmaps", lambda: plot_all_heatmaps(results_diag, results_adv)),
        ("bar_abcd", lambda: plot_gamma_bar_abcd(sim_diag, sim_adv, g_fixed=31, p_fixed=12)),
        ("bar_llm", lambda: plot_gamma_bar_llm(g_fixed=31, p_fixed=12, llm_results_dir=llm_results_dir)),
        ("plot1c", lambda: plot_gamma_vs_g_adversarial_llm_only(results_adv, llm_results_dir=llm_results_dir, p_fixed=None)),
        ("bar_large_llm", lambda: plot_gamma_bar_llm_1d(g_values=(63, 127), p_fixed=p_fixed, out="plot_gamma_bar_large_llm.pdf", llm_results_dir=llm_results_dir)),
        ("fit", lambda: fit_llm_to_estimator_d(llm_results_dir=llm_results_dir)),
    ]

    print(f"Running {len(tasks)} tasks with {workers} threads...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn): name for name, fn in tasks}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                fut.result()
            except Exception as e:
                import traceback
                error_msg = str(e)
                if "ParseException" in error_msg or "Parse" in str(type(e).__name__):
                    print(f"Task failed: {name}: ParseException (likely due to no-latex mode conflict)")
                    print(f"  Error details: {error_msg[:200]}")
                else:
                    print(f"Task failed: {name}: {e}")
                    traceback.print_exc()

    print("Done.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--diag", type=str, default=str(LLM_RESULTS_DIR / "simulation_full_diagnostic.json"))
    p.add_argument("--adv", type=str, default=str(LLM_RESULTS_DIR / "simulation_full_adversarial.json"))
    p.add_argument("--sim-diag", type=str, default=str(LLM_RESULTS_DIR / "simulation_simulation_diagnostic.json"))
    p.add_argument("--sim-adv", type=str, default=str(LLM_RESULTS_DIR / "simulation_simulation_adversarial.json"))
    p.add_argument("--p-fixed", type=int, default=12)
    p.add_argument("--llm-results-dir", type=str, default=str(LLM_RESULTS_DIR))
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--table-only", action="store_true",
                   help="Only export the tool-usage CSV; skip all plots.")
    p.add_argument("--table-out", type=str, default="tool_usage_table.csv",
                   help="Output path for the tool-usage CSV (relative to plots/ unless absolute).")
    p.add_argument("--table-g", type=str, default="63,127",
                   help="Comma-separated g values for the table (default: 63,127).")
    a = p.parse_args()

    if a.table_only:
        g_tuple = tuple(int(x) for x in a.table_g.split(",") if x.strip())
        export_tool_usage_table(
            output_csv=a.table_out,
            p_fixed=a.p_fixed,
            g_values=g_tuple,
            llm_results_dir=a.llm_results_dir,
        )
    else:
        main(a.diag, a.adv, a.sim_diag, a.sim_adv, a.p_fixed, a.llm_results_dir, a.workers)