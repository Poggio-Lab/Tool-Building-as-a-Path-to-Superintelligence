from __future__ import annotations

import json
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

    fig, ax = plt.subplots(figsize=(4, 3))
    max_y = y_min_clip
    all_g = set()

    # Plot estimators ABCD first
    # if filtered:
    #     xs_est = sorted({int(r["g"]) for r in filtered if int(r["g"]) > 0})
    #     by = {(g, a): [] for g in xs_est for a in "ABCD"}
    #     for r in filtered:
    #         g = int(r["g"])
    #         if g <= 0: continue
    #         for a in "ABCD":
    #             by[(g, a)].append(float(r[f"gamma_{a}"]))
    #
    #     for a in "ABCD":
    #         means, lows, highs = [], [], []
    #         for g in xs_est:
    #             vals = np.asarray(by[(g, a)], dtype=float)
    #             if vals.size == 0:
    #                 means.append(y_min_clip); lows.append(y_min_clip); highs.append(y_min_clip)
    #                 continue
    #             m = float(vals.mean())
    #             if a == "C":
    #                 lo = hi = m
    #             else:
    #                 k = float(vals.sum())
    #                 n = int(vals.size)
    #                 lo, hi = jeffreys_interval(k, n, alpha=0.05)
    #             means.append(max(m, y_min_clip))
    #             lows.append(max(lo, y_min_clip))
    #             highs.append(max(hi, y_min_clip))
    #
    #         c = COLORS.get_color(a)
    #         line, = ax.plot(xs_est, means, marker="o", markersize=3, label=f"Estimator {a}", color=c)
    #         ax.fill_between(xs_est, lows, highs, alpha=0.15, color=c)
    #         max_y = max(max_y, float(np.max(highs)))
    #         all_g.update(xs_est)

    # Filter LLMs to only include 3 small models (excluding 4B Instruct/qwen-4b instruct)
    small_llm_order = [
        ("4B Thinking", lambda s: "4b" in s and "thinking" in s),
        ("30B Thinking", lambda s: "30b" in s and "thinking" in s),
        ("30B Instruct", lambda s: "30b" in s and "instruct" in s),
    ]
    
    for label, pred in small_llm_order:
        name = next((n for n in sorted(llm.keys()) if pred(str(n).lower())), None)
        if name is None:
            continue
        recs = llm.get(name, [])
        
        c = COLORS.get_color(name)
        
        recs_p = [r for r in recs if int(r.get("p", -1)) == int(p_fixed)]
        if not recs_p: continue
        stats = compute_llm_gamma_stats_by_g(recs_p, alpha=0.05)
        xs = sorted(g for g in stats if g > 0)
        if not xs: continue
        m = np.maximum([stats[g]["mean"] for g in xs], y_min_clip)
        lo = np.maximum([stats[g]["lo"] for g in xs], y_min_clip)
        hi = np.maximum([stats[g]["hi"] for g in xs], y_min_clip)
        
        line, = ax.plot(xs, m, marker="^", markersize=3, linewidth=1, alpha=0.9, label=short_model_name(name), color=c)
        ax.fill_between(xs, lo, hi, alpha=0.1, color=c)
        max_y = max(max_y, float(np.max(hi)))
        all_g.update(xs)

    ax.axhline(ref, color="k", linestyle="--", label="Random", linewidth=0.8)
    ax.set_xlabel("Depth g")
    ax.set_ylabel("Probability Correct γ")
    ax.set_title(f"Bayesian Estimator Gamma Scaling (p={p_fixed}, d={d_max})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="upper right", frameon=True, borderpad=0.5, labelspacing=0.5)
    _set_log_xy(ax, sorted(all_g), y_min_clip * 0.8, max_y * 2.0)
    _axis_break(ax)
    _save(fig, "plot1c_gamma_vs_g_adversarial_llm.pdf")


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
def get_llm_gamma_data(excel_path: str | Path = "/Users/dkoplow/Downloads/gf2_code/llm_g_prompts_cleaned.xlsx"):
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
            ("ChatGPT 5.2 Thinking", "With Tools"),
            ("ChatGPT 5.2 Thinking", "No Tools"),
            ("Opus", "With Tools"),
            ("Opus", "No Tools"),
            ("Gemini", "With Tools"),
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
                "With Tools": {31: 1.0, 63: 1.0, 127: 0.9},
                "No Tools": {31: 0.9, 63: 0.5, 127: 0.0}
            },
            "Opus": {
                "With Tools": {31: 1.0, 63: 1.0, 127: 1.0},
                "No Tools": {31: 0.3, 63: 0.1, 127: 0.1}
            },
            "Gemini": {
                "With Tools": {31: 0.3, 63: 0.2, 127: 0.3},
                "No Tools": {31: 0.4, 63: 0.0, 127: 0.0}
            }
        }


# -------- bar: LLMs at fixed g --------
def plot_gamma_bar_llm(g_fixed: int = 31, p_fixed: int = 12, llm_prompts_file: str | Path = LLM_PROMPTS_FILE, llm_results_dir: str | Path = LLM_RESULTS_DIR):
    # Use hardcoded data instead of parsing
    llm_data = get_llm_gamma_data()
    
    frontier = ["ChatGPT 5.2 Thinking", "Opus", "Gemini"]
    counts = {m: {"With Tools": {"k": 0, "n": 0}, "No Tools": {"k": 0, "n": 0}} for m in frontier}
    
    # Convert hardcoded gamma values to counts (assuming n=10 for each)
    # This is a simplification - we use the gamma value directly as the mean
    for model in frontier:
        if model not in llm_data:
            continue
        for tool_type in ["With Tools", "No Tools"]:
            if tool_type not in llm_data[model]:
                continue
            if g_fixed not in llm_data[model][tool_type]:
                continue
            gamma_val = llm_data[model][tool_type][g_fixed]
            # Assume n=10 samples per condition (arbitrary but needed for Bayesian stats)
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

    small_order = [
        ("4B Thinking", lambda s: "4b" in s and "thinking" in s),
        ("4B Instruct", lambda s: "4b" in s and "instruct" in s),
        ("30B Thinking", lambda s: "30b" in s and "thinking" in s),
        ("30B Instruct", lambda s: "30b" in s and "instruct" in s),
    ]

    bar_labels, bar_means, bar_errs, bar_cols = [], [], [], []
    for label, pred in small_order:
        name = next((n for n in sorted(small.keys()) if pred(str(n).lower())), None)
        recs = small.get(name, [])
        rr = [r for r in recs if int(r.get("g", -1)) == g_fixed and int(r.get("p", -1)) == p_fixed]
        k = sum(1 for r in rr if recompute_is_correct_with_boxed_fallback(r))
        n = len(rr)
        if n:
            s = compute_bayesian_credible_stats(k, n, p_fixed, g_fixed, d_max, alpha=0.05)
            mean, lo, hi = s["corrected_mean"], s["cred_lo"], s["cred_hi"]
        else:
            mean, lo, hi = 0.0, 0.0, 0.0
        bar_labels.append(label)
        bar_means.append(mean)
        bar_errs.append((max(0.0, mean - lo), max(0.0, hi - mean)))
        bar_cols.append(COLORS.get_color(name or ""))

    variants = [
        ("ChatGPT (T.)", "ChatGPT 5.2 Thinking", "With Tools"),
        ("ChatGPT (N.T.)", "ChatGPT 5.2 Thinking", "No Tools"),
        ("Opus (T.)", "Opus", "With Tools"),
        ("Opus (N.T.)", "Opus", "No Tools"),
        ("Gemini (T.)", "Gemini", "With Tools"),
        ("Gemini (N.T.)", "Gemini", "No Tools"),
    ]
    hatches = []
    for label, model, pt in variants:
        k, n = counts[model][pt]["k"], counts[model][pt]["n"]
        if n:
            s = compute_bayesian_credible_stats(k, n, p_fixed, g_fixed, d_max, alpha=0.05)
            mean, lo, hi = s["corrected_mean"], s["cred_lo"], s["cred_hi"]
        else:
            mean, lo, hi = 0.0, 0.0, 0.0
        bar_labels.append(label)
        bar_means.append(mean)
        bar_errs.append((max(0.0, mean - lo), max(0.0, hi - mean)))
        bar_cols.append(COLORS.get_color(model))
        hatches.append("" if pt == "With Tools" else "xx")

    x = np.arange(len(bar_labels))
    yerr = np.asarray(bar_errs, dtype=float).T if bar_errs else None
    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(x, bar_means, width=0.5, color=bar_cols, yerr=yerr, capsize=3, error_kw={"linewidth": 0.8})
    for bar, hatch in zip(bars[len(small_order) :], hatches):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor("white")
            bar.set_linewidth(1.5)
        else:
            bar.set_edgecolor(bar.get_facecolor())
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=35, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-3, top=1.0)
    # Use FuncFormatter for scientific notation
    def format_y(x, pos):
        return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_y))
    ax.axhline(ref_val, color="k", linestyle="--", label="Random Guess", linewidth=0.8)
    ax.set_ylabel("Smallest Probability Correct γ")
    ax.set_xlabel("Model")
    ax.set_title(f"LLM Performance (p={p_fixed}, d={d_max}, g={g_fixed})")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "plot_gamma_bar_llm.pdf") 


# -------- frontier-only bars at multiple g --------
def plot_gamma_bar_llm_1d(g_values=(63, 127), p_fixed: int = 12, out: str = "plot_gamma_bar_large_llm.pdf", llm_prompts_file: str | Path = LLM_PROMPTS_FILE):
    # Use hardcoded data instead of parsing
    llm_data = get_llm_gamma_data()
    d_max = 4
    d = d_max - 1
    ref_val = 1.0 / nCr(p_fixed, d)
    y_min_clip = ref_val * 0.3

    variants = [
        ("ChatGPT (T.)", "ChatGPT 5.2 Thinking", "With Tools"),
        ("ChatGPT (N.T.)", "ChatGPT 5.2 Thinking", "No Tools"),
        ("Opus (T.)", "Opus", "With Tools"),
        ("Opus (N.T.)", "Opus", "No Tools"),
        ("Gemini (T.)", "Gemini", "With Tools"),
        ("Gemini (N.T.)", "Gemini", "No Tools"),
    ]

    stats: dict[str, dict] = {}
    for label, model, pt in variants:
        stats[label] = {"color": COLORS.get_color(model), "hatch": ("" if pt == "With Tools" else "xx"), "by_g": {}}
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

    variant_order = [v[0] for v in variants if stats.get(v[0], {}).get("by_g")]
    if not variant_order:
        return

    fig, ax = plt.subplots(figsize=(4, 3))
    n_variants = len(variant_order)
    x = np.arange(len(g_values))
    width = 0.8 / n_variants

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
                means.append(y_min_clip), el.append(0.0), eh.append(0.0)
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
            ax.text(
                b.get_x() + b.get_width() / 2,
                -0.08,
                label,
                rotation=35,
                ha="right",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=10,
            )

    random_line = ax.axhline(ref_val, color="k", linestyle="--", label="Random Guess", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["" for _ in g_values])
    ax.set_yscale("log")
    ax.set_ylim(bottom=y_min_clip * 0.8, top=2.0)
    # Use FuncFormatter to avoid math mode parsing issues
    def format_y(x, pos):
        return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_y))
    ax.set_ylabel("Bayesian Inference for $\\gamma_g$")
    # ax.set_xlabel("Depth g")
    ax.set_title("Frontier LLM $\\gamma_g$")

    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([f"g={g}" for g in g_values])
    ax_top.set_xlabel("Depth g")

    ax.legend(handles=[random_line], labels=["Random Guess"], fontsize=6, loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, out)


# -------- main with parallel plotting --------
def main(
    diag_file: str | Path = STORAGE_DIR / "results_full_diagnostic.json",
    adv_file: str | Path = STORAGE_DIR / "results_full_adversarial.json",
    sim_diag_file: str | Path = STORAGE_DIR / "results_simulation_diagnostic.json",
    sim_adv_file: str | Path = STORAGE_DIR / "results_simulation_adversarial.json",
    p_fixed: int = 12,
    llm_results_dir: str | Path = LLM_RESULTS_DIR,
    workers: int = 8,
):
    diag_path = resolve_output_path(diag_file, STORAGE_DIR)
    adv_path = resolve_output_path(adv_file, STORAGE_DIR)
    sim_diag_path = resolve_output_path(sim_diag_file, STORAGE_DIR)
    sim_adv_path = resolve_output_path(sim_adv_file, STORAGE_DIR)

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
        ("bar_large_llm", lambda: plot_gamma_bar_llm_1d(g_values=(63, 127), p_fixed=p_fixed, out="plot_gamma_bar_large_llm.pdf")),
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
    p.add_argument("--diag", type=str, default=str(STORAGE_DIR / "results_full_diagnostic.json"))
    p.add_argument("--adv", type=str, default=str(STORAGE_DIR / "results_full_adversarial.json"))
    p.add_argument("--sim-diag", type=str, default=str(STORAGE_DIR / "results_simulation_diagnostic.json"))
    p.add_argument("--sim-adv", type=str, default=str(STORAGE_DIR / "results_simulation_adversarial.json"))
    p.add_argument("--p-fixed", type=int, default=12)
    p.add_argument("--llm-results-dir", type=str, default=str(LLM_RESULTS_DIR))
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()

    main(a.diag, a.adv, a.sim_diag, a.sim_adv, a.p_fixed, a.llm_results_dir, a.workers)