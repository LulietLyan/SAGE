#!/usr/bin/env python3
"""Nature / Science style primitives for quantitative experiment figures.

Design contract (Nature-family quantitative grid):
- Final width ~89 mm (single) or ~183 mm (double); font 7–8 pt; Arial.
- Left+bottom spines only; no top/right; no default grid.
- Size sweeps use log2 x so power-of-two labels never collide.
- Hero method in deep blue; baselines in restrained neutrals/teals.
- Legends inside axes when space allows; frameless.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

METHODS = ("generic", "sme-fixed", "ours", "pytorch")
LABELS = {
    "generic": "TritonCPU",
    "sme-fixed": "Fixed SME",
    "ours": "Ours",
    "pytorch": "PyTorch",
}

# Suite v2 palette — richer but still print-safe / colorblind-friendly.
# Main triad: Ours (navy), TritonCPU (slate), PyTorch (coral).
# Ablation Fixed-SME: teal; gains/losses: emerald / rose.
COLORS = {
    "generic": "#5B6B7A",
    "sme-fixed": "#1B9E77",
    "ours": "#08519C",
    "pytorch": "#D95F02",
}
MARKERS = {"generic": "o", "sme-fixed": "s", "ours": "D", "pytorch": "^"}
LINESTYLES = {
    "generic": (0, (1.2, 1.2)),
    "sme-fixed": (0, (4, 1.6)),
    "ours": "-",
    "pytorch": (0, (2.2, 1.2)),
}
LINEWIDTHS = {"generic": 1.05, "sme-fixed": 1.15, "ours": 1.65, "pytorch": 1.2}
MARKERSIZES = {"generic": 3.4, "sme-fixed": 3.4, "ours": 4.2, "pytorch": 3.6}
ZORDERS = {"generic": 2, "sme-fixed": 3, "ours": 5, "pytorch": 4}

MAIN_METHODS = ("generic", "ours", "pytorch")
ABLATION_METHODS = ("sme-fixed", "ours")

SPEEDUP_STYLE = {
    "sme-fixed": dict(color="#1B9E77", marker="s", ls="-", lw=1.5, label="Ours / Fixed SME"),
    "pytorch": dict(
        color="#D95F02", marker="^", ls=(0, (2.2, 1.2)), lw=1.2, label="Ours / PyTorch"
    ),
    "generic": dict(
        color="#5B6B7A", marker="o", ls=(0, (1.2, 1.2)), lw=1.1, label="Ours / TritonCPU"
    ),
}

ACCENT = {
    "gain": "#2CA02C",
    "loss": "#C51B7D",
    "parity": "#B0B0B0",
    "band": "#DEEBF7",
    "ablation_fill": "#B2DFDB",
    "main_fill": "#9ECAE1",
}

OPERATOR_COLORS = {
    "GEMM": "#08519C",
    "Linear": "#3182BD",
    "BMM": "#6BAED6",
    "Attention": "#E6550D",
    "SwiGLU": "#FD8D3C",
    "KAN": "#756BB1",
}

PHASES = ("QK", "scale", "softmax", "PV")
PHASE_COLORS = ("#08519C", "#3182BD", "#FDAE6B", "#74C476")

# Nature single / double column (mm → inch)
MM = 1 / 25.4
FIG_SINGLE = (89 * MM, 68 * MM)
FIG_SINGLE_TALL = (89 * MM, 78 * MM)
FIG_DOUBLE = (183 * MM, 72 * MM)
FIG_DOUBLE_TALL = (183 * MM, 95 * MM)
FIG_BAR = (120 * MM, 78 * MM)
FIG_SUMMARY = (183 * MM, 110 * MM)
FIG_HEAT = (140 * MM, 95 * MM)

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "axes.titleweight": "regular",
    "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.solid_capstyle": "round",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.borderpad": 0.2,
    "legend.handlelength": 1.8,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
    "legend.labelspacing": 0.25,
}


def nature_axes(ax, *, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#272727")
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(colors="#272727", which="both")
    ax.grid(False)
    if grid:
        ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.4, linestyle="-")
        ax.set_axisbelow(True)


def log2_size_axis(ax, sizes: list[int], *, ylabel_pad: float = 2.0) -> None:
    """Place size sweeps on a base-2 log axis so labels never collide."""
    lo, hi = min(sizes), max(sizes)
    ax.set_xscale("log", base=2)
    ax.set_xlim(lo / 1.05, hi * 1.05)
    # Prefer exact powers of two present in data; fall back to log locator.
    powers = [s for s in sizes if s > 0 and (s & (s - 1)) == 0]
    if len(powers) >= 3:
        ticks = powers
        if len(ticks) > 6:
            # Keep endpoints + every other power.
            ticks = [powers[0], *powers[1:-1:2], powers[-1]]
            ticks = sorted(set(ticks))
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(round(v))}"))
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        ax.xaxis.set_major_locator(LogLocator(base=2))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(round(v))}"))
        ax.xaxis.set_minor_locator(NullLocator())


def sparse_size_ticks(sizes: list[int], max_ticks: int = 5) -> list[int]:
    prefer = [128, 256, 512, 1024, 2048, 4096]
    chosen = [s for s in prefer if s in sizes]
    if sizes[0] not in chosen:
        chosen = [sizes[0], *chosen]
    if sizes[-1] not in chosen:
        chosen = [*chosen, sizes[-1]]
    chosen = sorted(set(chosen))
    if len(chosen) <= max_ticks:
        return chosen
    step = max(1, len(chosen) // (max_ticks - 1))
    kept = chosen[::step]
    if chosen[-1] not in kept:
        kept.append(chosen[-1])
    return kept[:max_ticks]


def mark_indices(n: int, target: int = 6) -> list[int]:
    if n <= 0:
        return []
    step = max(1, (n + target - 1) // target)
    return sorted(set(range(0, n, step)) | {n - 1})


def nice_upper(value: float, *, pad: float = 1.08) -> float:
    import math

    if not math.isfinite(value) or value <= 0:
        return 1.0
    target = value * pad
    magnitude = 10 ** math.floor(math.log10(target))
    for step in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        candidate = step * magnitude
        if candidate >= target:
            return candidate
    return 10.0 * magnitude


def parity_line(ax, y: float = 1.0) -> None:
    ax.axhline(y, color="#B0B0B0", linewidth=0.55, linestyle=(0, (1.5, 1.2)), zorder=1)


def panel_label(ax, text: str, *, x: float = -0.14, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#272727",
    )


def save_figure(fig, output: Path, stem: str, *, dpi: int = 600) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(output / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def method_plot_kwargs(method: str, *, open_generic: bool = True) -> dict:
    face = "white" if method == "generic" and open_generic else COLORS[method]
    return dict(
        color=COLORS[method],
        linestyle=LINESTYLES[method],
        linewidth=LINEWIDTHS[method],
        marker=MARKERS[method],
        markersize=MARKERSIZES[method],
        markerfacecolor=face,
        markeredgecolor=COLORS[method],
        markeredgewidth=0.55,
        zorder=ZORDERS[method],
        label=LABELS[method],
    )
