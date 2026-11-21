#!/usr/bin/env python3
"""Plot the Linux single-thread result with the original M4 figure grammar.

The main Linux figure intentionally uses one single-NUMA thread budget, square
shapes only, the four methods shared with the M4 experiment, a linear size axis,
and the M4 serif palette.  Multi-thread and cross-NUMA views remain separate
diagnostics because the cited GPU papers do not define a CPU thread protocol
that can be transferred to SAGE.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "experiments"))
from plot_style import LABELS, save_figure


METHODS = ("generic", "sme-fixed", "ours", "pytorch")
COLORS = {
    "generic": "#9AA4B2",
    "sme-fixed": "#3CBBB0",
    "ours": "#5B8DEF",
    "pytorch": "#F2A65A",
}
MARKERS = {"generic": "o", "sme-fixed": "s", "ours": "D", "pytorch": "^"}
LINESTYLES = {
    "generic": (0, (1.1, 1.15)),
    "sme-fixed": (0, (4.2, 1.6)),
    "ours": "-",
    "pytorch": (0, (2.4, 1.15)),
}
LINEWIDTHS = {"generic": 1.2, "sme-fixed": 1.4, "ours": 2.1, "pytorch": 1.5}
MARKERSIZES = {"generic": 3.4, "sme-fixed": 3.5, "ours": 4.2, "pytorch": 3.8}
ZORDERS = {"generic": 2, "sme-fixed": 3, "ours": 5, "pytorch": 4}


def read_summary(path: Path) -> dict[str, dict[int, float]]:
    rows = defaultdict(dict)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["topology"] != "single" or int(row["threads"]) != 1:
                continue
            if not (row["m"] == row["n"] == row["k"]):
                continue
            method = row["method"]
            if method not in METHODS:
                continue
            rows[method][int(row["m"])] = float(row["logical_gflops"])
    missing = [method for method in METHODS if not rows[method]]
    if missing:
        raise ValueError(f"Missing square single-NUMA 1-thread methods: {missing}")
    sizes = set(rows["ours"])
    if any(set(rows[method]) != sizes for method in METHODS):
        raise ValueError("The four methods do not share the same square shape set")
    return rows


def journal_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#1A1A1A")
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(which="both", direction="out", colors="#1A1A1A")
    ax.yaxis.grid(True, which="major", color="#D8D8D8", linewidth=0.55)
    ax.yaxis.grid(True, which="minor", color="#EEEEEE", linewidth=0.35)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_axisbelow(True)


def plot(summary_path: Path, output: Path) -> None:
    data = read_summary(summary_path)
    sizes = sorted(data["ours"])
    ymax = max(max(data[method].values()) for method in METHODS)
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, ax = plt.subplots(figsize=(183 / 25.4, 100 / 25.4))
        for method in METHODS:
            color = COLORS[method]
            ax.plot(
                sizes,
                [data[method][size] for size in sizes],
                color=color,
                linestyle=LINESTYLES[method],
                linewidth=LINEWIDTHS[method],
                marker=MARKERS[method],
                markersize=MARKERSIZES[method],
                markerfacecolor="white" if method == "generic" else color,
                markeredgecolor=color,
                markeredgewidth=0.7,
                zorder=ZORDERS[method],
                label=LABELS[method],
            )
        ax.set_xlim(min(sizes) - 0.04 * (max(sizes) - min(sizes)), max(sizes) + 0.04 * (max(sizes) - min(sizes)))
        ticks = [size for size in (128, 512, 1024, 2048, 3072, 4096) if size in sizes]
        if len(ticks) < 4:
            ticks = sizes[:: max(1, len(sizes) // 5)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(size) for size in ticks])
        ax.set_xlabel(r"Square GEMM size $N$", labelpad=6, fontweight="bold")
        ax.set_ylabel(r"Throughput (GFLOP$\,s^{-1}$)", labelpad=7, fontweight="bold")
        ax.set_ylim(0, ymax * 1.12)
        journal_axes(ax)
        ax.legend(loc="upper right", frameon=False, handlelength=2.4, labelspacing=0.32)
        fig.tight_layout(pad=0.35)
        save_figure(fig, output.parent, output.stem, dpi=600)
    print(f"wrote {output.parent / output.stem} for {len(sizes)} square shapes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.summary, args.output)


if __name__ == "__main__":
    main()
