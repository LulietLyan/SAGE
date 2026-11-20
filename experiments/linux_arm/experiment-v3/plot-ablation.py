#!/usr/bin/env python3
"""Plot Linux schedule-selection regret with the main-branch figure style."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments"))
from plot_style import COLORS, FIG_DOUBLE, MARKERS, RC, nature_axes, parity_line, save_figure


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return sorted(rows, key=lambda row: (row["topology"], int(row["m"])))


def plot(rows: list[dict[str, str]], output: Path) -> None:
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE, sharey=True)
        for axis, topology, panel in zip(axes, ("single", "cross"), ("a", "b")):
            subset = [row for row in rows if row["topology"] == topology]
            sizes = [int(row["m"]) for row in subset]
            ratios = [float(row["sage_over_best_latency"]) for row in subset]
            axis.set_xscale("log", base=2)
            axis.set_xticks(sizes)
            axis.set_xticklabels([str(size) for size in sizes])
            axis.plot(
                sizes,
                ratios,
                color=COLORS["ours"],
                marker=MARKERS["ours"],
                linewidth=1.65,
                markersize=4.2,
                markerfacecolor=COLORS["ours"],
                markeredgecolor=COLORS["ours"],
                markeredgewidth=0.55,
                label="SAGE / best measured",
                zorder=5,
            )
            for size, ratio, row in zip(sizes, ratios, subset):
                axis.annotate(
                    f"{ratio:.3f}×\n{row['best_measured']}",
                    (size, ratio),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=COLORS["ours"],
                )
            parity_line(axis)
            axis.set_ylim(0.95, max(1.35, max(ratios) * 1.15))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            axis.set_xlabel(r"Square dimension $M{=}N{=}K$")
            axis.set_title("Single NUMA" if topology == "single" else "Cross NUMA")
            axis.legend(loc="upper left", fontsize=6, frameon=False, borderaxespad=0.2)
            nature_axes(axis)
            axis.text(
                -0.14,
                1.05,
                f"({panel})",
                transform=axis.transAxes,
                fontsize=8,
                fontweight="bold",
                va="bottom",
                ha="left",
                color="#272727",
            )
        axes[0].set_ylabel("SAGE latency / best measured candidate")
        fig.text(
            0.5,
            1.01,
            "Linux Arm v9 schedule-selection gap",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        save_figure(fig, output.parent, output.stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(load_rows(args.summaries), args.output)


if __name__ == "__main__":
    main()
