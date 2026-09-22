#!/usr/bin/env python3
"""Plot scalar stride-probe CSVs from an explicitly selected directory."""

import argparse
import csv
import statistics
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import RC, nature_axes, panel_label

plt.rcParams.update(RC)


def plot(root: Path):
    with (root / "low_level_stride.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(183 / 25.4, 78 / 25.4), gridspec_kw={"width_ratios": (1.35, 1)}
    )
    colors = {0: "#08519C", 4: "#3182BD", 8: "#6BAED6", 16: "#74C476", 32: "#FD8D3C", 64: "#BDBDBD"}
    for pad in [0, 4, 8, 16, 32, 64]:
        vals = []
        for c in sorted({int(r["columns"]) for r in rows}):
            vals.append(
                statistics.median(
                    float(r["gbps"])
                    for r in rows
                    if int(r["columns"]) == c and int(r["pad"]) == pad
                )
            )
        ax.plot(
            sorted({int(r["columns"]) for r in rows}),
            vals,
            marker="o",
            ms=2.5,
            lw=1.05,
            color=colors[pad],
            label=f"pad +{pad}",
        )
    ax.axvline(2048, color="#777", lw=0.6, ls=(0, (1.5, 1.3)))
    ax.set_xlabel("Logical columns")
    ax.set_ylabel("Scalar load bandwidth (GB s$^{-1}$)")
    nature_axes(ax, grid=True)
    ax.legend(ncol=2, fontsize=5.7, handlelength=1.5)
    panel_label(ax, "a")
    for phase in [0, 4, 8, 16]:
        vals = [
            statistics.median(
                float(r["gbps"])
                for r in rows
                if int(r["columns"]) == c and int(r["pad"]) == 0 and int(r["phase"]) == phase
            )
            for c in sorted({int(r["columns"]) for r in rows})
        ]
        bx.plot(
            sorted({int(r["columns"]) for r in rows}),
            vals,
            marker="o",
            ms=2.5,
            lw=1.05,
            label=f"phase {phase} floats",
        )
    bx.axvline(2048, color="#777", lw=0.6, ls=(0, (1.5, 1.3)))
    bx.set_xlabel("Logical columns; pad 0")
    bx.set_ylabel("Scalar load bandwidth (GB s$^{-1}$)")
    nature_axes(bx, grid=True)
    bx.legend(ncol=2, fontsize=5.7, handlelength=1.5)
    panel_label(bx, "b")
    fig.suptitle("Scalar load response to row stride and padding", fontsize=8.2, y=0.995)
    fig.tight_layout(pad=0.5)
    fig.savefig(root / "low-level-stride.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="directory containing low_level_stride.csv"
    )
    plot(parser.parse_args().root)


if __name__ == "__main__":
    main()
