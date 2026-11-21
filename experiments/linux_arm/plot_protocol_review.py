#!/usr/bin/env python3
"""Redraw Linux Arm protocol figures from existing summary and paired CSV files.

No benchmark is executed and no source CSV is modified.  The shape sweep uses
the geometric dimension (M*N*K)^(1/3) so rectangular cases remain visible in a
single axis.  Bands show the archived process min--max range.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from plot_style import COLORS, FIG_DOUBLE, FIG_DOUBLE_TALL, RC, nature_axes, parity_line, save_figure


BASELINE_COLORS = {
    "generic": COLORS["generic"],
    "sme-fixed": COLORS["sme-fixed"],
    "ours": COLORS["ours"],
    "openblas": "#1D4ED8",
    "pytorch": "#6535A6",
}
BASELINE_LABELS = {
    "generic": "TritonCPU",
    "sme-fixed": "Fixed SME",
    "ours": "SAGE",
    "openblas": "OpenBLAS ARMV8SVE",
    "pytorch": "PyTorch",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def effective_size(row: dict[str, str]) -> float:
    return math.prod(int(row[key]) for key in ("m", "n", "k")) ** (1.0 / 3.0)


def placement_key(row: dict[str, str]) -> tuple[str, int]:
    return row["topology"], int(row["threads"])


def plot_shape_sweep(summary_rows: list[dict[str, str]], output: Path) -> None:
    placements = [("single", 1), ("single", 8), ("cross", 8)]
    methods = ["generic", "sme-fixed", "ours", "openblas", "pytorch"]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=FIG_DOUBLE_TALL, sharey=False)
        for axis, placement in zip(axes, placements):
            subset = [r for r in summary_rows if placement_key(r) == placement]
            for method in methods:
                rows = sorted((r for r in subset if r["method"] == method), key=effective_size)
                if not rows:
                    continue
                x = [effective_size(r) for r in rows]
                y = [float(r["logical_gflops"]) for r in rows]
                low = [float(r["min_process_ns"]) for r in rows]
                high = [float(r["max_process_ns"]) for r in rows]
                flops = [2 * math.prod(int(r[key]) for key in ("m", "n", "k")) for r in rows]
                low_gflops = [f / hi for f, hi in zip(flops, high)]
                high_gflops = [f / lo for f, lo in zip(flops, low)]
                color = BASELINE_COLORS[method]
                marker = {"generic": "o", "sme-fixed": "s", "ours": "D", "openblas": "o", "pytorch": "^"}[method]
                axis.plot(x, y, color=color, marker=marker, linewidth=1.2 if method != "ours" else 1.65,
                          markersize=3.0 if method != "ours" else 4.0, label=BASELINE_LABELS.get(method, method))
                axis.fill_between(x, low_gflops, high_gflops, color=color, alpha=0.10, linewidth=0)
            axis.set_xscale("log", base=2)
            axis.set_xlabel(r"Effective dimension $(MNK)^{1/3}$")
            axis.set_title("Single NUMA · 1 thread" if placement == ("single", 1) else
                           "Single NUMA · 8 threads" if placement == ("single", 8) else
                           "Cross NUMA · 8 threads")
            nature_axes(axis)
            axis.legend(loc="upper left", fontsize=5.7, frameon=False)
            axis.grid(False)
        axes[0].set_ylabel("Logical GFLOP/s")
        fig.text(0.5, 1.01, "Linux Arm v9 complete-call shape sweep", ha="center", va="bottom", fontsize=8)
        save_figure(fig, output.parent, output.stem)


def plot_speedups(paired_rows: list[dict[str, str]], output: Path) -> None:
    placements = [("single", 1), ("single", 8), ("cross", 8)]
    baselines = ["generic", "sme-fixed", "openblas", "pytorch"]
    max_speedup = max(
        [float(row["paired_speedup"]) for row in paired_rows if row["candidate"] == "ours"] or [1.0]
    )
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=FIG_DOUBLE, sharey=True)
        for axis, placement in zip(axes, placements):
            subset = [r for r in paired_rows if placement_key(r) == placement and r["candidate"] == "ours"]
            for baseline in baselines:
                rows = sorted((r for r in subset if r["reference"] == baseline), key=effective_size)
                if not rows:
                    continue
                x = [effective_size(r) for r in rows]
                y = [float(r["paired_speedup"]) for r in rows]
                color = BASELINE_COLORS[baseline]
                marker = {"generic": "o", "sme-fixed": "s", "openblas": "o", "pytorch": "^"}[baseline]
                axis.plot(x, y, color=color, marker=marker, linewidth=1.15,
                          markersize=3.0, label=f"SAGE / {BASELINE_LABELS[baseline]}")
            parity_line(axis)
            axis.set_xscale("log", base=2)
            axis.set_xlabel(r"Effective dimension $(MNK)^{1/3}$")
            axis.set_title("Single NUMA · 1 thread" if placement == ("single", 1) else
                           "Single NUMA · 8 threads" if placement == ("single", 8) else
                           "Cross NUMA · 8 threads")
            axis.set_yscale("log", base=2)
            axis.set_ylim(0.25, 2 ** math.ceil(math.log2(max(2.0, max_speedup) * 1.05)))
            nature_axes(axis)
            axis.legend(loc="upper left", fontsize=5.7, frameon=False)
        axes[0].set_ylabel("Paired speedup  (SAGE / baseline)")
        fig.text(0.5, 1.01, "Linux Arm v9 paired complete-call speedups", ha="center", va="bottom", fontsize=8)
        save_figure(fig, output.parent, output.stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", nargs="+", required=True, type=Path)
    parser.add_argument("--paired", nargs="+", required=True, type=Path)
    parser.add_argument("--figures", required=True, type=Path)
    args = parser.parse_args()
    summary = [row for path in args.summary for row in read_rows(path)]
    paired = [row for path in args.paired for row in read_rows(path)]
    args.figures.mkdir(parents=True, exist_ok=True)
    plot_shape_sweep(summary, args.figures / "linux-shape-sweep")
    plot_speedups(paired, args.figures / "linux-paired-speedups")
    print(f"wrote {args.figures / 'linux-shape-sweep'} and {args.figures / 'linux-paired-speedups'}")


if __name__ == "__main__":
    main()
