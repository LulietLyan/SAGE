#!/usr/bin/env python3
"""Before/after plot for 2048-power-of-two GEMM cliffs (Ours).

Compares legacy ``results.csv`` / ``results_large.csv`` (before) against
``results_cliff*.csv`` (after padding / stride-conflict fix).
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import (
    COLORS,
    FIG_DOUBLE,
    FIG_DOUBLE_TALL,
    MAIN_METHODS,
    RC,
    log2_size_axis,
    mark_indices,
    method_plot_kwargs,
    nature_axes,
    nice_upper,
    panel_label,
    save_figure,
)

plt.rcParams.update(RC)

POW2_CLIFFS = (2048, 4096, 8192)


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _series(rows: list[dict], *, threads: int = 8, method: str = "ours") -> dict[int, float]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if int(row["threads"]) != threads or row["method"] != method:
            continue
        buckets[int(row["m"])].append(float(row["gflops"]))
    return {s: _median(vs) for s, vs in buckets.items()}


def _load_before(output: Path) -> dict[int, float]:
    merged: dict[int, list[float]] = defaultdict(list)
    for name in ("results.csv", "results_large.csv"):
        for row in _read(output / name):
            if int(row["threads"]) != 8 or row["method"] != "ours":
                continue
            # Prefer results_large when both exist: collect all then median
            merged[int(row["m"])].append(float(row["gflops"]))
    return {s: _median(vs) for s, vs in merged.items()}


def _load_after(output: Path) -> dict[int, float]:
    rows = _read(output / "results_cliff.csv") + _read(output / "results_cliff8192.csv")
    return _series(rows)


def _neighbor_ratio(series: dict[int, float], cliff: int) -> float | None:
    if cliff not in series:
        return None
    left = max((s for s in series if s < cliff), default=None)
    right = min((s for s in series if s > cliff), default=None)
    neighbors = [series[s] for s in (left, right) if s is not None]
    if not neighbors:
        return None
    ref = statistics.mean(neighbors)
    if ref <= 0:
        return None
    return series[cliff] / ref


def plot_cliff(output: Path) -> None:
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    before = _load_before(output)
    after = _load_after(output)
    if not after:
        raise SystemExit("results_cliff*.csv missing; run the cliff measurement first")

    # Focus window around measured after sizes, union with before for overlay
    focus = sorted(set(after) | {s for s in before if 1600 <= s <= 8500})
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE_TALL)

    ax = axes[0]
    for label, series, color, ls, marker in (
        ("Ours (before)", before, "#9ECAE1", (0, (2.2, 1.2)), "o"),
        ("Ours (after)", after, COLORS["ours"], "-", "D"),
    ):
        xs = [s for s in focus if s in series]
        if not xs:
            continue
        ys = [series[s] for s in xs]
        ax.plot(
            xs,
            ys,
            color=color,
            ls=ls,
            lw=1.5 if "after" in label else 1.1,
            label=label,
            zorder=3 if "after" in label else 2,
        )
        idxs = mark_indices(len(xs), 8)
        ax.plot(
            [xs[i] for i in idxs],
            [ys[i] for i in idxs],
            ls="none",
            marker=marker,
            ms=3.8 if "after" in label else 3.2,
            color=color,
            zorder=4 if "after" in label else 2,
        )
    for cliff in POW2_CLIFFS:
        if any(cliff in s for s in (before, after)):
            ax.axvline(cliff, color="#D0D0D0", lw=0.7, ls=(0, (1.5, 1.2)), zorder=1)
    log2_size_axis(ax, focus if focus else [2048, 8192])
    nature_axes(ax, grid=True)
    ax.set_xlabel("Square size $N$")
    ax.set_ylabel("Ours throughput (GFLOP/s)")
    ymax = max([*(before.get(s, 0) for s in focus), *(after.get(s, 0) for s in focus)], default=1)
    ax.set_ylim(0, nice_upper(ymax))
    ax.legend(loc="best", frameon=False)
    panel_label(ax, "a")

    # Neighbor-normalized bar at cliffs
    ax = axes[1]
    cliffs = [c for c in POW2_CLIFFS if c in after or c in before]
    x = np.arange(len(cliffs))
    width = 0.36
    before_r = [_neighbor_ratio(before, c) or float("nan") for c in cliffs]
    after_r = [_neighbor_ratio(after, c) or float("nan") for c in cliffs]
    ax.bar(
        x - width / 2,
        before_r,
        width,
        color="#9ECAE1",
        edgecolor="white",
        label="Before / neighbors",
    )
    ax.bar(
        x + width / 2,
        after_r,
        width,
        color=COLORS["ours"],
        edgecolor="white",
        label="After / neighbors",
    )
    ax.axhline(1.0, color="#B0B0B0", lw=0.7, ls=(0, (1.5, 1.2)))
    nature_axes(ax, grid=True)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in cliffs])
    ax.set_ylabel("Ours GFLOP/s ÷ mean of nearest neighbors")
    ax.set_xlabel("Power-of-two cliff $N$")
    ax.legend(loc="best", frameon=False)
    ax.set_ylim(0, nice_upper(max([v for v in before_r + after_r if math.isfinite(v)] + [1.2])))
    for i, (b, a) in enumerate(zip(before_r, after_r)):
        if math.isfinite(b):
            ax.text(
                i - width / 2,
                b + 0.02,
                f"{b:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                color="#4D4D4D",
            )
        if math.isfinite(a):
            ax.text(
                i + width / 2,
                a + 0.02,
                f"{a:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                color="#4D4D4D",
            )
    panel_label(ax, "b")
    fig.suptitle("2048-stride cliff fix · 8 threads · Ours", fontsize=8, y=1.02)
    fig.tight_layout()
    save_figure(fig, figures, "v2-gemm-cliff2048-before-after")

    # Main triad after-only around cliffs
    after_rows = _read(output / "results_cliff.csv") + _read(output / "results_cliff8192.csv")
    by_method = {m: _series(after_rows, method=m) for m in MAIN_METHODS}
    sizes = sorted(by_method["ours"])
    if sizes:
        fig, ax = plt.subplots(figsize=FIG_DOUBLE)
        for method in MAIN_METHODS:
            xs = [s for s in sizes if s in by_method[method]]
            ys = [by_method[method][s] for s in xs]
            kw = method_plot_kwargs(method)
            idxs = mark_indices(len(xs), 8)
            ax.plot(
                xs,
                ys,
                color=kw["color"],
                ls=kw["linestyle"],
                lw=kw["linewidth"],
                label=kw["label"],
                zorder=kw["zorder"],
            )
            ax.plot(
                [xs[i] for i in idxs],
                [ys[i] for i in idxs],
                ls="none",
                marker=kw["marker"],
                ms=kw["markersize"],
                mfc=kw["markerfacecolor"],
                mec=kw["markeredgecolor"],
                mew=kw["markeredgewidth"],
                zorder=kw["zorder"] + 1,
            )
        for cliff in POW2_CLIFFS:
            if cliff in sizes:
                ax.axvline(cliff, color="#D0D0D0", lw=0.7, ls=(0, (1.5, 1.2)), zorder=1)
        log2_size_axis(ax, sizes)
        nature_axes(ax, grid=True)
        ax.set_xlabel("Square size $N$")
        ax.set_ylabel("Throughput (GFLOP/s)")
        ax.legend(loc="best", frameon=False)
        ax.set_title("After fix · main triad around 2048-family sizes · 8 threads")
        ymax = max(by_method[m][s] for m in MAIN_METHODS for s in sizes if s in by_method[m])
        ax.set_ylim(0, nice_upper(ymax))
        fig.tight_layout()
        save_figure(fig, figures, "v2-gemm-cliff2048-triad")

    # Print summary for the chat
    print("Cliff neighbor ratios (Ours GFLOP/s ÷ mean neighbors):")
    for c in cliffs:
        b, a = _neighbor_ratio(before, c), _neighbor_ratio(after, c)
        b_s = f"{b:.3f}" if b is not None else "n/a"
        a_s = f"{a:.3f}" if a is not None else "n/a"
        abs_b = f"{before[c]:.1f}" if c in before else "n/a"
        abs_a = f"{after[c]:.1f}" if c in after else "n/a"
        print(f"  N={c}: before {abs_b} GFLOP/s ({b_s}× nbr) → after {abs_a} GFLOP/s ({a_s}× nbr)")
    print(f"Figures → {figures}/v2-gemm-cliff2048-*.{{pdf,png,svg}}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "output"
    )
    plot_cliff(parser.parse_args().output)


if __name__ == "__main__":
    main()
