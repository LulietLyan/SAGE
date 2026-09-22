#!/usr/bin/env python3
"""Nature-style narrative figures for schedule-selection claims.

Core conclusion: under identical Triton GEMM source, measured-cost schedule
selection (Ours) is consistently faster than Fixed SME; vs PyTorch/Accelerate
the gap remains honest and visible.
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
from matplotlib.ticker import MaxNLocator

from plot_style import (
    COLORS,
    FIG_BAR,
    FIG_DOUBLE,
    FIG_SINGLE,
    FIG_SINGLE_TALL,
    LABELS,
    METHODS,
    PHASE_COLORS,
    PHASES,
    RC,
    SPEEDUP_STYLE,
    log2_size_axis,
    mark_indices,
    method_plot_kwargs,
    nature_axes,
    nice_upper,
    panel_label,
    parity_line,
    save_figure,
)


def _read(path: Path) -> list[dict]:
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _geom_mean(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(v) for v in values))


def _paired_speedups(rows: list[dict], *, threads: int, size_key: str, latency_key: str) -> dict:
    by = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if int(row["threads"]) != threads:
            continue
        size = int(row[size_key])
        process = int(row.get("process", 0))
        by[size][row["method"]].append((process, float(row[latency_key])))
    result = {}
    for size, methods in sorted(by.items()):
        if "ours" not in methods:
            continue
        ours = {p: lat for p, lat in methods["ours"]}
        entry = {}
        for baseline in ("sme-fixed", "pytorch"):
            if baseline not in methods:
                continue
            ratios = [lat / ours[p] for p, lat in methods[baseline] if p in ours and ours[p] > 0]
            if ratios:
                entry[baseline] = _median(ratios)
        if entry:
            result[size] = entry
    return result


def _throughput(rows: list[dict], *, threads: int) -> dict:
    by = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if int(row["threads"]) != threads:
            continue
        by[int(row["m"])][row["method"]].append(float(row["gflops"]))
    return {s: {m: _median(v) for m, v in methods.items()} for s, methods in sorted(by.items())}


def plot_gemm_speedup(output: Path, rows: list[dict]) -> None:
    figures = output / "figures"
    for threads in sorted({int(r["threads"]) for r in rows}):
        speedups = _paired_speedups(rows, threads=threads, size_key="m", latency_key="median_ns")
        sizes = list(speedups)
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        for baseline in ("sme-fixed", "pytorch"):
            st = SPEEDUP_STYLE[baseline]
            ys = [speedups[s][baseline] for s in sizes]
            ax.plot(
                sizes,
                ys,
                color=st["color"],
                linestyle=st["ls"],
                linewidth=st["lw"],
                marker=st["marker"],
                markersize=3.4,
                markevery=mark_indices(len(sizes), 5),
                markerfacecolor=st["color"],
                markeredgewidth=0.5,
                label=st["label"],
                zorder=4 if baseline == "sme-fixed" else 3,
            )
        parity_line(ax)
        plotted = [speedups[s][b] for s in sizes for b in ("sme-fixed", "pytorch")]
        ax.set_ylim(0, nice_upper(max(plotted), pad=1.12))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        log2_size_axis(ax, sizes)
        ax.set_xlabel(r"Square dimension $M{=}N{=}K$")
        ax.set_ylabel("Paired speedup")
        nature_axes(ax)
        ax.legend(loc="upper left", fontsize=6)
        gm = _geom_mean([speedups[s]["sme-fixed"] for s in sizes])
        wins = sum(speedups[s]["sme-fixed"] > 1 for s in sizes)
        ax.text(
            0.98,
            0.04,
            f"Geom. mean vs Fixed SME = {gm:.2f}×  ({wins}/{len(sizes)})",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.8,
            color="#4D4D4D",
        )
        fig.suptitle(f"GEMM schedule gain  ·  {threads} thr.", fontsize=8, y=1.02)
        save_figure(fig, figures, f"fig-gemm-speedup-t{threads}")


def plot_gemm_panel(output: Path, rows: list[dict]) -> None:
    figures = output / "figures"
    for threads in sorted({int(r["threads"]) for r in rows}):
        series = _throughput(rows, threads=threads)
        speedups = _paired_speedups(rows, threads=threads, size_key="m", latency_key="median_ns")
        sizes = list(series)
        fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE, sharex=True)
        fig.subplots_adjust(wspace=0.32)
        ax0, ax1 = axes
        for method in METHODS:
            ys = [series[s][method] for s in sizes]
            ax0.plot(sizes, ys, markevery=mark_indices(len(sizes), 5), **method_plot_kwargs(method))
        ax0.set_ylabel("Throughput (GFLOP/s)")
        ax0.set_ylim(0, nice_upper(max(series[s][m] for s in sizes for m in METHODS)))
        ax0.yaxis.set_major_locator(MaxNLocator(nbins=4))
        nature_axes(ax0)
        ax0.legend(loc="upper left", fontsize=5.8)
        panel_label(ax0, "a")
        for baseline in ("sme-fixed", "pytorch"):
            st = SPEEDUP_STYLE[baseline]
            ys = [speedups[s][baseline] for s in sizes]
            ax1.plot(
                sizes,
                ys,
                color=st["color"],
                linestyle=st["ls"],
                linewidth=st["lw"],
                marker=st["marker"],
                markersize=3.2,
                markevery=mark_indices(len(sizes), 5),
                label=st["label"],
            )
        parity_line(ax1)
        plotted = [speedups[s][b] for s in sizes for b in ("sme-fixed", "pytorch")]
        ax1.set_ylim(0, nice_upper(max(plotted), pad=1.12))
        ax1.set_ylabel("Paired speedup")
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))
        nature_axes(ax1)
        ax1.legend(loc="upper left", fontsize=5.8)
        panel_label(ax1, "b")
        for ax in axes:
            log2_size_axis(ax, sizes)
            ax.set_xlabel(r"$M{=}N{=}K$")
        fig.suptitle(f"Square FP32 GEMM  ·  {threads} thr.", fontsize=8, y=1.04)
        save_figure(fig, figures, f"fig-gemm-panel-t{threads}")


def plot_planner_decisions(output: Path, rows: list[dict]) -> None:
    figures = output / "figures"
    ours = [r for r in rows if r["method"] == "ours"]
    for threads in sorted({int(r["threads"]) for r in ours}):
        subset = [r for r in ours if int(r["threads"]) == threads]
        sizes = sorted({int(r["m"]) for r in subset})

        def mode_at(field: str) -> list[int]:
            out = []
            for size in sizes:
                vals = [int(float(r[field])) for r in subset if int(r["m"]) == size]
                out.append(statistics.mode(vals))
            return out

        fig, axes = plt.subplots(3, 1, figsize=FIG_SINGLE_TALL, sharex=True)
        fig.subplots_adjust(hspace=0.38)
        for ax, values, ylabel, color in (
            (axes[0], mode_at("selected_threads"), "Team size", "#3775BA"),
            (axes[1], mode_at("block_m"), "Tile $M$", "#2E9E44"),
            (axes[2], mode_at("k_strip"), "K-strip", "#E28E2C"),
        ):
            ax.step(sizes, values, where="mid", color=color, linewidth=1.35)
            ax.scatter(
                sizes, values, s=11, color=color, zorder=3, edgecolors="white", linewidths=0.3
            )
            ax.set_ylabel(ylabel)
            uniq = sorted(set(values))
            ax.set_yticks(uniq)
            nature_axes(ax)
            lo, hi = min(uniq), max(uniq)
            span = hi - lo
            pad = (0.4 * span) if span else (0.15 * max(abs(hi), 1))
            ax.set_ylim(lo - pad, hi + pad)
        log2_size_axis(axes[-1], sizes)
        axes[-1].set_xlabel(r"Square dimension $M{=}N{=}K$")
        fig.suptitle(f"Ours planner decisions  ·  {threads} thr.", fontsize=8, y=1.01)
        save_figure(fig, figures, f"fig-planner-decisions-t{threads}")


def plot_attention_speedup(output: Path) -> None:
    path = output / "attention_results.csv"
    if not path.is_file():
        return
    rows = _read(path)
    figures = output / "figures"
    for threads in sorted({int(r["threads"]) for r in rows}):
        speedups = _paired_speedups(
            rows, threads=threads, size_key="sequence_length", latency_key="latency_ms"
        )
        lengths = list(speedups)
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        for baseline in ("sme-fixed", "pytorch"):
            st = SPEEDUP_STYLE[baseline]
            ys = [speedups[length][baseline] for length in lengths]
            ax.plot(
                lengths,
                ys,
                color=st["color"],
                linestyle=st["ls"],
                linewidth=st["lw"],
                marker=st["marker"],
                markersize=3.4,
                markevery=mark_indices(len(lengths), 5),
                label=st["label"],
            )
        parity_line(ax)
        plotted = [speedups[length][b] for length in lengths for b in ("sme-fixed", "pytorch")]
        ax.set_ylim(0, nice_upper(max(plotted), pad=1.12))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        log2_size_axis(ax, lengths)
        ax.set_xlabel("Sequence length $L$")
        ax.set_ylabel("Paired speedup")
        nature_axes(ax)
        ax.legend(loc="best", fontsize=6)
        fig.suptitle(f"Attention e2e gain  ·  {threads} thr.", fontsize=8, y=1.02)
        save_figure(fig, figures, f"fig-attention-speedup-t{threads}")


def plot_attention_phase_share(output: Path) -> None:
    path = output / "attention_phases.csv"
    if not path.is_file():
        return
    rows = _read(path)
    figures = output / "figures"
    for threads in sorted({int(r["threads"]) for r in rows}):
        lengths = sorted({int(r["sequence_length"]) for r in rows if int(r["threads"]) == threads})
        for length in [n for n in lengths if n in (256, 512, 1024)] or [lengths[-1]]:
            fig, ax = plt.subplots(figsize=FIG_BAR)
            y = list(range(len(METHODS)))
            left = [0.0] * len(METHODS)
            for phase, color in zip(PHASES, PHASE_COLORS):
                shares = []
                for method in METHODS:
                    vals = []
                    for ph in PHASES:
                        samples = [
                            float(r["latency_ms"])
                            for r in rows
                            if int(r["threads"]) == threads
                            and int(r["sequence_length"]) == length
                            and r["method"] == method
                            and r["phase"] == ph
                        ]
                        vals.append(_median(samples) if samples else 0.0)
                    total = sum(vals) or 1.0
                    shares.append(100.0 * vals[PHASES.index(phase)] / total)
                ax.barh(
                    y,
                    shares,
                    0.58,
                    left=left,
                    color=color,
                    edgecolor="white",
                    linewidth=0.4,
                    label=phase,
                )
                left = [a + b for a, b in zip(left, shares)]
            ax.set_yticks(y, [LABELS[m] for m in METHODS])
            ax.invert_yaxis()
            ax.set_xlim(0, 100)
            ax.set_xlabel("Stage share (%)")
            ax.xaxis.set_major_locator(MaxNLocator(5))
            nature_axes(ax, grid=True)
            ax.legend(loc="lower right", fontsize=5.8, ncol=2)
            fig.suptitle(
                f"Attention phase share  ·  $L={length}$, {threads} thr.", fontsize=8, y=1.02
            )
            save_figure(fig, figures, f"fig-attention-phase-share-t{threads}-l{length}")


def plot_operator_summary(output: Path) -> None:
    figures = output / "figures"
    blocks = []
    if (output / "results.csv").is_file():
        gemm = _read(output / "results.csv")
        for threads in sorted({int(r["threads"]) for r in gemm}):
            sp = _paired_speedups(gemm, threads=threads, size_key="m", latency_key="median_ns")
            blocks.append(
                (
                    f"GEMM\n{threads}t",
                    _geom_mean([v["sme-fixed"] for v in sp.values()]),
                    _geom_mean([v["pytorch"] for v in sp.values()]),
                )
            )
    if (output / "attention_results.csv").is_file():
        attn = _read(output / "attention_results.csv")
        for threads in sorted({int(r["threads"]) for r in attn}):
            sp = _paired_speedups(
                attn, threads=threads, size_key="sequence_length", latency_key="latency_ms"
            )
            blocks.append(
                (
                    f"Attn\n{threads}t",
                    _geom_mean([v["sme-fixed"] for v in sp.values()]),
                    _geom_mean([v["pytorch"] for v in sp.values()]),
                )
            )
    for name, path, keys in (
        ("KAN", output / "kan.csv", ("tokens", "in_features", "out_features", "spline_intervals")),
        ("SwiGLU", output / "swiglu.csv", ("tokens", "width", "hidden")),
    ):
        if not path.is_file():
            continue
        rows = _read(path)
        for threads in sorted({int(r["threads"]) for r in rows}):
            subset = [r for r in rows if int(r["threads"]) == threads]
            cases = sorted({tuple(int(r[k]) for k in keys) for r in subset})
            fixed, torch_r = [], []
            for case in cases:
                by = {
                    r["method"]: float(r["latency_ms"])
                    for r in subset
                    if tuple(int(r[k]) for k in keys) == case
                }
                if "ours" in by and "sme-fixed" in by and by["ours"] > 0:
                    fixed.append(by["sme-fixed"] / by["ours"])
                if "ours" in by and "pytorch" in by and by["ours"] > 0:
                    torch_r.append(by["pytorch"] / by["ours"])
            if fixed and torch_r:
                blocks.append((f"{name}\n{threads}t", _geom_mean(fixed), _geom_mean(torch_r)))
    if not blocks:
        return
    x = list(range(len(blocks)))
    width = 0.34
    fig, ax = plt.subplots(figsize=FIG_BAR)
    ax.bar(
        [i - width / 2 for i in x],
        [b[1] for b in blocks],
        width,
        color=COLORS["ours"],
        edgecolor="white",
        linewidth=0.4,
        label="Ours / Fixed SME",
    )
    ax.bar(
        [i + width / 2 for i in x],
        [b[2] for b in blocks],
        width,
        color=COLORS["pytorch"],
        edgecolor="white",
        linewidth=0.4,
        label="Ours / PyTorch",
    )
    parity_line(ax)
    ax.set_xticks(x, [b[0] for b in blocks])
    ax.set_ylabel("Geom.-mean paired speedup")
    ax.set_ylim(0, nice_upper(max(max(b[1], b[2]) for b in blocks), pad=1.15))
    nature_axes(ax, grid=True)
    ax.legend(loc="upper right", fontsize=6)
    fig.suptitle("Cross-operator schedule gain", fontsize=8, y=1.02)
    save_figure(fig, figures, "fig-operator-summary")


def plot_all(output: Path) -> None:
    output = Path(output)
    rows = _read(output / "results.csv")
    with plt.rc_context(RC):
        plot_gemm_speedup(output, rows)
        plot_gemm_panel(output, rows)
        plot_planner_decisions(output, rows)
        plot_attention_speedup(output)
        plot_attention_phase_share(output)
        plot_operator_summary(output)
    print(f"Nature-style figures → {output / 'figures'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "output"
    )
    plot_all(parser.parse_args().output)


if __name__ == "__main__":
    main()
