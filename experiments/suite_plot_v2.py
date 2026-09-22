#!/usr/bin/env python3
"""Suite v2 figures: main triad lines, Fixed-SME ablation, cross-operator summary.

Plot conventions:
- No heatmaps.
- Scale sweeps → line charts (GEMM / Linear / BMM / Attention / SwiGLU / KAN).
- Bars only for multi-step latency breakdowns (Attention phases).
- Large-N GEMM panel highlights GFLOP/s stability for Ours vs TritonCPU / PyTorch.
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
from matplotlib.ticker import MaxNLocator

from plot_style import (
    ABLATION_METHODS,
    ACCENT,
    COLORS,
    FIG_DOUBLE,
    FIG_DOUBLE_TALL,
    FIG_SINGLE,
    FIG_SUMMARY,
    LABELS,
    MAIN_METHODS,
    OPERATOR_COLORS,
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

plt.rcParams.update(RC)


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _geom_mean(values: list[float]) -> float:
    clean = [v for v in values if v > 0 and math.isfinite(v)]
    if not clean:
        return float("nan")
    return math.exp(statistics.mean(math.log(v) for v in clean))


def _resolve_gemm(output: Path) -> tuple[list[dict], int]:
    chunks = []
    for name in ("results_v2.csv", "results_large.csv", "results.csv"):
        rows = _read(output / name)
        if rows:
            chunks.extend(rows)
    if not chunks:
        return [], 8
    # Prefer v2/large rows when duplicate (size, process, method, threads) appear.
    best: dict[tuple, dict] = {}
    # Re-read with source priority
    best = {}
    for name, pri in (("results.csv", 1), ("results_v2.csv", 2), ("results_large.csv", 3)):
        for row in _read(output / name):
            key = (int(row["threads"]), int(row["m"]), int(row.get("process", 0)), row["method"])
            prev = best.get(key)
            if prev is None or pri >= prev[0]:
                best[key] = (pri, row)
    merged = [v[1] for v in best.values()]
    threads = sorted({int(r["threads"]) for r in merged})
    return merged, (8 if 8 in threads else threads[-1])


def _series_by_size(rows: list[dict], *, threads: int, metric: str) -> dict[str, dict[int, float]]:
    buckets: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if int(row["threads"]) != threads:
            continue
        size = int(row["m"])
        buckets[row["method"]][size].append(float(row[metric]))
    return {m: {s: _median(vs) for s, vs in sizes.items()} for m, sizes in buckets.items()}


def _paired_speedup_curve(
    rows: list[dict], *, threads: int, baselines=("sme-fixed", "pytorch", "generic")
):
    by = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if int(row["threads"]) != threads:
            continue
        by[int(row["m"])][row["method"]].append((int(row["process"]), float(row["median_ns"])))
    out = {b: {} for b in baselines}
    for size, methods in sorted(by.items()):
        if "ours" not in methods:
            continue
        ours = {p: ns for p, ns in methods["ours"]}
        for baseline in baselines:
            if baseline not in methods:
                continue
            ratios = [ns / ours[p] for p, ns in methods[baseline] if p in ours and ours[p] > 0]
            if ratios:
                out[baseline][size] = _median(ratios)
    return out


def _plot_lines(ax, sizes, series, methods, *, mark=8) -> None:
    for method in methods:
        if method not in series:
            continue
        xs = [s for s in sizes if s in series[method]]
        if not xs:
            continue
        ys = [series[method][s] for s in xs]
        kw = method_plot_kwargs(method)
        idxs = mark_indices(len(xs), mark)
        ax.plot(
            xs,
            ys,
            color=kw["color"],
            ls=kw["linestyle"],
            lw=kw["linewidth"],
            zorder=kw["zorder"],
            label=kw["label"],
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


def plot_gemm_main(rows: list[dict], figures: Path, *, threads: int) -> None:
    gflops = _series_by_size(rows, threads=threads, metric="gflops")
    latency = _series_by_size(rows, threads=threads, metric="median_ns")
    sizes = sorted(gflops.get("ours", {}))
    if not sizes:
        return
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE_TALL)
    _plot_lines(axes[0], sizes, gflops, MAIN_METHODS)
    lat_ms = {m: {s: v / 1e6 for s, v in d.items()} for m, d in latency.items()}
    _plot_lines(axes[1], sizes, lat_ms, MAIN_METHODS)
    for ax, series, ylabel in (
        (axes[0], gflops, "Throughput (GFLOP/s)"),
        (axes[1], lat_ms, "Latency (ms)"),
    ):
        log2_size_axis(ax, sizes)
        nature_axes(ax, grid=True)
        ax.set_xlabel("Square size $N$")
        ax.set_ylabel(ylabel)
        ymax = max(
            series[m][s] for m in MAIN_METHODS if m in series for s in sizes if s in series[m]
        )
        ax.set_ylim(0, nice_upper(ymax))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].legend(loc="upper left", frameon=False)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.suptitle(
        f"Main comparison · {threads} threads · TritonCPU / Ours / PyTorch", fontsize=8, y=1.02
    )
    fig.tight_layout()
    save_figure(fig, figures, f"v2-gemm-main-t{threads}")


def plot_gemm_large_stability(
    rows: list[dict], figures: Path, *, threads: int, floor: int = 2048
) -> None:
    """Zoom on large N: does Ours hold GFLOP/s while baselines diverge?"""
    gflops = _series_by_size(rows, threads=threads, metric="gflops")
    sizes = sorted(s for s in gflops.get("ours", {}) if s >= floor)
    if len(sizes) < 3:
        sizes = sorted(gflops.get("ours", {}))[-8:]
    if len(sizes) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE_TALL)
    _plot_lines(axes[0], sizes, gflops, MAIN_METHODS, mark=10)
    log2_size_axis(axes[0], sizes)
    nature_axes(axes[0], grid=True)
    axes[0].set_xlabel("Square size $N$")
    axes[0].set_ylabel("Throughput (GFLOP/s)")
    ymax = max(gflops[m][s] for m in MAIN_METHODS if m in gflops for s in sizes if s in gflops[m])
    axes[0].set_ylim(0, nice_upper(ymax))
    axes[0].legend(loc="best", frameon=False)
    # Relative to Ours median in-window as stability reference
    ours_vals = [gflops["ours"][s] for s in sizes]
    ref = statistics.median(ours_vals)
    axes[0].axhline(ref, color=ACCENT["parity"], lw=0.7, ls=(0, (2, 1.5)), zorder=1)
    axes[0].text(
        0.02,
        0.98,
        f"Ours median in window = {ref:.0f} GFLOP/s",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color="#4D4D4D",
    )
    panel_label(axes[0], "a")

    ax = axes[1]
    # Normalize each method by its own value at the first size ≥ floor (or first available)
    base_size = sizes[0]
    for method in MAIN_METHODS:
        if method not in gflops or base_size not in gflops[method]:
            continue
        base = gflops[method][base_size]
        if base <= 0:
            continue
        xs = [s for s in sizes if s in gflops[method]]
        ys = [gflops[method][s] / base for s in xs]
        kw = method_plot_kwargs(method)
        idxs = mark_indices(len(xs), 10)
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
    parity_line(ax, y=1.0)
    log2_size_axis(ax, sizes)
    nature_axes(ax, grid=True)
    ax.set_xlabel("Square size $N$")
    ax.set_ylabel(f"GFLOP/s relative to $N$={base_size}")
    ax.legend(loc="best", frameon=False)
    panel_label(ax, "b")
    hi = max(sizes)
    fig.suptitle(
        f"Large-N stability · {threads} threads · $N$={sizes[0]}…{hi}",
        fontsize=8,
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, figures, f"v2-gemm-large-stability-t{threads}")


def plot_gemm_ablation(rows: list[dict], figures: Path, *, threads: int) -> None:
    speedups = _paired_speedup_curve(rows, threads=threads, baselines=("sme-fixed", "generic"))
    gflops = _series_by_size(rows, threads=threads, metric="gflops")
    sizes = sorted(speedups.get("sme-fixed", {}))
    if not sizes:
        return
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE_TALL)
    _plot_lines(axes[0], sizes, gflops, ABLATION_METHODS)
    log2_size_axis(axes[0], sizes)
    nature_axes(axes[0], grid=True)
    axes[0].set_xlabel("Square size $N$")
    axes[0].set_ylabel("Throughput (GFLOP/s)")
    axes[0].legend(loc="upper left", frameon=False)
    panel_label(axes[0], "a")

    ax = axes[1]
    ys = [speedups["sme-fixed"][s] for s in sizes]
    ax.fill_between(
        sizes,
        1.0,
        ys,
        where=[y >= 1 for y in ys],
        color=ACCENT["ablation_fill"],
        alpha=0.55,
        interpolate=True,
    )
    ax.fill_between(
        sizes, 1.0, ys, where=[y < 1 for y in ys], color="#FBB4B9", alpha=0.45, interpolate=True
    )
    ax.plot(
        sizes,
        ys,
        color=COLORS["ours"],
        lw=1.6,
        marker="D",
        ms=3.8,
        markevery=mark_indices(len(sizes), 8),
        label="Ours / Fixed SME",
    )
    if speedups.get("generic"):
        gsz = sorted(speedups["generic"])
        ax.plot(
            gsz,
            [speedups["generic"][s] for s in gsz],
            color=COLORS["generic"],
            lw=1.1,
            ls=(0, (1.2, 1.2)),
            marker="o",
            ms=3.2,
            markevery=mark_indices(len(gsz), 8),
            label="Ours / TritonCPU",
        )
    parity_line(ax)
    log2_size_axis(ax, sizes)
    nature_axes(ax, grid=True)
    ax.set_xlabel("Square size $N$")
    ax.set_ylabel("Paired speedup (×)")
    ax.set_ylim(0, nice_upper(max(ys + [1.2])))
    ax.legend(loc="best", frameon=False)
    panel_label(ax, "b")
    gm = _geom_mean(ys)
    fig.suptitle(
        f"Ablation · schedule selection · {threads} threads · geom. mean Ours/Fixed SME = {gm:.2f}×",
        fontsize=8,
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, figures, f"v2-gemm-ablation-t{threads}")


def _metric_series(
    rows: list[dict],
    *,
    threads: int,
    x_key: str,
    metric_key: str,
    scale: float = 1.0,
) -> tuple[list[int], dict[str, list[float]]]:
    by = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if int(row["threads"]) != threads:
            continue
        by[int(float(row[x_key]))][row["method"]].append(float(row[metric_key]) * scale)
    xs = sorted(by)
    series = {m: [] for m in METHODS_ALL}
    for x in xs:
        for m in METHODS_ALL:
            series[m].append(_median(by[x][m]) if m in by[x] else float("nan"))
    return xs, series


METHODS_ALL = ("generic", "sme-fixed", "ours", "pytorch")


def plot_operator_lines(
    rows: list[dict],
    figures: Path,
    *,
    stem: str,
    title: str,
    threads: int,
    x_key: str,
    metric_key: str,
    ylabel: str,
    xlabel: str,
    methods=MAIN_METHODS,
    scale: float = 1.0,
    log_x: bool = True,
) -> None:
    xs, series = _metric_series(
        rows, threads=threads, x_key=x_key, metric_key=metric_key, scale=scale
    )
    if len(xs) < 2:
        return
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    for method in methods:
        ys = series[method]
        if all(not math.isfinite(y) for y in ys):
            continue
        kw = method_plot_kwargs(method)
        idxs = mark_indices(len(xs), 8)
        ax.plot(
            xs,
            ys,
            color=kw["color"],
            ls=kw["linestyle"],
            lw=kw["linewidth"],
            zorder=kw["zorder"],
            label=kw["label"],
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
    if log_x:
        log2_size_axis(ax, xs)
    nature_axes(ax, grid=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    finite = [y for m in methods for y in series[m] if math.isfinite(y)]
    ax.set_ylim(0, nice_upper(max(finite) if finite else 1.0))
    ax.legend(loc="best", frameon=False)
    ax.set_title(f"{title} · {threads} threads")
    fig.tight_layout()
    save_figure(fig, figures, stem)


def plot_speedup_lines(
    entries: list[dict],
    figures: Path,
    *,
    stem: str,
    title: str,
    threads: int,
    x_values: list[int] | None = None,
) -> None:
    if not entries:
        return
    xs = x_values if x_values is not None else list(range(1, len(entries) + 1))
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    for baseline in ("generic", "sme-fixed", "pytorch"):
        ys = [e.get(baseline, float("nan")) for e in entries]
        st = SPEEDUP_STYLE[baseline]
        ax.plot(
            xs,
            ys,
            color=st["color"],
            ls=st["ls"],
            lw=st["lw"],
            marker=st["marker"],
            ms=3.4,
            markevery=mark_indices(len(xs), 8),
            label=st["label"],
        )
    parity_line(ax)
    if x_values is not None and min(xs) > 0 and max(xs) / max(min(xs), 1) >= 4:
        log2_size_axis(ax, xs)
    nature_axes(ax, grid=True)
    ax.set_xlabel("Scale")
    ax.set_ylabel("Paired speedup of Ours (×)")
    finite = [
        e[b]
        for e in entries
        for b in ("generic", "sme-fixed", "pytorch")
        if b in e and math.isfinite(e[b])
    ]
    ax.set_ylim(0, nice_upper(max(finite) if finite else 1.5))
    ax.legend(loc="best", frameon=False, ncol=3)
    ax.set_title(f"{title} · {threads} threads")
    fig.tight_layout()
    save_figure(fig, figures, stem)


def _case_speedups(rows: list[dict], *, threads: int, latency_key: str, case_keys: list[str]):
    by = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if int(row["threads"]) != threads:
            continue
        key = tuple(int(row[k]) if str(row[k]).lstrip("-").isdigit() else row[k] for k in case_keys)
        by[key][row["method"]].append(float(row[latency_key]))
    result = []
    for key, methods in sorted(by.items()):
        if "ours" not in methods:
            continue
        ours = _median(methods["ours"])
        entry = {"case": key, "ours_ms": ours}
        for baseline in ("generic", "sme-fixed", "pytorch"):
            if baseline in methods and ours > 0:
                entry[baseline] = _median(methods[baseline]) / ours
        result.append(entry)
    return result


def plot_attention_phase_bars(
    phases: list[dict], figures: Path, *, threads: int, length: int
) -> None:
    """Bars are appropriate here: same sequence, different pipeline steps."""
    rows = [
        r
        for r in phases
        if int(r["threads"]) == threads
        and int(r.get("sequence_length", r.get("length", -1))) == length
    ]
    if not rows:
        return
    # Expect phase column
    if "phase" not in rows[0]:
        return
    by = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["phase"] not in PHASES:
            continue
        by[row["phase"]][row["method"]].append(
            float(row.get("latency_ms", float(row["median_ns"]) / 1e6))
        )
    if not by:
        return
    x = np.arange(len(PHASES))
    width = 0.2
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    for i, method in enumerate(METHODS_ALL):
        ys = [_median(by[p][method]) if method in by[p] else 0.0 for p in PHASES]
        ax.bar(
            x + (i - 1.5) * width,
            ys,
            width,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.3,
            label=LABELS[method],
        )
    nature_axes(ax, grid=True)
    ax.set_xticks(x)
    ax.set_xticklabels(list(PHASES))
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Attention phase")
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=6)
    ax.set_title(f"Attention phase latency · L={length} · {threads} threads")
    fig.tight_layout()
    save_figure(fig, figures, f"v2-attention-phases-t{threads}-l{length}")


def plot_summary(figures: Path, operator_gm: dict[str, dict[str, float]]) -> None:
    """Cleveland / slope summary — no heatmaps; no case-scale bars."""
    if not operator_gm:
        return
    ops = list(operator_gm.keys())
    baselines = ["generic", "sme-fixed", "pytorch"]
    base_labels = ["vs TritonCPU", "vs Fixed SME", "vs PyTorch"]

    fig, axes = plt.subplots(1, 2, figsize=FIG_SUMMARY, gridspec_kw={"width_ratios": [1.05, 1.0]})

    # (a) Cleveland-style dot plot
    ax = axes[0]
    y = np.arange(len(ops))
    for i, (b, lab) in enumerate(zip(baselines, base_labels)):
        xs = [operator_gm[op].get(b, float("nan")) for op in ops]
        ax.scatter(
            xs,
            y + (i - 1) * 0.12,
            s=28,
            color=COLORS[b],
            label=lab,
            zorder=3,
            edgecolors="white",
            linewidths=0.4,
        )
    for yi, op in enumerate(ops):
        vals = [operator_gm[op].get(b, float("nan")) for b in baselines]
        finite = [v for v in vals if math.isfinite(v)]
        if len(finite) >= 2:
            ax.plot([min(finite), max(finite)], [yi, yi], color="#D0D0D0", lw=0.8, zorder=1)
    parity_line(ax)
    nature_axes(ax, grid=True)
    ax.set_yticks(y)
    ax.set_yticklabels(ops)
    ax.set_xlabel("Geom. mean Ours / baseline (×)")
    ax.legend(loc="lower right", frameon=False, fontsize=6)
    ax.set_title("Per-operator paired effect")
    panel_label(ax, "a", x=-0.12)

    # (b) Slope chart across operators (same aggregates, line encoding)
    ax = axes[1]
    x = np.arange(len(ops))
    for b, lab in zip(baselines, base_labels):
        ys = [operator_gm[op].get(b, float("nan")) for op in ops]
        ax.plot(
            x,
            ys,
            color=COLORS[b],
            marker="o" if b == "generic" else ("s" if b == "sme-fixed" else "^"),
            ms=4.2,
            lw=1.25,
            ls="-" if b == "sme-fixed" else (0, (2.2, 1.2)),
            label=lab,
            zorder=3 if b != "generic" else 2,
        )
    parity_line(ax)
    nature_axes(ax, grid=True)
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=20, ha="right")
    ax.set_ylabel("Geom. mean Ours / baseline (×)")
    ax.legend(loc="best", frameon=False, fontsize=6)
    ax.set_title("Cross-operator profile")
    panel_label(ax, "b", x=-0.12)
    fig.suptitle(
        "Where schedule selection helps — and where Accelerate still wins", fontsize=8, y=1.01
    )
    fig.tight_layout()
    save_figure(fig, figures, "v2-summary-overview")

    # Workload-class: slope dots only
    native = [op for op in ops if op in ("GEMM", "Linear", "BMM", "Attention", "SwiGLU")]
    exotic = [op for op in ops if op in ("KAN",)]
    members = native + exotic
    if members:
        fig, ax = plt.subplots(figsize=FIG_DOUBLE)
        x = np.arange(len(members))
        for b, lab in zip(baselines, base_labels):
            ys = [operator_gm[op].get(b, float("nan")) for op in members]
            ax.plot(
                x,
                ys,
                color=COLORS[b],
                marker="o" if b == "generic" else ("s" if b == "sme-fixed" else "^"),
                ms=4.5,
                lw=1.3,
                label=lab,
            )
        if native and exotic:
            ax.axvline(len(native) - 0.5, color="#CCCCCC", lw=0.8, ls=(0, (2, 1.5)))
            ax.text(len(native) / 2 - 0.5, ax.get_ylim()[1] if False else 0, "", fontsize=1)
            ymax = nice_upper(
                max(
                    operator_gm[op].get(b, 0)
                    for op in members
                    for b in baselines
                    if math.isfinite(operator_gm[op].get(b, float("nan")))
                )
            )
            ax.set_ylim(0, ymax)
            ax.text(
                (len(native) - 1) / 2,
                ymax * 0.95,
                "BLAS-native / composed",
                ha="center",
                fontsize=6,
                color="#4D4D4D",
            )
            ax.text(
                len(native) + (len(exotic) - 1) / 2,
                ymax * 0.95,
                "BLAS-hostile",
                ha="center",
                fontsize=6,
                color="#4D4D4D",
            )
        parity_line(ax)
        nature_axes(ax, grid=True)
        ax.set_xticks(x)
        ax.set_xticklabels(members)
        ax.set_ylabel("Geom. mean speedup of Ours (×)")
        ax.legend(loc="best", frameon=False)
        ax.set_title("Workload class profile")
        fig.tight_layout()
        save_figure(fig, figures, "v2-summary-workload-class")


def plot_transfer_ablation(figures: Path, micro_gm: float, e2e: dict[str, float]) -> None:
    if not math.isfinite(micro_gm) or not e2e:
        return
    ops = list(e2e.keys())
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    xs = np.arange(len(ops) + 1)
    ys = [micro_gm] + [e2e[o] for o in ops]
    colors = [OPERATOR_COLORS.get("GEMM", COLORS["ours"])] + [
        OPERATOR_COLORS.get(o, "#999999") for o in ops
    ]
    labels = ["GEMM\n(micro)"] + ops
    ax.plot(xs, ys, color=COLORS["ours"], lw=1.4, marker="D", ms=5, zorder=3)
    ax.scatter(xs, ys, c=colors, s=36, zorder=4, edgecolors="white", linewidths=0.5)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(
            f"{y:.2f}×",
            (x, y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=5.5,
            color="#4D4D4D",
        )
    parity_line(ax)
    nature_axes(ax, grid=True)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Ours / Fixed SME (geom. mean ×)")
    ax.set_title("Transfer ablation · schedule gain dilutes in composites")
    ax.set_ylim(0, nice_upper(max(ys + [1.2])))
    fig.tight_layout()
    save_figure(fig, figures, "v2-ablation-transfer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "output"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # Drop obsolete heatmap artifacts
    for obsolete in figures.glob("v2-summary-heatmap.*"):
        obsolete.unlink(missing_ok=True)

    gemm_rows, threads = _resolve_gemm(output)
    operator_gm: dict[str, dict[str, float]] = {}
    transfer: dict[str, float] = {}

    if gemm_rows:
        plot_gemm_main(gemm_rows, figures, threads=threads)
        plot_gemm_large_stability(gemm_rows, figures, threads=threads, floor=2048)
        plot_gemm_ablation(gemm_rows, figures, threads=threads)
        speed = _paired_speedup_curve(gemm_rows, threads=threads)
        operator_gm["GEMM"] = {b: _geom_mean(list(d.values())) for b, d in speed.items() if d}
        transfer_micro = operator_gm["GEMM"].get("sme-fixed", float("nan"))
    else:
        transfer_micro = float("nan")

    linear = _read(output / "linear.csv")
    if linear:
        plot_operator_lines(
            linear,
            figures,
            stem=f"v2-linear-t{threads}",
            title="Linear (X@W)",
            threads=threads,
            x_key="tokens",
            metric_key="latency_ms",
            ylabel="Latency (ms)",
            xlabel="Tokens",
        )
        entries = _case_speedups(
            linear,
            threads=threads,
            latency_key="latency_ms",
            case_keys=["tokens", "in_features", "out_features"],
        )
        plot_speedup_lines(
            entries,
            figures,
            stem=f"v2-linear-speedup-t{threads}",
            title="Linear speedup",
            threads=threads,
            x_values=[e["case"][0] for e in entries],
        )
        operator_gm["Linear"] = {
            b: _geom_mean([e[b] for e in entries if b in e])
            for b in ("generic", "sme-fixed", "pytorch")
        }
        if "sme-fixed" in operator_gm["Linear"]:
            transfer["Linear"] = operator_gm["Linear"]["sme-fixed"]

    bmm = _read(output / "bmm.csv")
    if bmm:
        plot_operator_lines(
            bmm,
            figures,
            stem=f"v2-bmm-t{threads}",
            title="BMM",
            threads=threads,
            x_key="m",
            metric_key="latency_ms",
            ylabel="Latency (ms)",
            xlabel="Tile size $M$",
        )
        entries = _case_speedups(
            bmm, threads=threads, latency_key="latency_ms", case_keys=["batch", "m", "n", "k"]
        )
        plot_speedup_lines(
            entries,
            figures,
            stem=f"v2-bmm-speedup-t{threads}",
            title="BMM speedup",
            threads=threads,
            x_values=[e["case"][1] for e in entries],
        )
        operator_gm["BMM"] = {
            b: _geom_mean([e[b] for e in entries if b in e])
            for b in ("generic", "sme-fixed", "pytorch")
        }
        if "sme-fixed" in operator_gm["BMM"]:
            transfer["BMM"] = operator_gm["BMM"]["sme-fixed"]

    attn = _read(output / "attention_results.csv")
    if attn:
        full = list(attn)
        if "phase" in full[0]:
            full = [r for r in attn if r["phase"] == "full"]
        key = "latency_ms" if "latency_ms" in full[0] else "median_ns"
        scale = 1.0 if key == "latency_ms" else 1e-6
        case_key = next((c for c in ("sequence_length", "length", "seq") if c in full[0]), None)
        if case_key is not None:
            plot_operator_lines(
                full,
                figures,
                stem=f"v2-attention-t{threads}",
                title="Attention (e2e)",
                threads=threads,
                x_key=case_key,
                metric_key=key,
                ylabel="Latency (ms)",
                xlabel="Sequence length",
                scale=scale,
            )
            by = defaultdict(lambda: defaultdict(list))
            for row in full:
                if int(row["threads"]) != threads:
                    continue
                by[int(row[case_key])][row["method"]].append(float(row[key]) * scale)
            entries = []
            for length, methods in sorted(by.items()):
                if "ours" not in methods:
                    continue
                ours = _median(methods["ours"])
                e = {"case": (length,), "ours_ms": ours}
                for b in ("generic", "sme-fixed", "pytorch"):
                    if b in methods and ours > 0:
                        e[b] = _median(methods[b]) / ours
                entries.append(e)
            plot_speedup_lines(
                entries,
                figures,
                stem=f"v2-attention-speedup-t{threads}",
                title="Attention speedup",
                threads=threads,
                x_values=[e["case"][0] for e in entries],
            )
            operator_gm["Attention"] = {
                b: _geom_mean([e[b] for e in entries if b in e])
                for b in ("generic", "sme-fixed", "pytorch")
            }
            if "sme-fixed" in operator_gm["Attention"]:
                transfer["Attention"] = operator_gm["Attention"]["sme-fixed"]
            # Phase bars only (step latency)
            phases = _read(output / "attention_phases.csv")
            if phases:
                lengths = sorted(
                    {
                        int(r.get("sequence_length", r.get("length", 0)))
                        for r in phases
                        if int(r["threads"]) == threads
                    }
                )
                for L in (lengths[len(lengths) // 2], lengths[-1]) if lengths else []:
                    if L:
                        plot_attention_phase_bars(phases, figures, threads=threads, length=L)

    swiglu = _read(output / "swiglu.csv")
    if swiglu:
        plot_operator_lines(
            swiglu,
            figures,
            stem=f"v2-swiglu-t{threads}",
            title="SwiGLU",
            threads=threads,
            x_key="tokens",
            metric_key="latency_ms",
            ylabel="Latency (ms)",
            xlabel="Tokens",
        )
        entries = _case_speedups(
            swiglu,
            threads=threads,
            latency_key="latency_ms",
            case_keys=["tokens", "width", "hidden"],
        )
        plot_speedup_lines(
            entries,
            figures,
            stem=f"v2-swiglu-speedup-t{threads}",
            title="SwiGLU speedup",
            threads=threads,
            x_values=[e["case"][0] for e in entries],
        )
        operator_gm["SwiGLU"] = {
            b: _geom_mean([e[b] for e in entries if b in e])
            for b in ("generic", "sme-fixed", "pytorch")
        }
        if "sme-fixed" in operator_gm["SwiGLU"]:
            transfer["SwiGLU"] = operator_gm["SwiGLU"]["sme-fixed"]

    kan = _read(output / "kan.csv")
    if kan:
        plot_operator_lines(
            kan,
            figures,
            stem=f"v2-kan-t{threads}",
            title="KAN (BLAS-hostile)",
            threads=threads,
            x_key="tokens",
            metric_key="latency_ms",
            ylabel="Latency (ms)",
            xlabel="Tokens $L$",
        )
        plot_operator_lines(
            kan,
            figures,
            stem=f"v2-kan-gflops-t{threads}",
            title="KAN projection throughput",
            threads=threads,
            x_key="tokens",
            metric_key="projection_gflops",
            ylabel="Projection GFLOP/s",
            xlabel="Tokens $L$",
        )
        entries = _case_speedups(
            kan,
            threads=threads,
            latency_key="latency_ms",
            case_keys=["tokens", "in_features", "out_features", "spline_intervals"],
        )
        plot_speedup_lines(
            entries,
            figures,
            stem=f"v2-kan-speedup-t{threads}",
            title="KAN speedup",
            threads=threads,
            x_values=[e["case"][0] for e in entries],
        )
        operator_gm["KAN"] = {
            b: _geom_mean([e[b] for e in entries if b in e])
            for b in ("generic", "sme-fixed", "pytorch")
        }
        if "sme-fixed" in operator_gm["KAN"]:
            transfer["KAN"] = operator_gm["KAN"]["sme-fixed"]

    plot_summary(figures, operator_gm)
    plot_transfer_ablation(figures, transfer_micro, transfer)

    if operator_gm:
        fig, ax = plt.subplots(figsize=FIG_DOUBLE)
        ops = list(operator_gm.keys())
        x = np.arange(len(ops))
        width = 0.28
        for i, b in enumerate(("generic", "pytorch")):
            ys = [operator_gm[op].get(b, float("nan")) for op in ops]
            ax.bar(
                x + (i - 0.5) * width,
                ys,
                width,
                color=COLORS[b],
                label=SPEEDUP_STYLE[b]["label"],
                edgecolor="white",
            )
        parity_line(ax)
        nature_axes(ax, grid=True)
        ax.set_xticks(x)
        ax.set_xticklabels(ops)
        ax.set_ylabel("Geom. mean Ours / baseline (×)")
        ax.legend(loc="best", frameon=False)
        ax.set_title(f"Main triad across operators · {threads} threads")
        fig.tight_layout()
        save_figure(fig, figures, "v2-summary-main-triad")

    print(f"Suite v2 figures → {figures}")
    for p in sorted(figures.glob("v2-*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
