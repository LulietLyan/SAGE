#!/usr/bin/env python3
"""Nature-style KANLinear figures: line charts for scale sweeps.

Bar charts are reserved for multi-step latency breakdowns elsewhere
(Attention phases).  KAN reports end-to-end latency and projection GFLOP/s
as lines vs case scale (tokens).
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from plot_style import (
    FIG_DOUBLE,
    FIG_SINGLE,
    MAIN_METHODS,
    METHODS,
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


def _case_key(row: dict) -> tuple[int, int, int, int]:
    return (
        int(row["tokens"]),
        int(row["in_features"]),
        int(row["out_features"]),
        int(row["spline_intervals"]),
    )


def _plot_method_lines(ax, xs, series: dict[str, list[float]], methods=METHODS) -> None:
    for method in methods:
        if method not in series:
            continue
        ys = series[method]
        kw = method_plot_kwargs(method)
        idxs = mark_indices(len(xs), 7)
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


def plot(output: Path | str) -> None:
    output = Path(output)
    with (output / "kan.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("kan.csv is empty")
    threads = sorted({int(row["threads"]) for row in rows})
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        for thread in threads:
            selected = [row for row in rows if int(row["threads"]) == thread]
            cases = []
            for row in selected:
                key = _case_key(row)
                if key not in cases:
                    cases.append(key)
            cases.sort()
            # Prefer tokens as continuous x; fall back to case index if duplicates.
            tokens = [c[0] for c in cases]
            use_tokens = len(tokens) == len(set(tokens))
            xs = tokens if use_tokens else list(range(1, len(cases) + 1))
            xlabel = "Tokens $L$" if use_tokens else "Case index"

            series_lat = {m: [] for m in METHODS}
            series_gflops = {m: [] for m in METHODS}
            for case in cases:
                for method in METHODS:
                    match = next(
                        row
                        for row in selected
                        if row["method"] == method and _case_key(row) == case
                    )
                    series_lat[method].append(float(match["latency_ms"]))
                    series_gflops[method].append(float(match["projection_gflops"]))

            # Main triad latency + GFLOP/s
            fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)
            _plot_method_lines(axes[0], xs, series_lat, methods=MAIN_METHODS)
            _plot_method_lines(axes[1], xs, series_gflops, methods=MAIN_METHODS)
            for ax, ylabel in (
                (axes[0], "End-to-end latency (ms)"),
                (axes[1], "Projection throughput (GFLOP/s)"),
            ):
                if use_tokens:
                    log2_size_axis(ax, xs)
                else:
                    ax.set_xticks(xs)
                nature_axes(ax, grid=True)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ymax = (
                    max(max(series_lat[m]) for m in MAIN_METHODS)
                    if ax is axes[0]
                    else max(max(series_gflops[m]) for m in MAIN_METHODS)
                )
                ax.set_ylim(0, nice_upper(ymax))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            axes[0].legend(loc="upper left", frameon=False, fontsize=6)
            panel_label(axes[0], "a")
            panel_label(axes[1], "b")
            fig.suptitle(
                f"KANLinear · {thread} threads · TritonCPU / Ours / PyTorch", fontsize=8, y=1.02
            )
            fig.tight_layout()
            save_figure(fig, output, f"kan-t{thread}-latency")
            # also alias gflops stem for legacy copies
            fig2, ax2 = plt.subplots(figsize=FIG_SINGLE)
            _plot_method_lines(ax2, xs, series_gflops, methods=MAIN_METHODS)
            if use_tokens:
                log2_size_axis(ax2, xs)
            nature_axes(ax2, grid=True)
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel("Projection throughput (GFLOP/s)")
            ax2.set_ylim(0, nice_upper(max(max(series_gflops[m]) for m in MAIN_METHODS)))
            ax2.legend(loc="best", frameon=False, fontsize=6)
            ax2.set_title(f"KANLinear throughput · {thread} thr.")
            fig2.tight_layout()
            save_figure(fig2, output, f"kan-t{thread}-gflops")

            # Ablation speedup lines
            fig, ax = plt.subplots(figsize=FIG_SINGLE)
            fixed, torch_r, generic_r = [], [], []
            for case in cases:
                by = {
                    row["method"]: float(row["latency_ms"])
                    for row in selected
                    if _case_key(row) == case
                }
                fixed.append(by["sme-fixed"] / by["ours"])
                torch_r.append(by["pytorch"] / by["ours"])
                generic_r.append(by["generic"] / by["ours"])
            for baseline, ys in (
                ("sme-fixed", fixed),
                ("generic", generic_r),
                ("pytorch", torch_r),
            ):
                st = SPEEDUP_STYLE[baseline]
                ax.plot(
                    xs,
                    ys,
                    color=st["color"],
                    linestyle=st["ls"],
                    linewidth=st["lw"],
                    marker=st["marker"],
                    markersize=3.4,
                    markevery=mark_indices(len(xs), 7),
                    label=st["label"],
                )
            parity_line(ax)
            if use_tokens:
                log2_size_axis(ax, xs)
            else:
                ax.set_xticks(xs)
            ax.set_ylim(0, nice_upper(max(fixed + torch_r + generic_r), pad=1.12))
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Paired speedup of Ours (×)")
            nature_axes(ax, grid=True)
            ax.legend(loc="best", fontsize=6)
            gm = math.exp(statistics.mean(math.log(v) for v in fixed))
            ax.text(
                0.98,
                0.04,
                f"Geom. mean vs Fixed SME = {gm:.2f}×",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.8,
                color="#4D4D4D",
            )
            fig.suptitle(f"KANLinear schedule gain · {thread} thr.", fontsize=8, y=1.02)
            save_figure(fig, output, f"kan-t{thread}-speedup")

    for thread in threads:
        for stem in (f"kan-t{thread}-latency", f"kan-t{thread}-gflops", f"kan-t{thread}-speedup"):
            for ext in ("pdf", "svg", "png"):
                src = output / f"{stem}.{ext}"
                if src.is_file():
                    shutil.copy2(src, figures / f"fig-{stem}.{ext}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "output"
    )
    plot(parser.parse_args().output)


if __name__ == "__main__":
    main()
