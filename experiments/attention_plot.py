#!/usr/bin/env python3
"""Nature-style attention throughput, latency, and phase-share figures."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from plot_style import (
    FIG_BAR,
    FIG_SINGLE,
    LABELS,
    METHODS,
    PHASE_COLORS,
    PHASES,
    RC,
    log2_size_axis,
    mark_indices,
    method_plot_kwargs,
    nature_axes,
    nice_upper,
    save_figure,
)


def _read(path: Path) -> list[dict]:
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _median(rows: list[dict], metric: str) -> float:
    return float(statistics.median(float(row[metric]) for row in rows))


def plot(output: Path | str) -> None:
    output = Path(output)
    rows = _read(output / "attention_results.csv")
    phase_rows = _read(output / "attention_phases.csv")
    with plt.rc_context(RC):
        for thread in sorted({int(row["threads"]) for row in rows}):
            lengths = sorted(
                {int(row["sequence_length"]) for row in rows if int(row["threads"]) == thread}
            )
            marks = mark_indices(len(lengths), 5)
            for metric, ylabel in (
                ("gflops", "Throughput (GFLOP/s)"),
                ("latency_ms", "Latency (ms)"),
            ):
                fig, ax = plt.subplots(figsize=FIG_SINGLE)
                ymax = 0.0
                for method in METHODS:
                    vals = [
                        _median(
                            [
                                row
                                for row in rows
                                if int(row["threads"]) == thread
                                and int(row["sequence_length"]) == length
                                and row["method"] == method
                            ],
                            metric,
                        )
                        for length in lengths
                    ]
                    ymax = max(ymax, max(vals) if vals else 0.0)
                    ax.plot(lengths, vals, markevery=marks, **method_plot_kwargs(method))
                ax.set_ylim(0, nice_upper(ymax))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                log2_size_axis(ax, lengths)
                ax.set_xlabel("Sequence length $L$")
                ax.set_ylabel(ylabel)
                nature_axes(ax)
                ax.legend(loc="upper left", fontsize=6)
                fig.suptitle(f"Scaled dot-product attention  ·  {thread} thr.", fontsize=8, y=1.02)
                stem = f"attention-t{thread}-{'gflops' if metric == 'gflops' else 'latency'}"
                save_figure(fig, output, stem)

            for length in [n for n in lengths if n in (128, 256, 512, 1024)] or [lengths[-1]]:
                fig, ax = plt.subplots(figsize=FIG_BAR)
                y = list(range(len(METHODS)))
                left = [0.0] * len(METHODS)
                for phase, color in zip(PHASES, PHASE_COLORS):
                    vals = []
                    for method in METHODS:
                        vals.append(
                            _median(
                                [
                                    row
                                    for row in phase_rows
                                    if int(row["threads"]) == thread
                                    and int(row["sequence_length"]) == length
                                    and row["method"] == method
                                    and row["phase"] == phase
                                ],
                                "latency_ms",
                            )
                        )
                    ax.barh(
                        y,
                        vals,
                        0.58,
                        left=left,
                        color=color,
                        edgecolor="white",
                        linewidth=0.4,
                        label=phase,
                    )
                    left = [a + b for a, b in zip(left, vals)]
                ax.set_yticks(y, [LABELS[m] for m in METHODS])
                ax.invert_yaxis()
                ax.set_xlim(0, nice_upper(max(left)))
                ax.set_xlabel("Stage latency (ms)")
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                nature_axes(ax, grid=True)
                ax.legend(loc="lower right", fontsize=5.8, ncol=2)
                fig.suptitle(
                    f"Attention phase cost  ·  $L={length}$, {thread} thr.", fontsize=8, y=1.02
                )
                save_figure(fig, output, f"attention-phases-t{thread}-l{length}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "output"
    )
    plot(parser.parse_args().output)


if __name__ == "__main__":
    main()
