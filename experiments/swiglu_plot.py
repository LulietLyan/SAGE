#!/usr/bin/env python3
"""Nature-style SwiGLU latency, throughput, and speedup figures."""

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
    COLORS,
    FIG_SINGLE,
    LABELS,
    METHODS,
    RC,
    SPEEDUP_STYLE,
    nature_axes,
    nice_upper,
    parity_line,
    save_figure,
)


def _case_key(row: dict) -> tuple[int, int, int]:
    return int(row["tokens"]), int(row["width"]), int(row["hidden"])


def plot(output: Path | str) -> None:
    output = Path(output)
    path = output / "swiglu.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("swiglu.csv is empty")
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
            labels = [f"C{i + 1}" for i in range(len(cases))]
            y = list(range(len(cases)))
            height = 0.16
            offsets = [-(1.5 * height), -0.5 * height, 0.5 * height, 1.5 * height]
            for metric, xlabel, stem in (
                ("latency_ms", "End-to-end latency (ms)", f"swiglu-t{thread}-latency"),
                (
                    "projection_gflops",
                    "Projection throughput (GFLOP/s)",
                    f"swiglu-t{thread}-gflops",
                ),
            ):
                fig_h = max(70, 12 * len(cases) + 36)
                fig, ax = plt.subplots(figsize=(120 / 25.4, fig_h / 25.4))
                for method, offset in zip(METHODS, offsets):
                    values = []
                    for case in cases:
                        match = next(
                            row
                            for row in selected
                            if row["method"] == method and _case_key(row) == case
                        )
                        values.append(float(match[metric]))
                    ax.barh(
                        [v + offset for v in y],
                        values,
                        height=height,
                        color=COLORS[method],
                        edgecolor="white",
                        linewidth=0.35,
                        label=LABELS[method],
                    )
                ax.set_yticks(y, labels)
                ax.invert_yaxis()
                ax.set_xlabel(xlabel)
                ax.set_xlim(0, nice_upper(max(bar.get_width() for bar in ax.patches)))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                nature_axes(ax, grid=True)
                ax.legend(loc="lower right", fontsize=5.8)
                key_txt = "  ".join(
                    f"C{i + 1}: T={c[0]},d={c[1]},dff={c[2]}" for i, c in enumerate(cases)
                )
                fig.text(0.01, 0.01, key_txt, fontsize=4.8, color="#767676", ha="left")
                fig.suptitle(f"SwiGLU  ·  {thread} thr.", fontsize=8, y=0.995)
                save_figure(fig, output, stem)

            fig, ax = plt.subplots(figsize=FIG_SINGLE)
            xs = list(range(len(cases)))
            fixed, torch_r = [], []
            for case in cases:
                by = {
                    row["method"]: float(row["latency_ms"])
                    for row in selected
                    if _case_key(row) == case
                }
                fixed.append(by["sme-fixed"] / by["ours"])
                torch_r.append(by["pytorch"] / by["ours"])
            for baseline, ys in (("sme-fixed", fixed), ("pytorch", torch_r)):
                st = SPEEDUP_STYLE[baseline]
                ax.plot(
                    xs,
                    ys,
                    color=st["color"],
                    linestyle=st["ls"],
                    linewidth=st["lw"],
                    marker=st["marker"],
                    markersize=3.4,
                    label=st["label"],
                )
            parity_line(ax)
            ax.set_ylim(0, nice_upper(max(fixed + torch_r), pad=1.12))
            ax.set_xticks(xs, labels)
            ax.set_xlabel("SwiGLU case")
            ax.set_ylabel("Paired speedup")
            nature_axes(ax)
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
            fig.suptitle(f"SwiGLU schedule gain  ·  {thread} thr.", fontsize=8, y=1.02)
            save_figure(fig, output, f"swiglu-t{thread}-speedup")

    for thread in threads:
        for stem in (
            f"swiglu-t{thread}-latency",
            f"swiglu-t{thread}-gflops",
            f"swiglu-t{thread}-speedup",
        ):
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
