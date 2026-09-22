#!/usr/bin/env python3
"""Diagnostic figures for GEMM stride, scaling, and candidate-rank experiments.

The script is deliberately PDF-only: source CSVs and this script remain next to
figures so every plotted value is traceable without retaining raster previews.
"""

from __future__ import annotations
import argparse
import csv
import statistics
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.transforms import Bbox
from plot_style import (
    LABELS,
    RC,
    nature_axes,
    nice_upper,
    panel_label,
)

plt.rcParams.update(RC)

METHODS = ("generic", "sme-fixed", "ours", "pytorch")
CLIFF_GUIDES = (2048, 4096, 6144)

# Soft / fresh palette — still separable in print and grayscale-adjacent.
CLIFF_COLORS = {
    "generic": "#9AA4B2",
    "sme-fixed": "#3CBBB0",
    "ours": "#5B8DEF",
    "pytorch": "#F2A65A",
}
CLIFF_MARKERS = {"generic": "o", "sme-fixed": "s", "ours": "D", "pytorch": "^"}
CLIFF_LINESTYLES = {
    "generic": (0, (1.1, 1.15)),
    "sme-fixed": (0, (4.2, 1.6)),
    "ours": "-",
    "pytorch": (0, (2.4, 1.15)),
}
CLIFF_LINEWIDTHS = {"generic": 1.2, "sme-fixed": 1.4, "ours": 2.1, "pytorch": 1.5}
CLIFF_MARKERSIZES = {"generic": 3.4, "sme-fixed": 3.5, "ours": 4.2, "pytorch": 3.8}
CLIFF_ZORDERS = {"generic": 2, "sme-fixed": 3, "ours": 5, "pytorch": 4}
CLIFF_BAND = "#A8C5F0"


def rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def med(rows_, key, value="gflops"):
    groups = defaultdict(list)
    for r in rows_:
        groups[key(r)].append(float(r[value]))
    return {k: statistics.median(v) for k, v in groups.items()}


def save_pdf(
    fig,
    path,
    *,
    preview_png: Path | None = None,
    pad_inches: float = 0.03,
    pad_right: float = 0.0,
    pad_top: float = 0.0,
):
    """Save with optional extra figure margin on the open (right/top) sides.

    Axis data limits are left alone — whitespace is paper margin only.
    """
    fig.canvas.draw()
    # Matplotlib Agg: get_tightbbox() is already in inches (do not re-scale by dpi).
    tight = fig.get_tightbbox(fig.canvas.get_renderer()).frozen()
    bbox = Bbox.from_extents(
        tight.x0 - pad_inches,
        tight.y0 - pad_inches,
        tight.x1 + pad_inches + pad_right,
        tight.y1 + pad_inches + pad_top,
    )
    fig.savefig(path, bbox_inches=bbox)
    if preview_png is not None:
        fig.savefig(preview_png, dpi=240, bbox_inches=bbox)
    plt.close(fig)


@contextmanager
def _cliff_rc():
    """Times New Roman body + STIX math — classic journal typography."""
    overrides = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.55,
        "ytick.minor.width": 0.55,
        "xtick.major.size": 4.2,
        "ytick.major.size": 4.2,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(overrides):
        yield


def _cliff_xticks(sizes: list[int]) -> list[int]:
    """Equally spaced major ticks spanning the measured N range."""
    lo, hi = min(sizes), max(sizes)
    # Prefer a clean 1024 stride that lands on both endpoints of this sweep.
    step = 1024
    if (hi - lo) % step == 0:
        return list(range(lo, hi + 1, step))
    n = 7
    return [int(round(lo + i * (hi - lo) / (n - 1))) for i in range(n)]


def _padded_xlim(lo: float, hi: float, *, frac: float = 0.045) -> tuple[float, float]:
    """Small axis headroom so endpoint markers are not flush against the spine."""
    pad = (hi - lo) * frac
    return lo - pad, hi + pad


def _journal_axes(ax) -> None:
    """Left+bottom spines only; outward ticks; layered y-grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#1A1A1A")
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(
        which="major",
        direction="out",
        length=4.0,
        width=0.8,
        colors="#1A1A1A",
        top=False,
        right=False,
        pad=3.5,
    )
    ax.tick_params(
        which="minor",
        direction="out",
        length=2.0,
        width=0.55,
        colors="#1A1A1A",
        top=False,
        right=False,
    )
    ax.yaxis.grid(True, which="major", color="#D8D8D8", linewidth=0.55, linestyle="-")
    ax.yaxis.grid(True, which="minor", color="#EEEEEE", linewidth=0.35, linestyle="-")
    ax.set_axisbelow(True)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))


def cliff_figure(root, stem="dense"):
    """Four-method throughput sweep — line panel only (no neighbour-ratio bars)."""
    data = rows(root / f"{stem}.csv")
    out = root / "figures"
    out.mkdir(exist_ok=True)
    by_method = {
        m: med([r for r in data if r["method"] == m], lambda r: int(r["m"])) for m in METHODS
    }
    sizes = sorted(by_method["ours"])

    with _cliff_rc():
        fig, ax = plt.subplots(1, 1, figsize=(183 / 25.4, 100 / 25.4))

        lo, hi = min(sizes), max(sizes)
        # Soft cliff bands on a linear axis (fixed-width windows).
        band = 90
        for x in CLIFF_GUIDES:
            if lo <= x <= hi:
                ax.axvspan(x - band, x + band, color=CLIFF_BAND, alpha=0.18, zorder=0, lw=0)
                ax.axvline(x, color="#B7C9E8", lw=0.75, ls=(0, (2.5, 1.8)), zorder=1, alpha=0.9)

        ymax = 0.0
        for method in METHODS:
            series = by_method.get(method)
            if not series:
                continue
            xs = [s for s in sizes if s in series]
            ys = [series[s] for s in xs]
            ymax = max(ymax, max(ys))
            color = CLIFF_COLORS[method]
            face = "white" if method == "generic" else color
            ax.plot(
                xs,
                ys,
                color=color,
                ls=CLIFF_LINESTYLES[method],
                lw=CLIFF_LINEWIDTHS[method],
                marker=CLIFF_MARKERS[method],
                markersize=CLIFF_MARKERSIZES[method],
                markerfacecolor=face,
                markeredgecolor=color,
                markeredgewidth=0.7,
                zorder=CLIFF_ZORDERS[method],
                label=LABELS[method],
                solid_capstyle="round",
                solid_joinstyle="round",
                clip_on=False,
            )

        # Linear x, with modest headroom so endpoint markers are fully visible.
        ax.set_xscale("linear")
        ax.set_xlim(*_padded_xlim(lo, hi))
        ticks = _cliff_xticks(sizes)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])

        _journal_axes(ax)
        ax.set_xlabel(r"Square GEMM size $N$", labelpad=6, fontsize=11)
        ax.xaxis.label.set_fontfamily("serif")
        ax.xaxis.label.set_fontname("Times New Roman")
        ax.xaxis.label.set_fontweight("bold")
        ax.set_ylabel(r"Throughput (GFLOP$\,s^{-1}$)", labelpad=7, fontsize=11)
        ax.yaxis.label.set_fontfamily("serif")
        ax.yaxis.label.set_fontname("Times New Roman")
        ax.yaxis.label.set_fontweight("bold")
        ax.set_ylim(0, nice_upper(ymax, pad=1.12))

        ax.legend(
            loc="upper right",
            frameon=False,
            handlelength=2.4,
            borderpad=0.2,
            labelspacing=0.32,
            handletextpad=0.5,
            borderaxespad=0.4,
        )

        fig.tight_layout(pad=0.35)
        # Extra paper margin on the open right/top edges — not axis extension.
        save_pdf(
            fig,
            out / "padding-cliff.pdf",
            preview_png=out / "_preview_padding-cliff.png",
            pad_inches=0.04,
            pad_right=0.22,
            pad_top=0.16,
        )


def scaling_figure(root, stem="results_comm"):
    data = rows(root / f"{stem}.csv")
    out = root / "figures"
    out.mkdir(exist_ok=True)
    sizes = sorted({int(r["m"]) for r in data})
    threads = sorted({int(r["threads"]) for r in data})
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(183 / 25.4, 82 / 25.4), gridspec_kw={"width_ratios": (1.15, 1)}
    )
    for n in sizes:
        vals = []
        for t in threads:
            rs = [
                r
                for r in data
                if int(r["m"]) == n and int(r["threads"]) == t and r["method"] == "ours"
            ]
            vals.append(statistics.median(float(r["gflops"]) for r in rs))
        ax.plot(threads, vals, marker="o", lw=1.25, ms=3.2, label=f"{n}²")
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    nature_axes(ax, grid=True)
    ax.set_xlabel("Thread budget")
    ax.set_ylabel("Ours throughput (GFLOP s$^{-1}$)")
    ax.legend(ncol=2, loc="lower right", fontsize=5.8)
    panel_label(ax, "a")
    for n in sizes:
        vals = []
        for t in threads:
            ours = [
                float(r["median_ns"])
                for r in data
                if int(r["m"]) == n and int(r["threads"]) == t and r["method"] == "ours"
            ]
            torch = [
                float(r["median_ns"])
                for r in data
                if int(r["m"]) == n and int(r["threads"]) == t and r["method"] == "pytorch"
            ]
            vals.append(statistics.median(torch) / statistics.median(ours))
        bx.plot(threads, vals, marker="o", lw=1.25, ms=3.2, label=f"{n}²")
    bx.set_xscale("log", base=2)
    bx.set_xticks(threads)
    bx.set_xticklabels([str(t) for t in threads])
    nature_axes(bx, grid=True)
    bx.axhline(1, color="#777", lw=0.6, ls=(0, (1.5, 1.3)))
    bx.set_xlabel("Thread budget")
    bx.set_ylabel("PyTorch / Ours latency")
    panel_label(bx, "b")
    fig.suptitle("Measured scaling under a fixed compiler profile", fontsize=8.2, y=0.995)
    fig.tight_layout(pad=0.5)
    save_pdf(fig, out / "thread-scaling.pdf")


def rank_figure(root, stem="rank"):
    """Policy-rank throughput sweep — same publication style as cliff_figure."""
    data = rows(root / f"{stem}.csv")
    out = root / "figures"
    out.mkdir(exist_ok=True)
    methods = sorted({r["method"] for r in data}, key=lambda x: int(x.split("-")[1]))
    sizes = sorted({int(r["size"]) for r in data})

    # Fresh sequential hierarchy: best rank strongest, weaker ranks softer.
    rank_colors = {
        "Rank-1": "#5B8DEF",
        "Rank-5": "#3CBBB0",
        "Rank-10": "#7BC8A4",
        "Rank-20": "#F2A65A",
        "Rank-40": "#9AA4B2",
    }
    rank_markers = {
        "Rank-1": "D",
        "Rank-5": "o",
        "Rank-10": "s",
        "Rank-20": "^",
        "Rank-40": "v",
    }
    rank_ls = {
        "Rank-1": "-",
        "Rank-5": (0, (4.2, 1.6)),
        "Rank-10": (0, (2.4, 1.15)),
        "Rank-15": (0, (1.1, 1.15)),
        "Rank-20": (0, (2.4, 1.15)),
        "Rank-40": (0, (1.1, 1.15)),
    }
    rank_lw = {"Rank-1": 2.1, "Rank-5": 1.45, "Rank-10": 1.35, "Rank-20": 1.3, "Rank-40": 1.2}
    rank_ms = {"Rank-1": 4.2, "Rank-5": 3.4, "Rank-10": 3.3, "Rank-20": 3.5, "Rank-40": 3.2}

    with _cliff_rc():
        fig, ax = plt.subplots(1, 1, figsize=(183 / 25.4, 100 / 25.4))
        lo, hi = min(sizes), max(sizes)
        ymax = 0.0

        for method in methods:
            ys = [
                statistics.median(
                    float(r["gflops"])
                    for r in data
                    if r["method"] == method and int(r["size"]) == n
                )
                for n in sizes
            ]
            ymax = max(ymax, max(ys))
            color = rank_colors.get(method, "#9AA4B2")
            ax.plot(
                sizes,
                ys,
                color=color,
                ls=rank_ls.get(method, "-"),
                lw=rank_lw.get(method, 1.3),
                marker=rank_markers.get(method, "o"),
                markersize=rank_ms.get(method, 3.2),
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.7,
                zorder=5 if method == "Rank-1" else 3,
                label=method.replace("-", " "),
                solid_capstyle="round",
                solid_joinstyle="round",
                clip_on=False,
            )

        ax.set_xscale("linear")
        ax.set_xlim(*_padded_xlim(lo, hi))
        # Equidistant ticks on the measured sizes (not on the padded limits).
        step = 512
        ticks = list(range(lo, hi + 1, step)) if (hi - lo) % step == 0 else _cliff_xticks(sizes)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])

        _journal_axes(ax)
        ax.set_xlabel(r"Square GEMM size $N$", labelpad=6, fontsize=11)
        ax.xaxis.label.set_fontfamily("serif")
        ax.xaxis.label.set_fontname("Times New Roman")
        ax.xaxis.label.set_fontweight("bold")
        ax.set_ylabel(r"Throughput (GFLOP$\,s^{-1}$)", labelpad=7, fontsize=11)
        ax.yaxis.label.set_fontfamily("serif")
        ax.yaxis.label.set_fontname("Times New Roman")
        ax.yaxis.label.set_fontweight("bold")
        ax.set_ylim(0, nice_upper(ymax, pad=1.12))

        ax.legend(
            loc="upper right",
            frameon=False,
            handlelength=2.4,
            borderpad=0.2,
            labelspacing=0.32,
            handletextpad=0.5,
            borderaxespad=0.4,
        )

        fig.tight_layout(pad=0.35)
        save_pdf(
            fig,
            out / "rank-throughput.pdf",
            preview_png=out / "_preview_rank-throughput.png",
            pad_inches=0.04,
            pad_right=0.22,
            pad_top=0.16,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--kind", choices=("cliff", "scaling", "rank", "all"), default="all")
    p.add_argument("--cliff-stem", default="dense")
    p.add_argument("--scaling-stem", default="results_comm")
    p.add_argument("--rank-stem", default="rank")
    a = p.parse_args()
    if a.kind in ("cliff", "all"):
        cliff_figure(a.root, a.cliff_stem)
    if a.kind in ("scaling", "all"):
        scaling_figure(a.root, a.scaling_stem)
    if a.kind in ("rank", "all"):
        rank_figure(a.root, a.rank_stem)
    print("PDF figures:", *sorted(str(p) for p in (a.root / "figures").glob("*.pdf")), sep="\n")


if __name__ == "__main__":
    main()
