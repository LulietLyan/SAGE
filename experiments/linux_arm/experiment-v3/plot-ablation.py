#!/usr/bin/env python3
"""Plot measured-schedule regret from ablation-summary.csv files."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for summary in args.summaries:
        with summary.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    rows.sort(key=lambda row: (row["topology"], int(row["m"])))

    labels = [f"{row['topology']} {row['m']}" for row in rows]
    ratios = [float(row["sage_over_best_latency"]) for row in rows]
    colors = ["#2f6f9f" if row["topology"] == "single" else "#c2743c" for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        if args.output.suffix.lower() != ".svg":
            raise SystemExit("matplotlib is required for raster/PDF output. Use an .svg output for the dependency-free fallback.")
        write_svg(args.output, rows, labels, ratios, colors)
        return

    fig, axis = plt.subplots(figsize=(8.2, 4.6))
    bars = axis.bar(labels, ratios, color=colors, width=0.66)
    axis.axhline(1.0, color="#333333", linewidth=1.0)
    axis.set_ylabel("SAGE latency / best measured latency")
    axis.set_xlabel("NUMA placement and matrix size")
    axis.set_ylim(0.0, max(1.45, max(ratios) * 1.22))
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    axis.set_axisbelow(True)
    for bar, row in zip(bars, rows):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{float(row['sage_over_best_latency']):.3f}x\n{row['best_measured']}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    if args.output.suffix.lower() != ".pdf":
        fig.savefig(args.output.with_suffix(".pdf"))


def write_svg(path, rows, labels, ratios, colors):
    width, height = 900, 480
    left, top, right, bottom = 90, 38, 24, 92
    plot_width, plot_height = width - left - right, height - top - bottom
    limit = max(1.45, max(ratios) * 1.22)
    baseline = top + plot_height
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#222} .small{font-size:12px} .axis{stroke:#333;stroke-width:1} .grid{stroke:#ddd;stroke-width:1}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="22" text-anchor="middle" font-size="16">Linux Arm v9 ablation regret</text>',
        f'<text x="18" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 18 {top + plot_height / 2})" class="small">SAGE latency / best measured latency</text>',
    ]
    for tick in (0.0, 0.5, 1.0, 1.5):
        y = baseline - tick / limit * plot_height
        pieces.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        pieces.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="small">{tick:.1f}</text>')
    pieces.append(f'<line x1="{left}" y1="{baseline}" x2="{width-right}" y2="{baseline}" class="axis"/>')
    bar_width = plot_width / max(1, len(rows)) * 0.62
    for index, (row, label, ratio, color) in enumerate(zip(rows, labels, ratios, colors)):
        center = left + plot_width * (index + 0.5) / len(rows)
        x = center - bar_width / 2
        y = baseline - ratio / limit * plot_height
        pieces.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{baseline-y:.1f}" fill="{color}"/>')
        pieces.append(f'<text x="{center:.1f}" y="{y-8:.1f}" text-anchor="middle" class="small">{ratio:.3f}x</text>')
        pieces.append(f'<text x="{center:.1f}" y="{baseline+20:.1f}" text-anchor="middle" class="small">{label}</text>')
        pieces.append(f'<text x="{center:.1f}" y="{baseline+36:.1f}" text-anchor="middle" class="small">{row["best_measured"]}</text>')
    pieces.append(f'<text x="{width/2}" y="{height-16}" text-anchor="middle" class="small">NUMA placement and matrix size. Label below each bar is the best measured schedule.</text>')
    pieces.append('</svg>')
    path.write_text("\n".join(pieces))


if __name__ == "__main__":
    main()
