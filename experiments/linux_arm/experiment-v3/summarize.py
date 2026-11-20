#!/usr/bin/env python3
"""Verify window counts and summarize paired GEMM results by placement."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from collections import defaultdict
import sys


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def aggregate(run):
    env = json.loads((run / "environment.json").read_text())
    assert env["status"] == "complete", f"Incomplete run {run}: {env['status']}"
    windows = defaultdict(list)
    checks = {}
    key_names = "topology threads process m n k method".split()
    for row in read(run / "samples.csv"):
        key = tuple(row[k] for k in key_names)
        assert float(row["ns_per_call"]) > 0 and int(row["repetitions"]) > 0
        assert math.isclose(float(row["ns_per_call"]), int(row["elapsed_ns"]) / int(row["repetitions"]))
        windows[key].append(row)
    for row in read(run / "checks.csv"):
        key = tuple(row[k] for k in key_names)
        assert key not in checks and row["passed"] == "True", row
        checks[key] = row
    expected = set()
    for job in env["jobs"]:
        for m, n, k in job["shapes"]:
            for method in job["methods"]:
                expected.add(tuple(map(str, (job["topology"], job["threads"], job["process"], m, n, k, method))))
    assert expected == windows.keys() == checks.keys()
    hosts = {c["hostname"] for c in checks.values()}
    assert len(hosts) == 1, hosts
    hostname = next(iter(hosts))
    by_shape, paired = defaultdict(list), defaultdict(dict)
    for key, rows in windows.items():
        assert len(rows) == env["rounds"], (key, len(rows))
        assert set(int(r["round"]) for r in rows) == set(range(env["rounds"]))
        topology, threads, process, m, n, k, method = key
        group = topology, threads, m, n, k
        value = median(float(row["ns_per_call"]) for row in rows)
        check = checks[key]
        by_shape[group + (method,)].append((value, check))
        paired[group][process, method] = value

    summary, ratios = [], []
    for key, values in sorted(by_shape.items()):
        info = dict(zip(("topology", "threads", "m", "n", "k", "method"), key))
        flops = 2 * math.prod(int(info[dim]) for dim in ("m", "n", "k"))
        process_times = [item[0] for item in values]
        center = median(process_times)
        summary.append(dict(hostname=hostname, **info, processes=len(values), median_ns=center,
                            min_process_ns=min(process_times), max_process_ns=max(process_times),
                            process_mad_ns=median(abs(x - center) for x in process_times),
                            logical_gflops=flops / center,
                            max_fp32_abs=max(float(c["fp32_max_abs"]) for _, c in values),
                            max_fp64_abs=max(float(c["fp64_max_abs"]) for _, c in values),
                            max_gamma_ratio=max(float(c["fp64_gamma_2k_ratio"]) for _, c in values)))
    for group, timings in sorted(paired.items()):
        methods = sorted({m for _, m in timings})
        for candidate in methods:
            for reference in methods:
                if candidate == reference:
                    continue
                values = [timings[p, reference] / timings[p, method]
                          for p, method in timings if method == candidate and (p, reference) in timings]
                ratios.append(dict(zip(("topology", "threads", "m", "n", "k"), group),
                                   candidate=candidate, reference=reference, pair_count=len(values),
                                   paired_speedup=median(values), min_ratio=min(values), max_ratio=max(values)))
    return summary, ratios


def plot(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    root = Path(__file__).resolve().parents[3]
    for style_root in (root / "experiments", Path(__file__).resolve().parents[1]):
        if (style_root / "plot_style.py").is_file():
            sys.path.insert(0, str(style_root))
            break
    from plot_style import FIG_DOUBLE_TALL, RC, nature_axes, save_figure

    colors = dict(generic="#5B6B7A", **{"sme-fixed": "#1B9E77"}, ours="#08519C",
                  openblas="#1D4ED8", kblas="#941F48", pytorch="#6535A6")
    markers = dict(generic="o", **{"sme-fixed": "s"}, ours="D", openblas="o", kblas="P", pytorch="^")
    available = sorted({r["threads"] for r in summary}, key=int)
    budgets = sorted({available[0], available[-1]}, key=int)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, len(budgets), figsize=FIG_DOUBLE_TALL, squeeze=False, sharey="row")
        for column, t in enumerate(budgets):
            for method in sorted({r["method"] for r in summary}):
                rows = sorted(
                    (r for r in summary if r["threads"] == t and r["m"] == r["n"] == r["k"] and r["method"] == method),
                    key=lambda r: int(r["m"]),
                )
                if not rows:
                    continue
                x = [int(r["m"]) for r in rows]
                times = [float(r["median_ns"]) / 1e6 for r in rows]
                low = [float(r["min_process_ns"]) / 1e6 for r in rows]
                high = [float(r["max_process_ns"]) / 1e6 for r in rows]
                color = colors.get(method, "#555555")
                marker = markers.get(method, "o")
                axes[0, column].plot(x, [float(r["logical_gflops"]) for r in rows], label=method,
                                      color=color, marker=marker, markersize=3.2, linewidth=1.2)
                axes[1, column].plot(x, times, label=method, color=color, marker=marker,
                                     markersize=3.2, linewidth=1.2)
                axes[1, column].fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
            for row_index, axis in enumerate(axes[:, column]):
                axis.set_xscale("log", base=2)
                axis.set_xticks([1024, 2048, 4096])
                axis.set_xticklabels(["1024", "2048", "4096"])
                axis.set_xlabel(r"Square dimension $M{=}N{=}K$")
                axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
                nature_axes(axis)
                axis.legend(loc="upper left" if row_index == 0 else "lower right", fontsize=6,
                            frameon=False, borderaxespad=0.3)
            axes[0, column].set_title(f"{t} thread{'s' if int(t) != 1 else ''}")
        axes[0, 0].set_ylabel("Logical GFLOP/s")
        axes[1, 0].set_ylabel("Full-call latency (ms)")
        fig.suptitle(f"FP32 GEMM · {summary[0]['topology']} NUMA", fontsize=8, y=1.01)
        save_figure(fig, path.parent, path.stem)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--figures", required=True, type=Path)
    a = p.parse_args()
    summary, paired = aggregate(a.run)
    write(a.data / (a.run.name + "-summary.csv"), summary)
    write(a.data / (a.run.name + "-paired.csv"), paired)
    for topology in sorted({r["topology"] for r in summary}):
        plot([r for r in summary if r["topology"] == topology],
             a.figures / (a.run.name + f"-{topology}.png"))


if __name__ == "__main__":
    main()
