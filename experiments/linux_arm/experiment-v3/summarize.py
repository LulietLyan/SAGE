#!/usr/bin/env python3
"""Verify window counts and summarize paired GEMM results by placement."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from collections import defaultdict


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
    import matplotlib.pyplot as plt

    colors = dict(generic="#64748b", **{"sme-fixed": "#d97706"}, ours="#047857",
                  openblas="#1d4ed8", kblas="#941f48", pytorch="#6535a6")
    available = sorted({r["threads"] for r in summary}, key=int)
    budgets = sorted({available[0], available[-1]}, key=int)
    fig, axes = plt.subplots(2, len(budgets), figsize=(6 * len(budgets), 8), squeeze=False)
    for column, t in enumerate(budgets):
        for method in sorted({r["method"] for r in summary}):
            rows = sorted((r for r in summary if r["threads"] == t and r["m"] == r["n"] == r["k"]),
                          key=lambda r: int(r["m"]))
            rows = [r for r in rows if r["method"] == method]
            if not rows:
                continue
            x = [int(r["m"]) for r in rows]
            times = [r["median_ns"] / 1e6 for r in rows]
            low = [r["min_process_ns"] / 1e6 for r in rows]
            high = [r["max_process_ns"] / 1e6 for r in rows]
            color = colors.get(method)
            axes[0, column].plot(x, [r["logical_gflops"] for r in rows], label=method, color=color)
            axes[1, column].plot(x, times, label=method, color=color)
            axes[1, column].fill_between(x, low, high, color=color, alpha=0.13)
        axes[0, column].set_title(f"Requested threads: {t}")
        for ax in axes[:, column]:
            ax.grid(alpha=0.24)
            ax.set_xlabel("Square dimension M = N = K")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=False)
    axes[0, 0].set_ylabel("Logical GFLOP/s")
    for ax in axes[1, :]:
        ax.set_yscale("log")
    axes[1, 0].set_ylabel("Full-call latency (ms)")
    fig.suptitle(f"FP32 GEMM on {summary[0]['hostname']} · {summary[0]['topology']} NUMA")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


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
