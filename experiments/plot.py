#!/usr/bin/env python3
"""Plot measured square GEMM results in Nature-style quantitative panels.

Latency and throughput use linear y-axes. Size uses a base-2 log x-axis so
power-of-two tick labels never collide. The input CSV / environment contracts
are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from plot_style import (
    FIG_SINGLE,
    METHODS,
    RC,
    log2_size_axis,
    mark_indices,
    method_plot_kwargs,
    nature_axes,
    nice_upper,
    save_figure,
)

REQUIRED = {"threads", "process", "pid", "m", "n", "k", "method", "median_ns", "gflops"}


def load_results(path: Path) -> dict:
    environment_path = path.with_name("environment.json")
    protocol = {}
    if environment_path.is_file():
        environment = json.loads(environment_path.read_text())
        if not isinstance(environment, dict) or environment.get("status") != "complete":
            raise ValueError("The experiment is not complete; refusing to plot partial results")
        protocol = environment.get("protocol", environment)
        if not isinstance(protocol, dict):
            raise ValueError("The experiment protocol must be an object")
    groups = {}
    process_pids = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            try:
                threads, process, pid, m, n, k = (
                    int(row[key]) for key in ("threads", "process", "pid", "m", "n", "k")
                )
                latency, gflops = float(row["median_ns"]), float(row["gflops"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid numeric data on CSV line {line}") from error
            method = row["method"]
            if method not in METHODS:
                raise ValueError(f"Unknown method on CSV line {line}: {method}")
            if min(threads, pid, m, n, k) < 1 or process < 0:
                raise ValueError(f"Invalid thread/process/shape identity on CSV line {line}")
            if not m == n == k:
                raise ValueError("This plot requires one square GEMM family (M = N = K)")
            if not all(math.isfinite(value) and value > 0 for value in (latency, gflops)):
                raise ValueError(f"Nonpositive or nonfinite timing on CSV line {line}")
            if not math.isclose(gflops, 2 * m * n * k / latency, rel_tol=1e-5):
                raise ValueError(f"GFLOP/s disagrees with 2MNK / median_ns on CSV line {line}")
            key = threads, process
            if key in process_pids and process_pids[key] != pid:
                raise ValueError(f"Process {process} at thread budget {threads} has multiple PIDs")
            process_pids[key] = pid
            samples = groups.setdefault(threads, {}).setdefault(m, {}).setdefault(method, {})
            if process in samples:
                raise ValueError(
                    f"Duplicate process result: threads={threads}, N={m}, {method}, p={process}"
                )
            samples[process] = {"pid": pid, "gflops": gflops, "latency": latency / 1e6}
    if not groups:
        raise ValueError("The results CSV is empty")
    for key in ("sizes", "thread_budgets"):
        if key in protocol:
            expected = protocol[key]
            if (
                not isinstance(expected, list)
                or not expected
                or any(type(value) is not int or value < 1 for value in expected)
                or len(set(expected)) != len(expected)
            ):
                raise ValueError(f"Invalid protocol {key}: expected distinct positive integers")
    if "thread_budgets" in protocol and set(groups) != set(protocol["thread_budgets"]):
        raise ValueError("CSV thread budgets do not match the complete experiment protocol")
    if "processes" in protocol and (
        type(protocol["processes"]) is not int or protocol["processes"] < 1
    ):
        raise ValueError("Protocol processes must be a positive integer")
    for threads, sizes in groups.items():
        identities = {
            process: pid for (budget, process), pid in process_pids.items() if budget == threads
        }
        if "sizes" in protocol and set(sizes) != set(protocol["sizes"]):
            raise ValueError(f"Incomplete or unexpected sizes at thread budget {threads}")
        if "processes" in protocol and len(identities) != protocol["processes"]:
            raise ValueError(f"Incomplete or unexpected process count at thread budget {threads}")
        if len(set(identities.values())) != len(identities):
            raise ValueError(
                f"Thread budget {threads} reuses a PID across independent process labels"
            )
        for size, methods in sizes.items():
            if set(methods) != set(METHODS):
                raise ValueError(f"Missing one of four methods at threads={threads}, N={size}")
            for method, samples in methods.items():
                if set(samples) != set(identities):
                    raise ValueError(
                        f"Unbalanced process coverage at threads={threads}, N={size}, {method}"
                    )
    return groups


def summarize(groups: dict) -> dict:
    result = {}
    for threads, sizes in sorted(groups.items()):
        result[threads] = {}
        for size, methods in sorted(sizes.items()):
            result[threads][size] = {}
            for method in METHODS:
                samples = methods[method]
                stats = {"processes": len(samples)}
                for metric in ("gflops", "latency"):
                    values = [sample[metric] for sample in samples.values()]
                    stats[metric] = (statistics.median(values), min(values), max(values))
                result[threads][size][method] = stats
    return result


def plot(output: Path | str = Path(__file__).resolve().parents[1] / "output") -> dict:
    output = Path(output)
    summary = summarize(load_results(output / "results.csv"))
    with plt.rc_context(RC):
        for threads, rows in summary.items():
            sizes = list(rows)
            count = rows[sizes[0]][METHODS[0]]["processes"]
            marks = mark_indices(len(sizes), target=5)
            for metric in ("gflops", "latency"):
                fig, ax = plt.subplots(figsize=FIG_SINGLE)
                for method in METHODS:
                    values = [rows[size][method][metric] for size in sizes]
                    center, low, high = zip(*values)
                    kw = method_plot_kwargs(method)
                    ax.plot(sizes, center, markevery=marks, **kw)
                    ax.fill_between(
                        sizes, low, high, color=kw["color"], alpha=0.12, linewidth=0, zorder=1
                    )
                log2_size_axis(ax, sizes)
                ax.set_xlabel(r"Square dimension $M{=}N{=}K$")
                if metric == "gflops":
                    ax.set_ylabel("Throughput (GFLOP/s)")
                    ax.set_ylim(
                        0, nice_upper(max(rows[s][m]["gflops"][1] for s in sizes for m in METHODS))
                    )
                else:
                    ax.set_ylabel("Latency (ms)")
                    # Linear latency; inset for optimized methods when dynamic range is large.
                    peak_all = max(rows[s][m]["latency"][1] for s in sizes for m in METHODS)
                    ax.set_ylim(0, nice_upper(peak_all))
                    opt_peak = max(
                        rows[s][m]["latency"][1] for s in sizes for m in METHODS if m != "generic"
                    )
                    if peak_all / max(opt_peak, 1e-9) > 4:
                        axins = ax.inset_axes([0.14, 0.42, 0.52, 0.50])
                        for method in METHODS:
                            if method == "generic":
                                continue
                            values = [rows[size][method][metric] for size in sizes]
                            center, low, high = zip(*values)
                            kw = method_plot_kwargs(method)
                            axins.plot(
                                sizes,
                                center,
                                markevery=marks,
                                **{**kw, "label": None, "markersize": 2.6},
                            )
                            axins.fill_between(
                                sizes, low, high, color=kw["color"], alpha=0.12, linewidth=0
                            )
                        log2_size_axis(axins, sizes)
                        axins.set_ylim(0, nice_upper(opt_peak, pad=1.12))
                        axins.tick_params(labelsize=5.5, length=2.0, width=0.5)
                        axins.set_title("w/o TritonCPU", fontsize=6, pad=1.5)
                        nature_axes(axins)
                        for spine in axins.spines.values():
                            spine.set_visible(True)
                            spine.set_linewidth(0.5)
                            spine.set_color("#9A9A9A")
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
                nature_axes(ax)
                # Legend inside: lower-right for latency (curves rise), upper-left for throughput.
                loc = "upper left" if metric == "gflops" else "center right"
                ax.legend(loc=loc, fontsize=6, frameon=False, borderaxespad=0.4)
                fig.suptitle(
                    f"Square GEMM  ·  {threads} thread{'s' if threads != 1 else ''}",
                    fontsize=8,
                    y=1.02,
                )
                save_figure(fig, output, f"gemm-t{threads}-{metric}")
            noun = "process" if count == 1 else "processes"
            print(
                f"t{threads}: {len(sizes)} sizes, {count} independent {noun}; Nature-style exports done"
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "output"
    )
    plot(parser.parse_args().output)


if __name__ == "__main__":
    main()
