#!/usr/bin/env python3
"""Summarize scheduler regret against the measured candidate set."""
from collections import defaultdict
import argparse
import csv
import json
from pathlib import Path
from statistics import median


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert json.loads((a.run / "environment.json").read_text())["status"] == "complete"
    groups = defaultdict(dict)
    with (a.run / "process_results.csv").open() as f:
        for row in csv.DictReader(f):
            group = tuple(row[k] for k in ("topology", "threads", "m", "n", "k"))
            groups[group][int(row["process"]), row["method"]] = float(row["median_ns"])
    with a.output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topology", "threads", "m", "n", "k", "processes", "tested_schedules",
                    "best_measured", "sage_ns", "best_ns", "sage_over_best_latency",
                    "leave_one_process_out_sage_over_selected_latency"])
        for group, data in sorted(groups.items()):
            processes = sorted({p for p, _ in data})
            candidates = sorted({name for _, name in data})
            assert len(processes) >= 2 and "ours" in candidates
            centers = {name: median(data[p, name] for p in processes) for name in candidates}
            best = min(centers, key=centers.get)
            ratios = []
            for holdout in processes:
                chosen = min(candidates, key=lambda name: median(
                    data[p, name] for p in processes if p != holdout))
                ratios.append(data[holdout, "ours"] / data[holdout, chosen])
            w.writerow([*group, len(processes), len(candidates), best, centers["ours"], centers[best],
                        centers["ours"] / centers[best], median(ratios)])


if __name__ == "__main__":
    main()
