#!/usr/bin/env python3
"""Join complete, disjoint shape cohorts without mixing process pairs."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("runs", type=Path, nargs="+")
    a = p.parse_args()
    envs = [json.loads((r / "environment.json").read_text()) for r in a.runs]
    first = envs[0]
    fields = ("status", "methods", "rounds", "window", "processes", "profile_sha256")
    assert first["status"] == "complete"
    assert all([env[k] for k in fields] == [first[k] for k in fields] for env in envs)
    keys = set()
    placements = []
    for run, env in zip(a.runs, envs):
        for job in env["jobs"]:
            for shape in job["shapes"]:
                key = job["topology"], job["threads"], job["process"], tuple(shape)
                assert key not in keys, key
                keys.add(key)
            placements.append(dict(cohort=run.name, **job))
    a.output.mkdir(parents=True, exist_ok=False)
    for name in ("process_results.csv", "checks.csv", "samples.csv"):
        with (a.output / name).open("w", newline="") as target:
            writer = None
            for run in a.runs:
                with (run / name).open(newline="") as source:
                    reader = csv.DictReader(source)
                    if writer is None:
                        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                        writer.writeheader()
                    assert writer.fieldnames == reader.fieldnames
                    writer.writerows(reader)
    merged = {k: first[k] for k in fields}
    merged.update(jobs=placements, cohorts=[r.name for r in a.runs],
                  skips=[skip for env in envs for skip in env.get("skips", [])])
    (a.output / "environment.json").write_text(json.dumps(merged, indent=2) + "\n")


if __name__ == "__main__":
    main()
