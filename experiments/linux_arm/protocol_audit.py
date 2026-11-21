#!/usr/bin/env python3
"""Audit Linux Arm benchmark bundles without rerunning a benchmark.

The audit reads only archived CSV/JSON files.  It checks the evidence needed to
interpret a complete-call comparison: run status, host identity, process and
window counts, target-window coverage, correctness, and compile/first-call
records.  It writes a machine-readable JSON report and a short Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def audit_run(run: Path) -> dict:
    env_path = run / "environment.json"
    samples_path = run / "samples.csv"
    checks_path = run / "checks.csv"
    process_path = run / "process_results.csv"
    missing = [str(p.name) for p in (env_path, samples_path, checks_path, process_path) if not p.is_file()]
    result: dict = {"run": str(run), "status": "missing", "missing_files": missing}
    if missing:
        return result

    environment = json.loads(env_path.read_text())
    samples = read_csv(samples_path)
    checks = read_csv(checks_path)
    process_results = read_csv(process_path)
    expected_rounds = int(environment.get("rounds", 0))
    check_keys = ("topology", "threads", "process", "m", "n", "k", "method")

    sample_keys = [tuple(row[key] for key in check_keys) for row in samples]
    check_keys_seen = [tuple(row[key] for key in check_keys) for row in checks]
    groups = defaultdict(list)
    for row in samples:
        groups[tuple(row[key] for key in check_keys)].append(row)

    hosts = sorted({row.get("hostname", "") for row in checks})
    processes = sorted({row.get("process", "") for row in checks})
    passed_checks = sum(as_bool(row.get("passed", "false")) for row in checks)
    target_windows = sum(as_bool(row.get("target_window_met", "false")) for row in samples)
    first_call_present = sum(bool(row.get("first_call_ns")) for row in checks)
    compile_present = sum(bool(row.get("compile_ns")) for row in checks)
    window_counts = Counter(len(rows) for rows in groups.values())
    complete_rounds = expected_rounds > 0 and window_counts == Counter({expected_rounds: len(groups)})
    key_consistency = set(sample_keys) == set(check_keys_seen) and len(sample_keys) == len(set(sample_keys)) * expected_rounds
    all_passed = len(checks) > 0 and passed_checks == len(checks)
    status = "complete" if environment.get("status") == "complete" else str(environment.get("status", "unknown"))
    if status == "complete" and not (complete_rounds and key_consistency and all_passed):
        status = "incomplete-evidence"

    result.update(
        {
            "status": status,
            "environment_status": environment.get("status"),
            "topology": environment.get("topology"),
            "hostname": hosts[0] if len(hosts) == 1 else hosts,
            "host_count": len(hosts),
            "process_count": len(processes),
            "process_ids": processes,
            "sample_rows": len(samples),
            "check_rows": len(checks),
            "process_result_rows": len(process_results),
            "expected_rounds": expected_rounds,
            "window_count_distribution": {str(k): v for k, v in sorted(window_counts.items())},
            "target_windows": target_windows,
            "target_window_fraction": target_windows / len(samples) if samples else 0.0,
            "passed_checks": passed_checks,
            "all_checks_passed": all_passed,
            "compile_records": compile_present,
            "first_call_records": first_call_present,
            "key_consistency": key_consistency,
            "complete_rounds": complete_rounds,
            "notes": [
                "steady-state samples are separate from compile and first-call records",
                "process-level aggregation should use process_results.csv or summarize.py",
                "target_window_fraction is descriptive and is not a confidence interval",
            ],
        }
    )
    return result


def markdown(reports: list[dict]) -> str:
    lines = [
        "# Linux Arm benchmark protocol audit",
        "",
        "This report reads archived CSV/JSON files and does not run a benchmark.",
        "",
        "| run | status | host | processes | samples | checks | rounds | target-window coverage | compile/first-call |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        compile_first = f"{report.get('compile_records', 0)}/{report.get('first_call_records', 0)}"
        coverage = f"{report.get('target_window_fraction', 0.0):.1%}"
        host = report.get("hostname", "")
        if isinstance(host, list):
            host = ", ".join(host)
        lines.append(
            f"| `{Path(report['run']).name}` | {report['status']} | {host} | "
            f"{report.get('process_count', 0)} | {report.get('sample_rows', 0)} | "
            f"{report.get('check_rows', 0)} | {report.get('expected_rounds', 0)} | {coverage} | {compile_first} |"
        )
    lines.extend(
        [
            "",
            "The target-window fraction describes how many archived windows met the requested target. "
            "The process count is inferred from the check rows. Compile and first-call records are checked "
            "for presence and remain outside steady-state timing.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Archived run directories")
    parser.add_argument("--json", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--markdown", type=Path, required=True, help="Output Markdown path")
    args = parser.parse_args()
    reports = [audit_run(run) for run in args.runs]
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"runs": reports}, indent=2, sort_keys=True) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown(reports))
    print(json.dumps({"runs": len(reports), "statuses": Counter(r["status"] for r in reports)}, sort_keys=True))


if __name__ == "__main__":
    main()
