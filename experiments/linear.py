#!/usr/bin/env python3
"""Linear projection benchmark: X @ W (tokens × in → out).

PyTorch uses ``torch.mm`` and records the installed backend. Compiler modes
share the same Triton GEMM source; only ``ours`` selects the schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import measure_interleaved
from sme_plan.compiler_backend import compile_gemm, prepare

METHODS = ("generic", "sme-fixed", "ours", "pytorch")
# (tokens, in_features, out_features) — dense across typical LLM projection shapes.
CASES = [
    (32, 64, 128),
    (64, 128, 256),
    (96, 192, 384),
    (128, 256, 512),
    (160, 256, 768),
    (192, 384, 768),
    (256, 512, 1024),
    (320, 512, 1536),
    (384, 768, 1536),
    (448, 768, 2048),
    (512, 1024, 2048),
    (640, 1024, 3072),
    (768, 1536, 3072),
    (896, 1536, 4096),
    (1024, 2048, 4096),
    (1280, 2048, 4096),
    (1536, 2048, 4096),
    (2048, 2048, 4096),
]


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check(actual, reference):
    import torch

    delta = (actual - reference).abs()
    scale = 2e-5 * math.sqrt(max(1, reference.shape[1])) + 2e-4 * reference.abs()
    ratio = float((delta / scale).max())
    return {
        "correctness_passed": bool(torch.isfinite(actual).all()) and ratio <= 1.0,
        "max_abs": float(delta.max()),
        "tolerance_ratio": ratio,
    }


def build_calls(x, w, mode, threads, profile):
    import torch

    out = torch.empty((x.shape[0], w.shape[1]), dtype=x.dtype)
    if mode == "pytorch":

        def call():
            torch.mm(x, w, out=out)
            return out

        call.evidence = {"backend_verified": "", "executed_path": "pytorch torch.mm"}
        return call
    return compile_gemm(x, w, out, mode, threads, profile if mode == "ours" else None)


def _write_bundle(output, rows, samples, cases, budgets, rounds, window, *, status):
    write_csv(output / "linear.csv", rows)
    write_csv(output / "linear_samples.csv", samples)
    (output / "linear_environment.json").write_text(
        json.dumps(
            {
                "status": status,
                "operator": "Linear",
                "cases": [list(c) for c in cases],
                "case_fields": ["tokens", "in_features", "out_features"],
                "thread_budgets": list(budgets),
                "rounds": rounds,
                "window_s": window,
                "methods": list(METHODS),
                "method_observations": len(rows),
                "raw_windows": len(samples),
                "class": "BLAS-native",
            },
            indent=2,
        )
        + "\n"
    )


def run(output=WORKSPACE / "output", *, threads_list=(8,), rounds=3, window=0.015, quick=False):
    if len(threads_list) != 1:
        raise ValueError("run() accepts exactly one thread budget")
    threads = threads_list[0]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile = json.loads((output / "profile.json").read_text())
    cases = CASES[:4] if quick else CASES
    rows, samples = [], []
    torch = prepare(threads, ROOT / ".build" / "linear-jit-cache")
    for index, (tokens, din, dout) in enumerate(cases):
        g = torch.Generator().manual_seed(44011 + index)
        x = torch.randn((tokens, din), generator=g)
        w = torch.randn((din, dout), generator=g) * 0.02
        reference = x @ w
        calls = {name: build_calls(x, w, name, threads, profile) for name in METHODS}
        for name, call in calls.items():
            result = check(call(), reference)
            if not result["correctness_passed"]:
                raise RuntimeError(f"Linear correctness failed for {name}/{tokens}: {result}")
        timing = measure_interleaved(
            calls, rounds=rounds, target_s=window, seed=311 + index + 17 * threads, phase="linear"
        )
        flops = 2 * tokens * din * dout
        for name in METHODS:
            summary = timing["summary"][name]
            ev = getattr(calls[name], "evidence", {})
            rows.append(
                dict(
                    operator="Linear",
                    method=name,
                    threads=threads,
                    tokens=tokens,
                    in_features=din,
                    out_features=dout,
                    case_id=index,
                    latency_ms=summary["median_ns"] / 1e6,
                    flops=flops,
                    gflops=flops / summary["median_ns"],
                    torch_version=str(torch.__version__),
                    torch_threads=torch.get_num_threads(),
                    backend_verified=ev.get("backend_verified", ""),
                    executed_path=ev.get("executed_path", ""),
                    windows=summary["windows"],
                    **check(calls[name](), reference),
                )
            )
        for raw in timing["raw"]:
            samples.append(
                dict(
                    operator="Linear",
                    threads=threads,
                    tokens=tokens,
                    in_features=din,
                    out_features=dout,
                    **raw,
                )
            )
        print(
            json.dumps(
                {
                    "progress": "linear",
                    "threads": threads,
                    "case": [tokens, din, dout],
                    "done": len(rows),
                }
            ),
            flush=True,
        )
        prior, prior_samples = [], []
        if (output / "linear.csv").is_file():
            with (output / "linear.csv").open(newline="") as stream:
                prior = [row for row in csv.DictReader(stream) if int(row["threads"]) != threads]
        if (output / "linear_samples.csv").is_file():
            with (output / "linear_samples.csv").open(newline="") as stream:
                prior_samples = [
                    row for row in csv.DictReader(stream) if int(row["threads"]) != threads
                ]
        budgets = sorted({int(row["threads"]) for row in prior + rows} | {threads})
        _write_bundle(
            output,
            prior + rows,
            prior_samples + samples,
            cases,
            budgets,
            rounds,
            window,
            status="partial",
        )
    budgets = sorted({int(row["threads"]) for row in rows} | {threads})
    # merge complete status with any other budgets already on disk
    all_rows, all_samples = rows, samples
    if (output / "linear.csv").is_file():
        with (output / "linear.csv").open(newline="") as stream:
            all_rows = list(csv.DictReader(stream))
        with (output / "linear_samples.csv").open(newline="") as stream:
            all_samples = list(csv.DictReader(stream))
        budgets = sorted({int(row["threads"]) for row in all_rows})
    _write_bundle(output, all_rows, all_samples, cases, budgets, rounds, window, status="complete")
    return all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "output")
    parser.add_argument("--threads", type=int, nargs="+", default=[8])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--window", type=float, default=0.015)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--worker-budget", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_budget is not None:
        run(
            args.output,
            threads_list=(args.worker_budget,),
            rounds=args.rounds,
            window=args.window,
            quick=args.quick,
        )
        return
    for budget in args.threads:
        child = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output",
                str(args.output),
                "--rounds",
                str(args.rounds),
                "--window",
                str(args.window),
                "--worker-budget",
                str(budget),
            ]
            + (["--quick"] if args.quick else []),
            check=False,
        )
        if child.returncode:
            raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
