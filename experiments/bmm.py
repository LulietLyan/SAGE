#!/usr/bin/env python3
"""Batched matmul (BMM) benchmark.

PyTorch uses ``torch.bmm``; the installed backend is recorded.  Compiler modes lower each batch
slice to the same Triton GEMM entry used elsewhere; ``ours`` selects per-slice
schedules.  This isolates batched BLAS-native work without Attention softmax.
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
# (batch, M, N, K) — square-ish slices with growing batch / tile.
CASES = [
    (2, 64, 64, 64),
    (4, 64, 64, 64),
    (4, 96, 96, 96),
    (8, 96, 96, 96),
    (8, 128, 128, 128),
    (4, 160, 160, 160),
    (8, 192, 192, 192),
    (4, 256, 256, 256),
    (8, 256, 256, 256),
    (4, 320, 320, 320),
    (8, 384, 384, 384),
    (4, 448, 448, 448),
    (4, 512, 512, 512),
    (2, 640, 640, 640),
    (2, 768, 768, 768),
    (2, 896, 896, 896),
    (2, 1024, 1024, 1024),
    (1, 1280, 1280, 1280),
    (1, 1536, 1536, 1536),
    (1, 2048, 2048, 2048),
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
    scale = 2e-5 * math.sqrt(max(1, reference.shape[-1])) + 2e-4 * reference.abs()
    ratio = float((delta / scale).max())
    return {
        "correctness_passed": bool(torch.isfinite(actual).all()) and ratio <= 1.0,
        "max_abs": float(delta.max()),
        "tolerance_ratio": ratio,
    }


def build_calls(a, b, mode, threads, profile):
    import torch

    batch, m, k = a.shape
    n = b.shape[-1]
    out = torch.empty((batch, m, n), dtype=a.dtype)
    if mode == "pytorch":

        def call():
            torch.bmm(a, b, out=out)
            return out

        call.evidence = {"backend_verified": "", "executed_path": "pytorch torch.bmm"}
        return call

    slices = []
    for i in range(batch):
        slices.append(
            compile_gemm(
                a[i],
                b[i],
                out[i],
                mode,
                threads,
                profile if mode == "ours" else None,
            )
        )

    def call():
        for fn in slices:
            fn()
        return out

    call.evidence = getattr(slices[0], "evidence", {})
    return call


def _write_bundle(output, rows, samples, cases, budgets, rounds, window, *, status):
    write_csv(output / "bmm.csv", rows)
    write_csv(output / "bmm_samples.csv", samples)
    (output / "bmm_environment.json").write_text(
        json.dumps(
            {
                "status": status,
                "operator": "BMM",
                "cases": [list(c) for c in cases],
                "case_fields": ["batch", "m", "n", "k"],
                "thread_budgets": list(budgets),
                "rounds": rounds,
                "window_s": window,
                "methods": list(METHODS),
                "method_observations": len(rows),
                "raw_windows": len(samples),
                "class": "BLAS-native",
                "note": "Compiler path loops Triton GEMM over batch; PyTorch uses torch.bmm.",
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
    torch = prepare(threads, ROOT / ".build" / "bmm-jit-cache")
    for index, (batch, m, n, k) in enumerate(cases):
        g = torch.Generator().manual_seed(55021 + index)
        a = torch.randn((batch, m, k), generator=g)
        b = torch.randn((batch, k, n), generator=g)
        reference = torch.bmm(a, b)
        calls = {name: build_calls(a, b, name, threads, profile) for name in METHODS}
        for name, call in calls.items():
            result = check(call(), reference)
            if not result["correctness_passed"]:
                raise RuntimeError(f"BMM correctness failed for {name}/{batch}x{m}: {result}")
        timing = measure_interleaved(
            calls, rounds=rounds, target_s=window, seed=521 + index + 23 * threads, phase="bmm"
        )
        flops = 2 * batch * m * n * k
        for name in METHODS:
            summary = timing["summary"][name]
            ev = getattr(calls[name], "evidence", {})
            rows.append(
                dict(
                    operator="BMM",
                    method=name,
                    threads=threads,
                    batch=batch,
                    m=m,
                    n=n,
                    k=k,
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
            samples.append(dict(operator="BMM", threads=threads, batch=batch, m=m, n=n, k=k, **raw))
        print(
            json.dumps(
                {"progress": "bmm", "threads": threads, "case": [batch, m, n, k], "done": len(rows)}
            ),
            flush=True,
        )
        prior, prior_samples = [], []
        if (output / "bmm.csv").is_file():
            with (output / "bmm.csv").open(newline="") as stream:
                prior = [row for row in csv.DictReader(stream) if int(row["threads"]) != threads]
        if (output / "bmm_samples.csv").is_file():
            with (output / "bmm_samples.csv").open(newline="") as stream:
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
    all_rows, all_samples = rows, samples
    if (output / "bmm.csv").is_file():
        with (output / "bmm.csv").open(newline="") as stream:
            all_rows = list(csv.DictReader(stream))
        with (output / "bmm_samples.csv").open(newline="") as stream:
            all_samples = list(csv.DictReader(stream))
        budgets = sorted({int(row["threads"]) for row in all_rows})
    else:
        budgets = [threads]
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
