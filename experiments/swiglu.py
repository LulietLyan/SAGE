#!/usr/bin/env python3
"""Measure a vectorized SwiGLU block on compiler and PyTorch CPU paths.

SwiGLU is useful here because BLAS exposes the dense projections, while SiLU
and the gate multiplication are elementwise nonlinear work outside BLAS.  The
three compiler modes use the same Triton GEMM source for gate, up and down
projections; the nonlinear stages remain explicit and identical.  The PyTorch
baseline uses ``torch.mm`` for its projections and ``torch.nn.functional.silu``
for the gate.  Latency covers the complete synchronous block.
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
CASES = [
    (64, 128, 384),
    (96, 192, 576),
    (128, 256, 768),
    (160, 320, 960),
    (192, 384, 1152),
    (224, 416, 1248),
    (256, 384, 1152),
    (320, 448, 1344),
    (384, 448, 1344),
    (448, 512, 1536),
    (512, 512, 1536),
    (576, 576, 1728),
    (640, 576, 1728),
    (704, 640, 1920),
    (768, 640, 1920),
    (896, 704, 2112),
    (1024, 768, 2304),
    (1280, 896, 2688),
    (1536, 1024, 3072),
]


def make_problem(tokens, width, hidden, seed):
    import torch

    g = torch.Generator().manual_seed(seed)
    x = torch.randn((tokens, width), generator=g)
    wg = torch.randn((width, hidden), generator=g) * 0.02
    wu = torch.randn((width, hidden), generator=g) * 0.02
    wd = torch.randn((hidden, width), generator=g) * 0.02
    return x, wg.contiguous(), wu.contiguous(), wd.contiguous()


def build_calls(x, wg, wu, wd, mode, threads, profile):
    import torch

    m, width = x.shape
    hidden = wg.shape[1]
    if mode == "pytorch":
        out = torch.empty_like(x)

        def call():
            gate = torch.mm(x, wg)
            up = torch.mm(x, wu)
            value = torch.nn.functional.silu(gate) * up
            out.copy_(torch.mm(value, wd))
            return out

        call.evidence = {"backend_verified": "", "executed_path": "pytorch torch.mm"}
        return call

    gate = torch.empty((m, hidden), dtype=x.dtype)
    up = torch.empty_like(gate)
    value = torch.empty_like(gate)
    out = torch.empty_like(x)
    gate_call = compile_gemm(x, wg, gate, mode, threads, profile)
    up_call = compile_gemm(x, wu, up, mode, threads, profile)
    down_call = compile_gemm(value, wd, out, mode, threads, profile)

    def call():
        gate_call()
        up_call()
        torch.mul(torch.nn.functional.silu(gate), up, out=value)
        down_call()
        return out

    call.evidence = {
        "backend_verified": bool(
            gate_call.evidence.get("backend_verified")
            and up_call.evidence.get("backend_verified")
            and down_call.evidence.get("backend_verified")
        ),
        "executed_path": gate_call.evidence.get("executed_path", ""),
        "gate": gate_call.evidence,
        "up": up_call.evidence,
        "down": down_call.evidence,
    }
    return call


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


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_bundle(output, rows, samples, cases, budgets, rounds, window, *, status):
    write_csv(output / "swiglu.csv", rows)
    write_csv(output / "swiglu_samples.csv", samples)
    (output / "swiglu_environment.json").write_text(
        json.dumps(
            {
                "status": status,
                "operator": "SwiGLU",
                "cases": [list(case) for case in cases],
                "case_fields": ["tokens", "width", "hidden"],
                "thread_budgets": list(budgets),
                "rounds": rounds,
                "window_s": window,
                "methods": list(METHODS),
                "method_observations": len(rows),
                "raw_windows": len(samples),
                "note": "One OS process per thread budget (prepare-once constraint).",
            },
            indent=2,
        )
        + "\n"
    )


def run(output=WORKSPACE / "output", *, threads_list=(1,), rounds=3, window=0.01, quick=False):
    if len(threads_list) != 1:
        raise ValueError("run() accepts exactly one thread budget; main() spawns subprocesses")
    threads = threads_list[0]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile_path = output / "profile.json"
    profile = json.loads(profile_path.read_text()) if profile_path.is_file() else None
    cases = CASES[:3] if quick else CASES
    rows, samples = [], []
    torch = prepare(threads, ROOT / ".build" / "swiglu-jit-cache")
    for index, (tokens, width, hidden) in enumerate(cases):
        x, wg, wu, wd = make_problem(tokens, width, hidden, 12031 + index)
        reference = (torch.nn.functional.silu(torch.mm(x, wg)) * torch.mm(x, wu)).mm(wd)
        calls = {name: build_calls(x, wg, wu, wd, name, threads, profile) for name in METHODS}
        for name, call in calls.items():
            result = check(call(), reference)
            if not result["correctness_passed"]:
                raise RuntimeError(
                    f"SwiGLU correctness failed for {name}/{tokens}/t{threads}: {result}"
                )
        timing = measure_interleaved(
            calls, rounds=rounds, target_s=window, seed=907 + index + 19 * threads, phase="swiglu"
        )
        flops = 2 * tokens * (2 * width * hidden + hidden * width)
        for name in METHODS:
            summary = timing["summary"][name]
            ev = getattr(calls[name], "evidence", {})
            rows.append(
                dict(
                    operator="SwiGLU",
                    method=name,
                    threads=threads,
                    tokens=tokens,
                    width=width,
                    hidden=hidden,
                    latency_ms=summary["median_ns"] / 1e6,
                    projection_flops=flops,
                    projection_gflops=flops / summary["median_ns"],
                    gflops_scope="three dense projections; SiLU and gate multiply included in latency",
                    torch_version=str(torch.__version__),
                    torch_threads=torch.get_num_threads(),
                    torch_entry="torch.mm + torch.nn.functional.silu"
                    if name == "pytorch"
                    else "compiler-owned Triton GEMM",
                    backend_verified=ev.get("backend_verified", ""),
                    executed_path=ev.get("executed_path", ""),
                    windows=summary["windows"],
                    windows_meeting_target=summary["windows_meeting_target"],
                    **check(calls[name](), reference),
                )
            )
        for raw in timing["raw"]:
            samples.append(
                dict(
                    operator="SwiGLU",
                    threads=threads,
                    tokens=tokens,
                    width=width,
                    hidden=hidden,
                    **raw,
                )
            )
        print(
            json.dumps(
                {
                    "progress": "swiglu",
                    "threads": threads,
                    "case": [tokens, width, hidden],
                    "done_rows": len(rows),
                }
            ),
            flush=True,
        )
        prior, prior_samples = [], []
        if (output / "swiglu.csv").is_file():
            with (output / "swiglu.csv").open(newline="") as stream:
                prior = [row for row in csv.DictReader(stream) if int(row["threads"]) != threads]
        if (output / "swiglu_samples.csv").is_file():
            with (output / "swiglu_samples.csv").open(newline="") as stream:
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

    prior, prior_samples = [], []
    if (output / "swiglu.csv").is_file():
        with (output / "swiglu.csv").open(newline="") as stream:
            prior = [row for row in csv.DictReader(stream) if int(row["threads"]) != threads]
    if (output / "swiglu_samples.csv").is_file():
        with (output / "swiglu_samples.csv").open(newline="") as stream:
            prior_samples = [
                row for row in csv.DictReader(stream) if int(row["threads"]) != threads
            ]
    merged = prior + rows
    merged_samples = prior_samples + samples
    budgets = sorted({int(row["threads"]) for row in merged})
    _write_bundle(output, merged, merged_samples, cases, budgets, rounds, window, status="complete")
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "output")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--window", type=float, default=0.01)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if len(args.threads) > 1:
        for threads in args.threads:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output",
                str(args.output),
                "--threads",
                str(threads),
                "--rounds",
                str(args.rounds),
                "--window",
                str(args.window),
            ]
            if args.quick:
                cmd.append("--quick")
            subprocess.run(cmd, check=True)
        with (Path(args.output) / "swiglu.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = run(
            args.output,
            threads_list=tuple(args.threads),
            rounds=args.rounds,
            window=args.window,
            quick=args.quick,
        )
    print(
        json.dumps(
            {
                "operator": "SwiGLU",
                "rows": len(rows),
                "cases": len(CASES if not args.quick else CASES[:3]),
                "threads": args.threads,
                "output": str(Path(args.output).resolve() / "swiglu.csv"),
            }
        )
    )


if __name__ == "__main__":
    main()
