#!/usr/bin/env python3
"""Efficient-KAN KANLinear forward benchmark.

The operator is the efficient KAN variant (B-spline basis expansion followed
by a linear projection), with the optional SiLU/base projection retained.  A
complete timed call includes clamping, B-spline expansion, both projections,
and the final addition.  ``generic``, ``sme-fixed`` and ``ours`` use the same
compiled TritonCPU GEMM for the two projections; only ``ours`` lets the C++
planner select the local tile/layout/team.  ``pytorch`` uses ``torch.mm``.

The reported GFLOP/s is *projection GFLOP/s*: the two dense projections have
an exact 2*M*K*N count.  B-spline recurrence and SiLU contain non-linear
operations whose FLOP convention is not portable, so end-to-end latency is
the primary metric and the CSV records the excluded basis/activation work.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import measure_interleaved
from sme_plan.compiler_backend import compile_gemm, prepare


# (tokens L, in_features d, out_features h, spline intervals G).
# First four cases preserve the original protocol; the rest enlarge coverage
# across sequence length, width, and basis expansion while keeping both
# projections inside the compiler's verified dense GEMM shape envelope
# (including expanded spline width d*G).
CASES = [
    (128, 64, 256, 8),
    (256, 128, 256, 8),
    (384, 128, 512, 12),
    (512, 192, 512, 12),
    (64, 64, 128, 8),
    (96, 64, 192, 8),
    (160, 96, 256, 8),
    (192, 96, 256, 12),
    (224, 128, 256, 8),
    (320, 128, 384, 8),
    (448, 160, 384, 12),
    (576, 192, 512, 8),
    (640, 192, 512, 12),
    (768, 256, 512, 12),
    (896, 256, 768, 12),
    (1024, 256, 768, 16),
]
METHODS = ("generic", "sme-fixed", "ours", "pytorch")


def bspline_basis(x, intervals: int, degree: int = 3):
    """Evaluate clamped uniform B-splines, vectorized over ``x``."""
    import torch

    if intervals < 1 or degree < 0:
        raise ValueError("intervals and degree must be positive")
    x = x.clamp(0, 1)
    knots = torch.linspace(0, 1, intervals + 1, dtype=x.dtype, device=x.device)
    knots = torch.cat((knots[:1].repeat(degree), knots, knots[-1:].repeat(degree)))
    n = knots.numel() - 1
    basis = ((x[..., None] >= knots[:-1]) & (x[..., None] < knots[1:])).to(x.dtype)
    basis = torch.where(
        (x == 1)[..., None],
        torch.nn.functional.pad(torch.ones_like(x)[..., None], (n - 1, 0)),
        basis,
    )
    for d in range(1, degree + 1):
        count = n - d
        left_den = knots[d : d + count] - knots[:count]
        right_den = knots[d + 1 : d + 1 + count] - knots[1 : 1 + count]
        left_num = x[..., None] - knots[:count]
        right_num = knots[d + 1 : d + 1 + count] - x[..., None]
        left = torch.where(
            left_den > 0,
            left_num / torch.where(left_den > 0, left_den, torch.ones_like(left_den)),
            torch.zeros_like(left_num),
        )
        right = torch.where(
            right_den > 0,
            right_num / torch.where(right_den > 0, right_den, torch.ones_like(right_den)),
            torch.zeros_like(right_num),
        )
        basis = left * basis[..., :count] + right * basis[..., 1 : count + 1]
        n = count
    basis = torch.where(
        (x == 1)[..., None],
        torch.nn.functional.pad(torch.ones_like(x)[..., None], (n - 1, 0)),
        basis,
    )
    return basis


def make_problem(tokens, in_features, out_features, intervals, *, seed):
    import torch

    generator = torch.Generator().manual_seed(seed)
    x = torch.rand((tokens, in_features), generator=generator)
    base_weight = torch.randn((out_features, in_features), generator=generator) * 0.08
    q = intervals
    spline_weight = torch.randn((out_features, in_features * q), generator=generator) * 0.08
    return x, base_weight, spline_weight, q


def build_calls(x, base_weight, spline_weight, q, mode, threads, profile):
    import torch

    def basis_fn():
        return bspline_basis(x, q).reshape(x.shape[0], -1)

    base_a, base_b = torch.nn.functional.silu(x), base_weight.t().contiguous()
    spline_a = basis_fn()
    spline_b = spline_weight.t().contiguous()
    if mode == "pytorch":
        out = torch.empty((x.shape[0], base_weight.shape[0]))

        def call():
            basis = basis_fn()
            torch.mm(torch.nn.functional.silu(x), base_b, out=out)
            out.addmm_(basis, spline_b)
            return out

        call.evidence = {"executed_path": "pytorch", "backend_verified": ""}
        call.compile_ns = 0
        call.first_call_ns = 0
        return call

    base_out = torch.empty((x.shape[0], base_weight.shape[0]))
    spline_out = torch.empty_like(base_out)
    start = time.perf_counter_ns()
    base_call = compile_gemm(base_a, base_b, base_out, mode, threads, profile)
    spline_call = compile_gemm(spline_a, spline_b, spline_out, mode, threads, profile)
    compile_ns = time.perf_counter_ns() - start

    def call():
        basis = basis_fn()
        base_a.copy_(torch.nn.functional.silu(x))
        base_call()
        spline_a.copy_(basis)
        spline_call()
        base_out.add_(spline_out)
        return base_out

    ev = dict(
        mode=mode,
        backend_verified=bool(
            base_call.evidence.get("backend_verified")
            and spline_call.evidence.get("backend_verified")
        ),
        executed_path=base_call.evidence.get("executed_path", ""),
        compile_ns=compile_ns,
        first_call_ns=base_call.evidence.get("first_call_ns", 0)
        + spline_call.evidence.get("first_call_ns", 0),
        base_evidence=base_call.evidence,
        spline_evidence=spline_call.evidence,
    )
    call.evidence = ev
    return call


def check(actual, reference):
    import torch

    delta = (actual - reference).abs()
    scale = 2e-5 * math.sqrt(max(1, reference.shape[1])) + 2e-4 * reference.abs()
    ratio = float((delta / scale).max())
    return dict(
        correctness_passed=bool(torch.isfinite(actual).all() and ratio <= 1),
        max_abs=float(delta.max()),
        tolerance_ratio=ratio,
    )


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    output=WORKSPACE / "output",
    *,
    threads_list=(1,),
    rounds=3,
    window=0.01,
    quick=False,
):
    """Measure one thread budget per process (Torch/Triton prepare is once-only)."""
    if len(threads_list) != 1:
        raise ValueError("run() accepts exactly one thread budget; main() spawns subprocesses")
    threads = threads_list[0]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile_path = output / "profile.json"
    profile = json.loads(profile_path.read_text()) if profile_path.is_file() else None
    cases = CASES[:3] if quick else CASES
    rows, samples = [], []
    torch = prepare(threads, ROOT / ".build" / "kan-jit-cache")
    torch_backend = "Accelerate" if "accelerate" in torch.__config__.show().lower() else "unknown"
    for index, (tokens, in_features, out_features, intervals) in enumerate(cases):
        if in_features * intervals > 8192:
            raise ValueError(
                f"spline width too large for case {tokens, in_features, out_features, intervals}"
            )
        x, base_weight, spline_weight, q = make_problem(
            tokens, in_features, out_features, intervals, seed=9011 + index
        )
        basis = bspline_basis(x, intervals)
        reference = torch.nn.functional.silu(x).mm(base_weight.t()) + basis.reshape(tokens, -1).mm(
            spline_weight.t()
        )
        calls = {
            name: build_calls(x, base_weight, spline_weight, q, name, threads, profile)
            for name in METHODS
        }
        for name, call in calls.items():
            numerical = check(call(), reference)
            if not numerical["correctness_passed"]:
                raise RuntimeError(
                    f"KAN correctness failed for {name}/{tokens}/t{threads}: {numerical}"
                )
        timing = measure_interleaved(
            calls, rounds=rounds, target_s=window, seed=131 + index + 17 * threads, phase="kan"
        )
        projection_flops = 2 * tokens * in_features * out_features * (q + 1)
        for name in METHODS:
            summary = timing["summary"][name]
            call = calls[name]
            ev = getattr(call, "evidence", {})
            rows.append(
                dict(
                    operator="KANLinear-efficient-bspline",
                    method=name,
                    threads=threads,
                    torch_version=str(torch.__version__),
                    torch_threads=torch.get_num_threads(),
                    torch_entry="torch.mm" if name == "pytorch" else "compiler-owned Triton GEMM",
                    torch_backend=torch_backend,
                    tokens=tokens,
                    in_features=in_features,
                    out_features=out_features,
                    spline_intervals=intervals,
                    spline_degree=3,
                    basis_width=q,
                    median_ns=summary["median_ns"],
                    latency_ms=summary["median_ns"] / 1e6,
                    projection_flops=projection_flops,
                    projection_gflops=projection_flops / summary["median_ns"],
                    gflops_scope="two dense projections only; basis and SiLU included in latency",
                    windows=summary["windows"],
                    windows_meeting_target=summary["windows_meeting_target"],
                    compile_ns=ev.get("compile_ns", 0),
                    first_call_ns=ev.get("first_call_ns", 0),
                    backend_verified=ev.get("backend_verified", ""),
                    executed_path=ev.get("executed_path", ""),
                    **check(call(), reference),
                )
            )
        for raw in timing["raw"]:
            samples.append(
                dict(
                    operator="KANLinear-efficient-bspline",
                    threads=threads,
                    tokens=tokens,
                    in_features=in_features,
                    out_features=out_features,
                    spline_intervals=intervals,
                    **raw,
                )
            )
        del calls, x, base_weight, spline_weight, basis, reference
        gc.collect()
        print(
            json.dumps(
                {
                    "progress": "kan",
                    "threads": threads,
                    "case": [tokens, in_features, out_features, intervals],
                    "done_rows": len(rows),
                }
            ),
            flush=True,
        )
        prior = []
        prior_samples = []
        if (output / "kan.csv").is_file():
            with (output / "kan.csv").open(newline="") as stream:
                prior = [row for row in csv.DictReader(stream) if int(row["threads"]) != threads]
        if (output / "kan_samples.csv").is_file():
            with (output / "kan_samples.csv").open(newline="") as stream:
                prior_samples = [
                    row for row in csv.DictReader(stream) if int(row["threads"]) != threads
                ]
        budgets = sorted({int(row["threads"]) for row in prior + rows} | {threads})
        _write_kan_bundle(
            output,
            prior + rows,
            prior_samples + samples,
            cases,
            budgets,
            rounds,
            window,
            status="partial",
        )

    prior = []
    prior_samples = []
    if (output / "kan.csv").is_file():
        with (output / "kan.csv").open(newline="") as stream:
            prior = [row for row in csv.DictReader(stream) if int(row["threads"]) != threads]
    if (output / "kan_samples.csv").is_file():
        with (output / "kan_samples.csv").open(newline="") as stream:
            prior_samples = [
                row for row in csv.DictReader(stream) if int(row["threads"]) != threads
            ]
    merged = prior + rows
    merged_samples = prior_samples + samples
    budgets = sorted({int(row["threads"]) for row in merged})
    _write_kan_bundle(
        output, merged, merged_samples, cases, budgets, rounds, window, status="complete"
    )
    return merged


def _write_kan_bundle(output, rows, samples, cases, budgets, rounds, window, *, status):
    write_csv(output / "kan.csv", rows)
    write_csv(output / "kan_samples.csv", samples)
    (output / "kan_environment.json").write_text(
        json.dumps(
            {
                "status": status,
                "operator": "KANLinear-efficient-bspline",
                "cases": [list(case) for case in cases],
                "case_fields": ["tokens", "in_features", "out_features", "spline_intervals"],
                "thread_budgets": list(budgets),
                "rounds": rounds,
                "window_s": window,
                "methods": list(METHODS),
                "method_observations": len(rows),
                "raw_windows": len(samples),
                "note": "Expanded KAN protocol; one OS process per thread budget.",
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "output")
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 8],
        help="thread budgets to measure (default: 1 8)",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--window", type=float, default=0.01)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    # Each budget needs a fresh process: prepare() cannot run after Torch import.
    if len(args.threads) > 1:
        import subprocess

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
        with (Path(args.output) / "kan.csv").open(newline="") as stream:
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
                "operator": "KANLinear-efficient-bspline",
                "rows": len(rows),
                "cases": len(CASES if not args.quick else CASES[:3]),
                "threads": args.threads,
                "output": str(Path(args.output).resolve() / "kan.csv"),
            }
        )
    )


if __name__ == "__main__":
    main()
