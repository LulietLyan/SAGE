#!/usr/bin/env python3
"""Measured, phase-aware scaled dot-product attention on the four CPU paths.

The QK^T and PV matrix products use the same Triton GEMM source as the GEMM
experiment.  Softmax is deliberately kept as an explicit nonlinear phase so
that a GEMM result is not presented as an end-to-end attention result.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sme_plan.compiler_backend import prepare, compile_gemm, compiler_sha256, sha256, kernel
from common import measure_interleaved

METHODS = ("generic", "sme-fixed", "ours", "pytorch")
PHASES = ("QK", "scale", "softmax", "PV", "full")
DEFAULT_LENGTHS = [
    32,
    48,
    64,
    80,
    96,
    112,
    128,
    160,
    192,
    224,
    256,
    320,
    384,
    448,
    512,
    576,
    640,
    704,
    768,
    896,
    1024,
    1152,
    1280,
    1536,
    1792,
    2048,
]


def _csv(path, rows):
    if rows:
        with Path(path).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _check(actual, reference):
    import torch

    delta = (actual - reference).abs()
    scale = reference.abs()
    ratio = float((delta / (2e-4 * scale + 2e-5)).max())
    return {
        "passed": bool(torch.isfinite(actual).all()) and ratio <= 1.0,
        "max_abs": float(delta.max()),
        "tolerance_ratio": ratio,
    }


def worker(options):
    torch = prepare(options["threads"], options["cache"])
    profile = json.loads(Path(options["profile"]).read_text())
    if sha256(options["profile"]) != options["profile_file_sha256"]:
        raise ValueError("Calibration profile changed during the experiment")
    if profile["compilerSha256"] != compiler_sha256():
        raise ValueError("The compiler changed; recalibrate before measurement")
    source_sha = hashlib.sha256(kernel().src.encode()).hexdigest()
    if profile["sourceSha256"] != source_sha:
        raise ValueError("The Triton source changed; recalibrate before measurement")
    results, phases, checks = [], [], []
    identity = dict(
        pid=os.getpid(),
        python=platform.python_version(),
        torch_version=str(torch.__version__),
        torch_threads=torch.get_num_threads(),
        torch_entry="torch.nn.functional.scaled_dot_product_attention",
        torch_backend="Accelerate"
        if "accelerate" in torch.__config__.show().lower()
        else "unknown",
        source_sha256=source_sha,
        compiler_sha256=compiler_sha256(),
        profile_file_sha256=options["profile_file_sha256"],
    )
    for length in options["lengths"]:
        d = options["dim"]
        generator = torch.Generator().manual_seed(4711 + length)
        q = torch.randn(length, d, generator=generator)
        k = torch.randn(length, d, generator=generator)
        v = torch.randn(length, d, generator=generator)
        kt = k.t().contiguous()
        reference = torch.softmax(q @ kt / math.sqrt(d), dim=-1) @ v
        calls = {}
        phase_calls = {}
        for method in METHODS[:-1]:
            qk = torch.empty((length, length), dtype=q.dtype)
            pv = torch.empty((length, d), dtype=q.dtype)
            phase_scores = torch.randn((length, length), generator=generator)
            phase_scaled = torch.empty_like(phase_scores)
            phase_prob = torch.softmax(phase_scores, dim=-1).contiguous()
            phase_pv = torch.empty((length, d), dtype=q.dtype)
            qk_call = compile_gemm(
                q, kt, qk, method, options["threads"], profile if method == "ours" else None
            )
            # Compile PV against an initialized standalone phase input. The
            # full call uses the QK output, while phase timing never depends on
            # stale or mutated state from another phase.
            pv_call = compile_gemm(
                qk, v, pv, method, options["threads"], profile if method == "ours" else None
            )
            phase_pv_call = compile_gemm(
                phase_prob,
                v,
                phase_pv,
                method,
                options["threads"],
                profile if method == "ours" else None,
            )
            # Keep separate phase callables.  The full call performs all stages.
            phase_calls[method] = {
                "QK": qk_call,
                "scale": lambda inp=phase_scores, out=phase_scaled: torch.mul(
                    inp, 1.0 / math.sqrt(d), out=out
                ),
                "softmax": lambda inp=phase_scaled: torch.softmax(inp, dim=-1),
                "PV": phase_pv_call,
            }

            def make_full(qcall=qk_call, pcall=pv_call, qko=qk, pvo=pv):
                def run():
                    qcall()
                    qko.mul_(1.0 / math.sqrt(d))
                    qko.copy_(torch.softmax(qko, dim=-1))
                    pcall()
                    return pvo

                return run

            calls[method] = make_full()
            calls[method].evidence = dict(
                backend_verified=bool(
                    qk_call.evidence.get("backend_verified")
                    and pv_call.evidence.get("backend_verified")
                ),
                executed_path=qk_call.evidence.get("executed_path", ""),
                qk_evidence=qk_call.evidence,
                pv_evidence=pv_call.evidence,
            )
        out = torch.empty_like(reference)
        qk_pt = torch.empty((length, length), dtype=q.dtype)
        pv_pt = torch.empty_like(reference)
        phase_scores_pt = torch.randn((length, length), generator=generator)
        phase_scaled_pt = torch.empty_like(phase_scores_pt)
        phase_prob_pt = torch.softmax(phase_scores_pt, dim=-1).contiguous()

        def pt_qk(q=q, kt=kt, qk_pt=qk_pt):
            qk_pt.copy_(torch.mm(q, kt) / math.sqrt(d))
            return qk_pt

        def pt_softmax():
            return torch.softmax(phase_scaled_pt, dim=-1)

        def pt_scale():
            return torch.mul(phase_scores_pt, 1.0 / math.sqrt(d), out=phase_scaled_pt)

        def pt_pv(v=v, pv_pt=pv_pt):
            pv_pt.copy_(torch.mm(phase_prob_pt, v))
            return pv_pt

        def pt_full(q=q, k=k, v=v):
            # The complete PyTorch baseline uses its documented SDPA entry;
            # phase measurements above remain an explicit decomposition for
            # attribution and are not claimed to sum to this fused call.
            result = torch.nn.functional.scaled_dot_product_attention(
                q[None, None], k[None, None], v[None, None], dropout_p=0.0
            )
            out.copy_(result[0, 0])
            return out

        phase_calls["pytorch"] = {
            "QK": pt_qk,
            "scale": pt_scale,
            "softmax": pt_softmax,
            "PV": pt_pv,
        }
        calls["pytorch"] = pt_full
        common = dict(
            threads=options["threads"],
            process=options["process"],
            pid=os.getpid(),
            sequence_length=length,
            batch=1,
            heads=1,
            head_dim=d,
        )
        for method, call in calls.items():
            check = _check(call(), reference)
            if not check["passed"]:
                raise RuntimeError(f"attention correctness failed: {method}/{length}: {check}")
            evidence = getattr(call, "evidence", {})
            checks.append(
                dict(
                    **common,
                    method=method,
                    correctness_passed=check["passed"],
                    max_abs=check["max_abs"],
                    tolerance_ratio=check["tolerance_ratio"],
                    backend_verified=evidence.get("backend_verified", ""),
                    executed_path=evidence.get("executed_path", "pytorch"),
                    fmopa_count=evidence.get("fmopa_count", ""),
                )
            )
        full_timing = measure_interleaved(
            calls,
            rounds=options["rounds"],
            target_s=options["window"],
            seed=1009 + length + options["process"] * 17,
            phase="full",
        )
        # Phase timings are measured independently; this preserves the cost of
        # each stage while full timing remains the primary end-to-end result.
        phase_timing = {
            phase: measure_interleaved(
                {m: phase_calls[m][phase] for m in METHODS},
                rounds=options["rounds"],
                target_s=options["window"],
                seed=2003 + length + phase_index * 97 + options["process"],
                phase=phase,
            )
            for phase_index, phase in enumerate(("QK", "scale", "softmax", "PV"))
        }
        full_flops = 4 * length * length * d + 6 * length * length
        for method in METHODS:
            observed = full_timing["summary"][method]
            evidence = getattr(calls[method], "evidence", {})
            results.append(
                dict(
                    **common,
                    method=method,
                    median_ns=observed["median_ns"],
                    latency_ms=observed["median_ns"] / 1e6,
                    flops=full_flops,
                    gflops=full_flops / observed["median_ns"],
                    timing_windows=options["rounds"],
                    minimum_window_ns=observed["min_ns"],
                    maximum_window_ns=observed["max_ns"],
                    backend_verified=evidence.get("backend_verified", ""),
                    executed_path=evidence.get("executed_path", "pytorch"),
                    fmopa_count=evidence.get("fmopa_count", ""),
                    jit_binary_sha256=evidence.get("jit_binary_sha256", ""),
                    compiler_sha256=evidence.get("compiler_sha256", ""),
                    source_sha256=source_sha,
                )
            )
            for phase, timing in phase_timing.items():
                observed = timing["summary"][method]
                phase_flops = (
                    2 * length * length * d
                    if phase in ("QK", "PV")
                    else 5 * length * length
                    if phase == "softmax"
                    else length * length
                )
                phases.append(
                    dict(
                        **common,
                        method=method,
                        phase=phase,
                        median_ns=observed["median_ns"],
                        latency_ms=observed["median_ns"] / 1e6,
                        flops=phase_flops,
                        gflops=phase_flops / observed["median_ns"],
                        timing_windows=options["rounds"],
                    )
                )
        del calls, phase_calls, q, k, v, kt, qk_pt, pv_pt, reference
        gc.collect()
    return dict(
        results=results,
        phases=phases,
        checks=checks,
        identity=identity,
        torch_build_config=torch.__config__.show(),
    )


def run(
    output=WORKSPACE / "output",
    *,
    lengths=DEFAULT_LENGTHS,
    dim=64,
    threads=(1, 8),
    processes=3,
    rounds=3,
    window=0.015,
):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile_path = output / "profile.json"
    profile = json.loads(profile_path.read_text())
    profile_hash = sha256(profile_path)
    expected = dict(
        compiler_sha256=compiler_sha256(),
        source_sha256=profile["sourceSha256"],
        profile_file_sha256=profile_hash,
    )
    combined = {"results": [], "phases": [], "checks": []}
    environment = dict(
        status="running",
        operator="scaled_dot_product_attention",
        lengths=lengths,
        batch=1,
        heads=1,
        head_dim=dim,
        thread_budgets=list(threads),
        processes=processes,
        rounds=rounds,
        window_s=window,
        dtype="float32",
        flops="QK^T + scale + softmax + PV; approximate 5L^2 nonlinear FLOPs",
        scope="Complete synchronous attention call; QK, softmax, PV phase timings are separate.",
        compiler_sha256=expected["compiler_sha256"],
        source_sha256=expected["source_sha256"],
        profile_file_sha256=profile_hash,
        workers=[],
    )
    (output / "attention_environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    cache_directory = tempfile.TemporaryDirectory(prefix="attention-cache-", dir=ROOT / ".build")
    try:
        for process in range(processes):
            order = list(threads[process % len(threads) :]) + list(
                threads[: process % len(threads)]
            )
            for budget in order:
                options = dict(
                    threads=budget,
                    process=process,
                    lengths=lengths,
                    dim=dim,
                    rounds=rounds,
                    window=window,
                    profile=str(profile_path),
                    cache=cache_directory.name,
                    profile_file_sha256=profile_hash,
                )
                child = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--worker"],
                    input=json.dumps(options),
                    text=True,
                    capture_output=True,
                )
                if child.returncode:
                    raise RuntimeError(
                        f"Attention process {process}, budget {budget} failed:\n{child.stderr[-6000:]}"
                    )
                measured = json.loads(child.stdout)
                if any(measured["identity"].get(k) != v for k, v in expected.items()):
                    raise RuntimeError("Attention worker identity changed")
                for name in combined:
                    combined[name].extend(measured[name])
                    _csv(output / f"attention_{name}.csv", combined[name])
                environment["workers"].append(
                    dict(process=process, threads=budget, **measured["identity"])
                )
                (output / "attention_environment.json").write_text(
                    json.dumps(environment, indent=2) + "\n"
                )
                print(
                    f"Attention: process {process + 1}/{processes}, budget {budget}: {len(lengths)} lengths passed",
                    flush=True,
                )
        environment.update(
            status="complete",
            method_observations=len(combined["results"]),
            phase_observations=len(combined["phases"]),
        )
    except BaseException as error:
        environment.update(status="failed", error=str(error))
        raise
    finally:
        (output / "attention_environment.json").write_text(json.dumps(environment, indent=2) + "\n")
        cache_directory.cleanup()
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "output")
    parser.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--window", type=float, default=0.015)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
    else:
        run(
            args.output,
            lengths=args.lengths,
            dim=args.dim,
            threads=tuple(args.threads),
            processes=args.processes,
            rounds=args.rounds,
            window=args.window,
        )


if __name__ == "__main__":
    main()
