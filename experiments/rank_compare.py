#!/usr/bin/env python3
"""Candidate-rank diagnostic with repeated measurements in one process.

The legacy --processes argument counts repetitions, not independent workers.
Use the formal GEMM harness for independent-process performance evidence.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "experiments"))
from sme_plan.compiler_backend import compile_gemm, compiler_sha256, prepare
from common import check_gemm, measure_interleaved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--sizes", type=int, nargs="+", required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--processes", type=int, default=3)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--window", type=float, default=0.01)
    a = p.parse_args()
    a.output.mkdir(exist_ok=True)
    profile = json.loads(a.profile.read_text())
    torch = prepare(a.threads, a.output / ".jit-cache")
    modes = ["ours-rank-5", "ours-rank-10", "ours-rank-20", "ours-worst"]
    rows = []
    raw = []
    checks = []
    for proc in range(a.processes):
        for size in a.sizes:
            g = torch.Generator().manual_seed(71931 + size + proc * 101)
            x = torch.randn(size, size, generator=g)
            y = torch.randn(size, size, generator=g)
            ref = x @ y
            calls = {}
            info = {}
            for mode in modes:
                out = torch.empty_like(ref)
                fn = compile_gemm(x, y, out, mode, a.threads, profile)
                calls[mode] = fn
                md = fn.evidence["metadata"]
                count = int(md["sme_plan_candidate_count"])
                rank = int(md["sme_plan_selected_rank"])
                info[mode] = (count, rank, md)
                ck = check_gemm(fn(), ref, x, y)
                checks.append(
                    dict(
                        process=proc,
                        size=size,
                        mode=mode,
                        passed=ck["passed"],
                        candidate_count=count,
                        rank=rank,
                    )
                )
            t = measure_interleaved(
                calls,
                rounds=a.rounds,
                target_s=a.window,
                seed=173 + size + proc * 731,
                phase="rank",
            )
            for mode, (count, rank, md) in info.items():
                label = f"Rank-{count}" if mode == "ours-worst" else f"Rank-{rank}"
                s = t["summary"][mode]
                rows.append(
                    dict(
                        process=proc,
                        size=size,
                        method=label,
                        mode=mode,
                        candidate_count=count,
                        rank=rank,
                        median_ns=s["median_ns"],
                        gflops=2 * size**3 / s["median_ns"],
                        block_m=md.get("sme_plan_block_m", ""),
                        block_n=md.get("sme_plan_block_n", ""),
                        block_k=md.get("sme_plan_block_k", ""),
                        k_strip=md.get("sme_plan_k_strip", ""),
                        b_panel_pad=md.get("sme_plan_b_panel_pad", ""),
                        b_global_pad=md.get("sme_plan_b_global_pad", ""),
                    )
                )
                raw.extend(
                    dict(
                        process=proc,
                        size=size,
                        method=label,
                        **{k: v for k, v in r.items() if k != "method"},
                    )
                    for r in t["raw"]
                    if r["method"] == mode
                )
            print(f"Rank: process {proc + 1}/{a.processes}, size {size}", flush=True)
    for name, data in [("rank.csv", rows), ("rank_samples.csv", raw), ("rank_checks.csv", checks)]:
        with (a.output / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    (a.output / "environment_rank.json").write_text(
        json.dumps(
            dict(
                status="complete",
                profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),
                compiler_sha256=compiler_sha256(),
                sizes=a.sizes,
                threads=a.threads,
                processes=a.processes,
                rounds=a.rounds,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
