#!/usr/bin/env python3
"""Build the shape-specialized generic kernels in parallel without timing GEMM."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gemm import DEDUPE


def compile_worker(cache, configs):
    from sme_plan.compiler_backend import kernel, prepare

    torch = prepare(1, cache)
    for (m, n, k), budget in configs:
        a = torch.empty((m, k), dtype=torch.float32)
        b = torch.empty((k, n), dtype=torch.float32)
        c = torch.empty((m, n), dtype=torch.float32)
        grid = (((m + 31) // 32) * ((n + 63) // 64), 1, 1)
        begin = time.perf_counter_ns()
        fn = kernel().warmup(
            a, b, c, m, n, k, 32, 64, 16, grid=grid,
            num_cpu_threads=budget, sme_plan_mode="generic", sme_plan_profile="",
            enable_fast_math=False, enable_fp_fusion=True, assume_in_bounds=False, ukernels=None,
        )
        fn._init_handles()
        print(json.dumps(dict(m=m, n=n, k=k, threads=budget,
                              compilation_ns=time.perf_counter_ns()-begin,
                              cpu=list(os.sched_getaffinity(0)))), flush=True)


def run(args):
    cpus = sorted(os.sched_getaffinity(0))
    if args.phase == "calibration":
        from sme_plan.calibration import SIZES, TEAMS
        shapes, teams = [(s, s, s) for s in SIZES], TEAMS
    else:
        shapes, teams = DEDUPE, (1, 8)
    cases = list(itertools.product(shapes, teams))
    count = min(len(cpus), len(cases))
    args.cache.mkdir(parents=True, exist_ok=True)

    def build(index):
        env = dict(os.environ, OMP_PROC_BIND="spread", OMP_PLACES="{" + str(cpus[index]) + "}")
        subprocess.run(["taskset", "-c", str(cpus[index]), sys.executable, __file__, args.phase,
                        "--cache", str(args.cache), "--worker"],
                       input=json.dumps(cases[index::count]), text=True, env=env, check=True)
    with ThreadPoolExecutor(max_workers=count) as pool:
        list(pool.map(build, range(count)))


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("phase", choices=["calibration", "holdout"])
p.add_argument("--cache", type=Path, required=True)
p.add_argument("--worker", action="store_true")
args = p.parse_args()
if args.worker:
    compile_worker(args.cache, json.load(sys.stdin))
else:
    run(args)
