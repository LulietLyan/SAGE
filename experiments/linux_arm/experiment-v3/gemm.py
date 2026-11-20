#!/usr/bin/env python3
"""Run paired, full-call FP32 GEMM windows inside a Donau allocation."""
import argparse
import csv
import ctypes
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import statistics
import subprocess
import sys
import time

ROOT = Path(os.environ.get("SAGE_ROOT", Path(__file__).resolve().parents[1] / "sage"))
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "python")]
from common import measure_interleaved

SQUARES = (128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1152,
           1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096)
EXTRA = ((127, 1009, 513), (255, 4096, 1023), (4095, 257, 1025),
         (255, 255, 255), (257, 257, 257), (1009, 1023, 1031), (1057, 1057, 1057),
         (2047, 2047, 2047), (2049, 2049, 2049))
PROBES = {"fixed-control": 0, "vector-a": 10, "tile-m": 11, "k-strip": 12,
          "local-b": 13, "global-b": 14, "measured-tile": 1, "measured-strip": 2,
          "measured-local": 4, "measured-global": 8}
RANKS = ("ours-rank-2", "ours-rank-5", "ours-rank-10", "ours-rank-20", "ours-worst")
TRITON_METHODS = ("generic", "sme-fixed", "ours", *PROBES, *RANKS)
METHODS = (*TRITON_METHODS, "openblas", "kblas", "pytorch")
DEDUPE = tuple(dict.fromkeys(tuple((s, s, s) for s in SQUARES) + EXTRA))


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def csv_append(path, rows):
    if not rows:
        return
    new = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def cpu_ids(text):
    for spec in text.strip().split(","):
        if spec:
            edges = [int(n) for n in spec.split("-")]
            yield from range(edges[0], edges[-1] + 1)


def cpu_nodes():
    allowed = os.sched_getaffinity(0)
    return {
        int(p.name.removeprefix("node")): sorted(set(cpu_ids((p / "cpulist").read_text())) & allowed)
        for p in sorted(Path("/sys/devices/system/node").glob("node[0-9]*"))
    }


def node_free_bytes(node):
    for line in Path(f"/sys/devices/system/node/node{node}/meminfo").read_text().splitlines():
        if "MemFree:" in line:
            return int(line.split()[-2]) * 1024
    raise RuntimeError(f"missing NUMA node {node} MemFree")


@lru_cache(None)
def groups_available():
    return sorted(((k, v) for k, v in cpu_nodes().items() if v),
                  key=lambda pair: (-len(pair[1]), -node_free_bytes(pair[0]), pair[0]))


def placement(topology, budget):
    groups = groups_available()
    if topology == "single":
        node, cpus = groups[0]
        if len(cpus) < budget:
            raise ValueError(f"single NUMA needs {budget} allocated CPUs, got {groups}")
        return cpus[:budget], [node], "membind"
    left = (budget + 1) // 2
    if budget < 2 or len(groups) < 2 or len(groups[0][1]) < left or len(groups[1][1]) < budget - left:
        raise ValueError(f"cross NUMA needs a spread {budget} CPU allocation, got {groups}")
    (node_a, cpus_a), (node_b, cpus_b) = groups[:2]
    return cpus_a[:left] + cpus_b[:budget-left], [node_a, node_b], "interleave"


def bind_args(cpus, nodes, policy):
    cpu_list = ",".join(map(str, cpus))
    node_list = ",".join(map(str, nodes))
    return ["numactl", f"--physcpubind={cpu_list}", f"--{policy}={node_list}"]


def init_library(path, method, cpus, symbol, blas_int):
    threads = len(cpus)
    path = str(Path(path).resolve(strict=True))
    begin = time.perf_counter_ns()
    library = ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
    load_ns = time.perf_counter_ns() - begin
    query_prefix = "openblas_" if method == "openblas" else "Blas"
    config = {}
    for name in (query_prefix + "get_config", query_prefix + "get_corename"):
        fn = getattr(library, name, None)
        if fn is not None:
            fn.restype = ctypes.c_char_p
            config[name] = fn().decode()
    if any("USE64BITINT" in v for v in config.values()):
        raise ValueError("ELF reports a 64-bit BLAS integer build")
    setter = "openblas_set_num_threads" if method == "openblas" else "BlasSetNumThreads"
    getter = "openblas_get_num_threads" if method == "openblas" else "BlasGetNumThreads"
    if method == "openblas" and symbol.startswith("scipy_"):
        setter, getter = "scipy_" + setter, "scipy_" + getter
        for name in ("scipy_openblas_get_config", "scipy_openblas_get_corename"):
            fn = getattr(library, name)
            fn.restype = ctypes.c_char_p
            config[name] = fn().decode()
        if any("USE64BITINT" in v for v in config.values()):
            raise ValueError("ELF reports a 64-bit BLAS integer build")
    setter = os.environ.get("KBLAS_SET_THREADS", setter) if method == "kblas" else setter
    getter = os.environ.get("KBLAS_GET_THREADS", getter) if method == "kblas" else getter
    set_threads = getattr(library, setter, None)
    if set_threads is not None:
        set_threads.argtypes = [ctypes.c_int]
        set_threads.restype = None
        set_threads(threads)
    get_threads = getattr(library, getter, None)
    effective = get_threads() if get_threads is not None else None
    if effective is not None and effective != threads:
        raise RuntimeError(f"{method} thread budget {threads} differs from runtime report {effective}")
    binding = {}
    if method == "openblas":
        prefix = "scipy_" if symbol.startswith("scipy_") else ""
        parallel = getattr(library, prefix + "openblas_get_parallel")()
        config["parallel"] = parallel
        if parallel == 1:
            setter_fn = getattr(library, prefix + "openblas_setaffinity")
            getter_fn = getattr(library, prefix + "openblas_getaffinity")
            for fn in (setter_fn, getter_fn):
                fn.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
                fn.restype = ctypes.c_int
            word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
            Mask = ctypes.c_ulong * ((max(max(cpus) + 1, 1024) + word_bits - 1) // word_bits)
            for index, cpu in enumerate([*cpus[1:], cpus[0]]):
                mask = Mask()
                mask[cpu // word_bits] = 1 << (cpu % word_bits)
                observed = Mask()
                status = setter_fn(index, ctypes.sizeof(mask), ctypes.byref(mask))
                read_status = getter_fn(index, ctypes.sizeof(observed), ctypes.byref(observed))
                if status != 0 or read_status != 0 or list(observed) != list(mask):
                    raise RuntimeError(f"cannot pin/verify OpenBLAS thread {index} to CPU {cpu}")
                binding[index] = cpu
    i32, scalar, ptr = ctypes.c_int32, ctypes.c_float, ctypes.c_void_p
    size = ctypes.c_int64 if blas_int == 64 else i32
    gemm = getattr(library, symbol)
    gemm.argtypes = [i32, i32, i32, size, size, size, scalar, ptr, size, ptr, size,
                     scalar, ptr, size]
    gemm.restype = None
    return gemm, dict(library=path, sha256=file_sha(path), symbol=symbol, load_ns=load_ns,
                      blas_integer_bits=blas_int, runtime_threads=effective,
                      setter=setter if set_threads else None, getter=getter if get_threads else None,
                      config=config, pinned_blas_threads=binding)


def mappings():
    return sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                   if line.split()[-1].startswith("/") and ".so" in line.split()[-1]})


def validate(np, out, ref32, reference, scale, rows, k):
    delta = np.abs(out - ref32)
    finite = bool(np.isfinite(out).all())
    sampled_error = np.abs(out[rows].astype(np.float64) - reference)
    tolerance_ratio = float(np.max(delta / (2e-5 * math.sqrt(k) + 2e-4 * np.abs(ref32))))
    gamma = (2 * k * 2**-24) / (1 - 2 * k * 2**-24)
    gamma_ratio = float(np.max(sampled_error / (gamma * scale + 1e-30)))
    return dict(passed=finite and tolerance_ratio <= 1 and gamma_ratio <= 1, finite=finite,
                fp32_max_abs=float(delta.max()), fp32_tolerance_ratio=tolerance_ratio,
                fp64_max_abs=float(sampled_error.max()), fp64_gamma_2k_ratio=gamma_ratio)


def worker(job):
    import numpy as np

    budget = job["threads"]
    methods = job["methods"]
    assert sorted(os.sched_getaffinity(0)) == sorted(job["cpus"])
    if any(m in TRITON_METHODS for m in methods):
        from sme_plan.compiler_backend import compile_gemm, compiler_sha256, kernel, prepare
        torch = prepare(budget, job["cache"])
        profile = json.loads(Path(job["profile"]).read_text()) if job["profile"] else None
        if profile:
            profile_id = profile["compilerSha256"], profile["sourceSha256"]
            expected = compiler_sha256(), hashlib.sha256(kernel().src.encode()).hexdigest()
            if profile_id != expected:
                raise ValueError("profile compiler/operator mismatch")
    elif "pytorch" in methods:
        import torch
        torch.set_num_threads(budget)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
    else:
        torch = None

    libraries, provenance = {}, {}
    for name in ("openblas", "kblas"):
        if name in methods:
            default_symbol = "cblas_sgemm"
            libraries[name], provenance[name] = init_library(
                job[name], name, job["cpus"], job.get(name + "_symbol") or default_symbol,
                job.get(name + "_integer_bits", 32))
    if torch is not None:
        provenance["pytorch"] = dict(version=str(torch.__version__), threads=torch.get_num_threads(),
                                     interop=torch.get_num_interop_threads(),
                                     build=torch.__config__.show(), parallel=torch.__config__.parallel_info())
    out_dir = Path(job["output"])
    plan = list(job["shapes"])
    random.Random(1194 + job["process"]).shuffle(plan)
    for shape in plan:
        m, n, k = shape
        rng = np.random.default_rng(91471 + 17*m + 43*n + 101*k + 193*job["process"])
        a, b = rng.standard_normal((m, k), dtype=np.float32), rng.standard_normal((k, n), dtype=np.float32)
        rows = np.unique(np.linspace(0, m-1, num=min(16, m)).round().astype(int))
        a64, b64 = a[rows].astype(np.float64), b.astype(np.float64)
        reference = a64 @ b64
        np.abs(b64, out=b64)
        scale = np.abs(a64) @ b64
        del a64, b64
        ref32 = a @ b
        calls, checks = {}, []
        common = dict(topology=job["topology"], threads=budget, process=job["process"], pid=os.getpid(),
                      hostname=socket.gethostname(), m=m, n=n, k=k)
        for name in methods:
            out = np.empty((m, n), dtype=np.float32)
            out.fill(np.nan)
            evidence = {}
            if name in libraries:
                fn = libraries[name]
                if job["blas_layout"] == "column-alias":
                    # The same row-major output is C.T in column-major storage: B.T @ A.T.
                    args = (102, 111, 111, n, m, k, 1.0, b.ctypes.data, n, a.ctypes.data, k,
                            0.0, out.ctypes.data, n)
                else:
                    args = (101, 111, 111, m, n, k, 1.0, a.ctypes.data, k, b.ctypes.data, n,
                            0.0, out.ctypes.data, n)

                def call(fn=fn, args=args, out=out):
                    fn(*args)
                    return out
            else:
                ta, tb, tc = map(torch.from_numpy, (a, b, out))
                if name == "pytorch":
                    def call(a=ta, b=tb, out=tc):
                        torch.mm(a, b, out=out)
                else:
                    if name in PROBES:
                        mode = "calibration-probe"
                        params = dict(profile, calibrationProbeIndex=PROBES[name])
                    else:
                        mode = name
                        params = profile if name.startswith("ours") else None
                    call = compile_gemm(ta, tb, tc, mode, budget, params)
                    evidence = call.evidence
            if not evidence:
                begin = time.perf_counter_ns()
                call()
                evidence = dict(first_call_ns=time.perf_counter_ns()-begin, executed_path=name)
            error = validate(np, out, ref32, reference, scale, rows, k)
            checks.append(dict(**common, method=name, **error,
                               first_call_ns=evidence["first_call_ns"], compile_ns=evidence.get("compile_ns", ""),
                               executed_path=evidence["executed_path"], fmopa_count=evidence.get("fmopa_count", ""),
                               jit_binary_sha256=evidence.get("jit_binary_sha256", ""),
                               evidence_json=json.dumps(evidence, sort_keys=True)))
            calls[name] = call
        csv_append(out_dir / "checks.csv", checks)
        if not all(c["passed"] for c in checks):
            raise RuntimeError(f"FP32 GEMM correctness failed for {shape}: {checks}")
        timing = measure_interleaved(calls, rounds=job["rounds"], target_s=job["window"],
                                     seed=173+m+93*n+17*k+job["process"]*731+budget,
                                     phase="holdout")
        csv_append(out_dir / "samples.csv", [dict(**common, **row) for row in timing["raw"]])
        parsed = [dict(**common, method=name, gflops=2*m*n*k/row["median_ns"], **row)
                  for name, row in timing["summary"].items()]
        csv_append(out_dir / "process_results.csv", parsed)
        print(f"{job['topology']} t{budget} p{job['process']} {m}×{n}×{k}: " +
              " ".join(f"{x['method']}={x['gflops']:.1f}" for x in parsed), file=sys.stderr, flush=True)
    print(json.dumps(dict(provenance=provenance, mapped_libraries=mappings(),
                          cgroup=Path("/proc/self/cgroup").read_text(),
                          python=platform.python_version(), numpy=np.__version__,
                          cpu_affinity=sorted(os.sched_getaffinity(0)),
                          thread_affinities={t.name: sorted(os.sched_getaffinity(int(t.name)))
                                             for t in Path("/proc/self/task").iterdir()},
                          numa_policy=subprocess.check_output(["numactl", "--show"], text=True),
                          uname=platform.uname()._asdict())))


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="started", processes=args.processes, rounds=args.rounds, window=args.window,
                    methods=args.methods, scheduler_cpu_nodes=cpu_nodes(), jobs=[], skips=[],
                    profile_sha256=file_sha(args.profile) if args.profile else None)
    record = output / "environment.json"
    def save():
        record.write_text(json.dumps(manifest, indent=2) + "\n")
    save()
    if any(m == "ours" or m in PROBES or m in RANKS for m in args.methods) and not args.profile:
        raise ValueError("ours requires the independently measured Linux profile")
    if any(m in args.methods for m in ("openblas", "kblas") if getattr(args, m) is None):
        raise ValueError("supply the exact shared library for each requested BLAS method")
    shapes = [tuple(map(int, s.split("x"))) for s in args.shapes] if args.shapes else DEDUPE
    try:
        for index in range(args.processes):
            order = args.threads[index % len(args.threads):] + args.threads[:index % len(args.threads)]
            for topology in args.topology:
                for budget in order:
                    try:
                        cpus, nodes, policy = placement(topology, budget)
                    except ValueError as error:
                        manifest["skips"].append(dict(topology=topology, threads=budget, reason=str(error)))
                        continue
                    info = dict(topology=topology, threads=budget, process=index, cpus=cpus, nodes=nodes,
                                policy=policy, methods=args.methods, shapes=shapes,
                                rounds=args.rounds, window=args.window, profile=args.profile,
                                cache=str(output / "jit-cache"),
                                openblas=args.openblas, openblas_symbol=args.openblas_symbol,
                                blas_layout=args.blas_layout,
                                kblas=args.kblas, kblas_integer_bits=args.kblas_integer_bits,
                                output=str(output))
                    env = dict(os.environ, OMP_NUM_THREADS=str(budget), OMP_THREAD_LIMIT=str(budget),
                               OPENBLAS_DEFAULT_NUM_THREADS=str(budget), OPENBLAS_NUM_THREADS=str(budget),
                               MKL_NUM_THREADS=str(budget), OPENBLAS_MAIN_FREE="1", OMP_DYNAMIC="FALSE",
                               OMP_WAIT_POLICY="PASSIVE", OMP_PROC_BIND="spread",
                               OMP_PLACES=",".join("{" + str(cpu) + "}" for cpu in cpus),
                               KMP_BLOCKTIME="0",
                               PYTHONDONTWRITEBYTECODE="1")
                    cmd = bind_args(cpus, nodes, policy) + [sys.executable, __file__, "--worker"]
                    result = subprocess.run(cmd, input=json.dumps(info), env=env,
                                            text=True, stdout=subprocess.PIPE)
                    if result.returncode:
                        raise RuntimeError(f"worker failure {result.returncode}: {info}; {result.stdout}")
                    info["identity"] = json.loads(result.stdout.splitlines()[-1])
                    manifest["jobs"].append(info)
                    save()
        manifest["status"] = "complete" if manifest["jobs"] else "no_placeable_jobs"
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        save()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--output", type=Path)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=["openblas"])
    p.add_argument("--threads", nargs="+", type=int, default=[1, 8, 2, 4])
    p.add_argument("--topology", nargs="+", choices=["single", "cross"], default=["single", "cross"])
    p.add_argument("--processes", type=int, default=5)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--window", type=float, default=0.025)
    p.add_argument("--shapes", nargs="+")
    p.add_argument("--profile")
    p.add_argument("--openblas")
    p.add_argument("--openblas-symbol")
    p.add_argument("--blas-layout", choices=["row", "column-alias"], default="row")
    p.add_argument("--kblas")
    p.add_argument("--kblas-integer-bits", type=int, choices=[32, 64], default=32)
    return p


if __name__ == "__main__":
    options = parser().parse_args()
    if options.worker:
        worker(json.load(sys.stdin))
    elif options.output:
        run(options)
    else:
        raise SystemExit("--output is required")
