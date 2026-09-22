#!/usr/bin/env python3
"""Four methods, one Triton operator, independent processes, flat CSV output."""

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT
sys.path.insert(0, str(ROOT / "python"))
from sme_plan.compiler_backend import (
    prepare,
    compile_gemm,
    compiler_sha256,
    compiler_path,
    sha256,
    kernel,
)
from common import check_gemm, measure_interleaved

SIZES = [
    128,
    192,
    256,
    320,
    384,
    448,
    512,
    640,
    768,
    896,
    1024,
    1152,
    1280,
    1536,
    1792,
    2048,
    2304,
    2560,
    2816,
    3072,
    3328,
    3584,
    3840,
    4096,
]
# SME / Ours reject shapes below ~32 on the current compiler entry; dense v2
# starts there and continues past 4096 with midpoints between powers of two.
DENSE_SIZES = [
    32,
    40,
    48,
    56,
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
    640,
    768,
    896,
    1024,
    1280,
    1536,
    1792,
    2048,
    2560,
    3072,
    3584,
    4096,
    4608,
    5120,
    5632,
    6144,
    7168,
    8192,
]
# Dense probe of the large-N regime: GFLOP/s stability vs TritonCPU / PyTorch.
# Fixed SME rejects N=8192 on the current ConvertDot→ukernel path; keep ≤7168
# for four-method sweeps. Main triad can still be probed at 8192 separately.
LARGE_SIZES = [
    2048,
    2304,
    2560,
    2816,
    3072,
    3328,
    3584,
    3840,
    4096,
    4352,
    4608,
    4864,
    5120,
    5376,
    5632,
    5888,
    6144,
    6400,
    6656,
    6912,
    7168,
]
METHODS = ["generic", "sme-fixed", "ours", "pytorch"]
# Main-triad large probe (skips Fixed SME so N=8192 is reachable).
LARGE_TRIAD_SIZES = [
    4096,
    4352,
    4608,
    4864,
    5120,
    5376,
    5632,
    5888,
    6144,
    6400,
    6656,
    6912,
    7168,
    7424,
    7680,
    7936,
    8192,
]
PARAMETERS = {
    "selected_sme": "selected_sme",
    "selected_threads": "threads",
    "block_m": "block_m",
    "block_n": "block_n",
    "block_k": "block_k",
    "k_strip": "k_strip",
    "a_pack_vector4": "a_pack_vector4",
    "b_panel_pad": "b_panel_pad",
    "b_global_pad": "b_global_pad",
    "physical_ldb": "physical_ldb",
}


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write empty measurements")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def worker(options):
    torch = prepare(options["threads"], options["cache"])
    profile = json.loads(Path(options["profile"]).read_text())
    if sha256(options["profile"]) != options["profile_file_sha256"]:
        raise ValueError("Calibration profile changed during the experiment")
    if profile["compilerSha256"] != compiler_sha256():
        raise ValueError("The compiler changed; run experiments/calibrate.py first")
    source_sha = hashlib.sha256(kernel().src.encode()).hexdigest()
    if profile["sourceSha256"] != source_sha:
        raise ValueError("The Triton source changed; recalibrate before measurement")
    methods = list(options.get("methods") or METHODS)
    if not methods or any(m not in METHODS for m in methods):
        raise ValueError(f"Unsupported methods: {methods}")
    results, samples, checks = [], [], []
    identity = dict(
        pid=os.getpid(),
        python=platform.python_version(),
        torch_version=str(torch.__version__),
        torch_threads=torch.get_num_threads(),
        torch_entry="torch.mm(a,b,out=preallocated)",
        torch_backend="Accelerate"
        if "accelerate" in torch.__config__.show().lower()
        else "unknown",
        source_sha256=source_sha,
        compiler_sha256=compiler_sha256(),
        profile_file_sha256=options["profile_file_sha256"],
    )
    for size in options["sizes"]:
        generator = torch.Generator().manual_seed(91471 + size)
        a = torch.randn(size, size, generator=generator)
        b = torch.randn(size, size, generator=generator)
        reference = a @ b
        calls = {}
        for method in methods:
            if method == "pytorch":
                out = torch.empty_like(reference)
                calls["pytorch"] = lambda a=a, b=b, out=out: torch.mm(a, b, out=out)
            else:
                calls[method] = compile_gemm(
                    a,
                    b,
                    torch.empty_like(reference),
                    method,
                    options["threads"],
                    profile if method == "ours" else None,
                )
        common = dict(
            threads=options["threads"],
            process=options["process"],
            pid=os.getpid(),
            m=size,
            n=size,
            k=size,
        )
        numerical = {}
        for method, call in calls.items():
            check = check_gemm(call(), reference, a, b)
            if not check["passed"]:
                raise RuntimeError(f"{method}/{size} correctness failed: {check}")
            numerical[method] = check
            evidence = getattr(call, "evidence", {})
            checks.append(
                dict(
                    **common,
                    method=method,
                    correctness_passed=check["passed"],
                    fp64_max_abs=check["fp64_max_abs"],
                    fp64_gamma_2k_ratio=check["fp64_gamma_2k_ratio"],
                    backend_verified=evidence.get("backend_verified", ""),
                    executed_path=evidence.get("executed_path", "pytorch"),
                    fmopa_count=evidence.get("fmopa_count", ""),
                    jit_binary_sha256=evidence.get("jit_binary_sha256", ""),
                    compiler_sha256=evidence.get("compiler_sha256", ""),
                    source_sha256=source_sha,
                )
            )
        timing = measure_interleaved(
            calls,
            rounds=options["rounds"],
            target_s=options["window"],
            seed=173 + size + options["process"] * 731 + options["threads"],
            phase="measurement",
        )
        samples.extend(dict(**common, **row) for row in timing["raw"])
        for method in methods:
            observed = timing["summary"][method]
            evidence = getattr(calls[method], "evidence", {})
            metadata = evidence.get("metadata", {})
            results.append(
                dict(
                    **common,
                    method=method,
                    median_ns=observed["median_ns"],
                    gflops=2 * size**3 / observed["median_ns"],
                    timing_windows=options["rounds"],
                    minimum_window_ns=observed["min_ns"],
                    maximum_window_ns=observed["max_ns"],
                    compile_or_cache_lookup_ns=evidence.get("compile_ns", ""),
                    first_compiled_call_ns=evidence.get("first_call_ns", ""),
                    **{
                        name: metadata.get("sme_plan_" + key, "")
                        for name, key in PARAMETERS.items()
                    },
                    profile_sha256=evidence.get("profile_sha256", ""),
                    source_sha256=source_sha,
                    compiler_sha256=evidence.get("compiler_sha256", ""),
                    jit_binary_sha256=evidence.get("jit_binary_sha256", ""),
                    backend_verified=evidence.get("backend_verified", ""),
                    executed_path=evidence.get("executed_path", "pytorch"),
                    correctness_passed=True,
                    torch_version=identity["torch_version"],
                    torch_entry=identity["torch_entry"],
                )
            )
        del calls, a, b, reference
        gc.collect()
    return dict(
        results=results,
        samples=samples,
        checks=checks,
        identity=identity,
        torch_build_config=torch.__config__.show(),
    )


def run(
    output=WORKSPACE / "output",
    *,
    sizes=SIZES,
    threads=(1, 8),
    processes=5,
    rounds=4,
    window=0.025,
    stem="results",
    env_name="environment",
    clear_legacy_plots=True,
    methods=None,
):
    """Run GEMM sweep.

    ``stem`` prefixes CSV artifacts (``{stem}.csv``, ``{stem}_samples.csv``,
    ``{stem}_checks.csv``). Legacy stem ``results`` keeps the original names
    ``results.csv`` / ``samples.csv`` / ``checks.csv``.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile_path = output / "profile.json"
    profile = json.loads(profile_path.read_text())
    profile_file_sha256 = sha256(profile_path)
    expected_identity = dict(
        compiler_sha256=compiler_sha256(),
        source_sha256=profile["sourceSha256"],
        profile_file_sha256=profile_file_sha256,
    )
    build = json.loads((ROOT / ".build/build.json").read_text())
    if profile["compilerSha256"] != compiler_sha256():
        raise ValueError("Compiler/profile mismatch; recalibrate before starting")
    if (
        not sizes
        or sizes != sorted(set(sizes))
        or min(sizes) < 1
        or processes < 1
        or rounds < 1
        or window <= 0
    ):
        raise ValueError("Expected increasing sizes and positive process/window settings")
    if not threads or len(set(threads)) != len(threads) or min(threads) < 1:
        raise ValueError("Thread budgets must be distinct positive integers")
    methods = list(methods or METHODS)
    if not methods or any(m not in METHODS for m in methods) or len(set(methods)) != len(methods):
        raise ValueError(f"Invalid method list: {methods}")
    if stem == "results":
        result_csv, sample_csv, check_csv = "results.csv", "samples.csv", "checks.csv"
        report_md = "results.md"
    else:
        result_csv, sample_csv, check_csv = (
            f"{stem}.csv",
            f"{stem}_samples.csv",
            f"{stem}_checks.csv",
        )
        report_md = f"{stem}.md"
    environment = dict(
        status="running",
        started_ns=time.time_ns(),
        sizes=sizes,
        thread_budgets=list(threads),
        processes=processes,
        rounds=rounds,
        window_s=window,
        dtype="float32",
        stem=stem,
        result_csv=result_csv,
        methods=methods,
        compiler_sha256=compiler_sha256(),
        host=profile["device"]["host"],
        source_sha256=profile["sourceSha256"],
        profile_file_sha256=profile_file_sha256,
        triton_revision=build["revision"],
        patch_sha256=build["patch_sha256"],
        llvm_revision=build["llvm_revision"],
        vector_length_bits=profile["device"]["vectorLengthBits"],
        flops="2*M*N*K; GFLOP/s = FLOPs / median_ns",
        scope="Complete synchronous steady call: preparation, submission, computation and wait.",
        first_cost="Compile/cache lookup and first call recorded separately; shared phase cache, not all cold.",
        calibration_reuse="Measured coefficients only; calibration timings are not formal samples.",
        thread_scope="Requested budgets; physical SME unit count and Accelerate internal scheduling unknown.",
        note="SME/Ours require M,N,K≥32 on the current compiler entry; Fixed SME rejects N=8192.",
        workers=[],
    )
    env_path = output / f"{env_name}.json"

    def save_environment():
        env_path.write_text(json.dumps(environment, indent=2) + "\n")

    save_environment()
    if clear_legacy_plots and stem == "results":
        for pattern in ("gemm-t*-gflops.*", "gemm-t*-latency.*"):
            for path in output.glob(pattern):
                if path.suffix in (".pdf", ".svg", ".png"):
                    path.unlink()
    for name in (result_csv, sample_csv, check_csv, report_md):
        (output / name).unlink(missing_ok=True)
    combined = {name: [] for name in ("results", "samples", "checks")}
    paths = {"results": result_csv, "samples": sample_csv, "checks": check_csv}
    cache_directory = tempfile.TemporaryDirectory(prefix="gemm-cache-", dir=ROOT / ".build")
    cache = Path(cache_directory.name)
    try:
        for process in range(processes):
            order = list(threads[process % len(threads) :]) + list(
                threads[: process % len(threads)]
            )
            for budget in order:
                options = dict(
                    threads=budget,
                    process=process,
                    sizes=sizes,
                    rounds=rounds,
                    window=window,
                    methods=methods,
                    profile=str(profile_path),
                    cache=str(cache),
                    profile_file_sha256=profile_file_sha256,
                )
                child = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--worker"],
                    input=json.dumps(options),
                    text=True,
                    capture_output=True,
                )
                if child.returncode:
                    raise RuntimeError(
                        f"Process {process}, budget {budget} failed:\n{child.stderr[-6000:]}"
                    )
                measured = json.loads(child.stdout)
                if any(
                    measured["identity"].get(key) != value
                    for key, value in expected_identity.items()
                ):
                    raise RuntimeError(
                        "Worker used a different compiler, source or calibration profile"
                    )
                if (
                    environment.get("torch_build_config", measured["torch_build_config"])
                    != measured["torch_build_config"]
                ):
                    raise RuntimeError("PyTorch configuration changed between workers")
                environment["torch_build_config"] = measured["torch_build_config"]
                for name in combined:
                    combined[name].extend(measured[name])
                    write_csv(output / paths[name], combined[name])
                environment["workers"].append(
                    dict(process=process, threads=budget, **measured["identity"])
                )
                save_environment()
                print(
                    f"GEMM[{stem}]: process {process + 1}/{processes}, budget {budget}: {len(sizes)} sizes passed",
                    flush=True,
                )
        expected = len(sizes) * len(threads) * processes * len(methods)
        if len(combined["results"]) != expected or len(combined["samples"]) != expected * rounds:
            raise RuntimeError("Incomplete experiment")
        if (
            sha256(profile_path) != profile_file_sha256
            or sha256(compiler_path()) != expected_identity["compiler_sha256"]
        ):
            raise RuntimeError("Compiler or profile changed during measurement")
        environment.update(
            status="complete",
            completed_ns=time.time_ns(),
            method_observations=expected,
            raw_windows=expected * rounds,
        )
    except BaseException as error:
        environment.update(status="failed", error=str(error))
        raise
    finally:
        save_environment()
        cache_directory.cleanup()
    return combined["results"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "output")
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument(
        "--dense", action="store_true", help="use DENSE_SIZES (32…8192) instead of legacy SIZES"
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="use LARGE_SIZES (2048…7168 dense) for GFLOP/s stability",
    )
    parser.add_argument(
        "--large-triad",
        action="store_true",
        help="large-N main triad only (generic/ours/pytorch), includes 8192",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        choices=list(METHODS),
        help="subset of methods to measure",
    )
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--window", type=float, default=0.025)
    parser.add_argument(
        "--stem",
        type=str,
        default="results",
        help="CSV stem; use results_v2 for suite v2 without overwriting legacy results.csv",
    )
    parser.add_argument("--env-name", type=str, default="environment")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
    else:
        if args.large_triad:
            sizes = LARGE_TRIAD_SIZES
            methods = args.methods or ["generic", "ours", "pytorch"]
        elif args.large:
            sizes = LARGE_SIZES
            methods = args.methods
        elif args.dense:
            sizes = DENSE_SIZES
            methods = args.methods
        else:
            sizes = args.sizes
            methods = args.methods
        run(
            args.output,
            sizes=sizes,
            threads=args.threads,
            processes=args.processes,
            rounds=args.rounds,
            window=args.window,
            stem=args.stem,
            env_name=args.env_name,
            methods=methods,
        )


if __name__ == "__main__":
    main()
