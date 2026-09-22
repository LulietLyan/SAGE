#!/usr/bin/env python3
"""Build, calibrate when needed, measure and plot the current experiment."""

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT


def command(*arguments):
    subprocess.run([str(x) for x in arguments], cwd=ROOT, check=True)


def report(output):
    with (output / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with (output / "samples.csv").open() as stream:
        samples = list(csv.DictReader(stream))
    environment = json.loads((output / "environment.json").read_text())
    if environment["status"] != "complete":
        raise RuntimeError("The measurement did not finish; no report will be generated")
    methods = ["generic", "sme-fixed", "ours", "pytorch"]
    conclusions = []
    lookup = {(int(r["threads"]), int(r["m"]), int(r["process"]), r["method"]): r for r in rows}
    lines = [
        "# GEMM results",
        "",
        f"Measured compiler SHA-256: `{environment['compiler_sha256']}`.",
        "",
        "Same FP32 Triton source and original launch constants in all three compiler modes. "
        "C++ chooses the schedule; all required preparation, submission and waiting are timed.",
        "",
        f"{environment['processes']} independent processes per thread budget, {environment['rounds']} interleaved windows per method/size. "
        "Each process contributes its window median. Curves show process medians and min-max ranges, not confidence intervals.",
        "",
        f"Timing-window durations are targets, not guaranteed minimums: "
        f"{sum(float(s['elapsed_ns']) < float(s['target_s']) * 1e9 for s in samples)}/{len(samples)} windows were shorter than their target; "
        f"the shortest was {min(float(s['elapsed_ns']) for s in samples) / 1e6:.2f} ms. "
        "All windows and repetition-selection reasons are retained in samples.csv.",
        "",
        "Calibration is separate from measurement. Python fits device response coefficients; production schedule ranking and operand preparation belong to the compiler/native launcher.",
        "",
        "Physical SME unit count and Accelerate internal scheduling remain unknown. Thread budgets are configured before Torch/Triton initialization.",
        "",
    ]
    for budget in environment["thread_budgets"]:
        lines += [
            f"## {budget} thread budget",
            "",
            "| Size | TritonCPU | Fixed SME | Ours | PyTorch | Paired ours/fixed | Paired ours/PyTorch |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        ratios = {"sme-fixed": [], "pytorch": []}
        for size in environment["sizes"]:
            process_ids = sorted(
                {
                    int(r["process"])
                    for r in rows
                    if int(r["threads"]) == budget and int(r["m"]) == size
                }
            )
            values = [
                statistics.median(float(lookup[budget, size, p, m]["gflops"]) for p in process_ids)
                for m in methods
            ]
            speedups = [
                statistics.median(
                    float(lookup[budget, size, p, reference]["median_ns"])
                    / float(lookup[budget, size, p, "ours"]["median_ns"])
                    for p in process_ids
                )
                for reference in ("sme-fixed", "pytorch")
            ]
            for reference, speedup in zip(ratios, speedups):
                ratios[reference].append(speedup)
            lines.append(
                f"| {size} | "
                + " | ".join(f"{v:.1f}" for v in values)
                + " | "
                + " | ".join(f"{v:.3f}x" for v in speedups)
                + " |"
            )
        lines += [
            "",
            "Throughput: GFLOP/s = 2MNK / measured nanoseconds. Ratios are paired within processes before taking the median.",
            "",
        ]
        for reference, values in ratios.items():
            gm = math.exp(statistics.mean(math.log(v) for v in values))
            losses = [s for s, v in zip(environment["sizes"], values) if v <= 1]
            lines.append(
                f"- Ours / {reference}: geometric mean {gm:.3f}x; {sum(v > 1 for v in values)}/{len(values)} sizes faster. No gain at: {losses or 'none'}."
            )
            conclusions.append(
                f"{budget}-thread budget, ours / {reference}: {gm:.3f}x geometric mean; "
                f"{sum(v > 1 for v in values)}/{len(values)} sizes faster by paired process median."
            )
        lines += [
            "",
            f"[Throughput](gemm-t{budget}-gflops.pdf) · [Latency](gemm-t{budget}-latency.pdf)",
            "",
        ]
    lines += [
        "## First cost and scope",
        "",
        "For the three compiler modes, compile/cache lookup and first invocation costs are in results.csv. "
        "PyTorch cold-start cost is unmeasured. "
        "Workers share a JIT cache; these are not independent cold compilations. Python/Torch import is outside the first-invocation measurement. "
        "Calibration and build are one-time setup costs and are not subtracted from steady timings.",
        "",
        "The current compiler entry supports proven static, dense, row-major FP32 GEMM. "
        "It chooses pure CPU or pure SME; mixed schedules and automatic Attention/FFN/KAN transfer are not implemented in this entry. "
        "No general speedup or mathematical proof follows from this finite experiment.",
        "",
        "Selected tiles, K segments, packing, B layout and thread teams are recorded in results.csv. "
        "Backend verdicts and numerical errors are in checks.csv; raw timing windows are in samples.csv. "
        "Measurement identities are frozen in results.csv and environment.json; a later calibration does not change those measurements.",
        "",
    ]
    if environment.get("note") or environment.get("measured_version"):
        lines[2:2] = [
            str(environment.get("measured_version", "")),
            str(environment.get("note", "")),
            "",
        ]
    if environment.get("workers"):
        worker = environment["workers"][0]
        lines[2:2] = [
            f"PyTorch {worker['torch_version']}: `{worker['torch_entry']}`; reported BLAS backend: {worker['torch_backend']}. Full build configuration is in environment.json.",
            "",
        ]
    lines[2:2] = [
        "\n".join("- " + conclusion for conclusion in conclusions),
        "",
        "These are measured median effects across the listed square GEMMs; small differences are not claims of statistical significance.",
        "",
    ]
    ours = [r for r in rows if r["method"] == "ours"]
    if all(str(r["b_global_pad"]) == "0" and str(r["b_panel_pad"]) == "0" for r in ours):
        lines += [
            "All measured ours cases selected the original B layout. Padding support was not used and cannot explain the observed speedups.",
            "",
        ]
    (output / "results.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use the reduced smoke protocol for the selected experiment",
    )
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="redraw existing complete CSV data without compilation or timing",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="run compiler/runtime checks after setup, without the full GEMM sweep",
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        help="measure and plot scaled dot-product attention phases",
    )
    parser.add_argument(
        "--kan", action="store_true", help="measure the efficient-KAN KANLinear forward"
    )
    parser.add_argument("--swiglu", action="store_true", help="measure and plot the SwiGLU block")
    parser.add_argument(
        "--enrich-figures",
        action="store_true",
        help="derive schedule-selection narrative figures from existing CSVs",
    )
    parser.add_argument(
        "--suite-v2",
        action="store_true",
        help="8-thread dense GEMM + Linear/BMM + suite_plot_v2 narrative figures",
    )
    parser.add_argument("--output", type=Path, default=WORKSPACE / "output")
    args = parser.parse_args()
    output = (args.output if args.output.is_absolute() else WORKSPACE / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only and (args.test or args.recalibrate):
        parser.error("--plot-only cannot be combined with --test or --recalibrate")
    if args.plot_only:
        operators = (
            (args.attention, "attention_results.csv", "attention_plot.py"),
            (args.kan, "kan.csv", "kan_plot.py"),
            (args.swiglu, "swiglu.csv", "swiglu_plot.py"),
        )
        default_gemm = not (
            args.suite_v2 or args.enrich_figures or any(selected for selected, _, _ in operators)
        )
        if args.suite_v2:
            command(sys.executable, ROOT / "experiments/suite_plot_v2.py", "--output", output)
        for selected, data, plotter in operators:
            if selected or (args.enrich_figures and (output / data).is_file()):
                command(sys.executable, ROOT / "experiments" / plotter, "--output", output)
        if default_gemm or (args.enrich_figures and (output / "results.csv").is_file()):
            command(sys.executable, ROOT / "experiments/plot.py", "--output", output)
            report(output)
        if default_gemm or args.enrich_figures:
            command(sys.executable, ROOT / "experiments/enrich_figures.py", "--output", output)
        print(f"Figures: {output}")
        return
    if not args.plot_only:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError(
                "The current native implementation is validated on macOS arm64 with SME"
            )
        command(sys.executable, ROOT / "scripts/build.py")
        build = json.loads((ROOT / ".build/build.json").read_text())
        command(
            "cmake",
            "-S",
            ROOT,
            "-B",
            ROOT / ".build/core",
            "-G",
            "Ninja",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_CXX_COMPILER={build['cxx']}",
            f"-DCMAKE_ASM_COMPILER={build['cc']}",
            "-DCMAKE_BUILD_TYPE=Release",
        )
        command("cmake", "--build", ROOT / ".build/core", "--parallel", "2")
        command("ctest", "--test-dir", ROOT / ".build/core", "--output-on-failure")
        command(sys.executable, "-m", "unittest", "discover", "-s", ROOT / "tests/python")
        source_query = (
            "import sys,hashlib;sys.path.insert(0,'python');"
            "from sme_plan.compiler_backend import prepare,kernel;prepare(1);"
            "print(hashlib.sha256(kernel().src.encode()).hexdigest())"
        )
        source_sha = subprocess.check_output(
            [sys.executable, "-c", source_query], cwd=ROOT, text=True
        ).strip()
        profile_path = output / "profile.json"
        profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
        if (
            args.recalibrate
            or profile.get("compilerSha256") != build["compiler_sha256"]
            or profile.get("sourceSha256") != source_sha
        ):
            command(sys.executable, WORKSPACE / "experiments/calibrate.py", "--output", output)
        if args.test:
            command(sys.executable, ROOT / "tests/runtime.py", "--profile", profile_path)
            return
        if args.attention:
            attention_args = [
                sys.executable,
                WORKSPACE / "experiments/attention.py",
                "--output",
                output,
            ]
            if args.quick:
                attention_args += [
                    "--lengths",
                    "64",
                    "128",
                    "256",
                    "512",
                    "--processes",
                    "1",
                    "--rounds",
                    "2",
                ]
            command(*attention_args)
            command(sys.executable, WORKSPACE / "experiments/attention_plot.py", "--output", output)
        if args.kan:
            kan_args = [
                sys.executable,
                WORKSPACE / "experiments/kan.py",
                "--output",
                output,
                "--threads",
                "1",
                "8",
            ]
            if args.quick:
                kan_args += ["--quick", "--rounds", "2"]
            command(*kan_args)
            command(sys.executable, WORKSPACE / "experiments/kan_plot.py", "--output", output)
        if args.swiglu:
            swiglu_args = [
                sys.executable,
                WORKSPACE / "experiments/swiglu.py",
                "--output",
                output,
                "--threads",
                "1",
                "8",
            ]
            if args.quick:
                swiglu_args += ["--quick", "--rounds", "2"]
            command(*swiglu_args)
            command(sys.executable, WORKSPACE / "experiments/swiglu_plot.py", "--output", output)
        if args.suite_v2:
            gemm_v2 = [
                sys.executable,
                WORKSPACE / "experiments/gemm.py",
                "--output",
                output,
                "--dense",
                "--threads",
                "8",
                "--stem",
                "results_v2",
                "--env-name",
                "environment_v2",
            ]
            if args.quick:
                # Smoke runs use explicit sizes instead of the dense sweep.
                gemm_v2 = [
                    sys.executable,
                    WORKSPACE / "experiments/gemm.py",
                    "--output",
                    output,
                    "--sizes",
                    "32",
                    "64",
                    "128",
                    "256",
                    "512",
                    "1024",
                    "--threads",
                    "8",
                    "--processes",
                    "1",
                    "--rounds",
                    "2",
                    "--stem",
                    "results_v2",
                    "--env-name",
                    "environment_v2",
                ]
            else:
                gemm_v2 += ["--processes", "3", "--rounds", "3"]
            command(*gemm_v2)
            for script in ("linear.py", "bmm.py"):
                op = [
                    sys.executable,
                    WORKSPACE / "experiments" / script,
                    "--output",
                    output,
                    "--threads",
                    "8",
                ]
                if args.quick:
                    op += ["--quick", "--rounds", "2"]
                else:
                    op += ["--rounds", "3"]
                command(*op)
            command(sys.executable, WORKSPACE / "experiments/suite_plot_v2.py", "--output", output)
            print(f"Suite v2: {output / 'figures'}")
            return
        if args.attention or args.kan or args.swiglu:
            command(sys.executable, WORKSPACE / "experiments/enrich_figures.py", "--output", output)
            return
        benchmark = [sys.executable, WORKSPACE / "experiments/gemm.py", "--output", output]
        if args.quick:
            benchmark += ["--sizes", "128", "1024", "4096", "--processes", "1", "--rounds", "3"]
        command(*benchmark)
    command(sys.executable, WORKSPACE / "experiments/plot.py", "--output", output)
    report(output)
    command(sys.executable, WORKSPACE / "experiments/enrich_figures.py", "--output", output)
    print(f"Results: {output / 'results.md'}")
    print(f"Enrichment figures: {output / 'figures'}")


if __name__ == "__main__":
    main()
