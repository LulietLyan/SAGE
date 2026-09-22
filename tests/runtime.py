#!/usr/bin/env python3
"""Small correctness and compiler/launcher contract checks; no benchmarking.

Run after building the compiler and generating output/profile.json. Each thread
budget uses a fresh process. JIT caches and altered-source fixtures are temporary;
only pass/failure messages are retained on the terminal.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from sme_plan.compiler_backend import compile_gemm, compiler_path, kernel, prepare

SHAPE = (193, 257, 641)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_output(torch, a, b, output):
    aa, bb = a.double(), b.double()
    reference = aa @ bb
    absolute_sum = aa.abs() @ bb.abs()
    unit_roundoff = 2.0**-24
    ku = a.shape[1] * unit_roundoff
    bound = 4 * (ku / (1 - ku) * absolute_sum + unit_roundoff * reference.abs()) + 1e-30
    error = (output.double() - reference).abs()
    assert bool(torch.isfinite(output).all()), "Nonfinite GEMM output"
    assert bool((error <= bound).all()), "GEMM exceeds the FP32 accumulation error bound"
    torch.testing.assert_close(output.double(), reference, rtol=2e-4, atol=5e-5)


def check_case(torch, threads, profile, mode, probe=None):
    m, n, k = SHAPE
    generator = torch.Generator().manual_seed(7139)
    initial_a = torch.randn(m, k, generator=generator, dtype=torch.float32)
    initial_b = torch.randn(k, n, generator=generator, dtype=torch.float32)
    a, b = initial_a.clone(), initial_b.clone()
    output = torch.full((m, n), float("nan"), dtype=torch.float32)
    content = copy.deepcopy(profile) if mode in ("ours", "calibration-probe") else None
    if probe is not None:
        content["calibrationProbeIndex"] = probe
    call = compile_gemm(a, b, output, mode, threads, content)
    evidence, metadata = call.evidence, call.evidence["metadata"]
    label = mode if probe is None else f"probe-{probe}"
    assert evidence["backend_verified"] and not evidence["fallback"]
    assert not evidence["external_compute"]
    assert metadata["sme_plan_verified"] == 1
    assert metadata["sme_plan_mode"] == mode
    assert 1 <= metadata["sme_plan_threads"] <= threads
    if mode == "generic":
        assert evidence["executed_path"] == "generic-cpu" and evidence["fmopa_count"] == 0
    elif mode in ("sme-fixed", "calibration-probe"):
        assert evidence["executed_path"] == "compiler-native-sme"
        assert evidence["fmopa_count"] > 0 and evidence["direct_entry_calls_native_sme"]
    local_pad = metadata.get("sme_plan_b_panel_pad", 0)
    global_pad = metadata.get("sme_plan_b_global_pad", 0)
    assert (local_pad, global_pad) in ((0, 0), (16, 0), (0, 16))
    if probe is not None:
        assert (local_pad, global_pad) == {0: (0, 0), 4: (16, 0), 8: (0, 16)}[probe]
        # Probe identity stays in TTIR; the launcher exports its actual schedule.
        # Verify both instead of assuming every diagnostic attr is serialized.
        assert re.search(
            r"\bsme_plan\.calibration_probe\s*=\s*" + str(probe) + r"\s*:\s*i64\b",
            call.compiled.asm["ttir"],
        )
        fields = ("block_m", "block_n", "block_k", "k_strip", "a_pack_vector4")
        expected = (128, 256, 16, 0, 0) if probe == 0 else (256, 1024, 32, 512, 1)
        assert tuple(metadata[f"sme_plan_{field}"] for field in fields) == expected
    if global_pad:
        assert metadata["sme_plan_ldb"] == n
        assert metadata["sme_plan_ldc"] == n
        assert metadata["sme_plan_physical_ldb"] == n + global_pad
        assert metadata["sme_plan_global_b_workspace_bytes"] == 4 * k * (n + global_pad)
    check_output(torch, a, b, output)
    stats = call.compiled.run.get_global_b_stats
    runner = call.compiled[call.grid]

    def checked(lhs=a, rhs=b):
        output.fill_(float("nan"))
        before = stats()
        runner(lhs, rhs, output, *call.constants)
        after = stats()
        assert after["copy_call_count"] - before["copy_call_count"] == int(global_pad > 0)
        assert after["growth_count"] == before["growth_count"], "Same-shape workspace grew"
        if global_pad:
            assert after["last_copied_bytes"] == 4 * k * n
            assert after["last_requested_bytes"] == 4 * k * (n + global_pad)
            assert after["last_growth_ns"] == 0
            assert 1 <= after["last_copy_team_size"] <= metadata["sme_plan_threads"]
        check_output(torch, lhs, rhs, output)

    def rejected(operation, message, target=output):
        target.fill_(float("nan"))
        before = stats()
        try:
            operation()
        except ValueError as error:
            assert message in str(error), f"Unexpected rejection: {error}"
        else:
            raise AssertionError(f"Invalid launch was accepted: {message}")
        after = stats()
        assert bool(torch.isnan(target).all()), "Rejected launch modified output"
        assert after["copy_call_count"] == before["copy_call_count"]
        assert after["growth_count"] == before["growth_count"]

    checked()
    pointer = b.data_ptr()
    b.add_(0.25)
    assert b.data_ptr() == pointer and not torch.equal(b, initial_b)
    checked()
    b.zero_()
    checked()
    b.copy_(initial_b)
    a.zero_()
    checked()
    b.zero_()
    a[-1, -1], b[-1, -1] = 2.0, 3.0
    checked()
    assert output[-1, -1].item() == 6.0 and torch.count_nonzero(output).item() == 1
    a.copy_(initial_a)
    b.copy_(initial_b)
    checked(a, initial_b.clone())

    for axis in range(3):
        wrong_grid = list(call.grid)
        wrong_grid[axis] += 1
        rejected(
            lambda: call.compiled[tuple(wrong_grid)](a, b, output, *call.constants),
            "differs from compiler-proven grid",
        )
    original_metadata = call.compiled.packed_metadata

    def corrupt(fields, message):
        try:
            call.compiled.packed_metadata = original_metadata._replace(**fields)
            rejected(call, message)
        finally:
            call.compiled.packed_metadata = original_metadata

    corrupt({"sme_plan_threads": threads + 1}, "exceeds the caller thread budget")
    corrupt({"sme_plan_m": 2**62, "sme_plan_k": 8}, "storage byte count overflows")
    if global_pad:
        corrupt({"sme_plan_b_panel_pad": 16}, "Invalid compiler global-B padding contract")
        corrupt({"sme_plan_b_global_pad": 8}, "Invalid compiler global-B padding contract")

    for operand, elements, shape in (("A", m * k, (m, k)), ("B", k * n, (k, n))):
        for offset in (0, 16):
            storage = torch.randn(max(elements, offset + m * n), generator=generator)
            alias_input = storage[:elements].view(*shape)
            alias_output = storage[offset : offset + m * n].view(m, n)
            lhs, rhs = (alias_input, b) if operand == "A" else (a, alias_input)
            rejected(
                lambda: runner(lhs, rhs, alias_output, *call.constants),
                "must not overlap input A or B storage",
                alias_output,
            )
    shared = torch.randn(max(m * k, k * n), generator=generator)
    saved = shared.clone()
    checked(shared[: m * k].view(m, k), shared[: k * n].view(k, n))
    assert torch.equal(shared, saved), "Read-only A/B alias storage was modified"
    checked()
    print(
        f"PASS t{threads} {label}: tail, changed/zero B, final lane, aliases, grid/budget guards; "
        f"{evidence['executed_path']}",
        flush=True,
    )
    return evidence["source_sha256"]


def check_semantics(profile, temporary):
    """Only build TTIR for deliberately unsafe variants; never launch them."""
    from triton._C.libtriton import ir
    from triton.backends.cpu.compiler import CPUBackend
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver

    backend = CPUBackend(driver.active.get_current_target())
    base = textwrap.dedent(kernel().src)
    changes = {
        "output_plus_one": base.replace("],acc,", "],acc+1.0,"),
        "wrong_b_stride": base.replace("B+kk[:,None]*N+", "B+kk[:,None]*(N+1)+"),
        "missing_k_mask": base.replace("(rm[:,None]<M)&(kk[None,:]<K)", "(rm[:,None]<M)"),
        "nonzero_accumulator": base.replace(
            "tl.zeros((BM,BN),tl.float32)", "tl.full((BM,BN),1.0,tl.float32)"
        ),
        "extra_store": base.replace("    tl.store(C+", "    tl.store(C,0.0)\n    tl.store(C+"),
        "non_ieee_dot": base.replace("input_precision='ieee'", "input_precision='tf32'"),
    }

    def early(fn, mode):
        options = backend.parse_options(
            dict(
                num_cpu_threads=1,
                sme_plan_mode=mode,
                sme_plan_profile=json.dumps(profile, sort_keys=True) if mode == "ours" else "",
                enable_fast_math=False,
                enable_fp_fusion=True,
                assume_in_bounds=False,
                ukernels=None,
            )
        )
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)
        constants = dict(zip(("M", "N", "K", "BM", "BN", "BK"), (*SHAPE, 32, 64, 16)))
        source = ASTSource(fn, {"A": "*fp32", "B": "*fp32", "C": "*fp32"}, constexprs=constants)
        module = source.make_ir(
            backend.target,
            options,
            backend.get_codegen_implementation(options),
            backend.get_module_map(),
            context,
        )
        error = None
        try:
            backend.make_ttir(module, {}, options)
        except Exception as exception:
            error = exception
        return (
            error,
            module.get_int_attr("sme_plan.verified"),
            module.get_function(module.get_entry_func_name()).str_nodebug(),
        )

    for mode in ("off", "generic", "sme-fixed", "ours"):
        error, verified, _ = early(kernel(), mode)
        assert error is None, f"Nominal GEMM was rejected by {mode}: {error}"
        if mode != "off":
            assert verified == 1
    for name, source in changes.items():
        assert source != base, f"Semantic mutation did not change source: {name}"
        path = temporary / f"{name}.py"
        path.write_text("import triton\nimport triton.language as tl\n@triton.jit\n" + source)
        spec = importlib.util.spec_from_file_location(f"_runtime_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original = early(module.gemm, "off")
        generic = early(module.gemm, "generic")
        assert original[0] is None and generic[0] is None
        assert generic[1] == 0 and original[2] == generic[2], f"Generic changed {name} semantics"
        for mode in ("sme-fixed", "ours"):
            error, verified, _ = early(module.gemm, mode)
            assert error is not None and verified == 0, (
                f"{mode} accepted different semantics: {name}"
            )
    print(
        "PASS semantic proof: six altered kernels rejected by fixed/ours; generic preserves them",
        flush=True,
    )


def worker(args):
    assert digest(compiler_path()) == args.compiler_sha, "Compiler changed before worker"
    assert digest(args.profile) == args.profile_sha, "Profile changed before worker"
    profile = json.loads(args.profile.read_text())
    torch = prepare(args.worker, args.cache / f"t{args.worker}")
    sources = {
        check_case(torch, args.worker, profile, mode) for mode in ("generic", "sme-fixed", "ours")
    }
    for probe in (0, 4, 8):
        sources.add(check_case(torch, args.worker, profile, "calibration-probe", probe))
    assert len(sources) == 1, "Compiler modes did not use the same Triton source"
    if args.worker == 1:
        check_semantics(profile, args.cache)
    assert digest(compiler_path()) == args.compiler_sha, "Compiler changed during worker"
    assert digest(args.profile) == args.profile_sha, "Profile changed during worker"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "output/profile.json")
    parser.add_argument("--worker", type=int, choices=(1, 8), help=argparse.SUPPRESS)
    parser.add_argument("--cache", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--compiler-sha", help=argparse.SUPPRESS)
    parser.add_argument("--profile-sha", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.profile = args.profile.resolve()
    if args.worker:
        return worker(args)
    compiler_sha, profile_sha = digest(compiler_path()), digest(args.profile)
    with tempfile.TemporaryDirectory(prefix="runtime-checks-", dir=ROOT / ".build") as cache:
        for threads in (1, 8):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--profile",
                    str(args.profile),
                    "--worker",
                    str(threads),
                    "--cache",
                    cache,
                    "--compiler-sha",
                    compiler_sha,
                    "--profile-sha",
                    profile_sha,
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                check=True,
            )
    assert digest(compiler_path()) == compiler_sha, "Compiler changed during checks"
    assert digest(args.profile) == profile_sha, "Profile changed during checks"
    print("PASS runtime acceptance: independent t1/t8 workers; no benchmark samples", flush=True)


if __name__ == "__main__":
    main()
