"""One ordinary Triton GEMM; C++ chooses the schedule and prepares operands."""

from functools import lru_cache
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / ".build/triton"


def compiler_path():
    return STAGE / "python/triton/_C/libtriton.so"


def prepare(threads, cache=None):
    """Configure a fresh worker before importing either Torch or Triton."""
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("thread budget must be a positive integer")
    if "torch" in sys.modules or "triton" in sys.modules:
        raise RuntimeError("prepare must run once, before Torch/Triton imports")
    if not compiler_path().is_file():
        raise RuntimeError("Compiler is not built. Run ./run.sh or python scripts/build.py")
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "TRITON_CPU_MAX_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    os.environ.update(
        OMP_WAIT_POLICY="PASSIVE",
        OMP_DYNAMIC="FALSE",
        KMP_BLOCKTIME="0",
        OMP_STACKSIZE="16M",
        TRITON_CPU_UKERNELS_LIB="None",
        TRITON_BACKENDS_IN_TREE="1",
    )
    # Apple clang assembles the LLVM output and builds the small host launcher.
    # The compiler itself uses the separately pinned LLVM/MLIR toolchain.
    os.environ.setdefault("CC", "/usr/bin/clang" if sys.platform == "darwin" else "clang")
    os.environ.setdefault("CXX", "/usr/bin/clang++" if sys.platform == "darwin" else "clang++")
    if sys.platform == "darwin" and not os.environ.get("SDKROOT"):
        os.environ["SDKROOT"] = subprocess.check_output(
            ["xcrun", "--show-sdk-path"], text=True
        ).strip()
    for name in ("KMP_STACKSIZE", "TRITON_DISABLE_OPENMP", "TRITON_PLUGIN_PATHS"):
        os.environ.pop(name, None)
    for name in list(os.environ):
        if name.startswith(("TRITON_CPU_SME_", "TRITON_CPU_ARMV9_")):
            os.environ.pop(name, None)
    os.environ["TRITON_CACHE_DIR"] = str(cache or ROOT / ".build/jit-cache")
    os.environ["TRITON_LOCAL_LIBOMP_PATH"] = str(ROOT / ".build/openmp")
    sys.path.insert(0, str(STAGE / "python"))
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    import triton

    triton.runtime.driver.set_active_to_cpu()
    return torch


@lru_cache(None)
def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm(
        A,
        B,
        C,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pm = pid // tl.cdiv(N, BN)
        pn = pid % tl.cdiv(N, BN)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for start in range(0, K, BK):
            kk = start + rk
            av = tl.load(
                A + rm[:, None] * K + kk[None, :], (rm[:, None] < M) & (kk[None, :] < K), other=0.0
            )
            bv = tl.load(
                B + kk[:, None] * N + rn[None, :], (kk[:, None] < K) & (rn[None, :] < N), other=0.0
            )
            acc = tl.dot(av, bv, acc, input_precision="ieee")
        tl.store(C + rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))

    return gemm


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(None)
def compiler_sha256():
    return sha256(compiler_path())


@lru_cache(None)
def find_tool(name):
    candidates = [ROOT / ".build/llvm/bin" / name]
    if shutil.which("brew"):
        prefix = subprocess.check_output(["brew", "--prefix", "llvm"], text=True).strip()
        candidates.append(Path(prefix) / "bin" / name)
    found = shutil.which(name)
    if found:
        candidates.append(Path(found))
    for path in candidates:
        if path.is_file():
            return str(path)
    raise RuntimeError(f"{name} is required for backend checks (install LLVM)")


_AUDITS = {}


def _audit(compiled, selected_sme):
    """Inspect each unique JIT binary once; keep the verdict, not dump trees."""
    binary = next(
        Path(path) for name, path in compiled.metadata_group.items() if name.endswith(".so")
    )
    digest = sha256(binary)
    key = (digest, selected_sme)
    if key in _AUDITS:
        return dict(_AUDITS[key])
    assembly = subprocess.check_output(
        [find_tool("llvm-objdump"), "--disassemble", "--mattr=+sme", str(binary)], text=True
    )
    undefined = subprocess.check_output([shutil.which("nm") or "nm", "-u", str(binary)], text=True)
    llir = compiled.asm["llir"]
    entry = re.search(
        r"define[^\n]*@" + re.escape(compiled.metadata.name) + r"\([^\n]*\{\n(.*?)\n\}", llir, re.S
    )
    direct = bool(
        entry and re.search(r"\bcall[^\n@]*@__triton_cpu_sme_gemm_f32\w*\(", entry.group(1))
    )
    native = "__triton_cpu_sme_gemm_f32" in llir
    intrinsic = "llvm.aarch64.sme.mopa" in llir
    external = bool(re.search(r"(?i)(?:gemm|brgemm|cblas|kleidi|microkernel)", undefined))
    count = assembly.count("fmopa")
    verified = not external and (
        (native and intrinsic and direct and count > 0)
        if selected_sme
        else (not native and not intrinsic and count == 0)
    )
    if not verified:
        raise RuntimeError("Compiler plan and JIT compute path disagree; SME fallback is forbidden")
    result = dict(
        backend_verified=True,
        jit_binary_sha256=digest,
        fmopa_count=count,
        direct_entry_calls_native_sme=direct,
        external_compute=False,
        executed_path="compiler-native-sme" if selected_sme else "generic-cpu",
        fallback=False,
    )
    _AUDITS[key] = result
    return dict(result)


def compile_gemm(a, b, out, mode, threads, profile=None):
    """Compile once and return a synchronous complete-call callable.

    Every mode starts from the same operator, constants and grid. Compilation,
    first invocation and binary checks happen here, outside steady timing.
    """
    import torch

    rank_mode = mode.startswith("ours-rank-") or mode == "ours-worst"
    if mode not in ("generic", "sme-fixed", "ours", "calibration-probe") and not rank_mode:
        raise ValueError("unknown compiler mode")
    if any(
        x.device.type != "cpu" or x.dtype != torch.float32 or x.ndim != 2 or not x.is_contiguous()
        for x in (a, b, out)
    ):
        raise ValueError("supported input: dense row-major CPU FP32 matrices")
    m, k = a.shape
    kb, n = b.shape
    if kb != k or tuple(out.shape) != (m, n) or min(m, n, k) <= 0:
        raise ValueError("invalid GEMM shape")
    constants = (m, n, k, 32, 64, 16)
    grid = ((m + 31) // 32 * ((n + 63) // 64), 1, 1)
    content = json.dumps(profile, sort_keys=True, separators=(",", ":")) if profile else ""
    begin = time.perf_counter_ns()
    compiled = kernel().warmup(
        a,
        b,
        out,
        *constants,
        grid=grid,
        num_cpu_threads=threads,
        sme_plan_mode=mode,
        sme_plan_profile=content,
        enable_fast_math=False,
        enable_fp_fusion=True,
        assume_in_bounds=False,
        ukernels=None,
    )
    compiled._init_handles()
    compile_ns = time.perf_counter_ns() - begin
    runner = compiled[grid]
    before = compiled.run.get_global_b_stats()
    begin = time.perf_counter_ns()
    runner(a, b, out, *constants)
    first_ns = time.perf_counter_ns() - begin
    after = compiled.run.get_global_b_stats()
    metadata = compiled.metadata._asdict()
    selected_sme = metadata["sme_plan_selected_sme"] == 1
    if mode in ("sme-fixed", "calibration-probe") and not selected_sme:
        raise RuntimeError("An explicit SME mode must execute SME")
    global_pad = metadata.get("sme_plan_b_global_pad", 0)
    if after["copy_call_count"] - before["copy_call_count"] != int(global_pad > 0):
        raise RuntimeError("Selected global B preparation did not run exactly once")
    if global_pad and (
        after["last_copied_bytes"] != 4 * k * n
        or after["last_requested_bytes"] != 4 * k * (n + global_pad)
        or not 1 <= after["last_copy_team_size"] <= threads
    ):
        raise RuntimeError("Global B preparation violated shape or thread budget")
    evidence = dict(
        mode=mode,
        shape=[m, n, k],
        original_grid=grid,
        constants=constants,
        metadata={
            key: value
            for key, value in metadata.items()
            if key.startswith("sme_plan_") and key != "sme_plan_profile"
        },
        compile_ns=compile_ns,
        first_call_ns=first_ns,
        source_sha256=hashlib.sha256(kernel().src.encode()).hexdigest(),
        compiler_sha256=compiler_sha256(),
        profile_sha256=hashlib.sha256(content.encode()).hexdigest() if content else "",
        workspace_before_first_call=before,
        workspace_after_first_call=after,
        **_audit(compiled, selected_sme),
    )

    def call():
        runner(a, b, out, *constants)
        return out

    call.compiled = compiled
    call.evidence = evidence
    call.constants = constants
    call.grid = grid
    return call
