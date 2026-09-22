#!/usr/bin/env python3
"""Measure local calibration into three flat files; C++ owns GEMM schedules.

Run directly, or use --fit-only to reconstruct profile.json from the two CSVs.
Each calibration starts fresh calibration.csv/device.csv; the derived profile
is replaced only after all required processes, correctness and path checks pass.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sme_plan.calibration import (
    HOST_FIELDS,
    PROCESSES,
    SIZES,
    TEAMS,
    build_profile,
    canonical,
    read_csv,
)
from sme_plan.compiler_backend import compile_gemm, compiler_path, prepare
from common import check_gemm, measure_interleaved

NATIVE = ROOT / ".build/native"
CALIBRATION_FIELDS = (
    "process",
    "pid",
    "threads",
    "size",
    "m",
    "n",
    "k",
    "method",
    "status",
    "error",
    "round",
    "position",
    "repetitions",
    "elapsed_ns",
    "ns_per_call",
    "target_s",
    "calibration_ns",
    "repeat_reason",
    "target_window_met",
    "process_cpu_ns",
    "correct",
    "fp32_tolerance_ratio",
    "fp64_gamma_2k_ratio",
    "max_abs",
    "compiler_sha256",
    "source_sha256",
    "jit_binary_sha256",
    "backend_verified",
    "executed_path",
    "block_m",
    "block_n",
    "block_k",
    "k_strip",
    "a_pack_vector4",
    "b_panel_pad",
    "b_global_pad",
    "selected_threads",
    "selected_sme",
    "global_copy_verified",
    "compile_ns",
    "first_call_ns",
)
DEVICE_FIELDS = (
    "kind",
    "device_id",
    *HOST_FIELDS.values(),
    "vector_length_bits",
    "process",
    "pid",
    "threads",
    "actual_threads",
    "status",
    "error",
    "correct",
    "native_binary_sha256",
    "probe_binary_sha256",
    "compiler_sha256",
    "source_sha256",
    "jit_binary_sha256",
    "backend_verified",
    "case",
    "round",
    "position",
    "repetitions",
    "elapsed_ns",
    "bytes_per_invocation",
    "pilot_ns",
    "repeat_capped",
    "window_met_target",
    "wait_policy",
    "blocktime",
    "global_rows",
    "global_columns",
    "global_stride",
    "global_correct",
    "changed_global_correct",
    "workspace_initialization_ns",
    "conflict_columns",
    "conflict_pad_columns",
    "conflict_gbps",
    "padded_gbps",
    "conflict_severity",
    "is_conflict",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def message(error):
    return " ".join(str(error).replace(str(ROOT), ".").replace(str(Path.home()), "~").split())[
        :1200
    ]


def environment(threads):
    env = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        env[key] = str(threads)
    env.update(
        OMP_DYNAMIC="FALSE",
        OMP_STACKSIZE="16M",
        OMP_WAIT_POLICY="PASSIVE",
        KMP_BLOCKTIME="0",
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    env.pop("KMP_STACKSIZE", None)
    return env


class Table:
    def __init__(self, stream, fields):
        self.stream = stream
        self.writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        self.writer.writeheader()

    def append(self, rows):
        self.writer.writerows(rows)
        self.stream.flush()


def _probe():
    host = {}
    for key, field in HOST_FIELDS.items():
        result = subprocess.run(["/usr/sbin/sysctl", "-n", key], text=True, capture_output=True)
        if result.returncode and key in (
            "hw.model",
            "machdep.cpu.brand_string",
            "hw.memsize",
            "kern.osrelease",
        ):
            raise RuntimeError(f"cannot read required host fact {key}")
        host[field] = result.stdout.strip() if result.returncode == 0 else "unknown"
    result = subprocess.run([str(NATIVE / "sme-probe")], text=True, capture_output=True, timeout=20)
    if result.returncode:
        raise RuntimeError(
            f"SME execution probe failed ({result.returncode}): {message(result.stderr or result.stdout)}"
        )
    proof = json.loads(result.stdout)
    if (
        not proof.get("passed")
        or not proof.get("executed")
        or proof.get("vector_length_bytes") != 64
    ):
        raise RuntimeError(
            "the current compiler backend requires a correct 512-bit streaming SME probe"
        )
    identity = "native-device-" + hashlib.sha256(canonical(host)).hexdigest()[:20]
    return dict(
        kind="sme-probe",
        device_id=identity,
        **host,
        vector_length_bits=512,
        status="ok",
        correct=True,
        probe_binary_sha256=digest(NATIVE / "sme-probe"),
    )


def _native_process(facts, threads, process, rounds, target_ms, temporary):
    path = temporary / "native.json"
    command = [
        str(NATIVE / "compiler-device-calibration"),
        "--threads",
        str(threads),
        "--process-index",
        str(process),
        "--rounds",
        str(rounds),
        "--target-ms",
        str(target_ms),
        "--rows",
        "128",
        "--columns",
        "256",
        "--global-rows",
        "4096",
        "--global-columns",
        "4096",
        "--global-pad",
        "16",
        "--device-id",
        facts["device_id"],
        "--output",
        str(path),
    ]
    result = subprocess.run(command, text=True, capture_output=True, env=environment(threads))
    common = {
        key: value
        for key, value in facts.items()
        if key in ("device_id", *HOST_FIELDS.values(), "vector_length_bits", "probe_binary_sha256")
    }
    common.update(
        kind="native",
        threads=threads,
        process=process,
        native_binary_sha256=digest(NATIVE / "compiler-device-calibration"),
    )
    if result.returncode or not path.is_file():
        return [
            dict(
                **common,
                status="error",
                correct=False,
                error=message(result.stderr or result.stdout),
            )
        ], False
    raw = json.loads(path.read_text())
    path.unlink()
    env = raw["threadEnvironment"]
    good = (
        raw["correct"]
        and raw["actualThreads"] == threads
        and raw["requestedThreads"] == threads
        and raw["processIndex"] == process
        and raw["deviceIdentity"] == facts["device_id"]
        and raw["globalCopyCorrect"]
        and raw["changedGlobalInputCorrect"]
        and env["OMP_WAIT_POLICY"] == "PASSIVE"
        and env["KMP_BLOCKTIME"] == "0"
        and env["OMP_NUM_THREADS"] == str(threads)
        and env["OMP_THREAD_LIMIT"] == str(threads)
    )
    common.update(
        pid=raw["pid"],
        actual_threads=raw["actualThreads"],
        correct=raw["correct"],
        status="ok" if good else "error",
        error="" if good else "native correctness/team/wait-policy failed",
        wait_policy=env["OMP_WAIT_POLICY"],
        blocktime=env["KMP_BLOCKTIME"],
        global_rows=raw["globalCopyRows"],
        global_columns=raw["globalCopyColumns"],
        global_stride=raw["globalCopyStride"],
        global_correct=raw["globalCopyCorrect"],
        changed_global_correct=raw["changedGlobalInputCorrect"],
        workspace_initialization_ns=raw["globalWorkspaceInitializationNs"],
    )
    rows = [
        dict(
            **common,
            case=r["case"],
            round=r["round"],
            position=r["position"],
            repetitions=r["repetitions"],
            elapsed_ns=r["elapsedNs"],
            bytes_per_invocation=r["bytesPerInvocation"],
            pilot_ns=r["pilotNs"],
            repeat_capped=r["repeatCapped"],
            window_met_target=r["windowMetTarget"],
        )
        for r in raw["rawSamples"]
    ]
    for conflict in raw.get("rowStrideConflicts", []):
        row = dict(common)
        row.update(
            kind="row-stride-conflict",
            case=f"stride-{conflict['columns']}",
            conflict_columns=conflict["columns"],
            conflict_pad_columns=conflict["padColumns"],
            conflict_gbps=conflict["conflictGBps"],
            padded_gbps=conflict["paddedGBps"],
            conflict_severity=conflict["severity"],
            is_conflict=bool(conflict["conflict"]),
            status="ok" if good else "error",
            correct=bool(good and conflict.get("conflictGBps", 0) > 0),
        )
        rows.append(row)
    return rows, bool(good)


def _worker(job):
    rows, devices = [], []
    team, process = job["threads"], job["process"]
    current = dict(process=process, pid=os.getpid(), threads=team, size=0, method="bootstrap")
    try:
        torch = prepare(team, job["cache"])
        generator = torch.Generator().manual_seed(7301)
        a, b = (
            torch.randn(193, 129, generator=generator),
            torch.randn(129, 257, generator=generator),
        )
        fixed = compile_gemm(a, b, torch.empty(193, 257), "sme-fixed", team)
        for changed in (False, True):
            if changed:
                b.add_(0.25)
            if not check_gemm(fixed(), a @ b, a, b)["passed"]:
                raise RuntimeError("fixed SME bootstrap correctness failed")
        evidence = fixed.evidence
        if not evidence["backend_verified"] or evidence["executed_path"] != "compiler-native-sme":
            raise RuntimeError("fixed SME did not execute the verified compiler backend")
        facts = job["facts"]
        device = dict(
            identity=facts["device_id"],
            host={key: facts[field] for key, field in HOST_FIELDS.items()},
            vectorLengthBits=512,
            nativeSmeGemmVerified=True,
            supportsVectorAPack=True,
            supportsStreamingK=True,
            supportsPaddedB=True,
            supportsGlobalPaddedB=True,
            supportsGenericCPU=True,
        )
        bootstrap = dict(
            compilerSha256=evidence["compiler_sha256"],
            device=device,
            calibration=dict(
                deviceIdentity=device["identity"],
                evidenceId="fixed-sme-" + evidence["jit_binary_sha256"],
                teams=[],
            ),
        )
        devices.append(
            dict(
                kind="fixed-sme",
                device_id=device["identity"],
                process=process,
                pid=os.getpid(),
                threads=team,
                actual_threads=int(evidence["metadata"]["sme_plan_threads"]),
                correct=True,
                status="ok",
                backend_verified=True,
                compiler_sha256=evidence["compiler_sha256"],
                source_sha256=evidence["source_sha256"],
                jit_binary_sha256=evidence["jit_binary_sha256"],
            )
        )
        del a, b, fixed
        for size in SIZES:
            generator = torch.Generator().manual_seed(91471 + size)
            a, b = (
                torch.randn(size, size, generator=generator),
                torch.randn(size, size, generator=generator),
            )
            reference, calls, details = a @ b, {}, {}
            for method in ["generic", *(f"probe{i}" for i in range(10))]:
                current = dict(
                    process=process,
                    pid=os.getpid(),
                    threads=team,
                    size=size,
                    m=size,
                    n=size,
                    k=size,
                    method=method,
                )
                probe = (
                    None
                    if method == "generic"
                    else dict(bootstrap, calibrationProbeIndex=int(method[5:]))
                )
                try:
                    call = compile_gemm(
                        a,
                        b,
                        torch.empty_like(reference),
                        "generic" if method == "generic" else "calibration-probe",
                        team,
                        probe,
                    )
                except Exception as error:
                    if method == "probe5":
                        rows.append(dict(**current, status="unsupported", error=message(error)))
                        continue
                    raise
                if method == "probe5":
                    raise RuntimeError(
                        "probe5 unexpectedly passed the fixed C++ scratch resource gate"
                    )
                check = check_gemm(call(), reference, a, b)
                ev, md = call.evidence, call.evidence["metadata"]
                if (
                    not check["passed"]
                    or not ev["backend_verified"]
                    or int(md["sme_plan_threads"]) != team
                    or ev["executed_path"]
                    != ("generic-cpu" if method == "generic" else "compiler-native-sme")
                ):
                    raise RuntimeError("calibration candidate correctness/path/team check failed")
                global_pad = int(md["sme_plan_b_global_pad"])
                before, after = (
                    ev.get("workspace_before_first_call"),
                    ev.get("workspace_after_first_call"),
                )
                copied = (
                    isinstance(before, dict)
                    and isinstance(after, dict)
                    and after["copy_call_count"] - before["copy_call_count"] == 1
                    and after["last_requested_bytes"] == 4 * size * (size + global_pad)
                    and after["last_copied_bytes"] == 4 * size * size
                    and after["last_copy_team_size"] == team
                )
                if global_pad and not copied:
                    raise RuntimeError(
                        "global B copy was not executed exactly once by the selected team"
                    )
                details[method] = dict(
                    **current,
                    status="ok",
                    error="",
                    correct=True,
                    fp32_tolerance_ratio=check["fp32_tolerance_ratio"],
                    fp64_gamma_2k_ratio=check["fp64_gamma_2k_ratio"],
                    max_abs=check["max_abs"],
                    compiler_sha256=ev["compiler_sha256"],
                    source_sha256=ev["source_sha256"],
                    jit_binary_sha256=ev["jit_binary_sha256"],
                    backend_verified=ev["backend_verified"],
                    executed_path=ev["executed_path"],
                    block_m=int(md["sme_plan_block_m"]),
                    block_n=int(md["sme_plan_block_n"]),
                    block_k=int(md["sme_plan_block_k"]),
                    k_strip=int(md["sme_plan_k_strip"]),
                    a_pack_vector4=int(md["sme_plan_a_pack_vector4"]),
                    b_panel_pad=int(md["sme_plan_b_panel_pad"]),
                    b_global_pad=global_pad,
                    selected_threads=int(md["sme_plan_threads"]),
                    selected_sme=int(md["sme_plan_selected_sme"]),
                    global_copy_verified=bool(copied),
                    compile_ns=ev["compile_ns"],
                    first_call_ns=ev["first_call_ns"],
                )
                calls[method] = call

            def sample(window, details=details):
                timing = {
                    key: window[key]
                    for key in (
                        "round",
                        "position",
                        "repetitions",
                        "elapsed_ns",
                        "ns_per_call",
                        "target_s",
                        "calibration_ns",
                        "repeat_reason",
                        "target_window_met",
                        "process_cpu_ns",
                    )
                }
                rows.append(dict(**details[window["method"]], **timing))

            measure_interleaved(
                calls,
                rounds=job["rounds"],
                target_s=job["window"],
                seed=173 + size + process * 731 + team,
                phase="calibration",
                on_sample=sample,
            )
            del calls, details, a, b, reference, call
            gc.collect()
        return dict(passed=True, calibration=rows, device=devices)
    except Exception as error:
        rows.append(dict(**current, status="error", error=message(error)))
        return dict(passed=False, calibration=rows, device=devices, error=message(error))


def _save_profile(output):
    profile = build_profile(read_csv(output / "calibration.csv"), read_csv(output / "device.csv"))
    if profile["compilerSha256"] != digest(compiler_path()):
        raise RuntimeError("raw calibration belongs to a different compiler; it cannot be rebound")
    temporary = output / ".profile.pending"
    temporary.write_text(json.dumps(profile, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output / "profile.json")
    return output / "profile.json"


def run(
    output=WORKSPACE / "output", *, rounds=3, window=0.015, native_window_ms=2.0, fit_only=False
):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if fit_only:
        return _save_profile(output)
    if rounds < 3 or not all(math.isfinite(v) and v > 0 for v in (window, native_window_ms)):
        raise ValueError(
            "calibration requires at least three windows and positive target durations"
        )
    for name in ("sme-probe", "compiler-device-calibration"):
        if not (NATIVE / name).is_file():
            raise FileNotFoundError(f"build .build/native/{name} first")
    with (
        (output / "device.csv").open("w", newline="") as ds,
        (output / "calibration.csv").open("w", newline="") as cs,
    ):
        devices, samples = Table(ds, DEVICE_FIELDS), Table(cs, CALIBRATION_FIELDS)
        try:
            facts = _probe()
        except Exception as error:
            devices.append(
                [dict(kind="sme-probe", status="error", correct=False, error=message(error))]
            )
            raise
        devices.append([facts])
        with tempfile.TemporaryDirectory(prefix="sme-native-calibration-") as temporary:
            for process in range(PROCESSES):
                order = TEAMS[process % len(TEAMS) :] + TEAMS[: process % len(TEAMS)]
                for team in order:
                    rows, passed = _native_process(
                        facts, team, process, rounds, native_window_ms, Path(temporary)
                    )
                    devices.append(rows)
                    if not passed:
                        raise RuntimeError(
                            f"native calibration failed at team {team}, process {process}; see device.csv"
                        )
        with tempfile.TemporaryDirectory(prefix="calibration-cache-", dir=ROOT / ".build") as cache:
            for process in range(PROCESSES):
                order = TEAMS[process % len(TEAMS) :] + TEAMS[: process % len(TEAMS)]
                for team in order:
                    job = dict(
                        facts=facts,
                        threads=team,
                        process=process,
                        rounds=rounds,
                        window=window,
                        cache=cache,
                    )
                    child = subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()), "--worker"],
                        input=canonical(job).decode(),
                        text=True,
                        capture_output=True,
                        env=environment(team),
                    )
                    try:
                        result = json.loads(child.stdout)
                    except ValueError as error:
                        samples.append(
                            [
                                dict(
                                    process=process,
                                    threads=team,
                                    status="error",
                                    error=message(child.stderr or error),
                                )
                            ]
                        )
                        raise RuntimeError(
                            "calibration worker returned no valid sample payload"
                        ) from error
                    samples.append(result["calibration"])
                    devices.append(result["device"])
                    if child.returncode or not result["passed"]:
                        raise RuntimeError(
                            f"GEMM calibration failed at team {team}, process {process}: {result.get('error', '')}\n{child.stderr[-2000:]}"
                        )
                    print(
                        f"Calibration: team {team}, process {process + 1}/{PROCESSES} complete",
                        flush=True,
                    )
    return _save_profile(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "output")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--window", type=float, default=0.015)
    parser.add_argument("--native-window-ms", type=float, default=2.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        with contextlib.redirect_stdout(sys.stderr):
            result = _worker(json.load(sys.stdin))
        print(canonical(result).decode(), flush=True)
        return 0 if result["passed"] else 1
    profile = run(
        args.output,
        rounds=args.rounds,
        window=args.window,
        native_window_ms=args.native_window_ms,
        fit_only=args.fit_only,
    )
    print(f"Compiler profile: {profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
