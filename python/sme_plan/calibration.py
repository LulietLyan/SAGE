"""Device response fitting from flat raw samples; C++ selects every schedule.

Rates describe preparation-subtracted complete calls, not physical bandwidth or
arithmetic peaks. No shape-to-winner table is produced. The compiler requires an
exact macro-tile/packing/layout/full-K-or-streaming response family.
"""

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

TEAMS = (1, 2, 4, 8)
# Include the reproducible 3072 full-K working-set boundary as an explicit
# response knot.  Smooth panel-byte interpolation otherwise hides this
# discontinuity and can rank a full-K family above a measured K-strip family.
SIZES = (704, 1664, 3072, 3456, 4096)
PROCESSES = 3
HOST_FIELDS = {
    "hw.model": "host_model",
    "machdep.cpu.brand_string": "host_cpu",
    "hw.memsize": "host_memory_bytes",
    "kern.osrelease": "host_osrelease",
    "hw.logicalcpu": "host_logical_cpus",
    "hw.physicalcpu": "host_physical_cpus",
    "hw.optional.arm.FEAT_SME": "host_reported_sme",
    "hw.optional.arm.FEAT_SME2": "host_reported_sme2",
}


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def truth(value):
    if value in (True, 1, "True", "true", "1"):
        return True
    if value in (False, 0, "False", "false", "0"):
        return False
    raise ValueError(f"invalid boolean sample value: {value!r}")


def groups(rows, key):
    result = defaultdict(list)
    for row in rows:
        result[key(row)].append(row)
    return result


def features(shape, metadata, rates):
    """Fit features from compiled schedules without selecting candidates.

    A/B traffic counts repeated instruction lanes, including cached loads.
    reuse_m=4 changes visit order; it does not divide the B load count by four.
    Global B preparation is one shared copy including its own submit/wait.
    Preparation subtraction does not apply the C++ conflictScale heuristic;
    these residuals are not an exact inverse of the production estimator.
    """
    m, n, k = shape
    bm, bn, bk = (int(metadata[f"sme_plan_block_{axis}"]) for axis in "mnk")
    team = int(metadata["sme_plan_threads"])
    strip = int(metadata["sme_plan_k_strip"])
    vector = truth(metadata["sme_plan_a_pack_vector4"])
    local_pad, global_pad = (
        int(metadata[f"sme_plan_b_{kind}_pad"]) for kind in ("panel", "global")
    )
    sme = truth(metadata["sme_plan_selected_sme"])
    if min(m, n, k, bm, bn, bk, team) <= 0 or strip < 0 or (strip and strip % bk):
        raise ValueError("invalid compiled shape or K-strip metadata")
    if local_pad not in (0, 16) or global_pad not in (0, 16) or (local_pad and global_pad):
        raise ValueError(
            "the current lowering requires separate original/local/global B layouts at SVL=512"
        )
    if (local_pad or global_pad) and not sme:
        raise ValueError("B preparation is not an observed generic CPU strategy")

    def ceil(a, b):
        return (a + b - 1) // b

    tasks = ceil(m, bm) * ceil(n, bn)
    waves = ceil(tasks, team)
    occupancy = tasks / (waves * team)
    capacity = strip or ceil(k, bk) * bk
    strips = ceil(k, capacity)
    a_panel = 4 * bm * capacity if sme else 0
    local_panel = 4 * (bn + local_pad) * capacity if sme and local_pad else 0
    arithmetic = 2 * ceil(m, 32) * 32 * ceil(n, 32) * 32 * k if sme else 2 * m * n * k
    reads = (
        (
            4 * ceil(m, 32) * 32 * k * ceil(n, 32)
            + 4 * n * k * ceil(m, 32)
            + 4 * m * n * (2 * strips - 1)
        )
        if sme
        else 4 * (m * k + k * n + m * n)
    )
    a_pack = (
        8
        * m
        * k
        * ceil(n, bn)
        / rates["vectorAPackGBps" if vector else "scalarAPackGBps"]
        / occupancy
        if sme
        else 0
    )
    local_pack = 8 * ceil(m, bm) * k * n / rates["copyGBps"] / occupancy if sme and local_pad else 0
    global_pack = 8 * k * n / rates["globalBPackGBps"] if sme and global_pad else 0
    # taskOverheadNs is the marginal communication/dispatch cost per submitted
    # tile, measured from the empty-team versus empty-task experiment.  Charge
    # every output task; multiplying only by waves would undercount small-tile
    # communication by approximately the team size.
    # Keep calibration subtraction algebraically aligned with C++ estimation:
    # task dispatch is per submitted tile, while cross-worker communication is
    # per output wave.  Omitting the latter would fold barrier latency into the
    # fitted operand bandwidth and make the model shape dependent.
    preparation = (
        a_pack
        + local_pack
        + global_pack
        + rates["callOverheadNs"]
        + tasks * rates["taskOverheadNs"]
        + waves * rates.get("communicationNsPerWave", 0.0)
    )
    return dict(
        blockM=bm,
        blockN=bn,
        blockK=bk,
        vectorAPack=vector,
        bLayout="global" if global_pad else "local" if local_pad else "original",
        stripK=strip,
        selectedSme=sme,
        panelBytes=a_panel + local_panel,
        arithmeticFlops=arithmetic,
        referenceReadBytes=reads,
        occupancy=occupancy,
        knownPreparationNs=preparation,
    )


def _points(rows):
    points = []
    for panel, members in sorted(groups(rows, lambda r: r["panelBytes"]).items()):
        per_process = [
            median(r["response"] for r in items)
            for items in groups(members, lambda r: r["pid"]).values()
        ]
        points.append(dict(panelBytes=panel, operandBandwidthGBps=median(per_process)))
    return points


def _strategy(row):
    return (
        row["blockM"],
        row["blockN"],
        row["blockK"],
        row["vectorAPack"],
        row["bLayout"],
        bool(row["stripK"]),
    )


def strategy_responses(rows):
    if len({r["threads"] for r in rows}) > 1:
        raise ValueError("strategy responses must not mix thread teams")
    result = []
    for key, members in sorted(groups([r for r in rows if r["selectedSme"]], _strategy).items()):
        eligible = [r for r in members if r["eligibleForFit"]]
        if not eligible:
            continue  # Raw observations/coverage remain; no fabricated rate.
        bm, bn, bk, vector, layout, streaming = key
        result.append(
            dict(
                blockM=bm,
                blockN=bn,
                blockK=bk,
                aPackVector4=vector,
                bLayout=layout,
                streamingK=streaming,
                independentProcesses=len({r["pid"] for r in eligible}),
                panelResponses=_points(eligible),
            )
        )
    return result


def fit_team(rows, rates):
    if len({r["threads"] for r in rows}) != 1:
        raise ValueError("response fitting must not pool thread teams")
    sme = [r for r in rows if r["selectedSme"] and r["eligibleForFit"]]
    cpu = [r for r in rows if not r["selectedSme"] and r["eligibleForFit"]]
    if not sme or not cpu:
        raise ValueError("no identifiable SME or CPU response for a measured team")
    ceiling = max(
        median(r["compute"] for r in items)
        for items in groups(sme, lambda r: (r["size"], r["method"])).values()
    )
    cpu_rate = median(
        median(r["compute"] for r in items) for items in groups(cpu, lambda r: r["pid"]).values()
    )
    # Required parser fields; pooled rates are only a generic-CPU proxy or
    # diagnostics. Native C++ estimation uses strategyResponses exclusively.
    pooled = {}
    for layout, field in (
        ("original", "operandBandwidthGBps"),
        ("local", "paddedOperandBandwidthGBps"),
        ("global", "globalPaddedOperandBandwidthGBps"),
    ):
        points = _points([r for r in sme if r["bLayout"] == layout])
        pooled[field] = median(p["operandBandwidthGBps"] for p in points) if points else 0.0
    coverage = {}
    for native, field in ((True, "observedNativeTiles"), (False, "observedGenericTiles")):
        coverage[field] = [
            dict(
                blockM=bm, blockN=bn, blockK=bk, independentProcesses=len({r["pid"] for r in items})
            )
            for (bm, bn, bk), items in sorted(
                groups(
                    [r for r in rows if r["selectedSme"] == native],
                    lambda r: (r["blockM"], r["blockN"], r["blockK"]),
                ).items()
            )
        ]
    return dict(
        **rates,
        **pooled,
        **coverage,
        requestedThreads=rows[0]["threads"],
        actualThreads=rows[0]["threads"],
        independentRounds=len({r["pid"] for r in rows}),
        correct=True,
        pathVerified=True,
        smeGflops=ceiling,
        genericGflops=cpu_rate,
        # Communication is calibrated separately from GEMM responses;
        # _device_rates supplies it when barrier samples exist and
        # zero for legacy/raw profiles.
        stripOverheadNs=0.0,
        strategyResponses=strategy_responses(rows),
    )


def _device_rates(rows):
    rates = {}
    native = [row for row in rows if row["kind"] == "native"]
    # Barrier is an additive calibration signal.  Keep it optional so raw
    # tables produced before the communication microbenchmark remain valid.
    required_cases = {
        "scalarAPack",
        "vectorAPack",
        "copy",
        "globalBPack",
        "emptyTeam",
        "emptyTasks",
    }
    for team in TEAMS:
        members = [r for r in native if int(r["threads"]) == team]
        per_process = []
        identities = set()
        by_process = groups(members, lambda r: int(r["process"]))
        if set(by_process) != set(range(PROCESSES)):
            raise ValueError(f"team {team} needs three native calibration processes")
        for items in by_process.values():
            pids = {int(r["pid"]) for r in items}
            if len(pids) != 1 or identities.intersection(pids):
                raise ValueError("native process identities are inconsistent")
            identities.update(pids)
            for r in items:
                if (
                    r["status"] != "ok"
                    or int(r["actual_threads"]) != team
                    or not truth(r["correct"])
                    or r["wait_policy"] != "PASSIVE"
                    or str(r["blocktime"]) != "0"
                    or not truth(r["global_correct"])
                    or not truth(r["changed_global_correct"])
                    or (int(r["global_rows"]), int(r["global_columns"]), int(r["global_stride"]))
                    != (4096, 4096, 4112)
                ):
                    raise ValueError(
                        "native calibration failed or used a different runtime/copy contract"
                    )
            cases = groups(items, lambda r: r["case"])
            if not required_cases.issubset(cases):
                raise ValueError("native calibration cases are incomplete")
            times = {}
            for case in required_cases:
                samples = cases[case]
                if len(samples) < 3 or {int(r["round"]) for r in samples} != set(
                    range(len(samples))
                ):
                    raise ValueError("native timing windows are incomplete")
                times[case] = median(_duration(r) for r in samples)
            item = {}
            for case, field in (
                ("scalarAPack", "scalarAPackGBps"),
                ("vectorAPack", "vectorAPackGBps"),
                ("copy", "copyGBps"),
                ("globalBPack", "globalBPackGBps"),
            ):
                expected = 8 * 4096 * 4096 if case == "globalBPack" else 8 * 128 * 256 * team
                if {int(r["bytes_per_invocation"]) for r in cases[case]} != {expected}:
                    raise ValueError("native byte numerator differs from its declared workload")
                item[field] = expected / times[case]
            item["callOverheadNs"] = times["emptyTeam"]
            item["taskOverheadNs"] = max(0.0, (times["emptyTasks"] - times["emptyTeam"]) / 63.0)
            # Newer native calibrators include one explicit OpenMP barrier
            # per invocation.  Its launch-subtracted median is the
            # cross-worker communication latency paid per output wave.  Keep
            # this optional so pre-barrier raw tables and profiles remain
            # loadable with a conservative zero.
            if "barrier" in cases:
                barrier_samples = cases["barrier"]
                if len(barrier_samples) < 3 or {int(r["round"]) for r in barrier_samples} != set(
                    range(len(barrier_samples))
                ):
                    raise ValueError("native barrier timing windows are incomplete")
                barrier = median(_duration(r) for r in barrier_samples)
                item["communicationNsPerWave"] = max(0.0, barrier - times["emptyTeam"])
            else:
                item["communicationNsPerWave"] = 0.0
            per_process.append(item)
        rates[team] = {key: median(row[key] for row in per_process) for key in per_process[0]}
    return rates


def _duration(row):
    elapsed, repetitions = int(row["elapsed_ns"]), int(row["repetitions"])
    if elapsed <= 0 or repetitions <= 0:
        raise ValueError("timing samples need positive elapsed time and repetitions")
    duration = elapsed / repetitions
    if row.get("ns_per_call") not in (None, "") and not math.isclose(
        float(row["ns_per_call"]), duration, rel_tol=1e-12
    ):
        raise ValueError("derived per-call latency differs from raw elapsed/repetitions")
    return duration


def _row_stride_conflicts(device_rows):
    """Aggregate measured leading-dimension cliffs across native processes."""
    samples = [r for r in device_rows if r.get("kind") == "row-stride-conflict"]
    if not samples:
        return []
    by_columns = groups(samples, lambda r: int(r["conflict_columns"]))
    conflicts = []
    for columns, rows in sorted(by_columns.items()):
        if any(r["status"] != "ok" or not truth(r["correct"]) for r in rows):
            raise ValueError(f"row-stride conflict samples failed for columns={columns}")
        pads = {int(r["conflict_pad_columns"]) for r in rows}
        if len(pads) != 1 or next(iter(pads)) < 1:
            raise ValueError("row-stride conflict pad contract is inconsistent")
        severity = median(float(r["conflict_severity"]) for r in rows)
        if not math.isfinite(severity) or severity <= 0:
            raise ValueError("row-stride conflict severity is invalid")
        # Microbench noise can push conflict/padded slightly above 1.0.
        severity = min(severity, 1.0)
        votes = sum(1 for r in rows if truth(r["is_conflict"]))
        # Keep widths the native probe marked as cliffs on a majority of runs.
        if votes * 2 < len(rows) or severity >= 0.95:
            continue
        conflicts.append(
            dict(columns=columns, padColumns=next(iter(pads)), severity=min(severity, 0.9))
        )
    return conflicts


def build_profile(calibration_rows, device_rows):
    """Rebuild the sole compiler profile from calibration.csv and device.csv."""
    if not calibration_rows or not device_rows:
        raise ValueError("both raw calibration tables are required")
    probes = [r for r in device_rows if r["kind"] == "sme-probe"]
    if (
        len(probes) != 1
        or not truth(probes[0]["correct"])
        or int(probes[0]["vector_length_bits"]) != 512
    ):
        raise ValueError("this backend requires an executed, correct 512-bit SME probe")
    host = {key: str(probes[0][field]) for key, field in HOST_FIELDS.items()}
    identity = probes[0]["device_id"]
    if not identity or any(r["device_id"] != identity for r in device_rows):
        raise ValueError("device sample identities differ")
    rates = _device_rates(device_rows)
    conflicts = _row_stride_conflicts(device_rows)
    fixed = [r for r in device_rows if r["kind"] == "fixed-sme"]
    if {(int(r["threads"]), int(r["process"])) for r in fixed} != {
        (t, p) for t in TEAMS for p in range(PROCESSES)
    } or len(fixed) != 12:
        raise ValueError("fresh fixed-SME execution checks are incomplete")
    if any(
        r["status"] != "ok"
        or not truth(r["correct"])
        or not truth(r["backend_verified"])
        or int(r["actual_threads"]) != int(r["threads"])
        for r in fixed
    ):
        raise ValueError("fixed-SME compiler execution was not verified")
    compiler_hashes = {r["compiler_sha256"] for r in fixed}
    source_hashes = {r["source_sha256"] for r in fixed}
    expected_methods = {"generic"} | {f"probe{i}" for i in range(10) if i != 5}
    by_condition = groups(
        calibration_rows,
        lambda r: (int(r["threads"]), int(r["process"]), int(r["size"]), r["method"]),
    )
    expected = {
        (t, p, s, method)
        for t in TEAMS
        for p in range(PROCESSES)
        for s in SIZES
        for method in expected_methods | {"probe5"}
    }
    if set(by_condition) != expected:
        raise ValueError("same-operator calibration coverage is incomplete or unexpected")
    observations = []
    pids_by_process = defaultdict(set)
    for (team, process, size, method), samples in sorted(by_condition.items()):
        if method == "probe5":
            if (
                len(samples) != 1
                or samples[0]["status"] != "unsupported"
                or not samples[0]["error"]
            ):
                raise ValueError("resource-rejected probe5 must remain an explicit raw row")
            continue
        if len(samples) < 3 or {int(r["round"]) for r in samples} != set(range(len(samples))):
            raise ValueError("GEMM timing windows are incomplete")
        fields = (
            "block_m",
            "block_n",
            "block_k",
            "k_strip",
            "a_pack_vector4",
            "b_panel_pad",
            "b_global_pad",
            "selected_threads",
            "selected_sme",
            "jit_binary_sha256",
        )
        if len({tuple(str(row[field]) for field in fields) for row in samples}) != 1:
            raise ValueError("one measured candidate changed strategy or binary between windows")
        for r in samples:
            if (
                r["status"] != "ok"
                or not truth(r["correct"])
                or not truth(r["backend_verified"])
                or int(r["selected_threads"]) != team
                or tuple(int(r[axis]) for axis in ("m", "n", "k")) != (size, size, size)
                or truth(r["selected_sme"]) != (method != "generic")
                or (int(r["b_global_pad"]) and not truth(r["global_copy_verified"]))
                or r["executed_path"]
                != ("generic-cpu" if method == "generic" else "compiler-native-sme")
            ):
                raise ValueError("GEMM correctness, path or actual selected team differs")
            compiler_hashes.add(r["compiler_sha256"])
            source_hashes.add(r["source_sha256"])
            pids_by_process[team, process].add(int(r["pid"]))
        row = samples[0]
        metadata = {f"sme_plan_block_{axis}": int(row[f"block_{axis}"]) for axis in "mnk"}
        metadata.update(
            sme_plan_threads=team,
            sme_plan_k_strip=int(row["k_strip"]),
            sme_plan_a_pack_vector4=truth(row["a_pack_vector4"]),
            sme_plan_b_panel_pad=int(row["b_panel_pad"]),
            sme_plan_b_global_pad=int(row["b_global_pad"]),
            sme_plan_selected_sme=truth(row["selected_sme"]),
        )
        measured = median(_duration(r) for r in samples)
        f = features((size, size, size), metadata, rates[team])
        residual = measured - f["knownPreparationNs"]
        eligible = residual > max(1.0, 0.01 * measured)
        observations.append(
            dict(
                **f,
                threads=team,
                process=process,
                pid=int(row["pid"]),
                size=size,
                method=method,
                eligibleForFit=eligible,
                compute=f["arithmeticFlops"] / f["occupancy"] / residual if eligible else None,
                response=f["referenceReadBytes"] / f["occupancy"] / residual if eligible else None,
            )
        )
    if len(compiler_hashes) != 1 or len(source_hashes) != 1:
        raise ValueError("calibration must use one compiler and one Triton operator source")
    for team in TEAMS:
        identities = [pids_by_process[team, p] for p in range(PROCESSES)]
        if any(len(pids) != 1 for pids in identities) or len(set.union(*identities)) != PROCESSES:
            raise ValueError("GEMM calibration needs three independent processes per actual team")
    teams = [
        fit_team([r for r in observations if r["threads"] == team], rates[team]) for team in TEAMS
    ]

    # CSV text normalization makes an in-memory fit and --fit-only identical.
    def table(rows):
        fields = sorted({key for r in rows for key, value in r.items() if value not in (None, "")})
        return [
            {key: "" if r.get(key) is None else str(r.get(key, "")) for key in fields} for r in rows
        ]

    evidence = hashlib.sha256(canonical([table(calibration_rows), table(device_rows)])).hexdigest()
    device = dict(
        identity=identity,
        host=host,
        vectorLengthBits=512,
        nativeSmeGemmVerified=True,
        supportsVectorAPack=True,
        supportsStreamingK=True,
        supportsPaddedB=True,
        supportsGlobalPaddedB=True,
        supportsGenericCPU=True,
        rowStrideConflicts=conflicts,
    )
    return dict(
        compilerSha256=next(iter(compiler_hashes)),
        sourceSha256=next(iter(source_hashes)),
        device=device,
        calibration=dict(
            deviceIdentity=identity,
            evidenceId="device-response-" + evidence,
            minimumIndependentRounds=3,
            teams=teams,
        ),
        validity=dict(
            calibrationShapes=list(SIZES),
            calibrationProcessesPerTeam=3,
            fitRejectedObservations=sum(not r["eligibleForFit"] for r in observations),
            statisticalUnit="independent process median; windows nested within process",
            physicalBandwidthIdentified=False,
            computeOnlyThroughputIdentified=False,
            rowStrideConflicts=conflicts,
            scope="complete-call residual model; measured preparation transfers are estimates; taskOverheadNs is charged per output task; communicationNsPerWave is an effective wave contention penalty derived from an explicit OpenMP barrier microbenchmark and multiplied by modeled occupancy waves (the launcher barrier structure is not separately instrumented); strip overhead remains in residual; measured row-stride cliffs bias B/A leading-dimension costs",
            domain="exact strategy families; streaming panel interpolation within observed interval; full-K endpoint clamps",
            selection="C++ only; no Python schedule ranking or shape-to-winner table",
        ),
    )
