"""Complete-call timing and FP32/FP64 checks shared by calibration and GEMM."""

import math
import random
import statistics
import time


def measure_interleaved(
    calls, *, rounds, target_s, seed, min_reps=1, max_reps=10000, phase="holdout", on_sample=None
):
    """A window is one sample. Adaptive repeats are fixed before recorded rounds.

    Calls are synchronous and each does all required preparation and launch.
    No per-call clock cost is inserted into the timed loop. Raw elapsed nanoseconds
    and repeat counts are retained, not silently expanded into pseudo-samples.
    """
    if not calls or not all(callable(fn) for fn in calls.values()):
        raise ValueError("at least one named callable is required")
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if not isinstance(target_s, (int, float)) or not math.isfinite(target_s) or target_s <= 0:
        raise ValueError("target_s must be finite and positive")
    if type(min_reps) is not int or type(max_reps) is not int or not 1 <= min_reps <= max_reps:
        raise ValueError("repetition bounds must be positive ordered integers")
    rng = random.Random(seed)
    names = list(calls)
    pilots = {}
    repetitions = {}
    repeat_details = {}
    for name in rng.sample(names, len(names)):
        calls[name]()
        t = time.perf_counter_ns()
        calls[name]()
        elapsed = time.perf_counter_ns() - t
        pilots[name] = elapsed
        requested = math.ceil(target_s * 1e9 / max(1, elapsed))
        repetitions[name] = min(max_reps, max(min_reps, requested))
        repeat_details[name] = dict(
            repetitions_before_clip=requested,
            repeat_reason=(
                "maximum repetition cap"
                if requested > max_reps
                else "minimum repetition floor"
                if requested < min_reps
                else "ceil(target/calibration); no clipping"
            ),
        )
    raw = []
    base = rng.sample(names, len(names))
    for rnd in range(rounds):
        # Balanced cyclic order within each independent process, random starting permutation.
        order = base[rnd % len(base) :] + base[: rnd % len(base)]
        if rnd // len(base) % 2:
            order = list(reversed(order))
        for position, name in enumerate(order):
            reps = repetitions[name]
            cpu_before = time.process_time_ns()
            t = time.perf_counter_ns()
            for _ in range(reps):
                calls[name]()
            elapsed = time.perf_counter_ns() - t
            cpu_elapsed = time.process_time_ns() - cpu_before
            row = dict(
                phase=phase,
                round=rnd,
                position=position,
                method=name,
                repetitions=reps,
                elapsed_ns=elapsed,
                ns_per_call=elapsed / reps,
                target_s=target_s,
                calibration_ns=pilots[name],
                **repeat_details[name],
                process_cpu_ns=cpu_elapsed,
                process_cpu_over_wall=cpu_elapsed / elapsed,
                actual_window_over_target=elapsed / (target_s * 1e9),
                target_window_met=elapsed >= target_s * 1e9,
            )
            raw.append(row)
            if on_sample:
                on_sample(row)
    summary = {}
    for name in names:
        windows = [x for x in raw if x["method"] == name]
        values = [x["ns_per_call"] for x in windows]
        median = statistics.median(values)
        summary[name] = dict(
            median_ns=median,
            min_ns=min(values),
            max_ns=max(values),
            stdev_ns=statistics.stdev(values) if len(values) > 1 else None,
            mad_ns=statistics.median(abs(x - median) for x in values),
            windows=rounds,
            repetitions=repetitions[name],
            windows_meeting_target=sum(x["target_window_met"] for x in windows),
        )
    return dict(
        raw=raw,
        summary=summary,
        phase=phase,
        seed=seed,
        statistical_unit="timing window within an independent process; repetitions are not independent samples",
        cpu_time_scope="whole process around the wall window; average utilization, not a hard concurrency bound",
    )


def check_gemm(actual, reference, a, b):
    import torch

    if tuple(actual.shape) != tuple(reference.shape):
        return {"passed": False, "reason": "shape"}
    finite = bool(torch.isfinite(actual).all())
    reference_finite = bool(torch.isfinite(reference).all())
    if not finite or not reference_finite:
        return dict(
            passed=False,
            finite=finite,
            reference_finite=reference_finite,
            reason="nonfinite actual or reference; error scalars omitted to retain valid JSON",
        )
    delta = (actual - reference).abs()
    # FP32 rounding varies with tiling. Absolute term scales with sqrt(K),
    # while the FP64 sampled check is independently reported below.
    atol = 2e-5 * math.sqrt(a.shape[1])
    rtol = 2e-4
    ratio = float((delta / (atol + rtol * reference.abs())).max())
    rows = torch.linspace(0, a.shape[0] - 1, min(16, a.shape[0])).round().long().unique()
    exact = a[rows].double() @ b.double()
    sampled = (actual[rows].double() - exact).abs()
    scale = a[rows].double().abs() @ b.double().abs()
    unit = 2**-24
    gamma = (2 * a.shape[1] * unit) / (1 - 2 * a.shape[1] * unit)
    gamma_ratio = float((sampled / (gamma * scale + 1e-30)).max())
    return dict(
        passed=finite and ratio <= 1 and gamma_ratio <= 1,
        finite=finite,
        max_abs=float(delta.max()),
        fp32_tolerance_ratio=ratio,
        atol=atol,
        rtol=rtol,
        fp64_sample_rows=rows.tolist(),
        fp64_max_abs=float(sampled.max()),
        fp64_gamma_2k_ratio=gamma_ratio,
        scope="full output vs torch FP32; deterministic sampled rows vs FP64, conditional gamma diagnostic",
    )
