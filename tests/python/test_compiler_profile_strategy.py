"""Synthetic response-family regressions; these measure no kernel speed.

Python exports observed response points. C++ strategyResponseDomainTests cover
schedule admission, interpolation and endpoint clamps in the actual planner.
"""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from sme_plan.calibration import features, fit_team, strategy_responses


def observation(process=0, **changes):
    row = dict(
        threads=1,
        process=process,
        pid=100 + process,
        size=704,
        method="synthetic",
        selectedSme=True,
        eligibleForFit=True,
        blockM=128,
        blockN=1024,
        blockK=32,
        vectorAPack=True,
        bLayout="original",
        stripK=0,
        panelBytes=524288,
        response=100.0,
        compute=1000.0,
    )
    row.update(changes)
    return row


def cpu_observations():
    return [observation(p, selectedSme=False, method="generic", compute=50.0) for p in range(3)]


def family_key(row):
    return (
        row["blockM"],
        row["blockN"],
        row["blockK"],
        row["aPackVector4"],
        row["bLayout"],
        row["streamingK"],
    )


class CompilerProfileStrategyTest(unittest.TestCase):
    def test_equal_panel_does_not_pool_other_strategy_dimensions(self):
        variants = [
            {},
            {"blockM": 256},
            {"blockN": 256},
            {"blockK": 16},
            {"vectorAPack": False},
            {"bLayout": "local"},
            {"bLayout": "global"},
            {"stripK": 512},
        ]
        rows = [
            observation(p, response=10.0 * (i + 1), **variant)
            for i, variant in enumerate(variants)
            for p in range(3)
        ]
        strategies = strategy_responses(rows)
        self.assertEqual(len(strategies), len(variants))
        indexed = {family_key(row): row for row in strategies}
        for i, variant in enumerate(variants):
            row = observation(**variant)
            key = (
                row["blockM"],
                row["blockN"],
                row["blockK"],
                row["vectorAPack"],
                row["bLayout"],
                bool(row["stripK"]),
            )
            self.assertEqual(indexed[key]["independentProcesses"], 3)
            self.assertEqual(
                indexed[key]["panelResponses"],
                [dict(panelBytes=524288, operandBandwidthGBps=10.0 * (i + 1))],
            )

    def test_strip_points_share_family_but_shapes_do_not_inflate_processes(self):
        rows = [
            observation(
                p, size=size, stripK=strip, panelBytes=1024 * strip, response=10.0 * (p + 1) + strip
            )
            for p in range(3)
            for size in (704, 1664)
            for strip in (512, 1024)
        ]
        # More observations from one PID must not give that process more weight.
        rows += [deepcopy(row) for row in rows if row["process"] == 0] * 20
        strategies = strategy_responses(rows)
        self.assertEqual(len(strategies), 1)
        self.assertEqual(strategies[0]["independentProcesses"], 3)
        self.assertEqual(
            strategies[0]["panelResponses"],
            [
                dict(panelBytes=524288, operandBandwidthGBps=532.0),
                dict(panelBytes=1048576, operandBandwidthGBps=1044.0),
            ],
        )

    def test_rejected_residuals_remain_coverage_without_authorizing_response(self):
        rows = [
            observation(
                p,
                eligibleForFit=p != 2,
                response=100.0 if p != 2 else None,
                compute=1000.0 if p != 2 else None,
            )
            for p in range(3)
        ]
        original = deepcopy(rows)
        fitted = fit_team(rows + cpu_observations(), {})
        self.assertEqual(fitted["observedNativeTiles"][0]["independentProcesses"], 3)
        self.assertEqual(fitted["strategyResponses"][0]["independentProcesses"], 2)
        self.assertEqual(
            fitted["strategyResponses"][0]["panelResponses"],
            [dict(panelBytes=524288, operandBandwidthGBps=100.0)],
        )
        self.assertEqual(rows, original)
        rejected = [dict(row, eligibleForFit=False, response=None, compute=None) for row in rows]
        self.assertEqual(strategy_responses(rejected), [])
        other = [observation(p, blockM=256) for p in range(3)]
        fitted = fit_team(rejected + other + cpu_observations(), {})
        self.assertEqual([row["blockM"] for row in fitted["observedNativeTiles"]], [128, 256])
        self.assertEqual([row["blockM"] for row in fitted["strategyResponses"]], [256])

    def test_full_k_cannot_borrow_streaming_or_pooled_response(self):
        rows = [observation(p, stripK=512) for p in range(3)]
        fitted = fit_team(rows + cpu_observations(), {})
        fitted["operandBandwidthGBps"] = 1e12
        self.assertEqual(len(fitted["strategyResponses"]), 1)
        self.assertTrue(fitted["strategyResponses"][0]["streamingK"])
        self.assertNotIn("panelResponses", fitted)  # No pooled curve can become a native family.
        with_full = strategy_responses(rows + [observation(p, response=7.0) for p in range(3)])
        full = next(row for row in with_full if not row["streamingK"])
        self.assertEqual(full["panelResponses"][0]["operandBandwidthGBps"], 7.0)

    def test_streaming_exports_only_observed_interval_and_keeps_full_k_separate(self):
        rows = [
            observation(p, stripK=strip, panelBytes=1024 * strip)
            for p in range(3)
            for strip in (512, 1024)
        ]
        rows += [observation(p, panelBytes=131072, response=5.0) for p in range(3)]
        strategies = strategy_responses(rows)
        streaming = next(row for row in strategies if row["streamingK"])
        full = next(row for row in strategies if not row["streamingK"])
        self.assertEqual(
            [point["panelBytes"] for point in streaming["panelResponses"]], [524288, 1048576]
        )
        self.assertEqual(
            full["panelResponses"], [dict(panelBytes=131072, operandBandwidthGBps=5.0)]
        )
        # Missing intermediate/outside points are not synthesized in Python.
        self.assertNotIn(786432, [point["panelBytes"] for point in streaming["panelResponses"]])
        self.assertNotIn(131072, [point["panelBytes"] for point in streaming["panelResponses"]])

    def test_other_thread_team_cannot_supply_independence(self):
        rows = [observation(p, threads=2**p) for p in range(3)]
        with self.assertRaisesRegex(ValueError, "mix thread teams"):
            strategy_responses(rows)
        with self.assertRaisesRegex(ValueError, "pool thread teams"):
            fit_team(rows + cpu_observations(), {})

    def test_unidentifiable_backend_is_rejected_instead_of_fabricating_rates(self):
        rows = [observation(p) for p in range(3)] + cpu_observations()
        for native in (True, False):
            with self.subTest(selectedSme=native):
                missing = [
                    dict(row, eligibleForFit=False, response=None, compute=None)
                    if row["selectedSme"] == native
                    else row
                    for row in rows
                ]
                with self.assertRaisesRegex(ValueError, "no identifiable SME or CPU response"):
                    fit_team(missing, {})

    def test_features_subtract_per_wave_communication(self):
        metadata = {
            "sme_plan_block_m": 32,
            "sme_plan_block_n": 32,
            "sme_plan_block_k": 16,
            "sme_plan_threads": 2,
            "sme_plan_k_strip": 0,
            "sme_plan_a_pack_vector4": False,
            "sme_plan_b_panel_pad": 0,
            "sme_plan_b_global_pad": 0,
            "sme_plan_selected_sme": False,
        }
        rates = {
            "scalarAPackGBps": 1,
            "vectorAPackGBps": 1,
            "copyGBps": 1,
            "globalBPackGBps": 1,
            "callOverheadNs": 10,
            "taskOverheadNs": 2,
            "communicationNsPerWave": 0,
        }
        baseline = features((128, 128, 64), metadata, rates)
        rates["communicationNsPerWave"] = 50
        with_comm = features((128, 128, 64), metadata, rates)
        # 16 output tasks over a two-thread team produce eight waves.
        self.assertAlmostEqual(
            with_comm["knownPreparationNs"] - baseline["knownPreparationNs"], 8 * 50
        )


if __name__ == "__main__":
    unittest.main()
