"""Check plot routing without importing Torch or requiring an SME compiler."""

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("sage_run", ROOT / "scripts/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PlotRoutingTest(unittest.TestCase):
    def invoke(self, *flags):
        with tempfile.TemporaryDirectory() as output:
            with (
                patch("sys.argv", ["run.py", "--plot-only", "--output", output, *flags]),
                patch.object(runner, "command") as command,
                patch.object(runner, "report") as report,
                patch.object(runner.platform, "system", return_value="Windows"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                runner.main()
                return [
                    Path(call.args[1]).name for call in command.call_args_list
                ], report.call_count

    def test_default_redraws_gemm_and_report_without_building(self):
        scripts, reports = self.invoke()
        self.assertEqual(scripts, ["plot.py", "enrich_figures.py"])
        self.assertEqual(reports, 1)

    def test_operator_flags_redraw_only_the_selected_operators(self):
        for flag, expected in (
            ("--attention", "attention_plot.py"),
            ("--kan", "kan_plot.py"),
            ("--swiglu", "swiglu_plot.py"),
        ):
            with self.subTest(flag=flag):
                self.assertEqual(self.invoke(flag), ([expected], 0))

    def test_suite_and_operator_flags_are_both_respected(self):
        self.assertEqual(
            self.invoke("--suite-v2", "--attention"),
            (["suite_plot_v2.py", "attention_plot.py"], 0),
        )

    def test_plot_only_rejects_requested_measurement_work(self):
        for flag in ("--test", "--recalibrate"):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.invoke(flag)
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
