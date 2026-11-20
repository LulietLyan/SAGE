#!/usr/bin/env python3
"""Run Linux SAGE calibration under the allocation's selected NUMA policy."""
import os
from pathlib import Path
import subprocess
import sys

from gemm import ROOT, bind_args, placement

topology, output = sys.argv[1:3]
output = Path(output).resolve()
output.mkdir(parents=True, exist_ok=False)
cpus, nodes, policy = placement(topology, 8)
binding = bind_args(cpus, nodes, policy)
(out_log := output / "placement.log").write_text(
    " ".join(binding) + "\n" + subprocess.check_output(binding + ["numactl", "--show"], text=True))
env = dict(os.environ, OMP_PROC_BIND="spread",
           OMP_PLACES=",".join("{" + str(cpu) + "}" for cpu in cpus))
subprocess.run(binding + [sys.executable, str(ROOT / "experiments/calibrate.py"),
                         "--output", str(output), *sys.argv[3:]], check=True, env=env)
