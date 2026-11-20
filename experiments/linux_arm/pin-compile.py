"""Assign a compiler command to one CPU in its Donau allocation."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

directory = Path(os.environ["SAGE_BUILD_CPU_LOCKS"])
cpus = sorted(os.sched_getaffinity(0))
while True:
    for cpu in cpus:
        with (directory / str(cpu)).open("a") as slot:
            try:
                fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            os.sched_setaffinity(0, {cpu})
            raise SystemExit(subprocess.run(sys.argv[1:]).returncode)
    time.sleep(0.04)
