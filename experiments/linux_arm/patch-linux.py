"""Apply the Linux-only glue to the pinned SAGE and patched Triton snapshots."""
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
triton = root / ".build/triton"
here = Path(__file__).resolve().parent
for repo, patch in (
    (root, "sage-linux.patch"),
    (triton, "triton-linux.patch"),
    (triton, "triton-elf-sme.patch"),
    (triton, "triton-elf-abi-link.patch"),
):
    subprocess.run(["patch", "--batch", "--forward", "-p1", "-i", str(here / patch)],
                   cwd=repo, check=True)
