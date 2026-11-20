#!/usr/bin/env python3
"""Run only correctness and instruction checks on the Linux native compiler."""
import json
from pathlib import Path
import subprocess
import sys

from gemm import ROOT, placement, bind_args


def main():
    if sys.argv[1] != "--worker":
        out = Path(sys.argv[1]).resolve()
        out.mkdir(parents=True, exist_ok=True)
        for team in (1, 8):
            binding = bind_args(*placement("single", team))
            subprocess.run(binding + [sys.executable, __file__, "--worker", str(team), str(out)], check=True)
        return
    from sme_plan.compiler_backend import compile_gemm, find_tool, prepare
    from common import check_gemm
    team, out = int(sys.argv[2]), Path(sys.argv[3])
    torch = prepare(team, out / "jit-cache")
    print(f"Starting native validation with {team} threads", flush=True)
    g = torch.Generator().manual_seed(7301)
    results = []
    for dims in ((193, 257, 129), (128, 128, 128)):
        m, n, k = dims
        a = torch.randn((m, k), generator=g)
        b = torch.randn((k, n), generator=g)
        for mode in ("generic", "sme-fixed"):
            c = torch.empty((m, n))
            c.fill_(float("nan"))
            call = compile_gemm(a, b, c, mode, team)
            checked = []
            for change in (0, 1):
                rhs = b.clone()
                if change:
                    rhs.add_(0.25)
                    b.copy_(rhs)
                checked.append(check_gemm(call(), a @ b, a, b))
            assert all(t["passed"] for t in checked), checked
            evidence = call.evidence
            assert evidence["backend_verified"]
            assert (evidence["fmopa_count"] > 0) == (mode == "sme-fixed")
            jitted = next(Path(p) for n, p in call.compiled.metadata_group.items() if n.endswith(".so"))
            tag = f"{mode}-t{team}-{m}x{n}x{k}"
            (out / (tag + ".llir")).write_text(call.compiled.asm["llir"])
            listing = subprocess.check_output(
                [find_tool("llvm-objdump"), "--disassemble", "--mattr=+sme", str(jitted)], text=True)
            (out / (tag + ".s")).write_text(listing)
            results.append(dict(method=mode, checks=checked, **evidence))
            print(f"{tag} passed with {evidence['fmopa_count']} fmopa instructions", flush=True)
    (out / f"checks-t{team}.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"CPU and FP32 SME checks passed at thread budget {team}")


if __name__ == "__main__":
    main()
