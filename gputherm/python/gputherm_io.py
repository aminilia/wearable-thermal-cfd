"""Write gputherm spec files, run a solver binary, read the temperature field back."""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin"


@dataclass
class Spec:
    dx: list            # cell widths [m], one per cell along each axis
    dy: list
    dz: list
    regions: list       # (k, a, s): conductivity, w rho_b c_b, a T_a + q_m
    contacts: list = field(default_factory=list)   # (r1, r2, R'')
    boxes: list = field(default_factory=list)      # (i0, i1, j0, j1, k0, k1, region)
    bc: dict = field(default_factory=dict)         # side -> ("dirichlet", T) | ("flux", q) | ("robin", h, Tinf, eps)
    solver: tuple = (1e-12, 200000, 1e-10, 100)    # tol, maxit, picard_tol, picard_max

    @property
    def shape(self):
        return len(self.dz), len(self.dy), len(self.dx)   # (nz, ny, nx), C order of the output

    def write(self, path: Path):
        f = lambda v: " ".join(f"{x:.17g}" for x in v)
        lines = ["gputherm 1", f"grid {len(self.dx)} {len(self.dy)} {len(self.dz)}",
                 f"dx {f(self.dx)}", f"dy {f(self.dy)}", f"dz {f(self.dz)}",
                 f"regions {len(self.regions)}"]
        lines += [f(r) for r in self.regions]
        lines.append(f"contact {len(self.contacts)}")
        lines += [f"{a} {b} {R:.17g}" for a, b, R in self.contacts]
        lines.append(f"boxes {len(self.boxes)}")
        lines += [" ".join(str(int(v)) for v in b) for b in self.boxes]
        for side, bc in self.bc.items():
            lines.append(f"bc {side} {bc[0]} {f(bc[1:])}".rstrip())
        lines.append("solver " + " ".join(str(v) for v in self.solver))
        Path(path).write_text("\n".join(lines) + "\n")


def binary(backend="cpu") -> Path:
    b = BIN / f"gputherm_{backend}"
    if not b.exists():
        raise FileNotFoundError(f"{b} not built: run `make {backend}` in {ROOT}")
    return b


def run(spec: Spec, backend="cpu", workdir=None):
    """Solve `spec`; returns (summary dict, T array shaped (nz, ny, nx))."""
    with tempfile.TemporaryDirectory(dir=workdir) as d:
        sp, out = Path(d) / "case.spec", Path(d) / "T.bin"
        spec.write(sp)
        res = subprocess.run([str(binary(backend)), str(sp), "--out", str(out)],
                             capture_output=True, text=True, check=True)
        summary = json.loads(res.stdout.strip().splitlines()[-1])
        T = np.fromfile(out, dtype=np.float64).reshape(spec.shape)
    return summary, T


def bench(N: int, backend="cpu", tol=None):
    cmd = [str(binary(backend)), "--bench", str(N)]
    if tol:
        cmd += ["--tol", str(tol)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout.strip().splitlines()[-1])
