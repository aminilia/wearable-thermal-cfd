"""1D verification of the bioheat implementation (Pennes source, contact resistance,
convection + radiation boundary) against a semi-analytic solution.

Two columns through the tissue stack (core at 37 C at the bottom):

* ``bare``   - skin exposed to ambient: natural convection + radiation on top.
* ``device`` - a 1 mm plastic housing on the skin with contact resistance R'',
               and a uniform heat flux q'' entering the top of the housing.

The reference solution integrates  d/dy(k dT/dy) = -S(T),  S = w rho_b c_b (T_a - T) + q_m
layer by layer (exact within each layer, via a stiff ODE integrator at tight tolerance)
and shoots on the core heat flux to satisfy the top boundary condition.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .case import TEMPLATES, _copy_render, AIR
from .device3d import (BLOOD, HOUSING, SIGMA, TISSUE, _w, coupled_T, pennes_source,
                       solid_thermo)
from .foamio import Case
from .lattice import Axis, Lattice


@dataclass
class ColumnParams:
    kind: str = "bare"            # "bare" | "device"
    q_top: float = 200.0          # W/m2 into the housing top ("device")
    R_contact: float = 1e-3
    t_mm: float = 1.0
    h_amb: float = 5.0
    T_amb: float = 298.15
    eps: float = 0.98
    T_core: float = 310.15
    refine: float = 1.0
    nIter: int = 4000

    def layers(self):
        """(name, thickness_m, k, a = w rho_b c_b, q_m) from the core upwards."""
        out = [(n, th * 1e-3, k, w * BLOOD["rho"] * BLOOD["cp"], qm)
               for n, th, k, rho, cp, w, qm in TISSUE]
        if self.kind == "device":
            out.append(("housing", self.t_mm * 1e-3, HOUSING["k"], 0.0, 0.0))
        return out


# --------------------------------------------------------------------------- reference
def reference(p: ColumnParams, n_out=2000):
    L = p.layers()

    def shoot(q0, dense=False):
        """Integrate upward from the core with core heat flux q0 (W/m2, +y)."""
        T, q = p.T_core, q0
        ys, Ts = [], []
        y0 = 0.0
        for name, th, k, a, qm in L:
            if name == "housing":
                T = T - q * p.R_contact          # contact resistance: jump of q R''
            f = lambda y, s: [-s[1] / k, a * (BLOOD["T"] - s[0]) + qm]
            sol = solve_ivp(f, (y0, y0 + th), [T, q], method="Radau", rtol=1e-11, atol=1e-12,
                            dense_output=dense)
            if dense:
                yy = np.linspace(y0, y0 + th, max(20, int(n_out * th / 0.02)))
                ys.append(yy)
                Ts.append(sol.sol(yy)[0])
            T, q = sol.y[0, -1], sol.y[1, -1]
            y0 += th
        return T, q, (np.concatenate(ys), np.concatenate(Ts)) if dense else None

    def residual(q0):
        T, q, _ = shoot(q0)
        if p.kind == "bare":
            return q - (p.h_amb * (T - p.T_amb) + p.eps * SIGMA * (T**4 - p.T_amb**4))
        return q + p.q_top                        # flux leaving the top = -q_in

    q0 = brentq(residual, -5000, 5000, xtol=1e-12)
    T, q, prof = shoot(q0, dense=True)
    depth = sum(l[1] for l in TISSUE) * 1e-3
    y, Tp = prof
    return dict(q_core=q0, T_top=T, y=y - depth, T=Tp)


# --------------------------------------------------------------------------- OpenFOAM
def build_column(case_dir: Path, p: ColumnParams) -> Path:
    case_dir = Path(case_dir)
    if case_dir.exists():
        shutil.rmtree(case_dir)
    r = p.refine
    n = lambda b: max(1, int(round(b * r)))
    tissue = [t[0] for t in TISSUE]
    ylines, yc = [-sum(t[1] for t in TISSUE)], []
    base = {"inner": 60, "fat": 6, "dermis": 15, "epidermis": 8}
    for name, th, *_ in TISSUE:
        ylines.append(round(ylines[-1] + th, 9))
        yc.append(n(base[name]))
    names = list(tissue)
    if p.kind == "device":
        ylines.append(p.t_mm)
        yc.append(n(10))
        names.append("housing")
    lat = Lattice(Axis([0, 1], [1]), Axis(ylines, yc), Axis([0, 1], [1]),
                  zone=lambda i, j, k: names[j],
                  classify=lambda i, j, k, d: (("core", "wall") if d == "-y" else
                                               ("top", "wall") if d == "+y" else
                                               ("sides", "empty")))
    toks = dict(AIR, nIter=p.nIter, writeInterval=max(1, p.nIter // 10), k_solid=1, qvol=0)
    _copy_render(TEMPLATES / "common", case_dir, toks, skip_regions=("air", "heater"))
    c = case_dir
    (c / "system" / "blockMeshDict").write_text(lat.block_mesh_dict(f"1D bioheat column ({p.kind})"))
    _w(c / "constant" / "regionProperties", "dictionary", "regionProperties",
       f"regions ( fluid () solid ({' '.join(names)}) );\n")
    solid_sys = TEMPLATES / "common" / "system" / "heater"
    for name, th, k, rho, cp, w, qm in TISSUE:
        _w(c / "constant" / name / "thermophysicalProperties", "dictionary",
           "thermophysicalProperties", solid_thermo(k, rho, cp))
        if w > 0 or qm > 0:
            _w(c / "constant" / name / "fvOptions", "dictionary", "fvOptions",
               pennes_source(w, qm, cp))
    if p.kind == "device":
        _w(c / "constant" / "housing" / "thermophysicalProperties", "dictionary",
           "thermophysicalProperties", solid_thermo(HOUSING["k"], HOUSING["rho"], HOUSING["cp"]))
    for nm in names:
        _copy_render(solid_sys, c / "system" / nm, toks)
    top = names[-1]
    for i, nm in enumerate(names):
        ents = ["    sides           { type empty; }"]
        if i > 0:
            Rc = p.R_contact if {nm, names[i - 1]} == {"housing", "epidermis"} else None
            ents.append(f"    {nm}_to_{names[i - 1]}\n    {coupled_T('solidThermo', Rc)}")
        if i + 1 < len(names):
            Rc = p.R_contact if {nm, names[i + 1]} == {"housing", "epidermis"} else None
            ents.append(f"    {nm}_to_{names[i + 1]}\n    {coupled_T('solidThermo', Rc)}")
        if i == 0:
            ents.append(f"    core            {{ type fixedValue; value uniform {p.T_core:g}; }}")
        if nm == top:
            if p.kind == "bare":
                ents.append(f"""    top
    {{
        type            externalWallHeatFluxTemperature;
        mode            coefficient;
        h               uniform {p.h_amb:g};
        Ta              constant {p.T_amb:g};
        emissivity      {p.eps:g};
        kappaMethod     solidThermo;
        value           $internalField;
    }}""")
            else:
                ents.append(f"""    top
    {{
        type            externalWallHeatFluxTemperature;
        mode            flux;
        q               uniform {p.q_top:g};
        kappaMethod     solidThermo;
        value           $internalField;
    }}""")
        _w(c / "0.orig" / nm / "T", "volScalarField", "T",
           "dimensions      [0 0 0 1 0 0 0];\ninternalField   uniform 308;\n\nboundaryField\n{\n"
           + "\n".join(ents) + "\n}\n")
        _w(c / "0.orig" / nm / "p", "volScalarField", "p",
           "dimensions      [1 -1 -2 0 0 0 0];\ninternalField   uniform 1e5;\n\nboundaryField\n{\n"
           "    \".*\"            { type calculated; value $internalField; }\n"
           "    sides           { type empty; }\n}\n")
    (c / "Allrun").write_text("""#!/bin/sh
cd "${0%/*}" || exit 1
set -e
blockMesh                              > log.blockMesh 2>&1
splitMeshRegions -cellZones -overwrite > log.splitMeshRegions 2>&1
rm -rf 0 && cp -r 0.orig 0
chtMultiRegionSimpleFoam               > log.chtMultiRegionSimpleFoam 2>&1
""")
    (c / "Allrun").chmod(0o755)
    (c / "params.json").write_text(json.dumps(asdict(p)))
    return c


def column_profile(case_dir: Path, p: ColumnParams):
    """Cell-centre (y, T) through all regions, sorted by y [m, K]."""
    c = Case(case_dir)
    names = [t[0] for t in TISSUE] + (["housing"] if p.kind == "device" else [])
    ys, Ts = [], []
    for nm in names:
        m = c.mesh(nm)
        ys.append(m.C[:, 1])
        Ts.append(c.field("T", nm).internal)
    y, T = np.concatenate(ys), np.concatenate(Ts)
    o = np.argsort(y)
    top = c.field("T", names[-1]).boundary["top"][0]
    return y[o], T[o], float(top)
