"""Verify gputherm against the semi-analytic 1D bioheat solution (coolchan.bioheat).

Same two columns as Phase 2's OpenFOAM check (scripts/phase2_device.py column):
  bare   - skin exposed to the room: natural convection + radiation on top
  device - 1 mm housing on the skin with contact resistance R'', heat flux into its top
Each is solved on three grids (refine = 1, 2, 4 x the OpenFOAM column's cell counts),
and the observed order of accuracy is computed from the max cell-centre error.
The column is also extruded to a 3 x 3 x nz block with adiabatic sides, which must
reproduce the 1D answer (checks the x/y stencil and boundary bookkeeping).

    python python/verify_column.py                     # CPU
    python python/verify_column.py --backend cuda      # GPU
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root -> coolchan
from gputherm_io import Spec, run                                # noqa: E402
from coolchan.bioheat import ColumnParams, reference             # noqa: E402
from coolchan.device3d import BLOOD, HOUSING, SIGMA, TISSUE      # noqa: E402

BASE = {"inner": 60, "fat": 6, "dermis": 15, "epidermis": 8, "housing": 10}


def column_spec(p: ColumnParams, refine: int, lateral: int = 1):
    dz, regions, names = [], [], []
    for name, th, k, a, qm in p.layers():
        n = BASE[name] * refine
        dz += [th / n] * n
        regions.append((k, a, a * BLOOD["T"] + qm))
        names.append(name)
    boxes, k0 = [], 0
    for r, (name, th, *_ ) in enumerate(p.layers()):
        n = BASE[name] * refine
        boxes.append((0, lateral, 0, lateral, k0, k0 + n, r))
        k0 += n
    contacts = []
    if p.kind == "device":
        contacts.append((names.index("epidermis"), names.index("housing"), p.R_contact))
    top = ("robin", p.h_amb, p.T_amb, p.eps) if p.kind == "bare" else ("flux", p.q_top)
    w = 1e-3
    return Spec(dx=[w] * lateral, dy=[w] * lateral, dz=dz, regions=regions,
                contacts=contacts, boxes=boxes,
                bc={"zm": ("dirichlet", p.T_core), "zp": top})


def surface_T(p: ColumnParams, spec: Spec, T_top_cell: float):
    """Outer-surface temperature implied by the top cell and the top boundary condition."""
    k = spec.regions[-1][0]
    gc = 2 * k / spec.dz[-1]
    if p.kind == "device":
        return T_top_cell + p.q_top / gc
    Ts = T_top_cell
    for _ in range(50):
        hr = p.eps * SIGMA * (Ts**2 + p.T_amb**2) * (Ts + p.T_amb)
        Ts = (gc * T_top_cell + (p.h_amb + hr) * p.T_amb) / (gc + p.h_amb + hr)
    return Ts


def check(kind: str, backend: str, refines=(1, 2, 4)):
    p = ColumnParams(kind=kind)
    ref = reference(p)
    depth = sum(t[1] for t in TISSUE) * 1e-3
    rows = []
    for r in refines:
        spec = column_spec(p, r)
        summ, T = run(spec, backend)
        Tc = T[:, 0, 0]
        zc = np.cumsum(spec.dz) - 0.5 * np.asarray(spec.dz) - depth
        # the reference profile is dense per layer; interpolate inside each cell's layer
        Tref = np.interp(zc, ref["y"], ref["T"])
        err = np.abs(Tc - Tref)
        Ts = surface_T(p, spec, Tc[-1])
        rows.append(dict(refine=r, cells=len(Tc), max_err_mK=1e3 * err.max(),
                         top_err_mK=1e3 * abs(Ts - ref["T_top"]),
                         cg_iters=summ["cg_iters"], picard=summ["picard_iters"]))
    for a, b in zip(rows, rows[1:]):
        b["order"] = float(np.log(a["max_err_mK"] / b["max_err_mK"]) / np.log(2))
    # 3 x 3 lateral extrusion at the finest level: every column must match the 1D solve
    spec3 = column_spec(p, refines[-1], lateral=3)
    _, T3 = run(spec3, backend)
    spread = 1e3 * float(np.abs(T3 - T3[:, :1, :1]).max())
    _, T1 = run(column_spec(p, refines[-1]), backend)
    vs1d = 1e3 * float(np.abs(T3[:, 1, 1] - T1[:, 0, 0]).max())
    return dict(kind=kind, backend=backend, T_top_ref=ref["T_top"], grids=rows,
                extrusion_spread_mK=spread, extrusion_vs_1d_mK=vs1d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--json", type=Path, help="also write results here")
    a = ap.parse_args()
    out = [check(kind, a.backend) for kind in ("bare", "device")]
    ok = True
    for res in out:
        print(f"\n[{res['kind']}]  backend={res['backend']}  "
              f"T_surface(ref) = {res['T_top_ref'] - 273.15:.6f} C")
        print(f"  {'refine':>6} {'cells':>6} {'max err [mK]':>13} {'surface err [mK]':>17} "
              f"{'order':>6} {'CG its':>7} {'Picard':>6}")
        for g in res["grids"]:
            order = f"{g['order']:.2f}" if "order" in g else "-"
            print(f"  {g['refine']:>6} {g['cells']:>6} {g['max_err_mK']:>13.4f} "
                  f"{g['top_err_mK']:>17.4f} {order:>6} {g['cg_iters']:>7} {g['picard']:>6}")
        print(f"  3x3 extrusion: lateral spread {res['extrusion_spread_mK']:.2e} mK, "
              f"vs 1D {res['extrusion_vs_1d_mK']:.2e} mK")
        fine = res["grids"][-1]
        ok &= fine["max_err_mK"] < 1.0 and res["grids"][-1].get("order", 0) > 1.8 \
            and res["extrusion_vs_1d_mK"] < 1e-3
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
