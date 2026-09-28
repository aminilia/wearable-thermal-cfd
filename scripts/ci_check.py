#!/usr/bin/env python3
"""CI smoke test: run the coarse channel and coarse CHT cases and check them.

Runs natively if OpenFOAM is on the machine, or inside a container when
COOLCHAN_DOCKER_IMAGE is set (what the GitHub Actions job does).
Exits non-zero if any verification check fails.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coolchan.case import ChannelParams, ChtParams, build_channel, build_cht  # noqa: E402
from coolchan.cht import analyse_cht  # noqa: E402
from coolchan.runner import run_many  # noqa: E402
from coolchan.verification import NU_FD, FRE_FD, analyse_channel  # noqa: E402
from coolchan.bfs import BFSParams, build_bfs, reattachment, write_inlet_velocity  # noqa: E402
from coolchan.runner import run_case  # noqa: E402
from coolchan.bioheat import ColumnParams, build_column, column_profile, reference  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "ci"

CH = ChannelParams(ny=10, nIter=1000)
CHT = replace(ChtParams(), dy_mm=0.10, nIter=1500)
# reference value from the committed Stage-2 coarse-mesh run (results/stage2_summary.json)
REF = json.loads((REPO / "results" / "stage2_summary.json").read_text())["meshes"]["coarse"]


def main() -> int:
    build_channel(RUNS / "channel", CH)
    build_cht(RUNS / "cht", CHT)
    out = run_many([RUNS / "cht", RUNS / "channel"], workers=2)
    if any(isinstance(v, Exception) for v in out.values()):
        for k, v in out.items():
            print(k, v)
        return 1

    # Phase 1 smoke test: Gartling BFS on the coarse (20 cells/S) mesh
    gp = BFSParams(Re=800, S_mm=5.0, h_mm=5.0, Lin_S=0, Ld_S=60, n_S=20, nIter=20000)
    gd = build_bfs(RUNS / "gartling", gp)
    run_case(gd, "Allmesh")
    write_inlet_velocity(gd, gp)
    run_case(gd, "Allsolve")
    bf = reattachment(gd, gp)

    # Phase 2 smoke test: 1D bioheat columns vs semi-analytic (checks the Pennes source,
    # contact layer, radiation BC and the enthalpy reference on this OpenFOAM release)
    col_err = {}
    for kind, kw in (("bare", {}), ("device", dict(q_top=300.0, R_contact=5e-3))):
        cp_ = ColumnParams(kind=kind, **kw)
        cd = build_column(RUNS / f"column_{kind}", cp_)
        run_case(cd)
        y, T, _ = column_profile(cd, cp_)
        ref = reference(cp_)
        col_err[kind] = float(np.abs(T - np.interp(y, ref["y"], ref["T"])).max())

    ch = analyse_channel(RUNS / "channel", CH)
    ht = analyse_cht(RUNS / "cht", CHT)
    checks = {
        "channel: Nu_fd within 1.5 % of 140/17": (abs(ch.Nu_fd / NU_FD - 1), 0.015),
        "channel: fRe within 3 % of 96": (abs(ch.fRe / FRE_FD - 1), 0.03),
        "channel: u(y) within 3 % of Poiseuille": (ch.u_max_err, 0.03),
        "channel: energy balance < 1e-4": (abs(ch.energy_err), 1e-4),
        "cht: energy balance < 0.5 %": (abs(ht.energy_err), 5e-3),
        "cht: T_max within 1 K of reference": (abs(ht.T_max_C - REF["T_max_C"]), 1.0),
        "cht: iterative change of T_max < 1 mK": (ht.dTmax_iter, 1e-3),
        # coarse-mesh values are 3.2 % / 3.8 % / 0.8 % below Gartling (1990)
        "bfs: x1 within 5 % of Gartling 12.2": (abs(bf["x1"] / 12.2 - 1), 0.05),
        "bfs: x4 within 5 % of Gartling 9.7": (abs(bf["x4"] / 9.7 - 1), 0.05),
        "bfs: x5 within 5 % of Gartling 21.0": (abs(bf["x5"] / 21.0 - 1), 0.05),
        "bioheat bare column: max |error| < 1 mK": (col_err["bare"], 1e-3),
        "bioheat device column: max |error| < 1 mK": (col_err["device"], 1e-3),
    }
    ok = True
    for name, (val, tol) in checks.items():
        passed = val <= tol
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {name:45s} value={val:.3g}  tol={tol:g}")
    report = dict(channel=dict(Nu_fd=ch.Nu_fd, fRe=ch.fRe, u_max_err=ch.u_max_err,
                               energy_err=ch.energy_err),
                  cht={k: float(v) for k, v in ht.as_dict().items()},
                  bfs_gartling=bf, bioheat_column_err=col_err,
                  passed=bool(ok))
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "ci_report.json").write_text(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
