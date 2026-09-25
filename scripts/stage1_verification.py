#!/usr/bin/env python3
"""Stage 1 - code verification on a 2D parallel-plate channel.

1. Cold run (q'' = 0): velocity profile and pressure gradient vs plane Poiseuille.
2. Heated runs (uniform q'' on both plates) on four systematically refined
   meshes (r = 2): fully developed Nu_Dh vs 140/17 = 8.235.
3. Three-grid GCI (Celik et al. 2008) on Nu_fd, fRe and total pressure drop.

Usage:  python scripts/stage1_verification.py [--levels 10 20 40 80] [--skip-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coolchan.case import ChannelParams, build_channel, AIR  # noqa: E402
from coolchan.foamio import Case  # noqa: E402
from coolchan.runner import run_many  # noqa: E402
from coolchan.verification import (NU_FD, FRE_FD, analyse_channel, gci,  # noqa: E402
                                   poiseuille_u)
from coolchan import plots  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "stage1"
RES = REPO / "results"
FIG = REPO / "docs" / "figures"
ITERS = {10: 1000, 20: 1500, 40: 2500, 80: 4000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[10, 20, 40, 80])
    ap.add_argument("--skip-run", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    levels = sorted(a.levels)
    RES.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    cases = {f"heated_ny{n}": ChannelParams(ny=n, nIter=ITERS.get(n, 4000)) for n in levels}
    n_cold = levels[-2] if len(levels) > 1 else levels[-1]
    cases[f"cold_ny{n_cold}"] = ChannelParams(ny=n_cold, q_wall=0.0, nIter=ITERS.get(n_cold, 4000))

    if not a.skip_run:
        for name, p in cases.items():
            build_channel(RUNS / name, p)
        print(f"running {len(cases)} channel cases ...")
        # biggest first so the pool finishes together
        order = sorted(cases, key=lambda k: -cases[k].ny)
        out = run_many([RUNS / k for k in order], a.workers)
        bad = [k for k, v in out.items() if isinstance(v, Exception)]
        if bad:
            sys.exit(f"failed: {bad}")

    # ------------------------------------------------------------ analyse
    res = {}
    for name, p in cases.items():
        r = analyse_channel(RUNS / name, p)
        times = Case(RUNS / name).times()
        r_prev = analyse_channel(RUNS / name, p, time=times[-2]) if len(times) > 2 else r
        res[name] = (p, r, r_prev)

    cold_p, cold, _ = res[f"cold_ny{n_cold}"]
    summary = {
        "Re_Dh": cold_p.Re,
        "Pr": AIR["Pr_air"],
        "cold": {
            "ny": n_cold,
            "fRe": cold.fRe,
            "fRe_err_pct": 100 * (cold.fRe - FRE_FD) / FRE_FD,
            "u_profile_max_err_pct_of_Um": 100 * cold.u_max_err,
            "dpdx_Pa_per_m": cold.dpdx,
            "dpdx_exact": -12 * AIR["mu_air"] * cold_p.U_in / (cold_p.H_mm * 1e-3) ** 2,
        },
        "levels": [],
    }
    # one-way coupling check: flow in the heated run is identical to the cold run
    hot_same = RUNS / f"heated_ny{n_cold}"
    if hot_same.exists():
        Uc = Case(RUNS / f"cold_ny{n_cold}").field("U", "air").internal
        Uh = Case(hot_same).field("U", "air").internal
        summary["cold"]["max_dU_heated_vs_cold"] = float(np.abs(Uc - Uh).max())

    for n in levels:
        p, r, rp = res[f"heated_ny{n}"]
        summary["levels"].append(dict(
            ny=n, nx=p.nx, cells=p.nx * n, h_mm=p.H_mm / n,
            Nu_fd=r.Nu_fd, Nu_err_pct=100 * (r.Nu_fd - NU_FD) / NU_FD,
            fRe=r.fRe, fRe_err_pct=100 * (r.fRe - FRE_FD) / FRE_FD,
            dp_Pa=r.dp_total, u_max_err_pct=100 * r.u_max_err,
            energy_err=r.energy_err,
            iter_change_Nu=abs(r.Nu_fd - rp.Nu_fd),
        ))

    # GCI on the three finest meshes
    L = summary["levels"][-3:]
    h = [l["h_mm"] for l in L][::-1]          # fine, medium, coarse
    gcis = {}
    for key, exact in (("Nu_fd", NU_FD), ("fRe", FRE_FD), ("dp_Pa", None)):
        f1, f2, f3 = [l[key] for l in L][::-1]
        g = gci(f1, f2, f3, *h)
        gcis[key] = dict(phi1=f1, phi2=f2, phi3=f3, p=g.p, phi_ext=g.phi_ext,
                         e21_a_pct=100 * g.e21_a, gci_fine_pct=100 * g.gci_fine,
                         gci_medium_pct=100 * g.gci_medium,
                         asymptotic_ratio=g.asymptotic, oscillatory=g.oscillatory,
                         exact=exact,
                         ext_err_vs_exact_pct=(100 * (g.phi_ext - exact) / exact
                                               if exact else None))
    summary["gci"] = gcis
    (RES / "stage1_summary.json").write_text(json.dumps(summary, indent=2))
    write_gci_table(summary)
    make_figures(res, levels, cold_p, cold, summary)
    print(json.dumps({k: summary[k] for k in ("cold", "gci")}, indent=2))


def write_gci_table(s):
    rows = ["| Quantity | Grid 3 (coarse) | Grid 2 | Grid 1 (fine) | Observed order p "
            "| Richardson extrapolation | GCI<sub>fine</sub> | Analytical | Extrap. vs exact |",
            "|---|---|---|---|---|---|---|---|---|"]
    names = {"Nu_fd": "Nu<sub>Dh</sub> (fully developed)", "fRe": "f·Re (Darcy)",
             "dp_Pa": "Δp inlet→outlet [Pa]"}
    for k, g in s["gci"].items():
        ex = f"{g['exact']:.4g}" if g["exact"] else "-"
        ee = f"{g['ext_err_vs_exact_pct']:+.3f} %" if g["exact"] else "-"
        rows.append(f"| {names[k]} | {g['phi3']:.4f} | {g['phi2']:.4f} | {g['phi1']:.4f} "
                    f"| {g['p']:.2f} | {g['phi_ext']:.4f} | {g['gci_fine_pct']:.3f} % | {ex} | {ee} |")
    L = s["levels"][-3:]
    rows.append("")
    rows.append("Grids (cells across gap × along channel): " + ", ".join(
        f"{l['ny']}×{l['nx']}" for l in L) + "; refinement ratio r = 2.")
    (RES / "stage1_gci.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


def make_figures(res, levels, cold_p, cold, s):
    import matplotlib.pyplot as plt
    plots.setup()
    H = cold_p.H_mm * 1e-3

    # 1. velocity profile
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    yy = np.linspace(0, H, 200)
    ax.plot(poiseuille_u(yy, H, cold_p.U_in) / cold_p.U_in, yy / H, color=plots.INK2,
            lw=1.5, label="Analytical: 6η(1−η)")
    ax.plot(cold.u_prof / cold_p.U_in, cold.y_prof / H, "o", color=plots.C1, ms=5,
            mfc="none", mew=1.4, label=f"CFD, {cold_p.ny} cells across gap")
    ax.set_xlabel("u / U$_m$")
    ax.set_ylabel("y / H")
    ax.set_title(f"Cold run: velocity at x/D$_h$ = {cold.x_probe / (2 * H):.0f}")
    ax.legend(loc="lower right")
    ax.text(0.02, 0.98, f"max error {100 * cold.u_max_err:.2f} % of U$_m$\n"
            f"f·Re = {cold.fRe:.2f} (exact 96)", transform=ax.transAxes, va="top",
            fontsize=8.5, color=plots.INK2)
    plots.save(fig, FIG / "stage1_velocity_profile.png")

    # 2. local Nu along the channel
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    cols = [plots.C4, plots.C3, plots.C2, plots.C1][-len(levels):]
    for n, c in zip(levels, cols):
        p, r, _ = res[f"heated_ny{n}"]
        xp = r.x / (p.Dh * p.Re * AIR["Pr_air"])
        ax.plot(xp, r.Nu, color=c, lw=1.6, label=f"{n} cells across gap")
    ax.axhline(NU_FD, color=plots.INK, lw=1, ls="--")
    ax.text(0.98, NU_FD + 0.35, "fully developed: 140/17 = 8.235", ha="right",
            transform=ax.get_yaxis_transform(), fontsize=8.5, color=plots.INK)
    ax.set_xscale("log")
    ax.set_ylim(7, 20)
    ax.set_xlim(8e-4, 0.35)
    ax.set_xlabel("x$^+$ = x / (D$_h$ Re Pr)")
    ax.set_ylabel("Nu$_{Dh}$(x)")
    ax.set_title("Heated run: local Nusselt number, uniform flux on both plates")
    ax.legend(loc="upper right", title=None, bbox_to_anchor=(1, 0.88))
    plots.save(fig, FIG / "stage1_nusselt.png")

    # 3. grid convergence (two panels, one quantity each)
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.3))
    hs = np.array([l["h_mm"] for l in s["levels"]])
    for ax, key, exact, lab in ((axs[0], "Nu_fd", NU_FD, "Nu$_{Dh}$ fully developed"),
                                (axs[1], "fRe", FRE_FD, "f·Re")):
        v = np.array([l[key] for l in s["levels"]])
        g = s["gci"][key]
        ax.plot(hs, v, "o-", color=plots.C1, label="CFD")
        ax.plot(0, g["phi_ext"], "D", color=plots.C2, ms=7,
                label=f"Richardson extrap. (p = {g['p']:.2f})")
        ax.axhline(exact, color=plots.INK, lw=1, ls="--", label="analytical")
        ax.set_xlabel("cell height h [mm]")
        ax.set_title(lab)
        ax.set_xlim(-0.01, hs.max() * 1.1)
        ax.legend(loc="best")
    plots.save(fig, FIG / "stage1_grid_convergence.png")


if __name__ == "__main__":
    main()
