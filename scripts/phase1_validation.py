#!/usr/bin/env python3
"""Phase 1 - validation of the separated-flow physics that governs the heated block.

The flow behind the block in the design model is a backward-facing-step flow
(recirculation over ~8 mm behind a 1 mm step). This script validates that flow
physics in three steps:

  V1  gartling  Code-to-code benchmark: 2D BFS, ER = 2, Re = 800 (Gartling 1990),
                three meshes, GCI on x1, x4, x5.
  V2  armaly2d  Validation against experiment: 2D BFS, ER = 1.942, the seven
                Reynolds numbers tabulated from Armaly et al. (1983) + a mesh study,
                ASME V&V 20 comparison error E and validation uncertainty u_val.
  V3  armaly3d  Model-form attribution: 3D half-span runs with the real sidewalls
                (aspect ratio 18) at a low and a high Re, compared with 2D on the
                same mesh, to test whether the 2D discrepancy is the 2D assumption.

Usage:  python scripts/phase1_validation.py [gartling armaly2d armaly3d report]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coolchan.bfs import (BFSParams, build_bfs, reattachment, reattachment_iter_change,  # noqa: E402
                          reattachment_span, write_inlet_velocity)
from coolchan.foamio import Case  # noqa: E402
from coolchan.runner import run_case  # noqa: E402
from coolchan.verification import gci  # noqa: E402
from coolchan import plots  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "phase1"
DATA = REPO / "validation" / "data"
RES = REPO / "results" / "phase1"
FIG = REPO / "docs" / "figures"

ARMALY = pd.read_csv(DATA / "armaly1983_x1.csv", comment="#")
GARTLING = json.loads((DATA / "gartling1990.json").read_text())
S_A, H_A = 4.9, 5.2                      # Armaly step / inlet heights [mm]
RE_FAC = 2 * H_A / S_A                   # Re_2h = RE_FAC * Re_S

# Uncertainty assumptions (standard uncertainties, 1 sigma) - see validation/README.md
U_DATA_S = 0.20        # experimental + digitisation, in step heights
U_RE_REL = 0.02        # relative uncertainty of the experimental Reynolds number


def run(p: BFSParams, name: str) -> Path:
    """Build + mesh + solve unless an identical converged run exists."""
    d = RUNS / name
    pj = d / "params.json"
    log = d / "log.simpleFoam"
    if pj.exists() and log.exists() and json.loads(pj.read_text()) == asdict(p) \
            and "End" in log.read_text()[-300:]:
        return d
    build_bfs(d, p)
    run_case(d, "Allmesh")
    write_inlet_velocity(d, p)
    t = run_case(d, "Allsolve")
    print(f"  [ok] {name:28s} {p.n_cells():>8,d} cells  {t:6.0f} s", flush=True)
    return d


def result(d: Path, p: BFSParams) -> dict:
    r = reattachment(d, p)
    r["dx1_iter"] = reattachment_iter_change(d, p)
    r["n_cells"] = p.n_cells()
    r["iterations"] = int(float(Case(d).latest()))
    return r


# --------------------------------------------------------------------------- V1
def step_gartling():
    print("V1: Gartling benchmark (Re = 800, ER = 2)")
    levels = [20, 40, 80]
    rows = []
    for n in levels:
        p = BFSParams(Re=800, S_mm=5.0, h_mm=5.0, Lin_S=0, Ld_S=60, n_S=n,
                      nIter=20000, nProcs=2 if n >= 80 else 1)
        rows.append(dict(n_S=n, h_S=1 / n, **result(run(p, f"gartling_n{n}"), p)))
    df = pd.DataFrame(rows)
    out = {"levels": rows, "reference": GARTLING, "gci": {}}
    for k in ("x1", "x4", "x5"):
        f1, f2, f3 = df[k].iloc[::-1]
        g = gci(f1, f2, f3, 1 / 80, 1 / 40, 1 / 20)
        ref = GARTLING[f"{k}_over_S"]
        out["gci"][k] = dict(fine=f1, p=g.p, extrap=g.phi_ext, gci_fine_pct=100 * g.gci_fine,
                             ref=ref, fine_vs_ref_pct=100 * (f1 / ref - 1),
                             extrap_vs_ref_pct=100 * (g.phi_ext / ref - 1))
    (RES / "gartling.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out["gci"], indent=2))
    return out


# --------------------------------------------------------------------------- V2
def armaly_params(Re_S, n_S=40, **kw):
    base = dict(Re=float(Re_S) * RE_FAC, S_mm=S_A, h_mm=H_A, Lin_S=5, Ld_S=40, n_S=n_S,
                nIter=20000)
    base.update(kw)
    return BFSParams(**base)


def step_armaly2d():
    print("V2: Armaly et al. 2D sweep")
    # mesh study at the highest Reynolds number (hardest case)
    ReS_max = float(ARMALY.Re_S.max())
    ms = []
    for n in (20, 40, 80):
        p = armaly_params(ReS_max, n, nProcs=2 if n >= 80 else 1)
        ms.append(dict(n_S=n, **result(run(p, f"armaly2d_ReS{ReS_max:.0f}_n{n}"), p)))
    f1, f2, f3 = [m["x1"] for m in ms][::-1]
    g = gci(f1, f2, f3, 1 / 80, 1 / 40, 1 / 20)
    # production level n_S = 40: its GCI is GCI_medium from the same study
    gci_rel = g.gci_medium
    rows = []
    for ReS, x1e in zip(ARMALY.Re_S, ARMALY.x1_over_S):
        p = armaly_params(ReS, 40)
        r = result(run(p, f"armaly2d_ReS{ReS:.0f}_n40"), p)
        rows.append(dict(Re_S=ReS, Re_2h=ReS * RE_FAC, x1_exp=x1e, **r))
    df = pd.DataFrame(rows).sort_values("Re_S").reset_index(drop=True)
    # ASME V&V 20 validation metrics
    slope = np.gradient(df.x1, df.Re_S)                      # d x1 / d Re_S (from the model)
    df["E"] = df.x1 - df.x1_exp
    df["u_num"] = gci_rel * df.x1 / 2                         # GCI (95 %) -> standard unc.
    df["u_input"] = np.abs(slope) * U_RE_REL * df.Re_S
    df["u_D"] = U_DATA_S
    df["u_val"] = np.sqrt(df.u_num**2 + df.u_input**2 + df.u_D**2)
    df["model_error_detected"] = np.abs(df.E) > 2 * df.u_val
    df.to_csv(RES / "armaly2d.csv", index=False, float_format="%.5g")
    out = dict(mesh_study=ms, gci=dict(p=g.p, extrap=g.phi_ext, gci_fine_pct=100 * g.gci_fine,
                                       gci_medium_pct=100 * g.gci_medium),
               assumptions=dict(u_data_S=U_DATA_S, u_Re_rel=U_RE_REL))
    (RES / "armaly2d_mesh.json").write_text(json.dumps(out, indent=2, default=float))
    print(df[["Re_2h", "x1_exp", "x1", "E", "u_val", "model_error_detected"]].round(3))
    return df


# --------------------------------------------------------------------------- V3
W_HALF = 18 * (S_A + H_A) / 2            # Armaly: downstream aspect ratio 18


LEVELS_3D = [(12, 36), (16, 48), (20, 60)]      # (cells per S, spanwise cells), r = 1.25


def step_armaly3d(ReS_list=(141.49, 297.87)):
    print("V3: Armaly et al. 3D half-span (sidewalls) vs 2D on the same mesh")
    rows = []
    for ReS in ReS_list:
        for n, nz in LEVELS_3D:
            base = dict(n_S=n, aspect=2.5, Ld_S=30)
            p3 = replace(armaly_params(ReS, **base), W_half_mm=W_HALF, nz=nz, z_grading=0.08,
                         nProcs=2, nIter=8000)
            p2 = armaly_params(ReS, **base)
            name = f"armaly3d_ReS{ReS:.0f}_n{n}"
            r3 = result(run(p3, name), p3)
            r2 = result(run(p2, f"armaly3d_2dtwin_ReS{ReS:.0f}_n{n}"), p2)
            zs, x1z = reattachment_span(RUNS / name, p3)
            np.savetxt(RES / f"armaly3d_span_ReS{ReS:.0f}_n{n}.csv", np.column_stack([zs, x1z]),
                       delimiter=",", header="z_over_S,x1_over_S", comments="")
            rows.append(dict(Re_S=ReS, Re_2h=ReS * RE_FAC, n_S=n, nz=nz, x1_3d=r3["x1"],
                             x1_2d_same_mesh=r2["x1"], delta_3d=r3["x1"] - r2["x1"],
                             resid_3d=r3.get("max_residual"), dx1_iter_3d=r3["dx1_iter"],
                             iterations_3d=r3["iterations"], cells_3d=p3.n_cells()))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "armaly3d_levels.csv", index=False, float_format="%.5g")
    # GCI on the 3D centre-plane x1 and on the 3D-2D increment, per Reynolds number
    summ = []
    for ReS, g in df.groupby("Re_S"):
        g = g.sort_values("n_S", ascending=False)       # fine, medium, coarse
        hs = [1 / n for n in g.n_S]
        G3 = gci(*g.x1_3d, *hs)
        Gd = gci(*g.delta_3d, *hs)
        summ.append(dict(Re_S=ReS, Re_2h=ReS * RE_FAC, x1_3d_fine=g.x1_3d.iloc[0],
                         x1_3d_extrap=G3.phi_ext, p_3d=G3.p, gci_3d_pct=100 * G3.gci_fine,
                         delta_fine=g.delta_3d.iloc[0], delta_extrap=Gd.phi_ext, p_delta=Gd.p,
                         gci_delta_pct=100 * Gd.gci_fine, delta_oscillatory=Gd.oscillatory,
                         cells_fine=int(g.cells_3d.iloc[0])))
    sm = pd.DataFrame(summ)
    sm.to_csv(RES / "armaly3d.csv", index=False, float_format="%.5g")
    print(df.round(3).to_string())
    print(sm.round(3).to_string())
    return sm


# --------------------------------------------------------------------------- report
def step_report():
    import matplotlib.pyplot as plt
    plots.setup()
    g = json.loads((RES / "gartling.json").read_text())
    a2 = pd.read_csv(RES / "armaly2d.csv")
    a3 = pd.read_csv(RES / "armaly3d.csv") if (RES / "armaly3d.csv").exists() else None
    am = json.loads((RES / "armaly2d_mesh.json").read_text())

    # ---- tables
    L = ["| Quantity | 20 cells/S | 40 cells/S | 80 cells/S | Observed p | Richardson extrap. | GCI<sub>fine</sub> | Gartling (1990) | Extrap. vs Gartling |",
         "|---|---|---|---|---|---|---|---|---|"]
    lv = g["levels"]
    for k, lab in (("x1", "x<sub>1</sub>/S lower-wall reattachment"),
                   ("x4", "x<sub>4</sub>/S upper-wall separation"),
                   ("x5", "x<sub>5</sub>/S upper-wall reattachment")):
        G = g["gci"][k]
        L.append(f"| {lab} | {lv[0][k]:.3f} | {lv[1][k]:.3f} | {lv[2][k]:.3f} | {G['p']:.2f} | "
                 f"{G['extrap']:.3f} | {G['gci_fine_pct']:.2f} % | {G['ref']:.1f} | {G['extrap_vs_ref_pct']:+.2f} % |")
    (RES / "gartling_table.md").write_text("\n".join(L) + "\n")

    T = ["| Re (U<sub>m</sub>·2h/ν) | Experiment x<sub>1</sub>/S | 2D CFD x<sub>1</sub>/S | E = S − D | u<sub>num</sub> | u<sub>input</sub> | u<sub>D</sub> | u<sub>val</sub> | |E| > 2u<sub>val</sub>? |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in a2.iterrows():
        T.append(f"| {r.Re_2h:.0f} | {r.x1_exp:.2f} | {r.x1:.2f} | {r.E:+.2f} | {r.u_num:.2f} | "
                 f"{r.u_input:.2f} | {r.u_D:.2f} | {r.u_val:.2f} | {'**yes**' if r.model_error_detected else 'no'} |")
    (RES / "armaly2d_table.md").write_text("\n".join(T) + "\n")
    if a3 is not None:
        T3 = ["| Re | Experiment x<sub>1</sub>/S | 2D (40 cells/S) | 3D with sidewalls, centre-plane (finest) | 3D Richardson extrap. | GCI (3D) | 3D − 2D increment | E (3D) |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in a3.iterrows():
            e = np.interp(r.Re_S, a2.Re_S, a2.x1_exp)
            f = np.interp(r.Re_S, a2.Re_S, a2.x1)
            T3.append(f"| {r.Re_2h:.0f} | {e:.2f} | {f:.2f} | {r.x1_3d_fine:.2f} | {r.x1_3d_extrap:.2f} | "
                      f"{r.gci_3d_pct:.1f} % | {r.delta_extrap:+.2f} | {r.x1_3d_extrap - e:+.2f} |")
        (RES / "armaly3d_table.md").write_text("\n".join(T3) + "\n")

    # ---- figure: x1 vs Re + comparison error
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    ax.errorbar(a2.Re_2h, a2.x1_exp, yerr=2 * a2.u_D, fmt="o", color=plots.INK, mfc="white",
                mew=1.3, ms=6, capsize=3, label="Armaly et al. (1983), experiment (±2u$_D$)")
    ax.plot(a2.Re_2h, a2.x1, "-", color=plots.C1, label="2D CFD (40 cells/S)")
    ax.fill_between(a2.Re_2h, a2.x1 - 2 * a2.u_num, a2.x1 + 2 * a2.u_num, color=plots.C1,
                    alpha=0.15, lw=0)
    if a3 is not None:
        ax.errorbar(a3.Re_2h, a3.x1_3d_extrap, yerr=a3.gci_3d_pct / 100 * a3.x1_3d_extrap,
                    fmt="D", color=plots.C2, ms=7, capsize=3,
                    label="3D with sidewalls, centre-plane (extrap. ± GCI)")
    ax.axvspan(400, a2.Re_2h.max() * 1.05, color=plots.GRID, alpha=0.6, lw=0)
    ax.text(410, 2.8, "Armaly et al.: flow\nbecomes 3D (Re > 400)", fontsize=8, color=plots.INK2)
    ax.set_xlabel("Re = U$_m$·2h/ν")
    ax.set_ylabel("x$_1$/S  (lower-wall reattachment)")
    ax.set_title("(a) Reattachment length: model vs experiment")
    ax.set_xlim(0, a2.Re_2h.max() * 1.05)
    ax.legend(loc="upper left", fontsize=7.8)

    ax = axs[1]
    ax.fill_between(a2.Re_2h, -2 * a2.u_val, 2 * a2.u_val, color=plots.GRID, alpha=0.9, lw=0,
                    label="±2u$_{val}$ (V&V 20)")
    ax.axhline(0, color=plots.MUTED, lw=0.8)
    ax.plot(a2.Re_2h, a2.E, "o-", color=plots.C1, label="E = x$_1$(CFD) − x$_1$(exp), 2D")
    if a3 is not None:
        e3 = a3.x1_3d_extrap - np.interp(a3.Re_S, a2.Re_S, a2.x1_exp)
        ax.plot(a3.Re_2h, e3, "D", color=plots.C2, ms=7, label="E, 3D with sidewalls")
    ax.set_xlabel("Re = U$_m$·2h/ν")
    ax.set_ylabel("comparison error E [step heights]")
    ax.set_title("(b) Comparison error and validation uncertainty")
    ax.legend(loc="lower left", fontsize=7.8)
    plots.save(fig, FIG / "phase1_armaly_validation.png")

    # ---- figure: Gartling velocity field with bubbles
    d = RUNS / "gartling_n40"
    p = BFSParams(**json.loads((d / "params.json").read_text()))
    c = Case(d)
    m = c.mesh("")
    U = c.field("U", "").internal[:, 0]
    x, y = m.C[:, 0] / (p.S_mm * 1e-3), m.C[:, 1] / (p.S_mm * 1e-3)
    xu, yu = np.unique(np.round(x, 6)), np.unique(np.round(y, 6))
    img = np.full((len(yu), len(xu)), np.nan)
    img[np.searchsorted(yu, np.round(y, 6)), np.searchsorted(xu, np.round(x, 6))] = U / p.U_mean
    fig, ax = plt.subplots(figsize=(10, 2.4))
    im = ax.imshow(img, origin="lower", aspect="auto", extent=[0, xu.max(), 0, 2],
                   cmap=plots.DIV, vmin=-1.5, vmax=1.5, interpolation="bilinear")
    ax.contour(xu, yu, img, levels=[0], colors=plots.INK, linewidths=1.2)
    G = g["gci"]
    for k, yy in (("x1", 0.0), ("x4", 2.0), ("x5", 2.0)):
        ax.plot(G[k]["ref"], yy, "v" if yy else "^", color=plots.C4, mec=plots.INK, ms=9, clip_on=False)
    ax.set_xlim(0, 30)
    ax.set_xlabel("x / S")
    ax.set_ylabel("y / S")
    ax.set_title("V1: Gartling benchmark, Re = 800 — u/U$_m$ with u = 0 contour; "
                 "triangles = Gartling (1990) x$_1$, x$_4$, x$_5$")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.outline.set_visible(False)
    plots.save(fig, FIG / "phase1_gartling_field.png")

    # ---- figure: spanwise reattachment line (3D)
    if a3 is not None:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        for (ReS, col) in zip(a3.Re_S, (plots.C3, plots.C2)):
            sp = pd.read_csv(RES / f"armaly3d_span_ReS{ReS:.0f}_n{LEVELS_3D[-1][0]}.csv")
            ax.plot(sp.z_over_S, sp.x1_over_S, color=col, label=f"Re = {ReS * RE_FAC:.0f}")
        ax.axvline(W_HALF / S_A, color=plots.INK, lw=1.2)
        ax.text(W_HALF / S_A - 0.3, ax.get_ylim()[0] + 0.3, "sidewall", ha="right", fontsize=8.5)
        ax.set_xlabel("spanwise position z/S (0 = channel centre-plane)")
        ax.set_ylabel("x$_1$/S")
        ax.set_title("V3: reattachment line across the span (3D, half-span)")
        ax.legend(loc="best")
        plots.save(fig, FIG / "phase1_armaly3d_span.png")
    print("\n".join(L))
    print("\n".join(T))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["gartling", "armaly2d", "armaly3d", "report"])
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for s in a.steps:
        {"gartling": step_gartling, "armaly2d": step_armaly2d,
         "armaly3d": step_armaly3d, "report": step_report}[s]()


if __name__ == "__main__":
    main()
