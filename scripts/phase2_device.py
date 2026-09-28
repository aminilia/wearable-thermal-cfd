#!/usr/bin/env python3
"""Phase 2 - 3D wearable temple arm with the wearer's skin (Pennes bioheat).

Steps:
  column   verify the bioheat implementation (Pennes source, contact resistance,
           convection + radiation) on 1D tissue columns against a semi-analytic solution
  mesh     three-mesh GCI study of the nominal 3D design
  cases    design-mesh runs: nominal, idle (Q = 0), the Stage-3 recommendation,
           heater on the outer wall, and contact-resistance sensitivity
  report   tables and figures

Usage:  python scripts/phase2_device.py [column mesh cases report]
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
from coolchan.bioheat import ColumnParams, build_column, column_profile, reference  # noqa: E402
from coolchan.device3d import DeviceParams, TISSUE, analyse_device, build_device  # noqa: E402
from coolchan.foamio import Case  # noqa: E402
from coolchan.runner import run_case  # noqa: E402
from coolchan.verification import gci  # noqa: E402
from coolchan import plots  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "phase2"
RES = REPO / "results" / "phase2"
FIG = REPO / "docs" / "figures"
T_LIMIT = 43.0

NOMINAL = DeviceParams()                 # U = 1.5 m/s, H = 3 mm, Q = 0.5 W, heater on skin side
MESH = {"coarse": 0.75, "medium": 1.0, "fine": 1.33}
CASES = {
    "nominal": NOMINAL,
    "idle": replace(NOMINAL, Q_W=0.0),
    "stage3_recommended": replace(NOMINAL, U_in=1.75, H_mm=1.5),
    "heater_outer_wall": replace(NOMINAL, heater_side="outer"),
    "contact_R0": replace(NOMINAL, R_contact=0.0),      # plain coupling (no thin layer)
    "contact_R5e-3": replace(NOMINAL, R_contact=5e-3),
    "fan_3ms": replace(NOMINAL, U_in=3.0),
}


def run(p, name, builder=build_device) -> Path:
    d = RUNS / name
    pj = d / "params.json"
    log = d / "log.chtMultiRegionSimpleFoam"
    if pj.exists() and log.exists() and json.loads(pj.read_text()) == asdict(p) \
            and "End" in log.read_text()[-400:]:
        return d
    builder(d, p)
    t = run_case(d)
    print(f"  [ok] {name:26s} {t:7.0f} s", flush=True)
    return d


# --------------------------------------------------------------------------- column
def step_column():
    print("P2.1: 1D bioheat verification")
    rows, profiles = [], {}
    for kind, kw in (("bare", {}), ("device", dict(q_top=300.0, R_contact=5e-3))):
        ref = reference(ColumnParams(kind=kind, **kw))
        for r in (1, 2, 4):
            p = ColumnParams(kind=kind, refine=r, nIter=4000 * r, **kw)
            d = run(p, f"column_{kind}_r{r}", build_column)
            y, T, Ttop = column_profile(d, p)
            Tr = np.interp(y, ref["y"], ref["T"])
            rows.append(dict(kind=kind, refine=r, cells=len(y), T_top_cfd_C=Ttop - 273.15,
                             T_top_ref_C=ref["T_top"] - 273.15, err_top_K=Ttop - ref["T_top"],
                             max_abs_err_K=float(np.abs(T - Tr).max()), q_core_ref=ref["q_core"]))
            if r == 1:
                profiles[kind] = (y, T, ref)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "column_verification.csv", index=False, float_format="%.6g")
    print(df.round(5).to_string())
    import matplotlib.pyplot as plt
    plots.setup()
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (kind, (y, T, ref)) in zip(axs, profiles.items()):
        ax.plot(ref["T"] - 273.15, ref["y"] * 1e3, color=plots.INK2, lw=1.4, label="semi-analytic")
        ax.plot(T - 273.15, y * 1e3, "o", ms=3.5, color=plots.C1, mfc="none", label="OpenFOAM")
        yb = -sum(t[1] for t in TISSUE)
        for name, th, *_ in TISSUE:
            yb += th
            ax.axhline(yb, color=plots.GRID, lw=0.8, zorder=0)
        ax.set_xlabel("T [°C]")
        ax.set_title("bare skin, 25 °C still air" if kind == "bare"
                     else "1 mm housing, 300 W/m² in, R$_c$ = 5×10⁻³ m²K/W")
        ax.legend(loc="lower left")
    axs[0].set_ylabel("y [mm]  (0 = skin surface)")
    axs[0].set_ylim(-6, 1.3)
    plots.save(fig, FIG / "phase2_bioheat_verification.png")
    return df


# --------------------------------------------------------------------------- mesh
def step_mesh():
    print("P2.2: 3D mesh study (nominal design)")
    rows = []
    for lvl, r in MESH.items():
        p = replace(NOMINAL, refine=r, nIter=4000 if r > 1.2 else 3000)
        d = run(p, f"mesh_{lvl}")
        rows.append(dict(level=lvl, refine=r, **analyse_device(d, p)))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "mesh_study.csv", index=False, float_format="%.6g")
    h = [1 / r for r in (1.33, 1.0, 0.75)]
    out = {}
    for q in ("T_skin_max_C", "T_heater_max_C", "T_touch_max_C", "Q_to_skin_W", "dp_Pa"):
        f1, f2, f3 = (df.set_index("level").loc[l, q] for l in ("fine", "medium", "coarse"))
        off = 25.0 if q.endswith("_C") else 0.0      # GCI on the rise above ambient
        g = gci(f1 - off, f2 - off, f3 - off, *h)
        out[q] = dict(fine=f1, medium=f2, coarse=f3, p=g.p, extrap=g.phi_ext + off,
                      gci_fine_pct=100 * g.gci_fine, gci_medium_pct=100 * g.gci_medium,
                      oscillatory=g.oscillatory)
    (RES / "mesh_gci.json").write_text(json.dumps(out, indent=2, default=float))
    print(df.round(4).to_string())
    print(json.dumps(out, indent=1, default=float))
    return df


# --------------------------------------------------------------------------- cases
def step_cases():
    print("P2.3: design cases on the medium mesh")
    rows = []
    for name, p in CASES.items():
        d = run(p, f"case_{name}") if name != "nominal" else RUNS / "mesh_medium"
        r = analyse_device(d, p)
        rows.append(dict(case=name, U_in=p.U_in, H_mm=p.H_mm, Q_W=p.Q_W, R_contact=p.R_contact,
                         heater_side=p.heater_side, **r))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "cases.csv", index=False, float_format="%.6g")
    print(df[["case", "T_skin_max_C", "T_touch_max_C", "T_heater_max_C", "Q_to_air_W",
              "Q_to_skin_W", "Q_to_ambient_W", "device_energy_err", "max_residual"]].round(3).to_string())
    return df


# --------------------------------------------------------------------------- report
LABELS = {
    "nominal": "Nominal (1.5 m/s, 3 mm gap)",
    "idle": "Worn, idle (Q = 0)",
    "stage3_recommended": "Stage-3 pick (1.75 m/s, 1.5 mm)",
    "heater_outer_wall": "Heater on outer wall",
    "contact_R0": "Perfect skin contact",
    "contact_R5e-3": "Poor skin contact (5e-3)",
    "fan_3ms": "Fan 3 m/s",
}


def step_report():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plots.setup()
    cases = pd.read_csv(RES / "cases.csv")
    g = json.loads((RES / "mesh_gci.json").read_text())
    col = pd.read_csv(RES / "column_verification.csv")
    s2 = json.loads((REPO / "results" / "stage2_summary.json").read_text())

    # ---- tables
    TB = ["| Case | Skin max [°C] | Skin mean under device [°C] | Housing touch max [°C] | Heater max [°C] | "
         "Heat to air / skin / ambient [W] | Fan [mW] |", "|---|---|---|---|---|---|---|"]
    for _, r in cases.iterrows():
        flag = " ⚠" if r.T_skin_max_C > T_LIMIT else ""
        TB.append(f"| {LABELS[r.case]} | **{r.T_skin_max_C:.1f}**{flag} | {r.T_skin_mean_C:.1f} | {r.T_touch_max_C:.1f} | "
                 f"{r.T_heater_max_C:.1f} | {r.Q_to_air_W:.3f} / {r.Q_to_skin_W:.3f} / {r.Q_to_ambient_W:.3f} | "
                 f"{r.fan_mW:.2f} |")
    (RES / "cases_table.md").write_text("\n".join(TB) + "\n")
    M = ["| Output | coarse | medium | fine | Observed p | Richardson extrap. | GCI<sub>medium</sub> |",
         "|---|---|---|---|---|---|---|"]
    names = {"T_skin_max_C": "Skin max [°C]", "T_touch_max_C": "Housing touch max [°C]",
             "T_heater_max_C": "Heater max [°C]", "Q_to_skin_W": "Heat into skin [W]", "dp_Pa": "Δp [Pa]"}
    for q, G in g.items():
        M.append(f"| {names[q]} | {G['coarse']:.3f} | {G['medium']:.3f} | {G['fine']:.3f} | {G['p']:.2f} | "
                 f"{G['extrap']:.3f} | {G['gci_medium_pct']:.2f} % |")
    (RES / "mesh_table.md").write_text("\n".join(M) + "\n")
    C = ["| Column | Cells | T<sub>top</sub> OpenFOAM [°C] | T<sub>top</sub> semi-analytic [°C] | Max \\|error\\| [K] |",
         "|---|---|---|---|---|"]
    for _, r in col.iterrows():
        C.append(f"| {r.kind} | {r.cells} | {r.T_top_cfd_C:.4f} | {r.T_top_ref_C:.4f} | {r.max_abs_err_K:.1e} |")
    (RES / "column_table.md").write_text("\n".join(C) + "\n")

    # ---- skin temperature map (nominal, mirrored to full width)
    p = NOMINAL
    fig, axs = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    for ax, name in zip(axs, ("nominal", "heater_outer_wall")):
        d = RUNS / ("mesh_medium" if name == "nominal" else f"case_{name}")
        c = Case(d)
        m = c.mesh("epidermis")
        T = c.field("T", "epidermis")
        cells = m.patch_cells("skinExposed")
        pts, vals = [], []
        for patch in ("epidermis_to_housing", "skinExposed"):
            Cf = m.patch_centres(patch)
            pts.append(Cf[:, [0, 2]] * 1e3)
            vals.append(T.boundary[patch] - 273.15)
        P, V = np.concatenate(pts), np.concatenate(vals)
        P = np.vstack([P, P * [1, -1]])
        V = np.concatenate([V, V])
        tri = ax.tricontourf(P[:, 0], P[:, 1], V, levels=np.linspace(30, 50, 41), cmap=plots.DIV,
                             norm=__import__("matplotlib.colors", fromlist=["TwoSlopeNorm"]).TwoSlopeNorm(T_LIMIT, 30, 50),
                             extend="both")
        cs = ax.tricontour(P[:, 0], P[:, 1], V, levels=[T_LIMIT], colors=plots.INK, linewidths=1.6)
        hw = p.Wc_mm / 2 + p.t_mm
        ax.add_patch(Rectangle((0, -hw), p.L_mm, 2 * hw, fill=False, ec=plots.INK2, lw=0.8, ls="--"))
        ax.add_patch(Rectangle((p.Lup_mm, -p.Wb_mm / 2), p.Lb_mm, p.Wb_mm, fill=False, ec=plots.INK, lw=1))
        r = cases.set_index("case").loc[name]
        ax.set_title(f"{LABELS[name]} — peak skin {r.T_skin_max_C:.1f} °C", fontsize=9.5)
        ax.set_ylabel("z [mm]")
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlim(-p.margin_mm, p.L_mm + p.margin_mm)
    axs[1].set_xlabel("x [mm]  (airflow →)   ·   black line: 43 °C   ·   dashed: housing footprint   ·   box: heater")
    cb = fig.colorbar(tri, ax=axs, pad=0.01, shrink=0.9, ticks=[30, 35, 40, 43, 45, 50])
    cb.set_label("skin surface T [°C]")
    cb.outline.set_visible(False)
    plots.save(fig, FIG / "phase2_skin_map.png")

    # ---- heat budget + skin temperature by case
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw=dict(width_ratios=[1.1, 1]))
    sub = cases[cases.Q_W > 0].reset_index(drop=True)
    y = np.arange(len(sub))[::-1]
    ax = axs[0]
    pos, neg = np.zeros(len(sub)), np.zeros(len(sub))
    for col_, c_, lab in (("Q_to_air_W", plots.C1, "to air (fan)"), ("Q_to_skin_W", plots.C2, "into skin"),
                          ("Q_to_ambient_W", plots.C3, "housing → room")):
        v = sub[col_].to_numpy()
        base = np.where(v >= 0, pos, neg)
        ax.barh(y, v, left=base, color=c_, height=0.62, label=lab, edgecolor=plots.SURFACE, linewidth=1.5)
        pos += np.clip(v, 0, None)
        neg += np.clip(v, None, 0)
    ax.axvline(0, color=plots.INK2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[c] for c in sub.case], fontsize=8.5)
    ax.set_xlabel("heat flow [W]")
    ax.set_title("(a) Where the 0.5 W goes  (left of 0: skin heats the device)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    ax = axs[1]
    ax.barh(y, sub.T_skin_max_C, color=[plots.C2 if t > T_LIMIT else plots.C1 for t in sub.T_skin_max_C],
            height=0.62)
    idle = cases.set_index("case").loc["idle", "T_skin_max_C"]
    ax.axvline(T_LIMIT, color=plots.INK, lw=1.4)
    ax.axvline(idle, color=plots.MUTED, lw=1, ls="--")
    ax.text(T_LIMIT + 0.2, y.max() + 0.35, "43 °C", fontsize=8.5)
    ax.text(idle + 0.2, y.min() - 0.5, f"worn idle {idle:.1f} °C", fontsize=8, color=plots.INK2)
    for yy, t in zip(y, sub.T_skin_max_C):
        ax.text(t + 0.2, yy, f"{t:.1f}", va="center", fontsize=8.5, color=plots.INK)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(30, max(52, sub.T_skin_max_C.max() + 2))
    ax.set_xlabel("peak skin temperature [°C]")
    ax.set_title("(b) Peak skin temperature")
    plots.save(fig, FIG / "phase2_cases.png")
    print("\n".join(C + [""] + M + [""] + TB))
    # 2D vs 3D heater comparison for the README
    comp = dict(stage2_2d_heater_C=s2["meshes"]["medium"]["T_max_C"],
                phase2_3d_heater_C=float(cases.set_index("case").loc["nominal", "T_heater_max_C"]))
    (RES / "compare_2d_3d.json").write_text(json.dumps(comp, indent=2))
    print(comp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["column", "mesh", "cases", "report"])
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for s in a.steps:
        {"column": step_column, "mesh": step_mesh, "cases": step_cases, "report": step_report}[s]()


if __name__ == "__main__":
    main()
