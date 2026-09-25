#!/usr/bin/env python3
"""Stage 2 - conjugate heat transfer: heated aluminium block in a thin air channel.

Runs the nominal design on three meshes (r = 2) and reports
* peak solid temperature, mixed-mean outlet air temperature, pressure drop, fan power
* first-law energy balance (heat generated in the block vs. enthalpy rise of the air)
* three-grid GCI on each output -> picks the mesh used for the design study

Usage:  python scripts/stage2_cht.py [--skip-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coolchan.case import ChtParams, build_cht  # noqa: E402
from coolchan.cht import analyse_cht  # noqa: E402
from coolchan.foamio import Case  # noqa: E402
from coolchan.runner import run_many  # noqa: E402
from coolchan.verification import gci  # noqa: E402
from coolchan import plots  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "stage2"
RES = REPO / "results"
FIG = REPO / "docs" / "figures"

NOMINAL = ChtParams()           # U=1.5 m/s, H=3 mm, Q=0.5 W, k=200 W/m/K
MESHES = {"coarse": (0.10, 1500), "medium": (0.05, 2500), "fine": (0.025, 5000)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-run", action="store_true")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    cases = {k: replace(NOMINAL, dy_mm=dy, nIter=it) for k, (dy, it) in MESHES.items()}
    if not a.skip_run:
        for k, p in cases.items():
            build_cht(RUNS / k, p)
        out = run_many([RUNS / k for k in ("fine", "medium", "coarse")])
        if any(isinstance(v, Exception) for v in out.values()):
            sys.exit("a Stage-2 run failed")

    rows = {}
    for k, p in cases.items():
        r = analyse_cht(RUNS / k, p)
        rows[k] = dict(dy_mm=p.dy_mm, **{kk: float(v) for kk, v in r.as_dict().items()})
        print(k, rows[k])

    # linearity in Q: with constant properties T - T_in must scale exactly with Q
    lin = replace(cases["medium"], Q_W=2 * NOMINAL.Q_W)
    if not (RUNS / "linearity_2Q" / "0").exists() or not a.skip_run:
        build_cht(RUNS / "linearity_2Q", lin)
        run_many([RUNS / "linearity_2Q"])
    r2 = analyse_cht(RUNS / "linearity_2Q", lin)
    linearity = dict(Q_W=[NOMINAL.Q_W, lin.Q_W], R_KW=[rows["medium"]["R_KW"], float(r2.R_KW)],
                     rel_diff=float(r2.R_KW / rows["medium"]["R_KW"] - 1))
    print("linearity:", linearity)

    g = {}
    for q in ("T_max_C", "T_out_C", "dp_Pa"):
        f1, f2, f3 = rows["fine"][q], rows["medium"][q], rows["coarse"][q]
        # for temperatures use the rise above inlet so % errors are meaningful
        off = NOMINAL.T_in - 273.15 if q.startswith("T_") else 0.0
        G = gci(f1 - off, f2 - off, f3 - off, 0.025, 0.05, 0.10)
        g[q] = dict(fine=f1, medium=f2, coarse=f3, p=G.p, extrap=G.phi_ext + off,
                    gci_fine_pct=100 * G.gci_fine, gci_medium_pct=100 * G.gci_medium,
                    oscillatory=G.oscillatory, basis="rise above T_in" if off else "absolute")
    nom = NOMINAL
    summary = dict(
        nominal=dict(U_in=nom.U_in, H_mm=nom.H_mm, Q_W=nom.Q_W, k_solid=nom.k_solid,
                     T_in_C=nom.T_in - 273.15, Re_Dh=nom.Re, W_mm=nom.W_mm,
                     block_mm=[nom.Lb_mm, nom.hb_mm]),
        meshes=rows, gci=g, linearity_in_Q=linearity,
        design_mesh="medium",
    )
    (RES / "stage2_summary.json").write_text(json.dumps(summary, indent=2))
    write_table(summary)
    plot_field(RUNS / "medium", cases["medium"])
    print(json.dumps(g, indent=2))


def write_table(s):
    m = s["meshes"]
    lines = [
        "| Mesh | Cells | T<sub>max</sub> solid [°C] | T<sub>out</sub> air [°C] | Δp [Pa] | Fan power [mW] | Energy imbalance | Iterative ΔT<sub>max</sub> [K] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k in ("coarse", "medium", "fine"):
        r = m[k]
        lines.append(f"| {k} (Δy = {r['dy_mm']} mm) | {int(r['n_cells']):,} | {r['T_max_C']:.3f} | "
                     f"{r['T_out_C']:.3f} | {r['dp_Pa']:.3f} | {1e3 * r['fan_W']:.3f} | "
                     f"{100 * r['energy_err']:+.3f} % | {r['dTmax_iter']:.1e} |")
    lines += ["", "| Output | Observed order p | Richardson extrapolation | GCI<sub>fine</sub> | GCI<sub>medium</sub> |",
              "|---|---|---|---|---|"]
    names = {"T_max_C": "T<sub>max</sub> solid [°C]", "T_out_C": "T<sub>out</sub> air [°C]", "dp_Pa": "Δp [Pa]"}
    for q, G in s["gci"].items():
        lines.append(f"| {names[q]} | {G['p']:.2f} | {G['extrap']:.3f} | {G['gci_fine_pct']:.2f} % | "
                     f"{G['gci_medium_pct']:.2f} % |")
    lines.append("")
    lines.append("GCI for temperatures is computed on the rise above the inlet temperature. "
                 "T<sub>out</sub> is fixed by the energy balance (Q = ṁ c<sub>p</sub> ΔT), so its "
                 "mesh differences are at round-off level and its 'observed order' is not meaningful.")
    (RES / "stage2_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_field(case_dir, p: ChtParams):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plots.setup()
    c = Case(case_dir)
    xs, ys, Ts = [], [], []
    for reg in ("air", "heater"):
        m = c.mesh(reg)
        xs.append(m.C[:, 0] * 1e3)
        ys.append(m.C[:, 1] * 1e3)
        Ts.append(c.field("T", reg).internal - 273.15)
    x, y, T = map(np.concatenate, (xs, ys, Ts))
    # structured grid -> image
    xu, yu = np.unique(np.round(x, 6)), np.unique(np.round(y, 6))
    img = np.full((len(yu), len(xu)), np.nan)
    img[np.searchsorted(yu, np.round(y, 6)), np.searchsorted(xu, np.round(x, 6))] = T

    fig, ax = plt.subplots(figsize=(9, 2.6))
    im = ax.imshow(img, origin="lower", aspect="auto", cmap=plots.SEQ,
                   extent=[0, xu.max() + (xu[1] - xu[0]) / 2, 0, p.H_mm], interpolation="bilinear")
    ax.add_patch(Rectangle((p.Lup_mm, 0), p.Lb_mm, p.hb_mm, fill=False, ec=plots.INK, lw=1.2))
    cs = ax.contour(xu, yu, img, levels=[30, 35, 40, 45], colors=plots.INK, linewidths=0.6)
    ax.clabel(cs, fmt="%d °C", fontsize=7.5, inline_spacing=2)
    ax.text(p.Lup_mm + p.Lb_mm / 2, p.hb_mm / 2, "Al block", ha="center", va="center",
            fontsize=8.5, color="white")
    ax.set_xlabel("x [mm]  (flow →)")
    ax.set_ylabel("y [mm]\n(vertical scale ×10)")
    ax.grid(False)
    ax.set_title(f"Temperature, nominal design (U = {p.U_in} m/s, gap = {p.H_mm} mm, "
                 f"Q = {p.Q_W} W)")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("T [°C]")
    cb.outline.set_visible(False)
    plots.save(fig, FIG / "stage2_temperature_field.png")


if __name__ == "__main__":
    main()
