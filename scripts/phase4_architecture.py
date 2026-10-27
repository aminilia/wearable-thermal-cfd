#!/usr/bin/env python3
"""Phase 4 - redesign the thermal architecture of the temple arm.

Figure of merit: the largest heater power Q_max the arm can dissipate
continuously while ALL of
    * the skin stays at or below 43 C   (comfort, long-duration contact),
    * the outer housing stays at or below 48 C   (IEC 62368-1, 1 min - 8 h touch),
    * the heater (battery / SoC stand-in) stays at or below 60 C.
Q_max comes from two CFD runs per design (idle and 0.5 W) by face-wise
superposition, and is checked by a third run at Q_max for the final designs.

Steps:
  screen   six architectures on the design mesh
  opt      multi-objective Bayesian optimisation (ParEGO) of the best architecture:
           maximise Q_max, minimise fan power and spreader mass
  verify   re-run the chosen optima at Q = Q_max on the design mesh
  report   tables and figures

Usage:  python scripts/phase4_architecture.py [screen opt verify report]
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
from coolchan.device3d import DeviceParams, analyse_device, build_device, max_power  # noqa: E402
from coolchan.runner import run_case  # noqa: E402
from coolchan import plots  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "phase4"
P2RUNS = REPO / "runs" / "phase2"
RES = REPO / "results" / "phase4"
FIG = REPO / "docs" / "figures"
# skin: long-duration comfort; touch: IEC 62368-1 1 min-8 h contact; component: typical Li-ion
# cell discharge limit (60 C in cell datasheets) - the heater stands in for a battery/SoC
LIMITS = {"skin": 43.0, "touch": 48.0, "component": 60.0}
Q_REF = 0.5

BASE = DeviceParams()                                   # skin-side heater, 1.5 m/s, 3 mm gap
ARCH = {
    "A0": ("Heater on skin-side wall (Phase 2 baseline)", BASE),
    "A1": ("Skin-side + 0.5 mm insulation pad", replace(BASE, pad_mm=0.5)),
    "A2": ("Board-mounted mid-channel", replace(BASE, heater_side="centre")),
    "A3": ("Heater on outer wall", replace(BASE, heater_side="outer")),
    "A4": ("Outer wall + 0.3 mm Al spreader", replace(BASE, heater_side="outer", spreader_mm=0.3)),
    "A5": ("A4 with the fan off", replace(BASE, heater_side="outer", spreader_mm=0.3, U_in=0.0)),
}


# --------------------------------------------------------------------------- runs
def _find_existing(p: DeviceParams):
    for pj in list(P2RUNS.glob("*/params.json")) + list(RUNS.glob("*/params.json")):
        try:
            old = json.loads(pj.read_text())
            if not old.get("spreader_mm"):
                old.pop("k_spreader", None)          # field removed after the screening runs
            if old == asdict(p):
                log = pj.parent / "log.chtMultiRegionSimpleFoam"
                if log.exists() and "End" in log.read_text()[-400:]:
                    return pj.parent
        except Exception:
            pass
    return None


def run(p: DeviceParams, name: str) -> Path:
    d = _find_existing(p)
    if d is not None:
        return d
    d = RUNS / name
    build_device(d, p)
    t = run_case(d)
    print(f"  [ok] {name:34s} {t:6.0f} s", flush=True)
    return d


def evaluate(p: DeviceParams, name: str) -> dict:
    """Idle + loaded runs -> Q_max and the loaded-case outputs."""
    d0 = run(replace(p, Q_W=0.0), f"{name}_idle")
    d1 = run(replace(p, Q_W=Q_REF), f"{name}_q{Q_REF:g}")
    qm = max_power(d0, d1, Q_REF, LIMITS)
    r = analyse_device(d1, replace(p, Q_W=Q_REF))
    return dict(Q_max_W=qm["Q_max"], Q_max_skin_W=qm["skin"], Q_max_touch_W=qm["touch"],
                Q_max_component_W=qm["component"],
                binding=qm["binding"], fan_mW=r["fan_mW"], spreader_mass_g=r["spreader_mass_g"],
                T_skin_max_C=r["T_skin_max_C"], T_touch_outer_max_C=r["T_housing_outer_max_C"],
                T_heater_max_C=r["T_heater_max_C"], Q_to_air_W=r["Q_to_air_W"],
                Q_to_skin_W=r["Q_to_skin_W"], Q_to_ambient_W=r["Q_to_ambient_W"],
                energy_err=r["device_energy_err"], max_residual=r["max_residual"],
                dir_idle=str(d0), dir_loaded=str(d1))


# --------------------------------------------------------------------------- screen
def step_screen():
    print("P4.1: architecture screening (design mesh)")
    rows = []
    for key, (label, p) in ARCH.items():
        rows.append(dict(arch=key, label=label, **evaluate(p, f"screen_{key}")))
        print(f"  {key}: Q_max = {rows[-1]['Q_max_W']:.3f} W ({rows[-1]['binding']})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "screening.csv", index=False, float_format="%.6g")
    print(df[["arch", "Q_max_W", "Q_max_skin_W", "Q_max_touch_W", "Q_max_component_W", "binding", "fan_mW",
              "T_skin_max_C", "T_touch_outer_max_C", "energy_err"]].round(3).to_string())
    return df


# --------------------------------------------------------------------------- opt
OPT_BOUNDS = {"U_in": (0.0, 3.0), "H_mm": (2.3, 4.0), "spreader_mm": (0.1, 0.8)}   # air gap >= 0.5 mm
OPT_REFINE = 0.75            # coarse mesh for the search; optima are re-checked on the design mesh
N_INIT, N_ITER = 10, 8


def opt_params(x) -> DeviceParams:
    return replace(ARCH["A4"][1], U_in=float(x[0]), H_mm=float(x[1]), spreader_mm=float(x[2]),
                   refine=OPT_REFINE)


def _unit(X):
    lo = np.array([b[0] for b in OPT_BOUNDS.values()])
    hi = np.array([b[1] for b in OPT_BOUNDS.values()])
    return (np.asarray(X) - lo) / (hi - lo), lo, hi


def _objectives(df):
    """Minimisation form: -Q_max, fan power, spreader mass."""
    return np.column_stack([-df.Q_max_W, df.fan_mW, df.spreader_mass_g])


def pareto_mask(F):
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        keep[i] = not dom.any()
    return keep


def step_opt():
    """ParEGO (Knowles 2006): random Tchebycheff scalarisation + GP expected improvement."""
    from scipy.stats import norm, qmc
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    print("P4.2: multi-objective Bayesian optimisation of architecture A4")
    rng = np.random.default_rng(42)
    lo = np.array([b[0] for b in OPT_BOUNDS.values()])
    hi = np.array([b[1] for b in OPT_BOUNDS.values()])
    X0 = qmc.scale(qmc.LatinHypercube(d=3, seed=11, optimization="random-cd").random(N_INIT), lo, hi)
    csv = RES / "optimisation.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {tuple(np.round([r["U_in"], r["H_mm"], r["spreader_mm"]], 6)) for r in rows}

    def add(x, tag):
        key = tuple(np.round(x, 6))
        if key in done:
            return
        name = f"opt_U{x[0]:.3f}_H{x[1]:.3f}_s{x[2]:.3f}"
        r = evaluate(opt_params(x), name)
        rows.append(dict(U_in=x[0], H_mm=x[1], spreader_mm=x[2], source=tag, **r))
        done.add(key)
        pd.DataFrame(rows).to_csv(csv, index=False, float_format="%.6g")
        print(f"  {tag:8s} U={x[0]:.2f} H={x[1]:.2f} s={x[2]:.2f} -> Q_max={r['Q_max_W']:.3f} W "
              f"fan={r['fan_mW']:.2f} mW mass={r['spreader_mass_g']:.2f} g", flush=True)

    for x in X0:
        add(x, "LHS")
    n_bo = sum(r["source"] == "ParEGO" for r in rows)
    for it in range(n_bo, N_ITER):
        df = pd.DataFrame(rows)
        F = _objectives(df)
        Fn = (F - F.min(0)) / np.maximum(F.max(0) - F.min(0), 1e-12)
        lam = rng.dirichlet(np.ones(3))
        g = np.max(lam * Fn, axis=1) + 0.05 * (lam * Fn).sum(axis=1)
        Xu, _, _ = _unit(df[list(OPT_BOUNDS)].to_numpy())
        k = ConstantKernel(1.0) * Matern(np.ones(3), (0.05, 10.0), nu=2.5) + WhiteKernel(1e-4, (1e-8, 1e-1))
        gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=5, random_state=it).fit(Xu, g)
        C = rng.random((20000, 3))
        mu, sd = gp.predict(C, return_std=True)
        imp = g.min() - mu
        z = imp / np.maximum(sd, 1e-12)
        ei = imp * norm.cdf(z) + sd * norm.pdf(z)
        x = lo + C[np.argmax(ei)] * (hi - lo)
        add(x, "ParEGO")
    df = pd.DataFrame(rows)
    df["pareto"] = pareto_mask(_objectives(df))
    df.to_csv(csv, index=False, float_format="%.6g")
    print(df[["source", "U_in", "H_mm", "spreader_mm", "Q_max_W", "binding", "fan_mW",
              "spreader_mass_g", "pareto"]].round(3).to_string())
    return df


# --------------------------------------------------------------------------- verify
def pick_designs(df):
    """Three designs from the (coarse-mesh) Pareto set, chosen by explicit rules."""
    P = df[df.pareto].copy()
    picks = {}
    picks["max_power"] = P.loc[P.Q_max_W.idxmax()]
    quiet = P[P.fan_mW <= 0.1]
    if len(quiet):
        picks["quiet"] = quiet.loc[quiet.Q_max_W.idxmax()]
    bal = P[(P.fan_mW <= 1.0) & (P.spreader_mass_g <= 1.0)]
    if len(bal):
        picks["balanced"] = bal.loc[bal.Q_max_W.idxmax()]
    return picks


def step_verify():
    """Design-mesh re-evaluation of the picks, then a run AT Q_max to confirm all limits."""
    print("P4.3: verification of the chosen designs on the design mesh")
    df = pd.read_csv(RES / "optimisation.csv")
    rows = []
    for tag, r in pick_designs(df).items():
        p = replace(ARCH["A4"][1], U_in=float(r.U_in), H_mm=float(r.H_mm),
                    spreader_mm=float(r.spreader_mm))
        name = f"verify_{tag}"
        e = evaluate(p, name)
        pq = replace(p, Q_W=round(e["Q_max_W"], 4))
        d = run(pq, f"{name}_atQmax")
        a = analyse_device(d, pq)
        from coolchan.device3d import surface_temperatures
        st = surface_temperatures(d)
        rows.append(dict(design=tag, U_in=p.U_in, H_mm=p.H_mm, spreader_mm=p.spreader_mm,
                         Q_max_coarse_W=r.Q_max_W, Q_max_W=e["Q_max_W"], binding=e["binding"],
                         fan_mW=e["fan_mW"], spreader_mass_g=e["spreader_mass_g"],
                         at_Qmax_skin_C=float(st["skin"].max()) - 273.15,
                         at_Qmax_touch_C=float(st["touch"].max()) - 273.15,
                         at_Qmax_component_C=float(st["component"].max()) - 273.15,
                         at_Qmax_energy_err=a["device_energy_err"],
                         at_Qmax_residual=a["max_residual"]))
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "verified_designs.csv", index=False, float_format="%.6g")
    return out


# --------------------------------------------------------------------------- report
def step_report():
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    plots.setup()
    sc = pd.read_csv(RES / "screening.csv")
    op = pd.read_csv(RES / "optimisation.csv")
    vf = pd.read_csv(RES / "verified_designs.csv") if (RES / "verified_designs.csv").exists() else None

    # ---- tables
    T = ["| Architecture | Q<sub>max</sub> [W] | Limited by | Q at skin 43 °C | Q at touch 48 °C | Q at component 60 °C | Fan [mW] |",
         "|---|---|---|---|---|---|---|"]
    for _, r in sc.iterrows():
        T.append(f"| {r.arch}: {r.label} | **{r.Q_max_W:.2f}** | {r.binding} | {r.Q_max_skin_W:.2f} | "
                 f"{r.Q_max_touch_W:.2f} | {r.Q_max_component_W:.2f} | {abs(r.fan_mW):.2f} |")
    (RES / "screening_table.md").write_text("\n".join(T) + "\n")
    if vf is not None:
        V = ["| Design | Fan U [m/s] | Gap [mm] | Al spreader [mm] | Spreader mass [g] | Fan [mW] | Q<sub>max</sub> coarse → design mesh [W] | Limited by | At Q<sub>max</sub>: skin / touch / component [°C] |",
             "|---|---|---|---|---|---|---|---|---|"]
        for _, r in vf.iterrows():
            V.append(f"| {r.design} | {r.U_in:.2f} | {r.H_mm:.2f} | {r.spreader_mm:.2f} | {r.spreader_mass_g:.2f} | "
                     f"{r.fan_mW:.2f} | {r.Q_max_coarse_W:.3f} → **{r.Q_max_W:.3f}** | {r.binding} | "
                     f"{r.at_Qmax_skin_C:.2f} / {r.at_Qmax_touch_C:.2f} / {r.at_Qmax_component_C:.2f} |")
        (RES / "verified_table.md").write_text("\n".join(V) + "\n")
        print("\n".join(V))
    print("\n".join(T))

    # ---- screening figure
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    y = np.arange(len(sc))[::-1]
    ax.barh(y, sc.Q_max_W, color=[plots.C1 if a != "A0" else plots.MUTED for a in sc.arch], height=0.6)
    for yy, (_, r) in zip(y, sc.iterrows()):
        for col, mk, c in (("Q_max_skin_W", "o", plots.C2), ("Q_max_touch_W", "s", plots.C4),
                           ("Q_max_component_W", "D", plots.C3)):
            v = r[col]
            if v < 2.0:
                ax.plot(v, yy, mk, color=c, mec=plots.INK, mew=0.6, ms=7, zorder=3)
        ax.text(r.Q_max_W + 0.02, yy - 0.3, f"{r.Q_max_W:.2f} W", fontsize=8.5, color=plots.INK)
    ax.plot([], [], "o", color=plots.C2, mec=plots.INK, label="skin reaches 43 °C")
    ax.plot([], [], "s", color=plots.C4, mec=plots.INK, label="outer housing reaches 48 °C")
    ax.plot([], [], "D", color=plots.C3, mec=plots.INK, label="heater reaches 60 °C")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.arch}  {r.label}" for _, r in sc.iterrows()], fontsize=8.5)
    ax.set_xlabel("maximum continuous heater power Q$_{max}$ [W]   (bar = first limit reached)")
    ax.set_xlim(0, 2.0)
    ax.set_title("Architecture screening: power each layout can sustain (fan 1.5 m/s, 3 mm gap)")
    ax.legend(loc="lower right", fontsize=8)
    plots.save(fig, FIG / "phase4_screening.png")

    # ---- Pareto figure
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    fan = op.fan_mW.clip(lower=1e-3)
    norm = Normalize(op.spreader_mass_g.min(), op.spreader_mass_g.max())
    sca = ax.scatter(fan, op.Q_max_W, c=op.spreader_mass_g, cmap=plots.SEQ, norm=norm, s=46,
                     edgecolor=np.where(op.pareto, plots.INK, "none"), linewidth=1.2, zorder=3)
    S = op.sort_values("fan_mW")
    front = S[S.Q_max_W >= S.Q_max_W.cummax()]          # Q_max vs fan-power front
    ax.plot(front.fan_mW.clip(lower=1e-3), front.Q_max_W, color=plots.INK2, lw=0.9, ls="--", zorder=2,
            label="Q$_{max}$–fan power front")
    if vf is not None:
        for _, r in vf.iterrows():
            ax.plot(max(r.fan_mW, 1e-3), r.Q_max_W, "*", ms=15, color=plots.C2, mec=plots.INK, zorder=4)
            ax.annotate(r.design, (max(r.fan_mW, 1e-3), r.Q_max_W), xytext=(-8, 8), ha="right",
                        textcoords="offset points", fontsize=8.5)
    base = sc.set_index("arch").loc["A0"]
    ax.axhline(base.Q_max_W, color=plots.MUTED, lw=1, ls=":")
    ax.text(0.99, base.Q_max_W + 0.015, f"Phase 2 baseline {base.Q_max_W:.2f} W", fontsize=8,
            color=plots.INK2, ha="right", transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel("ideal fan power [mW]")
    ax.set_ylabel("Q$_{max}$ [W]")
    ax.set_title("ParEGO search: outer-wall heater + Al spreader (18 CFD designs)\n"
                 "stars: picks re-checked on the design mesh")
    ax.legend(loc="upper left", fontsize=8)
    cb = fig.colorbar(sca, ax=ax, pad=0.01)
    cb.set_label("spreader mass [g]")
    cb.outline.set_visible(False)
    plots.save(fig, FIG / "phase4_pareto.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["screen"])
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    steps = {"screen": step_screen, "opt": step_opt, "verify": step_verify, "report": step_report}
    for s in a.steps:
        steps[s]()


if __name__ == "__main__":
    main()
