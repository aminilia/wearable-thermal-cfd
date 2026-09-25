#!/usr/bin/env python3
"""Stage 3 - automated design study: LHS -> CFD -> surrogate -> Monte Carlo.

Steps (each can be run on its own; results are cached in results/):
  run    sample a Latin hypercube over (U, H, Q, k), build + run every case,
         parse the outputs into results/doe.csv (plus an independent hold-out set)
  fit    fit three surrogates, score by leave-one-out and on the hold-out set
  mc     Monte Carlo on the best surrogate -> exceedance-probability map,
         design map and the minimum-fan-power design that meets the risk target

Usage:
  python scripts/stage3_design.py all              # everything
  python scripts/stage3_design.py fit mc           # re-analyse cached CFD results
  python scripts/stage3_design.py run --n-train 30 --n-test 6
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace, asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coolchan.case import ChtParams, build_cht  # noqa: E402
from coolchan.cht import analyse_cht  # noqa: E402
from coolchan.runner import run_many  # noqa: E402
from coolchan import surrogate as S  # noqa: E402
from coolchan import plots  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "stage3"
RES = REPO / "results"
FIG = REPO / "docs" / "figures"

DESIGN_MESH = dict(dy_mm=0.05, aspect=4.0)   # "medium" mesh from Stage 2
UNC = S.Uncertainty()
RISK_TARGET = 0.01                           # accept designs with P(T_max > 43 C) <= 1 %


def n_iter(U):
    # low-velocity cases converge more slowly
    return int(2500 if U >= 1.0 else 3500)


# --------------------------------------------------------------------------- #
def step_run(n_train, n_test, workers):
    RES.mkdir(exist_ok=True)
    sets = {"train": S.lhs(n_train, seed=2026), "test": S.lhs(n_test, seed=7)}
    jobs = []
    for tag, X in sets.items():
        for i, row in enumerate(X):
            x = dict(zip(S.NAMES, map(float, row)))
            p = ChtParams(**x, **DESIGN_MESH, nIter=n_iter(x["U_in"]))
            name = f"{tag}_{i:02d}"
            jobs.append((tag, name, p))
    todo = []
    for tag, name, p in jobs:
        d = RUNS / name
        done = (d / "log.chtMultiRegionSimpleFoam").exists() and \
            "End" in (d / "log.chtMultiRegionSimpleFoam").read_text()[-200:]
        if done and json.loads((d / "params.json").read_text()) == asdict(p):
            continue                                   # cached
        build_cht(d, p)
        (d / "params.json").write_text(json.dumps(asdict(p)))
        todo.append(d)
    print(f"{len(jobs)} cases, {len(todo)} to run")
    # slowest (large gap, low U) first
    todo.sort(key=lambda d: -json.loads((d / "params.json").read_text())["H_mm"])
    out = run_many(todo, workers)
    rows = []
    for tag, name, p in jobs:
        if isinstance(out.get(name), Exception):
            continue
        r = analyse_cht(RUNS / name, p)
        rows.append(dict(set=tag, case=name, **{k: getattr(p, k) for k in S.NAMES},
                         T_in_C=p.T_in - 273.15, Re=p.Re, **r.as_dict()))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "doe.csv", index=False, float_format="%.6g")
    q = df[["energy_err", "dTmax_iter", "max_residual"]].abs().max()
    print("QC worst case:", q.to_dict())
    return df


# --------------------------------------------------------------------------- #
def step_fit():
    df = pd.read_csv(RES / "doe.csv")
    tr, te = df[df.set == "train"].reset_index(drop=True), df[df.set == "test"].reset_index(drop=True)
    out, preds = {}, {}
    for cls in (S.QuadraticRSM, S.GPTmax, S.GPResistance):
        l = S.loo(cls, tr)
        m = cls().fit(tr)
        h = m.predict(te) if len(te) else np.array([])
        out[cls.name] = dict(loo=S.metrics(tr.T_max_C, l),
                             holdout=S.metrics(te.T_max_C, h) if len(te) else None)
        preds[cls.name] = (l, h)
    (RES / "surrogate_metrics.json").write_text(json.dumps(out, indent=2))
    lines = ["| Surrogate | LOO RMSE [K] | LOO max error [K] | LOO R² | Hold-out RMSE [K] | Hold-out max error [K] |",
             "|---|---|---|---|---|---|"]
    for k, v in out.items():
        if not isinstance(v, dict):
            continue
        ho = v["holdout"] or dict(rmse=np.nan, max_abs=np.nan)
        lines.append(f"| {k} | {v['loo']['rmse']:.3f} | {v['loo']['max_abs']:.3f} | "
                     f"{v['loo']['r2']:.4f} | {ho['rmse']:.3f} | {ho['max_abs']:.3f} |")
    (RES / "surrogate_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot_parity(tr, te, preds)
    return out


def plot_parity(tr, te, preds):
    import matplotlib.pyplot as plt
    plots.setup()
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.5), sharex=True, sharey=True)
    lo, hi = min(tr.T_max_C.min(), 25), tr.T_max_C.max() * 1.03
    for ax, (name, (l, h)) in zip(axs, preds.items()):
        ax.plot([lo, hi], [lo, hi], color=plots.MUTED, lw=1)
        ax.axhline(43, color=plots.C2, lw=0.8, ls=":")
        ax.axvline(43, color=plots.C2, lw=0.8, ls=":")
        ax.plot(tr.T_max_C, l, "o", color=plots.C1, mfc="none", mew=1.4, label="training (leave-one-out)")
        if len(te):
            ax.plot(te.T_max_C, h, "D", color=plots.C2, ms=6, label="independent hold-out CFD")
        rmse = np.sqrt(np.mean((l - tr.T_max_C) ** 2))
        ax.set_title(name, fontsize=9.5)
        ax.text(0.04, 0.96, f"LOO RMSE {rmse:.2f} K", transform=ax.transAxes, va="top", fontsize=8.5,
                color=plots.INK2)
        ax.set_xlabel("CFD T$_{max}$ [°C]")
    axs[0].set_ylabel("surrogate T$_{max}$ [°C]")
    axs[0].legend(loc="lower right")
    plots.save(fig, FIG / "stage3_surrogate_parity.png")


# --------------------------------------------------------------------------- #
def step_mc():
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    df = pd.read_csv(RES / "doe.csv")
    tr = df[df.set == "train"]
    gp = S.GPResistance().fit(df)                  # final model uses every CFD run
    gpp = S.GPPressure().fit(df)

    nU, nH = 41, 41
    Ug = np.linspace(*S.BOUNDS["U_in"], nU)
    Hg = np.linspace(*S.BOUNDS["H_mm"], nH)
    HH, UU = np.meshgrid(Hg, Ug)
    k_nom = 0.5 * (UNC.k_lo + UNC.k_hi)
    Rn = gp.predict_R(np.column_stack([UU.ravel(), HH.ravel(), np.full(UU.size, k_nom)])).reshape(UU.shape)
    Tn = UNC.T_in_mean + UNC.Q_nom * Rn
    dp = gpp.predict(np.column_stack([UU.ravel(), HH.ravel()])).reshape(UU.shape)
    W = ChtParams().W_mm * 1e-3
    fan_mW = 1e3 * dp * UU * HH * 1e-3 * W          # ideal (dp x volume flow)

    print("Monte Carlo on", UU.size, "design points ...")
    P, T95, Qa = S.exceedance_probability(gp, UU, HH, UNC, n=20000, risk=RISK_TARGET)
    ok = P <= RISK_TARGET
    best = None
    if ok.any():
        i = np.argmin(np.where(ok, fan_mW, np.inf))
        iu, ih = np.unravel_index(i, UU.shape)
        best = dict(U_in=float(UU[iu, ih]), H_mm=float(HH[iu, ih]), P_exceed=float(P[iu, ih]),
                    T95_C=float(T95[iu, ih]), T_nominal_C=float(Tn[iu, ih]),
                    fan_mW=float(fan_mW[iu, ih]), dp_Pa=float(dp[iu, ih]),
                    Q_allow_W=float(Qa[iu, ih]))
    # allowable power at a few representative designs
    pts = []
    for U0 in (1.0, 2.0, 3.0):
        for H0 in (1.5, 2.5, 4.0):
            iu, ih = np.argmin(abs(Ug - U0)), np.argmin(abs(Hg - H0))
            pts.append(dict(U_in=U0, H_mm=H0, Q_allow_W=float(Qa[iu, ih]),
                            P_at_Qnom=float(P[iu, ih]), fan_mW=float(fan_mW[iu, ih]),
                            T_nominal_C=float(Tn[iu, ih])))
    out = dict(uncertainty=asdict(UNC), risk_target=RISK_TARGET, nominal_k=k_nom,
               best=best, fraction_of_space_safe_at_Qnom=float(ok.mean()),
               Q_allow_range_W=[float(Qa.min()), float(Qa.max())], table=pts)
    (RES / "monte_carlo.json").write_text(json.dumps(out, indent=2))
    np.savez(RES / "design_maps.npz", U=Ug, H=Hg, T_nominal=Tn, P_exceed=P, T95=T95,
             Q_allow=Qa, fan_mW=fan_mW)
    lines = ["| U [m/s] | Gap H [mm] | T<sub>max</sub> at 0.5 W, nominal inputs [°C] | "
             "P(T<sub>max</sub> > 43 °C) at 0.5 W | Max. power for ≤ 1 % risk [W] | Ideal fan power [mW] |",
             "|---|---|---|---|---|---|"]
    for r in pts:
        lines.append(f"| {r['U_in']:.1f} | {r['H_mm']:.1f} | {r['T_nominal_C']:.1f} | "
                     f"{100 * r['P_at_Qnom']:.1f} % | {r['Q_allow_W']:.2f} | {r['fan_mW']:.2f} |")
    (RES / "monte_carlo_table.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(out, indent=2))
    print("\n".join(lines))

    # ------------------------------------------------------------ design map
    plots.setup()
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True,
                            gridspec_kw=dict(wspace=0.22))
    plt.rcParams["axes.titlesize"] = 9.5
    ax = axs[0]
    norm = TwoSlopeNorm(vcenter=UNC.T_limit, vmin=min(Tn.min(), 31), vmax=max(Tn.max(), 55))
    cf = ax.contourf(HH, UU, Tn, levels=60, cmap=plots.DIV, norm=norm)
    cs = ax.contour(HH, UU, Tn, levels=[35, 39, 47, 51, 55], colors=plots.INK2, linewidths=0.5)
    ax.clabel(cs, fmt="%d °C", fontsize=7.5)
    c43 = ax.contour(HH, UU, Tn, levels=[UNC.T_limit], colors=plots.INK, linewidths=2)
    ax.clabel(c43, fmt="43 °C", fontsize=8)
    ax.plot(tr.H_mm, tr.U_in, "o", ms=4, color="white", mec=plots.INK, mew=0.8)
    ax.set_xlabel("channel gap H [mm]")
    ax.set_ylabel("inlet velocity U [m/s]")
    ax.set_title("(a) Peak temperature at nominal inputs\n"
                 f"Q = {UNC.Q_nom} W, k = {k_nom:.0f} W/m·K, T$_{{in}}$ = 25 °C  (dots: CFD runs)")
    ax.grid(False)
    cb = fig.colorbar(cf, ax=ax, pad=0.015, ticks=[31, 35, 39, 43, 47, 51, 55])
    cb.set_label("T$_{max}$ [°C]")
    cb.outline.set_visible(False)

    ax = axs[1]
    cf = ax.contourf(HH, UU, 100 * P, levels=np.linspace(0, 100, 21), cmap=plots.SEQ)
    c1 = ax.contour(HH, UU, 100 * P, levels=[1, 50], colors=[plots.INK, plots.INK2],
                    linewidths=[2, 0.8], linestyles=["-", "--"])
    ax.clabel(c1, fmt="%g %%", fontsize=8)
    cfan = ax.contour(HH, UU, fan_mW, levels=[0.1, 0.3, 1, 3], colors=plots.C2, linewidths=1)
    ax.clabel(cfan, fmt="%g mW", fontsize=7.5)
    if best:
        ax.plot(best["H_mm"], best["U_in"], "*", ms=16, color=plots.C4, mec=plots.INK, mew=0.8,
                clip_on=False, zorder=5)
        ax.annotate(f"lowest fan power with P ≤ 1 %\nH = {best['H_mm']:.2f} mm, "
                    f"U = {best['U_in']:.2f} m/s", (best["H_mm"], best["U_in"]),
                    xytext=(2.05, 1.05), fontsize=8, color="white",
                    arrowprops=dict(arrowstyle="-", color="white", lw=0.8))
    ax.set_xlabel("channel gap H [mm]")
    ax.set_title("(b) Monte Carlo risk P(T$_{max}$ > 43 °C) at 0.5 W\n"
                 "orange: ideal fan power")
    ax.grid(False)
    cb = fig.colorbar(cf, ax=ax, pad=0.015)
    cb.set_label("exceedance probability [%]")
    cb.outline.set_visible(False)

    ax = axs[2]
    lv = np.arange(0.2, 0.75, 0.05)
    cf = ax.contourf(HH, UU, Qa, levels=lv, cmap=plots.SEQ, extend="both")
    cq = ax.contour(HH, UU, Qa, levels=[0.3, 0.4, 0.5, 0.6], colors=plots.INK, linewidths=0.7)
    ax.clabel(cq, fmt="%.1f W", fontsize=8)
    ax.set_xlabel("channel gap H [mm]")
    ax.set_title("(c) Largest heat load with ≤ 1 % risk\n"
                 "(load ±10 %, k 150–220 W/m·K, T$_{in}$ 25±1.5 °C, model error)")
    ax.grid(False)
    cb = fig.colorbar(cf, ax=ax, pad=0.015)
    cb.set_label("allowable Q [W]")
    cb.outline.set_visible(False)
    plots.save(fig, FIG / "stage3_design_map.png")

    # ------------------------------------------------------------ histogram
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    rng = np.random.default_rng(3)
    n = 40000
    cmp_pts = [(ChtParams().U_in, ChtParams().H_mm, plots.C2, "Stage-2 nominal")]
    if best:
        cmp_pts.append((best["U_in"], best["H_mm"], plots.C1, "recommended"))
    for U0, H0, c, lab in cmp_pts:
        k = rng.uniform(UNC.k_lo, UNC.k_hi, n)
        Q = rng.normal(UNC.Q_nom, UNC.Q_cv * UNC.Q_nom, n)
        Tin = rng.normal(UNC.T_in_mean, UNC.T_in_sd, n)
        R, sd = gp.predict_R(np.column_stack([np.full(n, U0), np.full(n, H0), k]), return_std=True)
        T = Tin + Q * R * np.exp(sd * rng.standard_normal(n))
        pe = np.mean(T > UNC.T_limit)
        ax.hist(T, bins=80, density=True, histtype="stepfilled", alpha=0.35, color=c, ec=c, lw=1.2,
                label=f"{lab}: U = {U0:.2f} m/s, H = {H0:.2f} mm → P(>43 °C) = {100 * pe:.1f} %")
    ax.axvline(UNC.T_limit, color=plots.INK, lw=1.5)
    ax.text(UNC.T_limit + 0.3, ax.get_ylim()[1] * 0.95, "43 °C", va="top", fontsize=8.5)
    ax.set_xlabel("T$_{max}$ [°C]")
    ax.set_ylabel("probability density")
    ax.set_title("Monte Carlo distribution of peak temperature at 0.5 W")
    ax.legend(loc="upper left", fontsize=7.8, bbox_to_anchor=(0, -0.2))
    plots.save(fig, FIG / "stage3_mc_histogram.png")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["all"], choices=["all", "run", "fit", "mc"])
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=6)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    steps = {"run", "fit", "mc"} if "all" in a.steps else set(a.steps)
    if "run" in steps:
        step_run(a.n_train, a.n_test, a.workers)
    if "fit" in steps:
        step_fit()
    if "mc" in steps:
        step_mc()


if __name__ == "__main__":
    main()
