"""Post-processing of the conjugate heat-transfer (heated block) case."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .case import AIR, K_AIR, ChtParams
from .foamio import Case, final_residuals


@dataclass
class ChtResult:
    T_max_C: float          # peak solid temperature [C]
    T_out_C: float          # mixed-mean outlet air temperature [C]
    dp_Pa: float            # inlet - outlet pressure drop [Pa]
    fan_W: float            # ideal fan power dp * volumetric flow [W]
    R_KW: float             # thermal resistance (T_max - T_in)/Q [K/W]
    energy_err: float       # first-law imbalance / Q
    dTmax_iter: float       # |T_max(last) - T_max(previous write)| [K]
    max_residual: float     # largest initial residual at the last iteration
    n_cells: int

    def as_dict(self):
        return asdict(self)


def _tmax(c: Case, time: str) -> float:
    T = c.field("T", "heater", time)
    vals = [T.internal.max()] + [v.max() for v in T.boundary.values() if len(v)]
    return float(max(vals))


def analyse_cht(case_dir, p: ChtParams) -> ChtResult:
    c = Case(case_dir)
    times = c.times()
    t = times[-1]
    m = c.mesh("air")
    ms = c.mesh("heater")
    T = c.field("T", "air", t)
    U = c.field("U", "air", t)
    P = c.field("p", "air", t)
    rho, cp = AIR["rho_air"], AIR["cp_air"]

    dz = float(np.ptp(m.points[:, 2]))       # 2D slab thickness [m]
    scale = p.W_mm * 1e-3 / dz               # slab -> full spanwise depth

    def conv(name):
        Sf = m.patch_areas(name)
        md = rho * np.einsum("ij,ij->i", U.boundary[name], Sf)
        return md, (md * cp * T.boundary[name]).sum()

    md_in, H_in = conv("inlet")
    md_out, H_out = conv("outlet")
    ic = m.patch_cells("inlet")
    d = m.C[ic, 0] - m.patch_centres("inlet")[:, 0]
    A = np.linalg.norm(m.patch_areas("inlet"), axis=1)
    Q_cond_in = (K_AIR * (T.internal[ic] - T.boundary["inlet"]) / d * A).sum()
    Q_gen = p.qvol * ms.V.sum()
    energy_err = ((H_in + H_out + Q_cond_in) - Q_gen) / Q_gen

    T_out = (md_out * T.boundary["outlet"]).sum() / md_out.sum()
    Aout = np.linalg.norm(m.patch_areas("outlet"), axis=1)
    Ain = np.linalg.norm(m.patch_areas("inlet"), axis=1)
    p_in = (P.boundary["inlet"] * Ain).sum() / Ain.sum()
    p_out = (P.boundary["outlet"] * Aout).sum() / Aout.sum()
    dp = p_in - p_out
    vol_flow = -md_in.sum() / rho * scale
    Tmax = _tmax(c, t)
    dT_iter = abs(Tmax - _tmax(c, times[-2])) if len(times) > 2 else np.nan

    res = final_residuals(c.path / "log.chtMultiRegionSimpleFoam")
    return ChtResult(
        T_max_C=Tmax - 273.15,
        T_out_C=T_out - 273.15,
        dp_Pa=dp,
        fan_W=dp * vol_flow,
        R_KW=(Tmax - p.T_in) / p.Q_W,
        energy_err=energy_err,
        dTmax_iter=dT_iter,
        max_residual=max(res.values()) if res else np.nan,
        n_cells=m.n_cells + ms.n_cells,
    )
