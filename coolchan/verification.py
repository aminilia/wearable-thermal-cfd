"""Analytical references, channel post-processing and the Grid Convergence Index.

References
----------
* Plane Poiseuille flow between plates a distance H apart:
  u(y) = 6 U_m (y/H)(1 - y/H),  -dp/dx = 12 mu U_m / H^2,  f_Darcy Re_Dh = 96
* Fully developed laminar flow, both plates at the same uniform heat flux:
  Nu_Dh = 140/17 = 8.235  (Shah & London 1978; Incropera Table 8.1)
* GCI: Roache (1998); procedure of Celik et al., J. Fluids Eng. 130 (2008) 078001
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .case import AIR, K_AIR, ChannelParams
from .foamio import Case

NU_FD = 140.0 / 17.0   # 8.2353
FRE_FD = 96.0


def poiseuille_u(y, H, Um):
    eta = np.asarray(y) / H
    return 6.0 * Um * eta * (1.0 - eta)


# --------------------------------------------------------------------------- #
@dataclass
class ChannelResult:
    x: np.ndarray            # column x-positions [m]
    Nu: np.ndarray           # local Nusselt number Nu_Dh(x)
    Tb: np.ndarray           # bulk temperature [K]
    Tw: np.ndarray           # mean wall temperature [K]
    Tb_exact: np.ndarray     # first-law bulk temperature [K]
    y_prof: np.ndarray       # profile sample location (cell centres) [m]
    u_prof: np.ndarray       # u(y) at x_probe
    x_probe: float
    dpdx: float              # developed-region pressure gradient [Pa/m]
    fRe: float
    dp_total: float          # inlet - outlet area-averaged pressure [Pa]
    Nu_fd: float             # Nu averaged over the developed window
    u_max_err: float         # max |u - u_exact| / U_m at x_probe
    energy_err: float        # (Q_air - Q_walls) / Q_walls


def analyse_channel(case_dir, p: ChannelParams, window=(0.6, 0.9), x_probe=0.8,
                    time=None) -> ChannelResult:
    c = Case(case_dir)
    m = c.mesh("air")
    U = c.field("U", "air", time).internal[:, 0]
    T = c.field("T", "air", time)
    P = c.field("p", "air", time)
    C = m.C
    H, L = p.H_mm * 1e-3, p.L_mm * 1e-3
    Dh = 2 * H

    # structured columns: group cells by x-centre
    xr = np.round(C[:, 0], 12)
    xs = np.unique(xr)
    col = np.searchsorted(xs, xr)
    V = m.V                     # within a column dx is constant, so V ~ dy
    wU = U * V
    Tb = np.bincount(col, wU * T.internal) / np.bincount(col, wU)

    # wall temperature per column (both plates, same x)
    Tw = np.zeros_like(xs)
    for patch in ("bottomWall", "topWall"):
        xf = np.round(m.patch_centres(patch)[:, 0], 12)
        order = np.argsort(xf)
        Tw += T.boundary[patch][order]
    Tw /= 2.0
    q = p.q_wall
    with np.errstate(divide="ignore", invalid="ignore"):
        Nu = q * Dh / (K_AIR * (Tw - Tb))

    rho, cp = AIR["rho_air"], AIR["cp_air"]
    Tb_exact = p.T_in + 2 * q * xs / (rho * p.U_in * H * cp)

    # energy balance on the whole channel (per unit depth)
    def patch_flux(name, fld):
        Sf = m.patch_areas(name)
        Uf = c.field("U", "air", time).boundary[name]
        mdot = rho * np.einsum("ij,ij->i", Uf, Sf)
        return (mdot * cp * fld.boundary[name]).sum(), mdot.sum()
    Hin, _ = patch_flux("inlet", T)
    Hout, _ = patch_flux("outlet", T)
    dz = np.ptp(m.points[:, 2])
    Q_walls = 2 * q * L * dz
    # axial conduction back out through the fixed-T inlet (small but real at Pe~90)
    ic = m.patch_cells("inlet")
    d = m.C[ic, 0] - m.patch_centres("inlet")[:, 0]
    A = np.linalg.norm(m.patch_areas("inlet"), axis=1)
    Q_cond_in = (K_AIR * (T.internal[ic] - T.boundary["inlet"]) / d * A).sum()
    energy_err = ((Hout + Hin) + Q_cond_in - Q_walls) / Q_walls if q else 0.0

    # velocity profile at x_probe
    ip = np.argmin(abs(xs - x_probe * L))
    sel = col == ip
    y_prof, u_prof = C[sel, 1], U[sel]
    o = np.argsort(y_prof)
    y_prof, u_prof = y_prof[o], u_prof[o]
    u_max_err = np.max(abs(u_prof - poiseuille_u(y_prof, H, p.U_in))) / p.U_in

    # pressure: column-mean p, linear fit over the developed window
    pc = np.bincount(col, P.internal * V) / np.bincount(col, V)
    w = (xs > window[0] * L) & (xs < window[1] * L)
    dpdx = np.polyfit(xs[w], pc[w], 1)[0]
    fRe = (-dpdx) * Dh / (0.5 * rho * p.U_in**2) * p.Re
    dp_total = P.boundary["inlet"].mean() - P.boundary["outlet"].mean()

    Nu_fd = float(np.mean(Nu[w])) if q else float("nan")
    return ChannelResult(xs, Nu, Tb, Tw, Tb_exact, y_prof, u_prof, xs[ip], dpdx,
                         fRe, dp_total, Nu_fd, u_max_err, energy_err)


# --------------------------------------------------------------------------- #
@dataclass
class GCI:
    phi: tuple               # (fine, medium, coarse)
    h: tuple                 # representative cell sizes
    r21: float
    r32: float
    p: float                 # observed order of accuracy
    phi_ext: float           # Richardson-extrapolated value
    e21_a: float             # approximate relative error fine-medium
    e21_ext: float           # extrapolated relative error of the fine grid
    gci_fine: float          # GCI_fine (fraction)
    gci_medium: float
    asymptotic: float        # GCI_32 / (r^p GCI_21) ~ 1 in the asymptotic range
    oscillatory: bool


def gci(phi1, phi2, phi3, h1, h2, h3, Fs=1.25, p_max=None) -> GCI:
    """Celik et al. (2008) three-grid GCI; index 1 = finest."""
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = phi2 - phi1, phi3 - phi2
    osc = e21 * e32 < 0
    s = np.sign(e32 / e21) if e21 != 0 else 1.0
    # fixed-point iteration for p (handles non-constant r)
    p = abs(np.log(abs(e32 / e21))) / np.log(r21) if e21 != 0 else np.nan
    for _ in range(100):
        q = np.log((r21**p - s) / (r32**p - s))
        p_new = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    if p_max is not None:
        p = min(p, p_max)
    phi_ext = (r21**p * phi1 - phi2) / (r21**p - 1)
    ea21 = abs((phi1 - phi2) / phi1)
    eext21 = abs((phi_ext - phi1) / phi_ext)
    gci21 = Fs * ea21 / (r21**p - 1)
    ea32 = abs((phi2 - phi3) / phi2)
    gci32 = Fs * ea32 / (r32**p - 1)
    asym = gci32 / (r21**p * gci21) if gci21 else np.nan
    return GCI((phi1, phi2, phi3), (h1, h2, h3), r21, r32, p, phi_ext, ea21,
               eext21, gci21, gci32, asym, bool(osc))
