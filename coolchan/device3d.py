"""Phase 2: 3D wearable temple-arm model with the wearer's skin.

Assembly (half-width, symmetry plane at z = 0; x streamwise, y outward from skin):

    y  ^   ambient (h, emissivity)              housing top (PC)
       |   +-----------------------------------------------+
       |   |  air channel (fan-driven)    [heater block]   |  <- housing side wall at z = Wc..Wc+t
       |   +-----------------------------------------------+   housing bottom (PC), touches skin
     0 +===== epidermis / dermis / fat / inner tissue ==========  (Pennes bioheat)
       |                                                       core at 37 C (y = -depth)

* Heater: aluminium block (battery/SoC + spreader) on the skin-side housing wall.
* Housing <-> epidermis: thermal contact resistance R'' (thin layer in the coupled BC).
* Outer housing and exposed skin: natural convection h + grey-body radiation to ambient.
* Tissue: 4 layers, Pennes source  w*rho_b*c_b*(T_a - T) + q_m  in perfused layers.

Property values: In Compliance Magazine, "Reduced-order modeling of Pennes' bioheat
equation for thermal dose analysis" (Tables 1-2), see docs/phase2.md.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .case import AIR, K_AIR, TEMPLATES, _copy_render
from .foamio import Case, final_residuals
from .lattice import Axis, Lattice

# Sensible-enthalpy reference temperature. Every solid thermo dictionary written here
# sets "Tref 0; Href 0;" explicitly, so h = cp*T in every OpenFOAM release: v1912 ignores
# Tref and uses cp*T, while v2006+ default to Tref = Tstd = 298.15 K unless told otherwise.
# The Pennes source below depends on this reference, and the 1D bioheat verification
# (scripts/phase2_device.py column) checks it against a semi-analytic solution.
TSTD = 0.0
SIGMA = 5.670374419e-8
BLOOD = dict(rho=1060.0, cp=3770.0, T=310.15)

# name, thickness [mm], k, rho, cp, perfusion w [1/s], metabolic q_m [W/m3]  (deep -> surface)
TISSUE = [
    ("inner", 15.0, 0.50, 1000.0, 4000.0, 1.25e-3, 370.0),
    ("fat", 0.6, 0.19, 1000.0, 2500.0, 1.25e-3, 370.0),
    ("dermis", 1.5, 0.45, 1200.0, 3300.0, 1.25e-3, 370.0),
    ("epidermis", 0.4, 0.24, 1200.0, 3590.0, 0.0, 0.0),
]
HOUSING = dict(k=0.20, rho=1050.0, cp=2100.0)        # polycarbonate-type plastic


@dataclass
class DeviceParams:
    U_in: float = 1.5            # fan velocity at the inlet vent [m/s]
    H_mm: float = 3.0            # internal channel gap
    Q_W: float = 0.5             # heater power, full device [W]
    k_heater: float = 200.0      # aluminium spreader/battery stand-in
    R_contact: float = 1e-3      # housing-skin contact resistance [m2K/W] (assumed; see docs)
    T_amb: float = 298.15        # ambient air [K]
    h_amb: float = 5.0           # natural convection on outer surfaces [W/m2K]
    eps_housing: float = 0.9
    eps_skin: float = 0.98
    T_core: float = 310.15
    hb_mm: float = 1.0           # heater height
    Lb_mm: float = 20.0          # heater length
    Wb_mm: float = 8.0           # heater width (full)
    Lup_mm: float = 10.0
    Ldown_mm: float = 30.0
    Wc_mm: float = 12.0          # channel internal width (full)
    t_mm: float = 1.0            # housing wall thickness
    margin_mm: float = 10.0      # tissue beyond the housing footprint
    heater_side: str = "skin"    # "skin": on the skin-side housing wall; "outer": on the outer wall
    refine: float = 1.0          # multiplies every cell count (mesh study)
    nIter: int = 3000
    nProcs: int = 2

    @property
    def L_mm(self):
        return self.Lup_mm + self.Lb_mm + self.Ldown_mm

    @property
    def qvol(self):
        return self.Q_W / (self.Lb_mm * self.hb_mm * self.Wb_mm * 1e-9)

    @property
    def Re(self):
        return AIR["rho_air"] * self.U_in * 2 * self.H_mm * 1e-3 / AIR["mu_air"]

    # ------------------------------------------------------------- lattice
    def lattice(self) -> Lattice:
        r = self.refine
        n = lambda base: max(1, int(round(base * r)))
        M, L, t, H = self.margin_mm, self.L_mm, self.t_mm, self.H_mm
        x = Axis([-M, 0, self.Lup_mm, self.Lup_mm + self.Lb_mm, L, L + M],
                 [n(6), n(self.Lup_mm / 0.4), n(self.Lb_mm / 0.4), n(self.Ldown_mm / 0.5), n(6)],
                 [0.25, 1, 1, 1, 4.0])
        z = Axis([0, self.Wb_mm / 2, self.Wc_mm / 2, self.Wc_mm / 2 + t, self.Wc_mm / 2 + t + M],
                 [n(self.Wb_mm / 2 / 0.4), n((self.Wc_mm - self.Wb_mm) / 2 / 0.25), n(4), n(6)],
                 [1, 1, 1, 4.0])
        ylines, ycells, ygrad = [], [], []
        depth = sum(tl[1] for tl in TISSUE)
        y0 = -depth
        ylines.append(y0)
        base_tissue = {"inner": (10, 0.1), "fat": (3, 1), "dermis": (5, 1), "epidermis": (3, 1)}
        for name, th, *_ in TISSUE:
            y0 += th
            ylines.append(round(y0, 9))
            c, g = base_tissue[name]
            ycells.append(n(c))
            ygrad.append(g)
        if self.heater_side == "skin":
            chan = ((t + self.hb_mm, self.hb_mm / 0.1), (t + H, (H - self.hb_mm) / 0.125))
        else:
            chan = ((t + H - self.hb_mm, (H - self.hb_mm) / 0.125), (t + H, self.hb_mm / 0.1))
        for top, c in ((t, 4), *chan, (2 * t + H, 4)):
            ylines.append(top)
            ycells.append(n(c))
            ygrad.append(1)
        y = Axis(ylines, ycells, ygrad)
        tissue_names = [tl[0] for tl in TISSUE]

        def centre(ax, i):
            return 0.5 * (ax.lines[i] + ax.lines[i + 1])

        def zone(i, j, k):
            xc, yc, zc = centre(x, i), centre(y, j), centre(z, k)
            if yc < 0:
                return tissue_names[j]
            if not (0 < xc < L and zc < self.Wc_mm / 2 + t):
                return None
            if yc < t or yc > t + H or zc > self.Wc_mm / 2:
                return "housing"
            in_band = (yc < t + self.hb_mm) if self.heater_side == "skin" else (yc > t + H - self.hb_mm)
            if self.Lup_mm < xc < self.Lup_mm + self.Lb_mm and zc < self.Wb_mm / 2 and in_band:
                return "heater"
            return "air"

        def classify(i, j, k, d):
            zn = zone(i, j, k)
            if d == "-z" and k == 0:
                return "symmetry", "symmetryPlane"
            if zn == "air":
                if d == "-x":
                    return "inlet", "patch"
                if d == "+x":
                    return "outlet", "patch"
            if zn == "housing":
                return "housingOuter", "wall"
            if zn in tissue_names:
                if d == "-y" and j == 0:
                    return "core", "wall"
                if d == "+y":
                    return "skinExposed", "wall"
                return "tissueFar", "wall"
            raise ValueError(f"unclassified exterior face: zone {zn} block {(i, j, k)} dir {d}")

        return Lattice(x, y, z, zone, classify)

    def regions(self):
        solids = ["heater", "housing"] + [tl[0] for tl in TISSUE]
        return ["air"], solids


# --------------------------------------------------------------------------- #
HDR = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    location    "{loc}";
    object      {obj};
}}
"""


def _w(path: Path, cls: str, obj: str, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    loc = path.parent.name
    path.write_text(HDR.format(cls=cls, loc=loc, obj=obj) + "\n" + body)


def solid_thermo(k, rho, cp, mw=50.0):
    return f"""thermoType
{{
    type            heSolidThermo;
    mixture         pureMixture;
    transport       constIso;
    thermo          hConst;
    equationOfState rhoConst;
    specie          specie;
    energy          sensibleEnthalpy;
}}

mixture
{{
    specie          {{ molWeight {mw}; }}
    transport       {{ kappa {k:g}; }}
    thermodynamics  {{ Hf 0; Cp {cp:g}; Tref 0; Href 0; }}
    equationOfState {{ rho {rho:g}; }}
}}
"""


def pennes_source(w, qm, cp_tissue):
    """Pennes term w rho_b c_b (T_a - T) + q_m written for the sensible-enthalpy equation.

    h = cp (T - Tstd)  =>  S = [w rho_b c_b (T_a - Tstd) + q_m] + [-w rho_b c_b / cp] h
    """
    a = w * BLOOD["rho"] * BLOOD["cp"]
    su = a * (BLOOD["T"] - TSTD) + qm
    sp = -a / cp_tissue
    return f"""// Pennes bioheat: w rho_b c_b (T_a - T) + q_m, linearised in h (see coolchan/device3d.py)
pennes
{{
    type            scalarSemiImplicitSource;
    active          true;
    selectionMode   all;
    volumeMode      specific;
    injectionRateSuSp
    {{
        h           ({su:.12g} {sp:.12g});
    }}
}}
"""


def coupled_T(kappa_method, R_contact=None):
    layers = ""
    if R_contact:
        # thin layer of arbitrary thickness with kappa chosen to give R''
        d = 1e-4
        layers = f"\n        thicknessLayers ({d:g});\n        kappaLayers     ({d / R_contact:.6g});"
    return f"""{{
        type            compressible::turbulentTemperatureCoupledBaffleMixed;
        Tnbr            T;
        kappaMethod     {kappa_method};{layers}
        value           $internalField;
    }}"""


def ambient_T(p: DeviceParams, eps, kappa_method="solidThermo"):
    return f"""{{
        type            externalWallHeatFluxTemperature;
        mode            coefficient;
        h               uniform {p.h_amb:g};
        Ta              constant {p.T_amb:g};
        emissivity      {eps:g};
        kappaMethod     {kappa_method};
        value           $internalField;
    }}"""


# --------------------------------------------------------------------------- #
def build_device(case_dir: Path, p: DeviceParams) -> Path:
    case_dir = Path(case_dir)
    if case_dir.exists():
        shutil.rmtree(case_dir)
    lat = p.lattice()
    fluids, solids = p.regions()
    toks = dict(AIR, nIter=p.nIter, writeInterval=max(1, p.nIter // 10), k_solid=p.k_heater,
                qvol=p.qvol)
    # common: controlDict, top-level schemes, air + heater dictionaries
    _copy_render(TEMPLATES / "common", case_dir, toks)
    (case_dir / "system" / "blockMeshDict").write_text(
        lat.block_mesh_dict(f"wearable temple arm + skin, refine={p.refine}"))
    c = case_dir
    _w(c / "constant" / "regionProperties", "dictionary", "regionProperties",
       f"regions ( fluid ({' '.join(fluids)}) solid ({' '.join(solids)}) );\n")
    # solids other than the heater (heater comes from templates/common)
    _w(c / "constant" / "housing" / "thermophysicalProperties", "dictionary",
       "thermophysicalProperties", solid_thermo(HOUSING["k"], HOUSING["rho"], HOUSING["cp"]))
    for name, th, k, rho, cp, w, qm in TISSUE:
        _w(c / "constant" / name / "thermophysicalProperties", "dictionary",
           "thermophysicalProperties", solid_thermo(k, rho, cp))
        if w > 0 or qm > 0:
            _w(c / "constant" / name / "fvOptions", "dictionary", "fvOptions",
               pennes_source(w, qm, cp))
    heater_sys = case_dir / "system" / "heater"
    for s in solids:
        if s != "heater":
            shutil.copytree(heater_sys, case_dir / "system" / s, dirs_exist_ok=True)
    # parallel decomposition (same dict in every region)
    dec = f"numberOfSubdomains {p.nProcs};\nmethod          simple;\nsimpleCoeffs    {{ n ({p.nProcs} 1 1); delta 0.001; }}\n"
    for r in [""] + fluids + solids:
        _w(c / "system" / r / "decomposeParDict", "dictionary", "decomposeParDict", dec)
    write_fields(c, p)
    (c / "Allrun").write_text(ALLRUN.format(np=p.nProcs))
    (c / "Allrun").chmod(0o755)
    (c / "params.json").write_text(json.dumps(asdict(p)))
    return c


def _neighbours(region):
    """Coupled-patch neighbours of each region in this assembly."""
    tissue = [tl[0] for tl in TISSUE]
    nb = {"air": ["heater", "housing"], "heater": ["air", "housing"],
          "housing": ["air", "heater", "epidermis"]}
    for i, t in enumerate(tissue):
        nb[t] = ([tissue[i - 1]] if i > 0 else []) + ([tissue[i + 1]] if i + 1 < len(tissue) else [])
    nb["epidermis"].append("housing")
    return nb[region]


def write_fields(c: Path, p: DeviceParams):
    T0 = 305.0
    fluids, solids = p.regions()
    for reg in fluids + solids:
        km = "fluidThermo" if reg in fluids else "solidThermo"
        ents = []
        for nb in _neighbours(reg):
            Rc = p.R_contact if {reg, nb} == {"housing", "epidermis"} else None
            ents.append(f"    {reg}_to_{nb}\n    {coupled_T(km, Rc)}")
        ents.append("    symmetry        { type symmetryPlane; }")
        if reg == "air":
            ents += ["    inlet           { type fixedValue; value uniform %g; }" % p.T_amb,
                     "    outlet          { type inletOutlet; inletValue uniform %g; value $internalField; }" % p.T_amb]
        if reg == "housing":
            ents.append(f"    housingOuter\n    {ambient_T(p, p.eps_housing)}")
        if reg in [tl[0] for tl in TISSUE]:
            ents += [f"    skinExposed\n    {ambient_T(p, p.eps_skin)}",
                     "    tissueFar       { type zeroGradient; }",
                     f"    core            {{ type fixedValue; value uniform {p.T_core:g}; }}"]
        body = (f"dimensions      [0 0 0 1 0 0 0];\ninternalField   uniform {T0};\n\n"
                "boundaryField\n{\n" + "\n".join(ents) + "\n}\n")
        _w(c / "0.orig" / reg / "T", "volScalarField", "T", body)
        _w(c / "0.orig" / reg / "p", "volScalarField", "p",
           "dimensions      [1 -1 -2 0 0 0 0];\ninternalField   uniform 1e5;\n\nboundaryField\n{\n"
           "    \".*\"            { type calculated; value $internalField; }\n"
           "    symmetry        { type symmetryPlane; }\n}\n")
    U = p.U_in
    _w(c / "0.orig" / "air" / "U", "volVectorField", "U",
       f"dimensions      [0 1 -1 0 0 0 0];\ninternalField   uniform ({U} 0 0);\n\nboundaryField\n{{\n"
       f"    inlet           {{ type fixedValue; value uniform ({U} 0 0); }}\n"
       "    outlet          { type inletOutlet; inletValue uniform (0 0 0); value $internalField; }\n"
       "    \"air_to_.*\"     { type noSlip; }\n"
       "    symmetry        { type symmetryPlane; }\n}\n")
    _w(c / "0.orig" / "air" / "p_rgh", "volScalarField", "p_rgh",
       "dimensions      [1 -1 -2 0 0 0 0];\ninternalField   uniform 1e5;\n\nboundaryField\n{\n"
       "    inlet           { type fixedFluxPressure; value $internalField; }\n"
       "    outlet          { type fixedValue; value $internalField; }\n"
       "    \"air_to_.*\"     { type fixedFluxPressure; value $internalField; }\n"
       "    symmetry        { type symmetryPlane; }\n}\n")


ALLRUN = """#!/bin/sh
# Phase 2 device + skin model: mesh -> split into regions -> solve
cd "${{0%/*}}" || exit 1
set -e
blockMesh                              > log.blockMesh 2>&1
splitMeshRegions -cellZones -overwrite > log.splitMeshRegions 2>&1
rm -rf 0 && cp -r 0.orig 0
NP={np}
if [ "$NP" -gt 1 ]; then
    decomposePar -allRegions -force > log.decomposePar 2>&1
    mpirun --allow-run-as-root --oversubscribe -np $NP chtMultiRegionSimpleFoam -parallel > log.chtMultiRegionSimpleFoam 2>&1
    reconstructPar -allRegions > log.reconstructPar 2>&1
    rm -rf processor*
else
    chtMultiRegionSimpleFoam > log.chtMultiRegionSimpleFoam 2>&1
fi
echo "done: $(pwd)"
"""


# --------------------------------------------------------------------------- analysis
def _wall_flux(m, T, patch, k):
    """Heat flux leaving the region through ``patch`` (W, per modelled half) and face data."""
    sl = m.patch_slice(patch)
    Cf, Sf = m.face_geometry[0][sl], m.face_geometry[1][sl]
    A = np.linalg.norm(Sf, axis=1)
    n = Sf / A[:, None]
    cells = m.owner[sl]
    d = np.abs(np.einsum("ij,ij->i", Cf - m.C[cells], n))
    Tf = T.boundary[patch]
    q = k * (T.internal[cells] - Tf) / d          # W/m2, out of the region
    return (q * A).sum(), Tf, A, Cf


def analyse_device(case_dir, p: DeviceParams, time=None) -> dict:
    c = Case(case_dir)
    time = time or c.latest()
    rho, cp = AIR["rho_air"], AIR["cp_air"]
    kt = {t[0]: t[2] for t in TISSUE}
    out = {}
    # temperatures
    Th = c.field("T", "heater", time)
    out["T_heater_max_C"] = float(max(Th.internal.max(), *[v.max() for v in Th.boundary.values() if len(v)])) - 273.15
    me, Te = c.mesh("epidermis"), c.field("T", "epidermis", time)
    mh, Tho = c.mesh("housing"), c.field("T", "housing", time)
    Qskin_epi, Tskin, Askin, Cskin = _wall_flux(me, Te, "epidermis_to_housing", kt["epidermis"])
    Qskin, Ttouch, _, _ = _wall_flux(mh, Tho, "housing_to_epidermis", HOUSING["k"])
    out["T_skin_max_C"] = float(Tskin.max()) - 273.15
    out["T_skin_mean_C"] = float((Tskin * Askin).sum() / Askin.sum()) - 273.15
    out["T_touch_max_C"] = float(Ttouch.max()) - 273.15
    i = int(np.argmax(Tskin))
    out["skin_hotspot_xz_mm"] = [float(Cskin[i, 0] * 1e3), float(Cskin[i, 2] * 1e3)]
    Tout_h = Tho.boundary["housingOuter"]
    _, _, Ao, _ = _wall_flux(mh, Tho, "housingOuter", HOUSING["k"])
    out["T_housing_outer_max_C"] = float(Tout_h.max()) - 273.15
    Qamb = ((p.h_amb * (Tout_h - p.T_amb) + p.eps_housing * SIGMA * (Tout_h**4 - p.T_amb**4)) * Ao).sum()
    # air side
    ma, Ta_, Ua, Pa = c.mesh("air"), c.field("T", "air", time), c.field("U", "air", time), c.field("p", "air", time)

    def conv(patch):
        Sf = ma.patch_areas(patch)
        md = rho * np.einsum("ij,ij->i", Ua.boundary[patch], Sf)
        return md, (md * cp * Ta_.boundary[patch]).sum()
    md_in, H_in = conv("inlet")
    md_out, H_out = conv("outlet")
    ic = ma.patch_cells("inlet")
    d = ma.C[ic, 0] - ma.patch_centres("inlet")[:, 0]
    A = np.linalg.norm(ma.patch_areas("inlet"), axis=1)
    Q_cond_in = (K_AIR * (Ta_.internal[ic] - Ta_.boundary["inlet"]) / d * A).sum()
    Qair = H_in + H_out + Q_cond_in
    out["T_out_C"] = float((md_out * Ta_.boundary["outlet"]).sum() / md_out.sum()) - 273.15
    Ain, Aout = np.linalg.norm(ma.patch_areas("inlet"), axis=1), np.linalg.norm(ma.patch_areas("outlet"), axis=1)
    dp = (Pa.boundary["inlet"] * Ain).sum() / Ain.sum() - (Pa.boundary["outlet"] * Aout).sum() / Aout.sum()
    out["dp_Pa"] = float(dp)
    out["fan_mW"] = float(1e3 * dp * (-md_in.sum() / rho) * 2)
    # heat budget of the device (full device = 2 x half model)
    Qgen = p.qvol * c.mesh("heater").V.sum()
    out["Q_gen_W"] = float(2 * Qgen)
    out["Q_to_air_W"] = float(2 * Qair)
    out["Q_to_ambient_W"] = float(2 * Qamb)
    out["Q_to_skin_W"] = float(2 * Qskin)
    out["device_energy_err"] = float((Qair + Qamb + Qskin - Qgen) / max(Qgen, 1e-12)) if Qgen > 0 else float(Qair + Qamb + Qskin)
    out["contact_flux_mismatch"] = float(abs(Qskin + Qskin_epi) / max(abs(Qskin), 1e-12))
    res = final_residuals(Path(case_dir) / "log.chtMultiRegionSimpleFoam")
    out["max_residual"] = max(res.values()) if res else float("nan")
    out["n_cells"] = int(sum(c.mesh(r).n_cells for r in sum(p.regions(), [])))
    return out
