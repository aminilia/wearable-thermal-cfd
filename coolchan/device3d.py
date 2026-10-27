"""Phase 2: 3D wearable temple-arm model with the wearer's skin.

Assembly (half-width, symmetry plane at z = 0; x streamwise, y outward from skin):

    y  ^   ambient (h, emissivity)              housing top (PC)
       |   +-----------------------------------------------+
       |   |  air channel (fan-driven)    [heater block]   |  <- housing side wall at z = Wc..Wc+t
       |   +-----------------------------------------------+   housing bottom (PC), touches skin
     0 +===== epidermis / dermis / fat / inner tissue ==========  (Pennes bioheat)
       |                                                       core at 37 C (y = -depth)

* Heater: aluminium block (battery/SoC + spreader) on the skin-side housing wall.
* Housing <-> epidermis: thermal contact resistance R''. It is meshed as a thin solid
  region ("contact", CONTACT_MM thick) carved from the skin side of the housing wall, with
  a conductivity that gives exactly R'' plus the housing it replaces (see contact_kappa).
  A meshed layer behaves the same in every OpenFOAM release; the coupled BC's
  thicknessLayers/kappaLayers option does not (it was reworked after v1912).
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
CONTACT_MM = 0.05     # thickness of the meshed housing-skin contact layer


def contact_kappa(R_contact, t_mm=CONTACT_MM, k_host=HOUSING["k"]):
    """Conductivity of the contact layer: its t_mm of housing plus R'' in series."""
    d = t_mm * 1e-3
    return d / (R_contact + d / k_host)


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
    heater_side: str = "skin"    # "skin": on the skin-side wall; "outer": on the outer wall;
                                 # "centre": board-mounted in mid-channel (air on both sides)
    pad_mm: float = 0.0          # insulation pad between heater and skin-side wall ("skin" only)
    k_pad: float = 0.03          # polymer foam / aerogel-blanket class
    spreader_mm: float = 0.0     # aluminium spreader lining the inside of the outer wall (full
                                 # length), bonded to the heater: meshed as part of the heater
                                 # solid (same k), heat generated only in the heater block
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
        for top, c in self._channel_bands():
            ylines.append(top)
            ycells.append(n(c))
            ygrad.append(1)
        y = Axis(ylines, ycells, ygrad)
        tissue_names = [tl[0] for tl in TISSUE]

        def centre(ax, i):
            return 0.5 * (ax.lines[i] + ax.lines[i + 1])

        segs = self._channel_segments()

        def zone(i, j, k):
            xc, yc, zc = centre(x, i), centre(y, j), centre(z, k)
            if yc < 0:
                return tissue_names[j]
            if not (0 < xc < L and zc < self.Wc_mm / 2 + t):
                return None
            if yc < CONTACT_MM:
                return "contact"            # skin-side face of the housing, carries R''
            if yc < t or yc > t + H or zc > self.Wc_mm / 2:
                return "housing"
            kind = next(kd for lo, hi, kd, _ in segs if lo < yc < hi)
            if kind == "spreader":
                return "heater"         # one conducting solid with the heater (see spreader_mm)
            under = self.Lup_mm < xc < self.Lup_mm + self.Lb_mm and zc < self.Wb_mm / 2
            if kind in ("heater", "pad") and under:
                return kind
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
            if zn in ("housing", "heater", "contact"):  # heater: only the spreader ends at the vents
                return "housingOuter", "wall"
            if zn in tissue_names:
                if d == "-y" and j == 0:
                    return "core", "wall"
                if d == "+y":
                    return "skinExposed", "wall"
                return "tissueFar", "wall"
            raise ValueError(f"unclassified exterior face: zone {zn} block {(i, j, k)} dir {d}")

        return Lattice(x, y, z, zone, classify)

    # ------------------------------------------------------------- channel stack
    def _channel_segments(self):
        """(y_lo, y_hi, kind, base cell count) bands inside the housing cavity, bottom-up."""
        t, H, hb, pad, s = self.t_mm, self.H_mm, self.hb_mm, self.pad_mm, self.spreader_mm
        side = self.heater_side
        if pad and side != "skin":
            raise ValueError("an insulation pad is only modelled under a skin-side heater")
        top = t + H - s                      # top of the air cavity (below any spreader)
        if side == "skin":
            b = [(t, t + pad, "pad", max(3.0, pad / 0.1))] if pad else []
            b += [(t + pad, t + pad + hb, "heater", hb / 0.1),
                  (t + pad + hb, top, "air", (top - t - pad - hb) / 0.125)]
        elif side == "outer":
            b = [(t, top - hb, "air", (top - hb - t) / 0.125), (top - hb, top, "heater", hb / 0.1)]
        elif side == "centre":
            gap = (top - t - hb) / 2
            b = [(t, t + gap, "air", gap / 0.125), (t + gap, t + gap + hb, "heater", hb / 0.1),
                 (t + gap + hb, top, "air", gap / 0.125)]
        else:
            raise ValueError(side)
        if s:
            b.append((top, t + H, "spreader", max(3.0, s / 0.1)))
        if min(hi - lo for lo, hi, *_ in b) <= 0:
            raise ValueError("channel stack does not fit in the gap")
        return b

    def _channel_bands(self):
        t, H = self.t_mm, self.H_mm
        return [(CONTACT_MM, 1), (t, 4)] + [(hi, c) for lo, hi, kd, c in self._channel_segments()] + [(2 * t + H, 4)]

    @property
    def spreader_mass_g(self) -> float:
        """Full-device spreader mass (2700 kg/m3 aluminium)."""
        return 2700.0 * self.spreader_mm * self.L_mm * self.Wc_mm * 1e-9 * 1e3 if self.spreader_mm else 0.0

    def heater_box_m(self):
        """Bounding box of the heat-generating block [m] (half model, z >= 0)."""
        lo, hi, *_ = next(sg for sg in self._channel_segments() if sg[2] == "heater")
        return ((self.Lup_mm * 1e-3, lo * 1e-3, -1e-3),
                ((self.Lup_mm + self.Lb_mm) * 1e-3, hi * 1e-3, self.Wb_mm / 2 * 1e-3))

    def regions(self):
        solids = ["heater", "housing", "contact"]
        solids += ["pad"] if self.pad_mm else []
        return ["air"], solids + [tl[0] for tl in TISSUE]


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


def coupled_T(kappa_method):
    """Perfect thermal coupling between two regions (contact resistance is meshed instead)."""
    return f"""{{
        type            compressible::turbulentTemperatureCoupledBaffleMixed;
        Tnbr            T;
        kappaMethod     {kappa_method};
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
    _w(c / "constant" / "contact" / "thermophysicalProperties", "dictionary",
       "thermophysicalProperties",
       solid_thermo(contact_kappa(p.R_contact), HOUSING["rho"], HOUSING["cp"]))
    if p.pad_mm:
        _w(c / "constant" / "pad" / "thermophysicalProperties", "dictionary",
           "thermophysicalProperties", solid_thermo(p.k_pad, 100.0, 1300.0))
    if p.spreader_mm:
        # heat source only in the heater block, not in the bonded spreader
        _w(c / "constant" / "heater" / "fvOptions", "dictionary", "fvOptions", f"""heatSource
{{
    type            scalarSemiImplicitSource;
    active          true;
    selectionMode   cellSet;
    cellSet         heaterBlock;
    volumeMode      specific;
    injectionRateSuSp
    {{
        h           ({p.qvol:.10g} 0);
    }}
}}
""")
        b = p.heater_box_m()
        _w(c / "system" / "heater" / "topoSetDict", "dictionary", "topoSetDict", f"""actions
(
    {{
        name    heaterBlock;
        type    cellSet;
        action  new;
        source  boxToCell;
        sourceInfo {{ box ({b[0][0]:.9g} {b[0][1]:.9g} {b[0][2]:.9g}) ({b[1][0]:.9g} {b[1][1]:.9g} {b[1][2]:.9g}); }}
    }}
);
""")
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
    write_fields(c, p, lat.adjacency())
    if p.U_in == 0.0:
        # fan off: still air, no buoyancy -> solve energy only (pure conduction in the air)
        fv = c / "system" / "air" / "fvSolution"
        fv.write_text(fv.read_text().replace("momentumPredictor   yes;",
                                             "momentumPredictor   yes;\n    frozenFlow          yes;"))
    allrun = ALLRUN.format(np=p.nProcs)
    if p.spreader_mm:
        allrun = allrun.replace("rm -rf 0 && cp -r 0.orig 0\n",
                                "rm -rf 0 && cp -r 0.orig 0\ntopoSet -region heater > log.topoSet 2>&1\n")
    (c / "Allrun").write_text(allrun)
    (c / "Allrun").chmod(0o755)
    (c / "params.json").write_text(json.dumps(asdict(p)))
    return c


def write_fields(c: Path, p: DeviceParams, adjacency):
    T0 = 305.0
    fluids, solids = p.regions()
    for reg in fluids + solids:
        km = "fluidThermo" if reg in fluids else "solidThermo"
        ents = []
        nbs = sorted(next(iter(pr - {reg})) for pr in adjacency if reg in pr)
        for nb in nbs:
            ents.append(f"    {reg}_to_{nb}\n    {coupled_T(km)}")
        ents.append("    symmetry        { type symmetryPlane; }")
        if reg == "air":
            ents += ["    inlet           { type fixedValue; value uniform %g; }" % p.T_amb,
                     "    outlet          { type inletOutlet; inletValue uniform %g; value $internalField; }" % p.T_amb]
        if reg in ("housing", "contact") or (reg == "heater" and p.spreader_mm):
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


def _face_flux(m, T, patch, k):
    """Per-face heat flux [W/m2] leaving the region through ``patch``."""
    sl = m.patch_slice(patch)
    Cf, Sf = m.face_geometry[0][sl], m.face_geometry[1][sl]
    n = Sf / np.linalg.norm(Sf, axis=1)[:, None]
    cells = m.owner[sl]
    d = np.abs(np.einsum("ij,ij->i", Cf - m.C[cells], n))
    return k * (T.internal[cells] - T.boundary[patch]) / d


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
    mc, Tco = c.mesh("contact"), c.field("T", "contact", time)
    Qskin_epi, Tskin, Askin, Cskin = _wall_flux(me, Te, "epidermis_to_contact", kt["epidermis"])
    Qskin, Tcf, Acf, _ = _wall_flux(mc, Tco, "contact_to_epidermis", contact_kappa(p.R_contact))
    # housing surface at the skin = skin-side face + the jump q R'' across the contact
    qloc = np.zeros_like(Tcf) if p.R_contact == 0 else _face_flux(mc, Tco, "contact_to_epidermis",
                                                                    contact_kappa(p.R_contact))
    Ttouch = Tcf + qloc * p.R_contact
    out["T_skin_max_C"] = float(Tskin.max()) - 273.15
    out["T_skin_mean_C"] = float((Tskin * Askin).sum() / Askin.sum()) - 273.15
    out["T_touch_max_C"] = float(Ttouch.max()) - 273.15
    i = int(np.argmax(Tskin))
    out["skin_hotspot_xz_mm"] = [float(Cskin[i, 0] * 1e3), float(Cskin[i, 2] * 1e3)]
    Tout_h = Tho.boundary["housingOuter"]
    out["T_housing_outer_max_C"] = float(Tout_h.max()) - 273.15
    Qamb = 0.0
    for reg in ("housing", "heater", "contact"):
        mr, Tr = c.mesh(reg), c.field("T", reg, time)
        if "housingOuter" not in mr.patches or mr.patches["housingOuter"]["nFaces"] == 0:
            continue
        To = Tr.boundary["housingOuter"]
        Ao = np.linalg.norm(mr.patch_areas("housingOuter"), axis=1)
        Qamb += ((p.h_amb * (To - p.T_amb) + p.eps_housing * SIGMA * (To**4 - p.T_amb**4)) * Ao).sum()
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
    out["T_out_C"] = (float((md_out * Ta_.boundary["outlet"]).sum() / md_out.sum()) - 273.15
                      if abs(md_out.sum()) > 1e-15 else float("nan"))
    Ain, Aout = np.linalg.norm(ma.patch_areas("inlet"), axis=1), np.linalg.norm(ma.patch_areas("outlet"), axis=1)
    dp = (Pa.boundary["inlet"] * Ain).sum() / Ain.sum() - (Pa.boundary["outlet"] * Aout).sum() / Aout.sum()
    out["dp_Pa"] = float(dp)
    out["fan_mW"] = float(1e3 * dp * (-md_in.sum() / rho) * 2)
    # heat budget of the device (full device = 2 x half model)
    mhe = c.mesh("heater")
    (x0, y0, z0), (x1, y1, z1) = p.heater_box_m()
    Ch = mhe.C
    inb = ((Ch[:, 0] > x0) & (Ch[:, 0] < x1) & (Ch[:, 1] > y0) & (Ch[:, 1] < y1) & (Ch[:, 2] < z1))
    Qgen = p.qvol * mhe.V[inb].sum()
    out["Q_gen_W"] = float(2 * Qgen)
    out["Q_to_air_W"] = float(2 * Qair)
    out["Q_to_ambient_W"] = float(2 * Qamb)
    out["Q_to_skin_W"] = float(2 * Qskin)
    out["device_energy_err"] = float((Qair + Qamb + Qskin - Qgen) / max(Qgen, 1e-12)) if Qgen > 0 else float(Qair + Qamb + Qskin)
    out["contact_flux_mismatch"] = float(abs(Qskin + Qskin_epi) / max(abs(Qskin), 1e-12))
    res = final_residuals(Path(case_dir) / "log.chtMultiRegionSimpleFoam")
    out["max_residual"] = max(res.values()) if res else float("nan")
    out["n_cells"] = int(sum(c.mesh(r).n_cells for r in sum(p.regions(), [])))
    out["spreader_mass_g"] = p.spreader_mass_g
    return out


# --------------------------------------------------------------------------- limits
def surface_temperatures(case_dir, time=None) -> dict:
    """Face temperatures [K] on the two surfaces that carry a comfort/safety limit.

    skin     : every skin surface face (under the device and exposed)
    touch    : every outer housing face (what fingers touch when holding the arm)
    component: every heater cell and face (battery / SoC temperature)
    """
    c = Case(case_dir)
    Te = c.field("T", "epidermis", time)
    Th = c.field("T", "housing", time)
    Tc = c.field("T", "heater", time)
    Tk = c.field("T", "contact", time)
    skin = np.concatenate([Te.boundary["epidermis_to_contact"], Te.boundary["skinExposed"]])
    comp = np.concatenate([Tc.internal] + [Tc.boundary[k] for k in sorted(Tc.boundary)])
    touch = np.concatenate([Th.boundary["housingOuter"], Tk.boundary.get("housingOuter", [])])
    return dict(skin=skin, touch=touch, component=comp)


def max_power(case_idle, case_loaded, Q_ref, limits) -> dict:
    """Largest heater power for which every surface stays below its limit.

    Uses face-wise superposition T(x; Q) = T_idle(x) + Q r(x), exact for the linear
    parts of the model (conduction, convection, Pennes); grey-body radiation makes it
    approximate, so the result is checked by a CFD run at Q_max (see phase4 script).
    """
    a, b = surface_temperatures(case_idle), surface_temperatures(case_loaded)
    out = {}
    for key, lim in limits.items():
        T0, r = a[key], (b[key] - a[key]) / Q_ref
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(r > 0, (lim + 273.15 - T0) / r, np.inf)
        out[key] = float(max(q.min(), 0.0)) if np.all(T0 < lim + 273.15) else 0.0
    out["Q_max"] = min(out[k] for k in limits)
    out["binding"] = min(limits, key=lambda k: out[k])
    return out
