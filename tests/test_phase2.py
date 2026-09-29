"""Phase 2 helpers: lattice mesher, device geometry, bioheat reference solution."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coolchan import bioheat, device3d  # noqa: E402
from coolchan.lattice import Axis, Lattice  # noqa: E402


def test_lattice_skips_void_and_counts_cells():
    lat = Lattice(Axis([0, 1, 2], [2, 3]), Axis([0, 1], [4]), Axis([0, 1], [1]),
                  zone=lambda i, j, k: "a" if i == 0 else None,
                  classify=lambda i, j, k, d: ("wall", "wall"))
    d = lat.block_mesh_dict()
    assert d.count("hex (") == 1
    assert lat.n_cells() == 2 * 4 * 1


@pytest.mark.parametrize("side", ["skin", "outer"])
def test_device_lattice_every_exterior_face_classified(side):
    p = device3d.DeviceParams(heater_side=side)
    lat = p.lattice()
    txt = lat.block_mesh_dict()                  # raises if a face is unclassified
    for patch in ("inlet", "outlet", "housingOuter", "core", "skinExposed", "symmetry"):
        assert patch in txt
    zones = {lat.zone(i, j, k) for i in range(lat.x.n) for j in range(lat.y.n) for k in range(lat.z.n)}
    assert {"air", "heater", "housing", "epidermis", "dermis", "fat", "inner"} <= zones


def test_device_heat_source_integrates_to_Q():
    p = device3d.DeviceParams(Q_W=0.7)
    assert math.isclose(p.qvol * p.Lb_mm * p.hb_mm * p.Wb_mm * 1e-9, 0.7)


def test_pennes_source_linearisation():
    # S(T) = Su + Sp * cp * (T - TSTD) must equal w rho_b c_b (Ta - T) + qm for any T
    w, qm, cp = 1.25e-3, 370.0, 3300.0
    txt = device3d.pennes_source(w, qm, cp)
    su, sp = map(float, txt.split("h           (")[1].split(")")[0].split())
    for T in (300.0, 310.15, 320.0):
        S = su + sp * cp * (T - device3d.TSTD)
        exact = w * device3d.BLOOD["rho"] * device3d.BLOOD["cp"] * (device3d.BLOOD["T"] - T) + qm
        assert abs(S - exact) < 1e-3          # W/m3; Su ~ 1.5e6 so this is ~1e-9 relative


def test_reference_matches_closed_form_single_layer(monkeypatch):
    # one homogeneous perfused layer, convection only on top: closed-form cosh/sinh solution
    k, w, th = 0.5, 1.25e-3, 10.0
    monkeypatch.setattr(bioheat, "TISSUE", [("inner", th, k, 1000.0, 4000.0, w, 0.0)])
    p = bioheat.ColumnParams(kind="bare", h_amb=8.0, eps=0.0)
    ref = bioheat.reference(p)
    a = w * device3d.BLOOD["rho"] * device3d.BLOOD["cp"]
    m, L, Tb = math.sqrt(a / k), th * 1e-3, device3d.BLOOD["T"]
    A = p.T_core - Tb
    # -k T'(L) = h (T(L) - Tamb),  theta = A cosh(my) + B sinh(my)
    c, s = math.cosh(m * L), math.sinh(m * L)
    B = (-k * A * m * s - p.h_amb * (A * c + Tb - p.T_amb)) / (k * m * c + p.h_amb * s)
    T_top = Tb + A * c + B * s
    assert abs(ref["T_top"] - T_top) < 1e-6


# ---------------------------------------------------------------- Phase 4 geometry
@pytest.mark.parametrize("kw,expect", [
    (dict(pad_mm=0.5), {"pad"}),
    (dict(heater_side="outer", spreader_mm=0.3), set()),
    (dict(heater_side="centre"), set()),
])
def test_phase4_architectures_have_expected_regions(kw, expect):
    p = device3d.DeviceParams(**kw)
    lat = p.lattice()
    lat.block_mesh_dict()
    zones = {lat.zone(i, j, k) for i in range(lat.x.n) for j in range(lat.y.n) for k in range(lat.z.n)}
    assert expect <= zones
    assert set(p.regions()[1]) >= expect
    pairs = lat.adjacency()
    if "pad" in expect:           # pad separates heater from the skin-side wall
        assert frozenset(("heater", "housing")) not in pairs
        assert frozenset(("heater", "pad")) in pairs and frozenset(("pad", "housing")) in pairs
    if kw.get("spreader_mm"):      # spreader is part of the heater solid and lines the outer wall
        assert "spreader" not in p.regions()[1]
        assert frozenset(("heater", "housing")) in pairs
        (x0, y0, _), (x1, y1, z1) = p.heater_box_m()
        assert math.isclose((x1 - x0) * (y1 - y0) * z1 * 2, p.Lb_mm * p.hb_mm * p.Wb_mm * 1e-9)
    if kw.get("heater_side") == "centre":
        assert frozenset(("heater", "housing")) not in pairs


def test_pad_only_under_skin_side_heater():
    with pytest.raises(ValueError):
        device3d.DeviceParams(heater_side="outer", pad_mm=0.5).lattice()


def test_spreader_mass():
    p = device3d.DeviceParams(spreader_mm=0.3)
    assert math.isclose(p.spreader_mass_g, 2700 * 0.3e-3 * 60e-3 * 12e-3 * 1e3)
