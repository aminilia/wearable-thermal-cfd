"""Unit tests that do not need OpenFOAM."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coolchan import case as C  # noqa: E402
from coolchan import foamio as F  # noqa: E402
from coolchan import surrogate as S  # noqa: E402
from coolchan.verification import gci, poiseuille_u, NU_FD  # noqa: E402


# ---------------------------------------------------------------- templates
def test_render_replaces_and_rejects_missing():
    assert C.render("a {{x}} b {{ y }}", {"x": 1, "y": 2.5}) == "a 1 b 2.5"
    with pytest.raises(KeyError):
        C.render("{{nope}}", {})


@pytest.mark.parametrize("builder,params", [(C.build_channel, C.ChannelParams()),
                                            (C.build_cht, C.ChtParams())])
def test_build_leaves_no_tokens(tmp_path, builder, params):
    d = builder(tmp_path / "case", params)
    files = [f for f in d.rglob("*") if f.is_file()]
    assert files
    for f in files:
        assert not re.search(r"\{\{", f.read_text()), f


def test_cht_block_counts_and_source():
    p = C.ChtParams(H_mm=3.0, dy_mm=0.05, aspect=4)
    t = p.tokens()
    assert t["nyB"] == 20 and t["nyGap"] == 40 and t["nxB"] == 100
    # q''' * block volume * depth == Q
    assert np.isclose(p.qvol * 20e-3 * 1e-3 * 20e-3, p.Q_W)
    with pytest.raises(ValueError):
        C.ChtParams(H_mm=0.9).tokens()


# ---------------------------------------------------------------- parsing
class _Mesh:
    n_cells = 3
    patches = {"inlet": {"nFaces": 2}, "wall": {"nFaces": 3}, "front": {"nFaces": 0}}


FIELD = """FoamFile { version 2.0; format ascii; class volVectorField; object U; }
dimensions [0 1 -1 0 0 0 0];
internalField nonuniform List<vector> 3((1 0 0) (2 0 0) (3 0.5 0));
boundaryField
{
    inlet { type fixedValue; value uniform (0.5 0 0); }
    wall  { type noSlip; value nonuniform List<vector> 3{(0 0 0)}; }
    front { type empty; }
}
"""

SCALAR = """FoamFile { version 2.0; format ascii; class volScalarField; object T; }
internalField   uniform 300;
boundaryField
{
    "inlet"  { type fixedValue; value uniform 290; }
    wall     { type externalWallHeatFluxTemperature; q uniform 100;
               value nonuniform List<scalar> 3 ( 301 302 303 ) ; }
}
"""


def test_read_vector_field(tmp_path):
    f = tmp_path / "U"
    f.write_text(FIELD)
    fld = F.read_field(f, _Mesh())
    assert fld.internal.shape == (3, 3) and fld.internal[2, 1] == 0.5
    assert np.allclose(fld.boundary["inlet"], [[0.5, 0, 0]] * 2)
    assert np.allclose(fld.boundary["wall"], 0) and fld.boundary["wall"].shape == (3, 3)


def test_read_scalar_field_ignores_q_keyword(tmp_path):
    f = tmp_path / "T"
    f.write_text(SCALAR)
    fld = F.read_field(f, _Mesh())
    assert np.allclose(fld.internal, 300)
    assert np.allclose(fld.boundary["inlet"], 290)
    assert np.allclose(fld.boundary["wall"], [301, 302, 303])


# ---------------------------------------------------------------- verification
def test_gci_recovers_order_and_limit():
    h = np.array([0.025, 0.05, 0.1])
    phi = 8.235 + 3.0 * h**2
    g = gci(*phi, *h)
    assert abs(g.p - 2) < 1e-8
    assert abs(g.phi_ext - 8.235) < 1e-10
    assert abs(g.asymptotic - 1) < 1e-2


def test_gci_non_uniform_ratio():
    h = np.array([1.0, 1.5, 2.7])
    phi = 1.0 + 0.2 * h**1.5
    g = gci(*phi, *h)
    assert abs(g.p - 1.5) < 1e-6 and abs(g.phi_ext - 1.0) < 1e-8


def test_poiseuille_mean_is_um():
    y = np.linspace(0, 2e-3, 20001)
    assert abs(np.trapezoid(poiseuille_u(y, 2e-3, 0.7), y) / 2e-3 - 0.7) < 1e-8
    assert abs(NU_FD - 8.2353) < 1e-4


# ---------------------------------------------------------------- surrogate
def _synthetic(n, seed):
    X = S.lhs(n, seed)
    df = pd.DataFrame(X, columns=S.NAMES)
    R = 12.0 + 20.0 / df.U_in**0.6 + 3.0 * df.H_mm + 1500.0 / df.k_solid
    df["T_in_C"] = 25.0
    df["T_max_C"] = 25.0 + df.Q_W * R
    df["dp_Pa"] = 12 * 1.846e-5 * df.U_in * 0.06 / (df.H_mm * 1e-3) ** 2
    return df


def test_lhs_bounds_and_stratification():
    X = S.lhs(30, 0)
    for j, (lo, hi) in enumerate(S.BOUNDS.values()):
        assert X[:, j].min() >= lo and X[:, j].max() <= hi
        bins = np.floor((X[:, j] - lo) / (hi - lo) * 30).astype(int)
        assert len(np.unique(np.clip(bins, 0, 29))) == 30   # one sample per stratum


def test_physics_informed_gp_beats_generic_on_linear_in_Q_data():
    tr, te = _synthetic(30, 1), _synthetic(200, 2)
    e_phys = np.abs(S.GPResistance().fit(tr).predict(te) - te.T_max_C).max()
    e_rsm = np.abs(S.QuadraticRSM().fit(tr).predict(te) - te.T_max_C).max()
    assert e_phys < 0.5
    assert e_phys < e_rsm


def test_exceedance_probability_monotone_in_velocity():
    gp = S.GPResistance().fit(_synthetic(30, 3))
    U = np.array([0.6, 1.5, 2.9])
    unc = S.Uncertainty(Q_nom=0.7)
    P, _, Qa = S.exceedance_probability(gp, U, np.full(3, 2.5), unc, n=4000)
    assert P[0] >= P[1] >= P[2]
    assert Qa[0] <= Qa[1] <= Qa[2]
    # at Q_nom = Q_allow the exceedance probability must equal the risk level
    P2, _, _ = S.exceedance_probability(gp, U[1:2], np.full(1, 2.5),
                                        S.Uncertainty(Q_nom=float(Qa[1])), n=4000)
    assert abs(P2[0] - 0.01) < 2e-3
