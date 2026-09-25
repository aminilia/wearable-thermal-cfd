"""Tests for the backward-facing-step validation helpers (no OpenFOAM needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coolchan import bfs  # noqa: E402


def test_duct_profile_wide_duct_tends_to_plane_poiseuille():
    a, b = 1.0, 40.0                       # aspect ratio 40
    y = np.linspace(-a, a, 201)
    u = bfs.duct_profile(y, np.zeros_like(y), a, b)
    u /= u.max()
    assert np.allclose(u, 1 - y**2, atol=2e-3)


def test_duct_profile_no_slip_on_all_walls():
    a, b = 0.5, 2.0
    assert abs(bfs.duct_profile(np.array([a]), np.array([0.3]), a, b)[0]) < 1e-6
    assert abs(bfs.duct_profile(np.array([0.1]), np.array([b]), a, b)[0]) < 1e-6


def test_square_duct_centre_to_mean_ratio():
    # square duct: u_max / u_mean = 2.0962 (Shah & London)
    a = b = 1.0
    n = 400
    g = (np.arange(n) + 0.5) / n * 2 - 1
    Y, Z = np.meshgrid(g * a, g * b)
    u = bfs.duct_profile(Y.ravel(), Z.ravel(), a, b)
    assert abs(u.max() / u.mean() - 2.0962) < 5e-3


@pytest.mark.parametrize("w", [None, 90.9])
def test_block_mesh_dict_patches(w):
    p = bfs.BFSParams(Re=300, W_half_mm=w)
    d = bfs.block_mesh_dict(p)
    for patch in ("inlet", "outlet", "bottomWall", "stepWall", "topWall", "inletBottomWall"):
        assert patch in d
    if w is None:
        assert "frontAndBack" in d and "midPlane" not in d
    else:
        assert "midPlane" in d and "sideWall" in d
    assert p.n_cells() > 0


def test_reynolds_definition():
    p = bfs.BFSParams(Re=389.0)
    assert np.isclose(p.U_mean * 2 * p.h_mm * 1e-3 / bfs.NU_AIR, 389.0)
    assert abs(p.ER - 1.942) < 1e-3


def test_crossings_interpolates_zero():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    u = np.array([-1.0, -1.0, 1.0, 1.0])
    (xc, d), = bfs._crossings(x, u)
    assert np.isclose(xc, 1.5) and d == 1
