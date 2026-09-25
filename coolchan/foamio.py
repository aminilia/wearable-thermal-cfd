"""Minimal, dependency-free readers for ASCII OpenFOAM output.

Post-processing deliberately avoids OpenFOAM function objects: everything is
computed here from ``constant/<region>/polyMesh`` and ``<time>/<region>/<field>``.
That keeps the analysis identical across OpenFOAM releases (the CI image and
a local install can differ) and makes every reported number traceable to a
few lines of NumPy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)


def _strip(text: str) -> str:
    text = _COMMENT.sub("", text)
    # drop the FoamFile header
    m = re.search(r"FoamFile\s*\{", text)
    if m:
        end = _match_brace(text, m.end() - 1)
        text = text[end + 1:]
    return text


def _match_brace(s: str, i: int, open_="{", close="}") -> int:
    """Index of the bracket closing the one at s[i]."""
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return j
    raise ValueError("unbalanced brackets")


def _parse_list(s: str, i: int):
    """Parse ``N ( ... )`` or ``N{v}`` starting at s[i]. Returns (array, end)."""
    m = re.compile(r"\s*(\d+)\s*").match(s, i)
    n = int(m.group(1))
    j = m.end()
    if s[j] == "{":  # compact uniform list  N{value}
        k = _match_brace(s, j)
        inner = s[j + 1:k].replace("(", " ").replace(")", " ")
        v = np.array(inner.split(), float)
        return (np.full(n, v[0]) if v.size == 1 else np.tile(v, (n, 1))), k + 1
    assert s[j] == "(", s[j:j + 40]
    k = _match_brace(s, j, "(", ")")
    inner = s[j + 1:k]
    arr = np.array(inner.replace("(", " ").replace(")", " ").split(), float)
    if n and arr.size != n:
        arr = arr.reshape(n, -1)
    return arr, k + 1


def _parse_value(s: str, i: int, ncells: int | None = None):
    s_ = s[i:].lstrip()
    off = len(s) - len(s_) - i
    if s_.startswith("uniform"):
        rest = s_[len("uniform"):].lstrip()
        if rest.startswith("("):
            k = rest.index(")")
            v = np.array(rest[1:k].split(), float)
        else:
            v = float(rest.split(";")[0])
        return ("uniform", v)
    if s_.startswith("nonuniform"):
        m = re.compile(r"nonuniform\s+List<\w+>").match(s_)
        arr, _ = _parse_list(s_, m.end())
        return ("nonuniform", arr)
    raise ValueError(f"cannot parse value near: {s_[:60]!r}")


def _expand(val, n):
    kind, v = val
    if kind == "uniform":
        v = np.atleast_1d(np.asarray(v, float))
        return np.full(n, v[0]) if v.size == 1 else np.tile(v, (n, 1))
    return v


@dataclass
class Field:
    internal: np.ndarray
    boundary: dict  # patch -> ndarray of face values (may be missing for empty)


def read_field(path: Path, mesh: "PolyMesh") -> Field:
    s = _strip(Path(path).read_text())
    m = re.search(r"\binternalField\b", s)
    internal = _expand(_parse_value(s, m.end()), mesh.n_cells)
    boundary = {}
    m = re.search(r"\bboundaryField\s*\{", s)
    body = s[m.end():_match_brace(s, m.end() - 1)]
    pos = 0
    ent = re.compile(r'\s*("?[^\s{}"]+"?)\s*\{')
    while True:
        mm = ent.search(body, pos)
        if not mm:
            break
        name = mm.group(1).strip('"')
        end = _match_brace(body, mm.end() - 1)
        block = body[mm.end():end]
        vm = re.search(r"(?<![A-Za-z_])value\s", block)
        if vm and name in mesh.patches:
            nf = mesh.patches[name]["nFaces"]
            boundary[name] = _expand(_parse_value(block, vm.end()), nf)
        pos = end + 1
    return Field(internal, boundary)


class PolyMesh:
    """ASCII polyMesh with face/cell geometry computed in NumPy."""

    def __init__(self, mesh_dir: Path):
        d = Path(mesh_dir)
        self.dir = d
        self.points, _ = _parse_list(_strip((d / "points").read_text()), 0)
        self.points = self.points.reshape(-1, 3)
        fs = _strip((d / "faces").read_text())
        self.faces = self._read_faces(fs)
        self.owner = self._read_labels(d / "owner")
        self.neighbour = self._read_labels(d / "neighbour")
        self.patches = self._read_boundary(d / "boundary")
        self.n_cells = int(max(self.owner.max(), self.neighbour.max(initial=-1)) + 1)

    # ---------------------------------------------------------------- read
    @staticmethod
    def _read_labels(p: Path) -> np.ndarray:
        arr, _ = _parse_list(_strip(p.read_text()), 0)
        return np.asarray(arr, int).ravel()

    @staticmethod
    def _read_faces(s: str):
        m = re.compile(r"\s*(\d+)\s*\(").match(s)
        body = s[m.end():s.rindex(")")]
        faces = [list(map(int, f.split())) for f in re.findall(r"\d+\s*\(([^)]*)\)", body)]
        n = int(m.group(1))
        assert len(faces) == n, (len(faces), n)
        if all(len(f) == 4 for f in faces):
            return np.array(faces, int)
        return faces

    @staticmethod
    def _read_boundary(p: Path) -> dict:
        s = _strip(p.read_text())
        s = s[s.index("(") + 1:s.rindex(")")]
        out = {}
        for m in re.finditer(r"(\w+)\s*\{([^}]*)\}", s):
            d = dict(re.findall(r"(\w+)\s+([^;]+);", m.group(2)))
            out[m.group(1)] = dict(type=d["type"], nFaces=int(d["nFaces"]), startFace=int(d["startFace"]))
        return out

    # ------------------------------------------------------------ geometry
    @cached_property
    def face_geometry(self):
        """Face centres and area vectors (triangle fan about the vertex mean)."""
        if isinstance(self.faces, np.ndarray):
            P = self.points[self.faces]                   # (nf, 4, 3)
            c0 = P.mean(axis=1)
            nxt = np.roll(P, -1, axis=1)
            tri_c = (P + nxt + c0[:, None, :]) / 3.0
            tri_a = 0.5 * np.cross(P - c0[:, None, :], nxt - c0[:, None, :])
            mag = np.linalg.norm(tri_a, axis=2)
            Sf = tri_a.sum(axis=1)
            Cf = (tri_c * mag[..., None]).sum(axis=1) / mag.sum(axis=1)[:, None]
            return Cf, Sf
        Cf, Sf = [], []
        for f in self.faces:
            P = self.points[f]
            c0 = P.mean(axis=0)
            nxt = np.roll(P, -1, axis=0)
            tc = (P + nxt + c0) / 3.0
            ta = 0.5 * np.cross(P - c0, nxt - c0)
            mag = np.linalg.norm(ta, axis=1)
            Sf.append(ta.sum(axis=0))
            Cf.append((tc * mag[:, None]).sum(axis=0) / mag.sum())
        return np.array(Cf), np.array(Sf)

    @cached_property
    def cell_geometry(self):
        """Cell centroids and volumes by pyramid decomposition (as OpenFOAM)."""
        Cf, Sf = self.face_geometry
        nf_int = len(self.neighbour)
        own, nei = self.owner, self.neighbour
        # estimated centre = mean of face centres
        cnt = np.bincount(own, minlength=self.n_cells) + np.bincount(nei, minlength=self.n_cells)
        cEst = np.zeros((self.n_cells, 3))
        for k in range(3):
            cEst[:, k] = (np.bincount(own, Cf[:, k], self.n_cells)
                          + np.bincount(nei, Cf[:nf_int, k], self.n_cells)) / cnt
        def pyr(cells, sign, idx):
            h = np.einsum("ij,ij->i", Sf[idx], Cf[idx] - cEst[cells]) * sign
            v = h / 3.0
            cc = 0.75 * Cf[idx] + 0.25 * cEst[cells]
            return v, cc
        vo, co = pyr(own, 1.0, np.arange(len(own)))
        vn, cn = pyr(nei, -1.0, np.arange(nf_int))
        V = np.bincount(own, vo, self.n_cells) + np.bincount(nei, vn, self.n_cells)
        C = np.zeros((self.n_cells, 3))
        for k in range(3):
            C[:, k] = (np.bincount(own, vo * co[:, k], self.n_cells)
                       + np.bincount(nei, vn * cn[:, k], self.n_cells)) / V
        return C, V

    @property
    def C(self):
        return self.cell_geometry[0]

    @property
    def V(self):
        return self.cell_geometry[1]

    def patch_slice(self, name: str) -> slice:
        p = self.patches[name]
        return slice(p["startFace"], p["startFace"] + p["nFaces"])

    def patch_centres(self, name):
        return self.face_geometry[0][self.patch_slice(name)]

    def patch_areas(self, name):
        return self.face_geometry[1][self.patch_slice(name)]

    def patch_cells(self, name):
        return self.owner[self.patch_slice(name)]


# --------------------------------------------------------------------------- #
class Case:
    """Convenience wrapper: ``Case(dir).field('T', region='air')``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._mesh = {}

    def times(self) -> list[str]:
        ts = []
        for d in self.path.iterdir():
            try:
                float(d.name)
            except ValueError:
                continue
            if d.is_dir():
                ts.append(d.name)
        return sorted(ts, key=float)

    def latest(self) -> str:
        return self.times()[-1]

    def mesh(self, region: str = "") -> PolyMesh:
        if region not in self._mesh:
            self._mesh[region] = PolyMesh(self.path / "constant" / region / "polyMesh")
        return self._mesh[region]

    def field(self, name: str, region: str = "", time: str | None = None) -> Field:
        time = time or self.latest()
        return read_field(self.path / time / region / name, self.mesh(region))


# --------------------------------------------------------------------------- #
_RES = re.compile(r"Solving for (\w+), Initial residual = ([0-9.eE+-]+)")


def final_residuals(log: Path, tail_lines: int = 400) -> dict:
    """Initial residuals of the last iteration in a solver log (max over regions)."""
    lines = Path(log).read_text().splitlines()
    idx = [i for i, l in enumerate(lines) if l.startswith("Time = ")]
    block = lines[idx[-2]:idx[-1]] if len(idx) > 1 else lines[-tail_lines:]
    out: dict = {}
    for l in block:
        m = _RES.search(l)
        if m:
            out[m.group(1)] = max(out.get(m.group(1), 0.0), float(m.group(2)))
    return out
