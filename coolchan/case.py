"""Build OpenFOAM case directories from the token templates in ``templates/``.

A case is assembled from ``templates/common`` (region dictionaries shared by all
cases) plus a case-specific overlay (``templates/channel`` or ``templates/cht``).
Every ``{{token}}`` in every file is replaced; an unreplaced token is an error,
so a typo in a parameter name can never silently fall back to a default.
"""
from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "templates"
TOKEN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

# Dry air at 300 K (Incropera, Table A.4) - held constant in every run.
AIR = dict(rho_air=1.1614, cp_air=1007.0, mu_air=1.846e-5, Pr_air=0.707)
K_AIR = AIR["mu_air"] * AIR["cp_air"] / AIR["Pr_air"]  # 0.02629 W/m/K


def render(text: str, params: dict) -> str:
    def sub(m):
        key = m.group(1)
        if key not in params:
            raise KeyError(f"template token '{{{{{key}}}}}' has no value")
        return _fmt(params[key])

    return TOKEN.sub(sub, text)


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v)


def _copy_render(src: Path, dst: Path, params: dict, skip_regions=()):
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src)
        if any(part in skip_regions for part in rel.parts):
            continue
        out = dst / rel
        if f.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        text = f.read_text()
        out.write_text(render(text, params))
        if f.stat().st_mode & 0o111:
            out.chmod(f.stat().st_mode)


# --------------------------------------------------------------------------- #
# Stage 1: parallel-plate channel
# --------------------------------------------------------------------------- #
@dataclass
class ChannelParams:
    """Parallel-plate channel, both plates heated with the same flux."""

    H_mm: float = 2.0          # full plate spacing
    L_mm: float = 100.0        # channel length (L/Dh = 25)
    U_in: float = 0.5          # uniform inlet velocity [m/s]
    q_wall: float = 100.0      # wall heat flux [W/m^2]; 0 -> cold run
    T_in: float = 300.0        # inlet temperature [K]
    ny: int = 20               # cells across the gap
    aspect: float = 5.0        # dx/dy (cells are stretched streamwise)
    nIter: int = 4000

    @property
    def nx(self) -> int:
        dy = self.H_mm / self.ny
        return int(round(self.L_mm / (self.aspect * dy)))

    @property
    def Dh(self) -> float:
        return 2.0 * self.H_mm * 1e-3

    @property
    def Re(self) -> float:
        return AIR["rho_air"] * self.U_in * self.Dh / AIR["mu_air"]

    def tokens(self) -> dict:
        t = asdict(self)
        t.update(AIR)
        t.update(nx=self.nx, dz_mm=self.H_mm / self.ny,
                 writeInterval=max(1, self.nIter // 10))
        return t


def build_channel(case_dir: Path, p: ChannelParams) -> Path:
    case_dir = Path(case_dir)
    if case_dir.exists():
        shutil.rmtree(case_dir)
    toks = p.tokens()
    _copy_render(TEMPLATES / "common", case_dir, toks, skip_regions=("heater",))
    _copy_render(TEMPLATES / "channel", case_dir, toks)
    return case_dir


# --------------------------------------------------------------------------- #
# Stage 2/3: heated aluminium block in a channel (conjugate heat transfer)
# --------------------------------------------------------------------------- #
@dataclass
class ChtParams:
    """Heated block on the bottom wall of a thin air channel.

    Geometry (x streamwise, y across the gap), all in mm::

        y=H  +--------------------------------------------------+  top wall (adiabatic)
             |  air                                              |
        inlet|              +==============+                     | outlet
        y=hb |              |  heater      |                     |
        y=0  +--------------+==============+---------------------+  bottom wall (adiabatic)
             0            Lup          Lup+Lb                   Lup+Lb+Ldown

    The 2D model represents a slice of a channel of spanwise depth ``W_mm``;
    the block power ``Q_W`` is converted to a volumetric source over
    Lb * hb * W.
    """

    U_in: float = 1.5          # inlet (fan) velocity [m/s]
    H_mm: float = 3.0          # channel gap (wall to wall)
    Q_W: float = 0.5           # block power [W]
    k_solid: float = 200.0     # block conductivity [W/m/K]
    T_in: float = 298.15       # inlet air temperature [K] (25 C)
    hb_mm: float = 1.0         # block height
    Lb_mm: float = 20.0        # block length
    Lup_mm: float = 10.0       # upstream run
    Ldown_mm: float = 30.0     # downstream run
    W_mm: float = 20.0         # spanwise depth represented by the 2D slice
    dy_mm: float = 0.05        # target cell height
    aspect: float = 4.0        # dx/dy
    nIter: int = 3000

    # ----- derived --------------------------------------------------------
    @property
    def qvol(self) -> float:
        return self.Q_W / (self.Lb_mm * self.hb_mm * self.W_mm * 1e-9)

    @property
    def clearance_mm(self) -> float:
        return self.H_mm - self.hb_mm

    @property
    def Dh(self) -> float:
        return 2.0 * self.H_mm * 1e-3

    @property
    def Re(self) -> float:
        return AIR["rho_air"] * self.U_in * self.Dh / AIR["mu_air"]

    @property
    def mdot(self) -> float:
        """Mass flow through the full-depth channel [kg/s]."""
        return AIR["rho_air"] * self.U_in * self.H_mm * self.W_mm * 1e-6

    def tokens(self) -> dict:
        if self.H_mm <= self.hb_mm:
            raise ValueError("gap must be larger than block height")
        dy, dx = self.dy_mm, self.aspect * self.dy_mm
        n = lambda length, d: max(2, int(math.ceil(length / d - 1e-9)))
        t = asdict(self)
        t.update(AIR)
        t.update(
            qvol=self.qvol,
            x1=self.Lup_mm,
            x2=self.Lup_mm + self.Lb_mm,
            x3=self.Lup_mm + self.Lb_mm + self.Ldown_mm,
            nxUp=n(self.Lup_mm, dx),
            nxB=n(self.Lb_mm, dx),
            nxDown=n(self.Ldown_mm, dx),
            nyB=n(self.hb_mm, dy),
            nyGap=n(self.clearance_mm, dy),
            dz_mm=dy,
            writeInterval=max(1, self.nIter // 10),
        )
        return t


def build_cht(case_dir: Path, p: ChtParams) -> Path:
    case_dir = Path(case_dir)
    if case_dir.exists():
        shutil.rmtree(case_dir)
    toks = p.tokens()
    _copy_render(TEMPLATES / "common", case_dir, toks)
    _copy_render(TEMPLATES / "cht", case_dir, toks)
    return case_dir
