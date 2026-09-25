"""Run case ``Allrun`` scripts with an OpenFOAM environment, serially or in a pool.

The OpenFOAM environment is found in this order:

0. ``$COOLCHAN_DOCKER_IMAGE`` - run every case inside that image instead, e.g.
   ``opencfd/openfoam-default:2406`` (useful on macOS/Windows)
1. ``$COOLCHAN_FOAM_BASHRC`` - explicit path to an OpenFOAM ``etc/bashrc``
2. an already-sourced environment (``$WM_PROJECT_DIR`` set)
3. ``/usr/lib/openfoam/openfoam*/etc/bashrc`` (openfoam.com packages/Docker image)
4. ``/usr/share/openfoam`` (Debian/Ubuntu ``openfoam`` package)
"""
from __future__ import annotations

import glob
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def foam_env_prefix() -> str:
    rc = os.environ.get("COOLCHAN_FOAM_BASHRC")
    if rc:
        return f'. "{rc}" >/dev/null 2>&1 || true; '
    if os.environ.get("WM_PROJECT_DIR"):
        return ""
    cands = sorted(glob.glob("/usr/lib/openfoam/openfoam*/etc/bashrc"))
    if cands:
        return f'. "{cands[-1]}" >/dev/null 2>&1 || true; '
    if Path("/usr/share/openfoam/etc").is_dir():
        return "export WM_PROJECT_DIR=/usr/share/openfoam; "
    raise RuntimeError("No OpenFOAM installation found; set COOLCHAN_FOAM_BASHRC")


def run_case(case_dir: Path, script: str = "Allrun", timeout: float | None = None) -> float:
    """Run ``case_dir/script``; returns wall time in seconds. Raises on failure."""
    case_dir = Path(case_dir).resolve()
    t0 = time.time()
    image = os.environ.get("COOLCHAN_DOCKER_IMAGE")
    if image:
        inner = ('for f in /usr/lib/openfoam/openfoam*/etc/bashrc; do . "$f"; done '
                 f'>/dev/null 2>&1; ./{script}')
        argv = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
                "-e", "HOME=/tmp", "-v", f"{case_dir}:/case", "-w", "/case",
                "--entrypoint", "bash", image, "-c", inner]
    else:
        argv = ["bash", "-c", foam_env_prefix() + f'"{case_dir / script}"']
    res = subprocess.run(argv, cwd=case_dir, capture_output=True, text=True,
                         timeout=timeout)
    if res.returncode != 0:
        logs = sorted(case_dir.glob("log.*"), key=os.path.getmtime)
        tail = logs[-1].read_text()[-3000:] if logs else res.stderr
        raise RuntimeError(f"{case_dir.name}: {script} failed\n{tail}")
    return time.time() - t0


def run_many(case_dirs, workers: int | None = None, script: str = "Allrun"):
    """Run independent cases concurrently (each case is serial OpenFOAM)."""
    workers = workers or max(1, os.cpu_count() or 1)
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_case, d, script): d for d in case_dirs}
        for f in as_completed(futs):
            d = futs[f]
            try:
                out[Path(d).name] = f.result()
                print(f"  [ok] {Path(d).name}  {out[Path(d).name]:.0f}s", flush=True)
            except Exception as e:  # keep going; report at the end
                out[Path(d).name] = e
                print(f"  [FAIL] {Path(d).name}: {str(e).splitlines()[0]}", flush=True)
    return out
