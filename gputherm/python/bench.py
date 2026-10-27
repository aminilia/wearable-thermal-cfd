"""Benchmark gputherm on the N^3 device-on-skin block for each available backend.

Reports time per CG iteration and the effective memory bandwidth it implies. The
solver is memory-bound, so bandwidth against the GPU's peak is the honest figure of
merit (the roofline argument), not FLOP/s.

    python python/bench.py --sizes 32 64 128 192 256
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gputherm_io import BIN, bench  # noqa: E402

# Minimum DRAM traffic per cell per Jacobi-PCG iteration in the v0 kernels, in doubles:
#   stencil  diag,gx,gy,gz,p read + Ap write ........ 6
#   x += a p, r -= a Ap  (4 read, 2 write) .......... 6
#   z = r / diag  (2 read, 1 write) ................. 3
#   p = z + b p   (2 read, 1 write) ................. 3
#   three dot products (2 reads each) ............... 6
BYTES_PER_CELL_ITER = 24 * 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--backends", nargs="+", default=["cpu", "cuda"])
    ap.add_argument("--csv", type=Path, default=Path("bench_results.csv"))
    ap.add_argument("--plot", type=Path, default=Path("bench_results.png"))
    a = ap.parse_args()
    backends = [b for b in a.backends if (BIN / f"gputherm_{b}").exists()]
    if not backends:
        sys.exit("no solver binaries found; run `make cpu` and/or `make cuda`")

    rows = []
    for N in a.sizes:
        for b in list(backends):
            try:
                r = bench(N, b)
            except Exception as e:  # e.g. CUDA binary present but no GPU on this machine
                err = getattr(e, "stderr", "") or str(e)
                print(f"skipping backend {b}: {err.strip().splitlines()[-1]}")
                backends.remove(b)
                continue
            gbs = BYTES_PER_CELL_ITER * r["cells"] / (r["t_per_cg_iter_ms"] * 1e-3) / 1e9
            rows.append(dict(N=N, backend=b, cells=r["cells"], cg_iters=r["cg_iters"],
                             t_solve_s=r["t_solve_s"], t_assemble_s=r["t_assemble_s"],
                             ms_per_iter=r["t_per_cg_iter_ms"], eff_GBs=gbs, T_max=r["T_max"]))
            print(f"N={N:4d} {b:5s} cells={r['cells']:>10,d} iters={r['cg_iters']:>6d} "
                  f"solve={r['t_solve_s']:8.3f}s  {r['t_per_cg_iter_ms']:8.3f} ms/iter  "
                  f"~{gbs:6.1f} GB/s  Tmax={r['T_max']:.4f} K", flush=True)
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    if {"cpu", "cuda"} <= set(backends):
        print("\nspeedup (CPU / GPU solve time):")
        for N in a.sizes:
            c = next(r for r in rows if r["N"] == N and r["backend"] == "cpu")
            g = next(r for r in rows if r["N"] == N and r["backend"] == "cuda")
            print(f"  N={N:4d}  {c['t_solve_s'] / g['t_solve_s']:6.1f}x   "
                  f"|Tmax diff| = {abs(c['T_max'] - g['T_max']):.2e} K")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for b in backends:
        rr = [r for r in rows if r["backend"] == b]
        ax[0].loglog([r["cells"] for r in rr], [r["ms_per_iter"] for r in rr], "o-", label=b)
        ax[1].semilogx([r["cells"] for r in rr], [r["eff_GBs"] for r in rr], "o-", label=b)
    ax[0].set(xlabel="cells", ylabel="ms per CG iteration", title="Cost per iteration")
    ax[1].set(xlabel="cells", ylabel="effective GB/s", title="Achieved memory bandwidth")
    for x in ax:
        x.grid(True, which="both", alpha=0.3)
        x.legend()
    fig.tight_layout()
    fig.savefig(a.plot, dpi=150)
    print(f"\nwrote {a.csv} and {a.plot}")


if __name__ == "__main__":
    main()
