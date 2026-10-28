# gputherm: a GPU solver for conduction and bioheat in layered assemblies

The OpenFOAM model in this repository answers the wearable design question, but each 3D device run takes minutes on 2 cores, which is why Stage 3 could only afford 36 runs and a surrogate. `gputherm` is a purpose-built solver for the solid side of the problem (conduction, Pennes perfusion, contact resistance, convection and radiation boundaries) written in C++ and CUDA, so that full-physics design sweeps and Monte Carlo can run directly on a GPU.

It is general beyond wearables: any box-shaped, layered, multi-material assembly on a tensor-product grid (the grids `coolchan/lattice.py` already builds) fits, such as electronics packages, battery packs, or tissue heating.

## Equations and discretization

$$-\nabla\cdot(k\nabla T) + a\,T = s,\qquad a = w\rho_b c_b,\quad s = a\,T_a + q_m$$

* Cell-centred finite volumes on a structured grid with non-uniform spacing per axis.
* Region-wise $k$, harmonic face conductance, and contact resistance $R''$ between any pair of regions: $G = A / (d_P/2k_P + R'' + d_N/2k_N)$.
* Boundaries: adiabatic, fixed $T$, fixed flux, or convection plus radiation. Radiation is resolved by Picard iteration on $h_{rad} = \varepsilon\sigma(T_s^2+T_\infty^2)(T_s+T_\infty)$, which is exact at convergence.
* The result is a symmetric positive-definite 7-point stencil, solved matrix-free by Jacobi-preconditioned conjugate gradient. The CPU and GPU solvers apply the identical stencil to the identical assembled coefficients.

## Verification

`python python/verify_column.py` solves the same two 1D columns that Phase 2 uses to verify OpenFOAM's bioheat setup, against the same semi-analytic reference (`coolchan.bioheat.reference`), on three grids:

| Column | Cells | Max error | Surface error | Observed order |
|---|---|---|---|---|
| bare skin (convection + radiation) | 89 / 178 / 356 | 0.088 / 0.022 / **0.0055 mK** | 0.0054 mK | **2.00** |
| device (housing, contact resistance, flux) | 99 / 198 / 396 | 0.198 / 0.050 / **0.0124 mK** | 0.0124 mK | **2.00** |

The column extruded to 3 × 3 cells with adiabatic sides reproduces the 1D answer to within 10⁻⁵ mK, which checks the x/y stencil. The CUDA solver, run on a Colab GPU, reproduces the CPU errors digit for digit on all three grids for both columns (its 3 × 3 extrusion agrees with 1D to 7 × 10⁻⁶ mK), so the GPU port solves the same problem to the same accuracy.

## Results: CUDA v0 vs the OpenMP CPU reference

Same N³ device-on-skin problem and the same Jacobi-PCG, solved to a relative residual of 10⁻¹⁰, on a Google Colab T4 GPU and the Colab host CPU (OpenMP on all vCPUs). Raw numbers are in `results/bench_v0_colab.csv`.

| N | Cells | CPU solve [s] | GPU solve [s] | Speedup | GPU eff. bandwidth [GB/s] | \|T<sub>max</sub> diff\| [K] |
|---|---|---|---|---|---|---|
| 32 | 33 k | 0.199 | 0.045 | **4.5×** | 77 | 3 × 10⁻⁹ |
| 64 | 262 k | 2.93 | 0.299 | **9.8×** | 184 | 1 × 10⁻⁸ |
| 96 | 885 k | 16.1 | 1.23 | **13.1×** | 224 | 1 × 10⁻⁹ |
| 128 | 2.1 M | 59.5 | 3.53 | **16.9×** | 245 | 3 × 10⁻⁹ |
| 160 | 4.1 M | 143.8 | 8.35 | **17.2×** | 250 | 4 × 10⁻⁸ |

* **Correct:** CPU and GPU temperatures agree to 4 × 10⁻⁸ K or better at every size, and the CG iteration counts differ by at most 13 (reduction order changes rounding).
* **Memory-bound, as expected:** the GPU speedup grows with problem size as the launch overhead is amortised, and it levels off at about 250 GB/s. That is roughly 78 % of the T4's 320 GB/s peak, measured against a minimum-traffic model of 24 doubles per cell per iteration. It is an effective figure, not a profiler counter; Nsight Compute has not been run yet.
* **The CPU baseline is modest:** it reaches only 14–19 GB/s on the Colab host, so the speedups above are against a small shared CPU, not a tuned many-core node. Read them as "v0 vs this CPU", not as a general GPU advantage.
* **Headroom:** v0 still does three host round-trips per iteration and runs everything in FP64, which the T4 executes at 1/32 of its FP32 rate. Those are the first two roadmap items.

## Build and run

```bash
make cpu                      # g++ + OpenMP  -> bin/gputherm_cpu
make cuda ARCH=sm_75          # nvcc          -> bin/gputherm_cuda  (sm_75 = T4)
python python/verify_column.py --backend cpu      # or --backend cuda
python python/bench.py --sizes 32 64 128          # N^3 device-on-skin block
./bin/gputherm_cuda --bench 128                    # one run, JSON summary on stdout
./bin/gputherm_cpu case.spec --out T.bin           # any spec file (format in problem.hpp)
```

If no local GPU available, then, develop on a laptop, then run `notebooks/colab_gputherm.ipynb` on a free Colab or Kaggle GPU: it builds, verifies, benchmarks and profiles.

## Layout

```
include/gputherm/problem.hpp   spec reader, FV assembly, benchmark case, driver (shared)
src/cpu_main.cpp               reference Jacobi-PCG (OpenMP)
src/cuda_main.cu               v0 CUDA Jacobi-PCG
python/gputherm_io.py          write specs, run a binary, read T back
python/verify_column.py        verification against the semi-analytic bioheat solution
python/bench.py                CPU vs GPU timing and achieved bandwidth
notebooks/colab_gputherm.ipynb build + verify + benchmark + profile on a cloud GPU
```

## Roadmap

The v0 CUDA solver is a deliberately plain port, so every later step is measured against it. The solver is memory-bound, so progress is tracked as achieved DRAM bandwidth against the GPU's peak (roofline), plus wall-clock speedup.

1. **v0 (done): naive port.** One thread per cell, one kernel per vector operation, three dot products per iteration each copied to the host.
2. **Fewer round-trips.** Keep CG scalars on the device, fuse the x/r/z updates with the r·z and r·r reductions, and launch the loop as a CUDA Graph.
3. **Stencil bandwidth.** Compute face conductances on the fly from region ids instead of reading four coefficient arrays, and use 2.5D blocking (march in z, keep three planes in registers or shared memory).
4. **Precision.** FP32 stencil and vectors with FP64 reductions and iterative refinement. This matters most on a T4, whose FP64 rate is 1/32 of FP32.
5. **Better preconditioner.** Geometric multigrid on the tensor-product grid, to cut the iteration count that grows with N.
6. **Assembly on the GPU.** Move the Picard radiation update on device so nothing crosses PCIe inside the outer loop.
7. **Batched ensembles.** Solve thousands of parameter sets (gap, airflow-derived h, heater power, R'', tissue properties) in one launch, and run the Stage 3 Monte Carlo on full physics instead of a surrogate.
8. **Cross-check against Phase 2.** Drive the solver from `DeviceParams` with the air side represented by OpenFOAM-derived heat-transfer coefficients, and compare skin peaks with the 3D OpenFOAM model.
