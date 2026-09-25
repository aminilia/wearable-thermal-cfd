| Mesh | Cells | T<sub>max</sub> solid [°C] | T<sub>out</sub> air [°C] | Δp [Pa] | Fan power [mW] | Energy imbalance | Iterative ΔT<sub>max</sub> [K] |
|---|---|---|---|---|---|---|---|
| coarse (Δy = 0.1 mm) | 4,500 | 48.977 | 29.750 | 5.785 | 0.521 | -0.070 % | 0.0e+00 |
| medium (Δy = 0.05 mm) | 18,000 | 48.924 | 29.750 | 5.821 | 0.524 | -0.018 % | 4.0e-07 |
| fine (Δy = 0.025 mm) | 72,000 | 48.913 | 29.748 | 5.842 | 0.526 | -0.037 % | 2.1e-05 |

| Output | Observed order p | Richardson extrapolation | GCI<sub>fine</sub> | GCI<sub>medium</sub> |
|---|---|---|---|---|
| T<sub>max</sub> solid [°C] | 2.20 | 48.910 | 0.02 % | 0.08 % |
| T<sub>out</sub> air [°C] | 5.97 | 29.748 | 0.00 % | 0.00 % |
| Δp [Pa] | 0.75 | 5.873 | 0.66 % | 1.11 % |

GCI for temperatures is computed on the rise above the inlet temperature. T<sub>out</sub> is fixed by the energy balance (Q = ṁ c<sub>p</sub> ΔT), so its mesh differences are at round-off level and its 'observed order' is not meaningful.
