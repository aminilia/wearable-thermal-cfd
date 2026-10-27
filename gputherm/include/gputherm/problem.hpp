// gputherm: steady 3D conduction + Pennes bioheat on a tensor-product (lattice) grid.
//
//   -div(k grad T) + a T = s,     a = w rho_b c_b [W/m3K],  s = a T_a + q_m [W/m3]
//
// Cell-centred finite volumes on a structured grid with per-axis non-uniform cell
// widths (the same grids coolchan/lattice.py produces). Each cell belongs to a region
// (material); interfaces between regions may carry a contact resistance R'' [m2K/W].
// Face conductance between cells P and N:  G = A / (dP/(2kP) + R''_PN + dN/(2kN)),
// i.e. harmonic averaging, exact for piecewise-constant k.
//
// The discrete operator is a symmetric positive-definite 7-point stencil. It is stored
// "assembled-coefficient" style as five arrays (diag, gx, gy, gz, b), where gx[p] is the
// conductance between cell p and its +x neighbour. Both the CPU and the CUDA solvers
// apply this same stencil, so they solve bit-for-bit the same linear system.
//
// Boundary conditions per side: adiabatic, Dirichlet T, prescribed flux q (into the
// domain), or Robin with convection + radiation  q_out = h (Ts - Tinf) + eps sigma (Ts^4 - Tinf^4).
// Radiation is handled by Picard iteration on h_rad = eps sigma (Ts^2 + Tinf^2)(Ts + Tinf),
// which reproduces the T^4 law exactly at convergence.
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gputherm {

constexpr double SIGMA = 5.670374419e-8;

enum class BcType { Adiabatic, Dirichlet, Flux, Robin };
enum Side { XM = 0, XP, YM, YP, ZM, ZP };
static const char* SIDE_NAMES[6] = {"xm", "xp", "ym", "yp", "zm", "zp"};

struct Bc {
    BcType type = BcType::Adiabatic;
    double T = 0, q = 0, h = 0, Tinf = 0, eps = 0;
};

struct Region {
    double k;  // conductivity [W/mK]
    double a;  // perfusion coefficient w rho_b c_b [W/m3K]
    double s;  // source a T_a + q_m [W/m3]
};

struct Problem {
    int nx = 1, ny = 1, nz = 1;
    std::vector<double> dx, dy, dz;  // cell widths [m]
    std::vector<Region> regions;
    std::vector<double> Rc;          // nreg x nreg contact resistance [m2K/W]
    std::vector<int> reg;            // region id per cell
    std::array<Bc, 6> bc;
    double tol = 1e-10;              // CG: ||r|| / ||b||
    int maxit = 50000;
    double picard_tol = 1e-8;        // max |dT| between Picard iterations [K]
    int picard_max = 50;

    size_t n() const { return (size_t)nx * ny * nz; }
    size_t idx(int i, int j, int k) const { return i + (size_t)nx * (j + (size_t)ny * k); }
    double R(int a, int b) const { return Rc[(size_t)a * regions.size() + b]; }
    bool nonlinear() const {
        for (const auto& b : bc)
            if (b.type == BcType::Robin && b.eps > 0) return true;
        return false;
    }
};

// Assembled operator:  (A T)_p = diag_p T_p - sum_faces G_f T_nb,  rhs b.
struct System {
    std::vector<double> diag, gx, gy, gz, b;
    void resize(size_t n) {
        diag.assign(n, 0); gx.assign(n, 0); gy.assign(n, 0); gz.assign(n, 0); b.assign(n, 0);
    }
};

struct SolveStats {
    int iters = 0;
    double relres = 0;
};

// ------------------------------------------------------------------------- assembly
inline double face_g(double A, double dP, double kP, double dN, double kN, double R) {
    return A / (0.5 * dP / kP + R + 0.5 * dN / kN);
}

// Boundary contribution of one face of cell p with half-width d/2, area A.
// Tp is the current (Picard) estimate of the cell temperature, used for h_rad.
inline void boundary(const Bc& bc, double A, double d, double kP, double Tp, double& diag,
                     double& b) {
    switch (bc.type) {
        case BcType::Adiabatic: return;
        case BcType::Dirichlet: {
            double g = A / (0.5 * d / kP);
            diag += g;
            b += g * bc.T;
            return;
        }
        case BcType::Flux: b += bc.q * A; return;
        case BcType::Robin: {
            // Surface temperature consistent with the current cell value: a few fixed-point
            // steps of  Ts = (gc Tp + htot Tinf) / (gc + htot).
            double gc = 2.0 * kP / d, Ts = Tp, htot = bc.h;
            for (int it = 0; it < 4; ++it) {
                double hr = bc.eps * SIGMA * (Ts * Ts + bc.Tinf * bc.Tinf) * (Ts + bc.Tinf);
                htot = bc.h + hr;
                Ts = (gc * Tp + htot * bc.Tinf) / (gc + htot);
            }
            if (htot <= 0) return;
            double g = A / (1.0 / gc + 1.0 / htot);
            diag += g;
            b += g * bc.Tinf;
            return;
        }
    }
}

inline void assemble(const Problem& P, const std::vector<double>& T, System& S) {
    const size_t n = P.n();
    S.resize(n);
    // Volume terms
    for (int k = 0; k < P.nz; ++k)
        for (int j = 0; j < P.ny; ++j)
            for (int i = 0; i < P.nx; ++i) {
                size_t p = P.idx(i, j, k);
                const Region& r = P.regions[P.reg[p]];
                double V = P.dx[i] * P.dy[j] * P.dz[k];
                S.diag[p] = r.a * V;
                S.b[p] = r.s * V;
            }
    // Interior faces and boundaries
    for (int k = 0; k < P.nz; ++k)
        for (int j = 0; j < P.ny; ++j)
            for (int i = 0; i < P.nx; ++i) {
                size_t p = P.idx(i, j, k);
                int rp = P.reg[p];
                double kp = P.regions[rp].k;
                double Ax = P.dy[j] * P.dz[k], Ay = P.dx[i] * P.dz[k], Az = P.dx[i] * P.dy[j];
                if (i + 1 < P.nx) {
                    size_t q = p + 1; int rq = P.reg[q];
                    double g = face_g(Ax, P.dx[i], kp, P.dx[i + 1], P.regions[rq].k, P.R(rp, rq));
                    S.gx[p] = g; S.diag[p] += g; S.diag[q] += g;
                }
                if (j + 1 < P.ny) {
                    size_t q = p + P.nx; int rq = P.reg[q];
                    double g = face_g(Ay, P.dy[j], kp, P.dy[j + 1], P.regions[rq].k, P.R(rp, rq));
                    S.gy[p] = g; S.diag[p] += g; S.diag[q] += g;
                }
                if (k + 1 < P.nz) {
                    size_t q = p + (size_t)P.nx * P.ny; int rq = P.reg[q];
                    double g = face_g(Az, P.dz[k], kp, P.dz[k + 1], P.regions[rq].k, P.R(rp, rq));
                    S.gz[p] = g; S.diag[p] += g; S.diag[q] += g;
                }
                if (i == 0)         boundary(P.bc[XM], Ax, P.dx[i], kp, T[p], S.diag[p], S.b[p]);
                if (i == P.nx - 1)  boundary(P.bc[XP], Ax, P.dx[i], kp, T[p], S.diag[p], S.b[p]);
                if (j == 0)         boundary(P.bc[YM], Ay, P.dy[j], kp, T[p], S.diag[p], S.b[p]);
                if (j == P.ny - 1)  boundary(P.bc[YP], Ay, P.dy[j], kp, T[p], S.diag[p], S.b[p]);
                if (k == 0)         boundary(P.bc[ZM], Az, P.dz[k], kp, T[p], S.diag[p], S.b[p]);
                if (k == P.nz - 1)  boundary(P.bc[ZP], Az, P.dz[k], kp, T[p], S.diag[p], S.b[p]);
            }
}

// ------------------------------------------------------------------------- spec file
// Whitespace-separated tokens, '#' starts a comment. Sections (any order after grid):
//   grid nx ny nz
//   dx <nx widths, m>   dy <ny widths>   dz <nz widths>
//   regions R   then R lines:  k a s
//   contact C   then C lines:  r1 r2 R''        (symmetric)
//   boxes B     then B lines:  i0 i1 j0 j1 k0 k1 region   (half-open, applied in order)
//   bc <side> adiabatic | dirichlet T | flux q | robin h Tinf eps     side: xm xp ym yp zm zp
//   solver tol maxit picard_tol picard_max
// Cells not covered by a box belong to region 0.
inline Problem read_spec(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open spec " + path);
    std::stringstream clean;
    std::string line;
    while (std::getline(f, line)) {
        auto h = line.find('#');
        clean << (h == std::string::npos ? line : line.substr(0, h)) << "\n";
    }
    Problem P;
    std::string kw;
    auto vec = [&](std::vector<double>& v, int m) {
        v.resize(m);
        for (auto& x : v) clean >> x;
    };
    std::vector<std::array<int, 7>> boxes;
    bool have_grid = false;
    while (clean >> kw) {
        if (kw == "gputherm") { int ver; clean >> ver; }
        else if (kw == "grid") { clean >> P.nx >> P.ny >> P.nz; have_grid = true; }
        else if (kw == "dx") vec(P.dx, P.nx);
        else if (kw == "dy") vec(P.dy, P.ny);
        else if (kw == "dz") vec(P.dz, P.nz);
        else if (kw == "regions") {
            int m; clean >> m; P.regions.resize(m);
            for (auto& r : P.regions) clean >> r.k >> r.a >> r.s;
            P.Rc.assign((size_t)m * m, 0.0);
        } else if (kw == "contact") {
            int m; clean >> m;
            for (int c = 0; c < m; ++c) {
                int a, b; double R; clean >> a >> b >> R;
                size_t nr = P.regions.size();
                P.Rc[(size_t)a * nr + b] = P.Rc[(size_t)b * nr + a] = R;
            }
        } else if (kw == "boxes") {
            int m; clean >> m; boxes.resize(m);
            for (auto& b : boxes) for (auto& v : b) clean >> v;
        } else if (kw == "bc") {
            std::string side, type; clean >> side >> type;
            int s = -1;
            for (int q = 0; q < 6; ++q) if (side == SIDE_NAMES[q]) s = q;
            if (s < 0) throw std::runtime_error("bad bc side " + side);
            Bc& b = P.bc[s];
            if (type == "adiabatic") b.type = BcType::Adiabatic;
            else if (type == "dirichlet") { b.type = BcType::Dirichlet; clean >> b.T; }
            else if (type == "flux") { b.type = BcType::Flux; clean >> b.q; }
            else if (type == "robin") { b.type = BcType::Robin; clean >> b.h >> b.Tinf >> b.eps; }
            else throw std::runtime_error("bad bc type " + type);
        } else if (kw == "solver") {
            clean >> P.tol >> P.maxit >> P.picard_tol >> P.picard_max;
        } else throw std::runtime_error("unknown keyword " + kw);
    }
    if (!have_grid || (int)P.dx.size() != P.nx || (int)P.dy.size() != P.ny ||
        (int)P.dz.size() != P.nz || P.regions.empty())
        throw std::runtime_error("incomplete spec (need grid, dx, dy, dz, regions)");
    P.reg.assign(P.n(), 0);
    for (auto& b : boxes)
        for (int k = b[4]; k < b[5]; ++k)
            for (int j = b[2]; j < b[3]; ++j)
                for (int i = b[0]; i < b[1]; ++i) P.reg[P.idx(i, j, k)] = b[6];
    return P;
}

// ------------------------------------------------------------------------- benchmark
// N x N x N device-on-skin block: the Phase 2 tissue stack (coolchan/device3d.py), a 1 mm
// housing with contact resistance to the skin, and a 1.5 mm top layer holding a 0.5 W
// heater block. Uniform spacing, so thin layers are only resolved at larger N; it exists
// to measure performance, not to answer the design question.
inline Problem make_bench(int N) {
    Problem P;
    P.nx = P.ny = P.nz = N;
    const double L = 40e-3, H = 20e-3;
    P.dx.assign(N, L / N); P.dy.assign(N, L / N); P.dz.assign(N, H / N);
    const double rho_b = 1060.0, cp_b = 3770.0, Ta = 310.15;
    auto tissue = [&](double k, double w, double qm) {
        double a = w * rho_b * cp_b;
        return Region{k, a, a * Ta + qm};
    };
    // 0 inner, 1 fat, 2 dermis, 3 epidermis, 4 housing, 5 heater
    const double qvol = 0.5 / (20e-3 * 8e-3 * 1.5e-3);
    P.regions = {tissue(0.50, 1.25e-3, 370), tissue(0.19, 1.25e-3, 370),
                 tissue(0.45, 1.25e-3, 370), tissue(0.24, 0, 0),
                 Region{0.20, 0, 0}, Region{200.0, 0, qvol}};
    const size_t nr = P.regions.size();
    P.Rc.assign(nr * nr, 0.0);
    P.Rc[3 * nr + 4] = P.Rc[4 * nr + 3] = 1e-3;
    const double zb[] = {15.0e-3, 15.6e-3, 17.1e-3, 17.5e-3, 18.5e-3};  // layer tops
    P.reg.assign(P.n(), 0);
    for (int k = 0; k < N; ++k) {
        double zc = (k + 0.5) * H / N;
        int layer = 0;
        while (layer < 5 && zc > zb[layer]) ++layer;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                double xc = (i + 0.5) * L / N - L / 2, yc = (j + 0.5) * L / N - L / 2;
                int r = layer;
                if (layer == 5) r = (std::fabs(xc) < 10e-3 && std::fabs(yc) < 4e-3) ? 5 : 4;
                P.reg[P.idx(i, j, k)] = r;
            }
    }
    P.bc[ZM] = Bc{BcType::Dirichlet, 310.15};
    P.bc[ZP] = Bc{BcType::Robin, 0, 0, 10.0, 298.15, 0.9};
    return P;
}

// ------------------------------------------------------------------------- driver
using Clock = std::chrono::steady_clock;
inline double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// SolveFn: void(const Problem&, const System&, std::vector<double>& T, SolveStats&)
// solves A T = b in place, warm-started from the T it is given.
template <class SolveFn>
int run_driver(int argc, char** argv, const char* backend, SolveFn solve) {
    std::string spec, out;
    int bench = 0;
    double tol = -1;
    for (int a = 1; a < argc; ++a) {
        std::string s = argv[a];
        if (s == "--bench" && a + 1 < argc) bench = std::atoi(argv[++a]);
        else if (s == "--out" && a + 1 < argc) out = argv[++a];
        else if (s == "--tol" && a + 1 < argc) tol = std::atof(argv[++a]);
        else if (s == "-h" || s == "--help") {
            std::printf("usage: %s (SPEC | --bench N) [--out T.bin] [--tol 1e-10]\n", argv[0]);
            return 0;
        } else spec = s;
    }
    if (spec.empty() && bench <= 0) {
        std::fprintf(stderr, "need a spec file or --bench N (see --help)\n");
        return 2;
    }
    Problem P = bench > 0 ? make_bench(bench) : read_spec(spec);
    if (tol > 0) P.tol = tol;

    double T0 = 305.0;
    for (const auto& b : P.bc) if (b.type == BcType::Dirichlet) T0 = b.T;
    std::vector<double> T(P.n(), T0), Told;
    System S;
    double t_asm = 0, t_solve = 0, dT = 0;
    int cg_total = 0, picard = 0;
    SolveStats st;
    auto t_start = Clock::now();
    for (picard = 1; picard <= P.picard_max; ++picard) {
        auto t0 = Clock::now();
        assemble(P, T, S);
        auto t1 = Clock::now();
        Told = T;
        solve(P, S, T, st);
        auto t2 = Clock::now();
        t_asm += secs(t0, t1);
        t_solve += secs(t1, t2);
        cg_total += st.iters;
        dT = 0;
        for (size_t p = 0; p < T.size(); ++p) dT = std::max(dT, std::fabs(T[p] - Told[p]));
        if (!P.nonlinear() || dT < P.picard_tol) break;
    }
    double t_total = secs(t_start, Clock::now());
    if (!out.empty()) {
        std::ofstream f(out, std::ios::binary);
        f.write(reinterpret_cast<const char*>(T.data()), T.size() * sizeof(double));
    }
    double Tmax = *std::max_element(T.begin(), T.end());
    std::printf(
        "{\"backend\": \"%s\", \"nx\": %d, \"ny\": %d, \"nz\": %d, \"cells\": %zu, "
        "\"picard_iters\": %d, \"picard_dT\": %.3e, \"cg_iters\": %d, \"last_relres\": %.3e, "
        "\"t_assemble_s\": %.6f, \"t_solve_s\": %.6f, \"t_total_s\": %.6f, "
        "\"t_per_cg_iter_ms\": %.6f, \"T_max\": %.9f}\n",
        backend, P.nx, P.ny, P.nz, P.n(), std::min(picard, P.picard_max), dT, cg_total,
        st.relres, t_asm, t_solve, t_total, cg_total ? 1e3 * t_solve / cg_total : 0.0, Tmax);
    return 0;
}

}  // namespace gputherm
