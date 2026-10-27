// CPU reference solver: matrix-free Jacobi-preconditioned conjugate gradient on the
// 7-point stencil assembled in problem.hpp. OpenMP-parallel when built with -fopenmp.
// This is the baseline every GPU version is verified and benchmarked against.
#include "gputherm/problem.hpp"

using namespace gputherm;

namespace {

void apply(const Problem& P, const System& S, const double* x, double* y) {
    const int nx = P.nx, ny = P.ny, nz = P.nz;
    const size_t sy = nx, sz = (size_t)nx * ny;
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                size_t p = i + sy * j + sz * k;
                double v = S.diag[p] * x[p];
                if (i > 0)      v -= S.gx[p - 1] * x[p - 1];
                if (i < nx - 1) v -= S.gx[p] * x[p + 1];
                if (j > 0)      v -= S.gy[p - sy] * x[p - sy];
                if (j < ny - 1) v -= S.gy[p] * x[p + sy];
                if (k > 0)      v -= S.gz[p - sz] * x[p - sz];
                if (k < nz - 1) v -= S.gz[p] * x[p + sz];
                y[p] = v;
            }
}

double dot(size_t n, const double* a, const double* b) {
    double s = 0;
#pragma omp parallel for reduction(+ : s) schedule(static)
    for (size_t p = 0; p < n; ++p) s += a[p] * b[p];
    return s;
}

void pcg(const Problem& P, const System& S, std::vector<double>& T, SolveStats& st) {
    const size_t n = P.n();
    std::vector<double> r(n), z(n), p(n), Ap(n);
    double* x = T.data();
    apply(P, S, x, Ap.data());
#pragma omp parallel for schedule(static)
    for (size_t q = 0; q < n; ++q) {
        r[q] = S.b[q] - Ap[q];
        z[q] = r[q] / S.diag[q];
        p[q] = z[q];
    }
    const double bnorm = std::sqrt(dot(n, S.b.data(), S.b.data()));
    double rz = dot(n, r.data(), z.data());
    double rr = dot(n, r.data(), r.data());
    int it = 0;
    while (std::sqrt(rr) > P.tol * bnorm && it < P.maxit) {
        apply(P, S, p.data(), Ap.data());
        const double alpha = rz / dot(n, p.data(), Ap.data());
#pragma omp parallel for schedule(static)
        for (size_t q = 0; q < n; ++q) {
            x[q] += alpha * p[q];
            r[q] -= alpha * Ap[q];
            z[q] = r[q] / S.diag[q];
        }
        const double rz_new = dot(n, r.data(), z.data());
        rr = dot(n, r.data(), r.data());
        const double beta = rz_new / rz;
        rz = rz_new;
#pragma omp parallel for schedule(static)
        for (size_t q = 0; q < n; ++q) p[q] = z[q] + beta * p[q];
        ++it;
    }
    st.iters = it;
    st.relres = std::sqrt(rr) / bnorm;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run_driver(argc, argv, "cpu", pcg);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
