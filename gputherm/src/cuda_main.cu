// CUDA solver, version 0 ("naive"): the same Jacobi-preconditioned CG as cpu_main.cpp,
// one thread per cell, every vector operation its own kernel, and every dot product
// copied back to the host. It is deliberately straightforward so later versions have a
// clean baseline to beat. Known costs it leaves on the table (the optimization roadmap):
//   * 3 global reductions + 3 device->host copies per iteration (latency-bound at small N)
//   * separate kernels for x/r/z/p updates (each re-reads vectors from DRAM)
//   * the stencil reads 4 coefficient arrays + 7 neighbours per cell with no reuse
//   * FP64 everywhere: consumer/T4 GPUs run FP64 at 1/32 of FP32
//   * assembly (incl. the radiation update) still runs on the host each Picard step
#include "gputherm/problem.hpp"

#include <cuda_runtime.h>

using namespace gputherm;

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t e_ = (call);                                                          \
        if (e_ != cudaSuccess)                                                            \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) +     \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } while (0)

namespace {

constexpr int RED_THREADS = 256;

__global__ void k_apply(int nx, int ny, int nz, const double* __restrict__ diag,
                        const double* __restrict__ gx, const double* __restrict__ gy,
                        const double* __restrict__ gz, const double* __restrict__ x,
                        double* __restrict__ y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    const size_t sy = nx, sz = (size_t)nx * ny;
    const size_t p = i + sy * j + sz * k;
    double v = diag[p] * x[p];
    if (i > 0)      v -= gx[p - 1] * x[p - 1];
    if (i < nx - 1) v -= gx[p] * x[p + 1];
    if (j > 0)      v -= gy[p - sy] * x[p - sy];
    if (j < ny - 1) v -= gy[p] * x[p + sy];
    if (k > 0)      v -= gz[p - sz] * x[p - sz];
    if (k < nz - 1) v -= gz[p] * x[p + sz];
    y[p] = v;
}

// Grid-stride partial sums, shared-memory tree per block, one atomicAdd per block.
__global__ void k_dot(size_t n, const double* __restrict__ a, const double* __restrict__ b,
                      double* out) {
    __shared__ double sh[RED_THREADS];
    double s = 0;
    for (size_t p = blockIdx.x * (size_t)blockDim.x + threadIdx.x; p < n;
         p += (size_t)blockDim.x * gridDim.x)
        s += a[p] * b[p];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sh[0]);  // FP64 atomicAdd: sm_60+
}

__global__ void k_init(size_t n, const double* __restrict__ b, const double* __restrict__ Ax,
                       const double* __restrict__ diag, double* r, double* z, double* p) {
    size_t q = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (q >= n) return;
    r[q] = b[q] - Ax[q];
    z[q] = r[q] / diag[q];
    p[q] = z[q];
}

__global__ void k_update_xr(size_t n, double alpha, const double* __restrict__ p,
                            const double* __restrict__ Ap, double* x, double* r) {
    size_t q = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (q >= n) return;
    x[q] += alpha * p[q];
    r[q] -= alpha * Ap[q];
}

__global__ void k_precond(size_t n, const double* __restrict__ r,
                          const double* __restrict__ diag, double* z) {
    size_t q = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (q < n) z[q] = r[q] / diag[q];
}

__global__ void k_update_p(size_t n, double beta, const double* __restrict__ z, double* p) {
    size_t q = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (q < n) p[q] = z[q] + beta * p[q];
}

// Device buffers persist across Picard iterations (and are freed at exit).
struct Gpu {
    size_t n = 0;
    double *diag = nullptr, *gx = nullptr, *gy = nullptr, *gz = nullptr, *b = nullptr;
    double *x = nullptr, *r = nullptr, *z = nullptr, *p = nullptr, *Ap = nullptr, *red = nullptr;
    int red_blocks = 0;

    void alloc(size_t m) {
        if (m == n) return;
        free();
        n = m;
        for (double** a : {&diag, &gx, &gy, &gz, &b, &x, &r, &z, &p, &Ap})
            CUDA_CHECK(cudaMalloc(a, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&red, sizeof(double)));
        int dev, sms;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        red_blocks = std::min<size_t>(8 * sms, (n + RED_THREADS - 1) / RED_THREADS);
    }
    void free() {
        for (double* a : {diag, gx, gy, gz, b, x, r, z, p, Ap, red})
            if (a) cudaFree(a);
        diag = gx = gy = gz = b = x = r = z = p = Ap = red = nullptr;
        n = 0;
    }
    ~Gpu() { free(); }

    double dot(const double* a, const double* c) {
        CUDA_CHECK(cudaMemsetAsync(red, 0, sizeof(double)));
        k_dot<<<red_blocks, RED_THREADS>>>(n, a, c, red);
        double h;
        CUDA_CHECK(cudaMemcpy(&h, red, sizeof(double), cudaMemcpyDeviceToHost));
        return h;
    }
};

Gpu G;

void pcg_gpu(const Problem& P, const System& S, std::vector<double>& T, SolveStats& st) {
    const size_t n = P.n();
    G.alloc(n);
    const size_t bytes = n * sizeof(double);
    CUDA_CHECK(cudaMemcpy(G.diag, S.diag.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G.gx, S.gx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G.gy, S.gy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G.gz, S.gz.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G.b, S.b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G.x, T.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 sb(32, 4, 2);
    const dim3 sg((P.nx + sb.x - 1) / sb.x, (P.ny + sb.y - 1) / sb.y, (P.nz + sb.z - 1) / sb.z);
    const int vb = 256;
    const int vg = (int)((n + vb - 1) / vb);
    auto apply = [&](const double* in, double* outv) {
        k_apply<<<sg, sb>>>(P.nx, P.ny, P.nz, G.diag, G.gx, G.gy, G.gz, in, outv);
    };

    apply(G.x, G.Ap);
    k_init<<<vg, vb>>>(n, G.b, G.Ap, G.diag, G.r, G.z, G.p);
    CUDA_CHECK(cudaGetLastError());
    const double bnorm = std::sqrt(G.dot(G.b, G.b));
    double rz = G.dot(G.r, G.z);
    double rr = G.dot(G.r, G.r);
    int it = 0;
    while (std::sqrt(rr) > P.tol * bnorm && it < P.maxit) {
        apply(G.p, G.Ap);
        const double alpha = rz / G.dot(G.p, G.Ap);
        k_update_xr<<<vg, vb>>>(n, alpha, G.p, G.Ap, G.x, G.r);
        k_precond<<<vg, vb>>>(n, G.r, G.diag, G.z);
        const double rz_new = G.dot(G.r, G.z);
        rr = G.dot(G.r, G.r);
        k_update_p<<<vg, vb>>>(n, rz_new / rz, G.z, G.p);
        rz = rz_new;
        ++it;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(T.data(), G.x, bytes, cudaMemcpyDeviceToHost));
    st.iters = it;
    st.relres = std::sqrt(rr) / bnorm;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::fprintf(stderr, "GPU: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor,
                     prop.multiProcessorCount);
        CUDA_CHECK(cudaFree(nullptr));  // create the context outside the timed region
        int rc = run_driver(argc, argv, "cuda", pcg_gpu);
        G.free();
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
