/*
 * Runtime migration eval input: CUDA vector add.
 * Expected CUDA-side result: every output element equals a[i] + b[i].
 */
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__global__ void VecAddKernel(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    constexpr int n = 256;
    constexpr size_t bytes = n * sizeof(float);

    float *h_a = static_cast<float *>(std::malloc(bytes));
    float *h_b = static_cast<float *>(std::malloc(bytes));
    float *h_c = static_cast<float *>(std::malloc(bytes));
    if (!h_a || !h_b || !h_c) {
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
        h_c[i] = 0.0f;
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_a), bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_b), bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_c), bytes));
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    VecAddKernel<<<(n + 127) / 128, 128>>>(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            std::printf("FAIL idx=%d got=%f expected=%f\n", i, h_c[i], expected);
            return 1;
        }
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);
    std::printf("vector_add PASS n=%d\n", n);
    return 0;
}
