/*
 * Runtime migration eval input: stream, event, async memcpy and memset.
 * Expected CUDA-side result: device memset writes 0x5a to every byte.
 */
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

static int Check(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int main()
{
    constexpr size_t bytes = 4096;
    unsigned char *host = nullptr;
    unsigned char *out = nullptr;
    void *device = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;

    host = static_cast<unsigned char *>(std::malloc(bytes));
    out = static_cast<unsigned char *>(std::malloc(bytes));
    if (!host || !out) {
        return 1;
    }
    std::fill_n(host, bytes, 0x11);
    std::fill_n(out, bytes, 0);

    if (Check(cudaStreamCreate(&stream)) ||
        Check(cudaEventCreate(&event)) ||
        Check(cudaMalloc(&device, bytes)) ||
        Check(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream)) ||
        Check(cudaMemsetAsync(device, 0x5a, bytes, stream)) ||
        Check(cudaEventRecord(event, stream)) ||
        Check(cudaEventSynchronize(event)) ||
        Check(cudaMemcpy(out, device, bytes, cudaMemcpyDeviceToHost))) {
        return 1;
    }

    for (size_t i = 0; i < bytes; ++i) {
        if (out[i] != 0x5a) {
            std::printf("FAIL byte=%zu got=0x%02x\n", i, out[i]);
            return 1;
        }
    }

    Check(cudaFree(device));
    Check(cudaEventDestroy(event));
    Check(cudaStreamDestroy(stream));
    std::free(host);
    std::free(out);
    std::printf("stream_event_memcpy PASS bytes=%zu\n", bytes);
    return 0;
}
