/*
 * Runtime migration eval input: incrementally supported Runtime/Driver APIs.
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

static int CheckCuda(const char *name, cudaError_t err)
{
    if (err != cudaSuccess) {
        std::printf("FAIL %s: %s\n", name, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

static int CheckCu(const char *name, CUresult result)
{
    if (result != CUDA_SUCCESS) {
        std::printf("FAIL %s: CUresult %d\n", name, static_cast<int>(result));
        return 1;
    }
    return 0;
}

int main()
{
    int device = 0;
    if (CheckCuda("cudaSetDevice", cudaSetDevice(device))) {
        return 1;
    }

    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    if (CheckCuda("cudaStreamCreate", cudaStreamCreate(&stream)) ||
        CheckCuda("cudaEventCreate", cudaEventCreate(&event)) ||
        CheckCuda("cudaEventRecordWithFlags",
                  cudaEventRecordWithFlags(event, stream, cudaEventRecordDefault)) ||
        CheckCuda("cudaStreamSynchronize", cudaStreamSynchronize(stream))) {
        return 1;
    }

    CUcontext ctx = nullptr;
    unsigned int primaryFlags = 123;
    int active = -1;
    if (CheckCu("cuCtxGetCurrent", cuCtxGetCurrent(&ctx)) ||
        CheckCu("cuCtxSetCurrent", cuCtxSetCurrent(ctx)) ||
        CheckCu("cuDevicePrimaryCtxGetState",
                cuDevicePrimaryCtxGetState(device, &primaryFlags, &active))) {
        return 1;
    }

    void *devicePtr = nullptr;
    constexpr size_t bytes = 4096;
    if (CheckCuda("cudaMalloc", cudaMalloc(&devicePtr, bytes)) ||
        CheckCu("cuMemsetD32Async",
                cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(devicePtr), 0x01020304U,
                                 bytes / sizeof(unsigned int), reinterpret_cast<CUstream>(stream))) ||
        CheckCu("cuStreamWriteValue32",
                cuStreamWriteValue32(reinterpret_cast<CUstream>(stream),
                                     reinterpret_cast<CUdeviceptr>(devicePtr), 7, 0)) ||
        CheckCuda("cudaStreamSynchronize after driver writes", cudaStreamSynchronize(stream))) {
        return 1;
    }

    cudaError_t symbolErr = cudaGetSymbolAddress(nullptr, nullptr);
    cudaError_t copySymbolErr = cudaMemcpyToSymbol(nullptr, nullptr, 0, 0, cudaMemcpyHostToDevice);
    if (symbolErr != cudaErrorInvalidValue || copySymbolErr != cudaErrorInvalidValue) {
        std::printf("FAIL symbol invalid checks\n");
        return 1;
    }

    CheckCuda("cudaFree", cudaFree(devicePtr));
    CheckCuda("cudaEventDestroy", cudaEventDestroy(event));
    CheckCuda("cudaStreamDestroy", cudaStreamDestroy(stream));
    CheckCuda("cudaDeviceReset", cudaDeviceReset());
    std::printf("incremental_runtime_driver PASS flags=%u active=%d\n", primaryFlags, active);
    return 0;
}
