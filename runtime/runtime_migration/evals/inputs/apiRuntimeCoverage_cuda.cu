/*
 * Native CUDA version of apiRuntimeCoverage.
 *
 * Build on a CUDA machine, for example:
 *   nvcc -std=c++17 apiRuntimeCoverage_cuda.cu -lcuda -o apiRuntimeCoverage_cuda
 */

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

__global__ void ApiRuntimeCoverageKernel(int *out)
{
    *out = 42;
}

__global__ void ApiRuntimeCoverageKernelWithValue(int *out, int value)
{
    *out = value;
}

__device__ int g_apiRuntimeCoverageSymbol = 0;
__device__ int g_symbolSizeProbe[4] = {};
__device__ int g_capturedGraphSetterValue = 0;

__global__ void ApiRuntimeCoverageCapturedSetterKernel()
{
    g_capturedGraphSetterValue = 42;
}

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
__global__ void ApiRuntimeCoverageSetConditionalKernel(cudaGraphConditionalHandle handle)
{
    cudaGraphSetConditional(handle, 1);
}

__global__ void ApiRuntimeCoverageConditionalBodyKernel(int *out, cudaGraphConditionalHandle handle)
{
    *out = 314;
    cudaGraphSetConditional(handle, 0);
}
#endif

namespace {

int g_failures = 0;
int g_checks = 0;

cudaError_t GetStreamCaptureInfoCompat(cudaStream_t stream,
                                       cudaStreamCaptureStatus *status,
                                       unsigned long long *id,
                                       cudaGraph_t *graph,
                                       const cudaGraphNode_t **deps,
                                       size_t *depCount)
{
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    const cudaGraphEdgeData *edgeData = nullptr;
    return cudaStreamGetCaptureInfo(stream, status, id, graph, deps, &edgeData, depCount);
#else
    return cudaStreamGetCaptureInfo(stream, status, id, graph, deps, depCount);
#endif
}

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
cudaError_t GraphAddNodeCompat(cudaGraphNode_t *node,
                               cudaGraph_t graph,
                               const cudaGraphNode_t *deps,
                               size_t depCount,
                               cudaGraphNodeParams *params)
{
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    return cudaGraphAddNode(node, graph, deps, nullptr, depCount, params);
#else
    return cudaGraphAddNode(node, graph, deps, depCount, params);
#endif
}
#endif

struct IpcPayload {
    cudaIpcMemHandle_t memHandle;
    cudaIpcEventHandle_t eventHandle;
};

void PrintResult(const char *name, const char *state, const char *detail = "")
{
    std::printf("  %-42s %s%s%s\n", name, state, detail[0] ? " - " : "", detail);
}

std::string CudaErrorDetail(cudaError_t expected, cudaError_t actual)
{
    return std::string("expected ") + cudaGetErrorName(expected) + ", got " + cudaGetErrorName(actual);
}

std::string CuResultDetail(CUresult result)
{
    const char *errName = nullptr;
    cuGetErrorName(result, &errName);
    if (errName != nullptr) {
        return std::string(errName) + " (" + std::to_string(static_cast<int>(result)) + ")";
    }
    return std::string("CUresult ") + std::to_string(static_cast<int>(result));
}

std::string CuResultDetail(CUresult expected, CUresult actual)
{
    return std::string("expected ") + CuResultDetail(expected) + ", got " + CuResultDetail(actual);
}

bool ExpectSuccess(const char *name, cudaError_t err)
{
    ++g_checks;
    if (err == cudaSuccess) {
        PrintResult(name, "PASS");
        return true;
    }

    ++g_failures;
    PrintResult(name, "FAIL", cudaGetErrorString(err));
    return false;
}

bool ExpectIntValue(const char *name, int actual, int expected)
{
    ++g_checks;
    if (actual == expected) {
        PrintResult(name, "PASS");
        return true;
    }

    ++g_failures;
    char detail[96];
    std::snprintf(detail, sizeof(detail), "actual=%d expected=%d", actual, expected);
    PrintResult(name, "FAIL", detail);
    return false;
}

bool ExpectError(const char *name, cudaError_t err, cudaError_t expected)
{
    ++g_checks;
    if (err == expected) {
        PrintResult(name, "PASS", cudaGetErrorName(err));
        return true;
    }

    ++g_failures;
    std::string detail = CudaErrorDetail(expected, err);
    PrintResult(name, "FAIL", detail.c_str());
    return false;
}

bool ExpectErrorAny(const char *name, cudaError_t err, cudaError_t expectedA, cudaError_t expectedB)
{
    ++g_checks;
    if (err == expectedA || err == expectedB) {
        PrintResult(name, "PASS", cudaGetErrorName(err));
        return true;
    }

    ++g_failures;
    std::string detail = std::string("expected ") + cudaGetErrorName(expectedA) + " or " +
                         cudaGetErrorName(expectedB) + ", got " + cudaGetErrorName(err);
    PrintResult(name, "FAIL", detail.c_str());
    return false;
}

void Skip(const char *name, const char *reason)
{
    PrintResult(name, "SKIP", reason);
}

bool OptionalSuccess(const char *name, cudaError_t err)
{
    if (err == cudaSuccess) {
        ++g_checks;
        PrintResult(name, "PASS");
        return true;
    }

    Skip(name, cudaGetErrorString(err));
    return false;
}

bool OptionalCuSuccess(const char *name, CUresult result)
{
    if (result == CUDA_SUCCESS) {
        ++g_checks;
        PrintResult(name, "PASS");
        return true;
    }

    const char *errName = nullptr;
    cuGetErrorName(result, &errName);
    char detail[128];
    std::snprintf(detail, sizeof(detail), "%s (%d)",
                  errName == nullptr ? "CUresult" : errName, static_cast<int>(result));
    Skip(name, detail);
    return false;
}

bool ExpectCuSuccess(const char *name, CUresult result)
{
    ++g_checks;
    if (result == CUDA_SUCCESS) {
        PrintResult(name, "PASS");
        return true;
    }

    ++g_failures;
    std::string detail = CuResultDetail(result);
    PrintResult(name, "FAIL", detail.c_str());
    return false;
}

bool ExpectCuError(const char *name, CUresult result, CUresult expected)
{
    ++g_checks;
    if (result == expected) {
        PrintResult(name, "PASS");
        return true;
    }

    ++g_failures;
    std::string detail = CuResultDetail(expected, result);
    PrintResult(name, "FAIL", detail.c_str());
    return false;
}

bool ExpectCuErrorAny(const char *name, CUresult result, CUresult expectedA, CUresult expectedB)
{
    ++g_checks;
    if (result == expectedA || result == expectedB) {
        PrintResult(name, "PASS", CuResultDetail(result).c_str());
        return true;
    }

    ++g_failures;
    std::string detail = std::string("expected ") + CuResultDetail(expectedA) + " or " +
                         CuResultDetail(expectedB) + ", got " + CuResultDetail(result);
    PrintResult(name, "FAIL", detail.c_str());
    return false;
}

void CheckDeviceApis(int device)
{
    cudaDeviceProp prop{};
    ExpectSuccess("cudaGetDeviceProperties", cudaGetDeviceProperties(&prop, device));

    int attr = 0;
    ExpectSuccess("cudaDeviceGetAttribute ClockRate",
                  cudaDeviceGetAttribute(&attr, cudaDevAttrClockRate, device));

    size_t freeMem = 0;
    size_t totalMem = 0;
    ExpectSuccess("cudaMemGetInfo", cudaMemGetInfo(&freeMem, &totalMem));

    int driverVersion = 0;
    int runtimeVersion = 0;
    if (ExpectSuccess("cudaDriverGetVersion", cudaDriverGetVersion(&driverVersion))) {
        ++g_checks;
        if (driverVersion > 0) {
            PrintResult("cudaDriverGetVersion value", "PASS");
        } else {
            ++g_failures;
            PrintResult("cudaDriverGetVersion value", "FAIL", "expected positive version");
        }
    }
    if (ExpectSuccess("cudaRuntimeGetVersion", cudaRuntimeGetVersion(&runtimeVersion))) {
        ++g_checks;
        if (runtimeVersion > 0) {
            PrintResult("cudaRuntimeGetVersion value", "PASS");
        } else {
            ++g_failures;
            PrintResult("cudaRuntimeGetVersion value", "FAIL", "expected positive version");
        }
    }

    std::printf("  Device %d: %s, free/total memory: %zu / %zu\n",
                device, prop.name, freeMem, totalMem);
}

void CheckErrorApis()
{
    ExpectSuccess("cudaPeekAtLastError", cudaPeekAtLastError());
    ExpectSuccess("cudaGetLastError", cudaGetLastError());

    ++g_checks;
    const char *name = cudaGetErrorName(cudaErrorNotSupported);
    const char *text = cudaGetErrorString(cudaErrorNotSupported);
    if (name != nullptr && text != nullptr) {
        PrintResult("cudaGetErrorName/String", "PASS", name);
    } else {
        ++g_failures;
        PrintResult("cudaGetErrorName/String", "FAIL", "null text");
    }
}

void CheckDeviceStateApis()
{
    unsigned int flags = 0;
    OptionalSuccess("cudaSetDeviceFlags", cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync | cudaDeviceMapHost));
    OptionalSuccess("cudaGetDeviceFlags", cudaGetDeviceFlags(&flags));

    OptionalSuccess("cudaDeviceSetLimit", cudaDeviceSetLimit(cudaLimitStackSize, 4096));
    size_t limitValue = 0;
    OptionalSuccess("cudaDeviceGetLimit", cudaDeviceGetLimit(&limitValue, cudaLimitStackSize));
    ExpectError("cudaDeviceGetLimit null output",
                cudaDeviceGetLimit(nullptr, cudaLimitStackSize), cudaErrorInvalidValue);
    OptionalSuccess("cudaDeviceSetLimit heap",
                    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8 * 1024 * 1024));
    OptionalSuccess("cudaDeviceGetLimit heap",
                    cudaDeviceGetLimit(&limitValue, cudaLimitMallocHeapSize));

    cudaFuncCache cacheConfig = cudaFuncCachePreferNone;
    OptionalSuccess("cudaDeviceSetCacheConfig", cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    OptionalSuccess("cudaDeviceGetCacheConfig", cudaDeviceGetCacheConfig(&cacheConfig));

    int leastPriority = 0;
    int greatestPriority = 0;
    OptionalSuccess("cudaDeviceGetStreamPriorityRange",
                    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
}

void CheckAtomicCapabilityApis(int deviceCount)
{
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    cudaAtomicOperation operations[] = {
        cudaAtomicOperationIntegerAdd,
        cudaAtomicOperationCAS,
        cudaAtomicOperationFloatAdd
    };
    unsigned int capabilities[3] = {};

    OptionalSuccess("cudaDeviceGetHostAtomicCapabilities",
                    cudaDeviceGetHostAtomicCapabilities(capabilities, operations, 3, 0));
    ExpectError("cudaDeviceGetHostAtomicCapabilities null caps",
                cudaDeviceGetHostAtomicCapabilities(nullptr, operations, 3, 0), cudaErrorInvalidValue);
    ExpectError("cudaDeviceGetHostAtomicCapabilities zero count",
                cudaDeviceGetHostAtomicCapabilities(capabilities, operations, 0, 0), cudaErrorInvalidValue);
    cudaAtomicOperation invalidOperations[] = {static_cast<cudaAtomicOperation>(999)};
    ExpectError("cudaDeviceGetHostAtomicCapabilities bad op",
                cudaDeviceGetHostAtomicCapabilities(capabilities, invalidOperations, 1, 0), cudaErrorInvalidValue);

    if (deviceCount < 2) {
        Skip("cudaDeviceGetP2PAtomicCapabilities", "requires at least two visible devices");
    } else {
        OptionalSuccess("cudaDeviceGetP2PAtomicCapabilities",
                        cudaDeviceGetP2PAtomicCapabilities(capabilities, operations, 3, 0, 1));
    }
    ExpectError("cudaDeviceGetP2PAtomicCapabilities same device",
                cudaDeviceGetP2PAtomicCapabilities(capabilities, operations, 3, 0, 0), cudaErrorInvalidDevice);
#else
    (void)deviceCount;
    Skip("cudaDeviceGetHostAtomicCapabilities", "requires CUDA 13.x headers");
    Skip("cudaDeviceGetP2PAtomicCapabilities", "requires CUDA 13.x headers");
#endif
}

void CheckPeerAccessCapability(int deviceCount)
{
    ExpectError("cudaDeviceCanAccessPeer null output",
                cudaDeviceCanAccessPeer(nullptr, 0, 0), cudaErrorInvalidValue);

    if (deviceCount < 2) {
        Skip("cudaDeviceCanAccessPeer 0->1", "requires at least two visible devices");
        return;
    }

    int canAccessPeer = 0;
    if (ExpectSuccess("cudaDeviceCanAccessPeer 0->1",
                      cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1))) {
        ++g_checks;
        if (canAccessPeer == 0 || canAccessPeer == 1) {
            PrintResult("cudaDeviceCanAccessPeer result range", "PASS",
                        canAccessPeer ? "can access" : "cannot access");
        } else {
            ++g_failures;
            PrintResult("cudaDeviceCanAccessPeer result range", "FAIL", "expected 0 or 1");
        }
    }
}

void CheckPeerEnableAndMemcpyAsyncApis(int deviceCount)
{
    constexpr size_t bytes = 128;
    std::vector<unsigned char> hostIn(bytes);
    std::vector<unsigned char> hostOut(bytes, 0);
    for (size_t i = 0; i < bytes; ++i) {
        hostIn[i] = static_cast<unsigned char>(0x30U + (i % 64U));
    }

    void *src = nullptr;
    void *dst = nullptr;
    if (ExpectSuccess("peer async src cudaMalloc", cudaMalloc(&src, bytes)) &&
        ExpectSuccess("peer async dst cudaMalloc", cudaMalloc(&dst, bytes))) {
        ExpectSuccess("seed peer async src", cudaMemcpy(src, hostIn.data(), bytes, cudaMemcpyHostToDevice));

        ExpectSuccess("cudaMemcpyPeer same device", cudaMemcpyPeer(dst, 0, src, 0, bytes));
        ExpectSuccess("copy cudaMemcpyPeer dst back", cudaMemcpy(hostOut.data(), dst, bytes, cudaMemcpyDeviceToHost));
        ++g_checks;
        if (hostOut == hostIn) {
            PrintResult("cudaMemcpyPeer same device data", "PASS");
        } else {
            ++g_failures;
            PrintResult("cudaMemcpyPeer same device data", "FAIL", "data mismatch");
        }
        std::fill(hostOut.begin(), hostOut.end(), 0);

        cudaStream_t stream = nullptr;
        if (ExpectSuccess("stream for cudaMemcpyPeerAsync", cudaStreamCreate(&stream))) {
            ExpectSuccess("cudaMemcpyPeerAsync same device", cudaMemcpyPeerAsync(dst, 0, src, 0, bytes, stream));
            ExpectSuccess("sync cudaMemcpyPeerAsync", cudaStreamSynchronize(stream));
            ExpectSuccess("copy peer async dst back", cudaMemcpy(hostOut.data(), dst, bytes, cudaMemcpyDeviceToHost));
            ++g_checks;
            if (hostOut == hostIn) {
                PrintResult("cudaMemcpyPeerAsync same device data", "PASS");
            } else {
                ++g_failures;
                PrintResult("cudaMemcpyPeerAsync same device data", "FAIL", "data mismatch");
            }
            ExpectSuccess("destroy peer async stream", cudaStreamDestroy(stream));
        }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13020
        unsigned char *batchDstA = nullptr;
        unsigned char *batchDstB = nullptr;
        if (ExpectSuccess("cudaMalloc batch dst A", cudaMalloc(&batchDstA, bytes / 2)) &&
            ExpectSuccess("cudaMalloc batch dst B", cudaMalloc(&batchDstB, bytes / 2))) {
            void *dsts[2] = {batchDstA, batchDstB};
            const void *srcs[2] = {hostIn.data(), hostIn.data() + bytes / 2};
            size_t sizes[2] = {bytes / 2, bytes / 2};
            cudaMemcpyAttributes attrs[1]{};
            attrs[0].srcAccessOrder = cudaMemcpySrcAccessOrderStream;
            attrs[0].srcLocHint.type = cudaMemLocationTypeHost;
            attrs[0].dstLocHint.type = cudaMemLocationTypeDevice;
            size_t attrsIdxs[1] = {0};
            if (ExpectSuccess("stream for cudaMemcpyBatchAsync", cudaStreamCreate(&stream))) {
                if (ExpectSuccess("cudaMemcpyBatchAsync",
                                  cudaMemcpyBatchAsync(dsts, srcs, sizes, 2, attrs, attrsIdxs, 1, stream))) {
                    if (ExpectSuccess("sync cudaMemcpyBatchAsync", cudaStreamSynchronize(stream))) {
                        ExpectSuccess("copy batch dst A back",
                                      cudaMemcpy(hostOut.data(), batchDstA, bytes / 2, cudaMemcpyDeviceToHost));
                        ExpectSuccess("copy batch dst B back",
                                      cudaMemcpy(hostOut.data() + bytes / 2, batchDstB, bytes / 2,
                                                 cudaMemcpyDeviceToHost));
                        ++g_checks;
                        if (hostOut == hostIn) {
                            PrintResult("cudaMemcpyBatchAsync data", "PASS");
                        } else {
                            ++g_failures;
                            PrintResult("cudaMemcpyBatchAsync data", "FAIL", "data mismatch");
                        }
                    }
                }
                ExpectSuccess("destroy cudaMemcpyBatchAsync stream", cudaStreamDestroy(stream));
            }
        }
        if (batchDstA != nullptr) {
            ExpectSuccess("free batch dst A", cudaFree(batchDstA));
        }
        if (batchDstB != nullptr) {
            ExpectSuccess("free batch dst B", cudaFree(batchDstB));
        }
#else
        Skip("cudaMemcpyBatchAsync", "requires CUDA Runtime 13.2+");
#endif
    }
    if (src != nullptr) {
        ExpectSuccess("free peer async src", cudaFree(src));
    }
    if (dst != nullptr) {
        ExpectSuccess("free peer async dst", cudaFree(dst));
    }

    if (deviceCount < 2) {
        Skip("cudaDeviceEnablePeerAccess", "requires at least two visible devices");
        return;
    }

    int canAccessPeer = 0;
    if (!ExpectSuccess("cudaDeviceCanAccessPeer before enable",
                       cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1)) || canAccessPeer == 0) {
        Skip("cudaDeviceEnablePeerAccess", "device 0 cannot access device 1");
        return;
    }

    ExpectSuccess("cudaSetDevice for peer enable", cudaSetDevice(0));
    ExpectError("cudaDeviceEnablePeerAccess nonzero flags",
                cudaDeviceEnablePeerAccess(1, 1), cudaErrorInvalidValue);

    cudaError_t enableErr = cudaDeviceEnablePeerAccess(1, 0);
    if (enableErr == cudaSuccess || enableErr == cudaErrorPeerAccessAlreadyEnabled) {
        ++g_checks;
        PrintResult("cudaDeviceEnablePeerAccess 0->1", "PASS", cudaGetErrorName(enableErr));
        OptionalSuccess("cudaDeviceDisablePeerAccess 0->1", cudaDeviceDisablePeerAccess(1));
    } else {
        Skip("cudaDeviceEnablePeerAccess 0->1", cudaGetErrorString(enableErr));
    }
}

void CheckMemoryApis()
{
    constexpr size_t count = 4096;
    constexpr size_t bytes = count * sizeof(int);

    int *hostIn = nullptr;
    int *hostOut = nullptr;
    ExpectSuccess("cudaMallocHost hostIn", cudaMallocHost(&hostIn, bytes));
    ExpectSuccess("cudaMallocHost hostOut", cudaMallocHost(&hostOut, bytes));
    if (hostIn == nullptr || hostOut == nullptr) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        hostIn[i] = static_cast<int>(i);
        hostOut[i] = 0;
    }

    int *deviceA = nullptr;
    int *deviceB = nullptr;
    if (!ExpectSuccess("cudaMalloc deviceA", cudaMalloc(&deviceA, bytes)) ||
        !ExpectSuccess("cudaMalloc deviceB", cudaMalloc(&deviceB, bytes))) {
        cudaFreeHost(hostIn);
        cudaFreeHost(hostOut);
        return;
    }

    cudaPointerAttributes ptrAttr{};
    OptionalSuccess("cudaPointerGetAttributes", cudaPointerGetAttributes(&ptrAttr, deviceA));

    ExpectSuccess("cudaMemset", cudaMemset(deviceA, 0, bytes));
    ExpectSuccess("cudaMemcpy H2D", cudaMemcpy(deviceA, hostIn, bytes, cudaMemcpyHostToDevice));
    ExpectSuccess("cudaMemcpy D2D", cudaMemcpy(deviceB, deviceA, bytes, cudaMemcpyDeviceToDevice));
    ExpectSuccess("cudaMemcpy D2H", cudaMemcpy(hostOut, deviceB, bytes, cudaMemcpyDeviceToHost));

    ++g_checks;
    bool ok = true;
    for (size_t i = 0; i < count; ++i) {
        if (hostOut[i] != hostIn[i]) {
            ok = false;
            break;
        }
    }
    PrintResult("cudaMemcpy data", ok ? "PASS" : "FAIL");
    if (!ok) {
        ++g_failures;
    }

    cudaStream_t stream = nullptr;
    if (ExpectSuccess("stream for async memory", cudaStreamCreate(&stream))) {
        ExpectSuccess("cudaMemsetAsync", cudaMemsetAsync(deviceB, 0, bytes, stream));
        ExpectSuccess("cudaMemcpyAsync H2D", cudaMemcpyAsync(deviceA, hostIn, bytes, cudaMemcpyHostToDevice, stream));
        ExpectSuccess("cudaMemcpyAsync D2D", cudaMemcpyAsync(deviceB, deviceA, bytes, cudaMemcpyDeviceToDevice, stream));
        ExpectSuccess("cudaMemcpyAsync D2H", cudaMemcpyAsync(hostOut, deviceB, bytes, cudaMemcpyDeviceToHost, stream));
        ExpectSuccess("cudaStreamSynchronize async memory", cudaStreamSynchronize(stream));
        ExpectSuccess("destroy async memory stream", cudaStreamDestroy(stream));
    }

    constexpr size_t width = 64;
    constexpr size_t height = 8;
    void *pitched = nullptr;
    size_t pitch = 0;
    if (ExpectSuccess("cudaMallocPitch", cudaMallocPitch(&pitched, &pitch, width, height))) {
        std::vector<unsigned char> host2d(pitch * height, 0);
        ExpectSuccess("cudaMemset2D", cudaMemset2D(pitched, pitch, 0x5a, width, height));
        ExpectSuccess("cudaMemcpy2D",
                      cudaMemcpy2D(host2d.data(), pitch, pitched, pitch, width, height,
                                   cudaMemcpyDeviceToHost));

        cudaStream_t stream2d = nullptr;
        if (ExpectSuccess("stream for 2D async", cudaStreamCreate(&stream2d))) {
            std::fill(host2d.begin(), host2d.end(), 0);
            ExpectSuccess("cudaMemset2DAsync",
                          cudaMemset2DAsync(pitched, pitch, 0xa5, width, height, stream2d));
            ExpectSuccess("cudaMemcpy2DAsync",
                          cudaMemcpy2DAsync(host2d.data(), pitch, pitched, pitch, width, height,
                                            cudaMemcpyDeviceToHost, stream2d));
            ExpectSuccess("cudaStreamSynchronize 2D async", cudaStreamSynchronize(stream2d));
            ExpectSuccess("destroy 2D async stream", cudaStreamDestroy(stream2d));
        }
        ExpectSuccess("cudaFree pitched", cudaFree(pitched));
    }

    ExpectSuccess("cudaFree deviceA", cudaFree(deviceA));
    ExpectSuccess("cudaFree deviceB", cudaFree(deviceB));
    ExpectSuccess("cudaFreeHost hostIn", cudaFreeHost(hostIn));
    ExpectSuccess("cudaFreeHost hostOut", cudaFreeHost(hostOut));
}

void Check3DMemoryApis()
{
    constexpr size_t width = 16;
    constexpr size_t height = 4;
    constexpr size_t depth = 3;
    constexpr size_t pitch = 32;
    constexpr size_t total = pitch * height * depth;

    std::vector<unsigned char> src(total, 0);
    std::vector<unsigned char> dst(total, 0);
    std::vector<unsigned char> out(total, 0);
    for (size_t z = 0; z < depth; ++z) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                src[z * pitch * height + y * pitch + x] =
                    static_cast<unsigned char>(1 + z * 17 + y * 5 + x);
            }
        }
    }

    cudaMemcpy3DParms copy{};
    copy.srcPtr = make_cudaPitchedPtr(src.data(), pitch, pitch, height);
    copy.dstPtr = make_cudaPitchedPtr(dst.data(), pitch, pitch, height);
    copy.extent = make_cudaExtent(width, height, depth);
    copy.kind = cudaMemcpyHostToHost;

    if (ExpectSuccess("cudaMemcpy3D H2H", cudaMemcpy3D(&copy))) {
        ++g_checks;
        if (dst == src) {
            PrintResult("cudaMemcpy3D data", "PASS");
        } else {
            ++g_failures;
            PrintResult("cudaMemcpy3D data", "FAIL", "data mismatch");
        }
    }

    cudaStream_t stream = nullptr;
    if (!ExpectSuccess("stream for 3D memory", cudaStreamCreate(&stream))) {
        return;
    }

    std::fill(dst.begin(), dst.end(), 0);
    ExpectSuccess("cudaMemcpy3DAsync H2H", cudaMemcpy3DAsync(&copy, stream));
    ExpectSuccess("sync cudaMemcpy3DAsync", cudaStreamSynchronize(stream));

    unsigned char *deviceMem = nullptr;
    if (ExpectSuccess("cudaMalloc 3D memset buffer", cudaMalloc(&deviceMem, total))) {
        cudaPitchedPtr memsetPtr = make_cudaPitchedPtr(deviceMem, pitch, pitch, height);
        ExpectSuccess("cudaMemset3D", cudaMemset3D(memsetPtr, 0x5a, make_cudaExtent(width, height, depth)));
        ExpectSuccess("copy cudaMemset3D data", cudaMemcpy(out.data(), deviceMem, total, cudaMemcpyDeviceToHost));
        bool memsetOk = true;
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    memsetOk = memsetOk && out[z * pitch * height + y * pitch + x] == 0x5a;
                }
            }
        }
        ++g_checks;
        PrintResult("cudaMemset3D data", memsetOk ? "PASS" : "FAIL");
        if (!memsetOk) {
            ++g_failures;
        }

        ExpectSuccess("cudaMemset3DAsync",
                      cudaMemset3DAsync(memsetPtr, 0xa5, make_cudaExtent(width, height, depth), stream));
        ExpectSuccess("sync cudaMemset3DAsync", cudaStreamSynchronize(stream));
        ExpectSuccess("copy cudaMemset3DAsync data",
                      cudaMemcpy(out.data(), deviceMem, total, cudaMemcpyDeviceToHost));
        ExpectSuccess("cudaFree 3D memset buffer", cudaFree(deviceMem));
    }

    unsigned char *peerSrc = nullptr;
    unsigned char *peerDst = nullptr;
    if (ExpectSuccess("cudaMalloc 3D peer src", cudaMalloc(&peerSrc, total)) &&
        ExpectSuccess("cudaMalloc 3D peer dst", cudaMalloc(&peerDst, total))) {
        ExpectSuccess("seed cudaMemcpy3DPeer src",
                      cudaMemcpy(peerSrc, src.data(), total, cudaMemcpyHostToDevice));

        cudaMemcpy3DPeerParms peer{};
        peer.srcPtr = make_cudaPitchedPtr(peerSrc, pitch, pitch, height);
        peer.srcDevice = 0;
        peer.dstPtr = make_cudaPitchedPtr(peerDst, pitch, pitch, height);
        peer.dstDevice = 0;
        peer.extent = make_cudaExtent(width, height, depth);

        ExpectSuccess("cudaMemcpy3DPeer same device", cudaMemcpy3DPeer(&peer));
        ExpectSuccess("copy cudaMemcpy3DPeer dst",
                      cudaMemcpy(out.data(), peerDst, total, cudaMemcpyDeviceToHost));
        ++g_checks;
        if (out == src) {
            PrintResult("cudaMemcpy3DPeer data", "PASS");
        } else {
            ++g_failures;
            PrintResult("cudaMemcpy3DPeer data", "FAIL", "data mismatch");
        }

        std::fill(out.begin(), out.end(), 0);
        ExpectSuccess("clear cudaMemcpy3DPeerAsync dst", cudaMemset(peerDst, 0, total));
        ExpectSuccess("cudaMemcpy3DPeerAsync same device", cudaMemcpy3DPeerAsync(&peer, stream));
        ExpectSuccess("sync cudaMemcpy3DPeerAsync", cudaStreamSynchronize(stream));
        ExpectSuccess("copy cudaMemcpy3DPeerAsync dst",
                      cudaMemcpy(out.data(), peerDst, total, cudaMemcpyDeviceToHost));
        ++g_checks;
        if (out == src) {
            PrintResult("cudaMemcpy3DPeerAsync data", "PASS");
        } else {
            ++g_failures;
            PrintResult("cudaMemcpy3DPeerAsync data", "FAIL", "data mismatch");
        }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13020
        std::fill(out.begin(), out.end(), 0);
        ExpectSuccess("clear cudaMemcpy3DBatchAsync dst", cudaMemset(peerDst, 0, total));
        cudaMemcpy3DBatchOp batchOp{};
        batchOp.src.type = cudaMemcpyOperandTypePointer;
        batchOp.src.op.ptr.ptr = peerSrc;
        batchOp.src.op.ptr.rowLength = pitch;
        batchOp.src.op.ptr.layerHeight = height;
        batchOp.dst.type = cudaMemcpyOperandTypePointer;
        batchOp.dst.op.ptr.ptr = peerDst;
        batchOp.dst.op.ptr.rowLength = pitch;
        batchOp.dst.op.ptr.layerHeight = height;
        batchOp.extent = make_cudaExtent(width, height, depth);
        batchOp.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        if (ExpectSuccess("cudaMemcpy3DBatchAsync",
                          cudaMemcpy3DBatchAsync(1, &batchOp, 0, stream))) {
            if (ExpectSuccess("sync cudaMemcpy3DBatchAsync", cudaStreamSynchronize(stream))) {
                ExpectSuccess("copy cudaMemcpy3DBatchAsync dst",
                              cudaMemcpy(out.data(), peerDst, total, cudaMemcpyDeviceToHost));
                ++g_checks;
                if (out == src) {
                    PrintResult("cudaMemcpy3DBatchAsync data", "PASS");
                } else {
                    ++g_failures;
                    PrintResult("cudaMemcpy3DBatchAsync data", "FAIL", "data mismatch");
                }
            }
        }
#else
        Skip("cudaMemcpy3DBatchAsync", "requires CUDA Runtime 13.2+");
#endif
    }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13020
    cudaMemcpyAttributes attrs{};
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs.srcLocHint.type = cudaMemLocationTypeDevice;
    attrs.dstLocHint.type = cudaMemLocationTypeDevice;
    if (peerSrc != nullptr && peerDst != nullptr &&
        ExpectSuccess("clear cudaMemcpyWithAttributesAsync dst", cudaMemset(peerDst, 0, total)) &&
        ExpectSuccess("cudaMemcpyWithAttributesAsync",
                      cudaMemcpyWithAttributesAsync(peerDst, peerSrc, width, &attrs, stream))) {
        if (ExpectSuccess("sync cudaMemcpyWithAttributesAsync", cudaStreamSynchronize(stream))) {
            std::fill(out.begin(), out.end(), 0);
            ExpectSuccess("copy cudaMemcpyWithAttributesAsync dst",
                          cudaMemcpy(out.data(), peerDst, total, cudaMemcpyDeviceToHost));
            ++g_checks;
            if (std::equal(src.begin(), src.begin() + width, out.begin())) {
                PrintResult("cudaMemcpyWithAttributesAsync data", "PASS");
            } else {
                ++g_failures;
                PrintResult("cudaMemcpyWithAttributesAsync data", "FAIL", "data mismatch");
            }
        }
    }

    std::fill(out.begin(), out.end(), 0);
    cudaMemcpy3DBatchOp op{};
    op.src.type = cudaMemcpyOperandTypePointer;
    op.src.op.ptr.ptr = src.data();
    op.src.op.ptr.rowLength = pitch;
    op.src.op.ptr.layerHeight = height;
    op.src.op.ptr.locHint.type = cudaMemLocationTypeHost;
    op.dst.type = cudaMemcpyOperandTypePointer;
    op.dst.op.ptr.ptr = peerDst;
    op.dst.op.ptr.rowLength = pitch;
    op.dst.op.ptr.layerHeight = height;
    op.dst.op.ptr.locHint.type = cudaMemLocationTypeDevice;
    op.extent = make_cudaExtent(width, height, depth);
    op.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    if (peerDst != nullptr &&
        ExpectSuccess("clear cudaMemcpy3DWithAttributesAsync dst", cudaMemset(peerDst, 0, total)) &&
        ExpectSuccess("cudaMemcpy3DWithAttributesAsync",
                      cudaMemcpy3DWithAttributesAsync(&op, 0, stream))) {
        if (ExpectSuccess("sync cudaMemcpy3DWithAttrs", cudaStreamSynchronize(stream))) {
            ExpectSuccess("copy cudaMemcpy3DWithAttributesAsync dst",
                          cudaMemcpy(out.data(), peerDst, total, cudaMemcpyDeviceToHost));
            ++g_checks;
            if (out == src) {
                PrintResult("cudaMemcpy3DWithAttributesAsync data", "PASS");
            } else {
                ++g_failures;
                PrintResult("cudaMemcpy3DWithAttributesAsync data", "FAIL", "data mismatch");
            }
        }
    }
#else
    Skip("cudaMemcpyWithAttributesAsync", "requires CUDA Runtime 13.2+");
    Skip("cudaMemcpy3DWithAttributesAsync", "requires CUDA Runtime 13.2+");
#endif

    if (peerSrc != nullptr) {
        ExpectSuccess("free cudaMemcpy3DPeer src", cudaFree(peerSrc));
    }
    if (peerDst != nullptr) {
        ExpectSuccess("free cudaMemcpy3DPeer dst", cudaFree(peerDst));
    }

    ExpectSuccess("destroy 3D memory stream", cudaStreamDestroy(stream));
}

void CheckManagedMemoryApis(int device)
{
    constexpr size_t bytes = 4096;
    int *managed = nullptr;
    cudaError_t allocErr = cudaMallocManaged(&managed, bytes, cudaMemAttachGlobal);
    if (allocErr != cudaSuccess) {
        Skip("cudaMallocManaged", cudaGetErrorString(allocErr));
        return;
    }
    PrintResult("cudaMallocManaged", "PASS");

    managed[0] = 7;

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    cudaMemLocation location{};
    location.type = cudaMemLocationTypeDevice;
    location.id = device;
    OptionalSuccess("cudaMemPrefetchAsync_v2",
                    cudaMemPrefetchAsync(managed, bytes, location, 0, nullptr));
    OptionalSuccess("cudaMemPrefetchAsync",
                    cudaMemPrefetchAsync(managed, bytes, location, 0, nullptr));
    const void *prefetchPtrs[] = {managed};
    size_t prefetchSizes[] = {bytes};
    size_t locationIdxs[] = {0};
    OptionalSuccess("cudaMemPrefetchBatchAsync",
                    cudaMemPrefetchBatchAsync(prefetchPtrs, prefetchSizes, 1,
                                              &location, locationIdxs, 1, 0, nullptr));
#else
    OptionalSuccess("cudaMemPrefetchAsync", cudaMemPrefetchAsync(managed, bytes, device, nullptr));
    Skip("cudaMemPrefetchBatchAsync", "requires CUDA Runtime 13.x+");
#endif
    OptionalSuccess("sync after managed prefetch", cudaDeviceSynchronize());

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    OptionalSuccess("cudaMemAdvise read mostly",
                    cudaMemAdvise(managed, bytes, cudaMemAdviseSetReadMostly, location));
#else
    OptionalSuccess("cudaMemAdvise read mostly",
                    cudaMemAdvise(managed, bytes, cudaMemAdviseSetReadMostly, device));
#endif

    int readMostly = 0;
    if (OptionalSuccess("cudaMemRangeGetAttribute read mostly",
                        cudaMemRangeGetAttribute(&readMostly, sizeof(readMostly),
                                                 cudaMemRangeAttributeReadMostly, managed, bytes))) {
        ++g_checks;
        if (readMostly == 0 || readMostly == 1) {
            PrintResult("cudaMemRangeGetAttribute value", "PASS",
                        readMostly ? "readMostly=1" : "readMostly=0");
        } else {
            ++g_failures;
            PrintResult("cudaMemRangeGetAttribute value", "FAIL", "expected boolean value");
        }
    }

    int readMostlyBatch = 0;
    void *rangeData[] = {&readMostlyBatch};
    size_t rangeDataSizes[] = {sizeof(readMostlyBatch)};
    cudaMemRangeAttribute rangeAttrs[] = {cudaMemRangeAttributeReadMostly};
    OptionalSuccess("cudaMemRangeGetAttributes read mostly",
                    cudaMemRangeGetAttributes(rangeData, rangeDataSizes, rangeAttrs, 1, managed, bytes));

    void *hostPtr = nullptr;
    if (OptionalSuccess("cudaHostAlloc for flags",
                        cudaHostAlloc(&hostPtr, bytes, cudaHostAllocDefault))) {
        unsigned int hostFlags = 0;
        if (OptionalSuccess("cudaHostGetFlags", cudaHostGetFlags(&hostFlags, hostPtr))) {
            char detail[64];
            std::snprintf(detail, sizeof(detail), "flags=%u", hostFlags);
            PrintResult("cudaHostGetFlags value", "PASS", detail);
        }
        ExpectSuccess("cudaFreeHost managed flags host", cudaFreeHost(hostPtr));
    }

    ExpectSuccess("cudaFree managed", cudaFree(managed));
}

void CheckStreamApis()
{
    int leastPriority = 0;
    int greatestPriority = 0;
    OptionalSuccess("priority range for stream", cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

    cudaStream_t stream = nullptr;
    if (!ExpectSuccess("cudaStreamCreateWithFlags",
                       cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))) {
        return;
    }

    unsigned int flags = 0;
    OptionalSuccess("cudaStreamGetFlags", cudaStreamGetFlags(stream, &flags));

    int priority = 0;
    OptionalSuccess("cudaStreamGetPriority", cudaStreamGetPriority(stream, &priority));

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    unsigned long long streamId = 0;
    OptionalSuccess("cudaStreamGetId", cudaStreamGetId(stream, &streamId));
#else
    Skip("cudaStreamGetId", "requires CUDA 12.x headers");
#endif

    cudaError_t query = cudaStreamQuery(stream);
    if (query == cudaSuccess || query == cudaErrorNotReady) {
        ++g_checks;
        PrintResult("cudaStreamQuery", "PASS", cudaGetErrorName(query));
    } else {
        ++g_failures;
        PrintResult("cudaStreamQuery", "FAIL", cudaGetErrorString(query));
    }

    ExpectSuccess("cudaStreamSynchronize", cudaStreamSynchronize(stream));
    ExpectSuccess("cudaStreamDestroy", cudaStreamDestroy(stream));

    cudaStream_t priorityStream = nullptr;
    OptionalSuccess("cudaStreamCreateWithPriority",
                    cudaStreamCreateWithPriority(&priorityStream, cudaStreamNonBlocking, greatestPriority));
    if (priorityStream != nullptr) {
        ExpectSuccess("destroy priority stream", cudaStreamDestroy(priorityStream));
    }
}

void CheckEventApis()
{
    cudaStream_t stream = nullptr;
    if (!ExpectSuccess("event stream", cudaStreamCreate(&stream))) {
        return;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    ExpectSuccess("cudaEventCreate", cudaEventCreate(&start));
    ExpectSuccess("cudaEventCreateWithFlags", cudaEventCreateWithFlags(&stop, cudaEventDefault));

    ExpectSuccess("cudaEventRecord start", cudaEventRecord(start, stream));
    ExpectSuccess("cudaStreamWaitEvent", cudaStreamWaitEvent(stream, start, 0));
    ExpectSuccess("cudaEventRecord stop", cudaEventRecord(stop, stream));

    cudaError_t query = cudaEventQuery(stop);
    if (query == cudaSuccess || query == cudaErrorNotReady) {
        ++g_checks;
        PrintResult("cudaEventQuery", "PASS", cudaGetErrorName(query));
    } else {
        ++g_failures;
        PrintResult("cudaEventQuery", "FAIL", cudaGetErrorString(query));
    }

    ExpectSuccess("cudaEventSynchronize", cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    ExpectSuccess("cudaEventElapsedTime", cudaEventElapsedTime(&elapsed, start, stop));
    ExpectSuccess("cudaEventDestroy start", cudaEventDestroy(start));
    ExpectSuccess("cudaEventDestroy stop", cudaEventDestroy(stop));
    ExpectSuccess("destroy event stream", cudaStreamDestroy(stream));
}

void CheckHostRegistrationApis()
{
    constexpr size_t size = 4096;
    void *hostPtr = nullptr;
    if (posix_memalign(&hostPtr, 4096, size) != 0 || hostPtr == nullptr) {
        ++g_failures;
        PrintResult("posix_memalign", "FAIL");
        return;
    }
    std::memset(hostPtr, 0x3c, size);

    if (ExpectSuccess("cudaHostRegister default", cudaHostRegister(hostPtr, size, cudaHostRegisterDefault))) {
        ExpectSuccess("cudaHostUnregister default", cudaHostUnregister(hostPtr));
    }

    cudaError_t regErr = cudaHostRegister(hostPtr, size, cudaHostRegisterMapped);
    if (regErr != cudaSuccess) {
        Skip("cudaHostRegister mapped", cudaGetErrorString(regErr));
        std::free(hostPtr);
        return;
    }
    PrintResult("cudaHostRegister mapped", "PASS");

    void *deviceAlias = nullptr;
    ExpectSuccess("cudaHostGetDevicePointer", cudaHostGetDevicePointer(&deviceAlias, hostPtr, 0));
    ExpectSuccess("cudaHostUnregister mapped", cudaHostUnregister(hostPtr));

    std::free(hostPtr);
}

void HostCallback(void *userData)
{
    int *value = static_cast<int *>(userData);
    *value = 1234;
}

void CheckLaunchHostFuncApi()
{
    cudaStream_t stream = nullptr;
    if (!ExpectSuccess("stream for cudaLaunchHostFunc", cudaStreamCreate(&stream))) {
        return;
    }

    ExpectSuccess("cudaLaunchHostFunc null fn", cudaLaunchHostFunc(stream, nullptr, nullptr));

    int callbackValue = 0;
    if (OptionalSuccess("cudaLaunchHostFunc", cudaLaunchHostFunc(stream, HostCallback, &callbackValue))) {
        ExpectSuccess("sync cudaLaunchHostFunc", cudaStreamSynchronize(stream));
        ++g_checks;
        if (callbackValue == 1234) {
            PrintResult("cudaLaunchHostFunc callback", "PASS");
        } else {
            ++g_failures;
            PrintResult("cudaLaunchHostFunc callback", "FAIL", "callback did not run");
        }
    }

    ExpectSuccess("destroy host func stream", cudaStreamDestroy(stream));
}

void CheckLaunchKernelApi()
{
    void *invalidArgs[] = {nullptr};
    ExpectError("cudaLaunchKernel null func",
                cudaLaunchKernel(nullptr, dim3(1), dim3(1), invalidArgs, 0, nullptr),
                cudaErrorInvalidDeviceFunction);
    ExpectErrorAny("cudaLaunchKernel zero grid",
                   cudaLaunchKernel(reinterpret_cast<const void *>(ApiRuntimeCoverageKernel),
                                    dim3(0), dim3(1), invalidArgs, 0, nullptr),
                   cudaErrorInvalidConfiguration, cudaErrorInvalidValue);
    ExpectErrorAny("cudaLaunchKernel zero block",
                   cudaLaunchKernel(reinterpret_cast<const void *>(ApiRuntimeCoverageKernel),
                                    dim3(1), dim3(0), invalidArgs, 0, nullptr),
                   cudaErrorInvalidConfiguration, cudaErrorInvalidValue);

    int *deviceValue = nullptr;
    int hostValue = 0;
    if (!ExpectSuccess("cudaMalloc launch kernel value", cudaMalloc(&deviceValue, sizeof(int)))) {
        return;
    }

    cudaStream_t stream = nullptr;
    if (ExpectSuccess("stream for cudaLaunchKernel", cudaStreamCreate(&stream))) {
        void *args[] = {&deviceValue};
        ExpectSuccess("cudaLaunchKernel",
                      cudaLaunchKernel(reinterpret_cast<const void *>(ApiRuntimeCoverageKernel),
                                       dim3(1), dim3(1), args, 0, stream));
        ExpectSuccess("sync cudaLaunchKernel", cudaStreamSynchronize(stream));
        ExpectSuccess("copy cudaLaunchKernel result",
                      cudaMemcpy(&hostValue, deviceValue, sizeof(hostValue), cudaMemcpyDeviceToHost));
        ++g_checks;
        if (hostValue == 42) {
            PrintResult("cudaLaunchKernel result", "PASS");
        } else {
            ++g_failures;
            PrintResult("cudaLaunchKernel result", "FAIL", "unexpected value");
        }
        ExpectSuccess("destroy cudaLaunchKernel stream", cudaStreamDestroy(stream));
    }

    ExpectSuccess("free cudaLaunchKernel value", cudaFree(deviceValue));
}

int RunIpcChild(const char *payloadPath)
{
    FILE *fp = std::fopen(payloadPath, "rb");
    if (fp == nullptr) {
        std::printf("  child open IPC payload                  FAIL\n");
        return EXIT_FAILURE;
    }

    IpcPayload payload{};
    const size_t readItems = std::fread(&payload, sizeof(payload), 1, fp);
    std::fclose(fp);
    if (readItems != 1) {
        std::printf("  child read IPC payload                  FAIL\n");
        return EXIT_FAILURE;
    }

    int failures = 0;
    auto childCheck = [&failures](const char *name, cudaError_t err) -> bool {
        if (err == cudaSuccess) {
            std::printf("  %-34s PASS\n", name);
            return true;
        }
        ++failures;
        std::printf("  %-34s FAIL - %s\n", name, cudaGetErrorString(err));
        return false;
    };

    childCheck("child cudaSetDevice", cudaSetDevice(0));

    void *remotePtr = nullptr;
    cudaEvent_t remoteEvent = nullptr;
    cudaStream_t stream = nullptr;
    int observed = 0;
    bool openedMem = childCheck("child cudaIpcOpenMemHandle",
                                cudaIpcOpenMemHandle(&remotePtr, payload.memHandle,
                                                     cudaIpcMemLazyEnablePeerAccess));
    bool openedEvent = childCheck("child cudaIpcOpenEventHandle",
                                  cudaIpcOpenEventHandle(&remoteEvent, payload.eventHandle));
    bool streamReady = childCheck("child cudaStreamCreate", cudaStreamCreate(&stream));

    if (openedMem && openedEvent && streamReady) {
        childCheck("child cudaStreamWaitEvent", cudaStreamWaitEvent(stream, remoteEvent, 0));
        childCheck("child cudaMemcpyAsync",
                   cudaMemcpyAsync(&observed, remotePtr, sizeof(observed),
                                   cudaMemcpyDeviceToHost, stream));
        childCheck("child cudaStreamSynchronize", cudaStreamSynchronize(stream));
        if (observed == 20260910) {
            std::printf("  %-34s PASS - value=%d\n", "child ipc value", observed);
        } else {
            ++failures;
            std::printf("  %-34s FAIL - value=%d\n", "child ipc value", observed);
        }
    }

    if (openedMem) {
        childCheck("child cudaIpcCloseMemHandle", cudaIpcCloseMemHandle(remotePtr));
    }
    if (stream != nullptr) {
        childCheck("child cudaStreamDestroy", cudaStreamDestroy(stream));
    }

    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int RunVmmImportChild(const char *fdText)
{
    const int fd = std::atoi(fdText);
    int failures = 0;

    auto childCudaCheck = [&failures](const char *name, cudaError_t err) -> bool {
        if (err == cudaSuccess) {
            std::printf("  %-34s PASS\n", name);
            return true;
        }
        ++failures;
        std::printf("  %-34s FAIL - %s\n", name, cudaGetErrorName(err));
        return false;
    };

    auto childCheck = [&failures](const char *name, CUresult result) -> bool {
        if (result == CUDA_SUCCESS) {
            std::printf("  %-34s PASS\n", name);
            return true;
        }
        ++failures;
        std::string detail = CuResultDetail(result);
        std::printf("  %-34s FAIL - %s\n", name, detail.c_str());
        return false;
    };

    childCudaCheck("child cudaSetDevice vmm", cudaSetDevice(0));
    CUmemGenericAllocationHandle importedHandle = 0;
    if (childCheck("child cuMemImportFromShareableHandle",
                   cuMemImportFromShareableHandle(&importedHandle,
                                                  reinterpret_cast<void *>(static_cast<intptr_t>(fd)),
                                                  CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR))) {
        childCheck("child cuMemRelease imported", cuMemRelease(importedHandle));
    }

    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

void CheckThreadExchangeStreamCaptureModeApi()
{
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    if (ExpectSuccess("cudaThreadExchangeStreamCaptureMode",
                      cudaThreadExchangeStreamCaptureMode(&mode))) {
        ++g_checks;
        if (mode == cudaStreamCaptureModeGlobal ||
            mode == cudaStreamCaptureModeThreadLocal ||
            mode == cudaStreamCaptureModeRelaxed) {
            PrintResult("thread capture mode range", "PASS");
        } else {
            ++g_failures;
            PrintResult("thread capture mode range", "FAIL", "unexpected previous mode");
        }
        ExpectSuccess("restore thread capture mode",
                      cudaThreadExchangeStreamCaptureMode(&mode));
    }
}

void CheckBasicGraphCaptureApis()
{
    int *deviceValue = nullptr;
    int hostValue = 0;
    if (!ExpectSuccess("basic graph output malloc",
                       cudaMalloc(reinterpret_cast<void **>(&deviceValue), sizeof(int)))) {
        return;
    }

    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    if (ExpectSuccess("basic graph stream", cudaStreamCreate(&stream))) {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusInvalidated;
        ExpectError("cudaStreamIsCapturing null output",
                    cudaStreamIsCapturing(stream, nullptr), cudaErrorInvalidValue);
        if (ExpectSuccess("cudaStreamIsCapturing none", cudaStreamIsCapturing(stream, &status))) {
            ++g_checks;
            if (status == cudaStreamCaptureStatusNone) {
                PrintResult("cudaStreamIsCapturing none status", "PASS");
            } else {
                ++g_failures;
                PrintResult("cudaStreamIsCapturing none status", "FAIL", "unexpected status");
            }
        }

        if (ExpectSuccess("basic cudaStreamBeginCapture",
                          cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal))) {
            ExpectSuccess("cudaStreamIsCapturing active", cudaStreamIsCapturing(stream, &status));
            ApiRuntimeCoverageKernel<<<1, 1, 0, stream>>>(deviceValue);
            if (ExpectSuccess("basic cudaStreamEndCapture", cudaStreamEndCapture(stream, &graph)) &&
                graph != nullptr &&
                ExpectSuccess("cudaGraphDebugDotPrint",
                              cudaGraphDebugDotPrint(graph, "/tmp/apiRuntimeCoverage_cuda_graph.dot", 0)) &&
                ExpectSuccess("basic cudaGraphInstantiate",
                              cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0))) {
                ExpectSuccess("basic cudaGraphLaunch", cudaGraphLaunch(graphExec, stream));
                ExpectSuccess("sync basic cudaGraphLaunch", cudaStreamSynchronize(stream));
                ExpectSuccess("copy basic graph result",
                              cudaMemcpy(&hostValue, deviceValue, sizeof(hostValue), cudaMemcpyDeviceToHost));
                ++g_checks;
                if (hostValue == 42) {
                    PrintResult("basic cudaGraphLaunch result", "PASS");
                } else {
                    ++g_failures;
                    PrintResult("basic cudaGraphLaunch result", "FAIL", "unexpected value");
                }
                ExpectSuccess("basic cudaGraphExecDestroy", cudaGraphExecDestroy(graphExec));
                graphExec = nullptr;
            }
        }
        if (graphExec != nullptr) {
            ExpectSuccess("cleanup basic cudaGraphExecDestroy", cudaGraphExecDestroy(graphExec));
        }
        if (graph != nullptr) {
            ExpectSuccess("cleanup basic cudaGraphDestroy", cudaGraphDestroy(graph));
        }
        ExpectSuccess("destroy basic graph stream", cudaStreamDestroy(stream));
    }

    ExpectSuccess("free basic graph output", cudaFree(deviceValue));
}

void CheckIncrementalHostAlloc(size_t bytes)
{
    void *hostAllocPtr = nullptr;
    if (ExpectSuccess("cudaHostAlloc", cudaHostAlloc(&hostAllocPtr, bytes, cudaHostAllocPortable))) {
        ExpectSuccess("cudaFreeHost cudaHostAlloc", cudaFreeHost(hostAllocPtr));
    }
}

void CheckIncrementalEventRecord()
{
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    if (ExpectSuccess("incremental stream", cudaStreamCreate(&stream)) &&
        ExpectSuccess("incremental event", cudaEventCreate(&event))) {
        ExpectSuccess("cudaEventRecordWithFlags",
                      cudaEventRecordWithFlags(event, stream, cudaEventRecordDefault));
        ExpectSuccess("sync record-with-flags stream", cudaStreamSynchronize(stream));
        ExpectSuccess("destroy incremental event", cudaEventDestroy(event));
        ExpectSuccess("destroy incremental stream", cudaStreamDestroy(stream));
    }
}

void CheckStreamAttributeApis()
{
    cudaStream_t stream = nullptr;
    if (!ExpectSuccess("stream attribute stream", cudaStreamCreate(&stream))) {
        return;
    }

    cudaStreamAttrValue setValue{};
    cudaStreamAttrValue getValue{};

#if defined(__CUDACC__)
    if (std::getenv("CUDA_MAPPING_RUN_STREAM_ATTRIBUTE") == nullptr) {
        Skip("cudaStreamSetAttribute", "set CUDA_MAPPING_RUN_STREAM_ATTRIBUTE=1 to run CUDA access-policy path");
        Skip("cudaStreamGetAttribute", "set CUDA_MAPPING_RUN_STREAM_ATTRIBUTE=1 to run CUDA access-policy path");
    } else {
        setValue.accessPolicyWindow.base_ptr = nullptr;
        setValue.accessPolicyWindow.num_bytes = 0;
        setValue.accessPolicyWindow.hitRatio = 0.0;
        setValue.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
        setValue.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
        if (OptionalSuccess("cudaStreamSetAttribute",
                            cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &setValue))) {
            OptionalSuccess("cudaStreamGetAttribute",
                            cudaStreamGetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &getValue));
        }
    }
#else
    setValue.userCustomTag = 2026U;
    if (OptionalSuccess("cudaStreamSetAttribute",
                        cudaStreamSetAttribute(stream,
                                               static_cast<cudaStreamAttrID>(ACL_STREAM_ATTR_USER_CUSTOM_TAG),
                                               &setValue))) {
        OptionalSuccess("cudaStreamGetAttribute",
                        cudaStreamGetAttribute(stream,
                                               static_cast<cudaStreamAttrID>(ACL_STREAM_ATTR_USER_CUSTOM_TAG),
                                               &getValue));
    }
#endif

    ExpectSuccess("destroy stream attribute stream", cudaStreamDestroy(stream));
}

void CheckFunctionQueryApis()
{
    cudaFunction_t function = nullptr;
    OptionalSuccess("cudaGetFuncBySymbol",
                    cudaGetFuncBySymbol(&function,
                                        reinterpret_cast<const void *>(ApiRuntimeCoverageKernelWithValue)));

    const char *funcName = nullptr;
    if (OptionalSuccess("cudaFuncGetName",
                        cudaFuncGetName(&funcName,
                                        reinterpret_cast<const void *>(ApiRuntimeCoverageKernelWithValue)))) {
        ++g_checks;
        if (funcName != nullptr && funcName[0] != '\0') {
            PrintResult("cudaFuncGetName non-empty", "PASS", funcName);
        } else {
            ++g_failures;
            PrintResult("cudaFuncGetName non-empty", "FAIL", "empty name");
        }
    }

    size_t dynamicSmem = 0;
    if (OptionalSuccess("cudaOccupancyAvailableDynamicSMemPerBlock",
                        cudaOccupancyAvailableDynamicSMemPerBlock(&dynamicSmem,
                                                                  ApiRuntimeCoverageKernel,
                                                                  1,
                                                                  64))) {
        ++g_checks;
        if (dynamicSmem > 0) {
            char detail[96];
            std::snprintf(detail, sizeof(detail), "bytes=%zu", dynamicSmem);
            PrintResult("occupancy dynamic smem value", "PASS", detail);
        } else {
            ++g_failures;
            PrintResult("occupancy dynamic smem value", "FAIL", "expected nonzero bytes");
        }
    }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    size_t paramCount = 0;
    if (OptionalSuccess("cudaFuncGetParamCount",
                        cudaFuncGetParamCount(reinterpret_cast<const void *>(ApiRuntimeCoverageKernelWithValue),
                                              &paramCount))) {
        char detail[96];
        std::snprintf(detail, sizeof(detail), "paramCount=%zu", paramCount);
        PrintResult("cudaFuncGetParamCount value", "PASS", detail);
        if (paramCount > 0) {
            size_t offset = 0;
            size_t size = 0;
            ExpectSuccess("cudaFuncGetParamInfo first param",
                          cudaFuncGetParamInfo(reinterpret_cast<const void *>(ApiRuntimeCoverageKernelWithValue),
                                               0, &offset, &size));
        } else {
            Skip("cudaFuncGetParamInfo first param", "kernel reports zero params");
        }
    }
    ExpectError("cudaFuncGetParamCount null count",
                cudaFuncGetParamCount(reinterpret_cast<const void *>(ApiRuntimeCoverageKernelWithValue), nullptr),
                cudaErrorInvalidValue);
    ExpectError("cudaFuncGetParamInfo both outputs null",
                cudaFuncGetParamInfo(reinterpret_cast<const void *>(ApiRuntimeCoverageKernelWithValue),
                                     0, nullptr, nullptr),
                cudaErrorInvalidValue);
#else
    Skip("cudaFuncGetParamCount", "requires CUDA headers that declare cudaFuncGetParamCount");
    Skip("cudaFuncGetParamInfo", "requires CUDA headers that declare cudaFuncGetParamInfo");
#endif
}

void CheckLaunchHostFuncV2Api()
{
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    cudaStream_t stream = nullptr;
    if (!ExpectSuccess("stream for cudaLaunchHostFunc_v2", cudaStreamCreate(&stream))) {
        return;
    }

    ExpectSuccess("cudaLaunchHostFunc_v2 null fn",
                  cudaLaunchHostFunc_v2(stream, nullptr, nullptr, 0));

    int callbackValue = 0;
    if (OptionalSuccess("cudaLaunchHostFunc_v2",
                        cudaLaunchHostFunc_v2(stream, HostCallback, &callbackValue, 0))) {
        ExpectSuccess("sync cudaLaunchHostFunc_v2", cudaStreamSynchronize(stream));
        ++g_checks;
        if (callbackValue == 1234) {
            PrintResult("cudaLaunchHostFunc_v2 callback", "PASS");
        } else {
            ++g_failures;
            PrintResult("cudaLaunchHostFunc_v2 callback", "FAIL", "callback did not run");
        }
    }

    ExpectSuccess("destroy host func v2 stream", cudaStreamDestroy(stream));
#else
    Skip("cudaLaunchHostFunc_v2", "requires CUDA headers that declare cudaLaunchHostFunc_v2");
#endif
}

void CheckLaunchKernelExApi()
{
    int *deviceValue = nullptr;
    int hostValue = 0;
    constexpr int expectedValue = 2026;
    if (!ExpectSuccess("cudaMalloc launch kernel ex value", cudaMalloc(&deviceValue, sizeof(int)))) {
        return;
    }

    cudaStream_t stream = nullptr;
    if (ExpectSuccess("stream for cudaLaunchKernelEx", cudaStreamCreate(&stream))) {
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(1, 1, 1);
        config.blockDim = dim3(32, 1, 1);
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        config.attrs = nullptr;
        config.numAttrs = 0;

#if defined(__CUDACC__)
        cudaError_t launchErr = cudaLaunchKernelEx(&config, ApiRuntimeCoverageKernelWithValue,
                                                   deviceValue, expectedValue);
#else
        void *args[] = {&deviceValue, const_cast<int *>(&expectedValue)};
        cudaError_t launchErr = cudaLaunchKernelEx(&config,
                                                   reinterpret_cast<const void *>(ApiRuntimeCoverageKernelWithValue),
                                                   args);
#endif
        if (ExpectSuccess("cudaLaunchKernelEx", launchErr)) {
            ExpectSuccess("sync cudaLaunchKernelEx", cudaStreamSynchronize(stream));
            ExpectSuccess("copy cudaLaunchKernelEx result",
                          cudaMemcpy(&hostValue, deviceValue, sizeof(hostValue), cudaMemcpyDeviceToHost));
            ++g_checks;
            if (hostValue == expectedValue) {
                PrintResult("cudaLaunchKernelEx result", "PASS");
            } else {
                ++g_failures;
                PrintResult("cudaLaunchKernelEx result", "FAIL", "unexpected value");
            }
        }
        ExpectSuccess("destroy cudaLaunchKernelEx stream", cudaStreamDestroy(stream));
    }

    ExpectSuccess("free cudaLaunchKernelEx value", cudaFree(deviceValue));
}

void CheckIncrementalFunctionAndGraphApis()
{
    cudaFuncAttributes attrs{};
#if defined(__CUDACC__)
    ExpectSuccess("cudaFuncGetAttributes", cudaFuncGetAttributes(&attrs, ApiRuntimeCoverageKernel));
#else
    ExpectError("cudaFuncGetAttributes no CANN equivalent",
                cudaFuncGetAttributes(&attrs, reinterpret_cast<const void *>(ApiRuntimeCoverageKernel)),
                cudaErrorNotSupported);
#endif

    cudaGraph_t graph = nullptr;
    if (ExpectSuccess("cudaGraphCreate incremental", cudaGraphCreate(&graph, 0))) {
        size_t nodeCount = 0;
        ExpectSuccess("cudaGraphGetNodes", cudaGraphGetNodes(graph, nullptr, &nodeCount));

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
        int *conditionalOut = nullptr;
        cudaStream_t parentStream = nullptr;
        cudaStream_t bodyStream = nullptr;
        cudaGraph_t conditionalGraph = nullptr;
        cudaGraphExec_t conditionalExec = nullptr;
        cudaGraphConditionalHandle handle = 0;
        bool conditionalReady = false;
        bool parentCaptureOpen = false;
        bool bodyCaptureOpen = false;
        if (ExpectSuccess("conditional output malloc", cudaMalloc(reinterpret_cast<void **>(&conditionalOut), sizeof(int))) &&
            ExpectSuccess("conditional output init", cudaMemset(conditionalOut, 0, sizeof(int))) &&
            ExpectSuccess("conditional parent stream", cudaStreamCreate(&parentStream)) &&
            ExpectSuccess("conditional body stream", cudaStreamCreate(&bodyStream)) &&
            ExpectSuccess("conditional parent capture",
                          cudaStreamBeginCapture(parentStream, cudaStreamCaptureModeGlobal))) {
            parentCaptureOpen = true;
            cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
            const cudaGraphNode_t *deps = nullptr;
            size_t depCount = 0;
            if (ExpectSuccess("cudaStreamGetCaptureInfo conditional",
                              GetStreamCaptureInfoCompat(parentStream, &status, nullptr, &conditionalGraph,
                                                         &deps, &depCount)) &&
                ExpectSuccess("cudaGraphConditionalHandleCreate",
                              cudaGraphConditionalHandleCreate(&handle, conditionalGraph, 1,
                                                               cudaGraphCondAssignDefault))) {
                cudaGraphNode_t conditionalNode = nullptr;
                cudaGraphNodeParams conditionalParams{};
                conditionalParams.type = cudaGraphNodeTypeConditional;
                conditionalParams.conditional.handle = handle;
                conditionalParams.conditional.type = cudaGraphCondTypeWhile;
                conditionalParams.conditional.size = 1;
                if (ExpectSuccess("cudaGraphAddNode conditional",
                                  GraphAddNodeCompat(&conditionalNode, conditionalGraph, nullptr, 0,
                                                     &conditionalParams))) {
                    cudaGraph_t bodyGraph = conditionalParams.conditional.phGraph_out[0];
                    if (ExpectSuccess("cudaStreamBeginCaptureToGraph",
                                      cudaStreamBeginCaptureToGraph(bodyStream, bodyGraph, nullptr,
                                                                    nullptr, 0, cudaStreamCaptureModeGlobal))) {
                        bodyCaptureOpen = true;
                        ApiRuntimeCoverageConditionalBodyKernel<<<1, 1, 0, bodyStream>>>(conditionalOut, handle);
                        ExpectSuccess("cudaStreamEndCaptureToGraph", cudaStreamEndCapture(bodyStream, nullptr));
                        bodyCaptureOpen = false;
                    }
                    if (ExpectSuccess("conditional parent end capture",
                                      cudaStreamEndCapture(parentStream, &conditionalGraph)) &&
                        ExpectSuccess("conditional graph instantiate",
                                      cudaGraphInstantiate(&conditionalExec, conditionalGraph, nullptr, nullptr, 0))) {
                        conditionalReady = true;
                    }
                    parentCaptureOpen = false;
                }
            }
        }
        if (bodyCaptureOpen) {
            (void)cudaStreamEndCapture(bodyStream, nullptr);
        }
        if (parentCaptureOpen) {
            (void)cudaStreamEndCapture(parentStream, &conditionalGraph);
        }
        if (conditionalReady) {
            ExpectSuccess("cudaGraphSetConditional device path",
                          cudaGraphLaunch(conditionalExec, nullptr));
            ExpectSuccess("sync cudaGraphSetConditional graph", cudaDeviceSynchronize());
            int observed = 0;
            ExpectSuccess("conditional output copy",
                          cudaMemcpy(&observed, conditionalOut, sizeof(observed), cudaMemcpyDeviceToHost));
            ++g_checks;
            if (observed == 314) {
                PrintResult("conditional graph body result", "PASS");
            } else {
                ++g_failures;
                PrintResult("conditional graph body result", "FAIL", "unexpected body output");
            }
            ExpectSuccess("cudaGraphExecDestroy conditional", cudaGraphExecDestroy(conditionalExec));
        }
        if (bodyStream) {
            ExpectSuccess("destroy conditional body stream", cudaStreamDestroy(bodyStream));
        }
        if (parentStream) {
            ExpectSuccess("destroy conditional parent stream", cudaStreamDestroy(parentStream));
        }
        if (conditionalOut) {
            ExpectSuccess("free conditional output", cudaFree(conditionalOut));
        }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13020
        cudaGraph_t conditionalV2Graph = nullptr;
        if (ExpectSuccess("cudaGraphCreate conditional v2",
                          cudaGraphCreate(&conditionalV2Graph, 0))) {
            cudaGraphConditionalHandle conditionalV2Handle = 0;
            ExpectSuccess("cudaGraphConditionalHandleCreate_v2",
                          cudaGraphConditionalHandleCreate_v2(
                              &conditionalV2Handle, conditionalV2Graph, nullptr, 0, 0));
            ExpectSuccess("cudaGraphDestroy conditional v2",
                          cudaGraphDestroy(conditionalV2Graph));
        }
#else
        Skip("cudaGraphConditionalHandleCreate_v2", "requires CUDA Runtime 13.2+");
#endif
#else
        Skip("cudaGraphSetConditional", "requires CUDA 12.x conditional graph support");
#endif
        ExpectSuccess("cudaGraphDestroy incremental", cudaGraphDestroy(graph));
    }
}

void CheckIncrementalSymbolApis()
{
    int value = 77;
    ExpectSuccess("cudaMemcpyToSymbol", cudaMemcpyToSymbol(g_apiRuntimeCoverageSymbol, &value, sizeof(value)));

    void *symbolAddress = nullptr;
    if (ExpectSuccess("cudaGetSymbolAddress", cudaGetSymbolAddress(&symbolAddress, g_apiRuntimeCoverageSymbol))) {
        int observed = 0;
        ExpectSuccess("cudaMemcpy symbol value",
                      cudaMemcpy(&observed, symbolAddress, sizeof(observed), cudaMemcpyDeviceToHost));
        ++g_checks;
        if (observed == value) {
            PrintResult("cudaGetSymbolAddress value", "PASS");
        } else {
            ++g_failures;
            PrintResult("cudaGetSymbolAddress value", "FAIL", "unexpected symbol value");
        }
    }

    cudaStream_t stream = nullptr;
    if (!ExpectSuccess("symbol async stream", cudaStreamCreate(&stream))) {
        return;
    }

    value = 2026;
    ExpectSuccess("cudaMemcpyToSymbolAsync",
                  cudaMemcpyToSymbolAsync(g_apiRuntimeCoverageSymbol, &value, sizeof(value),
                                          0, cudaMemcpyHostToDevice, stream));
    ExpectSuccess("sync cudaMemcpyToSymbolAsync", cudaStreamSynchronize(stream));

    int asyncObserved = 0;
    ExpectSuccess("cudaMemcpyFromSymbol",
                  cudaMemcpyFromSymbol(&asyncObserved, g_apiRuntimeCoverageSymbol,
                                       sizeof(asyncObserved), 0, cudaMemcpyDeviceToHost));
    ++g_checks;
    if (asyncObserved == value) {
        PrintResult("cudaMemcpyFromSymbol data", "PASS");
    } else {
        ++g_failures;
        PrintResult("cudaMemcpyFromSymbol data", "FAIL", "unexpected symbol value");
    }

    asyncObserved = 0;
    ExpectSuccess("cudaMemcpyFromSymbolAsync",
                  cudaMemcpyFromSymbolAsync(&asyncObserved, g_apiRuntimeCoverageSymbol,
                                            sizeof(asyncObserved), 0, cudaMemcpyDeviceToHost, stream));
    ExpectSuccess("sync cudaMemcpyFromSymbolAsync", cudaStreamSynchronize(stream));
    ++g_checks;
    if (asyncObserved == value) {
        PrintResult("cudaMemcpyFromSymbolAsync data", "PASS");
    } else {
        ++g_failures;
        PrintResult("cudaMemcpyFromSymbolAsync data", "FAIL", "unexpected symbol value");
    }

    size_t symbolSize = 0;
    ExpectSuccess("cudaGetSymbolSize", cudaGetSymbolSize(&symbolSize, g_apiRuntimeCoverageSymbol));
    ++g_checks;
    if (symbolSize >= sizeof(value)) {
        char detail[96];
        std::snprintf(detail, sizeof(detail), "size=%zu", symbolSize);
        PrintResult("cudaGetSymbolSize value", "PASS", detail);
    } else {
        ++g_failures;
        PrintResult("cudaGetSymbolSize value", "FAIL", "smaller than copied symbol");
    }

    ExpectSuccess("destroy symbol async stream", cudaStreamDestroy(stream));
}

void CheckIncrementalCaptureInfo()
{
    cudaStream_t captureStream = nullptr;
    if (ExpectSuccess("capture info stream", cudaStreamCreate(&captureStream))) {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusInvalidated;
        cudaGraph_t graph = nullptr;
        const cudaGraphNode_t *deps = nullptr;
        size_t depCount = 0;
        ExpectSuccess("cudaStreamGetCaptureInfo",
                      GetStreamCaptureInfoCompat(captureStream, &status, nullptr, &graph, &deps, &depCount));
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
        ExpectSuccess("cudaStreamGetCaptureInfo edge data",
                      GetStreamCaptureInfoCompat(captureStream, &status, nullptr, &graph, &deps, &depCount));
#elif defined(CUDART_VERSION) && CUDART_VERSION >= 12030
        ExpectSuccess("cudaStreamGetCaptureInfo_v3",
                      cudaStreamGetCaptureInfo_v3(captureStream, &status, nullptr, &graph, &deps, nullptr, &depCount));
#elif defined(CUDART_VERSION) && CUDART_VERSION >= 11030
        ExpectSuccess("cudaStreamGetCaptureInfo_v3",
                      cudaStreamGetCaptureInfo_v2(captureStream, &status, nullptr, &graph, &deps, &depCount));
#else
        Skip("cudaStreamGetCaptureInfo_v3", "requires CUDA 11.3+ headers");
#endif
        ExpectSuccess("destroy capture info stream", cudaStreamDestroy(captureStream));
    }
}

void CheckIpcApis(const char *selfPath)
{
    constexpr int expectedValue = 20260910;
    constexpr size_t ipcBytes = 4096;
    char payloadPath[256];
    std::snprintf(payloadPath, sizeof(payloadPath), "/tmp/apiRuntimeCoverage_ipc_%ld.bin",
                  static_cast<long>(getpid()));

    int *deviceValue = nullptr;
    cudaEvent_t event = nullptr;
    IpcPayload payload{};
    bool memReady = false;
    bool eventReady = false;

    if (ExpectSuccess("ipc cudaMalloc", cudaMalloc(&deviceValue, ipcBytes)) &&
        ExpectSuccess("ipc seed value",
                      cudaMemcpy(deviceValue, &expectedValue, sizeof(expectedValue),
                                 cudaMemcpyHostToDevice))) {
        memReady = ExpectSuccess("cudaIpcGetMemHandle",
                                 cudaIpcGetMemHandle(&payload.memHandle, deviceValue));
    }

    if (ExpectSuccess("ipc event create",
                      cudaEventCreateWithFlags(&event, cudaEventDisableTiming | cudaEventInterprocess))) {
        eventReady = ExpectSuccess("cudaIpcGetEventHandle",
                                   cudaIpcGetEventHandle(&payload.eventHandle, event));
        ExpectSuccess("ipc event record", cudaEventRecord(event, nullptr));
        ExpectSuccess("ipc event sync before child", cudaEventSynchronize(event));
    }

    if (memReady && eventReady) {
        FILE *fp = std::fopen(payloadPath, "wb");
        if (fp != nullptr) {
            bool wrote = std::fwrite(&payload, sizeof(payload), 1, fp) == 1;
            std::fclose(fp);
            if (wrote) {
                PrintResult("write IPC payload", "PASS");
                ++g_checks;

                pid_t pid = fork();
                if (pid == 0) {
                    execl(selfPath, selfPath, "--ipc-child", payloadPath, static_cast<char *>(nullptr));
                    _exit(127);
                } else if (pid > 0) {
                    int status = 0;
                    if (waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                        PrintResult("cudaIpcOpen/Close child", "PASS", "exit=0");
                        ++g_checks;
                    } else {
                        PrintResult("cudaIpcOpen/Close child", "FAIL");
                        ++g_checks;
                        ++g_failures;
                    }
                } else {
                    PrintResult("fork IPC child", "FAIL");
                    ++g_checks;
                    ++g_failures;
                }
            } else {
                PrintResult("write IPC payload", "FAIL");
                ++g_checks;
                ++g_failures;
            }
        } else {
            PrintResult("open IPC payload", "FAIL");
            ++g_checks;
            ++g_failures;
        }
        std::remove(payloadPath);
    } else {
        Skip("cudaIpcOpen/Close child", "IPC handle export did not succeed");
    }

    if (event != nullptr) {
        ExpectSuccess("ipc event destroy", cudaEventDestroy(event));
    }
    if (deviceValue != nullptr) {
        ExpectSuccess("ipc cudaFree", cudaFree(deviceValue));
    }
}

void CheckIncrementalContext(int device)
{
    CUcontext context = nullptr;
    if (ExpectCuSuccess("cuCtxGetCurrent", cuCtxGetCurrent(&context))) {
        ExpectCuSuccess("cuCtxSetCurrent", cuCtxSetCurrent(context));
    }
    unsigned int primaryFlags = 0;
    int active = 0;
    ExpectCuSuccess("cuDevicePrimaryCtxGetState",
                    cuDevicePrimaryCtxGetState(device, &primaryFlags, &active));
}

void CheckDriverContextApis(int device)
{
    unsigned int flags = ~0U;
    int active = -1;
    if (ExpectCuSuccess("cuDevicePrimaryCtxGetState context",
                        cuDevicePrimaryCtxGetState(device, &flags, &active))) {
        ++g_checks;
        if (active == 0 || active == 1) {
            PrintResult("primary context active is boolean", "PASS");
        } else {
            ++g_failures;
            PrintResult("primary context active is boolean", "FAIL");
        }
    }

    CUcontext original = nullptr;
    ExpectCuSuccess("cuCtxGetCurrent before create", cuCtxGetCurrent(&original));

    CUcontext created = nullptr;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    if (!ExpectCuSuccess("cuCtxCreate regular context",
                         cuCtxCreate(&created, nullptr, 0, device))) {
        return;
    }
#else
    Skip("cuCtxCreate regular context", "four-argument API requires CUDA 13.0+ headers");
    return;
#endif
    ++g_checks;
    if (created != nullptr) {
        PrintResult("cuCtxCreate non-null handle", "PASS");
    } else {
        ++g_failures;
        PrintResult("cuCtxCreate non-null handle", "FAIL");
        return;
    }

    CUcontext current = nullptr;
    if (ExpectCuSuccess("cuCtxGetCurrent after create", cuCtxGetCurrent(&current))) {
        ++g_checks;
        if (current == created) {
            PrintResult("current equals created context", "PASS");
        } else {
            ++g_failures;
            PrintResult("current equals created context", "FAIL");
        }
    }

    CUdevice currentDevice = -1;
    if (ExpectCuSuccess("cuCtxGetDevice", cuCtxGetDevice(&currentDevice))) {
        ExpectIntValue("context device matches requested device", currentDevice, device);
    }

    int least = 9999;
    int greatest = 9999;
    if (ExpectCuSuccess("cuCtxGetStreamPriorityRange both",
                        cuCtxGetStreamPriorityRange(&least, &greatest))) {
        ++g_checks;
        if (greatest <= least) {
            PrintResult("stream priority range ordered", "PASS");
        } else {
            ++g_failures;
            PrintResult("stream priority range ordered", "FAIL");
        }
        int onlyGreatest = 9999;
        if (ExpectCuSuccess("cuCtxGetStreamPriorityRange null least",
                            cuCtxGetStreamPriorityRange(nullptr, &onlyGreatest))) {
            ExpectIntValue("single priority output matches", onlyGreatest, greatest);
        }
    }

    ExpectCuSuccess("cuCtxSetCurrent original", cuCtxSetCurrent(original));
    ExpectCuSuccess("cuCtxSetCurrent created", cuCtxSetCurrent(created));
    current = nullptr;
    if (ExpectCuSuccess("cuCtxGetCurrent after set", cuCtxGetCurrent(&current))) {
        ++g_checks;
        if (current == created) {
            PrintResult("restored context matches created", "PASS");
        } else {
            ++g_failures;
            PrintResult("restored context matches created", "FAIL");
        }
    }
    ExpectCuSuccess("cuCtxDestroy created", cuCtxDestroy(created));
}

void CheckIncrementalDriverWrites(size_t bytes)
{
    void *devicePtr = nullptr;
    cudaStream_t memsetStream = nullptr;
    if (ExpectSuccess("incremental cudaMalloc", cudaMalloc(&devicePtr, bytes)) &&
        ExpectSuccess("incremental memset stream", cudaStreamCreate(&memsetStream))) {
        ExpectCuSuccess("cuMemsetD32Async",
                        cuMemsetD32Async((CUdeviceptr)devicePtr, 0x01020304U,
                                         bytes / sizeof(uint32_t), (CUstream)memsetStream));
        ExpectCuSuccess("cuStreamWriteValue32",
                        cuStreamWriteValue32((CUstream)memsetStream, (CUdeviceptr)devicePtr, 7, 0));
        ExpectSuccess("sync incremental memset stream", cudaStreamSynchronize(memsetStream));
        ExpectSuccess("destroy incremental memset stream", cudaStreamDestroy(memsetStream));
        ExpectSuccess("free incremental devicePtr", cudaFree(devicePtr));
    }
}

void CheckIncrementalModuleApis()
{
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    static const char kPtx[] =
        ".version 7.0\n"
        ".target sm_70\n"
        ".address_size 64\n"
        ".visible .entry apiRuntimeCoveragePtxKernel() {\n"
        "  ret;\n"
        "}\n";

    if (ExpectCuSuccess("cuModuleLoadData", cuModuleLoadData(&module, kPtx))) {
        ExpectCuSuccess("cuModuleGetFunction",
                        cuModuleGetFunction(&function, module, "apiRuntimeCoveragePtxKernel"));
        ExpectCuSuccess("cuModuleUnload data", cuModuleUnload(module));
        module = nullptr;
    }

    const char *ptxPath = "/tmp/apiRuntimeCoverage_module.ptx";
    if (FILE *fp = std::fopen(ptxPath, "wb")) {
        std::fwrite(kPtx, 1, std::strlen(kPtx), fp);
        std::fclose(fp);
        if (ExpectCuSuccess("cuModuleLoad", cuModuleLoad(&module, ptxPath))) {
            ExpectCuSuccess("cuModuleUnload file", cuModuleUnload(module));
        }
        std::remove(ptxPath);
    } else {
        ++g_checks;
        ++g_failures;
        PrintResult("write PTX module file", "FAIL");
    }
}

void CheckGraphMemoryAndProfilerApis(int device)
{
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11040
    const cudaGraphMemAttributeType attrs[] = {
        cudaGraphMemAttrUsedMemCurrent,
        cudaGraphMemAttrUsedMemHigh,
        cudaGraphMemAttrReservedMemCurrent,
        cudaGraphMemAttrReservedMemHigh,
    };
    const char *names[] = {
        "cudaDeviceGetGraphMemAttribute used-current",
        "cudaDeviceGetGraphMemAttribute used-high",
        "cudaDeviceGetGraphMemAttribute reserved-current",
        "cudaDeviceGetGraphMemAttribute reserved-high",
    };
    for (size_t i = 0; i < sizeof(attrs) / sizeof(attrs[0]); ++i) {
        unsigned long long value = 0;
        OptionalSuccess(names[i], cudaDeviceGetGraphMemAttribute(device, attrs[i], &value));
    }
#else
    (void)device;
    Skip("cudaDeviceGetGraphMemAttribute", "requires CUDA Runtime 11.4+");
#endif

    cudaError_t profilerStart = cudaProfilerStart();
    if (profilerStart == cudaErrorProfilerDisabled ||
        profilerStart == cudaErrorProfilerNotInitialized) {
        Skip("cudaProfilerStart/cudaProfilerStop", cudaGetErrorString(profilerStart));
        (void)cudaGetLastError();
        return;
    }
    if (ExpectSuccess("cudaProfilerStart", profilerStart)) {
        ExpectSuccess("profiler enclosed synchronize", cudaDeviceSynchronize());
        ExpectSuccess("cudaProfilerStop", cudaProfilerStop());
    }
}

void CheckLibraryApis()
{
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    const char *imagePath = std::getenv("CUDA_MAPPING_LIBRARY_IMAGE");
    if (imagePath == nullptr || imagePath[0] == '\0') {
        const char *reason = "set CUDA_MAPPING_LIBRARY_IMAGE to a CUDA fatbin/CANN binary";
        Skip("cudaLibraryLoadFromFile", reason);
        Skip("cudaLibraryLoadData", reason);
        Skip("cudaLibraryGetKernelCount", reason);
        Skip("cudaLibraryEnumerateKernels", reason);
        Skip("cudaLibraryGetKernel", reason);
        Skip("cudaLibraryGetGlobal", reason);
        Skip("cudaLibraryUnload", reason);
        return;
    }
    const char *kernelName = std::getenv("CUDA_MAPPING_LIBRARY_KERNEL");
    const char *globalName = std::getenv("CUDA_MAPPING_LIBRARY_GLOBAL");
    const char *imageFile = std::strrchr(imagePath, '/');
    imageFile = imageFile == nullptr ? imagePath : imageFile + 1;
    const bool knownFixture = std::strcmp(imageFile, "mapping_library.fatbin") == 0;
    if (knownFixture) {
        if (kernelName == nullptr || kernelName[0] == '\0') {
            kernelName = "libraryKernel";
        }
        if (globalName == nullptr || globalName[0] == '\0') {
            globalName = "gLibraryGlobal";
        }
    }

    cudaLibrary_t library = nullptr;
    if (!ExpectSuccess("cudaLibraryLoadFromFile",
                       cudaLibraryLoadFromFile(&library, imagePath, nullptr, nullptr, 0,
                                               nullptr, nullptr, 0))) {
        return;
    }

    unsigned int count = 0;
    if (ExpectSuccess("cudaLibraryGetKernelCount",
                      cudaLibraryGetKernelCount(&count, library))) {
        ++g_checks;
        if (count > 0) {
            PrintResult("cudaLibraryGetKernelCount positive", "PASS");
            std::vector<cudaKernel_t> kernels(count);
            ExpectSuccess("cudaLibraryEnumerateKernels",
                          cudaLibraryEnumerateKernels(kernels.data(), count, library));
        } else {
            ++g_failures;
            PrintResult("cudaLibraryGetKernelCount positive", "FAIL", "count is zero");
        }
    }

    if (kernelName != nullptr && kernelName[0] != '\0') {
        cudaKernel_t kernel = nullptr;
        ExpectSuccess("cudaLibraryGetKernel",
                      cudaLibraryGetKernel(&kernel, library, kernelName));
    } else {
        Skip("cudaLibraryGetKernel", "set CUDA_MAPPING_LIBRARY_KERNEL for this image");
    }

    if (globalName != nullptr && globalName[0] != '\0') {
        void *global = nullptr;
        size_t globalBytes = 0;
        if (ExpectSuccess("cudaLibraryGetGlobal",
                          cudaLibraryGetGlobal(&global, &globalBytes, library,
                                               globalName))) {
            ++g_checks;
            if (global != nullptr && globalBytes > 0 &&
                (!knownFixture || globalBytes == sizeof(int))) {
                PrintResult("cudaLibraryGetGlobal address/size", "PASS");
            } else {
                ++g_failures;
                PrintResult("cudaLibraryGetGlobal address/size", "FAIL");
            }
        }
    } else {
        Skip("cudaLibraryGetGlobal", "set CUDA_MAPPING_LIBRARY_GLOBAL for this image");
    }
    ExpectSuccess("cudaLibraryUnload file", cudaLibraryUnload(library));

    std::ifstream input(imagePath, std::ios::binary);
    std::vector<char> image((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
    if (image.empty()) {
        ++g_checks;
        ++g_failures;
        PrintResult("cudaLibraryLoadData input", "FAIL", "image is empty");
        return;
    }
    library = nullptr;
    if (ExpectSuccess("cudaLibraryLoadData",
                      cudaLibraryLoadData(&library, image.data(), nullptr, nullptr, 0,
                                          nullptr, nullptr, 0))) {
        ExpectSuccess("cudaLibraryUnload data", cudaLibraryUnload(library));
    }
#else
    Skip("CUDA Library APIs", "requires CUDA Runtime 12.0+");
#endif
}

cudaError_t BuildGraphUpdateInput(cudaGraph_t *graph, cudaGraphNode_t *node,
                                  int *deviceOutput, int value)
{
    cudaError_t error = cudaGraphCreate(graph, 0);
    if (error != cudaSuccess) {
        return error;
    }
    void *args[] = {&deviceOutput, &value};
    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void *>(ApiRuntimeCoverageKernelWithValue);
    params.gridDim = dim3(1, 1, 1);
    params.blockDim = dim3(1, 1, 1);
    params.kernelParams = args;
    cudaError_t addError = cudaGraphAddKernelNode(node, *graph, nullptr, 0, &params);
    if (addError != cudaSuccess) {
        ExpectSuccess("destroy failed graph update input", cudaGraphDestroy(*graph));
        *graph = nullptr;
        *node = nullptr;
    }
    return addError;
}

void CheckAdvancedGraphMutationApis()
{
    int *deviceOutput = nullptr;
    if (!ExpectSuccess("advanced graph output malloc",
                       cudaMalloc(reinterpret_cast<void **>(&deviceOutput), sizeof(int)))) {
        return;
    }

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    cudaGraphNode_t kernelNode = nullptr;
    cudaGraphNode_t recordNode = nullptr;
    cudaGraphNode_t waitNode = nullptr;
    cudaEvent_t eventA = nullptr;
    cudaEvent_t eventB = nullptr;
    cudaGraphNodeType nodeType = cudaGraphNodeTypeEmpty;
    int value = 5;
    void *args[] = {&deviceOutput, &value};
    cudaKernelNodeParams kernelParams{};
    kernelParams.func = reinterpret_cast<void *>(ApiRuntimeCoverageKernelWithValue);
    kernelParams.gridDim = dim3(1, 1, 1);
    kernelParams.blockDim = dim3(1, 1, 1);
    kernelParams.kernelParams = args;

    if (!ExpectSuccess("advanced cudaGraphCreate", cudaGraphCreate(&graph, 0)) ||
        !ExpectSuccess("advanced event A", cudaEventCreateWithFlags(&eventA, cudaEventDisableTiming)) ||
        !ExpectSuccess("advanced event B", cudaEventCreateWithFlags(&eventB, cudaEventDisableTiming)) ||
        !ExpectSuccess("advanced cudaGraphAddKernelNode",
                       cudaGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelParams)) ||
        !ExpectSuccess("advanced cudaGraphAddEventRecordNode",
                       cudaGraphAddEventRecordNode(&recordNode, graph, &kernelNode, 1, eventA)) ||
        !ExpectSuccess("advanced cudaGraphAddEventWaitNode",
                       cudaGraphAddEventWaitNode(&waitNode, graph, &recordNode, 1, eventA)) ||
        !ExpectSuccess("advanced cudaGraphInstantiate",
                       cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0))) {
        goto cleanup;
    }

    if (!ExpectSuccess("cudaGraphExecEventRecordNodeSetEvent",
                       cudaGraphExecEventRecordNodeSetEvent(graphExec, recordNode, eventB)) ||
        !ExpectSuccess("cudaGraphExecEventWaitNodeSetEvent",
                       cudaGraphExecEventWaitNodeSetEvent(graphExec, waitNode, eventB))) {
        goto cleanup;
    }
    value = 7;
    ExpectSuccess("cudaGraphExecKernelNodeSetParams",
                  cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &kernelParams));

    if (ExpectSuccess("cudaGraphNodeGetType kernel",
                      cudaGraphNodeGetType(kernelNode, &nodeType))) {
        ++g_checks;
        if (nodeType == cudaGraphNodeTypeKernel) {
            PrintResult("cudaGraphNodeGetType kernel value", "PASS");
        } else {
            ++g_failures;
            PrintResult("cudaGraphNodeGetType kernel value", "FAIL");
        }
    }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13010
    {
        unsigned int graphId = 0;
        ExpectSuccess("cudaGraphExecGetId", cudaGraphExecGetId(graphExec, &graphId));
    }
#else
    Skip("cudaGraphExecGetId", "requires CUDA Runtime 13.1+");
#endif

    ExpectSuccess("advanced output clear", cudaMemset(deviceOutput, 0, sizeof(int)));
    ExpectSuccess("advanced cudaGraphLaunch", cudaGraphLaunch(graphExec, nullptr));
    ExpectSuccess("advanced graph synchronize", cudaDeviceSynchronize());
    {
        int observed = -1;
        if (ExpectSuccess("advanced graph output copy",
                          cudaMemcpy(&observed, deviceOutput, sizeof(observed),
                                     cudaMemcpyDeviceToHost))) {
            ExpectIntValue("advanced graph result", observed, 7);
        }
    }

    ExpectSuccess("cudaGraphNodeSetEnabled disable",
                  cudaGraphNodeSetEnabled(graphExec, kernelNode, 0));
    ExpectSuccess("advanced disabled output clear", cudaMemset(deviceOutput, 0, sizeof(int)));
    ExpectSuccess("advanced disabled launch", cudaGraphLaunch(graphExec, nullptr));
    ExpectSuccess("advanced disabled synchronize", cudaDeviceSynchronize());
    {
        int observed = -1;
        if (ExpectSuccess("advanced disabled output copy",
                          cudaMemcpy(&observed, deviceOutput, sizeof(observed),
                                     cudaMemcpyDeviceToHost))) {
            ExpectIntValue("cudaGraphNodeSetEnabled disabled result", observed, 0);
        }
    }
    ExpectSuccess("cudaGraphNodeSetEnabled enable",
                  cudaGraphNodeSetEnabled(graphExec, kernelNode, 1));

cleanup:
    if (graphExec != nullptr) {
        ExpectSuccess("advanced cudaGraphExecDestroy", cudaGraphExecDestroy(graphExec));
    }
    if (graph != nullptr) {
        ExpectSuccess("advanced cudaGraphDestroy", cudaGraphDestroy(graph));
    }
    if (eventA != nullptr) {
        ExpectSuccess("advanced event A destroy", cudaEventDestroy(eventA));
    }
    if (eventB != nullptr) {
        ExpectSuccess("advanced event B destroy", cudaEventDestroy(eventB));
    }

    cudaGraph_t graphA = nullptr;
    cudaGraph_t graphB = nullptr;
    cudaGraphNode_t nodeA = nullptr;
    cudaGraphNode_t nodeB = nullptr;
    graphExec = nullptr;
    if (ExpectSuccess("build cudaGraphExecUpdate graph A",
                      BuildGraphUpdateInput(&graphA, &nodeA, deviceOutput, 21)) &&
        ExpectSuccess("build cudaGraphExecUpdate graph B",
                      BuildGraphUpdateInput(&graphB, &nodeB, deviceOutput, 22)) &&
        ExpectSuccess("instantiate cudaGraphExecUpdate graph",
                      cudaGraphInstantiate(&graphExec, graphA, nullptr, nullptr, 0))) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
        cudaGraphExecUpdateResultInfo updateInfo{};
        if (ExpectSuccess("cudaGraphExecUpdate",
                          cudaGraphExecUpdate(graphExec, graphB, &updateInfo))) {
            ++g_checks;
            if (updateInfo.result == cudaGraphExecUpdateSuccess) {
                PrintResult("cudaGraphExecUpdate result", "PASS");
            } else {
                ++g_failures;
                PrintResult("cudaGraphExecUpdate result", "FAIL");
            }
        }
#else
        cudaGraphNode_t errorNode = nullptr;
        cudaGraphExecUpdateResult updateResult = cudaGraphExecUpdateError;
        if (ExpectSuccess("cudaGraphExecUpdate",
                          cudaGraphExecUpdate(graphExec, graphB, &errorNode,
                                              &updateResult))) {
            ++g_checks;
            if (updateResult == cudaGraphExecUpdateSuccess) {
                PrintResult("cudaGraphExecUpdate result", "PASS");
            } else {
                ++g_failures;
                PrintResult("cudaGraphExecUpdate result", "FAIL");
            }
        }
#endif
    }
    if (graphExec != nullptr) {
        ExpectSuccess("destroy cudaGraphExecUpdate exec", cudaGraphExecDestroy(graphExec));
    }
    if (graphA != nullptr) {
        ExpectSuccess("destroy cudaGraphExecUpdate graph A", cudaGraphDestroy(graphA));
    }
    if (graphB != nullptr) {
        ExpectSuccess("destroy cudaGraphExecUpdate graph B", cudaGraphDestroy(graphB));
    }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13010
    cudaGraph_t genericGraph = nullptr;
    cudaGraphExec_t genericExec = nullptr;
    cudaGraphNode_t genericNode = nullptr;
    value = 31;
    cudaGraphNodeParams genericParams{};
    genericParams.type = cudaGraphNodeTypeKernel;
    genericParams.kernel.functionType = cudaKernelFunctionTypeDeviceEntry;
    genericParams.kernel.func = reinterpret_cast<void *>(ApiRuntimeCoverageKernelWithValue);
    genericParams.kernel.gridDim = dim3(1, 1, 1);
    genericParams.kernel.blockDim = dim3(1, 1, 1);
    genericParams.kernel.kernelParams = args;
    if (ExpectSuccess("generic cudaGraphCreate", cudaGraphCreate(&genericGraph, 0)) &&
        ExpectSuccess("generic cudaGraphAddNode",
                      cudaGraphAddNode(&genericNode, genericGraph, nullptr, nullptr, 0,
                                       &genericParams)) &&
        ExpectSuccess("cudaGraphNodeSetParams",
                      cudaGraphNodeSetParams(genericNode, &genericParams)) &&
        ExpectSuccess("generic cudaGraphInstantiate",
                      cudaGraphInstantiate(&genericExec, genericGraph, nullptr, nullptr, 0))) {
        value = 32;
        ExpectSuccess("cudaGraphExecNodeSetParams",
                      cudaGraphExecNodeSetParams(genericExec, genericNode, &genericParams));
    }
    if (genericExec != nullptr) {
        ExpectSuccess("generic cudaGraphExecDestroy", cudaGraphExecDestroy(genericExec));
    }
    if (genericGraph != nullptr) {
        ExpectSuccess("generic cudaGraphDestroy", cudaGraphDestroy(genericGraph));
    }
#else
    Skip("cudaGraphNodeSetParams/cudaGraphExecNodeSetParams",
         "requires CUDA Runtime 13.1+");
#endif

    ExpectSuccess("free advanced graph output", cudaFree(deviceOutput));
}

void CheckCapturedGenericGraphSetters()
{
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13010
    cudaStream_t stream = nullptr;
    if (!ExpectSuccess("captured setter stream", cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))) {
        return;
    }
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    if (!ExpectSuccess("captured setter begin", cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal))) {
        cudaStreamDestroy(stream);
        return;
    }
    ApiRuntimeCoverageCapturedSetterKernel<<<1, 1, 0, stream>>>();
    if (!ExpectSuccess("captured setter end", cudaStreamEndCapture(stream, &graph))) {
        cudaStreamDestroy(stream);
        return;
    }

    size_t nodeCount = 0;
    if (!ExpectSuccess("captured setter node count", cudaGraphGetNodes(graph, nullptr, &nodeCount)) ||
        nodeCount == 0) {
        ++g_checks;
        ++g_failures;
        PrintResult("captured setter kernel node exists", "FAIL");
    } else {
        std::vector<cudaGraphNode_t> nodes(nodeCount);
        if (ExpectSuccess("captured setter nodes", cudaGraphGetNodes(graph, nodes.data(), &nodeCount))) {
            cudaGraphNode_t kernelNode = nullptr;
            for (cudaGraphNode_t node : nodes) {
                cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
                if (!ExpectSuccess("captured setter node type", cudaGraphNodeGetType(node, &type))) break;
                if (type == cudaGraphNodeTypeKernel) {
                    kernelNode = node;
                    break;
                }
            }
            ++g_checks;
            if (kernelNode == nullptr) {
                ++g_failures;
                PrintResult("captured setter kernel node exists", "FAIL");
            } else {
                PrintResult("captured setter kernel node exists", "PASS");
                cudaGraphNodeParams params{};
                params.type = cudaGraphNodeTypeKernel;
                params.kernel.functionType = cudaKernelFunctionTypeDeviceEntry;
                params.kernel.func = reinterpret_cast<void *>(ApiRuntimeCoverageCapturedSetterKernel);
                params.kernel.gridDim = dim3(1, 1, 1);
                params.kernel.blockDim = dim3(1, 1, 1);
                if (ExpectSuccess("cudaGraphNodeSetParams captured kernel",
                                  cudaGraphNodeSetParams(kernelNode, &params)) &&
                    ExpectSuccess("captured setter instantiate",
                                  cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0)) &&
                    ExpectSuccess("cudaGraphExecNodeSetParams captured kernel",
                                  cudaGraphExecNodeSetParams(graphExec, kernelNode, &params)) &&
                    ExpectSuccess("captured setter launch", cudaGraphLaunch(graphExec, stream)) &&
                    ExpectSuccess("captured setter synchronize", cudaStreamSynchronize(stream))) {
                    int output = 0;
                    if (ExpectSuccess("captured setter symbol read",
                                      cudaMemcpyFromSymbol(&output, g_capturedGraphSetterValue,
                                                           sizeof(output)))) {
                        ExpectIntValue("captured setter output", output, 42);
                    }
                }
            }
        }
    }
    if (graphExec != nullptr) ExpectSuccess("captured setter exec destroy", cudaGraphExecDestroy(graphExec));
    if (graph != nullptr) ExpectSuccess("captured setter graph destroy", cudaGraphDestroy(graph));
    ExpectSuccess("captured setter stream destroy", cudaStreamDestroy(stream));
#else
    Skip("captured generic Graph setters", "requires CUDA Runtime 13.1+");
#endif
}

void CheckIncrementalApiSurface(int device, const char *selfPath)
{
    constexpr size_t bytes = 4096;
    CheckIncrementalHostAlloc(bytes);
    CheckIncrementalEventRecord();
    CheckStreamAttributeApis();
    CheckFunctionQueryApis();
    CheckLaunchHostFuncV2Api();
    CheckLaunchKernelExApi();
    CheckIncrementalFunctionAndGraphApis();
    CheckIncrementalSymbolApis();
    CheckIncrementalCaptureInfo();
    CheckThreadExchangeStreamCaptureModeApi();
    CheckIpcApis(selfPath);
    CheckIncrementalContext(device);
    CheckIncrementalDriverWrites(bytes);
    CheckIncrementalModuleApis();
    CheckGraphMemoryAndProfilerApis(device);
    CheckLibraryApis();
    CheckAdvancedGraphMutationApis();
}

void CheckDriverVmmApis(int device, const char *selfPath)
{
    CUresult init = cuInit(0);
    if (!OptionalCuSuccess("cuInit", init)) {
        return;
    }

    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;

    size_t granularity = 0;
    if (!OptionalCuSuccess("cuMemGetAllocationGranularity",
                           cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM))) {
        return;
    }

    CUdeviceptr address = 0;
    if (!OptionalCuSuccess("cuMemAddressReserve", cuMemAddressReserve(&address, granularity, 0, 0, 0))) {
        return;
    }

    CUmemGenericAllocationHandle handle = 0;
    bool created = OptionalCuSuccess("cuMemCreate", cuMemCreate(&handle, granularity, &prop, 0));
    bool mapped = false;
    if (created) {
        mapped = OptionalCuSuccess("cuMemMap", cuMemMap(address, granularity, 0, handle, 0));
    }

    if (mapped) {
        CUmemAccessDesc accessDesc{};
        accessDesc.location = prop.location;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        OptionalCuSuccess("cuMemSetAccess", cuMemSetAccess(address, granularity, &accessDesc, 1));

        unsigned long long accessFlags = 0;
        OptionalCuSuccess("cuMemGetAccess", cuMemGetAccess(&accessFlags, &prop.location, address));

        OptionalCuSuccess("cuMemUnmap", cuMemUnmap(address, granularity));
    } else {
        Skip("cuMemSetAccess", "mapping did not succeed");
        Skip("cuMemGetAccess", "mapping did not succeed");
        Skip("cuMemUnmap", "mapping did not succeed");
    }

    if (created) {
        OptionalCuSuccess("cuMemRelease", cuMemRelease(handle));
    }
    OptionalCuSuccess("cuMemAddressFree", cuMemAddressFree(address, granularity));

    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    CUmemGenericAllocationHandle exportHandle = 0;
    if (OptionalCuSuccess("cuMemCreate export", cuMemCreate(&exportHandle, granularity, &prop, 0))) {
        int fd = -1;
        if (OptionalCuSuccess("cuMemExportToShareableHandle",
                              cuMemExportToShareableHandle(&fd, exportHandle,
                                                           CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0))) {
#if defined(__CUDACC__)
            CUmemGenericAllocationHandle importedHandle = 0;
            if (OptionalCuSuccess("cuMemImportFromShareableHandle",
                                  cuMemImportFromShareableHandle(&importedHandle,
                                                                 reinterpret_cast<void *>(static_cast<intptr_t>(fd)),
                                                                 CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR))) {
                OptionalCuSuccess("cuMemRelease imported", cuMemRelease(importedHandle));
            }
#else
            char fdText[32];
            std::snprintf(fdText, sizeof(fdText), "%d", fd);
            pid_t pid = fork();
            if (pid == 0) {
                execl(selfPath, selfPath, "--vmm-import-child", fdText, static_cast<char *>(nullptr));
                _exit(127);
            } else if (pid > 0) {
                int status = 0;
                if (waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                    ++g_checks;
                    PrintResult("cuMemImportFromShareableHandle child", "PASS", "exit=0");
                } else {
                    ++g_checks;
                    ++g_failures;
                    PrintResult("cuMemImportFromShareableHandle child", "FAIL");
                }
            } else {
                ++g_checks;
                ++g_failures;
                PrintResult("fork VMM import child", "FAIL");
            }
#endif
            if (fd >= 0) {
                close(fd);
            }
        }
        OptionalCuSuccess("cuMemRelease export", cuMemRelease(exportHandle));
    }
}


void CheckMemoryCopyAndSymbolBranches()
{
    size_t symbolSize = 0;
    if (ExpectSuccess("cudaGetSymbolSize before partial copy",
                      cudaGetSymbolSize(&symbolSize, g_symbolSizeProbe))) {
        ExpectIntValue("cudaGetSymbolSize declared bytes",
                       static_cast<int>(symbolSize), static_cast<int>(sizeof(g_symbolSizeProbe)));
    }
    int value = 123;
    ExpectSuccess("cudaMemcpyToSymbol partial",
                  cudaMemcpyToSymbol(g_symbolSizeProbe, &value, sizeof(value)));
    symbolSize = 0;
    if (ExpectSuccess("cudaGetSymbolSize after partial copy",
                      cudaGetSymbolSize(&symbolSize, g_symbolSizeProbe))) {
        ExpectIntValue("cudaGetSymbolSize after partial bytes",
                       static_cast<int>(symbolSize), static_cast<int>(sizeof(g_symbolSizeProbe)));
    }

    unsigned char hostIn[64] = {};
    for (size_t i = 0; i < sizeof(hostIn); ++i) {
        hostIn[i] = static_cast<unsigned char>(i + 1);
    }
    void *src = nullptr;
    void *dst = nullptr;
    void *hostOut = nullptr;
    cudaStream_t stream = nullptr;
    bool ready = ExpectSuccess("memory copy source alloc", cudaMalloc(&src, sizeof(hostIn))) &&
                 ExpectSuccess("memory copy destination alloc", cudaMalloc(&dst, sizeof(hostIn))) &&
                 ExpectSuccess("memory copy host alloc", cudaMallocHost(&hostOut, sizeof(hostIn))) &&
                 ExpectSuccess("memory copy stream create", cudaStreamCreate(&stream));
    if (ready) {
        ExpectSuccess("memory copy source seed",
                      cudaMemcpy(src, hostIn, sizeof(hostIn), cudaMemcpyHostToDevice));
        ExpectError("cudaMemcpyPeer invalid device IDs",
                    cudaMemcpyPeer(dst, 999, src, 888, sizeof(hostIn)), cudaErrorInvalidDevice);
        if (ExpectSuccess("cudaMemcpyPeer valid IDs",
                          cudaMemcpyPeer(dst, 0, src, 0, sizeof(hostIn)))) {
            ExpectSuccess("cudaMemcpyPeer data read",
                          cudaMemcpy(hostOut, dst, sizeof(hostIn), cudaMemcpyDeviceToHost));
            ExpectIntValue("cudaMemcpyPeer data",
                           std::memcmp(hostOut, hostIn, sizeof(hostIn)), 0);
        }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13020
        cudaMemcpyAttributes attrs{};
        attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        attrs.srcLocHint.type = cudaMemLocationTypeDevice;
        attrs.dstLocHint.type = cudaMemLocationTypeDevice;
        void *dsts[1] = {dst};
        const void *srcs[1] = {src};
        size_t sizes[1] = {sizeof(hostIn)};
        size_t attrsIdxs[1] = {0};
        if (ExpectSuccess("cudaMemcpyBatchAsync D2D",
                          cudaMemcpyBatchAsync(dsts, srcs, sizes, 1,
                                               &attrs, attrsIdxs, 1, stream))) {
            ExpectSuccess("cudaMemcpyBatchAsync D2D sync", cudaStreamSynchronize(stream));
            ExpectSuccess("cudaMemcpyBatchAsync D2D read",
                          cudaMemcpy(hostOut, dst, sizeof(hostIn), cudaMemcpyDeviceToHost));
            ExpectIntValue("cudaMemcpyBatchAsync D2D data",
                           std::memcmp(hostOut, hostIn, sizeof(hostIn)), 0);
        }

        cudaMemcpy3DBatchOp op{};
        op.src.type = cudaMemcpyOperandTypePointer;
        op.src.op.ptr.ptr = src;
        op.src.op.ptr.rowLength = sizeof(hostIn);
        op.src.op.ptr.layerHeight = 1;
        op.dst.type = cudaMemcpyOperandTypePointer;
        op.dst.op.ptr.ptr = dst;
        op.dst.op.ptr.rowLength = sizeof(hostIn);
        op.dst.op.ptr.layerHeight = 1;
        op.extent = make_cudaExtent(sizeof(hostIn), 1, 1);
        op.srcAccessOrder = cudaMemcpySrcAccessOrderInvalid;
        ExpectError("cudaMemcpy3DBatchAsync invalid order",
                    cudaMemcpy3DBatchAsync(1, &op, 0, stream), cudaErrorInvalidValue);
        op.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        op.flags = 0x80000000U;
        ExpectError("cudaMemcpy3DBatchAsync invalid flags",
                    cudaMemcpy3DBatchAsync(1, &op, 0, stream), cudaErrorInvalidValue);
        op.flags = 0;
        ExpectError("cudaMemcpy3DBatchAsync legacy NULL stream",
                    cudaMemcpy3DBatchAsync(1, &op, 0, nullptr), cudaErrorInvalidValue);
        ExpectError("cudaMemcpy3DWithAttributesAsync legacy NULL stream",
                    cudaMemcpy3DWithAttributesAsync(&op, 0, nullptr), cudaErrorInvalidValue);
        if (ExpectSuccess("cudaMemcpy3DBatchAsync D2D branch",
                          cudaMemcpy3DBatchAsync(1, &op, 0, stream))) {
            ExpectSuccess("cudaMemcpy3DBatchAsync branch sync", cudaStreamSynchronize(stream));
            ExpectSuccess("cudaMemcpy3DBatchAsync branch read",
                          cudaMemcpy(hostOut, dst, sizeof(hostIn), cudaMemcpyDeviceToHost));
            ExpectIntValue("cudaMemcpy3DBatchAsync branch data",
                           std::memcmp(hostOut, hostIn, sizeof(hostIn)), 0);
        }

        attrs.dstLocHint.type = cudaMemLocationTypeHost;
        std::memset(hostOut, 0, sizeof(hostIn));
        if (ExpectSuccess("cudaMemcpyWithAttributesAsync D2H",
                          cudaMemcpyWithAttributesAsync(hostOut, src, sizeof(hostIn),
                                                        &attrs, stream))) {
            ExpectSuccess("cudaMemcpyWithAttributesAsync D2H sync",
                          cudaStreamSynchronize(stream));
            ExpectIntValue("cudaMemcpyWithAttributesAsync D2H data",
                           std::memcmp(hostOut, hostIn, sizeof(hostIn)), 0);
        }
        ExpectError("cudaMemcpyWithAttributesAsync legacy NULL stream",
                    cudaMemcpyWithAttributesAsync(hostOut, src, sizeof(hostIn),
                                                  &attrs, nullptr), cudaErrorInvalidValue);
        attrs.dstLocHint.type = cudaMemLocationTypeDevice;
        ExpectError("cudaMemcpyBatchAsync legacy NULL stream",
                    cudaMemcpyBatchAsync(dsts, srcs, sizes, 1,
                                         &attrs, attrsIdxs, 1, nullptr), cudaErrorInvalidValue);

        unsigned char expected[sizeof(hostIn)] = {};
        std::memcpy(expected, hostIn, sizeof(hostIn));
        attrs.srcAccessOrder = cudaMemcpySrcAccessOrderDuringApiCall;
        attrs.srcLocHint.type = cudaMemLocationTypeHost;
        attrs.dstLocHint.type = cudaMemLocationTypeDevice;
        if (ExpectSuccess("cudaMemcpyWithAttributesAsync DuringApiCall",
                          cudaMemcpyWithAttributesAsync(dst, hostIn, sizeof(hostIn),
                                                        &attrs, stream))) {
            std::memset(hostIn, 0, sizeof(hostIn));
            ExpectSuccess("cudaMemcpyWithAttributesAsync DuringApiCall sync",
                          cudaStreamSynchronize(stream));
            ExpectSuccess("cudaMemcpyWithAttributesAsync DuringApiCall read",
                          cudaMemcpy(hostOut, dst, sizeof(hostIn), cudaMemcpyDeviceToHost));
            ExpectIntValue("cudaMemcpyWithAttributesAsync source lifetime",
                           std::memcmp(hostOut, expected, sizeof(hostIn)), 0);
        }
        std::memcpy(hostIn, expected, sizeof(hostIn));
        srcs[0] = hostIn;
        if (ExpectSuccess("cudaMemcpyBatchAsync DuringApiCall",
                          cudaMemcpyBatchAsync(dsts, srcs, sizes, 1,
                                               &attrs, attrsIdxs, 1, stream))) {
            std::memset(hostIn, 0, sizeof(hostIn));
            ExpectSuccess("cudaMemcpyBatchAsync DuringApiCall sync",
                          cudaStreamSynchronize(stream));
            ExpectSuccess("cudaMemcpyBatchAsync DuringApiCall read",
                          cudaMemcpy(hostOut, dst, sizeof(hostIn), cudaMemcpyDeviceToHost));
            ExpectIntValue("cudaMemcpyBatchAsync source lifetime",
                           std::memcmp(hostOut, expected, sizeof(hostIn)), 0);
        }
#endif
    }
    if (stream) ExpectSuccess("memory copy stream destroy", cudaStreamDestroy(stream));
    if (hostOut) ExpectSuccess("memory copy host free", cudaFreeHost(hostOut));
    if (dst) ExpectSuccess("memory copy destination free", cudaFree(dst));
    if (src) ExpectSuccess("memory copy source free", cudaFree(src));
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--ipc-child") == 0) {
        return RunIpcChild(argv[2]);
    }
    if (argc == 3 && std::strcmp(argv[1], "--vmm-import-child") == 0) {
        return RunVmmImportChild(argv[2]);
    }

    std::printf("%s Starting...\n\n", argv[0]);
    std::printf(" Native CUDA Runtime API Coverage\n\n");

    int deviceCount = 0;
    if (!ExpectSuccess("cudaGetDeviceCount", cudaGetDeviceCount(&deviceCount)) || deviceCount == 0) {
        std::printf("\nResult = FAIL\n");
        return EXIT_FAILURE;
    }

    int device = 0;
    ExpectSuccess("cudaSetDevice", cudaSetDevice(device));
    ExpectSuccess("cudaGetDevice", cudaGetDevice(&device));

    CheckDeviceApis(device);
    CheckErrorApis();
    CheckDeviceStateApis();
    CheckAtomicCapabilityApis(deviceCount);
    CheckPeerAccessCapability(deviceCount);
    CheckPeerEnableAndMemcpyAsyncApis(deviceCount);
    CheckMemoryApis();
    Check3DMemoryApis();
    CheckManagedMemoryApis(device);
    CheckStreamApis();
    CheckEventApis();
    CheckHostRegistrationApis();
    CheckLaunchHostFuncApi();
    CheckLaunchKernelApi();
    CheckBasicGraphCaptureApis();
    CheckIncrementalApiSurface(device, argv[0]);
    CheckCapturedGenericGraphSetters();
    CheckDriverVmmApis(device, argv[0]);
    CheckDriverContextApis(device);
    CheckMemoryCopyAndSymbolBranches();

    ExpectSuccess("cudaDeviceSynchronize", cudaDeviceSynchronize());
    ExpectSuccess("cudaDeviceReset", cudaDeviceReset());

    std::printf("\napiRuntimeCoverage_cuda, Runtime API checks = %d, Failures = %d\n",
                g_checks, g_failures);
    std::printf("Result = %s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
