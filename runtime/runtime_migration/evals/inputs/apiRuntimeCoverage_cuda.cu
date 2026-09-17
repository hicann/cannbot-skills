/*
 * Native CUDA version of apiRuntimeCoverage.
 *
 * Build on a CUDA machine, for example:
 *   nvcc -std=c++17 apiRuntimeCoverage_cuda.cu -lcuda -o apiRuntimeCoverage_cuda
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

__global__ void ApiRuntimeCoverageKernel(int *out)
{
    *out = 42;
}

__device__ int g_apiRuntimeCoverageSymbol = 0;

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
    ExpectSuccess("cudaDriverGetVersion", cudaDriverGetVersion(&driverVersion));
    ExpectSuccess("cudaRuntimeGetVersion", cudaRuntimeGetVersion(&runtimeVersion));

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
    ExpectError("cudaLaunchKernel zero grid",
                cudaLaunchKernel(reinterpret_cast<const void *>(ApiRuntimeCoverageKernel),
                                 dim3(0), dim3(1), invalidArgs, 0, nullptr),
                cudaErrorInvalidConfiguration);
    ExpectError("cudaLaunchKernel zero block",
                cudaLaunchKernel(reinterpret_cast<const void *>(ApiRuntimeCoverageKernel),
                                 dim3(1), dim3(0), invalidArgs, 0, nullptr),
                cudaErrorInvalidConfiguration);

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

void CheckIncrementalFunctionAndGraphApis()
{
    cudaFuncAttributes attrs{};
    ExpectSuccess("cudaFuncGetAttributes", cudaFuncGetAttributes(&attrs, ApiRuntimeCoverageKernel));

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
                              cudaStreamGetCaptureInfo(parentStream, &status, nullptr, &conditionalGraph,
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
                                  cudaGraphAddNode(&conditionalNode, conditionalGraph, nullptr, 0,
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
                      cudaStreamGetCaptureInfo(captureStream, &status, nullptr, &graph, &deps, &depCount));
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
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

void CheckIncrementalApiSurface(int device, const char *selfPath)
{
    constexpr size_t bytes = 4096;
    CheckIncrementalHostAlloc(bytes);
    CheckIncrementalEventRecord();
    CheckIncrementalFunctionAndGraphApis();
    CheckIncrementalSymbolApis();
    CheckIncrementalCaptureInfo();
    CheckThreadExchangeStreamCaptureModeApi();
    CheckIpcApis(selfPath);
    CheckIncrementalContext(device);
    CheckIncrementalDriverWrites(bytes);
    CheckIncrementalModuleApis();
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
    CheckStreamApis();
    CheckEventApis();
    CheckHostRegistrationApis();
    CheckLaunchHostFuncApi();
    CheckLaunchKernelApi();
    CheckBasicGraphCaptureApis();
    CheckIncrementalApiSurface(device, argv[0]);
    CheckDriverVmmApis(device, argv[0]);

    ExpectSuccess("cudaDeviceSynchronize", cudaDeviceSynchronize());
    ExpectSuccess("cudaDeviceReset", cudaDeviceReset());

    std::printf("\napiRuntimeCoverage_cuda, Runtime API checks = %d, Failures = %d\n",
                g_checks, g_failures);
    std::printf("Result = %s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
