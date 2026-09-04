/*
 * Native CUDA version of apiRuntimeCoverage.
 *
 * Build on a CUDA machine, for example:
 *   nvcc -std=c++17 apiRuntimeCoverage_cuda.cu -lcuda -o apiRuntimeCoverage_cuda
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

    cudaFuncCache cacheConfig = cudaFuncCachePreferNone;
    OptionalSuccess("cudaDeviceSetCacheConfig", cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    OptionalSuccess("cudaDeviceGetCacheConfig", cudaDeviceGetCacheConfig(&cacheConfig));

    int leastPriority = 0;
    int greatestPriority = 0;
    OptionalSuccess("cudaDeviceGetStreamPriorityRange",
                    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
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

void CheckLaunchKernelApi()
{
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

void CheckIncrementalApiSurface(int device)
{
    constexpr size_t bytes = 4096;
    CheckIncrementalHostAlloc(bytes);
    CheckIncrementalEventRecord();
    CheckIncrementalFunctionAndGraphApis();
    CheckIncrementalSymbolApis();
    CheckIncrementalCaptureInfo();
    CheckIncrementalContext(device);
    CheckIncrementalDriverWrites(bytes);
    CheckIncrementalModuleApis();
}

void CheckDriverVmmApis(int device)
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
}

}  // namespace

int main(int argc, char **argv)
{
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
    CheckMemoryApis();
    CheckStreamApis();
    CheckEventApis();
    CheckHostRegistrationApis();
    CheckLaunchKernelApi();
    CheckIncrementalApiSurface(device);
    CheckDriverVmmApis(device);

    ExpectSuccess("cudaDeviceSynchronize", cudaDeviceSynchronize());
    ExpectSuccess("cudaDeviceReset", cudaDeviceReset());

    std::printf("\napiRuntimeCoverage_cuda, Runtime API checks = %d, Failures = %d\n",
                g_checks, g_failures);
    std::printf("Result = %s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
