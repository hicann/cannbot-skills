/*
 * Additional CUDA Runtime API coverage checks for the CANN compatibility layer.
 */

#include "cann_runtime_compat.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void PrintResult(const char *name, const char *state, const char *detail = "")
{
    std::printf("  %-36s %s%s%s\n", name, state, detail[0] ? " - " : "", detail);
}

std::string CudaErrorDetail(cudaError_t expected, cudaError_t actual)
{
    return std::string("expected ") + cudaGetErrorName(expected) + ", got " + cudaGetErrorName(actual);
}

std::string CuResultDetail(CUresult result)
{
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

    std::string detail = CuResultDetail(result);
    Skip(name, detail.c_str());
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

void CheckErrorApis()
{
    ExpectSuccess("cudaPeekAtLastError", cudaPeekAtLastError());

    ++g_checks;
    const char *name = cudaGetErrorName(cudaErrorNotSupported);
    if (name != nullptr && std::strcmp(name, "cudaErrorNotSupported") == 0) {
        PrintResult("cudaGetErrorName", "PASS");
    } else {
        ++g_failures;
        PrintResult("cudaGetErrorName", "FAIL", name == nullptr ? "null" : name);
    }
}

void CheckDeviceStateApis()
{
    unsigned int flags = 0;
    ExpectSuccess("cudaSetDeviceFlags", cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync | cudaDeviceMapHost));
    if (ExpectSuccess("cudaGetDeviceFlags", cudaGetDeviceFlags(&flags)) &&
        flags != (cudaDeviceScheduleBlockingSync | cudaDeviceMapHost)) {
        ++g_failures;
        PrintResult("cudaGetDeviceFlags value", "FAIL", "stored flags mismatch");
    }

    ExpectSuccess("cudaDeviceSetLimit", cudaDeviceSetLimit(cudaLimitStackSize, 4096));
    size_t limitValue = 0;
    ExpectSuccess("cudaDeviceGetLimit", cudaDeviceGetLimit(&limitValue, cudaLimitStackSize));

    cudaFuncCache cacheConfig = cudaFuncCachePreferNone;
    ExpectSuccess("cudaDeviceSetCacheConfig", cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    ExpectSuccess("cudaDeviceGetCacheConfig", cudaDeviceGetCacheConfig(&cacheConfig));

    int leastPriority = 0;
    int greatestPriority = 0;
    ExpectSuccess("cudaDeviceGetStreamPriorityRange",
                  cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
}

void CheckStreamPriorityApis()
{
    int leastPriority = 0;
    int greatestPriority = 0;
    if (!ExpectSuccess("priority range for stream", cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority))) {
        return;
    }

    cudaStream_t stream = nullptr;
    int selectedPriority = greatestPriority;
    if (!ExpectSuccess("cudaStreamCreateWithPriority",
                       cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, selectedPriority))) {
        return;
    }

    unsigned long long streamId = 0;
    unsigned int flags = 0;
    int priority = 0;
    ExpectSuccess("cudaStreamGetId", cudaStreamGetId(stream, &streamId));
    ExpectSuccess("cudaStreamGetPriority", cudaStreamGetPriority(stream, &priority));
    ExpectSuccess("cudaStreamGetFlags", cudaStreamGetFlags(stream, &flags));
    ExpectSuccess("cudaStreamDestroy", cudaStreamDestroy(stream));
}

void CheckStreamCaptureApis()
{
    cudaStream_t stream = nullptr;
    if (!ExpectSuccess("stream for capture", cudaStreamCreate(&stream))) {
        return;
    }

    cudaStreamCaptureStatus status = cudaStreamCaptureStatusInvalidated;
    OptionalSuccess("cudaStreamIsCapturing", cudaStreamIsCapturing(stream, &status));

    cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal;
    OptionalSuccess("cudaThreadExchangeStreamCaptureMode", cudaThreadExchangeStreamCaptureMode(&mode));

    bool captureStarted = OptionalSuccess("cudaStreamBeginCapture", cudaStreamBeginCapture(stream));
    if (captureStarted) {
        OptionalSuccess("cudaStreamIsCapturing active", cudaStreamIsCapturing(stream, &status));
        cudaGraph_t graph = nullptr;
        OptionalSuccess("cudaStreamEndCapture", cudaStreamEndCapture(stream, &graph));
    } else {
        Skip("cudaStreamEndCapture", "begin capture did not succeed");
    }

    ExpectSuccess("destroy capture stream", cudaStreamDestroy(stream));
}

void CheckPitchAnd2DMemoryApis()
{
    constexpr size_t width = 64;
    constexpr size_t height = 8;
    void *devicePtr = nullptr;
    size_t pitch = 0;

    if (!ExpectSuccess("cudaMallocPitch", cudaMallocPitch(&devicePtr, &pitch, width, height))) {
        return;
    }

    std::vector<unsigned char> host(pitch * height, 0);
    ExpectSuccess("cudaMemset2D", cudaMemset2D(devicePtr, pitch, 0x5a, width, height));
    ExpectSuccess("cudaMemcpy2D after memset2D",
                  cudaMemcpy2D(host.data(), pitch, devicePtr, pitch, width, height, cudaMemcpyDeviceToHost));

    bool valid = true;
    for (size_t row = 0; row < height; ++row) {
        valid = valid && std::all_of(host.begin() + row * pitch,
                                     host.begin() + row * pitch + width,
                                     [](unsigned char v) { return v == 0x5a; });
    }
    ++g_checks;
    if (valid) {
        PrintResult("cudaMemset2D data", "PASS");
    } else {
        ++g_failures;
        PrintResult("cudaMemset2D data", "FAIL", "unexpected row contents");
    }

    cudaStream_t stream = nullptr;
    if (ExpectSuccess("stream for cudaMemset2DAsync", cudaStreamCreate(&stream))) {
        std::fill(host.begin(), host.end(), 0);
        ExpectSuccess("cudaMemset2DAsync", cudaMemset2DAsync(devicePtr, pitch, 0xa5, width, height, stream));
        ExpectSuccess("cudaStreamSynchronize", cudaStreamSynchronize(stream));
        ExpectSuccess("cudaMemcpy2D after memset2DAsync",
                      cudaMemcpy2D(host.data(), pitch, devicePtr, pitch, width, height, cudaMemcpyDeviceToHost));
        ExpectSuccess("destroy memset2DAsync stream", cudaStreamDestroy(stream));
    }

    ExpectSuccess("cudaFree pitched allocation", cudaFree(devicePtr));
}

void CheckManagedMemoryApis(int device)
{
    constexpr size_t count = 256;
    void *managedPtr = nullptr;
    ExpectError("cudaMallocManaged", cudaMallocManaged(&managedPtr, count, cudaMemAttachGlobal),
                cudaErrorNotSupported);
    ExpectError("cudaMemAdvise invalid", cudaMemAdvise(nullptr, count, cudaMemAdviseSetReadMostly, device),
                cudaErrorInvalidValue);
}

void CheckMemPrefetchMockApi(int device)
{
    cudaMemLocation location{};
    location.type = cudaMemLocationTypeDevice;
    location.id = device;
    ExpectError("cudaMemPrefetchAsync", cudaMemPrefetchAsync(nullptr, 0, location, 0, nullptr),
                cudaErrorNotSupported);
}

void CheckMemcpyBatchApi(int device)
{
    constexpr size_t bytes = 64;
    unsigned char srcA[bytes];
    unsigned char srcB[bytes];
    std::fill_n(srcA, sizeof(srcA), 0x11);
    std::fill_n(srcB, sizeof(srcB), 0x22);

    void *dstA = nullptr;
    void *dstB = nullptr;
    if (!ExpectSuccess("batch dst cudaMalloc A", cudaMalloc(&dstA, bytes))) {
        return;
    }
    if (!ExpectSuccess("batch dst cudaMalloc B", cudaMalloc(&dstB, bytes))) {
        cudaFree(dstA);
        return;
    }

    const void *dsts[] = {dstA, dstB};
    const void *srcs[] = {srcA, srcB};
    size_t sizes[] = {bytes, bytes};
    cudaMemcpyAttributes attrs[1]{};
    attrs[0].srcLocHint.type = cudaMemLocationTypeHost;
    attrs[0].srcLocHint.id = 0;
    attrs[0].dstLocHint.type = cudaMemLocationTypeDevice;
    attrs[0].dstLocHint.id = device;
    size_t attrIdxs[] = {0, 0};

    cudaStream_t stream = nullptr;
    if (ExpectSuccess("stream for cudaMemcpyBatchAsync", cudaStreamCreate(&stream))) {
        cudaError_t err = cudaMemcpyBatchAsync(dsts, srcs, sizes, 2, attrs, attrIdxs, 1, stream);
        if (err == cudaSuccess) {
            ++g_checks;
            PrintResult("cudaMemcpyBatchAsync", "PASS");
            ExpectSuccess("sync cudaMemcpyBatchAsync", cudaStreamSynchronize(stream));
        } else {
            Skip("cudaMemcpyBatchAsync", cudaGetErrorString(err));
        }
        ExpectSuccess("destroy batch stream", cudaStreamDestroy(stream));
    }

    ExpectSuccess("free batch dst A", cudaFree(dstA));
    ExpectSuccess("free batch dst B", cudaFree(dstB));
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
    std::fill_n(static_cast<unsigned char *>(hostPtr), size, 0x3c);

    cudaError_t regErr = cudaHostRegister(hostPtr, size, 0);
    if (regErr != cudaSuccess) {
        Skip("cudaHostRegister", cudaGetErrorString(regErr));
        std::free(hostPtr);
        return;
    }
    PrintResult("cudaHostRegister", "PASS");

    void *deviceAlias = nullptr;
    ExpectSuccess("cudaHostGetDevicePointer", cudaHostGetDevicePointer(&deviceAlias, hostPtr, 0));
    ExpectSuccess("cudaHostUnregister", cudaHostUnregister(hostPtr));
    std::free(hostPtr);
}

void CheckPeerApis(int deviceCount)
{
    constexpr size_t bytes = 64;
    unsigned char host[bytes];
    std::fill_n(host, sizeof(host), 0x7b);

    void *src = nullptr;
    void *dst = nullptr;
    if (ExpectSuccess("peer src cudaMalloc", cudaMalloc(&src, bytes)) &&
        ExpectSuccess("peer dst cudaMalloc", cudaMalloc(&dst, bytes))) {
        ExpectSuccess("seed peer src", cudaMemcpy(src, host, bytes, cudaMemcpyHostToDevice));
        ExpectSuccess("cudaMemcpyPeer same device", cudaMemcpyPeer(dst, 0, src, 0, bytes));

        cudaStream_t stream = nullptr;
        if (ExpectSuccess("stream for cudaMemcpyPeerAsync", cudaStreamCreate(&stream))) {
            ExpectSuccess("cudaMemcpyPeerAsync same device", cudaMemcpyPeerAsync(dst, 0, src, 0, bytes, stream));
            ExpectSuccess("sync cudaMemcpyPeerAsync", cudaStreamSynchronize(stream));
            ExpectSuccess("destroy peer stream", cudaStreamDestroy(stream));
        }
    }
    if (src != nullptr) {
        ExpectSuccess("free peer src", cudaFree(src));
    }
    if (dst != nullptr) {
        ExpectSuccess("free peer dst", cudaFree(dst));
    }

    if (deviceCount < 2) {
        Skip("peer access APIs", "requires at least two visible devices");
        return;
    }

    int canAccess = 0;
    if (!ExpectSuccess("cudaDeviceCanAccessPeer", cudaDeviceCanAccessPeer(&canAccess, 0, 1))) {
        return;
    }
    if (!canAccess) {
        Skip("cudaDeviceEnablePeerAccess", "device 0 cannot access device 1");
        Skip("cudaMemcpyPeer", "device 0 cannot access device 1");
        return;
    }

    cudaSetDevice(0);
    ExpectSuccess("cudaDeviceEnablePeerAccess", cudaDeviceEnablePeerAccess(1, 0));
    ExpectSuccess("cudaDeviceDisablePeerAccess", cudaDeviceDisablePeerAccess(1));
}

void CheckMempoolMockApis(int device)
{
    cudaMemPool_t pool = nullptr;
    cudaMemPoolProps props{};
    props.memPoolType = cudaMemPoolTypeDevice;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = device;

    ExpectError("cudaMemPoolCreate", cudaMemPoolCreate(&pool, &props), cudaErrorNotSupported);
    ExpectError("cudaDeviceGetDefaultMemPool", cudaDeviceGetDefaultMemPool(&pool, device), cudaErrorNotSupported);
    ExpectError("cudaDeviceSetMemPool", cudaDeviceSetMemPool(device, pool), cudaErrorNotSupported);
    ExpectError("cudaDeviceGetMemPool", cudaDeviceGetMemPool(&pool, device), cudaErrorNotSupported);
    ExpectError("cudaMemPoolDestroy", cudaMemPoolDestroy(pool), cudaErrorNotSupported);
    size_t threshold = 0;
    ExpectError("cudaMemPoolSetAttribute", cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold),
                cudaErrorNotSupported);
    ExpectError("cudaMemPoolGetAttribute", cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold),
                cudaErrorNotSupported);
    void *ptr = nullptr;
    ExpectError("cudaMemPoolMalloc", cudaMemPoolMalloc(&ptr, pool, 128), cudaErrorNotSupported);
    ExpectError("cudaMemPoolFree", cudaMemPoolFree(ptr, pool, nullptr), cudaErrorNotSupported);
    ExpectError("cudaMemPoolTrimTo", cudaMemPoolTrimTo(pool, 0), cudaErrorNotSupported);
    cudaMemAccessDesc desc{};
    desc.location = props.location;
    desc.access = cudaMemAccessReadWrite;
    ExpectError("cudaMemPoolSetAccess", cudaMemPoolSetAccess(pool, &desc, 1), cudaErrorNotSupported);
    cudaMemAccessFlags access = cudaMemAccessDefault;
    ExpectError("cudaMemPoolGetAccess", cudaMemPoolGetAccess(&access, pool, &props.location), cudaErrorNotSupported);
    ExpectError("cudaMallocAsync", cudaMallocAsync(&ptr, 128, nullptr), cudaErrorNotSupported);
    ExpectError("cudaFreeAsync", cudaFreeAsync(ptr, nullptr), cudaErrorNotSupported);
}

struct DriverVmmState {
    CUmemAllocationProp prop{};
    size_t granularity = 0;
    CUdeviceptr address = nullptr;
    CUmemGenericAllocationHandle handle = nullptr;
    bool created = false;
    bool mapped = false;
};

bool PrepareDriverVmmMapping(DriverVmmState *state, int device)
{
    state->prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    state->prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    state->prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    state->prop.location.id = device;
    if (!OptionalCuSuccess("cuMemGetAllocationGranularity",
                           cuMemGetAllocationGranularity(&state->granularity, &state->prop,
                                                         CU_MEM_ALLOC_GRANULARITY_MINIMUM))) {
        return false;
    }
    if (state->granularity == 0) {
        state->granularity = 2 * 1024 * 1024;
    }
    if (!OptionalCuSuccess("cuMemAddressReserve",
                           cuMemAddressReserve(&state->address, state->granularity, 0, nullptr, 0))) {
        return false;
    }
    state->created = OptionalCuSuccess("cuMemCreate", cuMemCreate(&state->handle, &state->prop, state->granularity));
    if (state->created) {
        state->mapped = OptionalCuSuccess("cuMemMap",
                                          cuMemMap(state->address, state->granularity, 0, state->handle, 0));
    }
    return true;
}

void CheckDriverVmmMappedAccess(const DriverVmmState &state)
{
    if (!state.mapped) {
        Skip("cuMemSetAccess", "mapping did not succeed");
        Skip("cuMemGetAccess", "mapping did not succeed");
        Skip("cuMemRetainAllocationHandle", "mapping did not succeed");
        Skip("cuMemUnmap", "mapping did not succeed");
        return;
    }
    CUmemAccessDesc accessDesc{};
    accessDesc.location = state.prop.location;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    OptionalCuSuccess("cuMemSetAccess", cuMemSetAccess(state.address, state.granularity, &accessDesc, 1));

    unsigned long long accessFlags = 0;
    OptionalCuSuccess("cuMemGetAccess", cuMemGetAccess(&accessFlags, &state.prop.location, state.address));

    CUmemGenericAllocationHandle retained = nullptr;
    OptionalCuSuccess("cuMemRetainAllocationHandle", cuMemRetainAllocationHandle(&retained, state.address));

    OptionalCuSuccess("cuMemUnmap", cuMemUnmap(state.address, state.granularity));
}

void CleanupDriverVmmMapping(const DriverVmmState &state)
{
    if (state.created) {
        OptionalCuSuccess("cuMemRelease", cuMemRelease(state.handle));
    }
    OptionalCuSuccess("cuMemAddressFree", cuMemAddressFree(state.address, state.granularity));
}

void CheckDriverVmmExportImport(CUmemAllocationProp prop, size_t granularity)
{
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    CUmemGenericAllocationHandle exportHandle = nullptr;
    if (!OptionalCuSuccess("cuMemCreate export handle", cuMemCreate(&exportHandle, &prop, granularity))) {
        return;
    }
    char shareableHandle[128] = {};
    if (OptionalCuSuccess("cuMemExportToShareableHandle",
                          cuMemExportToShareableHandle(shareableHandle, exportHandle,
                                                       CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0))) {
        CUmemGenericAllocationHandle importedHandle = nullptr;
        if (OptionalCuSuccess("cuMemImportFromShareableHandle",
                              cuMemImportFromShareableHandle(&importedHandle, shareableHandle,
                                                             CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR))) {
            OptionalCuSuccess("cuMemRelease imported handle", cuMemRelease(importedHandle));
        }
    }
    OptionalCuSuccess("cuMemRelease export handle", cuMemRelease(exportHandle));
}

void CheckDriverVmmApis(int device)
{
    DriverVmmState state{};
    if (!PrepareDriverVmmMapping(&state, device)) {
        return;
    }
    CheckDriverVmmMappedAccess(state);
    CleanupDriverVmmMapping(state);
    CheckDriverVmmExportImport(state.prop, state.granularity);
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

void CheckIncrementalInvalidInputs()
{
    ExpectError("cudaGetSymbolAddress invalid", cudaGetSymbolAddress(nullptr, nullptr), cudaErrorInvalidValue);
    ExpectError("cudaMemcpyToSymbol invalid",
                cudaMemcpyToSymbol(nullptr, nullptr, 0, 0, cudaMemcpyHostToDevice), cudaErrorInvalidValue);
    ExpectError("cudaFuncGetAttributes invalid", cudaFuncGetAttributes(nullptr, nullptr), cudaErrorInvalidValue);

    ExpectError("cudaGraphConditionalHandleCreate invalid",
                cudaGraphConditionalHandleCreate(nullptr, nullptr), cudaErrorInvalidValue);
    ExpectError("cudaGraphGetNodes invalid", cudaGraphGetNodes(nullptr, nullptr, nullptr), cudaErrorInvalidValue);
    ExpectError("cudaGraphSetConditional invalid", cudaGraphSetConditional(nullptr, 1), cudaErrorInvalidValue);
    ExpectError("cudaStreamBeginCaptureToGraph invalid",
                cudaStreamBeginCaptureToGraph(nullptr, nullptr, nullptr, nullptr, 0), cudaErrorInvalidValue);
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
                      cudaStreamGetCaptureInfo(captureStream, &status, nullptr, &graph, &deps, nullptr, &depCount));
        ExpectSuccess("cudaStreamGetCaptureInfo_v3",
                      cudaStreamGetCaptureInfo_v3(captureStream, &status, nullptr, &graph, &deps, nullptr, &depCount));
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

void CheckIncrementalModuleInvalidInputs()
{
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    ExpectCuError("cuModuleLoad invalid", cuModuleLoad(&module, nullptr), CUDA_ERROR_INVALID_VALUE);
    ExpectCuError("cuModuleLoadData invalid image", cuModuleLoadData(&module, "not an elf"),
                  CUDA_ERROR_INVALID_IMAGE);
    ExpectCuError("cuModuleGetFunction invalid", cuModuleGetFunction(&function, nullptr, "kernel"),
                  CUDA_ERROR_INVALID_VALUE);
    ExpectCuError("cuModuleUnload invalid", cuModuleUnload(nullptr), CUDA_ERROR_INVALID_VALUE);
}

void CheckIncrementalApiSurface(int device)
{
    constexpr size_t bytes = 4096;
    CheckIncrementalHostAlloc(bytes);
    CheckIncrementalEventRecord();
    CheckIncrementalInvalidInputs();
    CheckIncrementalCaptureInfo();
    CheckIncrementalContext(device);
    CheckIncrementalDriverWrites(bytes);
    CheckIncrementalModuleInvalidInputs();
}

}  // namespace

int main(int argc, char **argv)
{
    std::printf("%s Starting...\n\n", argv[0]);
    std::printf(" CUDA Runtime API Coverage for CANN Compatibility Layer\n\n");

    int deviceCount = 0;
    if (!ExpectSuccess("cudaGetDeviceCount", cudaGetDeviceCount(&deviceCount)) || deviceCount == 0) {
        std::printf("\nResult = FAIL\n");
        return EXIT_FAILURE;
    }

    int device = 0;
    ExpectSuccess("cudaSetDevice", cudaSetDevice(device));
    ExpectSuccess("cudaGetDevice", cudaGetDevice(&device));

    CheckErrorApis();
    CheckDeviceStateApis();
    CheckStreamPriorityApis();
    CheckStreamCaptureApis();
    CheckPitchAnd2DMemoryApis();
    CheckManagedMemoryApis(device);
    CheckMemPrefetchMockApi(device);
    CheckMemcpyBatchApi(device);
    CheckHostRegistrationApis();
    CheckPeerApis(deviceCount);
    CheckMempoolMockApis(device);
    CheckLaunchHostFuncApi();
    CheckIncrementalApiSurface(device);
    CheckDriverVmmApis(device);

    ExpectSuccess("cudaDeviceSynchronize", cudaDeviceSynchronize());
    ExpectSuccess("cudaDeviceReset", cudaDeviceReset());

    std::printf("\napiRuntimeCoverage, Runtime API checks = %d, Failures = %d\n", g_checks, g_failures);
    std::printf("Result = %s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
