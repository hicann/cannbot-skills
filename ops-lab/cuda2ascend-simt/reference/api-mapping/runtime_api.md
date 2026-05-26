# Runtime API Mapping

## Device Management

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| cudaChooseDevice |  |  |
| cudaCtxResetPersistingL2Cache |  |  |
| cudaDeviceFlushGPUDirectRDMAWrites |  |  |
| cudaDeviceGetAttribute |  |  |
| cudaDeviceGetByPCIBusId |  |  |
| cudaDeviceGetCacheConfig |  |  |
| cudaDeviceGetDefaultMemPool |  |  |
| cudaDeviceGetHostAtomicCapabilities |  |  |
| cudaDeviceGetLimit |  |  |
| cudaDeviceGetMemPool |  |  |
| cudaDeviceGetNvSciSyncAttributes |  |  |
| cudaDeviceGetPCIBusId |  |  |
| cudaDeviceGetStreamPriorityRange | aclrtDeviceGetStreamPriorityRange | aclError aclrtDeviceGetStreamPriorityRange(int32_t *leastPriority, int32_t *greatestPriority) |
| cudaDeviceGetTexture1DLinearMaxWidth |  |  |
| cudaDeviceRegisterAsyncNotification |  |  |
| cudaDeviceReset | aclrtResetDeviceForce | aclError aclrtResetDeviceForce(int32_t deviceId) |
| cudaDeviceSetCacheConfig |  |  |
| cudaDeviceSetLimit |  |  |
| cudaDeviceSetMemPool |  |  |
| cudaDeviceSynchronize | aclrtSynchronizeDevice | aclError aclrtSynchronizeDevice(void) |
| cudaDeviceUnregisterAsyncNotification |  |  |
| cudaGetDevice | aclrtGetDevice | aclError aclrtGetDevice(int32_t *deviceId) |
| cudaGetDeviceCount | aclrtGetDeviceCount | aclError aclrtGetDeviceCount(uint32_t *count) |
| cudaGetDeviceFlags |  |  |
| cudaGetDeviceProperties | aclrtGetDeviceInfo | aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value) |
| cudaInitDevice |  |  |
| cudaSetDevice | aclrtSetDevice | aclError aclrtSetDevice(int32_t deviceId) |
| cudaSetDeviceFlags |  |  |

## Error Handling

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| cudaGetErrorName |  |  |
| cudaGetErrorString |  |  |
| cudaGetLastError | aclrtGetLastError | aclError aclrtGetLastError(aclrtLastErrLevel level) |
| cudaPeekAtLastError | aclrtPeekAtLastError | aclError aclrtPeekAtLastError(aclrtLastErrLevel level) |

## Stream Management

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| cudaStreamAddCallback |  |  |
| cudaStreamAttachMemAsync |  |  |
| cudaStreamBeginCapture |  |  |
| cudaStreamBeginCaptureToGraph |  |  |
| cudaStreamCopyAttributes |  |  |
| cudaStreamCreate | aclrtCreateStream | aclError aclrtCreateStream(aclrtStream *stream) |
| cudaStreamCreateWithFlags | aclrtCreateStreamWithConfig | aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag) |
| cudaStreamCreateWithPriority | aclrtCreateStreamWithConfig | aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag) |
| cudaStreamDestroy | aclrtDestroyStream | aclError aclrtDestroyStream(aclrtStream stream) |
| cudaStreamEndCapture |  |  |
| cudaStreamGetAttribute |  |  |
| cudaStreamGetCaptureInfo |  |  |
| cudaStreamGetCaptureInfo_v3 |  |  |
| cudaStreamGetDevice |  |  |
| cudaStreamGetFlags |  |  |
| cudaStreamGetId |  |  |
| cudaStreamGetPriority | aclrtStreamGetPriority | aclError aclrtStreamGetPriority(aclrtStream stream, uint32_t *priority) |
| cudaStreamIsCapturing |  |  |
| cudaStreamQuery | aclrtStreamQuery | aclError aclrtStreamQuery(aclrtStream stream, aclrtStreamStatus *status) |
| cudaStreamSetAttribute |  |  |
| cudaStreamSynchronize | aclrtSynchronizeStream | aclError aclrtSynchronizeStream(aclrtStream stream) |
| cudaStreamUpdateCaptureDependencies |  |  |
| cudaStreamUpdateCaptureDependencies_v2 |  |  |
| cudaStreamWaitEvent | aclrtStreamWaitEvent | aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) |
| cudaThreadExchangeStreamCaptureMode |  |  |

## Event Management

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| cudaEventCreate | aclrtCreateEvent | aclError aclrtCreateEvent(aclrtEvent *event) |
| cudaEventCreateWithFlags | aclrtCreateEventWithFlag | aclError aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag) |
| cudaEventDestroy | aclrtDestroyEvent | aclError aclrtDestroyEvent(aclrtEvent event) |
| cudaEventElapsedTime | aclrtEventElapsedTime | aclError aclrtEventElapsedTime(float *ms, aclrtEvent startEvent, aclrtEvent endEvent) |
| cudaEventElapsedTime_v2 | aclrtEventElapsedTime | aclError aclrtEventElapsedTime(float *ms, aclrtEvent startEvent, aclrtEvent endEvent) |
| cudaEventQuery | aclrtQueryEventStatus | aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status) |
| cudaEventRecord | aclrtRecordEvent | aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) |
| cudaEventRecordWithFlags |  |  |
| cudaEventSynchronize | aclrtSynchronizeEvent | aclError aclrtSynchronizeEvent(aclrtEvent event) |

## Execution Control

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| cudaFuncGetAttributes |  |  |
| cudaFuncGetName |  |  |
| cudaFuncGetParamInfo |  |  |
| cudaFuncSetAttribute |  |  |
| cudaFuncSetCacheConfig |  |  |
| cudaGetFuncBySymbol |  |  |
| cudaGetKernel |  |  |
| cudaGetParameterBuffer |  |  |
| cudaGetParameterBufferV2 |  |  |
| cudaKernelSetAttributeForDevice |  |  |
| cudaLaunchCooperativeKernel |  |  |
| cudaLaunchCooperativeKernelMultiDevice |  |  |
| cudaLaunchHostFunc | aclrtLaunchHostFunc | aclError aclrtLaunchHostFunc(aclrtStream stream, aclrtHostFunc fn, void *args) |
| cudaLaunchKernel | aclrtLaunchKernelWithHostArgs | aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t numBlocks, aclrtStream stream, aclrtLaunchKernelCfg *cfg, void *hostArgs, size_t argsSize, aclrtPlaceHolderInfo *placeHolderArray, size_t placeHolderNum) |
| cudaLaunchKernelExC | aclrtLaunchKernelWithHostArgs | aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t numBlocks, aclrtStream stream, aclrtLaunchKernelCfg *cfg, void *hostArgs, size_t argsSize, aclrtPlaceHolderInfo *placeHolderArray, size_t placeHolderNum) |
| cudaSetDoubleForDevice |  |  |
| cudaSetDoubleForHost |  |  |

## Occupancy

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| cudaOccupancyAvailableDynamicSMemPerBlock |  |  |
| cudaOccupancyMaxActiveBlocksPerMultiprocessor | aclOccupancyMaxActiveBlocksPerVectorCoreFromDevice | aclError aclOccupancyMaxActiveBlocksPerVectorCoreFromDevice(int32_t *numBlocks, int32_t deviceId, int64_t threadsPerBlock, int64_t dynamicUbufBytesPerBlock, aclOccupancyResult *result) |
| cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags |  |  |
| cudaOccupancyMaxActiveClusters |  |  |
| cudaOccupancyMaxPotentialBlockSize |  |  |
| cudaOccupancyMaxPotentialBlockSizeVariableSMem |  |  |
| cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags |  |  |
| cudaOccupancyMaxPotentialBlockSizeWithFlags |  |  |
| cudaOccupancyMaxPotentialClusterSize |  |  |

## Memory Management

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| cudaFree | aclrtFree | aclError aclrtFree(void *devPtr) |
| cudaFreeArray |  |  |
| cudaFreeAsync |  |  |
| cudaFreeHost | aclrtFreeHost | aclError aclrtFreeHost(void *hostPtr) |
| cudaFreeMipmappedArray |  |  |
| cudaGetMipmappedArrayLevel |  |  |
| cudaGetSymbolAddress |  |  |
| cudaGetSymbolSize |  |  |
| cudaHostAlloc | aclrtMallocHost | aclError aclrtMallocHost(void **hostPtr, size_t size) |
| cudaHostGetDevicePointer | aclrtHostGetDevicePointer | aclError aclrtHostGetDevicePointer(void *pHost, void **pDevice, uint32_t flag) |
| cudaHostGetFlags |  |  |
| cudaHostRegister | aclrtHostRegister | aclError aclrtHostRegister(void *ptr, uint64_t size, aclrtHostRegisterType type, void **devPtr) |
| cudaHostUnregister | aclrtHostUnregister | aclError aclrtHostUnregister(void *ptr) |
| cudaMalloc | aclrtMalloc | aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy) |
| cudaMalloc3D |  |  |
| cudaMalloc3DArray |  |  |
| cudaMallocArray |  |  |
| cudaMallocAsync |  |  |
| cudaMallocFromPoolAsync |  |  |
| cudaMallocHost | aclrtMallocHost | aclError aclrtMallocHost(void **hostPtr, size_t size) |
| cudaMallocManaged |  |  |
| cudaMallocMipmappedArray |  |  |
| cudaMallocPitch |  |  |
| cudaMemAdvise |  |  |
| cudaMemAdvise_v2 |  |  |
| cudaMemDiscardAndPrefetchBatchAsync |  |  |
| cudaMemDiscardBatchAsync |  |  |
| cudaMemGetDefaultMemPool |  |  |
| cudaMemGetInfo | aclrtGetMemInfo | aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total) |
| cudaMemGetMemPool |  |  |
| cudaMemPoolCreate |  |  |
| cudaMemPoolDestroy |  |  |
| cudaMemPoolExportPointer |  |  |
| cudaMemPoolExportToShareableHandle |  |  |
| cudaMemPoolGetAccess |  |  |
| cudaMemPoolGetAttribute |  |  |
| cudaMemPoolImportFromShareableHandle |  |  |
| cudaMemPoolImportPointer |  |  |
| cudaMemPoolSetAccess |  |  |
| cudaMemPoolSetAttribute |  |  |
| cudaMemPoolTrimTo |  |  |
| cudaMemPrefetchAsync |  |  |
| cudaMemPrefetchAsync_v2 |  |  |
| cudaMemPrefetchBatchAsync |  |  |
| cudaMemRangeGetAttribute |  |  |
| cudaMemRangeGetAttributes |  |  |
| cudaMemSetMemPool |  |  |
| cudaMemcpy | aclrtMemcpy | aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) |
| cudaMemcpy2D |  |  |
| cudaMemcpy2DArrayToArray |  |  |
| cudaMemcpy2DAsync |  |  |
| cudaMemcpy2DFromArray |  |  |
| cudaMemcpy2DFromArrayAsync |  |  |
| cudaMemcpy2DToArray |  |  |
| cudaMemcpy2DToArrayAsync |  |  |
| cudaMemcpy3D |  |  |
| cudaMemcpy3DAsync |  |  |
| cudaMemcpy3DBatchAsync |  |  |
| cudaMemcpyArrayToArray |  |  |
| cudaMemcpyAsync | aclrtMemcpyAsync | aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind, aclrtStream stream) |
| cudaMemcpyBatchAsync | aclrtMemcpyBatchAsync | aclError aclrtMemcpyBatchAsync(void **dsts, size_t *destMaxs, void **srcs, size_t *sizes, size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes, size_t numAttrs, size_t *failIndex, aclrtStream stream) |
| cudaMemcpyFromArray |  |  |
| cudaMemcpyFromArrayAsync |  |  |
| cudaMemcpyFromSymbol |  |  |
| cudaMemcpyFromSymbolAsync |  |  |
| cudaMemcpyToArray |  |  |
| cudaMemcpyToArrayAsync |  |  |
| cudaMemcpyToSymbol |  |  |
| cudaMemcpyToSymbolAsync |  |  |
| cudaMemset | aclrtMemset | aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count) |
| cudaMemset2D |  |  |
| cudaMemset2DAsync |  |  |
| cudaMemset3D |  |  |
| cudaMemset3DAsync |  |  |
| cudaMemsetAsync | aclrtMemsetAsync | aclError aclrtMemsetAsync(void *devPtr, size_t maxCount, int32_t value, size_t count, aclrtStream stream) |
| cudaPointerGetAttributes | aclrtPointerGetAttributes | aclError aclrtPointerGetAttributes(const void *ptr, aclrtPtrAttributes *attributes) |
| make_cudaExtent |  |  |
| make_cudaPitchedPtr |  |  |
| make_cudaPos |  |  |

## Peer Device Memory Access

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| cudaDeviceCanAccessPeer | aclrtDeviceCanAccessPeer | aclError aclrtDeviceCanAccessPeer(int32_t *canAccessPeer, int32_t deviceId, int32_t peerDeviceId) |
| cudaDeviceDisablePeerAccess | aclrtDeviceDisablePeerAccess | aclError aclrtDeviceDisablePeerAccess(int32_t peerDeviceId) |
| cudaDeviceEnablePeerAccess | aclrtDeviceEnablePeerAccess | aclError aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags) |
| cudaDeviceGetP2PAtomicCapabilities |  |  |
| cudaDeviceGetP2PAttribute |  |  |
| cudaMemcpy3DPeer |  |  |
| cudaMemcpy3DPeerAsync |  |  |
| cudaMemcpyPeer |  |  |
| cudaMemcpyPeerAsync |  |  |

## Interprocess Communication

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| cudaIpcCloseMemHandle | aclrtIpcMemClose | aclError aclrtIpcMemClose(const char *key) |
| cudaIpcGetEventHandle | aclrtIpcGetEventHandle | aclError aclrtIpcGetEventHandle(aclrtEvent event, aclrtIpcEventHandle *handle) |
| cudaIpcGetMemHandle | aclrtIpcMemGetExportKey | aclError aclrtIpcMemGetExportKey(void *devPtr, size_t size, char *key, size_t len, uint64_t flags) |
| cudaIpcOpenEventHandle | aclrtIpcOpenEventHandle | aclError aclrtIpcOpenEventHandle(aclrtIpcEventHandle handle, aclrtEvent *event) |
| cudaIpcOpenMemHandle | aclrtIpcMemImportByKey | aclError aclrtIpcMemImportByKey(void **devPtr, const char *key, uint64_t flags) |

## Texture Reference Management

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| cudaArrayGetInfo |  |  |
| cudaArrayGetMemoryRequirements |  |  |
| cudaArrayGetPlane |  |  |
| cudaArrayGetSparseProperties |  |  |
| cudaCreateChannelDesc |  |  |
| cudaCreateTextureObject |  |  |
| cudaCreateTextureObject_v2 |  |  |
| cudaDestroyTextureObject |  |  |
| cudaGetChannelDesc |  |  |
| cudaGetTextureObjectResourceDesc |  |  |
| cudaGetTextureObjectResourceViewDesc |  |  |
| cudaGetTextureObjectTextureDesc |  |  |
| cudaGetTextureObjectTextureDesc_v2 |  |  |

## Surface Management

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| cudaCreateSurfaceObject |  |  |
| cudaDestroySurfaceObject |  |  |
| cudaGetSurfaceObjectResourceDesc |  |  |

## External Resource Interoperability

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| cudaDestroyExternalMemory |  |  |
| cudaDestroyExternalSemaphore |  |  |
| cudaExternalMemoryGetMappedBuffer |  |  |
| cudaExternalMemoryGetMappedMipmappedArray |  |  |
| cudaImportExternalMemory |  |  |
| cudaImportExternalSemaphore |  |  |
| cudaSignalExternalSemaphoresAsync |  |  |
| cudaWaitExternalSemaphoresAsync |  |  |

## Version Management

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| cudaDriverGetVersion |  |  |
| cudaRuntimeGetVersion |  |  |

## Profiler and Logging

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| cudaLogsCurrent |  |  |
| cudaLogsDumpToFile |  |  |
| cudaLogsDumpToMemory |  |  |
| cudaLogsRegisterCallback |  |  |
| cudaLogsUnregisterCallback |  |  |
| cudaProfilerStart |  |  |
| cudaProfilerStop |  |  |

## Driver Entry Point and Library Management

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| cudaGetDriverEntryPoint |  |  |
| cudaGetDriverEntryPointByVersion |  |  |
| cudaLibraryEnumerateKernels |  |  |
| cudaLibraryGetGlobal |  |  |
| cudaLibraryGetKernel |  |  |
| cudaLibraryGetKernelCount |  |  |
| cudaLibraryGetManaged |  |  |
| cudaLibraryGetUnifiedFunction |  |  |
| cudaLibraryLoadData |  |  |
| cudaLibraryLoadFromFile |  |  |
| cudaLibraryUnload |  |  |

## Other Runtime APIs

| CUDA API | Ascend API | Ascend signature |
|---|---|---|
| cudaSetValidDevices |  |  |
