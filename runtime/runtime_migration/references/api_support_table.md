# CUDA API 支持状态表

## 一、设备管理 API


| CUDA API                             | CANN API                              | 实现状态 | 说明                              |
| ------------------------------------ | ------------------------------------- | :------: | --------------------------------- |
| `cudaGetDeviceCount()`               | `aclrtGetDeviceCount()`               |    ✅    | 获取设备数量                      |
| `cudaSetDevice()`                    | `aclrtSetDevice()`                    |    ✅    | 设置当前设备                      |
| `cudaGetDevice()`                    | `aclrtGetDevice()`                    |    ✅    | 获取当前设备 ID                   |
| `cudaGetDeviceProperties()`          | `aclrtGetDeviceInfo()`                |    ✅    | 获取设备属性                      |
| `cudaDeviceGetAttribute()`           | Mock 实现                             |    ✅    | ComputeCapability, ComputeMode 等 |
| `cudaDeviceSynchronize()`            | `aclrtSynchronizeDevice()`            |    ✅    | 设备同步                          |
| `cudaDeviceReset()`                  | `aclrtResetDeviceForce()`             |    ✅    | 重置设备                          |
| `cudaSetDeviceFlags()`               | 内部存储                              |    ✅    | 设置设备标志                      |
| `cudaGetDeviceFlags()`               | 内部存储                              |    ✅    | 获取设备标志                      |
| `cudaDeviceSetLimit()`               | 内部存储                              |    ✅    | 设置设备限制                      |
| `cudaDeviceGetLimit()`               | Mock 实现                             |    ✅    | 获取设备限制                      |
| `cudaDeviceGetCacheConfig()`         | Mock 实现                             |    ✅    | 获取缓存配置                      |
| `cudaDeviceSetCacheConfig()`         | 内部存储                              |    ✅    | 设置缓存配置                      |
| `cudaDeviceGetStreamPriorityRange()` | `aclrtDeviceGetStreamPriorityRange()` |    ✅    | 获取流优先级范围                  |
| `cudaDeviceEnablePeerAccess()`       | `aclrtDeviceEnablePeerAccess()`       |    ✅    | 启用点对点访问                    |
| `cudaDeviceDisablePeerAccess()`      | `aclrtDeviceDisablePeerAccess()`      |    ✅    | 禁用点对点访问                    |
| `cudaDeviceCanAccessPeer()`          | `aclrtDeviceCanAccessPeer()`          |    ✅    | 检查点对点访问能力                |

## 二、内存管理 API


| CUDA API                     | CANN API                      | 实现状态 | 说明                   |
| ---------------------------- | ----------------------------- | :------: | ---------------------- |
| `cudaMalloc()`               | `aclrtMalloc()`               |    ✅    | 设备内存分配           |
| `cudaFree()`                 | `aclrtFree()`                 |    ✅    | 设备内存释放           |
| `cudaMallocPitch()`          | `aclrtMalloc()`               |    ✅    | 分配对齐内存           |
| `cudaMallocHost()`           | `aclrtMallocHost()`           |    ✅    | 分配页锁定主机内存     |
| `cudaFreeHost()`             | `aclrtFreeHost()`             |    ✅    | 释放页锁定主机内存     |
| `cudaMallocManaged()`        | `aclrtMemAllocManaged()`      |    ✅    | 统一内存分配 (UVM)     |
| `cudaMemcpy()`               | `aclrtMemcpy()`               |    ✅    | 同步内存拷贝           |
| `cudaMemcpyAsync()`          | `aclrtMemcpyAsync()`          |    ✅    | 异步内存拷贝           |
| `cudaMemcpy2D()`             | `aclrtMemcpy2d()`             |    ✅    | 2D 内存拷贝            |
| `cudaMemcpy2DAsync()`        | `aclrtMemcpy2dAsync()`        |    ✅    | 异步 2D 内存拷贝       |
| `cudaMemcpyPeer()`           | `aclrtMemcpy()`               |    ✅    | 点对点设备间拷贝       |
| `cudaMemcpyPeerAsync()`      | `aclrtMemcpyAsync()`          |    ✅    | 异步点对点拷贝         |
| `cudaMemcpyBatchAsync()`     | `aclrtMemcpyBatchAsyncV2()`   |   ✅*   | 批量拷贝 (CANN 9.1.0+) |
| `cudaMemset()`               | `aclrtMemset()`               |    ✅    | 内存初始化             |
| `cudaMemsetAsync()`          | `aclrtMemsetAsync()`          |    ✅    | 异步内存初始化         |
| `cudaMemset2D()`             | `aclrtMemset()` (逐行)        |    ✅    | 2D 内存初始化          |
| `cudaMemset2DAsync()`        | `aclrtMemsetAsync()` (逐行)   |    ✅    | 异步 2D 初始化         |
| `cudaMemGetInfo()`           | `aclrtGetMemInfo()`           |    ✅    | 获取内存信息           |
| `cudaPointerGetAttributes()` | `aclrtPointerGetAttributes()` |    ✅    | 获取指针属性           |
| `cudaHostRegister()`         | `aclrtHostRegisterV2()`       |    ✅    | 注册主机内存           |
| `cudaHostUnregister()`       | `aclrtHostUnregister()`       |    ✅    | 取消注册主机内存       |
| `cudaHostGetDevicePointer()` | `aclrtHostGetDevicePointer()` |    ✅    | 获取设备指针           |

## 三、内存池管理 API (不支持)


| CUDA API                        | 实现状态 | 说明                       |
| ------------------------------- | :-------: | -------------------------- |
| `cudaMemPoolCreate()`           | ⚠️ Mock | 返回 cudaErrorNotSupported |
| `cudaMemPoolDestroy()`          | ⚠️ Mock | 返回 cudaErrorNotSupported |
| `cudaMemPoolSetAttribute()`     | ⚠️ Mock | 返回 cudaErrorNotSupported |
| `cudaMemPoolGetAttribute()`     | ⚠️ Mock | 返回 cudaErrorNotSupported |
| `cudaMemPoolMalloc()`           | ⚠️ Mock | 返回 cudaErrorNotSupported |
| `cudaMemPoolFree()`             | ⚠️ Mock | 返回 cudaErrorNotSupported |
| `cudaMemPoolTrimTo()`           | ⚠️ Mock | 返回 cudaErrorNotSupported |
| `cudaMemPoolSetAccess()`        | ⚠️ Mock | 返回 cudaErrorNotSupported |
| `cudaMemPoolGetAccess()`        | ⚠️ Mock | 返回 cudaErrorNotSupported |
| `cudaDeviceGetDefaultMemPool()` | ⚠️ Mock | 返回 cudaErrorNotSupported |
| `cudaDeviceSetMemPool()`        | ⚠️ Mock | 返回 cudaErrorNotSupported |
| `cudaDeviceGetMemPool()`        | ⚠️ Mock | 返回 cudaErrorNotSupported |
| `cudaMallocAsync()`             | ⚠️ Mock | CANN 不支持异步内存分配    |
| `cudaFreeAsync()`               | ⚠️ Mock | CANN 不支持异步内存释放    |
| `cudaMemPrefetchAsync()`        | ⚠️ Mock | CANN 自动处理数据迁移      |

## 四、流管理 API


| CUDA API                                | CANN API                              | 实现状态 | 说明               |
| --------------------------------------- | ------------------------------------- | :------: | ------------------ |
| `cudaStreamCreate()`                    | `aclrtCreateStreamWithConfig()`       |    ✅    | 创建流             |
| `cudaStreamCreateWithFlags()`           | `aclrtCreateStreamWithConfig()`       |    ✅    | 创建流（带标志）   |
| `cudaStreamCreateWithPriority()`        | `aclrtCreateStreamWithConfig()`       |    ✅    | 创建流（带优先级） |
| `cudaStreamDestroy()`                   | `aclrtDestroyStream()`                |    ✅    | 销毁流             |
| `cudaStreamSynchronize()`               | `aclrtSynchronizeStream()`            |    ✅    | 流同步             |
| `cudaStreamQuery()`                     | `aclrtStreamQuery()`                  |    ✅    | 查询流状态         |
| `cudaStreamWaitEvent()`                 | `aclrtStreamWaitEvent()`              |    ✅    | 流等待事件         |
| `cudaStreamGetId()`                     | `aclrtStreamGetId()`                  |    ✅    | 获取流 ID          |
| `cudaStreamGetPriority()`               | `aclrtStreamGetPriority()`            |    ✅    | 获取流优先级       |
| `cudaStreamGetFlags()`                  | `aclrtStreamGetFlags()`               |    ✅    | 获取流标志         |
| `cudaStreamBeginCapture()`              | `aclmdlRICaptureBegin()`              |    ✅    | 开始流捕获         |
| `cudaStreamEndCapture()`                | `aclmdlRICaptureEnd()`                |    ✅    | 结束流捕获         |
| `cudaStreamIsCapturing()`               | `aclmdlRICaptureGetInfo()`            |    ✅    | 查询流捕获状态     |
| `cudaThreadExchangeStreamCaptureMode()` | `aclmdlRICaptureThreadExchangeMode()` |    ✅    | 交换流捕获模式     |

## 五、事件管理 API


| CUDA API                     | CANN API                       | 实现状态 | 说明               |
| ---------------------------- | ------------------------------ | :------: | ------------------ |
| `cudaEventCreate()`          | `aclrtCreateEventExWithFlag()` |    ✅    | 创建事件           |
| `cudaEventCreateWithFlags()` | `aclrtCreateEventExWithFlag()` |    ✅    | 创建事件（带标志） |
| `cudaEventDestroy()`         | `aclrtDestroyEvent()`          |    ✅    | 销毁事件           |
| `cudaEventRecord()`          | `aclrtRecordEvent()`           |    ✅    | 记录事件           |
| `cudaEventSynchronize()`     | `aclrtSynchronizeEvent()`      |    ✅    | 事件同步           |
| `cudaEventQuery()`           | `aclrtQueryEventStatus()`      |    ✅    | 查询事件状态       |
| `cudaEventElapsedTime()`     | `aclrtEventElapsedTime()`      |    ✅    | 计算事件耗时       |

## 六、IPC API


| CUDA API                   | CANN API                                | 实现状态 | 说明              |
| -------------------------- | --------------------------------------- | :------: | ----------------- |
| `cudaIpcGetMemHandle()`    | `aclrtMemExportToShareableHandleV2()`   |    ✅    | 获取 IPC 内存句柄 |
| `cudaIpcOpenMemHandle()`   | `aclrtMemImportFromShareableHandleV2()` |    ✅    | 打开 IPC 内存句柄 |
| `cudaIpcCloseMemHandle()`  | `aclrtFree()`                           |    ✅    | 关闭 IPC 内存句柄 |
| `cudaIpcGetEventHandle()`  | `aclrtIpcGetEventHandle()`              |    ✅    | 获取 IPC 事件句柄 |
| `cudaIpcOpenEventHandle()` | `aclrtIpcOpenEventHandle()`             |    ✅    | 打开 IPC 事件句柄 |

**⚠️ IPC 约束**：CANN 使用 opaque handle 而非 POSIX fd，跨进程传递必须使用共享内存。

## 七、库/模块管理 API


| CUDA API                    | CANN API                    | 实现状态 | 说明                       |
| --------------------------- | --------------------------- | :------: | -------------------------- |
| `cudaLibraryLoadFromFile()` | `aclrtBinaryLoadFromFile()` |    ✅    | 从文件加载库               |
| `cudaLibraryLoadData()`     | `aclrtBinaryLoadFromData()` |    ✅    | 从内存加载库               |
| `cudaLibraryUnload()`       | `aclrtBinaryUnload()`       |    ✅    | 卸载库                     |
| `cudaLibraryGetFunction()`  | `aclrtBinaryGetFunction()`  |    ✅    | 获取函数句柄               |
| `cudaLibraryGetGlobal()`    | `aclrtBinaryGetGlobal()`    |   ✅*   | 获取全局变量 (CANN 9.1.0+) |

## 八、执行控制 API


| CUDA API               | CANN API                | 实现状态 | 说明         |
| ---------------------- | ----------------------- | :------: | ------------ |
| `cudaLaunchHostFunc()` | `aclrtLaunchHostFunc()` |    ✅    | 主机函数回调 |

## 九、性能分析 API


| CUDA API              | CANN API                           | 实现状态 | 说明         |
| --------------------- | ---------------------------------- | :------: | ------------ |
| `cudaProfilerStart()` | `aclprofInit()` + `aclprofStart()` |    ✅    | 启动性能分析|
| `cudaProfilerStop()`  | `aclprofStop()`                    |    ✅    | 停止性能分析|

## 十、错误处理 API


| CUDA API                | 实现方式                 | 实现状态 | 说明           |
| ----------------------- | ------------------------ | :------: | -------------- |
| `cudaGetLastError()`    | `aclrtGetLastError()`    |    ✅    | 获取最后的错误 |
| `cudaPeekAtLastError()` | `aclrtPeekAtLastError()` |    ✅    | 查看最后的错误 |
| `cudaGetErrorName()`    | 内部实现                 |    ✅    | 获取错误名称   |
| `cudaGetErrorString()`  | 内部实现                 |    ✅    | 获取错误描述   |

## 十一、CUDA Driver VMM API


| CUDA Driver API                    | CANN API                                | 实现状态 | 说明             |
| ---------------------------------- | --------------------------------------- | :------: | ---------------- |
| `cuMemAddressReserve()`            | `aclrtReserveMemAddress()`              |    ✅    | 预留虚拟地址范围 |
| `cuMemAddressFree()`               | `aclrtReleaseMemAddress()`              |    ✅    | 释放虚拟地址范围 |
| `cuMemCreate()`                    | `aclrtMallocPhysical()`                 |    ✅    | 创建物理内存     |
| `cuMemRelease()`                   | `aclrtFreePhysical()`                   |    ✅    | 释放物理内存     |
| `cuMemExportToShareableHandle()`   | `aclrtMemExportToShareableHandleV2()`   |    ✅    | 导出可共享句柄   |
| `cuMemGetAccess()`                 | `aclrtMemGetAccess()`                   |    ✅    | 获取访问权限     |
| `cuMemSetAccess()`                 | `aclrtMemSetAccess()`                   |    ✅    | 设置访问权限     |
| `cuMemGetAllocationGranularity()`  | `aclrtMemGetAllocationGranularity()`    |    ✅    | 获取分配粒度     |
| `cuMemImportFromShareableHandle()` | `aclrtMemImportFromShareableHandleV2()` |    ✅    | 导入可共享句柄   |
| `cuMemMap()`                       | `aclrtMapMem()`                         |    ✅    | 映射物理内存     |
| `cuMemUnmap()`                     | `aclrtUnmapMem()`                       |    ✅    | 取消映射         |
| `cuMemRetainAllocationHandle()`    | `aclrtMemRetainAllocationHandle()`      |    ✅    | 保留分配句柄     |


## 不支持 API 替代方案建议


| 不支持 API             | 替代方案                             |
| ---------------------- | ------------------------------------ |
| `cudaMallocAsync`      | 使用`cudaMalloc` + 流同步            |
| `cudaFreeAsync`        | 使用`cudaFree` + 流同步              |
| `cudaMemPoolCreate`    | 自定义内存池管理，或使用`cudaMalloc` |
| `cudaMemPrefetchAsync` | CANN 自动管理，无需手动处理          |
| Texture/Surface API    | 使用普通内存 + 自定义纹理逻辑        |
