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
| `cudaHostAlloc()`            | `aclrtMallocHostAndRegister()` / `aclrtMallocHost()` + `aclrtHostRegisterV2()` |    ✅    | 分配 Host pinned 内存；mapped flag 会显式转换为 mapped+pinned，旧版缺少分配并注册接口时用 page-aligned host 内存加注册实现 |
| `cudaFreeHost()`             | `aclrtFreeHost()`             |    ✅    | 释放页锁定主机内存     |
| `cudaMemcpy()`               | `aclrtMemcpy()`               |    ✅    | 同步内存拷贝           |
| `cudaMemcpyAsync()`          | `aclrtMemcpyAsync()`          |    ✅    | 异步内存拷贝           |
| `cudaMemcpy2D()`             | `aclrtMemcpy2d()`             |    ✅    | 2D 内存拷贝            |
| `cudaMemcpy2DAsync()`        | `aclrtMemcpy2dAsync()`        |    ✅    | 异步 2D 内存拷贝       |
| `cudaMemcpyPeer()`           | `aclrtMemcpy()`               |    ✅    | 点对点设备间拷贝       |
| `cudaMemcpyPeerAsync()`      | `aclrtMemcpyAsync()` (`ACL_MEMCPY_DEVICE_TO_DEVICE`) |    ✅    | 异步点对点拷贝；当前仅支持同一个 PCIe Switch 内 Device 之间的内存复制 |
| `cudaMemcpyToSymbol()`       | `aclrtMemcpyToSymbol()`       |    ✅    | 向设备符号地址拷贝数据 |
| `cudaMemcpyBatchAsync()`     | `aclrtMemcpyBatchAsyncV2()`   |   ✅*   | 批量拷贝 (CANN 9.1.0+) |
| `cudaMemset()`               | `aclrtMemset()`               |    ✅    | 内存初始化             |
| `cudaMemsetAsync()`          | `aclrtMemsetAsync()`          |    ✅    | 异步内存初始化         |
| `cuMemsetD32Async()`         | `aclrtMemsetD32Async()`       |    ✅    | Driver D32 异步初始化；语义对齐 D32 写入 |
| `cudaMemset2D()`             | `aclrtMemset()` (逐行)        |    ✅    | 2D 内存初始化          |
| `cudaMemset2DAsync()`        | `aclrtMemsetAsync()` (逐行)   |    ✅    | 异步 2D 初始化         |
| `cudaMemGetInfo()`           | `aclrtGetMemInfo()`           |    ✅    | 获取内存信息           |
| `cudaPointerGetAttributes()` | `aclrtPointerGetAttributes()` |    ✅    | 获取指针属性           |
| `cudaGetSymbolAddress()`     | `aclrtGetSymbolAddress()`     |    ✅    | 获取设备符号地址       |
| `cudaHostRegister()`         | `aclrtHostRegisterV2()`       |    ✅    | 注册主机内存；显式转换 CUDA flags，default 走 pinned，mapped 走 mapped+pinned |
| `cudaHostUnregister()`       | `aclrtHostUnregister()`       |    ✅    | 取消注册主机内存；已覆盖 default 和 mapped 注册后的成功路径 |
| `cudaHostGetDevicePointer()` | `aclrtHostGetDevicePointer()` |    ✅    | 获取 mapped host 注册后的设备指针 |

## 三、SOMA/UVM 不支持 API

以下接口当前不做 CUDA 到 CANN 映射，不进入转测验收；兼容层如保留同名入口，仅用于编译兼容并返回 not supported。

| CUDA API | 实现状态 | 说明 |
| -------- | :------: | ---- |
| `cudaMallocManaged()` | ❌ 不支持 | CUDA UVM，CANN Runtime 当前不对标 |
| `cudaMemAdvise()` | ❌ 不支持 | CUDA UVM advise，CANN Runtime 当前不对标 |
| `cudaMemPrefetchAsync()` | ❌ 不支持 | CUDA UVM prefetch，CANN Runtime 当前不对标 |
| `cudaMemPrefetchAsync_v2()` | ❌ 不支持 | CUDA UVM prefetch v2，CANN Runtime 当前不对标 |
| `cudaMemRangeGetAttribute()` | ❌ 不支持 | CUDA UVM range attribute，CANN Runtime 当前不对标 |
| `cudaMemRangeGetAttributes()` | ❌ 不支持 | CUDA UVM range attributes，CANN Runtime 当前不对标 |
| `cudaMemPrefetchBatchAsync()` | ❌ 不支持 | CUDA UVM batch prefetch，CANN Runtime 当前不对标 |
| `cudaMemDiscardAndPrefetchBatchAsync()` | ❌ 不支持 | CUDA UVM discard/prefetch batch，CANN Runtime 当前不对标 |
| `cudaMemDiscardBatchAsync()` | ❌ 不支持 | CUDA UVM discard batch，CANN Runtime 当前不对标 |
| `cudaMallocAsync()` | ❌ 不支持 | CUDA SOMA，CANN Runtime 当前不对标 |
| `cudaFreeAsync()` | ❌ 不支持 | CUDA SOMA，CANN Runtime 当前不对标 |
| `cudaMemPoolCreate()` | ❌ 不支持 | CUDA SOMA mempool，CANN Runtime 当前不对标 |
| `cudaMemPoolDestroy()` | ❌ 不支持 | CUDA SOMA mempool，CANN Runtime 当前不对标 |
| `cudaMemPoolSetAttribute()` | ❌ 不支持 | CUDA SOMA mempool attribute，CANN Runtime 当前不对标 |
| `cudaMemPoolsetAttribute()` | ❌ 不支持 | 同 `cudaMemPoolSetAttribute()`，CANN Runtime 当前不对标 |
| `cudaMemPoolGetAttribute()` | ❌ 不支持 | CUDA SOMA mempool attribute，CANN Runtime 当前不对标 |
| `cudaMemPoolTrimTo()` | ❌ 不支持 | CUDA SOMA mempool trim，CANN Runtime 当前不对标 |
| `cudaMemPoolSetAccess()` | ❌ 不支持 | CUDA SOMA mempool access，CANN Runtime 当前不对标 |
| `cudaMemPoolsetAccess()` | ❌ 不支持 | 同 `cudaMemPoolSetAccess()`，CANN Runtime 当前不对标 |
| `cudaMemPoolGetAccess()` | ❌ 不支持 | CUDA SOMA mempool access，CANN Runtime 当前不对标 |
| `cudaMemPoolExportPointer()` | ❌ 不支持 | CUDA SOMA mempool export，CANN Runtime 当前不对标 |
| `cudaMemPoolExportToShareableHandle()` | ❌ 不支持 | CUDA SOMA mempool export，CANN Runtime 当前不对标 |
| `cudaMemPoolImportFromShareableHandle()` | ❌ 不支持 | CUDA SOMA mempool import，CANN Runtime 当前不对标 |
| `cudaMemPoolImportPointer()` | ❌ 不支持 | CUDA SOMA mempool import，CANN Runtime 当前不对标 |
| `cudaDeviceGetDefaultMemPool()` | ❌ 不支持 | CUDA SOMA default mempool，CANN Runtime 当前不对标 |
| `cudaDeviceGetMemPool()` | ❌ 不支持 | CUDA SOMA device mempool，CANN Runtime 当前不对标 |
| `cudaDeviceSetMemPool()` | ❌ 不支持 | CUDA SOMA device mempool，CANN Runtime 当前不对标 |
| `cudaMemGetDefaultMemPool()` | ❌ 不支持 | 同 default mempool 类接口，CANN Runtime 当前不对标 |
| `cudaMemGetMemPool()` | ❌ 不支持 | 同 device mempool 类接口，CANN Runtime 当前不对标 |
| `cudaMemSetMemPool()` | ❌ 不支持 | 同 device mempool 类接口，CANN Runtime 当前不对标 |
| `cuMemDiscardAndPrefetchBatchAsync()` | ❌ 不支持 | CUDA Driver UVM discard/prefetch batch，CANN Runtime 当前不对标 |
| `cuMemDiscardBatchAsync()` | ❌ 不支持 | CUDA Driver UVM discard batch，CANN Runtime 当前不对标 |
| `cuPointerGetAttributes()` | ❌ 不支持 | CUDA Driver pointer attributes 批量查询，CANN Runtime 当前不对标 |
| `cuPointerSetAttribute()` | ❌ 不支持 | CUDA Driver pointer attribute 设置，CANN Runtime 当前不对标 |

## 四、流管理 API


| CUDA API                                | CANN API                              | 实现状态 | 说明               |
| --------------------------------------- | ------------------------------------- | :------: | ------------------ |
| `cudaStreamCreate()`                    | `aclrtCreateStream()`                 |    ✅    | 创建流             |
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
| `cudaStreamBeginCaptureToGraph()`       | `aclmdlRICaptureToModelRIBegin()`     |    ✅*   | 开始捕获到已有 Model RI；目标 CANN 接口为试验特性，后续版本可能变更，不支持应用于生产环境 |
| `cudaStreamEndCapture()`                | `aclmdlRICaptureEnd()`                |    ✅    | 结束流捕获         |
| `cudaStreamCaptureStatus`               | `aclmdlRICaptureStatus`               |    ✅    | Stream capture 状态枚举显式映射：None/Active/Invalidated |
| `cudaStreamGetCaptureInfo()`            | `aclmdlRICaptureGetInfo()`            |    ✅    | 查询 capture 状态与 graph |
| `cudaStreamGetCaptureInfo_v3()`         | `aclmdlRICaptureGetInfo()`            |    ✅    | v3 查询接口兼容到 CANN capture info |
| `cudaStreamIsCapturing()`               | `aclmdlRICaptureGetInfo()`            |    ✅    | 查询流捕获状态     |
| `cudaThreadExchangeStreamCaptureMode()` | `aclmdlRICaptureThreadExchangeMode()` |    ✅    | 交换流捕获模式     |

## 五、事件管理 API


| CUDA API                     | CANN API                       | 实现状态 | 说明               |
| ---------------------------- | ------------------------------ | :------: | ------------------ |
| `cudaEventCreate()`          | `aclrtCreateEvent()`           |    ✅    | 创建事件           |
| `cudaEventCreateWithFlags()` | `aclrtCreateEventExWithFlag()` |    ✅    | 创建事件（带标志） |
| `cudaEventDestroy()`         | `aclrtDestroyEvent()`          |    ✅    | 销毁事件           |
| `cudaEventRecord()`          | `aclrtRecordEvent()`           |    ✅    | 记录事件           |
| `cudaEventRecordWithFlags()` | `aclrtRecordEventWithFlag()` |    ✅*   | 带 flags 记录事件；本机旧头缺声明时 default flag 回退 `aclrtRecordEvent`，external flag 返回 not supported |
| `cudaEventSynchronize()`     | `aclrtSynchronizeEvent()`      |    ✅    | 事件同步           |
| `cudaEventQuery()`           | `aclrtQueryEventStatus()`      |    ✅    | 查询事件状态       |
| `cudaEventElapsedTime()`     | `aclrtEventElapsedTime()`      |    ✅    | 计算事件耗时       |

## 六、IPC API


| CUDA API                   | CANN API                                | 实现状态 | 说明              |
| -------------------------- | --------------------------------------- | :------: | ----------------- |
| `cudaIpcGetMemHandle()`    | `aclrtIpcMemGetExportKey()`            |    ✅    | 获取 IPC 内存 key |
| `cudaIpcOpenMemHandle()`   | `aclrtIpcMemImportByKey()`             |    ✅    | 通过 IPC key 导入共享内存 |
| `cudaIpcCloseMemHandle()`  | `aclrtIpcMemClose()`                   |    ✅    | 关闭 IPC 共享内存 |
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
| `cuModuleLoad()`            | `aclrtBinaryLoadFromFile()` |    ✅    | Driver module 从文件加载   |
| `cuModuleLoadData()`        | `aclrtBinaryLoadFromData()` |    ✅    | Driver module 从内存 ELF 加载 |
| `cuModuleGetFunction()`     | `aclrtBinaryGetFunction()`  |    ✅    | Driver module 获取函数句柄 |
| `cuModuleUnload()`          | `aclrtBinaryUnLoad()`       |    ✅    | Driver module 卸载         |

## 八、执行控制 API


| CUDA API               | CANN API                | 实现状态 | 说明         |
| ---------------------- | ----------------------- | :------: | ------------ |
| `cudaLaunchHostFunc()` | `aclrtLaunchHostFunc()` |    ✅    | 主机函数回调 |
| `cudaFuncGetAttributes()` | `aclrtGetFunctionAttribute()` |    ✅    | 查询 CANN function 属性并填充 CUDA 属性结构 |
| `cudaLaunchKernel()`   | `aclrtLaunchKernelWithHostArgs()` / `aclrtLaunchKernelWithArgsArray()` / `aclrtLaunchSIMTKernelWithArgsArray()` / `aclrtLaunchSIMTKernelWithHostArgs()` |    ✅    | Kernel 启动；兼容层默认使用参数数组方式下发 |

## 九、性能分析 API


| CUDA API              | CANN API                           | 实现状态 | 说明         |
| --------------------- | ---------------------------------- | :------: | ------------ |
| `cudaProfilerStart()` | `aclprofInit()` + `aclprofStart()` |    ✅    | 启动性能分析|
| `cudaProfilerStop()`  | `aclprofStop()` + `aclprofFinalize()` |    ✅    | 停止性能分析并结束 Profiling |

## 十、错误处理 API


| CUDA API                | 实现方式                 | 实现状态 | 说明           |
| ----------------------- | ------------------------ | :------: | -------------- |
| `cudaGetLastError()`    | `aclrtGetLastError()`    |    ✅    | 获取最后的错误 |
| `cudaPeekAtLastError()` | `aclrtPeekAtLastError()` |    ✅    | 查看最后的错误 |
| `cudaGetErrorName()`    | 内部实现                 |    ✅    | 获取错误名称   |
| `cudaGetErrorString()`  | `aclGetRecentErrMsg()`  |    ✅    | 获取最近一次 ACL 错误描述；必要时结合本地错误文本包装完成 CUDA 错误字符串映射 |

## 十一、版本管理 API


| CUDA API                  | CANN API                | 实现状态 | 说明         |
| ------------------------- | ----------------------- | :------: | ------------ |
| `cudaRuntimeGetVersion()` | `aclsysGetVersionNum()` |    ✅    | 获取 Runtime 版本号 |

## 十二、Graph 管理 API


| CUDA API                   | CANN API                   | 实现状态 | 说明                |
| -------------------------- | -------------------------- | :------: | ------------------- |
| `cudaGraphDebugDotPrint()` | `aclmdlRIDebugJsonPrint()` |    ✅    | 导出 Graph/RI 调试信息 |
| `cudaGraphExecDestroy()`   | `aclmdlRIDestroy()`        |    ✅    | 销毁 Graph 执行实例 |
| `cudaGraphConditionalHandleCreate()` | `aclmdlRICondHandleCreate()` |    ✅*   | 创建条件 Graph handle；目标 CANN 接口为试验特性，后续版本可能变更，不支持应用于生产环境 |
| `cudaGraphAddNode()` (`cudaGraphNodeTypeConditional`) | `aclmdlRIAddCondTask()` |    ✅*   | 仅支持 conditional node 特例；`cudaGraphAddNode` 承载多种 task 类型，非 conditional node 不按此映射 |
| `cudaGraphGetNodes()`      | `aclmdlRIGetStreams()` + `aclmdlRIGetTasksByStream()` |    ✅*   | 通过 RI stream/task 汇总节点；目标 CANN 接口为试验特性，后续版本可能变更，不支持应用于生产环境 |
| `cudaGraphLaunch()`        | `aclmdlRIExecuteAsync()`   |    ✅    | 异步执行 Graph/RI |
| `cudaGraphSetConditional()` | `aclmdlRICondHandleGetCondPtr()` |    ✅*   | 设备侧设置条件值；CANN 侧通过条件 handle 取得设备条件指针后写入，`aclmdlRICondHandleGetCondPtr` 为试验特性，后续版本可能变更，不支持应用于生产环境 |

## 十三、CUDA Driver VMM API


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
| `cuCtxGetCurrent()`                | `aclrtGetCurrentContext()`              |    ✅    | 获取当前 Context |
| `cuCtxSetCurrent()`                | `aclrtSetCurrentContext()`              |    ✅    | 设置当前 Context |
| `cuDevicePrimaryCtxGetState()`     | `aclrtGetPrimaryCtxState()`             |    ✅    | 查询 primary context 状态 |
| `cuStreamWriteValue32()`           | `aclrtValueWrite()`                     |    ✅    | 在 stream 上写入 32-bit value |


## 不支持 API 替代方案建议


| 不支持 API             | 替代方案                             |
| ---------------------- | ------------------------------------ |
| SOMA/UVM API           | 当前不做映射、不进入转测；兼容层返回 not supported |
| Texture/Surface API    | 使用普通内存 + 自定义纹理逻辑        |

## 其他不支持 API


| 不支持 API | 类型 | 说明 |
| ---------- | ---- | ---- |
| `cudaFuncSetAttribute()` | CUDA Runtime | CANN Runtime 暂不支持，兼容层返回 `cudaErrorNotSupported` |
| `cudaLaunchCooperativeKernel()` | CUDA Runtime | CANN Runtime 暂不支持 cooperative launch，兼容层返回 `cudaErrorNotSupported` |
| `cuFuncSetCacheConfig()` | CUDA Driver | CANN Runtime 暂不支持，兼容层返回 `CUDA_ERROR_NOT_SUPPORTED` |
| `cudaGetDriverEntryPoint()` | CUDA Runtime | CANN Runtime 暂不支持 CUDA Driver entry point 查询，兼容层返回 `cudaErrorNotSupported` |
| `cudaGetDriverEntryPointByVersion()` | CUDA Runtime | CANN Runtime 暂不支持 CUDA Driver entry point 查询，兼容层返回 `cudaErrorNotSupported` |
| `cudaGraphAddNode()` | CUDA Runtime Graph | CANN Runtime 暂不支持手工添加通用 Graph Node；仅 `cudaGraphNodeTypeConditional` 特例可映射到 `aclmdlRIAddCondTask()`，其他类型兼容层返回 `cudaErrorNotSupported` |
| `cudaGraphAddNode_v2()` | CUDA Runtime Graph | CANN Runtime 暂不支持手工添加通用 Graph Node，兼容层返回 `cudaErrorNotSupported` |
| `cudaGraphDestroy()` | CUDA Runtime Graph | CANN Runtime 暂不支持 CUDA Graph 对象销毁语义，兼容层返回 `cudaErrorNotSupported` |
| `cudaGraphInstantiateWithFlags()` | CUDA Runtime Graph | CANN Runtime 暂不支持带 flags 的 Graph 实例化，兼容层返回 `cudaErrorNotSupported` |
| `cudaGraphNodeGetDependencies()` | CUDA Runtime Graph | CANN Runtime 暂不支持查询 CUDA Graph Node 依赖，兼容层返回 `cudaErrorNotSupported` |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor()` | CUDA Runtime Occupancy | 架构差异不支持 occupancy 查询，兼容层返回 `cudaErrorNotSupported` |
| `cudaOccupancyMaxPotentialBlockSize()` | CUDA Runtime Occupancy | 架构差异不支持 occupancy 查询，兼容层返回 `cudaErrorNotSupported` |
| `cudaStreamGetCaptureInfo_v2()` | CUDA Runtime Stream Capture | CANN Runtime 暂不支持 CUDA v2 capture info 语义，兼容层返回 `cudaErrorNotSupported` |
| `cudaStreamUpdateCaptureDependencies()` | CUDA Runtime Stream Capture | CANN Runtime 暂不支持更新 CUDA capture dependencies，兼容层返回 `cudaErrorNotSupported` |
| `cudaStreamUpdateCaptureDependencies_v2()` | CUDA Runtime Stream Capture | CANN Runtime 暂不支持更新 CUDA capture dependencies，兼容层返回 `cudaErrorNotSupported` |
| `cuCtxPopCurrent()` | CUDA Driver Context | CANN Runtime 暂不支持 Driver context stack 语义，兼容层返回 `CUDA_ERROR_NOT_SUPPORTED` |
| `cuCtxPushCurrent()` | CUDA Driver Context | CANN Runtime 暂不支持 Driver context stack 语义，兼容层返回 `CUDA_ERROR_NOT_SUPPORTED` |
| `cuDevicePrimaryCtxRetain()` | CUDA Driver Context | CANN Runtime 暂不支持 primary context retain 语义，兼容层返回 `CUDA_ERROR_NOT_SUPPORTED` |
| `cuGreenCtxCreate()` | CUDA Driver Green Context | CANN Runtime 暂不支持 Green Context，兼容层返回 `CUDA_ERROR_NOT_SUPPORTED` |
| `cuGreenCtxDestroy()` | CUDA Driver Green Context | CANN Runtime 暂不支持 Green Context，兼容层返回 `CUDA_ERROR_NOT_SUPPORTED` |
| `cuCtxFromGreenCtx()` | CUDA Driver Green Context | CANN Runtime 暂不支持 Green Context，兼容层返回 `CUDA_ERROR_NOT_SUPPORTED` |
| `cuDeviceGetDevResource()` | CUDA Driver Green Context | CANN Runtime 暂不支持 DevResource 查询，兼容层返回 `CUDA_ERROR_NOT_SUPPORTED` |
| `cuGreenCtxStreamCreate()` | CUDA Driver Green Context | CANN Runtime 暂不支持 Green Context Stream，兼容层返回 `CUDA_ERROR_NOT_SUPPORTED` |
| `cuModuleLoadDataEx()` | CUDA Driver Module/JIT | CANN Runtime 暂不支持 Driver JIT module load，兼容层返回 `CUDA_ERROR_NOT_SUPPORTED` |
| `cuLinkAddData()` | CUDA Driver Link/JIT | CANN Runtime 暂不支持 Driver JIT link data，兼容层返回 `CUDA_ERROR_NOT_SUPPORTED` |
| `cuMulticastAddDevice()` | CUDA Driver Multicast | CANN Runtime 暂不支持 Multicast，兼容层返回 `CUDA_ERROR_NOT_SUPPORTED` |
| `cuMulticastBindMem()` | CUDA Driver Multicast | CANN Runtime 暂不支持 Multicast，兼容层返回 `CUDA_ERROR_NOT_SUPPORTED` |
| `cuMulticastCreate()` | CUDA Driver Multicast | CANN Runtime 暂不支持 Multicast，兼容层返回 `CUDA_ERROR_NOT_SUPPORTED` |
| `cuMulticastUnbind()` | CUDA Driver Multicast | CANN Runtime 暂不支持 Multicast，兼容层返回 `CUDA_ERROR_NOT_SUPPORTED` |
| `cuTensorMapEncodeTiled()` | CUDA Driver Tensor Map | CANN Runtime 暂不支持 Tensor Map 编码，兼容层返回 `CUDA_ERROR_NOT_SUPPORTED` |
