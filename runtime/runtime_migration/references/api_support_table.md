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
| `cudaDeviceSetLimit()`               | `aclrtDeviceSetLimit()`               |    ✅    | 设置设备限制；当前映射 stack size 和 printf FIFO，其它 CUDA limit 返回 `cudaErrorUnsupportedLimit` |
| `cudaDeviceGetLimit()`               | `aclrtDeviceGetLimit()`               |    ✅    | 获取设备限制；当前映射 stack size 和 printf FIFO，其它 CUDA limit 返回 `cudaErrorUnsupportedLimit` |
| `cudaDeviceGetHostAtomicCapabilities()` | `aclrtDeviceGetHostAtomicCapabilities()` |    ✅    | 查询 Host atomic 能力；入参空指针、count 为 0、非法 operation 返回 `cudaErrorInvalidValue` |
| `cudaDeviceGetP2PAtomicCapabilities()` | `aclrtDeviceGetP2PAtomicCapabilities()` |    ✅    | 查询 P2P atomic 能力；同设备 src/dst 预校验返回 `cudaErrorInvalidDevice`，跨设备成功路径受设备数量和拓扑限制 |
| `cudaDeviceGetCacheConfig()`         | Mock 实现                             |    ✅    | 获取缓存配置                      |
| `cudaDeviceSetCacheConfig()`         | 内部存储                              |    ✅    | 设置缓存配置                      |
| `cudaDeviceGetStreamPriorityRange()` | `aclrtDeviceGetStreamPriorityRange()` |    ✅    | 获取流优先级范围                  |
| `cudaDeviceEnablePeerAccess()`       | `aclrtDeviceEnablePeerAccess()`       |    ✅    | 启用当前设备到 peer 设备的单向访问；flags 非 0 返回 `cudaErrorInvalidValue`，CANN 支持范围受产品和拓扑限制 |
| `cudaDeviceDisablePeerAccess()`      | `aclrtDeviceDisablePeerAccess()`      |    ✅    | 禁用点对点访问                    |
| `cudaDeviceCanAccessPeer()`          | `aclrtDeviceCanAccessPeer()`          |    ✅    | 检查点对点访问能力；空输出指针返回 `cudaErrorInvalidValue`，CANN 支持范围受产品和拓扑限制 |

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
| `cudaMemcpy3D()`             | `aclrtMemcpy2d()` (逐 slice)  |   ✅*   | 支持 linear pitched pointer；CUDA array/texture 3D 返回不支持 |
| `cudaMemcpy3DAsync()`        | `aclrtMemcpy2dAsync()` (逐 slice) | ✅* | 支持 linear pitched pointer；异步任务下发到同一 stream |
| `cudaMemcpy3DPeer()`         | `aclrtMemcpy2d()` (逐 slice, D2D) | ✅* | 支持 linear pitched pointer；跨 Device 成功路径受 CANN P2P/拓扑限制 |
| `cudaMemcpy3DPeerAsync()`    | `aclrtMemcpy2dAsync()` (逐 slice, D2D) | ✅* | 支持 linear pitched pointer；跨 Device 成功路径受 CANN P2P/拓扑限制 |
| `cudaMemcpyPeer()`           | `aclrtMemcpy()` (`ACL_MEMCPY_DEVICE_TO_DEVICE`) | ✅* | 同 Device D2D 已验证；跨 Device 受 CANN P2P/拓扑限制 |
| `cudaMemcpyPeerAsync()`      | `aclrtMemcpyAsync()` (`ACL_MEMCPY_DEVICE_TO_DEVICE`) |    ✅    | 异步点对点拷贝；当前仅支持同一个 PCIe Switch 内 Device 之间的内存复制 |
| `cudaMemcpyToSymbol()`       | `aclrtMemcpyToSymbol()`       |    ✅    | 向设备符号地址拷贝数据 |
| `cudaMemcpyToSymbolAsync()`  | `aclrtMemcpyToSymbolAsync()`  |   ✅*   | 异步向设备符号地址拷贝；C++ host fallback symbol 立即完成 |
| `cudaMemcpyFromSymbol()`     | `aclrtMemcpyFromSymbol()`     |    ✅    | 从设备符号地址拷贝数据 |
| `cudaMemcpyFromSymbolAsync()` | `aclrtMemcpyFromSymbolAsync()` | ✅*   | 异步从设备符号地址拷贝；C++ host fallback symbol 立即完成 |
| `cudaMemcpyBatchAsync()`     | `aclrtMemcpyBatchAsyncV2()` / 逐项 `aclrtMemcpyAsync()` fallback | ✅* | CUDA Runtime 13.2+；CANN 9.1.0+，根据 pointer attributes/location hint 推导 copy kind；batch 原生接口不可用时逐项 fallback |
| `cudaMemcpyWithAttributesAsync()` | `aclrtMemcpyAsync()`      |   ✅*   | 单段异步拷贝；当前忽略 CUDA access-order/location hint，仅使用 CANN default copy kind |
| `cudaMemcpy3DBatchAsync()`   | `aclrtMemcpy2dAsync()` (逐 op/逐 slice) | ✅* | 支持 pointer operand；CUDA array operand 返回不支持 |
| `cudaMemcpy3DWithAttributesAsync()` | `aclrtMemcpy2dAsync()` (单 op/逐 slice) | ✅* | 官方 batch op 形态；支持 pointer operand，flags 仅支持 0 |
| `cudaMemset()`               | `aclrtMemset()`               |    ✅    | 内存初始化             |
| `cudaMemsetAsync()`          | `aclrtMemsetAsync()`          |    ✅    | 异步内存初始化         |
| `cuMemsetD32Async()`         | `aclrtMemsetD32Async()`       |    ✅    | Driver D32 异步初始化；语义对齐 D32 写入 |
| `cudaMemset2D()`             | `aclrtMemset()` (逐行)        |    ✅    | 2D 内存初始化          |
| `cudaMemset2DAsync()`        | `aclrtMemsetAsync()` (逐行)   |    ✅    | 异步 2D 初始化         |
| `cudaMemset3D()`             | `aclrtMemset()` (逐 slice/逐行) | ✅* | 支持 linear pitched pointer |
| `cudaMemset3DAsync()`        | `aclrtMemsetAsync()` (逐 slice/逐行) | ✅* | 支持 linear pitched pointer；异步任务下发到同一 stream |
| `cudaMemGetInfo()`           | `aclrtGetMemInfo()`           |    ✅    | 获取内存信息           |
| `cudaPointerGetAttributes()` | `aclrtPointerGetAttributes()` |    ✅    | 获取指针属性           |
| `cudaGetSymbolAddress()`     | `aclrtGetSymbolAddress()`     |    ✅    | 获取设备符号地址       |
| `cudaGetSymbolSize()`        | `aclrtGetSymbolSize()`        |   ✅*   | 获取设备符号大小；C++ host fallback symbol 返回兼容层 backing size |
| `cudaHostRegister()`         | `aclrtHostRegisterV2()`       |    ✅    | 注册主机内存；显式转换 CUDA flags，default 走 pinned，mapped 走 mapped+pinned |
| `cudaHostUnregister()`       | `aclrtHostUnregister()`       |    ✅    | 取消注册主机内存；已覆盖 default 和 mapped 注册后的成功路径 |
| `cudaHostGetDevicePointer()` | `aclrtHostGetDevicePointer()` |    ✅    | 获取 mapped host 注册后的设备指针 |
| `cudaHostGetFlags()`         | `aclrtPointerGetAttributes()` |   ✅*   | 查询 Host 注册/分配指针并返回兼容 flags；CANN 不暴露原始 CUDA flags，当前成功路径返回 `cudaHostAllocDefault` |
| `cudaMallocManaged()`        | `aclrtMemAllocManaged()`      |   ✅*   | UVM 统一内存分配；仅 `cudaMemAttachGlobal`，受 CANN 产品 UVM 能力限制 |
| `cudaMemAdvise()`            | `aclrtMemManagedAdvise()`     |   ✅*   | UVM advise；显式转换 advice 与 device/location，受 CANN 产品 UVM 能力限制 |
| `cudaMemPrefetchAsync()`     | `aclrtMemManagedPrefetchAsync()` | ✅* | C++ device-id 重载和 v2 location 形态均委托到 CANN UVM prefetch；flags 仅支持 0 |
| `cudaMemPrefetchAsync_v2()`  | `aclrtMemManagedPrefetchAsync()` | ✅* | UVM prefetch v2；flags 仅支持 0，异步完成需同步 stream |
| `cudaMemRangeGetAttribute()` | `aclrtMemManagedGetAttr()`    |   ✅*   | UVM range attribute 查询；支持 read-mostly/preferred-location/accessed-by/last-prefetch-location 基础枚举映射 |
| `cudaMemRangeGetAttributes()` | `aclrtMemManagedGetAttrs()`  |   ✅*   | UVM range attributes 批量查询；支持范围同单属性查询 |
| `cudaMemPrefetchBatchAsync()` | `aclrtMemManagedPrefetchBatchAsync()` | ✅* | UVM 批量 prefetch；flags 仅支持 0，受 CANN 产品 UVM 能力限制 |

## 三、SOMA/UVM 不支持 API

以下接口当前不做 CUDA 到 CANN 映射，不进入转测验收；兼容层如保留同名入口，仅用于编译兼容并返回 not supported。已在上一节列为 `✅*` 的 UVM 基础接口按 CANN 产品能力条件支持，不属于本不支持清单。

| CUDA API | 实现状态 | 说明 |
| -------- | :------: | ---- |
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
| `cudaStreamSetAttribute()`              | `aclrtSetStreamAttribute()`           |    ✅*   | 设置 CANN stream 属性；`cudaStreamAttrID` / `cudaStreamAttrValue` 在兼容层中按 CANN `aclrtStreamAttr` / `aclrtStreamAttrValue` 直通 |
| `cudaStreamGetAttribute()`              | `aclrtGetStreamAttribute()`           |    ✅*   | 获取 CANN stream 属性；属性枚举语义以 CANN Runtime 为准 |
| `cudaStreamBeginCapture()`              | `aclmdlRICaptureBegin()`              |    ✅*   | 开始流捕获；目标 CANN Model RI 接口为试验特性，非法 capture mode 先返回 `cudaErrorInvalidValue` |
| `cudaStreamBeginCaptureToGraph()`       | `aclmdlRICaptureToModelRIBegin()`     |    ✅*   | 开始捕获到已有 Model RI；目标 CANN 接口为试验特性，后续版本可能变更，不支持应用于生产环境 |
| `cudaStreamEndCapture()`                | `aclmdlRICaptureEnd()`                |    ✅*   | 结束流捕获；目标 CANN Model RI 接口为试验特性 |
| `cudaStreamCaptureStatus`               | `aclmdlRICaptureStatus`               |    ✅    | Stream capture 状态枚举显式映射：None/Active/Invalidated |
| `cudaStreamGetCaptureInfo()`            | `aclmdlRICaptureGetInfo()`            |    ✅    | 查询 capture 状态与 graph；兼容层按参数个数适配 v2/v3 签名 |
| `cudaStreamGetCaptureInfo_v3()`         | `aclmdlRICaptureGetInfo()`            |    ✅    | v3 查询接口兼容到 CANN capture info |
| `cudaStreamIsCapturing()`               | `aclmdlRICaptureGetInfo()`            |    ✅*   | 查询流捕获状态；目标 CANN Model RI 接口为试验特性 |
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
| `cudaIpcGetMemHandle()`    | `aclrtIpcMemGetExportKey()`            |    ✅    | 获取 IPC 内存 key；CANN 侧验证需使用页对齐共享内存范围，部分产品可能返回不支持 |
| `cudaIpcOpenMemHandle()`   | `aclrtIpcMemImportByKey()`             |    ✅    | 通过 IPC key 导入共享内存；验收应使用 exec 子进程，避免 fork 后继承 CUDA/CANN 上下文 |
| `cudaIpcCloseMemHandle()`  | `aclrtIpcMemClose()`                   |    ✅    | 关闭 IPC 共享内存；导入进程先 close，导出进程后释放内存 |
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
| `cudaLaunchHostFunc()` | `aclrtLaunchHostFunc()` |    ✅    | 主机函数回调；空回调函数按 CUDA baseline 作为 no-op 返回 `cudaSuccess` |
| `cudaLaunchHostFunc_v2()` | `aclrtLaunchHostFunc()` |    ✅*   | v2 syncMode 当前按 CANN host callback 能力忽略；空回调函数作为 no-op 返回 `cudaSuccess` |
| `cudaFuncGetAttributes()` | `aclrtGetFunctionAttribute()` |    ✅    | 查询 CANN function 属性并填充 CUDA 属性结构 |
| `cudaFuncGetName()` | `aclrtGetFunctionName()` |    ✅*   | 查询 CANN function name；兼容层返回内部静态缓冲区指针 |
| `cudaFuncGetParamCount()` | `aclrtFunctionGetParamCount()` |    ✅*   | 查询 CANN function 参数个数 |
| `cudaFuncGetParamInfo()` | `aclrtFunctionGetParamInfo()` |    ✅*   | 查询 CANN function 参数 offset/size；两个输出指针不能同时为空 |
| `cudaLaunchKernel()`   | `aclrtLaunchKernelWithHostArgs()` / `aclrtLaunchKernelWithArgsArray()` / `aclrtLaunchSIMTKernelWithArgsArray()` / `aclrtLaunchSIMTKernelWithHostArgs()` |    ✅*   | Kernel 启动；兼容层默认使用参数数组方式下发，仅适用于 CANN 可识别的 kernel/function handle；CUDA `<<<>>>` kernel 迁移按 Host fallback 处理 |
| `cudaLaunchKernelEx()` | `aclrtLaunchKernelWithHostArgs()` |    ✅*   | 通过 `aclrtFunctionGetParamCount` / `aclrtFunctionGetParamInfo` 将 `void **args` 打包为 HostArgs 后下发；仅适用于 CANN 可识别的 function handle |

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
| `cudaGetErrorName()`    | 内部错误表 / `aclGetRecentErrMsg()` fallback |    ✅    | 已知 CUDA 错误码返回 CUDA 错误名称；未知错误码尝试返回最近一次 ACL 错误描述，否则返回 `cudaErrorUnknown` |
| `cudaGetErrorString()`  | `aclGetRecentErrMsg()`  |    ✅    | 获取最近一次 ACL 错误描述；必要时结合本地错误文本包装完成 CUDA 错误字符串映射 |

## 十一、版本管理 API


| CUDA API                  | CANN API                | 实现状态 | 说明         |
| ------------------------- | ----------------------- | :------: | ------------ |
| `cudaRuntimeGetVersion()` | `aclsysGetVersionNum()` |    ✅    | 获取 Runtime 版本号 |

## 十二、Graph 管理 API


| CUDA API                   | CANN API                   | 实现状态 | 说明                |
| -------------------------- | -------------------------- | :------: | ------------------- |
| `cudaGraphDebugDotPrint()` | `aclmdlRIDebugJsonPrint()` |    ✅*   | 导出 Graph/RI JSON 调试信息；目标 CANN Model RI 接口为试验特性 |
| `cudaGraphExecDestroy()`   | `aclmdlRIDestroy()`        |    ✅*   | 销毁 Graph 执行实例；目标 CANN Model RI 接口为试验特性 |
| `cudaGraphConditionalHandleCreate()` | `aclmdlRICondHandleCreate()` |    ✅*   | 创建条件 Graph handle；目标 CANN 接口为试验特性，后续版本可能变更，不支持应用于生产环境 |
| `cudaGraphAddNode()` (`cudaGraphNodeTypeConditional`) | `aclmdlRIAddCondTask()` |    ✅*   | 仅支持 conditional node 特例；`cudaGraphAddNode` 承载多种 task 类型，非 conditional node 不按此映射 |
| `cudaGraphGetNodes()`      | `aclmdlRIGetStreams()` + `aclmdlRIGetTasksByStream()` |    ✅*   | 通过 RI stream/task 汇总节点；目标 CANN 接口为试验特性，后续版本可能变更，不支持应用于生产环境 |
| `cudaGraphLaunch()`        | `aclmdlRIExecuteAsync()`   |    ✅*   | 异步执行 Graph/RI；目标 CANN Model RI 接口为试验特性 |
| `cudaGraphSetConditional()` | `aclmdlRICondHandleGetCondPtr()` |    ✅*   | 设备侧设置条件值；CANN 侧通过条件 handle 取得设备条件指针后写入，`aclmdlRICondHandleGetCondPtr` 为试验特性，后续版本可能变更，不支持应用于生产环境 |

## 十三、CUDA Driver VMM API


| CUDA Driver API                    | CANN API                                | 实现状态 | 说明             |
| ---------------------------------- | --------------------------------------- | :------: | ---------------- |
| `cuMemAddressReserve()`            | `aclrtReserveMemAddress()`              |    ✅    | 预留虚拟地址范围 |
| `cuMemAddressFree()`               | `aclrtReleaseMemAddress()`              |    ✅    | 释放虚拟地址范围 |
| `cuMemCreate()`                    | `aclrtMallocPhysical()`                 |    ✅    | 创建物理内存     |
| `cuMemRelease()`                   | `aclrtFreePhysical()`                   |    ✅    | 释放物理内存     |
| `cuMemExportToShareableHandle()`   | `aclrtMemExportToShareableHandleV2()`   |    ✅    | 导出可共享句柄；POSIX FD 路径需在 CUDA `int fd` 与 CANN `uint64_t` 句柄表示之间转换 |
| `cuMemGetAccess()`                 | `aclrtMemGetAccess()`                   |    ✅    | 获取访问权限     |
| `cuMemSetAccess()`                 | `aclrtMemSetAccess()`                   |    ✅*   | 设置访问权限；产品不支持或返回未映射 VMM access 错误时按 `CUDA_ERROR_NOT_SUPPORTED` 条件跳过 |
| `cuMemGetAllocationGranularity()`  | `aclrtMemGetAllocationGranularity()`    |    ✅    | 获取分配粒度     |
| `cuMemImportFromShareableHandle()` | `aclrtMemImportFromShareableHandleV2()` |    ✅    | 导入可共享句柄；CANN V2 不支持同进程 export/import 组合，验收应使用 exec 子进程 |
| `cuMemMap()`                       | `aclrtMapMem()`                         |    ✅    | 映射物理内存     |
| `cuMemUnmap()`                     | `aclrtUnmapMem()`                       |    ✅    | 取消映射         |
| `cuMemRetainAllocationHandle()`    | `aclrtMemRetainAllocationHandle()`      |    ✅    | 保留分配句柄     |
| `cuCtxGetCurrent()`                | `aclrtGetCurrentContext()`              |    ✅    | 获取当前 Context |
| `cuCtxSetCurrent()`                | `aclrtSetCurrentContext()`              |    ✅    | 设置当前 Context |
| `cuDevicePrimaryCtxGetState()`     | `aclrtGetPrimaryCtxState()`             |    ✅    | 查询 primary context 状态 |
| `cuStreamWriteValue32()`           | `aclrtValueWrite()`                     |    ✅    | 在 stream 上写入 32-bit value |
| `cudaOccupancyAvailableDynamicSMemPerBlock()` | `aclrtFunctionGetAvailDynUbufPerBlock()` | ✅* | 查询 CANN function 每 block 可用动态 UBuf；CUDA numBlocks/blockSize 仅做合法性校验 |


## 不支持 API 替代方案建议


| 不支持 API             | 替代方案                             |
| ---------------------- | ------------------------------------ |
| SOMA/扩展 UVM API      | CUDA SOMA 和 discard batch 类 UVM 扩展当前不做映射；Managed Memory 基础接口见内存管理 API 条件支持项 |
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
