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
| `cudaDeviceGetGraphMemAttribute()`   | `aclrtGetMemUsageInfo()`              |   ✅*   | 返回设备组件内存使用总量作为 Graph 内存近似值，并非 graph-only 统计；current/high 分别取当前/峰值总量，目标 API 在 Ascend 950 系列不支持 |

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
| `cudaMemcpyPeer()`           | `aclrtMemcpy()` (`ACL_MEMCPY_DEVICE_TO_DEVICE`) | ✅* | 校验设备编号及可查询的指针设备归属；同 Device D2D 已验证，跨 Device 受 CANN P2P/拓扑限制 |
| `cudaMemcpyPeerAsync()`      | `aclrtMemcpyAsync()` (`ACL_MEMCPY_DEVICE_TO_DEVICE`) |    ✅    | 异步点对点拷贝；当前仅支持同一个 PCIe Switch 内 Device 之间的内存复制 |
| `cudaMemcpyToSymbol()`       | `aclrtMemcpyToSymbol()`       |    ✅    | 向设备符号地址拷贝数据 |
| `cudaMemcpyToSymbolAsync()`  | `aclrtMemcpyToSymbolAsync()`  |   ✅*   | 异步向设备符号地址拷贝；C++ host fallback symbol 立即完成 |
| `cudaMemcpyFromSymbol()`     | `aclrtMemcpyFromSymbol()`     |    ✅    | 从设备符号地址拷贝数据 |
| `cudaMemcpyFromSymbolAsync()` | `aclrtMemcpyFromSymbolAsync()` | ✅*   | 异步从设备符号地址拷贝；C++ host fallback symbol 立即完成 |
| `cudaMemcpyBatchAsync()`     | `aclrtMemcpyBatchAsync()` / 逐项 `aclrtMemcpyAsync()` fallback | ✅* | 同方向 H2D/D2H 优先原生 batch，D2D/混合方向逐项回退；H2H 不支持；legacy NULL stream 返回参数错误；`DuringApiCall` 同步对应 stream；固定位置指针不受 location hint 覆盖 |
| `cudaMemcpyWithAttributesAsync()` | `aclrtMemcpyAsync()` (H2D/D2H/D2D) | ✅* | 根据指针位置选择方向；H2H 不支持；legacy NULL stream 返回参数错误；`DuringApiCall` 在返回前同步 stream，overlap flag 作为提示 |
| `cudaMemcpy3DBatchAsync()`   | `aclrtMemcpy2dAsync()` (逐 op/逐 slice) | ✅* | 支持 pointer operand；预校验每项 access-order/flags，legacy NULL stream 返回参数错误，`DuringApiCall` 同步 stream；CUDA array operand 返回不支持 |
| `cudaMemcpy3DWithAttributesAsync()` | `aclrtMemcpyBatchAsync()` / 逐项异步回退 (逐 row 展平) | ✅* | 支持 pointer operand；顶层 flags 仅支持 0；legacy NULL stream 返回参数错误；D2D 走逐项回退，H2H 与 array operand 不支持 |
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
| `cudaGetSymbolSize()`        | `aclrtGetSymbolSize()` / C++ `sizeof(T)` |   ✅*   | 真实设备符号走 CANN 查询；C++ host fallback 返回声明大小，复制不足或查询早于复制时亦相同 |
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
| `cudaLibraryUnload()`       | `aclrtBinaryUnLoad()`       |    ✅    | 卸载库                     |
| `cudaLibraryGetKernel()`    | `aclrtBinaryGetFunction()`  |    ✅    | 获取 kernel/function 句柄；保留旧 `cudaLibraryGetFunction` 别名以兼容既有迁移产物 |
| `cudaLibraryGetKernelCount()` | `aclrtBinaryGetFunctionCount()` | ✅* | 查询 kernel 数量；旧版 CANN 缺少符号时返回 `cudaErrorNotSupported` |
| `cudaLibraryEnumerateKernels()` | `aclrtBinaryEnumerateFunctions()` | ✅* | 枚举 kernel/function handle；旧版 CANN 缺少符号时返回 `cudaErrorNotSupported` |
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
| `cudaGetFuncBySymbol()` | `aclrtGetFuncBySymbol()` |   ✅*   | 由编译器注册的 device entry symbol 获取 CANN function handle；Ascend 910B3 / CANN 9.2.0 上混合编译的真实 Ascend C kernel 已验证成功，直接加载二进制文件不提供该 symbol；运行库缺少接口符号时返回 `cudaErrorNotSupported` |
| `cudaFuncGetAttributes()` | 无等价 CANN 属性集 | ❌ | CANN kernel type/ratio/scheduling 与 CUDA binaryVersion/maxThreadsPerBlock/cacheModeCA 不等价；真实 handle 返回 `cudaErrorNotSupported`，不得填充误导值 |
| `cudaFuncGetName()` | `aclrtGetFuncBySymbol()` + `aclrtGetFunctionName()` |    ✅*   | 先解析 symbol 或使用真实 CANN function handle；兼容层返回内部静态缓冲区指针 |
| `cudaFuncGetParamCount()` | `aclrtFunctionGetParamCount()` |    ✅*   | 查询 CANN function 参数个数 |
| `cudaFuncGetParamInfo()` | `aclrtFunctionGetParamInfo()` |    ✅*   | 查询 CANN function 参数 offset/size；两个输出指针不能同时为空 |
| `cudaLaunchKernel()`   | `aclrtLaunchKernelWithHostArgs()` / `aclrtLaunchKernelWithArgsArray()` / `aclrtLaunchSIMTKernelWithArgsArray()` / `aclrtLaunchSIMTKernelWithHostArgs()` |    ✅*   | Kernel 启动；兼容层默认使用参数数组方式下发，仅适用于 CANN 可识别的 kernel/function handle；CUDA `<<<>>>` kernel 迁移按 Host fallback 处理 |
| `cudaLaunchKernelEx()` | `aclrtLaunchKernelWithHostArgs()` |    ✅*   | 通过 `aclrtFunctionGetParamCount` / `aclrtFunctionGetParamInfo` 将 `void **args` 打包为 HostArgs 后下发；仅适用于 CANN 可识别的 function handle |

## 九、性能分析 API


| CUDA API              | CANN API                           | 实现状态 | 说明         |
| --------------------- | ---------------------------------- | :------: | ------------ |
| `cudaProfilerStart()` | `aclprofInit()` + `aclprofStart()` |    ✅    | 启动性能分析；结果目录读取 `CUDA_COMPAT_PROFILER_RESULT_PATH`，未设置时使用 `/tmp/cuda_compat_profiling` |
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
| `cudaGraphConditionalHandleCreate()` | `aclmdlRICondHandleCreate()` |    ✅*   | 创建条件 Graph handle；要求目标 CANN 支持条件句柄，Ascend 910B3 / CANN 9.2.0 上原生 capture 阶段调用也返回不支持；目标接口为试验特性 |
| `cudaGraphConditionalHandleCreate_v2()` | `aclmdlRICondHandleCreate()` | ✅* | `ctx` 为空或等于当前 CANN context，且目标 CANN 支持条件句柄时可用；Ascend 910B3 / CANN 9.2.0 原生 capture 阶段返回不支持 |
| `cudaGraphGetNodes()`      | `aclmdlRIGetStreams()` + `aclmdlRIGetTasksByStream()` |    ✅*   | 通过 RI stream/task 汇总节点；目标 CANN 接口为试验特性，后续版本可能变更，不支持应用于生产环境 |
| `cudaGraphLaunch()`        | `aclmdlRIExecuteAsync()`   |    ✅*   | 异步执行 Graph/RI；目标 CANN Model RI 接口为试验特性 |
| `cudaGraphSetConditional()` | `aclmdlRICondHandleGetCondPtr()` |    ✅*   | 设备侧设置条件值；CANN 侧通过条件 handle 取得设备条件指针后写入，`aclmdlRICondHandleGetCondPtr` 为试验特性，后续版本可能变更，不支持应用于生产环境 |
| `cudaGraphExecEventRecordNodeSetEvent()` | `aclmdlRITaskSetParams()` + `aclmdlRIUpdate()` | ✅* | 仅适用于目标 CANN 接受 EVENT_RECORD 更新的 AclGraph；CANN 9.2 实测拒绝该 task type，成功路径不可用 |
| `cudaGraphExecEventWaitNodeSetEvent()` | `aclmdlRITaskSetParams()` + `aclmdlRIUpdate()` | ✅* | 仅适用于目标 CANN 接受 EVENT_WAIT 更新的 AclGraph；CANN 9.2 实测拒绝该 task type，成功路径不可用 |
| `cudaGraphExecKernelNodeSetParams()` | `aclmdlRITaskSetParams()` + `aclmdlRIUpdate()` | ✅* | 支持真实捕获 kernel task 的 grid block 数更新；function 必须是可解析的 device entry symbol，或与已捕获 task 相同的真实 CANN handle；Ascend 910B3 / CANN 9.2.0 上已验证更新后 Graph 执行与输出数据；CUDA 参数数组、`extra`、动态 shared memory 更新返回不支持 |
| `cudaGraphExecNodeSetParams()` | `aclmdlRITaskSetParams()` + `aclmdlRIUpdate()` | ✅* | 真实 captured kernel task 成功路径已在 CANN 9.2/910B3 验证；event-record/event-wait task 更新成功路径不可用 |
| `cudaGraphNodeSetParams()` | `aclmdlRITaskSetParams()` | ✅* | 真实 captured kernel task 成功路径已在 CANN 9.2/910B3 验证；event task 返回不支持/无效参数 |
| `cudaGraphExecUpdate()` | `aclmdlRIUpdate()` | ✅* | 仅支持兼容层中 graph 与 graphExec 为同一 RI 的参数提交，不支持 CUDA 跨 graph 拓扑匹配 |
| `cudaGraphNodeGetType()` | `aclmdlRITaskGetType()` | ✅* | 映射 kernel/event/default/value task；CANN event-reset task 无安全 CUDA 类型映射时返回不支持 |
| `cudaGraphNodeSetEnabled()` | `aclmdlRITaskDisable()` + `aclmdlRIUpdate()` | ✅* | 仅支持 disable；CANN 无 re-enable 对等接口，`isEnabled=1` 返回不支持 |
| `cudaGraphExecGetId()` | `aclmdlRIGetId()` | ✅* | 返回 Model RI 的 32-bit ID；目标 CANN 接口为试验特性 |

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
| `cuCtxGetCurrent()`                | `aclrtGetCurrentContext()`              | ✅* | 无当前 Context 时将 CANN `ACL_ERROR_RT_CONTEXT_NULL` 归一为成功并输出空句柄；CUDA 13.3 GPU 与 CANN 9.2.0/910B3 成功路径通过 |
| `cuCtxSetCurrent()`                | `aclrtSetCurrentContext()` + 线程局部空 Context 状态 | ✅* | CANN 9.2 拒绝空 Context，兼容层用线程局部状态表示 CUDA 的 detach；选择非空 Context 或 Device 时清除；GPU/NPU 成功路径通过 |
| `cuDevicePrimaryCtxGetState()`     | `aclrtGetPrimaryCtxState()`             | ✅* | CANN 要求 `flags=NULL`，兼容层只能将 CUDA `flags` 输出置零，非零 flags 语义不可保留；GPU/NPU 活跃状态查询通过 |
| `cuCtxCreate()`                    | `aclrtCreateContext()`                 | ✅* | CUDA 13.x 四参数入口；仅接受空 `CUctxCreateParams` 和 `flags=0`，显式 Context 生命周期由调用方管理；GPU/NPU 成功路径通过 |
| `cuCtxDestroy()`                   | `aclrtDestroyContext()`                | ✅* | 仅销毁显式创建的 Context；不能销毁 `aclrtSetDevice` 隐式创建的默认 Context；GPU/NPU 成功路径通过 |
| `cuCtxGetDevice()`                 | `aclrtGetDevice()`                     | ✅* | 查询当前 Context 对应设备；要求当前线程已有 Context；GPU/NPU 成功路径通过 |
| `cuCtxGetStreamPriorityRange()`   | `aclrtDeviceGetStreamPriorityRange()`  | ✅* | 支持单个或两个输出指针；GPU 观察到 greatest=-5、least=0，NPU 观察到 greatest=0、least=7；各侧排序和单输出断言均通过 |
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
| `cudaGraphAddNode()` | CUDA Runtime Graph | CANN Runtime 不支持添加 CUDA Graph Node，包括 conditional node；兼容层返回 `cudaErrorNotSupported` |
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
