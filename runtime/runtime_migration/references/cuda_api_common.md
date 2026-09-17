# 常用 CUDA Runtime API 参考

## 设备管理 API

### cudaGetDeviceCount
```c
cudaError_t cudaGetDeviceCount(int *count);
```
- 功能：获取可用 CUDA 设备数量
- 参数：count - 设备数量输出指针
- 返回：cudaSuccess 或错误码

### cudaSetDevice
```c
cudaError_t cudaSetDevice(int device);
```
- 功能：设置当前线程的活动设备
- 参数：device - 设备 ID (0 到 count-1)
- 返回：cudaSuccess 或错误码

### cudaGetDevice
```c
cudaError_t cudaGetDevice(int *device);
```
- 功能：获取当前线程的活动设备 ID
- 参数：device - 设备 ID 输出指针
- 返回：cudaSuccess 或错误码

### cudaGetDeviceProperties
```c
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);
```
- 功能：获取设备属性
- 参数：prop - 属性结构体指针，device - 设备 ID
- 返回：cudaSuccess 或错误码

### cudaDeviceSynchronize
```c
cudaError_t cudaDeviceSynchronize(void);
```
- 功能：阻塞等待设备上所有任务完成
- 返回：cudaSuccess 或错误码

### cudaDeviceReset
```c
cudaError_t cudaDeviceReset(void);
```
- 功能：释放当前进程在设备上的所有资源
- 返回：cudaSuccess 或错误码

### cudaDeviceGetAttribute
```c
cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device);
```
- 功能：获取指定设备属性值
- 参数：value - 属性值输出，attr - 属性枚举，device - 设备 ID
- 返回：cudaSuccess 或错误码

### cudaDeviceEnablePeerAccess
```c
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
```
- 功能：启用当前设备对 peerDevice 的直接访问
- 参数：peerDevice - 目标设备，flags - 目前必须为 0
- 返回：cudaSuccess 或错误码，flags 非 0 时返回 `cudaErrorInvalidValue`

### cudaDeviceCanAccessPeer
```c
cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice);
```
- 功能：检查 device 是否可以直接访问 peerDevice 内存
- 参数：canAccessPeer - 结果输出 (1=可访问, 0=不可)，为空时返回 `cudaErrorInvalidValue`
- 返回：cudaSuccess 或错误码

### cudaDeviceSetLimit
```c
cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value);
```
- 功能：设置设备运行时资源限制
- 参数：limit - 资源限制枚举，value - 目标大小或数量
- 返回：cudaSuccess 或错误码；不支持的 limit 返回 `cudaErrorUnsupportedLimit`
- 兼容层口径：当前映射 `cudaLimitStackSize` 和 `cudaLimitPrintfFifoSize`，其余 CUDA limit 因 CANN 无等价 Runtime limit 返回 `cudaErrorUnsupportedLimit`

### cudaDeviceGetLimit
```c
cudaError_t cudaDeviceGetLimit(size_t *pValue, cudaLimit limit);
```
- 功能：查询设备运行时资源限制
- 参数：pValue - 输出指针，limit - 资源限制枚举
- 返回：cudaSuccess 或错误码；pValue 为空返回 `cudaErrorInvalidValue`
- 兼容层口径：当前映射 `cudaLimitStackSize` 和 `cudaLimitPrintfFifoSize`，其余 CUDA limit 返回 `cudaErrorUnsupportedLimit`

### cudaDeviceGetHostAtomicCapabilities
```c
cudaError_t cudaDeviceGetHostAtomicCapabilities(unsigned int *capabilities,
                                                const cudaAtomicOperation *operations,
                                                unsigned int count,
                                                int device);
```
- 功能：查询指定设备对 host atomic 操作的能力
- 参数：capabilities - 输出数组，operations - atomic operation 输入数组，count - 数组长度，device - 设备 ID
- 返回：cudaSuccess 或错误码；空指针、count 为 0、非法 operation 返回 `cudaErrorInvalidValue`
- 兼容层口径：operation 与 CANN SIMT atomic operation 数值一致；capability bit 需从 CANN 枚举转换为 CUDA 枚举，CANN 独有 scalar8/scalar16 能力不暴露为 CUDA bit

### cudaDeviceGetP2PAtomicCapabilities
```c
cudaError_t cudaDeviceGetP2PAtomicCapabilities(unsigned int *capabilities,
                                               const cudaAtomicOperation *operations,
                                               unsigned int count,
                                               int srcDevice,
                                               int dstDevice);
```
- 功能：查询 srcDevice 到 dstDevice 的 P2P atomic 能力
- 参数：capabilities - 输出数组，operations - atomic operation 输入数组，count - 数组长度，srcDevice/dstDevice - 源/目的设备 ID
- 返回：cudaSuccess 或错误码；空指针、count 为 0、非法 operation 返回 `cudaErrorInvalidValue`，srcDevice 与 dstDevice 相同返回 `cudaErrorInvalidDevice`
- 说明：跨设备成功路径依赖设备数量、产品和拓扑能力

---

## 内存管理 API

### cudaMalloc
```c
cudaError_t cudaMalloc(void **devPtr, size_t size);
```
- 功能：在设备上分配内存
- 参数：devPtr - 设备指针输出，size - 分配大小
- 返回：cudaSuccess 或错误码

### cudaFree
```c
cudaError_t cudaFree(void *devPtr);
```
- 功能：释放设备内存
- 参数：devPtr - 要释放的设备指针
- 返回：cudaSuccess 或错误码

### cudaMallocHost
```c
cudaError_t cudaMallocHost(void **ptr, size_t size);
```
- 功能：分配页锁定（pinned）主机内存
- 参数：ptr - 主机指针输出，size - 分配大小
- 返回：cudaSuccess 或错误码
- 说明：页锁定内存可提高 cudaMemcpy 性能

### cudaFreeHost
```c
cudaError_t cudaFreeHost(void *ptr);
```
- 功能：释放页锁定主机内存
- 参数：ptr - 要释放的主机指针
- 返回：cudaSuccess 或错误码

### cudaMemcpy
```c
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
```
- 功能：同步内存拷贝
- 参数：dst - 目标地址，src - 源地址，count - 拷贝大小，kind - 拷贝方向
- kind 值：
  - cudaMemcpyHostToDevice (0)
  - cudaMemcpyDeviceToHost (1)
  - cudaMemcpyDeviceToDevice (2)
- 返回：cudaSuccess 或错误码
- 说明：阻塞直到拷贝完成

### cudaMemcpyAsync
```c
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream);
```
- 功能：异步内存拷贝
- 参数：同 cudaMemcpy，增加 stream 参数
- 返回：cudaSuccess（立即返回）
- 说明：非阻塞，拷贝在流中执行

### cudaMemcpy2D
```c
cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
                         size_t spitch, size_t width, size_t height,
                         cudaMemcpyKind kind);
```
- 功能：2D 内存拷贝
- 参数：pitch - 行间距，width/height - 区域大小
- 返回：cudaSuccess 或错误码

### cudaMemcpyPeer
```c
cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src,
                           int srcDevice, size_t count);
```
- 功能：设备间内存拷贝（同步）
- 参数：dstDevice/srcDevice - 设备 ID
- 返回：cudaSuccess 或错误码

### cudaMemcpyPeerAsync
```c
cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src,
                                int srcDevice, size_t count, cudaStream_t stream);
```
- 功能：设备间内存拷贝（异步）
- 参数：dstDevice/srcDevice - 设备 ID，stream - 下发异步拷贝任务的流
- 返回：cudaSuccess 或错误码
- 说明：跨设备复制前通常需要确认 P2P 能力并启用对应方向的 peer access。迁移到 CANN 兼容层跨 Device 验证时，需要按 CANN P2P 约束调整用例顺序：先查询互通能力，再在两端 Device 双向 enable peer access，然后设置 `CUDA_COMPAT_DEVICE_MALLOC_POLICY=p2p` 并分配 P2P 源/目的内存，最后在目的 Device 的 stream 上执行 `cudaMemcpyPeerAsync` 并同步校验。普通同设备场景不设置该变量时仍使用默认设备内存策略。

### cudaMemset
```c
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
```
- 功能：初始化设备内存
- 参数：value - 设置值（单字节），count - 字节数
- 返回：cudaSuccess 或错误码

### cudaMemsetAsync
```c
cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
                             cudaStream_t stream);
```
- 功能：异步初始化设备内存
- 参数：同 cudaMemset，增加 stream
- 返回：cudaSuccess（立即返回）

### cudaMemGetInfo
```c
cudaError_t cudaMemGetInfo(size_t *free, size_t *total);
```
- 功能：获取设备内存信息
- 参数：free - 可用内存，total - 总内存
- 返回：cudaSuccess 或错误码

### cudaPointerGetAttributes
```c
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attributes,
                                      const void *ptr);
```
- 功能：获取指针属性
- 参数：attributes - 属性结构体，ptr - 查询指针
- 返回：cudaSuccess 或错误码

### cudaHostRegister
```c
cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags);
```
- 功能：注册主机内存为页锁定
- 参数：flags - cudaHostRegisterDefault 等
- 兼容层口径：`cudaHostRegisterDefault` 覆盖页锁定注册和 `cudaHostUnregister` 成功路径；需要 `cudaHostGetDevicePointer` 时使用 `cudaHostRegisterMapped`
- 返回：cudaSuccess 或错误码

### cudaHostUnregister
```c
cudaError_t cudaHostUnregister(void *ptr);
```
- 功能：取消页锁定注册
- 参数：ptr - 注册的指针
- 兼容层口径：释放 `cudaHostRegister` 成功注册的主机内存；注册内存本身仍由调用者用原分配方式释放
- 返回：cudaSuccess 或错误码

---

## 流管理 API

### cudaStreamCreate
```c
cudaError_t cudaStreamCreate(cudaStream_t *pStream);
```
- 功能：创建异步流
- 参数：pStream - 流句柄输出
- 返回：cudaSuccess 或错误码

### cudaStreamCreateWithFlags
```c
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags);
```
- 功能：带标志创建流
- flags：cudaStreamNonBlocking (1) - 不与默认流同步
- 返回：cudaSuccess 或错误码

### cudaStreamCreateWithPriority
```c
cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream,
                                          unsigned int flags, int priority);
```
- 功能：带优先级创建流
- priority：数值越小优先级越高
- 返回：cudaSuccess 或错误码

### cudaStreamDestroy
```c
cudaError_t cudaStreamDestroy(cudaStream_t stream);
```
- 功能：销毁流
- 参数：stream - 要销毁的流
- 返回：cudaSuccess 或错误码

### cudaStreamSynchronize
```c
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
```
- 功能：阻塞等待流完成
- 参数：stream - 要同步的流
- 返回：cudaSuccess 或错误码

### cudaStreamQuery
```c
cudaError_t cudaStreamQuery(cudaStream_t stream);
```
- 功能：查询流状态
- 返回：cudaSuccess（完成）或 cudaErrorNotReady（未完成）

### cudaStreamWaitEvent
```c
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                unsigned int flags);
```
- 功能：让流等待事件完成
- 参数：flags - 通常为 0
- 返回：cudaSuccess 或错误码

### cudaStreamBeginCapture
```c
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode);
```
- 功能：开始捕获指定 stream 上的后续任务
- 参数：mode 支持 `cudaStreamCaptureModeGlobal`、`cudaStreamCaptureModeThreadLocal`、`cudaStreamCaptureModeRelaxed`
- 兼容层口径：非法 mode 先返回 `cudaErrorInvalidValue`；CANN 侧基于 Model RI capture，默认 stream 和捕获期间同步/查询操作受 CANN 约束限制

### cudaStreamEndCapture
```c
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph);
```
- 功能：结束 stream capture，并通过 `pGraph` 返回捕获得到的 graph
- 兼容层口径：映射到 CANN Model RI；用于 `cudaStreamBeginCaptureToGraph` 的子图结束场景时允许按 CANN 语义传入空输出

### cudaStreamIsCapturing
```c
cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus *pCaptureStatus);
```
- 功能：查询 stream 当前是否处于 capture 状态
- 参数：`pCaptureStatus` 为空时返回 `cudaErrorInvalidValue`
- 兼容层口径：通过 CANN capture info 显式映射 None/Active/Invalidated 三种状态

---

## 事件管理 API

### cudaEventCreate
```c
cudaError_t cudaEventCreate(cudaEvent_t *event);
```
- 功能：创建事件
- 参数：event - 事件句柄输出
- 返回：cudaSuccess 或错误码

### cudaEventCreateWithFlags
```c
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
```
- 功能：带标志创建事件
- flags：
  - cudaEventDefault (0)
  - cudaEventBlockingSync (1) - 阻塞同步
  - cudaEventDisableTiming (2) - 禁用计时
  - cudaEventInterprocess (4) - 跨进程
- 返回：cudaSuccess 或错误码

### cudaEventDestroy
```c
cudaError_t cudaEventDestroy(cudaEvent_t event);
```
- 功能：销毁事件
- 参数：event - 要销毁的事件
- 返回：cudaSuccess 或错误码

### cudaEventRecord
```c
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
```
- 功能：在流中记录事件
- 参数：event - 事件，stream - 流（可为 NULL/0 表示默认流）
- 返回：cudaSuccess 或错误码

### cudaEventSynchronize
```c
cudaError_t cudaEventSynchronize(cudaEvent_t event);
```
- 功能：阻塞等待事件完成
- 参数：event - 要同步的事件
- 返回：cudaSuccess 或错误码

### cudaEventQuery
```c
cudaError_t cudaEventQuery(cudaEvent_t event);
```
- 功能：查询事件状态
- 返回：cudaSuccess（完成）或 cudaErrorNotReady（未完成）

### cudaEventElapsedTime
```c
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
```
- 功能：计算两事件间耗时
- 参数：ms - 耗时输出（毫秒），start/end - 开始/结束事件
- 返回：cudaSuccess 或错误码
- 要求：事件创建时未设置 cudaEventDisableTiming

---

## IPC API

### cudaIpcGetMemHandle
```c
cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr);
```
- 功能：获取 IPC 内存句柄
- 参数：handle - 句柄输出，devPtr - 设备指针
- 返回：cudaSuccess 或错误码

### cudaIpcOpenMemHandle
```c
cudaError_t cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle,
                                 unsigned int flags);
```
- 功能：打开 IPC 内存句柄
- 参数：flags - cudaIpcMemLazyEnablePeerAccess
- 返回：cudaSuccess 或错误码

### cudaIpcCloseMemHandle
```c
cudaError_t cudaIpcCloseMemHandle(void *devPtr);
```
- 功能：关闭 IPC 内存映射
- 参数：devPtr - IPC 映射的指针
- 返回：cudaSuccess 或错误码

### cudaIpcGetEventHandle
```c
cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event);
```
- 功能：获取 IPC 事件句柄
- 参数：handle - 句柄输出，event - 事件
- 返回：cudaSuccess 或错误码
- 要求：事件创建时设置 cudaEventInterprocess

### cudaIpcOpenEventHandle
```c
cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle);
```
- 功能：打开 IPC 事件句柄
- 参数：event - 事件输出，handle - IPC 句柄
- 返回：cudaSuccess 或错误码

---

## Profiler API

### cudaProfilerStart
```c
cudaError_t cudaProfilerStart(void);
```
- 功能：启动 profiler 数据收集
- 返回：cudaSuccess 或错误码

### cudaProfilerStop
```c
cudaError_t cudaProfilerStop(void);
```
- 功能：停止 profiler 数据收集
- 返回：cudaSuccess 或错误码

---

## Version API

### cudaRuntimeGetVersion
```c
cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);
```
- 功能：获取 CUDA Runtime 版本号
- 参数：runtimeVersion - 版本号输出
- 返回：cudaSuccess 或错误码

### cudaDriverGetVersion
```c
cudaError_t cudaDriverGetVersion(int *driverVersion);
```
- 功能：获取 CUDA Driver 版本号
- 参数：driverVersion - 版本号输出
- 返回：cudaSuccess 或错误码

---

## Stream Ordered Memory / UVM API

### cudaMallocAsync
```c
cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream);
```
- 功能：在指定 Stream 上异步申请内存
- CANN 对标结论：当前未真实对标 CUDA SOMA，兼容层返回 `cudaErrorNotSupported`

### cudaFreeAsync
```c
cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream);
```
- 功能：在指定 Stream 上异步释放内存
- CANN 对标结论：当前未真实对标 CUDA SOMA，兼容层返回 `cudaErrorNotSupported`

### cudaMemPoolSetAttribute / cudaMemPoolGetAttribute / cudaMemPoolTrimTo
```c
cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void *value);
cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void *value);
cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep);
```
- 功能：设置、查询和收缩 CUDA 内存池
- CANN 对标结论：当前未真实对标 CUDA SOMA，兼容层返回 `cudaErrorNotSupported`

### cudaMallocManaged
```c
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags);
```
- 功能：申请统一内存
- CANN 对标结论：当前未真实对标 CUDA UVM，兼容层返回 `cudaErrorNotSupported`

### cudaMemAdvise / cudaMemPrefetchAsync / cudaMemRangeGetAttribute(s)
```c
cudaError_t cudaMemAdvise(const void *devPtr, size_t count, cudaMemoryAdvise advice, int device);
cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count, cudaMemLocation location, unsigned int flags,
                                 cudaStream_t stream);
```
- 功能：CUDA UVM advise、prefetch 和 range attribute 查询
- CANN 对标结论：当前不做 CUDA UVM 映射，不进入转测验收；兼容层返回 `cudaErrorNotSupported`

---

## Kernel Launch API

### cudaLaunchKernel
```c
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem, cudaStream_t stream);
```
- 功能：启动 CUDA kernel
- 参数：func - kernel 入口，gridDim/blockDim - 网格与线程块维度，args - 参数数组，sharedMem - 动态共享内存大小，stream - 执行流
- 返回：cudaSuccess 或错误码
- 兼容层口径：空 `func` 返回 `cudaErrorInvalidDeviceFunction`，gridDim 或 blockDim 任一维度为 0 返回 `cudaErrorInvalidConfiguration`；CANN 侧只能启动其可识别的 kernel/function handle。普通 CUDA 源码中的 `kernel<<<...>>>` 迁移不代表 NPU device 侧执行，应按 `runtime_migration` 的 Host fallback 规则处理并在报告中标注。

### cudaLaunchHostFunc
```c
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData);
```
- 功能：在 stream 中插入 Host 回调任务
- 参数：`fn` 为空时按已验证 CUDA baseline 作为 no-op 返回 `cudaSuccess`
- 兼容层口径：映射到 CANN `aclrtLaunchHostFunc`；回调函数不要做资源申请/释放、stream/device 同步或继续下发任务，避免死锁或运行期错误

---

## Graph API

### cudaGraphDebugDotPrint
```c
cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char *path,
                                   unsigned int flags);
```
- 功能：导出 Graph 调试信息
- 参数：graph - Graph 对象，path - 输出路径，flags - 调试输出标志
- 返回：cudaSuccess 或错误码
- 兼容层口径：映射到 CANN `aclmdlRIDebugJsonPrint`，实际导出为 Model RI JSON 调试信息；空 graph 或 path 返回 `cudaErrorInvalidValue`

### cudaGraphExecDestroy
```c
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec);
```
- 功能：销毁 Graph 执行实例
- 参数：graphExec - Graph 执行实例
- 返回：cudaSuccess 或错误码
- 兼容层口径：映射到 CANN `aclmdlRIDestroy`；空执行实例返回 `cudaErrorInvalidValue`

### cudaGraphLaunch
```c
cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);
```
- 功能：在指定流上启动 Graph 执行实例
- 参数：graphExec - Graph 执行实例，stream - 执行流
- 返回：cudaSuccess 或错误码
- 兼容层口径：映射到 CANN `aclmdlRIExecuteAsync`；空执行实例返回 `cudaErrorInvalidValue`

### cudaGraphConditionalHandleCreate
```c
cudaError_t cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle *pHandle,
                                             cudaGraph_t graph,
                                             unsigned int defaultLaunchValue,
                                             unsigned int flags);
```
- 功能：为条件 Graph 节点创建条件 handle
- 返回：cudaSuccess 或错误码

### cudaGraphGetNodes
```c
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes);
```
- 功能：查询 Graph 中的节点列表
- 返回：cudaSuccess 或错误码

### cudaGraphAddNode (conditional node only)
```c
cudaError_t cudaGraphAddNode(cudaGraphNode_t *pGraphNode,
                             cudaGraph_t graph,
                             const cudaGraphNode_t *dependencies,
                             const cudaGraphEdgeData *dependencyData,
                             size_t numDependencies,
                             cudaGraphNodeParams *nodeParams);
```
- 功能：`nodeParams->type == cudaGraphNodeTypeConditional` 时创建条件 Graph 节点
- 约束：仅 conditional node 特例映射到 CANN Model RI 条件任务；其他 node type 不属于当前 Runtime 兼容范围
- 返回：cudaSuccess 或错误码

### cudaGraphSetConditional
```c
cudaError_t cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);
```
- 功能：设置条件 handle 的当前值
- 返回：cudaSuccess 或错误码

---

## 增量支持 API 摘要

本节记录新增的 CUDA Runtime/Driver API 面，兼容层已提供 CUDA to CANN Runtime 转换。

| CUDA API | 功能摘要 |
|---|---|
| `cudaEventRecordWithFlags` | 带 flags 记录 event |
| `cudaHostAlloc` | 分配 Host pinned 内存 |
| `cudaGetSymbolAddress` | 获取设备符号地址 |
| `cudaMemcpyToSymbol` | 向设备符号拷贝数据 |
| `cuMemsetD32Async` | Driver D32 异步 memset |
| `cudaFuncGetAttributes` | 查询 kernel/function 属性 |
| `cudaGraphConditionalHandleCreate` | 创建条件 Graph handle |
| `cudaGraphAddNode(cudaGraphNodeTypeConditional)` | 添加条件 Graph 节点 |
| `cudaGraphGetNodes` | 查询 Graph 节点 |
| `cudaGraphSetConditional` | 设置条件 handle 值 |
| `cudaStreamBeginCaptureToGraph` | 将 stream capture 接入已有 Graph |
| `cudaStreamCaptureStatus` | Stream capture 状态枚举；对应 CANN `aclmdlRICaptureStatus` |
| `cudaStreamGetCaptureInfo` | 查询 capture 信息 |
| `cudaStreamGetCaptureInfo_v3` | 查询 v3 capture 信息 |
| `cuCtxGetCurrent` | 获取当前 Driver context |
| `cuCtxSetCurrent` | 设置当前 Driver context |
| `cuDevicePrimaryCtxGetState` | 查询 primary context 状态 |
| `cuModuleGetFunction` | 从 module 获取 kernel function |
| `cuModuleLoad` | 从文件加载 module |
| `cuModuleLoadData` | 从内存加载 module |
| `cuModuleUnload` | 卸载 module |
| `cuStreamWriteValue32` | 在 stream 上写入 32-bit value |

## 不支持 API 摘要

以下接口当前无 CANN Runtime 对应能力，兼容层仅提供编译期 API 面并返回 `cudaErrorNotSupported` 或 `CUDA_ERROR_NOT_SUPPORTED`：

### CUDA Runtime 不支持接口

- `cudaMemPoolSetAccess`
- `cudaFuncSetAttribute`
- `cudaLaunchCooperativeKernel`
- `cudaGetDriverEntryPoint`
- `cudaGetDriverEntryPointByVersion`
- `cudaGraphAddNode` 非 conditional node 类型
- `cudaGraphAddNode_v2`
- `cudaGraphDestroy`
- `cudaGraphInstantiateWithFlags`
- `cudaGraphNodeGetDependencies`
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
- `cudaOccupancyMaxPotentialBlockSize`
- `cudaStreamGetCaptureInfo_v2`
- `cudaStreamUpdateCaptureDependencies`
- `cudaStreamUpdateCaptureDependencies_v2`

### CUDA Driver 不支持接口

- `cuFuncSetCacheConfig`
- `cuCtxPopCurrent`
- `cuCtxPushCurrent`
- `cuDevicePrimaryCtxRetain`
- `cuGreenCtxCreate`
- `cuGreenCtxDestroy`
- `cuCtxFromGreenCtx`
- `cuDeviceGetDevResource`
- `cuGreenCtxStreamCreate`
- `cuModuleLoadDataEx`
- `cuLinkAddData`
- `cuMulticastAddDevice`
- `cuMulticastBindMem`
- `cuMulticastCreate`
- `cuMulticastUnbind`
- `cuTensorMapEncodeTiled`

---

## 错误处理 API

### cudaGetLastError
```c
cudaError_t cudaGetLastError(void);
```
- 功能：获取并清除最后的错误
- 返回：最后的错误码

### cudaPeekAtLastError
```c
cudaError_t cudaPeekAtLastError(void);
```
- 功能：获取最后的错误（不清除）
- 返回：最后的错误码

### cudaGetErrorString
```c
const char *cudaGetErrorString(cudaError_t error);
```
- 功能：获取错误描述字符串
- 参数：error - 错误码
- 返回：错误描述字符串

### cudaGetErrorName
```c
const char *cudaGetErrorName(cudaError_t error);
```
- 功能：获取错误名称字符串
- 参数：error - 错误码
- 返回：错误名称字符串；兼容层对未知错误码会尝试返回最近一次 ACL 错误描述，否则返回 `cudaErrorUnknown`
