# 常用 CANN Runtime API 参考

## 设备管理 API

### aclrtGetDeviceCount
```c
aclError aclrtGetDeviceCount(uint32_t *count);
```
- 功能：获取可用设备数量
- 参数：count - 设备数量输出
- 返回：ACL_SUCCESS 或错误码

### aclrtSetDevice
```c
aclError aclrtSetDevice(int32_t deviceId);
```
- 功能：设置当前线程的活动设备
- 参数：deviceId - 设备 ID
- 返回：ACL_SUCCESS 或错误码

### aclrtGetDevice
```c
aclError aclrtGetDevice(int32_t *deviceId);
```
- 功能：获取当前线程的活动设备 ID
- 参数：deviceId - 设备 ID 输出
- 返回：ACL_SUCCESS 或错误码

### aclrtGetDeviceInfo
```c
aclError aclrtGetDeviceInfo(int32_t deviceId, aclrtDeviceInfoType infoType,
                            void *infoValue, size_t *infoValueLen);
```
- 功能：获取设备信息
- 参数：
  - deviceId - 设备 ID
  - infoType - 信息类型枚举
  - infoValue - 信息输出
  - infoValueLen - 信息长度
- infoType 常用值：
  - ACL_RT_DEVICE_INFO_TYPE_NAME - 设备名称
  - ACL_RT_DEVICE_INFO_TYPE_CORE_NUM - AI Core 数量
  - ACL_RT_DEVICE_INFO_TYPE_FREQUENCY - 频率
- 返回：ACL_SUCCESS 或错误码

### aclrtSynchronizeDevice
```c
aclError aclrtSynchronizeDevice(void);
```
- 功能：阻塞等待设备上所有任务完成
- 返回：ACL_SUCCESS 或错误码

### aclrtResetDeviceForce
```c
aclError aclrtResetDeviceForce(int32_t deviceId);
```
- 功能：强制释放设备资源
- 参数：deviceId - 设备 ID
- 返回：ACL_SUCCESS 或错误码

### aclrtDeviceEnablePeerAccess
```c
aclError aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags);
```
- 功能：启用当前 Device 到 peerDevice 的单向数据交互
- 参数：peerDeviceId - 目标设备 ID，不能与当前 Device 相同；flags - 保留参数，当前必须为 0
- 说明：双向访问需要分别在两个 Device 上调用；可先用 `aclrtDeviceCanAccessPeer` 查询能力，支持范围受产品和拓扑限制
- 返回：ACL_SUCCESS 或错误码

### aclrtDeviceCanAccessPeer
```c
aclError aclrtDeviceCanAccessPeer(int32_t *canAccess, int32_t deviceId,
                                   int32_t peerDeviceId);
```
- 功能：检查设备间直接访问能力
- 参数：canAccess - 结果输出 (1/0)，deviceId 与 peerDeviceId 不能相同
- 返回：ACL_SUCCESS 或错误码

### aclrtDeviceGetStreamPriorityRange
```c
aclError aclrtDeviceGetStreamPriorityRange(int32_t *leastPriority,
                                            int32_t *greatestPriority);
```
- 功能：获取流优先级范围
- 参数：leastPriority - 最低优先级(数值大)，greatestPriority - 最高优先级(数值小)
- 返回：ACL_SUCCESS 或错误码

### aclrtDeviceSetLimit / aclrtDeviceGetLimit
```c
aclError aclrtDeviceSetLimit(aclrtDeviceLimit limit, size_t value);
aclError aclrtDeviceGetLimit(aclrtDeviceLimit limit, size_t *value);
```
- 功能：设置或查询设备运行时资源限制
- 兼容层映射：`cudaLimitStackSize` 使用 `ACL_RT_DEV_LIMIT_SIMD_STACK_SIZE`；`cudaLimitPrintfFifoSize` 使用 `ACL_RT_DEV_LIMIT_SIMD_PRINTF_FIFO_SIZE_PER_CORE`
- 限制：CANN limit 类型与 CUDA 不完全等价；heap、device runtime sync depth、pending launch count、L2 fetch/persisting 等 CUDA limit 当前无等价 Runtime 映射，兼容层返回 `cudaErrorUnsupportedLimit`

### aclrtDeviceGetHostAtomicCapabilities
```c
aclError aclrtDeviceGetHostAtomicCapabilities(uint32_t *capabilities,
                                              const aclrtAtomicOperation *operations,
                                              uint32_t count,
                                              int32_t deviceId);
```
- 功能：查询 Host atomic operation 能力
- 兼容层映射：CUDA operation 0-12 映射到 CANN 对应 operation；CANN capability bit 转换为 CUDA capability bit
- 限制：CANN 的 scalar8/scalar16 能力没有 CUDA Runtime bit 位，兼容层不向上暴露

### aclrtDeviceGetP2PAtomicCapabilities
```c
aclError aclrtDeviceGetP2PAtomicCapabilities(uint32_t *capabilities,
                                             const aclrtAtomicOperation *operations,
                                             uint32_t count,
                                             int32_t srcDeviceId,
                                             int32_t dstDeviceId);
```
- 功能：查询两个设备之间的 P2P atomic operation 能力
- 说明：成功路径依赖产品、设备数量和拓扑；兼容层先按 CUDA 语义拦截同设备 src/dst，返回 `cudaErrorInvalidDevice`

---

## 内存管理 API

### aclrtMalloc
```c
aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy);
```
- 功能：在设备上分配内存
- 参数：
  - devPtr - 设备指针输出
  - size - 分配大小
  - policy - 分配策略
- policy 值：
  - ACL_MEM_MALLOC_HUGE_FIRST (0) - 优先 Huge Page
  - ACL_MEM_MALLOC_HUGE_ONLY (1) - 仅 Huge Page
  - ACL_MEM_MALLOC_NORMAL_ONLY (2) - 仅普通内存
- 返回：ACL_SUCCESS 或错误码

### aclrtFree
```c
aclError aclrtFree(void *devPtr);
```
- 功能：释放设备内存
- 参数：devPtr - 设备指针
- 返回：ACL_SUCCESS 或错误码

### aclrtMallocHost
```c
aclError aclrtMallocHost(void **hostPtr, size_t size);
```
- 功能：分配页锁定主机内存
- 参数：hostPtr - 主机指针输出，size - 分配大小
- 返回：ACL_SUCCESS 或错误码

### aclrtFreeHost
```c
aclError aclrtFreeHost(void *hostPtr);
```
- 功能：释放页锁定主机内存
- 参数：hostPtr - 主机指针
- 返回：ACL_SUCCESS 或错误码

### aclrtMemcpy
```c
aclError aclrtMemcpy(void *dst, size_t destMax, const void *src,
                      size_t count, aclrtMemcpyKind kind);
```
- 功能：同步内存拷贝
- 参数：
  - dst - 目标地址
  - destMax - 目标最大容量
  - src - 源地址
  - count - 拷贝大小
  - kind - 拷贝方向
- kind 值：
  - ACL_MEMCPY_HOST_TO_DEVICE (0)
  - ACL_MEMCPY_DEVICE_TO_HOST (1)
  - ACL_MEMCPY_DEVICE_TO_DEVICE (2)
- 返回：ACL_SUCCESS 或错误码

### aclrtMemcpyAsync
```c
aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src,
                           size_t count, aclrtMemcpyKind kind,
                           aclrtStream stream);
```
- 功能：异步内存拷贝
- 参数：同 aclrtMemcpy，增加 stream
- 返回：ACL_SUCCESS 或错误码；接口成功只表示任务下发成功，需同步 stream/device 后确认完成
- 说明：`ACL_MEMCPY_DEVICE_TO_DEVICE` 可表示同 Device 或跨 Device 复制；跨 Device 复制需先查询并启用 peer access，当前兼容层仅按同一个 PCIe Switch 内 Device 间复制进行条件支持。跨 Device P2P 验证时，应在分配 P2P 源/目的 device buffer 前，分别切到源 Device 和目的 Device 开启双向 peer access，并设置 `CUDA_COMPAT_DEVICE_MALLOC_POLICY=p2p`，使兼容层 `cudaMalloc` 使用 `ACL_MEM_MALLOC_HUGE_FIRST_P2P` 分配策略；复制应在目的 Device 上创建的 stream 中下发。若只开启单向访问或在普通内存分配后才 enable，可能在 stream 同步阶段返回设备/驱动不匹配错误。

### aclrtMemSetAccess
- 兼容层口径：用于 `cuMemSetAccess` 的 VMM 映射访问权限设置。部分产品或驱动组合可能返回未映射的 VMM access 错误；兼容层在该接口内把未映射错误收敛为 `CUDA_ERROR_NOT_SUPPORTED`，转测用例按条件 `SKIP` 处理，而不是暴露含糊的 `CUDA_ERROR_UNKNOWN`。

### aclrtMemcpy2d
```c
aclError aclrtMemcpy2d(void *dst, size_t dpitch, const void *src,
                       size_t spitch, size_t width, size_t height,
                       aclrtMemcpyKind kind);
```
- 功能：2D 内存拷贝（同步）
- 参数：pitch - 行间距，width/height - 区域大小
- 返回：ACL_SUCCESS 或错误码

### aclrtMemcpy2dAsync
```c
aclError aclrtMemcpy2dAsync(void *dst, size_t dpitch, const void *src,
                             size_t spitch, size_t width, size_t height,
                             aclrtMemcpyKind kind, aclrtStream stream);
```
- 功能：2D 异步内存拷贝
- 参数：同 aclrtMemcpy2d，增加 stream
- 返回：ACL_SUCCESS

### aclrtMemcpyBatchAsync
```c
aclError aclrtMemcpyBatchAsync(void **dsts, void **srcs, size_t *sizes,
                                size_t count, aclrtMemcpyKind kind,
                                aclrtStream stream);
```
- 功能：批量异步内存拷贝
- 参数：dsts/srcs - 地址数组，sizes - 大小数组，count - 数量
- 返回：ACL_SUCCESS

### aclrtMemset
```c
aclError aclrtMemset(void *devPtr, size_t count, int32_t value);
```
- 功能：初始化设备内存
- 参数：devPtr - 设备指针，count - 字节数，value - 设置值
- 返回：ACL_SUCCESS 或错误码

### aclrtMemsetAsync
```c
aclError aclrtMemsetAsync(void *devPtr, size_t count, int32_t value,
                           aclrtStream stream);
```
- 功能：异步初始化设备内存
- 参数：同 aclrtMemset，增加 stream
- 返回：ACL_SUCCESS

### aclrtGetMemInfo
```c
aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total);
```
- 功能：获取设备内存信息
- 参数：attr - 内存类型，free/total - 输出
- attr 值：ACL_MEM_ATTR_HBM, ACL_MEM_ATTR_DDR
- 返回：ACL_SUCCESS 或错误码

### aclrtPointerGetAttributes
```c
aclError aclrtPointerGetAttributes(size_t size, aclrtPointerAttributes *attributes,
                                    const void *ptr);
```
- 功能：获取指针属性
- 参数：size - 查询范围，attributes - 属性输出，ptr - 指针
- 返回：ACL_SUCCESS 或错误码

### aclrtHostRegisterV2
```c
aclError aclrtHostRegisterV2(void *hostPtr, size_t size,
                              aclrtHostMemRegisterPolicy policy);
```
- 功能：注册主机内存
- 参数：policy - 注册策略
- 返回：ACL_SUCCESS 或错误码

### aclrtHostRegister
```c
aclError aclrtHostRegister(void *ptr, uint64_t size,
                           aclrtHostRegisterType type, void **devPtr);
```
- 功能：将 Host 内存映射注册为 Device 可访问的内存地址
- 参数：type - 注册类型，devPtr - 映射后的 Device 地址输出
- 返回：ACL_SUCCESS 或错误码

### aclrtHostUnregister
```c
aclError aclrtHostUnregister(void *hostPtr);
```
- 功能：取消主机内存注册
- 参数：hostPtr - 主机指针
- 返回：ACL_SUCCESS 或错误码

---

## 流管理 API

### aclrtCreateStream
```c
aclError aclrtCreateStream(aclrtStream *stream);
```
- 功能：创建流
- 参数：stream - 流句柄输出
- 返回：ACL_SUCCESS 或错误码

### aclrtCreateStreamWithConfig
```c
aclError aclrtCreateStreamWithConfig(aclrtStream *stream, int32_t priority,
                                      uint64_t flag);
```
- 功能：带配置创建流
- 参数：priority - 优先级(0最高，7最低)，flag - 标志
- 返回：ACL_SUCCESS 或错误码

### aclrtDestroyStream
```c
aclError aclrtDestroyStream(aclrtStream stream);
```
- 功能：销毁流
- 参数：stream - 流句柄
- 返回：ACL_SUCCESS 或错误码

### aclrtSynchronizeStream
```c
aclError aclrtSynchronizeStream(aclrtStream stream);
```
- 功能：阻塞等待流完成
- 参数：stream - 流句柄
- 返回：ACL_SUCCESS 或错误码

### aclrtStreamQuery
```c
aclError aclrtStreamQuery(aclrtStream stream, aclrtStreamStatus *status);
```
- 功能：查询流状态
- 参数：status - 状态输出
- status 值：ACL_STREAM_STATUS_COMPLETE, ACL_STREAM_STATUS_NOT_READY
- 返回：ACL_SUCCESS 或错误码

### aclrtStreamWaitEvent
```c
aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event);
```
- 功能：让流等待事件
- 参数：stream - 流，event - 事件
- 返回：ACL_SUCCESS 或错误码

### aclrtStreamGetId
```c
aclError aclrtStreamGetId(aclrtStream stream, int32_t *streamId);
```
- 功能：获取流 ID
- 参数：streamId - ID 输出
- 返回：ACL_SUCCESS 或错误码

### aclrtStreamGetPriority
```c
aclError aclrtStreamGetPriority(aclrtStream stream, uint32_t *priority);
```
- 功能：获取流优先级
- 参数：priority - 优先级输出
- 返回：ACL_SUCCESS 或错误码

### aclrtStreamGetFlags
```c
aclError aclrtStreamGetFlags(aclrtStream stream, uint64_t *flag);
```
- 功能：获取流标志
- 参数：flag - 标志输出
- 返回：ACL_SUCCESS 或错误码

---

## 事件管理 API

### aclrtCreateEvent
```c
aclError aclrtCreateEvent(aclrtEvent *event);
```
- 功能：创建事件
- 参数：event - 事件句柄输出
- 返回：ACL_SUCCESS 或错误码

### aclrtCreateEventExWithFlag
```c
aclError aclrtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag);
```
- 功能：带标志创建事件
- 参数：event - 事件输出，flag - 标志组合
- flag 值：
  - ACL_EVENT_SYNC (0x01) - 支持同步
  - ACL_EVENT_TIME_LINE (0x02) - 支持计时
  - ACL_EVENT_IPC (0x04) - 支持跨进程
- 返回：ACL_SUCCESS 或错误码

### aclrtDestroyEvent
```c
aclError aclrtDestroyEvent(aclrtEvent event);
```
- 功能：销毁事件
- 参数：event - 事件句柄
- 返回：ACL_SUCCESS 或错误码

### aclrtRecordEvent
```c
aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream);
```
- 功能：在流中记录事件
- 参数：event - 事件，stream - 流
- 返回：ACL_SUCCESS 或错误码

### aclrtSynchronizeEvent
```c
aclError aclrtSynchronizeEvent(aclrtEvent event);
```
- 功能：阻塞等待事件完成
- 参数：event - 事件句柄
- 返回：ACL_SUCCESS 或错误码

### aclrtQueryEventStatus
```c
aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status);
```
- 功能：查询事件状态
- 参数：status - 状态输出
- status 值：ACL_EVENT_RECORDED_STATUS_COMPLETE, ACL_EVENT_RECORDED_STATUS_NOT_READY
- 返回：ACL_SUCCESS 或错误码

### aclrtEventElapsedTime
```c
aclError aclrtEventElapsedTime(float *ms, aclrtEvent start, aclrtEvent end);
```
- 功能：计算两事件间耗时
- 参数：ms - 耗时输出(毫秒)，start/end - 事件
- 返回：ACL_SUCCESS 或错误码
- 要求：事件创建时设置 ACL_EVENT_TIME_LINE

---

## IPC API

### aclrtIpcMemGetExportKey
```c
aclError aclrtIpcMemGetExportKey(void *devPtr, size_t size, char *key,
                                  size_t len, uint64_t flags);
```
- 功能：导出 IPC 共享内存 key
- 参数：
  - devPtr - 设备指针
  - size - 共享内存大小
  - key - key 输出，长度固定配置为 65
  - flags - 进程白名单校验配置
- 返回：ACL_SUCCESS 或错误码

### aclrtIpcMemImportByKey
```c
aclError aclrtIpcMemImportByKey(void **devPtr, const char *key, uint64_t flags);
```
- 功能：通过 key 导入 IPC 共享内存
- 参数：devPtr - Device 指针输出，key - 共享内存 key，flags - 是否开启两个 Device 之间的数据交互
- 返回：ACL_SUCCESS 或错误码

### aclrtIpcMemClose
```c
aclError aclrtIpcMemClose(const char *key);
```
- 功能：关闭 IPC 共享内存
- 参数：key - 共享内存 key
- 返回：ACL_SUCCESS 或错误码

### aclrtIpcGetEventHandle
```c
aclError aclrtIpcGetEventHandle(aclrtIpcEventHandle *handle, aclrtEvent event);
```
- 功能：导出 IPC 事件句柄
- 参数：handle - 句柄输出，event - 事件
- 返回：ACL_SUCCESS 或错误码
- 要求：事件创建时设置 ACL_EVENT_IPC

### aclrtIpcOpenEventHandle
```c
aclError aclrtIpcOpenEventHandle(aclrtEvent *event, aclrtIpcEventHandle handle);
```
- 功能：导入 IPC 事件句柄
- 参数：event - 事件输出，handle - IPC 句柄
- 返回：ACL_SUCCESS 或错误码

---

## Profiler API

### aclprofInit
```c
aclError aclprofInit(const char *profilerResultPath, size_t length);
```
- 功能：初始化 profiler
- 返回：ACL_SUCCESS 或错误码

### aclprofStart
```c
aclError aclprofStart(const aclprofConfig *profilerConfig);
```
- 功能：启动性能数据收集
- 返回：ACL_SUCCESS 或错误码

### aclprofStop
```c
aclError aclprofStop(const aclprofConfig *profilerConfig);
```
- 功能：停止性能数据收集
- 说明：与 aclprofStart 配对使用；cudaProfilerStop 兼容层还需配套调用 aclprofFinalize
- 返回：ACL_SUCCESS 或错误码

### aclprofFinalize
```c
aclError aclprofFinalize(void);
```
- 功能：释放 profiler 资源
- 返回：ACL_SUCCESS 或错误码

---

## Version API

### aclsysGetVersionNum
```c
aclError aclsysGetVersionNum(char *pkgName, int32_t *versionNum);
```
- 功能：查询软件包版本号
- 参数：pkgName - 包名，例如 "runtime"；versionNum - 版本号输出
- 返回：ACL_SUCCESS 或错误码

---

## Kernel Launch API

### aclrtLaunchKernelWithHostArgs
```c
aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t numBlocks,
                                       aclrtStream stream, aclrtLaunchKernelCfg *cfg,
                                       void *hostArgs, size_t argsSize,
                                       aclrtPlaceHolderInfo *placeHolderArray,
                                       size_t placeHolderNum);
```
- 功能：使用 Host 连续参数启动 Kernel
- 返回：ACL_SUCCESS 或错误码

### aclrtLaunchKernelWithArgsArray
```c
aclError aclrtLaunchKernelWithArgsArray(void *func, uint32_t numBlocks,
                                        aclrtStream stream, aclrtLaunchKernelCfg *cfg,
                                        void **args);
```
- 功能：使用 Host 参数数组启动 Kernel
- 约束：`func` 必须是 CANN Runtime 可识别的 kernel/function handle；`args` 中每个元素指向 Host 侧参数数据，顺序需与 kernel 参数顺序一致
- 返回：ACL_SUCCESS 或错误码

### aclrtLaunchSIMTKernelWithArgsArray
```c
aclError aclrtLaunchSIMTKernelWithArgsArray(void *func, dim3 gridDim, dim3 blockDim,
                                            size_t dynUbufSize, aclrtStream stream,
                                            aclrtLaunchKernelCfg *cfg, void **args);
```
- 功能：使用参数数组启动 SIMT Kernel
- 约束：本地 CANN 文档显示 SIMT launch 仅在部分产品支持；兼容层通过弱符号检查该接口是否由当前运行库导出，缺失时退回非 SIMT launch 路径
- 返回：ACL_SUCCESS 或错误码

### aclrtLaunchSIMTKernelWithHostArgs
```c
aclError aclrtLaunchSIMTKernelWithHostArgs(void *func, dim3 gridDim, dim3 blockDim,
                                           size_t dynUbufSize, aclrtStream stream,
                                           aclrtLaunchKernelCfg *cfg, void *hostArgs,
                                           size_t argsSize,
                                           aclrtPlaceHolderInfo *placeHolderArray,
                                           size_t placeHolderNum);
```
- 功能：使用 Host 连续参数启动 SIMT Kernel
- 约束：本地 CANN 文档显示 SIMT launch 仅在部分产品支持；当前 `cudaLaunchKernel` 兼容入口优先使用参数数组方式
- 返回：ACL_SUCCESS 或错误码

### aclrtLaunchHostFunc
```c
aclError aclrtLaunchHostFunc(aclrtStream stream, aclrtHostFunc fn, void *args);
```
- 功能：在 stream 任务队列中插入 Host 回调任务，回调会阻塞本 stream 后续任务执行
- 约束：同一 stream 上不应混用 `aclrtLaunchHostFunc` 与 `aclrtLaunchCallback`；回调函数内不应执行资源申请/释放、同步或任务下发等可能导致死锁的操作
- 返回：ACL_SUCCESS 或错误码

---

## Graph/RI API

### aclmdlRIDebugJsonPrint
```c
aclError aclmdlRIDebugJsonPrint(aclmdlRI modelRI, const char *path, uint32_t flags);
```
- 功能：导出模型运行实例调试信息
- 约束：Model RI/ACL Graph 场景接口，主要用于维测和转测；输出格式为 JSON，不是 CUDA DOT 格式的完全等价输出
- 返回：ACL_SUCCESS 或错误码

### aclmdlRIDestroy
```c
aclError aclmdlRIDestroy(aclmdlRI modelRI);
```
- 功能：销毁模型运行实例
- 约束：用于释放 capture/build 得到的 Model RI；需与 capture/build 生命周期配对
- 返回：ACL_SUCCESS 或错误码

### aclmdlRIExecuteAsync
```c
aclError aclmdlRIExecuteAsync(aclmdlRI modelRI, aclrtStream stream);
```
- 功能：异步执行模型运行实例
- 约束：执行 capture/build 得到的 Model RI；需通过 stream/device 同步确认任务完成
- 返回：ACL_SUCCESS 或错误码

### aclmdlRICondHandleCreate
```c
aclError aclmdlRICondHandleCreate(aclmdlRI modelRI, uint32_t defaultLaunchValue,
                                  aclmdlRICondHandleFlag flag,
                                  aclmdlRICondHandle *handle);
```
- 功能：创建条件 handle
- 约束：本接口在 CANN 文档中标注为试验特性，后续版本可能会存在变更，不支持应用于生产环境中
- 返回：ACL_SUCCESS 或错误码

### aclmdlRIAddCondTask
```c
aclError aclmdlRIAddCondTask(aclmdlRICondTaskParams params, aclrtStream stream, uint32_t flags);
```
- 功能：向 capture active 状态的 ACL graph stream 注册 IF/WHILE/SWITCH 条件任务，并输出子 Model RI
- 约束：用于 `cudaGraphAddNode(cudaGraphNodeTypeConditional)` 特例映射；通用 CUDA Graph node 不按此接口映射
- 返回：ACL_SUCCESS 或错误码

### aclmdlRIGetStreams / aclmdlRIGetTasksByStream
```c
aclError aclmdlRIGetStreams(aclmdlRI modelRI, aclrtStream *streams, uint32_t *numStreams);
aclError aclmdlRIGetTasksByStream(aclrtStream stream, aclmdlRITask *tasks, uint32_t *numTasks);
```
- 功能：查询 RI 中的 stream 与 task，用于兼容 Graph node 查询
- 约束：这两个接口在 CANN 文档中标注为试验特性，后续版本可能会存在变更，不支持应用于生产环境中
- 返回：ACL_SUCCESS 或错误码

### aclmdlRICondHandleGetCondPtr
```c
aclError aclmdlRICondHandleGetCondPtr(aclmdlRICondHandle handle, uint64_t **ptr);
```
- 功能：获取条件 handle 内部条件变量的 Device 内存地址，用于写入条件值
- 约束：本接口在 CANN 文档中标注为试验特性，后续版本可能会存在变更，不支持应用于生产环境中
- 返回：ACL_SUCCESS 或错误码

### aclrtValueWrite
```c
aclError aclrtValueWrite(void *devAddr, uint64_t value, uint32_t flag, aclrtStream stream);
```
- 功能：在 stream 上写入 device value
- 返回：ACL_SUCCESS 或错误码

---

## Graph/Capture API

### aclmdlRICaptureBegin
```c
aclError aclmdlRICaptureBegin(aclrtStream stream, aclmdlRICaptureMode mode);
```
- 功能：开始流捕获
- 参数：mode - ACL_MODEL_RI_CAPTURE_MODE_GLOBAL/THREAD_LOCAL/RELAXED
- 约束：CANN 文档说明捕获期间对 Stream、Event、Device、Context 的同步或查询可能导致捕获失败；默认 stream 不建议用于 capture
- 返回：ACL_SUCCESS 或错误码

### aclmdlRICaptureEnd
```c
aclError aclmdlRICaptureEnd(aclrtStream stream, aclmdlRI *modelRI);
```
- 功能：结束流捕获
- 参数：modelRI - 模型实例输出
- 返回：ACL_SUCCESS 或错误码

### aclmdlRICaptureGetInfo
```c
aclError aclmdlRICaptureGetInfo(aclrtStream stream,
                                 aclmdlRICaptureStatus *status,
                                 aclmdlRI *modelRI);
```
- 功能：获取捕获信息
- 参数：status - 状态输出
- 约束：CANN 错误信息示例中说明 `status` 与 `modelRI` 不能同时为空；兼容层 `cudaStreamIsCapturing` 至少传入 status 输出
- 返回：ACL_SUCCESS 或错误码

### aclmdlRICaptureStatus
```c
typedef enum {
    ACL_MODEL_RI_CAPTURE_STATUS_NONE = 0,
    ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE,
    ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED,
} aclmdlRICaptureStatus;
```
- 功能：表示 CANN Model RI stream capture 状态
- CUDA 映射：分别对应 `cudaStreamCaptureStatusNone`、`cudaStreamCaptureStatusActive`、`cudaStreamCaptureStatusInvalidated`

### aclmdlRICaptureToModelRIBegin
```c
aclError aclmdlRICaptureToModelRIBegin(aclrtStream stream, aclmdlRI modelRI,
                                       aclmdlRICaptureMode mode);
```
- 功能：开始捕获并接入已有 Model RI
- 约束：本接口在 CANN 文档中标注为试验特性，后续版本可能会存在变更，不支持应用于生产环境中
- 返回：ACL_SUCCESS 或错误码

---

## 增量 Runtime/Driver API

| CANN API | 对应 CUDA API | 功能摘要 |
|---|---|---|
| `aclrtRecordEventWithFlag` | `cudaEventRecordWithFlags` | 带 flags 记录 event |
| `aclrtMallocHostAndRegister` / `aclrtMallocHost` | `cudaHostAlloc` | Host pinned 内存分配 |
| `aclrtGetSymbolAddress` | `cudaGetSymbolAddress` | 获取设备符号地址 |
| `aclrtMemcpyToSymbol` | `cudaMemcpyToSymbol` | 向设备符号拷贝 |
| `aclrtMemsetD32Async` | `cuMemsetD32Async` | D32 异步 memset |
| `aclrtGetFunctionAttribute` | `cudaFuncGetAttributes` | 查询 function 属性 |
| `aclrtGetCurrentContext` | `cuCtxGetCurrent` | 获取当前 context |
| `aclrtSetCurrentContext` | `cuCtxSetCurrent` | 设置当前 context |
| `aclrtGetPrimaryCtxState` | `cuDevicePrimaryCtxGetState` | 查询 primary context |
| `aclrtBinaryLoadFromFile` | `cuModuleLoad` | 从文件加载 binary |
| `aclrtBinaryLoadFromData` | `cuModuleLoadData` | 从内存加载 binary |
| `aclrtBinaryGetFunction` | `cuModuleGetFunction` | 获取 function handle |
| `aclrtBinaryUnLoad` | `cuModuleUnload` | 卸载 binary |


## 错误码参考

| ACL 错误码 | 数值 | 说明 |
|-----------|------|------|
| ACL_SUCCESS | 0 | 成功 |
| ACL_ERROR_RT_PARAM_INVALID | 107000 | 参数无效 |
| ACL_ERROR_RT_INVALID_DEVICEID | 107001 | 设备 ID 无效 |
| ACL_ERROR_RT_FEATURE_NOT_SUPPORT | 207000 | 功能不支持 |
| ACL_ERROR_RT_MEMORY_ALLOCATION | 207001 | 内存分配失败 |
| ACL_ERROR_RT_MEMCPY_FAILURE | 207002 | 内存拷贝失败 |
| ACL_ERROR_RT_STREAM_TIMEOUT | 507000 | 流超时 |
| ACL_ERROR_RT_EVENT_TIMEOUT | 507001 | 事件超时 |

---

## 类型映射参考

| CUDA 类型 | CANN 类型 |
|-----------|-----------|
| cudaStream_t | aclrtStream |
| cudaEvent_t | aclrtEvent |
| cudaMemcpyKind | aclrtMemcpyKind |
| cudaDeviceProp | 自定义结构（需映射） |

---

## 标志映射参考

| CUDA 标志 | CANN 标志 |
|-----------|-----------|
| cudaEventDisableTiming | 不设置 ACL_EVENT_TIME_LINE |
| cudaEventInterprocess | ACL_EVENT_IPC |
| cudaStreamNonBlocking | flag = 0 |
| cudaHostRegisterDefault | ACL_HOST_REG_PINNED |
| cudaHostRegisterMapped | ACL_HOST_REG_MAPPED + ACL_HOST_REG_PINNED |
| cudaHostRegisterIoMemory | ACL_HOST_REG_IOMEMORY + ACL_HOST_REG_PINNED |
| cudaHostRegisterReadOnly | ACL_HOST_REG_READONLY + ACL_HOST_REG_PINNED |
