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
aclError aclrtDeviceEnablePeerAccess(int32_t peerDeviceId);
```
- 功能：启用对 peerDevice 的直接访问
- 参数：peerDeviceId - 目标设备 ID
- 返回：ACL_SUCCESS 或错误码

### aclrtDeviceCanAccessPeer
```c
aclError aclrtDeviceCanAccessPeer(int32_t *canAccess, int32_t deviceId,
                                   int32_t peerDeviceId);
```
- 功能：检查设备间直接访问能力
- 参数：canAccess - 结果输出 (1/0)
- 返回：ACL_SUCCESS 或错误码

### aclrtDeviceGetStreamPriorityRange
```c
aclError aclrtDeviceGetStreamPriorityRange(int32_t *leastPriority,
                                            int32_t *greatestPriority);
```
- 功能：获取流优先级范围
- 参数：leastPriority - 最低优先级(数值大)，greatestPriority - 最高优先级(数值小)
- 返回：ACL_SUCCESS 或错误码

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
- 返回：ACL_SUCCESS（立即返回）

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

### aclrtMemExportToShareableHandleV2
```c
aclError aclrtMemExportToShareableHandleV2(void *shareableHandle,
                                             void *devPtr,
                                             aclrtMemShareableHandleType handleType);
```
- 功能：导出 IPC 内存句柄
- 参数：
  - shareableHandle - 句柄输出(64字节)
  - devPtr - 设备指针
  - handleType - ACL_MEM_SHAREABLE_HANDLE_TYPE_LOCAL
- 返回：ACL_SUCCESS 或错误码

### aclrtMemImportFromShareableHandleV2
```c
aclError aclrtMemImportFromShareableHandleV2(void **devPtr,
                                              void *shareableHandle,
                                              aclrtMemShareableHandleType handleType);
```
- 功能：导入 IPC 内存句柄
- 参数：devPtr - 指针输出，shareableHandle - IPC 句柄
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
aclError aclprofInit(void);
```
- 功能：初始化 profiler
- 返回：ACL_SUCCESS 或错误码

### aclprofStart
```c
aclError aclprofStart(void);
```
- 功能：启动性能数据收集
- 返回：ACL_SUCCESS 或错误码

### aclprofStop
```c
aclError aclprofStop(void);
```
- 功能：停止性能数据收集
- 返回：ACL_SUCCESS 或错误码

### aclprofFinalize
```c
aclError aclprofFinalize(void);
```
- 功能：释放 profiler 资源
- 返回：ACL_SUCCESS 或错误码

---

## Graph/Capture API

### aclmdlRICaptureBegin
```c
aclError aclmdlRICaptureBegin(aclrtStream stream, aclmdlRICaptureMode mode);
```
- 功能：开始流捕获
- 参数：mode - ACL_MODEL_RI_CAPTURE_MODE_GLOBAL/THREAD_LOCAL/RELAXED
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
- 返回：ACL_SUCCESS 或错误码

---

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
| cudaHostRegisterDefault | ACL_HOST_MEM_REGISTER_DEFAULT