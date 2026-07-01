# CUDA <-> CANN 错误码映射参考

## 映射表（按 ACL 错误码排序）

| ACL 错误码 | ACL 错误名 | CUDA 错误码 | CUDA 错误名 |
|-----------|-----------|-------------|-------------|
| 0 | ACL_SUCCESS | 0 | cudaSuccess |
| 100000 | ACL_ERROR_INVALID_PARAM | 1 | cudaErrorInvalidValue |
| 100001 | ACL_ERROR_UNINITIALIZED | 3 | cudaErrorInitializationError |
| 100002 | ACL_ERROR_REPEAT_INITIALIZE | 3 | cudaErrorInitializationError |
| 100003 | ACL_ERROR_READ_FILE_FAILURE | 4 | cudaErrorMemoryAllocation |
| 107000 | ACL_ERROR_RT_PARAM_INVALID | 1 | cudaErrorInvalidValue |
| 107001 | ACL_ERROR_RT_INVALID_DEVICEID | 10 | cudaErrorInvalidDevice |
| 107002 | ACL_ERROR_RT_INVALID_STREAM | 14 | cudaErrorInvalidResourceHandle |
| 107003 | ACL_ERROR_RT_INVALID_EVENT | 14 | cudaErrorInvalidResourceHandle |
| 107004 | ACL_ERROR_RT_INVALID_MEM | 14 | cudaErrorInvalidResourceHandle |
| 107005 | ACL_ERROR_RT_CONTEXT_MISMATCH | 8 | cudaErrorInvalidContext |
| 107006 | ACL_ERROR_RT_STREAM_TIMEOUT | 15 | cudaErrorInvalidResourceHandle |
| 107007 | ACL_ERROR_RT_EVENT_TIMEOUT | 15 | cudaErrorInvalidResourceHandle |
| 107008 | ACL_ERROR_RT_DEVICE_TIMEOUT | 6 | cudaErrorMissingConfiguration |
| 107010 | ACL_ERROR_RT_DEV_ALREADY_RUNNING | 1 | cudaErrorInvalidValue |
| 107011 | ACL_ERROR_RT_DEV_NOT_RUNNING | 8 | cudaErrorInvalidContext |
| 107012 | ACL_ERROR_RT_DEV_RESET_FAILED | 8 | cudaErrorInvalidContext |
| 107013 | ACL_ERROR_RT_REGISTER_LIB_FAILED | 999 | cudaErrorUnknown |
| 107014 | ACL_ERROR_RT_CREATE_PROFILER_FAILED | 999 | cudaErrorUnknown |
| 107015 | ACL_ERROR_RT_PROFILER_ALREADY_RUN | 1 | cudaErrorInvalidValue |
| 107016 | ACL_ERROR_RT_PROFILER_NOT_RUN | 1 | cudaErrorInvalidValue |
| 107017 | ACL_ERROR_RT_QUERY_PROFILER_FAILED | 999 | cudaErrorUnknown |
| 107018 | ACL_ERROR_RT_SUBSCRIBE_PROFILER_FAILED | 999 | cudaErrorUnknown |
| 107019 | ACL_ERROR_RT_ALLOC_MEM_FAILED | 4 | cudaErrorMemoryAllocation |
| 107020 | ACL_ERROR_RT_FREE_MEM_FAILED | 4 | cudaErrorMemoryAllocation |
| 107021 | ACL_ERROR_RT_ALLOC_HOST_MEM_FAILED | 4 | cudaErrorMemoryAllocation |
| 107022 | ACL_ERROR_RT_FREE_HOST_MEM_FAILED | 4 | cudaErrorMemoryAllocation |
| 107023 | ACL_ERROR_RT_MALLOC_MEM_FAILED | 4 | cudaErrorMemoryAllocation |
| 107024 | ACL_ERROR_RT_MEMCPY_FAILED | 1 | cudaErrorInvalidValue |
| 107025 | ACL_ERROR_RT_MEMSET_FAILED | 1 | cudaErrorInvalidValue |
| 107026 | ACL_ERROR_RT_BIND_HANDLE_FAILED | 999 | cudaErrorUnknown |
| 107027 | ACL_ERROR_RT_DEV_TYPE_NOT_MATCH | 10 | cudaErrorInvalidDevice |
| 107028 | ACL_ERROR_RT_BIN_FILE_INVALID | 999 | cudaErrorUnknown |
| 107029 | ACL_ERROR_RT_MULTI_DEVICE_NOT_SUPPORT | 801 | cudaErrorNotSupported |
| 107030 | ACL_ERROR_RT_STREAM_NO_CB | 999 | cudaErrorUnknown |
| 107031 | ACL_ERROR_RT_EVENT_NO_CB | 999 | cudaErrorUnknown |
| 107032 | ACL_ERROR_RT_CB_IS_NULL | 1 | cudaErrorInvalidValue |
| 107033 | ACL_ERROR_RT_BIN_FILE_VER_MISMATCH | 999 | cudaErrorUnknown |
| 107034 | ACL_ERROR_RT_FEATURE_NOT_SUPPORT | 801 | cudaErrorNotSupported |
| 107035 | ACL_ERROR_RT_MEMORY_ALLOCATION | 4 | cudaErrorMemoryAllocation |
| 107036 | ACL_ERROR_RT_MEMCPY_FAILURE | 1 | cudaErrorInvalidValue |
| 107037 | ACL_ERROR_RT_INTERNAL_ERROR | 999 | cudaErrorUnknown |
| 107038 | ACL_ERROR_RT_ACL_PROFILER_NOT_INIT | 999 | cudaErrorUnknown |
| 107039 | ACL_ERROR_RT_PROCESS_TERMINATED | 999 | cudaErrorUnknown |
| 107040 | ACL_ERROR_RT_LAUNCH_FAILED | 999 | cudaErrorUnknown |
| 107041 | ACL_ERROR_RT_BIN_FILE_PATH_INVALID | 999 | cudaErrorUnknown |
| 107042 | ACL_ERROR_RT_LOAD_BINARY_FAILED | 999 | cudaErrorUnknown |
| 107043 | ACL_ERROR_RT_LAUNCH_BIN_FILE_FAILED | 999 | cudaErrorUnknown |
| 107044 | ACL_ERROR_RT_LIBRARY_NOT_FOUND | 999 | cudaErrorUnknown |
| 107045 | ACL_ERROR_RT_SYMBOL_NOT_FOUND | 999 | cudaErrorUnknown |
| 107046 | ACL_ERROR_RT_STREAM_DESTROY_FAILED | 999 | cudaErrorUnknown |
| 107047 | ACL_ERROR_RT_EVENT_DESTROY_FAILED | 999 | cudaErrorUnknown |
| 107048 | ACL_ERROR_RT_VMM_NOT_SUPPORT | 801 | cudaErrorNotSupported |
| 107049 | ACL_ERROR_RT_VMM_MAP_FAILED | 4 | cudaErrorMemoryAllocation |
| 107050 | ACL_ERROR_RT_VMM_UNMAP_FAILED | 4 | cudaErrorMemoryAllocation |
| 107051 | ACL_ERROR_RT_VMM_MALLOC_FAILED | 4 | cudaErrorMemoryAllocation |
| 107052 | ACL_ERROR_RT_VMM_FREE_FAILED | 4 | cudaErrorMemoryAllocation |
| 207000 | ACL_ERROR_RT_FEATURE_NOT_SUPPORT | 801 | cudaErrorNotSupported |
| 207001 | ACL_ERROR_RT_MEMORY_ALLOCATION | 4 | cudaErrorMemoryAllocation |
| 207002 | ACL_ERROR_RT_MEMCPY_FAILURE | 1 | cudaErrorInvalidValue |
| 507000 | ACL_ERROR_RT_STREAM_TIMEOUT | 6 | cudaErrorMissingConfiguration |
| 507001 | ACL_ERROR_RT_EVENT_TIMEOUT | 6 | cudaErrorMissingConfiguration |
| 507002 | ACL_ERROR_RT_DEVICE_TIMEOUT | 6 | cudaErrorMissingConfiguration |
| 507003 | ACL_ERROR_RT_KERNEL_TIMEOUT | 6 | cudaErrorMissingConfiguration |
| 507004 | ACL_ERROR_RT_OP_TIMEOUT | 6 | cudaErrorMissingConfiguration |

## CUDA 错误码参考

| CUDA 错误码 | 数值 | 说明 |
|-------------|------|------|
| cudaSuccess | 0 | 成功 |
| cudaErrorInvalidValue | 1 | 无效参数 |
| cudaErrorMemoryAllocation | 4 | 内存分配失败 |
| cudaErrorInitializationError | 3 | 初始化错误 |
| cudaErrorInvalidContext | 8 | 无效上下文 |
| cudaErrorInvalidDevice | 10 | 无效设备 |
| cudaErrorInvalidResourceHandle | 14 | 无效资源句柄 |
| cudaErrorMissingConfiguration | 6 | 缺少配置 |
| cudaErrorNotSupported | 801 | 功能不支持 |
| cudaErrorUnknown | 999 | 未知错误 |
| cudaErrorNotReady | 600 | 未就绪（异步查询） |

## 映射实现参考

### 二分查找实现

```c
typedef struct {
    int aclCode;
    cudaError_t cudaCode;
} acl2cudaMap_t;

static const acl2cudaMap_t g_acl2cudaTable[] = {
    {0, cudaSuccess},
    {100000, cudaErrorInvalidValue},
    {100001, cudaErrorInitializationError},
    {107000, cudaErrorInvalidValue},
    {107001, cudaErrorInvalidDevice},
    // ... 按 aclCode 排序
};

static const size_t g_acl2cudaTableSize = sizeof(g_acl2cudaTable) / sizeof(g_acl2cudaTable[0]);

static cudaError_t acl2cudaErrorLookup(int aclErr) {
    int left = 0, right = g_acl2cudaTableSize - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (g_acl2cudaTable[mid].aclCode == aclErr) {
            return g_acl2cudaTable[mid].cudaCode;
        }
        if (g_acl2cudaTable[mid].aclCode < aclErr) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return cudaErrorUnknown;
}
```

## 特殊错误处理

### 流/事件查询

- `ACL_STREAM_STATUS_NOT_READY` → `cudaErrorNotReady`
- `ACL_EVENT_RECORDED_STATUS_NOT_READY` → `cudaErrorNotReady`

### 不支持功能

- `ACL_ERROR_RT_FEATURE_NOT_SUPPORT` → `cudaErrorNotSupported`
- `ACL_ERROR_RT_MULTI_DEVICE_NOT_SUPPORT` → `cudaErrorNotSupported`
- `ACL_ERROR_RT_VMM_NOT_SUPPORT` → `cudaErrorNotSupported`

## 错误字符串映射

### CUDA 错误名称

| 错误码 | 名称字符串 |
|--------|-----------|
| cudaSuccess | "cudaSuccess" |
| cudaErrorInvalidValue | "cudaErrorInvalidValue" |
| cudaErrorMemoryAllocation | "cudaErrorMemoryAllocation" |
| cudaErrorNotReady | "cudaErrorNotReady" |
| cudaErrorNotSupported | "cudaErrorNotSupported" |
| cudaErrorUnknown | "cudaErrorUnknown" |

### CUDA 错误描述

| 错误码 | 描述字符串 |
|--------|-----------|
| cudaSuccess | "no error" |
| cudaErrorInvalidValue | "invalid argument" |
| cudaErrorMemoryAllocation | "out of memory" |
| cudaErrorNotReady | "device not ready" |
| cudaErrorNotSupported | "operation not supported" |
| cudaErrorUnknown | "unknown error" |