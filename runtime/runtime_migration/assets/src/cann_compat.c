/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */


/*
 * Project-local CUDA-to-CANN compatibility shim implementation.
 *
 * The implementation maps migrated sample calls onto CANN runtime APIs. Keep
 * this source free of vendor SDK header prose and copied documentation.
 */

#include "cann_compat_types.h"
#include "cann_compat_device.h"
#include "cann_compat_memory.h"
#include "cann_compat_symbol.h"
#include "acl/acl_prof.h"
#include "runtime/rt_external_kernel.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_COMPAT_MAX_MEMCPY_BATCH_ATTRS 1024U

cudaError_t cudaCompatRegisterSymbol(void *binHandle, const void *hostVar,
                                     const char *deviceVarName, size_t size,
                                     unsigned int flags)
{
    if (!binHandle || !hostVar || !deviceVarName || size == 0) {
        return cudaErrorInvalidValue;
    }
    rtRegisterVariable(binHandle, hostVar, deviceVarName, size, (uint32_t)flags, NULL);
    return cudaSuccess;
}

#ifdef CUDA_COMPAT_DEBUG_MODE
#define CUDA_COMPAT_DEBUG_LOG(acl_err) \
    do { \
        if (acl_err != ACL_SUCCESS) { \
            const char *errMsg = aclGetRecentErrMsg(); \
            fprintf(stderr, "[CUDA_COMPAT_DEBUG] %s:%d: %s failed with ACL error: %d\n", \
                    __FILE__, __LINE__, acl_err); \
            if (errMsg && errMsg[0] != '\0') { \
                fprintf(stderr, "[CUDA_COMPAT_DEBUG] %s:%d: Error message: %s\n", \
                        __FILE__, __LINE__, errMsg); \
            } \
        } \
    } while(0)
#else
#define CUDA_COMPAT_DEBUG_LOG(acl_err) ((void)0)
#endif

static size_t cudaCompatStringEnd(const char *dst, size_t dstSize)
{
    size_t pos = 0;
    while (pos < dstSize && dst[pos] != '\0')
    {
        pos++;
    }
    return pos;
}

static void cudaCompatAppendString(char *dst, size_t dstSize, const char *src)
{
    if (dst == NULL || dstSize == 0 || src == NULL)
    {
        return;
    }

    size_t pos = cudaCompatStringEnd(dst, dstSize);
    while (pos + 1 < dstSize && *src != '\0')
    {
        dst[pos++] = *src++;
    }
    dst[pos < dstSize ? pos : dstSize - 1] = '\0';
}

static void cudaCompatCopyString(char *dst, size_t dstSize, const char *src)
{
    if (dst == NULL || dstSize == 0)
    {
        return;
    }

    dst[0] = '\0';
    cudaCompatAppendString(dst, dstSize, src);
}

static void cudaCompatAppendLong(char *dst, size_t dstSize, long value)
{
    char digits[32];
    size_t count = 0;
    unsigned long magnitude;

    if (value < 0)
    {
        cudaCompatAppendString(dst, dstSize, "-");
        magnitude = (unsigned long)(-(value + 1)) + 1;
    }
    else
    {
        magnitude = (unsigned long)value;
    }

    do
    {
        digits[count++] = (char)('0' + (magnitude % 10));
        magnitude /= 10;
    } while (magnitude != 0 && count < sizeof(digits));

    while (count > 0)
    {
        char oneChar[2] = {digits[--count], '\0'};
        cudaCompatAppendString(dst, dstSize, oneChar);
    }
}

/* =================================================================
 * Global State
 * ================================================================= */
cudaCompatContext_t g_cuda_context = {
    .initialized = 0,
    .flags = cudaDeviceScheduleAuto,
    .profiler_initialized = 0,
    .profiler_running = 0};

/* =================================================================
 * Initialization
 * ================================================================= */

static const char *cuda_compat_get_config_path(const char *configPath)
{
    if (configPath != NULL)
    {
        return configPath;
    }
    return getenv("CUDA_COMPAT_INIT_FILE_PATH");
}

static cudaError_t cudaCompatInit(const char *configPath)
{
    if (g_cuda_context.initialized)
    {
        return cudaSuccess;
    }

    const char *effectivePath = cuda_compat_get_config_path(configPath);

    aclError ret = aclInit(effectivePath);
    CUDA_COMPAT_DEBUG_LOG(ret);
    if (ret != ACL_SUCCESS)
    {
        return acl2cudaError(ret);
    }

    g_cuda_context.initialized = 1;
    return cudaSuccess;
}

static cudaError_t cudaCompatFinalize(void)
{
    if (!g_cuda_context.initialized)
    {
        return cudaSuccess;
    }

    if (g_cuda_context.profiler_initialized)
    {
        if (g_cuda_context.profiler_running)
        {
            aclprofStop(NULL);
            g_cuda_context.profiler_running = 0;
        }
        aclprofFinalize();
        g_cuda_context.profiler_initialized = 0;
    }

    aclFinalize();

    g_cuda_context.initialized = 0;
    return cudaSuccess;
}

/* =================================================================
 * Auto Init / Finalize (constructor / destructor)
 * ================================================================= */

__attribute__((constructor)) static void cuda_compat_auto_init(void)
{
    cudaCompatInit(NULL);
}

__attribute__((destructor)) static void cuda_compat_auto_fini(void)
{
    cudaCompatFinalize();
}

/* =================================================================
 * Force Link Guard for Static Linking
 * ================================================================= */

const int _cuda_compat_force_link = 0;

/* =================================================================
 * Device Management Implementation
 * ================================================================= */

static void cudaCompatSetCoreProperties(cudaDeviceProp *prop, int device)
{
    int64_t value = 0;
    aclrtGetDeviceInfo(device, ACL_DEV_ATTR_AICORE_CORE_NUM, &value);
    prop->sharedMemPerBlock = value * 1024; // Approximation
    prop->multiProcessorCount = (int)value;
    prop->regsPerBlock = 65536; // Default value
    prop->maxThreadsDim[0] = value;
    prop->maxThreadsDim[1] = 1;
    prop->maxThreadsDim[2] = 1;
    prop->maxGridSize[0] = 65535;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
}


static void cudaCompatSetMemoryProperties(cudaDeviceProp *prop, int device)
{
    int64_t value = 0;
    aclError ret = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_WARP_SIZE, &value);
    prop->warpSize = ret == ACL_SUCCESS ? (int)value : 0;

    ret = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE, &value);
    prop->maxThreadsPerBlock = ret == ACL_SUCCESS ? (int)value : 0;

    ret = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_TOTAL_GLOBAL_MEM_SIZE, &value);
    prop->totalGlobalMem = ret == ACL_SUCCESS ? (size_t)value : 0;

    ret = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_L2_CACHE_SIZE, &value);
    prop->l2CacheSize = ret == ACL_SUCCESS ? (int)value : 0;
    prop->persistingL2CacheMaxSize = prop->l2CacheSize; // Assume all L2 can be used for persistence
}


static void cudaCompatSetDeviceName(cudaDeviceProp *prop, int device)
{
    const char *socName = aclrtGetSocName();
    if (socName && socName[0] != '\0')
    {
        int64_t aicoreNum = 0;
        aclError ret = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_AICORE_CORE_NUM, &aicoreNum);

        if (ret == ACL_SUCCESS)
        {
            // Assemble device name as {socName}-{aicoreNum}AIC
            cudaCompatCopyString(prop->name, sizeof(prop->name), socName);
            cudaCompatAppendString(prop->name, sizeof(prop->name), "-");
            cudaCompatAppendLong(prop->name, sizeof(prop->name), (long)aicoreNum);
            cudaCompatAppendString(prop->name, sizeof(prop->name), "AIC");
        }
        else
        {
            // Fallback to just SoC name if getting AI core number fails
            cudaCompatCopyString(prop->name, sizeof(prop->name), socName);
        }
    }
    else
    {
        // Fallback to default name if getting SoC name fails
        cudaCompatCopyString(prop->name, sizeof(prop->name), "ASCEND");
    }
}


static void cudaCompatSetFeatureProperties(cudaDeviceProp *prop)
{
    prop->streamPrioritiesSupported = 1;   // CANN supports stream priorities
    prop->major = MOCK_CUDA_MAJOR_VERSION; // mock cuda version
    prop->minor = MOCK_CUDA_MINOR_VERSION;
    prop->integrated = 1;                    // NPU is typically integrated
    prop->canMapHostMemory = 1;              // CANN supports host memory mapping
    prop->hostRegisterSupported = 1;         // CANN supports host memory registration
    prop->hostRegisterReadOnlySupported = 1; // CANN supports read-only host memory registration
    prop->ipcEventSupported = 1;             // CANN supports IPC events
    prop->managedMemory = 0;                 // CANN does not support unified memory
    prop->concurrentManagedAccess = 0;       // Not applicable
}


cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device)
{
    if (!prop)
    {
        return cudaErrorInvalidValue;
    }

    *prop = (cudaDeviceProp){0};
    cudaCompatSetCoreProperties(prop, device);
    cudaCompatSetMemoryProperties(prop, device);
    cudaCompatSetDeviceName(prop, device);
    cudaCompatSetFeatureProperties(prop);

    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device)
{
    if (!value)
    {
        return cudaErrorInvalidValue;
    }

    (void)(device);  // device parameter currently ignored

    switch (attr)
    {
        /* =================================================================
         * Compute Capability (Mock Values)
         * ================================================================= */
        case cudaDevAttrComputeCapabilityMajor:
            *value = MOCK_CUDA_MAJOR_VERSION;
            break;
        case cudaDevAttrComputeCapabilityMinor:
            *value = MOCK_CUDA_MINOR_VERSION;
            break;

        /* =================================================================
         * Compute Mode (Mock: Default - multiple threads can use device)
         * ================================================================= */
        case cudaDevAttrComputeMode:
            *value = cudaComputeModeDefault;
            break;

        /* =================================================================
         * Default fallback for other attributes
         * ================================================================= */
        default:
            *value = 0;
            break;
    }

    return cudaSuccess;
}

cudaError_t cudaDeviceGetLimit(size_t *pValue, cudaLimit limit)
{
    if (!pValue)
    {
        return cudaErrorInvalidValue;
    }

    // Return default values
    switch (limit)
    {
    case cudaLimitStackSize:
        *pValue = 1024;
        break;
    case cudaLimitMallocHeapSize:
        *pValue = 8 * 1024 * 1024;
        break;
    case cudaLimitPrintfFifoSize:
        *pValue = 1024 * 1024;
        break;
    default:
        *pValue = 0;
        break;
    }
    return cudaSuccess;
}

/* =================================================================
 * Memory Management Implementation
 * ================================================================= */

cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch,
                            size_t width, size_t height)
{
    if (width == 0 || height == 0)
    {
        *devPtr = NULL;
        *pitch = 0;
        return cudaSuccess;
    }

    const size_t alignment = 64; // CANN may require specific alignment
    *pitch = ((width + alignment - 1) / alignment) * alignment;

    size_t total_size = *pitch * height;
    return cudaMalloc(devPtr, total_size);
}

cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attributes,
                                     const void *ptr)
{
    if (!attributes || !ptr)
    {
        return cudaErrorInvalidValue;
    }

    *attributes = (cudaPointerAttributes){0};

    // Query pointer attributes from CANN
    aclrtPtrAttributes attr;
    aclError ret = aclrtPointerGetAttributes(ptr, &attr);
    if (ret != ACL_SUCCESS)
    {
        return acl2cudaError(ret);
    }

    switch (attr.location.type)
    {
    case ACL_MEM_LOCATION_TYPE_HOST:
        attributes->type = cudaMemoryTypeHost;
        attributes->hostPointer = (void *)ptr;
        break;
    case ACL_MEM_LOCATION_TYPE_DEVICE:
        attributes->type = cudaMemoryTypeDevice;
        attributes->device = attr.location.id;
        attributes->devicePointer = (void *)ptr;
        break;
    case ACL_MEM_LOCATION_TYPE_HOST_NUMA:
        attributes->type = cudaMemoryTypeHost;
        attributes->hostPointer = (void *)ptr;
        break;
    case ACL_MEM_LOCATION_TYPE_UNREGISTERED:
        attributes->type = cudaMemoryTypeUnregistered;
        break;
    }
    return cudaSuccess;
}

static const aclrtMemLocationType g_cuda2aclMemType[] = {
    ACL_MEM_LOCATION_TYPE_UNREGISTERED, // cudaMemLocationTypeInvalid
    ACL_MEM_LOCATION_TYPE_DEVICE,       // cudaMemLocationTypeDevice
    ACL_MEM_LOCATION_TYPE_HOST,         // cudaMemLocationTypeHost
    ACL_MEM_LOCATION_TYPE_HOST,         // cudaMemLocationTypeHostNuma
    ACL_MEM_LOCATION_TYPE_HOST          // cudaMemLocationTypeHostNumaCurrent
};

cudaError_t cudaMemcpyBatchAsync(const void **dsts, const void **srcs, const size_t *sizes, size_t count,
                                 cudaMemcpyAttributes *attrs, size_t *attrsIdxs, size_t numAttrs, cudaStream_t stream)
{
    if (aclrtMemcpyBatchAsyncV2 == NULL)
    {
        return cudaErrorNotSupported;
    }

    if (!dsts || !srcs || !sizes || !attrs || !attrsIdxs)
    {
        return cudaErrorInvalidValue;
    }
    if (numAttrs == 0 || numAttrs > CUDA_COMPAT_MAX_MEMCPY_BATCH_ATTRS ||
        numAttrs > SIZE_MAX / sizeof(aclrtMemcpyBatchAttr))
    {
        return cudaErrorInvalidValue;
    }

    aclrtMemcpyBatchAttr *cannAttrs = (aclrtMemcpyBatchAttr *)calloc(numAttrs, sizeof(aclrtMemcpyBatchAttr));
    if (!cannAttrs)
    {
        return cudaErrorMemoryAllocation;
    }
    /* Process each batch operation */
    for (size_t i = 0; i < numAttrs; i++)
    {
        if (attrs[i].srcLocHint.type >= sizeof(g_cuda2aclMemType) / sizeof(g_cuda2aclMemType[0]) ||
            attrs[i].dstLocHint.type >= sizeof(g_cuda2aclMemType) / sizeof(g_cuda2aclMemType[0]))
        {
            free(cannAttrs);
            return cudaErrorInvalidValue;
        }
        cannAttrs[i].srcLoc.id = (uint32_t)attrs[i].srcLocHint.id;
        cannAttrs[i].srcLoc.type = g_cuda2aclMemType[attrs[i].srcLocHint.type];
        cannAttrs[i].dstLoc.id = (uint32_t)attrs[i].dstLocHint.id;
        cannAttrs[i].dstLoc.type = g_cuda2aclMemType[attrs[i].dstLocHint.type];
    }

    aclError ret = aclrtMemcpyBatchAsyncV2((void **)dsts, (size_t *)sizes, (void **)srcs, (size_t *)sizes, count,
                                         cannAttrs, attrsIdxs, numAttrs, stream);
    free(cannAttrs);
    return acl2cudaError(ret);

}

/* =================================================================
 * Error Handling
 * ================================================================= */


typedef struct acl2cudaMap
{
    int aclCode;
    int cudaCode;
} acl2cudaMap_t;


static const acl2cudaMap_t g_acl2cudaTable[] = {
    // Success
    {0, cudaSuccess}, // ACL_RT_SUCCESS -> cudaSuccess
    {100000, cudaErrorInvalidValue},

    // Parameter and validation errors (107xxx series)
    {107000, cudaErrorInvalidValue},             // ACL_ERROR_RT_PARAM_INVALID
    {107001, cudaErrorInvalidDevice},            // ACL_ERROR_RT_INVALID_DEVICEID
    {107002, cudaErrorInvalidValue},             // ACL_ERROR_RT_CONTEXT_NULL
    {107003, cudaErrorInvalidValue},             // ACL_ERROR_RT_STREAM_CONTEXT
    {107004, cudaErrorInvalidValue},             // ACL_ERROR_RT_MODEL_CONTEXT
    {107005, cudaErrorInvalidValue},             // ACL_ERROR_RT_STREAM_MODEL
    {107006, cudaErrorInvalidValue},             // ACL_ERROR_RT_EVENT_TIMESTAMP_INVALID
    {107007, cudaErrorInvalidValue},             // ACL_ERROR_RT_EVENT_TIMESTAMP_REVERSAL
    {107008, cudaErrorMisalignedAddress},        // ACL_ERROR_RT_ADDR_UNALIGNED
    {107016, cudaErrorInvalidValue},             // ACL_ERROR_RT_INVALID_MEMORY_TYPE
    {107017, cudaErrorInvalidResourceHandle},    // ACL_ERROR_RT_INVALID_HANDLE
    {107018, cudaErrorInvalidValue},             // ACL_ERROR_RT_INVALID_MALLOC_TYPE
    {107019, cudaErrorLaunchTimeout},            // ACL_ERROR_RT_WAIT_TIMEOUT
    {107020, cudaErrorLaunchTimeout},            // ACL_ERROR_RT_TASK_TIMEOUT
    {107021, cudaErrorInvalidValue},             // ACL_ERROR_RT_SYSPARAMOPT_NOT_SET
    {107022, cudaErrorLaunchFailure},            // ACL_ERROR_RT_DEVICE_TASK_ABORT
    {107023, cudaErrorLaunchFailure},            // ACL_ERROR_RT_STREAM_ABORT
    {107024, cudaErrorStreamCaptureUnsupported}, // ACL_ERROR_RT_CAPTURE_DEPENDENCY
    {107025, cudaErrorStreamCaptureUnjoined},    // ACL_ERROR_RT_STREAM_UNJOINED
    {107026, cudaErrorCapturedEvent},            // ACL_ERROR_RT_MODEL_CAPTURED
    {107027, cudaErrorStreamCaptureUnsupported}, // ACL_ERROR_RT_STREAM_CAPTURED
    {107028, cudaErrorCapturedEvent},            // ACL_ERROR_RT_EVENT_CAPTURED
    {107029, cudaErrorStreamCaptureUnsupported}, // ACL_ERROR_RT_STREAM_NOT_CAPTURED
    {107030, cudaErrorStreamCaptureUnsupported}, // ACL_ERROR_RT_CAPTURE_MODE_NOT_SUPPORT
    {107031, cudaErrorStreamCaptureImplicit},    // ACL_ERROR_RT_STREAM_CAPTURE_IMPLICIT
    {107032, cudaErrorStreamCaptureMerge},       // ACL_ERROR_STREAM_CAPTURE_CONFLICT
    {107035, cudaErrorLaunchFailure},            // ACL_ERROR_RT_TASK_ABORT_STOP
    {107036, cudaErrorStreamCaptureUnmatched},   // ACL_ERROR_RT_STREAM_CAPTURE_UNMATCHED
    {107037, cudaErrorNotReady},                 // ACL_ERROR_RT_MODEL_RUNNING
    {107038, cudaErrorStreamCaptureWrongThread}, // ACL_ERROR_RT_STREAM_CAPTURE_WRONG_THREAD

    // Feature and resource errors (207xxx series)
    {207000, cudaErrorNotSupported},         // ACL_ERROR_RT_FEATURE_NOT_SUPPORT
    {207001, cudaErrorMemoryAllocation},     // ACL_ERROR_RT_MEMORY_ALLOCATION
    {207002, cudaErrorMemoryAllocation},     // ACL_ERROR_RT_MEMORY_FREE
    {207003, cudaErrorLaunchOutOfResources}, // ACL_ERROR_RT_AICORE_OVERFLOW
    {207004, cudaErrorNoDevice},             // ACL_ERROR_RT_NO_DEVICE
    {207005, cudaErrorLaunchOutOfResources}, // ACL_ERROR_RT_RESOURCE_ALLOC_FAIL
    {207006, cudaErrorNotPermitted},         // ACL_ERROR_RT_NO_PERMISSION
    {207007, cudaErrorLaunchOutOfResources}, // ACL_ERROR_RT_NO_EVENT_RESOURCE
    {207008, cudaErrorLaunchOutOfResources}, // ACL_ERROR_RT_NO_STREAM_RESOURCE
    {207009, cudaErrorLaunchOutOfResources}, // ACL_ERROR_RT_NO_NOTIFY_RESOURCE
    {207010, cudaErrorLaunchOutOfResources}, // ACL_ERROR_RT_NO_MODEL_RESOURCE
    {207011, cudaErrorLaunchOutOfResources}, // ACL_ERROR_RT_NO_CDQ_RESOURCE
    {207012, cudaErrorLaunchOutOfResources}, // ACL_ERROR_RT_OVER_LIMIT
    {207013, cudaErrorLaunchOutOfResources}, // ACL_ERROR_RT_QUEUE_EMPTY
    {207014, cudaErrorLaunchOutOfResources}, // ACL_ERROR_RT_QUEUE_FULL
    {207016, cudaErrorLaunchOutOfResources}, // ACL_ERROR_RT_AIVEC_OVERFLOW
    {207017, cudaErrorLaunchOutOfResources}, // ACL_ERROR_RT_OVERFLOW
    {207018, cudaErrorLaunchOutOfResources}, // ACL_ERROR_RT_DEVICE_OOM
    {207019, cudaErrorNotSupported},         // ACL_ERROR_RT_FEATURE_NOT_SUPPORT_UPDATE_OP

    // Internal and runtime errors (507xxx series)
    {507001, cudaErrorSystemDriverMismatch},        // ACL_ERROR_RT_TS_ERROR
    {507002, cudaErrorLaunchOutOfResources},        // ACL_ERROR_RT_STREAM_TASK_FULL
    {507003, cudaErrorLaunchOutOfResources},        // ACL_ERROR_RT_STREAM_TASK_EMPTY
    {507004, cudaErrorNotReady},                    // ACL_ERROR_RT_STREAM_NOT_COMPLETE
    {507006, cudaErrorNotReady},                    // ACL_ERROR_RT_EVENT_NOT_COMPLETE
    {507008, cudaErrorSystemDriverMismatch},        // ACL_ERROR_RT_SOC_VERSION
    {507009, cudaErrorNotSupported},                // ACL_ERROR_RT_TASK_TYPE_NOT_SUPPORT
    {507010, cudaErrorSystemNotReady},              // ACL_ERROR_RT_LOST_HEARTBEAT
    {507011, cudaErrorLaunchFailure},               // ACL_ERROR_RT_MODEL_EXECUTE
    {507012, cudaErrorTimeout},                     // ACL_ERROR_RT_REPORT_TIMEOUT
    {507014, cudaErrorLaunchTimeout},               // ACL_ERROR_RT_AICORE_TIMEOUT
    {507015, cudaErrorLaunchFailure},               // ACL_ERROR_RT_AICORE_EXCEPTION
    {507016, cudaErrorHardwareStackError},          // ACL_ERROR_RT_AICORE_TRAP_EXCEPTION
    {507017, cudaErrorLaunchTimeout},               // ACL_ERROR_RT_AICPU_TIMEOUT
    {507018, cudaErrorLaunchFailure},               // ACL_ERROR_RT_AICPU_EXCEPTION
    {507019, cudaErrorSystemDriverMismatch},        // ACL_ERROR_RT_AICPU_DATADUMP_RSP_ERR
    {507020, cudaErrorSystemDriverMismatch},        // ACL_ERROR_RT_AICPU_MODEL_RSP_ERR
    {507023, cudaErrorLaunchFailure},               // ACL_ERROR_RT_MODEL_ABORT_NORMAL
    {507032, cudaErrorLaunchOutOfResources},        // ACL_ERROR_RT_PROGRAM_USE_OUT
    {507034, cudaErrorLaunchTimeout},               // ACL_ERROR_RT_VECTOR_CORE_TIMEOUT
    {507035, cudaErrorLaunchFailure},               // ACL_ERROR_RT_VECTOR_CORE_EXCEPTION
    {507036, cudaErrorHardwareStackError},          // ACL_ERROR_RT_VECTOR_CORE_TRAP_EXCEPTION
    {507040, cudaErrorInvalidDevice},               // ACL_ERROR_RT_INVALID_DIEID
    {507042, cudaErrorHardwareStackError},          // ACL_ERROR_RT_AICORE_TRAP_READ_OVERFLOW
    {507043, cudaErrorHardwareStackError},          // ACL_ERROR_RT_AICORE_TRAP_WRITE_OVERFLOW
    {507044, cudaErrorHardwareStackError},          // ACL_ERROR_RT_VECTOR_CORE_TRAP_READ_OVERFLOW
    {507045, cudaErrorHardwareStackError},          // ACL_ERROR_RT_VECTOR_CORE_TRAP_WRITE_OVERFLOW
    {507046, cudaErrorLaunchTimeout},               // ACL_ERROR_RT_STREAM_SYNC_TIMEOUT
    {507047, cudaErrorLaunchTimeout},               // ACL_ERROR_RT_EVENT_SYNC_TIMEOUT
    {507048, cudaErrorLaunchTimeout},               // ACL_ERROR_RT_FFTS_PLUS_TIMEOUT
    {507049, cudaErrorLaunchFailure},               // ACL_ERROR_RT_FFTS_PLUS_EXCEPTION
    {507050, cudaErrorHardwareStackError},          // ACL_ERROR_RT_FFTS_PLUS_TRAP_EXCEPTION
    {507053, cudaErrorHardwareStackError},          // ACL_ERROR_RT_DEVICE_MEM_ERROR
    {507054, cudaErrorECCUncorrectable},            // ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR
    {507055, cudaErrorHardwareStackError},          // ACL_ERROR_RT_SUSPECT_DEVICE_MEM_ERROR
    {507902, cudaErrorSystemDriverMismatch},        // ACL_ERROR_RT_AICPU_INFO_LOAD_RSP_ERR
    {507903, cudaErrorStreamCaptureInvalidated},    // ACL_ERROR_RT_STREAM_CAPTURE_INVALIDATED
    {507905, cudaErrorTimeout},                     // ACL_ERROR_SNAPSHOT_LOCK_TIMEOUT
    {507910, cudaErrorHostMemoryAlreadyRegistered}, // ACL_ERROR_HOST_MEMORY_ALREADY_REGISTERED
    {507911, cudaErrorHostMemoryNotRegistered},     // ACL_ERROR_HOST_MEMORY_NOT_REGISTERED
};

static const int g_acl2cudaTableSize = sizeof(g_acl2cudaTable) / sizeof(g_acl2cudaTable[0]);


static cudaError_t acl2cudaErrorLookup(int aclErr)
{
    int left = 0;
    int right = g_acl2cudaTableSize - 1;

    while (left <= right)
    {
        int mid = left + (right - left) / 2;
        if (g_acl2cudaTable[mid].aclCode == aclErr)
        {
            return (cudaError_t)g_acl2cudaTable[mid].cudaCode;
        }
        else if (g_acl2cudaTable[mid].aclCode < aclErr)
        {
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }

    // Default to cudaErrorUnknown for unmapped error codes
    return cudaErrorUnknown;
}

cudaError_t acl2cudaError(aclError err)
{
#ifdef CUDA_COMPAT_DEBUG_MODE
    if (err != ACL_SUCCESS)
    {
        const char *errMsg = aclGetRecentErrMsg();
        fprintf(stderr, "[CUDA_COMPAT_DEBUG] ACL error: %d\n", err);
        if (errMsg && errMsg[0] != '\0')
        {
            fprintf(stderr, "[CUDA_COMPAT_DEBUG] ACL error message: %s\n", errMsg);
        }
        else
        {
            fprintf(stderr, "[CUDA_COMPAT_DEBUG] No detailed error message available\n");
        }
    }
#endif
    return acl2cudaErrorLookup(err);
}


typedef struct cudaErrorInfo
{
    int code;
    const char *name;
    const char *string;
} cudaErrorInfo_t;


static const cudaErrorInfo_t g_errorTable[] = {
    {0, "cudaSuccess", "no error"},
    {1, "cudaErrorInvalidValue", "invalid argument"},
    {2, "cudaErrorMemoryAllocation", "out of memory"},
    {3, "cudaErrorInitializationError", "initialization error"},
    {4, "cudaErrorCudartUnloading", "CUDA Runtime is unloading"},
    {5, "cudaErrorProfilerDisabled", "profiler disabled"},
    {6, "cudaErrorProfilerNotInitialized", "profiler not initialized"},
    {7, "cudaErrorProfilerAlreadyStarted", "profiler already started"},
    {8, "cudaErrorProfilerAlreadyStopped", "profiler already stopped"},
    {9, "cudaErrorInvalidConfiguration", "invalid configuration"},
    {12, "cudaErrorInvalidPitchValue", "invalid pitch value"},
    {13, "cudaErrorInvalidSymbol", "invalid symbol"},
    {16, "cudaErrorInvalidHostPointer", "invalid host pointer"},
    {17, "cudaErrorInvalidDevicePointer", "invalid device pointer"},
    {18, "cudaErrorInvalidTexture", "invalid texture"},
    {19, "cudaErrorInvalidTextureBinding", "invalid texture binding"},
    {20, "cudaErrorInvalidChannelDescriptor", "invalid channel descriptor"},
    {21, "cudaErrorInvalidMemcpyDirection", "invalid memcpy direction"},
    {22, "cudaErrorAddressOfConstant", "address of constant"},
    {23, "cudaErrorTextureFetchFailed", "texture fetch failed"},
    {24, "cudaErrorTextureNotBound", "texture not bound"},
    {25, "cudaErrorSynchronizationError", "synchronization error"},
    {26, "cudaErrorInvalidFilterSetting", "invalid filter setting"},
    {27, "cudaErrorInvalidNormSetting", "invalid norm setting"},
    {28, "cudaErrorMixedDeviceExecution", "mixed device execution"},
    {31, "cudaErrorNotYetImplemented", "not yet implemented"},
    {32, "cudaErrorMemoryValueTooLarge", "memory value too large"},
    {34, "cudaErrorStubLibrary", "stub library"},
    {35, "cudaErrorInsufficientDriver", "insufficient driver"},
    {36, "cudaErrorCallRequiresNewerDriver", "call requires newer driver"},
    {37, "cudaErrorInvalidSurface", "invalid surface"},
    {43, "cudaErrorDuplicateVariableName", "duplicate variable name"},
    {44, "cudaErrorDuplicateTextureName", "duplicate texture name"},
    {45, "cudaErrorDuplicateSurfaceName", "duplicate surface name"},
    {46, "cudaErrorDevicesUnavailable", "devices unavailable"},
    {49, "cudaErrorIncompatibleDriverContext", "incompatible driver context"},
    {52, "cudaErrorMissingConfiguration", "missing configuration"},
    {53, "cudaErrorPriorLaunchFailure", "prior launch failure"},
    {65, "cudaErrorLaunchMaxDepthExceeded", "launch max depth exceeded"},
    {66, "cudaErrorLaunchFileScopedTex", "launch file scoped tex"},
    {67, "cudaErrorLaunchFileScopedSurf", "launch file scoped surf"},
    {68, "cudaErrorSyncDepthExceeded", "sync depth exceeded"},
    {69, "cudaErrorLaunchPendingCountExceeded", "launch pending count exceeded"},
    {98, "cudaErrorInvalidDeviceFunction", "invalid device function"},
    {100, "cudaErrorNoDevice", "no CUDA-capable device detected"},
    {101, "cudaErrorInvalidDevice", "invalid device"},
    {102, "cudaErrorDeviceNotLicensed", "device not licensed"},
    {103, "cudaErrorSoftwareValidityNotEstablished", "software validity not established"},
    {127, "cudaErrorStartupFailure", "startup failure"},
    {200, "cudaErrorInvalidKernelImage", "invalid kernel image"},
    {201, "cudaErrorDeviceUninitialized", "device uninitialized"},
    {205, "cudaErrorMapBufferObjectFailed", "map buffer object failed"},
    {206, "cudaErrorUnmapBufferObjectFailed", "unmap buffer object failed"},
    {207, "cudaErrorArrayIsMapped", "array is mapped"},
    {208, "cudaErrorAlreadyMapped", "already mapped"},
    {209, "cudaErrorNoKernelImageForDevice", "no kernel image for device"},
    {210, "cudaErrorAlreadyAcquired", "already acquired"},
    {211, "cudaErrorNotMapped", "not mapped"},
    {212, "cudaErrorNotMappedAsArray", "not mapped as array"},
    {213, "cudaErrorNotMappedAsPointer", "not mapped as pointer"},
    {214, "cudaErrorECCUncorrectable", "ECC uncorrectable"},
    {215, "cudaErrorUnsupportedLimit", "unsupported limit"},
    {216, "cudaErrorDeviceAlreadyInUse", "device already in use"},
    {217, "cudaErrorPeerAccessUnsupported", "peer access unsupported"},
    {218, "cudaErrorInvalidPtx", "invalid PTX"},
    {219, "cudaErrorInvalidGraphicsContext", "invalid graphics context"},
    {220, "cudaErrorNvlinkUncorrectable", "NVLink uncorrectable"},
    {221, "cudaErrorJitCompilerNotFound", "JIT compiler not found"},
    {222, "cudaErrorUnsupportedPtxVersion", "unsupported PTX version"},
    {223, "cudaErrorJitCompilationDisabled", "JIT compilation disabled"},
    {224, "cudaErrorUnsupportedExecAffinity", "unsupported exec affinity"},
    {225, "cudaErrorUnsupportedDevSideSync", "unsupported device-side sync"},
    {226, "cudaErrorContained", "contained"},
    {300, "cudaErrorInvalidSource", "invalid source"},
    {301, "cudaErrorFileNotFound", "file not found"},
    {302, "cudaErrorSharedObjectSymbolNotFound", "shared object symbol not found"},
    {303, "cudaErrorSharedObjectInitFailed", "shared object init failed"},
    {304, "cudaErrorOperatingSystem", "operating system error"},
    {400, "cudaErrorInvalidResourceHandle", "invalid resource handle"},
    {401, "cudaErrorIllegalState", "illegal state"},
    {402, "cudaErrorLossyQuery", "lossy query"},
    {500, "cudaErrorSymbolNotFound", "symbol not found"},
    {600, "cudaErrorNotReady", "not ready"},
    {700, "cudaErrorIllegalAddress", "illegal address"},
    {701, "cudaErrorLaunchOutOfResources", "launch out of resources"},
    {702, "cudaErrorLaunchTimeout", "launch timeout"},
    {703, "cudaErrorLaunchIncompatibleTexturing", "launch incompatible texturing"},
    {704, "cudaErrorPeerAccessAlreadyEnabled", "peer access already enabled"},
    {705, "cudaErrorPeerAccessNotEnabled", "peer access not enabled"},
    {708, "cudaErrorSetOnActiveProcess", "set on active process"},
    {709, "cudaErrorContextIsDestroyed", "context is destroyed"},
    {710, "cudaErrorAssert", "assert"},
    {711, "cudaErrorTooManyPeers", "too many peers"},
    {712, "cudaErrorHostMemoryAlreadyRegistered", "host memory already registered"},
    {713, "cudaErrorHostMemoryNotRegistered", "host memory not registered"},
    {714, "cudaErrorHardwareStackError", "hardware stack error"},
    {715, "cudaErrorIllegalInstruction", "illegal instruction"},
    {716, "cudaErrorMisalignedAddress", "misaligned address"},
    {717, "cudaErrorInvalidAddressSpace", "invalid address space"},
    {718, "cudaErrorInvalidPc", "invalid PC"},
    {719, "cudaErrorLaunchFailure", "launch failure"},
    {720, "cudaErrorCooperativeLaunchTooLarge", "cooperative launch too large"},
    {721, "cudaErrorTensorMemoryLeak", "tensor memory leak"},
    {800, "cudaErrorNotPermitted", "not permitted"},
    {801, "cudaErrorNotSupported", "not supported"},
    {802, "cudaErrorSystemNotReady", "system not ready"},
    {803, "cudaErrorSystemDriverMismatch", "system driver mismatch"},
    {804, "cudaErrorCompatNotSupportedOnDevice", "compat not supported on device"},
    {805, "cudaErrorMpsConnectionFailed", "MPS connection failed"},
    {806, "cudaErrorMpsRpcFailure", "MPS RPC failure"},
    {807, "cudaErrorMpsServerNotReady", "MPS server not ready"},
    {808, "cudaErrorMpsMaxClientsReached", "MPS max clients reached"},
    {809, "cudaErrorMpsMaxConnectionsReached", "MPS max connections reached"},
    {810, "cudaErrorMpsClientTerminated", "MPS client terminated"},
    {811, "cudaErrorCdpNotSupported", "CDP not supported"},
    {812, "cudaErrorCdpVersionMismatch", "CDP version mismatch"},
    {900, "cudaErrorStreamCaptureUnsupported", "stream capture unsupported"},
    {901, "cudaErrorStreamCaptureInvalidated", "stream capture invalidated"},
    {902, "cudaErrorStreamCaptureMerge", "stream capture merge"},
    {903, "cudaErrorStreamCaptureUnmatched", "stream capture unmatched"},
    {904, "cudaErrorStreamCaptureUnjoined", "stream capture unjoined"},
    {905, "cudaErrorStreamCaptureIsolation", "stream capture isolation"},
    {906, "cudaErrorStreamCaptureImplicit", "stream capture implicit"},
    {907, "cudaErrorCapturedEvent", "captured event"},
    {908, "cudaErrorStreamCaptureWrongThread", "stream capture wrong thread"},
    {909, "cudaErrorTimeout", "timeout"},
    {910, "cudaErrorGraphExecUpdateFailure", "graph exec update failure"},
    {911, "cudaErrorExternalDevice", "external device"},
    {912, "cudaErrorInvalidClusterSize", "invalid cluster size"},
    {913, "cudaErrorFunctionNotLoaded", "function not loaded"},
    {914, "cudaErrorInvalidResourceType", "invalid resource type"},
    {915, "cudaErrorInvalidResourceConfiguration", "invalid resource configuration"},
    {917, "cudaErrorStreamDetached", "stream detached"},
    {999, "cudaErrorUnknown", "unknown error"},
    {10000, "cudaErrorApiFailureBase", "api failure base"},
};

static const int g_errorTableSize = sizeof(g_errorTable) / sizeof(g_errorTable[0]);


static const cudaErrorInfo_t *cudaFindErrorInfo(int error)
{
    int left = 0;
    int right = g_errorTableSize - 1;

    while (left <= right)
    {
        int mid = left + (right - left) / 2;
        if (g_errorTable[mid].code == error)
        {
            return &g_errorTable[mid];
        }
        else if (g_errorTable[mid].code < error)
        {
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }
    return NULL;
}

const char *cudaGetErrorName(cudaError_t error)
{
    const cudaErrorInfo_t *info = cudaFindErrorInfo(error);
    if (info != NULL)
    {
        return info->name;
    }
    return "cudaErrorUnknown";
}

const char *cudaGetErrorString(cudaError_t error)
{
    const cudaErrorInfo_t *info = cudaFindErrorInfo(error);
    if (info != NULL)
    {
        return info->string;
    }
    return "unknown error";
}

/* =================================================================
 * Profiler Management
 * ================================================================= */


cudaError_t cudaProfilerStart(void)
{
    // Initialize profiler on first call
    if (!g_cuda_context.profiler_initialized)
    {
        aclError ret = aclprofInit(NULL, 0); // Initialize with default config
        if (ret != ACL_SUCCESS)
        {
            return acl2cudaError(ret);
        }
        g_cuda_context.profiler_initialized = 1;
    }

    // Start profiling if not already running
    aclError ret = aclprofStart(NULL); // Start with default config
    if (ret != ACL_SUCCESS)
    {
        return acl2cudaError(ret);
    }
    g_cuda_context.profiler_running = 1;

    return cudaSuccess;
}


cudaError_t cudaProfilerStop(void)
{
    if (!g_cuda_context.profiler_initialized)
    {
        // Treat a stopped profiler as an idempotent success.
        return cudaSuccess;
    }

    if (g_cuda_context.profiler_running)
    {
        aclError ret = aclprofStop(NULL); // Stop with default config
        if (ret != ACL_SUCCESS)
        {
            return acl2cudaError(ret);
        }
        g_cuda_context.profiler_running = 0;
    }

    aclError ret = aclprofFinalize();
    if (ret != ACL_SUCCESS)
    {
        return acl2cudaError(ret);
    }

    g_cuda_context.profiler_initialized = 0;
    return cudaSuccess;
}
