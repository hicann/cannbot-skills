/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */


/*
 * Project-local compatibility types used by the CUDA-to-CANN shim.
 *
 * Symbol names and numeric constants are present for source compatibility with
 * migrated programs. Do not paste vendor SDK header prose or documentation into
 * this file.
 */

#ifndef CUDA_COMPAT_TYPES_H
#define CUDA_COMPAT_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "acl/acl_rt.h"
#include "cann_compat_device_prop_types.h"
#include "cann_compat_device_types.h"
#include "cann_compat_error_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define MOCK_CUDA_MAJOR_VERSION 13
#define MOCK_CUDA_MINOR_VERSION 3
#ifndef CUDART_VERSION
#define CUDART_VERSION 13030
#endif

#define cudaStreamNonDefault    0x00
#define cudaStreamNonBlocking   0x01

    /* =================================================================
     * Device Management Types
     * ================================================================= */

    /* =================================================================
     * Stream Types (Direct mapping)
     * ================================================================= */

    typedef aclrtStream cudaStream_t;

    typedef aclrtStreamAttr cudaStreamAttrID;
    typedef aclrtStreamAttrValue cudaStreamAttrValue;

#ifndef cudaStreamAttributeAccessPolicyWindow
#define cudaStreamAttributeAccessPolicyWindow ((cudaStreamAttrID)1)
#endif
#ifndef cudaStreamAttributeSynchronizationPolicy
#define cudaStreamAttributeSynchronizationPolicy ((cudaStreamAttrID)3)
#endif

    /* =================================================================
     * Event Types
     * ================================================================= */
    typedef aclrtEvent cudaEvent_t;

#define cudaEventDefault 0x00
#define cudaEventBlockingSync 0x01
#define cudaEventDisableTiming 0x02
#define cudaEventInterprocess 0x04

#define cudaEventRecordDefault 0x00
#define cudaEventRecordExternal 0x01

#define cudaEventWaitDefault 0x00
#define cudaEventWaitExternal 0x01

#ifdef __cplusplus
}
#endif

#include "cann_compat_memory_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* =================================================================
     * Limit Types
     * ================================================================= */

    typedef enum
    {
        cudaLimitStackSize = 0x00,
        cudaLimitPrintfFifoSize = 0x01,
        cudaLimitMallocHeapSize = 0x02,
        cudaLimitDevRuntimeSyncDepth = 0x03,
        cudaLimitDevRuntimePendingLaunchCount = 0x04,
        cudaLimitMaxL2FetchGranularity = 0x05,
        cudaLimitPersistingL2CacheSize = 0x06
    } cudaLimit;

    typedef enum
    {
        cudaAtomicOperationIntegerAdd = 0,
        cudaAtomicOperationIntegerMin = 1,
        cudaAtomicOperationIntegerMax = 2,
        cudaAtomicOperationIntegerIncrement = 3,
        cudaAtomicOperationIntegerDecrement = 4,
        cudaAtomicOperationAnd = 5,
        cudaAtomicOperationOr = 6,
        cudaAtomicOperationXOR = 7,
        cudaAtomicOperationExchange = 8,
        cudaAtomicOperationCAS = 9,
        cudaAtomicOperationFloatAdd = 10,
        cudaAtomicOperationFloatMin = 11,
        cudaAtomicOperationFloatMax = 12
    } cudaAtomicOperation;

    typedef enum
    {
        cudaAtomicCapabilitySigned = 1U << 0,
        cudaAtomicCapabilityUnsigned = 1U << 1,
        cudaAtomicCapabilityReduction = 1U << 2,
        cudaAtomicCapabilityScalar32 = 1U << 3,
        cudaAtomicCapabilityScalar64 = 1U << 4,
        cudaAtomicCapabilityScalar128 = 1U << 5,
        cudaAtomicCapabilityVector32x4 = 1U << 6
    } cudaAtomicCapability;

    /* =================================================================
     * Compute Mode
     * ================================================================= */


    typedef enum
    {
        cudaComputeModeDefault = 0,
        cudaComputeModeExclusive = 1,
        cudaComputeModeProhibited = 2,
        cudaComputeModeExclusiveProcess = 3
    } cudaComputeMode;

    /* =================================================================
     * Device Flags
     * ================================================================= */

    typedef enum
    {
        cudaDeviceScheduleAuto = 0,
        cudaDeviceScheduleSpin = 1,
        cudaDeviceScheduleYield = 2,
        cudaDeviceScheduleBlockingSync = 4,
        cudaDeviceMapHost = 8,
        cudaDeviceLmemResizeToMax = 16,
        cudaDeviceSyncMemops = 0x40000
    } cudaDeviceFlags;

    typedef enum
    {
        cudaFuncCachePreferNone = 0,
        cudaFuncCachePreferShared = 1,
        cudaFuncCachePreferL1 = 2,
        cudaFuncCachePreferEqual = 3
    } cudaFuncCache;

    typedef enum
    {
        cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
        cudaFuncAttributePreferredSharedMemoryCarveout = 9,
        cudaFuncAttributeMax = 10
    } cudaFuncAttribute;

    /* =================================================================
     * Function Attributes
     * ================================================================= */

    typedef struct
    {
        size_t sharedSizeBytes;
        size_t constSizeBytes;
        size_t localSizeBytes;
        int maxThreadsPerBlock;
        int numRegs;
        int ptxVersion;
        int binaryVersion;
        int cacheModeCA;
        int maxDynamicSharedSizeBytes;
        int preferredShmemCarveout;
        int clusterDimMustBeSet;
        int requiredClusterWidth;
        int requiredClusterHeight;
        int requiredClusterDepth;
        int clusterSchedulingPolicyPreference;
        int nonPortableClusterSizeAllowed;
        int reserved[16];
    } cudaFuncAttributes;

#ifdef __cplusplus
}
#endif

#include "cann_compat_graph_mempool_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* =================================================================
     * Internal State Management
     * ================================================================= */

    typedef struct cudaCompatContext
    {
        int initialized;
        uint32_t flags;
        int profiler_initialized;
        int profiler_running;
    } cudaCompatContext_t;

    extern cudaCompatContext_t g_cuda_context;

#define CUDA_COMPAT_FORCE_LINK() \
    extern const int _cuda_compat_force_link; \
    static volatile const int *_cuda_compat_force_link_ptr __attribute__((used)) = &_cuda_compat_force_link;

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_TYPES_H */
