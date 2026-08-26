/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */

/*
 * Project-local driver-style compatibility types for the CUDA-to-CANN shim.
 *
 * This file keeps the minimal API surface required by migrated sources. Avoid
 * copying vendor SDK comments, documentation, or installation-package content.
 */

#ifndef CUDA_COMPAT_CU_TYPES_H
#define CUDA_COMPAT_CU_TYPES_H

#include "acl/acl_rt.h"
#include "cann_compat_error_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* =================================================================
     * Device Pointer Type
     * ================================================================= */


    typedef void *CUdeviceptr;
    typedef int CUdevice;
    typedef void *CUcontext;
    typedef void *CUgreenCtx;
    typedef void *CUdevResource;
    typedef void *CUstream;
    typedef void *CUfunction;
    typedef void *CUmodule;
    typedef void *CUlinkState;
    typedef void *CUmulticastObject;
    typedef void *CUtensorMap;
    typedef unsigned int cuuint32_t;
    typedef unsigned long long cuuint64_t;

    typedef enum
    {
        CU_FUNC_CACHE_PREFER_NONE = 0,
        CU_FUNC_CACHE_PREFER_SHARED = 1,
        CU_FUNC_CACHE_PREFER_L1 = 2,
        CU_FUNC_CACHE_PREFER_EQUAL = 3
    } CUfunc_cache;

    typedef enum
    {
        CU_JIT_MAX_REGISTERS = 0,
        CU_JIT_THREADS_PER_BLOCK = 1,
        CU_JIT_WALL_TIME = 2,
        CU_JIT_INFO_LOG_BUFFER = 3,
        CU_JIT_ERROR_LOG_BUFFER = 5
    } CUjit_option;

    typedef enum
    {
        CU_JIT_INPUT_CUBIN = 0,
        CU_JIT_INPUT_PTX = 1,
        CU_JIT_INPUT_FATBINARY = 2,
        CU_JIT_INPUT_OBJECT = 3,
        CU_JIT_INPUT_LIBRARY = 4,
        CU_JIT_INPUT_NVVM = 5
    } CUjitInputType;

    typedef enum
    {
        CU_DEV_RESOURCE_TYPE_SM = 0
    } CUdevResourceType;

    typedef struct
    {
        unsigned int type;
        unsigned int flags;
        unsigned long long value;
    } CUdevResourceDesc;

    typedef struct
    {
        size_t size;
        unsigned long long flags;
    } CUmulticastObjectProp;

    typedef enum
    {
        CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,
        CU_TENSOR_MAP_DATA_TYPE_UINT16 = 1,
        CU_TENSOR_MAP_DATA_TYPE_UINT32 = 2,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 3
    } CUtensorMapDataType;

    typedef enum
    {
        CU_TENSOR_MAP_INTERLEAVE_NONE = 0
    } CUtensorMapInterleave;

    typedef enum
    {
        CU_TENSOR_MAP_SWIZZLE_NONE = 0
    } CUtensorMapSwizzle;

    typedef enum
    {
        CU_TENSOR_MAP_L2_PROMOTION_NONE = 0
    } CUtensorMapL2promotion;

    typedef enum
    {
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0
    } CUtensorMapFloatOOBfill;
    /* =================================================================
     * Memory Handle Types
     * ================================================================= */
    typedef aclrtDrvMemHandle CUmemGenericAllocationHandle;


    typedef enum
    {
        CU_MEM_LOCATION_TYPE_INVALID = 0x0,
        CU_MEM_LOCATION_TYPE_NONE = 0x0,
        CU_MEM_LOCATION_TYPE_DEVICE = 0x1,
        CU_MEM_LOCATION_TYPE_HOST = 0x2,
        CU_MEM_LOCATION_TYPE_HOST_NUMA = 0x3,
        CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT = 0x4,
        CU_MEM_LOCATION_TYPE_MAX = 0x7FFFFFFF
    } CUmemLocationType;


    typedef struct
    {
        CUmemLocationType type;
        int id;
    } CUmemLocation;


    typedef enum CUmemAccess_flags_enum
    {
        CU_MEM_ACCESS_FLAGS_PROT_NONE = 0x0,
        CU_MEM_ACCESS_FLAGS_PROT_READ = 0x1,
        CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3,
        CU_MEM_ACCESS_FLAGS_PROT_MAX = 0x7FFFFFFF
    } CUmemAccess_flags;


    typedef struct CUmemAccessDesc_st
    {
        CUmemLocation location;
        CUmemAccess_flags flags;
    } CUmemAccessDesc_v1;
    typedef CUmemAccessDesc_v1 CUmemAccessDesc;
    typedef enum
    {
        CU_MEM_ALLOCATION_TYPE_INVALID = 0x0,
        CU_MEM_ALLOCATION_TYPE_PINNED = 0x1,
        CU_MEM_ALLOCATION_TYPE_MANAGED = 0x2,
        CU_MEM_ALLOCATION_TYPE_MAX = 0x7FFFFFFF
    } CUmemAllocationType;


    typedef struct
    {
        unsigned char compressionType;
        unsigned char gpuDirectRDMACapable;
        unsigned short usage;
        unsigned char reserved[4];
    } CUmemAllocationProp_allocFlags_st;


    typedef enum CUmemAllocationHandleType_enum
    {
        CU_MEM_HANDLE_TYPE_NONE = 0x0,
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1,
        CU_MEM_HANDLE_TYPE_WIN32 = 0x2,
        CU_MEM_HANDLE_TYPE_WIN32_KMT = 0x4,
        CU_MEM_HANDLE_TYPE_FABRIC = 0x8,
        CU_MEM_HANDLE_TYPE_MAX = 0x7FFFFFFF
    } CUmemAllocationHandleType;

    typedef aclrtMemFabricHandle CUmemFabricHandle;
    typedef struct CUmemAllocationProp_st
    {
        CUmemAllocationType type;
        CUmemAllocationHandleType requestedHandleTypes;
        CUmemLocation location;
        void *win32HandleMetaData;
        struct
        {
            unsigned char compressionType;
            unsigned char gpuDirectRDMACapable;
            unsigned short usage;
            unsigned char reserved[4];
        } allocFlags;
    } CUmemAllocationProp_v1;
    typedef CUmemAllocationProp_v1 CUmemAllocationProp;
    typedef enum CUmemHandleType_enum
    {
        CU_MEM_HANDLE_TYPE_GENERIC = 0
    } CUmemHandleType;


    typedef enum CUmemAllocationGranularity_flags_enum
    {
        CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0x0,
        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1
    } CUmemAllocationGranularity_flags;

    /* =================================================================
     * Error Code Mapping
     * ================================================================= */


    static inline cudaError_t cu2cudaError(CUresult result)
    {
        // Cast to int first, then to cudaError_t
        // Note: CUDA Driver and Runtime use same error code values
        return (cudaError_t)(int)result;
    }


    static inline CUresult acl2cuError(aclError err)
    {
        return (CUresult)acl2cudaError(err);
    }

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_CU_TYPES_H */
