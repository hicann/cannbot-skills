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
