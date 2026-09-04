/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */



#ifndef CUDA_COMPAT_CU_VMM_H
#define CUDA_COMPAT_CU_VMM_H

#include "cann_compat_cu_types.h"
#include <stdint.h>
#include <stdlib.h>

#define CUDA_COMPAT_MAX_ACCESS_DESC_COUNT 1024U

#ifdef __cplusplus
extern "C"
{
#endif

    uint64_t get_elf_file_size(const void *elf_buf);

    /* =================================================================
     * Virtual Memory Management
     * ================================================================= */

    static inline CUresult cuMemShareTypeToCann(CUmemAllocationHandleType handleType,
                                                aclrtMemSharedHandleType *shareType)
    {
        if (shareType == NULL)
        {
            return CUDA_ERROR_INVALID_VALUE;
        }

        switch (handleType)
        {
        case CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR:
            *shareType = ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT;
            break;
        case CU_MEM_HANDLE_TYPE_FABRIC:
            *shareType = ACL_MEM_SHARE_HANDLE_TYPE_FABRIC;
            break;
        default:
            return CUDA_ERROR_INVALID_VALUE;
        }

        return CUDA_SUCCESS;
    }


    static inline CUresult cuMemAllocationPropToCann(
        const CUmemAllocationProp *cuProp,
        aclrtPhysicalMemProp *cannProp)
    {
        if (!cuProp || !cannProp)
        {
            return CUDA_ERROR_INVALID_VALUE;
        }

        // Map handle type
        switch (cuProp->requestedHandleTypes)
        {
        case CU_MEM_HANDLE_TYPE_NONE:
            cannProp->handleType = ACL_MEM_HANDLE_TYPE_NONE;
            break;
        case CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR:
            cannProp->handleType = ACL_MEM_HANDLE_TYPE_NONE;
            break;
        case CU_MEM_HANDLE_TYPE_FABRIC:
            cannProp->handleType = ACL_MEM_HANDLE_TYPE_NONE;
            break;
        default:
            return CUDA_ERROR_INVALID_VALUE;
        }

        // Map allocation type
        switch (cuProp->type)
        {
        case CU_MEM_ALLOCATION_TYPE_PINNED:
            cannProp->allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
            break;
        default:
            return CUDA_ERROR_INVALID_VALUE;
        }

        // Map location
        switch (cuProp->location.type)
        {
        case CU_MEM_LOCATION_TYPE_DEVICE:
            cannProp->location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
            break;
        case CU_MEM_LOCATION_TYPE_HOST:
            cannProp->location.type = ACL_MEM_LOCATION_TYPE_HOST;
            break;
        case CU_MEM_LOCATION_TYPE_HOST_NUMA:
            cannProp->location.type = ACL_MEM_LOCATION_TYPE_HOST_NUMA;
            break;
        default:
            return CUDA_ERROR_INVALID_VALUE;
        }
        cannProp->location.id = cuProp->location.id;

        cannProp->memAttr = ACL_MEM_NORMAL;
        // Reserve field is not used in CUDA, set to 0
        cannProp->reserve = 0;

        return CUDA_SUCCESS;
    }


    static inline CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size,
                                               size_t alignment, CUdeviceptr address,
                                               unsigned long long flags)
    {
        return acl2cuError(aclrtReserveMemAddress(ptr, size, alignment, (void *)address, flags));
    }


    static inline CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size)
    {
        (void)size; // Size not used by CANN
        return acl2cuError(aclrtReleaseMemAddress((void *)ptr));
    }


    static inline CUresult cuMemCreate(CUmemGenericAllocationHandle *handle,
                                       size_t size,
                                       const CUmemAllocationProp *prop,
                                       unsigned long long flags)
    {
        (void)flags;
        if (!handle || !prop)
        {
            return CUDA_ERROR_INVALID_VALUE;
        }

        // Convert CUDA properties to CANN properties
        aclrtPhysicalMemProp cannProp;
        CUresult ret = cuMemAllocationPropToCann(prop, &cannProp);
        if (ret != CUDA_SUCCESS)
        {
            return ret;
        }

        aclError aclRet = aclrtMallocPhysical(handle, size, &cannProp, 0);
        return acl2cuError(aclRet);
    }


    static inline CUresult cuMemRelease(CUmemGenericAllocationHandle handle)
    {
        return acl2cuError(aclrtFreePhysical(handle));
    }


     static inline CUresult cuMemExportToShareableHandle(void *shareableHandle,
                                                         CUmemGenericAllocationHandle handle,
                                                         CUmemAllocationHandleType handleType,
                                                         unsigned long long flags)
     {
         if (!shareableHandle)
         {
             return CUDA_ERROR_INVALID_VALUE;
         }

        aclrtMemSharedHandleType shareType = ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT;
        CUresult shareRet = cuMemShareTypeToCann(handleType, &shareType);
        if (shareRet != CUDA_SUCCESS)
        {
            return shareRet;
        }

        aclError ret = aclrtMemExportToShareableHandleV2(handle, ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION,
                                                         shareType, shareableHandle);
        return acl2cuError(ret);
    }


    static inline CUresult cuMemGetAccess(unsigned long long *flags,
                                          const CUmemLocation *location,
                                          CUdeviceptr ptr)
    {
        if (!location || ptr == 0)
        {
            return CUDA_ERROR_INVALID_VALUE;
        }

        aclrtMemLocation cannLocation;
        cannLocation.id = location->id;
        switch (location->type)
        {
        case CU_MEM_LOCATION_TYPE_DEVICE:
            cannLocation.type = ACL_MEM_LOCATION_TYPE_DEVICE;
            break;
        case CU_MEM_LOCATION_TYPE_HOST:
            cannLocation.type = ACL_MEM_LOCATION_TYPE_HOST;
            break;
        case CU_MEM_LOCATION_TYPE_HOST_NUMA:
            cannLocation.type = ACL_MEM_LOCATION_TYPE_HOST_NUMA;
            break;
        default:
            return CUDA_ERROR_INVALID_VALUE;
        }

        aclError ret = aclrtMemGetAccess((void *)ptr,
                                         &cannLocation, (uint64_t *)flags);
        return acl2cuError(ret);
    }


    static inline CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                                          const CUmemAccessDesc *desc,
                                          size_t count)
    {
        if (!desc || count == 0 || ptr == 0 || size == 0)
        {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (count > CUDA_COMPAT_MAX_ACCESS_DESC_COUNT ||
            count > SIZE_MAX / sizeof(aclrtMemAccessDesc))
        {
            return CUDA_ERROR_INVALID_VALUE;
        }

        aclrtMemAccessDesc *cannDesc = (aclrtMemAccessDesc *)malloc(count * sizeof(aclrtMemAccessDesc));
        if (!cannDesc)
        {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        for (int i = 0; i < count; i++)
        {
            cannDesc[i].flags = (aclrtMemAccessFlags)desc[i].flags;
            cannDesc[i].location.id = desc[i].location.id;
            switch (desc[i].location.type)
            {
            case CU_MEM_LOCATION_TYPE_DEVICE:
                cannDesc[i].location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
                break;
            case CU_MEM_LOCATION_TYPE_HOST:
                cannDesc[i].location.type = ACL_MEM_LOCATION_TYPE_HOST;
                break;
            case CU_MEM_LOCATION_TYPE_HOST_NUMA:
                cannDesc[i].location.type = ACL_MEM_LOCATION_TYPE_HOST_NUMA;
                break;
            default:
                free(cannDesc);
                return CUDA_ERROR_INVALID_VALUE;
            }
        }
        aclError ret = aclrtMemSetAccess((void *)ptr, size, cannDesc, count);
        free(cannDesc);
        return acl2cuError(ret);
    }


    static inline CUresult cuMemGetAllocationGranularity(size_t *granularity,
                                                         const CUmemAllocationProp *prop,
                                                         CUmemAllocationGranularity_flags option)
    {
        if (!granularity || !prop)
        {
            return CUDA_ERROR_INVALID_VALUE;
        }

        // Convert CUDA properties to CANN properties
        aclrtPhysicalMemProp cannProp;
        CUresult ret = cuMemAllocationPropToCann(prop, &cannProp);
        if (ret != CUDA_SUCCESS)
        {
            return ret;
        }

        aclrtMemGranularityOptions cannOption;
        switch (option)
        {
        case CU_MEM_ALLOC_GRANULARITY_MINIMUM:
            cannOption = ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM;
            break;
        case CU_MEM_ALLOC_GRANULARITY_RECOMMENDED:
            cannOption = ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED;
            break;
        default:
            return CUDA_ERROR_INVALID_VALUE;
        }

        aclError aclRet = aclrtMemGetAllocationGranularity(&cannProp, cannOption, granularity);
        return acl2cuError(aclRet);
    }


    static inline CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop,
                                                                  CUmemGenericAllocationHandle handle)
    {
        if (!prop || !handle)
        {
            return CUDA_ERROR_INVALID_VALUE;
        }

        // Get properties from CANN
        aclrtPhysicalMemProp cannProp;
        aclError ret = aclrtMemGetAllocationPropertiesFromHandle(handle, &cannProp);
        if (ret != ACL_SUCCESS)
        {
            return acl2cuError(ret);
        }

        // Convert CANN properties to CUDA properties
        // Map handle type
        if (cannProp.handleType==ACL_MEM_HANDLE_TYPE_NONE)
        {
            prop->requestedHandleTypes=CU_MEM_HANDLE_TYPE_NONE;
        }

        // Map allocation type
        switch (cannProp.allocationType)
        {
        case ACL_MEM_ALLOCATION_TYPE_PINNED:
            prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
            break;
        default:
            return CUDA_ERROR_INVALID_VALUE;
        }

        // Map location
        switch (cannProp.location.type)
        {
        case ACL_MEM_LOCATION_TYPE_DEVICE:
            prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            break;
        case ACL_MEM_LOCATION_TYPE_HOST:
            prop->location.type = CU_MEM_LOCATION_TYPE_HOST;
            break;
        case ACL_MEM_LOCATION_TYPE_HOST_NUMA:
            prop->location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
            break;
        default:
            return CUDA_ERROR_INVALID_VALUE;
        }
        prop->location.id = cannProp.location.id;

        // CANN doesn't have win32HandleMetaData and allocFlags
        // Set them to default values
        prop->win32HandleMetaData = NULL;
        prop->allocFlags.compressionType = 0;
        prop->allocFlags.gpuDirectRDMACapable = 0;
        prop->allocFlags.usage = 0;
        for (size_t i = 0; i < sizeof(prop->allocFlags.reserved); i++)
        {
            prop->allocFlags.reserved[i] = 0;
        }

        return CUDA_SUCCESS;
    }


     static inline CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle,
                                                           void *osHandle,
                                                           CUmemAllocationHandleType shHandleType)
     {
         if (!handle || !osHandle)
         {
             return CUDA_ERROR_INVALID_VALUE;
         }

        aclrtMemSharedHandleType shareType = ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT;
        CUresult shareRet = cuMemShareTypeToCann(shHandleType, &shareType);
        if (shareRet != CUDA_SUCCESS)
        {
            return shareRet;
        }

        aclError ret = aclrtMemImportFromShareableHandleV2(osHandle, shareType, 0, handle);
        return acl2cuError(ret);
    }


    static inline CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                                    CUmemGenericAllocationHandle handle,
                                    unsigned long long flags)
    {
        return acl2cuError(aclrtMapMem((void *)ptr, size, offset, handle, flags));
    }


    static inline CUresult cuMemUnmap(CUdeviceptr ptr, size_t size)
    {
        (void)size; // Size not used by CANN
        return acl2cuError(aclrtUnmapMem((void *)ptr));
    }


    static inline CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr)
    {
        return acl2cuError(aclrtMemRetainAllocationHandle(addr, handle));
    }


    static inline CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui,
                                            size_t N, CUstream hStream)
    {
        if (!dstDevice) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        aclError ret = aclrtMemsetD32Async((void *)dstDevice, N * sizeof(uint32_t),
                                           ui, N, (aclrtStream)hStream);
        return acl2cuError(ret);
    }

#ifdef __cplusplus
}
#endif

#include "cann_compat_cu_extra.h"

#endif /* CUDA_COMPAT_CU_VMM_H */
