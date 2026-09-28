/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */

#ifndef CUDA_COMPAT_MEMORY_QUERY_HOST_H
#define CUDA_COMPAT_MEMORY_QUERY_HOST_H

#include "cann_compat_safe.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /* =================================================================
     * Memory Query Operations
     * ================================================================= */


    static inline cudaError_t cudaMemGetInfo(size_t *free, size_t *total)
    {
        aclError ret = aclrtGetMemInfo(ACL_HBM_MEM, free, total);
        return acl2cudaError(ret);
    }


    cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attributes,
                                         const void *ptr);

    static inline cudaError_t cudaMemAdvise(const void *devPtr, size_t count,
                                            cudaMemoryAdvise advice, int device)
    {
        if (aclrtMemManagedAdvise == NULL) {
            return cudaErrorNotSupported;
        }
        if (!devPtr || count == 0) {
            return cudaErrorInvalidValue;
        }
        aclrtMemManagedAdviseType cannAdvice;
        cudaError_t adviceRet = cudaCompatMemAdviseToCann(advice, &cannAdvice);
        if (adviceRet != cudaSuccess) {
            return adviceRet;
        }
        aclrtMemManagedLocation location;
        cudaError_t locRet = cudaCompatDeviceToManagedLocation(device, &location);
        if (locRet != cudaSuccess) {
            return locRet;
        }
        aclError ret = aclrtMemManagedAdvise(devPtr, count, cannAdvice, location);
        return acl2cudaError(ret);
    }

    static inline cudaError_t cudaMemPrefetchAsync_v2(const void *devPtr, size_t count,
                                                      cudaMemLocation location, unsigned int flags,
                                                      cudaStream_t stream)
    {
        if (aclrtMemManagedPrefetchAsync == NULL) {
            return cudaErrorNotSupported;
        }
        if (!devPtr || count == 0 || flags != 0) {
            return cudaErrorInvalidValue;
        }
        aclrtMemManagedLocation cannLocation;
        cudaError_t locRet = cudaCompatMemLocationToCann(location, &cannLocation);
        if (locRet != cudaSuccess) {
            return locRet;
        }
        aclError ret = aclrtMemManagedPrefetchAsync(devPtr, count, cannLocation, flags, stream);
        return acl2cudaError(ret);
    }

    static inline cudaError_t cudaMemPrefetchBatchAsync(const void **devPtrs, size_t *sizes, size_t count,
                                                        cudaMemLocation *locations, size_t *locationIdxs,
                                                        size_t numLocations, unsigned long long flags,
                                                        cudaStream_t stream)
    {
        if (aclrtMemManagedPrefetchBatchAsync == NULL) {
            return cudaErrorNotSupported;
        }
        if (!devPtrs || !sizes || !locations || !locationIdxs || count == 0 || numLocations == 0 || flags != 0 ||
            numLocations > SIZE_MAX / sizeof(aclrtMemManagedLocation)) {
            return cudaErrorInvalidValue;
        }
        aclrtMemManagedLocation *cannLocations =
            (aclrtMemManagedLocation *)calloc(numLocations, sizeof(aclrtMemManagedLocation));
        if (!cannLocations) {
            return cudaErrorMemoryAllocation;
        }
        for (size_t i = 0; i < numLocations; ++i) {
            cudaError_t locRet = cudaCompatMemLocationToCann(locations[i], &cannLocations[i]);
            if (locRet != cudaSuccess) {
                free(cannLocations);
                return locRet;
            }
        }
        aclError ret = aclrtMemManagedPrefetchBatchAsync(devPtrs, sizes, count, cannLocations, locationIdxs,
                                                        numLocations, flags, stream);
        free(cannLocations);
        return acl2cudaError(ret);
    }

    static inline cudaError_t cudaMemRangeGetAttribute(void *data, size_t dataSize,
                                                       cudaMemRangeAttribute attribute,
                                                       const void *devPtr, size_t count)
    {
        if (aclrtMemManagedGetAttr == NULL) {
            return cudaErrorNotSupported;
        }
        if (!data || !devPtr || count == 0) {
            return cudaErrorInvalidValue;
        }
        aclrtMemManagedRangeAttribute cannAttribute;
        cudaError_t attrRet = cudaCompatMemRangeAttributeToCann(attribute, &cannAttribute);
        if (attrRet != cudaSuccess) {
            return attrRet;
        }
        aclError ret = aclrtMemManagedGetAttr(cannAttribute, devPtr, count, data, dataSize);
        return acl2cudaError(ret);
    }

    static inline cudaError_t cudaMemRangeGetAttributes(void **data, size_t *dataSizes,
                                                        cudaMemRangeAttribute *attributes,
                                                        size_t numAttributes, const void *devPtr,
                                                        size_t count)
    {
        if (aclrtMemManagedGetAttrs == NULL) {
            return cudaErrorNotSupported;
        }
        if (!data || !dataSizes || !attributes || !devPtr || count == 0 || numAttributes == 0 ||
            numAttributes > SIZE_MAX / sizeof(aclrtMemManagedRangeAttribute)) {
            return cudaErrorInvalidValue;
        }
        aclrtMemManagedRangeAttribute *cannAttributes =
            (aclrtMemManagedRangeAttribute *)calloc(numAttributes, sizeof(aclrtMemManagedRangeAttribute));
        if (!cannAttributes) {
            return cudaErrorMemoryAllocation;
        }
        for (size_t i = 0; i < numAttributes; ++i) {
            cudaError_t attrRet = cudaCompatMemRangeAttributeToCann(attributes[i], &cannAttributes[i]);
            if (attrRet != cudaSuccess) {
                free(cannAttributes);
                return attrRet;
            }
        }
        aclError ret = aclrtMemManagedGetAttrs(cannAttributes, numAttributes, devPtr, count, data, dataSizes);
        free(cannAttributes);
        return acl2cudaError(ret);
    }

    static inline cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost)
    {
        if (!pFlags || !pHost) {
            return cudaErrorInvalidValue;
        }
        cudaCompatHostAllocRecord record;
        if (cudaCompatFindHostAllocRecord(pHost, &record)) {
            *pFlags = record.flags;
            return cudaSuccess;
        }
        cudaPointerAttributes attributes;
        cudaError_t ret = cudaPointerGetAttributes(&attributes, pHost);
        if (ret != cudaSuccess) {
            return ret;
        }
        if (attributes.type != cudaMemoryTypeHost) {
            return cudaErrorInvalidValue;
        }
        *pFlags = cudaHostAllocDefault;
        return cudaSuccess;
    }

    /* =================================================================
     * Host Memory Registration
     * ================================================================= */

    static inline cudaError_t cudaCompatHostRegisterFlagsToCann(unsigned int cudaFlags,
                                                                uint32_t *cannFlags)
    {
        const unsigned int supportedFlags = cudaHostRegisterPortable |
                                            cudaHostRegisterMapped |
                                            cudaHostRegisterIoMemory |
                                            cudaHostRegisterReadOnly;
        if (!cannFlags || (cudaFlags & ~supportedFlags) != 0) {
            return cudaErrorInvalidValue;
        }

        uint32_t flags = ACL_HOST_REG_PINNED;
        if ((cudaFlags & cudaHostRegisterMapped) != 0) {
            flags |= ACL_HOST_REG_MAPPED;
        }
        if ((cudaFlags & cudaHostRegisterIoMemory) != 0) {
            flags |= ACL_HOST_REG_IOMEMORY;
        }
        if ((cudaFlags & cudaHostRegisterReadOnly) != 0) {
            flags |= ACL_HOST_REG_READONLY;
        }

        *cannFlags = flags;
        return cudaSuccess;
    }

    static inline cudaError_t cudaHostRegister(void *ptr, size_t size,
                                               unsigned int flags)
    {
        if (!ptr || size == 0) {
            return cudaErrorInvalidValue;
        }

        uint32_t cannFlags = 0;
        cudaError_t mapRet = cudaCompatHostRegisterFlagsToCann(flags, &cannFlags);
        if (mapRet != cudaSuccess) {
            return mapRet;
        }

        cudaError_t deviceRet = cudaCompatEnsureMemoryDevice();
        if (deviceRet != cudaSuccess) {
            return deviceRet;
        }

        aclError ret = aclrtHostRegisterV2(ptr, size, cannFlags);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaHostUnregister(void *ptr)
    {
        if (!ptr) {
            return cudaErrorInvalidValue;
        }
        cudaError_t deviceRet = cudaCompatEnsureMemoryDevice();
        if (deviceRet != cudaSuccess) {
            return deviceRet;
        }
        aclError ret = aclrtHostUnregister(ptr);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost,
                                                       unsigned int flags)
    {
        if (!pDevice || !pHost || flags != 0) {
            return cudaErrorInvalidValue;
        }
        cudaError_t deviceRet = cudaCompatEnsureMemoryDevice();
        if (deviceRet != cudaSuccess) {
            return deviceRet;
        }
        aclError ret = aclrtHostGetDevicePointer(pHost, pDevice, flags);
        return acl2cudaError(ret);
    }

#ifdef __cplusplus
}

static inline cudaError_t cudaMemAdvise(const void *devPtr, size_t count,
                                        cudaMemoryAdvise advice, cudaMemLocation location)
{
    if (location.type != cudaMemLocationTypeDevice) {
        return cudaErrorInvalidValue;
    }
    return cudaMemAdvise(devPtr, count, advice, location.id);
}

template <typename T>
static inline cudaError_t cudaMalloc(T **devPtr, size_t size)
{
    return cudaMalloc(reinterpret_cast<void **>(devPtr), size);
}

template <typename T>
static inline cudaError_t cudaMallocManaged(T **devPtr, size_t size,
                                            unsigned int flags = cudaMemAttachGlobal)
{
    return cudaMallocManaged(reinterpret_cast<void **>(devPtr), size, flags);
}

template <typename T>
static inline cudaError_t cudaMallocHost(T **ptr, size_t size)
{
    return cudaMallocHost(reinterpret_cast<void **>(ptr), size);
}

template <typename T>
static inline cudaError_t cudaHostAlloc(T **ptr, size_t size, unsigned int flags)
{
    return cudaHostAlloc(reinterpret_cast<void **>(ptr), size, flags);
}
#endif

#endif /* CUDA_COMPAT_MEMORY_QUERY_HOST_H */
