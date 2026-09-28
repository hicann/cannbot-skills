/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef CUDA_COMPAT_MEMORY_CORE_H
#define CUDA_COMPAT_MEMORY_CORE_H

#include "cann_compat_safe.h"
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /* =================================================================
     * Weak Symbol Declaration for Version Compatibility
     * ================================================================= */


    __attribute__((weak)) aclError aclrtMemcpyBatchAsyncV2(void **dsts, size_t *destMaxs, void **srcs, size_t *sizes,
        size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes, size_t numAttrs, aclrtStream stream);

    __attribute__((weak)) aclError aclrtMallocHostAndRegister(void **ptr, size_t size, uint32_t flag);

    __attribute__((weak)) aclError aclrtMemAllocManaged(void **ptr, uint64_t size, uint32_t flag);
    __attribute__((weak)) aclError aclrtMemManagedAdvise(const void *const ptr, uint64_t size,
        aclrtMemManagedAdviseType advise, aclrtMemManagedLocation location);
    __attribute__((weak)) aclError aclrtMemManagedPrefetchAsync(const void *ptr, size_t size,
        aclrtMemManagedLocation location, uint32_t flags, aclrtStream stream);
    __attribute__((weak)) aclError aclrtMemManagedPrefetchBatchAsync(const void **ptrs, size_t *sizes, size_t count,
        aclrtMemManagedLocation *prefetchLocs, size_t *prefetchLocIdxs, size_t numPrefetchLocs, uint64_t flags,
        aclrtStream stream);
    __attribute__((weak)) aclError aclrtMemManagedGetAttr(aclrtMemManagedRangeAttribute attribute,
        const void *ptr, size_t size, void *data, size_t dataSize);
    __attribute__((weak)) aclError aclrtMemManagedGetAttrs(aclrtMemManagedRangeAttribute *attributes,
        size_t numAttributes, const void *ptr, size_t size, void **data, size_t *dataSizes);

    static inline cudaError_t cudaCompatMemLocationToCann(cudaMemLocation location,
                                                          aclrtMemManagedLocation *cannLocation)
    {
        if (!cannLocation) {
            return cudaErrorInvalidValue;
        }
        switch (location.type) {
        case cudaMemLocationTypeInvalid:
            cannLocation->type = ACL_MEM_LOCATIONTYPE_INVALID;
            cannLocation->id = 0;
            return cudaSuccess;
        case cudaMemLocationTypeDevice:
            cannLocation->type = ACL_MEM_LOCATIONTYPE_DEVICE;
            cannLocation->id = location.id;
            return cudaSuccess;
        case cudaMemLocationTypeHost:
            cannLocation->type = ACL_MEM_LOCATIONTYPE_HOST;
            cannLocation->id = 0;
            return cudaSuccess;
        case cudaMemLocationTypeHostNuma:
            cannLocation->type = ACL_MEM_LOCATIONTYPE_HOST_NUMA;
            cannLocation->id = location.id;
            return cudaSuccess;
        case cudaMemLocationTypeHostNumaCurrent:
            cannLocation->type = ACL_MEM_LOCATIONTYPE_HOST_NUMA_CURRENT;
            cannLocation->id = 0;
            return cudaSuccess;
        default:
            return cudaErrorInvalidValue;
        }
    }

    static inline cudaError_t cudaCompatDeviceToManagedLocation(int device,
                                                               aclrtMemManagedLocation *cannLocation)
    {
        if (!cannLocation) {
            return cudaErrorInvalidValue;
        }
        if (device < 0) {
            cannLocation->type = ACL_MEM_LOCATIONTYPE_HOST;
            cannLocation->id = 0;
        } else {
            cannLocation->type = ACL_MEM_LOCATIONTYPE_DEVICE;
            cannLocation->id = device;
        }
        return cudaSuccess;
    }

    static inline cudaError_t cudaCompatMemAdviseToCann(cudaMemoryAdvise advice,
                                                        aclrtMemManagedAdviseType *cannAdvice)
    {
        if (!cannAdvice) {
            return cudaErrorInvalidValue;
        }
        switch (advice) {
        case cudaMemAdviseSetReadMostly:
            *cannAdvice = ACL_MEM_ADVISE_SET_READ_MOSTLY;
            return cudaSuccess;
        case cudaMemAdviseUnsetReadMostly:
            *cannAdvice = ACL_MEM_ADVISE_UNSET_READ_MOSTLY;
            return cudaSuccess;
        case cudaMemAdviseSetPreferredLocation:
            *cannAdvice = ACL_MEM_ADVISE_SET_PREFERRED_LOCATION;
            return cudaSuccess;
        case cudaMemAdviseUnsetPreferredLocation:
            *cannAdvice = ACL_MEM_ADVISE_UNSET_PREFERRED_LOCATION;
            return cudaSuccess;
        case cudaMemAdviseSetAccessedBy:
            *cannAdvice = ACL_MEM_ADVISE_SET_ACCESSED_BY;
            return cudaSuccess;
        case cudaMemAdviseUnsetAccessedBy:
            *cannAdvice = ACL_MEM_ADVISE_UNSET_ACCESSED_BY;
            return cudaSuccess;
        default:
            return cudaErrorInvalidValue;
        }
    }

    static inline cudaError_t cudaCompatMemRangeAttributeToCann(cudaMemRangeAttribute attribute,
                                                                aclrtMemManagedRangeAttribute *cannAttribute)
    {
        if (!cannAttribute) {
            return cudaErrorInvalidValue;
        }
        switch (attribute) {
        case cudaMemRangeAttributeReadMostly:
            *cannAttribute = ACL_MEM_RANGE_ATTRIBUTE_READ_MOSTLY;
            return cudaSuccess;
        case cudaMemRangeAttributePreferredLocation:
            *cannAttribute = ACL_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION;
            return cudaSuccess;
        case cudaMemRangeAttributeAccessedBy:
            *cannAttribute = ACL_MEM_RANGE_ATTRIBUTE_ACCESSED_BY;
            return cudaSuccess;
        case cudaMemRangeAttributeLastPrefetchLocation:
            *cannAttribute = ACL_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION;
            return cudaSuccess;
        default:
            return cudaErrorInvalidValue;
        }
    }

    static inline cudaError_t cudaCompatHostAllocFlagsToCann(unsigned int cudaFlags,
                                                             uint32_t *cannFlags)
    {
        const unsigned int supportedFlags = cudaHostAllocPortable |
                                            cudaHostAllocMapped |
                                            cudaHostAllocWriteCombined;
        if (!cannFlags || (cudaFlags & ~supportedFlags) != 0) {
            return cudaErrorInvalidValue;
        }

        uint32_t flags = ACL_HOST_REG_PINNED;
        if ((cudaFlags & cudaHostAllocMapped) != 0) {
            flags |= ACL_HOST_REG_MAPPED;
        }

        *cannFlags = flags;
        return cudaSuccess;
    }

    typedef struct cudaCompatHostAllocRecord {
        void *ptr;
        size_t size;
        int registered;
        int mallocHost;
        unsigned int flags;
    } cudaCompatHostAllocRecord;

    static inline cudaCompatHostAllocRecord *cudaCompatHostAllocRecords(void)
    {
        static cudaCompatHostAllocRecord records[64];
        return records;
    }

    static inline void cudaCompatRecordHostAlloc(void *ptr, size_t size, int registered, int mallocHost, unsigned int flags)
    {
        cudaCompatHostAllocRecord *records = cudaCompatHostAllocRecords();
        for (size_t i = 0; i < 64; ++i) {
            if (!records[i].ptr) {
                records[i].ptr = ptr;
                records[i].size = size;
                records[i].registered = registered;
                records[i].mallocHost = mallocHost;
                records[i].flags = flags;
                return;
            }
        }
    }

    static inline int cudaCompatFindHostAllocRecord(void *ptr, cudaCompatHostAllocRecord *record)
    {
        cudaCompatHostAllocRecord *records = cudaCompatHostAllocRecords();
        for (size_t i = 0; i < 64; ++i) {
            if (records[i].ptr == ptr) {
                if (record) {
                    *record = records[i];
                }
                return 1;
            }
        }
        return 0;
    }

    static inline int cudaCompatTakeHostAllocRecord(void *ptr, cudaCompatHostAllocRecord *record)
    {
        cudaCompatHostAllocRecord *records = cudaCompatHostAllocRecords();
        for (size_t i = 0; i < 64; ++i) {
            if (records[i].ptr == ptr) {
                if (record) {
                    *record = records[i];
                }
                records[i].ptr = NULL;
                records[i].size = 0;
                records[i].registered = 0;
                records[i].mallocHost = 0;
                records[i].flags = 0;
                return 1;
            }
        }
        return 0;
    }

    /* =================================================================
     * Memory Allocation/Deallocation
     * ================================================================= */

    static inline const char *cudaCompatSecureGetenv(const char *name)
    {
#if defined(__GLIBC__) && defined(__USE_GNU)
        return secure_getenv(name);
#else
        (void)name;
        return NULL;
#endif
    }

    static inline aclrtMemMallocPolicy cudaCompatGetDeviceMallocPolicy(void)
    {
        const char *policy = cudaCompatSecureGetenv("CUDA_COMPAT_DEVICE_MALLOC_POLICY");
        if (policy != NULL && (strcmp(policy, "p2p") == 0 || strcmp(policy, "P2P") == 0)) {
            return ACL_MEM_MALLOC_HUGE_FIRST_P2P;
        }
        return ACL_MEM_MALLOC_HUGE_FIRST;
    }

    static inline cudaError_t cudaCompatEnsureMemoryDevice(void)
    {
        int currentDevice = 0;
        aclError ret = aclrtGetDevice(&currentDevice);
        if (ret == ACL_SUCCESS) {
            return cudaSuccess;
        }

        uint32_t deviceCount = 0;
        aclError countRet = aclrtGetDeviceCount(&deviceCount);
        if (countRet != ACL_SUCCESS) {
            return acl2cudaError(countRet);
        }
        if (deviceCount == 0) {
            return cudaErrorNoDevice;
        }

        ret = aclrtSetDevice(0);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaMalloc(void **devPtr, size_t size)
    {
        cudaError_t deviceRet = cudaCompatEnsureMemoryDevice();
        if (deviceRet != cudaSuccess) {
            return deviceRet;
        }
        aclError ret = aclrtMalloc(devPtr, size, cudaCompatGetDeviceMallocPolicy());
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaFree(void *devPtr)
    {
        aclError ret = aclrtFree(devPtr);
        return acl2cudaError(ret);
    }


    cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch,
                                size_t width, size_t height);


    static inline cudaError_t cudaMallocHost(void **ptr, size_t size)
    {
        cudaError_t deviceRet = cudaCompatEnsureMemoryDevice();
        if (deviceRet != cudaSuccess) {
            return deviceRet;
        }
        aclError ret = aclrtMallocHost(ptr, size);
        cudaError_t cudaRet = acl2cudaError(ret);
        if (cudaRet == cudaSuccess) {
            cudaCompatRecordHostAlloc(*ptr, size, 0, 1, 0);
        }
        return cudaRet;
    }

    static inline cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags)
    {
        if (!ptr) {
            return cudaErrorInvalidValue;
        }
        cudaError_t deviceRet = cudaCompatEnsureMemoryDevice();
        if (deviceRet != cudaSuccess) {
            return deviceRet;
        }
        if (aclrtMallocHostAndRegister) {
            uint32_t cannFlags = 0;
            cudaError_t mapRet = cudaCompatHostAllocFlagsToCann(flags, &cannFlags);
            if (mapRet != cudaSuccess) {
                return mapRet;
            }
            aclError ret = aclrtMallocHostAndRegister(ptr, size, cannFlags);
            cudaError_t cudaRet = acl2cudaError(ret);
            if (cudaRet == cudaSuccess) {
                cudaCompatRecordHostAlloc(*ptr, size, 0, 1, flags);
            }
            return cudaRet;
        }
        if ((flags & cudaHostAllocMapped) != 0) {
            uint32_t cannFlags = 0;
            cudaError_t mapRet = cudaCompatHostAllocFlagsToCann(flags, &cannFlags);
            if (mapRet != cudaSuccess) {
                return mapRet;
            }

            void *hostPtr = NULL;
            if (posix_memalign(&hostPtr, 4096, size) != 0) {
                return cudaErrorMemoryAllocation;
            }

            aclError ret = aclrtHostRegisterV2(hostPtr, size, cannFlags);
            if (ret != ACL_SUCCESS) {
                free(hostPtr);
                return acl2cudaError(ret);
            }

            *ptr = hostPtr;
            cudaCompatRecordHostAlloc(hostPtr, size, 1, 0, flags);
            return cudaSuccess;
        }
        aclError ret = aclrtMallocHost(ptr, size);
        cudaError_t cudaRet = acl2cudaError(ret);
        if (cudaRet == cudaSuccess) {
            cudaCompatRecordHostAlloc(*ptr, size, 0, 1, flags);
        }
        return cudaRet;
    }


    static inline cudaError_t cudaFreeHost(void *ptr)
    {
        cudaError_t deviceRet = cudaCompatEnsureMemoryDevice();
        if (deviceRet != cudaSuccess) {
            return deviceRet;
        }
        cudaCompatHostAllocRecord record;
        if (cudaCompatTakeHostAllocRecord(ptr, &record)) {
            if (record.registered) {
                aclError unregRet = aclrtHostUnregister(ptr);
                if (unregRet != ACL_SUCCESS) {
                    return acl2cudaError(unregRet);
                }
            }
            if (record.mallocHost) {
                aclError freeRet = aclrtFreeHost(ptr);
                return acl2cudaError(freeRet);
            }
            free(ptr);
            return cudaSuccess;
        }
        aclError ret = aclrtFreeHost(ptr);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaMallocManaged(void **devPtr, size_t size
#ifdef __cplusplus
                                                , unsigned int flags = cudaMemAttachGlobal
#else
                                                , unsigned int flags
#endif
    )
    {
        if (!devPtr) {
            return cudaErrorInvalidValue;
        }
        if (aclrtMemAllocManaged == NULL) {
            *devPtr = NULL;
            return cudaErrorNotSupported;
        }
        if (flags != cudaMemAttachGlobal) {
            *devPtr = NULL;
            return cudaErrorInvalidValue;
        }
        cudaError_t deviceRet = cudaCompatEnsureMemoryDevice();
        if (deviceRet != cudaSuccess) {
            *devPtr = NULL;
            return deviceRet;
        }
        aclError ret = aclrtMemAllocManaged(devPtr, size, ACL_RT_MEM_ATTACH_GLOBAL);
        return acl2cudaError(ret);
    }

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_MEMORY_CORE_H */
