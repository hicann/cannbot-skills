/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */



#ifndef CUDA_COMPAT_MEMORY_H
#define CUDA_COMPAT_MEMORY_H

#include "cann_compat_types.h"
#include <stdlib.h>

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
    } cudaCompatHostAllocRecord;

    static inline cudaCompatHostAllocRecord *cudaCompatHostAllocRecords(void)
    {
        static cudaCompatHostAllocRecord records[64];
        return records;
    }

    static inline void cudaCompatRecordHostAlloc(void *ptr, size_t size, int registered)
    {
        cudaCompatHostAllocRecord *records = cudaCompatHostAllocRecords();
        for (size_t i = 0; i < 64; ++i) {
            if (!records[i].ptr) {
                records[i].ptr = ptr;
                records[i].size = size;
                records[i].registered = registered;
                return;
            }
        }
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
                return 1;
            }
        }
        return 0;
    }

    /* =================================================================
     * Memory Allocation/Deallocation
     * ================================================================= */


    static inline cudaError_t cudaMalloc(void **devPtr, size_t size)
    {
        aclError ret = aclrtMalloc(devPtr, size, ACL_MEM_MALLOC_HUGE_FIRST);
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
        aclError ret = aclrtMallocHost(ptr, size);
        return acl2cudaError(ret);
    }

    static inline cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags)
    {
        if (!ptr) {
            return cudaErrorInvalidValue;
        }
        if (aclrtMallocHostAndRegister) {
            uint32_t cannFlags = 0;
            cudaError_t mapRet = cudaCompatHostAllocFlagsToCann(flags, &cannFlags);
            if (mapRet != cudaSuccess) {
                return mapRet;
            }
            aclError ret = aclrtMallocHostAndRegister(ptr, size, cannFlags);
            return acl2cudaError(ret);
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
            cudaCompatRecordHostAlloc(hostPtr, size, 1);
            return cudaSuccess;
        }
        aclError ret = aclrtMallocHost(ptr, size);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaFreeHost(void *ptr)
    {
        cudaCompatHostAllocRecord record;
        if (cudaCompatTakeHostAllocRecord(ptr, &record)) {
            if (record.registered) {
                aclError unregRet = aclrtHostUnregister(ptr);
                if (unregRet != ACL_SUCCESS) {
                    return acl2cudaError(unregRet);
                }
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
        (void)size;
        (void)flags;
        *devPtr = NULL;
        return cudaErrorNotSupported;
    }

    /* =================================================================
     * Memory Copy Operations
     * ================================================================= */


    static inline cudaError_t cudaMemcpy(void *dst, const void *src,
                                         size_t count, cudaMemcpyKind kind)
    {
        aclError ret = aclrtMemcpy(dst, count, src, count, (aclrtMemcpyKind)kind);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaMemcpyAsync(void *dst, const void *src,
                                              size_t count, cudaMemcpyKind kind,
                                              cudaStream_t stream)
    {
        aclError ret = aclrtMemcpyAsync(dst, count, src, count, (aclrtMemcpyKind)kind, stream);
        return acl2cudaError(ret);
    }


    cudaError_t cudaMemcpyBatchAsync(const void **dsts, const void **srcs, const size_t *sizes, size_t count,
                                     cudaMemcpyAttributes *attrs, size_t *attrsIdxs, size_t numAttrs, cudaStream_t stream);


    static inline cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
                                           size_t spitch, size_t width, size_t height,
                                           cudaMemcpyKind kind)
    {
        aclError ret = aclrtMemcpy2d(dst, dpitch, src, spitch, width, height, (aclrtMemcpyKind)kind);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                                                size_t spitch, size_t width, size_t height,
                                                cudaMemcpyKind kind, cudaStream_t stream)
    {
        aclError ret = aclrtMemcpy2dAsync(dst, dpitch, src, spitch, width, height, (aclrtMemcpyKind)kind, stream);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaMemcpyPeer(void *dst, int dstDevice,
                                            const void *src, int srcDevice,
                                            size_t count)
    {
        (void)dstDevice;
        (void)srcDevice;

        aclError ret = aclrtMemcpy(dst, count, src, count,
                                    ACL_MEMCPY_DEVICE_TO_DEVICE);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice,
                                                  const void *src, int srcDevice,
                                                  size_t count, cudaStream_t stream)
    {
        (void)dstDevice;
        (void)srcDevice;
        // CANN supports device-to-device async copy only within the supported topology.
        aclError ret = aclrtMemcpyAsync(dst, count, src, count,
                                         ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaMemset(void *devPtr, int value, size_t count)
    {
        aclError ret = aclrtMemset(devPtr, count, value, count);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaMemsetAsync(void *devPtr, int value,
                                               size_t count, cudaStream_t stream)
    {
        aclError ret = aclrtMemsetAsync(devPtr, count, value, count, stream);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaMemset2D(void *devPtr, size_t pitch,
                                          int value, size_t width, size_t height)
    {
        // Validate parameters
        if (!devPtr || width == 0 || height == 0) {
            return cudaErrorInvalidValue;
        }

        // Set each row separately
        char *row_ptr = (char *)devPtr;
        for (size_t i = 0; i < height; i++) {
            aclError ret = aclrtMemset(row_ptr, width, value, width);
            if (ret != ACL_SUCCESS) {
                return acl2cudaError(ret);
            }
            row_ptr += pitch;
        }

        return cudaSuccess;
    }


    static inline cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch,
                                               int value, size_t width,
                                               size_t height, cudaStream_t stream)
    {
        // Validate parameters
        if (!devPtr || width == 0 || height == 0) {
            return cudaErrorInvalidValue;
        }

        // Set each row separately asynchronously
        char *row_ptr = (char *)devPtr;
        for (size_t i = 0; i < height; i++) {
            aclError ret = aclrtMemsetAsync(row_ptr, width, value, width, stream);
            if (ret != ACL_SUCCESS) {
                return acl2cudaError(ret);
            }
            row_ptr += pitch;
        }

        return cudaSuccess;
    }

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
        (void)devPtr;
        (void)count;
        (void)advice;
        (void)device;
        return cudaErrorNotSupported;
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

        aclError ret = aclrtHostRegisterV2(ptr, size, cannFlags);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaHostUnregister(void *ptr)
    {
        if (!ptr) {
            return cudaErrorInvalidValue;
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
        aclError ret = aclrtHostGetDevicePointer(pHost, pDevice, flags);
        return acl2cudaError(ret);
    }

#ifdef __cplusplus
}

template <typename T>
static inline cudaError_t cudaMalloc(T **devPtr, size_t size)
{
    return cudaMalloc(reinterpret_cast<void **>(devPtr), size);
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

#endif /* CUDA_COMPAT_MEMORY_H */
