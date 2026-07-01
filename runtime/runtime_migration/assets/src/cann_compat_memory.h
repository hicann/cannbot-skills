/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */



#ifndef CUDA_COMPAT_MEMORY_H
#define CUDA_COMPAT_MEMORY_H

#include "cann_compat_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* =================================================================
     * Weak Symbol Declaration for Version Compatibility
     * ================================================================= */


    __attribute__((weak)) aclError aclrtMemcpyBatchAsyncV2(void **dsts, size_t *destMaxs, void **srcs, size_t *sizes,
        size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes, size_t numAttrs, aclrtStream stream);

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


    static inline cudaError_t cudaFreeHost(void *ptr)
    {
        aclError ret = aclrtFreeHost(ptr);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags)
    {
        if (!devPtr) {
            return cudaErrorInvalidValue;
        }

        // Map CUDA flags to CANN flags (direct pass for now)
        uint32_t cannFlag = (uint32_t)flags;

        aclError ret = aclrtMemAllocManaged(devPtr, (uint64_t)size, cannFlag);
        return acl2cudaError(ret);
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
        // cann runtime can handle device-to-device copy without explicit device context management
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

    /* =================================================================
     * Host Memory Registration
     * ================================================================= */


    static inline cudaError_t cudaHostRegister(void *ptr, size_t size,
                                               unsigned int flags)
    {
        aclError ret = aclrtHostRegisterV2(ptr, size, flags);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaHostUnregister(void *ptr)
    {
        aclError ret = aclrtHostUnregister(ptr);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost,
                                                       unsigned int flags)
    {
        aclError ret = aclrtHostGetDevicePointer(pHost, pDevice, flags);
        return acl2cudaError(ret);
    }

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_MEMORY_H */
