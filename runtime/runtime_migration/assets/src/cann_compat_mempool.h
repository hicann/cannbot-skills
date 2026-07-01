/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */



#ifndef CUDA_COMPAT_MEMPOOL_H
#define CUDA_COMPAT_MEMPOOL_H

#include "cann_compat_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =================================================================
 * Memory Pool Creation and Destruction
 * ================================================================= */


static inline cudaError_t cudaMemPoolCreate(cudaMemPool_t *memPool, cudaMemPoolProps *props)
{
    if (!memPool || !props) {
        return cudaErrorInvalidValue;
    }

    // CANN does not support MemPool, return not supported
    // In future, could implement a simple pool wrapper around cudaMalloc
    return cudaErrorNotSupported;
}


static inline cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool)
{
    (void)memPool;
    return cudaErrorNotSupported;
}

/* =================================================================
 * Memory Pool Attributes
 * ================================================================= */


static inline cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool,
                                                   cudaMemPoolAttr attr,
                                                   void *value)
{
    (void)memPool;
    (void)attr;
    (void)value;
    return cudaErrorNotSupported;
}


static inline cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool,
                                                   cudaMemPoolAttr attr,
                                                   void *value)
{
    (void)memPool;
    (void)attr;
    (void)value;
    return cudaErrorNotSupported;
}

/* =================================================================
 * Memory Pool Allocation and Free
 * ================================================================= */


static inline cudaError_t cudaMemPoolMalloc(void **ptr,
                                             cudaMemPool_t memPool,
                                             size_t size)
{
    if (!ptr) {
        return cudaErrorInvalidValue;
    }

    (void)memPool;
    // Fallback to cudaMalloc, but return not supported for true pool behavior
    return cudaErrorNotSupported;
}


static inline cudaError_t cudaMemPoolFree(void *ptr,
                                           cudaMemPool_t memPool,
                                           cudaStream_t stream)
{
    (void)ptr;
    (void)memPool;
    (void)stream;
    return cudaErrorNotSupported;
}


static inline cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool,
                                             size_t size)
{
    (void)memPool;
    (void)size;
    return cudaErrorNotSupported;
}

/* =================================================================
 * Memory Pool Access Control
 * ================================================================= */


static inline cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool,
                                                const cudaMemAccessDesc *desc,
                                                size_t descCount)
{
    (void)memPool;
    (void)desc;
    (void)descCount;
    return cudaErrorNotSupported;
}


static inline cudaError_t cudaMemPoolGetAccess(cudaMemAccessFlags *access,
                                                cudaMemPool_t memPool,
                                                cudaMemLocation *location)
{
    if (!access) {
        return cudaErrorInvalidValue;
    }

    (void)memPool;
    (void)location;

    // Return default no access since pools are not supported
    *access = cudaMemAccessNone;
    return cudaErrorNotSupported;
}

/* =================================================================
 * Default Memory Pool
 * ================================================================= */


static inline cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool,
                                                       int device)
{
    if (!memPool) {
        return cudaErrorInvalidValue;
    }

    (void)device;

    // No default pool in CANN
    *memPool = NULL;
    return cudaErrorNotSupported;
}


static inline cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool)
{
    (void)device;
    (void)memPool;
    return cudaErrorNotSupported;
}


static inline cudaError_t cudaDeviceGetMemPool(cudaMemPool_t *memPool, int device)
{
    if (!memPool) {
        return cudaErrorInvalidValue;
    }

    (void)device;
    *memPool = NULL;
    return cudaErrorNotSupported;
}

/* =================================================================
 * Memory Pool Edge Access (Multi-Device)
 * ================================================================= */


static inline cudaError_t cudaMemPoolAddAccess(cudaMemPool_t memPool,
                                                const cudaMemAccessDesc *desc,
                                                size_t descCount)
{
    (void)memPool;
    (void)desc;
    (void)descCount;
    return cudaErrorNotSupported;
}


static inline cudaError_t cudaMemPoolRemoveAccess(cudaMemPool_t memPool,
                                                   const cudaMemAccessDesc *desc,
                                                   size_t descCount)
{
    (void)memPool;
    (void)desc;
    (void)descCount;
    return cudaErrorNotSupported;
}

/* =================================================================
 * Async Memory Allocation (MemPool-based)
 * ================================================================= */


static inline cudaError_t cudaMallocAsync(void **ptr,
                                           size_t size,
                                           cudaStream_t stream)
{
    if (!ptr) {
        return cudaErrorInvalidValue;
    }

    (void)size;
    (void)stream;

    // CANN does not support async memory allocation
    *ptr = NULL;
    return cudaErrorNotSupported;
}


static inline cudaError_t cudaFreeAsync(void *ptr, cudaStream_t stream)
{
    (void)ptr;
    (void)stream;
    return cudaErrorNotSupported;
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_MEMPOOL_H */