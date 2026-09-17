/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */



#ifndef CUDA_COMPAT_DEVICE_H
#define CUDA_COMPAT_DEVICE_H

#include "cann_compat_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =================================================================
 * Device Management APIs
 * ================================================================= */


static inline cudaError_t cudaGetDeviceCount(int *count) {
    uint32_t dev_count = 0;
    aclError ret = aclrtGetDeviceCount(&dev_count);
    if (ret == ACL_SUCCESS) {
        *count = (int)dev_count;
    }
    return acl2cudaError(ret);
}


static inline cudaError_t cudaSetDevice(int device) {
    aclError ret = aclrtSetDevice(device);
    return acl2cudaError(ret);
}


static inline cudaError_t cudaGetDevice(int *device) {
    aclError ret = aclrtGetDevice(device);
    return acl2cudaError(ret);
}


cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);


cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device);


static inline cudaError_t cudaDeviceReset(void) {
    int devId;
    aclError ret = aclrtGetDevice(&devId);
    if (ret != ACL_SUCCESS) {
        return acl2cudaError(ret);
    }
    ret = aclrtResetDeviceForce(devId);
    return acl2cudaError(ret);
}


static inline cudaError_t cudaDeviceSynchronize(void) {
    aclError ret = aclrtSynchronizeDevice();
    return acl2cudaError(ret);
}


static inline cudaError_t cudaSetDeviceFlags(unsigned int flags) {
    g_cuda_context.flags = flags;
    // CANN doesn't have direct flags, store for future reference
    return cudaSuccess;
}


static inline cudaError_t cudaGetDeviceFlags(unsigned int *flags) {
    *flags = g_cuda_context.flags;
    return cudaSuccess;
}


static inline cudaError_t cudaCompatLimitToAcl(cudaLimit limit, aclrtDeviceLimit *aclLimit) {
    if (aclLimit == NULL) {
        return cudaErrorInvalidValue;
    }

    switch (limit) {
        case cudaLimitStackSize:
            *aclLimit = ACL_RT_DEV_LIMIT_SIMD_STACK_SIZE;
            return cudaSuccess;
        case cudaLimitPrintfFifoSize:
            *aclLimit = ACL_RT_DEV_LIMIT_SIMD_PRINTF_FIFO_SIZE_PER_CORE;
            return cudaSuccess;
        default:
            return cudaErrorUnsupportedLimit;
    }
}


static inline cudaError_t cudaCompatLimitAclResult(aclError ret) {
    cudaError_t err = acl2cudaError(ret);
    return (err == cudaErrorNotSupported) ? cudaErrorUnsupportedLimit : err;
}


static inline cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value) {
    aclrtDeviceLimit aclLimit;
    cudaError_t err = cudaCompatLimitToAcl(limit, &aclLimit);
    if (err != cudaSuccess) {
        return err;
    }

    aclError ret = aclrtDeviceSetLimit(aclLimit, value);
    return cudaCompatLimitAclResult(ret);
}


cudaError_t cudaDeviceGetLimit(size_t *pValue, cudaLimit limit);


static inline cudaError_t cudaCompatAtomicOperationToAcl(cudaAtomicOperation operation,
                                                        aclrtAtomicOperation *aclOperation) {
    if (aclOperation == NULL) {
        return cudaErrorInvalidValue;
    }

    switch (operation) {
        case cudaAtomicOperationIntegerAdd:
        case cudaAtomicOperationIntegerMin:
        case cudaAtomicOperationIntegerMax:
        case cudaAtomicOperationIntegerIncrement:
        case cudaAtomicOperationIntegerDecrement:
        case cudaAtomicOperationAnd:
        case cudaAtomicOperationOr:
        case cudaAtomicOperationXOR:
        case cudaAtomicOperationExchange:
        case cudaAtomicOperationCAS:
        case cudaAtomicOperationFloatAdd:
        case cudaAtomicOperationFloatMin:
        case cudaAtomicOperationFloatMax:
            *aclOperation = (aclrtAtomicOperation)operation;
            return cudaSuccess;
        default:
            return cudaErrorInvalidValue;
    }
}


static inline unsigned int cudaCompatAtomicCapabilitiesToCuda(uint32_t aclCapabilities) {
    unsigned int cudaCapabilities = 0;
    if ((aclCapabilities & ACL_RT_ATOMIC_CAPABILITY_SIGNED) != 0) {
        cudaCapabilities |= cudaAtomicCapabilitySigned;
    }
    if ((aclCapabilities & ACL_RT_ATOMIC_CAPABILITY_UNSIGNED) != 0) {
        cudaCapabilities |= cudaAtomicCapabilityUnsigned;
    }
    if ((aclCapabilities & ACL_RT_ATOMIC_CAPABILITY_REDUCTION) != 0) {
        cudaCapabilities |= cudaAtomicCapabilityReduction;
    }
    if ((aclCapabilities & ACL_RT_ATOMIC_CAPABILITY_SCALAR32) != 0) {
        cudaCapabilities |= cudaAtomicCapabilityScalar32;
    }
    if ((aclCapabilities & ACL_RT_ATOMIC_CAPABILITY_SCALAR64) != 0) {
        cudaCapabilities |= cudaAtomicCapabilityScalar64;
    }
    if ((aclCapabilities & ACL_RT_ATOMIC_CAPABILITY_SCALAR128) != 0) {
        cudaCapabilities |= cudaAtomicCapabilityScalar128;
    }
    if ((aclCapabilities & ACL_RT_ATOMIC_CAPABILITY_VECTOR32X4) != 0) {
        cudaCapabilities |= cudaAtomicCapabilityVector32x4;
    }
    return cudaCapabilities;
}


static inline cudaError_t cudaCompatAllocCheckedArray(size_t count, size_t elementSize, void **buffer) {
    if (buffer == NULL || count == 0U || elementSize == 0U) {
        return cudaErrorInvalidValue;
    }
    if (count > (SIZE_MAX / elementSize)) {
        return cudaErrorInvalidValue;
    }
    size_t allocSize = count * elementSize;
    if (allocSize == 0U) {
        return cudaErrorInvalidValue;
    }
    *buffer = malloc(allocSize);
    if (*buffer == NULL) {
        return cudaErrorMemoryAllocation;
    }
    return cudaSuccess;
}


static inline cudaError_t cudaCompatPrepareAtomicOperations(aclrtAtomicOperation **aclOperations,
                                                           const cudaAtomicOperation *operations,
                                                           unsigned int count) {
    if (aclOperations == NULL || operations == NULL || count == 0U) {
        return cudaErrorInvalidValue;
    }
    void *buffer = NULL;
    cudaError_t err = cudaCompatAllocCheckedArray((size_t)count, sizeof(aclrtAtomicOperation), &buffer);
    if (err != cudaSuccess) {
        return err;
    }
    *aclOperations = (aclrtAtomicOperation *)buffer;

    for (unsigned int i = 0; i < count; ++i) {
        err = cudaCompatAtomicOperationToAcl(operations[i], &(*aclOperations)[i]);
        if (err != cudaSuccess) {
            free(*aclOperations);
            *aclOperations = NULL;
            return err;
        }
    }
    return cudaSuccess;
}


static inline cudaError_t cudaCompatPrepareAtomicCapabilityBuffers(
    aclrtAtomicOperation **aclOperations, uint32_t **aclCapabilities,
    const cudaAtomicOperation *operations, unsigned int count) {
    if (aclCapabilities == NULL) {
        return cudaErrorInvalidValue;
    }
    *aclCapabilities = NULL;

    cudaError_t err = cudaCompatPrepareAtomicOperations(aclOperations, operations, count);
    if (err != cudaSuccess) {
        return err;
    }
    void *buffer = NULL;
    cudaError_t sizeErr = cudaCompatAllocCheckedArray((size_t)count, sizeof(uint32_t), &buffer);
    if (sizeErr != cudaSuccess) {
        free(*aclOperations);
        *aclOperations = NULL;
        return sizeErr;
    }
    *aclCapabilities = (uint32_t *)buffer;
    return cudaSuccess;
}


static inline cudaError_t cudaCompatFinishAtomicCapabilities(unsigned int *capabilities,
                                                            uint32_t *aclCapabilities,
                                                            aclrtAtomicOperation *aclOperations,
                                                            unsigned int count,
                                                            aclError ret) {
    cudaError_t err = acl2cudaError(ret);
    if (err == cudaSuccess) {
        for (unsigned int i = 0; i < count; ++i) {
            capabilities[i] = cudaCompatAtomicCapabilitiesToCuda(aclCapabilities[i]);
        }
    }

    free(aclCapabilities);
    free(aclOperations);
    return err;
}


static inline cudaError_t cudaDeviceGetHostAtomicCapabilities(unsigned int *capabilities,
                                                              const cudaAtomicOperation *operations,
                                                              unsigned int count,
                                                              int device) {
    if (capabilities == NULL || operations == NULL || count == 0U) {
        return cudaErrorInvalidValue;
    }

    aclrtAtomicOperation *aclOperations = NULL;
    uint32_t *aclCapabilities = NULL;
    cudaError_t err = cudaCompatPrepareAtomicCapabilityBuffers(&aclOperations, &aclCapabilities,
                                                               operations, count);
    if (err != cudaSuccess) {
        return err;
    }

    aclError ret = aclrtDeviceGetHostAtomicCapabilities(aclCapabilities, aclOperations, count, device);
    return cudaCompatFinishAtomicCapabilities(capabilities, aclCapabilities, aclOperations, count, ret);
}


static inline cudaError_t cudaDeviceGetP2PAtomicCapabilities(unsigned int *capabilities,
                                                             const cudaAtomicOperation *operations,
                                                             unsigned int count,
                                                             int srcDevice,
                                                             int dstDevice) {
    if (capabilities == NULL || operations == NULL || count == 0U) {
        return cudaErrorInvalidValue;
    }
    if (srcDevice == dstDevice) {
        return cudaErrorInvalidDevice;
    }

    aclrtAtomicOperation *aclOperations = NULL;
    uint32_t *aclCapabilities = NULL;
    cudaError_t err = cudaCompatPrepareAtomicCapabilityBuffers(&aclOperations, &aclCapabilities,
                                                               operations, count);
    if (err != cudaSuccess) {
        return err;
    }

    aclError ret = aclrtDeviceGetP2PAtomicCapabilities(aclCapabilities, aclOperations, count,
                                                       srcDevice, dstDevice);
    return cudaCompatFinishAtomicCapabilities(capabilities, aclCapabilities, aclOperations, count, ret);
}


static inline cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache *pCacheConfig) {
    // CANN uses different cache model, return default
    *pCacheConfig = cudaFuncCachePreferNone;
    return cudaSuccess;
}


static inline cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig) {
    (void)(cacheConfig);
    // CANN handles cache internally
    return cudaSuccess;
}


static inline cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority,
                                                            int *greatestPriority) {
    aclError ret = aclrtDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
    return acl2cudaError(ret);
}

/* =================================================================
 * Peer Device Memory Access
 * ================================================================= */


static inline cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    if (flags != 0) {
        return cudaErrorInvalidValue;
    }
    aclError ret = aclrtDeviceEnablePeerAccess(peerDevice, flags);
    return acl2cudaError(ret);
}


static inline cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) {
    aclError ret = aclrtDeviceDisablePeerAccess(peerDevice);
    return acl2cudaError(ret);
}


static inline cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice) {
    if (canAccessPeer == NULL) {
        return cudaErrorInvalidValue;
    }
    aclError ret = aclrtDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
    return acl2cudaError(ret);
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_DEVICE_H */
