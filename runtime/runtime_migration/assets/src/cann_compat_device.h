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


static inline cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value){
    (void)(limit);
    (void)(value);
    // CANN handles limits internally, ignore for now
    return cudaSuccess;
}


cudaError_t cudaDeviceGetLimit(size_t *pValue, cudaLimit limit);


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
    aclError ret = aclrtDeviceEnablePeerAccess(peerDevice, flags);
    return acl2cudaError(ret);
}


static inline cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) {
    aclError ret = aclrtDeviceDisablePeerAccess(peerDevice);
    return acl2cudaError(ret);
}


static inline cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice) {
    aclError ret = aclrtDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
    return acl2cudaError(ret);
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_DEVICE_H */
