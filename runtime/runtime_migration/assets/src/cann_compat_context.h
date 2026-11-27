/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */

#ifndef CUDA_COMPAT_CONTEXT_H
#define CUDA_COMPAT_CONTEXT_H

#include "cann_compat_cu_types.h"
#include "cann_compat_types.h"
#include "acl/error_codes/rt_error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline CUresult cuCtxGetCurrent(CUcontext *pctx)
{
    if (!pctx) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pctx = NULL;
    if (cudaCompatContextIsDetached()) {
        return CUDA_SUCCESS;
    }
    aclrtContext context = NULL;
    aclError ret = aclrtGetCurrentContext(&context);
    if (ret == ACL_ERROR_RT_CONTEXT_NULL) {
        return CUDA_SUCCESS;
    }
    if (ret == ACL_SUCCESS) {
        *pctx = (CUcontext)context;
        cudaCompatContextSetDetached(0);
    }
    return acl2cuError(ret);
}

static inline CUresult cuCtxCreate(CUcontext *pctx,
                                   CUctxCreateParams *ctxCreateParams,
                                   unsigned int flags, CUdevice dev)
{
    if (!pctx) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pctx = NULL;
    /* CANN's context creation API has no CUDA scheduling, affinity or CIG
     * parameters. Reject these instead of silently changing their meaning. */
    if (ctxCreateParams != NULL || flags != 0) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    aclrtContext context = NULL;
    aclError ret = aclrtCreateContext(&context, (int32_t)dev);
    if (ret == ACL_SUCCESS) {
        *pctx = (CUcontext)context;
        cudaCompatContextSetDetached(0);
    }
    return acl2cuError(ret);
}


static inline CUresult cuCtxDestroy(CUcontext ctx)
{
    if (!ctx) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    return acl2cuError(aclrtDestroyContext((aclrtContext)ctx));
}


static inline CUresult cuCtxGetDevice(CUdevice *device)
{
    if (!device) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (cudaCompatContextIsDetached()) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    int32_t cannDevice = -1;
    aclError ret = aclrtGetDevice(&cannDevice);
    if (ret == ACL_SUCCESS) {
        *device = (CUdevice)cannDevice;
    }
    return acl2cuError(ret);
}


static inline CUresult cuCtxGetStreamPriorityRange(int *leastPriority,
                                                   int *greatestPriority)
{
    if (!leastPriority && !greatestPriority) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    int32_t least = 0;
    int32_t greatest = 0;
    aclError ret = aclrtDeviceGetStreamPriorityRange(&least, &greatest);
    if (ret == ACL_SUCCESS) {
        if (leastPriority) {
            *leastPriority = (int)least;
        }
        if (greatestPriority) {
            *greatestPriority = (int)greatest;
        }
    }
    return acl2cuError(ret);
}


static inline CUresult cuCtxSetCurrent(CUcontext ctx)
{
    if (!ctx) {
        cudaCompatContextSetDetached(1);
        return CUDA_SUCCESS;
    }
    aclError ret = aclrtSetCurrentContext((aclrtContext)ctx);
    if (ret == ACL_SUCCESS) {
        cudaCompatContextSetDetached(0);
    }
    return acl2cuError(ret);
}


static inline CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active)
{
    if (!flags || !active) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    int32_t cannActive = 0;
    aclError ret = aclrtGetPrimaryCtxState((int32_t)dev, NULL, &cannActive);
    if (ret == ACL_SUCCESS) {
        *flags = 0;
        *active = cannActive;
    }
    return acl2cuError(ret);
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_CONTEXT_H */
