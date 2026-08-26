/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */

#ifndef CUDA_COMPAT_CONTEXT_H
#define CUDA_COMPAT_CONTEXT_H

#include "cann_compat_cu_types.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline CUresult cuCtxGetCurrent(CUcontext *pctx)
{
    if (!pctx) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    aclrtContext context = NULL;
    aclError ret = aclrtGetCurrentContext(&context);
    if (ret == ACL_SUCCESS) {
        *pctx = (CUcontext)context;
    }
    return acl2cuError(ret);
}


static inline CUresult cuCtxSetCurrent(CUcontext ctx)
{
    return acl2cuError(aclrtSetCurrentContext((aclrtContext)ctx));
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
