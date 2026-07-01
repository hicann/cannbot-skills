/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */



#ifndef CUDA_COMPAT_EXEC_H
#define CUDA_COMPAT_EXEC_H

#include "cann_compat_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef void (*cudaHostFn_t)(void *userData);
    typedef cudaError_t (*cudaCompatHostKernelFn_t)(void *userData);


    static inline cudaError_t cudaCompatLaunchHostKernel(cudaCompatHostKernelFn_t fn,
                                                         void *userData)
    {
        if (!fn) {
            return cudaErrorInvalidValue;
        }
        return fn(userData);
    }


    static inline cudaError_t cudaLaunchHostFunc(cudaStream_t stream,
                                                 cudaHostFn_t fn,
                                                 void *userData)
    {
        aclError ret = aclrtLaunchHostFunc(stream, (aclrtHostFunc)fn, userData);
        return acl2cudaError(ret);
    }

    static inline cudaError_t cudaLaunchHostFunc_v2(cudaStream_t stream,
                                                    cudaHostFn_t fn,
                                                    void *userData,
                                                    unsigned int syncMode)
    {
        (void)syncMode; // CANN does not support different sync modes for host functions
        aclError ret = aclrtLaunchHostFunc(stream, (aclrtHostFunc)fn, userData);
        return acl2cudaError(ret);
    }

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_EXEC_H */
