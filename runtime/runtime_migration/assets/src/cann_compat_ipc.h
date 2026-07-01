/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */



#ifndef CUDA_COMPAT_IPC_H
#define CUDA_COMPAT_IPC_H

#include "cann_compat_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* =================================================================
     * IPC Memory API Functions
     * ================================================================= */


    cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr);


    cudaError_t cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags);


    cudaError_t cudaIpcCloseMemHandle(void *devPtr);

    /* =================================================================
     * IPC Event Handle Management
     * ================================================================= */


    static inline cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle,
                                                    cudaEvent_t event)
    {
        aclError ret = aclrtIpcGetEventHandle(event, handle);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event,
                                                     cudaIpcEventHandle_t handle)
    {
        aclError ret = aclrtIpcOpenEventHandle(handle, event);
        return acl2cudaError(ret);
    }

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_IPC_H */
