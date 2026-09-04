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

    static inline cudaError_t cudaCompatFuncGetHostFallbackAttributes(cudaFuncAttributes *attr)
    {
        if (!attr) {
            return cudaErrorInvalidValue;
        }
#ifdef __cplusplus
        *attr = cudaFuncAttributes();
#else
        *attr = (cudaFuncAttributes){0};
#endif
        return cudaSuccess;
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

    static inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *attr,
                                                    const void *func)
    {
        if (!attr || !func) {
            return cudaErrorInvalidValue;
        }
#ifdef __cplusplus
        *attr = cudaFuncAttributes();
#else
        *attr = (cudaFuncAttributes){0};
#endif

        int64_t attrValue = 0;
        aclError ret = aclrtGetFunctionAttribute((aclrtFuncHandle)func,
                                                 ACL_FUNC_ATTR_KERNEL_TYPE,
                                                 &attrValue);
        if (ret != ACL_SUCCESS) {
            return acl2cudaError(ret);
        }
        attr->binaryVersion = (int)attrValue;

        ret = aclrtGetFunctionAttribute((aclrtFuncHandle)func,
                                        ACL_FUNC_ATTR_KERNEL_RATIO,
                                        &attrValue);
        if (ret == ACL_SUCCESS) {
            attr->maxThreadsPerBlock = (int)attrValue;
        }

        ret = aclrtGetFunctionAttribute((aclrtFuncHandle)func,
                                        ACL_FUNC_ATTR_KERNEL_SCHED_MODE,
                                        &attrValue);
        if (ret == ACL_SUCCESS) {
            attr->cacheModeCA = (int)attrValue;
        }
        return cudaSuccess;
    }


    static inline uint32_t cudaCompatGridBlocks(dim3 gridDim)
    {
        uint64_t blocks = (uint64_t)gridDim.x * (uint64_t)gridDim.y * (uint64_t)gridDim.z;
        if (blocks == 0 || blocks > UINT32_MAX) {
            return 0;
        }
        return (uint32_t)blocks;
    }

    __attribute__((weak)) aclError aclrtLaunchSIMTKernelWithArgsArray(
        void *func, dim3 gridDim, dim3 blockDim, size_t dynUbufSize,
        aclrtStream stream, aclrtLaunchKernelCfg *cfg, void **args);

    static inline cudaError_t cudaLaunchKernel(const void *func,
                                               dim3 gridDim,
                                               dim3 blockDim,
                                               void **args,
                                               size_t sharedMem,
                                               cudaStream_t stream)
    {
        if (!func) {
            return cudaErrorInvalidDeviceFunction;
        }
        uint32_t numBlocks = cudaCompatGridBlocks(gridDim);
        if (numBlocks == 0) {
            return cudaErrorInvalidConfiguration;
        }
        if ((blockDim.x > 1 || blockDim.y > 1 || blockDim.z > 1) && aclrtLaunchSIMTKernelWithArgsArray) {
            aclError ret = aclrtLaunchSIMTKernelWithArgsArray((void *)func, gridDim, blockDim, sharedMem, stream, NULL, args);
            return acl2cudaError(ret);
        }
        aclError ret = aclrtLaunchKernelWithArgsArray((void *)func, numBlocks, stream, NULL, args);
        return acl2cudaError(ret);
    }

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_EXEC_H */
