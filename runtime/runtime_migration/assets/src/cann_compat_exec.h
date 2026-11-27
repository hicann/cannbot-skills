/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */



#ifndef CUDA_COMPAT_EXEC_H
#define CUDA_COMPAT_EXEC_H

#include "cann_compat_safe.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef void (*cudaHostFn_t)(void *userData);
    typedef cudaError_t (*cudaCompatHostKernelFn_t)(void *userData);

    __attribute__((weak)) aclError aclrtGetFuncBySymbol(
        const void *symbol, aclrtFuncHandle *funcHandle);


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
        return cudaErrorNotSupported;
    }

    static inline int cudaCompatLikelyInvalidFunctionHandle(const void *func)
    {
        /*
         * CANN function query/launch APIs require an aclrtFuncHandle obtained
         * from a real binary/module loading flow.  Very small non-null pointer
         * values are commonly used by smoke tests as fake handles; passing them
         * through to CANN may crash before an error code can be returned.
         */
        return ((uintptr_t)func) < 4096U;
    }


    static inline cudaError_t cudaGetFuncBySymbol(cudaFunction_t *functionPtr,
                                                  const void *symbolPtr)
    {
        if (!functionPtr || !symbolPtr) {
            return cudaErrorInvalidValue;
        }
        *functionPtr = NULL;
        if (cudaCompatLikelyInvalidFunctionHandle(symbolPtr)) {
            return cudaErrorInvalidDeviceFunction;
        }
        if (aclrtGetFuncBySymbol == NULL) {
            return cudaErrorNotSupported;
        }
        aclError ret = aclrtGetFuncBySymbol(symbolPtr, functionPtr);
        return acl2cudaError(ret);
    }


    static inline aclrtFuncHandle cudaCompatResolveFunctionHandle(const void *func)
    {
        aclrtFuncHandle handle = NULL;
        if (aclrtGetFuncBySymbol != NULL &&
            aclrtGetFuncBySymbol(func, &handle) == ACL_SUCCESS && handle != NULL) {
            return handle;
        }
        return (aclrtFuncHandle)func;
    }


    static inline cudaError_t cudaLaunchHostFunc(cudaStream_t stream,
                                                 cudaHostFn_t fn,
                                                 void *userData)
    {
        if (!fn) {
            return cudaSuccess;
        }
        aclError ret = aclrtLaunchHostFunc(stream, (aclrtHostFunc)fn, userData);
        return acl2cudaError(ret);
    }

    static inline cudaError_t cudaLaunchHostFunc_v2(cudaStream_t stream,
                                                    cudaHostFn_t fn,
                                                    void *userData,
                                                    unsigned int syncMode)
    {
        if (!fn) {
            return cudaSuccess;
        }
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
        if (cudaCompatLikelyInvalidFunctionHandle(func)) {
            return cudaErrorInvalidDeviceFunction;
        }
#ifdef __cplusplus
        *attr = cudaFuncAttributes();
#else
        *attr = (cudaFuncAttributes){0};
#endif

        /* CANN kernel type/ratio/scheduling do not represent CUDA binary
         * architecture, threads per block, or cache mode. */
        return cudaErrorNotSupported;
    }

    static inline cudaError_t cudaFuncGetName(const char **name, const void *func)
    {
        static char funcName[256];
        if (!name || !func) {
            return cudaErrorInvalidValue;
        }
        if (cudaCompatLikelyInvalidFunctionHandle(func)) {
            return cudaErrorInvalidDeviceFunction;
        }
        aclrtFuncHandle handle = cudaCompatResolveFunctionHandle(func);
        aclError ret = aclrtGetFunctionName(handle, (uint32_t)sizeof(funcName), funcName);
        if (ret != ACL_SUCCESS) {
            return acl2cudaError(ret);
        }
        *name = funcName;
        return cudaSuccess;
    }

    static inline cudaError_t cudaFuncGetParamCount(const void *func, size_t *paramCount)
    {
        if (!func || !paramCount) {
            return cudaErrorInvalidValue;
        }
        if (cudaCompatLikelyInvalidFunctionHandle(func)) {
            return cudaErrorInvalidDeviceFunction;
        }
        aclError ret = aclrtFunctionGetParamCount(func, paramCount);
        return acl2cudaError(ret);
    }

    static inline cudaError_t cudaFuncGetParamInfo(const void *func,
                                                   size_t paramIndex,
                                                   size_t *paramOffset,
                                                   size_t *paramSize)
    {
        if (!func || (!paramOffset && !paramSize)) {
            return cudaErrorInvalidValue;
        }
        if (cudaCompatLikelyInvalidFunctionHandle(func)) {
            return cudaErrorInvalidDeviceFunction;
        }
        aclError ret = aclrtFunctionGetParamInfo(func, paramIndex, paramOffset, paramSize);
        return acl2cudaError(ret);
    }

    static inline cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize,
                                                                        const void *func,
                                                                        int numBlocks,
                                                                        int blockSize)
    {
        if (!dynamicSmemSize || !func || numBlocks <= 0 || blockSize <= 0) {
            return cudaErrorInvalidValue;
        }
        if (cudaCompatLikelyInvalidFunctionHandle(func)) {
            return cudaErrorInvalidDeviceFunction;
        }
        aclError ret = aclrtFunctionGetAvailDynUbufPerBlock((void *)func, 0, dynamicSmemSize);
        return acl2cudaError(ret);
    }


    static inline uint32_t cudaCompatGridBlocks(dim3 gridDim)
    {
        uint64_t blocks = (uint64_t)gridDim.x * (uint64_t)gridDim.y * (uint64_t)gridDim.z;
        if (blocks == 0 || blocks > UINT32_MAX) {
            return 0;
        }
        return (uint32_t)blocks;
    }

    static inline int cudaCompatBlockDimValid(dim3 blockDim)
    {
        return blockDim.x != 0 && blockDim.y != 0 && blockDim.z != 0;
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
        if (numBlocks == 0 || !cudaCompatBlockDimValid(blockDim)) {
            return cudaErrorInvalidConfiguration;
        }
        if ((blockDim.x > 1 || blockDim.y > 1 || blockDim.z > 1) && aclrtLaunchSIMTKernelWithArgsArray) {
            aclError ret = aclrtLaunchSIMTKernelWithArgsArray((void *)func, gridDim, blockDim, sharedMem, stream, NULL, args);
            return acl2cudaError(ret);
        }
        aclError ret = aclrtLaunchKernelWithArgsArray((void *)func, numBlocks, stream, NULL, args);
        return acl2cudaError(ret);
    }

    static inline cudaError_t cudaCompatKernelHostArgsSize(const void *func,
                                                           void **args,
                                                           size_t paramCount,
                                                           size_t *totalSize)
    {
        if (!args) {
            return cudaErrorInvalidValue;
        }
        *totalSize = 0;
        for (size_t i = 0; i < paramCount; ++i) {
            size_t offset = 0;
            size_t size = 0;
            aclError ret = aclrtFunctionGetParamInfo(func, i, &offset, &size);
            if (ret != ACL_SUCCESS) {
                return acl2cudaError(ret);
            }
            if (!args[i]) {
                return cudaErrorInvalidValue;
            }
            size_t end = offset + size;
            if (end < offset) {
                return cudaErrorInvalidValue;
            }
            if (end > *totalSize) {
                *totalSize = end;
            }
        }
        return cudaSuccess;
    }

    static inline cudaError_t cudaCompatCopyKernelHostArgs(const void *func,
                                                           void **args,
                                                           size_t paramCount,
                                                           void *packed,
                                                           size_t totalSize)
    {
        for (size_t i = 0; i < paramCount; ++i) {
            size_t offset = 0;
            size_t size = 0;
            aclError ret = aclrtFunctionGetParamInfo(func, i, &offset, &size);
            if (ret != ACL_SUCCESS) {
                return acl2cudaError(ret);
            }
            if (offset > totalSize || size > totalSize - offset) {
                return cudaErrorInvalidValue;
            }
            cudaError_t copyRet = cudaCompatMemcpyChecked((char *)packed + offset,
                                                          totalSize - offset,
                                                          args[i], size);
            if (copyRet != cudaSuccess) {
                return copyRet;
            }
        }
        return cudaSuccess;
    }

    static inline cudaError_t cudaCompatPackKernelHostArgs(const void *func,
                                                           void **args,
                                                           void **hostArgs,
                                                           size_t *argsSize)
    {
        size_t paramCount = 0;
        aclError ret = aclrtFunctionGetParamCount(func, &paramCount);
        if (ret != ACL_SUCCESS) {
            return acl2cudaError(ret);
        }
        if (paramCount == 0) {
            *hostArgs = NULL;
            *argsSize = 0;
            return cudaSuccess;
        }

        size_t totalSize = 0;
        cudaError_t sizeRet = cudaCompatKernelHostArgsSize(func, args, paramCount, &totalSize);
        if (sizeRet != cudaSuccess) {
            return sizeRet;
        }
        void *packed = calloc(1, totalSize);
        if (!packed) {
            return cudaErrorMemoryAllocation;
        }
        cudaError_t copyRet = cudaCompatCopyKernelHostArgs(func, args, paramCount, packed, totalSize);
        if (copyRet != cudaSuccess) {
            free(packed);
            return copyRet;
        }
        *hostArgs = packed;
        *argsSize = totalSize;
        return cudaSuccess;
    }

    static inline cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t *config,
                                                 const void *func,
                                                 void **args)
    {
        if (!config || !func) {
            return cudaErrorInvalidValue;
        }
        uint32_t numBlocks = cudaCompatGridBlocks(config->gridDim);
        if (numBlocks == 0 || !cudaCompatBlockDimValid(config->blockDim)) {
            return cudaErrorInvalidConfiguration;
        }
        if (cudaCompatLikelyInvalidFunctionHandle(func)) {
            return cudaErrorInvalidDeviceFunction;
        }

        void *hostArgs = NULL;
        size_t argsSize = 0;
        cudaError_t packRet = cudaCompatPackKernelHostArgs(func, args, &hostArgs, &argsSize);
        if (packRet != cudaSuccess) {
            return packRet;
        }

        aclrtLaunchKernelCfg cfg;
        cfg.attrs = (aclrtLaunchKernelAttr *)config->attrs;
        cfg.numAttrs = (size_t)config->numAttrs;
        aclError ret = aclrtLaunchKernelWithHostArgs((aclrtFuncHandle)func, numBlocks, config->stream,
                                                     config->numAttrs > 0 ? &cfg : NULL,
                                                     hostArgs, argsSize, NULL, 0);
        free(hostArgs);
        return acl2cudaError(ret);
    }

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_EXEC_H */
