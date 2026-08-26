/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */

#ifndef CUDA_COMPAT_UNSUPPORTED_H
#define CUDA_COMPAT_UNSUPPORTED_H

#include "cann_compat_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    static inline cudaError_t cudaFuncSetAttribute(const void *func,
                                                   cudaFuncAttribute attr,
                                                   int value)
    {
        (void)func;
        (void)attr;
        (void)value;
        return cudaErrorNotSupported;
    }


    static inline cudaError_t cudaLaunchCooperativeKernel(const void *func,
                                                          dim3 gridDim,
                                                          dim3 blockDim,
                                                          void **args,
                                                          size_t sharedMem,
                                                          cudaStream_t stream)
    {
        (void)func;
        (void)gridDim;
        (void)blockDim;
        (void)args;
        (void)sharedMem;
        (void)stream;
        return cudaErrorNotSupported;
    }


    static inline cudaError_t cudaGetDriverEntryPoint(const char *symbol,
                                                      void **funcPtr,
                                                      unsigned long long flags)
    {
        (void)symbol;
        (void)flags;
        if (!funcPtr) {
            return cudaErrorInvalidValue;
        }
        *funcPtr = NULL;
        return cudaErrorNotSupported;
    }


    static inline cudaError_t cudaGetDriverEntryPointByVersion(const char *symbol,
                                                               void **funcPtr,
                                                               unsigned long long flags,
                                                               unsigned int cudaVersion)
    {
        (void)symbol;
        (void)flags;
        (void)cudaVersion;
        if (!funcPtr) {
            return cudaErrorInvalidValue;
        }
        *funcPtr = NULL;
        return cudaErrorNotSupported;
    }


    static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize)
    {
        (void)func;
        (void)blockSize;
        (void)dynamicSMemSize;
        if (!numBlocks) {
            return cudaErrorInvalidValue;
        }
        *numBlocks = 0;
        return cudaErrorNotSupported;
    }


    static inline cudaError_t cudaOccupancyMaxPotentialBlockSize(
        int *minGridSize, int *blockSize, const void *func,
        size_t dynamicSMemSize, int blockSizeLimit)
    {
        (void)func;
        (void)dynamicSMemSize;
        (void)blockSizeLimit;
        if (!minGridSize || !blockSize) {
            return cudaErrorInvalidValue;
        }
        *minGridSize = 0;
        *blockSize = 0;
        return cudaErrorNotSupported;
    }


    static inline cudaError_t cudaStreamGetCaptureInfo_v2(
        cudaStream_t stream, cudaStreamCaptureStatus *captureStatus,
        unsigned long long *id, cudaGraph_t *graph,
        const cudaGraphNode_t **dependencies, size_t *numDependencies)
    {
        (void)stream;
        if (captureStatus) {
            *captureStatus = cudaStreamCaptureStatusNone;
        }
        if (id) {
            *id = 0;
        }
        if (graph) {
            *graph = NULL;
        }
        if (dependencies) {
            *dependencies = NULL;
        }
        if (numDependencies) {
            *numDependencies = 0;
        }
        return cudaErrorNotSupported;
    }


    static inline cudaError_t cudaStreamUpdateCaptureDependencies(
        cudaStream_t stream, cudaGraphNode_t *dependencies,
        size_t numDependencies, unsigned int flags)
    {
        (void)stream;
        (void)dependencies;
        (void)numDependencies;
        (void)flags;
        return cudaErrorNotSupported;
    }


    static inline cudaError_t cudaStreamUpdateCaptureDependencies_v2(
        cudaStream_t stream, cudaGraphNode_t *dependencies,
        const cudaGraphEdgeData *dependencyData,
        size_t numDependencies, unsigned int flags)
    {
        (void)stream;
        (void)dependencies;
        (void)dependencyData;
        (void)numDependencies;
        (void)flags;
        return cudaErrorNotSupported;
    }

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_UNSUPPORTED_H */
