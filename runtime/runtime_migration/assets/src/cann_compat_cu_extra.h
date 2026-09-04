/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */

#ifndef CUDA_COMPAT_CU_EXTRA_H
#define CUDA_COMPAT_CU_EXTRA_H

#include "cann_compat_cu_types.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C"
{
#endif

    uint64_t get_elf_file_size(const void *elf_buf);

    typedef struct cudaCompatPtxModule_st {
        char magic[8];
    } cudaCompatPtxModule;

    typedef struct cudaCompatPtxFunction_st {
        char magic[8];
        CUmodule module;
        const char *name;
    } cudaCompatPtxFunction;

    static inline int cudaCompatIsPtxText(const char *image)
    {
        return image != NULL &&
               (strstr(image, ".version") != NULL || strstr(image, ".visible .entry") != NULL);
    }

    static inline int cudaCompatIsPtxModule(CUmodule module)
    {
        cudaCompatPtxModule *ptxModule = (cudaCompatPtxModule *)module;
        return ptxModule != NULL && memcmp(ptxModule->magic, "PTXMOD", 7) == 0;
    }

    static inline void cudaCompatSetMagic(char *dst, const char *src, size_t count)
    {
        for (size_t i = 0; i < count; ++i) {
            dst[i] = src[i];
        }
    }

    static inline CUresult cudaCompatCreatePtxModule(CUmodule *module)
    {
        cudaCompatPtxModule *ptxModule = (cudaCompatPtxModule *)calloc(1, sizeof(cudaCompatPtxModule));
        if (!ptxModule) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        cudaCompatSetMagic(ptxModule->magic, "PTXMOD", 7);
        *module = (CUmodule)ptxModule;
        return CUDA_SUCCESS;
    }

    static inline CUresult cuInit(unsigned int flags)
    {
        (void)flags;
        return CUDA_SUCCESS;
    }

    static inline CUresult cuGetErrorName(CUresult error, const char **pStr)
    {
        if (!pStr) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        switch (error) {
        case CUDA_SUCCESS:
            *pStr = "CUDA_SUCCESS";
            break;
        case CUDA_ERROR_INVALID_VALUE:
            *pStr = "CUDA_ERROR_INVALID_VALUE";
            break;
        case CUDA_ERROR_INVALID_IMAGE:
            *pStr = "CUDA_ERROR_INVALID_IMAGE";
            break;
        case CUDA_ERROR_INVALID_HANDLE:
            *pStr = "CUDA_ERROR_INVALID_HANDLE";
            break;
        case CUDA_ERROR_NOT_SUPPORTED:
            *pStr = "CUDA_ERROR_NOT_SUPPORTED";
            break;
        case CUDA_ERROR_UNKNOWN:
            *pStr = "CUDA_ERROR_UNKNOWN";
            break;
        default:
            *pStr = "CUDA_ERROR_UNKNOWN";
            break;
        }
        return CUDA_SUCCESS;
    }

    static inline CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
    {
        (void)hfunc;
        (void)config;
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    static inline CUresult cuCtxPopCurrent(CUcontext *pctx)
    {
        if (!pctx) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *pctx = NULL;
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    static inline CUresult cuCtxPushCurrent(CUcontext ctx)
    {
        (void)ctx;
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    static inline CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev)
    {
        (void)dev;
        if (!pctx) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *pctx = NULL;
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    static inline CUresult cuGreenCtxCreate(CUgreenCtx *phCtx, CUdevResourceDesc desc,
                                            CUdevice dev, unsigned int flags)
    {
        (void)desc;
        (void)dev;
        (void)flags;
        if (!phCtx) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *phCtx = NULL;
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    static inline CUresult cuGreenCtxDestroy(CUgreenCtx greenCtx)
    {
        (void)greenCtx;
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    static inline CUresult cuCtxFromGreenCtx(CUcontext *pctx, CUgreenCtx greenCtx)
    {
        (void)greenCtx;
        if (!pctx) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *pctx = NULL;
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    static inline CUresult cuDeviceGetDevResource(CUdevice dev, CUdevResource *resource,
                                                  CUdevResourceType type)
    {
        (void)dev;
        (void)type;
        if (!resource) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *resource = NULL;
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    static inline CUresult cuGreenCtxStreamCreate(CUstream *phStream, CUgreenCtx greenCtx,
                                                  unsigned int flags, int priority)
    {
        (void)greenCtx;
        (void)flags;
        (void)priority;
        if (!phStream) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *phStream = NULL;
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    static inline CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                                              unsigned int numOptions, CUjit_option *options,
                                              void **optionValues)
    {
        (void)image;
        (void)numOptions;
        (void)options;
        (void)optionValues;
        if (!module) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *module = NULL;
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    static inline CUresult cuModuleLoad(CUmodule *module, const char *fname)
    {
        if (!module || !fname) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        FILE *fp = fopen(fname, "rb");
        if (fp) {
            char probe[256];
            size_t bytes = fread(probe, 1, sizeof(probe) - 1, fp);
            fclose(fp);
            probe[bytes] = '\0';
            if (cudaCompatIsPtxText(probe)) {
                return cudaCompatCreatePtxModule(module);
            }
        }
        aclError ret = aclrtBinaryLoadFromFile(fname, NULL, (aclrtBinHandle *)module);
        return acl2cuError(ret);
    }

    static inline CUresult cuModuleLoadData(CUmodule *module, const void *image)
    {
        if (!module || !image) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (cudaCompatIsPtxText((const char *)image)) {
            return cudaCompatCreatePtxModule(module);
        }
        uint64_t elfSize = get_elf_file_size(image);
        if (elfSize == 0) {
            return CUDA_ERROR_INVALID_IMAGE;
        }
        aclError ret = aclrtBinaryLoadFromData(image, (size_t)elfSize, NULL,
                                               (aclrtBinHandle *)module);
        return acl2cuError(ret);
    }

    static inline CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                                               const char *name)
    {
        if (!hfunc || !hmod || !name) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (cudaCompatIsPtxModule(hmod)) {
            cudaCompatPtxFunction *func = (cudaCompatPtxFunction *)calloc(1, sizeof(cudaCompatPtxFunction));
            if (!func) {
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            cudaCompatSetMagic(func->magic, "PTXFUNC", 8);
            func->module = hmod;
            func->name = name;
            *hfunc = (CUfunction)func;
            return CUDA_SUCCESS;
        }
        aclError ret = aclrtBinaryGetFunction((aclrtBinHandle)hmod, name,
                                              (aclrtFuncHandle *)hfunc);
        return acl2cuError(ret);
    }

    static inline CUresult cuModuleUnload(CUmodule hmod)
    {
        if (!hmod) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (cudaCompatIsPtxModule(hmod)) {
            free(hmod);
            return CUDA_SUCCESS;
        }
        aclError ret = aclrtBinaryUnLoad((aclrtBinHandle)hmod);
        return acl2cuError(ret);
    }

    static inline CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr,
                                                uint32_t value, unsigned int flags)
    {
        if (!addr) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        aclError ret = aclrtValueWrite((void *)addr, (uint64_t)value, flags,
                                       (aclrtStream)stream);
        return acl2cuError(ret);
    }

    static inline CUresult cuLinkAddData(CUlinkState state, CUjitInputType type,
                                         void *data, size_t size, const char *name,
                                         unsigned int numOptions, CUjit_option *options,
                                         void **optionValues)
    {
        (void)state;
        (void)type;
        (void)data;
        (void)size;
        (void)name;
        (void)numOptions;
        (void)options;
        (void)optionValues;
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    static inline CUresult cuMulticastCreate(CUmemGenericAllocationHandle *handle,
                                             const CUmulticastObjectProp *prop)
    {
        (void)prop;
        if (!handle) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *handle = NULL;
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    static inline CUresult cuMulticastAddDevice(CUmemGenericAllocationHandle mcHandle,
                                                CUdevice dev)
    {
        (void)mcHandle;
        (void)dev;
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    static inline CUresult cuMulticastBindMem(CUmemGenericAllocationHandle mcHandle,
                                              size_t mcOffset,
                                              CUmemGenericAllocationHandle memHandle,
                                              size_t memOffset, size_t size,
                                              unsigned long long flags)
    {
        (void)mcHandle;
        (void)mcOffset;
        (void)memHandle;
        (void)memOffset;
        (void)size;
        (void)flags;
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    static inline CUresult cuMulticastUnbind(CUmemGenericAllocationHandle mcHandle,
                                             CUdevice dev, size_t mcOffset,
                                             size_t size)
    {
        (void)mcHandle;
        (void)dev;
        (void)mcOffset;
        (void)size;
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    static inline CUresult cuTensorMapEncodeTiled(
        CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
        cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
        const cuuint64_t *globalStrides, const cuuint32_t *boxDim,
        const cuuint32_t *elementStrides, CUtensorMapInterleave interleave,
        CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
        CUtensorMapFloatOOBfill oobFill)
    {
        (void)tensorDataType;
        (void)tensorRank;
        (void)globalAddress;
        (void)globalDim;
        (void)globalStrides;
        (void)boxDim;
        (void)elementStrides;
        (void)interleave;
        (void)swizzle;
        (void)l2Promotion;
        (void)oobFill;
        if (!tensorMap) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *tensorMap = NULL;
        return CUDA_ERROR_NOT_SUPPORTED;
    }

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_CU_EXTRA_H */
