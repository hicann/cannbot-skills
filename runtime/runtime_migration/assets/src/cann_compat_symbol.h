/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */

#ifndef CUDA_COMPAT_SYMBOL_H
#define CUDA_COMPAT_SYMBOL_H

#include "cann_compat_types.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol)
{
    if (!devPtr || !symbol) {
        return cudaErrorInvalidValue;
    }
    return acl2cudaError(aclrtGetSymbolAddress(symbol, devPtr));
}


static inline cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src,
                                             size_t count, size_t offset,
                                             cudaMemcpyKind kind)
{
    if (!symbol || !src) {
        return cudaErrorInvalidValue;
    }
    aclError ret = aclrtMemcpyToSymbol(symbol, src, count, offset, (aclrtMemcpyKind)kind);
    return acl2cudaError(ret);
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_SYMBOL_H */
