/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */

#ifndef CUDA_COMPAT_SAFE_H
#define CUDA_COMPAT_SAFE_H

#include "cann_compat_types.h"
#include <string.h>

#if defined(__has_include)
#if __has_include(<securec.h>)
#include <securec.h>
#define CUDA_COMPAT_HAS_SECUREC 1
#endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    static inline cudaError_t cudaCompatMemcpyChecked(void *dst, size_t dstSize,
                                                      const void *src, size_t count)
    {
        if (!dst || !src || count > dstSize) {
            return cudaErrorInvalidValue;
        }
        if (count == 0) {
            return cudaSuccess;
        }
#ifdef CUDA_COMPAT_HAS_SECUREC
        if (memcpy_s(dst, dstSize, src, count) != 0) {
            return cudaErrorInvalidValue;
        }
#else
        unsigned char *dstBytes = (unsigned char *)dst;
        const unsigned char *srcBytes = (const unsigned char *)src;
        for (size_t i = 0; i < count; ++i) {
            dstBytes[i] = srcBytes[i];
        }
#endif
        return cudaSuccess;
    }

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_SAFE_H */
