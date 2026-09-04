/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */

#ifndef CUDA_COMPAT_SYMBOL_H
#define CUDA_COMPAT_SYMBOL_H

#include "cann_compat_types.h"
#include <stdint.h>
#include <stdlib.h>

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

static inline cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol,
                                               size_t count, size_t offset,
                                               cudaMemcpyKind kind)
{
    if (!dst || !symbol) {
        return cudaErrorInvalidValue;
    }
    aclError ret = aclrtMemcpyFromSymbol(dst, count, symbol, count, offset,
                                         (aclrtMemcpyKind)kind);
    return acl2cudaError(ret);
}

cudaError_t cudaCompatRegisterSymbol(void *binHandle, const void *hostVar,
                                     const char *deviceVarName, size_t size,
                                     unsigned int flags);

#ifdef __cplusplus
}

struct cudaCompatSymbolRecord {
    const void *symbol;
    void *devicePtr;
    size_t size;
};

static inline cudaCompatSymbolRecord *cudaCompatSymbolRecords()
{
    static cudaCompatSymbolRecord records[64];
    return records;
}

static inline cudaCompatSymbolRecord *cudaCompatFindSymbolRecord(const void *hostSymbol, int create)
{
    cudaCompatSymbolRecord *records = cudaCompatSymbolRecords();
    cudaCompatSymbolRecord *freeSlot = nullptr;
    for (size_t i = 0; i < 64; ++i) {
        if (records[i].symbol == hostSymbol) {
            return &records[i];
        }
        if (!records[i].symbol && !freeSlot) {
            freeSlot = &records[i];
        }
    }
    return create ? freeSlot : nullptr;
}

template <typename T>
static inline cudaError_t cudaGetSymbolAddress(void **devPtr, const T &symbol)
{
    if (!devPtr) {
        return cudaErrorInvalidValue;
    }
    const void *hostSymbol = reinterpret_cast<const void *>(&symbol);
    cudaCompatSymbolRecord *record = cudaCompatFindSymbolRecord(hostSymbol, 0);
    if (record && record->devicePtr) {
        *devPtr = record->devicePtr;
        return cudaSuccess;
    }
    return cudaGetSymbolAddress(devPtr, hostSymbol);
}

template <typename T>
static inline cudaError_t cudaMemcpyToSymbol(const T &symbol, const void *src,
                                             size_t count, size_t offset = 0,
                                             cudaMemcpyKind kind = cudaMemcpyHostToDevice)
{
    if (!src || kind != cudaMemcpyHostToDevice) {
        return cudaMemcpyToSymbol(reinterpret_cast<const void *>(&symbol), src, count, offset, kind);
    }
    const void *hostSymbol = reinterpret_cast<const void *>(&symbol);
    cudaCompatSymbolRecord *slot = cudaCompatFindSymbolRecord(hostSymbol, 1);
    if (!slot) {
        return cudaErrorMemoryAllocation;
    }
    size_t needed = offset + count;
    if (!slot->devicePtr || slot->size < needed) {
        if (slot->devicePtr) {
            cudaError_t freeRet = cudaFree(slot->devicePtr);
            if (freeRet != cudaSuccess) {
                return freeRet;
            }
            slot->devicePtr = nullptr;
            slot->size = 0;
        }
        cudaError_t allocRet = cudaMalloc(&slot->devicePtr, needed);
        if (allocRet != cudaSuccess) {
            return allocRet;
        }
        slot->symbol = hostSymbol;
        slot->size = needed;
    }
    return cudaMemcpy(static_cast<uint8_t *>(slot->devicePtr) + offset, src, count, kind);
}

template <typename T>
static inline cudaError_t cudaMemcpyFromSymbol(void *dst, const T &symbol,
                                               size_t count, size_t offset = 0,
                                               cudaMemcpyKind kind = cudaMemcpyDeviceToHost)
{
    if (!dst || kind != cudaMemcpyDeviceToHost) {
        return cudaMemcpyFromSymbol(dst, reinterpret_cast<const void *>(&symbol), count, offset, kind);
    }
    void *devicePtr = nullptr;
    cudaError_t addrRet = cudaGetSymbolAddress(&devicePtr, symbol);
    if (addrRet != cudaSuccess) {
        return addrRet;
    }
    return cudaMemcpy(dst, static_cast<uint8_t *>(devicePtr) + offset, count, kind);
}
#endif

#endif /* CUDA_COMPAT_SYMBOL_H */
