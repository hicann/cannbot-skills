/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */



#include "cann_compat_ipc.h"
#include <pthread.h>
#include <stdint.h>

/* Hash table configuration */
#define CUDA_IPC_HASH_BUCKETS 256
#define CUDA_IPC_ENTRIES_PER_BUCKET 8


typedef struct cudaIpcImportEntry
{
    void *devPtr;                       // Device pointer (key for lookup)
    char key[CANN_IPC_MEM_HANDLE_SIZE]; // Export key from CANN
    uint32_t hash;                      // Cached hash value for validation
    volatile int used;                   // Whether entry is in use (atomic)
} cudaIpcImportEntry_t;

/* Hash shard with its own lock for parallel access */
typedef struct cudaIpcHashShard
{
    cudaIpcImportEntry_t entries[CUDA_IPC_ENTRIES_PER_BUCKET];
    pthread_rwlock_t lock;              // Read-write lock per shard
} cudaIpcHashShard_t;

/* IPC import tracking table with sharding */
static cudaIpcHashShard_t g_cudaIpcImportTable[CUDA_IPC_HASH_BUCKETS];

/* Initialize hash table (called once) */
static pthread_once_t g_ipcTableInit = PTHREAD_ONCE_INIT;


static void cudaIpcCopyKey(char *dst, size_t dstSize, const char *src)
{
    if (dst == NULL || dstSize == 0)
    {
        return;
    }

    size_t i = 0;
    if (src != NULL)
    {
        while (i + 1 < dstSize && src[i] != '\0')
        {
            dst[i] = src[i];
            i++;
        }
    }
    dst[i] = '\0';
}


static void cudaIpcClearKey(char *dst, size_t dstSize)
{
    if (dst == NULL)
    {
        return;
    }

    for (size_t i = 0; i < dstSize; i++)
    {
        dst[i] = '\0';
    }
}


static void cudaIpcInitHashTable(void)
{
    for (int i = 0; i < CUDA_IPC_HASH_BUCKETS; i++)
    {
        pthread_rwlock_init(&g_cudaIpcImportTable[i].lock, NULL);
        for (int j = 0; j < CUDA_IPC_ENTRIES_PER_BUCKET; j++)
        {
            g_cudaIpcImportTable[i].entries[j].used = 0;
            g_cudaIpcImportTable[i].entries[j].devPtr = NULL;
            g_cudaIpcImportTable[i].entries[j].hash = 0;
            cudaIpcClearKey(g_cudaIpcImportTable[i].entries[j].key,
                            sizeof(g_cudaIpcImportTable[i].entries[j].key));
        }
    }
}


static inline uint32_t cudaIpcHashPtr(void *ptr)
{
    uintptr_t addr = (uintptr_t)ptr;
    /* Golden ratio hash for good distribution */
    return (uint32_t)(addr * 2654435761u) % CUDA_IPC_HASH_BUCKETS;
}


static cudaIpcHashShard_t *cudaIpcGetShard(void *devPtr, uint32_t *bucket)
{
    pthread_once(&g_ipcTableInit, cudaIpcInitHashTable);

    uint32_t localBucket = cudaIpcHashPtr(devPtr);
    if (bucket != NULL)
    {
        *bucket = localBucket;
    }
    return &g_cudaIpcImportTable[localBucket];
}


cudaIpcImportEntry_t *cudaIpcFindImportEntry(void *devPtr)
{
    if (devPtr == NULL)
    {
        return NULL;
    }

    cudaIpcHashShard_t *shard = cudaIpcGetShard(devPtr, NULL);

    /* Read lock for concurrent access */
    pthread_rwlock_rdlock(&shard->lock);

    cudaIpcImportEntry_t *result = NULL;
    for (int i = 0; i < CUDA_IPC_ENTRIES_PER_BUCKET; i++)
    {
        cudaIpcImportEntry_t *entry = &shard->entries[i];
        /* Use volatile read for used flag */
        if (__atomic_load_n(&entry->used, __ATOMIC_ACQUIRE) &&
            entry->devPtr == devPtr)
        {
            result = entry;
            break;
        }
    }

    pthread_rwlock_unlock(&shard->lock);
    return result;
}


cudaError_t cudaIpcAddImportEntry(void *devPtr, const char *key)
{
    if (devPtr == NULL || key == NULL)
    {
        return cudaErrorInvalidValue;
    }

    uint32_t bucket = 0;
    cudaIpcHashShard_t *shard = cudaIpcGetShard(devPtr, &bucket);

    /* Write lock for modification */
    pthread_rwlock_wrlock(&shard->lock);

    /* First check if entry already exists */
    for (int i = 0; i < CUDA_IPC_ENTRIES_PER_BUCKET; i++)
    {
        cudaIpcImportEntry_t *entry = &shard->entries[i];
        if (entry->used && entry->devPtr == devPtr)
        {
            /* Update existing entry */
            cudaIpcCopyKey(entry->key, sizeof(entry->key), key);
            pthread_rwlock_unlock(&shard->lock);
            return cudaSuccess;
        }
    }

    /* Find free slot for new entry */
    for (int i = 0; i < CUDA_IPC_ENTRIES_PER_BUCKET; i++)
    {
        cudaIpcImportEntry_t *entry = &shard->entries[i];
        if (!entry->used)
        {
            entry->devPtr = devPtr;
            cudaIpcCopyKey(entry->key, sizeof(entry->key), key);
            entry->hash = bucket;
            /* Release store for used flag */
            __atomic_store_n(&entry->used, 1, __ATOMIC_RELEASE);
            pthread_rwlock_unlock(&shard->lock);
            return cudaSuccess;
        }
    }

    pthread_rwlock_unlock(&shard->lock);
    return cudaErrorUnknown; /* Shard full */
}


void cudaIpcRemoveImportEntry(void *devPtr)
{
    if (devPtr == NULL)
    {
        return;
    }

    cudaIpcHashShard_t *shard = cudaIpcGetShard(devPtr, NULL);

    /* Write lock for modification */
    pthread_rwlock_wrlock(&shard->lock);

    for (int i = 0; i < CUDA_IPC_ENTRIES_PER_BUCKET; i++)
    {
        cudaIpcImportEntry_t *entry = &shard->entries[i];
        if (entry->used && entry->devPtr == devPtr)
        {
            /* Clear entry */
            entry->used = 0;
            entry->devPtr = NULL;
            entry->hash = 0;
            cudaIpcClearKey(entry->key, sizeof(entry->key));
            break;
        }
    }

    pthread_rwlock_unlock(&shard->lock);
}

cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr)
{
    if (!handle || !devPtr)
    {
        return cudaErrorInvalidValue;
    }

    // Initialize handle
    *handle = (cudaIpcMemHandle_t){0};

    // Get memory size
    void *pbase = NULL;
    aclError ret;
    ret = aclrtMemGetAddressRange(devPtr, &pbase, &handle->size);
    if (ret != ACL_SUCCESS)
    {
        return acl2cudaError(ret);
    }
    // Get export key from CANN
    // CANN API: aclrtIpcMemGetExportKey(devPtr, size, key, len, flags)
    ret = aclrtIpcMemGetExportKey(devPtr,
                                  handle->size,
                                  handle->internal,
                                  CANN_IPC_MEM_HANDLE_SIZE,
                                  ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION); // flags
    if (ret != ACL_SUCCESS)
    {
        return acl2cudaError(ret);
    }

    return cudaSuccess;
}

cudaError_t cudaIpcOpenMemHandle(void **devPtr,
                                 cudaIpcMemHandle_t handle,
                                 unsigned int flags)
{
    if (!devPtr)
    {
        return cudaErrorInvalidValue;
    }

    // Import memory using export key from handle
    // CANN API: aclrtIpcMemImportByKey(&devPtr, key, flags)
    aclError ret = aclrtIpcMemImportByKey(devPtr, handle.internal, ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS);
    if (ret != ACL_SUCCESS)
    {
        return acl2cudaError(ret);
    }

    // Track the imported pointer-key mapping for cudaIpcCloseMemHandle
    ret = cudaIpcAddImportEntry(*devPtr, handle.internal);
    if (ret != cudaSuccess)
    {
        // Failed to track, but import succeeded - clean up
        aclrtIpcMemClose(handle.internal);
        *devPtr = NULL;
        return ret;
    }

    // Note: flags are currently ignored as CANN handles peer access differently
    // The cudaIpcMemLazyEnablePeerAccess flag is accepted for compatibility
    (void)flags; // Suppress unused parameter warning

    return cudaSuccess;
}

cudaError_t cudaIpcCloseMemHandle(void *devPtr)
{
    if (!devPtr)
    {
        return cudaErrorInvalidValue;
    }

    // Find the key associated with this device pointer
    cudaIpcImportEntry_t *entry = cudaIpcFindImportEntry(devPtr);
    if (entry == NULL)
    {
        // Not found - might be local memory or already closed
        return cudaErrorInvalidValue;
    }

    // Close the IPC memory using the key string
    // CANN API: aclrtIpcMemClose(key)
    aclError ret = aclrtIpcMemClose(entry->key);

    // Remove from tracking table
    cudaIpcRemoveImportEntry(devPtr);

    return acl2cudaError(ret);
}
