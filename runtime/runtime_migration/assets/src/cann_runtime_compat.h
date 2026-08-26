/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */


/*
 * Project-local CUDA-to-CANN compatibility shim.
 *
 * This header intentionally provides only the declarations and small wrappers
 * needed by migrated samples. Keep this tree free of vendor SDK header text,
 * copied documentation, and installation-package material.
 */

#ifndef CUDA_RUNTIME_COMPAT_H
#define CUDA_RUNTIME_COMPAT_H

#include "cann_compat_types.h"
#include "cann_compat_device.h"
#include "cann_compat_memory.h"
#include "cann_compat_mempool.h"
#include "cann_compat_stream.h"
#include "cann_compat_event.h"
#include "cann_compat_ipc.h"
#include "cann_compat_symbol.h"
#include "cann_compat_library.h"
#include "cann_compat_graph.h"
#include "cann_compat_cu_types.h"
#include "cann_compat_context.h"
#include "cann_compat_cu_vmm.h"
#include "cann_compat_exec.h"
#include "cann_compat_unsupported.h"

/* =================================================================
 * Convenience Macros for Common Operations
 * ================================================================= */

/* CUDA_CHECK macro for error handling */
#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

/* CUDA_CHECK_LAST for checking errors after kernel launches */
#define CUDA_CHECK_LAST()                                         \
    do                                                            \
    {                                                             \
        cudaError_t err = cudaGetLastError();                     \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

/* =================================================================
 * Additional CUDA Runtime APIs
 * ================================================================= */

#ifdef __cplusplus
extern "C"
{
#endif

/* =================================================================
 * Version Management
 * ================================================================= */


    static inline cudaError_t cudaRuntimeGetVersion(int *runtimeVersion)
    {
        if (!runtimeVersion)
        {
            return cudaErrorInvalidValue;
        }
        char pkgName[] = "runtime";
        int32_t versionNum = 0;
        aclError ret = aclsysGetVersionNum(pkgName, &versionNum);
        if (ret != ACL_SUCCESS) {
            return acl2cudaError(ret);
        }
        *runtimeVersion = versionNum;
        return cudaSuccess;
    }


    static inline cudaError_t cudaDriverGetVersion(int *driverVersion)
    {
        if (!driverVersion)
        {
            return cudaErrorInvalidValue;
        }
        char pkgName[] = "runtime";
        int32_t versionNum = 0;
        aclError ret = aclsysGetVersionNum(pkgName, &versionNum);
        if (ret != ACL_SUCCESS) {
            return acl2cudaError(ret);
        }
        *driverVersion = versionNum;
        return cudaSuccess;
    }

    /* =================================================================
     * Error Handling
     * ================================================================= */


    static inline cudaError_t cudaGetLastError(void)
    {
        // Get last error from CANN
        aclError err = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
        return acl2cudaError(err);
    }


    static inline cudaError_t cudaPeekAtLastError(void)
    {
        aclError err = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
        return acl2cudaError(err);
    }


    const char *cudaGetErrorName(cudaError_t error);


    const char *cudaGetErrorString(cudaError_t error);

    /* =================================================================
     * Profiler Management
     * ================================================================= */


    cudaError_t cudaProfilerStart(void);


    cudaError_t cudaProfilerStop(void);

    /* =================================================================
     * Memory Prefetch (for managed memory)
     * ================================================================= */


    static inline cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count,
                                                   cudaMemLocation location, unsigned int flags,
                                                   cudaStream_t stream)
    {
        // CANN handles data migration automatically
        // For now, this is a no-op
        (void)(devPtr);
        (void)(count);
        (void)(location);
        (void)(flags);
        (void)(stream);
        return cudaErrorNotSupported;
    }

    /* =================================================================
     * Advanced Features (Simplified)
     * ================================================================= */


    static inline cudaError_t cudaChooseDevice(int *device, const cudaDeviceProp *prop)
    {
        // Return first available device
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess || count == 0)
        {
            return cudaErrorNoDevice;
        }
        *device = 0;
        return cudaSuccess;
    }

#ifdef __cplusplus
}
#endif

/* =================================================================
 * C++ Specific Extensions
 * ==================================================================

#ifdef __cplusplus

namespace cuda {

// C++ exception class for CUDA errors
class error : public std::runtime_error {
public:
    explicit error(cudaError_t e) :
        std::runtime_error(cudaGetErrorString(e)),
        status_(e) {}

    cudaError_t status() const noexcept { return status_; }

private:
    cudaError_t status_;
};

// Check and throw on error
inline void check(cudaError_t e) {
    if (e != CUDASuccess) {
        throw error(e);
    }
}

} // namespace cuda

#endif /* __cplusplus */

#endif /* CUDA_RUNTIME_COMPAT_H */
