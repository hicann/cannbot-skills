/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */



#ifndef CUDA_COMPAT_LIBRARY_H
#define CUDA_COMPAT_LIBRARY_H

#include "cann_compat_types.h"
#include "acl/acl_rt.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* =================================================================
     * Weak Symbol Declaration for Version Compatibility
     * ================================================================= */


    __attribute__((weak)) aclError aclrtBinaryGetGlobal(aclrtBinHandle binHandle,
                                                        const char *name,
                                                        void **dptr,
                                                        size_t *bytes);

    /* =================================================================
     * Type Definitions
     * ================================================================= */


    typedef aclrtBinHandle cudaLibrary_t;


    typedef int cudaJitOption;


    typedef int cudaLibraryOption;

    static inline void cudaLibraryIgnoreOptions(cudaJitOption *jitOptions,
                                                void **jitOptionsValues,
                                                unsigned int numJitOptions,
                                                cudaLibraryOption *libraryOptions,
                                                void **libraryOptionValues,
                                                unsigned int numLibraryOptions)
    {
        (void)jitOptions;
        (void)jitOptionsValues;
        (void)numJitOptions;
        (void)libraryOptions;
        (void)libraryOptionValues;
        (void)numLibraryOptions;
    }

    /* =================================================================
     * Library Management
     * ================================================================= */


    cudaError_t cudaLibraryLoadData(cudaLibrary_t *library,
                                                  const void *code,
                                                  cudaJitOption *jitOptions,
                                                  void **jitOptionsValues,
                                                  unsigned int numJitOptions,
                                                  cudaLibraryOption *libraryOptions,
                                                  void **libraryOptionValues,
                                                  unsigned int numLibraryOptions);


    static inline cudaError_t cudaLibraryLoadFromFile(cudaLibrary_t *library,
                                                      const char *filename,
                                                      cudaJitOption *jitOptions,
                                                      void **jitOptionsValues,
                                                      unsigned int numJitOptions,
                                                      cudaLibraryOption *libraryOptions,
                                                      void **libraryOptionValues,
                                                      unsigned int numLibraryOptions)
    {
        cudaLibraryIgnoreOptions(jitOptions, jitOptionsValues, numJitOptions,
                                 libraryOptions, libraryOptionValues, numLibraryOptions);

        if (!library || !filename)
        {
            return cudaErrorInvalidValue;
        }

        aclError ret = aclrtBinaryLoadFromFile(filename, NULL, library);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaLibraryUnload(cudaLibrary_t library)
    {
        aclError ret = aclrtBinaryUnLoad(library);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaLibraryGetFunction(void **func,
                                                     cudaLibrary_t library,
                                                     const char *name)
    {
        if (!func || !name)
        {
            return cudaErrorInvalidValue;
        }

        aclError ret = aclrtBinaryGetFunction(library, name, (aclrtFuncHandle *)func);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaLibraryGetGlobal(void **dptr,
                                                    size_t *bytes,
                                                    cudaLibrary_t library,
                                                    const char *name)
    {
        if (aclrtBinaryGetGlobal == NULL)
        {
            return cudaErrorNotSupported;
        }
        if (!name)
        {
            return cudaErrorInvalidValue;
        }
        aclError ret = aclrtBinaryGetGlobal(library, name, dptr, bytes);
        return acl2cudaError(ret);
    }

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_LIBRARY_H */
