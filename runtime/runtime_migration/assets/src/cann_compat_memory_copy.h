/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */

#ifndef CUDA_COMPAT_MEMORY_COPY_H
#define CUDA_COMPAT_MEMORY_COPY_H

#include "cann_compat_safe.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* =================================================================
     * Memory Copy Operations
     * ================================================================= */


    static inline cudaError_t cudaMemcpy(void *dst, const void *src,
                                         size_t count, cudaMemcpyKind kind)
    {
        aclError ret = aclrtMemcpy(dst, count, src, count, (aclrtMemcpyKind)kind);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaMemcpyAsync(void *dst, const void *src,
                                              size_t count, cudaMemcpyKind kind,
                                              cudaStream_t stream)
    {
        aclError ret = aclrtMemcpyAsync(dst, count, src, count, (aclrtMemcpyKind)kind, stream);
        return acl2cudaError(ret);
    }


    cudaError_t cudaMemcpyBatchAsync(void *const *dsts, const void *const *srcs,
                                     const size_t *sizes, size_t count,
                                     cudaMemcpyAttributes *attrs, size_t *attrsIdxs, size_t numAttrs, cudaStream_t stream);


    static inline cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
                                           size_t spitch, size_t width, size_t height,
                                           cudaMemcpyKind kind)
    {
        aclError ret = aclrtMemcpy2d(dst, dpitch, src, spitch, width, height, (aclrtMemcpyKind)kind);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                                                size_t spitch, size_t width, size_t height,
                                                cudaMemcpyKind kind, cudaStream_t stream)
    {
        aclError ret = aclrtMemcpy2dAsync(dst, dpitch, src, spitch, width, height, (aclrtMemcpyKind)kind, stream);
        return acl2cudaError(ret);
    }

    static inline cudaError_t cudaCompatValidateMemcpy3DPtr(const cudaMemcpy3DParms *p)
    {
        if (!p || !p->srcPtr.ptr || !p->dstPtr.ptr || p->extent.width == 0 ||
            p->extent.height == 0 || p->extent.depth == 0) {
            return cudaErrorInvalidValue;
        }
        if (p->srcArray || p->dstArray) {
            return cudaErrorNotSupported;
        }
        if (p->srcPtr.pitch < p->extent.width || p->dstPtr.pitch < p->extent.width) {
            return cudaErrorInvalidValue;
        }
        if (p->srcPtr.ysize != 0 && p->srcPos.y + p->extent.height > p->srcPtr.ysize) {
            return cudaErrorInvalidValue;
        }
        if (p->dstPtr.ysize != 0 && p->dstPos.y + p->extent.height > p->dstPtr.ysize) {
            return cudaErrorInvalidValue;
        }
        return cudaSuccess;
    }

    static inline char *cudaCompatSlicePtr(cudaPitchedPtr ptr, cudaPos pos, size_t z)
    {
        return (char *)ptr.ptr + (pos.z + z) * ptr.pitch * ptr.ysize + pos.y * ptr.pitch + pos.x;
    }

    static inline int cudaCompatPtrLikelyHost(const void *ptr)
    {
        aclrtPtrAttributes attr;
        aclError ret = aclrtPointerGetAttributes(ptr, &attr);
        return ret != ACL_SUCCESS ||
               attr.location.type == ACL_MEM_LOCATION_TYPE_HOST ||
               attr.location.type == ACL_MEM_LOCATION_TYPE_HOST_NUMA ||
               attr.location.type == ACL_MEM_LOCATION_TYPE_UNREGISTERED;
    }

    static inline cudaError_t cudaCompatMemcpyHostRows(void *dst, size_t dstPitch,
                                                       const void *src, size_t srcPitch,
                                                       size_t width, size_t height)
    {
        for (size_t y = 0; y < height; ++y) {
            cudaError_t ret = cudaCompatMemcpyChecked((char *)dst + y * dstPitch,
                                                      width,
                                                      (const char *)src + y * srcPitch,
                                                      width);
            if (ret != cudaSuccess) {
                return ret;
            }
        }
        return cudaSuccess;
    }

    static inline cudaError_t cudaCompatMemcpyDeviceRows(void *dst, size_t dstPitch,
                                                         const void *src, size_t srcPitch,
                                                         size_t width, size_t height,
                                                         cudaStream_t stream)
    {
        for (size_t y = 0; y < height; ++y) {
            aclError rowRet = (stream == NULL) ?
                aclrtMemcpy((char *)dst + y * dstPitch, width,
                            (const char *)src + y * srcPitch, width,
                            ACL_MEMCPY_DEVICE_TO_DEVICE) :
                aclrtMemcpyAsync((char *)dst + y * dstPitch, width,
                                 (const char *)src + y * srcPitch, width,
                                 ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
            if (rowRet != ACL_SUCCESS) {
                return acl2cudaError(rowRet);
            }
        }
        return cudaSuccess;
    }

    static inline cudaError_t cudaCompatMemcpy3DImpl(const cudaMemcpy3DParms *p,
                                                     cudaStream_t stream)
    {
        cudaError_t valid = cudaCompatValidateMemcpy3DPtr(p);
        if (valid != cudaSuccess) {
            return valid;
        }
        for (size_t z = 0; z < p->extent.depth; ++z) {
            void *dst = cudaCompatSlicePtr(p->dstPtr, p->dstPos, z);
            const void *src = cudaCompatSlicePtr(p->srcPtr, p->srcPos, z);
            if (p->kind == cudaMemcpyHostToHost) {
                cudaError_t ret = cudaCompatMemcpyHostRows(dst, p->dstPtr.pitch, src, p->srcPtr.pitch,
                                                           p->extent.width, p->extent.height);
                if (ret != cudaSuccess) {
                    return ret;
                }
                continue;
            }
            if (p->kind == cudaMemcpyDeviceToDevice) {
                cudaError_t ret = cudaCompatMemcpyDeviceRows(dst, p->dstPtr.pitch, src, p->srcPtr.pitch,
                                                             p->extent.width, p->extent.height, stream);
                if (ret != cudaSuccess) {
                    return ret;
                }
                continue;
            }
            aclError ret = (stream == NULL) ?
                aclrtMemcpy2d(dst, p->dstPtr.pitch, src, p->srcPtr.pitch,
                              p->extent.width, p->extent.height,
                              (aclrtMemcpyKind)p->kind) :
                aclrtMemcpy2dAsync(dst, p->dstPtr.pitch, src, p->srcPtr.pitch,
                                   p->extent.width, p->extent.height,
                                   (aclrtMemcpyKind)p->kind, stream);
            if (ret != ACL_SUCCESS) {
                return acl2cudaError(ret);
            }
        }
        return cudaSuccess;
    }

    static inline cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms *p)
    {
        return cudaCompatMemcpy3DImpl(p, NULL);
    }

    static inline cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms *p, cudaStream_t stream)
    {
        return cudaCompatMemcpy3DImpl(p, stream);
    }

    static inline cudaError_t cudaCompatPeerParmsToMemcpy3D(const cudaMemcpy3DPeerParms *peer,
                                                            cudaMemcpy3DParms *copy)
    {
        if (!peer || !copy) {
            return cudaErrorInvalidValue;
        }
        (void)peer->srcDevice;
        (void)peer->dstDevice;
#ifdef __cplusplus
        *copy = cudaMemcpy3DParms();
#else
        *copy = (cudaMemcpy3DParms){0};
#endif
        copy->srcArray = peer->srcArray;
        copy->srcPos = peer->srcPos;
        copy->srcPtr = peer->srcPtr;
        copy->dstArray = peer->dstArray;
        copy->dstPos = peer->dstPos;
        copy->dstPtr = peer->dstPtr;
        copy->extent = peer->extent;
        copy->kind = cudaMemcpyDeviceToDevice;
        return cudaSuccess;
    }

    static inline cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms *p)
    {
        cudaMemcpy3DParms copy;
        cudaError_t ret = cudaCompatPeerParmsToMemcpy3D(p, &copy);
        if (ret != cudaSuccess) {
            return ret;
        }
        return cudaMemcpy3D(&copy);
    }

    static inline cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms *p, cudaStream_t stream)
    {
        cudaMemcpy3DParms copy;
        cudaError_t ret = cudaCompatPeerParmsToMemcpy3D(p, &copy);
        if (ret != cudaSuccess) {
            return ret;
        }
        return cudaMemcpy3DAsync(&copy, stream);
    }

    static inline cudaError_t cudaMemcpyWithAttributesAsync(void *dst, const void *src, size_t count,
                                                           cudaMemcpyAttributes *attrs, cudaStream_t stream)
    {
        if (!dst || !src || count == 0 || !attrs || !stream ||
            attrs->srcAccessOrder == cudaMemcpySrcAccessOrderInvalid ||
            attrs->srcAccessOrder >= cudaMemcpySrcAccessOrderMax ||
            (attrs->flags & ~((unsigned int)cudaMemcpyFlagPreferOverlapWithCompute)) != 0) {
            return cudaErrorInvalidValue;
        }
        int srcHost = cudaCompatPtrLikelyHost(src);
        int dstHost = cudaCompatPtrLikelyHost(dst);
        if (srcHost && dstHost) {
            return cudaErrorNotSupported; /* CANN has no stream-ordered H2H copy. */
        }
        aclrtMemcpyKind kind = srcHost ? ACL_MEMCPY_HOST_TO_DEVICE :
                               dstHost ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_DEVICE_TO_DEVICE;
        aclError ret = aclrtMemcpyAsync(dst, count, src, count, kind, stream);
        if (ret != ACL_SUCCESS) {
            return acl2cudaError(ret);
        }
        if (attrs->srcAccessOrder == cudaMemcpySrcAccessOrderDuringApiCall) {
            return acl2cudaError(aclrtSynchronizeStream(stream));
        }
        return cudaSuccess;
    }

    static inline cudaMemLocationType cudaCompatInferMemcpyLocation(cudaMemLocationType hint,
                                                                    const void *ptr)
    {
        if (hint == cudaMemLocationTypeHost) {
            return cudaMemLocationTypeHost;
        }
        if (hint == cudaMemLocationTypeDevice || hint == cudaMemLocationTypeInvalid) {
            return (hint == cudaMemLocationTypeInvalid && cudaCompatPtrLikelyHost(ptr)) ?
                cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
        }
        return hint;
    }

    static inline cudaMemcpyKind cudaCompatMemcpyKindFromLocations(cudaMemLocationType srcLoc,
                                                                  cudaMemLocationType dstLoc)
    {
        if (srcLoc == cudaMemLocationTypeHost && dstLoc == cudaMemLocationTypeHost) {
            return cudaMemcpyHostToHost;
        }
        if (srcLoc == cudaMemLocationTypeHost && dstLoc == cudaMemLocationTypeDevice) {
            return cudaMemcpyHostToDevice;
        }
        if (srcLoc == cudaMemLocationTypeDevice && dstLoc == cudaMemLocationTypeHost) {
            return cudaMemcpyDeviceToHost;
        }
        return cudaMemcpyDeviceToDevice;
    }

    static inline cudaError_t cudaCompatBatchPointerOperands(const cudaMemcpy3DBatchOp *op)
    {
        if (!op || op->extent.width == 0 || op->extent.height == 0 || op->extent.depth == 0) {
            return cudaErrorInvalidValue;
        }
        if (op->src.type != cudaMemcpyOperandTypePointer || op->dst.type != cudaMemcpyOperandTypePointer ||
            !op->src.op.ptr.ptr || !op->dst.op.ptr.ptr) {
            return cudaErrorNotSupported;
        }
        return cudaSuccess;
    }

    static inline cudaError_t cudaCompatMemcpy3DBatchOpToParms(const cudaMemcpy3DBatchOp *op,
                                                               cudaMemcpy3DParms *p)
    {
        if (!p) {
            return cudaErrorInvalidValue;
        }
        cudaError_t valid = cudaCompatBatchPointerOperands(op);
        if (valid != cudaSuccess) {
            return valid;
        }
#ifdef __cplusplus
        *p = cudaMemcpy3DParms();
#else
        *p = (cudaMemcpy3DParms){0};
#endif
        size_t srcRowLength = op->src.op.ptr.rowLength ? op->src.op.ptr.rowLength : op->extent.width;
        size_t dstRowLength = op->dst.op.ptr.rowLength ? op->dst.op.ptr.rowLength : op->extent.width;
        size_t srcLayerHeight = op->src.op.ptr.layerHeight ? op->src.op.ptr.layerHeight : op->extent.height;
        size_t dstLayerHeight = op->dst.op.ptr.layerHeight ? op->dst.op.ptr.layerHeight : op->extent.height;
        p->srcPtr = make_cudaPitchedPtr(op->src.op.ptr.ptr, srcRowLength, srcRowLength, srcLayerHeight);
        p->dstPtr = make_cudaPitchedPtr(op->dst.op.ptr.ptr, dstRowLength, dstRowLength, dstLayerHeight);
        p->extent = op->extent;
        cudaMemLocationType srcLoc = cudaCompatInferMemcpyLocation(op->src.op.ptr.locHint.type,
                                                                   op->src.op.ptr.ptr);
        cudaMemLocationType dstLoc = cudaCompatInferMemcpyLocation(op->dst.op.ptr.locHint.type,
                                                                   op->dst.op.ptr.ptr);
        p->kind = cudaCompatMemcpyKindFromLocations(srcLoc, dstLoc);
        return cudaSuccess;
    }

    static inline cudaError_t cudaMemcpy3DBatchAsync(size_t numOps, cudaMemcpy3DBatchOp *opList,
                                                     unsigned long long flags, cudaStream_t stream)
    {
        if (!opList || numOps == 0 || flags != 0 || !stream) {
            return cudaErrorInvalidValue;
        }
        /* Validate the whole batch before enqueueing any operation. */
        for (size_t i = 0; i < numOps; ++i) {
            if (opList[i].srcAccessOrder == cudaMemcpySrcAccessOrderInvalid ||
                opList[i].srcAccessOrder >= cudaMemcpySrcAccessOrderMax ||
                (opList[i].flags & ~((unsigned int)cudaMemcpyFlagPreferOverlapWithCompute)) != 0) {
                return cudaErrorInvalidValue;
            }
            cudaMemcpy3DParms p;
            cudaError_t valid = cudaCompatMemcpy3DBatchOpToParms(&opList[i], &p);
            if (valid != cudaSuccess) {
                return valid;
            }
            valid = cudaCompatValidateMemcpy3DPtr(&p);
            if (valid != cudaSuccess) {
                return valid;
            }
        }
        for (size_t i = 0; i < numOps; ++i) {
            cudaMemcpy3DParms p;
            cudaError_t ret = cudaCompatMemcpy3DBatchOpToParms(&opList[i], &p);
            if (ret != cudaSuccess) {
                return ret;
            }
            if (opList[i].srcAccessOrder == cudaMemcpySrcAccessOrderDuringApiCall) {
                ret = acl2cudaError(aclrtSynchronizeStream(stream));
                if (ret != cudaSuccess) {
                    return ret;
                }
            }
            ret = cudaMemcpy3DAsync(&p, stream);
            if (ret != cudaSuccess) {
                return ret;
            }
        }
        return cudaSuccess;
    }

    static inline cudaError_t cudaMemcpy3DWithAttributesAsync(cudaMemcpy3DBatchOp *op,
                                                              unsigned long long flags,
                                                              cudaStream_t stream)
    {
        cudaError_t valid = cudaCompatBatchPointerOperands(op);
        if (valid != cudaSuccess) {
            return valid;
        }
        if (flags != 0 || !stream ||
            op->srcAccessOrder == cudaMemcpySrcAccessOrderInvalid ||
            op->srcAccessOrder >= cudaMemcpySrcAccessOrderMax ||
            (op->flags & ~((unsigned int)cudaMemcpyFlagPreferOverlapWithCompute)) != 0) {
            return cudaErrorInvalidValue;
        }

        size_t srcRowLength = op->src.op.ptr.rowLength ? op->src.op.ptr.rowLength : op->extent.width;
        size_t dstRowLength = op->dst.op.ptr.rowLength ? op->dst.op.ptr.rowLength : op->extent.width;
        size_t srcLayerHeight = op->src.op.ptr.layerHeight ? op->src.op.ptr.layerHeight : op->extent.height;
        size_t dstLayerHeight = op->dst.op.ptr.layerHeight ? op->dst.op.ptr.layerHeight : op->extent.height;
        if (srcRowLength < op->extent.width || dstRowLength < op->extent.width ||
            srcLayerHeight < op->extent.height || dstLayerHeight < op->extent.height ||
            op->extent.depth > SIZE_MAX / op->extent.height) {
            return cudaErrorInvalidValue;
        }

        size_t copyCount = op->extent.depth * op->extent.height;
        if (copyCount > SIZE_MAX / sizeof(void *) || copyCount > SIZE_MAX / sizeof(size_t)) {
            return cudaErrorInvalidValue;
        }
        void **dsts = (void **)calloc(copyCount, sizeof(void *));
        const void **srcs = (const void **)calloc(copyCount, sizeof(void *));
        size_t *sizes = (size_t *)calloc(copyCount, sizeof(size_t));
        if (!dsts || !srcs || !sizes) {
            free(dsts);
            free(srcs);
            free(sizes);
            return cudaErrorMemoryAllocation;
        }

        size_t copyIndex = 0;
        for (size_t z = 0; z < op->extent.depth; ++z) {
            for (size_t y = 0; y < op->extent.height; ++y) {
                srcs[copyIndex] = (const char *)op->src.op.ptr.ptr +
                    (z * srcLayerHeight + y) * srcRowLength;
                dsts[copyIndex] = (char *)op->dst.op.ptr.ptr +
                    (z * dstLayerHeight + y) * dstRowLength;
                sizes[copyIndex] = op->extent.width;
                ++copyIndex;
            }
        }

        cudaMemcpyAttributes attr;
        attr.srcAccessOrder = op->srcAccessOrder;
        attr.srcLocHint = op->src.op.ptr.locHint;
        attr.dstLocHint = op->dst.op.ptr.locHint;
        attr.flags = op->flags;
        size_t attrsIndex = 0;
        cudaError_t ret = cudaMemcpyBatchAsync(dsts, srcs, sizes, copyCount,
                                               &attr, &attrsIndex, 1, stream);
        free(dsts);
        free(srcs);
        free(sizes);
        return ret;
    }


    static inline cudaError_t cudaMemcpyPeer(void *dst, int dstDevice,
                                            const void *src, int srcDevice,
                                            size_t count)
    {
        uint32_t deviceCount = 0;
        aclError countRet = aclrtGetDeviceCount(&deviceCount);
        if (countRet != ACL_SUCCESS) {
            return acl2cudaError(countRet);
        }
        if (dstDevice < 0 || srcDevice < 0 ||
            (uint32_t)dstDevice >= deviceCount || (uint32_t)srcDevice >= deviceCount) {
            return cudaErrorInvalidDevice;
        }

        aclrtPtrAttributes ptrAttr;
        if (aclrtPointerGetAttributes(dst, &ptrAttr) == ACL_SUCCESS &&
            ptrAttr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE &&
            ptrAttr.location.id != (uint32_t)dstDevice) {
            return cudaErrorInvalidDevice;
        }
        if (aclrtPointerGetAttributes(src, &ptrAttr) == ACL_SUCCESS &&
            ptrAttr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE &&
            ptrAttr.location.id != (uint32_t)srcDevice) {
            return cudaErrorInvalidDevice;
        }

        aclError ret = aclrtMemcpy(dst, count, src, count,
                                    ACL_MEMCPY_DEVICE_TO_DEVICE);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice,
                                                  const void *src, int srcDevice,
                                                  size_t count, cudaStream_t stream)
    {
        (void)dstDevice;
        (void)srcDevice;
        // CANN supports device-to-device async copy only within the supported topology.
        aclError ret = aclrtMemcpyAsync(dst, count, src, count,
                                         ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaMemset(void *devPtr, int value, size_t count)
    {
        aclError ret = aclrtMemset(devPtr, count, value, count);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaMemsetAsync(void *devPtr, int value,
                                               size_t count, cudaStream_t stream)
    {
        aclError ret = aclrtMemsetAsync(devPtr, count, value, count, stream);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaMemset2D(void *devPtr, size_t pitch,
                                          int value, size_t width, size_t height)
    {
        // Validate parameters
        if (!devPtr || width == 0 || height == 0) {
            return cudaErrorInvalidValue;
        }

        // Set each row separately
        char *row_ptr = (char *)devPtr;
        for (size_t i = 0; i < height; i++) {
            aclError ret = aclrtMemset(row_ptr, width, value, width);
            if (ret != ACL_SUCCESS) {
                return acl2cudaError(ret);
            }
            row_ptr += pitch;
        }

        return cudaSuccess;
    }


    static inline cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch,
                                               int value, size_t width,
                                               size_t height, cudaStream_t stream)
    {
        // Validate parameters
        if (!devPtr || width == 0 || height == 0) {
            return cudaErrorInvalidValue;
        }

        // Set each row separately asynchronously
        char *row_ptr = (char *)devPtr;
        for (size_t i = 0; i < height; i++) {
            aclError ret = aclrtMemsetAsync(row_ptr, width, value, width, stream);
            if (ret != ACL_SUCCESS) {
                return acl2cudaError(ret);
            }
            row_ptr += pitch;
        }

        return cudaSuccess;
    }

    static inline cudaError_t cudaCompatValidateMemset3D(cudaPitchedPtr pitchedDevPtr,
                                                         cudaExtent extent)
    {
        if (!pitchedDevPtr.ptr || extent.width == 0 || extent.height == 0 || extent.depth == 0 ||
            pitchedDevPtr.pitch < extent.width) {
            return cudaErrorInvalidValue;
        }
        if (pitchedDevPtr.ysize != 0 && extent.height > pitchedDevPtr.ysize) {
            return cudaErrorInvalidValue;
        }
        return cudaSuccess;
    }

    static inline cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value,
                                           cudaExtent extent)
    {
        cudaError_t valid = cudaCompatValidateMemset3D(pitchedDevPtr, extent);
        if (valid != cudaSuccess) {
            return valid;
        }
        for (size_t z = 0; z < extent.depth; ++z) {
            char *slice = (char *)pitchedDevPtr.ptr + z * pitchedDevPtr.pitch * pitchedDevPtr.ysize;
            for (size_t y = 0; y < extent.height; ++y) {
                aclError ret = aclrtMemset(slice + y * pitchedDevPtr.pitch, extent.width, value, extent.width);
                if (ret != ACL_SUCCESS) {
                    return acl2cudaError(ret);
                }
            }
        }
        return cudaSuccess;
    }

    static inline cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value,
                                                cudaExtent extent, cudaStream_t stream)
    {
        cudaError_t valid = cudaCompatValidateMemset3D(pitchedDevPtr, extent);
        if (valid != cudaSuccess) {
            return valid;
        }
        for (size_t z = 0; z < extent.depth; ++z) {
            char *slice = (char *)pitchedDevPtr.ptr + z * pitchedDevPtr.pitch * pitchedDevPtr.ysize;
            for (size_t y = 0; y < extent.height; ++y) {
                aclError ret = aclrtMemsetAsync(slice + y * pitchedDevPtr.pitch, extent.width,
                                                value, extent.width, stream);
                if (ret != ACL_SUCCESS) {
                    return acl2cudaError(ret);
                }
            }
        }
        return cudaSuccess;
    }

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_MEMORY_COPY_H */
