/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */

#ifndef CUDA_COMPAT_MEMORY_TYPES_H
#define CUDA_COMPAT_MEMORY_TYPES_H

    /* =================================================================
     * Memory Types
     * ================================================================= */
    typedef enum
    {
        cudaMemoryTypeUnregistered = 0,
        cudaMemoryTypeHost = 1,
        cudaMemoryTypeDevice = 2,
        cudaMemoryTypeManaged = 3
    } cudaMemoryType;

    typedef struct
    {
        cudaMemoryType type;
        int device;
        void *devicePointer;
        void *hostPointer;
    } cudaPointerAttributes;

    typedef enum
    {
        cudaMemLocationTypeInvalid = 0,
        cudaMemLocationTypeDevice = 1,
        cudaMemLocationTypeHost = 2,
        cudaMemLocationTypeHostNuma = 3,
        cudaMemLocationTypeHostNumaCurrent = 4
    } cudaMemLocationType;

    typedef struct
    {
        cudaMemLocationType type;
        int id;
    } cudaMemLocation;

    typedef enum
    {
        cudaMemAllocationTypeInvalid = 0,
        cudaMemAllocationTypePinned = 1,
        cudaMemAllocationTypeMax = 0x7fffffff
    } cudaMemAllocationType;

    typedef enum
    {
        cudaMemHandleTypeNone = 0,
        cudaMemHandleTypePosixFileDescriptor = 1,
        cudaMemHandleTypeWin32 = 2,
        cudaMemHandleTypeWin32Kmt = 4,
        cudaMemHandleTypeFabric = 8
    } cudaMemAllocationHandleType;

    typedef enum
    {
        cudaMemcpyHostToHost = 0,
        cudaMemcpyHostToDevice = 1,
        cudaMemcpyDeviceToHost = 2,
        cudaMemcpyDeviceToDevice = 3,
        cudaMemcpyDefault = 4
    } cudaMemcpyKind;

    typedef enum
    {
        cudaMemAttachGlobal = 1,
        cudaMemAttachHost = 2,
        cudaMemAttachSingle = 4
    } cudaMemAttachFlags;

    typedef enum
    {
        cudaMemAdviseSetReadMostly = 1,
        cudaMemAdviseUnsetReadMostly = 2,
        cudaMemAdviseSetPreferredLocation = 3,
        cudaMemAdviseUnsetPreferredLocation = 4,
        cudaMemAdviseSetAccessedBy = 5,
        cudaMemAdviseUnsetAccessedBy = 6
    } cudaMemoryAdvise;

    typedef enum
    {
        cudaMemRangeAttributeReadMostly = 1,
        cudaMemRangeAttributePreferredLocation = 2,
        cudaMemRangeAttributeAccessedBy = 3,
        cudaMemRangeAttributeLastPrefetchLocation = 4
    } cudaMemRangeAttribute;

    typedef enum
    {
        cudaHostAllocDefault = 0,
        cudaHostAllocPortable = 1,
        cudaHostAllocMapped = 2,
        cudaHostAllocWriteCombined = 4,
    } cudaHostAllocFlags;

    typedef enum
    {
        cudaHostRegisterDefault = 0,
        cudaHostRegisterPortable = 1,
        cudaHostRegisterMapped = 2,
        cudaHostRegisterIoMemory = 4,
        cudaHostRegisterReadOnly = 8
    } cudaHostRegisterFlags;

    typedef enum cudaMemcpySrcAccessOrder
    {
        cudaMemcpySrcAccessOrderInvalid = 0x0,
        cudaMemcpySrcAccessOrderStream = 0x1,
        cudaMemcpySrcAccessOrderDuringApiCall = 0x2,
        cudaMemcpySrcAccessOrderAny = 0x3,
        cudaMemcpySrcAccessOrderMax = 0x7FFFFFFF
    } cudaMemcpySrcAccessOrder;


    typedef struct
    {
        cudaMemcpySrcAccessOrder srcAccessOrder;
        cudaMemLocation srcLocHint;
        cudaMemLocation dstLocHint;
        unsigned int flags;
    } cudaMemcpyAttributes;

    typedef struct cudaArray *cudaArray_t;

    typedef struct
    {
        size_t x;
        size_t y;
        size_t z;
    } cudaPos;

    typedef struct
    {
        size_t width;
        size_t height;
        size_t depth;
    } cudaExtent;

    typedef struct
    {
        void *ptr;
        size_t pitch;
        size_t xsize;
        size_t ysize;
    } cudaPitchedPtr;

    typedef struct
    {
        cudaArray_t srcArray;
        cudaPos srcPos;
        cudaPitchedPtr srcPtr;
        cudaArray_t dstArray;
        cudaPos dstPos;
        cudaPitchedPtr dstPtr;
        cudaExtent extent;
        cudaMemcpyKind kind;
    } cudaMemcpy3DParms;

    typedef struct
    {
        cudaArray_t srcArray;
        cudaPos srcPos;
        cudaPitchedPtr srcPtr;
        int srcDevice;
        cudaArray_t dstArray;
        cudaPos dstPos;
        cudaPitchedPtr dstPtr;
        int dstDevice;
        cudaExtent extent;
    } cudaMemcpy3DPeerParms;

    typedef struct
    {
        cudaPitchedPtr pitchedDevPtr;
        int value;
        cudaExtent extent;
    } cudaMemset3DParms;

    typedef enum
    {
        cudaMemcpyOperandTypePointer = 0,
        cudaMemcpyOperandTypeArray = 1
    } cudaMemcpy3DOperandType;

    typedef struct
    {
        cudaMemcpy3DOperandType type;
        union
        {
            struct
            {
                void *ptr;
                size_t rowLength;
                size_t layerHeight;
                cudaMemLocation locHint;
            } ptr;
            struct
            {
                cudaArray_t array;
                cudaPos offset;
            } array;
        } op;
    } cudaMemcpy3DOperand;

    typedef struct
    {
        cudaMemcpy3DOperand src;
        cudaMemcpy3DOperand dst;
        cudaExtent extent;
        cudaMemcpySrcAccessOrder srcAccessOrder;
        cudaMemcpyAttributes attr;
    } cudaMemcpy3DBatchOp;

    static inline cudaPos make_cudaPos(size_t x, size_t y, size_t z)
    {
        cudaPos pos;
        pos.x = x;
        pos.y = y;
        pos.z = z;
        return pos;
    }

    static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d)
    {
        cudaExtent extent;
        extent.width = w;
        extent.height = h;
        extent.depth = d;
        return extent;
    }

    static inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz)
    {
        cudaPitchedPtr pitchedPtr;
        pitchedPtr.ptr = d;
        pitchedPtr.pitch = p;
        pitchedPtr.xsize = xsz;
        pitchedPtr.ysize = ysz;
        return pitchedPtr;
    }

#endif /* CUDA_COMPAT_MEMORY_TYPES_H */
