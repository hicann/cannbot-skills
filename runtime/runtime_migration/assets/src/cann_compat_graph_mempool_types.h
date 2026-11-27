/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */

#ifndef CUDA_COMPAT_GRAPH_MEMPOOL_TYPES_H
#define CUDA_COMPAT_GRAPH_MEMPOOL_TYPES_H

    /* =================================================================
     * IPC Memory Handle Types
     * ================================================================= */


#define CANN_IPC_MEM_HANDLE_SIZE 65

    typedef struct
    {
        char internal[CANN_IPC_MEM_HANDLE_SIZE]; // CANN export key
        size_t size;                             // Memory size
    } cudaIpcMemHandle_t;

    /* IPC memory flags */
#define cudaIpcMemLazyEnablePeerAccess 0x1

    /* =================================================================
     * IPC Event Handle Types
     * ================================================================= */

    typedef aclrtIpcEventHandle cudaIpcEventHandle_t;
    /* =================================================================
     * Stream Capture Types
     * ================================================================= */
    typedef enum
    {
        cudaStreamCaptureStatusNone = 0,       /* Not capturing */
        cudaStreamCaptureStatusActive = 1,     /* Currently capturing */
        cudaStreamCaptureStatusInvalidated = 2 /* Capture invalidated */
    } cudaStreamCaptureStatus;

    /* Stream capture mode */
    typedef enum
    {
        cudaStreamCaptureModeGlobal = 0,      /* Global capture mode */
        cudaStreamCaptureModeThreadLocal = 1, /* Thread-local capture mode */
        cudaStreamCaptureModeRelaxed = 2      /* Relaxed capture mode */
    } cudaStreamCaptureMode;

    /* CUDA graph (mapped to CANN's aclmdlRI capture/build result) */
    typedef aclmdlRI cudaGraph_t;
    typedef aclmdlRI cudaGraphExec_t;
    typedef void *cudaGraphNode_t;
    typedef aclmdlRICondHandle cudaGraphConditionalHandle;
    typedef aclrtContext cudaExecutionContext_t;
    typedef aclrtFuncHandle cudaFunction_t;
    typedef aclrtFuncHandle cudaKernel_t;

#if !defined(__VECTOR_TYPES_H__) && !defined(CANN_COMPAT_DIM3_DEFINED) && \
    (!defined(INC_EXTERNAL_ACL_ACL_RT_H_) || defined(__BISHENG_CCEC__))
#define CANN_COMPAT_DIM3_DEFINED
    typedef struct dim3 {
        unsigned int x;
        unsigned int y;
        unsigned int z;
#ifdef __cplusplus
        constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
#endif
    } dim3;
#endif

#define cudaGraphCondAssignDefault 0x1U

    typedef enum
    {
        cudaGraphNodeTypeKernel = 0,
        cudaGraphNodeTypeMemcpy = 1,
        cudaGraphNodeTypeMemset = 2,
        cudaGraphNodeTypeHost = 3,
        cudaGraphNodeTypeGraph = 4,
        cudaGraphNodeTypeEmpty = 5,
        cudaGraphNodeTypeWaitEvent = 6,
        cudaGraphNodeTypeEventRecord = 7,
        cudaGraphNodeTypeExtSemaphoreSignal = 8,
        cudaGraphNodeTypeExtSemaphoreWait = 9,
        cudaGraphNodeTypeMemAlloc = 10,
        cudaGraphNodeTypeMemFree = 11,
        cudaGraphNodeTypeBatchMemOp = 12,
        cudaGraphNodeTypeConditional = 13
    } cudaGraphNodeType;

    typedef enum
    {
        cudaGraphMemAttrUsedMemCurrent = 0,
        cudaGraphMemAttrUsedMemHigh = 1,
        cudaGraphMemAttrReservedMemCurrent = 2,
        cudaGraphMemAttrReservedMemHigh = 3
    } cudaGraphMemAttributeType;

    typedef enum
    {
        cudaGraphExecUpdateSuccess = 0,
        cudaGraphExecUpdateError = 1,
        cudaGraphExecUpdateErrorTopologyChanged = 2,
        cudaGraphExecUpdateErrorNodeTypeChanged = 3,
        cudaGraphExecUpdateErrorFunctionChanged = 4,
        cudaGraphExecUpdateErrorParametersChanged = 5,
        cudaGraphExecUpdateErrorNotSupported = 6,
        cudaGraphExecUpdateErrorUnsupportedFunctionChange = 7,
        cudaGraphExecUpdateErrorAttributesChanged = 8
    } cudaGraphExecUpdateResult;

    typedef struct
    {
        cudaGraphExecUpdateResult result;
        cudaGraphNode_t errorNode;
        cudaGraphNode_t errorFromNode;
    } cudaGraphExecUpdateResultInfo;

    typedef struct
    {
        void *func;
        dim3 gridDim;
        dim3 blockDim;
        unsigned int sharedMemBytes;
        void **kernelParams;
        void **extra;
    } cudaKernelNodeParams;

    typedef enum
    {
        cudaKernelFunctionTypeUnspecified = 0x00,
        cudaKernelFunctionTypeDeviceEntry = 0x01,
        cudaKernelFunctionTypeKernel = 0x02,
        cudaKernelFunctionTypeFunction = 0x03
    } cudaKernelFunctionType;

    typedef struct
    {
        cudaKernelFunctionType functionType;
        union
        {
            void *func;
            cudaKernel_t kern;
            cudaFunction_t cuFunc;
        };
        dim3 gridDim;
        dim3 blockDim;
        unsigned int sharedMemBytes;
        void **kernelParams;
        void **extra;
        cudaExecutionContext_t ctx;
    } cudaKernelNodeParamsV2;

    typedef struct
    {
        cudaEvent_t event;
        cudaExecutionContext_t ctx;
    } cudaEventRecordNodeParams;

    typedef struct
    {
        cudaEvent_t event;
        cudaExecutionContext_t ctx;
    } cudaEventWaitNodeParams;

    typedef enum
    {
        cudaGraphCondTypeIf = 0,
        cudaGraphCondTypeWhile = 1,
        cudaGraphCondTypeSwitch = 2
    } cudaGraphConditionalNodeType;

    typedef struct
    {
        cudaGraphConditionalHandle handle;
        cudaGraphConditionalNodeType type;
        unsigned int size;
        cudaGraph_t *phGraph_out;
    } cudaGraphConditionalNodeParams;

    typedef struct
    {
        cudaGraphNodeType type;
        int reserved0[3];
        union
        {
            cudaGraphConditionalNodeParams conditional;
            cudaKernelNodeParamsV2 kernel;
            cudaEventRecordNodeParams eventRecord;
            cudaEventWaitNodeParams eventWait;
            long long reserved1[29];
        };
        long long reserved2;
    } cudaGraphNodeParams;

    typedef struct cudaCompatGraphCaptureEntry_st
    {
        cudaGraph_t graph;
        cudaStream_t stream;
        struct cudaCompatGraphCaptureEntry_st *next;
    } cudaCompatGraphCaptureEntry;

    static inline cudaCompatGraphCaptureEntry **cudaCompatGraphCaptureRegistry(void)
    {
        static cudaCompatGraphCaptureEntry *head = NULL;
        return &head;
    }

    static inline void cudaCompatRegisterGraphCaptureStream(cudaGraph_t graph, cudaStream_t stream)
    {
        if (!graph || !stream) {
            return;
        }
        cudaCompatGraphCaptureEntry **head = cudaCompatGraphCaptureRegistry();
        for (cudaCompatGraphCaptureEntry *entry = *head; entry; entry = entry->next) {
            if (entry->graph == graph) {
                entry->stream = stream;
                return;
            }
        }
        cudaCompatGraphCaptureEntry *entry = (cudaCompatGraphCaptureEntry *)malloc(sizeof(cudaCompatGraphCaptureEntry));
        if (!entry) {
            return;
        }
        entry->graph = graph;
        entry->stream = stream;
        entry->next = *head;
        *head = entry;
    }

    static inline cudaStream_t cudaCompatFindGraphCaptureStream(cudaGraph_t graph)
    {
        cudaCompatGraphCaptureEntry **head = cudaCompatGraphCaptureRegistry();
        for (cudaCompatGraphCaptureEntry *entry = *head; entry; entry = entry->next) {
            if (entry->graph == graph) {
                return entry->stream;
            }
        }
        return NULL;
    }

    static inline void cudaCompatUnregisterGraphCaptureStream(cudaStream_t stream)
    {
        cudaCompatGraphCaptureEntry **head = cudaCompatGraphCaptureRegistry();
        cudaCompatGraphCaptureEntry **link = head;
        while (*link) {
            cudaCompatGraphCaptureEntry *entry = *link;
            if (entry->stream == stream) {
                *link = entry->next;
                free(entry);
            } else {
                link = &entry->next;
            }
        }
    }

    typedef aclrtLaunchKernelAttr cudaLaunchAttribute;

    typedef struct cudaLaunchConfig_t {
        dim3 gridDim;
        dim3 blockDim;
        size_t dynamicSmemBytes;
        cudaStream_t stream;
        cudaLaunchAttribute *attrs;
        unsigned int numAttrs;
    } cudaLaunchConfig_t;

#define cudaGraphDebugDotFlagsVerbose 0x1
#define cudaGraphDebugDotFlagsKernelNodeParams 0x4
#define cudaGraphDebugDotFlagsMemcpyNodeParams 0x8
#define cudaGraphDebugDotFlagsMemsetNodeParams 0x10
#define cudaGraphDebugDotFlagsHostNodeParams 0x20
#define cudaGraphDebugDotFlagsEventNodeParams 0x40
#define cudaGraphDebugDotFlagsExtSemasSignalNodeParams 0x80
#define cudaGraphDebugDotFlagsExtSemasWaitNodeParams 0x100
#define cudaGraphDebugDotFlagsKernelNodeAttributes 0x200
#define cudaGraphDebugDotFlagsHandles 0x400

    typedef enum
    {
        cudaGraphDependencyTypeDefault = 0,
        cudaGraphDependencyTypeProgrammatic = 1
    } cudaGraphDependencyType;

    typedef struct
    {
        cudaGraphNode_t from;
        cudaGraphNode_t to;
        cudaGraphDependencyType type;
    } cudaGraphEdgeData;
    /* =================================================================
     * Memory Pool Types (Mock Implementation)
     * ================================================================= */
    typedef struct cudaMemPool_st *cudaMemPool_t;


    typedef enum
    {
        cudaMemPoolTypeUnspecified = 0,
        cudaMemPoolTypeDevice = 1,
        cudaMemPoolTypeHost = 2
    } cudaMemPoolType;


    typedef struct
    {
        cudaMemAllocationType allocType;
        cudaMemAllocationHandleType handleTypes;
        cudaMemLocation location;
        void *win32SecurityAttributes;
        unsigned char cudaReserved[64];
        cudaMemPoolType memPoolType;
        size_t maxPageSize;
        size_t minPageSize;
        unsigned int reserved[4];
    } cudaMemPoolProps;


    typedef enum
    {
        cudaMemPoolAttrReservedMemCurrent = 0,
        cudaMemPoolAttrReservedMemHigh = 1,
        cudaMemPoolAttrUsedMemCurrent = 2,
        cudaMemPoolAttrUsedMemHigh = 3,
        cudaMemPoolAttrReleaseThreshold = 4,
        cudaMemPoolAttrReuseAllowOpportunistic = 5,
        cudaMemPoolAttrReuseAllowInternalDependencies = 6,
        cudaMemPoolAttrAccessPermissionMask = 7
    } cudaMemPoolAttr;


    typedef enum
    {
        cudaMemAccessDefault = 0,
        cudaMemAccessReadWrite = 1,
        cudaMemAccessRead = 2,
        cudaMemAccessNone = 3
    } cudaMemAccessFlags;


    typedef struct
    {
        cudaMemLocation location;
        cudaMemAccessFlags access;
    } cudaMemAccessDesc;

#endif /* CUDA_COMPAT_GRAPH_MEMPOOL_TYPES_H */
