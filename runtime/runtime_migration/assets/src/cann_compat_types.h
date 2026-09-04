/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */


/*
 * Project-local compatibility types used by the CUDA-to-CANN shim.
 *
 * Symbol names and numeric constants are present for source compatibility with
 * migrated programs. Do not paste vendor SDK header prose or documentation into
 * this file.
 */

#ifndef CUDA_COMPAT_TYPES_H
#define CUDA_COMPAT_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include "acl/acl_rt.h"
#include "cann_compat_device_types.h"
#include "cann_compat_error_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define MOCK_CUDA_MAJOR_VERSION 13
#define MOCK_CUDA_MINOR_VERSION 0
#ifndef CUDART_VERSION
#define CUDART_VERSION 13000
#endif

#define cudaStreamNonDefault    0x00
#define cudaStreamNonBlocking   0x01

    /* =================================================================
     * Device Management Types
     * ================================================================= */

    typedef aclrtUuid cudaUUID_t;
    typedef struct
    {
        char name[256];
        cudaUUID_t uuid;
        char luid[8];
        unsigned int luidDeviceNodeMask;
        size_t totalGlobalMem;
        size_t sharedMemPerBlock;
        int regsPerBlock;
        int warpSize;
        size_t memPitch;
        int maxThreadsPerBlock;
        int maxThreadsDim[3];
        int maxGridSize[3];
        size_t totalConstMem;
        int major;
        int minor;
        size_t textureAlignment;
        size_t texturePitchAlignment;
        int multiProcessorCount;
        int integrated;
        int canMapHostMemory;
        int maxTexture1D;
        int maxTexture1DMipmap;
        int maxTexture2D[2];
        int maxTexture2DMipmap[2];
        int maxTexture2DLinear[3];
        int maxTexture2DGather[2];
        int maxTexture3D[3];
        int maxTexture3DAlt[3];
        int maxTextureCubemap;
        int maxTexture1DLayered[2];
        int maxTexture2DLayered[3];
        int maxTextureCubemapLayered[2];
        int maxSurface1D;
        int maxSurface2D[2];
        int maxSurface3D[3];
        int maxSurface1DLayered[2];
        int maxSurface2DLayered[3];
        int maxSurfaceCubemap;
        int maxSurfaceCubemapLayered[2];
        size_t surfaceAlignment;
        int concurrentKernels;
        int ECCEnabled;
        int pciBusID;
        int pciDeviceID;
        int pciDomainID;
        int tccDriver;
        int asyncEngineCount;
        int unifiedAddressing;
        int memoryBusWidth;
        int l2CacheSize;
        int persistingL2CacheMaxSize;
        int maxThreadsPerMultiProcessor;
        int streamPrioritiesSupported;
        int globalL1CacheSupported;
        int localL1CacheSupported;
        size_t sharedMemPerMultiprocessor;
        int regsPerMultiprocessor;
        int managedMemory;
        int isMultiGpuBoard;
        int multiGpuBoardGroupID;
        int hostNativeAtomicSupported;
        int pageableMemoryAccess;
        int concurrentManagedAccess;
        int computePreemptionSupported;
        int canUseHostPointerForRegisteredMem;
        int cooperativeLaunch;
        size_t sharedMemPerBlockOptin;
        int pageableMemoryAccessUsesHostPageTables;
        int directManagedMemAccessFromHost;
        int maxBlocksPerMultiProcessor;
        int accessPolicyMaxWindowSize;
        size_t reservedSharedMemPerBlock;
        int hostRegisterSupported;
        int sparseCudaArraySupported;
        int hostRegisterReadOnlySupported;
        int timelineSemaphoreInteropSupported;
        int memoryPoolsSupported;
        int gpuDirectRDMASupported;
        unsigned int gpuDirectRDMAFlushWritesOptions;
        int gpuDirectRDMAWritesOrdering;
        unsigned int memoryPoolSupportedHandleTypes;
        int deferredMappingCudaArraySupported;
        int ipcEventSupported;
        int clusterLaunch;
        int unifiedFunctionPointers;
        int deviceNumaConfig;
        int deviceNumaId;
        int mpsEnabled;
        int hostNumaId;
        unsigned int gpuPciDeviceID;
        unsigned int gpuPciSubsystemID;
        int hostNumaMultinodeIpcSupported;
        int reserved[56];
    } cudaDeviceProp;

    /* =================================================================
     * Stream Types (Direct mapping)
     * ================================================================= */

    typedef aclrtStream cudaStream_t;
    /* =================================================================
     * Event Types
     * ================================================================= */
    typedef aclrtEvent cudaEvent_t;

#define cudaEventDefault 0x00
#define cudaEventBlockingSync 0x01
#define cudaEventDisableTiming 0x02
#define cudaEventInterprocess 0x04

#define cudaEventRecordDefault 0x00
#define cudaEventRecordExternal 0x01

#define cudaEventWaitDefault 0x00
#define cudaEventWaitExternal 0x01

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

    /* =================================================================
     * Limit Types
     * ================================================================= */

    typedef enum
    {
        cudaLimitStackSize = 0x00,
        cudaLimitPrintfFifoSize = 0x01,
        cudaLimitMallocHeapSize = 0x02,
        cudaLimitDevRuntimeSyncDepth = 0x03,
        cudaLimitDevRuntimePendingLaunchCount = 0x04,
        cudaLimitMaxL2FetchGranularity = 0x05,
        cudaLimitPersistingL2CacheSize = 0x06
    } cudaLimit;

    /* =================================================================
     * Compute Mode
     * ================================================================= */


    typedef enum
    {
        cudaComputeModeDefault = 0,
        cudaComputeModeExclusive = 1,
        cudaComputeModeProhibited = 2,
        cudaComputeModeExclusiveProcess = 3
    } cudaComputeMode;

    /* =================================================================
     * Device Flags
     * ================================================================= */

    typedef enum
    {
        cudaDeviceScheduleAuto = 0,
        cudaDeviceScheduleSpin = 1,
        cudaDeviceScheduleYield = 2,
        cudaDeviceScheduleBlockingSync = 4,
        cudaDeviceMapHost = 8,
        cudaDeviceLmemResizeToMax = 16,
        cudaDeviceSyncMemops = 0x40000
    } cudaDeviceFlags;

    typedef enum
    {
        cudaFuncCachePreferNone = 0,
        cudaFuncCachePreferShared = 1,
        cudaFuncCachePreferL1 = 2,
        cudaFuncCachePreferEqual = 3
    } cudaFuncCache;

    typedef enum
    {
        cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
        cudaFuncAttributePreferredSharedMemoryCarveout = 9,
        cudaFuncAttributeMax = 10
    } cudaFuncAttribute;

    /* =================================================================
     * Function Attributes
     * ================================================================= */

    typedef struct
    {
        size_t sharedSizeBytes;
        size_t constSizeBytes;
        size_t localSizeBytes;
        int maxThreadsPerBlock;
        int numRegs;
        int ptxVersion;
        int binaryVersion;
        int cacheModeCA;
        int maxDynamicSharedSizeBytes;
        int preferredShmemCarveout;
        int clusterDimMustBeSet;
        int requiredClusterWidth;
        int requiredClusterHeight;
        int requiredClusterDepth;
        int clusterSchedulingPolicyPreference;
        int nonPortableClusterSizeAllowed;
        int reserved[16];
    } cudaFuncAttributes;

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
        union
        {
            cudaGraphConditionalNodeParams conditional;
        };
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

    /* =================================================================
     * Internal State Management
     * ================================================================= */

    typedef struct cudaCompatContext
    {
        int initialized;
        uint32_t flags;
        int profiler_initialized;
        int profiler_running;
    } cudaCompatContext_t;

    extern cudaCompatContext_t g_cuda_context;

#define CUDA_COMPAT_FORCE_LINK() \
    extern const int _cuda_compat_force_link; \
    static volatile const int *_cuda_compat_force_link_ptr __attribute__((used)) = &_cuda_compat_force_link;

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_TYPES_H */
