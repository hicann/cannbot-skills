/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */

#ifndef CUDA_COMPAT_GRAPH_H
#define CUDA_COMPAT_GRAPH_H

#include "cann_compat_types.h"
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C"
{
#endif

    __attribute__((weak)) aclError aclrtGetFuncBySymbol(
        const void *symbol, aclrtFuncHandle *funcHandle);

    static inline cudaError_t cudaDeviceGetGraphMemAttribute(
        int device, cudaGraphMemAttributeType attr, void *value)
    {
        if (!value) {
            return cudaErrorInvalidValue;
        }
        if (attr < cudaGraphMemAttrUsedMemCurrent ||
            attr > cudaGraphMemAttrReservedMemHigh) {
            return cudaErrorInvalidValue;
        }

        enum { CUDA_COMPAT_MEM_USAGE_COMPONENTS = 64 };
        aclrtMemUsageInfo usage[CUDA_COMPAT_MEM_USAGE_COMPONENTS];
        memset(usage, 0, sizeof(usage));
        size_t outputCount = 0;
        aclError ret = aclrtGetMemUsageInfo(device, usage,
                                            CUDA_COMPAT_MEM_USAGE_COMPONENTS,
                                            &outputCount);
        if (ret != ACL_SUCCESS) {
            return acl2cudaError(ret);
        }
        if (outputCount > CUDA_COMPAT_MEM_USAGE_COMPONENTS) {
            outputCount = CUDA_COMPAT_MEM_USAGE_COMPONENTS;
        }

        uint64_t currentBytes = 0;
        uint64_t peakBytes = 0;
        for (size_t i = 0; i < outputCount; ++i) {
            if (UINT64_MAX - currentBytes < usage[i].curMemSize ||
                UINT64_MAX - peakBytes < usage[i].memPeakSize) {
                return cudaErrorInvalidValue;
            }
            currentBytes += usage[i].curMemSize;
            peakBytes += usage[i].memPeakSize;
        }
        *(uint64_t *)value =
            (attr == cudaGraphMemAttrUsedMemHigh ||
             attr == cudaGraphMemAttrReservedMemHigh) ? peakBytes : currentBytes;
        return cudaSuccess;
    }


    static inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec,
                                                   cudaGraph_t graph,
                                                   void *pErrorNode,
                                                   char *pLogBuffer,
                                                   size_t bufferSize)
    {
        (void)pErrorNode;
        (void)pLogBuffer;
        (void)bufferSize;
        if (!pGraphExec || !graph) {
            return cudaErrorInvalidValue;
        }
        *pGraphExec = graph;
        return cudaSuccess;
    }

    static inline cudaError_t cudaGraphCreate(cudaGraph_t *pGraph,
                                              unsigned int flags)
    {
        if (!pGraph) {
            return cudaErrorInvalidValue;
        }
        aclmdlRI graph = NULL;
        aclError ret = aclmdlRIBuildBegin(&graph, flags);
        if (ret != ACL_SUCCESS) {
            *pGraph = NULL;
            return acl2cudaError(ret);
        }
        *pGraph = graph;
        return cudaSuccess;
    }


    static inline cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec,
                                                            cudaGraph_t graph,
                                                            unsigned long long flags)
    {
        (void)graph;
        (void)flags;
        if (!pGraphExec) {
            return cudaErrorInvalidValue;
        }
        *pGraphExec = NULL;
        return cudaErrorNotSupported;
    }


    static inline cudaError_t cudaGraphAddNode_v1(cudaGraphNode_t *pGraphNode,
                                                  cudaGraph_t graph,
                                                  const cudaGraphNode_t *dependencies,
                                                  size_t numDependencies,
                                                  cudaGraphNodeParams *nodeParams)
    {
        (void)graph;
        (void)dependencies;
        (void)numDependencies;
        (void)nodeParams;
        if (!pGraphNode) {
            return cudaErrorInvalidValue;
        }
        *pGraphNode = NULL;
        return cudaErrorNotSupported;
    }


    static inline cudaError_t cudaGraphAddNode_v2(cudaGraphNode_t *pGraphNode,
                                                  cudaGraph_t graph,
                                                  const cudaGraphNode_t *dependencies,
                                                  const cudaGraphEdgeData *dependencyData,
                                                  size_t numDependencies)
    {
        (void)graph;
        (void)dependencies;
        (void)dependencyData;
        (void)numDependencies;
        if (!pGraphNode) {
            return cudaErrorInvalidValue;
        }
        *pGraphNode = NULL;
        return cudaErrorNotSupported;
    }

    static inline cudaError_t cudaGraphAddNode_v3(cudaGraphNode_t *pGraphNode,
                                                  cudaGraph_t graph,
                                                  const cudaGraphNode_t *dependencies,
                                                  const cudaGraphEdgeData *dependencyData,
                                                  size_t numDependencies,
                                                  cudaGraphNodeParams *nodeParams)
    {
        (void)dependencyData;
        return cudaGraphAddNode_v1(pGraphNode, graph, dependencies, numDependencies, nodeParams);
    }

#define CUDA_COMPAT_GRAPH_ADD_NODE_SELECT(_1, _2, _3, _4, _5, _6, NAME, ...) NAME
#define cudaGraphAddNode(...) \
    CUDA_COMPAT_GRAPH_ADD_NODE_SELECT(__VA_ARGS__, cudaGraphAddNode_v3, cudaGraphAddNode_v1)(__VA_ARGS__)


    static inline cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node,
                                                           cudaGraphNode_t *dependencies,
                                                           size_t *numDependencies)
    {
        (void)node;
        (void)dependencies;
        if (!numDependencies) {
            return cudaErrorInvalidValue;
        }
        *numDependencies = 0;
        return cudaErrorNotSupported;
    }

    static inline cudaError_t cudaGraphConditionalHandleCreate(
        cudaGraphConditionalHandle *pHandle,
        cudaGraph_t graph
#ifdef __cplusplus
        ,
        unsigned int defaultLaunchValue = 0,
        unsigned int flags = 0
#else
        ,
        unsigned int defaultLaunchValue,
        unsigned int flags
#endif
    )
    {
        if (!pHandle || !graph) {
            return cudaErrorInvalidValue;
        }
        /*
         * aclmdlRICondHandleCreate is documented by CANN as an experimental
         * Model RI API. It may change in future releases and is not intended
         * for production use.
         */
        aclmdlRICondHandleFlag cannFlag = ((flags & cudaGraphCondAssignDefault) != 0) ?
            ACL_MODEL_RI_COND_HANDLE_ASSIGN_DEFAULT : (aclmdlRICondHandleFlag)0;
        aclError ret = aclmdlRICondHandleCreate(graph, defaultLaunchValue, cannFlag, pHandle);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaGraphConditionalHandleCreate_v2(
        cudaGraphConditionalHandle *pHandle,
        cudaGraph_t graph,
        cudaExecutionContext_t ctx
#ifdef __cplusplus
        ,
        unsigned int defaultLaunchValue = 0,
        unsigned int flags = 0
#else
        ,
        unsigned int defaultLaunchValue,
        unsigned int flags
#endif
    )
    {
        if (ctx != NULL) {
            aclrtContext current = NULL;
            aclError ret = aclrtGetCurrentContext(&current);
            if (ret != ACL_SUCCESS) {
                return acl2cudaError(ret);
            }
            if (current != ctx) {
                return cudaErrorInvalidValue;
            }
        }
        return cudaGraphConditionalHandleCreate(pHandle, graph,
                                                defaultLaunchValue, flags);
    }

    static inline cudaError_t cudaCompatGraphGetStreams(cudaGraph_t graph,
                                                        aclrtStream **streams,
                                                        uint32_t *numStreams)
    {
        aclError ret = aclmdlRIGetStreams(graph, NULL, numStreams);
        if (ret != ACL_SUCCESS) {
            return acl2cudaError(ret);
        }
        if (*numStreams == 0) {
            *streams = NULL;
            return cudaSuccess;
        }

        *streams = (aclrtStream *)malloc((size_t)(*numStreams) * sizeof(aclrtStream));
        if (!*streams) {
            return cudaErrorMemoryAllocation;
        }
        uint32_t streamCapacity = *numStreams;
        ret = aclmdlRIGetStreams(graph, *streams, &streamCapacity);
        if (ret != ACL_SUCCESS) {
            free(*streams);
            *streams = NULL;
            return acl2cudaError(ret);
        }
        *numStreams = streamCapacity;
        return cudaSuccess;
    }

    static inline cudaError_t cudaCompatGraphCopyStreamTasks(
        aclrtStream stream, cudaGraphNode_t *nodes, size_t capacity,
        size_t *written, size_t *actualTotal)
    {
        uint32_t taskCount = 0;
        aclError ret = aclmdlRIGetTasksByStream(stream, NULL, &taskCount);
        if (ret != ACL_SUCCESS) {
            return acl2cudaError(ret);
        }
        *actualTotal += taskCount;
        if (!nodes || *written >= capacity || taskCount == 0) {
            return cudaSuccess;
        }

        aclmdlRITask *tasks = (aclmdlRITask *)malloc((size_t)taskCount * sizeof(aclmdlRITask));
        if (!tasks) {
            return cudaErrorMemoryAllocation;
        }
        uint32_t taskCapacity = taskCount;
        ret = aclmdlRIGetTasksByStream(stream, tasks, &taskCapacity);
        if (ret != ACL_SUCCESS) {
            free(tasks);
            return acl2cudaError(ret);
        }
        for (uint32_t j = 0; j < taskCapacity && *written < capacity; ++j) {
            nodes[*written] = (cudaGraphNode_t)tasks[j];
            ++(*written);
        }
        free(tasks);
        return cudaSuccess;
    }

    static inline cudaError_t cudaGraphGetNodes(cudaGraph_t graph,
                                                cudaGraphNode_t *nodes,
                                                size_t *numNodes)
    {
        if (!graph || !numNodes) {
            return cudaErrorInvalidValue;
        }

        aclrtStream *streams = NULL;
        uint32_t numStreams = 0;
        /*
         * aclmdlRIGetStreams and aclmdlRIGetTasksByStream are experimental
         * Model RI query APIs in CANN and are not intended for production use.
         */
        cudaError_t ret = cudaCompatGraphGetStreams(graph, &streams, &numStreams);
        if (ret != cudaSuccess) {
            return ret;
        }

        size_t capacity = nodes ? *numNodes : 0;
        size_t actualTotal = 0;
        size_t written = 0;
        for (uint32_t i = 0; i < numStreams; ++i) {
            ret = cudaCompatGraphCopyStreamTasks(streams[i], nodes, capacity,
                                                 &written, &actualTotal);
            if (ret != cudaSuccess) {
                free(streams);
                return ret;
            }
        }

        free(streams);

        *numNodes = actualTotal;
        return cudaSuccess;
    }


    static inline cudaError_t cudaCompatGraphTaskType(
        aclmdlRITaskType cannType, cudaGraphNodeType *cudaType)
    {
        if (!cudaType) {
            return cudaErrorInvalidValue;
        }
        switch (cannType) {
            case ACL_MODEL_RI_TASK_DEFAULT:
                *cudaType = cudaGraphNodeTypeEmpty;
                return cudaSuccess;
            case ACL_MODEL_RI_TASK_KERNEL:
                *cudaType = cudaGraphNodeTypeKernel;
                return cudaSuccess;
            case ACL_MODEL_RI_TASK_EVENT_RECORD:
                *cudaType = cudaGraphNodeTypeEventRecord;
                return cudaSuccess;
            case ACL_MODEL_RI_TASK_EVENT_WAIT:
                *cudaType = cudaGraphNodeTypeWaitEvent;
                return cudaSuccess;
            case ACL_MODEL_RI_TASK_VALUE_WRITE:
            case ACL_MODEL_RI_TASK_VALUE_WAIT:
                *cudaType = cudaGraphNodeTypeBatchMemOp;
                return cudaSuccess;
            default:
                return cudaErrorNotSupported;
        }
    }


    static inline cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node,
                                                   cudaGraphNodeType *pType)
    {
        if (!node || !pType) {
            return cudaErrorInvalidValue;
        }
        aclmdlRITaskType cannType = ACL_MODEL_RI_TASK_DEFAULT;
        aclError ret = aclmdlRITaskGetType((aclmdlRITask)node, &cannType);
        if (ret != ACL_SUCCESS) {
            return acl2cudaError(ret);
        }
        return cudaCompatGraphTaskType(cannType, pType);
    }


    static inline cudaError_t cudaCompatGraphSetEvent(
        cudaGraphNode_t node, cudaEvent_t event, aclmdlRITaskType expectedType)
    {
        if (!node || !event) {
            return cudaErrorInvalidValue;
        }
        aclmdlRITaskParams params;
        memset(&params, 0, sizeof(params));
        aclError ret = aclmdlRITaskGetParams((aclmdlRITask)node, &params);
        if (ret != ACL_SUCCESS) {
            return acl2cudaError(ret);
        }
        if (params.type != expectedType) {
            return cudaErrorInvalidValue;
        }
        if (expectedType == ACL_MODEL_RI_TASK_EVENT_RECORD) {
            params.eventRecordTaskParams.event = event;
        } else {
            params.eventWaitTaskParams.event = event;
        }
        ret = aclmdlRITaskSetParams((aclmdlRITask)node, &params);
        return acl2cudaError(ret);
    }


    static inline uint32_t cudaCompatGraphGridBlocks(dim3 gridDim)
    {
        uint64_t blocks = (uint64_t)gridDim.x * (uint64_t)gridDim.y *
                          (uint64_t)gridDim.z;
        return (blocks == 0 || blocks > UINT32_MAX) ? 0 : (uint32_t)blocks;
    }


    static inline cudaError_t cudaCompatGraphSetKernel(
        cudaGraphNode_t node, const void *func, dim3 gridDim, dim3 blockDim,
        unsigned int sharedMemBytes, void **kernelParams, void **extra)
    {
        if (!node || !func) {
            return cudaErrorInvalidValue;
        }
        if ((uintptr_t)func < 4096U) {
            return cudaErrorInvalidDeviceFunction;
        }
        if (blockDim.x == 0 || blockDim.y == 0 || blockDim.z == 0) {
            return cudaErrorInvalidConfiguration;
        }
        uint32_t numBlocks = cudaCompatGraphGridBlocks(gridDim);
        if (numBlocks == 0) {
            return cudaErrorInvalidConfiguration;
        }
        /* CANN's task setter cannot safely infer CUDA argument-array layouts. */
        if (sharedMemBytes != 0 || kernelParams != NULL || extra != NULL) {
            return cudaErrorNotSupported;
        }

        aclmdlRITaskParams params;
        memset(&params, 0, sizeof(params));
        aclError ret = aclmdlRITaskGetParams((aclmdlRITask)node, &params);
        if (ret != ACL_SUCCESS) {
            return acl2cudaError(ret);
        }
        if (params.type != ACL_MODEL_RI_TASK_KERNEL) {
            return cudaErrorInvalidValue;
        }

        aclrtFuncHandle handle = NULL;
        if (aclrtGetFuncBySymbol != NULL) {
            if (aclrtGetFuncBySymbol(func, &handle) != ACL_SUCCESS) {
                handle = NULL;
            }
        }
        /* A captured task already holds a validated CANN function handle.
         * Do not pass an unresolved CUDA symbol to CANN as a handle. */
        if (handle == NULL) {
            if ((aclrtFuncHandle)func != params.kernelTaskParams.funcHandle) {
                return cudaErrorNotSupported;
            }
            handle = params.kernelTaskParams.funcHandle;
        }
        params.kernelTaskParams.funcHandle = handle;
        params.kernelTaskParams.numBlocks = numBlocks;
        ret = aclmdlRITaskSetParams((aclmdlRITask)node, &params);
        return acl2cudaError(ret);
    }


    static inline const void *cudaCompatGraphV2Function(
        const cudaKernelNodeParamsV2 *params)
    {
        switch (params->functionType) {
            case cudaKernelFunctionTypeUnspecified:
            case cudaKernelFunctionTypeDeviceEntry:
                return params->func;
            case cudaKernelFunctionTypeKernel:
                return params->kern;
            case cudaKernelFunctionTypeFunction:
                return params->cuFunc;
            default:
                return NULL;
        }
    }


    static inline cudaError_t cudaGraphNodeSetParams(
        cudaGraphNode_t node, cudaGraphNodeParams *nodeParams)
    {
        if (!node || !nodeParams) {
            return cudaErrorInvalidValue;
        }
        switch (nodeParams->type) {
            case cudaGraphNodeTypeKernel:
                if (nodeParams->kernel.ctx != NULL) {
                    return cudaErrorNotSupported;
                }
                return cudaCompatGraphSetKernel(
                    node, cudaCompatGraphV2Function(&nodeParams->kernel),
                    nodeParams->kernel.gridDim, nodeParams->kernel.blockDim,
                    nodeParams->kernel.sharedMemBytes,
                    nodeParams->kernel.kernelParams, nodeParams->kernel.extra);
            case cudaGraphNodeTypeEventRecord:
                if (nodeParams->eventRecord.ctx != NULL) {
                    return cudaErrorNotSupported;
                }
                return cudaCompatGraphSetEvent(node, nodeParams->eventRecord.event,
                                               ACL_MODEL_RI_TASK_EVENT_RECORD);
            case cudaGraphNodeTypeWaitEvent:
                if (nodeParams->eventWait.ctx != NULL) {
                    return cudaErrorNotSupported;
                }
                return cudaCompatGraphSetEvent(node, nodeParams->eventWait.event,
                                               ACL_MODEL_RI_TASK_EVENT_WAIT);
            default:
                return cudaErrorNotSupported;
        }
    }

    static inline cudaError_t cudaGraphSetConditional(cudaGraphConditionalHandle handle,
                                                      unsigned int value)
    {
        if (!handle) {
            return cudaErrorInvalidValue;
        }
        uint64_t *condPtr = NULL;
        /*
         * aclmdlRICondHandleGetCondPtr is documented by CANN as experimental.
         * The compatibility wrapper uses it to obtain the condition value
         * address, then writes the value with aclrtValueWrite.
         */
        aclError ret = aclmdlRICondHandleGetCondPtr(handle, &condPtr);
        if (ret != ACL_SUCCESS) {
            return acl2cudaError(ret);
        }
        ret = aclrtValueWrite(condPtr, (uint64_t)value, 0, NULL);
        return acl2cudaError(ret);
    }

    static inline cudaError_t cudaCompatGraphConditionalHandleGetCondPtr(
        cudaGraphConditionalHandle handle, uint64_t **ptr)
    {
        if (!handle || !ptr) {
            return cudaErrorInvalidValue;
        }
        aclError ret = aclmdlRICondHandleGetCondPtr(handle, ptr);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph,
                                                     const char *path,
                                                     unsigned int flags)
    {
        if (!graph || !path) {
            return cudaErrorInvalidValue;
        }
        aclError ret = aclmdlRIDebugJsonPrint(graph, path, flags);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaCompatGraphCommitExec(cudaGraphExec_t graphExec,
                                                        cudaError_t setResult)
    {
        if (setResult != cudaSuccess) {
            return setResult;
        }
        aclError ret = aclmdlRIUpdate(graphExec);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaGraphExecEventRecordNodeSetEvent(
        cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaEvent_t event)
    {
        if (!graphExec) {
            return cudaErrorInvalidValue;
        }
        return cudaCompatGraphCommitExec(
            graphExec,
            cudaCompatGraphSetEvent(node, event,
                                    ACL_MODEL_RI_TASK_EVENT_RECORD));
    }


    static inline cudaError_t cudaGraphExecEventWaitNodeSetEvent(
        cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaEvent_t event)
    {
        if (!graphExec) {
            return cudaErrorInvalidValue;
        }
        return cudaCompatGraphCommitExec(
            graphExec,
            cudaCompatGraphSetEvent(node, event,
                                    ACL_MODEL_RI_TASK_EVENT_WAIT));
    }


    static inline cudaError_t cudaGraphExecKernelNodeSetParams(
        cudaGraphExec_t graphExec, cudaGraphNode_t node,
        const cudaKernelNodeParams *params)
    {
        if (!graphExec || !params) {
            return cudaErrorInvalidValue;
        }
        return cudaCompatGraphCommitExec(
            graphExec,
            cudaCompatGraphSetKernel(node, params->func, params->gridDim,
                                     params->blockDim, params->sharedMemBytes,
                                     params->kernelParams, params->extra));
    }


    static inline cudaError_t cudaGraphExecNodeSetParams(
        cudaGraphExec_t graphExec, cudaGraphNode_t node,
        cudaGraphNodeParams *nodeParams)
    {
        if (!graphExec) {
            return cudaErrorInvalidValue;
        }
        return cudaCompatGraphCommitExec(
            graphExec, cudaGraphNodeSetParams(node, nodeParams));
    }


    static inline cudaError_t cudaGraphExecUpdate(
        cudaGraphExec_t graphExec, cudaGraph_t graph,
        cudaGraphExecUpdateResultInfo *resultInfo)
    {
        if (resultInfo) {
            memset(resultInfo, 0, sizeof(*resultInfo));
            resultInfo->result = cudaGraphExecUpdateError;
        }
        if (!graphExec || !graph) {
            return cudaErrorInvalidValue;
        }
        if (graphExec != graph) {
            if (resultInfo) {
                resultInfo->result = cudaGraphExecUpdateErrorNotSupported;
            }
            return cudaErrorGraphExecUpdateFailure;
        }
        aclError ret = aclmdlRIUpdate(graphExec);
        if (ret != ACL_SUCCESS) {
            return acl2cudaError(ret);
        }
        if (resultInfo) {
            resultInfo->result = cudaGraphExecUpdateSuccess;
        }
        return cudaSuccess;
    }


    static inline cudaError_t cudaGraphNodeSetEnabled(
        cudaGraphExec_t graphExec, cudaGraphNode_t node,
        unsigned int isEnabled)
    {
        if (!graphExec || !node || isEnabled > 1) {
            return cudaErrorInvalidValue;
        }
        if (isEnabled != 0) {
            return cudaErrorNotSupported;
        }
        aclError ret = aclmdlRITaskDisable((aclmdlRITask)node);
        if (ret != ACL_SUCCESS) {
            return acl2cudaError(ret);
        }
        ret = aclmdlRIUpdate(graphExec);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaGraphExecGetId(cudaGraphExec_t graphExec,
                                                 unsigned int *graphId)
    {
        if (!graphExec || !graphId) {
            return cudaErrorInvalidValue;
        }
        aclError ret = aclmdlRIGetId(graphExec, graphId);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec,
                                              cudaStream_t stream)
    {
        if (!graphExec) {
            return cudaErrorInvalidValue;
        }
        aclError ret = aclmdlRIExecuteAsync(graphExec, stream);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec)
    {
        if (!graphExec) {
            return cudaErrorInvalidValue;
        }
        aclError ret = aclmdlRIDestroy(graphExec);
        return acl2cudaError(ret);
    }


    static inline cudaError_t cudaGraphDestroy(cudaGraph_t graph)
    {
        if (!graph) {
            return cudaErrorInvalidValue;
        }
        aclError ret = aclmdlRIDestroy(graph);
        return acl2cudaError(ret);
    }

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_GRAPH_H */
