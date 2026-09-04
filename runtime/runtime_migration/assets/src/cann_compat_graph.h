/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */

#ifndef CUDA_COMPAT_GRAPH_H
#define CUDA_COMPAT_GRAPH_H

#include "cann_compat_types.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif

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


    static inline cudaError_t cudaCompatGraphCondType(cudaGraphConditionalNodeType cudaType,
                                                      aclmdlRICondTaskType *cannType)
    {
        if (!cannType) {
            return cudaErrorInvalidValue;
        }
        switch (cudaType) {
        case cudaGraphCondTypeIf:
            *cannType = ACL_MODEL_RI_COND_TYPE_IF;
            return cudaSuccess;
        case cudaGraphCondTypeWhile:
            *cannType = ACL_MODEL_RI_COND_TYPE_WHILE;
            return cudaSuccess;
        case cudaGraphCondTypeSwitch:
            *cannType = ACL_MODEL_RI_COND_TYPE_SWITCH;
            return cudaSuccess;
        default:
            return cudaErrorInvalidValue;
        }
    }

    static inline cudaError_t cudaGraphAddNode(cudaGraphNode_t *pGraphNode,
                                               cudaGraph_t graph,
                                               const cudaGraphNode_t *dependencies,
                                               size_t numDependencies,
                                               cudaGraphNodeParams *nodeParams)
    {
        (void)dependencies;
        if (!pGraphNode || !graph || !nodeParams) {
            return cudaErrorInvalidValue;
        }
        *pGraphNode = NULL;
        if (numDependencies != 0) {
            return cudaErrorNotSupported;
        }
        if (nodeParams->type != cudaGraphNodeTypeConditional) {
            return cudaErrorNotSupported;
        }
        if (!nodeParams->conditional.handle || nodeParams->conditional.size == 0) {
            return cudaErrorInvalidValue;
        }

        cudaGraph_t *subGraphs = (cudaGraph_t *)calloc(nodeParams->conditional.size, sizeof(cudaGraph_t));
        if (!subGraphs) {
            return cudaErrorMemoryAllocation;
        }

        aclmdlRICondTaskType cannType = ACL_MODEL_RI_COND_TYPE_IF;
        cudaError_t typeRet = cudaCompatGraphCondType(nodeParams->conditional.type, &cannType);
        if (typeRet != cudaSuccess) {
            free(subGraphs);
            return typeRet;
        }

        cudaStream_t stream = cudaCompatFindGraphCaptureStream(graph);
        if (!stream) {
            free(subGraphs);
            return cudaErrorNotSupported;
        }

        aclmdlRICondTaskParams params;
        params.handle = nodeParams->conditional.handle;
        params.type = cannType;
        params.size = nodeParams->conditional.size;
        params.modelRIArray = (aclmdlRI *)subGraphs;
        aclError ret = aclmdlRIAddCondTask(params, stream, 0);
        if (ret != ACL_SUCCESS) {
            free(subGraphs);
            return acl2cudaError(ret);
        }

        nodeParams->conditional.phGraph_out = subGraphs;
        *pGraphNode = (cudaGraphNode_t)nodeParams->conditional.handle;
        return cudaSuccess;
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
