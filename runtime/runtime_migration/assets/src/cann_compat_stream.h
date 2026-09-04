/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */



#ifndef CUDA_COMPAT_STREAM_H
#define CUDA_COMPAT_STREAM_H

#include "cann_compat_types.h"
#include "cann_compat_event.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =================================================================
 * Stream Management
 * ================================================================= */


static inline cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
    aclError ret = aclrtCreateStream(pStream);
    return acl2cudaError(ret);
}


static inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
    (void)(flags);
    // least priority = 7 (lowest)
    aclError ret = aclrtCreateStreamWithConfig(pStream, 7, 0);
    return acl2cudaError(ret);
}



static inline cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream,
                                         unsigned int flags, int priority)
{
    (void)(flags);
    aclError ret = aclrtCreateStreamWithConfig(pStream, priority, 0);
    return acl2cudaError(ret);
}


static inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    aclError ret = aclrtDestroyStream(stream);
    return acl2cudaError(ret);
}


static inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    aclError ret = aclrtSynchronizeStream(stream);
    return acl2cudaError(ret);
}


static inline cudaError_t cudaStreamQuery(cudaStream_t stream) {
    aclrtStreamStatus status;
    aclError ret = aclrtStreamQuery(stream, &status);
    if (ret != ACL_SUCCESS) {
        return acl2cudaError(ret);
    }
    if (status == ACL_STREAM_STATUS_COMPLETE) {
        return cudaSuccess;
    }else if(status == ACL_STREAM_STATUS_NOT_READY){
        return cudaErrorNotReady;
    }
    return cudaErrorUnknown;
}


static inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream,
                                              cudaEvent_t event,
                                              unsigned int flags) {
    if ((flags & cudaEventWaitExternal) == cudaEventWaitExternal) {
        // CANN doesn't support external event waiting, return error
        return cudaErrorNotSupported;
    }
    aclError ret = aclrtStreamWaitEvent(stream, event);
    return acl2cudaError(ret);
}



static inline cudaError_t cudaStreamGetId(cudaStream_t hStream, unsigned long long *streamId) {
    int32_t cannStreamId;
    aclError ret = aclrtStreamGetId(hStream, &cannStreamId);
    if (ret == ACL_SUCCESS)
    {
        *streamId = (uint64_t)cannStreamId;
    }
    return acl2cudaError(ret);
}


static inline cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority)
{
    // Get stream priority from CANN
    uint32_t cannPriority;
    aclError ret = aclrtStreamGetPriority(hStream, &cannPriority);
    if (ret == ACL_SUCCESS)
    {
        *priority = (int)cannPriority;
    }
    return acl2cudaError(ret);
}


static inline cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags)
{
    aclError ret = aclrtStreamGetFlags(hStream, flags);
    return acl2cudaError(ret);
}

/* =================================================================
 * Stream Capture
 * ================================================================= */

static inline aclmdlRICaptureMode cudaCompatCaptureMode(cudaStreamCaptureMode mode)
{
    switch (mode) {
    case cudaStreamCaptureModeGlobal:
        return ACL_MODEL_RI_CAPTURE_MODE_GLOBAL;
    case cudaStreamCaptureModeThreadLocal:
        return ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL;
    case cudaStreamCaptureModeRelaxed:
        return ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
    default:
        return ACL_MODEL_RI_CAPTURE_MODE_GLOBAL;
    }
}

static inline cudaStreamCaptureStatus cudaCompatCaptureStatus(aclmdlRICaptureStatus status)
{
    switch (status) {
    case ACL_MODEL_RI_CAPTURE_STATUS_NONE:
        return cudaStreamCaptureStatusNone;
    case ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE:
        return cudaStreamCaptureStatusActive;
    case ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED:
        return cudaStreamCaptureStatusInvalidated;
    default:
        return cudaStreamCaptureStatusInvalidated;
    }
}

static inline cudaError_t cudaStreamBeginCapture(cudaStream_t stream,
#ifdef __cplusplus
                                                 cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal
#else
                                                 cudaStreamCaptureMode mode
#endif
)
{
    aclError ret = aclmdlRICaptureBegin(stream, cudaCompatCaptureMode(mode));
    if (ret == ACL_SUCCESS) {
        aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
        aclmdlRI modelRI = NULL;
        if (aclmdlRICaptureGetInfo(stream, &status, &modelRI) == ACL_SUCCESS &&
            status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
            cudaCompatRegisterGraphCaptureStream(modelRI, stream);
        }
    }
    return acl2cudaError(ret);
}



static inline cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph)
{
    aclError ret = aclmdlRICaptureEnd(stream, (aclmdlRI *)pGraph);
    if (ret == ACL_SUCCESS) {
        cudaCompatUnregisterGraphCaptureStream(stream);
    }
    return acl2cudaError(ret);
}

static inline cudaError_t cudaStreamBeginCaptureToGraph(cudaStream_t stream,
                                                        cudaGraph_t graph,
                                                        const cudaGraphNode_t *dependencies,
                                                        const cudaGraphEdgeData *dependencyData,
                                                        size_t numDependencies,
#ifdef __cplusplus
                                                        cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal
#else
                                                        cudaStreamCaptureMode mode
#endif
)
{
    (void)dependencies;
    (void)dependencyData;
    (void)numDependencies;
    if (!graph) {
        return cudaErrorInvalidValue;
    }
    /*
     * aclmdlRICaptureToModelRIBegin is documented by CANN as an experimental
     * Model RI API. It may change in future releases and is not intended for
     * production use.
     */
    aclError ret = aclmdlRICaptureToModelRIBegin(stream, graph, cudaCompatCaptureMode(mode));
    if (ret == ACL_SUCCESS) {
        cudaCompatRegisterGraphCaptureStream(graph, stream);
    }
    return acl2cudaError(ret);
}

static inline cudaError_t cudaCompatStreamGetCaptureInfo(cudaStream_t stream,
                                                         cudaStreamCaptureStatus *captureStatus_out,
                                                         unsigned long long *id_out,
                                                         cudaGraph_t *graph_out,
                                                         const cudaGraphNode_t **dependencies_out,
                                                         const cudaGraphEdgeData **edgeData_out,
                                                         size_t *numDependencies_out)
{
    if (!captureStatus_out) {
        return cudaErrorInvalidValue;
    }

    aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI modelRI = NULL;
    aclError ret = aclmdlRICaptureGetInfo(stream, &status, &modelRI);
    if (ret != ACL_SUCCESS) {
        return acl2cudaError(ret);
    }

    *captureStatus_out = cudaCompatCaptureStatus(status);
    if (id_out) {
        *id_out = 0;
    }
    if (graph_out) {
        *graph_out = modelRI;
    }
    if (status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
        cudaCompatRegisterGraphCaptureStream(modelRI, stream);
    }
    if (dependencies_out) {
        *dependencies_out = NULL;
    }
    if (edgeData_out) {
        *edgeData_out = NULL;
    }
    if (numDependencies_out) {
        *numDependencies_out = 0;
    }
    return cudaSuccess;
}

static inline cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream,
                                                   cudaStreamCaptureStatus *captureStatus_out,
                                                   unsigned long long *id_out,
                                                   cudaGraph_t *graph_out,
                                                   const cudaGraphNode_t **dependencies_out,
                                                   size_t *numDependencies_out)
{
    return cudaCompatStreamGetCaptureInfo(stream, captureStatus_out, id_out, graph_out,
                                          dependencies_out, NULL, numDependencies_out);
}

static inline cudaError_t cudaStreamGetCaptureInfo_v3(cudaStream_t stream,
                                                      cudaStreamCaptureStatus *captureStatus_out,
                                                      unsigned long long *id_out,
                                                      cudaGraph_t *graph_out,
                                                      const cudaGraphNode_t **dependencies_out,
                                                      const cudaGraphEdgeData **edgeData_out,
                                                      size_t *numDependencies_out)
{
    return cudaCompatStreamGetCaptureInfo(stream, captureStatus_out, id_out, graph_out,
                                          dependencies_out, edgeData_out, numDependencies_out);
}



static inline cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus *pCaptureStatus)
{
    if (!pCaptureStatus) {
        return cudaErrorInvalidValue;
    }
    aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI modelRI = NULL;
    aclError ret = aclmdlRICaptureGetInfo(stream, &status, &modelRI);
    if (ret == ACL_SUCCESS) {
        *pCaptureStatus = cudaCompatCaptureStatus(status);
    }
    return acl2cudaError(ret);
}


static inline cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode *pMode)
{
    if (!pMode)
    {
        return cudaErrorInvalidValue;
    }

    if (*pMode != cudaStreamCaptureModeGlobal &&
        *pMode != cudaStreamCaptureModeThreadLocal &&
        *pMode != cudaStreamCaptureModeRelaxed) {
        return cudaErrorInvalidValue;
    }
    aclmdlRICaptureMode cannMode = cudaCompatCaptureMode(*pMode);

    // Exchange mode with CANN
    aclError ret = aclmdlRICaptureThreadExchangeMode(&cannMode);
    if (ret != ACL_SUCCESS)
    {
        return acl2cudaError(ret);
    }

    // Map previous mode back to CUDA mode
    switch (cannMode)
    {
    case ACL_MODEL_RI_CAPTURE_MODE_GLOBAL:
        *pMode = cudaStreamCaptureModeGlobal;
        break;
    case ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL:
        *pMode = cudaStreamCaptureModeThreadLocal;
        break;
    case ACL_MODEL_RI_CAPTURE_MODE_RELAXED:
        *pMode = cudaStreamCaptureModeRelaxed;
        break;
    default:
        *pMode = cudaStreamCaptureModeRelaxed; // Default fallback
        break;
    }

    return cudaSuccess;
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_STREAM_H */
