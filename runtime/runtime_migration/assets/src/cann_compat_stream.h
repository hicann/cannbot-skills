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
    // least priority = 7 (lowest)
    aclError ret = aclrtCreateStreamWithConfig(pStream, 7, 0);
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


static inline cudaError_t cudaStreamBeginCapture(cudaStream_t stream)
{
    // Use GLOBAL mode default
    aclError ret = aclmdlRICaptureBegin(stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL);
    return acl2cudaError(ret);
}



static inline cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph)
{
    aclError ret = aclmdlRICaptureEnd(stream, (aclmdlRI *)pGraph);
    return acl2cudaError(ret);
}



static inline cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus *pCaptureStatus)
{
    aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI modelRI;
    aclError ret = aclmdlRICaptureGetInfo(stream, &status, &modelRI);
    *pCaptureStatus = (cudaStreamCaptureStatus)status;
    return acl2cudaError(ret);
}


static inline cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode *pMode)
{
    if (!pMode)
    {
        return cudaErrorInvalidValue;
    }

    aclmdlRICaptureMode cannMode;
    // Map CUDA mode to CANN mode
    switch (*pMode)
    {
    case cudaStreamCaptureModeGlobal:
        cannMode = ACL_MODEL_RI_CAPTURE_MODE_GLOBAL;
        break;
    case cudaStreamCaptureModeThreadLocal:
        cannMode = ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL;
        break;
    case cudaStreamCaptureModeRelaxed:
        cannMode = ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
        break;
    default:
        return cudaErrorInvalidValue;
    }

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