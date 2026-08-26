/*
 * Copyright (C) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 */



#ifndef CUDA_COMPAT_EVENT_H
#define CUDA_COMPAT_EVENT_H

#include "cann_compat_types.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((weak)) aclError aclrtRecordEventWithFlag(aclrtEvent event, aclrtStream stream, uint32_t flag);

/* =================================================================
 * Event Management
 * ================================================================= */


static inline cudaError_t cudaEventCreate(cudaEvent_t *event) {
    aclError ret = aclrtCreateEvent(event);
    return acl2cudaError(ret);
}


static inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    uint32_t cannFlags = ACL_EVENT_SYNC;
    if ((flags & cudaEventDisableTiming) == 0) {
        cannFlags |= ACL_EVENT_TIME_LINE;
    }
    if ((flags & cudaEventInterprocess) > 0) {
        cannFlags = ACL_EVENT_IPC; // 不支持 ACL_EVENT_IPC 与其他flag 相或
    }
    aclError ret = aclrtCreateEventExWithFlag(event, cannFlags);
    return acl2cudaError(ret);
}


static inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
    aclError ret = aclrtDestroyEvent(event);
    return acl2cudaError(ret);
}


static inline cudaError_t cudaEventRecord(cudaEvent_t event,
                                          cudaStream_t stream) {
    aclError ret = aclrtRecordEvent(event, stream);
    return acl2cudaError(ret);
}

static inline cudaError_t cudaEventRecordWithFlags(cudaEvent_t event,
                                                   cudaStream_t stream,
                                                   unsigned int flags)
{
    uint32_t cannFlags = ((flags & cudaEventRecordExternal) != 0) ?
        ACL_EVENT_RECORD_EXTERNAL : ACL_EVENT_RECORD_DEFAULT;

    if (aclrtRecordEventWithFlag) {
        aclError ret = aclrtRecordEventWithFlag(event, stream, cannFlags);
        return acl2cudaError(ret);
    }

    if (cannFlags != ACL_EVENT_RECORD_DEFAULT) {
        return cudaErrorNotSupported;
    }
    aclError ret = aclrtRecordEvent(event, stream);
    return acl2cudaError(ret);
}

static inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    aclError ret = aclrtSynchronizeEvent(event);
    return acl2cudaError(ret);
}


static inline cudaError_t cudaEventQuery(cudaEvent_t event) {
    aclrtEventRecordedStatus status;
    aclError ret = aclrtQueryEventStatus(event, &status);
    if (ret != ACL_SUCCESS) {
        return acl2cudaError(ret);
    }
    if (status == ACL_EVENT_RECORDED_STATUS_COMPLETE) {
        return cudaSuccess;
    } else if (status == ACL_EVENT_RECORDED_STATUS_NOT_READY) {
        return cudaErrorNotReady;
    }
    return cudaErrorUnknown;
}


static inline cudaError_t cudaEventElapsedTime(float *ms,
                                                cudaEvent_t start,
                                                cudaEvent_t end) {
    aclError ret = aclrtEventElapsedTime(ms, start, end);
    return acl2cudaError(ret);
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_COMPAT_EVENT_H */
