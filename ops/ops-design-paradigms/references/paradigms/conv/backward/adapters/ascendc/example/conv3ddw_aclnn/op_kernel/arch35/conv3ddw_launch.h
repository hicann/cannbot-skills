/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV3DDW_LAUNCH_H
#define CONV3DDW_LAUNCH_H

#include "conv3ddw_tiling_data.h"

enum {
    CONV3DDW_FP16 = 0,
    CONV3DDW_BF16 = 1
};
enum {
    CONV3DDW_NCDHW = 0,
    CONV3DDW_NDHWC = 1,
    CONV3DDW_DHWCN = 2
};

/* 规格已按逻辑 NCDHW 轴义归一；sd/sh/sw、dd/dh/dw 分别为步长和膨胀率。
 * x/dedy 类型与格式配对，输出始终 FP32；filter_size 的解析由调用适配完成。
 */
struct Conv3DdwSpec {
    uint32_t batch, cin, di, hi, wi, cout, dout, ho, wo;
    uint32_t dk, hk, wk, sd, sh, sw, dd, dh, dw;
    uint32_t padFront, padBack, padUp, padDown, padLeft, padRight;
    uint32_t groups, xFormat, yFormat, dtype;
};

struct Conv3DdwPlatformCaps {
    uint32_t aicNum, aivNum;
    uint64_t l0aBytes, l0bBytes, l0cBytes, l1Bytes;
};

#ifdef __cplusplus
extern "C" {
#endif

/* 返回 0 表示合法计划；失败时不发布可启动参数，并填写具体原因。 */
int Conv3DdwSolveTilingWithCaps(const Conv3DdwSpec* spec, const Conv3DdwPlatformCaps* caps, Conv3DdwTilingData* tiling,
                                char* error, uint32_t errorSize);
int Conv3DdwQueryPlatform(Conv3DdwPlatformCaps* caps);
int Conv3DdwSolveTiling(const Conv3DdwSpec* spec, Conv3DdwTilingData* tiling, char* error, uint32_t errorSize);

/* 每次 launch 内部先清零 dedw，再运行 MN_StreamK；两种 dtype 共用参数与 FP32 输出。 */
void LaunchConv3DdwFp16(uint32_t blocks, void* stream, void* x, void* filterSize, void* dedy, void* dedw,
                        void* workspace, void* tiling);
void LaunchConv3DdwBf16(uint32_t blocks, void* stream, void* x, void* filterSize, void* dedy, void* dedw,
                        void* workspace, void* tiling);

#ifdef __cplusplus
}
#endif
#endif
