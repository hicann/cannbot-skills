/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3ddx_tiling.h
 * \brief 三维卷积输入梯度的 Host 规格与 Tiling 求解接口。
 *
 * Conv3DdxSpec 按逻辑 NCDHW 轴义记录形状，属性只保留 D/H/W 分量。
 * Conv3DdxTiling 保存分发信息；payload 中的 TilingData 单独传给 Kernel。
 */

#ifndef CONV3DDX_TILING_H
#define CONV3DDX_TILING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CONV3DDX_DTYPE_FP16 0u
#define CONV3DDX_DTYPE_BF16 1u
#define CONV3DDX_FMT_NCDHW 0u
#define CONV3DDX_FMT_NDHWC 1u

/* small-shape、standard、l1-resident 的 Host 策略编码。 */
#define CONV3DDX_TPL_SMALL_SHAPE 0u
#define CONV3DDX_TPL_INNER_PRODUCT 1u
#define CONV3DDX_TPL_FULL_LOAD 2u
#define CONV3DDX_OK 0

#define CONV3DDX_LOADB2_TRANSPOSE_AND_REVERSE 1u
#define CONV3DDX_LOADB2_REVERSE_ONLY 2u
#define CONV3DDX_LOADB2_TRANSPOSE_ONLY 3u
#define CONV3DDX_TILING_KEY(loadB2) (0x1000000u + (uint32_t)(loadB2))

/* 逻辑规格：dedx/dedy 几何按 NCDHW 语义记录；属性取空间分量（升维后统一为 3D 问题） */
struct Conv3DdxSpec {
    uint32_t n;            /* dedx batch（与 dedy batch 一致） */
    uint32_t cin;          /* dedx 通道（= filter CinG × groups） */
    uint32_t di, hi, wi;   /* dedx 空间几何 */
    uint32_t cout;         /* dedy 通道（= filter Cout） */
    uint32_t dout, ho, wo; /* dedy 空间几何 */
    uint32_t kd, kh, kw;   /* 核几何 */
    uint32_t sd, sh, sw;   /* 步长（空间分量） */
    uint32_t padDf, padDb; /* D 轴 pad (front, back) */
    uint32_t padHf, padHb; /* H 轴 pad */
    uint32_t padWf, padWb; /* W 轴 pad */
    uint32_t dd, dh, dw;   /* 空洞率（空间分量） */
    uint32_t groups;       /* 分组数 */
    uint32_t fmt;          /* dedy/dedx 物理格式，CONV3DDX_FMT_* */
    uint32_t filterDtype;  /* weight dtype，CONV3DDX_DTYPE_* */
    uint32_t dedyDtype;    /* dedy dtype，CONV3DDX_DTYPE_* */
};

/* Host 求解结果：分发信息与 TilingData 负载；Kernel 只读取 payload。 */
struct Conv3DdxTiling {
    /* ---- Host 分发信息 ---- */
    uint32_t blockDim;       /* 启动核数 */
    uint32_t tilingKey;      /* 五参数编码：2^24 + loadB2 */
    uint32_t templateId;     /* CONV3DDX_TPL_* */
    uint32_t loadB2;         /* CONV3DDX_LOADB2_* */
    uint32_t workspaceBytes; /* 设备 workspace 字节数（0 表示无） */
    uint32_t initOutputFlag; /* 1 = 需要输出预清零（DX 无贡献位语义） */
    uint32_t reserved[1];    /* 头部对齐保留，置 0 */
    /* ---- 不透明负载：算子实现自定义的共享字段区（≤ 1008 字节） ---- */
    uint8_t payload[1008];
};

/* 成功返回 CONV3DDX_OK；规格或资源不满足时返回负值。 */
int Conv3DdxSolveTiling(const struct Conv3DdxSpec* spec, struct Conv3DdxTiling* tiling);

#ifdef __cplusplus
}
#endif

#endif /* CONV3DDX_TILING_H */
