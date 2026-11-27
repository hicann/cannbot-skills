/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file conv3dtranspose_tiling.h
 * \brief 三维转置卷积的 Host 规格与 Tiling 求解接口。
 *
 * Conv3DtSpec 按逻辑 NCDHW 轴义记录形状，属性只保留 D/H/W 分量。
 * Conv3DtTiling 保存分发信息；payload 中的 TilingData 单独传给 Kernel。
 */

#ifndef CONV3DT_TILING_H
#define CONV3DT_TILING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CONV3DT_DTYPE_FP16 0u
#define CONV3DT_DTYPE_BF16 1u
#define CONV3DT_FMT_NCDHW 0u
#define CONV3DT_FMT_NDHWC 1u

/* small-shape、standard、l1-resident 的 Host 策略编码。 */
#define CONV3DT_TPL_SMALL_SHAPE 0u
#define CONV3DT_TPL_INNER_PRODUCT 1u
#define CONV3DT_TPL_FULL_LOAD 2u
#define CONV3DT_OK 0

#define CONV3DT_LOADB2_TRANSPOSE_AND_REVERSE 1u
#define CONV3DT_LOADB2_REVERSE_ONLY 2u
#define CONV3DT_LOADB2_TRANSPOSE_ONLY 3u
#define CONV3DT_TILING_KEY(loadB2) (0x1000000u + (uint32_t)(loadB2))

/* 逻辑规格：y/dedy(x) 几何按 NCDHW 语义记录；属性取空间分量（升维后统一为 3D 问题）。
 * 对象绑定（scene §1）：x→dedy(N,Cout,Dout,Ho,Wo)、filter→weight(Cout,CinG,Dk,Hk,Wk)、
 * y→dedx(N,Cin,Di,Hi,Wi)、bias(Cin,)。 */
struct Conv3DtSpec {
    uint32_t n;            /* y batch（与 x batch 一致） */
    uint32_t cin;          /* y 通道（= filter CinG × groups，bias 长度） */
    uint32_t di, hi, wi;   /* y 空间几何（显式 input_size 目标形状） */
    uint32_t cout;         /* x 通道（= filter Cout） */
    uint32_t dout, ho, wo; /* x 空间几何 */
    uint32_t kd, kh, kw;   /* 核几何 */
    uint32_t sd, sh, sw;   /* 步长（空间分量） */
    uint32_t padDf, padDb; /* D 轴 pad (front, back) */
    uint32_t padHf, padHb; /* H 轴 pad */
    uint32_t padWf, padWb; /* W 轴 pad */
    uint32_t dd, dh, dw;   /* 空洞率（空间分量） */
    uint32_t qd, qh, qw;   /* output_padding（空间分量；q>=0 且 q<stride or q<dilation） */
    uint32_t groups;       /* 分组数 */
    uint32_t fmt;          /* x/y 物理格式，CONV3DT_FMT_* */
    uint32_t filterDtype;  /* weight dtype，CONV3DT_DTYPE_* */
    uint32_t dedyDtype;    /* x dtype，CONV3DT_DTYPE_* */
    uint32_t hasBias;      /* 0 = 无 bias；1 = 有 bias（biasElems 必须等于 cin） */
    uint32_t biasElems;    /* bias 元素数，有 bias 时等于 cin */
};

/* Host 求解结果：分发信息与 TilingData 负载；Kernel 只读取 payload。 */
struct Conv3DtTiling {
    /* ---- Host 分发信息 ---- */
    uint32_t blockDim;       /* 启动核数 */
    uint32_t tilingKey;      /* 五参数编码：2^24 + loadB2（(loadB2,0,0,true,0) 实例） */
    uint32_t templateId;     /* CONV3DT_TPL_* */
    uint32_t loadB2;         /* CONV3DT_LOADB2_* */
    uint32_t workspaceBytes; /* 设备 workspace 字节数（0 表示无） */
    uint32_t initOutputFlag; /* 1 = 需要输出预初始化（转置无贡献位写 bias 值语义） */
    uint32_t hasBiasFlag;    /* 与 spec.hasBias 一致 */
    uint32_t reserved[1];    /* 头部对齐保留，置 0 */
    /* ---- 不透明负载：算子实现自定义的共享字段区（≤ 992 字节） ---- */
    uint8_t payload[992];
};

/* 成功返回 CONV3DT_OK；规格或资源不满足时返回负值。 */
int Conv3DtSolveTiling(const struct Conv3DtSpec* spec, struct Conv3DtTiling* tiling);

#ifdef __cplusplus
}
#endif

#endif /* CONV3DT_TILING_H */
