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
 * \file conv3ddx_tiling_data.h
 * \brief conv3ddx 直调工程 Host/Device 共享 TilingData（POD，随 launch 拷贝 GM 传入）
 *
 * 字段命名与单位遵循卷积反向范式：
 *   - generic/input/tiling/tiling.md §9（Host 传给 Kernel 的参数）
 *   - adapters/ascendc/input/tilingkey-templates.md §3（共享字段）
 * 生成协议：三种 Tiling 模板（small_shape / inner_product / full_load）为 Host 运行时计划，
 * Kernel 以同一套 baseM·baseN·baseK 与 singleCoreM·singleCoreCin·singleCoreDin·stepKa·stepKb 字段统一消费；
 * dtype、format、loadB2Condition 绑定编译实例。
 *
 * 初始化协议（preprocessing §5.2）：主计算完整写回变体——
 * needCompute 输出块由主计算经 Fixpipe 全量写回有效 M/N；无有效乘积的块（needCompute=false）
 * 由 Kernel 显式写零覆盖本块有效区域，等价于“可为整个 dedx 省去独立预清零”的合法形态；
 * initOutputFlag 仍按 preprocessing §5.2 规则由 Host 发布（信息性标志，20 例设计推演均为 0）。
 */

#ifndef CONV3DDX_TILING_DATA_H
#define CONV3DDX_TILING_DATA_H

#include <stdint.h>

/* 常量（arch35 / DAV_3510 平台绑定；单位：字节；与求解器平台查询一致性由求解器校验） */
constexpr uint32_t CONV3DDX_C0 = 16u;        /* dedy 通道分形粒度 C0 */
constexpr uint32_t CONV3DDX_FRACTAL = 16u;   /* m0/n0 分形粒度 */
constexpr uint32_t CONV3DDX_ELEM_BYTES = 2u; /* fp16/bf16 输入元素宽度 */
constexpr uint32_t CONV3DDX_ACC_BYTES = 4u;  /* fp32 部分和宽度 */
constexpr uint32_t CONV3DDX_L1_BYTES = 512u * 1024u;
constexpr uint32_t CONV3DDX_L0A_BYTES = 64u * 1024u;
constexpr uint32_t CONV3DDX_L0B_BYTES = 64u * 1024u;
constexpr uint32_t CONV3DDX_L0C_BYTES = 256u * 1024u;
constexpr uint32_t CONV3DDX_MAX_CORES = 36u; /* Ascend950 系 AIC 上限（运行时按平台查询收窄） */

/* 指令字段上限（8 位无符号 pad / 16 位 M/K 扩展） */
constexpr uint32_t CONV3DDX_MAX_PAD = 255u;
constexpr uint32_t CONV3DDX_MAX_DILATION = 255u;
constexpr uint32_t CONV3DDX_MAX_STRIDE = 63u;

/* loadB2Condition（tilingkey-templates §2） */
constexpr uint32_t CONV3DDX_TPL_TRANSPOSE_AND_REVERSE = 1u;
constexpr uint32_t CONV3DDX_TPL_REVERSE_ONLY = 2u;
constexpr uint32_t CONV3DDX_TPL_TRANSPOSE_ONLY = 3u;

/*!
 * Kernel 消费的共享 Tiling 负载：由 Conv3DdxSolveTiling 求解后写入 Conv3DdxTiling.payload，
 * 整体 H2D 拷贝后随 <<<blocks,nullptr,stream>>> 以 GM 指针传入 kernel。
 * 约束：保持 POD、4 字节自然对齐、总长 ≤ 1008 字节（Conv3DdxTiling.payload 容量）。
 */
struct Conv3DdxTilingData {
    /* ---- 几何与属性（generic/input/tiling/tiling.md §9.1） ---- */
    uint32_t batch;        /* N */
    uint32_t cin;          /* dedx 总通道 */
    uint32_t cout;         /* dedy 总通道 */
    uint32_t cinG;         /* 组内 dedx 通道 */
    uint32_t coutG;        /* 组内 dedy 通道 */
    uint32_t group;        /* G */
    uint32_t di, hi, wi;   /* dedx 空间 */
    uint32_t dout, ho, wo; /* dedy 空间 */
    uint32_t dk, hk, wk;   /* 核采样点数 */
    uint32_t strideD, strideH, strideW;
    uint32_t dilationD, dilationH, dilationW;
    uint32_t padFront, padBack; /* D 轴正向 pad (front, back) */
    uint32_t padUp, padDown;    /* H 轴 */
    uint32_t padLeft, padRight; /* W 轴 */
    int32_t backpropPadUp;      /* 反向边界（有符号，b^- = KeffH-1-padUp） */
    int32_t backpropPadDown;    /* b^+ = KeffH-1-padDown + rhoH */
    int32_t backpropPadLeft;    /* b^- = KeffW-1-padLeft */
    int32_t backpropPadRight;   /* b^+ = KeffW-1-padRight + rhoW */
    /* ---- 派生量 ---- */
    uint32_t c0;         /* 分形粒度，恒 16 */
    uint32_t mValue;     /* alignUp(hi*wi, m0) */
    uint32_t nValue;     /* alignUp(cinG, n0) */
    uint32_t kValue;     /* alignUp(coutG,C0)*hk*wk */
    uint32_t kIter;      /* ceilDiv(kValue, baseK) */
    uint32_t tailK;      /* kValue - (kIter-1)*baseK */
    uint32_t hkWk;       /* hk*wk */
    uint32_t dkHkWk;     /* 核体积 dk*hk*wk */
    uint32_t khDilation; /* KeffH = (hk-1)*dilationH+1 */
    uint32_t kwDilation; /* KeffW = (wk-1)*dilationW+1 */
    uint32_t hoExpand;   /* (ho-1)*strideH+1 */
    uint32_t woExpand;   /* (wo-1)*strideW+1 */
    uint32_t doHoWo;     /* dout*ho*wo（dedy 通道步幅） */
    uint32_t hiWi;       /* hi*wi */
    uint32_t diHiWi;     /* di*hi*wi */
    /* ---- 计算块（L0 基本块） ---- */
    uint32_t baseM, baseN, baseK;
    /* ---- 任务划分（generic/input/compute-flow §6.1） ---- */
    uint32_t coreNum; /* 启动计算核数（= blockDim） */
    uint32_t singleCoreM;
    uint32_t singleCoreCin;
    uint32_t singleCoreDin;
    uint32_t singleCoreCout;  /* 恒 = coutG */
    uint32_t singleCoreBatch; /* 恒 1 */
    uint32_t singleCoreGroup; /* 恒 1 */
    uint32_t singleIterateDk; /* 恒 = dk */
    uint64_t totalTasks;      /* batch*group*dCount*mCount*nCount */
    /* ---- L1 驻留 ---- */
    uint32_t stepKa, stepKb;
    uint32_t aSlotBytes; /* 单份 L1A 槽字节数（32B 对齐） */
    uint32_t bSlotBytes; /* 单份 L1B 槽字节数（32B 对齐） */
    /* ---- Buffer 份数与执行标志 ---- */
    uint32_t al1Pbuffer, bl1Pbuffer, al0Pbuffer, bl0Pbuffer, cl0Pbuffer;
    uint32_t initOutputFlag; /* preprocessing §5.2 判定（1=存在无贡献位置） */
    uint32_t enableFullLoad; /* l1-resident 置 1 */
    uint32_t iterateOrder;   /* 恒 1 */
    uint32_t fmtIsNdhwc;     /* dedy/dedx/weight 物理格式（0=NCDHW，1=NDHWC，运行时分支） */
};

static_assert(sizeof(Conv3DdxTilingData) <= 1008, "payload exceeds Conv3DdxTiling capacity");

#endif /* CONV3DDX_TILING_DATA_H */
