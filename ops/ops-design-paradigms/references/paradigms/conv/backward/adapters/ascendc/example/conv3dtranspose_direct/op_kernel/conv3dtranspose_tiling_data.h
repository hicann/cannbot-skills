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
 * \file conv3dtranspose_tiling_data.h
 * \brief conv3dtranspose 直调工程 Host/Device 共享 TilingData（POD，随 launch 拷贝 GM 传入）
 *
 * 字段命名与单位遵循卷积反向范式：
 *   - generic/input/tiling/tiling.md §9（Host 传给 Kernel 的参数）
 *   - adapters/ascendc/tilingkey-templates.md §3（共享字段）
 * 生成协议：三种 Tiling 模板（small_shape / inner_product / full_load）为 Host 运行时计划，
 * Kernel 以同一套 baseM·baseN·baseK 与 singleCoreM·singleCoreCin·singleCoreDin·stepKa·stepKb 字段统一消费；
 * dtype、format、loadB2Condition 绑定编译实例；bias 存在性由 launch 的 bias 指针决定（hasBiasFlag
 * 不入 tilingKey，与内置算子一致），isBiasFullLoad 为运行时通路字段。
 *
 * 转置卷积差异（preprocessing §2.1/§4.2/§4.3/§5.3、tiling.md §5.1）：
 *   - output_padding（q）与显式目标形状余量（rho）由 Host 折入反向边界 backpropPad*（b⁺=Keff-1-p⁺+q+rho）
 *     与目标几何 di/hi/wi；Kernel 不感知 q/rho；
 *   - bias 通路字段：hasBiasFlag（与 launch bias 指针一致）、isBiasFullLoad（UpdateIsBiasFullLoad 在最终
 *     baseN/singleCoreCin 形成后判定）、biasAllocBytes（L1/BT 实际分配字节数）。
 *
 * 初始化协议（sync-implementation §4 + preprocessing §5.2）：主计算完整写回变体——needCompute 输出块由
 * 主计算经 Fixpipe 全量写回有效 M/N；无有效乘积的块（needCompute=false）无 bias 时显式写零、带 bias 时
 * 经 ComputeForBias（LoadL0Zero 全补零窗口 + 带 BT 的零乘积 MMAD + 正常 Fixpipe 写回）产生 bias 值输出
 * （scene §2.7：每个输出位置含 bias 语义，包括无有效乘积位置）；initOutputFlag 仍按 preprocessing §5.2
 * 转置条件由 Host 发布（信息性标志）。
 */

#ifndef CONV3DT_TILING_DATA_H
#define CONV3DT_TILING_DATA_H

#include <stdint.h>

/* 常量（arch35 / DAV_3510 平台绑定；单位：字节；与求解器平台查询一致性由求解器校验） */
constexpr uint32_t CONV3DT_C0 = 16u;        /* dedy 通道分形粒度 C0 */
constexpr uint32_t CONV3DT_FRACTAL = 16u;   /* m0/n0 分形粒度 */
constexpr uint32_t CONV3DT_ELEM_BYTES = 2u; /* fp16/bf16 输入元素宽度 */
constexpr uint32_t CONV3DT_BIAS_BYTES = 4u; /* fp32 bias 元素宽度 */
constexpr uint32_t CONV3DT_ACC_BYTES = 4u;  /* fp32 部分和宽度 */
constexpr uint32_t CONV3DT_L1_BYTES = 512u * 1024u;
constexpr uint32_t CONV3DT_L0A_BYTES = 64u * 1024u;
constexpr uint32_t CONV3DT_L0B_BYTES = 64u * 1024u;
constexpr uint32_t CONV3DT_L0C_BYTES = 256u * 1024u;
constexpr uint32_t CONV3DT_MAX_CORES = 36u; /* Ascend950 系 AIC 上限（运行时按平台查询收窄） */

/* 指令字段上限（8 位无符号 pad / 16 位 M/K 扩展） */
constexpr uint32_t CONV3DT_MAX_PAD = 255u;
constexpr uint32_t CONV3DT_MAX_DILATION = 255u;
constexpr uint32_t CONV3DT_MAX_STRIDE = 63u;

/* loadB2Condition（tilingkey-templates §2） */
constexpr uint32_t CONV3DT_TPL_TRANSPOSE_AND_REVERSE = 1u;
constexpr uint32_t CONV3DT_TPL_REVERSE_ONLY = 2u;
constexpr uint32_t CONV3DT_TPL_TRANSPOSE_ONLY = 3u;

/*!
 * Kernel 消费的共享 Tiling 负载：由 Conv3DtSolveTiling 求解后写入 Conv3DtTiling.payload，
 * 整体 H2D 拷贝后随 <<<blocks,nullptr,stream>>> 以 GM 指针传入 kernel。
 * 约束：保持 POD、4 字节自然对齐、总长 ≤ 992 字节（Conv3DtTiling.payload 容量；
 * 路由头 8×uint32=32B 之后为 payload）。
 */
struct Conv3DtTilingData {
    /* ---- 几何与属性（tiling.md §9.1） ---- */
    uint32_t batch;        /* N */
    uint32_t cin;          /* dedx 总通道（bias 长度） */
    uint32_t cout;         /* dedy 总通道 */
    uint32_t cinG;         /* 组内 dedx 通道 */
    uint32_t coutG;        /* 组内 dedy 通道 */
    uint32_t group;        /* G */
    uint32_t di, hi, wi;   /* dedx 空间（显式 input_size 目标形状，含 q 扩展） */
    uint32_t dout, ho, wo; /* dedy 空间 */
    uint32_t dk, hk, wk;   /* 核采样点数（q 不改变核采样点数与 K 长度） */
    uint32_t strideD, strideH, strideW;
    uint32_t dilationD, dilationH, dilationW;
    uint32_t padFront, padBack; /* D 轴正向 pad (front, back) */
    uint32_t padUp, padDown;    /* H 轴 */
    uint32_t padLeft, padRight; /* W 轴 */
    int32_t backpropPadUp;      /* 反向边界（有符号）；转置场景 b^- = KeffH-1-padUp */
    int32_t backpropPadDown;    /* b^+ = KeffH-1-padDown+qH+rhoH（q+rho 已折入） */
    int32_t backpropPadLeft;    /* b^- = KeffW-1-padLeft */
    int32_t backpropPadRight;   /* b^+ = KeffW-1-padRight+qW+rhoW */
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
    /* ---- 任务划分（compute-flow §6.1） ---- */
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
    /* ---- bias 通路（bias-bt-path.md；hasBiasFlag 不入 tilingKey） ---- */
    uint32_t hasBiasFlag;    /* 与 launch bias 指针一致（1=带 bias） */
    uint32_t isBiasFullLoad; /* 1=任务通道段全量装载，0=按当前 N 子块装载 */
    uint32_t biasAllocBytes; /* L1/BT 实际分配字节数 alignUp(biasUse*4,64) */
    /* ---- Buffer 份数与执行标志 ---- */
    uint32_t al1Pbuffer, bl1Pbuffer, al0Pbuffer, bl0Pbuffer, cl0Pbuffer;
    uint32_t initOutputFlag; /* preprocessing §5.2 转置条件判定（1=存在无贡献位置） */
    uint32_t enableFullLoad; /* l1-resident 置 1 */
    uint32_t iterateOrder;   /* 恒 1 */
    uint32_t fmtIsNdhwc;     /* dedy/dedx/weight 物理格式（0=NCDHW，1=NDHWC，运行时分支） */
};

static_assert(sizeof(Conv3DtTilingData) <= 992, "payload exceeds Conv3DtTiling capacity");

#endif /* CONV3DT_TILING_DATA_H */
