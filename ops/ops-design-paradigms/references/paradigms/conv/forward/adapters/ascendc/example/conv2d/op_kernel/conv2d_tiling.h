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
 * \file conv2d_tiling.h
 * \brief Conv2D Host 侧 Tiling 计算（泛化实现）
 *
 * 推导顺序（tiling.md §0 固定前步约束后步）：
 *   ① 指令预检 → ② 输出/MNK → ③ 多核切分 → ④ L0/L1 切分 → ⑤ fullLoad 判定
 *
 * 泛化能力（覆盖范式验证范围 verification.md §2.1）：
 *   - 1D/2D/3D（1D 升维 H=1, 3D 含 D 轴外层循环 + D 向软件填零）
 *   - 任意 stride (1~63) / dilation (1~255) / 非对称 pad (< kernel)
 *   - fullLoad 状态：BL1 驻留（含 ALL 退化态）/ NONE 流式，B 侧驻留自动判定
 *   - K 侧 L1 分块（kChunk），支持大通道 / 大核场景
 *   - 多核 batch × dout × M 三维绑核（逐维贪心 + CeilDiv 兜底；范式完整三步法见 tiling.md §2.3）
 */

#ifndef CONV2D_TILING_H
#define CONV2D_TILING_H

#include "conv2d_tiling_data.h"

inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

inline uint32_t AlignB(uint32_t a, uint32_t b)
{
    return CeilDiv(a, b) * b;
}

inline uint32_t FloorAlign(uint32_t a, uint32_t b)
{
    return a / b * b;
}

/// 输出尺寸：Ho = (Hi + padH + padT - dkH) / sH + 1（dkH 为膨胀后核尺寸 = (Kh-1)·dH+1）
inline uint32_t CalcOutSize(uint32_t inSize, uint32_t padHead, uint32_t padTail, uint32_t dilKernel, uint32_t stride)
{
    const uint32_t dk = dilKernel;
    return (inSize + padHead + padTail >= dk) ? (inSize + padHead + padTail - dk) / stride + 1 : 0;
}

/// 膨胀后核尺寸：dkH = (Kh-1)·dH + 1
inline uint32_t CalcDk(uint32_t kernel, uint32_t dilation)
{
    return (kernel - 1) * dilation + 1;
}

/// 最坏情形行推理（tiling.md §3.3）：中间列起块比行对齐多跨最多 1 行。
/// 仅用作 L1 预算上界；kernel 侧逐块精确装载按下边界钳制（compute-flow.md §6.3）
inline uint32_t CalcHiLoadWorst(uint32_t mBlk, uint32_t wo, uint32_t sh, uint32_t dkH, uint32_t hi)
{
    const uint32_t hoSpanMax = CeilDiv(mBlk + wo - 1, wo);
    uint32_t hiLoad = (hoSpanMax - 1) * sh + dkH;
    return (hiLoad < hi) ? hiLoad : hi;
}

/**
 * \brief 计算 Conv2D Tiling 参数（泛化版）
 * \return 0 成功；1 空输出；-1 超出能力范围（非法输入/指令超限或 L1 下限不可行）
 */
inline int CalcConv2DTiling(const Conv2DSpec& inSpec, Conv2DTilingData& td)
{
    td.spec = inSpec;

    // ── 0. 输入防御 + 指令限制预检（compute-flow.md §4.1 约束表）──
    if (inSpec.sh == 0 || inSpec.sw == 0 || inSpec.sd == 0) {
        return -1; // stride 必须 ≥ 1（0 会使输出公式除零）
    }
    if (inSpec.sh > LOAD3D_MAX_STRIDE || inSpec.sw > LOAD3D_MAX_STRIDE || inSpec.sd > LOAD3D_MAX_STRIDE ||
        inSpec.dh > LOAD3D_MAX_DILATION || inSpec.dw > LOAD3D_MAX_DILATION || inSpec.dd > LOAD3D_MAX_DILATION ||
        inSpec.padT > LOAD3D_MAX_PAD || inSpec.padB > LOAD3D_MAX_PAD || inSpec.padL > LOAD3D_MAX_PAD ||
        inSpec.padR > LOAD3D_MAX_PAD) {
        return -1;
    }
    const uint32_t dkH = CalcDk(inSpec.kh, inSpec.dh);
    const uint32_t dkW = CalcDk(inSpec.kw, inSpec.dw);
    const uint32_t dkD = CalcDk(inSpec.kd, inSpec.dd);
    if (dkH > LOAD3D_MAX_KERNEL || dkW > LOAD3D_MAX_KERNEL || dkD > LOAD3D_MAX_KERNEL) {
        return -1;
    }

    // ── ① 输出尺寸与 M/N/K（compute-flow.md §1.2 + language.md §4）──
    //     CalcOutSize 使用膨胀后核尺寸 dkD/dkH/dkW（传入原始核会算错 dilation 场景）
    td.dout = CalcOutSize(inSpec.di, inSpec.padDH, inSpec.padDT, dkD, inSpec.sd);
    td.ho = CalcOutSize(inSpec.hi, inSpec.padT, inSpec.padB, dkH, inSpec.sh);
    td.wo = CalcOutSize(inSpec.wi, inSpec.padL, inSpec.padR, dkW, inSpec.sw);
    td.ciAlign = AlignB(inSpec.ci, FRACTAL_K);
    td.coAlign = AlignB(inSpec.co, FRACTAL_N);
    td.k = td.ciAlign * inSpec.kh * inSpec.kw;
    td.m = td.ho * td.wo;

    if (td.dout == 0 || td.ho == 0 || td.wo == 0) {
        td.emptyOut = 1;
        td.blockDim = 1;
        td.singleCoreM = td.m > 0 ? td.m : 1;
        td.mBlk = td.singleCoreM;
        td.nBlk = td.coAlign;
        td.kChunk = td.k;
        td.kL0 = td.k;
        td.hiLoadL1 = 1;
        td.al1Size = 128;
        td.bl1Size = 128;
        td.bResident = 0;
        return 1; // 空输出：合法，调用方跳过执行
    }
    td.emptyOut = 0;

    // ── ② 多核切分：batch × dout × M 三维绑核（tiling.md §2.1）──
    //     简化实现：do → batch → M 逐维贪心绑定剩余核数，CeilDiv 兜底；
    //     范式完整三步法（绑满 → 因子整除 → 代价模型）见 tiling.md §2.3
    td.doDim = (td.dout < AIC_CORE_NUM) ? td.dout : AIC_CORE_NUM;
    uint32_t restCores = AIC_CORE_NUM / td.doDim;
    td.batchDim = (inSpec.n < restCores) ? inSpec.n : restCores;
    restCores /= (td.batchDim > 0) ? td.batchDim : 1;
    td.mDim = (td.m < restCores) ? (td.m > 0 ? td.m : 1) : restCores;

    td.blockDim = td.batchDim * td.doDim * td.mDim;
    td.singleCoreBatch = CeilDiv(inSpec.n, td.batchDim);
    td.singleCoreDo = CeilDiv(td.dout, td.doDim);
    td.singleCoreM = CeilDiv(td.m, td.mDim);

    // ── ③ L0 切分（tiling.md §3.2：容量不等式下取最大块）──
    //     mBlk 搜索：L0C 上限向下按 M0 步进（末步允许 < M0，MMAD 的 m 自动 AlignUp）
    uint32_t mCapL0C = FloorAlign(L0C_SIZE_BYTE / 4 / td.coAlign, FRACTAL_M); // fp32 累加 ×4B
    if (mCapL0C > 2048) {
        mCapL0C = 2048; // LOAD3D 16bit 起点限制
    }
    td.mBlk = (td.singleCoreM < mCapL0C) ? td.singleCoreM : mCapL0C;
    if (td.mBlk == 0) {
        td.mBlk = FRACTAL_M;
    }

    // 16bit 指令限制：LOAD3D L1 图积 Hi_L1×Wi ≤ 32767（同时覆盖 DN2NZ nValue/dstNzC0Stride 16bit）。
    // 超限时收缩 mBlk（行带变窄）；收到最小仍超限 → 超出本实现能力
    // （范式口径：切 HW 模式或软件搬运，见 tiling.md §1.4/§3.5，本参考实现不覆盖）
    while (td.mBlk > FRACTAL_M &&
           CalcHiLoadWorst(td.mBlk, td.wo, inSpec.sh, dkH, inSpec.hi) * inSpec.wi > LOAD3D_MAX_HIWI) {
        td.mBlk /= 2;
    }
    if (CalcHiLoadWorst(td.mBlk, td.wo, inSpec.sh, dkH, inSpec.hi) * inSpec.wi > LOAD3D_MAX_HIWI) {
        return -1;
    }

    // nBlk：Co^align 全载优先，超 L1 预算时缩半；上界受 BT 段容量约束（bias-bt-path.md §5：
    // N 块超 BT 容量时分段装载由 kernel 的 nBlk 循环承接，故 nBlk ≤ BT_MAX_NBLK 即可）
    td.nBlk = (td.coAlign < BT_MAX_NBLK) ? td.coAlign : BT_MAX_NBLK;
    td.kChunk = td.k;

    // kL0：L0A/L0B 容量允许的最大 K0 倍数
    const uint32_t mAlign = AlignB(td.mBlk, FRACTAL_M);
    const uint32_t kCapL0A = FloorAlign(L0A_SIZE_BYTE / (mAlign * sizeof(uint16_t)), FRACTAL_K);
    const uint32_t kCapL0B = FloorAlign(L0B_SIZE_BYTE / (AlignB(td.nBlk, FRACTAL_N) * sizeof(uint16_t)), FRACTAL_K);
    uint32_t kCap = (kCapL0A < kCapL0B) ? kCapL0A : kCapL0B;
    td.kL0 = AlignB((td.kChunk < kCap) ? td.kChunk : kCap, FRACTAL_K);
    if (td.kL0 < FRACTAL_K) {
        td.kL0 = FRACTAL_K;
    }

    // ── ④ L1 切分 + ⑤ fullLoad 判定（tiling.md §3.3）──
    //     迭代收缩直到 A+B ≤ L1 预算
    td.hiLoadL1 = CalcHiLoadWorst(td.mBlk, td.wo, inSpec.sh, dkH, inSpec.hi);
    const uint32_t biasL1 = AlignB(inSpec.co, FRACTAL_N) * sizeof(uint16_t);
    const uint32_t budget = L1_SIZE_BYTE - L1_MARGIN_BYTE - biasL1;

    uint32_t guard = 0;
    while (guard++ < 256) {
        const uint32_t cinA = td.kChunk / (inSpec.kh * inSpec.kw); // A 侧通道分形
        const uint32_t aL1 = cinA * td.hiLoadL1 * inSpec.wi * sizeof(uint16_t);
        const uint32_t bL1 = td.kChunk * td.nBlk * sizeof(uint16_t);
        td.al1Size = AlignB(aL1, 128);
        td.bl1Size = AlignB(bL1, 128);

        if (td.al1Size + td.bl1Size <= budget) {
            break; // 可行
        }

        // 分级退化（tiling.md §3.5，代价从小到大）
        if (td.nBlk > FRACTAL_N) {
            // 级 0a：缩 N 块
            td.nBlk = (td.nBlk > 2 * FRACTAL_N) ? FloorAlign(td.nBlk / 2, FRACTAL_N) : FRACTAL_N;
        } else if (td.kChunk > FRACTAL_K * inSpec.kh * inSpec.kw) {
            // 级 0b：缩 K 分块（cinA 变小 → A 侧变小；B 侧重装次数增加）
            const uint32_t kUnit = FRACTAL_K * inSpec.kh * inSpec.kw;
            td.kChunk = (td.kChunk > 2 * kUnit) ? FloorAlign(td.kChunk / 2, kUnit) : kUnit;
            td.hiLoadL1 = CalcHiLoadWorst(td.mBlk, td.wo, inSpec.sh, dkH, inSpec.hi);
        } else {
            // L1 下限仍超预算 → 超出能力（tiling.md §3.5 不可行判定）
            return -1;
        }
    }
    if (guard >= 256) {
        return -1;
    }

    // fullLoad 判定（tiling.md §3.3 四态的参考实现裁剪：BL1 驻留（含 ALL 退化态）/ NONE）
    const uint32_t bFullL1 = td.k * td.coAlign * sizeof(uint16_t);

    td.bResident = 0;

    // BL1 驻留：weight 全量可容纳，且 A 侧最坏行带（全通道）同时容纳
    // （仅 2D/1D；3D 须按 kd 切片逐片装载，见 compute-flow.md §2.2。
    //   coAlign 受 BT 段容量约束：驻留态 nBlk=coAlign 须 ≤ BT_MAX_NBLK，
    //   否则保持 NONE 流式（nBlk 循环分段），功能不变仅少一次驻留收益）
    if (inSpec.kd == 1 && td.coAlign <= BT_MAX_NBLK &&
        bFullL1 + td.ciAlign * td.hiLoadL1 * inSpec.wi * sizeof(uint16_t) <= budget) {
        td.bResident = 1;
        td.nBlk = td.coAlign;
        td.kChunk = td.k;
    }

    // 驻留恢复全量后按最终参数重算 L1 尺寸（防沿用收缩循环的旧值导致 buffer 分配过小）
    {
        const uint32_t cinA = td.kChunk / (inSpec.kh * inSpec.kw);
        const uint32_t aL1 = cinA * td.hiLoadL1 * inSpec.wi * sizeof(uint16_t);
        const uint32_t bL1 = td.kChunk * td.nBlk * sizeof(uint16_t);
        td.al1Size = AlignB(aL1, 128);
        td.bl1Size = AlignB(bL1, 128);
    }

    // kL0 在 kChunk 基础上重算（kChunk 可能因退化缩小）
    td.kL0 = AlignB((td.kChunk < kCap) ? td.kChunk : kCap, FRACTAL_K);
    if (td.kL0 < FRACTAL_K) {
        td.kL0 = FRACTAL_K;
    }

    return 0;
}

#endif // CONV2D_TILING_H
