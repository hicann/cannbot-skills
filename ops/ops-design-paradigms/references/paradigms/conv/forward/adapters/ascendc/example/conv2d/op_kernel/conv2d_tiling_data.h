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
 * \file conv2d_tiling_data.h
 * \brief Conv2D TilingData（POD，Host 与 Kernel 共享）
 *
 * 支持范围（范式 verification.md §2.1 验收标准）：
 *   空间维 1D/2D/3D（升维处理）、fp16、NCHW/NCDHW/NCW、pad<kernel、groups=1、
 *   stride 1~63、dilation 1~255。
 */

#ifndef CONV2D_TILING_DATA_H
#define CONV2D_TILING_DATA_H

#include <cstdint>

constexpr uint32_t FRACTAL_M = 16;
constexpr uint32_t FRACTAL_N = 16;
constexpr uint32_t FRACTAL_K = 16;

constexpr uint32_t L1_SIZE_BYTE = 512 * 1024;
constexpr uint32_t L0A_SIZE_BYTE = 64 * 1024;
constexpr uint32_t L0B_SIZE_BYTE = 64 * 1024;
constexpr uint32_t L0C_SIZE_BYTE = 256 * 1024;
constexpr uint32_t AIC_CORE_NUM = 32;
constexpr uint32_t BT_SIZE_BYTE = 4 * 1024;

/// L1 预留余量：覆盖 bias 驻留与对齐碎片（tiling.md §0 产出三件套之一）
constexpr uint32_t L1_MARGIN_BYTE = 4 * 1024;

/// 指令限制（compute-flow.md §4.1 约束汇总表）
constexpr uint32_t LOAD3D_MAX_STRIDE = 63;
constexpr uint32_t LOAD3D_MAX_DILATION = 255;
constexpr uint32_t LOAD3D_MAX_PAD = 255;
constexpr uint32_t LOAD3D_MAX_KERNEL = 511;
/// LOAD3D L1 图 15bit 积上限：Hi_L1×Wi ≤ 32767（Hi_L1 ≥ 1 时同时覆盖 DN2NZ nValue/dstNzC0Stride 的 16bit 上限）
constexpr uint32_t LOAD3D_MAX_HIWI = 32767;

/// BT 段 N 上限：BT 容量按 L0C 类型（4B）折算（bias-bt-path.md §5）
constexpr uint32_t BT_MAX_NBLK = BT_SIZE_BYTE / 4;

struct Conv2DSpec {
    uint32_t n, ci, di, hi, wi;                    // 输入
    uint32_t co, kd, kh, kw;                       // 卷积核
    uint32_t sd, sh, sw;                           // 步长
    uint32_t padDH, padDT, padT, padB, padL, padR; // 填充（逐维头尾）
    uint32_t dd, dh, dw;                           // 空洞率
};

struct Conv2DTilingData {
    Conv2DSpec spec;

    uint32_t dout, ho, wo; // 输出尺寸
    uint32_t m;            // M = Ho × Wo（Do 独立外层循环）
    uint32_t k; // 单 kd 切片 K = Ci^align × Kh × Kw；3D 总归约 = Kd × k（kernel 按 kd 切片循环）
    uint32_t ciAlign, coAlign;

    // 核间切分（tiling.md §2）
    uint32_t batchDim, mDim, doDim; // 绑核维数（batch × dout × M）
    uint32_t blockDim;              // 总核数
    uint32_t singleCoreM;           // 每核 M 规模
    uint32_t singleCoreBatch;
    uint32_t singleCoreDo;

    // 核内切分（tiling.md §3）
    uint32_t mBlk;             // M 块大小（跨行 M 块，可 < 对齐值）
    uint32_t nBlk;             // N 的 L1 块（Co^align 取切分后的值，受 BT 段容量上限约束）
    uint32_t kChunk;           // K 的 L1 分块
    uint32_t kL0;              // K 的 L0 步进
    uint32_t hiLoadL1;         // A 侧 L1 单次装载的输入行数（最坏情形行推理）
    uint32_t al1Size, bl1Size; // A/B 侧 L1 实际分配字节数

    // 全载状态（tiling.md §3.3；参考实现裁剪为 BL1 驻留/NONE 两态，A 侧驻留为范式扩展点）
    uint32_t bResident; // 1 = weight 驻留 L1（跨块复用；含 ALL 退化态）

    uint32_t emptyOut; // 1 = 空输出（合法，跳过执行）
};

#endif // CONV2D_TILING_DATA_H
