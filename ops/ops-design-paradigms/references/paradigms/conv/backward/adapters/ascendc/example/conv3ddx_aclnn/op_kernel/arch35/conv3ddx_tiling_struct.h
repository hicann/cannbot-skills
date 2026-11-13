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
 * \file conv3ddx_tiling_struct.h
 * \brief Conv3ddxTilingData 共享结构体（Host/Device 同一份定义）
 *
 * 唯一事实源：design/TilingData.md「TilingData C++ 结构体定义」（§1）逐字段转写——
 * 字段集合、顺序、类型与注释与该文档一致（无缺失、无新增字段）；语义/单位/取值域见
 * 该文档 §2「字段表」。Host 侧填充（design/HostTiling.md §7）与 Kernel 侧消费
 * （design/Kernel.md「Kernel 入口」经 GET_TILING_DATA_WITH_STRUCT 读取）共用本定义。
 *
 * 反向边界注释口径：HostTiling.md §3.3 派生量表为 b⁻/b⁺ 公式的数学权威——
 * b⁺ = Keff − 1 − pad⁺ + ρ（ρ 为尾部余数 (L_in + p⁻ + p⁺ − Keff) mod s）；TilingData.md
 * §1 行内注释把尾项写为 (stride−1)，是 ρ = s−1 的交付谱系特例（20 例逐轴满足），一般
 * 合法域（不对称 pad 等）以 ρ 形式为准（HostTiling.md §3.3 跨文档同步点登记；设计收口
 * 时同步修订 TilingData.md 行内注释，不改字段布局/类型/语义域）。
 */

#ifndef CONV3DDX_TILING_STRUCT_H
#define CONV3DDX_TILING_STRUCT_H

#include <stdint.h>

/*
 * arch35 / DAV_3510 平台绑定参考值（单位：字节）。权威值以 TilingFunc 运行时
 * platformInfo 查询为准，Host 求解前后与这组参考值交叉核对（不一致以查询值为准并记录）。
 */
constexpr uint32_t CONV3DDX_C0 = 16u;        /* dedy 通道分形粒度 C0（恒 16） */
constexpr uint32_t CONV3DDX_FRACTAL = 16u;   /* m0/n0 分形粒度 */
constexpr uint32_t CONV3DDX_ELEM_BYTES = 2u; /* fp16/bf16 输入元素宽度（本交付无 fp32） */
constexpr uint32_t CONV3DDX_ACC_BYTES = 4u;  /* fp32 部分和宽度 */
constexpr uint32_t CONV3DDX_L1_BYTES = 512u * 1024u;
constexpr uint32_t CONV3DDX_L0A_BYTES = 64u * 1024u;
constexpr uint32_t CONV3DDX_L0B_BYTES = 64u * 1024u;
constexpr uint32_t CONV3DDX_L0C_BYTES = 256u * 1024u;
constexpr uint32_t CONV3DDX_MAX_CORES = 36u; /* Ascend950 系 AIC 上限（运行时按平台查询收窄） */

/* 指令字段上限（8 位无符号 pad / 搬运扩展） */
constexpr uint32_t CONV3DDX_MAX_PAD = 255u;
constexpr uint32_t CONV3DDX_MAX_DILATION = 255u;
constexpr uint32_t CONV3DDX_MAX_STRIDE = 63u;

/*
 * Kernel 消费的共享 Tiling 负载：Host 求解（三模板优先序 + loadB2 判定）后填充，
 * 经注册发布整体传入 kernel。约束：保持 POD、4 字节自然对齐、无未初始化槽位
 * （Host 填充前对整结构 memset 清零，尾部不预留保留字段）。
 */
struct Conv3ddxTilingData {
    /* ---- 几何与属性（规格归一化后装载；generic/tiling §9.1） ---- */
    uint32_t batch;                     /* N（批次） */
    uint32_t cin;                       /* dedx 总通道（= cinG * group） */
    uint32_t cout;                      /* dedy 总通道（= filter 的 Cout 维） */
    uint32_t cinG;                      /* 组内 dedx 通道（= cin / group；本交付 group=1 时等于 cin） */
    uint32_t coutG;                     /* 组内 dedy 通道（= cout / group） */
    uint32_t group;                     /* 分组数 G（本交付用例恒 1，域 [1,65535]） */
    uint32_t di, hi, wi;                /* dedx 空间长度（D/H/W，NCDHW 语义；NDHWC 由 data_format 归一） */
    uint32_t dout, ho, wo;              /* dedy 空间长度 */
    uint32_t dk, hk, wk;                /* 核采样点数（filter 的 kD/kH/kW） */
    uint32_t strideD, strideH, strideW; /* 空间步长（属性空间分量） */
    uint32_t dilationD, dilationH, dilationW; /* 空间膨胀率 */
    uint32_t padFront, padBack;               /* D 轴正向 pad（front/back） */
    uint32_t padUp, padDown;                  /* H 轴正向 pad（top/bottom） */
    uint32_t padLeft, padRight;               /* W 轴正向 pad（left/right） */
    int32_t backpropPadUp;    /* H 轴反向下边界 b^- = (hk-1)*dilationH + 1 - padUp（有符号，可负） */
    int32_t backpropPadDown;  /* H 轴反向上边界 b^+ = (hk-1)*dilationH + 1 - padDown + (strideH-1) */
    int32_t backpropPadLeft;  /* W 轴反向下边界 b^- = (wk-1)*dilationW + 1 - padLeft */
    int32_t backpropPadRight; /* W 轴反向上边界 b^+ = (wk-1)*dilationW + 1 - padRight + (strideW-1) */
    /* ---- 派生量（generic/tiling §3） ---- */
    uint32_t c0;         /* dedy 通道分形粒度，恒 16（= CONV3DDX_C0） */
    uint32_t mValue;     /* alignUp(hi*wi, m0)，M 向选块长度上限 */
    uint32_t nValue;     /* alignUp(cinG, n0)，N 向选块长度上限 */
    uint32_t kValue;     /* alignUp(coutG, C0) * hk * wk，K 向归约总长 */
    uint32_t kIter;      /* ceilDiv(kValue, baseK)，K 迭代次数 */
    uint32_t tailK;      /* kValue - (kIter-1)*baseK，末段有效 K 长（≥1 且 ≤ baseK） */
    uint32_t hkWk;       /* hk * wk，核高宽积（K 窗口基数） */
    uint32_t dkHkWk;     /* dk * hk * wk，核体积（loadB2 判定输入） */
    uint32_t khDilation; /* 有效核高 KeffH = (hk-1)*dilationH + 1 */
    uint32_t kwDilation; /* 有效核宽 KeffW = (wk-1)*dilationW + 1 */
    uint32_t hoExpand;   /* (ho-1)*strideH + 1，dedy 展开逻辑高 */
    uint32_t woExpand;   /* (wo-1)*strideW + 1，dedy 展开逻辑宽 */
    uint32_t doHoWo;     /* dout*ho*wo，dedy 单通道空间元素数（通道步幅） */
    uint32_t hiWi;       /* hi * wi */
    uint32_t diHiWi;     /* di * hi * wi，dedx 单通道空间元素数 */
    /* ---- 计算块（L0 基本块，generic/tiling §4） ---- */
    uint32_t baseM; /* M 向块宽（16 的倍数，min(起点,mValue,L0A 容量截断) 后经 N/M 扩张调整） */
    uint32_t baseN; /* N 向块宽（16 的倍数） */
    uint32_t baseK; /* K 向块长（C0 的倍数，K 窗口配对结果；≥ C0） */
    /* ---- 任务划分（generic/compute-flow §6 / generic/tiling §8） ---- */
    uint32_t coreNum; /* 启动计算核数（= SetBlockDim；运行时 platformInfo 查询，≤ CONV3DDX_MAX_CORES） */
    uint32_t singleCoreM;     /* 单任务空间长度（起点粒度；< baseM 时块收缩对齐回 baseM） */
    uint32_t singleCoreCin;   /* 单任务 dedx 通道段长（= 最终 baseN 的通道划分） */
    uint32_t singleCoreDin;   /* 单任务深度段长（本执行配置恒 1，D 轴任务逐层分发） */
    uint32_t singleCoreCout;  /* 单任务 dedy 通道范围（恒 = coutG，K 形成时乘 hk*wk 并按 C0 对齐） */
    uint32_t singleCoreBatch; /* 恒 1（每任务一个 batch） */
    uint32_t singleCoreGroup; /* 恒 1（每任务一个 group） */
    uint32_t singleIterateDk; /* 恒 = dk（核深度在 K 循环外完整归约） */
    uint64_t totalTasks;      /* 任务总数 = batch * group * dCount * mCount * nCount（宽整数发布） */
    /* ---- L1 驻留（generic/tiling §5/§6） ---- */
    uint32_t stepKa;     /* dedy 一路驻留供给的 K 计算块数（驻留长 stepKa*baseK；≤ 65535/baseK） */
    uint32_t stepKb;     /* weight 一路驻留供给的 K 计算块数（驻留长 stepKb*baseK） */
    uint32_t aSlotBytes; /* 单份 L1A 槽字节数（32B 对齐；驻留 dedy 预算） */
    uint32_t bSlotBytes; /* 单份 L1B 槽字节数（32B 对齐；驻留 weight 预算） */
    /* ---- Buffer 份数与执行标志 ---- */
    uint32_t al1Pbuffer; /* L1A（dedy 驻留）份数，取 1 或 2 */
    uint32_t bl1Pbuffer; /* L1B（weight 驻留）份数，取 1 或 2 */
    uint32_t al0Pbuffer; /* L0A（fmap 窗口）份数，取 1 或 2 */
    uint32_t bl0Pbuffer; /* L0B（反向权重）份数，取 1 或 2 */
    uint32_t cl0Pbuffer; /* L0C（dedx 部分和）份数，取 1 或 2（L0C 容量富余时取 2） */
    uint32_t initOutputFlag; /* 输出初始化要求（1=存在无贡献位置；本交付为信息性标志，见约束表） */
    uint32_t enableFullLoad; /* 完整 K 权重驻留开关（仅 full_load 求解器置 1，其余恒 0） */
    uint32_t iterateOrder;   /* L1 驻留遍历序（当前执行配置恒 1） */
};

#endif /* CONV3DDX_TILING_STRUCT_H */
