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
 * \file conv3dtranspose.h
 * \brief Conv3DTranspose 算子 kernel 模板类（三 loadB2 权重变换共用骨架 + bias BT 通路，arch35 / AIC-only Cube 通路）
 *
 * 实现真值（design 文档集为权威依据，实现不偏离设计）：
 *   - design/Kernel.md §2「Kernel 类声明」/§3「调度框架」/§4「Init」/§5「Process（每种 dtype 独立）」/
 *     §6「Sync」/§7「CopyIn A 路径 + B 路径共享帧 + CopyInBrc bias BT 通路」/§8「CopyOut」/
 *     §9.1–§9.9——跨分支共享帧（入口三步协议、任务解码轴序、UpdateL1ComputeInfo 行带交集 +
 *     深度采样、EnsureA1/IssueLoadToA2/Load3D、Mmad FP32 L0C 累加（bias 经 BT (C2) 以 FP32 折入
 *     首个有效 MMAD 的 L0C 初值）、Fixpipe 写回、无贡献块完整写回协议（ComputeForBias /
 *     ComputeZero，D-1 修复：禁用裸标量 GM 写）、L0 槽对 MTE1_M/M_MTE1 事件对、L1B enableFullLoad
 *     驻留）逐字实装；
 *   - design/branches/DESIGN-BRANCH-16777217.md §5.2（loadB2Condition=1 转置加逆序：GM→L1B 三布局
 *     参数表 + L1B 槽清零「补 co 尾」+ LoadData2DParamsV2 逆序源偏移（转置与逆序同时生效））——
 *     本分支（branch-1 bring-up）的实现义务，ifTranspose=true 编译期常量；
 *     DESIGN-BRANCH-16777218.md §5.2（loadB2Condition=2 仅逆序：整行逆序装载 + 负 srcStride
 *     窗口分段）、DESIGN-BRANCH-16777219.md §5.2（loadB2Condition=3 仅转置：核体积 1 三布局
 *     归一装载 + cout1 分形正序）——ASCENDC_TPL_SEL 在每个 .o 内实例化全部 3 个 sub-kernel，
 *     三臂均须可编译（本节点门禁不运行另两 key 的用例，不改动其契约）。
 *
 * 参考实现：范式 conv3dtranspose_direct（20 例客户谱系验证全绿直调工程——转置几何 + bias BT
 * 通路 + ComputeZero/ComputeForBias；注册形态按 Kernel.md §1 差异登记适配：format 提升为编译期
 * 模板实参 FMT_FILTER/FMT_DY/FMT_Y（三 filter 布局经 if constexpr 静态分派）、GET_TILING_DATA_WITH_STRUCT
 * 整体读入、AIC-only 入口、bias dtype 经 BIAST 模板参数与 DTYPE_BIAS 外层宏绑定（两口径 sizing
 * 见 Init））。DHWCN 装载参数为同族注册工程 conv3ddx_aclnn 已落地的设备验证修正值（本算子的
 * DHWCN 独立执行记录由矩阵 #38/V-13 回填，DESIGN-BRANCH-16777217.md §5.2 证据状态）。
 *
 * 片上级数（范式边界，DESIGN-BRANCH-16777217.md §4）：GM→L1A/L1B→L0A/L0B→L0C→GM（bias 另经
 * L1→BT），UB 不承载主计算数据；dtype fp16/bf16（2B）输入、FP32 L0C（4B）累加、F322F16/F322BF16
 * CAST_RINT 写出；无 FP32/vec/C04 兜底路径（本交付执行配置）；主计算完整写回（无 InitOutput、
 * 无融合 AIV 清零、无 AIV 启动配比依赖）。
 */

#ifndef CONV3DTRANSPOSE_KERNEL_H
#define CONV3DTRANSPOSE_KERNEL_H

#include "kernel_operator.h"
#include "conv3dtranspose_tiling_data.h"
#include "conv3dtranspose_tiling_key.h" /* TPL_* 常量 + ASCENDC_TPL_SEL 实例集合 */

using namespace AscendC;

/* ---------------- 文件局部标量小工具（Kernel.md §2：与 SDK API 名称分开的本地定义） ---------------- */

__aicore__ inline uint32_t CeilDivL(uint64_t a, uint64_t b)
{
    return b == 0 ? 0 : static_cast<uint32_t>((a + b - 1) / b);
}
__aicore__ inline uint32_t AlignUp16L(uint32_t a)
{
    return (a + 15u) >> 4 << 4;
}
__aicore__ inline uint32_t CeilDiv16L(uint32_t a)
{
    return (a + 15u) >> 4;
}
__aicore__ inline uint32_t AlignUpTo(uint32_t a, uint32_t b)
{
    return CeilDivL(a, b) * b;
}
constexpr uint32_t BLOCK_SIZE_ELEM = 256u;                    /* 512B 分形 = fp16/bf16 256 输入元素
                                                                 （范式 instruction-mapping §5.1 步幅单位） */
constexpr uint64_t BIAS_HELD_INVALID = 0xFFFFFFFFFFFFFFFFull; /* bias 全量持有身份哨兵 */

/*
 * 注册组合 format 标签：取值 = 对应 ge::Format 枚举值（本交付 CANN 9.3.0
 * include/graph/c_types.h 实测：C_FORMAT_NDHWC=27、C_FORMAT_NCDHW=30、C_FORMAT_DHWCN=31），
 * 与 codegen 按 OpDef 组合注入的 FORMAT_FILTER / FORMAT_X / FORMAT_Y 宏值一致
 * （registry-invoke def 驱动模式）；Kernel.md 9.9 校验项 7 的构建期核定对象。
 */
constexpr uint32_t CONV3DTRANSPOSE_FMT_NDHWC = 27u;
constexpr uint32_t CONV3DTRANSPOSE_FMT_NCDHW = 30u;
constexpr uint32_t CONV3DTRANSPOSE_FMT_DHWCN = 31u;

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST = float>
class Conv3dtranspose {
public:
    /* 接口契约断言：受支持组合内 x/filter/y 恒同 dtype（Interface.md「算器规格(受支持
     * dtype 组合)」等 dtype 配对，混合 dtype 不支持）；y 随 x 严格配对（Interface.md
     * 「数据 Format 支持」）；bias 为独立 fp16/fp32 类型（不参与 y 的 dtype 推导，
     * 不设与 T 的同 dtype 断言）。 */
    static_assert(IsSameType<FT, DYT>::value && IsSameType<DYT, YT>::value,
                  "filter/x/y must share one dtype in supported combos");
    static_assert(FMT_DY == FMT_Y, "y format must pair with x format");
    using T = FT;

    /* 编译期布局标签（x/y 双格式、filter 三布局；DHWCN filter 恒路由 16777217，
     * BranchRoute.md「tilingKey 路由表」） */
    static constexpr bool kDyNd = (FMT_DY == CONV3DTRANSPOSE_FMT_NDHWC);
    static constexpr bool kFiltNcdhw = (FMT_FILTER == CONV3DTRANSPOSE_FMT_NCDHW);
    static constexpr bool kFiltNd = (FMT_FILTER == CONV3DTRANSPOSE_FMT_NDHWC);
    static constexpr bool kIsBf16 = IsSameType<T, bfloat16_t>::value;

    __aicore__ inline Conv3dtranspose() {}
    __aicore__ inline ~Conv3dtranspose() {}

    /* 入口三步协议（范式 kernel-template §2.1/§2.3：Init → Process → End 各调用一次；
     * Init 末位 bias 参数默认 nullptr（DX 入口可省略），本算子入口恒显式传入；
     * Process 按入口 bias 指针分路 hasBias；End 由入口调用一次，Process 内不重复调用 */
    __aicore__ inline void Init(GM_ADDR filter, GM_ADDR x, GM_ADDR y, const Conv3DTransposeTilingData& td,
                                GM_ADDR bias = nullptr);
    template <bool hasBias>
    __aicore__ inline void Process();
    __aicore__ inline void End();

private:
    /* ---- 调度与块级门控（跨分支共享） ---- */
    __aicore__ inline void UpdateL1ComputeInfo(); /* 行带交集 + 深度采样 → needCompute_ */
    template <bool hasBias>
    __aicore__ inline void ComputeOneBlock(); /* 单输出块完整归约（含 bias 折入）+ 写回 */
    template <bool hasBias>
    __aicore__ inline void NoNeedComputeHandle(); /* 无贡献块完整写回（9.6） */
    /* ---- 显式形状余量 rho 的零裕域写回（Kernel.md 9.9 校验项 4 尾块/边界域收口） ---- */
    __aicore__ inline void ResolveRhoMargin(); /* 逐轴 corner/E 求解（Init 一次） */
    template <bool hasBias>
    __aicore__ inline void WriteMarginZones(uint64_t spaceStart, uint64_t spaceUse); /* H/W 裕域覆盖写回 */
    template <bool hasBias>
    __aicore__ inline void WriteMarginRange(uint64_t segLo, uint64_t segHi, uint64_t domLo, uint64_t domHi,
                                            uint64_t mTotal); /* 分段分块写回帧 */
    /* ---- A 路径：dedy(x) GM→L1A→L0A（跨分支共享） ---- */
    __aicore__ inline void EnsureA1(uint32_t curDoutIdx, uint32_t kIdx);
    __aicore__ inline void ReleaseA1IfSegmentEnd(uint32_t curDoutIdx, uint32_t kIdx);
    __aicore__ inline void FreeA1Tensor();
    __aicore__ inline void IssueLoadToA2(uint64_t kStart);
    /* ---- B 路径：weight(filter) GM→L1B→L0B（共享帧；参数构造体归分支设计文件 §5.2） ---- */
    __aicore__ inline void EnsureB1(uint32_t kd, uint32_t kIdx);
    __aicore__ inline void ReleaseB1IfSegmentEnd(uint32_t kIdx);
    __aicore__ inline void FreeB1Tensor();
    __aicore__ inline void IssueLoadToB2(uint64_t kStart);
    /* ---- bias BT 通路（仅 hasBias 实例执行；范式 bias-bt-path） ---- */
    __aicore__ inline void LoadBiasToL1(uint64_t gmStart, uint32_t biasUse);
    __aicore__ inline void CopyBiasL1ToBT(uint32_t biasUse);
    __aicore__ inline void FullLoadBias();
    __aicore__ inline void LoadBiasToBT();
    __aicore__ inline void ComputeBTBias();
    __aicore__ inline void FreeBiasTensor();
    /* ---- 无贡献块零窗口（ComputeForBias / ComputeZero 共用；bias-bt-path §5） ---- */
    __aicore__ inline void EnsureZeroL1Held();
    __aicore__ inline void ComputeForBias(LocalTensor<float>& useC1);
    __aicore__ inline void ComputeZero(LocalTensor<float>& useC1);
    __aicore__ inline void IssueLoadL0ZeroA2();
    __aicore__ inline void IssueLoadL0ZeroB2();
    /* ---- 计算与写回（跨分支共享） ---- */
    template <bool hasBias>
    __aicore__ inline void IssueMmad(LocalTensor<float>& useC1, bool firstContribution);
    __aicore__ inline void LoadL0c2Gm(LocalTensor<float>& c1v);
    __aicore__ inline void FillZero(const LocalTensor<T>& buf, uint32_t slotBytes);

    /* ---- 状态字段（跨分支共享集；三 key 实例化同一字段布局） ---- */
    Conv3DTransposeTilingData td_;
    GlobalTensor<T> wGm_;        /* filter（weight） */
    GlobalTensor<T> dyGm_;       /* x（dedy） */
    GlobalTensor<T> xGm_;        /* y（dedx） */
    GlobalTensor<BIAST> biasGm_; /* bias（存在时绑定） */
    bool hasBiasGm_ = false;     /* 入口 bias 指针存在性（与 Process<hasBias> 恒一致） */
    TPipe pipe_;
    TQue<TPosition::A1, 2> queAl1_;      /* L1A：dedy 驻留（队列深度 2，实际份数 al1Pbuffer） */
    TQue<TPosition::B1, 2> queBl1_;      /* L1B：weight 驻留（队列深度 2，实际份数 bl1Pbuffer） */
    TBuf<TPosition::A2> a2Buf_;          /* L0A：fmap 窗口 */
    TBuf<TPosition::B2> b2Buf_;          /* L0B：weight_rot */
    TQue<TPosition::CO1, 1> queCl0Ping_; /* L0C 槽 0（每槽独立 CO1 队列，深度 1） */
    TQue<TPosition::CO1, 1> queCl0Pong_; /* L0C 槽 1（cl0Pbuffer=2 时启用） */
    TQue<TPosition::A1, 1> biasL1Que_;   /* bias L1（存在时创建；通道连续 BIAST） */
    TQue<TPosition::C2, 1> biasBTQue_;   /* bias BT（存在时创建；FP32 部分和口径） */
    LocalTensor<T> a2Base_;
    LocalTensor<T> b2Base_;
    LocalTensor<T> a1v_;
    LocalTensor<T> b1v_;
    LocalTensor<BIAST> biasL1Held_;
    LocalTensor<float> biasBTHeld_;
    LoadData3DParamsV2<T> load3d_;
    LoadData2DParamsV2 load2dv2_;
    LoadData3DParamsV2<T> load3dZero_; /* 零窗口专用独立实例（不干扰常规窗口状态） */
    LoadData2DParamsV2 load2dZero_;
    uint32_t coreId_ = 0;
    uint32_t coreNum_ = 1;
    int64_t savedCtrlFloatBit_ = 1; /* Init 时保存的 CTRL[48]（FLOAT 饱和位，End 恢复；单粒度位访问） */
    int64_t savedCtrlStrategyBit_ = 0; /* Init 时保存的 CTRL[60]（饱和策略位，End 恢复） */
    uint32_t hoWo_ = 0;                /* ho·wo（dedy 行内元素数） */
    uint32_t inputSlotCount_ = 1;      /* L0A/L0B 输入槽对数：两路同时双缓冲为 2，否则 1 */
    uint32_t l0aHalfElem_ = 0;         /* L0A 半区元素数（al0Pbuffer≥2 时非 0） */
    uint32_t l0bHalfElem_ = 0;         /* L0B 半区元素数（bl0Pbuffer≥2 时非 0） */
    uint32_t cSlotCount_ = 1;          /* L0C 槽数（cl0Pbuffer） */
    uint32_t l0Slot_ = 0;              /* 当前 L0 输入槽对 */
    uint32_t cSlot_ = 0;               /* 当前 L0C 槽 */
    uint64_t batchIdx_ = 0;
    uint64_t groupIdx_ = 0;
    uint32_t curDinIdx_ = 0;            /* 当前 dedx 深度 */
    uint32_t mStart_ = 0;               /* 当前输出块空间起点 */
    uint32_t cinStart_ = 0;             /* 当前 N 块 dedx 组内通道起点 */
    uint32_t taskChannelStart_ = 0;     /* 当前任务 dedx 通道段起点（全量 bias 偏移基准） */
    uint32_t baseUseM_ = 0;             /* 有效 M（窗口裁剪后） */
    uint32_t baseUseN_ = 0;             /* 有效 N */
    uint32_t baseUseK_ = 0;             /* 本轮有效 K */
    uint32_t mCompute_ = 0;             /* alignUp(baseUseM_,16)：指令对齐 M */
    uint32_t nCompute_ = 0;             /* alignUp(baseUseN_,16)：L0 物理排布口径 */
    uint32_t blockBaseN_ = 1;           /* ceilDiv(baseUseN_,16)：N 分形数 */
    bool needCompute_ = false;          /* 行带与深度采样联合判定 */
    int32_t curHoIdx_ = 0;              /* 当前块首行在 dedy_expand 中的行号（可负） */
    uint32_t curHoSize_ = 0;            /* 驻留行带行数 */
    uint32_t woExpandLocal_ = 0;        /* 裁剪后 dedy 展开宽 */
    uint32_t a1SegFirst_ = 0xFFFFFFFFu; /* L1A 父段身份：首 kIdx 段 */
    uint32_t a1Dout_ = 0xFFFFFFFFu;     /* L1A 父段身份：curDoutIdx */
    uint32_t aSegStartK_ = 0;           /* L1A 父段完整 K 起点 */
    bool a1Valid_ = false;
    uint32_t channelSize_ = 0;          /* L1A 描述的 dedy 通道范围 */
    uint32_t b1SegFirst_ = 0xFFFFFFFFu; /* L1B 父段身份四元组（segFirst/kd/cin/group） */
    uint32_t bSegStartK_ = 0;
    uint32_t b1IdKd_ = 0xFFFFFFFFu;
    uint32_t b1IdCin_ = 0xFFFFFFFFu;
    uint32_t b1IdGroup_ = 0xFFFFFFFFu;
    bool b1Valid_ = false;
    uint32_t curCout1Cout0Size_ = 0; /* L1B 父段 Cout 对齐量 */
    uint32_t curLoadKbl1_ = 0;       /* L1B 父段 K 跨度（元素） */
    /* bias 状态（范式 bias-bt-path §1/§4/§6；仅 hasBiasGm_ 时有效） */
    bool biasL1HeldValid_ = false;
    bool biasBTHeldValid_ = false;
    uint64_t biasHeldStart_ = BIAS_HELD_INVALID; /* 全量 bias 持有身份（GM 起点） */
    uint64_t biasTaskStart_ = 0;                 /* 当前任务 bias GM 起点（groupIdx·cinG+channelStart） */
    uint32_t biasTaskUse_ = 0;                   /* 当前任务有效通道数（channelUse） */
    bool fullLoadBiasFlag_ = false;              /* 当前任务是否需重载全量 bias（身份变化） */
    uint32_t biasOffset_ = 0;                    /* 当前输出块在 BT 内的元素偏移 */
    bool computeBiasOnce_ = false;               /* 当前输出块是否已引入 bias（块首复位） */
    /* rho 零裕域（spec.yaml math_semantics「rho 余量对应插零域未命中位置，写 0/bias」；
     * 轴进入 golden 参考语义的 torch 不可表达角落（opSpec = q+rho ≥ s 且 ≥ d 且
     * opMin ≥ s 且 ≥ d）时，位置 ≥ E = (Lx-1)·s + Keff − min(pF,pB) − pF 恒写 0/bias，
     * E 由既有 TilingData 字段（lx/ly/stride/dilation/kEff/pads）设备侧推导，
     * Host 与 TilingData 契约零变更；非角落轴不设裕域（对 rho=0 与可表达 rho 组合
     * 零扰动）。Init 求解一次，字段三轴共享。 */
    bool dMarginActive_ = false; /* D 轴零裕域生效（深度 ≥ dMarginStart_ 全裕域） */
    bool hMarginActive_ = false; /* H 轴零裕域生效（行 ≥ hMarginStart_ 全裕域） */
    bool wMarginActive_ = false; /* W 轴零裕域生效（列 ≥ wMarginStart_ 全裕域） */
    uint32_t dMarginStart_ = 0u; /* D 轴裕域起点（含） */
    uint32_t hMarginStart_ = 0u; /* H 轴裕域起点（含） */
    uint32_t wMarginStart_ = 0u; /* W 轴裕域起点（含） */
};

/* ================================ Init（Kernel.md §4） ================================ */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::Init(
    GM_ADDR filter, GM_ADDR x, GM_ADDR y, const Conv3DTransposeTilingData& td, GM_ADDR bias)
{
    td_ = td;
    wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(filter));
    dyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    /* fp16 溢出饱和（DESIGN-BRANCH-16777217.md §5.4/§7、spec.yaml extreme_inputs「fp16 上溢边界」
     * 与 Overview.md「特殊值与数值稳定性」溢出条目）：Fixpipe quantPre=F322F16 的降精度写出默认
     * INF_NAN 模（CANN Fixpipe 文档 quantPre 表），构造性极端输入终值越出 fp16 表示域时写出
     * ±Inf（非有限值硬门禁违例；case00503/504 首次设备复现——V-14 谱系此前从未执行）。
     * 打开全局浮点饱和（CTRL[48]=0 + USE_GLOBAL 策略，AscendC SetSaturationFlag/SetSaturationStrategy，
     * 枚举注释「floating-point computation and precision conversion」）使该精度转换按输出 dtype
     * 表示域饱和（fp16 ±65504，与 golden 同口径）；End 按保存值恢复 CTRL 位域，不向核外泄漏。
     * 主计算不受影响：L0C FP32 累加量级（用例域 |sum|≤4.2e3，极端构造 ≤4.6e8）远低于 fp32 饱和界，
     * 饱和仅作用于越出 fp16 表示域的写出值，域内值仍按 CAST_RINT 最近偶舍入。 */
    savedCtrlFloatBit_ = GetCtrlSpr<48, 48>();
    savedCtrlStrategyBit_ = GetCtrlSpr<60, 60>();
    SetSaturationStrategy<OverrideStrategy::USE_GLOBAL>();
    SetSaturationFlag<SaturationMode::FLOAT>(true);

    /* bias 为运行时指针（nullptr=无 bias）：存在时才创建两路 bias 队列资源
     * （范式 buffer-and-types §3.1「无 bias 时不创建两路 bias 队列资源」；
     * TilingData.md「范式特有约束」bias 存在性指针契约行）；入口按同一指针分路
     * Process<true>/<false>，hasBiasGm_ 与 hasBias 编译期分路恒一致 */
    hasBiasGm_ = (bias != nullptr);
    if (hasBiasGm_) {
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BIAST*>(bias));
    }

    coreId_ = GetBlockIdx();
    coreNum_ = td.coreNum;

    hoWo_ = td.ho * td.wo;
    inputSlotCount_ = (td.al0Pbuffer >= 2 && td.bl0Pbuffer >= 2) ? 2u : 1u;
    l0aHalfElem_ = (td.al0Pbuffer >= 2) ? (CONV3DTRANSPOSE_L0A_BYTES / 2 / sizeof(T)) : 0;
    l0bHalfElem_ = (td.bl0Pbuffer >= 2) ? (CONV3DTRANSPOSE_L0B_BYTES / 2 / sizeof(T)) : 0;
    cSlotCount_ = (td.cl0Pbuffer >= 2) ? 2u : 1u;

    /* 五路 Buffer 分配（范式 buffer-and-types §3；TQue 形态与内置基线同族一致：
     * A1/B1 深度 2、L0A/L0B 整块 TBuf、CO1 每槽独立队列深度 1） */
    pipe_.InitBuffer(queAl1_, td.al1Pbuffer, td.aSlotBytes);
    pipe_.InitBuffer(queBl1_, td.bl1Pbuffer, td.bSlotBytes);
    pipe_.InitBuffer(a2Buf_, CONV3DTRANSPOSE_L0A_BYTES);
    pipe_.InitBuffer(b2Buf_, CONV3DTRANSPOSE_L0B_BYTES);
    pipe_.InitBuffer(queCl0Ping_, 1, CONV3DTRANSPOSE_L0C_BYTES / cSlotCount_);
    if (cSlotCount_ == 2) {
        pipe_.InitBuffer(queCl0Pong_, 1, CONV3DTRANSPOSE_L0C_BYTES / cSlotCount_);
    }
    if (hasBiasGm_) {
        /* bias 双队列各一份（范式 bias-bt-path §1）。两槽字节数口径不同：
         * L1 槽按 bias 元素宽度口径消费 Host 发布的 biasAllocBytes
         * （TilingData.md「字段表」：alignUp(biasUse·biasBytes,64)，biasBytes=
         * sizeof(BIAST)）；BT 槽按折入通路 4B 部分和口径（TilingData.md 头注
         * 「L0C / BT 折入通路恒按 4B 部分和消费」；biasUse 基数与 HostTiling.md
         * 「填 TilingData + 设 TilingKey」bias 通路行同式）。BIAST=float 时两口径
         * 同值（= biasAllocBytes，与直调参考实现一致）；BIAST=half 的扩展义务
         * 见 Kernel.md 9.9 校验项 9。 */
        const uint32_t biasUse = (td.isBiasFullLoad != 0) ? td.singleCoreCin : td.baseN;
        pipe_.InitBuffer(biasL1Que_, 1, td.biasAllocBytes);
        pipe_.InitBuffer(
            biasBTQue_, 1,
            AlignUpTo(biasUse * CONV3DTRANSPOSE_ACC_BYTES, CONV3DTRANSPOSE_BT_COVER_BYTES)); /* F-02 具名 */
    }
    a2Base_ = a2Buf_.Get<T>();
    b2Base_ = b2Buf_.Get<T>();

    /* Load3D 参数初值（范式 instruction-mapping §4.1；L1A 已按正向 stride 插零，
     * 故 H/W 步长固定 1，保留 dilation） */
    load3d_.strideW = 1;
    load3d_.strideH = 1;
    load3d_.filterW = static_cast<uint8_t>(td.wk);
    load3d_.filterH = static_cast<uint8_t>(td.hk);
    load3d_.filterSizeW = td.wk > 255u;
    load3d_.filterSizeH = td.hk > 255u;
    load3d_.dilationFilterW = static_cast<uint8_t>(td.dilationW);
    load3d_.dilationFilterH = static_cast<uint8_t>(td.dilationH);
    load3d_.enTranspose = false;
    load3d_.enSmallK = false;
    load3d_.fMatrixCtrl = false;
    load3d_.padValue = static_cast<T>(0);

    /* Load2D 参数初值（范式 instruction-mapping §5.1；ifTranspose 是分支编译期常量，
     * 其余字段在 IssueLoadToB2 内按分支改写——三分支参数表归分支设计文件 §5.2） */
    load2dv2_.mStartPosition = 0;
    load2dv2_.kStartPosition = 0;
    load2dv2_.mStep = 1;
    load2dv2_.kStep = 1;
    load2dv2_.dstStride = 1;
    load2dv2_.ifTranspose = (LOADB2 != TPL_REVERSE_ONLY);
    load2dv2_.sid = 0;

    /* ComputeForBias / ComputeZero 专用 LoadL0Zero 参数（范式 bias-bt-path §5 参数表）：
     * 独立实例，不干扰常规窗口状态 */
    load3dZero_.strideW = 1;
    load3dZero_.strideH = 1;
    load3dZero_.filterW = 1;
    load3dZero_.filterH = 1;
    load3dZero_.filterSizeW = false;
    load3dZero_.filterSizeH = false;
    load3dZero_.dilationFilterW = 1;
    load3dZero_.dilationFilterH = 1;
    load3dZero_.enTranspose = false;
    load3dZero_.enSmallK = false;
    load3dZero_.fMatrixCtrl = false;
    load3dZero_.padValue = static_cast<T>(0);
    load3dZero_.l1H = 1;
    load3dZero_.l1W = 32;
    load3dZero_.padList[0] = 0;    /* 左 */
    load3dZero_.padList[1] = 0;    /* 右 */
    load3dZero_.padList[2] = 255u; /* 上：补零窗口——网格 H 扩至 256 行，全部落入上侧补零区 */
    load3dZero_.padList[3] = 0;    /* 下 */
    load3dZero_.channelSize = static_cast<uint16_t>(CONV3DTRANSPOSE_C0);
    load3dZero_.mStartPt = 0;
    load3dZero_.kStartPt = 0;

    load2dZero_.mStartPosition = 0;
    load2dZero_.kStartPosition = 0;
    load2dZero_.mStep = 1;
    load2dZero_.kStep = 1;
    load2dZero_.srcStride = 1;
    load2dZero_.dstStride = 1;
    load2dZero_.ifTranspose = true;
    load2dZero_.sid = 0;

    /* L1 持有状态复位（含 bias 状态） */
    a1Valid_ = false;
    b1Valid_ = false;
    a1SegFirst_ = 0xFFFFFFFFu;
    a1Dout_ = 0xFFFFFFFFu;
    b1SegFirst_ = 0xFFFFFFFFu;
    b1IdKd_ = 0xFFFFFFFFu;
    b1IdCin_ = 0xFFFFFFFFu;
    b1IdGroup_ = 0xFFFFFFFFu;
    l0Slot_ = 0;
    cSlot_ = 0;
    biasL1HeldValid_ = false;
    biasBTHeldValid_ = false;
    biasHeldStart_ = BIAS_HELD_INVALID;
    biasTaskStart_ = 0;
    biasTaskUse_ = 0;
    fullLoadBiasFlag_ = false;
    biasOffset_ = 0;
    computeBiasOnce_ = false;
    /* rho 零裕域求解（三轴；见字段组注释——q+rho 由 ly 与既有几何反解：
     * opSpec = ly − [(lx−1)·s − pF − pB + kEff]，与 Host 折入 backpropPadRight 的
     * q+rho 同源，设备不感知 q/rho 拆分） */
    ResolveRhoMargin();
}

/* rho 零裕域逐轴求解（Init 尾调用；corner/E 公式与冻结 golden 的
 * 「torch 约束角落（op=0 + 尾部零填充）」分支逐值对齐——cpu 侧已对全部
 * 裕域行核验 margin ≡ 0/bias，证据见 run-l1-100pct.RECORD.md）：
 *   opSpec = q + rho = ly − [(lx−1)·s − pF − pB + kEff]
 *   m = min(pF,pB)，eB = pB − m，opMin = max(0, opSpec − eB)
 *   corner ⟺ opSpec ≥ s ∧ opSpec ≥ d ∧ opMin ≥ s ∧ opMin ≥ d
 *   E = (lx−1)·s + kEff − m − pF（torch op=0 原始域在 y 坐标的界，含 eF 折算）
 * corner 且 E < ly 时轴裕域 [max(E,0), ly) 恒写 0/bias；E ≥ ly 时无裕域。 */
template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::ResolveRhoMargin()
{
    const Conv3DTransposeTilingData& td = td_;
    /* D 轴：lx=dout / ly=di / kEff=(dk-1)*dilationD+1 / pF=padFront / pB=padBack */
    {
        const int64_t kEff = (int64_t)(td.dk - 1) * td.dilationD + 1;
        const int64_t pF = td.padFront;
        const int64_t pB = td.padBack;
        const int64_t opSpec = (int64_t)td.di - ((int64_t)(td.dout - 1) * td.strideD - pF - pB + kEff);
        const int64_t m = pF < pB ? pF : pB;
        const int64_t eB = pB - m;
        int64_t opMin = opSpec - eB;
        if (opMin < 0) {
            opMin = 0;
        }
        const int64_t e = (int64_t)(td.dout - 1) * td.strideD + kEff - m - pF;
        if (opSpec >= (int64_t)td.strideD && opSpec >= (int64_t)td.dilationD && opMin >= (int64_t)td.strideD &&
            opMin >= (int64_t)td.dilationD && e < (int64_t)td.di) {
            dMarginActive_ = true;
            dMarginStart_ = e < 0 ? 0u : static_cast<uint32_t>(e);
        }
    }
    /* H 轴：lx=ho / ly=hi / kEff=khDilation / pF=padUp / pB=padDown */
    {
        const int64_t kEff = td.khDilation;
        const int64_t pF = td.padUp;
        const int64_t pB = td.padDown;
        const int64_t opSpec = (int64_t)td.hi - ((int64_t)(td.ho - 1) * td.strideH - pF - pB + kEff);
        const int64_t m = pF < pB ? pF : pB;
        const int64_t eB = pB - m;
        int64_t opMin = opSpec - eB;
        if (opMin < 0) {
            opMin = 0;
        }
        const int64_t e = (int64_t)(td.ho - 1) * td.strideH + kEff - m - pF;
        if (opSpec >= (int64_t)td.strideH && opSpec >= (int64_t)td.dilationH && opMin >= (int64_t)td.strideH &&
            opMin >= (int64_t)td.dilationH && e < (int64_t)td.hi) {
            hMarginActive_ = true;
            hMarginStart_ = e < 0 ? 0u : static_cast<uint32_t>(e);
        }
    }
    /* W 轴：lx=wo / ly=wi / kEff=kwDilation / pF=padLeft / pB=padRight */
    {
        const int64_t kEff = td.kwDilation;
        const int64_t pF = td.padLeft;
        const int64_t pB = td.padRight;
        const int64_t opSpec = (int64_t)td.wi - ((int64_t)(td.wo - 1) * td.strideW - pF - pB + kEff);
        const int64_t m = pF < pB ? pF : pB;
        const int64_t eB = pB - m;
        int64_t opMin = opSpec - eB;
        if (opMin < 0) {
            opMin = 0;
        }
        const int64_t e = (int64_t)(td.wo - 1) * td.strideW + kEff - m - pF;
        if (opSpec >= (int64_t)td.strideW && opSpec >= (int64_t)td.dilationW && opMin >= (int64_t)td.strideW &&
            opMin >= (int64_t)td.dilationW && e < (int64_t)td.wi) {
            wMarginActive_ = true;
            wMarginStart_ = e < 0 ? 0u : static_cast<uint32_t>(e);
        }
    }
}

/* ================================ Process（Kernel.md §5，含 §3 任务解码） ================================ */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
template <bool hasBias>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::Process()
{
    const Conv3DTransposeTilingData& td = td_;
    const uint64_t mCount = CeilDivL(static_cast<uint64_t>(td.hi) * td.wi, td.singleCoreM);
    const uint64_t nCount = CeilDivL(td.cinG, td.singleCoreCin);
    const uint64_t dCount = CeilDivL(td.di, td.singleCoreDin);
    /* 任务轴序（范式 compute-flow §6.1）：batch/group 在外层，内层从慢到快按条件三选一 */
    uint64_t cnt0 = 0;
    uint64_t cnt1 = 0;
    uint64_t cnt2 = 0;
    uint32_t ax0 = 0;
    uint32_t ax1 = 0;
    uint32_t ax2 = 0; /* 轴编码：0=空间 M、1=深度 D、2=通道 N */
    if (td.dk > 1) {
        ax0 = 0;
        ax1 = 1;
        ax2 = 2;
        cnt0 = mCount;
        cnt1 = dCount;
        cnt2 = nCount;
    } else if ((td.hk > 1) || (mCount > nCount)) {
        ax0 = 1;
        ax1 = 0;
        ax2 = 2;
        cnt0 = dCount;
        cnt1 = mCount;
        cnt2 = nCount;
    } else {
        ax0 = 1;
        ax1 = 2;
        ax2 = 0;
        cnt0 = dCount;
        cnt1 = nCount;
        cnt2 = mCount;
    }
    const uint64_t bgCount = static_cast<uint64_t>(td.batch) * td.group;
    const uint64_t totalTasks = bgCount * cnt0 * cnt1 * cnt2;

    for (uint64_t taskId = coreId_; taskId < totalTasks; taskId += coreNum_) {
        uint64_t t = taskId;
        const uint64_t idx2 = t % cnt2;
        t /= cnt2;
        const uint64_t idx1 = t % cnt1;
        t /= cnt1;
        const uint64_t idx0 = t % cnt0;
        t /= cnt0;
        const uint64_t bg = t % bgCount;
        const uint64_t batchIdx = bg / td.group;
        const uint64_t groupIdx = bg % td.group;

        uint64_t mIdx = 0;
        uint64_t dIdx = 0;
        uint64_t nIdx = 0;
        if (ax0 == 0) {
            mIdx = idx0;
        } else if (ax0 == 1) {
            dIdx = idx0;
        } else {
            nIdx = idx0;
        }
        if (ax1 == 0) {
            mIdx = idx1;
        } else if (ax1 == 1) {
            dIdx = idx1;
        } else {
            nIdx = idx1;
        }
        if (ax2 == 0) {
            mIdx = idx2;
        } else if (ax2 == 1) {
            dIdx = idx2;
        } else {
            nIdx = idx2;
        }

        const uint64_t spaceStart = mIdx * td.singleCoreM;
        const uint64_t mTotal = static_cast<uint64_t>(td.hi) * td.wi;
        const uint64_t spaceUse = (spaceStart + td.singleCoreM <= mTotal) ? td.singleCoreM : (mTotal - spaceStart);
        const uint64_t channelStart = nIdx * td.singleCoreCin;
        const uint64_t channelUse =
            (channelStart + td.singleCoreCin <= td.cinG) ? td.singleCoreCin : (td.cinG - channelStart);
        const uint64_t depthStart = dIdx * td.singleCoreDin;
        const uint64_t depthUse = (depthStart + td.singleCoreDin <= td.di) ? td.singleCoreDin : (td.di - depthStart);
        batchIdx_ = batchIdx;
        groupIdx_ = groupIdx;

        /* bias 任务级装载（范式 kernel-template §2.2 SetBias / CalcBiasFullLoadFlag /
         * 全量重载三阶段内联；bias-bt-path §1）：全量模式按任务通道段一次装载双持有；
         * 身份变化（组/通道段）时重载；M/深度变化不改变身份 */
        if constexpr (hasBias) {
            taskChannelStart_ = static_cast<uint32_t>(channelStart);
            biasTaskStart_ = groupIdx * td.cinG + channelStart;
            biasTaskUse_ = static_cast<uint32_t>(channelUse);
            fullLoadBiasFlag_ = (biasTaskStart_ != biasHeldStart_);
            if (td.isBiasFullLoad != 0 && fullLoadBiasFlag_) {
                FreeBiasTensor();
                FullLoadBias();
            }
        }

        /* 任务内顺序：深度 → dedx 通道块 → 空间块（范式 kernel-template §2.2） */
        for (uint64_t d = 0; d < depthUse; ++d) {
            curDinIdx_ = static_cast<uint32_t>(depthStart + d);
            for (uint64_t n = 0; n < channelUse; n += td.baseN) {
                cinStart_ = static_cast<uint32_t>(channelStart + n);
                baseUseN_ = (n + td.baseN <= channelUse) ? td.baseN : static_cast<uint32_t>(channelUse - n);
                nCompute_ = AlignUp16L(baseUseN_);
                blockBaseN_ = CeilDiv16L(baseUseN_);
                if (blockBaseN_ == 0) {
                    blockBaseN_ = 1;
                }
                for (uint64_t m = 0; m < spaceUse; m += td.baseM) {
                    mStart_ = static_cast<uint32_t>(spaceStart + m);
                    baseUseM_ = (m + td.baseM <= spaceUse) ? td.baseM : static_cast<uint32_t>(spaceUse - m);
                    mCompute_ = AlignUp16L(baseUseM_);
                    computeBiasOnce_ = false;
                    UpdateL1ComputeInfo();
                    if (!needCompute_) {
                        NoNeedComputeHandle<hasBias>();
                        continue;
                    }
                    ComputeOneBlock<hasBias>();
                }
                /* H/W 轴零裕域覆盖写回（rho 零裕域，见字段组注释）：乘积块写回完成后，
                 * 对本任务空间域内的裕域位置以 NoNeedComputeHandle 同通路覆写 0/bias
                 * ——同核 Fixpipe 单管道按发射序串行，覆写在乘积写之后落地（9.6 通路
                 * 复用；D 轴裕域整层已由上方门控承接，深度入裕域时无需重复覆写）。
                 * 触达条件限 hMarginActive_/wMarginActive_（仅 rho>0 的 torch 不可
                 * 表达角落组合触达，对既有全绿组合零扰动）。 */
                if ((hMarginActive_ || wMarginActive_) && (!dMarginActive_ || curDinIdx_ < dMarginStart_)) {
                    WriteMarginZones<hasBias>(spaceStart, spaceUse);
                }
            }
        }
    }
    FreeB1Tensor();
    FreeA1Tensor();
    if constexpr (hasBias) {
        FreeBiasTensor();
    }
}

/* ================================ End（Kernel.md §6） ================================ */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::End()
{
    /* 收尾（范式 sync-implementation §1.2）：Process 已归还全部持有张量；End 对各实际
     * 创建队列调用 FreeAllEvent，等待各槽归还事件（最后一次 MTE1 读取 / Fixpipe 读取
     * 已提交）并释放队列事件；M_MTE1/MTE1_M 固定事件仍由 TPipe 析构等待并释放（含
     * 未使用槽保留的初始令牌）。仅对 InitBuffer 实际创建的队列调用（CO1 Pong 按
     * cSlotCount_、bias 双队列按 hasBiasGm_ 门控，与 Init 创建条件一致）。 */
    queAl1_.FreeAllEvent();
    queBl1_.FreeAllEvent();
    queCl0Ping_.FreeAllEvent();
    if (cSlotCount_ == 2) {
        queCl0Pong_.FreeAllEvent();
    }
    if (hasBiasGm_) {
        biasL1Que_.FreeAllEvent();
        biasBTQue_.FreeAllEvent();
    }
    /* 恢复 Init 保存的 CTRL[48]/CTRL[60]（FLOAT 饱和位/策略位），全局饱和设置不向本核
     * 后续 kernel 泄漏（固定事件仍由 TPipe 析构等待并释放） */
    SetCtrlSpr<48, 48>(savedCtrlFloatBit_);
    SetCtrlSpr<60, 60>(savedCtrlStrategyBit_);
}

/* ================================ 块级门控（Kernel.md §3 UpdateL1ComputeInfo） ================================ */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::UpdateL1ComputeInfo()
{
    const Conv3DTransposeTilingData& td = td_;
    /* D 轴零裕域门控（rho 零裕域，见字段组注释）：深度 ≥ dMarginStart_ 整层恒写
     * 0/bias（整层无乘积域），与 needCompute_=false 的 NoNeedComputeHandle 通路
     * 一致（9.6 完整写回协议）；先于深度采样判定，行带字段同口径置空。 */
    if (dMarginActive_ && curDinIdx_ >= dMarginStart_) {
        curHoSize_ = 0;
        needCompute_ = false;
        a1SegFirst_ = 0xFFFFFFFFu;
        a1Dout_ = 0xFFFFFFFFu;
        return;
    }
    /* 深度采样预判：任一 kd 使 dTmp 非负、整除 strideD 且商 < dout 即存在有效贡献（9.3） */
    bool dValid = false;
    for (uint32_t kd = 0; kd < td.dk; ++kd) {
        const int64_t dTmp = static_cast<int64_t>(curDinIdx_) + td.padFront - static_cast<int64_t>(kd) * td.dilationD;
        if (dTmp >= 0 && (td.strideD == 1 || dTmp % td.strideD == 0) &&
            dTmp < static_cast<int64_t>(td.dout) * td.strideD) {
            dValid = true;
            break;
        }
    }
    /* 行带交集：窗口包围行带 [curHoIdx_, curHoIdx_+hiCal+(hk-1)·dilationH) 与
     * [0, hoExpand) 的交（范式 compute-flow §5.2.1，负上边界先裁剪） */
    const uint32_t curHiIdx = mStart_ / td.wi;
    const uint32_t curWiIdx = mStart_ % td.wi;
    curHoIdx_ = static_cast<int32_t>(curHiIdx) - td.backpropPadUp;
    const uint32_t hiCal = CeilDivL(static_cast<uint64_t>(baseUseM_) + curWiIdx, td.wi);
    const int64_t endHoIdx = static_cast<int64_t>(curHoIdx_) + hiCal + (td.hk - 1) * td.dilationH;
    if (dValid && curHoIdx_ < static_cast<int32_t>(td.hoExpand) && endHoIdx > 0) {
        const uint32_t realStart = curHoIdx_ < 0 ? 0u : static_cast<uint32_t>(curHoIdx_);
        const uint32_t realEnd =
            static_cast<uint32_t>(endHoIdx) > td.hoExpand ? td.hoExpand : static_cast<uint32_t>(endHoIdx);
        curHoSize_ = realEnd - realStart;
        needCompute_ = true;
    } else {
        curHoSize_ = 0;
        needCompute_ = false;
    }
    /* 新输出块：L1A 父段身份失效（L1B 是否跨块保留由 enableFullLoad 决定，9.7） */
    a1SegFirst_ = 0xFFFFFFFFu;
    a1Dout_ = 0xFFFFFFFFu;
}

/* ================================ 单输出块主计算（Kernel.md §5 ComputeOneBlock） ================================ */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
template <bool hasBias>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::ComputeOneBlock()
{
    const Conv3DTransposeTilingData& td = td_;
    auto& q = (cSlot_ == 0) ? queCl0Ping_ : queCl0Pong_;
    LocalTensor<float> useC1 = q.template AllocTensor<float>();
    bool firstContribution = true;
    for (uint32_t kd = 0; kd < td.dk; ++kd) {
        /* 深度采样：dTmp 有符号检查（9.3） */
        const int64_t dTmp = static_cast<int64_t>(curDinIdx_) + td.padFront - static_cast<int64_t>(kd) * td.dilationD;
        if (dTmp < 0 || (td.strideD > 1 && dTmp % td.strideD != 0) ||
            dTmp >= static_cast<int64_t>(td.dout) * td.strideD) {
            continue;
        }
        const uint32_t curDoutIdx = static_cast<uint32_t>(td.strideD > 1 ? dTmp / td.strideD : dTmp);
        for (uint64_t kIdx = 0; kIdx < td.kIter; ++kIdx) {
            const uint64_t kStart = kIdx * td.baseK;
            baseUseK_ = (kIdx + 1 == td.kIter) ? td.tailK : td.baseK;
            EnsureA1(curDoutIdx, static_cast<uint32_t>(kIdx));
            EnsureB1(kd, static_cast<uint32_t>(kIdx));
            if constexpr (hasBias) {
                if (firstContribution) {
                    ComputeBTBias();
                }
            }
            /* L0 输入槽交接（范式 sync-implementation §1 稳态事件对）。
             * F-05（code-review 登记，2026-11-05）：SetFlag/WaitFlag 为 ISASI 类接口，目标代际
             * 950 可用 AscendC::Mutex::Lock/Unlock；暂缓迁移依据——本算子五路 TQue/TBuf 与事件环
             * 深度耦合（Mutex 不得与 AllocTensor/EnQue 等资源管理接口同用），且当前序列与官方
             * adv_api Conv3DBackpropInput 同构（功能正确性无风险）；重构迭代时再迁移。 */
            WaitFlag<HardEvent::M_MTE1>(l0Slot_);
            IssueLoadToA2(kStart);
            IssueLoadToB2(kStart);
            SetFlag<HardEvent::MTE1_M>(l0Slot_);
            WaitFlag<HardEvent::MTE1_M>(l0Slot_);
            IssueMmad<hasBias>(useC1, firstContribution);
            firstContribution = false;
            SetFlag<HardEvent::M_MTE1>(l0Slot_);
            l0Slot_ = (l0Slot_ + 1) % inputSlotCount_;
            /* 段末释放：父段最后一次 MTE1 读取已提交后由队列发布归还依赖（6） */
            ReleaseA1IfSegmentEnd(curDoutIdx, static_cast<uint32_t>(kIdx));
            ReleaseB1IfSegmentEnd(static_cast<uint32_t>(kIdx));
        }
    }
    if constexpr (hasBias) {
        if (!computeBiasOnce_) {
            /* 本块无实际乘加（防御分支：needCompute=true 时至少一轮 K 乘加）——纯 bias 输出 */
            ComputeForBias(useC1);
            firstContribution = false;
        }
        /* 分块模式：本块 MMAD 读取已提交后归还 BT（bias-bt-path §3） */
        if (td.isBiasFullLoad == 0 && biasBTHeldValid_) {
            biasBTQue_.FreeTensor(biasBTHeld_);
            biasBTHeldValid_ = false;
        }
    }
    if (!firstContribution) {
        q.EnQue(useC1);
        LocalTensor<float> c1v = q.template DeQue<float>();
        LoadL0c2Gm(c1v);
        q.FreeTensor(c1v);
    } else {
        /* 防御分支：needCompute_=true 时深度采样保证至少一次有效 MMAD，
         * 此路径设计上不可达，仅保证 L0C 槽成对平衡 */
        q.FreeTensor(useC1);
    }
    cSlot_ = (cSlot_ + 1) % cSlotCount_;
}

/* ================================ 无贡献块完整写回（Kernel.md §5 NoNeedComputeHandle，9.6 协议） ================ */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
template <bool hasBias>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::NoNeedComputeHandle()
{
    auto& q = (cSlot_ == 0) ? queCl0Ping_ : queCl0Pong_;
    LocalTensor<float> useC1 = q.template AllocTensor<float>();
    if constexpr (hasBias) {
        /* 带 bias：LoadL0Zero + 带 BT 的零乘积 MMAD 得 bias 广播，正常 Fixpipe 写回（9.6） */
        ComputeForBias(useC1);
    } else {
        /* 无 bias：LoadL0Zero + 零初始化 MMAD 得全零块，正常 Fixpipe 写回（9.6；
         * D-1 修复：不使用裸标量 GM 写——标量写的行级读改写与并发 Fixpipe 写回在
         * NCDHW 布局同 cache 行上竞争，实测致主计算写回丢失） */
        ComputeZero(useC1);
    }
    q.EnQue(useC1);
    LocalTensor<float> c1v = q.template DeQue<float>();
    LoadL0c2Gm(c1v);
    q.FreeTensor(c1v);
    cSlot_ = (cSlot_ + 1) % cSlotCount_;
    if constexpr (hasBias) {
        if (td_.isBiasFullLoad == 0 && biasBTHeldValid_) {
            biasBTQue_.FreeTensor(biasBTHeld_);
            biasBTHeldValid_ = false;
        }
    }
}

/* ================================ rho 零裕域覆盖写回（H/W 轴） ================================ */

/*
 * 对当前任务空间域 [spaceStart, spaceStart+spaceUse) 内的 H/W 轴零裕域位置，以
 * NoNeedComputeHandle 同通路（ComputeForBias/ComputeZero + Fixpipe）覆写 0/bias：
 *   - H 轴裕域：行 ≥ hMarginStart_ 即 M 坐标 [hMarginStart_·wi, hi·wi)——空间连续尾段；
 *   - W 轴裕域：列 ≥ wMarginStart_ 即逐行 M 坐标 [r·wi + wMarginStart_, (r+1)·wi)——
 *     周期性行尾段，逐行分段、每段按 baseM 分块；
 *   - 与任务空间域求交（空间任务划分 singleCoreM 不保证行对齐，逐段双侧裁剪）；
 *   - 每段写回走既有 NoNeedComputeHandle 帧（mStart_/baseUseM_/mCompute_ 尾块双口径
 *     + cSlot/l0Slot 轮转 + BT 分块装载/归还），与乘积写回同管道按发射序串行。
 * 无裕域轴（marginActive_=false）不触达；E ≥ Ly 时 ResolveRhoMargin 已置不活跃。
 */
template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
template <bool hasBias>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::WriteMarginZones(
    uint64_t spaceStart, uint64_t spaceUse)
{
    const Conv3DTransposeTilingData& td = td_;
    const uint64_t mTotal = static_cast<uint64_t>(td.hi) * td.wi;
    const uint64_t domLo = spaceStart;
    const uint64_t domHi = spaceStart + spaceUse;

    if (hMarginActive_) {
        /* H 轴裕域：连续尾段 [hMarginStart_·wi, hi·wi) */
        WriteMarginRange<hasBias>(static_cast<uint64_t>(hMarginStart_) * td.wi, mTotal, domLo, domHi, mTotal);
    }
    if (wMarginActive_ && wMarginStart_ < td.wi) {
        /* W 轴裕域：逐行尾段 [r·wi + wMarginStart_, (r+1)·wi) */
        for (uint64_t r = 0; r < td.hi; ++r) {
            WriteMarginRange<hasBias>(r * td.wi + wMarginStart_, (r + 1) * td.wi, domLo, domHi, mTotal);
        }
    }
}

/* 通用分段写回：[segLo, segHi) 与任务空间域 [domLo, domHi) / [0, mTotal) 求交后
 * 按 baseM 分块，逐块改写 mStart_/baseUseM_/mCompute_ 尾块双口径现场并走
 * NoNeedComputeHandle（lambda 不继承 __aicore__，展开为成员方法）。 */
template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
template <bool hasBias>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::WriteMarginRange(
    uint64_t segLo, uint64_t segHi, uint64_t domLo, uint64_t domHi, uint64_t mTotal)
{
    const Conv3DTransposeTilingData& td = td_;
    if (segLo < domLo) {
        segLo = domLo;
    }
    if (segHi > domHi) {
        segHi = domHi;
    }
    if (segHi > mTotal) {
        segHi = mTotal;
    }
    for (uint64_t p = segLo; p < segHi; p += td.baseM) {
        const uint64_t use64 = (p + td.baseM <= segHi) ? td.baseM : (segHi - p);
        mStart_ = static_cast<uint32_t>(p);
        baseUseM_ = static_cast<uint32_t>(use64);
        mCompute_ = AlignUp16L(baseUseM_);
        computeBiasOnce_ = false;
        NoNeedComputeHandle<hasBias>();
    }
}

/* ================================ A 路径（dedy：GM→L1A→L0A，Kernel.md §7） ================================ */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::EnsureA1(
    uint32_t curDoutIdx, uint32_t kIdx)
{
    const Conv3DTransposeTilingData& td = td_;
    /* 父段命中复用：segFirst 与 curDoutIdx 均未变时沿用驻留片段（范式 kernel-template §4.3） */
    const uint32_t segFirst = (kIdx / td.stepKa) * td.stepKa;
    if (segFirst == a1SegFirst_ && curDoutIdx == a1Dout_ && a1Valid_) {
        return;
    }
    FreeA1Tensor();
    a1SegFirst_ = segFirst;
    a1Dout_ = curDoutIdx;
    const uint64_t segStartK = static_cast<uint64_t>(segFirst) * td.baseK;
    const uint32_t segIters = (td.kIter - segFirst) < td.stepKa ? (td.kIter - segFirst) : td.stepKa;
    const uint32_t segUseK = static_cast<uint32_t>(static_cast<uint64_t>(segIters) * td.baseK < td.kValue - segStartK ?
                                                       static_cast<uint64_t>(segIters) * td.baseK :
                                                       td.kValue - segStartK);
    aSegStartK_ = static_cast<uint32_t>(segStartK);

    /* 父段通道范围：段起点分形对齐（K 窗口配对，HostTiling.md「K 窗口配对」） */
    const uint32_t coutIdx = static_cast<uint32_t>(segStartK / td.hkWk);
    uint32_t coutSize = segUseK / td.hkWk;
    const uint32_t coutRemain = td.coutG - coutIdx;
    coutSize = coutSize < coutRemain ? coutSize : coutRemain;
    const uint32_t coutAligned = AlignUp16L(coutSize);
    const uint32_t coutGAligned = AlignUp16L(td.coutG);
    channelSize_ = coutAligned < coutGAligned ? coutAligned : coutGAligned;
    if (channelSize_ == 0) {
        channelSize_ = CONV3DTRANSPOSE_C0;
    }

    /* W 裁剪：负反向边界先裁剪真实 dedy（9.8；范式 compute-flow §5.2.1 woEnd 带符号口径；
     * b⁻/b⁺ 已含 q+rho 折入，设备不感知 q/rho） */
    const int32_t bpLeft = td.backpropPadLeft;
    const int32_t bpRight = td.backpropPadRight;
    const uint32_t woExpandRaw = td.woExpand;
    uint32_t woStart = 0;
    uint32_t woEnd = CeilDivL(woExpandRaw, td.strideW);
    bool resetWoExpand = false;
    if (bpLeft < 0) {
        resetWoExpand = true;
        woStart = CeilDivL(static_cast<uint32_t>(-bpLeft), td.strideW);
    }
    if (bpRight < 0) {
        resetWoExpand = true;
        /* woEnd = ceilDiv(expandedW + backpropPadRight, strideW)：带符号相加，负右边界收窄
         * 有效列（对 bpRight 取绝对值会虚增 realWo，致 GM 读越界与 L1A 行宽错位——
         * 范式 compute-flow §5.2.1 修复口径，高风险复跑点，9.9 校验项 4）；
         * 合法规格恒有 expandedW+bpRight = wi+padLeft ≥ 1（转置输出几何恒等式，含 q+rho），
         * int64 相加后转无符号安全 */
        woEnd = CeilDivL(static_cast<uint64_t>(static_cast<int64_t>(woExpandRaw) + bpRight), td.strideW);
    }
    uint32_t woExpandLocal = woExpandRaw;
    uint32_t realWo = td.wo;
    if (resetWoExpand) {
        woExpandLocal = (woEnd <= woStart) ? 1u : (woEnd - woStart - 1) * td.strideW + 1u;
        realWo = woEnd > woStart ? (woEnd - woStart) : 0u;
    }
    woExpandLocal_ = woExpandLocal;

    /* stride 相位：首个真实 dedy 行在父段内的行偏移与其绝对行号 */
    const uint32_t realStart = curHoIdx_ < 0 ? 0u : static_cast<uint32_t>(curHoIdx_);
    const uint32_t hSkip =
        (td.strideH > 1 && realStart % td.strideH != 0) ? (AlignUpTo(realStart, td.strideH) - realStart) : 0u;
    const uint32_t hLoop = (curHoSize_ > hSkip) ? CeilDivL(curHoSize_ - hSkip, td.strideH) : 0u;
    const uint32_t firstRealRow = CeilDivL(realStart, td.strideH);

    const uint64_t a1DstOffset = static_cast<uint64_t>(hSkip) * woExpandLocal * CONV3DTRANSPOSE_C0;

    /* FMatrix（padList 顺序左、右、上、下；下侧取 255 哨兵，范式 instruction-mapping §4.2）
     * F-03（code-review 整改，2026-11-05 带外授权）：装载前域断言兜底——Host 已对四侧反向
     * 边界 >255 响亮拒绝（tiling 校验层），此处断言超域在调试期暴露，杜绝静默截断。 */
    const int32_t padL =
        bpLeft < 0 ? static_cast<int32_t>(CeilDivL(static_cast<uint32_t>(-bpLeft), td.strideW) * td.strideW) + bpLeft :
                     bpLeft;
    const int32_t padR =
        bpRight < 0 ? static_cast<int32_t>(woExpandRaw) + bpRight - static_cast<int32_t>((woEnd - 1) * td.strideW + 1) :
                      bpRight;
    const int32_t padT = curHoIdx_ < 0 ? -curHoIdx_ : 0;
    CONV3DTRANSPOSE_FIELD_ASSERT(padL >= 0 && padL <= 255);
    CONV3DTRANSPOSE_FIELD_ASSERT(padR >= 0 && padR <= 255);
    CONV3DTRANSPOSE_FIELD_ASSERT(padT >= 0 && padT <= 255);
    load3d_.l1H = static_cast<uint16_t>(curHoSize_);
    load3d_.l1W = static_cast<uint16_t>(woExpandLocal);
    load3d_.padList[0] = static_cast<uint8_t>(padL);
    load3d_.padList[1] = static_cast<uint8_t>(padR);
    load3d_.padList[2] = static_cast<uint8_t>(padT);
    load3d_.padList[3] = 255u;
    load3d_.channelSize = static_cast<uint16_t>(channelSize_);

    /* 槽清零后搬入真实值（Fill → PipeBarrier<PIPE_MTE2> → DataCopy → EnQue，范式 sync §5）。
     * F-06（code-review 登记，2026-11-05）：Nd2Nz/Dn2Nz 分形装载为 GM→L1A/L1B 唯一可用 API
     * （DataCopyPad 无分形变体、无等价迁移路径）；目的侧对齐由槽 FillZero 预清零 + 对齐
     * 步幅参数构造性保证，非对齐正确性缺陷不适用。后续新增非分形搬运一律 DataCopyPad。 */
    LocalTensor<T> a1Alloc = queAl1_.template AllocTensor<T>();
    FillZero(a1Alloc, td.aSlotBytes);
    const uint64_t coGlobal = groupIdx_ * td.coutG + coutIdx;
    const uint64_t dyBatchOff = kDyNd ? batchIdx_ * static_cast<uint64_t>(td.dout) * hoWo_ * td.cout :
                                        batchIdx_ * static_cast<uint64_t>(td.cout) * td.doHoWo;
    uint64_t srcBase = dyBatchOff;
    if constexpr (!kDyNd) {
        srcBase += coGlobal * td.doHoWo + static_cast<uint64_t>(curDoutIdx) * hoWo_ +
                   static_cast<uint64_t>(firstRealRow) * td.wo + woStart;
    } else {
        srcBase += static_cast<uint64_t>(curDoutIdx) * hoWo_ * td.cout +
                   static_cast<uint64_t>(firstRealRow) * td.wo * td.cout + static_cast<uint64_t>(woStart) * td.cout +
                   coGlobal;
    }
    if (hLoop > 0) {
        if (td.strideW * td.strideH > 1 || realWo != td.wo || resetWoExpand) {
            /* 逐行搬入：目的步幅表达 W/H 前放大（范式 instruction-mapping §2.2 参数表） */
            if constexpr (!kDyNd) {
                Dn2NzParams p;
                p.dnNum = static_cast<uint16_t>(hLoop);
                p.nValue = static_cast<uint16_t>(realWo);
                p.dValue = coutSize;
                p.srcDnMatrixStride = td.wo;
                p.srcDValue = td.doHoWo;
                p.dstNzC0Stride = static_cast<uint16_t>(curHoSize_ * woExpandLocal);
                p.dstNzNStride = static_cast<uint16_t>(td.strideW);
                p.dstNzMatrixStride = static_cast<uint32_t>(td.strideH) * woExpandLocal * CONV3DTRANSPOSE_C0;
                DataCopy(a1Alloc[a1DstOffset], dyGm_[srcBase], p);
            } else {
                Nd2NzParams p;
                p.ndNum = static_cast<uint16_t>(hLoop);
                p.nValue = static_cast<uint16_t>(realWo);
                p.dValue = coutSize;
                p.srcNdMatrixStride = static_cast<uint64_t>(td.wo) * td.cout;
                p.srcDValue = td.cout;
                p.dstNzC0Stride = static_cast<uint16_t>(curHoSize_ * woExpandLocal);
                p.dstNzNStride = static_cast<uint16_t>(td.strideW);
                p.dstNzMatrixStride = static_cast<uint32_t>(td.strideH) * woExpandLocal * CONV3DTRANSPOSE_C0;
                DataCopy(a1Alloc[a1DstOffset], dyGm_[srcBase], p);
            }
        } else {
            /* 无插零/裁剪：连续行合成单矩阵 */
            if constexpr (!kDyNd) {
                Dn2NzParams p;
                p.dnNum = 1;
                p.nValue = static_cast<uint16_t>(hLoop * td.wo);
                p.dValue = coutSize;
                p.srcDnMatrixStride = 0;
                p.srcDValue = td.doHoWo;
                p.dstNzC0Stride = static_cast<uint16_t>(curHoSize_ * woExpandLocal);
                p.dstNzNStride = 1;
                p.dstNzMatrixStride = 0;
                DataCopy(a1Alloc[a1DstOffset], dyGm_[srcBase], p);
            } else {
                Nd2NzParams p;
                p.ndNum = 1;
                p.nValue = static_cast<uint16_t>(hLoop * td.wo);
                p.dValue = coutSize;
                p.srcNdMatrixStride = 0;
                p.srcDValue = td.cout;
                p.dstNzC0Stride = static_cast<uint16_t>(curHoSize_ * woExpandLocal);
                p.dstNzNStride = 1;
                p.dstNzMatrixStride = 0;
                DataCopy(a1Alloc[a1DstOffset], dyGm_[srcBase], p);
            }
        }
    }
    queAl1_.EnQue(a1Alloc);
    a1v_ = queAl1_.template DeQue<T>();
    a1Valid_ = true;
}

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::ReleaseA1IfSegmentEnd(
    uint32_t curDoutIdx, uint32_t kIdx)
{
    const Conv3DTransposeTilingData& td = td_;
    const uint32_t segFirst = (kIdx / td.stepKa) * td.stepKa;
    const uint32_t segIters = (td.kIter - segFirst) < td.stepKa ? (td.kIter - segFirst) : td.stepKa;
    if (kIdx + 1 == segFirst + segIters && a1Valid_) {
        (void)curDoutIdx;
        queAl1_.FreeTensor(a1v_);
        a1Valid_ = false;
    }
}

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::FreeA1Tensor()
{
    if (a1Valid_) {
        queAl1_.FreeTensor(a1v_);
        a1Valid_ = false;
    }
}

/* L1A→L0A 窗口提取（Load3D 硬件 im2col；H/W 步长固定 1，dilation 保留；
 * SDK 版本分支按范式 instruction-mapping §4.2/§4.3 成对选择 Repeat 参数与 LoadData 重载） */
template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::IssueLoadToA2(
    uint64_t kStart)
{
    const Conv3DTransposeTilingData& td = td_;
    const uint32_t kLocalA = static_cast<uint32_t>(kStart - aSegStartK_);
    load3d_.kStartPt = static_cast<uint16_t>(kLocalA);
    load3d_.kExtension = static_cast<uint16_t>(baseUseK_);
    load3d_.mExtension = static_cast<uint16_t>(baseUseM_);
    /* F-03 整改：mStartPt 域断言兜底（Host 已限 wi 落入 uint16 域，超域调试期暴露） */
    const uint32_t mStartPtVal = mStart_ % td.wi;
    CONV3DTRANSPOSE_FIELD_ASSERT(mStartPtVal <= 65535u);
    load3d_.mStartPt = static_cast<uint16_t>(mStartPtVal);
    LocalTensor<T> a2 = a2Base_;
    if (l0Slot_ == 1) {
        a2 = a2[l0aHalfElem_];
    }
#if defined(ASC_DEVKIT_VERSION_NUM) && ASC_DEVKIT_VERSION_NUM >= 90000000
    LoadDataRepeatParamWithStride rp = {0, 1, 0, static_cast<uint16_t>(CeilDiv16L(baseUseM_))};
    SetLoadDataRepeatWithStride(rp);
    LoadDataWithStride(a2, a1v_, load3d_);
#else
    LoadDataRepeatParam rp = {0, 1, 0};
    SetLoadDataRepeat(rp);
    LoadData(a2, a1v_, load3d_);
#endif
}

/* 零填充（Fill 协议：InitConstValue 后 PipeBarrier<PIPE_MTE2> 保证零填充先于真实搬入；
 * 范式 sync-implementation §5 与 instruction-mapping §2.2） */
template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::FillZero(
    const LocalTensor<T>& buf, uint32_t slotBytes)
{
    InitConstValueParams<T> z;
    z.repeatTimes = 1;
    /* F-02 具名：DataBlock 32B 架构不变量 */
    z.blockNum = static_cast<uint16_t>(slotBytes / CONV3DTRANSPOSE_DATA_BLOCK_BYTES);
    z.dstGap = 0;
    z.initValue = static_cast<T>(0);
    InitConstValue(buf, z);
    PipeBarrier<PIPE_MTE2>();
}

/* ================================ B 路径（weight：GM→L1B→L0B） ================================
 * 共享帧（Kernel.md §7「B 路径」帧表）：装载触发 = 父段身份四元组
 * (segFirst, kd, cinStart_, groupIdx_) 任一变化或 b1Valid_=false 时重装、全等沿用；
 * 父段参数 / 通道量 / 槽清零 / 队列交接 / 归还（enableFullLoad=1 时段末跳过，9.7）跨分支共享。
 * GM→L1B 的 srcBase 与 Nd2Nz/Dn2Nz 参数、L1B→L0B 的 LoadData2DParamsV2 字段按
 * LOADB2 分支臂构造（DESIGN-BRANCH-16777217/8/9.md §5.2 各自的参数表与完整代码）。
 */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::EnsureB1(uint32_t kd,
                                                                                                        uint32_t kIdx)
{
    const Conv3DTransposeTilingData& td = td_;
    /* 父段命中复用：身份四元组 (segFirst, kd, cinStart_, groupIdx_) 全等即沿用（enableFullLoad
     * 驻留语义，Kernel.md 9.7） */
    const uint32_t segFirst = (kIdx / td.stepKb) * td.stepKb;
    if (segFirst == b1SegFirst_ && b1Valid_ && b1IdKd_ == kd && b1IdCin_ == cinStart_ && b1IdGroup_ == groupIdx_) {
        return;
    }
    FreeB1Tensor();
    b1SegFirst_ = segFirst;
    b1IdKd_ = kd;
    b1IdCin_ = cinStart_;
    b1IdGroup_ = groupIdx_;
    const uint64_t segStartK = static_cast<uint64_t>(segFirst) * td.baseK;
    const uint32_t segIters = (td.kIter - segFirst) < td.stepKb ? (td.kIter - segFirst) : td.stepKb;
    const uint32_t segUseK = static_cast<uint32_t>(static_cast<uint64_t>(segIters) * td.baseK < td.kValue - segStartK ?
                                                       static_cast<uint64_t>(segIters) * td.baseK :
                                                       td.kValue - segStartK);
    bSegStartK_ = static_cast<uint32_t>(segStartK);

    const uint32_t coutIdx = static_cast<uint32_t>(segStartK / td.hkWk);
    uint32_t coutSize = segUseK / td.hkWk;
    const uint32_t coutRemain = td.coutG - coutIdx;
    coutSize = coutSize < coutRemain ? coutSize : coutRemain;
    curCout1Cout0Size_ = AlignUp16L(coutSize);
    curLoadKbl1_ = curCout1Cout0Size_ * td.hkWk;

    const uint32_t cinRemain = td.cinG - cinStart_;
    const uint32_t cinSize = td.baseN < cinRemain ? td.baseN : cinRemain;

    const uint64_t coGlobal = groupIdx_ * td.coutG + coutIdx;
    uint64_t srcBase = 0;
    if constexpr (LOADB2 == TPL_TRANSPOSE_AND_REVERSE) {
        /* 转置加逆序（tilingKey 16777217，DESIGN-BRANCH-16777217.md §5.2 参数表：
         * NCDHW/NDHWC 为范式与直调参考已验证值，DHWCN 为同族注册工程修正值——co 最快轴） */
        if constexpr (FMT_FILTER == CONV3DTRANSPOSE_FMT_NCDHW) {
            /* filter NCDHW：(Cout, CinG, kD, kH, kW)——固定 (co,kd) 下 (kh,kw) 连续 */
            srcBase = coGlobal * td.cinG * td.dkHkWk + static_cast<uint64_t>(cinStart_) * td.dkHkWk +
                      static_cast<uint64_t>(kd) * td.hkWk;
        } else if constexpr (FMT_FILTER == CONV3DTRANSPOSE_FMT_NDHWC) {
            /* filter NDHWC：(Cout, kD, kH, kW, CinG)——固定 (co,kd) 下 ci 连续
             * （本 key 进入几何：最终 baseN < hkWk，NDHWC 谓词不成立） */
            srcBase = coGlobal * td.cinG * td.dkHkWk + static_cast<uint64_t>(kd) * td.hkWk * td.cinG + cinStart_;
        } else {
            /* filter DHWCN：(kD, kH, kW, CinG, Cout)——co 为最内连续轴，恒路由本 key */
            srcBase = (static_cast<uint64_t>(kd) * td.hkWk * td.cinG + cinStart_) * td.cout + coGlobal;
        }
    } else if constexpr (LOADB2 == TPL_TRANSPOSE_ONLY) {
        /* 仅转置（tilingKey 16777219，核体积 1，DESIGN-BRANCH-16777219.md §5.2）：
         * NCDHW/NDHWC 在核体积 1 下同为 (co, ci) 行主序；DHWCN 为 (ci, co) 列主序 */
        if constexpr (FMT_FILTER == CONV3DTRANSPOSE_FMT_NCDHW || FMT_FILTER == CONV3DTRANSPOSE_FMT_NDHWC) {
            srcBase = coGlobal * td.cinG + cinStart_;
        } else { /* DHWCN：Dn2Nz 互补结构（co/ci 互换） */
            srcBase = static_cast<uint64_t>(cinStart_) * td.cout + coGlobal;
        }
    } else {
        /* 仅逆序（tilingKey 16777218，DESIGN-BRANCH-16777218.md §5.2(c)）；
         * DHWCN filter 两仅逆序谓词恒不成立、永不路由本 key（BranchRoute.md）——
         * 该 (LOADB2=2 × DHWCN) 实例化组合在设计路由域外，此处按 NCDHW 同构装载
         * 保证模板全实例化可编译，运行期不可达。 */
        if constexpr (FMT_FILTER == CONV3DTRANSPOSE_FMT_NDHWC) {
            srcBase = coGlobal * td.cinG * td.dkHkWk + static_cast<uint64_t>(kd) * td.hkWk * td.cinG + cinStart_;
        } else {
            srcBase = coGlobal * td.cinG * td.dkHkWk + static_cast<uint64_t>(cinStart_) * td.dkHkWk +
                      static_cast<uint64_t>(kd) * td.hkWk;
        }
    }

    /* 槽清零：装载前对整个槽 FillZero（尾部补零职责：转置加逆序补 co 尾、仅逆序补 ci 尾，
     * 另一条分形轴由转换补齐——DESIGN-BRANCH-16777217/8.md §5.2 槽清零职责）；
     * bSlotBytes 覆盖 stepKb·baseK 的预留 K 跨度（≥ 实际 curLoadKbl1_），预留超出段
     * 同样由 FillZero 保持 0（Load2D 只按 curLoadKbl1_ 寻址，清零为防御性不变式） */
    LocalTensor<T> b1Alloc = queBl1_.template AllocTensor<T>();
    FillZero(b1Alloc, td.bSlotBytes);
    if (coutSize > 0 && cinSize > 0) {
        if constexpr (LOADB2 == TPL_TRANSPOSE_AND_REVERSE) {
            /* 转置加逆序：Dn2Nz（NCDHW）/ Nd2Nz（NDHWC）/ Dn2Nz（DHWCN）→ 同一 L1B 父布局
             * (Cin1, Hk, Wk, Cout1, Cout0, Cin0)（DESIGN-BRANCH-16777217.md §5.2）。
             * 目的步幅按源分形结构分两型：NCDHW/NDHWC 臂（co 矩阵 × 核高宽行 × ci 列）；
             * DHWCN 臂（window 矩阵 × co 行 × ci 列，co 为 DHWCN 最内连续轴）——两型目的
             * 地址式逐地址一致，L1B→L0B 消费端（LoadData2D 逆序窗口寻址）无需分支。 */
            if constexpr (FMT_FILTER == CONV3DTRANSPOSE_FMT_NCDHW) {
                Dn2NzParams p; /* wGm_(GM, T) → L1B 槽 (T) */
                p.dnNum = static_cast<uint16_t>(coutSize);
                p.nValue = static_cast<uint16_t>(td.hkWk);
                p.dValue = cinSize;
                p.srcDnMatrixStride = static_cast<uint64_t>(td.cinG) * td.dkHkWk;
                p.srcDValue = td.dkHkWk;
                p.dstNzC0Stride = static_cast<uint16_t>(td.hkWk * curCout1Cout0Size_);
                p.dstNzNStride = static_cast<uint16_t>(curCout1Cout0Size_);
                p.dstNzMatrixStride = CONV3DTRANSPOSE_C0;
                DataCopy(b1Alloc, wGm_[srcBase], p);
            } else if constexpr (FMT_FILTER == CONV3DTRANSPOSE_FMT_NDHWC) {
                Nd2NzParams p; /* wGm_(GM, T) → L1B 槽 (T) */
                p.ndNum = static_cast<uint16_t>(coutSize);
                p.nValue = static_cast<uint16_t>(td.hkWk);
                p.dValue = cinSize;
                p.srcNdMatrixStride = static_cast<uint64_t>(td.cinG) * td.dkHkWk;
                p.srcDValue = td.cinG;
                p.dstNzC0Stride = static_cast<uint16_t>(td.hkWk * curCout1Cout0Size_);
                p.dstNzNStride = static_cast<uint16_t>(curCout1Cout0Size_);
                p.dstNzMatrixStride = CONV3DTRANSPOSE_C0;
                DataCopy(b1Alloc, wGm_[srcBase], p);
            } else {
                /* DHWCN：Dn2Nz——window 矩阵 × co 连续行 × ci 跨 cout 列；目的侧逐地址落统一父布局
                 * （同族注册工程 sf_001 修正值：Nd2Nz 把 ci 放在隐含连续的 D 轴与 DHWCN 实际轴序
                 * 不符，为精度错位根因，修正为 Dn2Nz 结构后通过；本算子沿用修正值） */
                Dn2NzParams p; /* wGm_(GM, T) → L1B 槽 (T) */
                p.dnNum = static_cast<uint16_t>(td.hkWk);
                p.nValue = static_cast<uint16_t>(coutSize);
                p.dValue = static_cast<uint16_t>(cinSize);
                p.srcDnMatrixStride = static_cast<uint64_t>(td.cinG) * td.cout;
                p.srcDValue = static_cast<uint64_t>(td.cout);
                p.dstNzC0Stride = static_cast<uint16_t>(td.hkWk * curCout1Cout0Size_);
                p.dstNzNStride = 1;
                p.dstNzMatrixStride = static_cast<uint32_t>(curCout1Cout0Size_) * CONV3DTRANSPOSE_C0;
                DataCopy(b1Alloc, wGm_[srcBase], p);
            }
        } else if constexpr (LOADB2 == TPL_TRANSPOSE_ONLY) {
            /* 仅转置：单矩阵装载（核体积 1），NCDHW/NDHWC 用 Nd2Nz、DHWCN 用互补 Dn2Nz
             * → 同一 L1B (Cin1, Cout1, Cout0, Cin0)（DESIGN-BRANCH-16777219.md §5.2） */
            if constexpr (FMT_FILTER == CONV3DTRANSPOSE_FMT_NCDHW || FMT_FILTER == CONV3DTRANSPOSE_FMT_NDHWC) {
                Nd2NzParams p;
                p.ndNum = 1;
                p.nValue = static_cast<uint16_t>(coutSize);
                p.dValue = cinSize;
                p.srcNdMatrixStride = 0;
                p.srcDValue = td.cinG;
                p.dstNzC0Stride = static_cast<uint16_t>(curCout1Cout0Size_);
                p.dstNzNStride = 1;
                p.dstNzMatrixStride = 0;
                DataCopy(b1Alloc, wGm_[srcBase], p);
            } else { /* DHWCN：Dn2Nz（n=co 行在 DHWCN 源中 co 连续、d=ci 步幅 cout） */
                Dn2NzParams p;
                p.dnNum = 1;
                p.nValue = static_cast<uint16_t>(coutSize);
                p.dValue = cinSize;
                p.srcDnMatrixStride = 0;
                p.srcDValue = td.cout;
                p.dstNzC0Stride = static_cast<uint16_t>(curCout1Cout0Size_);
                p.dstNzNStride = 1;
                p.dstNzMatrixStride = 0;
                DataCopy(b1Alloc, wGm_[srcBase], p);
            }
        } else {
            /* 仅逆序：NCDHW 整行逆序装载（Dn2Nz）/ NDHWC 紧凑通道行（Dn2Nz）
             * → L1B (Cout1, Hk, Wk, Cin1, Cin0, Cout0)（DESIGN-BRANCH-16777218.md §5.2(c)；
             * DHWCN 组合设计不可达，按 NCDHW 同构） */
            const uint32_t cinAligned = AlignUp16L(cinSize);
            if constexpr (FMT_FILTER == CONV3DTRANSPOSE_FMT_NDHWC) {
                Dn2NzParams p;
                p.dnNum = static_cast<uint16_t>(td.hkWk);
                p.dValue = coutSize;
                p.nValue = static_cast<uint16_t>(cinSize);
                p.srcDnMatrixStride = td.cinG;
                p.srcDValue = static_cast<uint64_t>(td.cinG) * td.dkHkWk;
                p.dstNzC0Stride = static_cast<uint16_t>(td.hkWk * cinAligned);
                p.dstNzNStride = 1;
                p.dstNzMatrixStride = cinAligned * CONV3DTRANSPOSE_C0;
                DataCopy(b1Alloc, wGm_[srcBase], p);
            } else {
                Dn2NzParams p;
                p.dnNum = static_cast<uint16_t>(cinSize);
                p.nValue = static_cast<uint16_t>(td.hkWk);
                p.dValue = coutSize;
                p.srcDnMatrixStride = td.dkHkWk;
                p.srcDValue = static_cast<uint64_t>(td.cinG) * td.dkHkWk;
                p.dstNzC0Stride = static_cast<uint16_t>(td.hkWk * cinAligned);
                p.dstNzNStride = static_cast<uint16_t>(cinAligned);
                p.dstNzMatrixStride = CONV3DTRANSPOSE_C0;
                DataCopy(b1Alloc, wGm_[srcBase], p);
            }
        }
    }
    queBl1_.EnQue(b1Alloc);
    b1v_ = queBl1_.template DeQue<T>();
    b1Valid_ = true;
}

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::ReleaseB1IfSegmentEnd(
    uint32_t kIdx)
{
    const Conv3DTransposeTilingData& td = td_;
    if (td.enableFullLoad != 0) {
        return; /* 驻留：跨输出块/任务复用，身份变化或结束时释放（Kernel.md 9.7） */
    }
    const uint32_t segFirst = (kIdx / td.stepKb) * td.stepKb;
    const uint32_t segIters = (td.kIter - segFirst) < td.stepKb ? (td.kIter - segFirst) : td.stepKb;
    if (kIdx + 1 == segFirst + segIters && b1Valid_) {
        queBl1_.FreeTensor(b1v_);
        b1Valid_ = false;
    }
}

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::FreeB1Tensor()
{
    if (b1Valid_) {
        queBl1_.FreeTensor(b1v_);
        b1Valid_ = false;
    }
}

/* L1B→L0B 装载指令（LoadData2DParamsV2；K 配对约束：L0B 矩阵中对应 K 元素与 L0A
 * 指向同一 dedy 通道与核高宽位置，Kernel.md 9.2）；三种准备方式最终形成同一
 * ZN(K,N) 片段 (K1, Cin1, Cin0, k0)。 */
template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::IssueLoadToB2(
    uint64_t kStart)
{
    const Conv3DTransposeTilingData& td = td_;
    LocalTensor<T> b2 = b2Base_;
    if (l0Slot_ == 1) {
        b2 = b2[l0bHalfElem_];
    }
    const uint32_t kLocalB = static_cast<uint32_t>(kStart - bSegStartK_);
    const uint32_t bKStart = kLocalB / CONV3DTRANSPOSE_C0;
    const uint32_t curKRepeat = CeilDiv16L(baseUseK_);
    if constexpr (LOADB2 == TPL_TRANSPOSE_AND_REVERSE) {
        /* 转置加逆序（tilingKey 16777217，DESIGN-BRANCH-16777217.md §5.2；ifTranspose=true
         * 由 Init 编译期设置，此处不改写）：每个 K 分形一次 LoadData，读全部 blockBaseN_
         * 个 dedx 通道分形；核空间逆序（rot180 空间分量）由源偏移 hkWk−1−r%hkWk 表达，
         * 深度分量 rot180 由 kd 采样与 srcBase 的原 kd 坐标承载（9.3，不重复反转） */
        load2dv2_.mStartPosition = 0;
        load2dv2_.kStartPosition = 0;
        load2dv2_.mStep = 1;
        load2dv2_.kStep = static_cast<uint16_t>(blockBaseN_);
        load2dv2_.srcStride = static_cast<int32_t>(curLoadKbl1_ / CONV3DTRANSPOSE_C0);
        load2dv2_.dstStride = 1;
        uint32_t dstOffset = 0;
        for (uint32_t i = 0; i < curKRepeat; ++i) {
            const uint32_t r = bKStart + i;
            const uint32_t idxC1out = r / td.hkWk;
            const uint32_t srcOffset =
                idxC1out * BLOCK_SIZE_ELEM + (td.hkWk - 1 - (r % td.hkWk)) * curCout1Cout0Size_ * CONV3DTRANSPOSE_C0;
            LoadData(b2[dstOffset], b1v_[srcOffset], load2dv2_); /* b1v_(L1B, T) → b2(L0B, T) */
            dstOffset += blockBaseN_ * BLOCK_SIZE_ELEM;
        }
    } else if constexpr (LOADB2 == TPL_TRANSPOSE_ONLY) {
        /* 仅转置（tilingKey 16777219，ifTranspose=true）：K 分形 = cout1 分形，源偏移按
         * 分形正序递增（核体积 1，空间逆序为恒等；DESIGN-BRANCH-16777219.md §5.2） */
        load2dv2_.mStartPosition = 0;
        load2dv2_.kStartPosition = 0;
        load2dv2_.mStep = 1;
        load2dv2_.kStep = static_cast<uint16_t>(blockBaseN_);
        load2dv2_.srcStride = static_cast<int32_t>(curLoadKbl1_ / CONV3DTRANSPOSE_C0);
        load2dv2_.dstStride = 1;
        uint32_t dstOffset = 0;
        for (uint32_t i = 0; i < curKRepeat; ++i) {
            const uint32_t idxC1out = bKStart + i;
            const uint32_t srcOffset = idxC1out * BLOCK_SIZE_ELEM;
            LoadData(b2[dstOffset], b1v_[srcOffset], load2dv2_);
            dstOffset += blockBaseN_ * BLOCK_SIZE_ELEM;
        }
    } else {
        /* 仅逆序（tilingKey 16777218，ifTranspose=false，三 key 唯一）：按 Cout1 窗口分段
         * 提交，负 srcStride 沿 K 逆序读取——mStartPosition 的 (windowEnd−1−first) 因子把
         * 起点放到本 Cout1 窗口内被需求位置的逆序对侧，等价 H/W 两轴各自逆序
         * （rot180 核空间分量；DESIGN-BRANCH-16777218.md §5.2(d)） */
        const uint32_t bKEnd = bKStart + CeilDiv16L(baseUseK_);
        load2dv2_.mStep = static_cast<uint16_t>(blockBaseN_);
        load2dv2_.srcStride = -static_cast<int32_t>(blockBaseN_);
        load2dv2_.dstStride = static_cast<uint16_t>(blockBaseN_);
        uint32_t dstOffset = 0;
        const uint32_t wBegin = bKStart / td.hkWk;
        const uint32_t wEnd = CeilDivL(bKEnd, td.hkWk);
        for (uint32_t co1w = wBegin; co1w < wEnd; ++co1w) {
            const uint32_t windowBegin = co1w * td.hkWk;
            const uint32_t windowEnd = windowBegin + td.hkWk;
            const uint32_t first = bKStart > windowBegin ? bKStart : windowBegin;
            const uint32_t last = bKEnd < windowEnd ? bKEnd : windowEnd;
            load2dv2_.kStartPosition = windowBegin;
            load2dv2_.mStartPosition = (windowEnd - 1 - first) * blockBaseN_;
            load2dv2_.kStep = static_cast<uint16_t>(last - first);
            LoadData(b2[dstOffset], b1v_[0], load2dv2_);
            dstOffset += (last - first) * blockBaseN_ * BLOCK_SIZE_ELEM;
        }
    }
}

/* ================================ bias BT 通路（Kernel.md §7 CopyInBrc，跨分支共享） ================================
 */

/* GM→L1（DataCopyPad，含 64B 覆盖预置零）→ EnQue/DeQue；调用方决定持有或复制后归还 */
template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::LoadBiasToL1(
    uint64_t gmStart, uint32_t biasUse)
{
    LocalTensor<BIAST> b1 = biasL1Que_.template AllocTensor<BIAST>();
    const uint32_t validBytes = biasUse * sizeof(BIAST);
    /* BT 按 64B 覆盖读取；GM→L1 的 DataCopyPad 只补到 32B——32B 块数为奇数时先置零
     * 将被读取的 64B 对齐区域（bias-bt-path §2），Fill 与 DataCopyPad 间按 MTE2 排序 */
    if (((validBytes + CONV3DTRANSPOSE_DATA_BLOCK_BYTES - 1u) / CONV3DTRANSPOSE_DATA_BLOCK_BYTES) % 2u ==
        1u) { /* F-02 具名 */
        InitConstValueParams<BIAST> z;
        z.repeatTimes = 1;
        z.blockNum = static_cast<uint16_t>((validBytes + CONV3DTRANSPOSE_BT_COVER_BYTES - 1u) /
                                           CONV3DTRANSPOSE_BT_COVER_BYTES * 2u); /* F-02 具名 */
        z.dstGap = 0;
        z.initValue = static_cast<BIAST>(0);
        InitConstValue(b1, z);
        PipeBarrier<PIPE_MTE2>();
    }
    DataCopyParams cp;
    cp.blockCount = 1;
    cp.blockLen = static_cast<uint16_t>(validBytes); /* 字节（bias-bt-path §2） */
    cp.srcStride = 0;
    cp.dstStride = 0;
    DataCopyPadParams pp;
    pp.isPad = true;
    pp.leftPadding = 0;
    pp.rightPadding = static_cast<uint8_t>(AlignUpTo(validBytes, CONV3DTRANSPOSE_DATA_BLOCK_BYTES) / sizeof(BIAST) -
                                           biasUse); /* F-02 具名 */
    pp.paddingValue = 0;
    DataCopyPad(b1, biasGm_[gmStart], cp, pp);
    biasL1Que_.EnQue(b1);
    biasL1Held_ = biasL1Que_.template DeQue<BIAST>();
    biasL1HeldValid_ = true;
}

/* L1→BT（DataCopy，32B 块计）→ EnQue/DeQue，持有 BT */
template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::CopyBiasL1ToBT(
    uint32_t biasUse)
{
    const uint32_t validBytes = biasUse * sizeof(BIAST);
    LocalTensor<float> bt = biasBTQue_.template AllocTensor<float>();
    DataCopyParams cp;
    cp.blockCount = 1;
    /* F-02 具名 + F-06 登记：blockLen 单位为 32B 块数（非字节），恒 64B 对齐偶数块（源槽
     * 64B 覆盖区已预置零）；DataCopyParams 与 DataCopyExtParams 的 blockLen 单位差异是
     * 条款列出的最常见混淆源，防后续维护按字节口径误改 */
    cp.blockLen =
        static_cast<uint16_t>((validBytes + CONV3DTRANSPOSE_BT_COVER_BYTES - 1u) / CONV3DTRANSPOSE_BT_COVER_BYTES * 2u);
    cp.srcStride = 0;
    cp.dstStride = 0;
    /* BIAST=float 时为同型拷贝（直调参考实现同式）；BIAST=half 时为 L1→C2 的类型
     * 拓宽拷贝（内置基线 conv_bp_sub_func.h FullLoadBias/LoadBiasToBT 同构），
     * 重载存在性与覆盖义务见 Kernel.md 9.9 校验项 9 */
    DataCopy(bt, biasL1Held_, cp);
    biasBTQue_.EnQue(bt);
    biasBTHeld_ = biasBTQue_.template DeQue<float>();
    biasBTHeldValid_ = true;
}

/* 全量装载：任务通道段 GM→L1→BT，双持有（身份变化或任务结束时释放） */
template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::FullLoadBias()
{
    LoadBiasToL1(biasTaskStart_, biasTaskUse_);
    CopyBiasL1ToBT(biasTaskUse_);
    biasHeldStart_ = biasTaskStart_;
}

/* 分块装载：当前 N 子块 GM→L1→BT，复制到 BT 后归还 L1（bias-bt-path §3） */
template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::LoadBiasToBT()
{
    const uint64_t gmStart = groupIdx_ * td_.cinG + cinStart_;
    LoadBiasToL1(gmStart, baseUseN_);
    CopyBiasL1ToBT(baseUseN_);
    biasL1Que_.FreeTensor(biasL1Held_);
    biasL1HeldValid_ = false;
}

/* 当前输出块的 BT 就绪（范式 kernel-template §2.2：首乘加前调用）：
 * 全量模式取任务通道段内偏移（防御路径先释放可能持有的旧句柄再整备）；
 * 分块模式逐 N 子块装载，BT 偏移 0 */
template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::ComputeBTBias()
{
    if (td_.isBiasFullLoad != 0) {
        if (!biasBTHeldValid_) {
            FreeBiasTensor();
            FullLoadBias();
        }
        biasOffset_ = cinStart_ - taskChannelStart_;
    } else {
        if (biasBTHeldValid_) {
            biasBTQue_.FreeTensor(biasBTHeld_);
            biasBTHeldValid_ = false;
        }
        LoadBiasToBT();
        biasOffset_ = 0;
    }
}

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::FreeBiasTensor()
{
    if (biasBTHeldValid_) {
        biasBTQue_.FreeTensor(biasBTHeld_);
        biasBTHeldValid_ = false;
    }
    if (biasL1HeldValid_) {
        biasL1Que_.FreeTensor(biasL1Held_);
        biasL1HeldValid_ = false;
    }
    biasHeldStart_ = BIAS_HELD_INVALID;
}

/* ================================ 无贡献块零通路（Kernel.md §8，9.6 完整写回协议） ================================ */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::EnsureZeroL1Held()
{
    /* 零窗口 L1 持有（ComputeForBias/ComputeZero 共用）：无有效驻留时申请 A1/B1 槽、
     * FillZero 后持有；身份置失效（0xFFFFFFFF 哨兵），后续 EnsureA1/EnsureB1 按新段重载。
     * 窗口只读补零区/积恒零，槽内容不参与结果（B 侧 FillZero 兼防未定义 L1 内容中的
     * 非有限值进入 0*x 乘积）。 */
    if (!a1Valid_) {
        LocalTensor<T> a1 = queAl1_.template AllocTensor<T>();
        FillZero(a1, td_.aSlotBytes);
        queAl1_.EnQue(a1);
        a1v_ = queAl1_.template DeQue<T>();
        a1Valid_ = true;
        a1SegFirst_ = 0xFFFFFFFFu;
        a1Dout_ = 0xFFFFFFFFu;
    }
    if (!b1Valid_) {
        LocalTensor<T> b1 = queBl1_.template AllocTensor<T>();
        FillZero(b1, td_.bSlotBytes);
        queBl1_.EnQue(b1);
        b1v_ = queBl1_.template DeQue<T>();
        b1Valid_ = true;
        b1SegFirst_ = 0xFFFFFFFFu;
        b1IdKd_ = 0xFFFFFFFFu;
        b1IdCin_ = 0xFFFFFFFFu;
        b1IdGroup_ = 0xFFFFFFFFu;
    }
}

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::ComputeForBias(
    LocalTensor<float>& useC1)
{
    /* 无贡献块的纯 bias 输出（bias-bt-path §5）：LoadL0Zero 全补零窗口准备 L0A/L0B
     * （A 侧窗口全部落入上侧补零区、B 侧槽内容与零乘积无关），带 BT 的零乘积 MMAD
     * 得 bias 广播，K 临时取 C0；随后按正常 M→F 协议 Fixpipe 写回。 */
    ComputeBTBias();
    EnsureZeroL1Held();
    WaitFlag<HardEvent::M_MTE1>(l0Slot_);
    IssueLoadL0ZeroA2();
    IssueLoadL0ZeroB2();
    SetFlag<HardEvent::MTE1_M>(l0Slot_);
    WaitFlag<HardEvent::MTE1_M>(l0Slot_);
    LocalTensor<T> a2 = a2Base_;
    LocalTensor<T> b2 = b2Base_;
    if (l0Slot_ == 1) {
        a2 = a2[l0aHalfElem_];
        b2 = b2[l0bHalfElem_];
    }
    MmadParams mp;
    mp.m = static_cast<uint16_t>(mCompute_);
    mp.n = static_cast<uint16_t>(baseUseN_);
    mp.k = static_cast<uint16_t>(CONV3DTRANSPOSE_C0); /* 纯 bias MMAD 的 K 临时取 C0 */
    mp.unitFlag = 0;
    mp.kDirectionAlign = false;
    mp.cmatrixSource = true;
    mp.cmatrixInitVal = false;
    Mmad(useC1, a2, b2, biasBTHeld_[biasOffset_], mp);
    computeBiasOnce_ = true;
    SetFlag<HardEvent::M_MTE1>(l0Slot_);
    l0Slot_ = (l0Slot_ + 1) % inputSlotCount_;
}

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::ComputeZero(
    LocalTensor<float>& useC1)
{
    /* 无贡献块的零值输出（ComputeForBias 的无 bias 变体，D-1 修复）：LoadL0Zero 全补零
     * 窗口准备 L0A/L0B（A 侧窗口全部落入上侧补零区 → 乘积恒零），零初始化 MMAD
     * （cmatrixInitVal=true）得全零块，K 临时取 C0；随后按正常 M→F 协议 Fixpipe 写回
     * ——写回通道与主计算一致，消除裸标量 GM 写（见 9.6 修复登记）。 */
    EnsureZeroL1Held();
    WaitFlag<HardEvent::M_MTE1>(l0Slot_);
    IssueLoadL0ZeroA2();
    IssueLoadL0ZeroB2();
    SetFlag<HardEvent::MTE1_M>(l0Slot_);
    WaitFlag<HardEvent::MTE1_M>(l0Slot_);
    LocalTensor<T> a2 = a2Base_;
    LocalTensor<T> b2 = b2Base_;
    if (l0Slot_ == 1) {
        a2 = a2[l0aHalfElem_];
        b2 = b2[l0bHalfElem_];
    }
    MmadParams mp;
    mp.m = static_cast<uint16_t>(mCompute_);
    mp.n = static_cast<uint16_t>(baseUseN_);
    mp.k = static_cast<uint16_t>(CONV3DTRANSPOSE_C0); /* 零输出 MMAD 的 K 临时取 C0 */
    mp.unitFlag = 0;
    mp.kDirectionAlign = false;
    mp.cmatrixSource = false;
    mp.cmatrixInitVal = true; /* 零初始化 L0C 后累加零窗口乘积 → 全零块 */
    Mmad(useC1, a2, b2, mp);
    SetFlag<HardEvent::M_MTE1>(l0Slot_);
    l0Slot_ = (l0Slot_ + 1) % inputSlotCount_;
}

/* LoadL0Zero 两路装载（bias-bt-path §5 参数表）：A 侧 l1H=1/l1W=32、filter 1×1、
 * padList{0,0,255,0}（网格 H 扩至 256 行全部落入上侧补零区）、channelSize=C0、
 * mExtension=alignUp(baseM,16)、kExtension=C0；B 侧单次 Load2D 读取槽首 blockBaseN_
 * 个分形（内容与零乘积无关，形状满足 K0×N）。 */
template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::IssueLoadL0ZeroA2()
{
    load3dZero_.mExtension = static_cast<uint16_t>(mCompute_);
    load3dZero_.kExtension = static_cast<uint16_t>(CONV3DTRANSPOSE_C0);
    LocalTensor<T> a2 = a2Base_;
    if (l0Slot_ == 1) {
        a2 = a2[l0aHalfElem_];
    }
#if defined(ASC_DEVKIT_VERSION_NUM) && ASC_DEVKIT_VERSION_NUM >= 90000000
    LoadDataRepeatParamWithStride rp = {0, 1, 0, static_cast<uint16_t>(CeilDiv16L(mCompute_))};
    SetLoadDataRepeatWithStride(rp);
    LoadDataWithStride(a2, a1v_, load3dZero_);
#else
    LoadDataRepeatParam rp = {0, 1, 0};
    SetLoadDataRepeat(rp);
    LoadData(a2, a1v_, load3dZero_);
#endif
}

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::IssueLoadL0ZeroB2()
{
    LocalTensor<T> b2 = b2Base_;
    if (l0Slot_ == 1) {
        b2 = b2[l0bHalfElem_];
    }
    load2dZero_.kStep = static_cast<uint16_t>(blockBaseN_);
    LoadData(b2, b1v_[0], load2dZero_);
}

/* ================================ MMAD（Kernel.md §9.4，FP32 L0C 累加口径含 bias BT 初值）
 * ================================ */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
template <bool hasBias>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::IssueMmad(
    LocalTensor<float>& useC1, bool firstContribution)
{
    LocalTensor<T> a2 = a2Base_;
    LocalTensor<T> b2 = b2Base_;
    if (l0Slot_ == 1) {
        a2 = a2[l0aHalfElem_];
        b2 = b2[l0bHalfElem_];
    }
    MmadParams mp;
    mp.m = static_cast<uint16_t>(mCompute_); /* 任务末 M 块向 16 对齐（9.1 双口径） */
    mp.n = static_cast<uint16_t>(baseUseN_);
    mp.k = static_cast<uint16_t>(baseUseK_);
    mp.unitFlag = 0;
    mp.kDirectionAlign = false;
    if constexpr (hasBias) {
        if (firstContribution && !computeBiasOnce_) {
            /* 首次有效贡献：带 BT 的 MMAD 引入 bias 初值（cmatrixSource=1、initVal=false），
             * 后续 K/kd 从 L0C 累加；每个输出块恰好一次（bias-bt-path §4） */
            mp.cmatrixSource = true;
            mp.cmatrixInitVal = false;
            Mmad(useC1, a2, b2, biasBTHeld_[biasOffset_], mp);
            computeBiasOnce_ = true;
            return;
        }
        mp.cmatrixSource = false;
        mp.cmatrixInitVal = false;
        Mmad(useC1, a2, b2, mp);
    } else {
        mp.cmatrixSource = false;
        mp.cmatrixInitVal = firstContribution; /* 首次有效贡献清零 L0C，其后原址累加 */
        Mmad(useC1, a2, b2, mp);
    }
}

/* ================================ 写回（Kernel.md §8 CopyOut） ================================ */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2,
          typename BIAST>
__aicore__ inline void Conv3dtranspose<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2, BIAST>::LoadL0c2Gm(
    LocalTensor<float>& c1v)
{
    const Conv3DTransposeTilingData& td = td_;
    const uint64_t ciGlobal = groupIdx_ * td.cinG + cinStart_;
    const uint64_t xBatchOff = kDyNd ? batchIdx_ * static_cast<uint64_t>(td.di) * td.hiWi * td.cin :
                                       batchIdx_ * static_cast<uint64_t>(td.cin) * td.diHiWi;
    uint64_t dstOffset = xBatchOff;
    if constexpr (!kDyNd) {
        /* y 为 NCDHW：COLUMN_MAJOR / NZ2DN，dstStride = di·hi·wi */
        dstOffset += ciGlobal * td.diHiWi + static_cast<uint64_t>(curDinIdx_) * td.hiWi + mStart_;
        FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR> fp;
        fp.mSize = static_cast<uint16_t>(baseUseM_);
        fp.nSize = static_cast<uint16_t>(baseUseN_);
        fp.srcStride = static_cast<uint16_t>(mCompute_);
        fp.dstStride = td.diHiWi;
        fp.params.dnNum = 1;
        fp.params.srcNzMatrixStride = 0;
        fp.params.dstDnMatrixStride = 0;
        fp.params.srcNzC0Stride = 1;
        fp.quantPre = kIsBf16 ? QuantMode_t::F322BF16 : QuantMode_t::F322F16;
        fp.reluEn = false;
        fp.unitFlag = 0;
        Fixpipe<T, float, CFG_COLUMN_MAJOR>(xGm_[dstOffset], c1v, fp);
    } else {
        /* y 为 NDHWC：ROW_MAJOR / NZ2ND，dstStride = cin（总通道，非 cinG——尾块陷阱） */
        dstOffset += (static_cast<uint64_t>(curDinIdx_) * td.hiWi + mStart_) * td.cin + ciGlobal;
        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fp;
        fp.mSize = static_cast<uint16_t>(baseUseM_);
        fp.nSize = static_cast<uint16_t>(baseUseN_);
        fp.srcStride = static_cast<uint16_t>(mCompute_);
        fp.dstStride = td.cin;
        fp.params.ndNum = 1;
        fp.quantPre = kIsBf16 ? QuantMode_t::F322BF16 : QuantMode_t::F322F16;
        fp.reluEn = false;
        fp.unitFlag = 0;
        Fixpipe<T, float, CFG_ROW_MAJOR>(xGm_[dstOffset], c1v, fp);
    }
}

#endif /* CONV3DTRANSPOSE_KERNEL_H */
