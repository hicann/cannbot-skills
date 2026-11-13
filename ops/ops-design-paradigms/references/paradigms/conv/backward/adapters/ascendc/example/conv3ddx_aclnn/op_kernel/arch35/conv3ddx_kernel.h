/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3ddx_kernel.h
 * \brief Conv3ddx 算子 kernel 模板类（三 loadB2 权重变换共用骨架，arch35 / AIC-only Cube 通路）
 *
 * 实现真值（design 文档集为权威依据，实现不偏离设计）：
 *   - design/Kernel.md §2「Kernel 类声明」/§3「调度框架」/§4「Init」/§5「Process」/
 *     §6「Sync」/§7「CopyIn A 路径 + B 路径共享帧」/§8「CopyOut」/§9.1–§9.8——跨分支共享帧
 *     （入口三步协议、任务解码轴序、UpdateL1ComputeInfo 行带交集 + 深度采样、
 *     EnsureA1/IssueLoadToA2/Load3D、Mmad FP32 L0C 累加、Fixpipe 写回、无贡献块
 *     完整写回协议（ComputeZeroBlock）、L0 槽对 MTE1_M/M_MTE1 事件对）逐字实装；
 *   - design/branches/DESIGN-BRANCH-16777217.md §5.2（loadB2Condition=1 转置加逆序：
 *     GM→L1B 三布局参数表 + L1B 补 co 尾 FillZero + LoadData2DParamsV2 逆序源偏移）、
 *     DESIGN-BRANCH-16777218.md §5.2（loadB2Condition=2 仅逆序：整行逆序装载 + 负
 *     srcStride 窗口分段）、DESIGN-BRANCH-16777219.md §5.2（loadB2Condition=3 仅转置：
 *     核体积 1 三布局归一装载 + cout1 分形正序）——三个 tilingKey 的 B 路径分支臂
 *     （ASCENDC_TPL_SEL 三组合实例化同一模板类，三臂均须可编译）；
 *   - 参考实现：范式 conv3ddx_direct（20 例验证全绿直调工程）——注册形态按 Kernel.md
 *     §1 差异登记适配（format 提升为编译期模板实参 FMT_FILTER/FMT_DY/FMT_Y，三 filter
 *     布局经 if constexpr 静态分派；DHWCN 装载参数为分支设计登记的推导值）。
 *
 * 片上级数（范式边界，design/branches/DESIGN-BRANCH-16777217.md §4）：GM→L1A/L1B→
 * L0A/L0B→L0C→GM，UB 不承载主计算数据；dtype fp16/bf16（2B）输入、FP32 L0C（4B）累加、
 * F322F16/F322BF16 写出；无 bias、无 vec/C04/Kahan 兜底（本交付执行配置）。
 */

#ifndef CONV3DDX_KERNEL_H
#define CONV3DDX_KERNEL_H

#include "kernel_operator.h"
#include "conv3ddx_tiling_struct.h"
#include "conv3ddx_struct.h" /* TPL_* 常量 */

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
constexpr uint32_t BLOCK_SIZE_ELEM = 256u; /* 512B 分形 = fp16/bf16 256 输入元素
                                              （instruction-mapping §5.1 步幅单位） */

/*
 * 注册组合 format 标签：取值 = 对应 ge::Format 枚举值（本交付 CANN 9.3.0
 * include/graph/c_types.h 实测：C_FORMAT_NDHWC=27、C_FORMAT_NCDHW=30、C_FORMAT_DHWCN=31），
 * 与 codegen 按 OpDef 组合注入的 FORMAT_FILTER / FORMAT_OUT_BACKPROP / FORMAT_Y 宏值一致
 * （registry-invoke def 驱动模式：构建系统生成 -DFORMAT_X=<ge::Format 值> 编译宏）；
 * Kernel.md 9.8 校验项 6 的构建期核定对象。
 */
constexpr uint32_t CONV3DDX_FMT_NDHWC = 27u;
constexpr uint32_t CONV3DDX_FMT_NCDHW = 30u;
constexpr uint32_t CONV3DDX_FMT_DHWCN = 31u;

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2>
class Conv3ddxKernel {
public:
    /* 接口契约断言：受支持组合内三路恒同 dtype（Interface.md「算子规格(受支持 dtype 组合)」
     * 等 dtype 配对，混合 dtype 不支持）；y 随 out_backprop 严格配对
     * （Interface.md「数据 Format 支持」）。 */
    static_assert(IsSameType<FT, DYT>::value && IsSameType<DYT, YT>::value,
                  "filter/out_backprop/y must share one dtype in supported combos");
    static_assert(FMT_DY == FMT_Y, "y format must pair with out_backprop format");
    using T = FT;

    /* 编译期布局标签（dedy/y 双格式、filter 三布局；DHWCN filter 恒路由 16777217，
     * BranchRoute.md「tilingKey 路由表」） */
    static constexpr bool kDyNd = (FMT_DY == CONV3DDX_FMT_NDHWC);
    static constexpr bool kFiltNcdhw = (FMT_FILTER == CONV3DDX_FMT_NCDHW);
    static constexpr bool kFiltNd = (FMT_FILTER == CONV3DDX_FMT_NDHWC);
    static constexpr bool kIsBf16 = IsSameType<T, bfloat16_t>::value;

    __aicore__ inline Conv3ddxKernel() {}
    __aicore__ inline ~Conv3ddxKernel() {}

    /* 入口三步协议（范式 kernel-template §2.1：Init → Process → End，各调用一次；
     * End 由入口调用一次，Process 内不重复调用） */
    __aicore__ inline void Init(GM_ADDR filter, GM_ADDR outBackprop, GM_ADDR y, const Conv3ddxTilingData& td);
    __aicore__ inline void Process();
    __aicore__ inline void End();

private:
    /* ---- 调度与块级门控（跨分支共享） ---- */
    __aicore__ inline void UpdateL1ComputeInfo(); /* 行带交集 + 深度采样 → needCompute_ */
    __aicore__ inline void ComputeOneBlock();     /* 单输出块完整归约 + 写回 */
    __aicore__ inline void ComputeZeroBlock();    /* 无贡献块显式写零（完整写回协议，Kernel.md 9.5） */
    /* ---- A 路径：dedy GM→L1A→L0A（跨分支共享） ---- */
    __aicore__ inline void EnsureA1(uint32_t curDoutIdx, uint32_t kIdx);
    __aicore__ inline void ReleaseA1IfSegmentEnd(uint32_t curDoutIdx, uint32_t kIdx);
    __aicore__ inline void FreeA1Tensor();
    __aicore__ inline void IssueLoadToA2(uint64_t kStart);
    /* ---- B 路径：weight GM→L1B→L0B（共享帧；参数构造体归分支设计文件 §5.2） ---- */
    __aicore__ inline void EnsureB1(uint32_t kd, uint32_t kIdx, bool zeroFill = false);
    __aicore__ inline void ReleaseB1IfSegmentEnd(uint32_t kIdx);
    __aicore__ inline void FreeB1Tensor();
    __aicore__ inline void IssueLoadToB2(uint64_t kStart);
    /* ---- 计算与写回（跨分支共享） ---- */
    __aicore__ inline void IssueMmad(LocalTensor<float>& useC1, bool firstContribution);
    __aicore__ inline void LoadL0c2Gm(LocalTensor<float>& c1v);
    __aicore__ inline void FillZero(const LocalTensor<T>& buf, uint32_t slotBytes);

    /* ---- 状态字段（跨分支共享集；三 key 实例化同一字段布局） ---- */
    Conv3ddxTilingData td_;
    GlobalTensor<T> wGm_;  /* filter */
    GlobalTensor<T> dyGm_; /* out_backprop（dedy） */
    GlobalTensor<T> xGm_;  /* y（dedx） */
    TPipe pipe_;
    TQue<TPosition::A1, 2> queAl1_;      /* L1A：dedy 驻留（队列深度 2，实际份数 al1Pbuffer） */
    TQue<TPosition::B1, 2> queBl1_;      /* L1B：weight 驻留（队列深度 2，实际份数 bl1Pbuffer） */
    TBuf<TPosition::A2> a2Buf_;          /* L0A：fmap 窗口 */
    TBuf<TPosition::B2> b2Buf_;          /* L0B：weight_rot */
    TQue<TPosition::CO1, 1> queCl0Ping_; /* L0C 槽 0（每槽独立 CO1 队列，深度 1） */
    TQue<TPosition::CO1, 1> queCl0Pong_; /* L0C 槽 1（cl0Pbuffer=2 时启用） */
    LocalTensor<T> a2Base_;
    LocalTensor<T> b2Base_;
    LocalTensor<T> a1v_;
    LocalTensor<T> b1v_;
    LoadData3DParamsV2<T> load3d_;
    LoadData2DParamsV2 load2dv2_;
    uint32_t coreId_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t hoWo_ = 0;           /* ho·wo（dedy 行内元素数） */
    uint32_t inputSlotCount_ = 1; /* L0A/L0B 输入槽对数：两路同时双缓冲为 2，否则 1 */
    uint32_t l0aHalfElem_ = 0;    /* L0A 半区元素数（al0Pbuffer≥2 时非 0） */
    uint32_t l0bHalfElem_ = 0;    /* L0B 半区元素数（bl0Pbuffer≥2 时非 0） */
    uint32_t cSlotCount_ = 1;     /* L0C 槽数（cl0Pbuffer） */
    uint32_t l0Slot_ = 0;         /* 当前 L0 输入槽对 */
    uint32_t cSlot_ = 0;          /* 当前 L0C 槽 */
    uint64_t batchIdx_ = 0;
    uint64_t groupIdx_ = 0;
    uint32_t curDinIdx_ = 0;            /* 当前 dedx 深度 */
    uint32_t mStart_ = 0;               /* 当前输出块空间起点 */
    uint32_t cinStart_ = 0;             /* 当前 N 块 dedx 组内通道起点 */
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
};

/* ================================ Init（Kernel.md §4） ================================ */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2>
__aicore__ inline void Conv3ddxKernel<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2>::Init(
    GM_ADDR filter, GM_ADDR outBackprop, GM_ADDR y, const Conv3ddxTilingData& td)
{
    td_ = td;
    wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(filter));
    dyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outBackprop));
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    coreId_ = GetBlockIdx();
    coreNum_ = td.coreNum;

    hoWo_ = td.ho * td.wo;
    inputSlotCount_ = (td.al0Pbuffer >= 2 && td.bl0Pbuffer >= 2) ? 2u : 1u;
    l0aHalfElem_ = (td.al0Pbuffer >= 2) ? (CONV3DDX_L0A_BYTES / 2 / sizeof(T)) : 0;
    l0bHalfElem_ = (td.bl0Pbuffer >= 2) ? (CONV3DDX_L0B_BYTES / 2 / sizeof(T)) : 0;
    cSlotCount_ = (td.cl0Pbuffer >= 2) ? 2u : 1u;

    /* 五路 Buffer 分配（范式 buffer-and-types §3；TQue 形态与内置一致：
     * A1/B1 深度 2、L0A/L0B 整块 TBuf、CO1 每槽独立队列深度 1） */
    pipe_.InitBuffer(queAl1_, td.al1Pbuffer, td.aSlotBytes);
    pipe_.InitBuffer(queBl1_, td.bl1Pbuffer, td.bSlotBytes);
    pipe_.InitBuffer(a2Buf_, CONV3DDX_L0A_BYTES);
    pipe_.InitBuffer(b2Buf_, CONV3DDX_L0B_BYTES);
    pipe_.InitBuffer(queCl0Ping_, 1, CONV3DDX_L0C_BYTES / cSlotCount_);
    if (cSlotCount_ == 2) {
        pipe_.InitBuffer(queCl0Pong_, 1, CONV3DDX_L0C_BYTES / cSlotCount_);
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

    /* Load2D 参数初值（范式 instruction-mapping §5.1；ifTranspose 是分支编译期常量：
     * 仅逆序分支（LOADB2=TPL_REVERSE_ONLY）为 false，其余两分支为 true） */
    load2dv2_.mStartPosition = 0;
    load2dv2_.kStartPosition = 0;
    load2dv2_.mStep = 1;
    load2dv2_.kStep = 1;
    load2dv2_.dstStride = 1;
    load2dv2_.ifTranspose = (LOADB2 != TPL_REVERSE_ONLY);
    load2dv2_.sid = 0;

    /* L1 持有状态复位 */
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
}

/* ================================ Process（Kernel.md §5，含 §3 任务解码） ================================ */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2>
__aicore__ inline void Conv3ddxKernel<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2>::Process()
{
    const Conv3ddxTilingData& td = td_;
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
                    UpdateL1ComputeInfo();
                    if (!needCompute_) {
                        ComputeZeroBlock();
                        continue;
                    }
                    ComputeOneBlock();
                }
            }
        }
    }
    FreeB1Tensor();
    FreeA1Tensor();
}

/* ================================ End（TPipe 析构完成事件收尾，Kernel.md §6） ================================ */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2>
__aicore__ inline void Conv3ddxKernel<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2>::End()
{
    /* 全部队列成对平衡（Alloc/EnQue/DeQue/FreeTensor），TPipe 析构完成事件收尾
     * （含未使用槽保留的初始令牌）；无跨核旗语、无 SyncAllCores（AIC_ONLY 任务并行模型）。 */
}

/* ================================ 块级门控（Kernel.md §3 UpdateL1ComputeInfo） ================================ */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2>
__aicore__ inline void Conv3ddxKernel<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2>::UpdateL1ComputeInfo()
{
    const Conv3ddxTilingData& td = td_;
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
     * [0, hoExpand) 的交（范式 compute-flow §5.2.1，负上边界先裁剪）。
     * 行带与深度采样解耦计算：仅深度采样失败（dValid=false，纯零深度平面）的块
     * 保留真实行带（curHoSize_>0）——ComputeZeroBlock 的 A 路径装载窗口因此与
     * 有贡献块完全同构（合法 Load3D 参数），不再依赖 curHoSize_=0 的短路。 */
    const uint32_t curHiIdx = mStart_ / td.wi;
    const uint32_t curWiIdx = mStart_ % td.wi;
    curHoIdx_ = static_cast<int32_t>(curHiIdx) - td.backpropPadUp;
    const uint32_t hiCal = CeilDivL(static_cast<uint64_t>(baseUseM_) + curWiIdx, td.wi);
    const int64_t endHoIdx = static_cast<int64_t>(curHoIdx_) + hiCal + (td.hk - 1) * td.dilationH;
    const bool bandValid = curHoIdx_ < static_cast<int32_t>(td.hoExpand) && endHoIdx > 0;
    if (bandValid) {
        const uint32_t realStart = curHoIdx_ < 0 ? 0u : static_cast<uint32_t>(curHoIdx_);
        const uint32_t realEnd =
            static_cast<uint32_t>(endHoIdx) > td.hoExpand ? td.hoExpand : static_cast<uint32_t>(endHoIdx);
        curHoSize_ = realEnd - realStart;
    } else {
        curHoSize_ = 0;
    }
    needCompute_ = dValid && bandValid;
    /* 新输出块：L1A 父段身份失效（L1B 是否跨块保留由 enableFullLoad 决定，9.6） */
    a1SegFirst_ = 0xFFFFFFFFu;
    a1Dout_ = 0xFFFFFFFFu;
}

/* ================================ 单输出块主计算（Kernel.md §5 ComputeOneBlock） ================================ */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2>
__aicore__ inline void Conv3ddxKernel<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2>::ComputeOneBlock()
{
    const Conv3ddxTilingData& td = td_;
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
            /* L0 输入槽交接（范式 sync-implementation §1 稳态事件对） */
            WaitFlag<HardEvent::M_MTE1>(l0Slot_);
            IssueLoadToA2(kStart);
            IssueLoadToB2(kStart);
            SetFlag<HardEvent::MTE1_M>(l0Slot_);
            WaitFlag<HardEvent::MTE1_M>(l0Slot_);
            IssueMmad(useC1, firstContribution);
            firstContribution = false;
            SetFlag<HardEvent::M_MTE1>(l0Slot_);
            l0Slot_ = (l0Slot_ + 1) % inputSlotCount_;
            /* 段末释放：父段最后一次 MTE1 读取已提交后由队列发布归还依赖（§6） */
            ReleaseA1IfSegmentEnd(curDoutIdx, static_cast<uint32_t>(kIdx));
            ReleaseB1IfSegmentEnd(static_cast<uint32_t>(kIdx));
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

/* ================================ 无贡献块零通路（Kernel.md 9.5 完整写回协议） ================================
 *
 * 无贡献块（needCompute_=false：纯零深度平面 / 空行带）的本块有效 M/N 区域仍须显式写出精确 0
 * （完整写回协议，语义与设计 §9.5 一致：不依赖输出预清零、覆盖本块全部有效区域）。写回机制
 * 与 §9.5 参考实现的逐元素标量 GM 写不同——本实现取「零权重单次 MMAD + Fixpipe」通路：
 *
 *   B 路径以 zeroFill 装载（L1B 整槽 FillZero、跳过真实 weight 搬运）→ LoadData 后 L0B 恒为 0；
 *   A 路径按有贡献块同构参数装载真实 dedy 窗口（curDoutIdx=0，有限值）→ 单次
 *   Mmad(cmatrixInitVal=true) 得 C = A·0 = 0（fp32 精确 +0）→ 经 LoadL0c2Gm 的 Fixpipe 写出。
 *
 * 机制偏离登记（相对 §9.5 逐元素标量写）：arch35 上标量 GM 写与并发 Fixpipe CO2GM 写 /
 * 相邻任务标量写在共享 32B 行上互相覆盖（never-written 楔形缺损，L1 门禁 11 红行根因，
 * 证据见 run-l1-100pct.RECORD.md §5），全部输出写统一走 Fixpipe 引擎后该冲突消除
 * （同参数全 Fixpipe 写回经 pkB/pkB2 与 E1 禁写对照验证为正确）。
 */
template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2>
__aicore__ inline void Conv3ddxKernel<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2>::ComputeZeroBlock()
{
    const Conv3ddxTilingData& td = td_;
    auto& q = (cSlot_ == 0) ? queCl0Ping_ : queCl0Pong_;
    LocalTensor<float> useC1 = q.template AllocTensor<float>();
    /* 空行带块合成 1 行顶部行带：A 路径装载窗口（l1H≥1 + padList 下侧哨兵零填）
     * 与有贡献块同构，保证 Load3D 参数合法；装载内容与本块零结果无关（B 恒 0） */
    if (curHoSize_ == 0) {
        curHoIdx_ = 0;
        curHoSize_ = 1;
    }
    baseUseK_ = (td.kIter == 1) ? td.tailK : td.baseK;
    EnsureA1(0u, 0u);
    EnsureB1(0u, 0u, true);
    /* L0 输入槽交接（与 ComputeOneBlock 稳态事件对一致，范式 sync-implementation §1） */
    WaitFlag<HardEvent::M_MTE1>(l0Slot_);
    IssueLoadToA2(0);
    IssueLoadToB2(0);
    SetFlag<HardEvent::MTE1_M>(l0Slot_);
    WaitFlag<HardEvent::MTE1_M>(l0Slot_);
    IssueMmad(useC1, true);
    SetFlag<HardEvent::M_MTE1>(l0Slot_);
    l0Slot_ = (l0Slot_ + 1) % inputSlotCount_;
    ReleaseA1IfSegmentEnd(0u, 0u);
    ReleaseB1IfSegmentEnd(0u);
    q.EnQue(useC1);
    LocalTensor<float> c1v = q.template DeQue<float>();
    LoadL0c2Gm(c1v);
    q.FreeTensor(c1v);
    cSlot_ = (cSlot_ + 1) % cSlotCount_;
}

/* ================================ A 路径（dedy：GM→L1A→L0A，Kernel.md §7） ================================ */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2>
__aicore__ inline void Conv3ddxKernel<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2>::EnsureA1(uint32_t curDoutIdx,
                                                                                                uint32_t kIdx)
{
    const Conv3ddxTilingData& td = td_;
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
        channelSize_ = CONV3DDX_C0;
    }

    /* W 裁剪：负反向边界先裁剪真实 dedy（9.7；范式 compute-flow §5.2.1 woEnd 带符号口径） */
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
         * 范式 compute-flow §5.2.1 修复口径，高风险复跑点，9.8 校验项 4） */
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
    const uint64_t a1DstOffset = static_cast<uint64_t>(hSkip) * woExpandLocal * CONV3DDX_C0;

    /* FMatrix：padList 顺序左、右、上、下；下侧取 255 哨兵（范式 instruction-mapping §4.2） */
    load3d_.l1H = static_cast<uint16_t>(curHoSize_);
    load3d_.l1W = static_cast<uint16_t>(woExpandLocal);
    load3d_.padList[0] = static_cast<uint8_t>(
        bpLeft < 0 ? static_cast<int32_t>(CeilDivL(static_cast<uint32_t>(-bpLeft), td.strideW) * td.strideW) + bpLeft :
                     bpLeft);
    load3d_.padList[1] = static_cast<uint8_t>(bpRight < 0 ? static_cast<int32_t>(woExpandRaw) + bpRight -
                                                                static_cast<int32_t>((woEnd - 1) * td.strideW + 1) :
                                                            bpRight);
    load3d_.padList[2] = static_cast<uint8_t>(curHoIdx_ < 0 ? -curHoIdx_ : 0);
    load3d_.padList[3] = 255u;
    load3d_.channelSize = static_cast<uint16_t>(channelSize_);

    /* 槽清零后搬入真实值（Fill → PipeBarrier<PIPE_MTE2> → DataCopy → EnQue，范式 sync §5） */
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
                p.dstNzMatrixStride = static_cast<uint32_t>(td.strideH) * woExpandLocal * CONV3DDX_C0;
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
                p.dstNzMatrixStride = static_cast<uint32_t>(td.strideH) * woExpandLocal * CONV3DDX_C0;
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

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2>
__aicore__ inline void Conv3ddxKernel<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2>::ReleaseA1IfSegmentEnd(
    uint32_t curDoutIdx, uint32_t kIdx)
{
    const Conv3ddxTilingData& td = td_;
    (void)curDoutIdx;
    const uint32_t segFirst = (kIdx / td.stepKa) * td.stepKa;
    const uint32_t segIters = (td.kIter - segFirst) < td.stepKa ? (td.kIter - segFirst) : td.stepKa;
    if (kIdx + 1 == segFirst + segIters && a1Valid_) {
        queAl1_.FreeTensor(a1v_);
        a1Valid_ = false;
    }
}

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2>
__aicore__ inline void Conv3ddxKernel<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2>::FreeA1Tensor()
{
    if (a1Valid_) {
        queAl1_.FreeTensor(a1v_);
        a1Valid_ = false;
    }
}

/* L1A→L0A 窗口提取（Load3D 硬件 im2col；H/W 步长固定 1，dilation 保留；
 * SDK 版本分支按范式 instruction-mapping §4.2/§4.3 成对选择 Repeat 参数与 LoadData 重载） */
template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2>
__aicore__ inline void Conv3ddxKernel<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2>::IssueLoadToA2(uint64_t kStart)
{
    const Conv3ddxTilingData& td = td_;
    const uint32_t kLocalA = static_cast<uint32_t>(kStart - aSegStartK_);
    load3d_.kStartPt = static_cast<uint16_t>(kLocalA);
    load3d_.kExtension = static_cast<uint16_t>(baseUseK_);
    load3d_.mExtension = static_cast<uint16_t>(baseUseM_);
    load3d_.mStartPt = static_cast<uint16_t>(mStart_ % td.wi);
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
template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2>
__aicore__ inline void Conv3ddxKernel<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2>::FillZero(
    const LocalTensor<T>& buf, uint32_t slotBytes)
{
    InitConstValueParams<T> z;
    z.repeatTimes = 1;
    /* DataBlock 32B 架构不变量经系统变量 API 引用（KERNEL-5：禁止裸 32 参与块对齐计算） */
    z.blockNum = static_cast<uint16_t>(slotBytes / static_cast<uint32_t>(GetDataBlockSizeInBytes()));
    z.dstGap = 0;
    z.initValue = static_cast<T>(0);
    InitConstValue(buf, z);
    PipeBarrier<PIPE_MTE2>();
}

/* ================================ B 路径（weight：GM→L1B→L0B） ================================
 * 共享帧（Kernel.md §7「B 路径」帧表）：装载触发 = 父段身份四元组
 * (segFirst, kd, cinStart_, groupIdx_) 任一变化或 b1Valid_=false 时重装、全等沿用；
 * 父段参数 / 通道量 / 槽清零 / 队列交接 / 归还（enableFullLoad=1 时段末跳过，9.6）跨分支共享。
 * GM→L1B 的 srcBase 与 Nd2Nz/Dn2Nz 参数、L1B→L0B 的 LoadData2DParamsV2 字段按
 * LOADB2 分支臂构造（DESIGN-BRANCH-16777217/8/9.md §5.2 各自的参数表与完整代码）。
 */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2>
__aicore__ inline void Conv3ddxKernel<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2>::EnsureB1(uint32_t kd,
                                                                                                uint32_t kIdx,
                                                                                                bool zeroFill)
{
    const Conv3ddxTilingData& td = td_;
    /* 父段命中复用：身份四元组全等即沿用（enableFullLoad 驻留语义，Kernel.md 9.6）；
     * zeroFill 槽以 kd 哨兵 0xFFFFFFFF 参与匹配与登记——真实 kd（<dk）永不命中零槽、
     * 零块也永不命中真实 weight 驻留槽（否则零块会按真实 weight 计算），仅连续
     * 无贡献块复用同一零槽，后续有贡献块必然重装真实 weight */
    const uint32_t idKd = zeroFill ? 0xFFFFFFFFu : kd;
    const uint32_t segFirst = (kIdx / td.stepKb) * td.stepKb;
    if (segFirst == b1SegFirst_ && b1Valid_ && b1IdKd_ == idKd && b1IdCin_ == cinStart_ && b1IdGroup_ == groupIdx_) {
        return;
    }
    FreeB1Tensor();
    b1SegFirst_ = segFirst;
    b1IdKd_ = idKd;
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
        /* 转置加逆序（tilingKey 16777217，DESIGN-BRANCH-16777217.md §5.2 参数表；
         * NCDHW/NDHWC 为范式与参考实现已验证值，DHWCN 为设计登记推导值（co 最快轴）） */
        if constexpr (FMT_FILTER == CONV3DDX_FMT_NCDHW) {
            srcBase = coGlobal * td.cinG * td.dkHkWk + static_cast<uint64_t>(cinStart_) * td.dkHkWk +
                      static_cast<uint64_t>(kd) * td.hkWk;
        } else if constexpr (FMT_FILTER == CONV3DDX_FMT_NDHWC) {
            srcBase = coGlobal * td.cinG * td.dkHkWk + static_cast<uint64_t>(kd) * td.hkWk * td.cinG + cinStart_;
        } else { /* FMT_FILTER == CONV3DDX_FMT_DHWCN：DHWCN filter 恒路由本 key（推导值） */
            srcBase = (static_cast<uint64_t>(kd) * td.hkWk * td.cinG + cinStart_) * td.cout + coGlobal;
        }
    } else if constexpr (LOADB2 == TPL_TRANSPOSE_ONLY) {
        /* 仅转置（tilingKey 16777219，核体积 1，DESIGN-BRANCH-16777219.md §5.2）：
         * NCDHW/NDHWC 在核体积 1 下同为 (co, ci) 行主序；DHWCN 为 (ci, co) 列主序 */
        if constexpr (FMT_FILTER == CONV3DDX_FMT_NCDHW || FMT_FILTER == CONV3DDX_FMT_NDHWC) {
            srcBase = coGlobal * td.cinG + cinStart_;
        } else { /* DHWCN：Dn2Nz 互补结构（co/ci 互换，镜像基线 LoadToB1TransposeForDkHkWkIsOne） */
            srcBase = static_cast<uint64_t>(cinStart_) * td.cout + coGlobal;
        }
    } else {
        /* 仅逆序（tilingKey 16777218，DESIGN-BRANCH-16777218.md §5.2(a)）；
         * DHWCN filter 两仅逆序谓词恒不成立、永不路由本 key（BranchRoute.md）——
         * 该 (LOADB2=2 × DHWCN) 实例化组合在设计路由域外，此处按 NCDHW 同构装载
         * 保证模板全实例化可编译（参考实现同款 else 口径），运行期不可达。 */
        if constexpr (FMT_FILTER == CONV3DDX_FMT_NDHWC) {
            srcBase = coGlobal * td.cinG * td.dkHkWk + static_cast<uint64_t>(kd) * td.hkWk * td.cinG + cinStart_;
        } else {
            srcBase = coGlobal * td.cinG * td.dkHkWk + static_cast<uint64_t>(cinStart_) * td.dkHkWk +
                      static_cast<uint64_t>(kd) * td.hkWk;
        }
    }

    /* 槽清零：装载前对整个槽 FillZero（FillZero 全槽，另一条分形轴尾部由转换补齐；
     * 转置加逆序补 co 尾 / 仅逆序补 ci 尾，DESIGN-BRANCH-16777217/8.md §5.2(c)）；
     * zeroFill=true 时跳过真实 weight 搬运，整槽保持 FillZero 的 0（无贡献块零通路，
     * Kernel.md 9.5——L0B 恒 0 使单次 MMAD 得 C=A·0=0） */
    LocalTensor<T> b1Alloc = queBl1_.template AllocTensor<T>();
    FillZero(b1Alloc, td.bSlotBytes);
    if (!zeroFill && coutSize > 0 && cinSize > 0) {
        if constexpr (LOADB2 == TPL_TRANSPOSE_AND_REVERSE) {
            /* 转置加逆序：Dn2Nz（NCDHW）/ Nd2Nz（NDHWC/DHWCN）→ 同一 L1B 父布局
             * (Cin1,Hk,Wk,Cout1,Cout0,Cin0)（DESIGN-BRANCH-16777217.md §5.2） */
            if constexpr (FMT_FILTER == CONV3DDX_FMT_NCDHW) {
                Dn2NzParams p;
                p.dnNum = static_cast<uint16_t>(coutSize);
                p.nValue = static_cast<uint16_t>(td.hkWk);
                p.dValue = cinSize;
                p.srcDnMatrixStride = static_cast<uint64_t>(td.cinG) * td.dkHkWk;
                p.srcDValue = td.dkHkWk;
                p.dstNzC0Stride = static_cast<uint16_t>(td.hkWk * curCout1Cout0Size_);
                p.dstNzNStride = static_cast<uint16_t>(curCout1Cout0Size_);
                p.dstNzMatrixStride = CONV3DDX_C0;
                DataCopy(b1Alloc, wGm_[srcBase], p);
            } else if constexpr (FMT_FILTER == CONV3DDX_FMT_NDHWC) {
                Nd2NzParams p;
                p.ndNum = static_cast<uint16_t>(coutSize);
                p.nValue = static_cast<uint16_t>(td.hkWk);
                p.dValue = cinSize;
                p.srcNdMatrixStride = static_cast<uint64_t>(td.cinG) * td.dkHkWk;
                p.srcDValue = td.cinG;
                p.dstNzC0Stride = static_cast<uint16_t>(td.hkWk * curCout1Cout0Size_);
                p.dstNzNStride = static_cast<uint16_t>(curCout1Cout0Size_);
                p.dstNzMatrixStride = CONV3DDX_C0;
                DataCopy(b1Alloc, wGm_[srcBase], p);
            } else { /* DHWCN：Dn2NzParams —— DHWCN 源 (kd,kh,kw,CinG,Cout) 的三轴中
                      * 唯一连续轴是 cout，故取 d=cin（srcDValue=cout，步长语义正确）、
                      * n=co（Dn2Nz 要求 n 源连续，co 恰为 DHWCN 最内连续轴）、
                      * 矩阵轴 m=window（srcDnMatrixStride=cinG·cout）。
                      * 目的侧仍落本分支统一 L1B 父布局
                      * (Cin1,Hk,Wk,Cout1,Cout0,Cin0)（地址式
                      * c·dstNzC0Stride·16 + w·coutAligned·16 + k·16 + n）：
                      * dstNzNStride=1（co 步长 16 元素）、dstNzMatrixStride=
                      * coutAligned·16（window 步长）、dstNzC0Stride=hkWk·coutAligned
                      * （Cin1 步长）——与 NCDHW/NDHWC 两臂的目的布局逐地址一致，
                      * L1B→L0B 消费端（LoadData2D 逆序窗口寻址）无需分支。
                      * 原「设计登记推导值」用 Nd2Nz 把 cin 放在隐含连续的 D 轴
                      * （步长 1），实测（sf_001 dump 复现，105/112 错位、c=0 恒对；
                      * CinG=1 时不可见故 sf_007 旧码也过）为根因。 */
                Dn2NzParams p;
                p.dnNum = static_cast<uint16_t>(td.hkWk);
                p.nValue = static_cast<uint16_t>(coutSize);
                p.dValue = static_cast<uint16_t>(cinSize);
                p.srcDnMatrixStride = static_cast<uint64_t>(td.cinG) * td.cout;
                p.srcDValue = static_cast<uint64_t>(td.cout);
                p.dstNzC0Stride = static_cast<uint16_t>(td.hkWk * curCout1Cout0Size_);
                p.dstNzNStride = 1;
                p.dstNzMatrixStride = static_cast<uint32_t>(curCout1Cout0Size_) * CONV3DDX_C0;
                DataCopy(b1Alloc, wGm_[srcBase], p);
            }
        } else if constexpr (LOADB2 == TPL_TRANSPOSE_ONLY) {
            /* 仅转置：单矩阵装载（核体积 1），NCDHW/NDHWC 用 Nd2Nz、DHWCN 用互补 Dn2Nz
             * → 同一 L1B (Cin1,Cout1,Cout0,Cin0)（DESIGN-BRANCH-16777219.md §5.2） */
            if constexpr (FMT_FILTER == CONV3DDX_FMT_NCDHW || FMT_FILTER == CONV3DDX_FMT_NDHWC) {
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
            } else { /* DHWCN：Dn2NzParams（n=co 行在 DHWCN 源中 co 连续、d=ci 步幅 cout） */
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
            /* 仅逆序：NCDHW 整行逆序装载（Dn2NzTranspose）/ NDHWC 紧凑通道行
             * （内部使用 Dn2NzParams）→ L1B (Cout1,Hk,Wk,Cin1,Cin0,Cout0)
             * （DESIGN-BRANCH-16777218.md §5.2(b)；DHWCN 组合设计不可达，按 NCDHW 同构） */
            const uint32_t cinAligned = AlignUp16L(cinSize);
            if constexpr (FMT_FILTER == CONV3DDX_FMT_NDHWC) {
                Dn2NzParams p;
                p.dnNum = static_cast<uint16_t>(td.hkWk);
                p.dValue = coutSize;
                p.nValue = static_cast<uint16_t>(cinSize);
                p.srcDnMatrixStride = td.cinG;
                p.srcDValue = static_cast<uint64_t>(td.cinG) * td.dkHkWk;
                p.dstNzC0Stride = static_cast<uint16_t>(td.hkWk * cinAligned);
                p.dstNzNStride = 1;
                p.dstNzMatrixStride = cinAligned * CONV3DDX_C0;
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
                p.dstNzMatrixStride = CONV3DDX_C0;
                DataCopy(b1Alloc, wGm_[srcBase], p);
            }
        }
    }
    queBl1_.EnQue(b1Alloc);
    b1v_ = queBl1_.template DeQue<T>();
    b1Valid_ = true;
}

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2>
__aicore__ inline void Conv3ddxKernel<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2>::ReleaseB1IfSegmentEnd(
    uint32_t kIdx)
{
    const Conv3ddxTilingData& td = td_;
    if (td.enableFullLoad != 0) {
        return; /* 驻留：跨输出块/任务复用，身份变化或结束时释放（Kernel.md 9.6） */
    }
    const uint32_t segFirst = (kIdx / td.stepKb) * td.stepKb;
    const uint32_t segIters = (td.kIter - segFirst) < td.stepKb ? (td.kIter - segFirst) : td.stepKb;
    if (kIdx + 1 == segFirst + segIters && b1Valid_) {
        queBl1_.FreeTensor(b1v_);
        b1Valid_ = false;
    }
}

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2>
__aicore__ inline void Conv3ddxKernel<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2>::FreeB1Tensor()
{
    if (b1Valid_) {
        queBl1_.FreeTensor(b1v_);
        b1Valid_ = false;
    }
}

/* L1B→L0B 装载指令（LoadData2DParamsV2；K 配对约束：L0B 矩阵中对应 K 元素与 L0A
 * 指向同一 dedy 通道与核高宽位置，Kernel.md 9.2）；三种准备方式最终形成同一
 * ZN(K,N) 片段 (K1, Cin1, Cin0, k0)。 */
template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2>
__aicore__ inline void Conv3ddxKernel<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2>::IssueLoadToB2(uint64_t kStart)
{
    const Conv3ddxTilingData& td = td_;
    LocalTensor<T> b2 = b2Base_;
    if (l0Slot_ == 1) {
        b2 = b2[l0bHalfElem_];
    }
    const uint32_t kLocalB = static_cast<uint32_t>(kStart - bSegStartK_);
    const uint32_t bKStart = kLocalB / CONV3DDX_C0;
    const uint32_t curKRepeat = CeilDiv16L(baseUseK_);
    if constexpr (LOADB2 == TPL_TRANSPOSE_AND_REVERSE) {
        /* 转置加逆序（ifTranspose=true 由 Init 编译期设置）：每个 K 分形一次 LoadData，
         * 核空间逆序（rot180 空间分量）由源偏移 hkWk−1−r%hkWk 表达，深度分量由 kd 采样
         * 与 srcBase 的原 kd 坐标承载（DESIGN-BRANCH-16777217.md §5.2） */
        load2dv2_.mStartPosition = 0;
        load2dv2_.kStartPosition = 0;
        load2dv2_.mStep = 1;
        load2dv2_.kStep = static_cast<uint16_t>(blockBaseN_);
        load2dv2_.srcStride = static_cast<int32_t>(curLoadKbl1_ / CONV3DDX_C0);
        load2dv2_.dstStride = 1;
        uint32_t dstOffset = 0;
        for (uint32_t i = 0; i < curKRepeat; ++i) {
            const uint32_t r = bKStart + i;
            const uint32_t idxC1out = r / td.hkWk;
            const uint32_t srcOffset =
                idxC1out * BLOCK_SIZE_ELEM + (td.hkWk - 1 - (r % td.hkWk)) * curCout1Cout0Size_ * CONV3DDX_C0;
            LoadData(b2[dstOffset], b1v_[srcOffset], load2dv2_);
            dstOffset += blockBaseN_ * BLOCK_SIZE_ELEM;
        }
    } else if constexpr (LOADB2 == TPL_TRANSPOSE_ONLY) {
        /* 仅转置（ifTranspose=true）：K 分形 = cout1 分形，源偏移按分形正序递增
         * （核体积 1，空间逆序为恒等；DESIGN-BRANCH-16777219.md §5.2） */
        load2dv2_.mStartPosition = 0;
        load2dv2_.kStartPosition = 0;
        load2dv2_.mStep = 1;
        load2dv2_.kStep = static_cast<uint16_t>(blockBaseN_);
        load2dv2_.srcStride = static_cast<int32_t>(curLoadKbl1_ / CONV3DDX_C0);
        load2dv2_.dstStride = 1;
        uint32_t dstOffset = 0;
        for (uint32_t i = 0; i < curKRepeat; ++i) {
            const uint32_t idxC1out = bKStart + i;
            const uint32_t srcOffset = idxC1out * BLOCK_SIZE_ELEM;
            LoadData(b2[dstOffset], b1v_[srcOffset], load2dv2_);
            dstOffset += blockBaseN_ * BLOCK_SIZE_ELEM;
        }
    } else {
        /* 仅逆序（ifTranspose=false，三 key 唯一）：按 Cout1 窗口分段提交，负 srcStride
         * 沿 K 逆序读取——mStartPosition 的 (windowEnd−1−first) 因子把起点放到本 Cout1
         * 窗口内被需求位置的逆序对侧，等价 H/W 两轴各自逆序（rot180 核空间分量；
         * DESIGN-BRANCH-16777218.md §5.2(d)） */
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

/* ================================ MMAD（Kernel.md §9.4，FP32 L0C 累加口径） ================================ */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2>
__aicore__ inline void Conv3ddxKernel<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2>::IssueMmad(
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
    mp.cmatrixSource = false;
    mp.cmatrixInitVal = firstContribution; /* 首次有效贡献清零 L0C，其后原址累加 */
    Mmad(useC1, a2, b2, mp);
}

/* ================================ 写回（Kernel.md §8 CopyOut） ================================ */

template <typename FT, uint32_t FMT_FILTER, typename DYT, uint32_t FMT_DY, typename YT, uint32_t FMT_Y, uint8_t LOADB2>
__aicore__ inline void Conv3ddxKernel<FT, FMT_FILTER, DYT, FMT_DY, YT, FMT_Y, LOADB2>::LoadL0c2Gm(
    LocalTensor<float>& c1v)
{
    const Conv3ddxTilingData& td = td_;
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
        /* y 为 NDHWC：ROW_MAJOR / NZ2ND，dstStride = cin（总通道，非 cinG） */
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

#endif /* CONV3DDX_KERNEL_H */
