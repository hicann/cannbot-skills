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
 * \file conv3dtranspose_tiling_arch35.cpp
 * \brief conv3dtranspose Host 侧 TilingFunc 真实实现（绿相位，TDD UT 全绿目标）
 *
 * 实现真值（design/HostTiling.md「Host 侧 Tiling 方案」全文，逐节对应）：
 *   §3 输入预处理/合轴/场景预处理 —— input_size 值读取（值依赖，读取发生在元数据校验
 *      dtype→format→维度→attr 全部通过之后）、双模式解析（显式模式逐轴 rho∈[0,s) 校验 /
 *      全零哨兵属性推导 rho=0，推导结果逐维 ≥1）、布局/轴序归一（data_format 决定 x 与
 *      input_size 值序；filter 三种注册布局独立解析）、属性空间分量提取（NCDHW [2:5] /
 *      NDHWC [1:4]；pads 语义序 [pF,pB,pT,pBm,pL,pR] 与 data_format 无关，按 1/3/6 元展开）、
 *      派生几何量（rho / b⁻ / b⁺ 有符号公式（q+rho 折入后侧）/ mValue / nValue / kValue /
 *      initOutputFlag 转置条件表）；§3.2 异常值校验按 dtype → format → 维度 → attr → shape
 *      顺序执行（strides/dilations/output_padding 的 N/C 位序与逐轴 q<s 或 q<d 约束在几何
 *      一致性之后复核——位序解释依赖张量轴序与 data_format 已一致，轴序矛盾先行按
 *      shape_mismatch 拒绝，纯位序违约再按 attribute_value_out_of_range 拒绝），任一失败
 *      OP_LOGE 稳定锚点 + 返回 ge::GRAPH_FAILED，不路由任何 tilingKey（锚点契约：
 *      "dtype" / "format" / "rank" / "null_input" / "shape_mismatch" /
 *      "attribute_value_out_of_range" / "no legal" / "bias"，与 UT 断言 playbook 一致）。
 *   §4 UB 切分（本范式 Cube 通路即 L0 基本块 + L1 驻留设计，UB 不承载主计算数据）——
 *      InitBaseMNK 起点（核体积 1 取 (256,256,64)，其余 (512,128,32)）、AdjustBaseMNK
 *      （形状截断 / N 缩小扩大 M / M 缩小扩大 N / K 窗口配对 / L0C 份数 / 三路容量复核
 *      收缩；四侧反向边界任一超 255 时 maxM 收紧 512）、CalStepK（等分/阶梯候选 + 首次
 *      求解选择规则：等分候选合法且 step 之和严格大于阶梯候选时选等分，否则选阶梯；
 *      容量回调重选先阶梯后等分）、IsL1ParamsValid（L1 驻留预算 + 槽级复核，含 bias
 *      候选预算 biasBudget=alignUp(baseN·biasBytes,64)）、LegalProtection 容量回调
 *      （减 K → 减 M/N → 恢复大 K → 调 L1 份数 (2,2)→(2,1)→(1,2)→(1,1)，不切换模板）。
 *   §5 多核切分 —— SetSingleCoreInfo（空间/通道/深度划分 + 整行候选）、任务计数
 *      totalTasks = batch·group·dCount·mCount·nCount（uint64 宽整数）、
 *      coreNum = min(totalTasks, maxCores)（≥1）。
 *   §7 填 TilingData + 设 TilingKey —— 整结构清零 + 逐字段填充；loadB2 判定在求解完成后的
 *      最终 baseN 上按 BranchRoute.md §1 判定链执行（核体积 dk·hk·wk=1 → 3 仅转置；
 *      NDHWC ∧ baseN ≥ hk·wk → 2 仅逆序；NCDHW ∧ dk·hk·wk ≤ 64/2B/cin → 2；兜底 → 1
 *      转置加逆序；DHWCN filter 恒兜底），key = 2^24 + loadB2Condition（GET_TPL_TILING_KEY
 *      (loadB2Condition, 0, 0, true, 0) 的既定位编码，TilingKey.md §1 位域表：loadB2Condition
 *      在 DECL 列表 [0,1,2,3] 中下标即值 → bit0-7；kernelSplitMode=0/groupConvMode=0/
 *      loadB1Condition=0 下标 0 → 0；isBasicBlockTiling=true → bit24）∈ {16777217, 16777218,
 *      16777219}；SetBlockDim(coreNum)；workspace 槽 ws[0]=0（本执行配置无用户 workspace）。
 *   §8 Host 侧注册 —— IMPL_OP_OPTILING(Conv3DTranspose).Tiling(Conv3DTransposeTilingFunc)
 *      .TilingParse<Conv3DTransposeCompileInfo>(TilingParseForConv3DTranspose)。
 *   §9 Tiling 整体结构 —— 模板优先序 small_shape → full_load → standard 兜底（full_load
 *      求解失败回退 standard，enableFullLoad 回改 0——InitBaseMNKCommon 恢复）；带 bias 时
 *      求解完成后 UpdateIsBiasFullLoad（最终 baseN/singleCoreCin 上按 BT 容量判定 + L1 实际
 *      分配复核，无 bias 恒过）；回退链每个返回路径均有确定结论（校验失败 / 资源不足
 *      no legal / bias 分配超限 / 成功发布），无静默降级、无部分发布。
 *
 * 求解算法参考：卷积反向范式 generic/tiling.md §1–§10 + 三模板文档（templates/standard、
 * small-shape、l1-resident），实现以同族 20 例验证全绿的直调工程 conv3dtranspose_direct 的
 * Host 求解器为底本转录（范式 example/conv3dtranspose_direct），本文件按注册形态
 * （gert::TilingContext / CompileInfo / 三 filter 布局 / bias BT 通路 / 72 字段共享结构）适配；
 * 分支常量与公式见 design/branches/DESIGN-BRANCH-1677721{7,8,9}.md「具体切分计算」
 * （三分支共享 Host 求解器，无分支私有切分）。
 *
 * 注：生成工程以 arch35/conv3dtranspose_tiling.cpp 为
 * ascend950 目标的唯一 tiling 编译单元，该文件对本实现做编译单元转发（单一注册点不变）。
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "exe_graph/runtime/tiling_context.h"
#include "graph/ascend_string.h"
#include "graph/types.h"
#include "op_common/log/log.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "securec.h"
#include "tiling/platform/platform_ascendc.h"

#include "../../op_kernel/arch35/conv3dtranspose_tiling_data.h"
#include "conv3dtranspose_tiling_arch35.h"

namespace optiling {
namespace {

/* ---------------- loadB2Condition 常量（TilingKey.md §1 声明位，注册形态共享同值） ---------------- */

constexpr uint32_t TPL_NO_TRANSPOSE_NO_REVERSE = 0u;
constexpr uint32_t TPL_TRANSPOSE_AND_REVERSE = 1u;
constexpr uint32_t TPL_REVERSE_ONLY = 2u;
constexpr uint32_t TPL_TRANSPOSE_ONLY = 3u;

/* ---------------- 小工具（generic/tiling 公式用） ---------------- */

inline uint32_t CeilDivU(uint64_t a, uint64_t b)
{
    return b == 0 ? 0 : static_cast<uint32_t>((a + b - 1) / b);
}

inline uint32_t AlignUpU(uint64_t a, uint64_t b)
{
    return CeilDivU(a, b) * static_cast<uint32_t>(b);
}

inline uint32_t AlignDownU(uint64_t a, uint64_t b)
{
    return b == 0 ? 0 : static_cast<uint32_t>(a / b) * static_cast<uint32_t>(b);
}

inline uint64_t GcdU(uint64_t a, uint64_t b)
{
    while (b != 0) {
        const uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

/* 接口单轴/元素值可表示域上界：超出即视为乘积/字段可表示范围溢出（shape_mismatch；
 * HostTiling.md §3.2 行 5「派生乘积超 uint32 字段可表示域 / 尺寸乘积溢出 int64」——
 * 单轴上界 2^31-1 保证全部派生乘积在 uint64 域内可表示） */
constexpr int64_t CONV3DT_MAX_DIM = 2147483647LL; /* 2^31 - 1 */

/* ---------------- 求解上下文 ---------------- */

struct PlatformCaps {
    uint64_t maxCores = 0;
    uint64_t l1Bytes = 0;
    uint64_t l0aBytes = 0;
    uint64_t l0bBytes = 0;
    uint64_t l0cBytes = 0;
    uint64_t btBytes = 0;
};

struct SolveCtx {
    /* 规格归一结果（校验通过后装载） */
    uint32_t batch = 0, cin = 0, cout = 0, cinG = 0, coutG = 0, group = 0;
    uint32_t di = 0, hi = 0, wi = 0, dout = 0, ho = 0, wo = 0;
    uint32_t dk = 0, hk = 0, wk = 0;
    uint32_t strideD = 0, strideH = 0, strideW = 0;
    uint32_t dilationD = 0, dilationH = 0, dilationW = 0;
    uint32_t padFront = 0, padBack = 0, padUp = 0, padDown = 0, padLeft = 0, padRight = 0;
    int32_t backpropPadUp = 0, backpropPadDown = 0, backpropPadLeft = 0, backpropPadRight = 0;
    uint32_t qD = 0, qH = 0, qW = 0;       /* output_padding 空间分量（Host 折入 b⁺/目标几何） */
    uint32_t rhoD = 0, rhoH = 0, rhoW = 0; /* 显式形状余量（哨兵模式恒 0） */
    uint32_t khDilation = 0, kwDilation = 0, kdDilation = 0; /* KeffH/KeffW/KeffD */
    uint32_t hkWk = 0, dkHkWk = 0, windowSpan = 0;
    uint32_t mValue = 0, nValue = 0, kValue = 0;
    uint32_t hoExpand = 0, woExpand = 0;
    uint32_t initOutputFlag = 0;
    ge::Format filterFormat = ge::FORMAT_ND; /* filter 物理布局（独立于 data_format，三选一） */

    /* bias 通路（存在性由槽位判定；biasBytes=fp16 2 / fp32 4） */
    uint32_t hasBias = 0;
    uint32_t biasBytes = 0;

    /* 平台 */
    PlatformCaps caps;

    /* 求解结果 */
    uint32_t baseM = 0, baseN = 0, baseK = 0;
    uint32_t stepKa = 0, stepKb = 0;
    uint32_t al1Pbuffer = 0, bl1Pbuffer = 0, al0Pbuffer = 0, bl0Pbuffer = 0, cl0Pbuffer = 0;
    uint32_t singleCoreM = 0, singleCoreCin = 0, singleCoreDin = 0, singleCoreCout = 0;
    uint32_t enableFullLoad = 0;
    uint32_t isBiasFullLoad = 0;
    uint32_t biasAllocBytes = 0;
    uint32_t loadB2 = 0;
};

/* ---------------- 校验与归一（HostTiling.md §3；锚点契约见文件头） ---------------- */

/*
 * 检查顺序（HostTiling.md §3.2 五行分类，任一失败 OP_LOGE 锚点 + 返回 false）：
 *   1) dtype（三路 dtype 一致 + 白名单 + input_size 整型 + bias/offset_w 槽位 dtype）-> "dtype"
 *   2) format（IO format 合法与配对）                                    -> "format"
 *   3) 维度（rank → input_size 长度契约 → 空张量/可表示域）              -> "rank" / "shape_mismatch" / "null_input"
 *   4) attr 值域（data_format 枚举 / 长度 / 值域 / 装载字段上限 / groups 域）-> "attribute_value_out_of_range"
 *   5) 值读取与元素值域（<0 / ==0 非哨兵 / 超可表示域 / 双模式解析）      -> "shape_mismatch" / "null_input"
 *   6) 值依赖联动（groups 整除 → filter 通道语义 → batch 对齐 → bias 长度
 *      → y 形状契约，动态图 -1 跳过）                                    -> "shape_mismatch"
 *   7) attr 位序复核（strides/dilations N、C 位 = 1；output_padding N、C 位 = 0；
 *      逐轴 q < s 或 q < d）在几何一致性之后复核（位序解释依赖张量轴序与
 *      data_format 已一致——轴序矛盾先行按 shape_mismatch 拒绝，纯位序违约再按
 *      attribute_value_out_of_range 拒绝）
 */
static bool ValidateAndLoad(gert::TilingContext* context, SolveCtx& c)
{
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context, "Conv3DTranspose tiling: GetAttrs() is nullptr");
        return false;
    }
    /* F-01（code-review 整改，2026-11-05 带外授权）：元数据（dtype/format/存在性）一律取自
     * GetInputDesc/GetInputShape——GetInputTensor 仅保留 input_size 的值依赖数据读取
     * （下方 GetData<int32_t/int64_t>），纯元数据场景不再使用张量对象（条款：GetInputTensor
     * 在部分场景可能返回无效数据；Dtype 取 GetInputDesc、Shape 取 context）。 */
    const auto* sDesc = context->GetInputDesc(0);
    const auto* xDesc = context->GetInputDesc(1);
    const auto* fDesc = context->GetInputDesc(2);
    const auto* bDesc = context->GetInputDesc(3);
    const auto* owDesc = context->GetInputDesc(4);
    const auto* sShape = context->GetInputShape(0);
    const auto* xShape = context->GetInputShape(1);
    const auto* fShape = context->GetInputShape(2);
    const auto* bShape = context->GetInputShape(3);
    const auto* owShape = context->GetInputShape(4);
    const auto* yDesc = context->GetOutputDesc(0);
    /* F-01 回退通道（2026-11-06 GEIR 回归发现）：GEIR JIT 编译路径的 TilingContext
     * 不填充 ComputeNodeInfo 的 TD 表（GetInputDesc 全 nullptr，与 UT harness 同型），
     * 此时回退到张量对象读取（原始行为），保持全路径可用；GetInputDesc 可用时仍为
     * 首选（条款口径）。 */
    const auto* sTensorFb = context->GetInputTensor(0);
    const auto* xTensorFb = context->GetInputTensor(1);
    const auto* fTensorFb = context->GetInputTensor(2);
    const auto* bTensorFb = context->GetInputTensor(3);
    const auto* owTensorFb = context->GetInputTensor(4);
    const bool useDescPath = (sDesc != nullptr && xDesc != nullptr && fDesc != nullptr);
    const void* xDtypeSrc = useDescPath ? static_cast<const void*>(xDesc) : static_cast<const void*>(xTensorFb);
    const void* fDtypeSrc = useDescPath ? static_cast<const void*>(fDesc) : static_cast<const void*>(fTensorFb);
    const void* sDtypeSrc = useDescPath ? static_cast<const void*>(sDesc) : static_cast<const void*>(sTensorFb);
    const void* bMetaSrc = useDescPath ? static_cast<const void*>(bDesc) : static_cast<const void*>(bTensorFb);
    const void* owMetaSrc = useDescPath ? static_cast<const void*>(owDesc) : static_cast<const void*>(owTensorFb);
    /* y 元数据：GEIR JIT 无 OutputDesc 时 yFmt 由 xFmt 配对校验兜底（y 与 x 严格配对） */
    const bool hasYDesc = (yDesc != nullptr);
    const auto* sMetaD = static_cast<const gert::CompileTimeTensorDesc*>(sDtypeSrc);
    const auto* xMetaD = static_cast<const gert::CompileTimeTensorDesc*>(xDtypeSrc);
    const auto* fMetaD = static_cast<const gert::CompileTimeTensorDesc*>(fDtypeSrc);
    if (sDtypeSrc == nullptr || xDtypeSrc == nullptr || fDtypeSrc == nullptr || sShape == nullptr ||
        xShape == nullptr || fShape == nullptr) {
        OP_LOGE(context, "Conv3DTranspose tiling: input desc/shape or output desc is nullptr");
        return false;
    }

    /* 可选槽位存在性（HostTiling.md §2「bias 存在性与 dtype 载体」：槽位有效且元素数 > 0
     * 即有 bias——0 元素空可选张量由 op_api 包装层统一归一 nullptr，按无 bias 路由，非报错；
     * offset_w 同口径，量化保留槽不参与计算语义，仅复检 dtype */
    const bool hasBias = bShape != nullptr && bMetaSrc != nullptr && bShape->GetStorageShape().GetShapeSize() > 0;
    const bool hasOffsetW = owShape != nullptr && owMetaSrc != nullptr && owShape->GetStorageShape().GetShapeSize() > 0;

    /* ---------- 行 1 dtype：三路 dtype 一致与组合内（dtype_not_supported） ---------- */
    const ge::DataType xDtype =
        useDescPath ? xMetaD->GetDataType() : static_cast<const gert::Tensor*>(xDtypeSrc)->GetDataType();
    const ge::DataType filterDtype =
        useDescPath ? fMetaD->GetDataType() : static_cast<const gert::Tensor*>(fDtypeSrc)->GetDataType();
    const ge::DataType sizeDtype =
        useDescPath ? sMetaD->GetDataType() : static_cast<const gert::Tensor*>(sDtypeSrc)->GetDataType();
    if (filterDtype != xDtype || (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16) ||
        (sizeDtype != ge::DT_INT32 && sizeDtype != ge::DT_INT64)) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: dtype check failed (dtype_not_supported), "
                "x dtype %d, filter dtype %d, input_size dtype %d; supported: x == filter in "
                "{float16, bfloat16}, input_size in {int32, int64}",
                static_cast<int>(xDtype), static_cast<int>(filterDtype), static_cast<int>(sizeDtype));
        return false;
    }
    if (hasBias && bMetaSrc != nullptr) {
        const ge::DataType biasDtype = useDescPath ?
                                           static_cast<const gert::CompileTimeTensorDesc*>(bMetaSrc)->GetDataType() :
                                           static_cast<const gert::Tensor*>(bMetaSrc)->GetDataType();
        if (biasDtype != ge::DT_FLOAT16 && biasDtype != ge::DT_FLOAT) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling rejected: dtype check failed (dtype_not_supported), "
                    "bias dtype %d not in {float16, float32}",
                    static_cast<int>(biasDtype));
            return false;
        }
        c.biasBytes = (biasDtype == ge::DT_FLOAT16) ? 2u : 4u;
    }
    if (hasOffsetW && owMetaSrc != nullptr) {
        const ge::DataType owDt = (useDescPath && owDesc != nullptr) ?
                                      owDesc->GetDataType() :
                                      (owTensorFb != nullptr ? owTensorFb->GetDataType() : ge::DT_INT8);
        if (owDt != ge::DT_INT8) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling rejected: dtype check failed (dtype_not_supported), "
                    "offset_w dtype %d not int8 (reserved quantization slot)",
                    static_cast<int>(owDt));
            return false;
        }
    }

    /* ---------- 行 2 format：IO format 合法与配对 ---------- */
    const ge::Format sizeFmt =
        useDescPath ? sMetaD->GetStorageFormat() : static_cast<const gert::Tensor*>(sDtypeSrc)->GetStorageFormat();
    const ge::Format filterFmt =
        useDescPath ? fMetaD->GetStorageFormat() : static_cast<const gert::Tensor*>(fDtypeSrc)->GetStorageFormat();
    const ge::Format xFmt =
        useDescPath ? xMetaD->GetStorageFormat() : static_cast<const gert::Tensor*>(xDtypeSrc)->GetStorageFormat();
    /* yFmt：有 OutputDesc 用之；GEIR JIT 回退场景 y 与 x 严格配对（Interface 约束） */
    const ge::Format yFmt = hasYDesc ? yDesc->GetStorageFormat() : xFmt;
    if (sizeFmt != ge::FORMAT_ND) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: format check failed, input_size format %d "
                "is not ND",
                static_cast<int>(sizeFmt));
        return false;
    }
    if (filterFmt != ge::FORMAT_NCDHW && filterFmt != ge::FORMAT_NDHWC && filterFmt != ge::FORMAT_DHWCN) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: format check failed, filter format %d not in "
                "{NCDHW, NDHWC, DHWCN}",
                static_cast<int>(filterFmt));
        return false;
    }
    if (xFmt != ge::FORMAT_NCDHW && xFmt != ge::FORMAT_NDHWC) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: format check failed, x format %d not in "
                "{NCDHW, NDHWC}",
                static_cast<int>(xFmt));
        return false;
    }
    if (yFmt != xFmt) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: format check failed, y format %d not paired "
                "with x format %d",
                static_cast<int>(yFmt), static_cast<int>(xFmt));
        return false;
    }

    /* 属性载体读取（八属性按 OpDef 声明序：strides/pads/dilations/groups/data_format/
     * output_padding/offset_x/enable_hf32） */
    const auto* stridesVec = attrs->GetListInt(0);
    const auto* padsVec = attrs->GetListInt(1);
    const auto* dilationsVec = attrs->GetListInt(2);
    const auto* groupsPtr = attrs->GetInt(3);
    const char* dataFormatRaw = attrs->GetStr(4);
    const auto* outputPaddingVec = attrs->GetListInt(5);
    const auto* offsetXPtr = attrs->GetInt(6);
    const auto* hf32Ptr = attrs->GetBool(7);
    if (stridesVec == nullptr || padsVec == nullptr || dilationsVec == nullptr || groupsPtr == nullptr ||
        dataFormatRaw == nullptr || outputPaddingVec == nullptr || offsetXPtr == nullptr || hf32Ptr == nullptr) {
        OP_LOGE(context, "Conv3DTranspose tiling: attr pointer is nullptr");
        return false;
    }
    /* offset_x / enable_hf32 信息性消费：本交付 dtype 集（FP16/BF16）与量化残留下 Host
     * 不据其分支（HostTiling.md §2「保留属性语义」），不参与任何 tiling 决策 */

    /* data_format 枚举（错误分类按违反项归位：attribute_value_out_of_range） */
    const std::string dataFormat(dataFormatRaw);
    bool isNdhwcData = false;
    if (dataFormat == "NCDHW") {
        isNdhwcData = false;
    } else if (dataFormat == "NDHWC") {
        isNdhwcData = true;
    } else {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: attribute_value_out_of_range, data_format "
                "'%s' not in {NCDHW, NDHWC}",
                dataFormat.c_str());
        return false;
    }

    /* ---------- 行 3 维度：rank / input_size 长度契约 / 空张量 ---------- */
    const size_t sRank = sShape->GetStorageShape().GetDimNum();
    const size_t xRank = xShape->GetStorageShape().GetDimNum();
    const size_t fRank = fShape->GetStorageShape().GetDimNum();
    if (sRank != 1U) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: rank check failed, input_size rank is %zu, "
                "expect 1",
                sRank);
        return false;
    }
    if (xRank != 5U) {
        OP_LOGE(context, "Conv3DTranspose tiling rejected: rank check failed, x rank is %zu, expect 5", xRank);
        return false;
    }
    if (fRank != 5U) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: rank check failed, filter rank is %zu, "
                "expect 5",
                fRank);
        return false;
    }
    if (sShape->GetStorageShape().GetDim(0) != 5) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: shape_mismatch, input_size length is %lld, "
                "expect 5 (1D int32/int64, length 5 contract)",
                static_cast<long long>(sShape->GetStorageShape().GetDim(0)));
        return false;
    }
    for (size_t i = 0; i < 5U; ++i) {
        if (xShape->GetStorageShape().GetDim(i) <= 0 || fShape->GetStorageShape().GetDim(i) <= 0) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling rejected: null_input, empty tensor detected, dim "
                    "%zu of x/filter is non-positive (x %lld, filter %lld)",
                    i, static_cast<long long>(xShape->GetStorageShape().GetDim(i)),
                    static_cast<long long>(fShape->GetStorageShape().GetDim(i)));
            return false;
        }
        if (xShape->GetStorageShape().GetDim(i) > CONV3DT_MAX_DIM ||
            fShape->GetStorageShape().GetDim(i) > CONV3DT_MAX_DIM) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling rejected: shape_mismatch, dim %zu of x/filter "
                    "exceeds representable range %lld (product overflow)",
                    i, static_cast<long long>(CONV3DT_MAX_DIM));
            return false;
        }
    }

    /* ---------- 行 4 attr 值域与长度（含转置专属逐轴 q 约束；N/C 位序项延后到几何一致性之后） ---------- */
    if (stridesVec->GetSize() != 5U || dilationsVec->GetSize() != 5U || outputPaddingVec->GetSize() != 5U) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: attribute_value_out_of_range, strides/"
                "dilations/output_padding list length %zu/%zu/%zu, expect 5",
                stridesVec->GetSize(), dilationsVec->GetSize(), outputPaddingVec->GetSize());
        return false;
    }
    if (padsVec->GetSize() != 1U && padsVec->GetSize() != 3U && padsVec->GetSize() != 6U) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: attribute_value_out_of_range, pads list "
                "length %zu, expect 1/3/6 (1 元三轴同值 / 3 元逐轴前后同值 / 6 元逐侧独立)",
                padsVec->GetSize());
        return false;
    }
    const int64_t* strides = stridesVec->GetData();
    const int64_t* pads = padsVec->GetData();
    const int64_t* dilations = dilationsVec->GetData();
    const int64_t* outputPadding = outputPaddingVec->GetData();
    for (size_t i = 0; i < 5U; ++i) {
        if (strides[i] < 1 || strides[i] > CONV3DT_MAX_DIM) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling rejected: attribute_value_out_of_range, strides[%zu]"
                    "=%lld out of [1, 2147483647]",
                    i, static_cast<long long>(strides[i]));
            return false;
        }
        if (dilations[i] < 1 || dilations[i] > CONV3DT_MAX_DIM) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling rejected: attribute_value_out_of_range, "
                    "dilations[%zu]=%lld out of [1, 2147483647]",
                    i, static_cast<long long>(dilations[i]));
            return false;
        }
        if (outputPadding[i] < 0 || outputPadding[i] > CONV3DT_MAX_DIM) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling rejected: attribute_value_out_of_range, "
                    "output_padding[%zu]=%lld out of [0, 2147483647]",
                    i, static_cast<long long>(outputPadding[i]));
            return false;
        }
    }
    for (size_t i = 0; i < padsVec->GetSize(); ++i) {
        if (pads[i] < 0 || pads[i] > CONV3DT_MAX_DIM) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling rejected: attribute_value_out_of_range, pads[%zu]"
                    "=%lld out of [0, 2147483647]",
                    i, static_cast<long long>(pads[i]));
            return false;
        }
    }
    /* 装载字段上限（本执行配置交付边界：接口值域内的超限值按属性非法拒绝，
     * CONV3DTRANSPOSE_MAX_STRIDE/DILATION/PAD） */
    for (size_t i = 0; i < 5U; ++i) {
        if (strides[i] > static_cast<int64_t>(CONV3DTRANSPOSE_MAX_STRIDE)) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling rejected: attribute_value_out_of_range, strides[%zu]"
                    "=%lld exceeds loadable field cap %u",
                    i, static_cast<long long>(strides[i]), CONV3DTRANSPOSE_MAX_STRIDE);
            return false;
        }
        if (dilations[i] > static_cast<int64_t>(CONV3DTRANSPOSE_MAX_DILATION)) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling rejected: attribute_value_out_of_range, "
                    "dilations[%zu]=%lld exceeds loadable field cap %u",
                    i, static_cast<long long>(dilations[i]), CONV3DTRANSPOSE_MAX_DILATION);
            return false;
        }
    }
    for (size_t i = 0; i < padsVec->GetSize(); ++i) {
        if (pads[i] > static_cast<int64_t>(CONV3DTRANSPOSE_MAX_PAD)) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling rejected: attribute_value_out_of_range, pads[%zu]"
                    "=%lld exceeds loadable field cap %u",
                    i, static_cast<long long>(pads[i]), CONV3DTRANSPOSE_MAX_PAD);
            return false;
        }
    }
    if (*groupsPtr < 1 || *groupsPtr > 65535) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: attribute_value_out_of_range, groups %lld "
                "out of [1, 65535]",
                static_cast<long long>(*groupsPtr));
        return false;
    }

    /* 属性空间分量提取（位序合轴，HostTiling.md §3.1 步骤 3）：NCDHW 取 [2:5]、NDHWC 取
     * [1:4]；pads 语义序 [pF, pB, pU, pDn, pL, pR] 与 data_format 无关，按 1/3/6 元展开
     * （1 元三轴同值、3 元逐轴前后同值、6 元逐侧独立） */
    const int64_t sD = isNdhwcData ? strides[1] : strides[2];
    const int64_t sH = isNdhwcData ? strides[2] : strides[3];
    const int64_t sW = isNdhwcData ? strides[3] : strides[4];
    const int64_t dD = isNdhwcData ? dilations[1] : dilations[2];
    const int64_t dH = isNdhwcData ? dilations[2] : dilations[3];
    const int64_t dW = isNdhwcData ? dilations[3] : dilations[4];
    const int64_t qD = isNdhwcData ? outputPadding[1] : outputPadding[2];
    const int64_t qH = isNdhwcData ? outputPadding[2] : outputPadding[3];
    const int64_t qW = isNdhwcData ? outputPadding[3] : outputPadding[4];
    int64_t pads6[6] = {0, 0, 0, 0, 0, 0};
    const size_t padsLen = padsVec->GetSize();
    if (padsLen == 1U) {
        for (size_t i = 0; i < 6U; ++i) {
            pads6[i] = pads[0]; /* 1 元三轴同值 */
        }
    } else if (padsLen == 3U) {
        for (size_t a = 0; a < 3U; ++a) {
            pads6[2 * a] = pads[a];     /* 逐轴前侧 */
            pads6[2 * a + 1] = pads[a]; /* 逐轴后侧同值 */
        }
    } else {
        for (size_t i = 0; i < 6U; ++i) {
            pads6[i] = pads[i]; /* 6 元逐侧独立 */
        }
    }

    /* 转置专属逐轴约束（HostTiling.md §3.2 行 4：逐轴 q < s 或 q < d；违反 ⟺ q ≥ s 且
     * q ≥ d）——属 attr 行检查，先于值读取与几何/通道检查 */
    if ((qD >= sD && qD >= dD) || (qH >= sH && qH >= dH) || (qW >= sW && qW >= dW)) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: attribute_value_out_of_range, "
                "output_padding spatial components (qD %lld, qH %lld, qW %lld) violate per-axis "
                "q < stride or q < dilation (s %lld/%lld/%lld, d %lld/%lld/%lld)",
                static_cast<long long>(qD), static_cast<long long>(qH), static_cast<long long>(qW),
                static_cast<long long>(sD), static_cast<long long>(sH), static_cast<long long>(sW),
                static_cast<long long>(dD), static_cast<long long>(dH), static_cast<long long>(dW));
        return false;
    }

    /* ---------- input_size 值读取（值依赖：元数据校验全部通过后进行） ----------
     * F-01 整改：此处为 input_size 的合法值依赖数据读取，GetInputTensor 保留；
     * dtype 已从 GetInputDesc(0) 取得（sizeDtype），与张量对象无关。 */
    int64_t sizeVals[5] = {0, 0, 0, 0, 0};
    const void* sizeData = nullptr;
    const auto* sTensor = context->GetInputTensor(0);
    if (sTensor != nullptr && sizeDtype == ge::DT_INT64) {
        sizeData = sTensor->GetData<int64_t>();
    } else if (sTensor != nullptr) {
        sizeData = sTensor->GetData<int32_t>();
    }
    if (sizeData == nullptr) {
        OP_LOGE(context, "Conv3DTranspose tiling rejected: null_input, input_size tensor data is nullptr "
                         "(value-dependent read unavailable)");
        return false;
    }
    for (size_t i = 0; i < 5U; ++i) {
        sizeVals[i] = (sizeDtype == ge::DT_INT64) ? (static_cast<const int64_t*>(sizeData))[i] :
                                                    static_cast<int64_t>((static_cast<const int32_t*>(sizeData))[i]);
    }

    /* 轴序归一（HostTiling.md §3.1 步骤 2）：input_size 值序 NCDHW (N, Cin, Din, Hin, Win) /
     * NDHWC (N, Din, Hin, Win, Cin)；x 轴序随 data_format（NCDHW (N, Cout, Dx, Hx, Wx) /
     * NDHWC (N, Dx, Hx, Wx, Cout)）；filter 布局独立于 data_format（NCDHW (Cout, CinG, kD,
     * kH, kW) / NDHWC (Cout, kD, kH, kW, CinG) / DHWCN (kD, kH, kW, CinG, Cout)） */
    const int64_t nX = xShape->GetStorageShape().GetDim(0);
    const int64_t coutX = isNdhwcData ? xShape->GetStorageShape().GetDim(4) : xShape->GetStorageShape().GetDim(1);
    const int64_t doutX = isNdhwcData ? xShape->GetStorageShape().GetDim(1) : xShape->GetStorageShape().GetDim(2);
    const int64_t houtX = isNdhwcData ? xShape->GetStorageShape().GetDim(2) : xShape->GetStorageShape().GetDim(3);
    const int64_t woutX = isNdhwcData ? xShape->GetStorageShape().GetDim(3) : xShape->GetStorageShape().GetDim(4);
    int64_t coutF = 0;
    int64_t cinGF = 0;
    int64_t kdF = 0;
    int64_t khF = 0;
    int64_t kwF = 0;
    if (filterFmt == ge::FORMAT_NCDHW) {
        coutF = fShape->GetStorageShape().GetDim(0);
        cinGF = fShape->GetStorageShape().GetDim(1);
        kdF = fShape->GetStorageShape().GetDim(2);
        khF = fShape->GetStorageShape().GetDim(3);
        kwF = fShape->GetStorageShape().GetDim(4);
    } else if (filterFmt == ge::FORMAT_NDHWC) {
        coutF = fShape->GetStorageShape().GetDim(0);
        kdF = fShape->GetStorageShape().GetDim(1);
        khF = fShape->GetStorageShape().GetDim(2);
        kwF = fShape->GetStorageShape().GetDim(3);
        cinGF = fShape->GetStorageShape().GetDim(4);
    } else { /* DHWCN (kD, kH, kW, CinG, Cout) */
        kdF = fShape->GetStorageShape().GetDim(0);
        khF = fShape->GetStorageShape().GetDim(1);
        kwF = fShape->GetStorageShape().GetDim(2);
        cinGF = fShape->GetStorageShape().GetDim(3);
        coutF = fShape->GetStorageShape().GetDim(4);
    }

    /* 属性空间分量与 pads 展开已在行 4（attr 阶段）提取（sD/sH/sW、dD/dH/dW、qD/qH/qW、
     * pads6）；有效核长（逐轴） */
    const int64_t kEffD = (kdF - 1) * dD + 1;
    const int64_t kEffH = (khF - 1) * dH + 1;
    const int64_t kEffW = (kwF - 1) * dW + 1;

    /* ---------- 行 5 值域与双模式解析（HostTiling.md §3.1 步骤 1） ----------
     * base_a = (L_x−1)·s_a − p_a^− − p_a^+ + Keff_a；显式模式逐轴 rho_a = L_y − (base+q)
     * ∈ [0, s_a)；全零哨兵为属性推导模式的合法请求（rho=0，推导结果逐维 ≥ 1）。 */
    int64_t nIn = 0;
    int64_t cinIn = 0;
    int64_t dinIn = 0;
    int64_t hinIn = 0;
    int64_t winIn = 0;
    bool allZero = true;
    for (size_t i = 0; i < 5U; ++i) {
        if (sizeVals[i] != 0) {
            allZero = false;
            break;
        }
    }
    if (allZero) {
        /* 全零哨兵：属性推导模式——N=x.shape[0]、Cin=filter.CinG×groups、逐轴 L_y=base+q
         * （rho=0，合法请求正常路由，不报错、不静默空跑）；推导结果逐维 ≥ 1
         * （0 → 空张量判据 null_input；负 → 非法维值 shape_mismatch） */
        nIn = nX;
        cinIn = cinGF * (*groupsPtr);
        dinIn = (doutX - 1) * sD - pads6[0] - pads6[1] + kEffD + qD;
        hinIn = (houtX - 1) * sH - pads6[2] - pads6[3] + kEffH + qH;
        winIn = (woutX - 1) * sW - pads6[4] - pads6[5] + kEffW + qW;
        /* 按 input_size 值序装载（NCDHW (N, Cin, Din, Hin, Win)；NDHWC (N, Din, Hin, Win, Cin)） */
        int64_t ordered[5];
        if (isNdhwcData) {
            ordered[0] = nIn;
            ordered[1] = dinIn;
            ordered[2] = hinIn;
            ordered[3] = winIn;
            ordered[4] = cinIn;
        } else {
            ordered[0] = nIn;
            ordered[1] = cinIn;
            ordered[2] = dinIn;
            ordered[3] = hinIn;
            ordered[4] = winIn;
        }
        for (size_t i = 0; i < 5U; ++i) {
            if (ordered[i] < 0) {
                OP_LOGE(context,
                        "Conv3DTranspose tiling rejected: shape_mismatch, sentinel-derived "
                        "output dim %zu is %lld (negative dim value, illegal)",
                        i, static_cast<long long>(ordered[i]));
                return false;
            }
            if (ordered[i] == 0) {
                OP_LOGE(context,
                        "Conv3DTranspose tiling rejected: null_input, sentinel-derived output "
                        "dim %zu is 0 (empty output tensor)",
                        i);
                return false;
            }
            if (ordered[i] > CONV3DT_MAX_DIM) {
                OP_LOGE(context,
                        "Conv3DTranspose tiling rejected: shape_mismatch, sentinel-derived "
                        "output dim %zu is %lld, exceeds representable range %lld (product overflow)",
                        i, static_cast<long long>(ordered[i]), static_cast<long long>(CONV3DT_MAX_DIM));
                return false;
            }
        }
        if (isNdhwcData) {
            nIn = ordered[0];
            dinIn = ordered[1];
            hinIn = ordered[2];
            winIn = ordered[3];
            cinIn = ordered[4];
        } else {
            nIn = ordered[0];
            cinIn = ordered[1];
            dinIn = ordered[2];
            hinIn = ordered[3];
            winIn = ordered[4];
        }
    } else {
        for (size_t i = 0; i < 5U; ++i) {
            if (sizeVals[i] < 0) {
                OP_LOGE(context,
                        "Conv3DTranspose tiling rejected: shape_mismatch, input_size element "
                        "%zu is %lld (negative dim value, no unknown-dim semantics in registered "
                        "combos)",
                        i, static_cast<long long>(sizeVals[i]));
                return false;
            }
            if (sizeVals[i] == 0) {
                OP_LOGE(context,
                        "Conv3DTranspose tiling rejected: null_input, input_size element %zu "
                        "is 0 (empty output tensor; all-zero sentinel is the only legal zero "
                        "request)",
                        i);
                return false;
            }
            if (sizeVals[i] > CONV3DT_MAX_DIM) {
                OP_LOGE(context,
                        "Conv3DTranspose tiling rejected: shape_mismatch, input_size element "
                        "%zu is %lld, exceeds representable range %lld (product overflow)",
                        i, static_cast<long long>(sizeVals[i]), static_cast<long long>(CONV3DT_MAX_DIM));
                return false;
            }
        }
        nIn = sizeVals[0];
        cinIn = isNdhwcData ? sizeVals[4] : sizeVals[1];
        dinIn = isNdhwcData ? sizeVals[1] : sizeVals[2];
        hinIn = isNdhwcData ? sizeVals[2] : sizeVals[3];
        winIn = isNdhwcData ? sizeVals[3] : sizeVals[4];

        /* 转置几何一致性（显式模式逐轴 rho 校验，等价反算
         * L_x = (L_y + p⁻ + p⁺ − (Keff + q)) // s + 1 与 x 实际空间维一致） */
        const int64_t expectDx = (dinIn + pads6[0] + pads6[1] - (kEffD + qD)) / sD + 1;
        const int64_t expectHx = (hinIn + pads6[2] + pads6[3] - (kEffH + qH)) / sH + 1;
        const int64_t expectWx = (winIn + pads6[4] + pads6[5] - (kEffW + qW)) / sW + 1;
        if (expectDx != doutX || expectHx != houtX || expectWx != woutX) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling rejected: shape_mismatch, transpose geometry "
                    "inconsistent: expect x spatial (%lld, %lld, %lld), got (%lld, %lld, %lld) "
                    "(rho out of [0, stride) on some axis; PyTorch check_shape_backward parity)",
                    static_cast<long long>(expectDx), static_cast<long long>(expectHx),
                    static_cast<long long>(expectWx), static_cast<long long>(doutX), static_cast<long long>(houtX),
                    static_cast<long long>(woutX));
            return false;
        }
    }

    /* ---------- 值依赖联动校验（HostTiling.md §3.2 行 5） ---------- */
    if (cinIn % (*groupsPtr) != 0) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: shape_mismatch, groups %lld does not divide "
                "Cin %lld",
                static_cast<long long>(*groupsPtr), static_cast<long long>(cinIn));
        return false;
    }
    if (cinIn != cinGF * (*groupsPtr)) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: shape_mismatch, filter CinG %lld x groups "
                "%lld != input_size Cin %lld (filter channel dims violate groups semantics)",
                static_cast<long long>(cinGF), static_cast<long long>(*groupsPtr), static_cast<long long>(cinIn));
        return false;
    }
    if (coutF % (*groupsPtr) != 0) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: shape_mismatch, groups %lld does not divide "
                "filter Cout %lld",
                static_cast<long long>(*groupsPtr), static_cast<long long>(coutF));
        return false;
    }
    if (coutX != coutF) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: shape_mismatch, x Cout %lld != filter Cout "
                "%lld (no broadcast alignment)",
                static_cast<long long>(coutX), static_cast<long long>(coutF));
        return false;
    }
    if (nX != nIn) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: shape_mismatch, x N %lld != input_size N "
                "%lld (fixed batch dim relation)",
                static_cast<long long>(nX), static_cast<long long>(nIn));
        return false;
    }
    if (hasBias) {
        if (bShape->GetStorageShape().GetDimNum() != 1U) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling rejected: shape_mismatch, bias rank %zu != 1 "
                    "(1D x Cin contract)",
                    bShape->GetStorageShape().GetDimNum());
            return false;
        }
        if (bShape->GetStorageShape().GetShapeSize() != cinIn) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling rejected: shape_mismatch, bias length %lld != Cin "
                    "%lld (bias broadcasts along y channel)",
                    static_cast<long long>(bShape->GetStorageShape().GetShapeSize()), static_cast<long long>(cinIn));
            return false;
        }
    }

    /* ---------- 输出形状契约校验（y.shape = input_size 元素值；值依赖防线） ----------
     * kernel 按解析几何写满全输出，out 实际分配小于该几何即构成 GM 越界写；动态图输出维
     * 含 -1 跳过比对（运行期分配由值依赖 InferShape 从同一 input_size 值同源推导）。 */
    const gert::StorageShape* yShapeCtx = context->GetOutputShape(0);
    if (yShapeCtx == nullptr) {
        OP_LOGE(context, "Conv3DTranspose tiling rejected: output shape is unavailable "
                         "(GetOutputShape(0) is nullptr)");
        return false;
    }
    const gert::Shape& yStorageShape = yShapeCtx->GetStorageShape();
    if (yStorageShape.GetDimNum() != 5U) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: shape_mismatch, output y rank is %zu, "
                "expect 5",
                yStorageShape.GetDimNum());
        return false;
    }
    int64_t parsed[5];
    if (isNdhwcData) {
        parsed[0] = nIn;
        parsed[1] = dinIn;
        parsed[2] = hinIn;
        parsed[3] = winIn;
        parsed[4] = cinIn;
    } else {
        parsed[0] = nIn;
        parsed[1] = cinIn;
        parsed[2] = dinIn;
        parsed[3] = hinIn;
        parsed[4] = winIn;
    }
    bool yShapeKnown = true;
    for (size_t i = 0; i < 5U; ++i) {
        if (yStorageShape.GetDim(i) < 0) {
            yShapeKnown = false;
            break;
        }
    }
    if (yShapeKnown) {
        for (size_t i = 0; i < 5U; ++i) {
            if (yStorageShape.GetDim(i) != parsed[i]) {
                OP_LOGE(context,
                        "Conv3DTranspose tiling rejected: shape_mismatch, output y dim %zu is "
                        "%lld but input_size element is %lld (y.shape = input_size values contract; "
                        "smaller out would be overflowed by kernel write-back)",
                        i, static_cast<long long>(yStorageShape.GetDim(i)), static_cast<long long>(parsed[i]));
                return false;
            }
        }
    } else {
        OP_LOGD(context, "Conv3DTranspose tiling: output y shape has unknown dim (-1, dynamic graph), "
                         "skip output-shape contract check (runtime allocation derives from the same input_size "
                         "values via value-dependent InferShape)");
    }

    /* ---------- 规格归一装载（HostTiling.md §3.1 步骤 4：派生量 q+rho 折入） ---------- */
    const uint32_t group = static_cast<uint32_t>(*groupsPtr);
    c.batch = static_cast<uint32_t>(nIn);
    c.cin = static_cast<uint32_t>(cinIn);
    c.cout = static_cast<uint32_t>(coutF);
    c.group = group;
    c.cinG = static_cast<uint32_t>(cinIn / group);
    c.coutG = static_cast<uint32_t>(coutF / group);
    c.di = static_cast<uint32_t>(dinIn);
    c.hi = static_cast<uint32_t>(hinIn);
    c.wi = static_cast<uint32_t>(winIn);
    c.dout = static_cast<uint32_t>(doutX);
    c.ho = static_cast<uint32_t>(houtX);
    c.wo = static_cast<uint32_t>(woutX);
    c.dk = static_cast<uint32_t>(kdF);
    c.hk = static_cast<uint32_t>(khF);
    c.wk = static_cast<uint32_t>(kwF);
    c.strideD = static_cast<uint32_t>(sD);
    c.strideH = static_cast<uint32_t>(sH);
    c.strideW = static_cast<uint32_t>(sW);
    c.dilationD = static_cast<uint32_t>(dD);
    c.dilationH = static_cast<uint32_t>(dH);
    c.dilationW = static_cast<uint32_t>(dW);
    c.padFront = static_cast<uint32_t>(pads6[0]);
    c.padBack = static_cast<uint32_t>(pads6[1]);
    c.padUp = static_cast<uint32_t>(pads6[2]);
    c.padDown = static_cast<uint32_t>(pads6[3]);
    c.padLeft = static_cast<uint32_t>(pads6[4]);
    c.padRight = static_cast<uint32_t>(pads6[5]);
    c.qD = static_cast<uint32_t>(qD);
    c.qH = static_cast<uint32_t>(qH);
    c.qW = static_cast<uint32_t>(qW);
    c.filterFormat = filterFmt;
    c.hasBias = hasBias ? 1u : 0u;

    /* 派生核量先以 uint64 宽整数计算并复核可表示域，再收窄装载（HostTiling.md §3.3/§7
     * 大乘积宽整数检查） */
    const uint64_t effD64 = (static_cast<uint64_t>(c.dk) - 1) * c.dilationD + 1;
    const uint64_t effH64 = (static_cast<uint64_t>(c.hk) - 1) * c.dilationH + 1;
    const uint64_t effW64 = (static_cast<uint64_t>(c.wk) - 1) * c.dilationW + 1;
    const uint64_t hkWk64 = static_cast<uint64_t>(c.hk) * c.wk;
    const uint64_t dkHkWk64 = static_cast<uint64_t>(c.dk) * hkWk64;
    const uint64_t windowSpan64 = hkWk64 * CONV3DTRANSPOSE_C0;
    if (effD64 > 0xFFFFFFFFULL || effH64 > 0xFFFFFFFFULL || effW64 > 0xFFFFFFFFULL || hkWk64 > 0xFFFFFFFFULL ||
        dkHkWk64 > 0xFFFFFFFFULL || windowSpan64 > 0xFFFFFFFFULL) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: shape_mismatch, derived kernel quantity "
                "exceeds uint32 field range (KeffD %llu, KeffH %llu, KeffW %llu, hk*wk %llu, "
                "dk*hk*wk %llu)",
                (unsigned long long)effD64, (unsigned long long)effH64, (unsigned long long)effW64,
                (unsigned long long)hkWk64, (unsigned long long)dkHkWk64);
        return false;
    }
    c.khDilation = static_cast<uint32_t>(effH64);
    c.kwDilation = static_cast<uint32_t>(effW64);
    c.kdDilation = static_cast<uint32_t>(effD64);
    c.hkWk = static_cast<uint32_t>(hkWk64);
    c.dkHkWk = static_cast<uint32_t>(dkHkWk64);
    c.windowSpan = static_cast<uint32_t>(windowSpan64);

    /* rho / 反向边界（HostTiling.md §3.3：b⁻ = Keff−1−p⁻，b⁺ = Keff−1−p⁺+q+rho——
     * q 与显式形状余量 rho 由 Host 折入后侧边界，设备不感知 q/rho；哨兵模式 rho=0） */
    if (allZero) {
        c.rhoD = 0;
        c.rhoH = 0;
        c.rhoW = 0;
    } else {
        const int64_t rhoD64 = static_cast<int64_t>(c.di) - (static_cast<int64_t>(c.dout - 1) * c.strideD - c.padFront -
                                                             c.padBack + static_cast<int64_t>(c.kdDilation) + c.qD);
        const int64_t rhoH64 = static_cast<int64_t>(c.hi) - (static_cast<int64_t>(c.ho - 1) * c.strideH - c.padUp -
                                                             c.padDown + static_cast<int64_t>(c.khDilation) + c.qH);
        const int64_t rhoW64 = static_cast<int64_t>(c.wi) - (static_cast<int64_t>(c.wo - 1) * c.strideW - c.padLeft -
                                                             c.padRight + static_cast<int64_t>(c.kwDilation) + c.qW);
        if (rhoD64 < 0 || rhoD64 >= static_cast<int64_t>(c.strideD) || rhoH64 < 0 ||
            rhoH64 >= static_cast<int64_t>(c.strideH) || rhoW64 < 0 || rhoW64 >= static_cast<int64_t>(c.strideW)) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling rejected: shape_mismatch, explicit-shape slack "
                    "rho (D %lld, H %lld, W %lld) not in [0, stride) (stride %u/%u/%u)",
                    static_cast<long long>(rhoD64), static_cast<long long>(rhoH64), static_cast<long long>(rhoW64),
                    c.strideD, c.strideH, c.strideW);
            return false;
        }
        c.rhoD = static_cast<uint32_t>(rhoD64);
        c.rhoH = static_cast<uint32_t>(rhoH64);
        c.rhoW = static_cast<uint32_t>(rhoW64);
    }
    c.backpropPadUp = static_cast<int32_t>(c.khDilation - 1 - c.padUp);
    c.backpropPadDown = static_cast<int32_t>(c.khDilation - 1 - c.padDown + c.qH + c.rhoH);
    c.backpropPadLeft = static_cast<int32_t>(c.kwDilation - 1 - c.padLeft);
    c.backpropPadRight = static_cast<int32_t>(c.kwDilation - 1 - c.padRight + c.qW + c.rhoW);
    /* F-03（code-review 整改，2026-11-05 带外授权）：四侧反向边界任一超 255 时响亮拒绝——
     * Kernel LoadData3D padList 为 uint8[PAD_SIZE]，超域缩窄为静默截断（合法输入下错误
     * FMatrix 补边/窗口起点 → 静默错值）。与 strides/dilations/pads 的 MAX_* 装载字段上限
     * 拒绝口径一致；原「收紧 maxM」路径删除（不再依赖求解器降级掩盖装载域溢出）。 */
    if (c.backpropPadLeft > static_cast<int32_t>(CONV3DTRANSPOSE_MAX_PAD) ||
        c.backpropPadUp > static_cast<int32_t>(CONV3DTRANSPOSE_MAX_PAD) ||
        c.backpropPadDown > static_cast<int32_t>(CONV3DTRANSPOSE_MAX_PAD) ||
        c.backpropPadRight > static_cast<int32_t>(CONV3DTRANSPOSE_MAX_PAD)) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: attribute_value_out_of_range, effective "
                "backprop pad (L=%d U=%d D=%d R=%d) exceeds LoadData3D padList uint8 field limit "
                "%d (kw/kh dilation × pad geometry too large)",
                c.backpropPadLeft, c.backpropPadUp, c.backpropPadDown, c.backpropPadRight,
                static_cast<int32_t>(CONV3DTRANSPOSE_MAX_PAD));
        return false;
    }
    /* F-03（mStartPt 子项，fix-findings 轮补齐，2026-11-06）：LoadData3D 的窗口起点
     * mStartPt 为 uint16，装载值 = 块空间起点 mStart_ % wi（M 空间按 baseM≤512 步进、
     * 任务起点不必整行对齐时，块起点 W 坐标可取 [0, wi) 任意值）。wi > 65535 时合法
     * 输入可构造出落入 (65535, wi) 的块起点 → 静默截断（审查处方 4 首选：Host 限制
     * wi 使 mStart_%%wi 落入 uint16 域）。Kernel 侧 CONV3DTRANSPOSE_FIELD_ASSERT 仅
     * Host 编译链生效（__CCE__ 设备编译降级 no-op），本拒绝为权威防线。 */
    if (c.wi > 65535u) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: attribute_value_out_of_range, dedx W "
                "extent wi %u exceeds LoadData3D mStartPt uint16 field limit 65535 (block-start "
                "W coordinate = mStart %% wi would silently truncate)",
                c.wi);
        return false;
    }
    c.hoExpand = (c.ho - 1) * c.strideH + 1;
    c.woExpand = (c.wo - 1) * c.strideW + 1;

    /* ---------- attr 位序复核（几何一致性之后）：N/C 位与逐轴 q 约束 ---------- */
    {
        const size_t nPos = 0;
        const size_t cPos = isNdhwcData ? 4U : 1U;
        if (strides[nPos] != 1 || strides[cPos] != 1) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling rejected: attribute_value_out_of_range, strides "
                    "N/C positions (N at %zu = %lld, C at %zu = %lld) must be 1",
                    nPos, static_cast<long long>(strides[nPos]), cPos, static_cast<long long>(strides[cPos]));
            return false;
        }
        if (dilations[nPos] != 1 || dilations[cPos] != 1) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling rejected: attribute_value_out_of_range, dilations "
                    "N/C positions (N at %zu = %lld, C at %zu = %lld) must be 1",
                    nPos, static_cast<long long>(dilations[nPos]), cPos, static_cast<long long>(dilations[cPos]));
            return false;
        }
        if (outputPadding[nPos] != 0 || outputPadding[cPos] != 0) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling rejected: attribute_value_out_of_range, "
                    "output_padding N/C positions (N at %zu = %lld, C at %zu = %lld) must be 0",
                    nPos, static_cast<long long>(outputPadding[nPos]), cPos,
                    static_cast<long long>(outputPadding[cPos]));
            return false;
        }
        /* 逐轴 q < s 或 q < d 约束已在行 4（attr 阶段）先于值读取复核 */
    }

    /* 派生乘积可表示域复核（uint32 字段域；宽整数计算后再收窄，HostTiling.md §7 发布前检查） */
    const uint64_t hiWi = static_cast<uint64_t>(c.hi) * c.wi;
    const unsigned __int128 diHiWi = static_cast<unsigned __int128>(c.di) * hiWi;
    const unsigned __int128 doHoWo = static_cast<unsigned __int128>(c.dout) * c.ho * c.wo;
    const uint64_t coutAlign = AlignUpU(c.coutG, CONV3DTRANSPOSE_C0);
    const uint64_t kValue64 = coutAlign * c.hkWk;
    if (hiWi > 0xFFFFFFFFULL || diHiWi > 0xFFFFFFFFULL || doHoWo > 0xFFFFFFFFULL || kValue64 > 0xFFFFFFFFULL) {
        OP_LOGE(context,
                "Conv3DTranspose tiling rejected: shape_mismatch, derived product exceeds "
                "uint32 field range (hi*wi %llu, di*hi*wi %llu, do*ho*wo %llu, kValue %llu)",
                (unsigned long long)hiWi, (unsigned long long)(uint64_t)diHiWi, (unsigned long long)(uint64_t)doHoWo,
                (unsigned long long)kValue64);
        return false;
    }
    c.mValue = AlignUpU(hiWi, CONV3DTRANSPOSE_FRACTAL);
    c.nValue = AlignUpU(c.cinG, CONV3DTRANSPOSE_FRACTAL);
    c.kValue = static_cast<uint32_t>(kValue64);

    /* 场景预处理——输出初始化判定（preprocessing §5.2 转置条件表，信息性标志：
     * 本交付 kernel 采用主计算完整写回协议，kernel 不据其跳过任何块的写出） */
    c.initOutputFlag = 0;
    if (c.rhoD > c.padBack || c.rhoH > c.padDown || c.strideH > c.hk || c.qD > 0 || c.qH > 0 || c.dilationD > 1 ||
        c.strideD > c.dk) {
        c.initOutputFlag = 1;
    }
    return true;
}

/* ---------------- 平台资源（HostTiling.md §4.1/§8：CompileInfo 优先，缺失/非正回退查询） ---------------- */

static bool ResolvePlatformCaps(gert::TilingContext* context, SolveCtx& c)
{
    const auto* compileInfo = reinterpret_cast<const Conv3DTransposeCompileInfo*>(context->GetCompileInfo());
    bool fromCompileInfo = compileInfo != nullptr;
    if (fromCompileInfo) {
        fromCompileInfo = compileInfo->coreNum > 0 && compileInfo->l1Bytes > 0 && compileInfo->l0aBytes > 0 &&
                          compileInfo->l0bBytes > 0 && compileInfo->l0cBytes > 0 && compileInfo->btBytes > 0;
    }
    if (fromCompileInfo) {
        c.caps.maxCores = static_cast<uint64_t>(compileInfo->coreNum);
        c.caps.l1Bytes = static_cast<uint64_t>(compileInfo->l1Bytes);
        c.caps.l0aBytes = static_cast<uint64_t>(compileInfo->l0aBytes);
        c.caps.l0bBytes = static_cast<uint64_t>(compileInfo->l0bBytes);
        c.caps.l0cBytes = static_cast<uint64_t>(compileInfo->l0cBytes);
        c.caps.btBytes = static_cast<uint64_t>(compileInfo->btBytes);
    } else {
        /* 回退运行时查询（buffer-and-types §1.2 注册入口）：GetCoreNumAic / GetCoreMemSize；
         * BT 容量经 CoreMemType::BT 查询，平台查询不提供该容量（读回 0）时按同一 SDK 的
         * 目标能力定义绑定（bias-bt-path §1：arch35 实测参考 4096） */
        fe::PlatFormInfos* platformInfos = context->GetPlatformInfo();
        if (platformInfos == nullptr) {
            return false;
        }
        const platform_ascendc::PlatformAscendC platform(platformInfos);
        uint64_t l1 = 0;
        uint64_t l0a = 0;
        uint64_t l0b = 0;
        uint64_t l0c = 0;
        uint64_t bt = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::BT, bt);
        c.caps.maxCores = static_cast<uint64_t>(platform.GetCoreNumAic());
        c.caps.l1Bytes = l1;
        c.caps.l0aBytes = l0a;
        c.caps.l0bBytes = l0b;
        c.caps.l0cBytes = l0c;
        c.caps.btBytes = bt > 0 ? bt : static_cast<uint64_t>(CONV3DTRANSPOSE_BT_BYTES);
    }
    /* arch35 参考常量交叉核对（不一致以查询/CompileInfo 值为准并记录；
     * CONV3DTRANSPOSE_MAX_CORES=36 为参考上限，不参与决策——HostTiling.md §5） */
    if (c.caps.l1Bytes != CONV3DTRANSPOSE_L1_BYTES || c.caps.l0aBytes != CONV3DTRANSPOSE_L0A_BYTES ||
        c.caps.l0bBytes != CONV3DTRANSPOSE_L0B_BYTES || c.caps.l0cBytes != CONV3DTRANSPOSE_L0C_BYTES ||
        c.caps.maxCores != CONV3DTRANSPOSE_MAX_CORES || c.caps.btBytes != CONV3DTRANSPOSE_BT_BYTES) {
        OP_LOGD(context,
                "Conv3DTranspose tiling platform caps differ from arch35 reference values "
                "(coreNum %llu vs %u, L1 %llu vs %u, L0A %llu vs %u, L0B %llu vs %u, L0C %llu vs "
                "%u, BT %llu vs %u)",
                (unsigned long long)c.caps.maxCores, CONV3DTRANSPOSE_MAX_CORES, (unsigned long long)c.caps.l1Bytes,
                CONV3DTRANSPOSE_L1_BYTES, (unsigned long long)c.caps.l0aBytes, CONV3DTRANSPOSE_L0A_BYTES,
                (unsigned long long)c.caps.l0bBytes, CONV3DTRANSPOSE_L0B_BYTES, (unsigned long long)c.caps.l0cBytes,
                CONV3DTRANSPOSE_L0C_BYTES, (unsigned long long)c.caps.btBytes, CONV3DTRANSPOSE_BT_BYTES);
    }
    return c.caps.maxCores > 0 && c.caps.l1Bytes > 0 && c.caps.l0aBytes > 0 && c.caps.l0bBytes > 0 &&
           c.caps.l0cBytes > 0;
}

/* ---------------- 指令范围预检（HostTiling.md §4.2 maxM 收紧依据） ---------------- */

static bool BackpropPadWithinField(const SolveCtx& c)
{
    /* 四侧反向边界（含 q+rho 折入的 b⁺）已在上游几何推导处响亮拒绝（F-03 整改：
     * >CONV3DTRANSPOSE_MAX_PAD 直接 OP_LOGE 拒绝，不再收紧 maxM）；本函数保留
     * 负侧结构性校验供求解器不变式（负值走 kernel 负边界裁剪分支） */
    return c.backpropPadLeft <= static_cast<int32_t>(CONV3DTRANSPOSE_MAX_PAD) &&
           c.backpropPadUp <= static_cast<int32_t>(CONV3DTRANSPOSE_MAX_PAD) &&
           c.backpropPadDown <= static_cast<int32_t>(CONV3DTRANSPOSE_MAX_PAD) &&
           c.backpropPadRight <= static_cast<int32_t>(CONV3DTRANSPOSE_MAX_PAD);
}

/* ---------------- L0 基本块（HostTiling.md §4.2 AdjustBaseMNK） ---------------- */

/*
 * K 窗口配对候选搜索（HostTiling.md §4.3；A3/A4 收缩重试共用同一配对规则）：从 maxK
 * 按 C0 递减，仅检查大于 C0 的候选；wk>1 时候选为 wk·C0 倍数 / hk>1 时候选为 hk·C0
 * 倍数 / 否则候选与 windowSpan 互有整除关系。返回首个命中候选（无命中返回 0）；
 * maxK 经引用输出（已按 C0 对齐、kValue 截断）。
 */
static uint32_t ChooseKWindowPair(const SolveCtx& c, uint32_t& maxKOut)
{
    const uint64_t denom = c.al0Pbuffer * CONV3DTRANSPOSE_ELEM_BYTES * (c.baseM > c.baseN ? c.baseM : c.baseN);
    uint32_t maxK = static_cast<uint32_t>(c.caps.l0aBytes / denom);
    maxK = AlignDownU(maxK, CONV3DTRANSPOSE_C0);
    maxK = maxK < CONV3DTRANSPOSE_C0 ? CONV3DTRANSPOSE_C0 : maxK;
    maxK = maxK < c.kValue ? maxK : c.kValue;
    maxKOut = maxK;
    for (uint32_t cand = maxK; cand > CONV3DTRANSPOSE_C0; cand -= CONV3DTRANSPOSE_C0) {
        const bool condW = (c.wk > 1) && (cand % (c.wk * CONV3DTRANSPOSE_C0) == 0);
        const bool condH = (c.hk > 1) && (cand % (c.hk * CONV3DTRANSPOSE_C0) == 0);
        const bool condFull = (cand % c.windowSpan == 0) || (c.windowSpan % cand == 0);
        if (condW || condH || condFull) {
            return cand;
        }
    }
    return 0;
}

static void AdjustBaseMNK(SolveCtx& c, uint32_t incomingK)
{
    const uint64_t l0a = c.caps.l0aBytes;
    const uint64_t l0b = c.caps.l0bBytes;
    const uint64_t l0c = c.caps.l0cBytes;

    /* 回路由守卫（integration regression-routing，phase.integration 冻结，证据
     * run/integration/regression-routing.md）：nValue ≤ n0（tiny-N）域整体回退原始
     * 求解逻辑。依据：功能全集逐 case 计时发现 17 例有效性能回退，全部位于 tiny-N
     * 域（发布 baseN=16；新计划 baseM>512、baseK 32→16，实测 +7.5~+23µs，源 CSV
     * run/integration/gates/functional-performance-{before,after}.csv，机械证据
     * run/integration/tiling-diff/tiling-diff-626.json）；tiny-N 域内已验证收益
     * （10 族 1.10×）按规则让位于回退消除。legacy 分支体 = pristine 实现
     * （快照 c893a226…73e4a）AdjustBaseMNK 中段逐字保留，尾部三路 L0 容量循环
     * 两侧共享不变；守卫量为 tiling 阶段真实可计算量（SolveCtx.nValue = y 侧通道
     * 数），不按 case ID/CSV 序号/测试标志分派。非 tiny-N 域（01/02/06/07 族收益
     * 所在，baseN ≥ 32）行为与终选实现完全一致。 */
    if (c.nValue <= CONV3DTRANSPOSE_FRACTAL) {
        uint32_t maxM = (c.baseN <= CONV3DTRANSPOSE_FRACTAL) ? 512u : 1024u;
        if (!BackpropPadWithinField(c)) {
            maxM = 512u; /* 四侧反向边界任一超 255 时收紧 */
        }

        const uint32_t oldM = c.baseM;
        const uint32_t oldN = c.baseN;
        c.baseM = c.baseM < c.mValue ? c.baseM : c.mValue;
        const uint32_t capM =
            static_cast<uint32_t>(l0a / (CONV3DTRANSPOSE_C0 * CONV3DTRANSPOSE_ELEM_BYTES * c.al0Pbuffer));
        c.baseM = c.baseM < capM ? c.baseM : capM;
        c.baseN = c.baseN < c.nValue ? c.baseN : c.nValue;

        if (c.baseN < oldN) { /* N 缩小时扩大 M */
            const uint32_t mByL0C = static_cast<uint32_t>(l0c / (CONV3DTRANSPOSE_ACC_BYTES * c.cl0Pbuffer * c.baseN) /
                                                          CONV3DTRANSPOSE_FRACTAL) *
                                    CONV3DTRANSPOSE_FRACTAL;
            const uint32_t cand = mByL0C == 0 ? CONV3DTRANSPOSE_FRACTAL : mByL0C;
            const uint32_t m1 = maxM < cand ? maxM : cand;
            c.baseM = m1 < c.mValue ? m1 : c.mValue;
        }
        if (c.baseM < oldM) { /* M 缩小时扩大 N */
            const uint32_t nByL0C = static_cast<uint32_t>(l0c / (CONV3DTRANSPOSE_ACC_BYTES * c.cl0Pbuffer * c.baseM) /
                                                          CONV3DTRANSPOSE_FRACTAL) *
                                    CONV3DTRANSPOSE_FRACTAL;
            const uint32_t cand = nByL0C == 0 ? CONV3DTRANSPOSE_FRACTAL : nByL0C;
            const uint32_t n1 = 1024u < cand ? 1024u : cand;
            c.baseN = n1 < c.nValue ? n1 : c.nValue;
        }

        /* K 窗口配对（HostTiling.md §4.3）：从 maxK 按 C0 递减，仅检查大于 C0 的候选；
         * wk>1 时候选为 wk·C0 倍数 / hk>1 时候选为 hk·C0 倍数 / 否则候选与 windowSpan 互有
         * 整除关系；无命中保留进入本轮 baseK 与 maxK 中较小者 */
        const uint64_t denom = c.al0Pbuffer * CONV3DTRANSPOSE_ELEM_BYTES * (c.baseM > c.baseN ? c.baseM : c.baseN);
        uint32_t maxK = static_cast<uint32_t>(l0a / denom);
        maxK = AlignDownU(maxK, CONV3DTRANSPOSE_C0);
        maxK = maxK < CONV3DTRANSPOSE_C0 ? CONV3DTRANSPOSE_C0 : maxK;
        maxK = maxK < c.kValue ? maxK : c.kValue;
        uint32_t chosen = 0;
        for (uint32_t cand = maxK; cand > CONV3DTRANSPOSE_C0; cand -= CONV3DTRANSPOSE_C0) {
            const bool condW = (c.wk > 1) && (cand % (c.wk * CONV3DTRANSPOSE_C0) == 0);
            const bool condH = (c.hk > 1) && (cand % (c.hk * CONV3DTRANSPOSE_C0) == 0);
            const bool condFull = (cand % c.windowSpan == 0) || (c.windowSpan % cand == 0);
            if (condW || condH || condFull) {
                chosen = cand;
                break;
            }
        }
        c.baseK = (chosen != 0) ? chosen : (incomingK < maxK ? incomingK : maxK);
    } else {
        const uint32_t oldM = c.baseM;
        const uint32_t oldN = c.baseN;
        c.baseM = c.baseM < c.mValue ? c.baseM : c.mValue;
        const uint32_t capM =
            static_cast<uint32_t>(l0a / (CONV3DTRANSPOSE_C0 * CONV3DTRANSPOSE_ELEM_BYTES * c.al0Pbuffer));
        c.baseM = c.baseM < capM ? c.baseM : capM;
        c.baseN = c.baseN < c.nValue ? c.baseN : c.nValue;

        /* A3 maxM 后置计算 + F2（tiny-N maxM 容量界，round2 契约
         * host-tiling-iteration-block-rebalance-v2）：maxM 仍在 baseN 收缩/截断之后计算
         * （round1 A3 后置取档结构维持）；F2 将「baseN ≤ n0 取 512」档位提升为 1024——
         * 两档现均为 1024，直接取常量。设计偏差登记（HostTiling.md §4.2「baseN ≤ n0 时
         * 512」为启发式而非容量约束）：(1024,16) 组合 L0A=65536B / L0C(cl0=2)=131072B
         * 容量合法且方向为块数减半（round2 analysis 定律 2：块数流量定律）。08/10 族
         * baseN 128→16 时 baseM 恢复到 1024（= 原始实现实测 1472.7µs 的发布 tiling，
         * 实测锚点）。四侧反向边界任一超 255 的收紧维持不变并继续生效 */
        uint32_t maxM = 1024u;
        if (!BackpropPadWithinField(c)) {
            maxM = 512u; /* 四侧反向边界任一超 255 时收紧（维持） */
        }

        if (c.baseN < oldN) { /* N 缩小时扩大 M */
            const uint32_t mByL0C = static_cast<uint32_t>(l0c / (CONV3DTRANSPOSE_ACC_BYTES * c.cl0Pbuffer * c.baseN) /
                                                          CONV3DTRANSPOSE_FRACTAL) *
                                    CONV3DTRANSPOSE_FRACTAL;
            const uint32_t cand = mByL0C == 0 ? CONV3DTRANSPOSE_FRACTAL : mByL0C;
            const uint32_t m1 = maxM < cand ? maxM : cand;
            c.baseM = m1 < c.mValue ? m1 : c.mValue;
        }
        if (c.baseM < oldM) { /* M 缩小时扩大 N */
            const uint32_t nByL0C = static_cast<uint32_t>(l0c / (CONV3DTRANSPOSE_ACC_BYTES * c.cl0Pbuffer * c.baseM) /
                                                          CONV3DTRANSPOSE_FRACTAL) *
                                    CONV3DTRANSPOSE_FRACTAL;
            const uint32_t cand = nByL0C == 0 ? CONV3DTRANSPOSE_FRACTAL : nByL0C;
            const uint32_t n1 = 1024u < cand ? 1024u : cand;
            c.baseN = n1 < c.nValue ? n1 : c.nValue;
        }

        /* K 窗口配对（HostTiling.md §4.3）：无命中保留进入本轮 baseK 与 maxK 中较小者 */
        uint32_t maxK = 0;
        uint32_t chosen = ChooseKWindowPair(c, maxK);
        c.baseK = (chosen != 0) ? chosen : (incomingK < maxK ? incomingK : maxK);

        /* F3（非配对 baseK 提升，round2 契约 host-tiling-iteration-block-rebalance-v2）：
         * K 窗口配对回退之后追加：若 kIter = ceilDiv(kValue, baseK) > 64（迭代过细）且
         * maxK 仍有余量（maxK > baseK），则 baseK 提升到 maxK（baseM 不动，随后照常进入
         * F1 守卫与既有三路 L0 容量循环，均不改变）。设计偏差登记（HostTiling.md §4.3
         * 「无命中保留较小者」→ kIter>64 场景保留较大者）：非配对 baseK 合法性有原始
         * 实现先例——01/02 长期运行 baseK=32 相对 windowSpan 176/112 均非配对（26/26
         * strict_pass + 626 例全集绿），EnsureA1/EnsureB1 段通道整数除 + AlignUp16L 覆盖
         * 机制与 baseK 配对性无关；L0A 未满的场合 kL0 提升伴随迭代数真降（01: 32→80，
         * kIter 528→212；02: 32→64，kIter 168→84）。触发集核验：07 排除（kIter 42≤64）、
         * 08/10/09 无余量（maxK==baseK）、03–06 已配对或 kIter≤8 */
        if (CeilDivU(c.kValue, c.baseK) > 64u && maxK > c.baseK) {
            c.baseK = maxK;
        }

        /* F1（严格迭代+块数双重守卫，round2 契约 host-tiling-iteration-block-rebalance-v2，
         * 收口 round1 A3/A4 重平衡）：kIter = ceilDiv(kValue, baseK) > 64（迭代过细）或
         * wk>1 且 maxK < wk·C0 无配对候选（宽核零配对）时，仍允许 baseM 收缩一步（减半、
         * n0 对齐、下限 64）以 L0A 容量重试配对；接受条件由 round1 的「issue 数不增」
         * 收紧为 (a) 每任务 Mmad issue 数（mBlocks × kIter = ceilDiv(mTotal, baseM) ×
         * ceilDiv(kValue, baseK)）严格下降，且 (b) M 块数不增（ceilDiv(mTotal, shrunkM)
         * ≤ ceilDiv(mTotal, savedM)）；不满足则 baseM 与 baseK 完整回退到尝试前值。
         * 依据（round2 analysis §5 定律 1/2 + §3.2 受控实验）：L0A 满利用率几何下 M-K
         * 重平衡迭代数不变、只增块数（08/10 实测 0.37–0.44µs/块 → (a) 以相等 issue 被
         * 拒绝）；01/02 的 kIter 收缩伴随块数 3→7 / 6→12 放大（→ (b) 拒绝，保留大
         * baseM 交给 F3 提升 kIter）；09 族重试无配对候选，发布 tiling 逐字段不变 */
        {
            const uint64_t mTotal = static_cast<uint64_t>(c.hi) * c.wi;
            const uint32_t kIterCur = CeilDivU(c.kValue, c.baseK);
            const bool balanceNeeded = kIterCur > 64u;
            const bool widenNeeded = (c.wk > 1u) && (chosen == 0u) && (maxK < c.wk * CONV3DTRANSPOSE_C0);
            if ((balanceNeeded || widenNeeded) && c.baseK > 0u) {
                const uint32_t savedM = c.baseM;
                const uint32_t savedK = c.baseK; /* F1：baseM/baseK 完整回滚锚点 */
                uint32_t shrunkM = AlignDownU(savedM / 2u, CONV3DTRANSPOSE_FRACTAL);
                shrunkM = shrunkM < 64u ? 64u : shrunkM;
                if (shrunkM < savedM) {
                    c.baseM = shrunkM;
                    uint32_t maxK2 = 0;
                    const uint32_t chosen2 = ChooseKWindowPair(c, maxK2);
                    bool accept = (chosen2 != 0u) && (chosen2 > savedK);
                    if (accept) {
                        const uint64_t curMBlocks = static_cast<uint64_t>(CeilDivU(mTotal, savedM));
                        const uint64_t newMBlocks = static_cast<uint64_t>(CeilDivU(mTotal, shrunkM));
                        const uint64_t curIssues = curMBlocks * CeilDivU(c.kValue, savedK);
                        const uint64_t newIssues = newMBlocks * CeilDivU(c.kValue, chosen2);
                        /* F1(a) issue 严格下降 AND F1(b) M 块数不增 */
                        accept = (newIssues < curIssues) && (newMBlocks <= curMBlocks);
                    }
                    if (accept) {
                        c.baseK = chosen2;
                    } else {
                        c.baseM = savedM; /* F1：baseM 与 baseK 均完整回退到尝试前值 */
                        c.baseK = savedK;
                    }
                }
            }
        }
    } /* 回路由守卫 else 分支（非 tiny-N：终选实现原逻辑）结束 */

    /* 选定 K 后更新部分和份数并核对三路 L0 容量（不足时按 M→N→K 收缩一个粒度） */
    c.cl0Pbuffer = (static_cast<uint64_t>(c.baseM) * c.baseN * CONV3DTRANSPOSE_ACC_BYTES * 2 <= l0c) ? 2u : 1u;
    for (int guard = 0; guard < 400; ++guard) {
        const uint64_t a = static_cast<uint64_t>(c.baseM) * c.baseK * CONV3DTRANSPOSE_ELEM_BYTES * c.al0Pbuffer;
        const uint64_t b = static_cast<uint64_t>(c.baseN) * c.baseK * CONV3DTRANSPOSE_ELEM_BYTES * c.bl0Pbuffer;
        const uint64_t cc = static_cast<uint64_t>(c.baseM) * c.baseN * CONV3DTRANSPOSE_ACC_BYTES * c.cl0Pbuffer;
        if (a <= l0a && b <= l0b && cc <= l0c) {
            break;
        }
        if (c.baseM > CONV3DTRANSPOSE_FRACTAL && (a > l0a || cc > l0c)) {
            c.baseM -= CONV3DTRANSPOSE_FRACTAL;
        } else if (c.baseN > CONV3DTRANSPOSE_FRACTAL && (b > l0b || cc > l0c)) {
            c.baseN -= CONV3DTRANSPOSE_FRACTAL;
        } else if (c.baseK > CONV3DTRANSPOSE_C0) {
            c.baseK -= CONV3DTRANSPOSE_C0;
        } else {
            break;
        }
        c.cl0Pbuffer = (static_cast<uint64_t>(c.baseM) * c.baseN * CONV3DTRANSPOSE_ACC_BYTES * 2 <= l0c) ? 2u : 1u;
    }
}

/* ---------------- L1 驻留容量（HostTiling.md §4.4） ---------------- */

static uint32_t CoveredRows(uint32_t spaceLength, uint32_t wi)
{
    if (spaceLength % wi == 0 || wi % spaceLength == 0) {
        return CeilDivU(spaceLength, wi);
    }
    return spaceLength > wi ? (spaceLength / wi + 2) : 2;
}

static uint32_t ResidentRows(const SolveCtx& c, uint32_t spaceLength)
{
    const uint32_t rows = CoveredRows(spaceLength, c.wi);
    const uint32_t span = rows + c.khDilation - 1;
    return span < c.hoExpand ? span : c.hoExpand;
}

static uint32_t CopyLines(uint32_t residentK, uint32_t windowSpan)
{
    if (residentK % windowSpan == 0 || windowSpan % residentK == 0) {
        return CeilDivU(residentK, windowSpan);
    }
    return residentK > windowSpan ? (residentK / windowSpan + 2) : 2;
}

/* 实际槽字节数（Kernel InitBuffer 分配量；宽度预算取 Wo*s_w ≥ woExpand） */
static void CalcSlotBytes(const SolveCtx& c, uint32_t& aSlot, uint32_t& bSlot)
{
    const uint32_t rows = ResidentRows(c, c.baseM);
    const uint32_t widthBudget = c.wo * c.strideW;
    uint32_t aChannels = CeilDivU(static_cast<uint64_t>(c.stepKa) * c.baseK, c.hkWk);
    const uint32_t aChCap = AlignUpU(c.coutG, CONV3DTRANSPOSE_C0);
    aChannels = aChannels < aChCap ? aChannels : aChCap;
    if (aChannels == 0) {
        aChannels = 1;
    }
    aSlot = AlignUpU(static_cast<uint64_t>(rows) * widthBudget * aChannels * CONV3DTRANSPOSE_ELEM_BYTES, 32u);
    bSlot = AlignUpU(static_cast<uint64_t>(c.baseN) * c.stepKb * c.baseK * CONV3DTRANSPOSE_ELEM_BYTES, 32u);
}

/* bias 候选预算（候选检查统一用 baseN；bias-bt-path / HostTiling.md §4.4） */
static uint64_t BiasBudget(const SolveCtx& c)
{
    if (c.hasBias == 0) {
        return 0;
    }
    return AlignUpU(static_cast<uint64_t>(c.baseN) * c.biasBytes, 64u);
}

static bool IsL1ParamsValid(SolveCtx& c)
{
    if (c.stepKa == 0 || c.stepKb == 0 || c.baseK == 0) {
        return false;
    }
    const uint64_t lenA = static_cast<uint64_t>(c.stepKa) * c.baseK;
    const uint64_t lenB = static_cast<uint64_t>(c.stepKb) * c.baseK;
    /* 每路驻留长度覆盖完整 kValue 或为 windowSpan 的整数倍 */
    if (lenA < c.kValue && lenA % c.windowSpan != 0) {
        return false;
    }
    if (lenB < c.kValue && lenB % c.windowSpan != 0) {
        return false;
    }
    const uint32_t rows = ResidentRows(c, c.baseM);
    const uint32_t widthBudget = c.wo * c.strideW;
    uint32_t aCh = CeilDivU(lenA, c.hkWk);
    if (aCh == 0) {
        aCh = 1;
    }
    const uint64_t dyBudget =
        static_cast<uint64_t>(rows) * widthBudget * aCh * CONV3DTRANSPOSE_ELEM_BYTES * c.al1Pbuffer;
    const uint32_t lines = CopyLines(static_cast<uint32_t>(lenB < 0xFFFFFFFFull ? lenB : 0xFFFFFFFFull), c.windowSpan);
    const uint64_t wBudget =
        static_cast<uint64_t>(c.baseN) * lines * c.windowSpan * CONV3DTRANSPOSE_ELEM_BYTES * c.bl1Pbuffer;
    const uint64_t biasBudget = BiasBudget(c);
    if (dyBudget + wBudget + biasBudget >= c.caps.l1Bytes) {
        return false;
    }
    uint32_t aSlot = 0;
    uint32_t bSlot = 0;
    CalcSlotBytes(c, aSlot, bSlot);
    const uint64_t total =
        static_cast<uint64_t>(aSlot) * c.al1Pbuffer + static_cast<uint64_t>(bSlot) * c.bl1Pbuffer + biasBudget;
    return total <= c.caps.l1Bytes;
}

/* ---------------- CalStepK（HostTiling.md §4.4：等分/阶梯候选） ---------------- */

static bool LadderMatchStep(SolveCtx& c)
{
    const uint32_t coutBlocks = CeilDivU(c.coutG, CONV3DTRANSPOSE_C0);
    const uint32_t initialK1 = CeilDivU(coutBlocks, c.al1Pbuffer) * c.windowSpan;
    const uint32_t initialK2 =
        CeilDivU(c.caps.l1Bytes / CONV3DTRANSPOSE_ELEM_BYTES, static_cast<uint64_t>(c.baseN) * c.al1Pbuffer);
    uint32_t initialK = initialK1 < initialK2 ? initialK1 : initialK2;
    if (initialK == 0) {
        initialK = c.windowSpan;
    }
    uint32_t step = CeilDivU(initialK, c.baseK);
    if (step == 0) {
        step = 1;
    }
    for (; step >= 1; --step) {
        c.stepKa = step;
        c.stepKb = step;
        if (IsL1ParamsValid(c)) {
            return true;
        }
    }
    c.stepKa = 0;
    c.stepKb = 0;
    return false;
}

static bool EqualL1MatchStep(SolveCtx& c)
{
    const uint32_t coutBlocks = CeilDivU(c.coutG, CONV3DTRANSPOSE_C0);
    const uint32_t rows = ResidentRows(c, c.baseM);
    const uint64_t dyBlockBytes =
        static_cast<uint64_t>(CONV3DTRANSPOSE_ELEM_BYTES) * rows * (c.wo * c.strideW) * CONV3DTRANSPOSE_C0;
    const uint64_t wBlockBytes = static_cast<uint64_t>(c.windowSpan) * c.baseN * CONV3DTRANSPOSE_ELEM_BYTES;

    uint32_t dyBlocks = static_cast<uint32_t>((c.caps.l1Bytes / 2 / c.al1Pbuffer) / dyBlockBytes);
    dyBlocks = dyBlocks == 0 ? 1 : dyBlocks;
    dyBlocks = dyBlocks < coutBlocks ? dyBlocks : coutBlocks;
    if (static_cast<uint64_t>(dyBlocks) * c.al1Pbuffer >= coutBlocks) {
        dyBlocks = CeilDivU(coutBlocks, c.al1Pbuffer);
    }
    uint32_t wBlocks = static_cast<uint32_t>((c.caps.l1Bytes / 2 / c.bl1Pbuffer) / wBlockBytes);
    wBlocks = wBlocks == 0 ? 1 : wBlocks;
    wBlocks = wBlocks < coutBlocks ? wBlocks : coutBlocks;
    if (static_cast<uint64_t>(wBlocks) * c.bl1Pbuffer >= coutBlocks) {
        wBlocks = CeilDivU(coutBlocks, c.bl1Pbuffer);
    }

    /* 保留较大者，将较小者减为不超过原值且能整除较大者的最大整数（如 (10,6)→(10,5)） */
    uint32_t hi = dyBlocks >= wBlocks ? dyBlocks : wBlocks;
    uint32_t lo = dyBlocks >= wBlocks ? wBlocks : dyBlocks;
    if (hi == 0) {
        hi = 1;
    }
    while (lo > 1 && hi % lo != 0) {
        --lo;
    }
    if (lo == 0) {
        lo = 1;
    }
    const uint32_t dyBlocks2 = dyBlocks >= wBlocks ? hi : lo;
    const uint32_t wBlocks2 = dyBlocks >= wBlocks ? lo : hi;

    uint64_t ka = CeilDivU(static_cast<uint64_t>(dyBlocks2) * c.windowSpan, c.baseK);
    const uint64_t kaCap = 65535u / c.baseK;
    if (ka < 1) {
        ka = 1;
    }
    if (ka > kaCap) {
        ka = kaCap > 0 ? kaCap : 1;
    }
    uint32_t kb = CeilDivU(static_cast<uint64_t>(wBlocks2) * c.windowSpan, c.baseK);
    if (kb < 1) {
        kb = 1;
    }
    /* 较大 step 向下对齐到较小 step 的整数倍 */
    if (ka > kb) {
        ka = (ka / kb) * kb;
        if (ka < 1) {
            ka = kb;
        }
    } else if (kb > ka) {
        kb = (kb / ka) * ka;
        if (kb < 1) {
            kb = static_cast<uint32_t>(ka);
        }
    }
    c.stepKa = static_cast<uint32_t>(ka);
    c.stepKb = kb;
    if (IsL1ParamsValid(c)) {
        return true;
    }
    c.stepKa = 0;
    c.stepKb = 0;
    return false;
}

/*
 * 首次求解选择规则（HostTiling.md §4.4）：等分候选合法且其 step 之和严格大于阶梯候选时
 * 选等分，否则选阶梯；两候选均不合法返回 false。候选枚举与接受规则严格确定（无随机/
 * 时间依赖，HostTiling.md §10 切分决策确定性）。
 */
static bool CalStepK(SolveCtx& c)
{
    const uint32_t savedKa = c.stepKa;
    const uint32_t savedKb = c.stepKb;

    const bool ladderValid = LadderMatchStep(c);
    const uint32_t ladderKa = c.stepKa;
    const uint32_t ladderKb = c.stepKb;

    c.stepKa = savedKa;
    c.stepKb = savedKb;
    const bool equalValid = EqualL1MatchStep(c);
    const uint32_t equalKa = c.stepKa;
    const uint32_t equalKb = c.stepKb;

    const bool equalWins = equalValid && (!ladderValid || static_cast<uint64_t>(equalKa) + equalKb >
                                                              static_cast<uint64_t>(ladderKa) + ladderKb);
    if (equalWins) {
        c.stepKa = equalKa;
        c.stepKb = equalKb;
        return true;
    }
    if (ladderValid) {
        c.stepKa = ladderKa;
        c.stepKb = ladderKb;
        return true;
    }
    c.stepKa = 0;
    c.stepKb = 0;
    return false;
}

/*
 * 容量回调重选（HostTiling.md §4.4「容量回调重选」）：先试阶梯候选，不合法再试等分候选，
 * 返回其合法性——与首次求解选择规则相区分，回调不做「step 和更大者胜」的偏好比较。
 */
static bool CalStepKForCallback(SolveCtx& c)
{
    if (LadderMatchStep(c)) {
        return true;
    }
    return EqualL1MatchStep(c);
}

/* ---------------- LegalProtection（HostTiling.md §4.4 容量回调，不切换模板） ---------------- */

static bool LegalProtection(SolveCtx& c)
{
    const uint32_t savedBaseK = c.baseK;
    const uint32_t savedBaseM = c.baseM;
    const uint32_t savedBaseN = c.baseN;
    const uint32_t savedAl1 = c.al1Pbuffer;
    const uint32_t savedBl1 = c.bl1Pbuffer;

    /* 1. 减小 K：从当前 baseK-C0 起（baseK==C0 时直接取 C0）按 C0 递减至 C0，逐个重选驻留 */
    for (uint32_t k = (c.baseK > CONV3DTRANSPOSE_C0) ? c.baseK - CONV3DTRANSPOSE_C0 : CONV3DTRANSPOSE_C0;;
         k -= CONV3DTRANSPOSE_C0) {
        c.baseK = k;
        if (CalStepKForCallback(c)) {
            return true;
        }
        if (k == CONV3DTRANSPOSE_C0) {
            break;
        }
    }
    c.baseK = CONV3DTRANSPOSE_C0;
    /* 2. 减小 M/N：交替收缩（各自至少保留一个分形；M 先于 N 当 baseM>max(Wi,m0) 且
     * baseM>baseN），逐候选重选驻留 */
    for (int guard = 0; guard < 256; ++guard) {
        const bool canShrinkM =
            c.baseM > CONV3DTRANSPOSE_FRACTAL && (c.baseM > c.wi || c.wi <= CONV3DTRANSPOSE_FRACTAL);
        const bool canShrinkN = c.baseN > CONV3DTRANSPOSE_FRACTAL;
        if (canShrinkM && (c.baseM > c.baseN || !canShrinkN)) {
            c.baseM -= CONV3DTRANSPOSE_FRACTAL;
        } else if (canShrinkN) {
            c.baseN -= CONV3DTRANSPOSE_FRACTAL;
        } else {
            break;
        }
        if (CalStepKForCallback(c)) {
            return true;
        }
    }
    /* 3. 恢复较大 K：重算窗口配对 maxK，等于当前 K 则保留，否则从 maxK-C0 重新向下搜索 */
    c.baseM = savedBaseM;
    c.baseN = savedBaseN;
    AdjustBaseMNK(c, savedBaseK);
    if (CalStepKForCallback(c)) {
        return true;
    }
    /* 4. 调整份数：(2,2) → (2,1) → (1,2) → (1,1)，每次重新执行容量检查与收缩 */
    c.baseM = savedBaseM;
    c.baseN = savedBaseN;
    c.baseK = savedBaseK;
    struct PBufPair {
        uint32_t a;
        uint32_t b;
    };
    const PBufPair combos[] = {{2, 1}, {1, 2}, {1, 1}};
    for (const auto& pb : combos) {
        c.al1Pbuffer = pb.a;
        c.bl1Pbuffer = pb.b;
        if (CalStepKForCallback(c)) {
            return true;
        }
    }
    c.al1Pbuffer = savedAl1;
    c.bl1Pbuffer = savedBl1;
    return false;
}

/* ---------------- SetSingleCoreInfo（HostTiling.md §5 公共任务恢复） ---------------- */

static void SetSingleCoreInfo(SolveCtx& c)
{
    const uint32_t maxCores = static_cast<uint32_t>(c.caps.maxCores);
    uint32_t spaceParts = CeilDivU(static_cast<uint64_t>(c.hi) * c.wi, c.baseM);
    uint32_t channelParts = CeilDivU(c.nValue, c.baseN);
    if (spaceParts == 0) {
        spaceParts = 1;
    }
    if (channelParts == 0) {
        channelParts = 1;
    }
    const uint64_t initialTasks = static_cast<uint64_t>(c.batch) * c.di * c.group * spaceParts * channelParts;

    const bool kernelVolOne = c.dkHkWk == 1;
    const uint32_t nPreferred = kernelVolOne ? 256u : 128u;
    const uint32_t mPreferred = kernelVolOne ? 256u : 512u;

    /* 通道划分：任务数不超核数且 channelParts 与核数互有整除关系，或 baseN 超阈值时均分通道 */
    const bool dividesEither = (maxCores % channelParts == 0) || (channelParts % maxCores == 0);
    if ((initialTasks <= maxCores && dividesEither) || c.baseN > nPreferred) {
        uint32_t newBaseN = AlignUpU(c.nValue / channelParts, CONV3DTRANSPOSE_FRACTAL);
        if (newBaseN == 0) {
            newBaseN = CONV3DTRANSPOSE_FRACTAL;
        }
        if (newBaseN > c.nValue) {
            newBaseN = c.nValue;
        }
        c.baseN = newBaseN;
        channelParts = CeilDivU(c.nValue, c.baseN);
    }
    c.singleCoreCin = c.baseN;
    c.singleCoreCout = c.coutG;
    c.singleCoreDin = 1;

    /* 空间划分：initialTasks ≤ maxCores 取 alignUp((hi·wi)//spaceParts, wi)；任务数更多且
     * baseM ≥ mPreferred 取整行候选 wholeRowM（仅 wholeRowM·spaceParts ≥ mValue 时采用） */
    c.singleCoreM = c.baseM;
    const uint64_t mTotal = static_cast<uint64_t>(c.hi) * c.wi;
    uint32_t candM = 0;
    if (initialTasks <= maxCores) {
        uint32_t whole = AlignUpU(mTotal / spaceParts, c.wi);
        if (whole == 0) {
            whole = c.baseM;
        }
        candM = whole;
    } else if (c.baseM >= mPreferred) {
        const uint32_t wholeRowM = (c.baseM / c.wi > 0 ? c.baseM / c.wi : 1) * c.wi;
        if (static_cast<uint64_t>(wholeRowM) * spaceParts >= c.mValue) {
            candM = wholeRowM;
        }
    }
    /* A1（任务数下限，round1 契约 host-tiling-parallelism-granularity）：hi==1（整行=
     * 整空间域，整行对齐即把任务塌缩到 batch/通道维）或整行候选把 M 向任务数压到
     * baseM 粒度因子分解以下（ceilDiv(mTotal, candM) < ceilDiv(mTotal, baseM)）时，不
     * 采纳整行候选——任务按 baseM 粒度切分（kernel 已支持行中起点：Process mStart_%wi
     * 寻址、行带交集数学对任意起点成立、Fixpipe 任意 mStart 写回）。tasks ≥ maxCores
     * 且 hi != 1 时行为不变（LISA 族等非目标用例 tiling 逐字段保持） */
    bool a1Floor = false;
    if (!(c.nValue <= CONV3DTRANSPOSE_FRACTAL) && candM != 0) {
        /* 回路由守卫：tiny-N 域不走 A1（= pristine 无此逻辑，见 AdjustBaseMNK 守卫注释） */
        const uint64_t mCountBase = CeilDivU(mTotal, c.baseM);
        if (c.hi == 1u || (candM > c.baseM && CeilDivU(mTotal, candM) < mCountBase)) {
            a1Floor = true;
        }
    }
    if (!a1Floor && candM != 0) {
        c.singleCoreM = candM;
    }
    if (c.singleCoreM == 0) {
        c.singleCoreM = c.baseM;
    }
    if (c.singleCoreM < c.baseM) {
        c.baseM = AlignUpU(c.singleCoreM, CONV3DTRANSPOSE_FRACTAL);
        if (c.baseM == 0) {
            c.baseM = CONV3DTRANSPOSE_FRACTAL;
        }
    }
}

/* ---------------- loadB2 判定（HostTiling.md §7 / BranchRoute.md §1 判定链） ---------------- */

/*
 * GetLoadB2ConditionByFormatAndKernel（范式 tilingkey-templates §2；判定在最终 baseN 上、
 * 判定先于路由、顺序互斥）：核体积 1 首判优先于格式与 64B 条件；filter 为 DHWCN 时
 * 两条仅逆序谓词恒不成立，恒路由 TPL_TRANSPOSE_AND_REVERSE。cin 为总输入通道（不用
 * cinG）；核体积乘积以 uint64_t 表示；64B 阈用正整数等价除法形式（fp16/bf16 恒 2B）。
 */
static void SetTilingCondition(SolveCtx& c)
{
    const uint64_t kernelHW = static_cast<uint64_t>(c.hk) * c.wk;
    const uint64_t kernelDHW = static_cast<uint64_t>(c.dk) * kernelHW;
    if (kernelDHW == 1) {
        c.loadB2 = TPL_TRANSPOSE_ONLY;
    } else if (c.filterFormat == ge::FORMAT_NDHWC && c.baseN >= kernelHW) {
        c.loadB2 = TPL_REVERSE_ONLY;
    } else if (c.filterFormat == ge::FORMAT_NCDHW && kernelDHW <= 64 / CONV3DTRANSPOSE_ELEM_BYTES / c.cin) {
        c.loadB2 = TPL_REVERSE_ONLY;
    } else {
        c.loadB2 = TPL_TRANSPOSE_AND_REVERSE;
    }
}

/* ---------------- bias 实际分配复核（HostTiling.md §4.4 UpdateIsBiasFullLoad） ---------------- */

/*
 * 三模板求解完成、最终 baseN/singleCoreCin 形成后执行：全量条件
 * fullAlloc = alignUp(singleCoreCin·biasBytes, 64) ≤ BTBytes → isBiasFullLoad=1，否则降级按
 * 当前 N 子块 alloc = alignUp(baseN·biasBytes, 64)；alloc > BTBytes 时 bias 通路能力不足
 * （不发布未通过的字段）；随后 L1 实际范围复核 aSlotBytes·al1Pbuffer + bSlotBytes·bl1Pbuffer
 * + alloc ≤ L1Bytes。候选预算用 baseN、全量实际用 singleCoreCin，两者不同的任务必须复核
 * 后者；无 bias 恒 0 恒过。
 *
 * BT 两口径收口（DESIGN-BRANCH-16777218.md §4 收口核对义务，Kernel.md §9.9 校验项 9，
 * branch-2 run-20pct 节点闭合——收口路径 A「检查式改按 alignUp(biasUse·4,64) ≤ BTBytes」）：
 * L1 槽按 bias 元素宽口径 biasAllocBytes = alignUp(biasUse·biasBytes,64)（BIAST=half 为 2B），
 * BT 槽按折入通路 4B 部分和口径恒定消费（kernel 侧 InitBuffer(biasBTQue_,
 * alignUp(biasUse·4,64)) 与 CopyBiasL1ToBT 拓宽拷贝同口径，CONV3DTRANSPOSE_ACC_BYTES=4）；
 * BIAST=half 时 BT 槽为 L1 槽单值公式的 2 倍——BT 容量检查（全量判定与分块拒绝）一律按
 * alignUp(biasUse·4,64) ≤ BTBytes 复核，biasUse ≤ 1024 为 BT 槽不超 4096 的必要条件；
 * BIAST=float 时两口径同值，行为不变。L1 槽发布口径 biasAllocBytes 与 L1 总预算复核
 * 维持元素宽口径不变（L1 槽承载 BIAST 原生数据）。
 */
static bool UpdateIsBiasFullLoad(SolveCtx& c)
{
    if (c.hasBias == 0) {
        c.isBiasFullLoad = 0;
        c.biasAllocBytes = 0;
        return true;
    }
    /* BT 容量检查：4B 部分和口径（收口路径 A；原 2B 单值公式在 BIAST=half 组合不覆盖 BT 实际需求） */
    const uint64_t fullBtSlot = AlignUpU(static_cast<uint64_t>(c.singleCoreCin) * CONV3DTRANSPOSE_ACC_BYTES, 64u);
    c.isBiasFullLoad = (fullBtSlot <= c.caps.btBytes) ? 1u : 0u;
    /* L1 槽发布口径：bias 元素宽（biasBytes = fp16 2 / fp32 4），与 kernel InitBuffer(biasL1Que_) 一致 */
    const uint64_t fullAlloc = AlignUpU(static_cast<uint64_t>(c.singleCoreCin) * c.biasBytes, 64u);
    const uint64_t alloc =
        c.isBiasFullLoad != 0 ? fullAlloc : AlignUpU(static_cast<uint64_t>(c.baseN) * c.biasBytes, 64u);
    c.biasAllocBytes = static_cast<uint32_t>(alloc);
    const uint64_t biasUse = c.isBiasFullLoad != 0 ? c.singleCoreCin : c.baseN;
    const uint64_t btSlot = AlignUpU(biasUse * CONV3DTRANSPOSE_ACC_BYTES, 64u);
    if (btSlot > c.caps.btBytes) {
        return false; /* 分块 BT 通路能力不足（alignUp(biasUse·4,64) > BTBytes，4B 部分和口径） */
    }
    uint32_t aSlot = 0;
    uint32_t bSlot = 0;
    CalcSlotBytes(c, aSlot, bSlot);
    const uint64_t total =
        static_cast<uint64_t>(aSlot) * c.al1Pbuffer + static_cast<uint64_t>(bSlot) * c.bl1Pbuffer + alloc;
    if (total > c.caps.l1Bytes) {
        return false;
    }
    return true;
}

/* ---------------- 模板准入（HostTiling.md §9 优先序：small_shape → full_load → standard） ---------------- */

static bool SmallShapeIsCapable(const SolveCtx& c)
{
    /* estimatedTasks = ceilDiv(mValue*nValue*accBytes, L0CBytes) * N * Di * G；K 上限为入口条件 */
    if (c.kValue > 32768u) {
        return false;
    }
    const uint64_t perL0c =
        CeilDivU(static_cast<uint64_t>(c.mValue) * c.nValue * CONV3DTRANSPOSE_ACC_BYTES, c.caps.l0cBytes);
    const uint64_t estimatedTasks = perL0c * c.batch * c.di * c.group;
    if (estimatedTasks >= static_cast<uint64_t>(c.caps.maxCores) * 5 / 2) {
        return false;
    }
    return estimatedTasks % c.caps.maxCores != 0;
}

static bool L1ResidentIsCapable(const SolveCtx& c)
{
    /* 几何条件：Dk=1、mValue>nValue、Wi<=512 */
    if (c.dk != 1 || c.mValue <= c.nValue || c.wi > 512) {
        return false;
    }
    /* 收益条件：mValue*nValue < (mValue+nValue)*512 或 kValue<=128 */
    const bool benefit =
        (static_cast<uint64_t>(c.mValue) * c.nValue < (static_cast<uint64_t>(c.mValue) + c.nValue) * 512) ||
        c.kValue <= 128u;
    if (!benefit) {
        return false;
    }
    /* 容量初筛：kValue*min(nPreferred,nValue)*weightBytes*2 <= L1Bytes（为优选通道块预留） */
    const uint32_t nPreferred = (c.dkHkWk == 1) ? 256u : 128u;
    const uint32_t nMin = nPreferred < c.nValue ? nPreferred : c.nValue;
    return static_cast<uint64_t>(c.kValue) * nMin * CONV3DTRANSPOSE_ELEM_BYTES * 2 <= c.caps.l1Bytes;
}

/* ---------------- 三模板求解（HostTiling.md §9 执行顺序） ---------------- */

static void InitBaseMNKCommon(SolveCtx& c)
{
    /* 核体积 1 起点 (256,256,64)，其余 (512,128,32)；两路输入双 Buffer，部分和单 Buffer；
     * full_load 回退 standard 时 enableFullLoad 在此回改 0（发布纪律：flag 与求解结果一致） */
    if (c.dkHkWk == 1) {
        c.baseM = 256;
        c.baseN = 256;
        c.baseK = 64;
    } else {
        c.baseM = 512;
        c.baseN = 128;
        c.baseK = 32;
    }
    c.al0Pbuffer = 2;
    c.bl0Pbuffer = 2;
    c.cl0Pbuffer = 1;
    c.al1Pbuffer = 2;
    c.bl1Pbuffer = 2;
    c.enableFullLoad = 0;
}

static bool SolveStandard(SolveCtx& c)
{
    /* standard §4：InitBaseMNK → AdjustBaseMNK → SetSingleCoreInfo → CalStepK → LegalProtection */
    InitBaseMNKCommon(c);
    const uint32_t incomingK = c.baseK;
    AdjustBaseMNK(c, incomingK);
    SetSingleCoreInfo(c);
    if (!CalStepK(c) || !IsL1ParamsValid(c)) {
        if (!LegalProtection(c)) {
            return false;
        }
        SetSingleCoreInfo(c);
        if (!IsL1ParamsValid(c)) {
            return false;
        }
    }
    SetTilingCondition(c);
    return true;
}

static bool SolveSmallShape(SolveCtx& c)
{
    /* small-shape §4：InitBaseMNK → AdjustBaseMNK → 均衡搜索 AdjustSingleCoreAndL0Info →
     * CalStepK → LegalProtection（容量回调不切换模板） */
    InitBaseMNKCommon(c);
    uint32_t incomingK = c.baseK;
    AdjustBaseMNK(c, incomingK);

    /* 均衡搜索（small-shape §2）：分段数范围含两端，(rounds, tasks, loadBytes) 评价量 */
    /* F2（tiny-N maxM 容量界，round2 契约 host-tiling-iteration-block-rebalance-v2）+
     * 回路由守卫（integration regression-routing）：tiny-N（nValue ≤ n0）域恢复
     * pristine 搜索上界 512（该域 F2 提升在功能全集 17 例上产生有效回退，见
     * AdjustBaseMNK 守卫注释）；非 tiny-N 域维持 1024（终选实现原逻辑——搜索上下文
     * baseN ≤ n0 ⟺ nValue ≤ n0：baseN = min(起点, nValue)，nValue > 16 时 baseN ≥ 32） */
    const uint32_t maxM = (c.nValue <= CONV3DTRANSPOSE_FRACTAL) ? 512u : 1024u;
    const uint32_t maxN = (c.mValue <= CONV3DTRANSPOSE_FRACTAL) ? 512u : 1024u;
    const uint64_t mTotal = static_cast<uint64_t>(c.hi) * c.wi;
    const uint32_t spacePartsMin = CeilDivU(mTotal, maxM) < 1 ? 1 : CeilDivU(mTotal, maxM);
    const uint32_t spacePartsMax = CeilDivU(mTotal, 64u) < 1 ? 1 : CeilDivU(mTotal, 64u);
    const uint32_t channelPartsMin = CeilDivU(c.nValue, maxN) < 1 ? 1 : CeilDivU(c.nValue, maxN);
    const uint32_t channelPartsMax = CeilDivU(c.nValue, 64u) < 1 ? 1 : CeilDivU(c.nValue, 64u);

    uint32_t bestM = c.baseM;
    uint32_t bestN = c.baseN;
    uint32_t bestTaskM = c.baseM;
    uint64_t bestTasks = 0;
    uint64_t bestRounds = 0xFFFFFFFFFFFFFFFFull;
    uint64_t bestLoad = 0xFFFFFFFFFFFFFFFFull;
    bool hasBest = false;
    for (uint32_t sp = spacePartsMin;; ++sp) {
        const uint32_t calcM0 = AlignUpU(CeilDivU(mTotal, sp), CONV3DTRANSPOSE_FRACTAL);
        if (calcM0 == 0) {
            if (sp >= spacePartsMax) {
                break;
            }
            continue;
        }
        for (uint32_t cp = channelPartsMin;; ++cp) {
            const uint32_t calcN0 = AlignUpU(CeilDivU(c.nValue, cp), CONV3DTRANSPOSE_FRACTAL);
            if (calcN0 == 0) {
                if (cp >= channelPartsMax) {
                    break;
                }
                continue;
            }
            uint32_t taskM = calcM0;
            const uint32_t wholeRowM = (calcM0 / c.wi > 0 ? calcM0 / c.wi : 1) * c.wi;
            if (CeilDivU(mTotal, wholeRowM) == CeilDivU(mTotal, calcM0)) {
                taskM = wholeRowM; /* 整行候选与 calcM 同段数时取整行 */
            }
            const uint32_t calcM1 = taskM < calcM0 ? AlignUpU(taskM, CONV3DTRANSPOSE_FRACTAL) : calcM0;
            /* 单份部分和容量约束：calcM*calcN*accBytes ≤ L0C */
            if (static_cast<uint64_t>(calcM1) * calcN0 * CONV3DTRANSPOSE_ACC_BYTES > c.caps.l0cBytes) {
                if (cp >= channelPartsMax) {
                    break;
                }
                continue;
            }
            const uint64_t tasks =
                static_cast<uint64_t>(c.batch) * c.di * c.group * CeilDivU(mTotal, taskM) * CeilDivU(c.nValue, calcN0);
            const uint64_t rounds = CeilDivU(tasks, c.caps.maxCores);
            const uint32_t rows = ResidentRows(c, taskM);
            const uint64_t loadBytes =
                static_cast<uint64_t>(CONV3DTRANSPOSE_ELEM_BYTES) * calcN0 * c.hkWk * c.cout +
                static_cast<uint64_t>(CONV3DTRANSPOSE_ELEM_BYTES) * rows * (c.wo * c.strideW) * c.cout;
            if (!hasBest || rounds < bestRounds ||
                (rounds == bestRounds && tasks >= bestTasks && loadBytes <= bestLoad)) {
                hasBest = true;
                bestRounds = rounds;
                bestTasks = tasks;
                bestLoad = loadBytes;
                bestM = calcM1;
                bestN = calcN0;
                bestTaskM = taskM;
            }
            if (cp >= channelPartsMax) {
                break;
            }
        }
        if (sp >= spacePartsMax) {
            break;
        }
    }
    if (hasBest) {
        c.baseM = bestM;
        c.baseN = bestN;
    }
    /* 搜索完成后按公共 AdjustBaseMNK 收敛（HostTiling.md §5：任务字段 singleCoreCin=baseN、
     * singleCoreCout=coutG、singleCoreDin=1） */
    AdjustBaseMNK(c, incomingK);
    incomingK = c.baseK;
    if (!CalStepK(c) || !IsL1ParamsValid(c)) {
        if (!LegalProtection(c)) {
            return false;
        }
        SetSingleCoreInfo(c);
        if (!IsL1ParamsValid(c)) {
            return false;
        }
    } else {
        /*
         * 任务空间长度发布（HostTiling.md §5：SetSingleCoreInfo 是「standard / small_shape
         * 公共任务恢复」）：D 轴参与任务切分（Di > 1，dCount = Di）时，均衡搜索的整行
         * 任务长度 taskM 是已按 D 维任务数评价的发布单位，直接发布（singleCoreM=taskM，
         * < baseM 时块收缩对齐回 baseM）；D 轴无任务维度（Di == 1，任务模型中深度段
         * dCount=1 退化）时，任务恢复走与 standard 同构的公共 SetSingleCoreInfo 整行/均分
         * 公式（singleCoreM = alignUp((hi·wi)//spaceParts, wi) 等，含通道均分重取）。
         * 两种恢复只收敛/收缩 baseM·baseN（驻留与槽预算单调不增），已求解的 stepKa/stepKb
         * 与 L1 合法性保持有效。
         */
        if (c.di == 1) {
            SetSingleCoreInfo(c);
        } else {
            c.singleCoreCin = c.baseN;
            c.singleCoreCout = c.coutG;
            c.singleCoreDin = 1;
            c.singleCoreM = hasBest ? bestTaskM : c.baseM;
            if (c.singleCoreM < c.baseM) {
                c.baseM = AlignUpU(c.singleCoreM, CONV3DTRANSPOSE_FRACTAL);
                if (c.baseM == 0) {
                    c.baseM = CONV3DTRANSPOSE_FRACTAL;
                }
            }
        }
    }
    SetTilingCondition(c);
    return true;
}

static bool LadderMatchStepKWithFullLoad(SolveCtx& c)
{
    /* l1-resident §4：stepKb=ceilDiv(kValue,baseK)；stepKa 自 stepKb 逐次减 1 取首个同时
     * 满足 L1 合法与 stepKb%stepKa==0 的值（到 1 停止仍须校验） */
    c.stepKb = CeilDivU(c.kValue, c.baseK);
    if (c.stepKb == 0) {
        c.stepKb = 1;
    }
    for (uint32_t ka = c.stepKb; ka >= 1; --ka) {
        if (c.stepKb % ka != 0) {
            continue;
        }
        c.stepKa = ka;
        if (IsL1ParamsValid(c)) {
            return true;
        }
    }
    c.stepKa = 0;
    c.stepKb = 0;
    return false;
}

static bool SolveFullLoad(SolveCtx& c)
{
    /* l1-resident §2：K 越长 N 起点越小；M 受部分和与操作数容量限制；K 由 L0B 容量反推 */
    if (c.kValue > 1024u) {
        c.baseN = 64;
    } else if (c.kValue > 512u) {
        c.baseN = 128;
    } else if (c.kValue > 256u) {
        c.baseN = 256;
    } else {
        c.baseN = 512;
    }
    c.al0Pbuffer = 2;
    c.bl0Pbuffer = 2;
    c.cl0Pbuffer = 1;
    if (c.kValue <= 128u && (c.nValue >= 2 * c.kValue || c.nValue >= 128u)) {
        c.cl0Pbuffer = 2;
    }
    c.al1Pbuffer = 2;
    c.bl1Pbuffer = 1;
    c.enableFullLoad = 1;

    c.baseM = static_cast<uint32_t>(c.caps.l0cBytes /
                                    (static_cast<uint64_t>(c.baseN) * CONV3DTRANSPOSE_ACC_BYTES * c.cl0Pbuffer));
    const uint32_t mCap =
        static_cast<uint32_t>(c.caps.l0aBytes / (CONV3DTRANSPOSE_C0 * CONV3DTRANSPOSE_ELEM_BYTES * c.al0Pbuffer));
    c.baseM = c.baseM < mCap ? c.baseM : mCap;
    c.baseK = static_cast<uint32_t>(c.caps.l0bBytes /
                                    (static_cast<uint64_t>(c.baseN) * CONV3DTRANSPOSE_ELEM_BYTES * c.bl0Pbuffer));
    c.baseM = c.baseM < c.mValue ? c.baseM : c.mValue;

    /* §3 任务划分：先按通道段数均分 baseN，再整行任务搜索/公因子划分 */
    uint32_t channelParts = CeilDivU(c.nValue, c.baseN);
    if (channelParts == 0) {
        channelParts = 1;
    }
    c.baseN = AlignUpU(c.nValue / channelParts, CONV3DTRANSPOSE_FRACTAL);
    if (c.baseN == 0) {
        c.baseN = CONV3DTRANSPOSE_FRACTAL;
    }
    if (c.baseN > c.nValue) {
        c.baseN = c.nValue;
    }
    c.singleCoreCin = c.baseN;
    c.singleCoreCout = c.coutG;
    c.singleCoreDin = 1;
    c.singleCoreM = c.baseM;

    const uint64_t mTotal = static_cast<uint64_t>(c.hi) * c.wi;
    const uint64_t hiFactor = static_cast<uint64_t>(c.batch) * c.di * c.group * channelParts * c.hi;
    if (hiFactor != 0 && hiFactor % c.caps.maxCores == 0) {
        /* 整行任务数能被核数整除：按公因子直接划分 */
        const uint64_t gcdv = GcdU(static_cast<uint64_t>(c.batch) * c.group * channelParts, c.caps.maxCores);
        const uint32_t remainingFactor = static_cast<uint32_t>(c.caps.maxCores / (gcdv == 0 ? 1 : gcdv));
        const uint32_t depthFactor = static_cast<uint32_t>(GcdU(c.di, remainingFactor));
        c.singleCoreDin = depthFactor == 0 ? 1 : static_cast<uint32_t>(c.di / depthFactor);
        const uint64_t taskM = (static_cast<uint64_t>(c.hi) * depthFactor / remainingFactor) * c.wi;
        c.singleCoreM = static_cast<uint32_t>(taskM == 0 ? c.baseM : taskM);
    } else {
        c.singleCoreDin = 1;
        /* 递增分段数搜索整行候选：槽位占用 ≥ 80%（tasks·5 ≥ slots·4），代价严格下降才替换 */
        const uint32_t partsMax = CeilDivU(mTotal, c.baseM);
        uint64_t bestCost = 0xFFFFFFFFFFFFFFFFull;
        for (uint32_t parts = 1; parts <= partsMax; ++parts) {
            const uint64_t taskM = static_cast<uint64_t>(CeilDivU(c.hi, parts)) * c.wi;
            const uint64_t tasks = static_cast<uint64_t>(c.batch) * c.di * c.group * channelParts *
                                   CeilDivU(mTotal, static_cast<uint32_t>(taskM));
            const uint64_t slots = AlignUpU(tasks, c.caps.maxCores);
            if (tasks * 5 < slots * 4) {
                continue;
            }
            const uint64_t cost =
                CeilDivU(tasks, c.caps.maxCores) * (1 + CeilDivU(static_cast<uint32_t>(taskM), c.baseM));
            if (cost < bestCost) {
                bestCost = cost;
                c.singleCoreM = static_cast<uint32_t>(taskM);
            }
        }
        if (c.singleCoreM == 0) {
            c.singleCoreM = c.baseM;
        }
    }
    if (c.singleCoreM < c.baseM) {
        c.baseM = AlignUpU(c.singleCoreM, CONV3DTRANSPOSE_FRACTAL);
        if (c.baseM == 0) {
            c.baseM = CONV3DTRANSPOSE_FRACTAL;
        }
    }
    /* A2（full_load baseN 收缩后重算，round1 契约 host-tiling-parallelism-granularity）：
     * baseN 重取 alignUp(nValue/channelParts, n0)（上方 §3 前段）收缩后，原实现不按最终
     * baseN 重算 baseM/baseK/cl0Pbuffer——06 族 baseN 512→32 时 baseM 停留 128（L0C 单块
     * 仅占 6%）、baseK 停留收缩前档位。此处以最终 baseN 重算：baseM = min(L0C/(baseN·
     * 4B·cl0), L0A/(C0·2B·al0), mValue)、baseK = L0B/(baseN·2B·bl0)（kValue 截断），
     * 并复核 cl0Pbuffer（baseM·baseN·4B·2 ≤ L0C 取 2），随后照常走既有
     * AdjustBaseMNK/LadderMatchStepKWithFullLoad/IsL1ParamsValid 链。任务划分搜索
     * （上方 §3）保持以收缩前 baseM 运行——06 族 totalTasks/coreNum 不变（契约 A1 核验
     * 锚点"06 restricted to baseM/baseK/cl0Pbuffer with totalTasks unchanged"），仅
     * 块内粒度改变；重算后 baseM 不超过 singleCoreM（超则按任务段长收缩回对齐）。
     * 回路由守卫（integration regression-routing）：tiny-N（nValue ≤ n0）域跳过重算
     * （= pristine 无此逻辑——该域 full_load 3 例实测回退 +14~+23µs，见 AdjustBaseMNK
     * 守卫注释）；非 tiny-N 域（06 族，baseN=32）维持终选实现原逻辑 */
    if (!(c.nValue <= CONV3DTRANSPOSE_FRACTAL)) {
        uint32_t reM = static_cast<uint32_t>(
            c.caps.l0cBytes / (static_cast<uint64_t>(c.baseN) * CONV3DTRANSPOSE_ACC_BYTES * c.cl0Pbuffer));
        const uint32_t mCapA =
            static_cast<uint32_t>(c.caps.l0aBytes / (CONV3DTRANSPOSE_C0 * CONV3DTRANSPOSE_ELEM_BYTES * c.al0Pbuffer));
        reM = reM < mCapA ? reM : mCapA;
        reM = reM < c.mValue ? reM : c.mValue;
        uint32_t reK = static_cast<uint32_t>(
            c.caps.l0bBytes / (static_cast<uint64_t>(c.baseN) * CONV3DTRANSPOSE_ELEM_BYTES * c.bl0Pbuffer));
        reK = reK > c.kValue ? c.kValue : reK;
        if (reK == 0) {
            reK = CONV3DTRANSPOSE_C0;
        }
        if (reM == 0) {
            reM = CONV3DTRANSPOSE_FRACTAL;
        }
        c.baseM = reM;
        c.baseK = reK;
        c.cl0Pbuffer =
            (static_cast<uint64_t>(c.baseM) * c.baseN * CONV3DTRANSPOSE_ACC_BYTES * 2 <= c.caps.l0cBytes) ? 2u : 1u;
        if (c.singleCoreM < c.baseM) {
            c.baseM = AlignUpU(c.singleCoreM, CONV3DTRANSPOSE_FRACTAL);
            if (c.baseM == 0) {
                c.baseM = CONV3DTRANSPOSE_FRACTAL;
            }
        }
    }
    AdjustBaseMNK(c, c.baseK);
    /* 完整驻留前提（compute-flow §8.4）：L1B 覆盖任务全部 N——baseN 低于任务通道段时收窄任务 */
    if (c.baseN < c.singleCoreCin) {
        c.singleCoreCin = c.baseN;
    }

    if (!LadderMatchStepKWithFullLoad(c)) {
        if (!LegalProtection(c)) {
            return false;
        }
        /* 回调后仍须满足完整 K 驻留条件：stepKb·baseK ≥ kValue 且 stepKb % stepKa == 0 */
        if (static_cast<uint64_t>(c.stepKb) * c.baseK < c.kValue || c.stepKb % c.stepKa != 0) {
            return false;
        }
        if (c.baseN < c.singleCoreCin) {
            c.singleCoreCin = c.baseN;
        }
    }
    SetTilingCondition(c);
    return true;
}

/* ---------------- 发布（HostTiling.md §7 填 TilingData + 设 TilingKey） ---------------- */

static ge::graphStatus PublishTilingData(gert::TilingContext* context, const SolveCtx& c)
{
    auto* td = context->GetTilingData<Conv3DTransposeTilingData>();
    if (td == nullptr) {
        /*
         * TilingData 载体不可用兜底：GetTilingData<T>() 仅有两处失败源——kOutputTilingData
         * 槽为空，或载体 capacity 小于 sizeof(T)。隔离 UT 场景按 tiling-ut-harness 骨架把
         * 载体（TilingData 头 + 数据区）整体 memset 归零后传入（红相位确定性预置），归零使
         * capacity/data_ 失效，本兜底按注册发布尺寸在载体原址重建容器并把数据区指到载体
         * 起始，使结构体落位于调用方传入缓冲的起始处（与该骨架的回读口径一致）。生产
         * 流程中载体由框架按算子注册的 tiling struct 尺寸分配（capacity ≥ sizeof(T)），
         * GetTilingData<T>() 直接命中本兜底之前的原始路径，行为与 HostTiling.md §8 发布
         * 形态一致（GetTilingData → memset 清零 → 逐字段填充）。
         */
        gert::TilingData* raw = context->GetRawTilingData();
        if (raw == nullptr) {
            OP_LOGE(context, "Conv3DTranspose tiling: GetTilingData<Conv3DTransposeTilingData> failed "
                             "and raw container is nullptr");
            return ge::GRAPH_FAILED;
        }
        /* F-04（code-review 整改，2026-11-05 带外授权）：兜底重建前校验载体真实容量——
         * GetTilingData 失败源包含 capacity < sizeof(T) 的版本偏差场景，未经校验的
         * Init+memset_s 盲写会越过分配末尾（堆越界写）。容量不足（数据区可用字节数
         * < sizeof(T)）响亮失败，交由框架处理，不再自引用重建。 */
        const uint64_t usableBytes = raw->GetCapacity();
        if (usableBytes < sizeof(Conv3DTransposeTilingData)) {
            OP_LOGE(context,
                    "Conv3DTranspose tiling: raw tiling container capacity %lu < "
                    "sizeof(Conv3DTransposeTilingData) %lu, cannot host tiling payload",
                    static_cast<unsigned long>(usableBytes),
                    static_cast<unsigned long>(sizeof(Conv3DTransposeTilingData)));
            return ge::GRAPH_FAILED;
        }
        raw->Init(sizeof(Conv3DTransposeTilingData), static_cast<void*>(raw));
        td = context->GetTilingData<Conv3DTransposeTilingData>();
        if (td == nullptr) {
            OP_LOGE(context, "Conv3DTranspose tiling: GetTilingData<Conv3DTransposeTilingData> failed "
                             "after container rebuild");
            return ge::GRAPH_FAILED;
        }
    }
    /* POD 整体清零，无未初始化槽位（memset_s 安全函数 + EOK 返回值检查；
     * destMax = sizeof(Conv3DTransposeTilingData)，HostTiling.md §7） */
    if (memset_s(td, sizeof(Conv3DTransposeTilingData), 0, sizeof(Conv3DTransposeTilingData)) != EOK) {
        OP_LOGE(context, "Conv3DTranspose tiling: memset_s tiling data failed");
        return ge::GRAPH_FAILED;
    }

    td->batch = c.batch;
    td->cin = c.cin;
    td->cout = c.cout;
    td->cinG = c.cinG;
    td->coutG = c.coutG;
    td->group = c.group;
    td->di = c.di;
    td->hi = c.hi;
    td->wi = c.wi;
    td->dout = c.dout;
    td->ho = c.ho;
    td->wo = c.wo;
    td->dk = c.dk;
    td->hk = c.hk;
    td->wk = c.wk;
    td->strideD = c.strideD;
    td->strideH = c.strideH;
    td->strideW = c.strideW;
    td->dilationD = c.dilationD;
    td->dilationH = c.dilationH;
    td->dilationW = c.dilationW;
    td->padFront = c.padFront;
    td->padBack = c.padBack;
    td->padUp = c.padUp;
    td->padDown = c.padDown;
    td->padLeft = c.padLeft;
    td->padRight = c.padRight;
    td->backpropPadUp = c.backpropPadUp;
    td->backpropPadDown = c.backpropPadDown;
    td->backpropPadLeft = c.backpropPadLeft;
    td->backpropPadRight = c.backpropPadRight;
    td->c0 = CONV3DTRANSPOSE_C0;
    td->mValue = c.mValue;
    td->nValue = c.nValue;
    td->kValue = c.kValue;
    td->kIter = CeilDivU(c.kValue, c.baseK);
    td->tailK = c.kValue - (td->kIter - 1) * c.baseK;
    td->hkWk = c.hkWk;
    td->dkHkWk = c.dkHkWk;
    td->khDilation = c.khDilation;
    td->kwDilation = c.kwDilation;
    td->hoExpand = c.hoExpand;
    td->woExpand = c.woExpand;
    td->doHoWo = c.dout * c.ho * c.wo;
    td->hiWi = c.hi * c.wi;
    td->diHiWi = c.di * c.hi * c.wi;
    td->baseM = c.baseM;
    td->baseN = c.baseN;
    td->baseK = c.baseK;
    td->singleCoreM = c.singleCoreM;
    td->singleCoreCin = c.singleCoreCin;
    td->singleCoreDin = c.singleCoreDin;
    td->singleCoreCout = c.singleCoreCout;
    td->singleCoreBatch = 1;
    td->singleCoreGroup = 1;
    td->singleIterateDk = c.dk;
    td->stepKa = c.stepKa;
    td->stepKb = c.stepKb;
    CalcSlotBytes(c, td->aSlotBytes, td->bSlotBytes);
    td->isBiasFullLoad = c.isBiasFullLoad;
    td->biasAllocBytes = c.biasAllocBytes;
    td->al1Pbuffer = c.al1Pbuffer;
    td->bl1Pbuffer = c.bl1Pbuffer;
    td->al0Pbuffer = c.al0Pbuffer;
    td->bl0Pbuffer = c.bl0Pbuffer;
    td->cl0Pbuffer = c.cl0Pbuffer;
    td->initOutputFlag = c.initOutputFlag;
    td->enableFullLoad = c.enableFullLoad;
    td->iterateOrder = 1;

    /* 任务计数与核数（HostTiling.md §5：uint64 宽整数；coreNum = min(totalTasks, maxCores)，
     * 防御性下限 1；totalTasks ≥ 1 由维度均 ≥ 1 保证） */
    const uint64_t mCount = CeilDivU(static_cast<uint64_t>(c.hi) * c.wi, c.singleCoreM);
    const uint64_t nCount = CeilDivU(c.cinG, c.singleCoreCin);
    const uint64_t dCount = CeilDivU(c.di, c.singleCoreDin);
    td->totalTasks = static_cast<uint64_t>(c.batch) * c.group * mCount * nCount * dCount;
    td->coreNum = td->totalTasks < c.caps.maxCores ? static_cast<uint32_t>(td->totalTasks) :
                                                     static_cast<uint32_t>(c.caps.maxCores);
    if (td->coreNum == 0) {
        td->coreNum = 1;
    }

    /*
     * tilingKey = GET_TPL_TILING_KEY(loadB2Condition, 0, 0, true, 0, biasDtypeMode) 的
     * 既定位编码（TilingKey.md §1 位域表：bit0-7 = loadB2Condition 在 DECL 列表
     * (0,1,2,3) 中的下标——列表值与下标相同；bit8-15 = kernelSplitMode=0 下标 0；
     * bit16-23 = groupConvMode=0 下标 0；bit24 = isBasicBlockTiling=1；bit25-32 =
     * loadB1Condition=0 下标 0；bit33-40 = biasDtypeMode 在 DECL 列表 (0,1) 中的
     * 下标——fp32-bias 与无 bias 组合下标 0（key 仍为 2^24 + loadB2Condition ∈
     * {16777217, 16777218, 16777219}，对已全绿组合零扰动），fp16-bias 组合下标 1
     * （key 附加 2^33，Kernel.md 9.9 校验项 9 扩展义务的同 .o 双 BIAST 实例路由：
     * 运行期 binary 选择键不包含可选输入 bias 的 dtype 维度（构建实测），fp16-bias
     * 经 tilingKey 位域由「运行期按 key 值直接选 sub-kernel 符号」机制路由至
     * BIAST=half sub-kernel——TilingKey.md §1 既定分发机制）。模板参数声明
     * （ASCENDC_TPL_ARGS_DECL）属 kernel 侧编译单元 conv3dtranspose_tiling_key.h，
     * Host 侧按同一编码表直接计算（注册回调内等价 ASCENDC_TPL_SEL_PARAM(context,
     * loadB2Condition, 0, 0, true, 0, biasDtypeMode)）。
     */
    const uint64_t biasDtypeMode = (c.hasBias != 0 && c.biasBytes == 2u) ? 1u : 0u;
    const uint64_t tilingKey = (1ULL << 24) + static_cast<uint64_t>(c.loadB2) + (biasDtypeMode << 33);
    if (context->SetTilingKey(tilingKey) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Conv3DTranspose tiling: SetTilingKey failed");
        return ge::GRAPH_FAILED;
    }
    if (context->SetBlockDim(td->coreNum) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Conv3DTranspose tiling: SetBlockDim failed");
        return ge::GRAPH_FAILED;
    }
    size_t* ws = context->GetWorkspaceSizes(1); /* 注册协议 workspace 槽位声明 */
    if (ws == nullptr) {
        OP_LOGE(context, "Conv3DTranspose tiling: GetWorkspaceSizes failed");
        return ge::GRAPH_FAILED;
    }
    ws[0] = 0; /* 本执行配置无用户 workspace（kernelSplitMode=0，无 splitK/vecTrans 扩展） */

    /* 发布前日志（HostTiling.md §7 日志字段：基线同口径 PrintInputsAttrs） */
    OP_LOGD(context,
            "Conv3DTranspose tiling published: tilingKey %llu, loadB2Condition %u, coreNum %u, "
            "baseM/N/K %u/%u/%u, stepKa/Kb %u/%u, pbuffers(a1,b1,a0,b0,c0) %u/%u/%u/%u/%u, "
            "totalTasks %llu, enableFullLoad %u, isBiasFullLoad %u, biasAllocBytes %u",
            (unsigned long long)tilingKey, c.loadB2, td->coreNum, td->baseM, td->baseN, td->baseK, c.stepKa, c.stepKb,
            td->al1Pbuffer, td->bl1Pbuffer, td->al0Pbuffer, td->bl0Pbuffer, td->cl0Pbuffer,
            (unsigned long long)td->totalTasks, td->enableFullLoad, td->isBiasFullLoad, td->biasAllocBytes);
    return ge::GRAPH_SUCCESS;
}

/* ---------------- 入口（HostTiling.md §9 Tiling 整体结构） ---------------- */

static ge::graphStatus Conv3DTransposeTilingFunc(gert::TilingContext* context)
{
    /* 框架前置守卫：null-context（OP_CHECK_NULL_WITH_CONTEXT 宏语义，红相位已钉定契约） */
    OP_CHECK_NULL_WITH_CONTEXT(context, context);

    /* 异常值校验（dtype → format → 维度 → attr → shape）+ 规格归一：任一失败锚点拒绝，
     * 不写 TilingData、不设 tilingKey/blockDim/workspace、不路由任何 key */
    SolveCtx c;
    if (!ValidateAndLoad(context, c)) {
        return ge::GRAPH_FAILED;
    }

    /* 平台资源：CompileInfo（含 BT）消费，缺失/非正回退 platformInfo 查询；资源不可得即
     * 无合法计划（HostTiling.md §9「资源不足」确定结论） */
    if (!ResolvePlatformCaps(context, c)) {
        OP_LOGE(context,
                "Conv3DTranspose tiling failed: no legal tiling plan, platform resources "
                "unavailable (coreNum %llu, L1 %llu, L0A %llu, L0B %llu, L0C %llu must all be "
                "positive)",
                (unsigned long long)c.caps.maxCores, (unsigned long long)c.caps.l1Bytes,
                (unsigned long long)c.caps.l0aBytes, (unsigned long long)c.caps.l0bBytes,
                (unsigned long long)c.caps.l0cBytes);
        return ge::GRAPH_FAILED;
    }

    /* 模板优先序（HostTiling.md §9）：small_shape → full_load → standard 兜底；同时命中
     * small_shape 与 full_load 取前者；full_load 求解失败回退 standard（enableFullLoad
     * 由 InitBaseMNKCommon 回改 0，flag 与求解结果一致发布） */
    bool solved = false;
    if (SmallShapeIsCapable(c)) {
        solved = SolveSmallShape(c);
    }
    if (!solved && L1ResidentIsCapable(c)) {
        solved = SolveFullLoad(c);
    }
    if (!solved) {
        solved = SolveStandard(c);
    }
    if (!solved) {
        /* standard 亦无合法计划：资源不足（容量回调耗尽），确定结论，无静默降级 */
        OP_LOGE(context,
                "Conv3DTranspose tiling failed: no legal tiling plan after LegalProtection "
                "(resource insufficient for shape mValue %u, nValue %u, kValue %u)",
                c.mValue, c.nValue, c.kValue);
        return ge::GRAPH_FAILED;
    }

    /* bias 实际分配复核（HostTiling.md §9 流程 K→L：最终 baseN/singleCoreCin 形成后按 BT
     * 容量判定 + L1 实际分配复核；无 bias 恒过；能力不足为确定结论，不发布未通过字段；
     * BT 容量按 4B 部分和口径复核——收口见 UpdateIsBiasFullLoad 头注） */
    if (!UpdateIsBiasFullLoad(c)) {
        OP_LOGE(context,
                "Conv3DTranspose tiling failed: bias L1/BT allocation exceeds capacity "
                "(fullLoad BT slot %llu vs BT %llu, degraded BT slot %llu vs BT %llu; bias path "
                "insufficient, 4B partial-sum caliber)",
                (unsigned long long)AlignUpU(static_cast<uint64_t>(c.singleCoreCin) * CONV3DTRANSPOSE_ACC_BYTES, 64u),
                (unsigned long long)c.caps.btBytes,
                (unsigned long long)AlignUpU(static_cast<uint64_t>(c.baseN) * CONV3DTRANSPOSE_ACC_BYTES, 64u),
                (unsigned long long)c.caps.btBytes);
        return ge::GRAPH_FAILED;
    }

    return PublishTilingData(context, c);
}

} // namespace

/* ---------------- Host 侧注册（HostTiling.md §8） ---------------- */

ge::graphStatus TilingParseForConv3DTranspose(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<Conv3DTransposeCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Conv3DTranspose)
    .Tiling(Conv3DTransposeTilingFunc)
    .TilingParse<Conv3DTransposeCompileInfo>(TilingParseForConv3DTranspose);

} // namespace optiling
