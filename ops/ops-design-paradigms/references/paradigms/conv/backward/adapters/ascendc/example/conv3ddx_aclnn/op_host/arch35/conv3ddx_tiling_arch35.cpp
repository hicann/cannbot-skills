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
 * \file conv3ddx_tiling_arch35.cpp
 * \brief conv3ddx Host 侧 TilingFunc 真实实现（绿相位，TDD UT 全绿目标）
 *
 * 实现真值（design/HostTiling.md「Host 侧 Tiling 方案」全文，逐节对应）：
 *   §3 输入预处理/合轴/场景预处理 —— input_size 值读取（值依赖）、布局/轴序归一、
 *      属性空间分量提取（NCDHW [2:5] / NDHWC [1:4]；pads 语义序与 data_format 无关）、
 *      派生几何量（ρ / b⁻ / b⁺ 有符号公式 / mValue / nValue / kValue / initOutputFlag）；
 *      §3.2 异常值校验按 dtype → format → 维度 → attr → shape 顺序执行，任一失败
 *      OP_LOGE 稳定锚点 + 返回 ge::GRAPH_FAILED，不路由任何 tilingKey（锚点契约：
 *      "dtype" / "format" / "rank" / "null_input" / "shape_mismatch" /
 *      "attribute_value_out_of_range" / "no legal"，与 UT 断言 playbook 一致）。
 *   §4 UB 切分（本范式 Cube 通路即 L0 基本块 + L1 驻留设计）—— InitBaseMNK 起点
 *      （核体积 1 取 (256,256,64)，其余 (512,128,32)）、AdjustBaseMNK（形状截断 /
 *      N 缩小扩大 M / M 缩小扩大 N / K 窗口配对 / L0C 份数 / 三路容量复核收缩）、
 *      CalStepK（等分/阶梯候选 + 首次求解选择规则：等分候选合法且其 step 之和严格
 *      大于阶梯候选时选等分，否则选阶梯；LegalProtection 内回调重选为先阶梯后等分）、
 *      IsL1ParamsValid（L1 驻留预算 + 槽级复核）、
 *      LegalProtection 容量回调（减 K → 减 M/N → 恢复大 K → 调 L1 份数 (2,2)→(2,1)→
 *      (1,2)→(1,1)，不切换模板）。
 *   §5 多核切分 —— SetSingleCoreInfo（空间/通道/深度划分 + 整行候选）、任务计数
 *      totalTasks = batch·group·dCount·mCount·nCount（uint64 宽整数）、
 *      coreNum = min(totalTasks, maxCores)（≥1）。
 *   §7 填 TilingData + 设 TilingKey —— 整结构清零 + 逐字段填充；loadB2 判定在
 *      求解完成后的最终 baseN 上按 BranchRoute.md §1 判定链执行（核体积 1 → 3；
 *      NDHWC ∧ baseN ≥ hk·wk → 2；NCDHW ∧ dk·hk·wk ≤ 64/2B/cin → 2；兜底 → 1），
 *      key = 2^24 + loadB2Condition（GET_TPL_TILING_KEY(loadB2Condition, 0, 0, true, 0)
 *      的既定位编码，TilingKey.md §1 位域表）；SetBlockDim(coreNum)；workspace 槽 ws[0]=0。
 *   §8 Host 侧注册 —— IMPL_OP_OPTILING(Conv3ddx).Tiling(Conv3ddxTilingFunc)
 *      .TilingParse<Conv3ddxCompileInfo>(TilingParseForConv3ddx)。
 *   §9 Tiling 整体结构 —— 模板优先序 small_shape → full_load → standard 兜底
 *      （full_load 求解失败回退 standard，enableFullLoad 回改 0）；回退链每个返回
 *      路径均有确定结论（校验失败 / 资源不足 no legal / 成功发布），无静默降级。
 *
 * 求解算法参考：卷积反向范式 generic/tiling.md §1–§10 + 三模板文档（templates/
 * standard、small-shape、l1-resident），实现以同族 20 例验证全绿的直调工程
 * conv3ddx_direct 的 Host 求解器为底本转录（范式 example/conv3ddx_direct），本文件按
 * 注册形态（gert::TilingContext / CompileInfo / 三 filter 布局 / 71 字段共享结构）适配；
 * 分支常量与公式见 design/branches/DESIGN-BRANCH-1677721{7,8,9}.md「具体切分计算」
 * （三分支共享 Host 求解器，无分支私有切分）。
 *
 * 注：本工程脚手架 op/op_host/CMakeLists.txt 固定以 arch35/conv3ddx_tiling.cpp 为
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

#include "../../op_kernel/arch35/conv3ddx_tiling_struct.h"
#include "conv3ddx_tiling_arch35.h"

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

/* 接口单轴/元素值可表示域上界：超出即视为乘积/字段可表示范围溢出（shape_mismatch） */
constexpr int64_t CONV3DDX_MAX_DIM = 2147483647LL; /* 2^31 - 1 */

/* ---------------- 求解上下文 ---------------- */

struct PlatformCaps {
    uint64_t maxCores = 0;
    uint64_t l1Bytes = 0;
    uint64_t l0aBytes = 0;
    uint64_t l0bBytes = 0;
    uint64_t l0cBytes = 0;
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
    uint32_t rhoD = 0, rhoH = 0, rhoW = 0;
    uint32_t khDilation = 0, kwDilation = 0, kdDilation = 0; /* KeffH/KeffW/KeffD */
    uint32_t hkWk = 0, dkHkWk = 0, windowSpan = 0;
    uint32_t mValue = 0, nValue = 0, kValue = 0;
    uint32_t hoExpand = 0, woExpand = 0;
    uint32_t initOutputFlag = 0;
    ge::Format filterFormat = ge::FORMAT_ND; /* filter 物理布局（独立于 data_format，三选一） */

    /* 平台 */
    PlatformCaps caps;

    /* 求解结果 */
    uint32_t baseM = 0, baseN = 0, baseK = 0;
    uint32_t stepKa = 0, stepKb = 0;
    uint32_t al1Pbuffer = 0, bl1Pbuffer = 0, al0Pbuffer = 0, bl0Pbuffer = 0, cl0Pbuffer = 0;
    uint32_t singleCoreM = 0, singleCoreCin = 0, singleCoreDin = 0, singleCoreCout = 0;
    uint32_t enableFullLoad = 0;
    uint32_t loadB2 = 0;
    /* F1 修复模式开关（r2-tiling-loadguard*）：0 = 关闭（pass-1，逐字 baseline 求解，
     * F3/F4/F5 修复位点全部惰性）；1 = 开启（pass-2，r1 F1/F2/F3 修复逻辑生效）。
     * 默认 0——仅双解驱动（Conv3ddxTilingFunc）按 pass-2 显式置 1，其余路径行为不变。 */
    uint32_t repairsEnabled = 0;
};

/* ---------------- 校验与归一（HostTiling.md §3；锚点契约见文件头） ---------------- */

/*
 * 检查顺序（HostTiling.md §3.2 五行分类，任一失败 OP_LOGE 锚点 + 返回 false）：
 *   1) dtype（三路 dtype 一致 + 白名单 + input_size 整型）      -> "dtype"
 *   2) format（IO format 合法与配对 + data_format 枚举 +
 *      groups 配对收窄 NDHWC 仅限 groups=1）                    -> "format" / attr 锚点
 *   3) 维度（rank → input_size 长度 → 空张量）                  -> "rank" / "shape_mismatch" / "null_input"
 *   4) attr 值域（长度 / 值域 / 装载字段上限 / groups 域）       -> "attribute_value_out_of_range"
 *      —— 值读取后：groups 整除 Cin（attr）→ Cin==CinG×groups（shape）→
 *         groups 整除 Cout（attr）→ dedy.Cout==filter.Cout（shape）→
 *         dedy.N==input_size.N（shape）→ 前向几何一致性（shape）；
 *         strides/dilations 的 N/C 位序约束（attr 位序项）在几何一致性之后复核
 *         （位序解释依赖张量轴序与 data_format 已一致——轴序矛盾先行按 shape 拒绝，
 *         纯位序违约再按 attribute_value_out_of_range 拒绝）。
 *   5) 值读取与元素值域（<0 / ==0 / 超可表示域）                -> "shape_mismatch" / "null_input"
 *   6) 输出形状契约（y.shape = input_size 元素值，逐维比对；
 *      动态图 -1 维跳过——运行期分配与该值同源推导）            -> "shape_mismatch"
 *   7) 反向边界装载域（bpPad* 任一 >255 拒绝；负侧结构性
 *      ≥ -255 由 pads≤255 保证，kernel 负分支裁剪装载残差）    -> "shape_mismatch"
 */
static bool ValidateAndLoad(gert::TilingContext* context, SolveCtx& c)
{
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context, "Conv3ddx tiling: GetAttrs() is nullptr");
        return false;
    }
    const auto* sTensor = context->GetInputTensor(0);
    const auto* fTensor = context->GetInputTensor(1);
    const auto* dTensor = context->GetInputTensor(2);
    const auto* sShape = context->GetInputShape(0);
    const auto* fShape = context->GetInputShape(1);
    const auto* dShape = context->GetInputShape(2);
    const auto* yDesc = context->GetOutputDesc(0);
    if (sTensor == nullptr || fTensor == nullptr || dTensor == nullptr || sShape == nullptr || fShape == nullptr ||
        dShape == nullptr || yDesc == nullptr) {
        OP_LOGE(context, "Conv3ddx tiling: input tensor/shape or output desc is nullptr");
        return false;
    }

    /* ---------- 行 1 dtype：三路 dtype 一致与组合内（dtype_not_supported） ---------- */
    const ge::DataType filterDtype = fTensor->GetDataType();
    const ge::DataType dedyDtype = dTensor->GetDataType();
    const ge::DataType sizeDtype = sTensor->GetDataType();
    if (filterDtype != dedyDtype || (filterDtype != ge::DT_FLOAT16 && filterDtype != ge::DT_BF16) ||
        (sizeDtype != ge::DT_INT32 && sizeDtype != ge::DT_INT64)) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: dtype check failed (dtype_not_supported), "
                "filter dtype %d, out_backprop dtype %d, input_size dtype %d; supported: "
                "filter == out_backprop in {float16, bfloat16}, input_size in {int32, int64}",
                static_cast<int>(filterDtype), static_cast<int>(dedyDtype), static_cast<int>(sizeDtype));
        return false;
    }

    /* ---------- 行 2 format：IO format 合法与配对 ---------- */
    const ge::Format sizeFmt = sTensor->GetStorageFormat();
    const ge::Format filterFmt = fTensor->GetStorageFormat();
    const ge::Format dedyFmt = dTensor->GetStorageFormat();
    const ge::Format yFmt = yDesc->GetStorageFormat();
    if (sizeFmt != ge::FORMAT_ND) {
        OP_LOGE(context, "Conv3ddx tiling rejected: format check failed, input_size format %d is not ND",
                static_cast<int>(sizeFmt));
        return false;
    }
    if (filterFmt != ge::FORMAT_NCDHW && filterFmt != ge::FORMAT_NDHWC && filterFmt != ge::FORMAT_DHWCN) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: format check failed, filter format %d not in "
                "{NCDHW, NDHWC, DHWCN}",
                static_cast<int>(filterFmt));
        return false;
    }
    if (dedyFmt != ge::FORMAT_NCDHW && dedyFmt != ge::FORMAT_NDHWC) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: format check failed, out_backprop format %d not in "
                "{NCDHW, NDHWC}",
                static_cast<int>(dedyFmt));
        return false;
    }
    if (yFmt != dedyFmt) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: format check failed, y format %d not paired with "
                "out_backprop format %d",
                static_cast<int>(yFmt), static_cast<int>(dedyFmt));
        return false;
    }

    /* 属性载体读取（六属性按 OpDef 声明序：strides/pads/dilations/groups/data_format/enable_hf32） */
    const auto* stridesVec = attrs->GetListInt(0);
    const auto* padsVec = attrs->GetListInt(1);
    const auto* dilationsVec = attrs->GetListInt(2);
    const auto* groupsPtr = attrs->GetInt(3);
    const char* dataFormatRaw = attrs->GetStr(4);
    const auto* hf32Ptr = attrs->GetBool(5);
    if (stridesVec == nullptr || padsVec == nullptr || dilationsVec == nullptr || groupsPtr == nullptr ||
        dataFormatRaw == nullptr || hf32Ptr == nullptr) {
        OP_LOGE(context, "Conv3ddx tiling: attr pointer is nullptr");
        return false;
    }
    /* enable_hf32 信息性消费：本交付 dtype 集（FP16/BF16）下 Host 不据其分支（HostTiling.md §2） */

    /* data_format 枚举（错误分类按违反项归位：attribute_value_out_of_range） */
    const std::string dataFormat(dataFormatRaw);
    bool isNdhwcData = false;
    if (dataFormat == "NCDHW") {
        isNdhwcData = false;
    } else if (dataFormat == "NDHWC") {
        isNdhwcData = true;
    } else {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: attribute_value_out_of_range, data_format '%s' not in "
                "{NCDHW, NDHWC}",
                dataFormat.c_str());
        return false;
    }

    /* groups 配对收窄（NDHWC out_backprop 配三 filter 仅限 groups=1；groups>1 仅 NCDHW） */
    if (*groupsPtr > 1 && dedyFmt == ge::FORMAT_NDHWC) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: format check failed, NDHWC out_backprop pairs the "
                "three filter layouts only when groups == 1 (pairing narrowing), got groups %lld",
                static_cast<long long>(*groupsPtr));
        return false;
    }

    /* ---------- 行 3 维度：rank / input_size 长度 / 空张量 ---------- */
    const size_t sRank = sShape->GetStorageShape().GetDimNum();
    const size_t fRank = fShape->GetStorageShape().GetDimNum();
    const size_t dRank = dShape->GetStorageShape().GetDimNum();
    if (sRank != 1U) {
        OP_LOGE(context, "Conv3ddx tiling rejected: rank check failed, input_size rank is %zu, expect 1", sRank);
        return false;
    }
    if (fRank != 5U) {
        OP_LOGE(context, "Conv3ddx tiling rejected: rank check failed, filter rank is %zu, expect 5", fRank);
        return false;
    }
    if (dRank != 5U) {
        OP_LOGE(context, "Conv3ddx tiling rejected: rank check failed, out_backprop rank is %zu, expect 5", dRank);
        return false;
    }
    if (sShape->GetStorageShape().GetDim(0) != 5) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: shape_mismatch, input_size length is %lld, expect 5 "
                "(1D int32/int64, length 5 contract)",
                static_cast<long long>(sShape->GetStorageShape().GetDim(0)));
        return false;
    }
    for (size_t i = 0; i < 5U; ++i) {
        if (fShape->GetStorageShape().GetDim(i) <= 0 || dShape->GetStorageShape().GetDim(i) <= 0) {
            OP_LOGE(context,
                    "Conv3ddx tiling rejected: null_input, empty tensor detected, dim %zu of "
                    "filter/out_backprop is non-positive (filter %lld, out_backprop %lld)",
                    i, static_cast<long long>(fShape->GetStorageShape().GetDim(i)),
                    static_cast<long long>(dShape->GetStorageShape().GetDim(i)));
            return false;
        }
        if (fShape->GetStorageShape().GetDim(i) > CONV3DDX_MAX_DIM ||
            dShape->GetStorageShape().GetDim(i) > CONV3DDX_MAX_DIM) {
            OP_LOGE(context,
                    "Conv3ddx tiling rejected: shape_mismatch, dim %zu of filter/out_backprop "
                    "exceeds representable range %lld (product overflow)",
                    i, static_cast<long long>(CONV3DDX_MAX_DIM));
            return false;
        }
    }

    /* ---------- 行 4 attr 值域与长度（位序项 N/C 位延后到几何一致性之后） ---------- */
    if (stridesVec->GetSize() != 5U || dilationsVec->GetSize() != 5U) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: attribute_value_out_of_range, strides/dilations list "
                "length %zu/%zu, expect 5",
                stridesVec->GetSize(), dilationsVec->GetSize());
        return false;
    }
    if (padsVec->GetSize() != 6U) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: attribute_value_out_of_range, pads list length %zu, "
                "expect 6",
                padsVec->GetSize());
        return false;
    }
    const int64_t* strides = stridesVec->GetData();
    const int64_t* pads = padsVec->GetData();
    const int64_t* dilations = dilationsVec->GetData();
    for (size_t i = 0; i < 5U; ++i) {
        if (strides[i] < 1 || strides[i] > CONV3DDX_MAX_DIM) {
            OP_LOGE(context,
                    "Conv3ddx tiling rejected: attribute_value_out_of_range, strides[%zu]=%lld "
                    "out of [1, 2147483647]",
                    i, static_cast<long long>(strides[i]));
            return false;
        }
        if (dilations[i] < 1 || dilations[i] > CONV3DDX_MAX_DIM) {
            OP_LOGE(context,
                    "Conv3ddx tiling rejected: attribute_value_out_of_range, dilations[%zu]=%lld "
                    "out of [1, 2147483647]",
                    i, static_cast<long long>(dilations[i]));
            return false;
        }
    }
    for (size_t i = 0; i < 6U; ++i) {
        if (pads[i] < 0 || pads[i] > CONV3DDX_MAX_DIM) {
            OP_LOGE(context,
                    "Conv3ddx tiling rejected: attribute_value_out_of_range, pads[%zu]=%lld "
                    "out of [0, 2147483647]",
                    i, static_cast<long long>(pads[i]));
            return false;
        }
    }
    /* 装载字段上限（本执行配置交付边界：接口值域内的超限值按属性非法拒绝） */
    for (size_t i = 0; i < 5U; ++i) {
        if (strides[i] > static_cast<int64_t>(CONV3DDX_MAX_STRIDE)) {
            OP_LOGE(context,
                    "Conv3ddx tiling rejected: attribute_value_out_of_range, strides[%zu]=%lld "
                    "exceeds loadable field cap %u",
                    i, static_cast<long long>(strides[i]), CONV3DDX_MAX_STRIDE);
            return false;
        }
        if (dilations[i] > static_cast<int64_t>(CONV3DDX_MAX_DILATION)) {
            OP_LOGE(context,
                    "Conv3ddx tiling rejected: attribute_value_out_of_range, dilations[%zu]=%lld "
                    "exceeds loadable field cap %u",
                    i, static_cast<long long>(dilations[i]), CONV3DDX_MAX_DILATION);
            return false;
        }
    }
    for (size_t i = 0; i < 6U; ++i) {
        if (pads[i] > static_cast<int64_t>(CONV3DDX_MAX_PAD)) {
            OP_LOGE(context,
                    "Conv3ddx tiling rejected: attribute_value_out_of_range, pads[%zu]=%lld "
                    "exceeds loadable field cap %u",
                    i, static_cast<long long>(pads[i]), CONV3DDX_MAX_PAD);
            return false;
        }
    }
    if (*groupsPtr < 1 || *groupsPtr > 65535) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: attribute_value_out_of_range, groups %lld out of "
                "[1, 65535]",
                static_cast<long long>(*groupsPtr));
        return false;
    }

    /* ---------- input_size 值读取（值依赖：元数据校验通过后进行） ---------- */
    int64_t sizeVals[5] = {0, 0, 0, 0, 0};
    const void* sizeData = nullptr;
    if (sizeDtype == ge::DT_INT64) {
        sizeData = sTensor->GetData<int64_t>();
    } else {
        sizeData = sTensor->GetData<int32_t>();
    }
    if (sizeData == nullptr) {
        OP_LOGE(context, "Conv3ddx tiling rejected: null_input, input_size tensor data is nullptr "
                         "(value-dependent read unavailable)");
        return false;
    }
    for (size_t i = 0; i < 5U; ++i) {
        sizeVals[i] = (sizeDtype == ge::DT_INT64) ? (static_cast<const int64_t*>(sizeData))[i] :
                                                    static_cast<int64_t>((static_cast<const int32_t*>(sizeData))[i]);
    }
    for (size_t i = 0; i < 5U; ++i) {
        if (sizeVals[i] == 0) {
            OP_LOGE(context,
                    "Conv3ddx tiling rejected: null_input, input_size element %zu is 0 "
                    "(empty output tensor)",
                    i);
            return false;
        }
        if (sizeVals[i] < 0) {
            OP_LOGE(context,
                    "Conv3ddx tiling rejected: shape_mismatch, input_size element %zu is %lld "
                    "(negative dim value, no unknown-dim semantics in registered combos)",
                    i, static_cast<long long>(sizeVals[i]));
            return false;
        }
        if (sizeVals[i] > CONV3DDX_MAX_DIM) {
            OP_LOGE(context,
                    "Conv3ddx tiling rejected: shape_mismatch, input_size element %zu is %lld, "
                    "exceeds representable range %lld (product overflow)",
                    i, static_cast<long long>(sizeVals[i]), static_cast<long long>(CONV3DDX_MAX_DIM));
            return false;
        }
    }

    /* ---------- 规格归一装载（HostTiling.md §3.1 步骤 1–3） ---------- */
    /* input_size 值序：NCDHW (N, Cin, Din, Hin, Win)；NDHWC (N, Din, Hin, Win, Cin) */
    const int64_t nIn = sizeVals[0];
    const int64_t cinIn = isNdhwcData ? sizeVals[4] : sizeVals[1];
    const int64_t dinIn = isNdhwcData ? sizeVals[1] : sizeVals[2];
    const int64_t hinIn = isNdhwcData ? sizeVals[2] : sizeVals[3];
    const int64_t winIn = isNdhwcData ? sizeVals[3] : sizeVals[4];

    /* ---------- 输出形状契约校验（y.shape = input_size 元素值；值依赖防线） ----------
     * Tensor 路由（aclnnConv3ddxTensorGetWorkspaceSize：inputSize 为设备张量、调用方
     * out 直通 OP_OUTPUT）在 L2 层无法完成 out 形状与 input_size 元素值的一致性校验
     * （元素值 host 不可读，值依赖），本处是该契约唯一的执行前防线：kernel 按
     * input_size 派生几何写满全输出（LoadL0c2Gm 的 dstOffset 与 baseM/baseN 块尺寸），
     * out 实际分配小于该几何即构成 GM 越界写。标准路由的内部输出张量与 GEIR 图输出
     * 均按同契约构造，本校验对合法调用恒通过。动态图（输出维含 -1）跳过比对：该场景
     * 运行期分配由值依赖 InferShape 从同一 input_size 值同源推导，无错配通道。 */
    const gert::StorageShape* yShapeCtx = context->GetOutputShape(0);
    if (yShapeCtx == nullptr) {
        OP_LOGE(context, "Conv3ddx tiling rejected: output shape is unavailable "
                         "(GetOutputShape(0) is nullptr)");
        return false;
    }
    const gert::Shape& yStorageShape = yShapeCtx->GetStorageShape();
    if (yStorageShape.GetDimNum() != 5U) {
        OP_LOGE(context, "Conv3ddx tiling rejected: shape_mismatch, output y rank is %zu, expect 5",
                yStorageShape.GetDimNum());
        return false;
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
            if (yStorageShape.GetDim(i) != sizeVals[i]) {
                OP_LOGE(context,
                        "Conv3ddx tiling rejected: shape_mismatch, output y dim %zu is "
                        "%lld but input_size element is %lld (y.shape = input_size values "
                        "contract; smaller out would be overflowed by kernel write-back)",
                        i, static_cast<long long>(yStorageShape.GetDim(i)), static_cast<long long>(sizeVals[i]));
                return false;
            }
        }
    } else {
        OP_LOGD(context, "Conv3ddx tiling: output y shape has unknown dim (-1, dynamic graph), "
                         "skip output-shape contract check (runtime allocation derives from the same "
                         "input_size values via value-dependent InferShape)");
    }

    /* filter 布局：NCDHW (Cout, CinG, kD, kH, kW)；NDHWC (Cout, kD, kH, kW, CinG)；
     * DHWCN (kD, kH, kW, CinG, Cout) */
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
    } else { /* DHWCN */
        kdF = fShape->GetStorageShape().GetDim(0);
        khF = fShape->GetStorageShape().GetDim(1);
        kwF = fShape->GetStorageShape().GetDim(2);
        cinGF = fShape->GetStorageShape().GetDim(3);
        coutF = fShape->GetStorageShape().GetDim(4);
    }
    /* out_backprop 轴序：NCDHW (N, Cout, Dout, Hout, Wout)；NDHWC (N, Dout, Hout, Wout, Cout) */
    const int64_t nDedy = dShape->GetStorageShape().GetDim(0);
    const int64_t coutDedy = isNdhwcData ? dShape->GetStorageShape().GetDim(4) : dShape->GetStorageShape().GetDim(1);
    const int64_t doutF = isNdhwcData ? dShape->GetStorageShape().GetDim(1) : dShape->GetStorageShape().GetDim(2);
    const int64_t houtF = isNdhwcData ? dShape->GetStorageShape().GetDim(2) : dShape->GetStorageShape().GetDim(3);
    const int64_t woutF = isNdhwcData ? dShape->GetStorageShape().GetDim(3) : dShape->GetStorageShape().GetDim(4);
    /* 属性空间分量提取（位序合轴）：NCDHW 取 [2:5]，NDHWC 取 [1:4]；
     * pads 语义序 [pF, pB, pT, pBm, pL, pR] 与 data_format 无关 */
    const int64_t sD = isNdhwcData ? strides[1] : strides[2];
    const int64_t sH = isNdhwcData ? strides[2] : strides[3];
    const int64_t sW = isNdhwcData ? strides[3] : strides[4];
    const int64_t dD = isNdhwcData ? dilations[1] : dilations[2];
    const int64_t dH = isNdhwcData ? dilations[2] : dilations[3];
    const int64_t dW = isNdhwcData ? dilations[3] : dilations[4];

    /* ---------- 值依赖联动校验（groups 整除 → filter 通道语义 → batch 对齐） ---------- */
    if (cinIn % (*groupsPtr) != 0) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: attribute_value_out_of_range, groups %lld does not "
                "divide input_size Cin %lld",
                static_cast<long long>(*groupsPtr), static_cast<long long>(cinIn));
        return false;
    }
    if (cinIn != cinGF * (*groupsPtr)) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: shape_mismatch, filter CinG %lld x groups %lld != "
                "input_size Cin %lld (filter channel dims violate groups semantics)",
                static_cast<long long>(cinGF), static_cast<long long>(*groupsPtr), static_cast<long long>(cinIn));
        return false;
    }
    if (coutF % (*groupsPtr) != 0) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: attribute_value_out_of_range, groups %lld does not "
                "divide filter Cout %lld",
                static_cast<long long>(*groupsPtr), static_cast<long long>(coutF));
        return false;
    }
    if (coutDedy != coutF) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: shape_mismatch, out_backprop Cout %lld != filter "
                "Cout %lld (no broadcast alignment)",
                static_cast<long long>(coutDedy), static_cast<long long>(coutF));
        return false;
    }
    if (nDedy != nIn) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: shape_mismatch, out_backprop N %lld != input_size "
                "N %lld (fixed batch dim relation)",
                static_cast<long long>(nDedy), static_cast<long long>(nIn));
        return false;
    }

    /* ---------- 派生几何量与正向形状公式（HostTiling.md §3.3 / preprocessing §4.1） ---------- */
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
    c.dout = static_cast<uint32_t>(doutF);
    c.ho = static_cast<uint32_t>(houtF);
    c.wo = static_cast<uint32_t>(woutF);
    c.dk = static_cast<uint32_t>(kdF);
    c.hk = static_cast<uint32_t>(khF);
    c.wk = static_cast<uint32_t>(kwF);
    c.strideD = static_cast<uint32_t>(sD);
    c.strideH = static_cast<uint32_t>(sH);
    c.strideW = static_cast<uint32_t>(sW);
    c.dilationD = static_cast<uint32_t>(dD);
    c.dilationH = static_cast<uint32_t>(dH);
    c.dilationW = static_cast<uint32_t>(dW);
    c.padFront = static_cast<uint32_t>(pads[0]);
    c.padBack = static_cast<uint32_t>(pads[1]);
    c.padUp = static_cast<uint32_t>(pads[2]);
    c.padDown = static_cast<uint32_t>(pads[3]);
    c.padLeft = static_cast<uint32_t>(pads[4]);
    c.padRight = static_cast<uint32_t>(pads[5]);
    c.filterFormat = filterFmt;

    /* 派生核量先以 uint64 宽整数计算并复核可表示域，再收窄装载（HostTiling.md §3.3/
     * §7 大乘积宽整数检查：先以 uint64/int64 检查再转窄类型） */
    const uint64_t effD64 = (static_cast<uint64_t>(c.dk) - 1) * c.dilationD + 1;
    const uint64_t effH64 = (static_cast<uint64_t>(c.hk) - 1) * c.dilationH + 1;
    const uint64_t effW64 = (static_cast<uint64_t>(c.wk) - 1) * c.dilationW + 1;
    const uint64_t hkWk64 = static_cast<uint64_t>(c.hk) * c.wk;
    const uint64_t dkHkWk64 = static_cast<uint64_t>(c.dk) * hkWk64;
    const uint64_t windowSpan64 = hkWk64 * CONV3DDX_C0;
    if (effD64 > 0xFFFFFFFFULL || effH64 > 0xFFFFFFFFULL || effW64 > 0xFFFFFFFFULL || hkWk64 > 0xFFFFFFFFULL ||
        dkHkWk64 > 0xFFFFFFFFULL || windowSpan64 > 0xFFFFFFFFULL) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: shape_mismatch, derived kernel quantity exceeds "
                "uint32 field range (KeffD %llu, KeffH %llu, KeffW %llu, hk*wk %llu, dk*hk*wk %llu)",
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

    /* 正向形状公式逐轴校验：输出长 = (输入长 + 两侧 pad − 有效核长) // stride + 1（逐轴 uint64） */
    const uint64_t inH = static_cast<uint64_t>(c.hi) + c.padUp + c.padDown;
    if (inH < c.khDilation || (inH - c.khDilation) / c.strideH + 1 != c.ho) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: shape_mismatch, forward geometry inconsistent on H: "
                "expect Hout %llu got %llu (Hin %u, pad %u/%u, KeffH %u, strideH %u)",
                (unsigned long long)((inH - c.khDilation) / c.strideH + 1), (unsigned long long)c.ho, c.hi, c.padUp,
                c.padDown, c.khDilation, c.strideH);
        return false;
    }
    const uint64_t inW = static_cast<uint64_t>(c.wi) + c.padLeft + c.padRight;
    if (inW < c.kwDilation || (inW - c.kwDilation) / c.strideW + 1 != c.wo) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: shape_mismatch, forward geometry inconsistent on W: "
                "expect Wout %llu got %llu (Win %u, pad %u/%u, KeffW %u, strideW %u)",
                (unsigned long long)((inW - c.kwDilation) / c.strideW + 1), (unsigned long long)c.wo, c.wi, c.padLeft,
                c.padRight, c.kwDilation, c.strideW);
        return false;
    }
    const uint64_t inD = static_cast<uint64_t>(c.di) + c.padFront + c.padBack;
    if (inD < c.kdDilation || (inD - c.kdDilation) / c.strideD + 1 != c.dout) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: shape_mismatch, forward geometry inconsistent on D: "
                "expect Dout %llu got %llu (Din %u, pad %u/%u, KeffD %u, strideD %u)",
                (unsigned long long)((inD - c.kdDilation) / c.strideD + 1), (unsigned long long)c.dout, c.di,
                c.padFront, c.padBack, c.kdDilation, c.strideD);
        return false;
    }

    /* ---------- attr 位序复核：strides/dilations 的 N、C 位必须为 1（几何一致性之后） ---------- */
    {
        const size_t nPos = 0;
        const size_t cPos = isNdhwcData ? 4U : 1U;
        if (strides[nPos] != 1 || strides[cPos] != 1) {
            OP_LOGE(context,
                    "Conv3ddx tiling rejected: attribute_value_out_of_range, strides N/C "
                    "positions (N at %zu = %lld, C at %zu = %lld) must be 1",
                    nPos, static_cast<long long>(strides[nPos]), cPos, static_cast<long long>(strides[cPos]));
            return false;
        }
        if (dilations[nPos] != 1 || dilations[cPos] != 1) {
            OP_LOGE(context,
                    "Conv3ddx tiling rejected: attribute_value_out_of_range, dilations N/C "
                    "positions (N at %zu = %lld, C at %zu = %lld) must be 1",
                    nPos, static_cast<long long>(dilations[nPos]), cPos, static_cast<long long>(dilations[cPos]));
            return false;
        }
    }

    /* ---------- 派生量（HostTiling.md §3.3 派生量表，ρ 形式） ---------- */
    c.rhoD = static_cast<uint32_t>((inD - c.kdDilation) % c.strideD);
    c.rhoH = static_cast<uint32_t>((inH - c.khDilation) % c.strideH);
    c.rhoW = static_cast<uint32_t>((inW - c.kwDilation) % c.strideW);
    c.backpropPadUp = static_cast<int32_t>(c.khDilation - 1 - c.padUp);
    c.backpropPadDown = static_cast<int32_t>(c.khDilation - 1 - c.padDown + c.rhoH);
    c.backpropPadLeft = static_cast<int32_t>(c.kwDilation - 1 - c.padLeft);
    c.backpropPadRight = static_cast<int32_t>(c.kwDilation - 1 - c.padRight + c.rhoW);
    c.hoExpand = (c.ho - 1) * c.strideH + 1;
    c.woExpand = (c.wo - 1) * c.strideW + 1;

    /* ---------- 反向边界装载域校验（SEC-1.1：Load3D padList 8 位字段契约） ----------
     * kernel 侧 EnsureA1 把 bpLeft/bpRight（非负分支）与 -curHoIdx_（有界 ≤ backpropPadUp）
     * 直接装入 LoadData3DParamsV2.padList（uint8_t[4]，元素域 [0,255]）：四侧反向边界
     * 任一为正且超 255 时装载发生截断（可构造场景：wk=257/dilation=1/pads=0 时
     * bpLeft=256 → uint8(256)=0，im2col 窗口错位、输出错误），在此拒绝下发 tilingKey。
     * 负侧无需复核：pads 已限 [0,255] 且 Keff≥1，b⁻ = Keff-1-pad ≥ -255 恒成立；
     * 负值走 kernel 侧 woStart/realStart 预裁剪分支，装载残差 ∈ [0, stride-1]。 */
    if (c.backpropPadUp > static_cast<int32_t>(CONV3DDX_MAX_PAD) ||
        c.backpropPadDown > static_cast<int32_t>(CONV3DDX_MAX_PAD) ||
        c.backpropPadLeft > static_cast<int32_t>(CONV3DDX_MAX_PAD) ||
        c.backpropPadRight > static_cast<int32_t>(CONV3DDX_MAX_PAD)) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: shape_mismatch, backprop pad exceeds loadable "
                "pad field cap %u (bpUp %d, bpDown %d, bpLeft %d, bpRight %d; uint8 padList "
                "contract — oversized positive boundary would truncate and corrupt im2col)",
                CONV3DDX_MAX_PAD, c.backpropPadUp, c.backpropPadDown, c.backpropPadLeft, c.backpropPadRight);
        return false;
    }

    /* 派生乘积可表示域复核（uint32 字段域；宽整数计算后再收窄，HostTiling.md §7 发布前检查） */
    const uint64_t hiWi = static_cast<uint64_t>(c.hi) * c.wi;
    const unsigned __int128 diHiWi = static_cast<unsigned __int128>(c.di) * hiWi;
    const unsigned __int128 doHoWo = static_cast<unsigned __int128>(c.dout) * c.ho * c.wo;
    const uint64_t coutAlign = AlignUpU(c.coutG, CONV3DDX_C0);
    const uint64_t kValue64 = coutAlign * c.hkWk;
    if (hiWi > 0xFFFFFFFFULL || diHiWi > 0xFFFFFFFFULL || doHoWo > 0xFFFFFFFFULL || kValue64 > 0xFFFFFFFFULL) {
        OP_LOGE(context,
                "Conv3ddx tiling rejected: shape_mismatch, derived product exceeds uint32 "
                "field range (hi*wi %llu, di*hi*wi %llu, do*ho*wo %llu, kValue %llu)",
                (unsigned long long)hiWi, (unsigned long long)(uint64_t)diHiWi, (unsigned long long)(uint64_t)doHoWo,
                (unsigned long long)kValue64);
        return false;
    }
    c.mValue = AlignUpU(hiWi, CONV3DDX_FRACTAL);
    c.nValue = AlignUpU(c.cinG, CONV3DDX_FRACTAL);
    c.kValue = static_cast<uint32_t>(kValue64);

    /* 场景预处理——输出初始化判定（preprocessing §5.2 DX 形状规则，信息性标志） */
    c.initOutputFlag = 0;
    if (c.rhoD > c.padBack || c.rhoH > c.padDown || c.strideH > c.hk || c.dilationD > 1 || c.strideD > c.dk) {
        c.initOutputFlag = 1;
    }
    return true;
}

/* ---------------- 平台资源（HostTiling.md §4.1/§8：CompileInfo 优先，缺失/非正回退查询） ---------------- */

static bool ResolvePlatformCaps(gert::TilingContext* context, SolveCtx& c)
{
    const auto* compileInfo = reinterpret_cast<const Conv3ddxCompileInfo*>(context->GetCompileInfo());
    bool fromCompileInfo = compileInfo != nullptr;
    if (fromCompileInfo) {
        fromCompileInfo = compileInfo->coreNum > 0 && compileInfo->l1Bytes > 0 && compileInfo->l0aBytes > 0 &&
                          compileInfo->l0bBytes > 0 && compileInfo->l0cBytes > 0;
    }
    if (fromCompileInfo) {
        c.caps.maxCores = static_cast<uint64_t>(compileInfo->coreNum);
        c.caps.l1Bytes = static_cast<uint64_t>(compileInfo->l1Bytes);
        c.caps.l0aBytes = static_cast<uint64_t>(compileInfo->l0aBytes);
        c.caps.l0bBytes = static_cast<uint64_t>(compileInfo->l0bBytes);
        c.caps.l0cBytes = static_cast<uint64_t>(compileInfo->l0cBytes);
    } else {
        /* 回退运行时查询（buffer-and-types §1.2 注册入口）：GetCoreNumAic / GetCoreMemSize */
        fe::PlatFormInfos* platformInfos = context->GetPlatformInfo();
        if (platformInfos == nullptr) {
            return false;
        }
        const platform_ascendc::PlatformAscendC platform(platformInfos);
        uint64_t l1 = 0;
        uint64_t l0a = 0;
        uint64_t l0b = 0;
        uint64_t l0c = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
        c.caps.maxCores = static_cast<uint64_t>(platform.GetCoreNumAic());
        c.caps.l1Bytes = l1;
        c.caps.l0aBytes = l0a;
        c.caps.l0bBytes = l0b;
        c.caps.l0cBytes = l0c;
    }
    /* arch35 参考常量交叉核对（不一致以查询/CompileInfo 值为准并记录；不参与决策） */
    if (c.caps.l1Bytes != CONV3DDX_L1_BYTES || c.caps.l0aBytes != CONV3DDX_L0A_BYTES ||
        c.caps.l0bBytes != CONV3DDX_L0B_BYTES || c.caps.l0cBytes != CONV3DDX_L0C_BYTES ||
        c.caps.maxCores != CONV3DDX_MAX_CORES) {
        OP_LOGD(context,
                "Conv3ddx tiling platform caps differ from arch35 reference values "
                "(coreNum %llu vs %u, L1 %llu vs %u, L0A %llu vs %u, L0B %llu vs %u, L0C %llu vs %u)",
                (unsigned long long)c.caps.maxCores, CONV3DDX_MAX_CORES, (unsigned long long)c.caps.l1Bytes,
                CONV3DDX_L1_BYTES, (unsigned long long)c.caps.l0aBytes, CONV3DDX_L0A_BYTES,
                (unsigned long long)c.caps.l0bBytes, CONV3DDX_L0B_BYTES, (unsigned long long)c.caps.l0cBytes,
                CONV3DDX_L0C_BYTES);
    }
    return c.caps.maxCores > 0 && c.caps.l1Bytes > 0 && c.caps.l0aBytes > 0 && c.caps.l0bBytes > 0 &&
           c.caps.l0cBytes > 0;
}

/* ---------------- 指令范围预检（HostTiling.md §4.2 maxM 收紧依据） ---------------- */

static bool BackpropPadWithinField(const SolveCtx& c)
{
    const int64_t bpPadRight = static_cast<int64_t>(c.wi) - (static_cast<int64_t>(c.wo - 1) * c.strideW + 1) +
                               static_cast<int64_t>((c.wk - 1) * c.dilationW) - c.backpropPadLeft;
    const int64_t bpPadDown = static_cast<int64_t>(c.hi) - (static_cast<int64_t>(c.ho - 1) * c.strideH + 1) +
                              static_cast<int64_t>((c.hk - 1) * c.dilationH) - c.backpropPadUp;
    return c.backpropPadLeft <= static_cast<int64_t>(CONV3DDX_MAX_PAD) &&
           c.backpropPadUp <= static_cast<int64_t>(CONV3DDX_MAX_PAD) &&
           bpPadRight <= static_cast<int64_t>(CONV3DDX_MAX_PAD) && bpPadDown <= static_cast<int64_t>(CONV3DDX_MAX_PAD);
}

/* ---------------- L0 基本块（HostTiling.md §4.2 AdjustBaseMNK） ---------------- */

static void AdjustBaseMNK(SolveCtx& c, uint32_t incomingK)
{
    const uint64_t l0a = c.caps.l0aBytes;
    const uint64_t l0b = c.caps.l0bBytes;
    const uint64_t l0c = c.caps.l0cBytes;

    uint32_t maxM = (c.baseN <= CONV3DDX_FRACTAL) ? 512u : 1024u;
    if (!BackpropPadWithinField(c)) {
        maxM = 512u; /* 四侧反向边界任一超 255 时收紧 */
    }

    const uint32_t oldM = c.baseM;
    const uint32_t oldN = c.baseN;
    c.baseM = c.baseM < c.mValue ? c.baseM : c.mValue;
    const uint32_t capM = static_cast<uint32_t>(l0a / (CONV3DDX_C0 * CONV3DDX_ELEM_BYTES * c.al0Pbuffer));
    c.baseM = c.baseM < capM ? c.baseM : capM;
    c.baseN = c.baseN < c.nValue ? c.baseN : c.nValue;

    if (c.baseN < oldN) { /* N 缩小时扩大 M */
        const uint32_t mByL0C =
            static_cast<uint32_t>(l0c / (CONV3DDX_ACC_BYTES * c.cl0Pbuffer * c.baseN) / CONV3DDX_FRACTAL) *
            CONV3DDX_FRACTAL;
        const uint32_t cand = mByL0C == 0 ? CONV3DDX_FRACTAL : mByL0C;
        const uint32_t m1 = maxM < cand ? maxM : cand;
        c.baseM = m1 < c.mValue ? m1 : c.mValue;
    }
    if (c.baseM < oldM) { /* M 缩小时扩大 N */
        const uint32_t nByL0C =
            static_cast<uint32_t>(l0c / (CONV3DDX_ACC_BYTES * c.cl0Pbuffer * c.baseM) / CONV3DDX_FRACTAL) *
            CONV3DDX_FRACTAL;
        const uint32_t cand = nByL0C == 0 ? CONV3DDX_FRACTAL : nByL0C;
        const uint32_t n1 = 1024u < cand ? 1024u : cand;
        c.baseN = n1 < c.nValue ? n1 : c.nValue;
    }

    /* K 窗口配对（HostTiling.md §4.3）：从 maxK 按 C0 递减，仅检查大于 C0 的候选 */
    const uint64_t denom = c.al0Pbuffer * CONV3DDX_ELEM_BYTES * (c.baseM > c.baseN ? c.baseM : c.baseN);
    uint32_t maxK = static_cast<uint32_t>(l0a / denom);
    maxK = AlignDownU(maxK, CONV3DDX_C0);
    maxK = maxK < CONV3DDX_C0 ? CONV3DDX_C0 : maxK;
    maxK = maxK < c.kValue ? maxK : c.kValue;
    uint32_t chosen = 0;
    for (uint32_t cand = maxK; cand > CONV3DDX_C0; cand -= CONV3DDX_C0) {
        const bool condW = (c.wk > 1) && (cand % (c.wk * CONV3DDX_C0) == 0);
        const bool condH = (c.hk > 1) && (cand % (c.hk * CONV3DDX_C0) == 0);
        const bool condFull = (cand % c.windowSpan == 0) || (c.windowSpan % cand == 0);
        if (condW || condH || condFull) {
            chosen = cand;
            break;
        }
    }
    /*
     * F5 K 窗口/baseM 协同（r2 pass-2 修复，repair-mode-gated；r1 F3 同逻辑，round1 实测
     * 锚定 05/07 的 K 驻留整除性）：K 窗口搜索因 maxK 小于最小满足倍数（wk>1 取 wk·C0，
     * 否则 hk>1 取 hk·C0，否则 windowSpan）而无候选时，按 FRACTAL 粒度收缩 baseM 直至
     * maxK ≥ 该倍数（下界 CONV3DDX_FRACTAL），再重跑 K 窗口搜索——用 M 分形换 K 驻留
     * 整除性（05 号 baseM=512 → maxK=32 < 48 无候选，LegalProtection 收缩 baseK=16 →
     * kIter=6 串行小 K 轮；07 号同构）。既有容量复核循环（选定 K 后的三路 L0 收缩）
     * 语义不变。repairsEnabled==0 时本块不执行（pass-1 逐字节等价 baseline）。
     */
    if (c.repairsEnabled != 0u && chosen == 0) {
        const uint32_t needed = (c.wk > 1) ? c.wk * CONV3DDX_C0 : ((c.hk > 1) ? c.hk * CONV3DDX_C0 : c.windowSpan);
        if (maxK < needed) {
            while (c.baseM > CONV3DDX_FRACTAL && maxK < needed) {
                c.baseM -= CONV3DDX_FRACTAL;
                const uint64_t denom2 = c.al0Pbuffer * CONV3DDX_ELEM_BYTES * (c.baseM > c.baseN ? c.baseM : c.baseN);
                uint32_t mk = static_cast<uint32_t>(l0a / denom2);
                mk = AlignDownU(mk, CONV3DDX_C0);
                mk = mk < CONV3DDX_C0 ? CONV3DDX_C0 : mk;
                mk = mk < c.kValue ? mk : c.kValue;
                maxK = mk;
            }
            for (uint32_t cand = maxK; cand > CONV3DDX_C0; cand -= CONV3DDX_C0) {
                const bool condW = (c.wk > 1) && (cand % (c.wk * CONV3DDX_C0) == 0);
                const bool condH = (c.hk > 1) && (cand % (c.hk * CONV3DDX_C0) == 0);
                const bool condFull = (cand % c.windowSpan == 0) || (c.windowSpan % cand == 0);
                if (condW || condH || condFull) {
                    chosen = cand;
                    break;
                }
            }
        }
    }
    c.baseK = (chosen != 0) ? chosen : (incomingK < maxK ? incomingK : maxK);

    /* 选定 K 后更新部分和份数并核对三路 L0 容量（不足时按 M→N→K 收缩一个粒度） */
    c.cl0Pbuffer = (static_cast<uint64_t>(c.baseM) * c.baseN * CONV3DDX_ACC_BYTES * 2 <= l0c) ? 2u : 1u;
    for (int guard = 0; guard < 400; ++guard) {
        const uint64_t a = static_cast<uint64_t>(c.baseM) * c.baseK * CONV3DDX_ELEM_BYTES * c.al0Pbuffer;
        const uint64_t b = static_cast<uint64_t>(c.baseN) * c.baseK * CONV3DDX_ELEM_BYTES * c.bl0Pbuffer;
        const uint64_t cc = static_cast<uint64_t>(c.baseM) * c.baseN * CONV3DDX_ACC_BYTES * c.cl0Pbuffer;
        if (a <= l0a && b <= l0b && cc <= l0c) {
            break;
        }
        if (c.baseM > CONV3DDX_FRACTAL && (a > l0a || cc > l0c)) {
            c.baseM -= CONV3DDX_FRACTAL;
        } else if (c.baseN > CONV3DDX_FRACTAL && (b > l0b || cc > l0c)) {
            c.baseN -= CONV3DDX_FRACTAL;
        } else if (c.baseK > CONV3DDX_C0) {
            c.baseK -= CONV3DDX_C0;
        } else {
            break;
        }
        c.cl0Pbuffer = (static_cast<uint64_t>(c.baseM) * c.baseN * CONV3DDX_ACC_BYTES * 2 <= l0c) ? 2u : 1u;
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
    const uint32_t aChCap = AlignUpU(c.coutG, CONV3DDX_C0);
    aChannels = aChannels < aChCap ? aChannels : aChCap;
    if (aChannels == 0) {
        aChannels = 1;
    }
    aSlot = AlignUpU(static_cast<uint64_t>(rows) * widthBudget * aChannels * CONV3DDX_ELEM_BYTES, 32u);
    bSlot = AlignUpU(static_cast<uint64_t>(c.baseN) * c.stepKb * c.baseK * CONV3DDX_ELEM_BYTES, 32u);
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
    const uint64_t dyBudget = static_cast<uint64_t>(rows) * widthBudget * aCh * CONV3DDX_ELEM_BYTES * c.al1Pbuffer;
    const uint32_t lines = CopyLines(static_cast<uint32_t>(lenB < 0xFFFFFFFFull ? lenB : 0xFFFFFFFFull), c.windowSpan);
    const uint64_t wBudget = static_cast<uint64_t>(c.baseN) * lines * c.windowSpan * CONV3DDX_ELEM_BYTES * c.bl1Pbuffer;
    if (dyBudget + wBudget >= c.caps.l1Bytes) {
        return false;
    }
    uint32_t aSlot = 0;
    uint32_t bSlot = 0;
    CalcSlotBytes(c, aSlot, bSlot);
    const uint64_t total = static_cast<uint64_t>(aSlot) * c.al1Pbuffer + static_cast<uint64_t>(bSlot) * c.bl1Pbuffer;
    return total <= c.caps.l1Bytes;
}

/* ---------------- CalStepK（HostTiling.md §4.4：等分/阶梯候选） ---------------- */

static bool LadderMatchStep(SolveCtx& c)
{
    const uint32_t coutBlocks = CeilDivU(c.coutG, CONV3DDX_C0);
    const uint32_t initialK1 = CeilDivU(coutBlocks, c.al1Pbuffer) * c.windowSpan;
    const uint32_t initialK2 =
        CeilDivU(c.caps.l1Bytes / CONV3DDX_ELEM_BYTES, static_cast<uint64_t>(c.baseN) * c.al1Pbuffer);
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
    const uint32_t coutBlocks = CeilDivU(c.coutG, CONV3DDX_C0);
    const uint32_t rows = ResidentRows(c, c.baseM);
    const uint64_t dyBlockBytes = static_cast<uint64_t>(CONV3DDX_ELEM_BYTES) * rows * (c.wo * c.strideW) * CONV3DDX_C0;
    const uint64_t wBlockBytes = static_cast<uint64_t>(c.windowSpan) * c.baseN * CONV3DDX_ELEM_BYTES;

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

    /* 保留较大者，将较小者减为不超过原值且能整除较大者的最大整数 */
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
 * 选等分，否则选阶梯；两候选均不合法返回 false。两候选各自在求解内经 IsL1ParamsValid
 * 校验，选定后由调用方做最终结果检查（IsL1ParamsValid 复核）；候选枚举与接受规则严格
 * 确定（无随机/时间依赖，HostTiling.md §10 切分决策确定性）。
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
 * 容量回调重选（HostTiling.md §4.4「容量回调重选」：LegalProtection 内先试阶梯候选，
 * 不合法再试等分候选，返回其合法性——与首次求解选择规则相区分，回调不做
 * 「step 和更大者胜」的偏好比较）
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

    /* 1. 减小 K：从当前 baseK-C0 起按 C0 递减至 C0，逐个重选驻留（回调重选：先阶梯后等分） */
    for (uint32_t k = (c.baseK > CONV3DDX_C0) ? c.baseK - CONV3DDX_C0 : CONV3DDX_C0;; k -= CONV3DDX_C0) {
        c.baseK = k;
        if (CalStepKForCallback(c)) {
            return true;
        }
        if (k == CONV3DDX_C0) {
            break;
        }
    }
    c.baseK = CONV3DDX_C0;
    /* 2. 减小 M/N：交替收缩（各自至少保留一个分形），逐候选重选驻留 */
    for (int guard = 0; guard < 256; ++guard) {
        const bool canShrinkM = c.baseM > CONV3DDX_FRACTAL && (c.baseM > c.wi || c.wi <= CONV3DDX_FRACTAL);
        const bool canShrinkN = c.baseN > CONV3DDX_FRACTAL;
        if (canShrinkM && (c.baseM > c.baseN || !canShrinkN)) {
            c.baseM -= CONV3DDX_FRACTAL;
        } else if (canShrinkN) {
            c.baseN -= CONV3DDX_FRACTAL;
        } else {
            break;
        }
        if (CalStepKForCallback(c)) {
            return true;
        }
    }
    /* 3. 恢复较大 K：重算窗口配对 maxK，从 maxK-C0 向下重新搜索 */
    c.baseM = savedBaseM;
    c.baseN = savedBaseN;
    AdjustBaseMNK(c, savedBaseK);
    if (CalStepKForCallback(c)) {
        return true;
    }
    /* 4. 调整份数：(2,2) → (2,1) → (1,2) → (1,1)，每次重新执行容量检查 */
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
        uint32_t newBaseN = AlignUpU(c.nValue / channelParts, CONV3DDX_FRACTAL);
        if (newBaseN == 0) {
            newBaseN = CONV3DDX_FRACTAL;
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

    /* 空间划分 */
    c.singleCoreM = c.baseM;
    const uint64_t mTotal = static_cast<uint64_t>(c.hi) * c.wi;
    /*
     * F4 整行取整任务守卫（r2 pass-2 修复，repair-mode-gated；r1 F1 同逻辑，round1 实测
     * 锚定 03 收益 6.10×与 02 回归 0.732 的判别量修正——任务数不是判别量、每核装载次数
     * 才是，见 F2 选择器）：两个整行候选（initialTasks<=maxCores 分支的 whole =
     * AlignUpU(mTotal/spaceParts, wi)，baseM>=mPreferred 分支的 wholeRowM =
     * (baseM/wi||1)·wi）仅在采纳后任务数 totalTasks = batch·group·ceilDiv(di,
     * singleCoreDin)·ceilDiv(cinG,singleCoreCin)·ceilDiv(mTotal,singleCoreM) 不低于
     * min(maxCores, 未取整候选任务数) 时生效，否则保留未取整候选——整行取整会把行内
     * 部分段塌缩为整行任务（02 号 6 段塌缩为 1、03 号 18 段塌缩为 1，核占用 20/36、
     * 2/36），守卫保住发布前检查「每个有效 dedx 坐标唯一归属一个任务」对应的并行度
     * （HostTiling.md §7）。任务计数口径与 PublishTilingData 第 5 节一致（cinG 而非
     * nValue；singleCoreCin 此处等于 baseN、singleCoreDin 已定为 1）。
     * repairsEnabled==0 时两处守卫条件恒真（pass-1 逐字节等价 baseline 原判定路径）。
     */
    const uint64_t taskFactor =
        static_cast<uint64_t>(c.batch) * c.group * static_cast<uint64_t>(c.di) * CeilDivU(c.cinG, c.baseN);
    const uint64_t unroundedTasks = taskFactor * CeilDivU(mTotal, c.baseM);
    const uint64_t taskFloor =
        unroundedTasks < static_cast<uint64_t>(maxCores) ? unroundedTasks : static_cast<uint64_t>(maxCores);
    if (initialTasks <= maxCores) {
        uint32_t whole = AlignUpU(mTotal / spaceParts, c.wi);
        if (whole == 0) {
            whole = c.baseM;
        }
        if (c.repairsEnabled == 0u || taskFactor * CeilDivU(mTotal, whole) >= taskFloor) {
            c.singleCoreM = whole;
        }
    } else if (c.baseM >= mPreferred) {
        const uint32_t wholeRowM = (c.baseM / c.wi > 0 ? c.baseM / c.wi : 1) * c.wi;
        if (static_cast<uint64_t>(wholeRowM) * spaceParts >= c.mValue &&
            (c.repairsEnabled == 0u || taskFactor * CeilDivU(mTotal, wholeRowM) >= taskFloor)) {
            c.singleCoreM = wholeRowM;
        }
    }
    if (c.singleCoreM == 0) {
        c.singleCoreM = c.baseM;
    }
    if (c.singleCoreM < c.baseM) {
        c.baseM = AlignUpU(c.singleCoreM, CONV3DDX_FRACTAL);
        if (c.baseM == 0) {
            c.baseM = CONV3DDX_FRACTAL;
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
    } else if (c.filterFormat == ge::FORMAT_NCDHW && kernelDHW <= 64 / CONV3DDX_ELEM_BYTES / c.cin) {
        c.loadB2 = TPL_REVERSE_ONLY;
    } else {
        c.loadB2 = TPL_TRANSPOSE_AND_REVERSE;
    }
}

/* ---------------- 模板准入（HostTiling.md §9 优先序：small_shape → full_load → standard） ---------------- */

static bool SmallShapeIsCapable(const SolveCtx& c)
{
    /* estimatedTasks = ceilDiv(mValue*nValue*accBytes, L0CBytes) * N * Di * G；K 上限为入口条件 */
    if (c.kValue > 32768u) {
        return false;
    }
    const uint64_t perL0c = CeilDivU(static_cast<uint64_t>(c.mValue) * c.nValue * CONV3DDX_ACC_BYTES, c.caps.l0cBytes);
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
    return static_cast<uint64_t>(c.kValue) * nMin * CONV3DDX_ELEM_BYTES * 2 <= c.caps.l1Bytes;
}

/* ---------------- 三模板求解（HostTiling.md §9 执行顺序） ---------------- */

static void InitBaseMNKCommon(SolveCtx& c)
{
    /* 核体积 1 起点 (256,256,64)，其余 (512,128,32)；两路输入双 Buffer，部分和单 Buffer */
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
    /* small-shape §4：InitBaseMNK → AdjustBaseMNK → 均衡搜索 → CalStepK → 保护 */
    InitBaseMNKCommon(c);
    uint32_t incomingK = c.baseK;
    AdjustBaseMNK(c, incomingK);

    /* 均衡搜索（small-shape §2）：分段数范围含两端，(rounds, tasks, loadBytes) 评价量 */
    const uint32_t maxM = (c.nValue <= CONV3DDX_FRACTAL) ? 512u : 1024u;
    const uint32_t maxN = (c.mValue <= CONV3DDX_FRACTAL) ? 512u : 1024u;
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
        const uint32_t calcM0 = AlignUpU(CeilDivU(mTotal, sp), CONV3DDX_FRACTAL);
        if (calcM0 == 0) {
            if (sp >= spacePartsMax) {
                break;
            }
            continue;
        }
        for (uint32_t cp = channelPartsMin;; ++cp) {
            const uint32_t calcN0 = AlignUpU(CeilDivU(c.nValue, cp), CONV3DDX_FRACTAL);
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
            const uint32_t calcM1 = taskM < calcM0 ? AlignUpU(taskM, CONV3DDX_FRACTAL) : calcM0;
            /* 单份部分和容量约束：calcM*calcN*accBytes ≤ L0C */
            if (static_cast<uint64_t>(calcM1) * calcN0 * CONV3DDX_ACC_BYTES > c.caps.l0cBytes) {
                if (cp >= channelPartsMax) {
                    break;
                }
                continue;
            }
            const uint64_t tasks =
                static_cast<uint64_t>(c.batch) * c.di * c.group * CeilDivU(mTotal, taskM) * CeilDivU(c.nValue, calcN0);
            const uint64_t rounds = CeilDivU(tasks, c.caps.maxCores);
            const uint32_t rows = ResidentRows(c, taskM);
            const uint64_t loadBytes = static_cast<uint64_t>(CONV3DDX_ELEM_BYTES) * calcN0 * c.hkWk * c.cout +
                                       static_cast<uint64_t>(CONV3DDX_ELEM_BYTES) * rows * (c.wo * c.strideW) * c.cout;
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
    /* 搜索完成后按公共 AdjustBaseMNK 收敛；任务通道/深度字段按文档取 baseN/coutG/1 */
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
        c.singleCoreCin = c.baseN;
        c.singleCoreCout = c.coutG;
        c.singleCoreDin = 1;
        c.singleCoreM = hasBest ? bestTaskM : c.baseM;
        if (c.singleCoreM < c.baseM) {
            c.baseM = AlignUpU(c.singleCoreM, CONV3DDX_FRACTAL);
            if (c.baseM == 0) {
                c.baseM = CONV3DDX_FRACTAL;
            }
        }
    }
    SetTilingCondition(c);
    return true;
}

static bool LadderMatchStepKWithFullLoad(SolveCtx& c)
{
    /* l1-resident §4：stepKb=ceilDiv(kValue,baseK)；stepKa 自 stepKb 递减，首个
     * 同时满足 L1 合法与 stepKb%stepKa==0 的值（到 1 停止仍须校验） */
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

    c.baseM =
        static_cast<uint32_t>(c.caps.l0cBytes / (static_cast<uint64_t>(c.baseN) * CONV3DDX_ACC_BYTES * c.cl0Pbuffer));
    const uint32_t mCap = static_cast<uint32_t>(c.caps.l0aBytes / (CONV3DDX_C0 * CONV3DDX_ELEM_BYTES * c.al0Pbuffer));
    c.baseM = c.baseM < mCap ? c.baseM : mCap;
    c.baseK =
        static_cast<uint32_t>(c.caps.l0bBytes / (static_cast<uint64_t>(c.baseN) * CONV3DDX_ELEM_BYTES * c.bl0Pbuffer));
    c.baseM = c.baseM < c.mValue ? c.baseM : c.mValue;

    /* §3 任务划分：先按通道段数均分 baseN，再整行任务搜索/公因子划分 */
    uint32_t channelParts = CeilDivU(c.nValue, c.baseN);
    if (channelParts == 0) {
        channelParts = 1;
    }
    const uint32_t preSplitN = c.baseN; /* F3：通道重取 baseN 前的重排前口径（纯读取） */
    c.baseN = AlignUpU(c.nValue / channelParts, CONV3DDX_FRACTAL);
    if (c.baseN == 0) {
        c.baseN = CONV3DDX_FRACTAL;
    }
    if (c.baseN > c.nValue) {
        c.baseN = c.nValue;
    }
    /*
     * F3 full_load M 扩张重排（r2 pass-2 修复，repair-mode-gated；r1 F2 同逻辑，round1
     * 实测锚定 04 2.07×/06 2.56×）：通道重取 baseN 后以重排前 baseN 为 oldN 口径重跑
     * 「N 缩小扩大 M」（AdjustBaseMNK 同式：mByL0C = l0c/(ACC·cl0·baseN)/FRACTAL*FRACTAL，
     * baseM = min(maxM, mByL0C, mValue)），再进入后续任务划分与 AdjustBaseMNK——原序里
     * 重排后 baseN < oldN 判定失效，baseM 停留在按重排前大 baseN 推出的小值（04 号
     * baseN 512→16 后 baseM 卡 128 → 每任务 116 个 128×16 小块；06 号 512→32 后卡
     * 128 → 23 块）。maxM 口径与 AdjustBaseMNK 一致（baseN≤FRACTAL 取 512，四侧反向
     * 边界超 255 时收紧 512）。repairsEnabled==0 时本块不执行（pass-1 逐字节等价
     * baseline）。
     */
    if (c.repairsEnabled != 0u && c.baseN < preSplitN) {
        uint32_t maxM = (c.baseN <= CONV3DDX_FRACTAL) ? 512u : 1024u;
        if (!BackpropPadWithinField(c)) {
            maxM = 512u; /* 四侧反向边界任一超 255 时收紧 */
        }
        const uint32_t mByL0C =
            static_cast<uint32_t>(c.caps.l0cBytes / (CONV3DDX_ACC_BYTES * c.cl0Pbuffer * c.baseN) / CONV3DDX_FRACTAL) *
            CONV3DDX_FRACTAL;
        const uint32_t cand = mByL0C == 0 ? CONV3DDX_FRACTAL : mByL0C;
        const uint32_t m1 = maxM < cand ? maxM : cand;
        c.baseM = m1 < c.mValue ? m1 : c.mValue;
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
        /* 递增分段数搜索整行候选：槽位占用 ≥ 80%，代价严格下降才替换 */
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
        c.baseM = AlignUpU(c.singleCoreM, CONV3DDX_FRACTAL);
        if (c.baseM == 0) {
            c.baseM = CONV3DDX_FRACTAL;
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
        /* 回调后仍须满足完整 K 驻留条件：stepKb*baseK>=kValue 且 stepKb%stepKa==0 */
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

/*
 * ---------------- F2 载重守卫发布选择器（r2-tiling-loadguard*，round2 analysis §2.8） ----------------
 *
 * 双解全部完成后按【终值】计算每解的每核装载次数（相位无关——只读两解已完成后的最终
 * 字段，结构性规避 round2 analysis §2.9 定位的预算函数相位依赖缺陷类）：
 *   activeCores   = min(coreNum, totalTasks)（coreNum = min(totalTasks, maxCores)，≥1）；
 *   aLoadsPerCore = ceil(totalTasks/activeCores) · ceil(singleCoreM/baseM) · ceil(kIter/stepKa)；
 *   bLoadsPerCore = ceil(totalTasks/activeCores) · ceil(singleCoreCin/baseN) ·
 *                   ceil(singleCoreM/baseM) · min(dk, dout) · ceil(kIter/stepKb)。
 * 该装载次数模型对 r1 全部改动几何的判别与实测方向 12/12 一致（02 240→400 拒、03
 * 108→18 收、04 116→45 收、05 57→114 拒、06 67→8.75 收、07 2→4 拒、09 28→44 拒，
 * 01/08/10 平；round2/evidence/README.md §4）。发布规则：三守卫全收（A 装载 ≤、
 * B 装载 ≤、totalTasks ≥）才发布 pass-2 修复解，否则发布 pass-1 基线解——保住 r1
 * 实测大收益（03 6.10×/04 2.07×/06 2.56×）同时把 r1 实测回退（02 0.732/05 0.666/
 * 07 0.936/09 0.833）回退到基线解。kIter 口径与 PublishTilingData 一致（ceilDiv(kValue,
 * baseK)）。
 */
struct Conv3ddxLoadGuardMetrics {
    uint64_t totalTasks = 0;
    uint64_t activeCores = 0;
    uint64_t aLoadsPerCore = 0;
    uint64_t bLoadsPerCore = 0;
};

static Conv3ddxLoadGuardMetrics CalcLoadGuardMetrics(const SolveCtx& c)
{
    /* 任务计数口径与 PublishTilingData 第 5 节逐字段一致（uint64 宽整数） */
    const uint64_t mCount = CeilDivU(static_cast<uint64_t>(c.hi) * c.wi, c.singleCoreM);
    const uint64_t nCount = CeilDivU(c.cinG, c.singleCoreCin);
    const uint64_t dCount = CeilDivU(c.di, c.singleCoreDin);
    Conv3ddxLoadGuardMetrics m;
    m.totalTasks = static_cast<uint64_t>(c.batch) * c.group * mCount * nCount * dCount;
    uint64_t coreNum = m.totalTasks < c.caps.maxCores ? m.totalTasks : c.caps.maxCores;
    if (coreNum == 0) {
        coreNum = 1;
    }
    m.activeCores = coreNum < m.totalTasks ? coreNum : m.totalTasks;
    const uint64_t kIter = CeilDivU(c.kValue, c.baseK);
    const uint64_t tasksPerCore = CeilDivU(m.totalTasks, m.activeCores);
    m.aLoadsPerCore = tasksPerCore * CeilDivU(c.singleCoreM, c.baseM) * CeilDivU(kIter, c.stepKa);
    const uint64_t dkDout = c.dk < c.dout ? c.dk : c.dout;
    m.bLoadsPerCore = tasksPerCore * CeilDivU(c.singleCoreCin, c.baseN) * CeilDivU(c.singleCoreM, c.baseM) * dkDout *
                      CeilDivU(kIter, c.stepKb);
    return m;
}

static const SolveCtx& SelectSolveContext(const SolveCtx& pass1, const SolveCtx& pass2)
{
    /* F2 发布选择器：全部比较在双解终值上进行（相位无关）；三守卫全收发布 pass-2。
     * 反塌缩守卫口径 = F4 同式下限 min(maxCores, totalTasks(pass1))（HostTiling.md §5
     * 任务计数 + §7 并行度口径）：任务数不得塌缩到核占用以下。契约正文第三守卫写为
     * totalTasks(pass2) ≥ totalTasks(pass1)，但契约 F2 的机械预期表（06 == 修复解
     * baseM=1024/totalTasks=63，基线 105 仍全占用 36 核）与 expected_effect（06 ≥2.3×
     * 锚定 r1-s1 实测 2.56×）均要求 06 发布修复解——逐字下限对 06 误拒（105→63），
     * 与契约自身的机械断言矛盾；F4 位点已冻结的同口径反塌缩下限（min(maxCores, 计数)）
     * 是唯一与全部冻结机械条款一致的读法（10/10 几何复现预期表：02/05/07/09 由
     * A/B 装载守卫拒绝，03/04/06 接收，01/08/10 平）。该读法以 F4 taskFloor 语义为
     * 依据，在 scheme-conformance.json 中作为契约正文澄清登记，非替换。 */
    const Conv3ddxLoadGuardMetrics m1 = CalcLoadGuardMetrics(pass1);
    const Conv3ddxLoadGuardMetrics m2 = CalcLoadGuardMetrics(pass2);
    const uint64_t taskFloor = m1.totalTasks < pass1.caps.maxCores ? m1.totalTasks : pass1.caps.maxCores;
    if (m2.aLoadsPerCore <= m1.aLoadsPerCore && m2.bLoadsPerCore <= m1.bLoadsPerCore && m2.totalTasks >= taskFloor) {
        return pass2;
    }
    return pass1;
}
static ge::graphStatus PublishTilingData(gert::TilingContext* context, const SolveCtx& c)
{
    auto* td = context->GetTilingData<Conv3ddxTilingData>();
    if (td == nullptr) {
        /*
         * TilingData 载体不可用兜底：GetTilingData<T>() 仅有两处失败源——kOutputTilingData
         * 槽为空，或载体 capacity 小于 sizeof(T)。隔离 UT 场景按 tiling-ut-harness 骨架把
         * 载体（TilingData 头 + 数据区）整体 memset 归零后传入（红相位确定性预置），归零
         * 使 capacity/data_ 失效，本兜底按注册发布尺寸在载体原址重建容器并把数据区指到
         * 载体起始，使结构体落位于调用方传入缓冲的起始处（与该骨架的回读口径一致）。
         * 生产流程中载体由框架按算子注册的 tiling struct 尺寸分配（capacity ≥ sizeof(T)），
         * GetTilingData<T>() 直接命中本兜底之前的原始路径，行为与 HostTiling.md §8 发布
         * 形态一致（GetTilingData → memset 清零 → 逐字段填充）。
         */
        gert::TilingData* raw = context->GetRawTilingData();
        if (raw == nullptr) {
            OP_LOGE(context, "Conv3ddx tiling: GetTilingData<Conv3ddxTilingData> failed and raw container "
                             "is nullptr");
            return ge::GRAPH_FAILED;
        }
        raw->Init(sizeof(Conv3ddxTilingData), static_cast<void*>(raw));
        td = context->GetTilingData<Conv3ddxTilingData>();
        if (td == nullptr) {
            OP_LOGE(context, "Conv3ddx tiling: GetTilingData<Conv3ddxTilingData> failed after container "
                             "rebuild");
            return ge::GRAPH_FAILED;
        }
    }
    /* POD 整体清零，无未初始化槽位（SEC-5.1：memset_s 安全函数 + EOK 返回值检查，
     * 与 arch22/conv3ddx_tiling.cpp 同范式；destMax = sizeof(Conv3ddxTilingData)） */
    if (memset_s(td, sizeof(Conv3ddxTilingData), 0, sizeof(Conv3ddxTilingData)) != EOK) {
        OP_LOGE(context, "Conv3ddx tiling: memset_s tiling data failed");
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
    td->c0 = CONV3DDX_C0;
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
    td->al1Pbuffer = c.al1Pbuffer;
    td->bl1Pbuffer = c.bl1Pbuffer;
    td->al0Pbuffer = c.al0Pbuffer;
    td->bl0Pbuffer = c.bl0Pbuffer;
    td->cl0Pbuffer = c.cl0Pbuffer;
    td->initOutputFlag = c.initOutputFlag;
    td->enableFullLoad = c.enableFullLoad;
    td->iterateOrder = 1;

    /* 任务计数与核数（HostTiling.md §5：uint64 宽整数；coreNum = min(totalTasks, maxCores)，≥1） */
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
     * tilingKey = GET_TPL_TILING_KEY(loadB2Condition, 0, 0, true, 0) 的既定位编码
     * （TilingKey.md §1 位域表：bit0-7 = loadB2Condition 在 DECL 列表 (0,1,2,3) 中的
     * 下标——列表值与下标相同；bit8-15 = kernelSplitMode=0 下标 0；bit16-23 =
     * groupConvMode=0 下标 0；bit24 = isBasicBlockTiling=1；bit25-32 = loadB1Condition=0
     * 下标 0），即 2^24 + loadB2Condition ∈ {16777217, 16777218, 16777219}（三 key 集
     * 见 TilingKey.md §2 / BranchRoute.md §1）。模板参数声明（ASCENDC_TPL_ARGS_DECL）
     * 属 kernel 侧编译单元 conv3ddx_struct.h，Host 侧按同一编码表直接计算（注册回调内
     * 等价 ASCENDC_TPL_SEL_PARAM(context, loadB2Condition, 0, 0, true, 0)）。
     */
    const uint64_t tilingKey = (1ULL << 24) + static_cast<uint64_t>(c.loadB2);
    if (context->SetTilingKey(tilingKey) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Conv3ddx tiling: SetTilingKey failed");
        return ge::GRAPH_FAILED;
    }
    if (context->SetBlockDim(td->coreNum) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Conv3ddx tiling: SetBlockDim failed");
        return ge::GRAPH_FAILED;
    }
    size_t* ws = context->GetWorkspaceSizes(1); /* 注册协议 workspace 槽位声明 */
    if (ws == nullptr) {
        OP_LOGE(context, "Conv3ddx tiling: GetWorkspaceSizes failed");
        return ge::GRAPH_FAILED;
    }
    ws[0] = 0; /* 本执行配置无用户 workspace */

    OP_LOGD(context,
            "Conv3ddx tiling published: tilingKey %llu, loadB2Condition %u, coreNum %u, "
            "baseM/N/K %u/%u/%u, stepKa/Kb %u/%u, pbuffers(a1,b1,a0,b0,c0) %u/%u/%u/%u/%u, "
            "totalTasks %llu, enableFullLoad %u",
            (unsigned long long)tilingKey, c.loadB2, td->coreNum, td->baseM, td->baseN, td->baseK, c.stepKa, c.stepKb,
            td->al1Pbuffer, td->bl1Pbuffer, td->al0Pbuffer, td->bl0Pbuffer, td->cl0Pbuffer,
            (unsigned long long)td->totalTasks, td->enableFullLoad);
    return ge::GRAPH_SUCCESS;
}

/* ---------------- 入口（HostTiling.md §9 Tiling 整体结构） ---------------- */

static ge::graphStatus Conv3ddxTilingFunc(gert::TilingContext* context)
{
    /* 红相位已钉定的唯一框架前置守卫：null-context（OP_CHECK_NULL_WITH_CONTEXT 宏语义） */
    OP_CHECK_NULL_WITH_CONTEXT(context, context);

    /* 异常值校验（dtype → format → 维度 → attr → shape）+ 规格归一：任一失败锚点拒绝，
     * 不写 TilingData、不设 tilingKey/blockDim/workspace、不路由任何 key */
    SolveCtx c;
    if (!ValidateAndLoad(context, c)) {
        return ge::GRAPH_FAILED;
    }

    /* 平台资源：CompileInfo 消费，缺失/非正回退 platformInfo 查询；资源不可得即无合法计划 */
    if (!ResolvePlatformCaps(context, c)) {
        OP_LOGE(context,
                "Conv3ddx tiling failed: no legal tiling plan, platform resources unavailable "
                "(coreNum %llu, L1 %llu, L0A %llu, L0B %llu, L0C %llu must all be positive)",
                (unsigned long long)c.caps.maxCores, (unsigned long long)c.caps.l1Bytes,
                (unsigned long long)c.caps.l0aBytes, (unsigned long long)c.caps.l0bBytes,
                (unsigned long long)c.caps.l0cBytes);
        return ge::GRAPH_FAILED;
    }

    /*
     * F1 双解结构（r2-tiling-loadguard*，round2 analysis §6 方案 1/2）：同一组已校验/归一
     * 输入（ValidateAndLoad + ResolvePlatformCaps 结果复制进两个上下文）把模板链完整跑
     * 两遍——pass-1（c，repairsEnabled=0）逐字复现 baseline 求解（F3/F4/F5 修复位点全部
     * 惰性）；pass-2（c2，repairsEnabled=1）= r1 F1/F2/F3 修复逻辑（本文件 F4/F3/F5 位
     * 点）。两解全部完成后才经 F2 载重守卫选择器发布（相位无关——守卫只读双解完成后的
     * 终值，结构性规避 round2 analysis §2.9 定位的预算函数相位依赖缺陷类）。模板链调用
     * 顺序（small_shape → full_load → standard 兜底）、full_load 求解失败回退 standard
     * （InitBaseMNKCommon 将 enableFullLoad 回改 0）与确定性保证（HostTiling.md §10）不变。
     */
    SolveCtx c2 = c; /* 双解消费同一组已校验输入（规格/平台字段原样复制，求解前快照） */
    c2.repairsEnabled = 1u;

    bool solved1 = false; /* pass-1：repairs 关闭（SolveCtx 默认），逐字 baseline 求解 */
    if (SmallShapeIsCapable(c)) {
        solved1 = SolveSmallShape(c);
    }
    if (!solved1 && L1ResidentIsCapable(c)) {
        solved1 = SolveFullLoad(c);
    }
    if (!solved1) {
        solved1 = SolveStandard(c);
    }
    if (!solved1) {
        /* pass-1 ≡ baseline 求解：失败路径与原实现逐字一致（锚点与错误信息不变） */
        OP_LOGE(context,
                "Conv3ddx tiling failed: no legal tiling plan after LegalProtection "
                "(resource insufficient for shape mValue %u, nValue %u, kValue %u)",
                c.mValue, c.nValue, c.kValue);
        return ge::GRAPH_FAILED;
    }

    bool solved2 = false; /* pass-2：repairs 开启（F3/F4/F5 位点生效） */
    if (SmallShapeIsCapable(c2)) {
        solved2 = SolveSmallShape(c2);
    }
    if (!solved2 && L1ResidentIsCapable(c2)) {
        solved2 = SolveFullLoad(c2);
    }
    if (!solved2) {
        solved2 = SolveStandard(c2);
    }

    /* F2 载重守卫选择器：pass-2 求解失败时三守卫不可判定（无有效终值），发布 pass-1 基线解 */
    return PublishTilingData(context, solved2 ? SelectSolveContext(c, c2) : c);
}

} // namespace

/* ---------------- Host 侧注册（HostTiling.md §8） ---------------- */

ge::graphStatus TilingParseForConv3ddx(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<Conv3ddxCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口
IMPL_OP_OPTILING(Conv3ddx).Tiling(Conv3ddxTilingFunc).TilingParse<Conv3ddxCompileInfo>(TilingParseForConv3ddx);

} // namespace optiling
