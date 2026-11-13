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
 * \file conv3ddx_infershape.cpp
 * \brief Conv3ddx 算子形状推导实现（GEIR OpType Conv3ddx）
 *
 * 事实源：design/InferShapeDtype.md「InferShape」节（spec.outputs.y 为 data_dependent_shape）。
 * 核心语义：
 *   - y.shape = input_size 的元素值（1D int32/int64、长度 5；值序随 data_format），
 *     输出对输入 0 值依赖（InputsDataDependency({0})，REQUIREMENTS §5.4）；
 *   - input_size 元素值不可见（动态图占位）或 filter 形状未知时，输出 5 维全部置 -1
 *     （与内置基线 Conv3DBackpropInputV2 同口径）；
 *   - unknown rank（[-2]）/ unknown shape（含 -1）场景输出全 -1，rank 固定 5；
 *   - 静态场景逐维做 batch/通道/正向几何一致性校验（任一不满足报 shape_mismatch、
 *     null_input 或 attribute_value_out_of_range，判据见 design/InferShapeDtype.md
 *     「逐维对齐与几何一致性校验」「边界与失败判据」）；
 *   - filter 布局取 desc origin format：三种注册布局（NCDHW/NDHWC/DHWCN）直接解析；
 *     origin format 为 ND（布局未声明，如 GE/TTK 以 ND desc 馈入 DHWCN 权重）时按三种
 *     注册布局做唯一一致性消歧（通道对齐 + 前向几何逐轴一致），候选必须恰好一个——
 *     零个或多个均按 shape_mismatch 响亮拒绝，绝不静默猜测布局。
 */

#include <cstdint>
#include <string>

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {
namespace {
// 输入/输出序（与 conv3ddx_proto.h REG_OP / conv3ddx_def.cpp OpDef 声明一致）
constexpr size_t kInputSizeIdx = 0U;   // 输入 0：input_size（整型形状描述符，值依赖）
constexpr size_t kFilterIdx = 1U;      // 输入 1：filter（权重）
constexpr size_t kOutBackpropIdx = 2U; // 输入 2：out_backprop（输出梯度 dedy）
constexpr size_t kOutputIdx = 0U;      // 输出 0：y（输入梯度 dedx）
constexpr size_t kRank5 = 5U;          // filter/out_backprop/y 固定 5D
constexpr size_t kInputSizeLen = 5U;   // input_size 1D 长度 5（接口契约，REQUIREMENTS §8.2.1）
constexpr int64_t kUnknownDim = -1;

// 属性声明序（OpDef：strides/pads/dilations/groups/data_format/enable_hf32）
constexpr size_t kAttrStrides = 0U;
constexpr size_t kAttrPads = 1U;
constexpr size_t kAttrDilations = 2U;
constexpr size_t kAttrGroups = 3U;
constexpr size_t kAttrDataFormat = 4U;
constexpr size_t kStridesLen = 5U;
constexpr size_t kPadsLen = 6U;
constexpr size_t kDilationsLen = 5U;

// 参与几何运算的外部值可表示域上界（SEC-1.3：先收窄再运算）：strides/dilations/
// pads/groups/形状维均限制在 [0/1, 2^31-1]（与 tiling 层 CONV3DDX_MAX_DIM 同口径），
// 使 kernelExtent = dilation*(kernel-1)+1、numerator = inSize+pads-kernelExtent、
// cinPerGroup*groups 等有符号中间量恒 < 2^62，消除有符号整数溢出 UB。
constexpr int64_t kMaxDimValue = 2147483647;

// -1（未知维）与 -2（unknown rank）均为负值，统一按未知处理
bool HasUnknownDim(const gert::Shape& shape)
{
    for (size_t idx = 0; idx < shape.GetDimNum(); ++idx) {
        if (shape.GetDim(idx) < 0) {
            return true;
        }
    }
    return false;
}

// 输出置为全未知（rank 固定 5，注册期只能确定 rank 与 format 族）
void SetUnknownShape(gert::Shape& shape)
{
    shape.SetDimNum(kRank5);
    for (size_t idx = 0; idx < kRank5; ++idx) {
        shape.SetDim(idx, kUnknownDim);
    }
}

// input_size 常量数据是否 Host 可见（ValuesDataDependency 声明后由运行时馈入）
bool IsConstDataVisible(const gert::Tensor& tensor)
{
    return tensor.GetAddr() != nullptr;
}

// filter 语义维（规范序 Cout/CinG/kD/kH/kW）按 desc 的 origin format 解析；
// filter 三种注册布局独立于 data_format（spec shape_constraints.global_constraints）
struct FilterSemanticDims {
    int64_t cout;
    int64_t cinPerGroup;
    int64_t kD;
    int64_t kH;
    int64_t kW;
};

bool GetFilterSemanticDims(const gert::Shape& filterShape, const ge::Format filterFormat, FilterSemanticDims& dims)
{
    if (filterShape.GetDimNum() != kRank5) {
        return false;
    }
    switch (filterFormat) {
        case ge::FORMAT_NCDHW: // (Cout, Cin/groups, kD, kH, kW)
            dims.cout = filterShape.GetDim(0);
            dims.cinPerGroup = filterShape.GetDim(1);
            dims.kD = filterShape.GetDim(2);
            dims.kH = filterShape.GetDim(3);
            dims.kW = filterShape.GetDim(4);
            return true;
        case ge::FORMAT_NDHWC: // (Cout, kD, kH, kW, Cin/groups)
            dims.cout = filterShape.GetDim(0);
            dims.kD = filterShape.GetDim(1);
            dims.kH = filterShape.GetDim(2);
            dims.kW = filterShape.GetDim(3);
            dims.cinPerGroup = filterShape.GetDim(4);
            return true;
        case ge::FORMAT_DHWCN: // (kD, kH, kW, Cin/groups, Cout)
            dims.kD = filterShape.GetDim(0);
            dims.kH = filterShape.GetDim(1);
            dims.kW = filterShape.GetDim(2);
            dims.cinPerGroup = filterShape.GetDim(3);
            dims.cout = filterShape.GetDim(4);
            return true;
        default:
            return false;
    }
}

// 前向卷积几何：Dout/Hout/Wout 同构公式；无有效输出时返回负值
int64_t ForwardOutSize(const int64_t inSize, const int64_t padFront, const int64_t padBack, const int64_t kernel,
                       const int64_t dilation, const int64_t stride)
{
    const int64_t kernelExtent = dilation * (kernel - 1) + 1;
    const int64_t numerator = inSize + padFront + padBack - kernelExtent;
    if (numerator < 0) {
        return -1;
    }
    return numerator / stride + 1;
}

// 三种注册布局的候选语义维（枚举序 NCDHW / NDHWC / DHWCN；调用前提 rank=5）
void MakeFilterCandidates(const gert::Shape& filterShape, FilterSemanticDims (&cands)[3])
{
    // NCDHW (Cout, CinG, kD, kH, kW)
    cands[0] = {filterShape.GetDim(0), filterShape.GetDim(1), filterShape.GetDim(2), filterShape.GetDim(3),
                filterShape.GetDim(4)};
    // NDHWC (Cout, kD, kH, kW, CinG)
    cands[1] = {filterShape.GetDim(0), filterShape.GetDim(4), filterShape.GetDim(1), filterShape.GetDim(2),
                filterShape.GetDim(3)};
    // DHWCN (kD, kH, kW, CinG, Cout)
    cands[2] = {filterShape.GetDim(4), filterShape.GetDim(3), filterShape.GetDim(0), filterShape.GetDim(1),
                filterShape.GetDim(2)};
}

// ND（布局未声明）filter 消歧判据：候选布局下通道对齐 + 前向几何逐轴全部一致
// （静默判定，不落日志——消歧结论由调用方统一报告）
struct CandidateCtx {
    int64_t cin;
    int64_t groups;
    int64_t dedyChannel;
    int64_t dedyD;
    int64_t dedyH;
    int64_t dedyW;
    int64_t dIn;
    int64_t hIn;
    int64_t wIn;
    const int64_t* strides;
    const int64_t* dilations;
    const int64_t* pads;
    size_t spatialBase;
};

bool FilterCandidateConsistent(const FilterSemanticDims& cand, const CandidateCtx& ctx)
{
    if (ctx.cin % ctx.groups != 0 || cand.cinPerGroup * ctx.groups != ctx.cin) {
        return false;
    }
    if (cand.cout % ctx.groups != 0) {
        return false;
    }
    if (ctx.dedyChannel != cand.cout) {
        return false;
    }
    const int64_t expectedD = ForwardOutSize(ctx.dIn, ctx.pads[0], ctx.pads[1], cand.kD, ctx.dilations[ctx.spatialBase],
                                             ctx.strides[ctx.spatialBase]);
    const int64_t expectedH = ForwardOutSize(ctx.hIn, ctx.pads[2], ctx.pads[3], cand.kH,
                                             ctx.dilations[ctx.spatialBase + 1U], ctx.strides[ctx.spatialBase + 1U]);
    const int64_t expectedW = ForwardOutSize(ctx.wIn, ctx.pads[4], ctx.pads[5], cand.kW,
                                             ctx.dilations[ctx.spatialBase + 2U], ctx.strides[ctx.spatialBase + 2U]);
    if (expectedD < 1 || expectedD != ctx.dedyD) {
        return false;
    }
    if (expectedH < 1 || expectedH != ctx.dedyH) {
        return false;
    }
    if (expectedW < 1 || expectedW != ctx.dedyW) {
        return false;
    }
    return true;
}

// 从 input_size 元素值读取 5 个维值（int32/int64 双分支；其余 dtype 返回失败）
bool ReadInputSizeValues(const gert::Tensor& inputSizeTensor, int64_t (&values)[kInputSizeLen])
{
    const ge::DataType dtype = inputSizeTensor.GetDataType();
    if (dtype == ge::DT_INT32) {
        const int32_t* data = inputSizeTensor.GetData<int32_t>();
        if (data == nullptr) {
            return false;
        }
        for (size_t idx = 0; idx < kInputSizeLen; ++idx) {
            values[idx] = static_cast<int64_t>(data[idx]);
        }
        return true;
    }
    if (dtype == ge::DT_INT64) {
        const int64_t* data = inputSizeTensor.GetData<int64_t>();
        if (data == nullptr) {
            return false;
        }
        for (size_t idx = 0; idx < kInputSizeLen; ++idx) {
            values[idx] = data[idx];
        }
        return true;
    }
    return false;
}

// 静态形状与属性的一致性校验（design/InferShapeDtype.md「逐维对齐与几何一致性校验」）。
// 调用前提：input_size 维值已读取（yShape 已按其赋值），filter/out_backprop 形状全静态。
ge::graphStatus ValidateConv3ddxConsistency(gert::InferShapeContext* context, const int64_t (&inSize)[kInputSizeLen])
{
    const char* opName = (context == nullptr) ? nullptr : context->GetNodeName();

    const gert::Shape* filterShape = context->GetInputShape(kFilterIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterShape);
    const gert::Shape* dedyShape = context->GetInputShape(kOutBackpropIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, dedyShape);

    // rank 契约：filter/out_backprop 恒为 5D（rank_range [5,5]）
    OP_CHECK_IF(filterShape->GetDimNum() != kRank5,
                OP_LOGE(opName, "Conv3ddx InferShape failed: shape_mismatch, filter rank is %zu, expect 5",
                        filterShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dedyShape->GetDimNum() != kRank5,
                OP_LOGE(opName, "Conv3ddx InferShape failed: shape_mismatch, out_backprop rank is %zu, expect 5",
                        dedyShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    // 空张量：任一输入含 0 维即 null_input（无静默空跑路径，与内置基线同口径）
    for (size_t idx = 0; idx < kRank5; ++idx) {
        OP_CHECK_IF(filterShape->GetDim(idx) < 1 || dedyShape->GetDim(idx) < 1,
                    OP_LOGE(opName,
                            "Conv3ddx InferShape failed: null_input, empty tensor detected, dim %zu of "
                            "filter/out_backprop is non-positive (filter %lld, out_backprop %lld)",
                            idx, static_cast<long long>(filterShape->GetDim(idx)),
                            static_cast<long long>(dedyShape->GetDim(idx))),
                    return ge::GRAPH_FAILED);
    }

    // 形状维可表示域上界（SEC-1.3 值域收窄，先于一切几何运算）：filter/out_backprop
    // 各维 ≤ 2^31-1 —— kernelExtent = dilation*(kernel-1)+1 等有符号乘加的安全前提
    // （input_size 维值的上界已在 InferShapeConv3ddx 读取后同口径拒绝）
    for (size_t idx = 0; idx < kRank5; ++idx) {
        OP_CHECK_IF(filterShape->GetDim(idx) > kMaxDimValue || dedyShape->GetDim(idx) > kMaxDimValue,
                    OP_LOGE(opName,
                            "Conv3ddx InferShape failed: shape_mismatch, dim %zu of filter/out_backprop "
                            "exceeds representable range %lld (filter %lld, out_backprop %lld)",
                            idx, static_cast<long long>(kMaxDimValue), static_cast<long long>(filterShape->GetDim(idx)),
                            static_cast<long long>(dedyShape->GetDim(idx))),
                    return ge::GRAPH_FAILED);
    }

    // 属性载体（六属性按 OpDef 声明序）
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* stridesVec = attrs->GetListInt(kAttrStrides);
    const auto* padsVec = attrs->GetListInt(kAttrPads);
    const auto* dilationsVec = attrs->GetListInt(kAttrDilations);
    const int64_t* groupsPtr = attrs->GetInt(kAttrGroups);
    const char* dataFormatRaw = attrs->GetStr(kAttrDataFormat);
    OP_CHECK_IF(stridesVec == nullptr || padsVec == nullptr || dilationsVec == nullptr || groupsPtr == nullptr ||
                    dataFormatRaw == nullptr,
                OP_LOGE(opName, "Conv3ddx InferShape failed: attribute_value_out_of_range, attr pointer is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(stridesVec->GetSize() != kStridesLen,
                OP_LOGE(opName,
                        "Conv3ddx InferShape failed: attribute_value_out_of_range, strides size is %zu, "
                        "expect 5",
                        stridesVec->GetSize()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(padsVec->GetSize() != kPadsLen,
                OP_LOGE(opName,
                        "Conv3ddx InferShape failed: attribute_value_out_of_range, pads size is %zu, "
                        "expect 6",
                        padsVec->GetSize()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dilationsVec->GetSize() != kDilationsLen,
                OP_LOGE(opName,
                        "Conv3ddx InferShape failed: attribute_value_out_of_range, dilations size is %zu, "
                        "expect 5",
                        dilationsVec->GetSize()),
                return ge::GRAPH_FAILED);

    // data_format 枚举：决定 input_size 值序、out_backprop 轴序与 strides/dilations 位序
    const std::string dataFormat(dataFormatRaw);
    bool isNdhwc = false;
    if (dataFormat == "NCDHW") {
        isNdhwc = false;
    } else if (dataFormat == "NDHWC") {
        isNdhwc = true;
    } else {
        OP_LOGE(opName,
                "Conv3ddx InferShape failed: attribute_value_out_of_range, data_format '%s' not in "
                "{NCDHW, NDHWC}",
                dataFormat.c_str());
        return ge::GRAPH_FAILED;
    }

    // strides/dilations 的 N、C 位必须为 1（NCDHW 取位 0/1，NDHWC 取位 0/4）
    const int64_t* strides = stridesVec->GetData();
    const int64_t* dilations = dilationsVec->GetData();
    // pads 语义序 [D前, D后, H上, H下, W左, W右]，与 data_format 无关（支持单侧不对称）
    const int64_t* pads = padsVec->GetData();
    const size_t nPos = 0U;
    const size_t cPos = isNdhwc ? 4U : 1U;
    OP_CHECK_IF(strides[nPos] != 1 || strides[cPos] != 1,
                OP_LOGE(opName,
                        "Conv3ddx InferShape failed: attribute_value_out_of_range, strides N/C positions "
                        "must be 1, got strides[%zu]=%lld, strides[%zu]=%lld",
                        nPos, static_cast<long long>(strides[nPos]), cPos, static_cast<long long>(strides[cPos])),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dilations[nPos] != 1 || dilations[cPos] != 1,
                OP_LOGE(opName,
                        "Conv3ddx InferShape failed: attribute_value_out_of_range, dilations N/C "
                        "positions must be 1, got dilations[%zu]=%lld, dilations[%zu]=%lld",
                        nPos, static_cast<long long>(dilations[nPos]), cPos, static_cast<long long>(dilations[cPos])),
                return ge::GRAPH_FAILED);

    // 属性值域收窄（SEC-1.3：外部 int64 属性先收窄到 [0/1, 2^31-1] 再进入几何运算；
    // 与 tiling 层口径一致，消除 kernelExtent/通道乘积的有符号溢出 UB）
    for (size_t idx = 0; idx < kStridesLen; ++idx) {
        OP_CHECK_IF(strides[idx] < 1 || strides[idx] > kMaxDimValue,
                    OP_LOGE(opName,
                            "Conv3ddx InferShape failed: attribute_value_out_of_range, strides[%zu]=%lld "
                            "out of [1, %lld]",
                            idx, static_cast<long long>(strides[idx]), static_cast<long long>(kMaxDimValue)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(dilations[idx] < 1 || dilations[idx] > kMaxDimValue,
                    OP_LOGE(opName,
                            "Conv3ddx InferShape failed: attribute_value_out_of_range, dilations[%zu]=%lld "
                            "out of [1, %lld]",
                            idx, static_cast<long long>(dilations[idx]), static_cast<long long>(kMaxDimValue)),
                    return ge::GRAPH_FAILED);
    }
    for (size_t idx = 0; idx < kPadsLen; ++idx) {
        OP_CHECK_IF(pads[idx] < 0 || pads[idx] > kMaxDimValue,
                    OP_LOGE(opName,
                            "Conv3ddx InferShape failed: attribute_value_out_of_range, pads[%zu]=%lld "
                            "out of [0, %lld]",
                            idx, static_cast<long long>(pads[idx]), static_cast<long long>(kMaxDimValue)),
                    return ge::GRAPH_FAILED);
    }

    // 语义维定位（值序/轴序随 data_format，见 layout_contract.variants）
    //   NCDHW: input_size=[N,Cin,Din,Hin,Win]，dedy=(N,Cout,Dout,Hout,Wout)，空间属性位 strides[2:5]
    //   NDHWC: input_size=[N,Din,Hin,Win,Cin]，dedy=(N,Dout,Hout,Wout,Cout)，空间属性位 strides[1:4]
    const int64_t batch = inSize[0];
    const int64_t cin = isNdhwc ? inSize[4] : inSize[1];
    const int64_t dIn = isNdhwc ? inSize[1] : inSize[2];
    const int64_t hIn = isNdhwc ? inSize[2] : inSize[3];
    const int64_t wIn = isNdhwc ? inSize[3] : inSize[4];
    const size_t spatialBase = isNdhwc ? 1U : 2U; // dedy 空间维起始轴 / strides 空间位起始
    const size_t dedyChannelIdx = isNdhwc ? 4U : 1U;

    // groups：取值下界 1（先于 filter 段读取——ND 消歧判据需要 groups 参与）；
    // 上界同 SEC-1.3 值域收窄口径（cinPerGroup*groups 乘积安全前提）
    const int64_t groups = *groupsPtr;
    OP_CHECK_IF(groups < 1,
                OP_LOGE(opName, "Conv3ddx InferShape failed: attribute_value_out_of_range, groups %lld < 1",
                        static_cast<long long>(groups)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(groups > kMaxDimValue,
                OP_LOGE(opName,
                        "Conv3ddx InferShape failed: attribute_value_out_of_range, groups %lld exceeds "
                        "representable range %lld",
                        static_cast<long long>(groups), static_cast<long long>(kMaxDimValue)),
                return ge::GRAPH_FAILED);

    // filter 语义维（布局独立于 data_format，按 desc origin format 解析）：
    //   - origin format 为三种注册布局（NCDHW/NDHWC/DHWCN）之一：单一解释，走既有校验；
    //   - origin format 为 ND（布局未声明，如 GE/TTK 馈入的 ND desc）：按三种注册布局做
    //     唯一一致性消歧——通道对齐 + 前向几何逐轴一致的候选必须恰好一个；零个（均不一致）
    //     或多个（歧义）均按 shape_mismatch 响亮拒绝，绝不静默猜测布局。
    const auto* filterDesc = context->GetInputDesc(kFilterIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterDesc);
    FilterSemanticDims fDims = {0, 0, 0, 0, 0};
    const ge::Format filterFormat = filterDesc->GetOriginFormat();
    if (GetFilterSemanticDims(*filterShape, filterFormat, fDims)) {
        OP_LOGD(opName, "Conv3ddx InferShape: filter layout specified by origin format %d",
                static_cast<int32_t>(filterFormat));
    } else if (filterFormat == ge::FORMAT_ND) {
        FilterSemanticDims cands[3] = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
        MakeFilterCandidates(*filterShape, cands);
        const CandidateCtx candCtx = {cin,
                                      groups,
                                      dedyShape->GetDim(dedyChannelIdx),
                                      dedyShape->GetDim(spatialBase),
                                      dedyShape->GetDim(spatialBase + 1U),
                                      dedyShape->GetDim(spatialBase + 2U),
                                      dIn,
                                      hIn,
                                      wIn,
                                      strides,
                                      dilations,
                                      pads,
                                      spatialBase};
        size_t consistentIdx = 3U;
        size_t consistentCount = 0U;
        for (size_t ci = 0U; ci < 3U; ++ci) {
            if (FilterCandidateConsistent(cands[ci], candCtx)) {
                if (consistentCount == 0U) {
                    consistentIdx = ci;
                }
                ++consistentCount;
            }
        }
        if (consistentCount == 0U) {
            OP_LOGE(opName, "Conv3ddx InferShape failed: shape_mismatch, filter layout unspecified (origin format "
                            "ND) and no registered layout interpretation is consistent (NCDHW/NDHWC/DHWCN all violate "
                            "channel/geometry consistency)");
            return ge::GRAPH_FAILED;
        }
        if (consistentCount > 1U) {
            OP_LOGE(opName,
                    "Conv3ddx InferShape failed: shape_mismatch, filter layout unspecified (origin format "
                    "ND) and interpretation is ambiguous (%zu registered layouts consistent), refusing to guess",
                    consistentCount);
            return ge::GRAPH_FAILED;
        }
        fDims = cands[consistentIdx];
        OP_LOGD(opName,
                "Conv3ddx InferShape: filter layout unspecified (origin format ND), uniquely resolved to "
                "registered layout index %zu (0=NCDHW, 1=NDHWC, 2=DHWCN) by consistency disambiguation",
                consistentIdx);
        // 布局消歧唯一成立后，把消歧出的注册布局回写进本上下文的 filter desc
        // （CompileTimeTensorDesc 提供 SetOriginFormat/SetStorageFormat；经 const_cast
        // 去除 GetInputDesc 的只读限定）。origin 与 storage 同值改写：filter 内存数据
        // 即按该布局生成，形状各维不变，仅布局标签由未声明（ND）改为真实注册布局。
        // 注（fix-2/fix-3 实测）：KernelRunContextBuilder 的 arena 为逐调用拷贝，
        // 本改写不回传 ge::OpDesc——FE 引擎检查与 tiling 读到的仍由 op info 注册层
        // 决定（ND filter 经 fix-3 构建接线归一化为首个注册布局 DHWCN）。本改写使
        // InferShape 上下文内 fDims 消歧结论与 desc 标签一致，并在 CANN 上下文语义
        // 变化（改写回传激活）时即为等价的注册布局承接；消歧不唯一的分支已在上方
        // 响亮拒绝，绝不猜测布局。
        static constexpr ge::Format kResolvedLayouts[3] = {ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_DHWCN};
        auto* mutableFilterDesc = const_cast<gert::CompileTimeTensorDesc*>(filterDesc);
        if (mutableFilterDesc != nullptr) {
            const ge::Format resolved = kResolvedLayouts[consistentIdx];
            mutableFilterDesc->SetOriginFormat(resolved);
            mutableFilterDesc->SetStorageFormat(resolved);
            OP_LOGD(opName,
                    "Conv3ddx InferShape: filter desc normalized to registered layout %d "
                    "(origin=storage) in infer context",
                    static_cast<int32_t>(resolved));
        }
    } else {
        OP_LOGE(opName,
                "Conv3ddx InferShape failed: attribute_value_out_of_range, filter origin format "
                "%d not in {NCDHW, NDHWC, DHWCN}",
                static_cast<int32_t>(filterFormat));
        return ge::GRAPH_FAILED;
    }

    // groups 与通道收缩对齐（指定布局路径既有校验；ND 消歧路径下已按候选判据成立）
    OP_CHECK_IF(
        cin % groups != 0 || fDims.cinPerGroup * groups != cin,
        OP_LOGE(opName,
                "Conv3ddx InferShape failed: shape_mismatch, input Cin %lld not divisible by "
                "groups %lld or filter Cin dim %lld * groups != Cin",
                static_cast<long long>(cin), static_cast<long long>(groups), static_cast<long long>(fDims.cinPerGroup)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(fDims.cout % groups != 0,
                OP_LOGE(opName,
                        "Conv3ddx InferShape failed: shape_mismatch, filter Cout %lld not divisible by "
                        "groups %lld",
                        static_cast<long long>(fDims.cout), static_cast<long long>(groups)),
                return ge::GRAPH_FAILED);

    // 批次一致：input_size[0] == out_backprop.shape[0]（两种格式 N 均为轴 0）
    OP_CHECK_IF(dedyShape->GetDim(0) != batch,
                OP_LOGE(opName,
                        "Conv3ddx InferShape failed: shape_mismatch, out_backprop batch %lld != "
                        "input_size batch %lld",
                        static_cast<long long>(dedyShape->GetDim(0)), static_cast<long long>(batch)),
                return ge::GRAPH_FAILED);

    // 通道收缩对齐：out_backprop 通道维 == filter 语义 Cout 维
    OP_CHECK_IF(dedyShape->GetDim(dedyChannelIdx) != fDims.cout,
                OP_LOGE(opName,
                        "Conv3ddx InferShape failed: shape_mismatch, out_backprop channel %lld != filter "
                        "Cout %lld",
                        static_cast<long long>(dedyShape->GetDim(dedyChannelIdx)), static_cast<long long>(fDims.cout)),
                return ge::GRAPH_FAILED);

    // 前向几何一致性（Dout/Hout/Wout 逐轴；out_backprop 空间维必须逐一相等）
    const int64_t sD = strides[spatialBase];
    const int64_t sH = strides[spatialBase + 1U];
    const int64_t sW = strides[spatialBase + 2U];
    const int64_t dD = dilations[spatialBase];
    const int64_t dH = dilations[spatialBase + 1U];
    const int64_t dW = dilations[spatialBase + 2U];
    const int64_t expectedDOut = ForwardOutSize(dIn, pads[0], pads[1], fDims.kD, dD, sD);
    const int64_t expectedHOut = ForwardOutSize(hIn, pads[2], pads[3], fDims.kH, dH, sH);
    const int64_t expectedWOut = ForwardOutSize(wIn, pads[4], pads[5], fDims.kW, dW, sW);
    OP_CHECK_IF(expectedDOut < 1 || expectedDOut != dedyShape->GetDim(spatialBase),
                OP_LOGE(opName,
                        "Conv3ddx InferShape failed: shape_mismatch, forward geometry Dout expected %lld "
                        "but out_backprop dim is %lld (Din=%lld, pads=[%lld,%lld], kD=%lld, dD=%lld, sD=%lld)",
                        static_cast<long long>(expectedDOut), static_cast<long long>(dedyShape->GetDim(spatialBase)),
                        static_cast<long long>(dIn), static_cast<long long>(pads[0]), static_cast<long long>(pads[1]),
                        static_cast<long long>(fDims.kD), static_cast<long long>(dD), static_cast<long long>(sD)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        expectedHOut < 1 || expectedHOut != dedyShape->GetDim(spatialBase + 1U),
        OP_LOGE(opName,
                "Conv3ddx InferShape failed: shape_mismatch, forward geometry Hout expected %lld "
                "but out_backprop dim is %lld (Hin=%lld, pads=[%lld,%lld], kH=%lld, dH=%lld, sH=%lld)",
                static_cast<long long>(expectedHOut), static_cast<long long>(dedyShape->GetDim(spatialBase + 1U)),
                static_cast<long long>(hIn), static_cast<long long>(pads[2]), static_cast<long long>(pads[3]),
                static_cast<long long>(fDims.kH), static_cast<long long>(dH), static_cast<long long>(sH)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        expectedWOut < 1 || expectedWOut != dedyShape->GetDim(spatialBase + 2U),
        OP_LOGE(opName,
                "Conv3ddx InferShape failed: shape_mismatch, forward geometry Wout expected %lld "
                "but out_backprop dim is %lld (Win=%lld, pads=[%lld,%lld], kW=%lld, dW=%lld, sW=%lld)",
                static_cast<long long>(expectedWOut), static_cast<long long>(dedyShape->GetDim(spatialBase + 2U)),
                static_cast<long long>(wIn), static_cast<long long>(pads[4]), static_cast<long long>(pads[5]),
                static_cast<long long>(fDims.kW), static_cast<long long>(dW), static_cast<long long>(sW)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace

// InferShape{OpType}：y.shape = input_size 元素值（值依赖推导 + 一致性校验）
static ge::graphStatus InferShapeConv3ddx(gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    gert::Shape* yShape = context->GetOutputShape(kOutputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    const gert::Tensor* inputSizeTensor = context->GetInputTensor(kInputSizeIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSizeTensor);

    // unknown rank（[-2]）/ unknown shape（含 -1）：输出全 -1，rank 固定 5（与内置基线同口径）
    const gert::Shape& inputSizeShape = inputSizeTensor->GetOriginShape();
    if (HasUnknownDim(inputSizeShape)) {
        SetUnknownShape(*yShape);
        OP_LOGD(context->GetNodeName(), "Conv3ddx InferShape: input_size unknown, y shape set to all -1 (rank 5)");
        return ge::GRAPH_SUCCESS;
    }

    // input_size 接口契约：1D 且长度 5（REQUIREMENTS §8.2.1；违反报 shape_mismatch）
    OP_CHECK_IF(inputSizeShape.GetDimNum() != 1U || static_cast<size_t>(inputSizeShape.GetShapeSize()) != kInputSizeLen,
                OP_LOGE(context->GetNodeName(),
                        "Conv3ddx InferShape failed: shape_mismatch, input_size must be "
                        "1D with length 5, got rank %zu, element num %lld",
                        inputSizeShape.GetDimNum(), static_cast<long long>(inputSizeShape.GetShapeSize())),
                return ge::GRAPH_FAILED);

    const gert::Shape* filterShape = context->GetInputShape(kFilterIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterShape);

    // 动态场景：filter 形状未知或 input_size 常量数据不可见（图编译期占位输入）时，
    // 具体维值无法读取，输出全 -1 交给运行时（值依赖）推导
    if (HasUnknownDim(*filterShape) || !IsConstDataVisible(*inputSizeTensor)) {
        SetUnknownShape(*yShape);
        OP_LOGD(context->GetNodeName(), "Conv3ddx InferShape: input_size const data invisible or filter shape "
                                        "unknown, y shape set to all -1 (rank 5)");
        return ge::GRAPH_SUCCESS;
    }

    // 读取 5 个维值（int32/int64；其他 dtype 违反形状描述符契约）
    int64_t inSize[kInputSizeLen] = {0, 0, 0, 0, 0};
    OP_CHECK_IF(!ReadInputSizeValues(*inputSizeTensor, inSize),
                OP_LOGE(context->GetNodeName(),
                        "Conv3ddx InferShape failed: dtype_not_supported, input_size dtype "
                        "%d not in {DT_INT32, DT_INT64}",
                        static_cast<int32_t>(inputSizeTensor->GetDataType())),
                return ge::GRAPH_FAILED);

    // 维值边界：元素值为 0 即空张量（null_input，无静默空跑路径）；负值为非法维
    // （shape_mismatch）；超可表示域上界拒绝（shape_mismatch，SEC-1.3 值域收窄——
    // y 形状与后续几何运算不携带超域值）
    for (size_t idx = 0; idx < kInputSizeLen; ++idx) {
        OP_CHECK_IF(inSize[idx] < 0,
                    OP_LOGE(context->GetNodeName(),
                            "Conv3ddx InferShape failed: shape_mismatch, input_size[%zu] "
                            "= %lld is negative",
                            idx, static_cast<long long>(inSize[idx])),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(inSize[idx] == 0,
                    OP_LOGE(context->GetNodeName(),
                            "Conv3ddx InferShape failed: null_input, input_size[%zu] = 0 "
                            "(empty tensor, no silent empty-run path)",
                            idx),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(inSize[idx] > kMaxDimValue,
                    OP_LOGE(context->GetNodeName(),
                            "Conv3ddx InferShape failed: shape_mismatch, input_size[%zu] "
                            "= %lld exceeds representable range %lld",
                            idx, static_cast<long long>(inSize[idx]), static_cast<long long>(kMaxDimValue)),
                    return ge::GRAPH_FAILED);
    }

    // 输出 shape = input_size 元素值（data_dependent_shape；值序随 data_format 由校验段消费）
    yShape->SetDimNum(kRank5);
    for (size_t idx = 0; idx < kRank5; ++idx) {
        yShape->SetDim(idx, inSize[idx]);
    }

    // 静态场景逐维一致性校验；filter/out_backprop 含未知维（部分动态）时跳过，交由运行时复推
    const gert::Shape* dedyShape = context->GetInputShape(kOutBackpropIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, dedyShape);
    if (!HasUnknownDim(*filterShape) && !HasUnknownDim(*dedyShape)) {
        const ge::graphStatus ret = ValidateConv3ddxConsistency(context, inSize);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }

    OP_LOGD(context->GetNodeName(),
            "Conv3ddx InferShape: y shape set from input_size values "
            "(N=%lld, C=%lld, D=%lld, H=%lld, W=%lld)",
            static_cast<long long>(yShape->GetDim(0)), static_cast<long long>(yShape->GetDim(1)),
            static_cast<long long>(yShape->GetDim(2)), static_cast<long long>(yShape->GetDim(3)),
            static_cast<long long>(yShape->GetDim(4)));
    return ge::GRAPH_SUCCESS;
}

// 注册：IMPL_OP_INFERSHAPE({OpType}).InferShape(InferShape{OpType})，声明输入 0 值依赖
IMPL_OP_INFERSHAPE(Conv3ddx).InferShape(InferShapeConv3ddx).InputsDataDependency({0});

} // namespace ops
