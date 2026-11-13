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
 * \file conv3dtranspose_infershape.cpp
 * \brief Conv3DTranspose 算子形状推导实现（GEIR OpType Conv3DTranspose）
 *
 * 事实源：design/InferShapeDtype.md「InferShape」节（spec.outputs.y 为 data_dependent_shape）。
 * 核心语义：
 *   - y.shape = input_size 的元素值（1D int32/int64、长度 5；值序随 data_format），
 *     输出对输入 0 值依赖（InputsDataDependency({0})，REQUIREMENTS §5.6）；
 *   - input_size 元素值不可见（动态图占位）或 filter 形状未知时，输出 5 维全部置 -1
 *     （与内置基线 Conv3DTranspose 同口径），交由运行时复推；
 *   - unknown rank（[-2]）/ unknown shape（含 -1）场景输出全 -1，rank 固定 5；
 *   - 零输出哨兵（input_size 元素值全零，内置口径 CheckOutputAllZero）：属性驱动兜底
 *     —— N 取 x.shape[0]、Cin 取 filter 语义 Cin/G 维 × groups、逐空间轴
 *     L_y = s*(L_x-1) + q + (d*(K-1)+1) - (p^- + p^+)（rho=0，REQUIREMENTS §8.1）；
 *   - 静态场景逐维做 batch/通道/转置几何一致性校验（任一不满足报 shape_mismatch、
 *     null_input 或 attribute_value_out_of_range，判据见 design/InferShapeDtype.md
 *     「逐维对齐与几何一致性校验」「边界与失败判据」）：
 *       N：input_size[0] == x.shape[0]；
 *       Cin：input_size 通道位 == filter 语义 Cin/G 维 × groups；
 *       Cout：x 通道维 == filter 语义 Cout 维；
 *       空间轴：L_x = (L_y + p^- + p^+ - (Keff + q)) // s + 1 逐一等于 x 实际空间维
 *       （等价于逐轴显式形状余量 0 <= rho < s，spec shape_constraints.global_constraints）；
 *       bias：存在时 bias.shape == (Cin,)；
 *   - filter 布局取 desc origin format：三种注册布局（NCDHW/NDHWC/DHWCN）直接解析，
 *     独立于 data_format；origin format 为 ND（布局未声明）时按三种注册布局做唯一一致
 *     性消歧（通道对齐 + 转置几何逐轴一致），候选必须恰好一个——零个或多个均按
 *     shape_mismatch 响亮拒绝，绝不静默猜测布局。
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
// 输入/输出序（与 conv3dtranspose_proto.h REG_OP / conv3dtranspose_def.cpp OpDef 声明一致）
constexpr size_t kInputSizeIdx = 0U; // 输入 0：input_size（整型形状描述符，值依赖）
constexpr size_t kXIdx = 1U;         // 输入 1：x（转置卷积输入特征图，dx 语义 dedy）
constexpr size_t kFilterIdx = 2U;    // 输入 2：filter（权重）
constexpr size_t kBiasIdx = 3U;      // 输入 3：bias（OPTIONAL_INPUT，可选逐通道偏置）
constexpr size_t kOutputIdx = 0U;    // 输出 0：y（转置卷积输出，dx 语义 dedx）
constexpr size_t kRank5 = 5U;        // x/filter/y 固定 5D
constexpr size_t kInputSizeLen = 5U; // input_size 1D 长度 5（接口契约，REQUIREMENTS §8.2.1）
constexpr int64_t kUnknownDim = -1;

// 属性声明序（OpDef：strides/pads/dilations/groups/data_format/output_padding/offset_x/
// enable_hf32；offset_x 与 enable_hf32 为保留属性，不参与形状推导校验）
constexpr size_t kAttrStrides = 0U;
constexpr size_t kAttrPads = 1U;
constexpr size_t kAttrDilations = 2U;
constexpr size_t kAttrGroups = 3U;
constexpr size_t kAttrDataFormat = 4U;
constexpr size_t kAttrOutputPadding = 5U;
constexpr size_t kStridesLen = 5U;
constexpr size_t kDilationsLen = 5U;
constexpr size_t kOutputPaddingLen = 5U;

// 参与几何运算的外部值可表示域上界（SEC-1.3：先收窄再运算；与 tiling 层
// CONV3DT_MAX_DIM 同口径）：strides/dilations/pads/output_padding/groups/形状维均限制
// 在 [0/1, 2^31-1]，使 Keff = dilation*(kernel-1)+1、numerator、cinPerGroup*groups 等
// 有符号中间量恒 < 2^62，消除有符号整数溢出 UB。
constexpr int64_t kMaxDimValue = 2147483647;
// groups 取值域上界（spec attributes.groups machine_constraint，REQUIREMENTS §8.2.2）
constexpr int64_t kMaxGroups = 65535;

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

// 转置几何反算（Host 反算检查形式，design/InferShapeDtype.md「逐维对齐与几何一致性校验」）：
// 由目标输出维 L_y 反算期望的 x 空间维；分子为负时返回负值
//   L_x = ( L_y + p^- + p^+ - (Keff + q) ) // s + 1，Keff = d*(K-1)+1
// 与逐轴显式形状余量判据等价：0 <= rho < s，rho = L_y - [s*(L_x-1) + q + Keff - (p^-+p^+)]
int64_t TransposeBackInputSize(const int64_t outSize, const int64_t padFront, const int64_t padBack,
                               const int64_t kernel, const int64_t dilation, const int64_t stride,
                               const int64_t outputPad)
{
    const int64_t kernelExtent = dilation * (kernel - 1) + 1;
    const int64_t numerator = outSize + padFront + padBack - (kernelExtent + outputPad);
    if (numerator < 0) {
        return -1;
    }
    return numerator / stride + 1;
}

// 零输出哨兵属性推导（design/InferShapeDtype.md「DynamicShape」属性驱动兜底）：
// 由 x 空间维推 y 目标维（属性推导模式 rho=0）
//   L_y = s*(L_x-1) + q + (d*(K-1)+1) - (p^- + p^+)
int64_t TransposeDerivedOutSize(const int64_t inSize, const int64_t padFront, const int64_t padBack,
                                const int64_t kernel, const int64_t dilation, const int64_t stride,
                                const int64_t outputPad)
{
    const int64_t kernelExtent = dilation * (kernel - 1) + 1;
    return stride * (inSize - 1) + outputPad + kernelExtent - (padFront + padBack);
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

// 属性集（解析 + 校验后的取位视图；strides/dilations/output_padding 位序与 pads 语义序
// 随 data_format 物理位序，design/InferShapeDtype.md「逐维对齐与几何一致性校验」）
struct TransposeAttrSet {
    bool isNdhwc;
    size_t spatialBase;           // strides/dilations/output_padding 空间位起始（NCDHW=2 / NDHWC=1）
    size_t xChannelIdx;           // x 通道轴（NCDHW=1 / NDHWC=4）
    const int64_t* strides;       // 5 元
    const int64_t* dilations;     // 5 元
    int64_t pads6[6];             // pads 按 1/3/6 元展开后的逐侧值 [pF,pB,pT,pBm,pL,pR]
    const int64_t* outputPadding; // 5 元
    int64_t groups;
};

// 属性解析与校验（主路径与零输出哨兵共用；判据见 design/InferShapeDtype.md
// 「边界与失败判据」属性各行为：attribute_value_out_of_range）
ge::graphStatus ParseTransposeAttrs(gert::InferShapeContext* context, TransposeAttrSet& as)
{
    const char* opName = (context == nullptr) ? nullptr : context->GetNodeName();
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* stridesVec = attrs->GetListInt(kAttrStrides);
    const auto* padsVec = attrs->GetListInt(kAttrPads);
    const auto* dilationsVec = attrs->GetListInt(kAttrDilations);
    const int64_t* groupsPtr = attrs->GetInt(kAttrGroups);
    const char* dataFormatRaw = attrs->GetStr(kAttrDataFormat);
    const auto* outputPaddingVec = attrs->GetListInt(kAttrOutputPadding);
    OP_CHECK_IF(stridesVec == nullptr || padsVec == nullptr || dilationsVec == nullptr || groupsPtr == nullptr ||
                    dataFormatRaw == nullptr || outputPaddingVec == nullptr,
                OP_LOGE(opName, "Conv3DTranspose InferShape failed: attribute_value_out_of_range, attr "
                                "pointer is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(stridesVec->GetSize() != kStridesLen,
                OP_LOGE(opName,
                        "Conv3DTranspose InferShape failed: attribute_value_out_of_range, strides "
                        "size is %zu, expect 5",
                        stridesVec->GetSize()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dilationsVec->GetSize() != kDilationsLen,
                OP_LOGE(opName,
                        "Conv3DTranspose InferShape failed: attribute_value_out_of_range, dilations "
                        "size is %zu, expect 5",
                        dilationsVec->GetSize()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputPaddingVec->GetSize() != kOutputPaddingLen,
                OP_LOGE(opName,
                        "Conv3DTranspose InferShape failed: attribute_value_out_of_range, "
                        "output_padding size is %zu, expect 5",
                        outputPaddingVec->GetSize()),
                return ge::GRAPH_FAILED);
    // pads 支持 1/3/6 元展开（1 元三轴同值、3 元逐轴前后同值、6 元逐侧独立，
    // REQUIREMENTS §5.6/§5.10.1；语义序与 data_format 无关）
    const size_t padsLen = padsVec->GetSize();
    OP_CHECK_IF(padsLen != 1U && padsLen != 3U && padsLen != 6U,
                OP_LOGE(opName,
                        "Conv3DTranspose InferShape failed: attribute_value_out_of_range, pads "
                        "size is %zu, expect 1/3/6",
                        padsLen),
                return ge::GRAPH_FAILED);

    // data_format 枚举：决定 input_size 值序、x 轴序与 strides/dilations/output_padding 位序
    const std::string dataFormat(dataFormatRaw);
    if (dataFormat == "NCDHW") {
        as.isNdhwc = false;
    } else if (dataFormat == "NDHWC") {
        as.isNdhwc = true;
    } else {
        OP_LOGE(opName,
                "Conv3DTranspose InferShape failed: attribute_value_out_of_range, data_format "
                "'%s' not in {NCDHW, NDHWC}",
                dataFormat.c_str());
        return ge::GRAPH_FAILED;
    }
    as.spatialBase = as.isNdhwc ? 1U : 2U;
    as.xChannelIdx = as.isNdhwc ? 4U : 1U;

    const int64_t* strides = stridesVec->GetData();
    const int64_t* dilations = dilationsVec->GetData();
    const int64_t* pads = padsVec->GetData();
    const int64_t* outputPadding = outputPaddingVec->GetData();
    as.strides = strides;
    as.dilations = dilations;
    as.outputPadding = outputPadding;

    // strides/dilations 的 N、C 位必须为 1；output_padding 的 N、C 位必须为 0
    // （NCDHW 取位 0/1，NDHWC 取位 0/4，REQUIREMENTS §8.2.2）
    const size_t nPos = 0U;
    const size_t cPos = as.isNdhwc ? 4U : 1U;
    OP_CHECK_IF(strides[nPos] != 1 || strides[cPos] != 1,
                OP_LOGE(opName,
                        "Conv3DTranspose InferShape failed: attribute_value_out_of_range, strides "
                        "N/C positions must be 1, got strides[%zu]=%lld, strides[%zu]=%lld",
                        nPos, static_cast<long long>(strides[nPos]), cPos, static_cast<long long>(strides[cPos])),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dilations[nPos] != 1 || dilations[cPos] != 1,
                OP_LOGE(opName,
                        "Conv3DTranspose InferShape failed: attribute_value_out_of_range, dilations "
                        "N/C positions must be 1, got dilations[%zu]=%lld, dilations[%zu]=%lld",
                        nPos, static_cast<long long>(dilations[nPos]), cPos, static_cast<long long>(dilations[cPos])),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        outputPadding[nPos] != 0 || outputPadding[cPos] != 0,
        OP_LOGE(opName,
                "Conv3DTranspose InferShape failed: attribute_value_out_of_range, "
                "output_padding N/C positions must be 0, got output_padding[%zu]=%lld, "
                "output_padding[%zu]=%lld",
                nPos, static_cast<long long>(outputPadding[nPos]), cPos, static_cast<long long>(outputPadding[cPos])),
        return ge::GRAPH_FAILED);

    // 属性值域收窄（SEC-1.3：外部 int64 属性先收窄到 [0/1, 2^31-1] 再进入几何运算；
    // 与 tiling 层口径一致，消除 Keff/通道乘积的有符号溢出 UB）
    for (size_t idx = 0; idx < kStridesLen; ++idx) {
        OP_CHECK_IF(strides[idx] < 1 || strides[idx] > kMaxDimValue,
                    OP_LOGE(opName,
                            "Conv3DTranspose InferShape failed: attribute_value_out_of_range, "
                            "strides[%zu]=%lld out of [1, %lld]",
                            idx, static_cast<long long>(strides[idx]), static_cast<long long>(kMaxDimValue)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(dilations[idx] < 1 || dilations[idx] > kMaxDimValue,
                    OP_LOGE(opName,
                            "Conv3DTranspose InferShape failed: attribute_value_out_of_range, "
                            "dilations[%zu]=%lld out of [1, %lld]",
                            idx, static_cast<long long>(dilations[idx]), static_cast<long long>(kMaxDimValue)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(outputPadding[idx] < 0 || outputPadding[idx] > kMaxDimValue,
                    OP_LOGE(opName,
                            "Conv3DTranspose InferShape failed: attribute_value_out_of_range, "
                            "output_padding[%zu]=%lld out of [0, %lld]",
                            idx, static_cast<long long>(outputPadding[idx]), static_cast<long long>(kMaxDimValue)),
                    return ge::GRAPH_FAILED);
    }
    for (size_t idx = 0; idx < padsLen; ++idx) {
        OP_CHECK_IF(pads[idx] < 0 || pads[idx] > kMaxDimValue,
                    OP_LOGE(opName,
                            "Conv3DTranspose InferShape failed: attribute_value_out_of_range, "
                            "pads[%zu]=%lld out of [0, %lld]",
                            idx, static_cast<long long>(pads[idx]), static_cast<long long>(kMaxDimValue)),
                    return ge::GRAPH_FAILED);
    }

    // groups 取值域 [1, 65535]（spec attributes.groups machine_constraint）
    const int64_t groups = *groupsPtr;
    OP_CHECK_IF(groups < 1 || groups > kMaxGroups,
                OP_LOGE(opName,
                        "Conv3DTranspose InferShape failed: attribute_value_out_of_range, groups "
                        "%lld out of [1, %lld]",
                        static_cast<long long>(groups), static_cast<long long>(kMaxGroups)),
                return ge::GRAPH_FAILED);
    as.groups = groups;

    // pads 1/3/6 元展开 → 逐侧 6 元（语义序 [pD_front, pD_back, pH_top, pH_bottom,
    // pW_left, pW_right]，与 data_format 无关）
    if (padsLen == 1U) {
        for (size_t idx = 0; idx < 6U; ++idx) {
            as.pads6[idx] = pads[0]; // 1 元三轴同值
        }
    } else if (padsLen == 3U) {
        for (size_t axis = 0; axis < 3U; ++axis) {
            as.pads6[2 * axis] = pads[axis];      // 逐轴前侧
            as.pads6[2 * axis + 1U] = pads[axis]; // 逐轴后侧同值
        }
    } else {
        for (size_t idx = 0; idx < 6U; ++idx) {
            as.pads6[idx] = pads[idx]; // 6 元逐侧独立
        }
    }

    // 转置专属逐轴约束：q < s 或 q < d（违反 ⟺ q ≥ s 且 q ≥ d；PyTorch §4.4 同口径，
    // REQUIREMENTS §2.1.2/§8.2.2；空间分量位序随 data_format）
    const size_t b = as.spatialBase;
    const int64_t sD = strides[b];
    const int64_t sH = strides[b + 1U];
    const int64_t sW = strides[b + 2U];
    const int64_t dD = dilations[b];
    const int64_t dH = dilations[b + 1U];
    const int64_t dW = dilations[b + 2U];
    const int64_t qD = outputPadding[b];
    const int64_t qH = outputPadding[b + 1U];
    const int64_t qW = outputPadding[b + 2U];
    OP_CHECK_IF((qD >= sD && qD >= dD) || (qH >= sH && qH >= dH) || (qW >= sW && qW >= dW),
                OP_LOGE(opName,
                        "Conv3DTranspose InferShape failed: attribute_value_out_of_range, "
                        "output_padding spatial components (qD %lld, qH %lld, qW %lld) violate per-axis "
                        "q < stride or q < dilation (s %lld/%lld/%lld, d %lld/%lld/%lld)",
                        static_cast<long long>(qD), static_cast<long long>(qH), static_cast<long long>(qW),
                        static_cast<long long>(sD), static_cast<long long>(sH), static_cast<long long>(sW),
                        static_cast<long long>(dD), static_cast<long long>(dH), static_cast<long long>(dW)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ND（布局未声明）filter 消歧判据：候选布局下通道对齐 + 转置几何逐轴全部一致
// （静默判定，不落日志——消歧结论由调用方统一报告）
struct CandidateCtx {
    int64_t cin; // y 通道维真值（input_size 通道位）
    int64_t groups;
    int64_t xChannel; // x 通道维（= filter 语义 Cout）
    int64_t xD;
    int64_t xH;
    int64_t xW;
    int64_t yD; // y 空间维真值（input_size 空间位）
    int64_t yH;
    int64_t yW;
    const TransposeAttrSet* attrs;
};

bool FilterCandidateConsistent(const FilterSemanticDims& cand, const CandidateCtx& ctx)
{
    if (ctx.cin % ctx.groups != 0 || cand.cinPerGroup * ctx.groups != ctx.cin) {
        return false;
    }
    if (cand.cout % ctx.groups != 0) {
        return false;
    }
    if (ctx.xChannel != cand.cout) {
        return false;
    }
    const TransposeAttrSet& as = *ctx.attrs;
    const size_t b = as.spatialBase;
    const int64_t expectedD = TransposeBackInputSize(ctx.yD, as.pads6[0], as.pads6[1], cand.kD, as.dilations[b],
                                                     as.strides[b], as.outputPadding[b]);
    const int64_t expectedH = TransposeBackInputSize(ctx.yH, as.pads6[2], as.pads6[3], cand.kH, as.dilations[b + 1U],
                                                     as.strides[b + 1U], as.outputPadding[b + 1U]);
    const int64_t expectedW = TransposeBackInputSize(ctx.yW, as.pads6[4], as.pads6[5], cand.kW, as.dilations[b + 2U],
                                                     as.strides[b + 2U], as.outputPadding[b + 2U]);
    if (expectedD < 1 || expectedD != ctx.xD) {
        return false;
    }
    if (expectedH < 1 || expectedH != ctx.xH) {
        return false;
    }
    if (expectedW < 1 || expectedW != ctx.xW) {
        return false;
    }
    return true;
}

// 哨兵派生：按候选布局从 x 空间维属性推导 y 各维（rho=0；值序随 data_format）
void DeriveSentinelDims(const FilterSemanticDims& cand, const TransposeAttrSet& as, const gert::Shape& xShape,
                        int64_t (&yDims)[kRank5])
{
    const size_t b = as.spatialBase;
    const int64_t dOut = TransposeDerivedOutSize(xShape.GetDim(b), as.pads6[0], as.pads6[1], cand.kD, as.dilations[b],
                                                 as.strides[b], as.outputPadding[b]);
    const int64_t hOut = TransposeDerivedOutSize(xShape.GetDim(b + 1U), as.pads6[2], as.pads6[3], cand.kH,
                                                 as.dilations[b + 1U], as.strides[b + 1U], as.outputPadding[b + 1U]);
    const int64_t wOut = TransposeDerivedOutSize(xShape.GetDim(b + 2U), as.pads6[4], as.pads6[5], cand.kW,
                                                 as.dilations[b + 2U], as.strides[b + 2U], as.outputPadding[b + 2U]);
    const int64_t cin = cand.cinPerGroup * as.groups;
    yDims[0] = xShape.GetDim(0);
    if (as.isNdhwc) {
        yDims[1] = dOut;
        yDims[2] = hOut;
        yDims[3] = wOut;
        yDims[4] = cin;
    } else {
        yDims[1] = cin;
        yDims[2] = dOut;
        yDims[3] = hOut;
        yDims[4] = wOut;
    }
}

// bias 存在性与长度校验（design/InferShapeDtype.md「broadcast 对齐规则」：
// bias.shape == (Cin,) 精确等于 y 通道数，否则 shape_mismatch）
// 存在性口径与 tiling 层一致：槽位实例化（GetOptionalInputTensor 非 nullptr）且
// 元素数 > 0（0 元素空可选张量按无 bias 路由，非报错）
ge::graphStatus ValidateBiasLength(gert::InferShapeContext* context, const int64_t cin)
{
    const char* opName = (context == nullptr) ? nullptr : context->GetNodeName();
    const gert::Tensor* biasTensor = context->GetOptionalInputTensor(kBiasIdx);
    const gert::Shape* biasShape = context->GetInputShape(kBiasIdx);
    const bool hasBias =
        biasTensor != nullptr && biasShape != nullptr && biasShape->GetDimNum() > 0 && biasShape->GetShapeSize() > 0;
    if (!hasBias) {
        OP_LOGD(opName, "Conv3DTranspose InferShape: bias absent (optional slot unbound or empty)");
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(biasShape->GetDimNum() != 1U,
                OP_LOGE(opName,
                        "Conv3DTranspose InferShape failed: shape_mismatch, bias rank is %zu, "
                        "expect 1 (per-channel vector of length Cin)",
                        biasShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    // bias 含未知维（部分动态）时跳过长度比对，交由运行时复推
    if (HasUnknownDim(*biasShape)) {
        OP_LOGD(opName, "Conv3DTranspose InferShape: bias shape unknown, length check deferred");
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(biasShape->GetDim(0) != cin,
                OP_LOGE(opName,
                        "Conv3DTranspose InferShape failed: shape_mismatch, bias length %lld != "
                        "y channel Cin %lld (filter Cin dim * groups)",
                        static_cast<long long>(biasShape->GetDim(0)), static_cast<long long>(cin)),
                return ge::GRAPH_FAILED);
    OP_LOGD(opName, "Conv3DTranspose InferShape: bias length %lld matches Cin",
            static_cast<long long>(biasShape->GetDim(0)));
    return ge::GRAPH_SUCCESS;
}

// 静态形状与属性的一致性校验（design/InferShapeDtype.md「逐维对齐与几何一致性校验」）。
// 调用前提：input_size 维值已读取（yShape 已按其赋值），x/filter 形状全静态。
ge::graphStatus ValidateConv3DTransposeConsistency(gert::InferShapeContext* context,
                                                   const int64_t (&inSize)[kInputSizeLen])
{
    const char* opName = (context == nullptr) ? nullptr : context->GetNodeName();

    const gert::Shape* filterShape = context->GetInputShape(kFilterIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterShape);
    const gert::Shape* xShape = context->GetInputShape(kXIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // rank 契约：x/filter 恒为 5D（rank_range [5,5]）
    OP_CHECK_IF(filterShape->GetDimNum() != kRank5,
                OP_LOGE(opName,
                        "Conv3DTranspose InferShape failed: shape_mismatch, filter rank is %zu, "
                        "expect 5",
                        filterShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape->GetDimNum() != kRank5,
                OP_LOGE(opName,
                        "Conv3DTranspose InferShape failed: shape_mismatch, x rank is %zu, "
                        "expect 5",
                        xShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    // 空张量：任一输入含 0 维即 null_input（无静默空跑路径，TF 同族口径）
    for (size_t idx = 0; idx < kRank5; ++idx) {
        OP_CHECK_IF(
            filterShape->GetDim(idx) < 1 || xShape->GetDim(idx) < 1,
            OP_LOGE(opName,
                    "Conv3DTranspose InferShape failed: null_input, empty tensor detected, "
                    "dim %zu of filter/x is non-positive (filter %lld, x %lld)",
                    idx, static_cast<long long>(filterShape->GetDim(idx)), static_cast<long long>(xShape->GetDim(idx))),
            return ge::GRAPH_FAILED);
    }

    // 形状维可表示域上界（SEC-1.3 值域收窄，先于一切几何运算）：x/filter 各维 ≤ 2^31-1
    // —— Keff = dilation*(kernel-1)+1 等有符号乘加的安全前提（input_size 维值的上界
    // 已在 InferShapeConv3DTranspose 读取后同口径拒绝）
    for (size_t idx = 0; idx < kRank5; ++idx) {
        OP_CHECK_IF(filterShape->GetDim(idx) > kMaxDimValue || xShape->GetDim(idx) > kMaxDimValue,
                    OP_LOGE(opName,
                            "Conv3DTranspose InferShape failed: shape_mismatch, dim %zu of "
                            "filter/x exceeds representable range %lld (filter %lld, x %lld)",
                            idx, static_cast<long long>(kMaxDimValue), static_cast<long long>(filterShape->GetDim(idx)),
                            static_cast<long long>(xShape->GetDim(idx))),
                    return ge::GRAPH_FAILED);
    }

    // 属性解析与校验（八属性按 OpDef 声明序；offset_x/enable_hf32 保留不参与）
    TransposeAttrSet as = {};
    const ge::graphStatus attrRet = ParseTransposeAttrs(context, as);
    if (attrRet != ge::GRAPH_SUCCESS) {
        return attrRet;
    }

    // 语义维定位（值序/轴序随 data_format，见 layout_contract.variants）：
    //   NCDHW: input_size=[N,Cin,Din,Hin,Win]，x=(N,Cout,Dx,Hx,Wx)，空间属性位 strides[2:5]
    //   NDHWC: input_size=[N,Din,Hin,Win,Cin]，x=(N,Dx,Hx,Wx,Cout)，空间属性位 strides[1:4]
    const int64_t batch = inSize[0];
    const int64_t cin = as.isNdhwc ? inSize[4] : inSize[1];
    const int64_t yD = as.isNdhwc ? inSize[1] : inSize[2];
    const int64_t yH = as.isNdhwc ? inSize[2] : inSize[3];
    const int64_t yW = as.isNdhwc ? inSize[3] : inSize[4];

    // filter 语义维（布局独立于 data_format，按 desc origin format 解析）：
    //   - origin format 为三种注册布局（NCDHW/NDHWC/DHWCN）之一：单一解释，走既有校验；
    //   - origin format 为 ND（布局未声明）：按三种注册布局做唯一一致性消歧——通道对齐 +
    //     转置几何逐轴一致的候选必须恰好一个；零个（均不一致）或多个（歧义）均按
    //     shape_mismatch 响亮拒绝，绝不静默猜测布局。
    const auto* filterDesc = context->GetInputDesc(kFilterIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterDesc);
    FilterSemanticDims fDims = {0, 0, 0, 0, 0};
    const ge::Format filterFormat = filterDesc->GetOriginFormat();
    if (GetFilterSemanticDims(*filterShape, filterFormat, fDims)) {
        OP_LOGD(opName, "Conv3DTranspose InferShape: filter layout specified by origin format %d",
                static_cast<int32_t>(filterFormat));
    } else if (filterFormat == ge::FORMAT_ND) {
        FilterSemanticDims cands[3] = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
        MakeFilterCandidates(*filterShape, cands);
        const CandidateCtx candCtx = {cin,
                                      as.groups,
                                      xShape->GetDim(as.xChannelIdx),
                                      xShape->GetDim(as.spatialBase),
                                      xShape->GetDim(as.spatialBase + 1U),
                                      xShape->GetDim(as.spatialBase + 2U),
                                      yD,
                                      yH,
                                      yW,
                                      &as};
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
            OP_LOGE(opName, "Conv3DTranspose InferShape failed: shape_mismatch, filter layout "
                            "unspecified (origin format ND) and no registered layout interpretation is "
                            "consistent (NCDHW/NDHWC/DHWCN all violate channel/geometry consistency)");
            return ge::GRAPH_FAILED;
        }
        if (consistentCount > 1U) {
            OP_LOGE(opName,
                    "Conv3DTranspose InferShape failed: shape_mismatch, filter layout "
                    "unspecified (origin format ND) and interpretation is ambiguous (%zu registered "
                    "layouts consistent), refusing to guess",
                    consistentCount);
            return ge::GRAPH_FAILED;
        }
        fDims = cands[consistentIdx];
        OP_LOGD(opName,
                "Conv3DTranspose InferShape: filter layout unspecified (origin format ND), "
                "uniquely resolved to registered layout index %zu (0=NCDHW, 1=NDHWC, 2=DHWCN) by "
                "consistency disambiguation",
                consistentIdx);
        // 布局消歧唯一成立后，把消歧出的注册布局回写进本上下文的 filter desc
        // （CompileTimeTensorDesc 提供 SetOriginFormat/SetStorageFormat；经 const_cast 去除
        // GetInputDesc 的只读限定）。origin 与 storage 同值改写：filter 内存数据即按该布局
        // 生成，形状各维不变，仅布局标签由未声明（ND）改为真实注册布局。
        // 注：本改写不回传 ge::OpDesc（arena 为逐调用拷贝）——FE 引擎检查与 tiling 读到的
        // 仍由 op info 注册层决定；改写使 InferShape 上下文内消歧结论与 desc 标签一致，
        // 在 CANN 上下文语义变化（改写回传激活）时即为等价的注册布局承接；消歧不唯一的
        // 分支已在上方响亮拒绝，绝不猜测布局。
        static constexpr ge::Format kResolvedLayouts[3] = {ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_DHWCN};
        auto* mutableFilterDesc = const_cast<gert::CompileTimeTensorDesc*>(filterDesc);
        if (mutableFilterDesc != nullptr) {
            const ge::Format resolved = kResolvedLayouts[consistentIdx];
            mutableFilterDesc->SetOriginFormat(resolved);
            mutableFilterDesc->SetStorageFormat(resolved);
            OP_LOGD(opName,
                    "Conv3DTranspose InferShape: filter desc normalized to registered layout "
                    "%d (origin=storage) in infer context",
                    static_cast<int32_t>(resolved));
        }
    } else {
        OP_LOGE(opName,
                "Conv3DTranspose InferShape failed: attribute_value_out_of_range, filter origin "
                "format %d not in {NCDHW, NDHWC, DHWCN}",
                static_cast<int32_t>(filterFormat));
        return ge::GRAPH_FAILED;
    }

    // groups 与通道收缩对齐（指定布局路径既有校验；ND 消歧路径下已按候选判据成立）
    OP_CHECK_IF(cin % as.groups != 0 || fDims.cinPerGroup * as.groups != cin,
                OP_LOGE(opName,
                        "Conv3DTranspose InferShape failed: shape_mismatch, y channel Cin %lld "
                        "not divisible by groups %lld or filter Cin dim %lld * groups != Cin",
                        static_cast<long long>(cin), static_cast<long long>(as.groups),
                        static_cast<long long>(fDims.cinPerGroup)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(fDims.cout % as.groups != 0,
                OP_LOGE(opName,
                        "Conv3DTranspose InferShape failed: shape_mismatch, filter Cout %lld "
                        "not divisible by groups %lld",
                        static_cast<long long>(fDims.cout), static_cast<long long>(as.groups)),
                return ge::GRAPH_FAILED);

    // 批次一致：input_size[0] == x.shape[0]（两种格式 N 均为轴 0）
    OP_CHECK_IF(xShape->GetDim(0) != batch,
                OP_LOGE(opName,
                        "Conv3DTranspose InferShape failed: shape_mismatch, x batch %lld != "
                        "input_size batch %lld",
                        static_cast<long long>(xShape->GetDim(0)), static_cast<long long>(batch)),
                return ge::GRAPH_FAILED);

    // 通道收缩对齐：x 通道维（= 转置卷积输入的 Cout 侧）== filter 语义 Cout 维
    OP_CHECK_IF(xShape->GetDim(as.xChannelIdx) != fDims.cout,
                OP_LOGE(opName,
                        "Conv3DTranspose InferShape failed: shape_mismatch, x channel %lld != "
                        "filter Cout %lld",
                        static_cast<long long>(xShape->GetDim(as.xChannelIdx)), static_cast<long long>(fDims.cout)),
                return ge::GRAPH_FAILED);

    // 转置几何一致性（反算检查形式，逐轴；x 空间维必须逐一等于反算结果）
    const size_t b = as.spatialBase;
    const int64_t sD = as.strides[b];
    const int64_t sH = as.strides[b + 1U];
    const int64_t sW = as.strides[b + 2U];
    const int64_t dD = as.dilations[b];
    const int64_t dH = as.dilations[b + 1U];
    const int64_t dW = as.dilations[b + 2U];
    const int64_t qD = as.outputPadding[b];
    const int64_t qH = as.outputPadding[b + 1U];
    const int64_t qW = as.outputPadding[b + 2U];
    const int64_t expectedDIn = TransposeBackInputSize(yD, as.pads6[0], as.pads6[1], fDims.kD, dD, sD, qD);
    const int64_t expectedHIn = TransposeBackInputSize(yH, as.pads6[2], as.pads6[3], fDims.kH, dH, sH, qH);
    const int64_t expectedWIn = TransposeBackInputSize(yW, as.pads6[4], as.pads6[5], fDims.kW, dW, sW, qW);
    OP_CHECK_IF(expectedDIn < 1 || expectedDIn != xShape->GetDim(as.spatialBase),
                OP_LOGE(opName,
                        "Conv3DTranspose InferShape failed: shape_mismatch, transposed geometry "
                        "Din expected %lld but x dim is %lld (Dy=%lld, pads=[%lld,%lld], kD=%lld, dD=%lld, "
                        "sD=%lld, qD=%lld)",
                        static_cast<long long>(expectedDIn), static_cast<long long>(xShape->GetDim(as.spatialBase)),
                        static_cast<long long>(yD), static_cast<long long>(as.pads6[0]),
                        static_cast<long long>(as.pads6[1]), static_cast<long long>(fDims.kD),
                        static_cast<long long>(dD), static_cast<long long>(sD), static_cast<long long>(qD)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        expectedHIn < 1 || expectedHIn != xShape->GetDim(as.spatialBase + 1U),
        OP_LOGE(opName,
                "Conv3DTranspose InferShape failed: shape_mismatch, transposed geometry "
                "Hin expected %lld but x dim is %lld (Hy=%lld, pads=[%lld,%lld], kH=%lld, dH=%lld, "
                "sH=%lld, qH=%lld)",
                static_cast<long long>(expectedHIn), static_cast<long long>(xShape->GetDim(as.spatialBase + 1U)),
                static_cast<long long>(yH), static_cast<long long>(as.pads6[2]), static_cast<long long>(as.pads6[3]),
                static_cast<long long>(fDims.kH), static_cast<long long>(dH), static_cast<long long>(sH),
                static_cast<long long>(qH)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        expectedWIn < 1 || expectedWIn != xShape->GetDim(as.spatialBase + 2U),
        OP_LOGE(opName,
                "Conv3DTranspose InferShape failed: shape_mismatch, transposed geometry "
                "Win expected %lld but x dim is %lld (Wy=%lld, pads=[%lld,%lld], kW=%lld, dW=%lld, "
                "sW=%lld, qW=%lld)",
                static_cast<long long>(expectedWIn), static_cast<long long>(xShape->GetDim(as.spatialBase + 2U)),
                static_cast<long long>(yW), static_cast<long long>(as.pads6[4]), static_cast<long long>(as.pads6[5]),
                static_cast<long long>(fDims.kW), static_cast<long long>(dW), static_cast<long long>(sW),
                static_cast<long long>(qW)),
        return ge::GRAPH_FAILED);

    // bias 长度校验：bias.shape == (Cin,)（可选输入，按 y 通道广播）
    const ge::graphStatus biasRet = ValidateBiasLength(context, cin);
    if (biasRet != ge::GRAPH_SUCCESS) {
        return biasRet;
    }
    return ge::GRAPH_SUCCESS;
}

// 零输出哨兵：input_size 元素值全零时按属性公式推导 y（design/InferShapeDtype.md
// 「DynamicShape」属性驱动兜底，与内置 InferShapeForConv3DTransposeZeroOutput 同构）。
// 调用前提：input_size 五个元素值均为 0（常量数据可见），x/filter 形状全静态。
ge::graphStatus InferShapeSentinelByAttrs(gert::InferShapeContext* context, gert::Shape* yShape)
{
    const char* opName = (context == nullptr) ? nullptr : context->GetNodeName();
    const gert::Shape* filterShape = context->GetInputShape(kFilterIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterShape);
    const gert::Shape* xShape = context->GetInputShape(kXIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    TransposeAttrSet as = {};
    const ge::graphStatus attrRet = ParseTransposeAttrs(context, as);
    if (attrRet != ge::GRAPH_SUCCESS) {
        return attrRet;
    }

    const auto* filterDesc = context->GetInputDesc(kFilterIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterDesc);
    FilterSemanticDims fDims = {0, 0, 0, 0, 0};
    const ge::Format filterFormat = filterDesc->GetOriginFormat();
    if (GetFilterSemanticDims(*filterShape, filterFormat, fDims)) {
        OP_LOGD(opName,
                "Conv3DTranspose InferShape: zero-output sentinel, filter layout specified by "
                "origin format %d",
                static_cast<int32_t>(filterFormat));
    } else if (filterFormat == ge::FORMAT_ND) {
        // 布局未声明：按候选布局做通道对齐 + 派生维全 ≥1 的一致性消歧（y 未知，几何
        // 由候选自派生）；候选必须恰好一个，零个/多个均按 shape_mismatch 响亮拒绝
        FilterSemanticDims cands[3] = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
        MakeFilterCandidates(*filterShape, cands);
        size_t consistentIdx = 3U;
        size_t consistentCount = 0U;
        for (size_t ci = 0U; ci < 3U; ++ci) {
            if (cands[ci].cout % as.groups != 0 || xShape->GetDim(as.xChannelIdx) != cands[ci].cout) {
                continue;
            }
            int64_t yDims[kRank5] = {0, 0, 0, 0, 0};
            DeriveSentinelDims(cands[ci], as, *xShape, yDims);
            bool allPositive = true;
            for (size_t idx = 0; idx < kRank5; ++idx) {
                if (yDims[idx] < 1) {
                    allPositive = false;
                    break;
                }
            }
            if (allPositive) {
                if (consistentCount == 0U) {
                    consistentIdx = ci;
                }
                ++consistentCount;
            }
        }
        if (consistentCount != 1U) {
            OP_LOGE(opName,
                    "Conv3DTranspose InferShape failed: shape_mismatch, zero-output sentinel "
                    "with unspecified filter layout (origin format ND): %zu registered layout "
                    "interpretations consistent (expect exactly 1), refusing to guess",
                    consistentCount);
            return ge::GRAPH_FAILED;
        }
        fDims = cands[consistentIdx];
        OP_LOGD(opName,
                "Conv3DTranspose InferShape: zero-output sentinel, filter layout unspecified "
                "(origin format ND), uniquely resolved to registered layout index %zu",
                consistentIdx);
    } else {
        OP_LOGE(opName,
                "Conv3DTranspose InferShape failed: attribute_value_out_of_range, filter origin "
                "format %d not in {NCDHW, NDHWC, DHWCN}",
                static_cast<int32_t>(filterFormat));
        return ge::GRAPH_FAILED;
    }

    // 属性公式推导（rho=0）：N = x.shape[0]；Cin = filter 语义 Cin/G 维 × groups；
    // 逐空间轴 L_y = s*(L_x-1) + q + (d*(K-1)+1) - (p^- + p^+)
    int64_t yDims[kRank5] = {0, 0, 0, 0, 0};
    DeriveSentinelDims(fDims, as, *xShape, yDims);
    for (size_t idx = 0; idx < kRank5; ++idx) {
        OP_CHECK_IF(yDims[idx] == 0,
                    OP_LOGE(opName,
                            "Conv3DTranspose InferShape failed: null_input, zero-output sentinel "
                            "derived y dim %zu is 0 (empty tensor, no silent empty-run path)",
                            idx),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(yDims[idx] < 0,
                    OP_LOGE(opName,
                            "Conv3DTranspose InferShape failed: shape_mismatch, zero-output "
                            "sentinel derived y dim %zu is %lld (negative dim value)",
                            idx, static_cast<long long>(yDims[idx])),
                    return ge::GRAPH_FAILED);
    }
    yShape->SetDimNum(kRank5);
    for (size_t idx = 0; idx < kRank5; ++idx) {
        yShape->SetDim(idx, yDims[idx]);
    }

    // bias 长度校验（Cin 为派生值）
    const int64_t cin = fDims.cinPerGroup * as.groups;
    const ge::graphStatus biasRet = ValidateBiasLength(context, cin);
    if (biasRet != ge::GRAPH_SUCCESS) {
        return biasRet;
    }
    OP_LOGD(opName,
            "Conv3DTranspose InferShape: zero-output sentinel, y shape derived from attributes "
            "(N=%lld, C=%lld, D=%lld, H=%lld, W=%lld)",
            static_cast<long long>(yShape->GetDim(0)), static_cast<long long>(yShape->GetDim(1)),
            static_cast<long long>(yShape->GetDim(2)), static_cast<long long>(yShape->GetDim(3)),
            static_cast<long long>(yShape->GetDim(4)));
    return ge::GRAPH_SUCCESS;
}
// input_size 维值校验（负值 / 超可表示域上界 / 零维清点；SEC-1.3 值域收窄）。
// 主路径与"常量数据可见 + filter 形状未知"的动态路径共用：维值一经可读，
// 值级契约立即生效，不因几何信息缺失而推迟（空张量/负维/超域值无静默路径）。
// 返回 ge::GRAPH_SUCCESS 表示五个维值全部合法（zeroCount 回填零值个数）。
ge::graphStatus ValidateInputSizeDimValues(gert::InferShapeContext* context, const int64_t (&inSize)[kInputSizeLen],
                                           size_t& zeroCount)
{
    zeroCount = 0U;
    for (size_t idx = 0; idx < kInputSizeLen; ++idx) {
        OP_CHECK_IF(inSize[idx] < 0,
                    OP_LOGE(context->GetNodeName(),
                            "Conv3DTranspose InferShape failed: shape_mismatch, "
                            "input_size[%zu] = %lld is negative",
                            idx, static_cast<long long>(inSize[idx])),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(inSize[idx] > kMaxDimValue,
                    OP_LOGE(context->GetNodeName(),
                            "Conv3DTranspose InferShape failed: shape_mismatch, "
                            "input_size[%zu] = %lld exceeds representable range %lld",
                            idx, static_cast<long long>(inSize[idx]), static_cast<long long>(kMaxDimValue)),
                    return ge::GRAPH_FAILED);
        if (inSize[idx] == 0) {
            ++zeroCount;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 属性校验入口（动态图补齐）：属性在图编译期恒可见（与形状是否 -1 无关），
// 任何属性违约（strides/pads/dilations/groups/data_format/output_padding 的
// 位序、值域、逐轴 q<s 或 q<d 约束）先于一切"未知维早退"响亮拒绝——
// 动态图编译期不再放过非法属性（geir-run-100pct 修复：src-dyn 组合下
// data_format 违约行曾因早退未校验而静默执行）。
ge::graphStatus ValidateAttrsEarly(gert::InferShapeContext* context)
{
    TransposeAttrSet probe = {};
    return ParseTransposeAttrs(context, probe);
}
} // namespace

// InferShape{OpType}：y.shape = input_size 元素值（值依赖推导 + 一致性校验）
static ge::graphStatus InferShapeConv3DTranspose(gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    gert::Shape* yShape = context->GetOutputShape(kOutputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    const gert::Tensor* inputSizeTensor = context->GetInputTensor(kInputSizeIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSizeTensor);

    // 属性校验前移（动态图补齐）：见 ValidateAttrsEarly 注释
    {
        const ge::graphStatus earlyAttrRet = ValidateAttrsEarly(context);
        if (earlyAttrRet != ge::GRAPH_SUCCESS) {
            return earlyAttrRet;
        }
    }

    // unknown rank（[-2]）/ unknown shape（含 -1）：输出全 -1，rank 固定 5（与内置基线同口径）
    // F-01 整改：shape 元数据取自 context（GetInputShape），不经张量对象；inputSizeTensor
    // 保留用于后续合法值依赖数据读取（ReadInputSizeValues 的 GetData）。
    const gert::Shape* inputSizeShapePtr = context->GetInputShape(kInputSizeIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSizeShapePtr);
    const gert::Shape& inputSizeShape = *inputSizeShapePtr;
    if (HasUnknownDim(inputSizeShape)) {
        SetUnknownShape(*yShape);
        OP_LOGD(context->GetNodeName(), "Conv3DTranspose InferShape: input_size unknown, y shape set to "
                                        "all -1 (rank 5)");
        return ge::GRAPH_SUCCESS;
    }

    // input_size 接口契约：1D 且长度 5（REQUIREMENTS §8.2.1；4 元 2D 升维路径不在本
    // ACLNN 契约内，违反报 shape_mismatch）
    OP_CHECK_IF(inputSizeShape.GetDimNum() != 1U || static_cast<size_t>(inputSizeShape.GetShapeSize()) != kInputSizeLen,
                OP_LOGE(context->GetNodeName(),
                        "Conv3DTranspose InferShape failed: shape_mismatch, "
                        "input_size must be 1D with length 5, got rank %zu, element num %lld",
                        inputSizeShape.GetDimNum(), static_cast<long long>(inputSizeShape.GetShapeSize())),
                return ge::GRAPH_FAILED);

    const gert::Shape* filterShape = context->GetInputShape(kFilterIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterShape);
    const gert::Shape* xShape = context->GetInputShape(kXIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // 动态场景：filter 形状未知（图编译期占位输入）时，几何一致性无法检查；
    // 但 input_size 常量数据一经可见（Const 值张量，静态/动态图编译期均可见），
    // 维值级契约立即生效：dtype/负值/超域/零维先行校验，y 直接按元素值落形
    // （与内置基线值依赖口径一致），几何一致性交由运行时复推
    // （geir-run-100pct 修复：src-dyn 组合下空张量行曾因早退未读值而静默执行）。
    if (HasUnknownDim(*filterShape) || !IsConstDataVisible(*inputSizeTensor)) {
        if (IsConstDataVisible(*inputSizeTensor)) {
            int64_t inSizeEarly[kInputSizeLen] = {0, 0, 0, 0, 0};
            OP_CHECK_IF(!ReadInputSizeValues(*inputSizeTensor, inSizeEarly),
                        OP_LOGE(context->GetNodeName(),
                                "Conv3DTranspose InferShape failed: dtype_not_supported, "
                                "input_size dtype %d not in {DT_INT32, DT_INT64}",
                                static_cast<int32_t>(inputSizeTensor->GetDataType())),
                        return ge::GRAPH_FAILED);
            size_t zeroCountEarly = 0U;
            const ge::graphStatus dimRet = ValidateInputSizeDimValues(context, inSizeEarly, zeroCountEarly);
            if (dimRet != ge::GRAPH_SUCCESS) {
                return dimRet;
            }
            // 零输出哨兵（全零）：属性驱动兜底需要 filter 形状，未知时交运行时复推
            if (zeroCountEarly == kInputSizeLen) {
                SetUnknownShape(*yShape);
                OP_LOGD(context->GetNodeName(), "Conv3DTranspose InferShape: zero-output sentinel with "
                                                "unknown filter shape, y shape set to all -1 (rank 5)");
                return ge::GRAPH_SUCCESS;
            }
            // 元素值为 0（非全零哨兵）：该维即空张量（null_input，无静默空跑路径）
            for (size_t idx = 0; idx < kInputSizeLen; ++idx) {
                OP_CHECK_IF(inSizeEarly[idx] == 0,
                            OP_LOGE(context->GetNodeName(),
                                    "Conv3DTranspose InferShape failed: null_input, "
                                    "input_size[%zu] = 0 (empty tensor, no silent empty-run path)",
                                    idx),
                            return ge::GRAPH_FAILED);
            }
            // 维值合法：y.shape = input_size 元素值（data_dependent_shape）
            yShape->SetDimNum(kRank5);
            for (size_t idx = 0; idx < kRank5; ++idx) {
                yShape->SetDim(idx, inSizeEarly[idx]);
            }
            OP_LOGD(context->GetNodeName(),
                    "Conv3DTranspose InferShape: filter shape unknown, y shape set "
                    "from input_size values (N=%lld, C=%lld, D=%lld, H=%lld, W=%lld), geometry deferred",
                    static_cast<long long>(yShape->GetDim(0)), static_cast<long long>(yShape->GetDim(1)),
                    static_cast<long long>(yShape->GetDim(2)), static_cast<long long>(yShape->GetDim(3)),
                    static_cast<long long>(yShape->GetDim(4)));
            return ge::GRAPH_SUCCESS;
        }
        SetUnknownShape(*yShape);
        OP_LOGD(context->GetNodeName(), "Conv3DTranspose InferShape: input_size const data invisible or "
                                        "filter shape unknown, y shape set to all -1 (rank 5)");
        return ge::GRAPH_SUCCESS;
    }

    // 读取 5 个维值（int32/int64；其他 dtype 违反形状描述符契约）
    int64_t inSize[kInputSizeLen] = {0, 0, 0, 0, 0};
    OP_CHECK_IF(!ReadInputSizeValues(*inputSizeTensor, inSize),
                OP_LOGE(context->GetNodeName(),
                        "Conv3DTranspose InferShape failed: dtype_not_supported, "
                        "input_size dtype %d not in {DT_INT32, DT_INT64}",
                        static_cast<int32_t>(inputSizeTensor->GetDataType())),
                return ge::GRAPH_FAILED);

    // 维值边界：负值为非法维（shape_mismatch）；超可表示域上界拒绝（shape_mismatch，
    // SEC-1.3 值域收窄——y 形状与后续几何运算不携带超域值）
    size_t zeroCount = 0U;
    {
        const ge::graphStatus dimRet = ValidateInputSizeDimValues(context, inSize, zeroCount);
        if (dimRet != ge::GRAPH_SUCCESS) {
            return dimRet;
        }
    }

    // 零输出哨兵（input_size 元素值全零，内置口径 CheckOutputAllZero）：属性驱动兜底，
    // y.shape 不由元素值直出（REQUIREMENTS §5.6/§8.1）；x/filter 形状未知时交运行时复推
    if (zeroCount == kInputSizeLen) {
        if (HasUnknownDim(*filterShape) || HasUnknownDim(*xShape)) {
            SetUnknownShape(*yShape);
            OP_LOGD(context->GetNodeName(), "Conv3DTranspose InferShape: zero-output sentinel with "
                                            "dynamic x/filter shape, y shape set to all -1 (rank 5)");
            return ge::GRAPH_SUCCESS;
        }
        return InferShapeSentinelByAttrs(context, yShape);
    }

    // 元素值为 0（非全零哨兵）：该维即空张量（null_input，无静默空跑路径）
    for (size_t idx = 0; idx < kInputSizeLen; ++idx) {
        OP_CHECK_IF(inSize[idx] == 0,
                    OP_LOGE(context->GetNodeName(),
                            "Conv3DTranspose InferShape failed: null_input, "
                            "input_size[%zu] = 0 (empty tensor, no silent empty-run path)",
                            idx),
                    return ge::GRAPH_FAILED);
    }

    // 输出 shape = input_size 元素值（data_dependent_shape；值序随 data_format 由校验段消费）
    yShape->SetDimNum(kRank5);
    for (size_t idx = 0; idx < kRank5; ++idx) {
        yShape->SetDim(idx, inSize[idx]);
    }

    // 静态场景逐维一致性校验；x/filter 含未知维（部分动态）时跳过，交由运行时复推
    if (!HasUnknownDim(*filterShape) && !HasUnknownDim(*xShape)) {
        const ge::graphStatus ret = ValidateConv3DTransposeConsistency(context, inSize);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }

    OP_LOGD(context->GetNodeName(),
            "Conv3DTranspose InferShape: y shape set from input_size values "
            "(N=%lld, C=%lld, D=%lld, H=%lld, W=%lld)",
            static_cast<long long>(yShape->GetDim(0)), static_cast<long long>(yShape->GetDim(1)),
            static_cast<long long>(yShape->GetDim(2)), static_cast<long long>(yShape->GetDim(3)),
            static_cast<long long>(yShape->GetDim(4)));
    return ge::GRAPH_SUCCESS;
}

// 注册：IMPL_OP_INFERSHAPE({OpType}).InferShape(InferShape{OpType})，声明输入 0 值依赖
// （InferShape 读取 input_size 元素值推导 y 形状，REQUIREMENTS §5.6）
IMPL_OP_INFERSHAPE(Conv3DTranspose).InferShape(InferShapeConv3DTranspose).InputsDataDependency({0});

} // namespace ops
