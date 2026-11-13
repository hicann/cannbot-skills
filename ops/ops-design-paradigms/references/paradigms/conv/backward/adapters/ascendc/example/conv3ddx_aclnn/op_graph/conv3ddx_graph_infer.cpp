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
 * \file conv3ddx_graph_infer.cpp
 * \brief Conv3ddx 图模式推导注册（GEIR OpType Conv3ddx）
 *
 * 事实源：design/InferShapeDtype.md「InferDataType」节（spec.outputs.y.dtype_rule：
 * y.dtype = np.promote_types(filter.dtype, out_backprop.dtype)）。
 * 受支持组合内 filter 与 out_backprop 恒为同一 dtype，上式退化为
 * y.dtype = out_backprop.dtype（REQUIREMENTS §8.2.1：y 的 dtype/格式随 outBackprop）；
 * 混合 dtype 配对不做隐式提升，直接按失败判据拒绝（dtype_not_supported）。
 * 本文件编入 libcust_opsproto_rt2.0.so（cust_op_proto），GE 图编译经
 * ASCEND_CUSTOM_OPP_PATH 扫描加载时完成 IMPL_OP(Conv3ddx).InferDataType 注册。
 *
 * InferFormat / OpSelectFormat 注册（本文件，GE 图编译注册表装载）：
 * filter 布局取 desc origin format；TTK GEIR 图构建器无法表达 DHWCN（无映射，
 * 静默落为 FORMAT_ND）。fix-2 三通道实测（plog-fix2/DECISIVE_DIGEST.md）：
 * InferFormat 在本图拓扑（单算子 + Data 输入 + NetOutput 未锁形）不可达；
 * OpSelectFormat 的 JSON 仅被 FE IsNeedSkipOpJudge 消费（tileFwkOp 键查询，
 * 缺键告警后按非 tile-framework op 继续，不影响任何门禁），不参与格式选择；
 * InferShape 上下文改写不回传 ge::OpDesc。有效层是 op info 注册表本身——
 * 由 op/op_host/CMakeLists.txt 的 fix-3 包构建接线后处理 autogen ini
 * （DHWCN 优先组合序 + 2 个 ND 接受组合）实现：ND filter 经 FE 组合匹配
 * 回退归一化为首个注册布局（DHWCN），归一化正确性由 InferShape 唯一消歧
 * （shape_mismatch 响亮拒绝）与 tiling 几何校验（E90003 响亮拒绝）守卫。
 * 本文件两注册保留：判据与接线同源（唯一一致性消歧），在 CANN 调用链变化
 * （InferFormat 可达拓扑 / select-format 路径激活）时即可直接承接布局归一
 * 化，行为与接线一致；当前执行路径上二者均无行为影响（fix-3 零回归实测）。
 */

#include <cstdint>
#include <string>

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "exe_graph/runtime/exe_res_generation_context.h"
#include "graph/infer_format_context.h"
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

// 属性声明序（OpDef：strides/pads/dilations/groups/data_format/enable_hf32）
constexpr size_t kAttrStrides = 0U;
constexpr size_t kAttrPads = 1U;
constexpr size_t kAttrDilations = 2U;
constexpr size_t kAttrGroups = 3U;
constexpr size_t kAttrDataFormat = 4U;
constexpr size_t kStridesLen = 5U;
constexpr size_t kPadsLen = 6U;
constexpr size_t kDilationsLen = 5U;

// 参与几何运算的外部值可表示域上界（SEC-1.3：先收窄再运算；与 op_host/
// conv3ddx_infershape.cpp 同口径）：strides/dilations/pads/groups/形状维均限制在
// [0/1, 2^31-1]，使 FilterCandidateConsistent/ForwardOutSize 的有符号中间量恒 < 2^62。
// 本文件两处消费点（InferFormat/OpSelectFormat）为非破坏性路径：值越界时不改写任何
// format/列表，保持默认返回——响亮拒绝由 InferShape（同判据值域校验）执行。
constexpr int64_t kMaxDimValue = 2147483647;

// 值域守卫：input_size 维值 / filter·dedy·y 形状维 / strides / dilations / pads /
// groups 全部落在运算安全域内才允许进入候选一致性消歧（乘加溢出 UB 防护）
bool ExternalValuesInRange(const int64_t (&inSize)[kInputSizeLen], const gert::Shape& filterShape,
                           const gert::Shape& dedyShape, const int64_t* strides, const int64_t* dilations,
                           const int64_t* pads, const int64_t groups)
{
    for (size_t idx = 0; idx < kInputSizeLen; ++idx) {
        if (inSize[idx] > kMaxDimValue) {
            return false;
        }
    }
    for (size_t idx = 0; idx < filterShape.GetDimNum(); ++idx) {
        if (filterShape.GetDim(idx) > kMaxDimValue) {
            return false;
        }
    }
    for (size_t idx = 0; idx < dedyShape.GetDimNum(); ++idx) {
        if (dedyShape.GetDim(idx) > kMaxDimValue) {
            return false;
        }
    }
    for (size_t idx = 0; idx < kStridesLen; ++idx) {
        if (strides[idx] < 1 || strides[idx] > kMaxDimValue || dilations[idx] < 1 || dilations[idx] > kMaxDimValue) {
            return false;
        }
    }
    for (size_t idx = 0; idx < kPadsLen; ++idx) {
        if (pads[idx] < 0 || pads[idx] > kMaxDimValue) {
            return false;
        }
    }
    return groups >= 1 && groups <= kMaxDimValue;
}

// ---- 以下消歧判据与 op_host/conv3ddx_infershape.cpp（fix-1）逐条同源 ----

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

// filter 语义维（规范序 Cout/CinG/kD/kH/kW）；三种注册布局的候选解释（枚举序
// NCDHW / NDHWC / DHWCN，与消歧结果到 ge::Format 的映射一一对应）
struct FilterSemanticDims {
    int64_t cout;
    int64_t cinPerGroup;
    int64_t kD;
    int64_t kH;
    int64_t kW;
};

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
} // namespace

// InferDataTypeFor{OpType}：y.dtype = out_backprop.dtype（等 dtype 配对下与
// promote_types(filter.dtype, out_backprop.dtype) 同值）；混合配对直接拒绝
static ge::graphStatus InferDataTypeForConv3ddx(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const ge::DataType filterDtype = context->GetInputDataType(kFilterIdx);
    const ge::DataType outBackpropDtype = context->GetInputDataType(kOutBackpropIdx);

    // 等配对约束：混合 dtype（如 fp16 + bf16）不提升、不插 cast，按失败判据拒绝
    OP_CHECK_IF(filterDtype != outBackpropDtype,
                OP_LOGE(context->GetNodeName(),
                        "Conv3ddx InferDataType failed: dtype_not_supported, filter "
                        "dtype %d does not pair with out_backprop dtype %d (equal-dtype pairing required, no "
                        "implicit promotion)",
                        static_cast<int32_t>(filterDtype), static_cast<int32_t>(outBackpropDtype)),
                return ge::GRAPH_FAILED);

    // input_size（int32/int64）为形状描述符，不参与浮点 dtype 推导；
    // 内部 FP32 L0C 累加是计算精度约定，不改变输出 dtype 结论
    const ge::graphStatus ret = context->SetOutputDataType(kOutputIdx, outBackpropDtype);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(),
                        "Conv3ddx InferDataType failed: SetOutputDataType for y returned "
                        "%d",
                        static_cast<int32_t>(ret)),
                return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Conv3ddx InferDataType: y dtype %d set from out_backprop",
            static_cast<int32_t>(outBackpropDtype));
    return ge::GRAPH_SUCCESS;
}

// 图模式推导注册：IMPL_OP({OpType}).InferDataType(InferDataTypeFor{OpType})
IMPL_OP(Conv3ddx).InferDataType(InferDataTypeForConv3ddx);

// InferFormatFor{OpType}：filter 布局未声明（origin format ND，如 TTK GEIR 构建器
// 无法表达 DHWCN 而落为 ND）时的注册布局归一化。改写仅发生在「消歧唯一成立」时；
// 其余一切场景（指定布局、非 ND 未声明布局、消歧零候选/歧义、数据不可见、属性
// 异常）一律不动任何 format 并返回成功——拒绝语义仍由 InferShape 响亮报出。
// 注（fix-2/fix-3 实测）：本函数经 IMPL_OP 注册装载成功，但在当前图拓扑下 GE
// format_refiner 不可达（BackInferProcess 跳过 ND 输入、AnchorsInferProcess 需
// 下游锁形）；现行 ND→注册布局归一化由 op info 注册层的 fix-3 构建接线承担
// （见文件头注释）。本实现保留为拓扑可达 / 调用链激活时的等价承接路径。
static ge::graphStatus InferFormatForConv3ddx(gert::InferFormatContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    gert::StorageFormat* filterFmt = context->GetInputFormat(kFilterIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterFmt);

    const ge::Format filterOrigin = filterFmt->GetOriginFormat();
    // 指定布局：三种注册布局之一，行为保持——不改动
    if (filterOrigin == ge::FORMAT_NCDHW || filterOrigin == ge::FORMAT_NDHWC || filterOrigin == ge::FORMAT_DHWCN) {
        return ge::GRAPH_SUCCESS;
    }
    // 未声明且非 ND 的布局（如 NCHW）：不改动；InferShape 按 attribute_value_out_of_range 拒绝
    if (filterOrigin != ge::FORMAT_ND) {
        return ge::GRAPH_SUCCESS;
    }

    // ---- origin format 为 ND：唯一一致性消歧（判据与 infershape fix-1 同源） ----
    // 前置载体齐全性：任一缺失/异常即保持 ND 不动（InferShape 按各自判据响亮拒绝）
    const gert::Shape* filterShape = context->GetInputShape(kFilterIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterShape);
    const gert::Shape* dedyShape = context->GetInputShape(kOutBackpropIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, dedyShape);
    const gert::Tensor* inputSizeTensor = context->GetInputTensor(kInputSizeIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSizeTensor);

    // rank 契约 / 未知维：不改动（InferShape 按 shape_mismatch 或走动态未知路径）
    if (filterShape->GetDimNum() != kRank5 || dedyShape->GetDimNum() != kRank5) {
        return ge::GRAPH_SUCCESS;
    }
    if (HasUnknownDim(*filterShape) || HasUnknownDim(*dedyShape)) {
        return ge::GRAPH_SUCCESS;
    }
    // input_size 常量数据不可见（图编译期占位输入）：不改动（InferShape 走值依赖动态路径）
    if (inputSizeTensor->GetAddr() == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    // input_size 接口契约：1D 且长度 5；违反不改动（InferShape 按 shape_mismatch 拒绝）
    const gert::Shape& inputSizeShape = inputSizeTensor->GetOriginShape();
    if (inputSizeShape.GetDimNum() != 1U || static_cast<size_t>(inputSizeShape.GetShapeSize()) != kInputSizeLen) {
        return ge::GRAPH_SUCCESS;
    }

    // 维值读取：dtype 不在 {DT_INT32, DT_INT64} 或任一维值非正（空张量/负维）
    // 不改动（InferShape 按 dtype_not_supported / null_input / shape_mismatch 拒绝）
    int64_t inSize[kInputSizeLen] = {0, 0, 0, 0, 0};
    if (!ReadInputSizeValues(*inputSizeTensor, inSize)) {
        return ge::GRAPH_SUCCESS;
    }
    for (size_t idx = 0; idx < kInputSizeLen; ++idx) {
        if (inSize[idx] < 1) {
            return ge::GRAPH_SUCCESS;
        }
    }
    // 属性载体（六属性按 OpDef 声明序）：异常即不改动（InferShape 按
    // attribute_value_out_of_range / shape_mismatch 响亮拒绝）
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* stridesVec = attrs->GetListInt(kAttrStrides);
    const auto* padsVec = attrs->GetListInt(kAttrPads);
    const auto* dilationsVec = attrs->GetListInt(kAttrDilations);
    const int64_t* groupsPtr = attrs->GetInt(kAttrGroups);
    const char* dataFormatRaw = attrs->GetStr(kAttrDataFormat);
    if (stridesVec == nullptr || padsVec == nullptr || dilationsVec == nullptr || groupsPtr == nullptr ||
        dataFormatRaw == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (stridesVec->GetSize() != kStridesLen || padsVec->GetSize() != kPadsLen ||
        dilationsVec->GetSize() != kDilationsLen) {
        return ge::GRAPH_SUCCESS;
    }
    const std::string dataFormat(dataFormatRaw);
    bool isNdhwc = false;
    if (dataFormat == "NCDHW") {
        isNdhwc = false;
    } else if (dataFormat == "NDHWC") {
        isNdhwc = true;
    } else {
        return ge::GRAPH_SUCCESS; // data_format 非法：不改动（InferShape 拒绝）
    }
    const int64_t groups = *groupsPtr;
    if (groups < 1) {
        return ge::GRAPH_SUCCESS; // groups 非法：不改动（InferShape 拒绝）
    }
    const int64_t* strides = stridesVec->GetData();
    const int64_t* dilations = dilationsVec->GetData();
    const int64_t* pads = padsVec->GetData();
    // SEC-1.3 值域守卫：任一外部值超 [0/1, 2^31-1] 运算安全域即不进入候选消歧
    // （保持 ND 不动；InferShape 同判据值域校验将响亮拒绝，避免有符号乘加溢出 UB）
    if (!ExternalValuesInRange(inSize, *filterShape, *dedyShape, strides, dilations, pads, groups)) {
        return ge::GRAPH_SUCCESS;
    }
    // strides/dilations N、C 位非 1：不改动（InferShape 按 attribute_value_out_of_range 拒绝）
    const size_t nPos = 0U;
    const size_t cPos = isNdhwc ? 4U : 1U;
    if (strides[nPos] != 1 || strides[cPos] != 1 || dilations[nPos] != 1 || dilations[cPos] != 1) {
        return ge::GRAPH_SUCCESS;
    }

    // 语义维定位（值序/轴序随 data_format，与 infershape 同口径）
    const int64_t cin = isNdhwc ? inSize[4] : inSize[1];
    const int64_t dIn = isNdhwc ? inSize[1] : inSize[2];
    const int64_t hIn = isNdhwc ? inSize[2] : inSize[3];
    const int64_t wIn = isNdhwc ? inSize[3] : inSize[4];
    const size_t spatialBase = isNdhwc ? 1U : 2U;
    const size_t dedyChannelIdx = isNdhwc ? 4U : 1U;
    // 批次一致前置：input_size[0] != out_backprop.shape[0] 时不改动（InferShape 拒绝）
    if (dedyShape->GetDim(0) != inSize[0]) {
        return ge::GRAPH_SUCCESS;
    }

    // 三候选一致性计数（通道对齐 + 前向几何逐轴）
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

    // 消歧不唯一（零候选或歧义）：保持 ND 不动——InferShape 按 shape_mismatch 响亮拒绝
    if (consistentCount != 1U) {
        return ge::GRAPH_SUCCESS;
    }

    // 唯一一致：归一化为该注册布局（origin 与 storage 同值改写；filter 的内存数据
    // 即按该布局生成，形状各维不变，仅布局标签由未声明改为真实注册布局）
    static constexpr ge::Format kRegisteredLayouts[3] = {ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_DHWCN};
    const ge::Format resolved = kRegisteredLayouts[consistentIdx];
    filterFmt->SetOriginFormat(resolved);
    filterFmt->SetStorageFormat(resolved);
    OP_LOGD(context->GetNodeName(),
            "Conv3ddx InferFormat: filter layout unspecified (origin format ND), "
            "normalized to registered layout %d by unique consistency disambiguation",
            static_cast<int32_t>(resolved));
    return ge::GRAPH_SUCCESS;
}

// 图模式 format 推导注册：IMPL_OP({OpType}).InferFormat(InferFormatFor{OpType})。
// 输入 0 值依赖声明与 infershape 注册一致（format 上下文构建据此绑定 input_size
// 常量数据，消歧需要读取其元素值）。
IMPL_OP(Conv3ddx).InferFormat(InferFormatForConv3ddx).InputsDataDependency({0});

// OpSelectFormatFor{OpType}：FE 图编译 Setcheck 阶段按注册表调用的格式选择函数。
// 与 InferFormat 同判据地解析 filter 布局：input_size 元素值在本上下文不可见，
// 但 y 的 origin shape 恒等于 input_size 元素值（本算子 InferShape 的输出契约，
// 且 TTK 建图时 output desc shape 亦取自同一值），据此完成同一套唯一一致性消歧。
// 返回的 JSON 为算子支持的 dtype/format 列表（dtype 拼写按 FE String2DataType
// 契约：串 toupper 后拼 "DT_" 走 SerialStringToDataType——float16→DT_FLOAT16、
// bf16→DT_BF16；format 同为 serial 拼写）：
//   - filter 指定布局（NCDHW/NDHWC/DHWCN）：仅回显该布局；
//   - filter 布局未声明（ND）且消歧唯一：仅返回消歧出的注册布局；
//   - filter 布局未声明且消歧不唯一（零候选/歧义/形状未知——动态编译）或载体
//     异常：返回全部三种注册布局（规范序 NCDHW 优先）——不猜测、不缩小 FE 的
//     选择域，错误布局由 tiling 几何一致性响亮拒绝。
// 其余输入/输出回显注册集合：input_size 恒 ND；out_backprop/y 为 {NCDHW, NDHWC}
// （与 OpDef 组合一致，y 与 out_backprop 严格配对的语义由列表内容保持）。
// 注（fix-2/fix-3 实测）：本 JSON 在当前 FE 栈仅被 IsNeedSkipOpJudge 的
// tileFwkOp 键查询消费（缺键 → 告警 + 按非 tile-framework op 继续，无门禁影响），
// 不驱动格式改写；ND 归一化由 fix-3 注册接线承担。本实现保留为 select-format
// 路径激活时的等价承接（其候选序 NCDHW 优先与本接线 DHWCN 优先的差异仅在
// 「消歧不唯一的动态编译」域上出现，该域由 InferShape 先行拒绝，不出现在
// 受支持组合的执行路径上）。
static ge::graphStatus OpSelectFormatForConv3ddx(const gert::OpCheckContext* context, ge::AscendString& result)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);

    // filter 支持布局列表（默认：三种注册布局 × 2 dtype，规范序 NCDHW 优先）
    std::string filterFormats = "NCDHW,NCDHW,NDHWC,NDHWC,DHWCN,DHWCN";
    std::string filterDtypes = "float16,bf16,float16,bf16,float16,bf16";

    const auto* filterDesc = context->GetInputDesc(kFilterIdx);
    const gert::StorageShape* filterShape = context->GetInputShape(static_cast<int64_t>(kFilterIdx));
    const gert::StorageShape* dedyShape = context->GetInputShape(static_cast<int64_t>(kOutBackpropIdx));
    const gert::StorageShape* yShape = context->GetOutputShape(0);
    if (filterDesc != nullptr && filterShape != nullptr && dedyShape != nullptr && yShape != nullptr) {
        const ge::Format filterOrigin = filterDesc->GetOriginFormat();
        const char* resolved = nullptr;
        if (filterOrigin == ge::FORMAT_NCDHW) {
            resolved = "NCDHW";
        } else if (filterOrigin == ge::FORMAT_NDHWC) {
            resolved = "NDHWC";
        } else if (filterOrigin == ge::FORMAT_DHWCN) {
            resolved = "DHWCN";
        } else if (filterOrigin == ge::FORMAT_ND) {
            // 布局未声明：唯一一致性消歧（判据与 infershape fix-1 同源；
            // cin/Din/Hin/Win 取自 y 的 origin shape —— 恒等于 input_size 元素值）
            const gert::Shape& fShape = filterShape->GetOriginShape();
            const gert::Shape& dShape = dedyShape->GetOriginShape();
            const gert::Shape& yOriShape = yShape->GetOriginShape();
            bool canDisambiguate =
                (fShape.GetDimNum() == kRank5 && dShape.GetDimNum() == kRank5 && yOriShape.GetDimNum() == kRank5 &&
                 !HasUnknownDim(fShape) && !HasUnknownDim(dShape) && !HasUnknownDim(yOriShape));
            if (canDisambiguate) {
                for (size_t idx = 0; idx < kRank5 && canDisambiguate; ++idx) {
                    canDisambiguate = fShape.GetDim(idx) > 0 && dShape.GetDim(idx) > 0 && yOriShape.GetDim(idx) > 0;
                }
            }
            if (canDisambiguate) {
                const gert::RuntimeAttrs* attrs = context->GetAttrs();
                if (attrs != nullptr) {
                    const auto* stridesVec = attrs->GetListInt(kAttrStrides);
                    const auto* padsVec = attrs->GetListInt(kAttrPads);
                    const auto* dilationsVec = attrs->GetListInt(kAttrDilations);
                    const int64_t* groupsPtr = attrs->GetInt(kAttrGroups);
                    const char* dataFormatRaw = attrs->GetStr(kAttrDataFormat);
                    if (stridesVec != nullptr && padsVec != nullptr && dilationsVec != nullptr &&
                        groupsPtr != nullptr && dataFormatRaw != nullptr && stridesVec->GetSize() == kStridesLen &&
                        padsVec->GetSize() == kPadsLen && dilationsVec->GetSize() == kDilationsLen) {
                        const std::string dataFormat(dataFormatRaw);
                        bool isNdhwc = false;
                        bool formatOk = false;
                        if (dataFormat == "NCDHW") {
                            isNdhwc = false;
                            formatOk = true;
                        } else if (dataFormat == "NDHWC") {
                            isNdhwc = true;
                            formatOk = true;
                        }
                        const int64_t groups = *groupsPtr;
                        const int64_t* strides = stridesVec->GetData();
                        const int64_t* dilations = dilationsVec->GetData();
                        const int64_t* pads = padsVec->GetData();
                        const size_t nPos = 0U;
                        const size_t cPos = isNdhwc ? 4U : 1U;
                        if (formatOk && groups >= 1 && strides[nPos] == 1 && strides[cPos] == 1 &&
                            dilations[nPos] == 1 && dilations[cPos] == 1 && dShape.GetDim(0) == yOriShape.GetDim(0)) {
                            // SEC-1.3 值域守卫：y 形状维（≡ input_size 元素值）/ filter·
                            // dedy 形状维 / strides / dilations / pads / groups 任一超出
                            // [0/1, 2^31-1] 运算安全域即跳过消歧（保持默认三布局列表；
                            // InferShape 同判据值域校验将响亮拒绝，避免有符号乘加溢出 UB）
                            int64_t inSizeFromY[kInputSizeLen] = {yOriShape.GetDim(0), yOriShape.GetDim(1),
                                                                  yOriShape.GetDim(2), yOriShape.GetDim(3),
                                                                  yOriShape.GetDim(4)};
                            if (!ExternalValuesInRange(inSizeFromY, fShape, dShape, strides, dilations, pads, groups)) {
                                resolved = nullptr;
                            } else {
                                // 语义维定位（y.shape ≡ input_size 元素值，值序随 data_format）
                                const int64_t cin = isNdhwc ? yOriShape.GetDim(4) : yOriShape.GetDim(1);
                                const int64_t dIn = isNdhwc ? yOriShape.GetDim(1) : yOriShape.GetDim(2);
                                const int64_t hIn = isNdhwc ? yOriShape.GetDim(2) : yOriShape.GetDim(3);
                                const int64_t wIn = isNdhwc ? yOriShape.GetDim(3) : yOriShape.GetDim(4);
                                const size_t spatialBase = isNdhwc ? 1U : 2U;
                                const size_t dedyChannelIdx = isNdhwc ? 4U : 1U;
                                FilterSemanticDims cands[3] = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
                                MakeFilterCandidates(fShape, cands);
                                const CandidateCtx candCtx = {cin,
                                                              groups,
                                                              dShape.GetDim(dedyChannelIdx),
                                                              dShape.GetDim(spatialBase),
                                                              dShape.GetDim(spatialBase + 1U),
                                                              dShape.GetDim(spatialBase + 2U),
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
                                if (consistentCount == 1U) {
                                    // 消歧唯一：仅返回该注册布局（FE 将 ND 改写为该布局）
                                    static constexpr const char* kLayoutNames[3] = {"NCDHW", "NDHWC", "DHWCN"};
                                    resolved = kLayoutNames[consistentIdx];
                                }
                            }
                        }
                    }
                }
            }
        }
        // 未声明且非 ND 的布局（如 NCHW）：保持默认三布局列表；该类行先于本阶段
        // 被 InferShape 按 attribute_value_out_of_range 拒绝，不受本列表影响。
        if (resolved != nullptr) {
            filterFormats = std::string(resolved) + "," + resolved;
            filterDtypes = "float16,bf16";
        }
    }

    const std::string json =
        "{"
        "\"input0\": {\"name\": \"input_size\", \"dtype\": \"int32,int64\", \"format\": \"ND,ND\","
        " \"unknownshape_format\": \"ND,ND\"},"
        "\"input1\": {\"name\": \"filter\", \"dtype\": \"" +
        filterDtypes + "\", \"format\": \"" + filterFormats + "\", \"unknownshape_format\": \"" + filterFormats +
        "\"},"
        "\"input2\": {\"name\": \"out_backprop\", \"dtype\": \"float16,bf16,float16,bf16\","
        " \"format\": \"NCDHW,NCDHW,NDHWC,NDHWC\", \"unknownshape_format\": \"NCDHW,NCDHW,NDHWC,NDHWC\"},"
        "\"output0\": {\"name\": \"y\", \"dtype\": \"float16,bf16,float16,bf16\","
        " \"format\": \"NCDHW,NCDHW,NDHWC,NDHWC\", \"unknownshape_format\": \"NCDHW,NCDHW,NDHWC,NDHWC\"}"
        "}";
    result = ge::AscendString(json.c_str());
    return ge::GRAPH_SUCCESS;
}

// 图模式格式选择注册：IMPL_OP({OpType}).OpSelectFormat(OpSelectFormatFor{OpType})。
// DynamicFormatFlag(true) 已在 OpDef 声明（conv3ddx_def.cpp）；本注册经
// DefaultOpImplSpaceRegistryV2 供 FE Setcheck/SelectTbeOpFormat 查找调用。
IMPL_OP(Conv3ddx).OpSelectFormat(OpSelectFormatForConv3ddx);

} // namespace ops
