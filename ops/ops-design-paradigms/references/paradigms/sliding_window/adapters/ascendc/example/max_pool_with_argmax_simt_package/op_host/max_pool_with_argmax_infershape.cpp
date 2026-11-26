/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* DRAFT — 范式参考模板，未编译、未上板 */

/*!
 * \file max_pool_with_argmax_infershape.cpp
 * \brief 正向输出 shape 推导：窗口公式（floor 模式）+ 布局分派
 */

#include <string>
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr uint32_t IDX_X = 0;
static constexpr uint32_t IDX_Y = 0;
static constexpr uint32_t INDEX_KSIZE = 0;
static constexpr uint32_t INDEX_STRIDES = 1;
static constexpr uint32_t INDEX_PADS = 2;
static constexpr uint32_t INDEX_DILATIONS = 3;
static constexpr uint32_t INDEX_DATA_FORMAT = 4;
static constexpr size_t DIM_NUM = 4;
static constexpr size_t ATTR_LIST_SHAPE_SIZE = 2;

// 窗口公式与 tiling 侧同源（window-math.md §1：两处推导必须同源核对）
static inline int64_t PoolOutDim(int64_t in, int64_t pad, int64_t k, int64_t s, int64_t d)
{
    int64_t span = (k - 1) * d + 1;
    return (in + 2 * pad - span) / s + 1; // floor 模式
}

static ge::graphStatus InferShape4MaxPoolWithArgmax(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_IF(xShape->GetDimNum() != DIM_NUM, OP_LOGE(context, "x must be 4D"), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_KSIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    OP_CHECK_IF(ksize->GetSize() != ATTR_LIST_SHAPE_SIZE, OP_LOGE(context, "ksize must have 2 elements"),
                return ge::GRAPH_FAILED);
    auto ksizeData = static_cast<const int64_t*>(ksize->GetData());
    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    auto stridesData = static_cast<const int64_t*>(strides->GetData());
    auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_PADS);
    OP_CHECK_NULL_WITH_CONTEXT(context, pads);
    auto padsData = static_cast<const int64_t*>(pads->GetData());
    auto dilations = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_DILATIONS);
    OP_CHECK_NULL_WITH_CONTEXT(context, dilations);
    auto dilationsData = static_cast<const int64_t*>(dilations->GetData());
    const char* dataFormat = attrs->GetAttrPointer<char>(INDEX_DATA_FORMAT);
    std::string formatStr = (dataFormat != nullptr) ? std::string(dataFormat) : std::string("NCHW");
    OP_CHECK_IF(formatStr != "NCHW" && formatStr != "NHWC", OP_LOGE(context, "data_format must be NCHW or NHWC"),
                return ge::GRAPH_FAILED);
    bool isNhwc = (formatStr == "NHWC");

    int64_t n = xShape->GetDim(0);
    int64_t c = isNhwc ? xShape->GetDim(3) : xShape->GetDim(1);
    int64_t inH = isNhwc ? xShape->GetDim(1) : xShape->GetDim(2);
    int64_t inW = isNhwc ? xShape->GetDim(2) : xShape->GetDim(3);

    int64_t outH = PoolOutDim(inH, padsData[0], ksizeData[0], stridesData[0], dilationsData[0]);
    int64_t outW = PoolOutDim(inW, padsData[1], ksizeData[1], stridesData[1], dilationsData[1]);
    OP_CHECK_IF(outH < 1 || outW < 1, OP_LOGE(context, "output dims must be >= 1"), return ge::GRAPH_FAILED);

    gert::Shape* yShape = context->GetOutputShape(IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDim(0, n);
    yShape->SetDim(1, isNhwc ? outH : c);
    yShape->SetDim(2, isNhwc ? outW : outH);
    yShape->SetDim(3, isNhwc ? c : outW);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaxPoolWithArgmax).InferShape(InferShape4MaxPoolWithArgmax);

} // namespace ops
