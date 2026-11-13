/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// =============================================================================
// L2 implementation, stage 1 (plan) + stage 2 (run) —— conv3ddx 手写 ACLNN。
// 签名契约：op_api/aclnn_conv3ddx.h（build/autogen/aclnn_conv3ddx.h 的逐字副
// 本，参数名/顺序/const/标量映射均不得改动）。校验规格为包内黑盒 L2 异常集
// （op/tests/blackbox/testcases/conv3ddx_l2_exception.csv 的 13 行 mutation）
// 与 spec/design Interface.md 的接口契约：
//   - dtype：filter/outBackprop ∈ {fp16,bf16} 且等 dtype；out 与 outBackprop
//     同 dtype（dtype_not_supported / dtype_combo_mismatch）。
//   - format：filter ∈ {NCDHW,NDHWC,DHWCN}；outBackprop/out ∈ {NCDHW,NDHWC}
//     且 out 与 outBackprop 严格配对（attribute_value_out_of_range）。
//   - rank/长度：filter/outBackprop/out 恒 5D；inputSize 恒 1D×5（标准路由
//     aclIntArray 长度 5；Tensor 路由张量形状 [5]）（shape_mismatch）。
//   - 空 tensor：任一 tensor 含 0 维即报错（null_input，无静默空跑路径 ——
//     spec boundary 与内置基线同口径；本算子不设 early-return success）。
//   - 属性：strides/dilations 长度 5、pads 长度 6、值域 [1/0, INT32_MAX]，
//     N/C 位必须为 1；groups ∈ [1, 65535]；dataFormat ∈ {NCDHW, NDHWC}
//     （attribute_value_out_of_range；前向几何一致性由 tiling 深层校验）。
//   - 空参默认（与生成实现逐语义对齐）：dilationsOptional == nullptr →
//     [1,1,1,1,1]；dataFormatOptional == nullptr → "NDHWC"；groups /
//     enableHf32 为必选标量（C 形参按值到达，无空态）。
// =============================================================================

#include "aclnn_conv3ddx.h"
#include "conv3ddx.h"

#include <initializer_list>
#include <string>

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace {

constexpr int64_t CONV3DDX_GROUPS_MIN = 1;
constexpr int64_t CONV3DDX_GROUPS_MAX = 65535;
constexpr int64_t CONV3DDX_LIST_VALUE_MAX = 2147483647; // 950 机器域上界（装载字段口径）
constexpr size_t CONV3DDX_TENSOR_RANK = 5UL;
constexpr size_t CONV3DDX_INPUT_SIZE_LEN = 5UL;

const std::initializer_list<op::DataType> CONV3DDX_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                         op::DataType::DT_BF16};
const std::initializer_list<op::DataType> CONV3DDX_SIZE_DTYPE_LIST = {op::DataType::DT_INT32, op::DataType::DT_INT64};

// 属性默认值（OpDef 声明的 GE 默认；空参路径逐语义对齐生成实现）
constexpr int64_t CONV3DDX_DILATIONS_DEFAULT[5] = {1, 1, 1, 1, 1};
const char* const CONV3DDX_DATA_FORMAT_NCDHW = "NCDHW";
const char* const CONV3DDX_DATA_FORMAT_NDHWC = "NDHWC";

bool IsSupportFormat(const aclTensor* tensor, const std::initializer_list<op::Format>& formatList)
{
    auto fmt = tensor->GetStorageFormat();
    for (auto support : formatList) {
        if (fmt == support) {
            return true;
        }
    }
    return false;
}

bool CheckNotNull(const aclTensor* filter, const aclTensor* outBackprop, const aclTensor* out)
{
    OP_CHECK_NULL(filter, return false);
    OP_CHECK_NULL(outBackprop, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

bool CheckDtypeValid(const aclTensor* filter, const aclTensor* outBackprop, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(filter, CONV3DDX_DTYPE_SUPPORT_LIST, return false);
    // 等 dtype 配对：混合 dtype（如 fp16 filter + bf16 dedy）不支持
    OP_CHECK_DTYPE_NOT_SAME(filter, outBackprop, return false);
    // y 的 dtype 随 out_backprop（spec dtype_rule 在等 dtype 配对下退化同 dtype）
    OP_CHECK_DTYPE_NOT_MATCH(out, outBackprop->GetDataType(), return false);
    return true;
}

bool CheckFormatValid(const aclTensor* filter, const aclTensor* outBackprop, const aclTensor* out)
{
    // filter 三种注册布局独立于 data_format（OpDef 组合 1-24）
    OP_CHECK(IsSupportFormat(filter, {op::Format::FORMAT_NCDHW, op::Format::FORMAT_NDHWC, op::Format::FORMAT_DHWCN}),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "filter format %d not in {NCDHW, NDHWC, DHWCN}.",
                     static_cast<int>(filter->GetStorageFormat())),
             return false);
    OP_CHECK(IsSupportFormat(outBackprop, {op::Format::FORMAT_NCDHW, op::Format::FORMAT_NDHWC}),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outBackprop format %d not in {NCDHW, NDHWC}.",
                     static_cast<int>(outBackprop->GetStorageFormat())),
             return false);
    // y 与 out_backprop 严格配对（tiling 同口径校验 yFmt == dedyFmt，提前到 L2）
    OP_CHECK(out->GetStorageFormat() == outBackprop->GetStorageFormat(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out format %d not paired with outBackprop format %d.",
                     static_cast<int>(out->GetStorageFormat()), static_cast<int>(outBackprop->GetStorageFormat())),
             return false);
    return true;
}

// 空 tensor：任一输入/输出含 0 维即拒绝（null_input；无静默空跑路径）
bool CheckNotEmpty(const aclTensor* filter, const aclTensor* outBackprop, const aclTensor* out)
{
    OP_CHECK(!filter->IsEmpty(), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "filter is empty tensor."), return false);
    OP_CHECK(!outBackprop->IsEmpty(), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outBackprop is empty tensor."), return false);
    OP_CHECK(!out->IsEmpty(), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out is empty tensor."), return false);
    return true;
}

bool CheckShapeValid(const aclTensor* filter, const aclTensor* outBackprop, const aclTensor* out)
{
    OP_CHECK_WRONG_DIMENSION(filter, CONV3DDX_TENSOR_RANK, return false);
    OP_CHECK_WRONG_DIMENSION(outBackprop, CONV3DDX_TENSOR_RANK, return false);
    OP_CHECK_WRONG_DIMENSION(out, CONV3DDX_TENSOR_RANK, return false);
    return true;
}

// input_size 载体（Tensor 路由）：1D×5 int32/int64 ND（OpDef 契约）
bool CheckInputSizeTensorValid(const aclTensor* inputSize)
{
    OP_CHECK(inputSize->GetViewShape().GetDimNum() == 1UL,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputSize tensor rank %zu != 1.", inputSize->GetViewShape().GetDimNum()),
             return false);
    OP_CHECK(inputSize->GetViewShape().GetDim(0) == static_cast<int64_t>(CONV3DDX_INPUT_SIZE_LEN),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputSize tensor length %ld != 5.",
                     static_cast<long>(inputSize->GetViewShape().GetDim(0))),
             return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(inputSize, CONV3DDX_SIZE_DTYPE_LIST, return false);
    OP_CHECK(inputSize->GetStorageFormat() == op::Format::FORMAT_ND,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputSize tensor format %d != ND.",
                     static_cast<int>(inputSize->GetStorageFormat())),
             return false);
    return true;
}

// input_size 载体（标准路由）：aclIntArray 恒 5 元素且为正维度值
// （长度 4 等接口契约违例 → shape_mismatch；0/负值 → null_input/shape_mismatch）
bool CheckInputSizeArrayValid(const aclIntArray* inputSize)
{
    OP_CHECK(inputSize->Size() == CONV3DDX_INPUT_SIZE_LEN,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputSize length %lu != 5 (1D x 5 contract).",
                     static_cast<unsigned long>(inputSize->Size())),
             return false);
    for (uint64_t i = 0; i < CONV3DDX_INPUT_SIZE_LEN; i++) {
        OP_CHECK((*inputSize)[i] > 0,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputSize[%lu] = %ld must be a positive dim.",
                         static_cast<unsigned long>(i), static_cast<long>((*inputSize)[i])),
                 return false);
        OP_CHECK((*inputSize)[i] <= CONV3DDX_LIST_VALUE_MAX,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputSize[%lu] = %ld exceeds representable range.",
                         static_cast<unsigned long>(i), static_cast<long>((*inputSize)[i])),
                 return false);
    }
    return true;
}

// 调用方 out 的形状契约：y.shape = input_size 元素值（共享 helper 推导后比对）
bool CheckOutShapeValid(const aclTensor* out, const aclIntArray* inputSize)
{
    op::Shape expected;
    if (!l0op::Conv3ddxOutShape(inputSize, expected)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out shape derivation failed: inputSize must hold 5 values.");
        return false;
    }
    OP_CHECK(out->GetViewShape() == expected,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out shape mismatch: y.shape must equal inputSize values."),
             return false);
    return true;
}

// 属性契约（OpDef 声明序读取；950 机器域与 spec attributes 同口径）：
// N/C 位序随 dataFormat —— NCDHW 位 [0,1]；NDHWC 位 [0,4]（N 位与 C 位）
bool CheckAttrsValid(const aclIntArray* strides, const aclIntArray* pads, const aclIntArray* dilations, int64_t groups,
                     const char* dataFormat)
{
    OP_CHECK(strides->Size() == 5UL,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "strides length %lu != 5.", static_cast<unsigned long>(strides->Size())),
             return false);
    OP_CHECK(
        dilations->Size() == 5UL,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilations length %lu != 5.", static_cast<unsigned long>(dilations->Size())),
        return false);
    OP_CHECK(pads->Size() == 6UL,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pads length %lu != 6.", static_cast<unsigned long>(pads->Size())),
             return false);
    for (uint64_t i = 0; i < 5UL; i++) {
        OP_CHECK((*strides)[i] >= 1 && (*strides)[i] <= CONV3DDX_LIST_VALUE_MAX,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "strides[%lu] = %ld out of range [1, 2147483647].",
                         static_cast<unsigned long>(i), static_cast<long>((*strides)[i])),
                 return false);
        OP_CHECK((*dilations)[i] >= 1 && (*dilations)[i] <= CONV3DDX_LIST_VALUE_MAX,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilations[%lu] = %ld out of range [1, 2147483647].",
                         static_cast<unsigned long>(i), static_cast<long>((*dilations)[i])),
                 return false);
    }
    for (uint64_t i = 0; i < 6UL; i++) {
        OP_CHECK((*pads)[i] >= 0 && (*pads)[i] <= CONV3DDX_LIST_VALUE_MAX,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pads[%lu] = %ld out of range [0, 2147483647].",
                         static_cast<unsigned long>(i), static_cast<long>((*pads)[i])),
                 return false);
    }
    OP_CHECK(groups >= CONV3DDX_GROUPS_MIN && groups <= CONV3DDX_GROUPS_MAX,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groups %ld out of range [1, 65535].", static_cast<long>(groups)),
             return false);
    // dataFormat ∈ {NCDHW, NDHWC}；N/C 位必须为 1（REQUIREMENTS §8.2.2）
    bool isNdhwc = false;
    OP_CHECK(std::string(dataFormat) == std::string(CONV3DDX_DATA_FORMAT_NCDHW) ||
                 std::string(dataFormat) == std::string(CONV3DDX_DATA_FORMAT_NDHWC),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dataFormat '%s' not in {NCDHW, NDHWC}.", dataFormat), return false);
    isNdhwc = (std::string(dataFormat) == std::string(CONV3DDX_DATA_FORMAT_NDHWC));
    const uint64_t ncPos[2] = {0UL, isNdhwc ? 4UL : 1UL}; // N 位恒 0；C 位随位序
    for (uint64_t pos : ncPos) {
        OP_CHECK((*strides)[pos] == 1,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "strides[%lu] = %ld, N/C positions must be 1.",
                         static_cast<unsigned long>(pos), static_cast<long>((*strides)[pos])),
                 return false);
        OP_CHECK((*dilations)[pos] == 1,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilations[%lu] = %ld, N/C positions must be 1.",
                         static_cast<unsigned long>(pos), static_cast<long>((*dilations)[pos])),
                 return false);
    }
    return true;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnConv3ddxGetWorkspaceSize(const aclIntArray* inputSize, const aclTensor* filter,
                                          const aclTensor* outBackprop, const aclIntArray* strides,
                                          const aclIntArray* pads, const aclIntArray* dilationsOptional, int64_t groups,
                                          char* dataFormatOptional, bool enableHf32, const aclTensor* out,
                                          uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    // DFX_IN 必须携带全部形参（含属性）：L2_DFX_PHASE_1 的 executor 缓存按
    // API 名 + IN/OUT 元组生成键 —— 属性缺席会使"同张量签名、不同属性"的
    // 顺序调用命中同一缓存计划（官方 convolution_backward 同款全参入键）。
    L2_DFX_PHASE_1(aclnnConv3ddx,
                   DFX_IN(inputSize, filter, outBackprop, strides, pads, dilationsOptional, groups, dataFormatOptional,
                          enableHf32),
                   DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 必选指针（空参语义与生成实现一致：required 槽位 nullptr → 161001）
    CHECK_RET(CheckNotNull(filter, outBackprop, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(inputSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(strides != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(pads != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 可选参数空态 → OpDef 声明默认（生成实现同语义：null → 默认，不报错）
    const aclIntArray* dilations = dilationsOptional;
    if (dilations == nullptr) {
        auto defArr = uniqueExecutor->AllocIntArray(CONV3DDX_DILATIONS_DEFAULT, 5UL);
        CHECK_RET(defArr != nullptr, ACLNN_ERR_INNER_NULLPTR);
        dilations = defArr;
    }
    const char* dataFormat = (dataFormatOptional != nullptr) ? dataFormatOptional : CONV3DDX_DATA_FORMAT_NDHWC;

    CHECK_RET(CheckDtypeValid(filter, outBackprop, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormatValid(filter, outBackprop, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(filter, outBackprop, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNotEmpty(filter, outBackprop, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckInputSizeArrayValid(inputSize), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOutShapeValid(out, inputSize), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAttrsValid(strides, pads, dilations, groups, dataFormat), ACLNN_ERR_PARAM_INVALID);

    // 本算子无静默空跑路径：空 tensor 在 CheckNotEmpty 已拒绝，无 early-return。

    // 非连续输入先做连续化（每个数据输入各一次；输出侧由 ViewCopy 回填）
    auto filterContiguous = l0op::Contiguous(filter, uniqueExecutor.get());
    CHECK_RET(filterContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outBackpropContiguous = l0op::Contiguous(outBackprop, uniqueExecutor.get());
    CHECK_RET(outBackpropContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 属性装载为 executor 持有副本（stage-2 启动时生命周期归 executor）；
    // dataFormat 归一为受检枚举字面量（NCDHW/NDHWC，静态存储期）
    const char* dataFormatNorm = (std::string(dataFormat) == std::string(CONV3DDX_DATA_FORMAT_NCDHW)) ?
                                     CONV3DDX_DATA_FORMAT_NCDHW :
                                     CONV3DDX_DATA_FORMAT_NDHWC;
    auto stridesArr = uniqueExecutor->AllocIntArray(strides->GetData(), strides->Size());
    CHECK_RET(stridesArr != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto padsArr = uniqueExecutor->AllocIntArray(pads->GetData(), pads->Size());
    CHECK_RET(padsArr != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto dilationsArr = uniqueExecutor->AllocIntArray(dilations->GetData(), dilations->Size());
    CHECK_RET(dilationsArr != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 核心计算经 L0（标准路由）：属性值 OpDef 声明序传入
    auto result = l0op::Conv3ddx(inputSize, filterContiguous, outBackpropContiguous, stridesArr, padsArr, dilationsArr,
                                 groups, dataFormatNorm, enableHf32, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // L0 结果回填调用方 out（可能非连续）：每输出一次 ViewCopy
    CHECK_RET(l0op::ViewCopy(result, out, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnConv3ddxTensorGetWorkspaceSize(const aclTensor* inputSize, const aclTensor* filter,
                                                const aclTensor* outBackprop, const aclIntArray* strides,
                                                const aclIntArray* pads, const aclIntArray* dilationsOptional,
                                                int64_t groups, char* dataFormatOptional, bool enableHf32,
                                                const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnConv3ddxTensor,
                   DFX_IN(inputSize, filter, outBackprop, strides, pads, dilationsOptional, groups, dataFormatOptional,
                          enableHf32),
                   DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CHECK_RET(CheckNotNull(filter, outBackprop, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(inputSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(strides != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(pads != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    const aclIntArray* dilations = dilationsOptional;
    if (dilations == nullptr) {
        auto defArr = uniqueExecutor->AllocIntArray(CONV3DDX_DILATIONS_DEFAULT, 5UL);
        CHECK_RET(defArr != nullptr, ACLNN_ERR_INNER_NULLPTR);
        dilations = defArr;
    }
    const char* dataFormat = (dataFormatOptional != nullptr) ? dataFormatOptional : CONV3DDX_DATA_FORMAT_NDHWC;

    CHECK_RET(CheckDtypeValid(filter, outBackprop, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormatValid(filter, outBackprop, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(filter, outBackprop, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNotEmpty(filter, outBackprop, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckInputSizeTensorValid(inputSize), ACLNN_ERR_PARAM_INVALID);
    // Tensor 路由：out 形状无法在 host 与设备张量 inputSize 的元素值比对（值依赖），
    // 该契约（y.shape = input_size 元素值）由 tiling 层 ValidateAndLoad 的输出形状
    // 逐维比对防线执行（值依赖读取后拒绝错配，防 GM 越界写）；L2 层仅做静态项校验。
    CHECK_RET(CheckAttrsValid(strides, pads, dilations, groups, dataFormat), ACLNN_ERR_PARAM_INVALID);

    auto inputSizeContiguous = l0op::Contiguous(inputSize, uniqueExecutor.get());
    CHECK_RET(inputSizeContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto filterContiguous = l0op::Contiguous(filter, uniqueExecutor.get());
    CHECK_RET(filterContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outBackpropContiguous = l0op::Contiguous(outBackprop, uniqueExecutor.get());
    CHECK_RET(outBackpropContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const char* dataFormatNorm = (std::string(dataFormat) == std::string(CONV3DDX_DATA_FORMAT_NCDHW)) ?
                                     CONV3DDX_DATA_FORMAT_NCDHW :
                                     CONV3DDX_DATA_FORMAT_NDHWC;
    auto stridesArr = uniqueExecutor->AllocIntArray(strides->GetData(), strides->Size());
    CHECK_RET(stridesArr != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto padsArr = uniqueExecutor->AllocIntArray(pads->GetData(), pads->Size());
    CHECK_RET(padsArr != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto dilationsArr = uniqueExecutor->AllocIntArray(dilations->GetData(), dilations->Size());
    CHECK_RET(dilationsArr != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Tensor 路由：input_size 设备张量与调用方 out 直通 L0（生成实现同语义）
    auto result =
        l0op::Conv3ddxTensor(inputSizeContiguous, filterContiguous, outBackpropContiguous, stridesArr, padsArr,
                             dilationsArr, groups, dataFormatNorm, enableHf32, out, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnConv3ddx(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnConv3ddx);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
