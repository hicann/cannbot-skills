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
// L2 implementation, stage 1 (plan) + stage 2 (run) —— conv3dtranspose 手写 ACLNN。
// 签名契约：op_api/aclnn_conv3dtranspose.h（REQUIREMENTS §8.2.1 /
// design target_aclnn_header.h 基准——参数名/顺序/const/标量映射逐字保留，
// 不得改动）。校验规格为包内黑盒 L2 异常集（op/tests/blackbox/testcases/
// conv3dtranspose_l2_exception.csv 的 19 行 mutation）与 spec/design
// Interface.md 的接口契约：
//   - dtype：x/filter/y ∈ {fp16,bf16} 且 x==filter（dtype_not_supported /
//     dtype_combo_mismatch）；inputSize ∈ {int32,int64}（input_size_dtype_invalid）；
//     bias ∈ {fp16,fp32}（可选，空指针/空张量 = 无 bias）。
//   - format：inputSize/bias ND；x ∈ {NCDHW,NDHWC}；filter ∈
//     {NCDHW,NDHWC,DHWCN}；y 与 x 严格配对（attribute_value_out_of_range）。
//   - rank/长度：x/filter/y 恒 5D；inputSize 恒 1D×5（input_size_len6）；
//     bias 恒 1D（长度 == Cin 的值依赖校验由 tiling 执行——bias_length_mismatch
//     行经 tiling 拒绝）。
//   - 空 tensor：inputSize/x/filter/y 任一含 0 维即报错（null_input，无静默
//     空跑路径——spec boundary 与同族 conv3ddx 同口径；本算子不设
//     early-return success）。bias 空张量（0 元素）归一为 nullptr 走无 bias
//     路由（HostTiling.md §2：可选 bias 存在性由 op_api 包装层归一）。
//   - 属性：strides/dilations/outputPadding 长度 5、pads 长度 6、值域
//     [1/0, INT32_MAX]（950 装载字段口径），N/C 位序随 dataFormat（NCDHW 位
//     [0,1]；NDHWC 位 [0,4]——strides/dilations 必 1、outputPadding 必 0）；
//     逐轴 q < s 或 q < d（output_padding_violation）；groups ∈ [1,65535]；
//     dataFormat ∈ {NCDHW,NDHWC}（data_format_invalid）；offset_x/enable_hf32
//     为保留/按值属性（无空态，无 L2 约束——深层语义归 tiling）。
//   - 空参默认（与 OpDef 声明默认逐语义对齐）：dilations == nullptr →
//     {1,1,1,1,1}；outputPadding == nullptr → {0,0,0,0,0}；dataFormat ==
//     nullptr → "NDHWC"；strides/pads 必选（nullptr → 161001）。
//   - offsetW：保留槽（"保留，不启用"，REQUIREMENTS §8.2.2）——非空指针即
//     拒绝（冻结 golden 同口径：offset_w 传入即 out of contract）。
//   - 值依赖（y.shape = inputSize 元素值 / 转置几何 / bias 长度）：host 侧
//     不可读设备张量值，L2 只做静态项校验；值依赖契约由 tiling 层
//     ValidateAndLoad 读取设备侧 input_size 值后执行（geom_wx_mismatch 行经
//     tiling 拒绝）。
// =============================================================================

#include "aclnn_conv3dtranspose.h"
#include "conv3dtranspose.h"

#include <initializer_list>
#include <string>

#include "acl/acl_rt.h"
#include "exe_graph/runtime/tensor_data.h"
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

constexpr int64_t CONV3DTRANSPOSE_GROUPS_MIN = 1;
constexpr int64_t CONV3DTRANSPOSE_GROUPS_MAX = 65535;
constexpr int64_t CONV3DTRANSPOSE_LIST_VALUE_MAX = 2147483647; // 950 机器域上界（装载字段口径）
constexpr size_t CONV3DTRANSPOSE_TENSOR_RANK = 5UL;
constexpr size_t CONV3DTRANSPOSE_INPUT_SIZE_LEN = 5UL;
constexpr size_t CONV3DTRANSPOSE_STRIDES_LEN = 5UL;
constexpr size_t CONV3DTRANSPOSE_PADS_LEN = 6UL;
constexpr size_t CONV3DTRANSPOSE_OUTPUT_PADDING_LEN = 5UL;

const std::initializer_list<op::DataType> CONV3DTRANSPOSE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                                op::DataType::DT_BF16};
const std::initializer_list<op::DataType> CONV3DTRANSPOSE_BIAS_DTYPE_LIST = {op::DataType::DT_FLOAT16,
                                                                             op::DataType::DT_FLOAT};
const std::initializer_list<op::DataType> CONV3DTRANSPOSE_SIZE_DTYPE_LIST = {op::DataType::DT_INT32,
                                                                             op::DataType::DT_INT64};

// 属性默认值（OpDef 声明的 GE 默认；空参路径逐语义对齐）
constexpr int64_t CONV3DTRANSPOSE_DILATIONS_DEFAULT[5] = {1, 1, 1, 1, 1};
constexpr int64_t CONV3DTRANSPOSE_OUTPUT_PADDING_DEFAULT[5] = {0, 0, 0, 0, 0};
const char* const CONV3DTRANSPOSE_DATA_FORMAT_NCDHW = "NCDHW";
const char* const CONV3DTRANSPOSE_DATA_FORMAT_NDHWC = "NDHWC";

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

// 必选指针（空参语义与生成实现一致：required 槽位 nullptr → 161001）；
// bias（可选）与 offsetW（保留，不启用——非空即拒）不入本表
bool CheckNotNull(const aclTensor* inputSize, const aclTensor* x, const aclTensor* filter, const aclTensor* y)
{
    OP_CHECK_NULL(inputSize, return false);
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(filter, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

// bias 非空时的 dtype 白名单（fp16/fp32——独立 bias dtype 组合）；y 的 dtype
// 经共享 helper（Conv3DTransposeCheckOut）与 x 配对校验
bool CheckDtypeValid(const aclTensor* inputSize, const aclTensor* x, const aclTensor* filter, const aclTensor* bias)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(inputSize, CONV3DTRANSPOSE_SIZE_DTYPE_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x, CONV3DTRANSPOSE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(filter, CONV3DTRANSPOSE_DTYPE_SUPPORT_LIST, return false);
    // 等 dtype 配对：混合 dtype（fp16 x + bf16 filter）不支持
    OP_CHECK_DTYPE_NOT_SAME(x, filter, return false);
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, CONV3DTRANSPOSE_BIAS_DTYPE_LIST, return false);
    }
    return true;
}

// filter 三种注册布局独立于 data_format（OpDef 组合 25-48/49-72）；
// x 布局随 data_format 于 {NCDHW,NDHWC}；y 与 x 严格配对经共享 helper
bool CheckFormatValid(const aclTensor* inputSize, const aclTensor* x, const aclTensor* filter, const aclTensor* bias)
{
    OP_CHECK(
        inputSize->GetStorageFormat() == op::Format::FORMAT_ND,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputSize format %d != ND.", static_cast<int>(inputSize->GetStorageFormat())),
        return false);
    OP_CHECK(
        IsSupportFormat(x, {op::Format::FORMAT_NCDHW, op::Format::FORMAT_NDHWC}),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x format %d not in {NCDHW, NDHWC}.", static_cast<int>(x->GetStorageFormat())),
        return false);
    OP_CHECK(IsSupportFormat(filter, {op::Format::FORMAT_NCDHW, op::Format::FORMAT_NDHWC, op::Format::FORMAT_DHWCN}),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "filter format %d not in {NCDHW, NDHWC, DHWCN}.",
                     static_cast<int>(filter->GetStorageFormat())),
             return false);
    if (bias != nullptr) {
        OP_CHECK(bias->GetStorageFormat() == op::Format::FORMAT_ND,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias format %d != ND.", static_cast<int>(bias->GetStorageFormat())),
                 return false);
    }
    return true;
}

// rank 契约：input_size 1D×5（接口契约，REQUIREMENTS §8.2.1）；x/filter 恒 5D；
// bias 恒 1D（长度 == Cin 的值依赖校验归 tiling）；y 恒 5D 经共享 helper
bool CheckShapeValid(const aclTensor* inputSize, const aclTensor* x, const aclTensor* filter, const aclTensor* bias)
{
    OP_CHECK(inputSize->GetViewShape().GetDimNum() == 1UL,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputSize rank %zu != 1.", inputSize->GetViewShape().GetDimNum()),
             return false);
    OP_CHECK(inputSize->GetViewShape().GetDim(0) == static_cast<int64_t>(CONV3DTRANSPOSE_INPUT_SIZE_LEN),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputSize length %ld != 5 (1D x 5 contract).",
                     static_cast<long>(inputSize->GetViewShape().GetDim(0))),
             return false);
    OP_CHECK_WRONG_DIMENSION(x, CONV3DTRANSPOSE_TENSOR_RANK, return false);
    OP_CHECK_WRONG_DIMENSION(filter, CONV3DTRANSPOSE_TENSOR_RANK, return false);
    if (bias != nullptr) {
        OP_CHECK(bias->GetViewShape().GetDimNum() == 1UL,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias rank %zu != 1.", bias->GetViewShape().GetDimNum()),
                 return false);
    }
    return true;
}

// 空 tensor：必选输入/输出含 0 维即拒绝（null_input；无静默空跑路径）。
// bias 空张量不入本表（0 元素可选 bias 由 op_api 归一 nullptr，按无 bias 路由）
bool CheckNotEmpty(const aclTensor* inputSize, const aclTensor* x, const aclTensor* filter, const aclTensor* y)
{
    OP_CHECK(!inputSize->IsEmpty(), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputSize is empty tensor."), return false);
    OP_CHECK(!x->IsEmpty(), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x is empty tensor."), return false);
    OP_CHECK(!filter->IsEmpty(), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "filter is empty tensor."), return false);
    OP_CHECK(!y->IsEmpty(), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y is empty tensor."), return false);
    return true;
}

// 属性契约（OpDef 声明序读取；950 机器域与 spec attributes 同口径）：
// N/C 位序随 dataFormat —— NCDHW 位 [0,1]；NDHWC 位 [0,4]
bool CheckAttrsValid(const aclIntArray* strides, const aclIntArray* pads, const aclIntArray* dilations, int64_t groups,
                     const char* dataFormat, const aclIntArray* outputPadding)
{
    OP_CHECK(strides->Size() == CONV3DTRANSPOSE_STRIDES_LEN,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "strides length %lu != 5.", static_cast<unsigned long>(strides->Size())),
             return false);
    OP_CHECK(
        dilations->Size() == CONV3DTRANSPOSE_STRIDES_LEN,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilations length %lu != 5.", static_cast<unsigned long>(dilations->Size())),
        return false);
    OP_CHECK(pads->Size() == CONV3DTRANSPOSE_PADS_LEN,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pads length %lu != 6.", static_cast<unsigned long>(pads->Size())),
             return false);
    OP_CHECK(outputPadding->Size() == CONV3DTRANSPOSE_OUTPUT_PADDING_LEN,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outputPadding length %lu != 5.",
                     static_cast<unsigned long>(outputPadding->Size())),
             return false);
    for (uint64_t i = 0; i < 5UL; i++) {
        OP_CHECK((*strides)[i] >= 1 && (*strides)[i] <= CONV3DTRANSPOSE_LIST_VALUE_MAX,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "strides[%lu] = %ld out of range [1, 2147483647].",
                         static_cast<unsigned long>(i), static_cast<long>((*strides)[i])),
                 return false);
        OP_CHECK((*dilations)[i] >= 1 && (*dilations)[i] <= CONV3DTRANSPOSE_LIST_VALUE_MAX,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilations[%lu] = %ld out of range [1, 2147483647].",
                         static_cast<unsigned long>(i), static_cast<long>((*dilations)[i])),
                 return false);
        OP_CHECK((*outputPadding)[i] >= 0 && (*outputPadding)[i] <= CONV3DTRANSPOSE_LIST_VALUE_MAX,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outputPadding[%lu] = %ld out of range [0, 2147483647].",
                         static_cast<unsigned long>(i), static_cast<long>((*outputPadding)[i])),
                 return false);
    }
    for (uint64_t i = 0; i < 6UL; i++) {
        OP_CHECK((*pads)[i] >= 0 && (*pads)[i] <= CONV3DTRANSPOSE_LIST_VALUE_MAX,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pads[%lu] = %ld out of range [0, 2147483647].",
                         static_cast<unsigned long>(i), static_cast<long>((*pads)[i])),
                 return false);
    }
    OP_CHECK(groups >= CONV3DTRANSPOSE_GROUPS_MIN && groups <= CONV3DTRANSPOSE_GROUPS_MAX,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groups %ld out of range [1, 65535].", static_cast<long>(groups)),
             return false);
    // dataFormat ∈ {NCDHW, NDHWC}（REQUIREMENTS §8.2.2）
    bool isNdhwc = false;
    OP_CHECK(std::string(dataFormat) == std::string(CONV3DTRANSPOSE_DATA_FORMAT_NCDHW) ||
                 std::string(dataFormat) == std::string(CONV3DTRANSPOSE_DATA_FORMAT_NDHWC),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dataFormat '%s' not in {NCDHW, NDHWC}.", dataFormat), return false);
    isNdhwc = (std::string(dataFormat) == std::string(CONV3DTRANSPOSE_DATA_FORMAT_NDHWC));
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
        OP_CHECK((*outputPadding)[pos] == 0,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outputPadding[%lu] = %ld, N/C positions must be 0.",
                         static_cast<unsigned long>(pos), static_cast<long>((*outputPadding)[pos])),
                 return false);
    }
    // 逐轴 output_padding 约束 q < s 或 q < d（PyTorch §4.4 同口径，spec 全局约束）；
    // 空间位随 dataFormat：NCDHW 位 [2,4]；NDHWC 位 [1,3]
    uint64_t spatialPos[3] = {2UL, 3UL, 4UL};
    if (isNdhwc) {
        spatialPos[0] = 1UL;
        spatialPos[1] = 2UL;
        spatialPos[2] = 3UL;
    }
    for (uint64_t pos : spatialPos) {
        OP_CHECK(
            (*outputPadding)[pos] < (*strides)[pos] || (*outputPadding)[pos] < (*dilations)[pos],
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outputPadding[%lu] = %ld violates q < stride(%ld) or q < dilation(%ld).",
                    static_cast<unsigned long>(pos), static_cast<long>((*outputPadding)[pos]),
                    static_cast<long>((*strides)[pos]), static_cast<long>((*dilations)[pos])),
            return false);
    }
    return true;
}

} // namespace

namespace {

/**
 * @brief 值依赖桥接：调用方 inputSize 载体 → OpDef 输入 0 的装载张量
 *
 * 契约背景：冻结调用方（黑盒 CSV/TTK aclnn 绑定、REQUIREMENTS §8.2.1 目标
 * ABI）把 inputSize 作为设备 aclTensor 传入；而 tiling 的值依赖读取
 * （GetInputTensor(0)->GetData）在本 SDK 的 aclnn 执行器直调路径不做隐式
 * Device→Host 物化（GE 图编译路径经 TilingInputsDataDependency/InputsData
 * Dependency 机制物化；执行器直调把设备地址原样放入 tiling 上下文——实测
 * host 解引用设备地址 SIGSEGV，见任务报告）。kernel 侧对 input_size 地址
 * 不消费（(void)input_size，其值由 Host 解析进 tilingData）。
 *
 * 桥接（与官方卷积反向族/同族 conv3ddx 标准路由的 GetOutputSize 装载机制
 * 同构——aclIntArray → executor 持有宿主常量张量）：
 *   - 调用方张量已在宿主（IsOnHost，如宿主常量）：直通（Contiguous 已保证
 *     紧凑排布）；
 *   - 设备张量：一次 5 元素（20/40 字节）同步 D2H 读取（aclrtMemcpy——
 *     本 L2 唯一的显式数据搬运，仅针对值依赖形状描述符；算子数据的搬运
 *     仍全部经 executor 的 Contiguous/启动列表）后构造 executor 持有的
 *     1D×5 int32/int64 ND 宿主常量张量（dtype 随调用方描述符）。
 *     调用方设备张量仅作数值来源，不进启动列表。
 *
 * @return 装载张量（executor 持有或宿主直通）；失败返回 nullptr
 */
const aclTensor* Conv3DTransposeInputSizeTensor(const aclTensor* inputSizeContiguous, aclOpExecutor* executor)
{
    if (gert::TensorPlacementUtils::IsOnHost(inputSizeContiguous->GetPlacement())) {
        return inputSizeContiguous;
    }
    constexpr size_t kSizeLen = 5UL;
    const bool isInt64 = (inputSizeContiguous->GetDataType() == op::DataType::DT_INT64);
    int32_t raw32[kSizeLen] = {0, 0, 0, 0, 0};
    int64_t values[kSizeLen] = {0, 0, 0, 0, 0};
    const size_t bytes = isInt64 ? kSizeLen * sizeof(int64_t) : kSizeLen * sizeof(int32_t);
    auto memcpyRet = aclrtMemcpy(isInt64 ? static_cast<void*>(values) : static_cast<void*>(raw32), bytes,
                                 inputSizeContiguous->GetData(), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (memcpyRet != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                "Conv3DTranspose value-depend bridge failed: read inputSize device data, aclrtMemcpy ERROR: %d.",
                static_cast<int>(memcpyRet));
        return nullptr;
    }
    if (!isInt64) {
        for (size_t i = 0; i < kSizeLen; i++) {
            values[i] = static_cast<int64_t>(raw32[i]);
        }
    }
    auto sizeArr = executor->AllocIntArray(values, kSizeLen);
    OP_CHECK_NULL(sizeArr, return nullptr);
    auto inputSizeTensor = executor->ConvertToTensor(sizeArr, inputSizeContiguous->GetDataType());
    OP_CHECK_NULL(inputSizeTensor, return nullptr);
    // ConvertToTensor 缺省 format 与 OpDef 注册（ND）对齐，显式三处设置避免缺省漂移
    const_cast<aclTensor*>(inputSizeTensor)->SetStorageFormat(op::Format::FORMAT_ND);
    const_cast<aclTensor*>(inputSizeTensor)->SetViewFormat(op::Format::FORMAT_ND);
    const_cast<aclTensor*>(inputSizeTensor)->SetOriginalFormat(op::Format::FORMAT_ND);
    return inputSizeTensor;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnConv3DTransposeGetWorkspaceSize(const aclTensor* inputSize, const aclTensor* x,
                                                 const aclTensor* filter, const aclTensor* bias,
                                                 const aclTensor* offsetW, const aclIntArray* strides,
                                                 const aclIntArray* pads, const aclIntArray* dilations, int64_t groups,
                                                 char* dataFormat, const aclIntArray* outputPadding, int64_t offsetX,
                                                 bool enableHf32, aclTensor* y, uint64_t* workspaceSize,
                                                 aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    // DFX_IN 必须携带全部形参（含属性）：L2_DFX_PHASE_1 的 executor 缓存按
    // API 名 + IN/OUT 元组生成键 —— 属性缺席会使"同张量签名、不同属性"的
    // 顺序调用命中同一缓存计划（官方 convolution_backward 同款全参入键）。
    L2_DFX_PHASE_1(aclnnConv3DTranspose,
                   DFX_IN(inputSize, x, filter, bias, offsetW, strides, pads, dilations, groups, dataFormat,
                          outputPadding, offsetX, enableHf32),
                   DFX_OUT(y));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 必选指针（空参语义与生成实现一致：required 槽位 nullptr → 161001）
    CHECK_RET(CheckNotNull(inputSize, x, filter, y), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(strides != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(pads != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    // offsetW：保留槽（保留，不启用——REQUIREMENTS §8.2.2）；非空指针即拒绝
    // （冻结 golden 同口径：offset_w 传入即 out of contract）
    CHECK_RET(offsetW == nullptr, ACLNN_ERR_PARAM_INVALID);

    // 可选属性空态 → OpDef 声明默认（生成实现同语义：null → 默认，不报错）
    const aclIntArray* dilationsArr = dilations;
    if (dilationsArr == nullptr) {
        auto defArr = uniqueExecutor->AllocIntArray(CONV3DTRANSPOSE_DILATIONS_DEFAULT, 5UL);
        CHECK_RET(defArr != nullptr, ACLNN_ERR_INNER_NULLPTR);
        dilationsArr = defArr;
    }
    const aclIntArray* outputPaddingArr = outputPadding;
    if (outputPaddingArr == nullptr) {
        auto defArr = uniqueExecutor->AllocIntArray(CONV3DTRANSPOSE_OUTPUT_PADDING_DEFAULT, 5UL);
        CHECK_RET(defArr != nullptr, ACLNN_ERR_INNER_NULLPTR);
        outputPaddingArr = defArr;
    }
    const char* dataFormatStr = (dataFormat != nullptr) ? dataFormat : CONV3DTRANSPOSE_DATA_FORMAT_NDHWC;

    CHECK_RET(CheckDtypeValid(inputSize, x, filter, bias), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormatValid(inputSize, x, filter, bias), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(inputSize, x, filter, bias), ACLNN_ERR_PARAM_INVALID);
    // y↔x 静态契约（共享 helper，与 L0 直通防线同源）：rank 5 / dtype 随 x /
    // 存储格式与 x 严格配对
    CHECK_RET(l0op::Conv3DTransposeCheckOut(x, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNotEmpty(inputSize, x, filter, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAttrsValid(strides, pads, dilationsArr, groups, dataFormatStr, outputPaddingArr),
              ACLNN_ERR_PARAM_INVALID);

    // bias 空张量归一 nullptr（HostTiling.md §2：0 元素可选 bias 由 op_api
    // 包装层统一归一，按无 bias 路由，非报错）
    const aclTensor* biasNorm = bias;
    if (biasNorm != nullptr && biasNorm->IsEmpty()) {
        biasNorm = nullptr;
    }

    // 本算子无静默空跑路径：空 tensor 在 CheckNotEmpty 已拒绝，无 early-return。

    // 非连续输入先做连续化（每个数据输入各一次；输出侧 y 由调用方持有，
    // 值依赖直通形态——无 ViewCopy 回填）
    auto inputSizeContiguous = l0op::Contiguous(inputSize, uniqueExecutor.get());
    CHECK_RET(inputSizeContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto filterContiguous = l0op::Contiguous(filter, uniqueExecutor.get());
    CHECK_RET(filterContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* biasContiguous = biasNorm;
    if (biasNorm != nullptr) {
        biasContiguous = l0op::Contiguous(biasNorm, uniqueExecutor.get());
        CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 值依赖桥接：inputSize 装载为 OpDef 输入 0 的宿主常量张量（设备张量
    // 读取 5 元素值；tiling 的值依赖读取需宿主地址，kernel 不消费该地址）
    auto inputSizeTensor = Conv3DTransposeInputSizeTensor(inputSizeContiguous, uniqueExecutor.get());
    CHECK_RET(inputSizeTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 属性装载为 executor 持有副本（stage-2 启动时生命周期归 executor）；
    // dataFormat 归一为受检枚举字面量（NCDHW/NDHWC，静态存储期）
    const char* dataFormatNorm = (std::string(dataFormatStr) == std::string(CONV3DTRANSPOSE_DATA_FORMAT_NCDHW)) ?
                                     CONV3DTRANSPOSE_DATA_FORMAT_NCDHW :
                                     CONV3DTRANSPOSE_DATA_FORMAT_NDHWC;
    auto stridesArr = uniqueExecutor->AllocIntArray(strides->GetData(), strides->Size());
    CHECK_RET(stridesArr != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto padsArr = uniqueExecutor->AllocIntArray(pads->GetData(), pads->Size());
    CHECK_RET(padsArr != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto dilationsCopy = uniqueExecutor->AllocIntArray(dilationsArr->GetData(), dilationsArr->Size());
    CHECK_RET(dilationsCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outputPaddingCopy = uniqueExecutor->AllocIntArray(outputPaddingArr->GetData(), outputPaddingArr->Size());
    CHECK_RET(outputPaddingCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 核心计算经 L0（标准路由，值依赖）：inputSize 装载张量（宿主常量，
    // 见 Conv3DTransposeInputSizeTensor 桥接）、bias（可空）与调用方 y 直通
    // （y.shape = input_size 元素值契约由 tiling 层 ValidateAndLoad 在值依赖
    // 读取后执行）
    auto result = l0op::Conv3DTranspose(inputSizeTensor, xContiguous, filterContiguous, biasContiguous, stridesArr,
                                        padsArr, dilationsCopy, groups, dataFormatNorm, outputPaddingCopy, offsetX,
                                        enableHf32, y, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 无 ViewCopy：result 即调用方 y（值依赖直通形态，L0 原样返回）
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnConv3DTranspose(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnConv3DTranspose);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
