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
 * \file custom_gather_tiling.cpp
 * \brief Tiling implementation for custom_gather
 *
 * MDE 2/3 设计要点：
 *   - 核数两步法（PER_CORE_MIN=1024 下限保护）切分 outSize，VF 内建网格全局步长
 *   - ValidateShape 校验通过后调用 MergeAxis() 通用合轴（分段 reshape 合并），
 *     后续元信息全部按合并 shape 计算
 *   - TilingKey 仅编码场景位 idxMode（禁止枚举 dtype）；无用户 workspace；
 *     值依赖（SE 5.8）：否
 */

#include <array>
#include <climits>

#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/custom_gather_tiling_data.h"
#include "../../op_kernel/arch35/custom_gather_tiling_key.h"

namespace optiling {

constexpr uint32_t DCACHE_SIZE = 32 * 1024; // MDE 4.4：DCache 32KB（UB 仅作参数中转约 256B）
constexpr uint32_t STATIC_UB_ESTIMATE = 0;  // 无 __ubuf__ 静态数组（全部 LocalMemAllocator 动态分配）
constexpr int64_t PER_CORE_MIN = 1024;      // MDE 2.1：下限保护，32 对齐
constexpr int32_t MAX_RANK = CUSTOM_GATHER_MAX_RANK; // MDE 3.1：strides 数组长度
constexpr int32_t ATTR_DIM = 0; // dim 在属性数组中的下标（与 def.cpp Attr 注册顺序一致）

struct CustomGatherCompileInfo {};

// --- 命名常量（SE 2.2 输入顺序）---
constexpr int32_t INPUT_INPUT = 0;
constexpr int32_t INPUT_INDEX = 1;

// --- ValidateDtype（SE 2.2 / 5.7）---
static ge::graphStatus ValidateDtype(gert::TilingContext* context)
{
    // SE §2.2: input dtype ∈ {float16, float32, float64, bfloat16, int32, int64}
    auto inputDesc = context->GetInputDesc(INPUT_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(inputDtype != ge::DT_FLOAT16 && inputDtype != ge::DT_FLOAT && inputDtype != ge::DT_DOUBLE &&
                    inputDtype != ge::DT_BF16 && inputDtype != ge::DT_INT32 && inputDtype != ge::DT_INT64,
                OP_LOGE(context, "input dtype must be fp16/fp32/fp64/bf16/int32/int64, got %d", inputDtype),
                return ge::GRAPH_FAILED);

    // SE §2.2: index dtype ∈ {int32, int64}
    auto indexDesc = context->GetInputDesc(INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indexDesc);
    ge::DataType indexDtype = indexDesc->GetDataType();
    OP_CHECK_IF(indexDtype != ge::DT_INT32 && indexDtype != ge::DT_INT64,
                OP_LOGE(context, "index dtype must be int32/int64, got %d", indexDtype), return ge::GRAPH_FAILED);

    // SE §5.5: output.dtype == input.dtype，由 def.cpp DataType 映射保证，无需额外校验
    return ge::GRAPH_SUCCESS;
}

// --- ValidateAttr（SE 2.4：dim 归一化 + 范围校验）---
static ge::graphStatus ValidateAttr(gert::TilingContext* context, int32_t& dim, int32_t& inputRank)
{
    auto inputShape = context->GetInputShape(INPUT_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    size_t r = inputShape->GetStorageShape().GetDimNum();
    inputRank = static_cast<int32_t>(r);

    // SE §2.4: dim ∈ [-R, R)，默认 0；负值 +R 归一化到 [0, R)
    int64_t dimRaw = 0;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const auto* dimPtr = attrs->GetAttrPointer<int64_t>(ATTR_DIM);
        if (dimPtr != nullptr) {
            dimRaw = *dimPtr;
        }
    }
    int64_t dimNorm = (dimRaw < 0) ? dimRaw + static_cast<int64_t>(r) : dimRaw;
    OP_CHECK_IF(dimNorm < 0 || dimNorm >= static_cast<int64_t>(r),
                OP_LOGE(context, "dim %ld out of range [0, %zu)", dimNorm, r), return ge::GRAPH_FAILED);
    dim = static_cast<int32_t>(dimNorm);
    return ge::GRAPH_SUCCESS;
}

// --- MergeAxis：通用合轴（分段 reshape 合并，MDE 3.3 step 4.5）---
// 切段规则：dim 维单独保留；其余维切成极大可合并段——段 [a..b] 可合并 ⟺
//   段内除段头 a 外 index.size[d]==input.size[d]（段身相等）；段头可广播豁免
//   （其 size 不进任何步长）；段长 ≥2 才合并，单根段原样保留。
// 不变量：outSize / gatherDimSize / inStrideDim 均不随合轴改变。
// 收益：解码除法次数 rank-1 → mergeRank-1（无段可合时不变）。
static ge::graphStatus MergeAxis(gert::TilingContext* context, const std::array<int64_t, MAX_RANK>& xShape,
                                 const std::array<int64_t, MAX_RANK>& idxShape, int32_t rank, int32_t& dim,
                                 std::array<int64_t, MAX_RANK>& xMerge, std::array<int64_t, MAX_RANK>& idxMerge,
                                 int32_t& mergeRank)
{
    int32_t outIdx = 0;   // 合并后轴写指针
    int32_t dimShift = 0; // dim 前合并段累计减少的维数 Σ(段长-1)
    int32_t i = 0;
    while (i < rank) {
        if (i == dim) {
            // dim 维单独保留
            xMerge[outIdx] = xShape[i];
            idxMerge[outIdx] = idxShape[i];
            ++outIdx;
            ++i;
            continue;
        }
        // 段尾扩展：只查右侧维段身相等（j+1 处 idx==x 才可并入段身；广播维/dim 维打断切段）
        int32_t j = i;
        while (j + 1 < rank && j + 1 != dim && idxShape[j + 1] == xShape[j + 1]) {
            ++j;
        }
        if (j > i) {
            // 合并段 [i..j]（长 ≥ 2）：两空间分别 prod（段头广播时可不同）
            int64_t xProd = 1;
            int64_t idxProd = 1;
            for (int32_t d = i; d <= j; ++d) {
                // 段 prod ≤ 原张量元素数，此处为异常 shape 的显式防御
                OP_CHECK_IF(xShape[d] != 0 && xProd > INT64_MAX / xShape[d],
                            OP_LOGE(context, "input merge prod overflow at dim %d", d), return ge::GRAPH_FAILED);
                OP_CHECK_IF(idxShape[d] != 0 && idxProd > INT64_MAX / idxShape[d],
                            OP_LOGE(context, "index merge prod overflow at dim %d", d), return ge::GRAPH_FAILED);
                xProd *= xShape[d];
                idxProd *= idxShape[d];
            }
            xMerge[outIdx] = xProd;
            idxMerge[outIdx] = idxProd;
            if (i < dim) {
                dimShift += j - i; // dim 前段合并 → dim 前移该段减少的维数（段长-1）
            }
            ++outIdx;
        } else {
            // 单根段原样保留（可为广播维，其 size 不进任何步长）
            xMerge[outIdx] = xShape[i];
            idxMerge[outIdx] = idxShape[i];
            ++outIdx;
        }
        i = j + 1;
    }
    dim -= dimShift;
    mergeRank = outIdx;
    return ge::GRAPH_SUCCESS;
}

// --- ValidateShape（SE 2.5 / 1.4 / 5.4）+ 合轴 + 元信息计算（MDE 3.2 / 3.3）---
// dim/inputRank 均为入出参：入参为原始归一化值，出参为 MergeAxis 合并后视图值
// （后续 ComputeTiling / kernel 分派全部使用合并后值）
static ge::graphStatus ValidateShape(gert::TilingContext* context, int32_t& dim, int32_t& inputRank, int64_t& outSize,
                                     int64_t& gatherDimSize, int64_t& inStrideDim,
                                     std::array<int64_t, MAX_RANK>& outStrides,
                                     std::array<int64_t, MAX_RANK>& inStrides)
{
    auto inputShape = context->GetInputShape(INPUT_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto indexShape = context->GetInputShape(INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indexShape);

    const gert::Shape& inputStorage = inputShape->GetStorageShape();
    const gert::Shape& indexStorage = indexShape->GetStorageShape();

    // SE §2.5/§1.4: input rank ∈ [1, 8]（MDE 9.1 上限）
    OP_CHECK_IF(inputStorage.GetDimNum() != static_cast<size_t>(inputRank) || inputRank < 1 || inputRank > MAX_RANK,
                OP_LOGE(context, "input rank %d must be in [1, %d]", inputRank, MAX_RANK), return ge::GRAPH_FAILED);

    // SE §2.5/§5.4: index rank == input rank
    OP_CHECK_IF(indexStorage.GetDimNum() != static_cast<size_t>(inputRank),
                OP_LOGE(context, "index rank %zu != input rank %d", indexStorage.GetDimNum(), inputRank),
                return ge::GRAPH_FAILED);

    // 原始 shape 提取 + SE §5.4 广播约束校验（d != dim 时 index.size(d) <= input.size(d)）
    std::array<int64_t, MAX_RANK> xShape{};
    std::array<int64_t, MAX_RANK> idxShape{};
    for (int32_t d = 0; d < inputRank; ++d) {
        xShape[d] = static_cast<int64_t>(inputStorage.GetDim(static_cast<size_t>(d)));
        idxShape[d] = static_cast<int64_t>(indexStorage.GetDim(static_cast<size_t>(d)));
        if (d == dim) {
            continue;
        }
        OP_CHECK_IF(idxShape[d] > xShape[d],
                    OP_LOGE(context, "index.size(%d)=%ld > input.size(%d)=%ld", d, idxShape[d], d, xShape[d]),
                    return ge::GRAPH_FAILED);
    }

    // MDE 3.3 step 4.5: 通用合轴（宽定义切段 reshape 合并），后续全部按合并 shape 计算
    std::array<int64_t, MAX_RANK> xMerge{};
    std::array<int64_t, MAX_RANK> idxMerge{};
    int32_t mergeRank = inputRank;
    OP_CHECK_IF(MergeAxis(context, xShape, idxShape, inputRank, dim, xMerge, idxMerge, mergeRank) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "MergeAxis failed"), return ge::GRAPH_FAILED);

    // MDE 3.2: 元信息计算（合并后视图）
    outSize = 1;
    for (int32_t d = 0; d < mergeRank; ++d) {
        outSize *= idxMerge[d];
    }
    gatherDimSize = xMerge[dim];

    // MDE 3.2: outStrides[d] = prod(idxMerge[d+1:])，inStrides[d] = prod(xMerge[d+1:])
    for (int32_t d = 0; d < MAX_RANK; ++d) {
        outStrides[d] = 1;
        inStrides[d] = 1;
    }
    int64_t oStride = 1;
    for (int32_t d = mergeRank - 1; d >= 0; --d) {
        outStrides[d] = oStride;
        oStride *= idxMerge[d];
    }
    int64_t iStride = 1;
    for (int32_t d = mergeRank - 1; d >= 0; --d) {
        inStrides[d] = iStride;
        iStride *= xMerge[d];
    }
    inStrideDim = inStrides[dim]; // 合轴不变量：= prod(input.shape[dim+1:])
    inputRank = mergeRank;
    return ge::GRAPH_SUCCESS;
}

// --- ValidateInputs 调度器 ---
static ge::graphStatus ValidateInputs(gert::TilingContext* context, int32_t& dim, int32_t& inputRank, int64_t& outSize,
                                      int64_t& gatherDimSize, int64_t& inStrideDim,
                                      std::array<int64_t, MAX_RANK>& outStrides,
                                      std::array<int64_t, MAX_RANK>& inStrides)
{
    // SE §2.2/§5.7 dtype 白名单
    OP_CHECK_IF(ValidateDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateDtype failed"),
                return ge::GRAPH_FAILED);
    // SE §2.4 dim 归一化 + 范围校验
    OP_CHECK_IF(ValidateAttr(context, dim, inputRank) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateAttr failed"),
                return ge::GRAPH_FAILED);
    // SE §2.5/§1.4/§5.4 shape 校验 + 合轴（MDE 3.3 step 4.5）+ 元信息计算（合并后视图）
    OP_CHECK_IF(ValidateShape(context, dim, inputRank, outSize, gatherDimSize, inStrideDim, outStrides, inStrides) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateShape failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// DFX: 打印 TilingData 全部标量字段（仅 Debug 构建生效，skill R28）
static void DumpTilingData(gert::TilingContext* context, const CustomGatherTilingData* tiling)
{
    OP_LOGD(context,
            "CustomGatherTilingData: dim=%d, inputRank=%d, outSize=%ld, gatherDimSize=%ld, "
            "inStrideDim=%ld, outStrides=[8 elements], inStrides=[8 elements]",
            tiling->dim, tiling->inputRank, tiling->outSize, tiling->gatherDimSize, tiling->inStrideDim);
}

// TilingKey 生成（MDE 3.2：仅编码场景位 idxMode，禁止枚举 dtype）
static uint64_t GetTilingKey(bool use64Bit)
{
    uint64_t idxMode = use64Bit ? CUSTOM_GATHER_IDX64 : CUSTOM_GATHER_IDX32;
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(idxMode));
}

// 核数两步法 + 元信息填充（MDE 2.1 / 3.2）
static ge::graphStatus ComputeTiling(gert::TilingContext* context, CustomGatherTilingData* tiling, int32_t dim,
                                     int32_t inputRank, int64_t outSize, int64_t gatherDimSize, int64_t inStrideDim,
                                     const std::array<int64_t, MAX_RANK>& outStrides,
                                     const std::array<int64_t, MAX_RANK>& inStrides, int64_t coreNum)
{
    tiling->dim = dim;
    tiling->inputRank = inputRank;
    tiling->outSize = outSize;
    tiling->gatherDimSize = gatherDimSize;
    tiling->inStrideDim = inStrideDim;
    for (int32_t d = 0; d < MAX_RANK; ++d) {
        tiling->outStrides[d] = outStrides[d];
        tiling->inStrides[d] = inStrides[d];
    }

    // MDE 2.1: 核数两步法 + 下限保护
    int64_t blockFactor = (outSize + coreNum - 1) / coreNum;
    if (blockFactor < PER_CORE_MIN) {
        blockFactor = PER_CORE_MIN;
    }
    int64_t needCoreNum = (outSize + blockFactor - 1) / blockFactor;
    if (needCoreNum > coreNum) {
        needCoreNum = coreNum;
    }
    if (needCoreNum <= 0) {
        needCoreNum = 1; // 空输出兜底（MDE 2.1），保证 VF 正常空转退出
    }
    context->SetBlockDim(needCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CustomGatherTilingFunc(gert::TilingContext* context)
{
    // 1. 校验 + 合轴 + 元信息计算（MDE 3.2/3.3，dim/inputRank 出参为合并后视图值）
    int32_t dim = 0;
    int32_t inputRank = 0;
    int64_t outSize = 0;
    int64_t gatherDimSize = 0;
    int64_t inStrideDim = 0;
    std::array<int64_t, MAX_RANK> outStrides{};
    std::array<int64_t, MAX_RANK> inStrides{};
    OP_CHECK_IF(ValidateInputs(context, dim, inputRank, outSize, gatherDimSize, inStrideDim, outStrides, inStrides) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateInputs failed"), return ge::GRAPH_FAILED);

    // 2. 平台信息
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // 3. use64Bit 判定（MDE 3.3 step 6）：outSize/inputNum 超 INT32_MAX 即切换
    //    64 位；indexDtype 不强制（int64 index 越界语义由 kernel 原值比较保障，
    //    小张量走 IDX32 特化）。outStrides/inStrides/gatherDimSize 均被
    //    outSize/inputNum 覆盖，无需单独判定；合轴不改变 outSize/inputNum
    int64_t inputNum = context->GetInputShape(INPUT_INPUT)->GetStorageShape().GetShapeSize();
    bool use64Bit = (outSize > INT32_MAX) || (inputNum > INT32_MAX);

    // 4. tiling 计算
    CustomGatherTilingData* tiling = context->GetTilingData<CustomGatherTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    if (memset_s(tiling, sizeof(CustomGatherTilingData), 0, sizeof(CustomGatherTilingData)) != EOK) {
        OP_LOGE(context, "set tiling data error");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(ComputeTiling(context, tiling, dim, inputRank, outSize, gatherDimSize, inStrideDim, outStrides,
                              inStrides, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ComputeTiling failed"), return ge::GRAPH_FAILED);

    // 5. workspace：无用户 workspace，仅系统 workspace（必须无条件分配）
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = static_cast<size_t>(sysWorkspaceSize);

    // 6. local memory（UB 仅用于 TilingData → VF 参数中转，DCache 预留 32KB）
    DumpTilingData(context, tiling);
    OP_CHECK_IF(ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE, OP_LOGE(context, "ubSize %lu too small", ubSize),
                return ge::GRAPH_FAILED);
    context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));

    // 7. tiling key（仅场景模式 idxMode，不含 dtype）
    context->SetTilingKey(GetTilingKey(use64Bit));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCustomGather([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// SE 5.8：无需值依赖，不注册 TilingInputsDataDependency
IMPL_OP_OPTILING(CustomGather)
    .Tiling(CustomGatherTilingFunc)
    .TilingParse<CustomGatherCompileInfo>(TilingParseForCustomGather);

} // namespace optiling
