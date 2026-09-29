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
 * \file euclidean_norm_tiling_arch35.cpp
 * \brief EuclideanNorm 算子 host tiling 实现（新范式）。
 */

#include <algorithm>
#include <set>
#include <string>
#include <vector>
#include <cstring>

#include "euclidean_norm_tiling_arch35.h"

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "graph/utils/type_utils.h"
#include "../../op_kernel/arch35/euclidean_norm_tiling_data.h"
#include "../../op_kernel/arch35/euclidean_norm_tiling_key.h"

namespace optiling {

// 输入索引：x = 0，axes = 1（与 TilingInputsDataDependency({1}) 的值依赖声明对应）
static constexpr size_t INPUT_X_IDX = 0;
static constexpr size_t INPUT_AXES_IDX = 1;

// A/R 规整后轴数合法区间：上限取通用类别 MAX_PATTERN_RANK = 9。
static constexpr int32_t MIN_AXIS_NUM = 2;
static constexpr int32_t MAX_AXIS_NUM = MAX_PATTERN_RANK;

// cacheBuf（kernel 二分树缓存）固定 16KB 预算；FP32 单元素字节数（中间精度换算）。
static constexpr int64_t CACHE_BUF_BYTES = 16 * 1024;
static constexpr int64_t FP32_BYTES = 4;

// UB 预算系数：pre 侧共 3 个 buffer（preIn + preRes + preResTail = P_PRE + P_PRE_EXT），
// post 侧 1 个（outBuf = P_POST）。
static constexpr int64_t P_PRE = 2;
static constexpr int64_t P_PRE_EXT = 1;
static constexpr int64_t P_POST = 1;

// A/R 轴模式化后偶位 A、奇位 R，相邻同类型轴（同为 A 或同为 R）的固定间距。
static constexpr int32_t AXIS_INTERVAL = 2;
// Group 触发阈值分母：A 迭代总数 ≤ 核数/2（A 并行度不足半数核）且 R 迭代 >1 时走 Group 2D 分核。
static constexpr int64_t GROUP_CORE_RATIO = 2;

// 空 tensor 三分类：NORMAL（正常）/ EMPTY_A（非归约维含 0 → 输出 0 元素）/
// EMPTY_R（仅归约维含 0 → 每输出 = sqrt(0) = 0）。
enum class EuclideanNormEmptyKind {
    NORMAL = 0,
    EMPTY_A,
    EMPTY_R,
};

// tiling 全程工作状态：一次 TilingFunc 调用内的全部中间量，按生产阶段分区。
struct EuclideanNormCtx {
    // ── 平台参数（GetPlatformInfo 填充）──
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    int64_t blockSize = 0;
    int64_t cacheLineSize = 0;

    // ── 输入信息（GetShapeAndDtype / ParseAxesTensor 填充）──
    ge::DataType xDtype = ge::DT_UNDEFINED;
    int64_t dtypeSize = 0;
    int64_t maxDtypeSize = 0;
    std::vector<int64_t> xShape;
    std::vector<int64_t> reduceAxes;

    // ── 空 tensor 分类（ClassifyEmptyTensor 填充）──
    EuclideanNormEmptyKind emptyKind = EuclideanNormEmptyKind::NORMAL;
    int64_t aTotalEmpty = 0;

    // ── A/R 规整模式（BuildInitialAxisList + PreprocessPattern 填充）──
    std::vector<int64_t> axisShape;
    std::vector<bool> isReduceAxis;
    int32_t axisNum = 0;
    bool isTailR = false;

    // ── UB 切分参数（ComputeAUbFactor / ComputeRUbFactor / ExpandAIfRFullyLoaded 填充）──
    int32_t aSplitIdx = 0;
    int32_t rSplitIdx = 0;
    int64_t aUbFactor = 0;
    int64_t rUbFactor = 0;
    int64_t rUbFactorAlign = 0;
    int64_t innerAProdAlign = 0;
    int64_t innerRProdAlign = 0;

    // ── A 方向多核参数（ComputeFusedALoopSplit 填充）──
    int64_t aLoopCntTotal = 0;
    int64_t aSplitChunkCnt = 0;
    int64_t aBigCoreLoopCnt = 0;
    int64_t aSmallCoreLoopCnt = 0;
    int32_t aBigCoreCnt = 0;
    int32_t usedCoreNum = 0;

    // ── R 方向迭代数（ComputeRLoopCnt 填充）──
    int64_t rLoopCntTotal = 0;

    // ── UB buffer 尺寸（ComputeUbSizes 填充）──
    int64_t preBufSize = 0;
    int64_t postBufSize = 0;

    // ── Group 2D 分核参数（ComputeGroupSplit 填充）──
    bool isGroup = false;
    int64_t rGroupCnt = 0;
};

// 获取平台硬件参数：AIV 核数、UB 容量、32B block 大小、cache line 大小。
// 全走 platform 接口（禁止写死），任一参数为 0 视为环境异常直接失败。
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ctx.coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(ctx.coreNum == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "platform.coreNum",
                                                         "GetCoreNumAiv returned 0"),
                return ge::GRAPH_FAILED);

    uint64_t ub = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ctx.ubSize = static_cast<int64_t>(ub);
    OP_CHECK_IF(ctx.ubSize == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "platform.ubSize",
                                                         "GetCoreMemSize(UB) returned 0"),
                return ge::GRAPH_FAILED);

    ctx.blockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context));
    ctx.cacheLineSize = static_cast<int64_t>(Ops::Base::GetCacheLineSize(context));
    OP_CHECK_IF(ctx.blockSize == 0 || ctx.cacheLineSize == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "platform.blockSize/cacheLineSize",
                                                         "BlockSize=" + std::to_string(ctx.blockSize) +
                                                             " or cacheLineSize=" + std::to_string(ctx.cacheLineSize) +
                                                             " is 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 读取输入 x 的 shape 与 dtype：rank 0（标量）视作 [1]；负 dim（动态 shape 防御）报错；
// dtype 白名单 {fp16, bf16, fp32, int32}；派生 dtypeSize 与 maxDtypeSize（fp32 中间精度
// 使 b16 输入也按 4 字节预算 UB）
static ge::graphStatus GetShapeAndDtype(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    auto xShapePtr = context->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape& xS = xShapePtr->GetStorageShape();
    const size_t rank = xS.GetDimNum();
    ctx.xShape.clear();
    if (rank == 0) {
        ctx.xShape.push_back(1);
    } else {
        for (size_t i = 0; i < rank; ++i) {
            OP_CHECK_IF(
                xS.GetDim(i) < 0,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context->GetNodeName(), "x", std::to_string(xS.GetDim(i)),
                    "dim[" + std::to_string(i) + "] is negative (dynamic shape must be instantiated before tiling)"),
                return ge::GRAPH_FAILED);
            ctx.xShape.push_back(xS.GetDim(i));
        }
    }

    auto xDesc = context->GetInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    ctx.xDtype = xDesc->GetDataType();
    const std::set<ge::DataType> supportedDtypes = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32};
    OP_CHECK_IF(supportedDtypes.count(ctx.xDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x",
                                                      ge::TypeUtils::DataTypeToSerialString(ctx.xDtype).c_str(),
                                                      "only DT_FLOAT16/DT_BF16/DT_FLOAT/DT_INT32 are supported"),
                return ge::GRAPH_FAILED);
    ctx.dtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(ctx.xDtype));
    ctx.maxDtypeSize = std::max(ctx.dtypeSize, FP32_BYTES);

    std::string shapeStr = "[";
    for (size_t i = 0; i < ctx.xShape.size(); ++i) {
        if (i > 0) {
            shapeStr += ", ";
        }
        shapeStr += std::to_string(ctx.xShape[i]);
    }
    shapeStr += "]";
    OP_LOGI(context, "EuclideanNorm input: xShape=%s xDtype=%d dtypeSize=%ld", shapeStr.c_str(),
            static_cast<int>(ctx.xDtype), ctx.dtypeSize);
    return ge::GRAPH_SUCCESS;
}

// 归一化并追加单个 axis：负索引 +xRank 归一到 [0, xRank)；越界/重复打 ERROR 并返回 false。
static bool PushOneAxis(const char* nodeName, int64_t v, int64_t idx, int64_t xRank, std::set<int64_t>& seen,
                        std::vector<int64_t>& axesOut)
{
    if (v < -xRank || v >= xRank) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "axes", std::to_string(v),
                                              "axes[" + std::to_string(idx) + "] out of range [-" +
                                                  std::to_string(xRank) + ", " + std::to_string(xRank) + ")");
        return false;
    }
    const int64_t norm = (v < 0) ? (v + xRank) : v;
    if (!seen.insert(norm).second) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "axes", std::to_string(norm),
                                              "duplicate axis in axes (input idx=" + std::to_string(idx) + ")");
        return false;
    }
    axesOut.push_back(norm);
    return true;
}

// 按 T（int32/int64）逐元素读取 axes 并归一化入 axesOut——模板化消 dtype 双分支循环重复
template <typename T>
static bool LoadAxesLoop(const char* nodeName, const T* data, int64_t axesNum, int64_t xRank, std::set<int64_t>& seen,
                         std::vector<int64_t>& axesOut)
{
    axesOut.reserve(static_cast<size_t>(axesNum));
    for (int64_t i = 0; i < axesNum; ++i) {
        if (!PushOneAxis(nodeName, static_cast<int64_t>(data[i]), i, xRank, seen, axesOut)) {
            return false;
        }
    }
    return true;
}

// 解析值依赖输入 axes 的具体数值到 ctx.reduceAxes（升序去重）：
// 空 axes → full reduce 展开 [0..rank-1]（与 infershape 端空 axes 语义一致）；
// 负索引归一化 v+rank；越界 / 重复 → GRAPH_FAILED；
// 末尾 sort 升序保证合轴确定性（bitwise_reproducible）。
static ge::graphStatus ParseAxesTensor(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    const gert::Tensor* axesTensor = context->GetInputTensor(INPUT_AXES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, axesTensor);
    const int64_t axesNum = axesTensor->GetShapeSize();
    const int64_t xRank = static_cast<int64_t>(ctx.xShape.size());
    ctx.reduceAxes.clear();

    if (axesNum == 0) {
        ctx.reduceAxes.resize(static_cast<size_t>(xRank));
        for (int64_t i = 0; i < xRank; ++i) {
            ctx.reduceAxes[static_cast<size_t>(i)] = i;
        }
        return ge::GRAPH_SUCCESS;
    }

    const char* nodeName = context->GetNodeName();
    std::set<int64_t> seen;
    const ge::DataType dt = axesTensor->GetDataType();
    if (dt == ge::DT_INT32) {
        const int32_t* data = axesTensor->GetData<int32_t>();
        OP_CHECK_IF(
            data == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "axes", "axes const tensor data ptr is null (int32)"),
            return ge::GRAPH_FAILED);
        if (!LoadAxesLoop<int32_t>(nodeName, data, axesNum, xRank, seen, ctx.reduceAxes)) {
            return ge::GRAPH_FAILED; // 失败细节已在 LoadAxesLoop 内打 ERROR
        }
    } else if (dt == ge::DT_INT64) {
        const int64_t* data = axesTensor->GetData<int64_t>();
        OP_CHECK_IF(
            data == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "axes", "axes const tensor data ptr is null (int64)"),
            return ge::GRAPH_FAILED);
        if (!LoadAxesLoop<int64_t>(nodeName, data, axesNum, xRank, seen, ctx.reduceAxes)) {
            return ge::GRAPH_FAILED; // 失败细节已在 LoadAxesLoop 内打 ERROR
        }
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "axes", ge::TypeUtils::DataTypeToSerialString(dt).c_str(),
                                              "only int32/int64 are supported");
        return ge::GRAPH_FAILED;
    }
    std::sort(ctx.reduceAxes.begin(), ctx.reduceAxes.end());
    return ge::GRAPH_SUCCESS;
}

// 初始化轴列表：axisShape 拷贝 xShape，按 reduceAxes 给各轴打 R（归约轴）标记，其余为 A（保留轴）。
// 在 ClassifyEmptyTensor 之前执行，使空分类能在原始轴语义上判零维。
static void BuildInitialAxisList(EuclideanNormCtx& ctx)
{
    const size_t rank = ctx.xShape.size();
    ctx.axisShape = ctx.xShape;
    ctx.isReduceAxis.assign(rank, false);
    for (int64_t reduceAxis : ctx.reduceAxes) {
        ctx.isReduceAxis[static_cast<size_t>(reduceAxis)] = true;
    }
}

// A/R 规整第 1 步（共 4 步，顺序不可换）：删除所有 size-1 轴（对归约数值无贡献；
// A=1 不影响输出、R=1 退化为 |x|）；全删空则保留 [A1] 占位。
static void DropSizeOneAxes(EuclideanNormCtx& ctx)
{
    std::vector<int64_t> newShape;
    std::vector<bool> newIsR;
    for (size_t i = 0; i < ctx.axisShape.size(); ++i) {
        if (ctx.axisShape[i] != 1) {
            newShape.push_back(ctx.axisShape[i]);
            newIsR.push_back(ctx.isReduceAxis[i]);
        }
    }
    if (newShape.empty()) {
        newShape.push_back(1);
        newIsR.push_back(false);
    }
    ctx.axisShape = std::move(newShape);
    ctx.isReduceAxis = std::move(newIsR);
}

// A/R 规整第 2 步：相邻同类型轴（同 A 或同 R）乘积合并——A 合并 = 输出维乘积、
// R 合并 = 归约长度乘积（ND 连续布局下数值等价），压缩为最短交替模式。
static void FuseAxis(EuclideanNormCtx& ctx)
{
    std::vector<int64_t> fusedShape;
    std::vector<bool> fusedIsR;
    for (size_t i = 0; i < ctx.axisShape.size(); ++i) {
        if (!fusedShape.empty() && fusedIsR.back() == ctx.isReduceAxis[i]) {
            fusedShape.back() *= ctx.axisShape[i];
        } else {
            fusedShape.push_back(ctx.axisShape[i]);
            fusedIsR.push_back(ctx.isReduceAxis[i]);
        }
    }
    ctx.axisShape = std::move(fusedShape);
    ctx.isReduceAxis = std::move(fusedIsR);
}

// A/R 规整第 3 步：首轴为 R 时在最前补一根 size-1 的 A 轴，保证模式以 A 起头
// （偶位 A / 奇位 R 下标不变量的前提）。
static void PadLeadingOneA(EuclideanNormCtx& ctx)
{
    if (!ctx.axisShape.empty() && ctx.isReduceAxis.front()) {
        ctx.axisShape.insert(ctx.axisShape.begin(), 1);
        ctx.isReduceAxis.insert(ctx.isReduceAxis.begin(), false);
    }
}

// A/R 规整第 4 步：纯 A 退化（无 R 轴）时补 R——全 1 退化 [A1] 末尾补 R=1（成 tail-R）；
// 其余前置 [A1, R1]（成 tail-A，保持最内轴为 A 的内存连续性——禁止末尾补 R=1，
// 否则最内轴变成长度 1 的 R，每 A 元素独占 1 元素 burst，搬运效率崩塌）。
static void PadRIfPureA(EuclideanNormCtx& ctx)
{
    bool hasR = false;
    for (bool isReduce : ctx.isReduceAxis) {
        if (isReduce) {
            hasR = true;
            break;
        }
    }
    if (!hasR) {
        if (ctx.axisShape.size() == 1 && ctx.axisShape[0] == 1) {
            ctx.axisShape.push_back(1);
            ctx.isReduceAxis.push_back(true);
        } else {
            ctx.axisShape.insert(ctx.axisShape.begin(), {1, 1});
            ctx.isReduceAxis.insert(ctx.isReduceAxis.begin(), {false, true});
        }
    }
}

// 空 tensor 三分类（在原始 shape 上扫描零维，先于规整执行以免合轴/预算退化）：
// A 轴含 0 → EMPTY_A（输出 0 元素）；仅 R 轴含 0 → EMPTY_R（空归约 sum=0 → 每输出 = sqrt(0) = 0，
// aTotalEmpty = 非零 A 轴乘积 = 输出元素数）；否则 NORMAL 走主 tiling 路径。
static void ClassifyEmptyTensor(EuclideanNormCtx& ctx)
{
    bool hasZeroA = false;
    bool hasZeroR = false;
    int64_t aTotal = 1;
    for (size_t i = 0; i < ctx.xShape.size(); ++i) {
        const bool isR = ctx.isReduceAxis[i];
        const int64_t sz = ctx.xShape[i];
        if (!isR) {
            if (sz == 0) {
                hasZeroA = true;
            } else {
                aTotal *= sz;
            }
        } else {
            if (sz == 0) {
                hasZeroR = true;
            }
        }
    }
    if (hasZeroA) {
        ctx.emptyKind = EuclideanNormEmptyKind::EMPTY_A;
        ctx.aTotalEmpty = 0;
    } else if (hasZeroR) {
        ctx.emptyKind = EuclideanNormEmptyKind::EMPTY_R;
        ctx.aTotalEmpty = aTotal;
    } else {
        ctx.emptyKind = EuclideanNormEmptyKind::NORMAL;
        ctx.aTotalEmpty = 0;
    }
}

// A/R 规整总装（仅 NORMAL 路径）：依次执行去 1 → 合轴 → 补 leading A → 补 R 增广，
// 校验 axisNum ∈ [MIN_AXIS_NUM, MAX_AXIS_NUM] 与"偶位 A / 奇位 R"不变量（防御性自检），
// 按轴数奇偶定 isTailR：偶数轴 → tail-R（最内轴为 R），奇数轴 → tail-A。
static ge::graphStatus PreprocessPattern(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    DropSizeOneAxes(ctx);
    FuseAxis(ctx);
    PadLeadingOneA(ctx);
    PadRIfPureA(ctx);

    ctx.axisNum = static_cast<int32_t>(ctx.axisShape.size());
    OP_CHECK_IF(ctx.axisNum < MIN_AXIS_NUM || ctx.axisNum > MAX_AXIS_NUM,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "axisNum",
                    "after A/R pattern regularization, axisNum=" + std::to_string(ctx.axisNum) + " out of [" +
                        std::to_string(MIN_AXIS_NUM) + ", " + std::to_string(MAX_AXIS_NUM) + "]"),
                return ge::GRAPH_FAILED);

    for (int32_t i = 0; i < ctx.axisNum; ++i) {
        const bool wantR = (i % AXIS_INTERVAL == 1);
        OP_CHECK_IF(ctx.isReduceAxis[static_cast<size_t>(i)] != wantR,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "isReduceAxis",
                                                             "axis[" + std::to_string(i) +
                                                                 "] type mismatch after A/R regularization "
                                                                 "(expected alternating [A,R] pattern)"),
                    return ge::GRAPH_FAILED);
    }
    ctx.isTailR = (ctx.axisNum % AXIS_INTERVAL == 0);
    std::string axisStr = "[";
    for (int32_t i = 0; i < ctx.axisNum; ++i) {
        if (i > 0) {
            axisStr += ", ";
        }
        axisStr += std::to_string(ctx.axisShape[static_cast<size_t>(i)]);
        axisStr += ctx.isReduceAxis[static_cast<size_t>(i)] ? "(R)" : "(A)";
    }
    axisStr += "]";
    OP_LOGD(context->GetNodeName(), "PreprocessPattern: axisNum=%d isTailR=%d axes=%s", ctx.axisNum,
            static_cast<int>(ctx.isTailR), axisStr.c_str());
    return ge::GRAPH_SUCCESS;
}

// 计算规整后各轴在输入 GM 上的步长：stride[i] = ∏_{j>i} axisShape[j]
// （ND 连续布局，最内轴 stride=1），供 kernel 侧 Unravel 解码 GM 偏移。
static void ComputeAxisStrides(const EuclideanNormCtx& ctx, int64_t outStride[])
{
    int64_t strideAcc = 1;
    for (int32_t i = ctx.axisNum - 1; i >= 0; --i) {
        outStride[i] = strideAcc;
        strideAcc *= ctx.axisShape[static_cast<size_t>(i)];
    }
}

// 返回最后一个 A 轴（最大偶下标）的位置；规整后恒存在，兜底返回 0（防御）。
static int32_t LastAAxisIdx(const EuclideanNormCtx& ctx)
{
    for (int32_t i = ctx.axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 0) {
            return i;
        }
    }
    return 0;
}

// 返回最后一个 R 轴（最大奇下标）的位置；规整后恒存在，兜底返回 1（防御）。
static int32_t LastRAxisIdx(const EuclideanNormCtx& ctx)
{
    for (int32_t i = ctx.axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 1) {
            return i;
        }
    }
    return 1;
}

// 计算全部 A 轴长度乘积 = 输出元素总数 aTotal（Group workspace 列数 / 扩 A 上限）。
static int64_t TotalAProd(const EuclideanNormCtx& ctx)
{
    int64_t total = 1;
    for (int32_t i = 0; i < ctx.axisNum; i += AXIS_INTERVAL) {
        total *= ctx.axisShape[static_cast<size_t>(i)];
    }
    return total;
}

// 计算 aSplitIdx 内侧 A 轴的对齐乘积 innerAProdAlign：仅 tail-A 的最内 A 轴（搬运 burst 轴）
// 按 32B 元素数（bsElem）向上对齐，其余轴原值累乘；aSplit 变化后需重算。
static void ComputeInnerAProdAlign(EuclideanNormCtx& ctx)
{
    const int64_t bsElem = ctx.blockSize / ctx.dtypeSize;
    ctx.innerAProdAlign = 1;
    for (int32_t k = ctx.aSplitIdx + AXIS_INTERVAL; k < ctx.axisNum; k += AXIS_INTERVAL) {
        if (k == ctx.axisNum - 1 && !ctx.isTailR) {
            ctx.innerAProdAlign *= Ops::Base::CeilAlign(ctx.axisShape[static_cast<size_t>(k)], bsElem);
        } else {
            ctx.innerAProdAlign *= ctx.axisShape[static_cast<size_t>(k)];
        }
    }
}

// UB 切分 Step 1：以 1 条 cache line 为初始预算确定 A 切分（aSplitIdx / aUbFactor），
// 使 A 单元（aUbFactor × innerAProdAlign）驻留单条 cache line（Reduce 主循环热数据局部性）。
// 从最内轴向外逐轴累乘，超预算即停：停在 A 轴 → 按剩余预算切该轴 chunk；
// 停在 R 轴 → 退到外侧 A 轴、因子 1；全部装得下 → 不切（aSplitIdx=0）。
static void ComputeAUbFactor(EuclideanNormCtx& ctx)
{
    const int64_t bsElem = ctx.blockSize / ctx.dtypeSize;
    const int64_t maxInnerAInitElem = ctx.cacheLineSize / ctx.dtypeSize;
    const int64_t cachelineTmp = (maxInnerAInitElem / bsElem) * bsElem;

    int64_t product = 1;
    int32_t idx = ctx.axisNum - 1;
    while (idx >= 0) {
        int64_t axisSize = (idx == ctx.axisNum - 1) ?
                               Ops::Base::CeilAlign(ctx.axisShape[static_cast<size_t>(idx)], bsElem) :
                               ctx.axisShape[static_cast<size_t>(idx)];
        if (product * axisSize > cachelineTmp) {
            break;
        }
        product *= axisSize;
        idx--;
    }

    if (idx < 0) {
        ctx.aSplitIdx = 0;
        ctx.aUbFactor = ctx.axisShape[0];
    } else if (idx % AXIS_INTERVAL == 0) {
        ctx.aSplitIdx = idx;
        ctx.aUbFactor = std::min(cachelineTmp / product, ctx.axisShape[static_cast<size_t>(idx)]);
        ctx.aUbFactor = std::max<int64_t>(ctx.aUbFactor, 1);
    } else {
        ctx.aSplitIdx = idx - 1;
        ctx.aUbFactor = 1;
    }

    ComputeInnerAProdAlign(ctx);
}

// 反解单次 R 迭代可用元素上限 rIMax（UB 预算不等式的逆）：
// (ubSize − 16KB cacheBuf − outBuf) / (3 个 pre buffer × aUnit × maxDtypeSize)；
// aUnit 非法或预算不足时返回 -1（由调用方报 TILING_FAIL）。
static int64_t ComputeRiMax(const EuclideanNormCtx& ctx)
{
    const int64_t ubAvailable = ctx.ubSize - CACHE_BUF_BYTES;
    const int64_t aUnit = ctx.aUbFactor * ctx.innerAProdAlign;
    const int64_t postBufSize = Ops::Base::CeilAlign(aUnit * ctx.maxDtypeSize, ctx.blockSize);
    const int64_t aOnlyBytes = P_POST * postBufSize;
    const int64_t bytesPerRElem = (P_PRE + P_PRE_EXT) * aUnit * ctx.maxDtypeSize;
    if (aUnit <= 0 || bytesPerRElem <= 0) {
        return -1;
    }
    const int64_t numer = ubAvailable - aOnlyBytes;
    if (numer <= 0) {
        return -1;
    }
    return numer / bytesPerRElem;
}

// UB 切分 Step 2：在 rIMax 预算内从最内 R 轴向外吸收 R 轴（得 innerRProdAlign 与 rSplitIdx），
// 再定 rUbFactor；切在最内 R 轴（isTailR 且 rSplitIdx==lastR）时做 32B burst 对齐得
// rUbFactorAlign——全载 CeilAlign / 部分 FloorAlign（对齐到 0 报错），对齐后超预算回退
// FloorAlign（无条件 FloorAlign 会误判 TILING_FAIL）；actual 与 padded 之差由 kernel 侧清零兜底。
static ge::graphStatus ComputeRUbFactor(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    const int32_t lastR = LastRAxisIdx(ctx);
    const int64_t rIMax = ComputeRiMax(ctx);
    OP_CHECK_IF(rIMax < 1,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "rIMax",
                    "R_i_max=" + std::to_string(rIMax) + " < 1 (aUbFactor=" + std::to_string(ctx.aUbFactor) +
                        ", innerAProdAlign=" + std::to_string(ctx.innerAProdAlign) + ")"),
                return ge::GRAPH_FAILED);

    const int64_t bsElem = ctx.blockSize / ctx.dtypeSize;
    ctx.innerRProdAlign = 1;
    ctx.rSplitIdx = lastR;
    while (ctx.rSplitIdx > 1) {
        int64_t axisSize = (ctx.rSplitIdx == lastR && ctx.isTailR) ?
                               Ops::Base::CeilAlign(ctx.axisShape[static_cast<size_t>(ctx.rSplitIdx)], bsElem) :
                               ctx.axisShape[static_cast<size_t>(ctx.rSplitIdx)];
        if (axisSize * ctx.innerRProdAlign > rIMax) {
            break;
        }
        ctx.innerRProdAlign *= axisSize;
        ctx.rSplitIdx -= AXIS_INTERVAL;
    }

    const int64_t rAxisSize = ctx.axisShape[static_cast<size_t>(ctx.rSplitIdx)];
    ctx.rUbFactor = std::min(rIMax / ctx.innerRProdAlign, rAxisSize);
    ctx.rUbFactor = std::max<int64_t>(ctx.rUbFactor, 1);

    const bool isBurstTailR = (ctx.isTailR && ctx.rSplitIdx == lastR);
    if (isBurstTailR) {
        if (ctx.rUbFactor < rAxisSize) {
            ctx.rUbFactor = Ops::Base::FloorAlign(ctx.rUbFactor, bsElem);
            OP_CHECK_IF(
                ctx.rUbFactor == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "rUbFactor",
                                                         "rUbFactor floor-aligned to 0 (tail-R burst alignment)"),
                return ge::GRAPH_FAILED);
            ctx.rUbFactorAlign = ctx.rUbFactor;
        } else {
            ctx.rUbFactorAlign = Ops::Base::CeilAlign(ctx.rUbFactor, bsElem);
            if (ctx.rUbFactorAlign * ctx.innerRProdAlign > rIMax) {
                ctx.rUbFactor = Ops::Base::FloorAlign(ctx.rUbFactor, bsElem);
                OP_CHECK_IF(ctx.rUbFactor == 0,
                            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                                context->GetNodeName(), "rUbFactor",
                                "rUbFactor floor-aligned to 0 (retry after CeilAlign overflow)"),
                            return ge::GRAPH_FAILED);
                ctx.rUbFactorAlign = ctx.rUbFactor;
            }
        }
    } else {
        ctx.rUbFactorAlign = ctx.rUbFactor;
    }
    OP_LOGD(context->GetNodeName(),
            "ComputeRUbFactor: rSplitIdx=%d rUbFactor=%ld rUbFactorAlign=%ld innerRProdAlign=%ld", ctx.rSplitIdx,
            ctx.rUbFactor, ctx.rUbFactorAlign, ctx.innerRProdAlign);
    return ge::GRAPH_SUCCESS;
}

// 判定整个 R 空间是否单次迭代全载：rUbFactor 占满 rSplit 轴长度，且 rSplitIdx 外侧无 R 轴
// （即 rLoopCntTotal 将为 1）——全载时才值得把剩余 UB 让给 A。
static bool RIsFullyLoaded(const EuclideanNormCtx& ctx)
{
    if (ctx.rUbFactor != ctx.axisShape[static_cast<size_t>(ctx.rSplitIdx)]) {
        return false;
    }
    for (int32_t i = ctx.rSplitIdx - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 1) {
            return false;
        }
    }
    return true;
}

// R 全载时把 UB 预算反解为 A 上限：每 A lane 耗 3×rPadded 个 pre 元素 + 1 个 out 元素，
// aUnitMax = ubAvailable / (3 × rPaddedElems × maxDtypeSize + maxDtypeSize)；预算非法返回 -1。
static int64_t SolveAUnitMax(const EuclideanNormCtx& ctx)
{
    const int64_t ubAvailable = ctx.ubSize - CACHE_BUF_BYTES;
    const int64_t rPaddedElems = ctx.rUbFactorAlign * ctx.innerRProdAlign;
    const int64_t coeff = (P_PRE + P_PRE_EXT) * rPaddedElems * ctx.maxDtypeSize + P_POST * ctx.maxDtypeSize;
    if (coeff <= 0) {
        return -1;
    }
    return ubAvailable / coeff;
}

// UB 切分 Step 3：R 全载时用剩余 UB 扩张 A 单元，减少 A 迭代次数。
// aUnitMax 受三重约束：UB 反解值 / totalA / cacheBuf 行宽上限（16KB÷4 = 4096，cacheBuf 硬约束）；
// tail-A 时按 32B 元素数向下对齐（最内 A 轴是 burst 轴）；从 lastA 向内重新累积定
// aSplitIdx / aUbFactor 并重算 innerAProdAlign；无利可图（≤ 当前值）则保持不变。
static ge::graphStatus ExpandAIfRFullyLoaded(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    if (!RIsFullyLoaded(ctx)) {
        return ge::GRAPH_SUCCESS;
    }

    const int64_t bsElem = ctx.blockSize / ctx.dtypeSize;
    const int64_t cacheLaneLimit = CACHE_BUF_BYTES / FP32_BYTES;
    int64_t totalA = TotalAProd(ctx);

    int64_t aUnitMax = std::min(SolveAUnitMax(ctx), totalA);
    aUnitMax = std::min(aUnitMax, cacheLaneLimit);
    if (!ctx.isTailR) {
        aUnitMax = Ops::Base::FloorAlign(aUnitMax, bsElem);
    }

    const int64_t curAUnit = ctx.aUbFactor * ctx.innerAProdAlign;
    if (aUnitMax <= curAUnit) {
        return ge::GRAPH_SUCCESS;
    }

    int64_t product = 1;
    int32_t idx = LastAAxisIdx(ctx);
    while (idx >= 0) {
        int64_t axisSize = (idx == ctx.axisNum - 1 && !ctx.isTailR) ?
                               Ops::Base::CeilAlign(ctx.axisShape[static_cast<size_t>(idx)], bsElem) :
                               ctx.axisShape[static_cast<size_t>(idx)];
        if (product * axisSize > aUnitMax) {
            break;
        }
        product *= axisSize;
        idx -= AXIS_INTERVAL;
    }

    if (idx < 0) {
        ctx.aSplitIdx = 0;
        ctx.aUbFactor = ctx.axisShape[0];
    } else {
        ctx.aSplitIdx = idx;
        ctx.aUbFactor = std::min(aUnitMax / product, ctx.axisShape[static_cast<size_t>(idx)]);
        ctx.aUbFactor = std::max<int64_t>(ctx.aUbFactor, 1);
    }

    ComputeInnerAProdAlign(ctx);
    OP_LOGD(context->GetNodeName(), "ExpandAIfRFullyLoaded: triggered, aSplitIdx=%d aUbFactor=%ld innerAProdAlign=%ld",
            ctx.aSplitIdx, ctx.aUbFactor, ctx.innerAProdAlign);
    return ge::GRAPH_SUCCESS;
}

// A 方向多核切分：aLoopCntTotal = aSplitIdx 外侧 A 轴乘积 × 切分轴 chunk 数（外层 A 轴
// 视作一个混合进制循环，即 "fused"）；按核数做大小核均分——前 aLoopCntTotal%coreNum 个核
// 多干 1 轮；usedCoreNum = 全核（小核轮数>0）或仅大核数。迭代-核映射固定，保证
// bitwise_reproducible。
static void ComputeFusedALoopSplit(EuclideanNormCtx& ctx)
{
    int64_t outerAProd = 1;
    for (int32_t i = 0; i < ctx.aSplitIdx; i += AXIS_INTERVAL) {
        outerAProd *= ctx.axisShape[static_cast<size_t>(i)];
    }
    const int64_t aSplitAxisSize = ctx.axisShape[static_cast<size_t>(ctx.aSplitIdx)];
    ctx.aSplitChunkCnt = Ops::Base::CeilDiv(aSplitAxisSize, ctx.aUbFactor);
    ctx.aLoopCntTotal = outerAProd * ctx.aSplitChunkCnt;

    ctx.aSmallCoreLoopCnt = ctx.aLoopCntTotal / ctx.coreNum;
    ctx.aBigCoreCnt = static_cast<int32_t>(ctx.aLoopCntTotal % ctx.coreNum);
    ctx.aBigCoreLoopCnt = ctx.aSmallCoreLoopCnt + (ctx.aBigCoreCnt > 0 ? 1 : 0);
    ctx.usedCoreNum = (ctx.aSmallCoreLoopCnt > 0) ? static_cast<int32_t>(ctx.coreNum) : ctx.aBigCoreCnt;
    ctx.usedCoreNum = std::max(ctx.usedCoreNum, 1);
}

// R 方向总迭代数：rLoopCntTotal = rSplitIdx 外侧 R 轴乘积 × CeilDiv(切分轴长, rUbFactor)；
// 供 kernel Init 推导二分树参数（bisectionPos/Tail/cacheCount）与 Group 触发判定。
static void ComputeRLoopCnt(EuclideanNormCtx& ctx)
{
    int64_t outerR = 1;
    for (int32_t i = 1; i < ctx.rSplitIdx; i += AXIS_INTERVAL) {
        outerR *= ctx.axisShape[static_cast<size_t>(i)];
    }
    const int64_t rChunks = Ops::Base::CeilDiv(ctx.axisShape[static_cast<size_t>(ctx.rSplitIdx)], ctx.rUbFactor);
    ctx.rLoopCntTotal = outerR * rChunks;
}

// 计算 UB buffer 尺寸：preBufSize = aUnit × rPaddedElems × maxDtypeSize
// （preIn / preRes / preResTail 三个 buffer 同尺寸，统一按 maxDtypeSize 预算，b16 也按 4B 计）；
// postBufSize = CeilAlign(aUnit × maxDtypeSize, 32B)。
static void ComputeUbSizes(EuclideanNormCtx& ctx)
{
    const int64_t aUnit = ctx.aUbFactor * ctx.innerAProdAlign;
    const int64_t rPaddedElems = ctx.rUbFactorAlign * ctx.innerRProdAlign;
    ctx.preBufSize = aUnit * rPaddedElems * ctx.maxDtypeSize;
    ctx.postBufSize = Ops::Base::CeilAlign(aUnit * ctx.maxDtypeSize, ctx.blockSize);
}

// Group 2D 分核触发判定：A 迭代总数 ≤ 核数/GROUP_CORE_RATIO（A 方向并行度吃不满半数核）
// 且 R 迭代数 > 1（R 方向有多轮迭代可切）——半数核闲置且 R 有并行度时才值得 2D 分核。
static bool ShouldUseGroup(const EuclideanNormCtx& ctx)
{
    if (ctx.aLoopCntTotal > ctx.coreNum / GROUP_CORE_RATIO) {
        return false;
    }
    if (ctx.rLoopCntTotal <= 1) {
        return false;
    }
    return true;
}

// Group 2D 分核参数：totalOuter = aLoopCntTotal × rLoopCntTotal 按核均分得 numBlocks，
// 再把 numBlocks 对齐到 aLoopCntTotal 的整数倍（CeilAlign 超核数则 FloorAlign）——保证每核
// = 整数个完整 A chunk × 一段完整 R 区间（Phase1 核号分解 blockIdx = aChunkIdx×rGroupCnt +
// rChunkIdx 依赖此性质）；得 usedCoreNum 与 rGroupCnt（R 分组数 = Phase2 workspace 行数）。
static void ComputeGroupSplit(EuclideanNormCtx& ctx)
{
    int64_t totalOuter = ctx.aLoopCntTotal * ctx.rLoopCntTotal;
    int64_t perCoreNum = Ops::Base::CeilDiv(totalOuter, ctx.coreNum);
    int64_t numBlocks = Ops::Base::CeilDiv(totalOuter, perCoreNum);

    if (Ops::Base::CeilAlign(numBlocks, ctx.aLoopCntTotal) <= ctx.coreNum) {
        numBlocks = Ops::Base::CeilAlign(numBlocks, ctx.aLoopCntTotal);
    } else {
        numBlocks = Ops::Base::FloorAlign(numBlocks, ctx.aLoopCntTotal);
    }

    ctx.usedCoreNum = static_cast<int32_t>(numBlocks);
    ctx.rGroupCnt = numBlocks / ctx.aLoopCntTotal;
    ctx.isGroup = true;
}

// 申报 workspace 大小：Group 时用户 workspace = rGroupCnt × aTotal × sizeof(float)
// （[rGroupCnt 行, aTotal 列] fp32 行优先 dense 部分和矩阵，Phase1 写入 / Phase2 RA 归约读出），
// 叠加系统 workspace（GetLibApiWorkSpaceSize）；normal / empty 仅需系统部分。
static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context, const EuclideanNormCtx& ctx)
{
    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    size_t usrSize = 0;
    if (ctx.isGroup) {
        int64_t aTotal = TotalAProd(ctx);
        usrSize = static_cast<size_t>(ctx.rGroupCnt) * static_cast<size_t>(aTotal) * sizeof(float);
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    ws[0] = usrSize + sysWorkspaceSize;
    OP_LOGI(context, "Set ws size:%lu, usrSize:%lu", ws[0], usrSize);
    return ge::GRAPH_SUCCESS;
}

// 把 ctx 全量写入 EuclideanNormTilingData：memset_s 清零后逐字段填充（未用轴槽
// axisShape=1 / axisStride=0，定长数组约定），cacheBufUbSize 直填固定 16KB；
// axisStride 现场由 ComputeAxisStrides 计算；随后打全量 INFO 日志（pattern / 切分 / 多核参数）。
static ge::graphStatus FillAndLogTilingData(gert::TilingContext* context, const EuclideanNormCtx& ctx)
{
    EuclideanNormTilingData* td = context->GetTilingData<EuclideanNormTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    OP_CHECK_IF(memset_s(td, sizeof(EuclideanNormTilingData), 0, sizeof(EuclideanNormTilingData)) != EOK,
                OP_LOGE(context, "Memset tilingdata error"), return ge::GRAPH_FAILED);

    int64_t axisStride[MAX_PATTERN_RANK] = {0};
    ComputeAxisStrides(ctx, axisStride);

    td->axisNum = ctx.axisNum;
    for (int32_t i = 0; i < MAX_PATTERN_RANK; ++i) {
        td->axisShape[i] = (i < ctx.axisNum) ? ctx.axisShape[static_cast<size_t>(i)] : 1;
        td->axisStride[i] = (i < ctx.axisNum) ? axisStride[i] : 0;
    }
    td->aLoopCntTotal = ctx.aLoopCntTotal;
    td->aSplitChunkCnt = ctx.aSplitChunkCnt;
    td->aBigCoreLoopCnt = ctx.aBigCoreLoopCnt;
    td->aSmallCoreLoopCnt = ctx.aSmallCoreLoopCnt;
    td->aBigCoreCnt = ctx.aBigCoreCnt;
    td->usedCoreNum = ctx.usedCoreNum;
    td->aSplitIdx = ctx.aSplitIdx;
    td->rSplitIdx = ctx.rSplitIdx;
    td->aUbFactor = ctx.aUbFactor;
    td->rUbFactor = ctx.rUbFactor;
    td->rUbFactorAlign = ctx.rUbFactorAlign;
    td->innerAProdAlign = ctx.innerAProdAlign;
    td->innerRProdAlign = ctx.innerRProdAlign;
    td->rLoopCntTotal = ctx.rLoopCntTotal;
    td->preBufSize = ctx.preBufSize;
    td->postBufSize = ctx.postBufSize;
    td->cacheBufUbSize = CACHE_BUF_BYTES;
    td->rGroupCnt = ctx.rGroupCnt;

    OP_LOGI(context, "EuclideanNorm tiling: dtype=%d axisNum=%d isTailR=%d usedCoreNum=%d isGroup=%d",
            static_cast<int>(ctx.xDtype), ctx.axisNum, static_cast<int>(ctx.isTailR), ctx.usedCoreNum,
            static_cast<int>(ctx.isGroup));
    OP_LOGI(context, "  aSplitIdx=%d aUbFactor=%ld rSplitIdx=%d rUbFactor=%ld rUbFactorAlign=%ld", td->aSplitIdx,
            td->aUbFactor, td->rSplitIdx, td->rUbFactor, td->rUbFactorAlign);
    OP_LOGI(context, "  innerAProdAlign=%ld innerRProdAlign=%ld rLoopCntTotal=%ld", td->innerAProdAlign,
            td->innerRProdAlign, td->rLoopCntTotal);
    OP_LOGI(context, "  UB: preBuf=%ld postBuf=%ld cache=%ld", td->preBufSize, td->postBufSize, td->cacheBufUbSize);
    OP_LOGI(context, "  aLoopCntTotal=%ld aSplitChunkCnt=%ld aBigCoreLoopCnt=%ld aSmallCoreLoopCnt=%ld aBigCoreCnt=%d",
            td->aLoopCntTotal, td->aSplitChunkCnt, td->aBigCoreLoopCnt, td->aSmallCoreLoopCnt, td->aBigCoreCnt);
    OP_LOGI(context, "  rGroupCnt=%ld", td->rGroupCnt);
    std::string shapeStr = "[";
    for (int32_t i = 0; i < td->axisNum; ++i) {
        if (i > 0) {
            shapeStr += ", ";
        }
        shapeStr += std::to_string(td->axisShape[i]);
    }
    shapeStr += "]";
    OP_LOGI(context, "  axisShape=%s", shapeStr.c_str());
    std::string strideStr = "[";
    for (int32_t i = 0; i < td->axisNum; ++i) {
        if (i > 0) {
            strideStr += ", ";
        }
        strideStr += std::to_string(td->axisStride[i]);
    }
    strideStr += "]";
    OP_LOGI(context, "  axisStride=%s", strideStr.c_str());
    return ge::GRAPH_SUCCESS;
}

// EMPTY_R 输出填充切分（无需读输入，仅向 y 写 aTotal 个 0）：
// aUbFactor = clamp(max(每核 ≥4KB 防碎片, CeilDiv(aTotal, coreNum)), 单 buf ≤64KB 上限, aTotal)；
// 输出区间按大小核协议均分；postBufSize = CeilAlign(max(aUbFactor×maxDtypeSize, 32B), 32B)；
// preBufSize = 0（不申请任何输入侧 buffer）。
static void ComputeEmptyRTiling(EuclideanNormCtx& ctx)
{
    constexpr int64_t MAX_SINGLE_UB_BYTES = 64 * 1024;
    const int64_t maxUbFactor =
        std::min(MAX_SINGLE_UB_BYTES / ctx.maxDtypeSize, ctx.ubSize / P_POST / ctx.maxDtypeSize);

    constexpr int64_t MIN_BYTES_PER_CORE = 4096;
    const int64_t minAPerCore = Ops::Base::CeilDiv(MIN_BYTES_PER_CORE, ctx.maxDtypeSize);
    const int64_t aTotal = ctx.aTotalEmpty;

    int64_t aUbFactor = std::max(minAPerCore, Ops::Base::CeilDiv(aTotal, ctx.coreNum));
    aUbFactor = std::min(aUbFactor, maxUbFactor);
    aUbFactor = std::min(aUbFactor, aTotal);
    aUbFactor = std::max<int64_t>(aUbFactor, 1);

    const int64_t aLoopCntTotal = Ops::Base::CeilDiv(aTotal, aUbFactor);
    const int64_t aSmallCoreLoopCnt = aLoopCntTotal / ctx.coreNum;
    const int32_t aBigCoreCnt = static_cast<int32_t>(aLoopCntTotal % ctx.coreNum);
    const int64_t aBigCoreLoopCnt = aSmallCoreLoopCnt + (aBigCoreCnt > 0 ? 1 : 0);
    const int32_t usedCoreNum = (aSmallCoreLoopCnt > 0) ? static_cast<int32_t>(ctx.coreNum) : std::max(aBigCoreCnt, 1);

    ctx.aUbFactor = aUbFactor;
    ctx.aLoopCntTotal = aLoopCntTotal;
    ctx.aSplitChunkCnt = aLoopCntTotal;
    ctx.aSmallCoreLoopCnt = aSmallCoreLoopCnt;
    ctx.aBigCoreLoopCnt = aBigCoreLoopCnt;
    ctx.aBigCoreCnt = aBigCoreCnt;
    ctx.usedCoreNum = usedCoreNum;

    int64_t outRaw = aUbFactor * ctx.maxDtypeSize;
    outRaw = std::max(outRaw, ctx.blockSize);
    ctx.postBufSize = Ops::Base::CeilAlign(outRaw, ctx.blockSize);
    ctx.preBufSize = 0;
}

// 填充 EuclideanNormEmptyTilingData：memset 清零后，EMPTY_A 全零直接返回（kernel 全核早退）；
// EMPTY_R 填输出切分 7 字段（usedCoreNum / aTotal / aUbFactor / 大小核三参 / postBufSize）。
static ge::graphStatus FillEmptyTilingData(gert::TilingContext* context, const EuclideanNormCtx& ctx)
{
    EuclideanNormEmptyTilingData* td = context->GetTilingData<EuclideanNormEmptyTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    OP_CHECK_IF(memset_s(td, sizeof(EuclideanNormEmptyTilingData), 0, sizeof(EuclideanNormEmptyTilingData)) != EOK,
                OP_LOGE(context, "Memset tilingdata error"), return ge::GRAPH_FAILED);

    if (ctx.emptyKind == EuclideanNormEmptyKind::EMPTY_A) {
        OP_LOGI(context, "EuclideanNorm EMPTY_A: dtype=%d usedCoreNum=0", static_cast<int>(ctx.xDtype));
        return ge::GRAPH_SUCCESS;
    }

    td->aTotal = ctx.aTotalEmpty;
    td->usedCoreNum = ctx.usedCoreNum;
    td->aUbFactor = ctx.aUbFactor;
    td->aBigCoreCnt = ctx.aBigCoreCnt;
    td->aBigCoreLoopCnt = ctx.aBigCoreLoopCnt;
    td->aSmallCoreLoopCnt = ctx.aSmallCoreLoopCnt;
    td->postBufSize = ctx.postBufSize;

    OP_LOGI(context, "EuclideanNorm EMPTY_R: dtype=%d aTotal=%ld aUbFactor=%ld usedCoreNum=%d postBuf=%ld",
            static_cast<int>(ctx.xDtype), ctx.aTotalEmpty, td->aUbFactor, td->usedCoreNum, td->postBufSize);
    return ge::GRAPH_SUCCESS;
}

// 空 tensor 快速路径总装（主流程短路出口，跳过整套 normal 切分）：
// EMPTY_R → ComputeEmptyRTiling；EMPTY_A → usedCoreNum=0（kernel 全核零操作）→
// FillEmptyTilingData → 固定 tilingKey(isGroup=0, isEmptyTensor=1) →
// SetBlockDim(max(usedCoreNum,1))（EMPTY_A 启 1 核即退）→ SetWorkspaceSize（仅系统 ws）。
static ge::graphStatus HandleEmptyTensor(gert::TilingContext* context, EuclideanNormCtx& ctx)
{
    if (ctx.emptyKind == EuclideanNormEmptyKind::EMPTY_R) {
        ComputeEmptyRTiling(ctx);
        OP_LOGD(context->GetNodeName(), "EmptyR: aUbFactor=%ld usedCoreNum=%d postBufSize=%ld", ctx.aUbFactor,
                ctx.usedCoreNum, ctx.postBufSize);
    } else {
        ctx.usedCoreNum = 0;
    }
    OP_CHECK_IF(FillEmptyTilingData(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(0U), static_cast<uint64_t>(1U));
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(static_cast<uint32_t>(std::max(ctx.usedCoreNum, 1)));

    OP_CHECK_IF(SetWorkspaceSize(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// TilingFunc 主入口（IMPL_OP_OPTILING 注册），编排全链：
// ① 平台/输入/axes 解析 → ② 空分类（非 NORMAL 走 HandleEmptyTensor 直接返回）→
// ③ A/R 规整 → ④ A 切分 → R 切分 → R 全载扩 A（UB 三步算法）→
// ⑤ A 多核均分 + R 迭代数 → ⑥ Group 判定（触发则 ComputeGroupSplit + SetScheduleMode(1)
//    batch mode，kernel SyncAll 依赖）→ ⑦ UB 尺寸 + 填 TilingData →
// ⑧ SetTilingKey(isGroup, isEmptyTensor=0) + SetBlockDim + SetWorkspaceSize。
static ge::graphStatus EuclideanNormTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "Begin EuclideanNormTilingFunc");
    EuclideanNormCtx ctx;

    OP_CHECK_IF(GetPlatformInfo(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetShapeAndDtype(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseAxesTensor(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    BuildInitialAxisList(ctx);
    ClassifyEmptyTensor(ctx);

    if (ctx.emptyKind != EuclideanNormEmptyKind::NORMAL) {
        return HandleEmptyTensor(context, ctx);
    }

    OP_CHECK_IF(PreprocessPattern(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    ComputeAUbFactor(ctx);
    OP_LOGD(context->GetNodeName(), "ComputeAUbFactor: aSplitIdx=%d aUbFactor=%ld innerAProdAlign=%ld", ctx.aSplitIdx,
            ctx.aUbFactor, ctx.innerAProdAlign);
    OP_CHECK_IF(ComputeRUbFactor(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(ExpandAIfRFullyLoaded(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    ComputeFusedALoopSplit(ctx);
    ComputeRLoopCnt(ctx);
    OP_LOGD(context->GetNodeName(), "MultiCore: aLoopCntTotal=%ld aSplitChunkCnt=%ld usedCoreNum=%d rLoopCntTotal=%ld",
            ctx.aLoopCntTotal, ctx.aSplitChunkCnt, ctx.usedCoreNum, ctx.rLoopCntTotal);
    if (ShouldUseGroup(ctx)) {
        ComputeGroupSplit(ctx);
        OP_LOGD(context->GetNodeName(), "Group: rGroupCnt=%ld usedCoreNum=%d", ctx.rGroupCnt, ctx.usedCoreNum);
        OP_CHECK_IF(context->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Failed to set ScheduleMode!"), return ge::GRAPH_FAILED);
    }
    ComputeUbSizes(ctx);
    OP_LOGD(context->GetNodeName(), "UbSizes: preBufSize=%ld postBufSize=%ld", ctx.preBufSize, ctx.postBufSize);

    OP_CHECK_IF(FillAndLogTilingData(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(static_cast<uint64_t>(ctx.isGroup ? 1U : 0U), static_cast<uint64_t>(0U));
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(static_cast<uint32_t>(std::max(ctx.usedCoreNum, 1)));

    OP_CHECK_IF(SetWorkspaceSize(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// TilingParse 空实现：EuclideanNormCompileInfo 为空结构（本算子无跨次编译缓存信息），
// 恒返回成功；后续若做 binary 复用，缓存字段挂 CompileInfo 并在此解析。
static ge::graphStatus TilingParseForEuclideanNorm([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EuclideanNorm)
    .Tiling(EuclideanNormTilingFunc)
    .TilingParse<EuclideanNormCompileInfo>(TilingParseForEuclideanNorm)
    .TilingInputsDataDependency({1});

} // namespace optiling
