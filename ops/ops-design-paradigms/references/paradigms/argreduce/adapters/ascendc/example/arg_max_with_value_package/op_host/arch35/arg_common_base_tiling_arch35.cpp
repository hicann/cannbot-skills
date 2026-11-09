/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file arg_common_base_tiling_arch35.cpp
 * \brief arg_common_base_tiling_arch35
 */

#include "log/log.h"
#include "arg_common_base_tiling_arch35.h"
#include "util/math_util.h"
#include "util/const_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"

using namespace AscendC;
namespace optiling {
static const uint64_t INPUT_IDX_X = 0;
static const uint64_t INPUT_IDX_DIMENSION = 1;
static const uint64_t OUTPUT_IDX_INDICE = 0;
static const uint64_t INDEX_DIMENSION = 0;
static const uint64_t WORK_SPACE_SIZE = static_cast<uint64_t>(16) * 1024 * 1024;
static const uint64_t VALUE_TWO = 2;
static const uint64_t DOUBLE_BUFFER_NUM = 2;
static const uint64_t TEMP_BUFFER_NUM = 3;
static const uint64_t MAX_EXTRA_BLOCK = 17;
static const uint64_t PROCESS_SIZE = 1024;
static const uint64_t VL_SIZE = 256;
static const uint64_t ALIGN_128 = 128;
static const uint64_t MIN_CUT_SIZE = 128;
static const uint64_t COMPACT_NEXTA_BYTES = 64;
static const uint64_t MAX_NEXTA_SIZE = 32;
static const uint64_t BLOCK_SIZE = 32;
static const uint64_t MAX_VALUE_UINT16 = std::numeric_limits<uint16_t>::max();
static const uint64_t MAX_VALUE_INT32 = std::numeric_limits<int32_t>::max();
static const uint64_t MAX_SIZE_USING_SINGLE_CORE = static_cast<uint64_t>(4 * 1024);
static const uint64_t CUTR_MIN_MOV = static_cast<uint64_t>(4 * 1024);
static const uint64_t CUTR_MIN_SIZE = 4;
static const float AR_BLOCK_NUM_FACTOR = 0.85f;
static constexpr int BF16_BUFFER_MULTIPLIER = 2; // BF16数据类型因float32中间结果所需的缓冲区倍数

ge::graphStatus ArgCommonBaseTilingArch35::Init(const uint64_t& coreNum, const uint64_t& ubSize,
    const uint64_t& vRegSize)
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start init ArgCommonBaseTilingArch35.");
    coreNum_ = coreNum;
    originCoreNum_ = coreNum; // 保存物理核数，GROUP_REDUCE 全链路使用
    if (coreNum_ == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "coreNum",
                                              std::to_string(coreNum_).c_str(),
                                              "The value of coreNum must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    ubSize_ = ubSize;
    if (ubSize_ == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "ubSize", std::to_string(ubSize_).c_str(),
                                              "The value of ubSize must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    vRegSize_ = vRegSize;
    if (vRegSize_ == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "vRegSize",
                                              std::to_string(vRegSize_).c_str(),
                                              "The value of vRegSize must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ArgCommonBaseTilingArch35::RunArgMaxTiling(bool withValue)
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start running Tiling4ArgMaxWithValue.");
    if (GetShapeInfo(withValue) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // UB split
    if (CalcSplitInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // fill data
    FillTilingData();
    // print data
    PrintTilingData();
    // set block dim and tilingKey
    if (tilingData_.get_realCoreNum() == 0) {
        tilingContext_->SetBlockDim(1);
    } else {
        tilingContext_->SetBlockDim(tilingData_.get_realCoreNum());
    }
    if (tilingKey_ == static_cast<uint64_t>(ArgMaxWithValueTilingMode::GROUP_REDUCE)) {
        tilingContext_->SetScheduleMode(1);
    }
    tilingContext_->SetTilingKey(tilingData_.get_tilingKey());
    size_t* workspaces = tilingContext_->GetWorkspaceSizes(1);
    workspaces[0] = workSpaceSize_ + WORK_SPACE_SIZE;
    OP_LOGD(tilingContext_->GetNodeName(), "Tiling4ArgMaxWithValue success.");
    return ge::GRAPH_SUCCESS;
}

uint64_t ArgCommonBaseTilingArch35::CalcCutNextA(uint64_t cutNextASize)
{
    uint64_t nextASize = tilingData_.get_nextASize();
    cutNextASize = std::min(cutNextASize, nextASize);
    if (cutNextASize <= static_cast<uint64_t>(1)) {
        return 1;
    }
    for (uint64_t i = 0; i < nextASize; i++) {
        if (Ops::Base::CeilDiv(nextASize, cutNextASize) * tilingData_.get_aSize() < coreNum_ && cutNextASize > 1) {
            cutNextASize--;
        } else {
            break;
        }
    }
    if (nextASize % Ops::Base::CeilDiv(nextASize, cutNextASize) == 0) {
        cutNextASize = nextASize / Ops::Base::CeilDiv(nextASize, cutNextASize);
    } else {
        cutNextASize = Ops::Base::CeilDiv(nextASize, Ops::Base::CeilDiv(nextASize, cutNextASize));
    }

    return cutNextASize;
}

ge::graphStatus ArgCommonBaseTilingArch35::GetShapeInfo(bool withValue)
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering ArgCommonBaseTilingArch35 GetShapeInfo.");
    auto xTensor = tilingContext_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, xTensor);
    auto xShape = Ops::Base::EnsureNotScalar(xTensor->GetStorageShape());
    xDimNum_ = xShape.GetDimNum();
    auto xDesc = tilingContext_->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, xDesc);
    auto xDtype = xDesc->GetDataType();
    eleLenInBytes_ = ge::GetSizeByDataType(xDtype);
    valueDtype_ = xDtype;

    auto indiceDesc = tilingContext_->GetOutputDesc(OUTPUT_IDX_INDICE);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, indiceDesc);
    auto indiceDtype = indiceDesc->GetDataType();
    eleLenIndiceBytes_ = ge::GetSizeByDataType(indiceDtype);
    indexDtypeSize_ = eleLenIndiceBytes_;
    indiceDtype_ = indiceDtype;
    if (eleLenInBytes_ == static_cast<uint64_t>(0) || eleLenIndiceBytes_ == static_cast<uint64_t>(0)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            tilingContext_->GetNodeName(), "x or indice",
            ("xDtypeSize=" + std::to_string(eleLenInBytes_) + ", indiceDtypeSize=" + std::to_string(eleLenIndiceBytes_))
                .c_str(),
            "The data type size must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    elementPerBlock_ = BLOCK_SIZE / eleLenInBytes_;
    if (ge::DT_BF16 == xDtype) {
        eleLenInBytes_ *= VALUE_TWO;
    }
    if (ge::DT_INT64 == indiceDtype && ge::DT_INT32 == xDtype) {
        eleLenIndiceBytes_ *= VALUE_TWO;
    } else if (ge::DT_INT32 == indiceDtype && ge::DT_INT64 == xDtype) {
        eleLenIndiceBytes_ *= (VALUE_TWO * VALUE_TWO);
    }

    valueDtypeSize_ = ge::GetSizeByDataType(valueDtype_);
    if (valueDtype_ == ge::DT_INT64) {
        t2Size_ = static_cast<uint64_t>(sizeof(int64_t));
    } else {
        t2Size_ = static_cast<uint64_t>(sizeof(int32_t));
    }
    indiceDtypeSize_ = ge::GetSizeByDataType(indiceDtype_);

    if (GetDimension(withValue) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (SetShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ArgCommonBaseTilingArch35::GetDimension(bool withValue)
{
    if (withValue) {
        auto* attrs = tilingContext_->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);
        const auto* dimension = attrs->GetAttrPointer<int64_t>(INDEX_DIMENSION);
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, dimension);
        dimension_ = *dimension;
    } else {
        if (xDimNum_ == 1) {
            dimension_ = 0;
            OP_LOGI(tilingContext_->GetNodeName(), "the input shape len is %ld", xDimNum_);
        } else {
            if (!(Ops::Base::GetConstInt(tilingContext_, INPUT_IDX_DIMENSION, dimension_))) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "dimension", "null",
                                                      "Failed to get const data for dimension.");
                return ge::GRAPH_FAILED;
            }
        }
        OP_LOGI(tilingContext_->GetNodeName(), "The dimension is %ld", dimension_);
    }
    if (dimension_ >= xDimNum_ || dimension_ + xDimNum_ < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "dimension",
                                              std::to_string(dimension_).c_str(),
                                              ("The value of dimension must be within the range [-" +
                                               std::to_string(xDimNum_) + ", " + std::to_string(xDimNum_ - 1) + "].")
                                                  .c_str());
        return ge::GRAPH_FAILED;
    }
    dimension_ = dimension_ < 0 ? xDimNum_ + dimension_ : dimension_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ArgCommonBaseTilingArch35::SetShapeInfo()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering ArgCommonBaseTilingArch35 SetShapeInfo.");
    auto xTensor = tilingContext_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, xTensor);
    auto xShape = Ops::Base::EnsureNotScalar(xTensor->GetStorageShape());

    uint64_t aSize = 1;
    uint64_t nextASize = 1;
    for (int64_t i = 0; i < dimension_; i++) {
        aSize *= xShape.GetDim(i);
    }
    for (int64_t i = dimension_ + 1; i < xDimNum_; i++) {
        nextASize *= xShape.GetDim(i);
    }
    if (xShape.GetDim(dimension_) == 1) {
        tilingKey_ = static_cast<uint64_t>(ArgMaxWithValueTilingMode::COPY_ONLY);
    } else if (dimension_ == xDimNum_ - static_cast<int64_t>(1) || nextASize == static_cast<uint64_t>(1)) {
        tilingKey_ = static_cast<uint64_t>(ArgMaxWithValueTilingMode::AR_CUT_A);
    } else {
        tilingKey_ = static_cast<uint64_t>(ArgMaxWithValueTilingMode::ARA_CUT_A);
    }

    // 空tensor
    if (aSize * nextASize * xShape.GetDim(dimension_) == 0) {
        isEmptyTensor_ = true;
        return ge::GRAPH_SUCCESS;
    } else if (xShape.GetDim(dimension_) < 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            tilingContext_->GetNodeName(), "x",
            ("aSize=" + std::to_string(aSize) + ", nextASize=" + std::to_string(nextASize) +
             ", R=" + std::to_string(xShape.GetDim(dimension_)))
                .c_str(),
            "The value of the reduction dimension must be greater than or equal to 0.");
        return ge::GRAPH_FAILED;
    }

    tilingData_.set_aSize(aSize);
    tilingData_.set_nextASize(nextASize);
    tilingData_.set_rSize(xShape.GetDim(dimension_));

    // 动态核数：每 4KB 输入数据启用一个核（空/负 tensor 已在上文 return）。
    // originCoreNum_ 保留物理核数供 GROUP_REDUCE 全链路使用。
    uint64_t dataBytes = aSize * nextASize * static_cast<uint64_t>(xShape.GetDim(dimension_)) * eleLenInBytes_;
    uint64_t dynamicCore = Ops::Base::CeilDiv(dataBytes, MAX_SIZE_USING_SINGLE_CORE);
    coreNum_ = std::max(static_cast<uint64_t>(1), std::min(dynamicCore, originCoreNum_));

    // 新增性能分支
    SetShapeInfoHighPerf();

    if ((dimension_ == xDimNum_ - 1 || nextASize == static_cast<uint64_t>(1)) &&
        (tilingKey_ == static_cast<uint64_t>(ArgMaxWithValueTilingMode::GROUP_REDUCE)) &&
        aSize > AR_BLOCK_NUM_FACTOR * originCoreNum_) {
        tilingKey_ = static_cast<uint64_t>(ArgMaxWithValueTilingMode::AR_CUT_A);
    }

    ubElement_ = (ubSize_ / VALUE_TWO - PROCESS_SIZE * (eleLenInBytes_ + eleLenIndiceBytes_) * VALUE_TWO - VL_SIZE) /
                 eleLenInBytes_;

    return ge::GRAPH_SUCCESS;
}

void ArgCommonBaseTilingArch35::SetShapeInfoHighPerf()
{
    // copy only分支直接返回
    if (tilingData_.get_rSize() == 1) {
        return;
    }

    uint64_t allASize = tilingData_.get_aSize() * tilingData_.get_nextASize();
    uint64_t arSize = tilingData_.get_rSize() * tilingData_.get_nextASize();
    uint64_t maxSize = (ubSize_ / VALUE_TWO - PROCESS_SIZE * (eleLenInBytes_ + eleLenIndiceBytes_)) / VL_SIZE;
    // RA接口方案，处理R较小A较大大的场景
    if (tilingData_.get_aSize() == 1) { // RA场景
        if (tilingData_.get_nextASize() >= coreNum_ * VL_SIZE ||
            (tilingData_.get_nextASize() > 1 && tilingData_.get_rSize() < VL_SIZE)) {
            tilingKey_ = static_cast<uint64_t>(ArgMaxWithValueTilingMode::RA_CUT_A);
        }
    } else if (tilingData_.get_nextASize() == 1) { // AR场景
        // ArgMaxGatherRa 的 gather 步长 = R*eleBytes；当其 32B 对齐时产生 UB bank 冲突，
        // 故排除 R*eleBytes 为 32 倍数的场景，回退 AR_CUT_A(ArgMaxV1 非 gather)。
        if ((tilingData_.get_rSize() * eleLenInBytes_ < vRegSize_) &&
            (tilingData_.get_aSize() * eleLenInBytes_ >= coreNum_ * vRegSize_) &&
            ((tilingData_.get_rSize() * eleLenInBytes_) % BLOCK_SIZE != 0)) {
            tilingKey_ = static_cast<uint64_t>(ArgMaxWithValueTilingMode::AR_GATHER);
        }
    } else if (allASize * eleLenInBytes_ >= coreNum_ * vRegSize_) { // ARA场景
        if ((tilingData_.get_rSize() * eleLenInBytes_ < vRegSize_) ||
            (arSize * eleLenInBytes_ >= ubSize_ / VALUE_TWO && tilingData_.get_rSize() < maxSize)) {
            tilingKey_ = static_cast<uint64_t>(ArgMaxWithValueTilingMode::ARA_GATHER);
        }
    }

    // GroupReduce方案，处理R很大，A轴较小的场景，R的范围不超过int32最大值
    // GROUP_REDUCE 有全核 SyncAll 同步，核数判断/分核/workspace 全链路使用 originCoreNum_ 保证自洽
    if ((tilingData_.get_rSize() * eleLenInBytes_ >= originCoreNum_ * VL_SIZE) &&
        (tilingData_.get_rSize() < MAX_VALUE_INT32) &&
        ((tilingData_.get_nextASize() == 1 && tilingData_.get_aSize() < originCoreNum_) ||
         (tilingData_.get_nextASize() > 1 && allASize < originCoreNum_ * VALUE_TWO))) {
        uint64_t outAAlign = Ops::Base::CeilDiv(allASize, elementPerBlock_) * elementPerBlock_;
        uint64_t workSpaceSize = outAAlign * VALUE_TWO * sizeof(int32_t) * originCoreNum_;
        if (ge::DT_INT64 == valueDtype_) {
            workSpaceSize = workSpaceSize * VALUE_TWO;
        }
        workSpaceSize_ = Ops::Base::CeilDiv(workSpaceSize, VL_SIZE) * VL_SIZE; // 向上256B对齐
        tilingKey_ = static_cast<uint64_t>(ArgMaxWithValueTilingMode::GROUP_REDUCE);
    }

    // ARA A轴和ANEXT轴分核方案，处理A和NEXTA较大场景
    // nextA 不切核时 (aSize > coreNum) 门槛放宽到 64B，切核时保持 MIN_CUT_SIZE
    // int64 暂未做 compact 优化，不适用放宽
    if (static_cast<uint64_t>(tilingKey_) == static_cast<uint64_t>(ArgMaxWithValueTilingMode::ARA_CUT_A) &&
        allASize * eleLenInBytes_ >= coreNum_ * MIN_CUT_SIZE) {
        uint64_t minNextABytes = (tilingData_.get_aSize() > coreNum_ && valueDtype_ != ge::DT_INT64) ?
                                     COMPACT_NEXTA_BYTES :
                                     MIN_CUT_SIZE;
        if (tilingData_.get_nextASize() * eleLenInBytes_ >= minNextABytes) {
            tilingKey_ = static_cast<uint64_t>(ArgMaxWithValueTilingMode::ARA_CUT_A_AND_NEXT_A);
        }
    }
}

ge::graphStatus ArgCommonBaseTilingArch35::CalcSplitInfo()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering ArgCommonBaseTilingArch35 CalcSplitInfo.");
    if (isEmptyTensor_) { // empty tensor
        realCoreNum_ = static_cast<uint64_t>(0);
        return ge::GRAPH_SUCCESS;
    }

    // 动态核数已在 SetShapeInfo 前移计算（每 4KB 一核），此处不再单独判断单核。

    switch (tilingKey_) {
        case static_cast<uint64_t>(ArgMaxWithValueTilingMode::COPY_ONLY):
            if (CalcSplitInfoForCopyOnly() != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            break;
        case static_cast<uint64_t>(ArgMaxWithValueTilingMode::AR_CUT_A):
            if (CalcSplitInfoForAr() != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            break;
        case static_cast<uint64_t>(ArgMaxWithValueTilingMode::ARA_CUT_A):
            if (CalcSplitInfoForArAAndTryMode4() != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            break;
        case static_cast<uint64_t>(ArgMaxWithValueTilingMode::ARA_CUT_A_AND_NEXT_A):
            if (CalcSplitInfoForArACutAAndNextA() != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            break;
        case static_cast<uint64_t>(ArgMaxWithValueTilingMode::RA_CUT_A):
            if (CalcSplitInfoForRa() != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            break;
        case static_cast<uint64_t>(ArgMaxWithValueTilingMode::AR_GATHER):
            if (CalcSplitInfoForGatherRa() != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            break;
        case static_cast<uint64_t>(ArgMaxWithValueTilingMode::ARA_GATHER):
            if (CalcSplitInfoForGatherArA() != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            break;
        case static_cast<uint64_t>(ArgMaxWithValueTilingMode::GROUP_REDUCE):
            if (CalcSplitInfoForGroupReduce() != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            break;
        default:
            break;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ArgCommonBaseTilingArch35::CalcSplitInfoForArAAndTryMode4()
{
    if (CalcSplitInfoForArA() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (aRaMode_ == static_cast<uint64_t>(ARAMode::ARA_MODE4)) {
        if (TryRouteAraMode4() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ArgCommonBaseTilingArch35::CalcSplitInfoForAr()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering ArgCommonBaseTilingArch35 CalcSplitInfoForAr.");
    if (tilingData_.get_aSize() <= coreNum_) {
        realCoreNum_ = tilingData_.get_aSize();
        blkFactor_ = static_cast<uint64_t>(1);
        blkTailFactor_ = static_cast<uint64_t>(0);
    } else {
        realCoreNum_ = (coreNum_ <= 0) ? 1 : coreNum_;
        blkFactor_ = tilingData_.get_aSize() / realCoreNum_;
        blkTailFactor_ = tilingData_.get_aSize() % realCoreNum_;
    }
    uint64_t factor = blkTailFactor_ == static_cast<uint64_t>(0) ? blkFactor_ : blkFactor_ + static_cast<uint64_t>(1);

    // 修改AR模板ubElement_
    uint16_t vlSize = 1;
    if (ge::DT_INT64 == valueDtype_) {
        vlSize = vRegSize_ / sizeof(valueDtype_) * VALUE_TWO;
    } else if (ge::DT_BF16 == valueDtype_) {
        vlSize = static_cast<uint16_t>(vRegSize_ / sizeof(float));
    } else {
        vlSize = vRegSize_ / sizeof(valueDtype_);
    }
    uint64_t alignNum = Ops::Base::CeilDiv(static_cast<uint64_t>(cutASize_) * static_cast<uint64_t>(cutRSize_),
                                           elementPerBlock_) *
                            elementPerBlock_ +
                        vlSize;
    uint64_t ubInSize = ubSize_;
    if (ge::DT_BF16 == valueDtype_) {
        ubInSize -= alignNum * sizeof(float);
    }

    if (!(ge::DT_INT32 == valueDtype_ || ge::DT_INT64 == valueDtype_)) {
        ubInSize -= processSize_ * sizeof(indiceDtype_);
    }
    ubElement_ = (ubInSize / VALUE_TWO - PROCESS_SIZE * (eleLenInBytes_ + eleLenIndiceBytes_) - VL_SIZE) /
                 eleLenInBytes_;

    if (tilingData_.get_rSize() < ubElement_ && tilingData_.get_rSize() < MAX_VALUE_UINT16) { // not need cut R
        uint64_t cutASize = std::min(ubElement_ / tilingData_.get_rSize(), PROCESS_SIZE);
        cutASize_ = static_cast<uint16_t>(std::min(factor, cutASize));
        cutRSize_ = static_cast<uint16_t>(tilingData_.get_rSize());
    } else { // need cut R
        uint64_t vlCnt = vRegSize_ / eleLenInBytes_;
        uint64_t rTempSize = std::min(ubElement_, MAX_VALUE_UINT16);
        rTempSize = std::min(rTempSize, tilingData_.get_rSize());
        rTempSize = rTempSize <= vlCnt ? rTempSize : (rTempSize / vlCnt * vlCnt);
        cutRSize_ = static_cast<uint16_t>(rTempSize);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ArgCommonBaseTilingArch35::CalcSplitInfoForArACutAAndNextA()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering ArgCommonBaseTilingArch35 CalcSplitInfoForArACutAAndNextA.");
    // A 轴分核
    blkFactor_ = Ops::Base::CeilDiv(static_cast<uint64_t>(tilingData_.get_aSize()), coreNum_);
    auto blkNumA = Ops::Base::CeilDiv(static_cast<uint64_t>(tilingData_.get_aSize()), blkFactor_);
    blkTailFactor_ = tilingData_.get_aSize() - (blkNumA - 1) * blkFactor_;
    if (blkNumA == static_cast<uint64_t>(0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "blkNumA", std::to_string(blkNumA).c_str(),
                                              "The value of blkNumA must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    uint64_t minDtypeSize = std::min(valueDtypeSize_, indiceDtypeSize_);
    if (minDtypeSize == static_cast<uint64_t>(0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "minDtypeSize",
                                              std::to_string(minDtypeSize).c_str(),
                                              "The value of minDtypeSize must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    // 优先尝试 flat-copy 路径
    uint64_t nextAAlign = Ops::Base::CeilAlign(tilingData_.get_nextASize(), BLOCK_SIZE / minDtypeSize);
    uint64_t ubNeededFlat = DOUBLE_BUFFER_NUM * tilingData_.get_rSize() * nextAAlign * valueDtypeSize_ +
                            VALUE_TWO * nextAAlign * (valueDtypeSize_ + indiceDtypeSize_) +
                            nextAAlign * (valueDtypeSize_ + t2Size_);
    if (ubNeededFlat <= ubSize_ - MAX_EXTRA_BLOCK * BLOCK_SIZE) {
        return CalcSplitInfoForArACutAAndNextAFlat(blkNumA, nextAAlign);
    }
    return CalcSplitInfoForArACutAAndNextANormal(blkNumA, minDtypeSize);
}

ge::graphStatus ArgCommonBaseTilingArch35::CalcSplitInfoForArACutAAndNextAFlat(uint64_t blkNumA, uint64_t nextAAlign)
{
    useFlatPath_ = 1;
    blkFactor2nd_ = tilingData_.get_nextASize();
    blkTailFactor2nd_ = blkFactor2nd_;
    blkNum2nd_ = 1;
    realCoreNum_ = blkNumA;
    cutNextASize_ = static_cast<uint16_t>(tilingData_.get_nextASize());
    cutRSize_ = static_cast<uint16_t>(tilingData_.get_rSize());
    uint64_t usableUbSize = ubSize_ - MAX_EXTRA_BLOCK * BLOCK_SIZE;
    uint64_t oneRowUb = DOUBLE_BUFFER_NUM * tilingData_.get_rSize() * nextAAlign * valueDtypeSize_ +
                        VALUE_TWO * nextAAlign * (valueDtypeSize_ + indiceDtypeSize_) +
                        nextAAlign * (valueDtypeSize_ + t2Size_);
    if (oneRowUb == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "oneRowUb",
                                              "oneRowUb must be greater than 0.", "");
        return ge::GRAPH_FAILED;
    }
    cutASize_ = static_cast<uint16_t>(usableUbSize / oneRowUb);
    cutASize_ = std::max(cutASize_, static_cast<uint16_t>(1));
    cutASize_ = std::min(static_cast<uint64_t>(cutASize_), blkFactor_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ArgCommonBaseTilingArch35::CalcSplitInfoForArACutAAndNextANormal(uint64_t blkNumA,
    uint64_t minDtypeSize)
{
    // ANext 分核
    auto blkNumANext = std::min(static_cast<uint64_t>(tilingData_.get_nextASize()), coreNum_ / blkNumA);
    blkFactor2nd_ = Ops::Base::CeilDiv(static_cast<uint64_t>(tilingData_.get_nextASize()), blkNumANext);
    blkNumANext = Ops::Base::CeilDiv(static_cast<uint64_t>(tilingData_.get_nextASize()), blkFactor2nd_);
    blkTailFactor2nd_ = tilingData_.get_nextASize() - (blkNumANext - 1) * blkFactor2nd_;
    realCoreNum_ = blkNumA * blkNumANext;
    // 切 nextA
    uint64_t usableUbSize = ubSize_ - MAX_EXTRA_BLOCK * BLOCK_SIZE;
    cutNextASize_ = Ops::Base::FloorDiv(
        usableUbSize, DOUBLE_BUFFER_NUM * (valueDtypeSize_ + valueDtypeSize_ + indiceDtypeSize_) + valueDtypeSize_ +
                          t2Size_ + t2Size_);
    cutNextASize_ = std::max(cutNextASize_, static_cast<uint16_t>(1));
    cutNextASize_ = std::min(static_cast<uint64_t>(cutNextASize_), blkFactor2nd_);
    // 切 R
    uint64_t nextAAlign = Ops::Base::CeilAlign(static_cast<uint64_t>(cutNextASize_), BLOCK_SIZE / minDtypeSize);
    usableUbSize = Ops::Base::FloorDiv(ubSize_, nextAAlign) - TEMP_BUFFER_NUM * valueDtypeSize_ -
                   DOUBLE_BUFFER_NUM * (t2Size_ + indiceDtypeSize_);
    cutRSize_ = Ops::Base::FloorDiv(usableUbSize, DOUBLE_BUFFER_NUM * valueDtypeSize_);
    cutRSize_ = std::max(cutRSize_, static_cast<uint16_t>(1));
    cutRSize_ = std::min(static_cast<uint64_t>(cutRSize_), tilingData_.get_rSize());
    // 取 A
    uint64_t oneRAUbSize = nextAAlign *
                           (DOUBLE_BUFFER_NUM * cutRSize_ * valueDtypeSize_ + TEMP_BUFFER_NUM * valueDtypeSize_ +
                            DOUBLE_BUFFER_NUM * indiceDtypeSize_ + DOUBLE_BUFFER_NUM * t2Size_);
    cutASize_ = Ops::Base::FloorDiv(ubSize_, oneRAUbSize);
    cutASize_ = std::max(cutASize_, static_cast<uint16_t>(1));
    cutASize_ = std::min(static_cast<uint64_t>(cutASize_), blkFactor_);
    return ge::GRAPH_SUCCESS;
}

void ArgCommonBaseTilingArch35::CalcCutA()
{
    uint64_t arSize = tilingData_.get_rSize() * tilingData_.get_nextASize();
    realCoreNum_ = coreNum_;
    blkFactor_ = tilingData_.get_aSize() / realCoreNum_;
    blkTailFactor_ = tilingData_.get_aSize() % realCoreNum_;
    uint64_t factor = Ops::Base::CeilDiv(tilingData_.get_aSize(), realCoreNum_);
    uint64_t cutASize = std::min(ubElement_, MAX_VALUE_UINT16) / arSize;
    cutASize_ = static_cast<uint16_t>(std::min(factor, cutASize));
    cutNextASize_ = static_cast<uint16_t>(tilingData_.get_nextASize());
    cutRSize_ = static_cast<uint16_t>(tilingData_.get_rSize());
}

void ArgCommonBaseTilingArch35::CalcCutRA()
{
    uint64_t vlCnt = vRegSize_ / eleLenInBytes_;
    uint64_t allASize = tilingData_.get_aSize() * tilingData_.get_nextASize();
    // 前移 realCoreNum_：allASize 不足满核时按实际任务数（吸收原 MODE5/105），
    // 分母用 realCoreNum_ 避免 allASize<=coreNum_ 时 maxCutNextA=0 报错。
    realCoreNum_ = allASize <= coreNum_ ? allASize : coreNum_;
    uint64_t maxCutNextA = allASize / realCoreNum_;
    maxCutNextA = std::min(maxCutNextA, tilingData_.get_nextASize());
    uint64_t cutNextASize = std::min(ubElement_ / tilingData_.get_rSize(), PROCESS_SIZE);
    cutNextASize = cutNextASize <= MIN_CUT_SIZE / eleLenInBytes_ ? MIN_CUT_SIZE / eleLenInBytes_ : cutNextASize;
    cutNextASize = cutNextASize <= vlCnt ? cutNextASize : (cutNextASize / vlCnt * vlCnt);
    cutNextASize = std::min(cutNextASize, maxCutNextA);
    if (cutNextASize == static_cast<uint64_t>(0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "cutNextASize",
                                              std::to_string(cutNextASize).c_str(),
                                              "The value of cutNextASize must be greater than 0.");
        return;
    }
    uint64_t rTempSize = ubElement_ / cutNextASize;
    rTempSize = std::min(rTempSize, MAX_VALUE_UINT16);
    if (rTempSize >= tilingData_.get_rSize()) {
        rTempSize = tilingData_.get_rSize();
    } else {
        rTempSize = rTempSize <= vlCnt ? rTempSize : (rTempSize / vlCnt * vlCnt);
    }
    cutNextASize_ = static_cast<uint16_t>(cutNextASize);
    cutRSize_ = static_cast<uint16_t>(rTempSize);
    blkFactor_ = tilingData_.get_aSize() * Ops::Base::CeilDiv(tilingData_.get_nextASize(), cutNextASize) / realCoreNum_;
    blkTailFactor_ = tilingData_.get_aSize() * Ops::Base::CeilDiv(tilingData_.get_nextASize(), cutNextASize) %
                     realCoreNum_;
}

ge::graphStatus ArgCommonBaseTilingArch35::CalcSplitInfoForArA()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering ArgCommonBaseTilingArch35 CalcSplitInfoForArA.");
    uint64_t arSize = tilingData_.get_rSize() * tilingData_.get_nextASize();
    // After axis combination, the dimension is recorded as A * R * nextA.
    if (tilingData_.get_aSize() >= coreNum_ && arSize <= ubElement_) { // 切A，A大于核数且R * nextA 在UB全载
        if (tilingData_.get_nextASize() * eleLenInBytes_ < MAX_NEXTA_SIZE * VALUE_TWO) {
            aRaMode_ = static_cast<uint64_t>(ARAMode::ARA_MODE1);
        } else {
            aRaMode_ = static_cast<uint64_t>(ARAMode::ARA_MODE2);
        }
        CalcCutA();
    } else if (tilingData_.get_aSize() >= coreNum_ && tilingData_.get_nextASize() * eleLenInBytes_ <
                                                          MAX_NEXTA_SIZE) { // nextA较小且R * nextA UB放不下的场景，切R
        aRaMode_ = static_cast<uint64_t>(ARAMode::ARA_MODE3);
        CalcCutA();
        uint64_t rTempSize = ubElement_ / tilingData_.get_nextASize();
        rTempSize = std::min(rTempSize, MAX_VALUE_UINT16);
        rTempSize = std::min(rTempSize, tilingData_.get_rSize());
        cutRSize_ = static_cast<uint16_t>(rTempSize);
        cutASize_ = static_cast<uint16_t>(1);
    } else { // 其余所有 ARA 场景统一走 MODE4（吸收原 MODE5/105 与 MODE6/106）
        // CalcCutRA 已修复 allASize<=coreNum_ 的分核，并按 R 是否放得下自动全载/切 R 迭代
        aRaMode_ = static_cast<uint64_t>(ARAMode::ARA_MODE4);
        CalcCutRA();
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t ArgCommonBaseTilingArch35::CalcUnitUbSpace()
{
    uint64_t divideFactor = tilingData_.get_rSize() * eleLenInBytes_ + eleLenIndiceBytes_ + eleLenInBytes_;
    // 单位空间包含数据类型转换预留的UB空间
    if (ge::DT_BF16 == valueDtype_) {
        divideFactor += tilingData_.get_rSize() * sizeof(float) + sizeof(float);
    }
    if (ge::DT_INT64 == indiceDtype_) {
        divideFactor += sizeof(int32_t);
    }

    return divideFactor;
}

ge::graphStatus ArgCommonBaseTilingArch35::CalcSplitInfoForGatherRa()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering ArgCommonBaseTilingArch35 CalcSplitInfoForGatherRa.");
    realCoreNum_ = coreNum_;
    blkFactor_ = tilingData_.get_aSize() / realCoreNum_;
    blkTailFactor_ = tilingData_.get_aSize() % realCoreNum_;
    uint64_t factor = blkTailFactor_ == static_cast<uint64_t>(0) ? blkFactor_ : blkFactor_ + static_cast<uint64_t>(1);
    // Only cut A, cutRSize_ is rSize
    uint64_t vlCnt = vRegSize_ / eleLenInBytes_;
    uint64_t aSize = CalcUnitUbSpace();
    if (aSize == static_cast<uint64_t>(0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "unitUbSpace",
                                              std::to_string(aSize).c_str(),
                                              "The value of unitUbSpace must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    uint64_t calcASize = ubSize_ / VALUE_TWO / aSize;
    uint64_t cutASize = calcASize <= vlCnt ? calcASize : (calcASize / vlCnt * vlCnt); // 向下对齐到VLSIZE
    cutASize_ = static_cast<uint16_t>(std::min(factor, cutASize));
    cutRSize_ = static_cast<uint16_t>(tilingData_.get_rSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ArgCommonBaseTilingArch35::CalcSplitInfoForRa()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering ArgCommonBaseTilingArch35 CalcSplitInfoForRa.");
    // NextA must be greater than cornNum * VL
    realCoreNum_ = coreNum_;
    blkFactor_ = tilingData_.get_nextASize() / realCoreNum_;
    blkTailFactor_ = tilingData_.get_nextASize() % realCoreNum_;
    uint64_t factor = blkTailFactor_ == static_cast<uint64_t>(0) ? blkFactor_ : blkFactor_ + static_cast<uint64_t>(1);

    uint64_t vlCnt = VL_SIZE / eleLenInBytes_;
    uint64_t aSize = CalcUnitUbSpace();
    if (aSize == static_cast<uint64_t>(0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "unitUbSpace",
                                              std::to_string(aSize).c_str(),
                                              "The value of unitUbSpace must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    uint64_t calcASize = ubSize_ / VALUE_TWO / aSize;
    if (calcASize >= factor) { // Only cut nextA, cutRSize_ is rSize
        cutNextASize_ = static_cast<uint16_t>(factor);
        cutRSize_ = static_cast<uint16_t>(tilingData_.get_rSize());
    } else if (calcASize >= processSize_) {
        uint64_t cutNextASize = calcASize <= vlCnt ? calcASize : (calcASize / vlCnt * vlCnt);
        cutNextASize_ = static_cast<uint16_t>(cutNextASize);
        cutRSize_ = static_cast<uint16_t>(tilingData_.get_rSize());
    } else { // need cut nextA and r
        uint64_t cutNextASize = std::min(factor, processSize_);
        uint64_t outAAlign = Ops::Base::CeilDiv(cutNextASize, elementPerBlock_) * elementPerBlock_;
        uint64_t cutRSize = (ubSize_ / VALUE_TWO - VL_SIZE -
                             outAAlign * VALUE_TWO * (eleLenInBytes_ + eleLenIndiceBytes_)) /
                            outAAlign / eleLenInBytes_;
        if (ge::DT_BF16 == valueDtype_) {
            cutRSize = (ubSize_ / VALUE_TWO - VL_SIZE -
                        outAAlign * VALUE_TWO * (eleLenInBytes_ * VALUE_TWO + eleLenIndiceBytes_)) /
                       outAAlign / eleLenInBytes_;
        }
        cutNextASize_ = static_cast<uint16_t>(cutNextASize);
        cutRSize_ = static_cast<uint16_t>(std::min(tilingData_.get_rSize(), cutRSize));
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ArgCommonBaseTilingArch35::CalcSplitInfoForGatherArA()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering ArgCommonBaseTilingArch35 CalcSplitInfoForGatherArA.");
    uint64_t arSize = CalcUnitUbSpace() * Ops::Base::CeilDiv(tilingData_.get_nextASize(), elementPerBlock_) *
                      elementPerBlock_;
    uint64_t factor = 0;
    uint64_t calcASize = 0;
    if (tilingData_.get_aSize() >= coreNum_ && arSize < ubSize_ / VALUE_TWO) { // Cores are divided based on A
        realCoreNum_ = coreNum_;
        blkFactor_ = tilingData_.get_aSize() / realCoreNum_;
        blkTailFactor_ = tilingData_.get_aSize() % realCoreNum_;
        factor = blkTailFactor_ == static_cast<uint64_t>(0) ? blkFactor_ : blkFactor_ + static_cast<uint64_t>(1);
        calcASize = ubSize_ / VALUE_TWO / arSize;

        cutASize_ = static_cast<uint16_t>(std::min(factor, calcASize));
        cutNextASize_ = static_cast<uint16_t>(tilingData_.get_nextASize());
        cutRSize_ = static_cast<uint16_t>(tilingData_.get_rSize());
    } else { // Cores are divided based on A * nextA
        realCoreNum_ = coreNum_;
        blkFactor_ = tilingData_.get_aSize() * tilingData_.get_nextASize() / realCoreNum_;
        cutASize_ = static_cast<uint16_t>(1);
        uint64_t aSize = CalcUnitUbSpace();
        if (aSize == static_cast<uint64_t>(0)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "unitUbSpace",
                                                  std::to_string(aSize).c_str(),
                                                  "The value of unitUbSpace must be greater than 0.");
            return ge::GRAPH_FAILED;
        }
        ubElement_ = ubSize_ / VALUE_TWO / aSize;
        uint64_t cutNextASize = static_cast<uint16_t>(std::min(blkFactor_, ubElement_));

        // 计算cutNextASize使其充分启用多核
        cutNextASize = CalcCutNextA(cutNextASize);
        uint64_t AlignCutNextASize = Ops::Base::CeilDiv(cutNextASize, elementPerBlock_) * elementPerBlock_;
        uint64_t realMaxUbSize = (ubSize_ / VALUE_TWO - AlignCutNextASize * (eleLenInBytes_ + eleLenIndiceBytes_));
        if (tilingData_.get_rSize() * AlignCutNextASize * eleLenInBytes_ > realMaxUbSize) {
            cutNextASize = Ops::Base::FloorDiv(cutNextASize, elementPerBlock_) * elementPerBlock_;
        }
        blkFactor_ = tilingData_.get_aSize() * Ops::Base::CeilDiv(tilingData_.get_nextASize(), cutNextASize) /
                     realCoreNum_;
        blkTailFactor_ = tilingData_.get_aSize() * Ops::Base::CeilDiv(tilingData_.get_nextASize(), cutNextASize) %
                         realCoreNum_;
        cutNextASize_ = static_cast<uint16_t>(cutNextASize);
        cutRSize_ = static_cast<uint16_t>(tilingData_.get_rSize());
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ArgCommonBaseTilingArch35::CalcSplitInfoForCopyOnly()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering ArgCommonBaseTilingArch35 CalcSplitInfoForCopyOnly.");
    uint64_t sumSize = tilingData_.get_aSize() * tilingData_.get_nextASize();
    if (sumSize <= coreNum_) {
        realCoreNum_ = sumSize;
        blkFactor_ = static_cast<uint64_t>(1);
        blkTailFactor_ = static_cast<uint64_t>(0);
    } else {
        realCoreNum_ = (coreNum_ <= 0) ? 1 : coreNum_;
        blkFactor_ = sumSize / realCoreNum_;
        blkTailFactor_ = sumSize % realCoreNum_;
    }

    cutASize_ = static_cast<uint16_t>(
        ((ubSize_ - PROCESS_SIZE * eleLenIndiceBytes_ - VL_SIZE) / (VALUE_TWO * eleLenInBytes_)));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ArgCommonBaseTilingArch35::CalcSplitInfoForGroupReduce()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering ArgCommonBaseTilingArch35 CalcSplitInfoForGroupReduce.");
    uint64_t aSize = tilingData_.get_aSize();
    uint64_t rSize = tilingData_.get_rSize();
    uint64_t nextASize = tilingData_.get_nextASize();
    uint64_t allASize = aSize * nextASize;
    uint64_t nextAAlign = Ops::Base::CeilDiv(nextASize, elementPerBlock_) * elementPerBlock_;
    uint64_t outAAlign = Ops::Base::CeilDiv(allASize, elementPerBlock_) * elementPerBlock_;

    realCoreNum_ = originCoreNum_; // GROUP_REDUCE 全核 SyncAll，使用物理核数
    if (realCoreNum_ > rSize) {
        realCoreNum_ = rSize;
    }
    blkFactor_ = rSize / realCoreNum_;
    blkTailFactor_ = rSize % realCoreNum_;
    uint64_t maxBlkPerCore = blkTailFactor_ == static_cast<uint64_t>(0) ? blkFactor_ :
                                                                          blkFactor_ + static_cast<uint64_t>(1);

    bool isRaSplit = (nextASize * eleLenInBytes_ >= ALIGN_128) || (blkFactor_ * eleLenInBytes_ < VL_SIZE);
    isRaSplit_ = isRaSplit ? 1 : 0;
    bool fallbackToRa = false;

    if (!isRaSplit) {
        if (CalcGroupReduceArBranch(aSize, nextASize, outAAlign, maxBlkPerCore, fallbackToRa) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    if (isRaSplit || fallbackToRa) {
        if (CalcGroupReduceRaBranch(aSize, nextASize, nextAAlign, maxBlkPerCore) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    cutASize_ = static_cast<uint16_t>(aSize);
    cutNextASize_ = static_cast<uint16_t>(nextASize);

    return ge::GRAPH_SUCCESS;
}

void ArgCommonBaseTilingArch35::AddExtraBufferNeed(uint64_t& fixedNeed, int& isBfloatNum, uint64_t alignSize)
{
    if (ge::DT_BF16 == valueDtype_) {
        fixedNeed += alignSize * DOUBLE_BUFFER_NUM * sizeof(float);
        isBfloatNum = BF16_BUFFER_MULTIPLIER;
    }
    if ((indiceDtypeSize_ != t2Size_) || (indiceDtype_ != ge::DT_INT32 && indiceDtype_ != ge::DT_INT64)) {
        fixedNeed += alignSize * indiceDtypeSize_;
    }
}

ge::graphStatus ArgCommonBaseTilingArch35::CalcGroupReduceArBranch(uint64_t aSize, uint64_t nextASize,
    uint64_t outAAlign, uint64_t maxBlkPerCore, bool& fallbackToRa)
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CalcGroupReduceArBranch.");
    uint64_t allASize = aSize * nextASize;
    uint64_t vlCnt = vRegSize_ / eleLenInBytes_;

    uint64_t workSpaceSize = outAAlign * VALUE_TWO * sizeof(int32_t) * originCoreNum_;
    if (ge::DT_INT64 == valueDtype_) {
        workSpaceSize = workSpaceSize * VALUE_TWO;
    }
    workSpaceSize_ = Ops::Base::CeilDiv(workSpaceSize, VL_SIZE) * VL_SIZE;

    uint64_t fixedNeed = outAAlign * DOUBLE_BUFFER_NUM * (eleLenInBytes_ + t2Size_) +
                         realCoreNum_ * outAAlign * t2Size_ + VL_SIZE;
    uint64_t inQueueXNeedForOne = std::max(VL_SIZE * allASize / eleLenInBytes_, realCoreNum_ * outAAlign);
    uint64_t inQueueXNeed = inQueueXNeedForOne * DOUBLE_BUFFER_NUM * eleLenInBytes_;
    int isBfloatNum = 1;
    AddExtraBufferNeed(fixedNeed, isBfloatNum, outAAlign);

    if (ubSize_ >= fixedNeed + inQueueXNeed * isBfloatNum) {
        loopANum_ = 1;
        cutAPerLoop_ = static_cast<uint16_t>(aSize);

        uint64_t ubAvail = (ubSize_ - fixedNeed) / isBfloatNum / DOUBLE_BUFFER_NUM;
        uint64_t cutRSize = ubAvail / (aSize * nextASize * eleLenInBytes_);
        cutRSize = cutRSize <= vlCnt ? cutRSize : (cutRSize / vlCnt * vlCnt);
        cutRSize = std::min(cutRSize, MAX_VALUE_UINT16);
        cutRSize = std::min(cutRSize, maxBlkPerCore);
        cutRSize_ = static_cast<uint16_t>(cutRSize);
        gatherBlockSize_ = static_cast<uint16_t>(nextASize);
    } else {
        fallbackToRa = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ArgCommonBaseTilingArch35::CalcGroupReduceRaBranch(uint64_t aSize, uint64_t nextASize,
    uint64_t nextAAlign, uint64_t maxBlkPerCore)
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CalcGroupReduceRaBranch.");
    isRaSplit_ = 1;

    uint64_t gatherBlock = ALIGN_128 / eleLenInBytes_;
    if (gatherBlock > nextASize) {
        gatherBlock = nextASize;
    }
    uint64_t gatherBlockAlign = Ops::Base::CeilDiv(gatherBlock, elementPerBlock_) * elementPerBlock_;
    gatherBlockSize_ = static_cast<uint16_t>(gatherBlockAlign);

    uint64_t fixedNeed = nextAAlign * DOUBLE_BUFFER_NUM * (eleLenInBytes_ + t2Size_) +
                         realCoreNum_ * gatherBlockAlign * t2Size_ + VL_SIZE;
    int isBfloatNum = 1;
    AddExtraBufferNeed(fixedNeed, isBfloatNum, nextAAlign);

    loopANum_ = static_cast<uint16_t>(aSize);
    cutAPerLoop_ = 1;
    uint64_t ubAvail = (ubSize_ - fixedNeed) / isBfloatNum / DOUBLE_BUFFER_NUM;
    ubAvail = ubAvail / BLOCK_SIZE * BLOCK_SIZE;
    uint64_t cutRSize = ubAvail / (nextASize * eleLenInBytes_);
    cutRSize = std::min(cutRSize, MAX_VALUE_UINT16);
    cutRSize = std::min(cutRSize, maxBlkPerCore);
    cutRSize_ = static_cast<uint16_t>(cutRSize);

    uint64_t perCoreValueBytes = Ops::Base::CeilDiv(nextASize * eleLenInBytes_, ALIGN_128) * ALIGN_128;
    uint64_t perCoreIndiceBytes = Ops::Base::CeilDiv(nextASize * t2Size_, ALIGN_128) * ALIGN_128;
    uint64_t workSpaceSize = aSize * realCoreNum_ * (perCoreValueBytes + perCoreIndiceBytes);
    workSpaceSize_ = Ops::Base::CeilDiv(workSpaceSize, VL_SIZE) * VL_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ArgCommonBaseTilingArch35::TryRouteAraMode4()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering ArgCommonBaseTilingArch35 TryRouteAraMode4.");
    uint64_t cutNextA = CalcCutNextA(static_cast<uint64_t>(tilingData_.get_nextASize()));
    if (cutNextA != tilingData_.get_nextASize() && tilingData_.get_nextASize() * eleLenInBytes_ >= BLOCK_SIZE &&
        tilingData_.get_rSize() >= coreNum_ * CUTR_MIN_SIZE) {
        if (cutNextA * eleLenInBytes_ >= BLOCK_SIZE * VALUE_TWO) {
            tilingKey_ = static_cast<uint64_t>(ArgMaxWithValueTilingMode::ARA_CUT_A_AND_NEXT_A);
            if (CalcSplitInfoForArACutAAndNextA() != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        } else {
            // 改道 GROUP_REDUCE：全核 SyncAll，门槛用 originCoreNum_ 与后续分核口径一致
            if (tilingData_.get_rSize() / originCoreNum_ * tilingData_.get_nextASize() * eleLenInBytes_ >=
                CUTR_MIN_MOV) {
                tilingKey_ = static_cast<uint64_t>(ArgMaxWithValueTilingMode::GROUP_REDUCE);
                if (CalcSplitInfoForGroupReduce() != ge::GRAPH_SUCCESS) {
                    return ge::GRAPH_FAILED;
                }
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

void ArgCommonBaseTilingArch35::FillTilingData()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering ArgCommonBaseTilingArch35 FillTilingData.");
    tilingData_.set_cutASize(cutASize_);
    tilingData_.set_cutRSize(cutRSize_);
    tilingData_.set_cutNextASize(cutNextASize_);
    tilingData_.set_realCoreNum(realCoreNum_);
    tilingData_.set_blkFactor(blkFactor_);
    tilingData_.set_blkTailFactor(blkTailFactor_);
    tilingData_.set_tilingKey(tilingKey_);
    tilingData_.set_aRaMode(aRaMode_);
    tilingData_.set_workSpaceSize(workSpaceSize_);
    tilingData_.set_blkFactor2nd(blkFactor2nd_);
    tilingData_.set_blkTailFactor2nd(blkTailFactor2nd_);
    tilingData_.set_blkNum2nd(blkNum2nd_);
    tilingData_.set_loopANum(loopANum_);
    tilingData_.set_cutAPerLoop(cutAPerLoop_);
    tilingData_.set_isRaSplit(isRaSplit_);
    tilingData_.set_gatherBlockSize(gatherBlockSize_);
    tilingData_.set_useFlatPath(useFlatPath_);

    tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                             tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

void ArgCommonBaseTilingArch35::PrintTilingData()
{
    OP_LOGI(tilingContext_->GetNodeName(),
            "tilingData is aSize = %lu, cutASize = %hu, rSize = %lu, cutRSize = %hu, nextASize = %lu, \
            cutNextASize = %hu, realCoreNum = %lu, blkFactor = %lu, blkTailFactor = %lu, blkFactor2nd = %lu,\
            blkTailFactor2nd = %lu, blkNum2nd=%lu,tilingKey = %lu, \
            aRaMode = %lu, workSpaceSize = %lu, loopANum = %hu, cutAPerLoop = %hu, isRaSplit = %u, \
            gatherBlockSize = %hu, Tiling4ArgMaxWithValue ends.",
            tilingData_.get_aSize(), tilingData_.get_cutASize(), tilingData_.get_rSize(), tilingData_.get_cutRSize(),
            tilingData_.get_nextASize(), tilingData_.get_cutNextASize(), tilingData_.get_realCoreNum(),
            tilingData_.get_blkFactor(), tilingData_.get_blkTailFactor(), tilingData_.get_blkFactor2nd(),
            tilingData_.get_blkTailFactor2nd(), tilingData_.get_blkNum2nd(), tilingData_.get_tilingKey(),
            tilingData_.get_aRaMode(), tilingData_.get_workSpaceSize(), tilingData_.get_loopANum(),
            tilingData_.get_cutAPerLoop(), tilingData_.get_isRaSplit(), tilingData_.get_gatherBlockSize());
}

ge::graphStatus ArgOpsTilingForAscendC(gert::TilingContext* context, const uint64_t& coreNum, const uint64_t& ubSize,
                                       bool withValue, const uint64_t& vRegSize)
{
    ArgCommonBaseTilingArch35 tilingObject(context);
    if (tilingObject.Init(coreNum, ubSize, vRegSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunArgMaxTiling(withValue);
}
} // namespace optiling
