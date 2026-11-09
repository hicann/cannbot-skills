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
 * \file arg_common_base_tiling_arch35.h
 * \brief arg_common_base_tiling_arch35
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_ARG_COMMON_BASE_TILING_ARCH35_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_ARG_COMMON_BASE_TILING_ARCH35_H

#include <cstdint>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
ge::graphStatus ArgOpsTilingForAscendC(gert::TilingContext* context, const uint64_t& coreNum, const uint64_t& ubSize,
                                       bool withValue, const uint64_t& vRegSize);

BEGIN_TILING_DATA_DEF(ArgMaxWithValueTilingData)
TILING_DATA_FIELD_DEF(uint64_t, aSize);
TILING_DATA_FIELD_DEF(uint16_t, cutASize);
TILING_DATA_FIELD_DEF(uint64_t, rSize);
TILING_DATA_FIELD_DEF(uint16_t, cutRSize);
TILING_DATA_FIELD_DEF(uint64_t, nextASize);
TILING_DATA_FIELD_DEF(uint16_t, cutNextASize);
TILING_DATA_FIELD_DEF(uint64_t, realCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, blkFactor);
TILING_DATA_FIELD_DEF(uint64_t, blkTailFactor);
TILING_DATA_FIELD_DEF(uint64_t, blkFactor2nd);
TILING_DATA_FIELD_DEF(uint64_t, blkTailFactor2nd);
TILING_DATA_FIELD_DEF(uint64_t, blkNum2nd);
TILING_DATA_FIELD_DEF(uint64_t, tilingKey);
TILING_DATA_FIELD_DEF(uint64_t, aRaMode);
TILING_DATA_FIELD_DEF(uint16_t, loopANum);
TILING_DATA_FIELD_DEF(uint16_t, cutAPerLoop);
TILING_DATA_FIELD_DEF(uint8_t, isRaSplit);
TILING_DATA_FIELD_DEF(uint16_t, gatherBlockSize);
TILING_DATA_FIELD_DEF(uint8_t, useFlatPath);
TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize);
END_TILING_DATA_DEF;

class ArgCommonBaseTilingArch35 {
public:
    explicit ArgCommonBaseTilingArch35(gert::TilingContext* context) : tilingContext_(context) {};
    ge::graphStatus Init(const uint64_t& coreNum, const uint64_t& ubSize, const uint64_t& vRegSize);
    ge::graphStatus RunArgMaxTiling(bool withValue);

private:
    ge::graphStatus GetShapeInfo(bool withValue);
    ge::graphStatus GetDimension(bool withValue);
    ge::graphStatus SetShapeInfo();
    ge::graphStatus CalcSplitInfo();
    ge::graphStatus CalcSplitInfoForAr();
    ge::graphStatus CalcSplitInfoForArA();
    ge::graphStatus CalcSplitInfoForArAAndTryMode4();
    ge::graphStatus CalcSplitInfoForArACutAAndNextA();
    ge::graphStatus CalcSplitInfoForArACutAAndNextAFlat(uint64_t blkNumA, uint64_t nextAAlign);
    ge::graphStatus CalcSplitInfoForArACutAAndNextANormal(uint64_t blkNumA, uint64_t minDtypeSize);
    ge::graphStatus CalcSplitInfoForGatherRa();
    ge::graphStatus CalcSplitInfoForGatherArA();
    ge::graphStatus CalcSplitInfoForRa();
    ge::graphStatus CalcSplitInfoForCopyOnly();
    ge::graphStatus CalcSplitInfoForGroupReduce();
    ge::graphStatus CalcGroupReduceArBranch(uint64_t aSize, uint64_t nextASize, uint64_t outAAlign,
                                            uint64_t maxBlkPerCore, bool& fallbackToRa);
    ge::graphStatus CalcGroupReduceRaBranch(uint64_t aSize, uint64_t nextASize, uint64_t nextAAlign,
                                            uint64_t maxBlkPerCore);
    void AddExtraBufferNeed(uint64_t& fixedNeed, int& isBfloatNum, uint64_t alignSize);
    ge::graphStatus TryRouteAraMode4();
    void SetShapeInfoHighPerf();
    void FillTilingData();
    void PrintTilingData();
    void CalcCutA();
    void CalcCutRA();
    uint64_t CalcCutNextA(uint64_t cutNextASize);
    uint64_t CalcUnitUbSpace();

private:
    ArgMaxWithValueTilingData tilingData_;
    gert::TilingContext* tilingContext_ = nullptr;
    int64_t xDimNum_ = 1;
    int64_t dimension_ = 0;
    uint64_t coreNum_ = 0;
    uint64_t originCoreNum_ = 0; // 物理核数(不随每4KB一核的动态核数变化)，GROUP_REDUCE 全链路使用
    uint64_t ubSize_ = 0;
    uint64_t realCoreNum_ = 1;
    uint64_t tilingKey_ = 0;
    uint64_t blkFactor_ = 0;
    uint64_t blkTailFactor_ = 0;
    uint64_t blkFactor2nd_ = 0;
    uint64_t blkTailFactor2nd_ = 0;
    uint64_t blkNum2nd_ = 1;
    uint64_t eleLenInBytes_ = 1;
    uint64_t eleLenIndiceBytes_ = 1;
    uint64_t indexDtypeSize_ = 1;
    uint64_t processSize_ = 1024;
    uint64_t ubElement_ = 1;
    uint16_t cutASize_ = 1;
    uint16_t cutRSize_ = 1;
    uint16_t cutNextASize_ = 1;
    uint16_t aRaMode_ = 101;
    uint64_t vRegSize_ = 1;
    uint64_t elementPerBlock_ = 1;
    ge::DataType valueDtype_ = ge::DataType::DT_MAX;
    ge::DataType indiceDtype_ = ge::DataType::DT_MAX;
    bool isEmptyTensor_ = false;
    uint64_t workSpaceSize_ = 0;
    uint64_t valueDtypeSize_ = 0;
    uint64_t t2Size_ = 0;
    uint64_t indiceDtypeSize_ = 0;
    uint16_t loopANum_ = 1;
    uint16_t cutAPerLoop_ = 1;
    uint8_t isRaSplit_ = 0;
    uint16_t gatherBlockSize_ = 1;
    uint8_t useFlatPath_ = 0;
};

struct ArgMaxWithValueCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

enum class ArgMaxWithValueTilingMode : uint64_t {
    AR_CUT_A = 10001,             // AR
    ARA_CUT_A = 10002,            // ARA
    ARA_CUT_A_AND_NEXT_A = 10012, // ARA 双轴切核
    COPY_ONLY = 10003,            // R = 1
    AR_GATHER = 20001,            // AR_GATHER
    ARA_GATHER = 20002,           // ARA_GATHER
    RA_CUT_A = 20003,             // RA
    GROUP_REDUCE = 30001          // Group Reduce
};

enum class ARAMode : uint64_t {
    ARA_MODE1 = 101,
    ARA_MODE2 = 102,
    ARA_MODE3 = 103,
    ARA_MODE4 = 104,
    ARA_MODE5 = 105,
    ARA_MODE6 = 106
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_ARG_COMMON_BASE_TILING_ARCH35_H
