/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_ONE_CORE_H
#define SORT_ONE_CORE_H

#include "sort_common.h"
#include "sort_tiling.h"

namespace KvSort {
using namespace AscendC;

class SortOneCore
{
public:
    __aicore__ inline SortOneCore() {};
    __aicore__ inline void Init(
        GM_ADDR keys, GM_ADDR values, GM_ADDR sortedKeys, GM_ADDR sortedValues,
        const SortKernelTiling* tiling, TPipe* tPipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyIn();
    __aicore__ inline void SortCompute();
    __aicore__ inline void CopyOut();

protected:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> copyInQueue;
    TQue<QuePosition::VECOUT, 1> copyOutQueue;
    TBuf<TPosition::VECCALC> tempBuffer;
    TBuf<TPosition::VECCALC> sortedBuffer;

    GlobalTensor<int32_t> keysGm;
    GlobalTensor<int32_t> valuesGm;
    GlobalTensor<int32_t> sortedKeysGm;
    GlobalTensor<int32_t> sortedValuesGm;

    int64_t totalLength;
    int64_t sortNum;
    int64_t tileLength;
    int64_t coreNum;
};

__aicore__ inline void SortOneCore::Init(
    GM_ADDR keys, GM_ADDR values, GM_ADDR sortedKeys, GM_ADDR sortedValues,
    const SortKernelTiling* tiling, TPipe* tPipe)
{
    this->pipe = tPipe;
    this->totalLength = tiling->totalLength;
    this->tileLength = AlignUp(this->totalLength, sizeof(int32_t));
    this->sortNum = CeilDiv(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    this->coreNum = tiling->coreNum;

    keysGm.SetGlobalBuffer((__gm__ int32_t*)keys, this->tileLength);
    valuesGm.SetGlobalBuffer((__gm__ int32_t*)values, this->tileLength);
    sortedKeysGm.SetGlobalBuffer((__gm__ int32_t*)sortedKeys, this->tileLength);
    sortedValuesGm.SetGlobalBuffer((__gm__ int32_t*)sortedValues, this->tileLength);

    int64_t kvFactor = 2;
    int64_t buffSize = this->sortNum * sizeof(int32_t) * kvFactor;
    pipe->InitBuffer(copyInQueue, 1, buffSize);
    pipe->InitBuffer(copyOutQueue, 1, buffSize);
    pipe->InitBuffer(tempBuffer, buffSize);
    pipe->InitBuffer(sortedBuffer, buffSize);
}

__aicore__ inline void SortOneCore::CopyIn()
{
    LocalTensor<int32_t> inLocal = copyInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(this->totalLength * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(inLocal[0], keysGm, dataCopyParams, dataCopyPadParams);
    DataCopyPad(inLocal[this->sortNum], valuesGm, dataCopyParams, dataCopyPadParams);
    copyInQueue.EnQue(inLocal);
}

__aicore__ inline void SortOneCore::SortCompute()
{
    LocalTensor<int32_t> inLocal = copyInQueue.DeQue<int32_t>();
    LocalTensor<int32_t> keysLocal = inLocal[0];
    LocalTensor<float> keysFp32 = keysLocal.ReinterpretCast<float>();
    Cast(keysFp32, keysLocal, RoundMode::CAST_ROUND, this->tileLength);
    Muls(keysFp32, keysFp32, (float)-1, this->tileLength);

    int64_t duplicateNum = this->totalLength % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = this->totalLength - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(keysFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }

    LocalTensor<float> concatLocal;
    LocalTensor<float> tempTensor = tempBuffer.Get<float>(GetSortLen<float>(this->sortNum));
    Concat(concatLocal, keysFp32, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);

    LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortNum));
    LocalTensor<uint32_t> valuesLocal = inLocal[this->sortNum].ReinterpretCast<uint32_t>();
    Sort<float, true>(sortedLocal, concatLocal, valuesLocal, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);

    LocalTensor<float> outLocal = copyOutQueue.AllocTensor<float>();
    LocalTensor<float> sortedKeysFp32 = outLocal[0];
    LocalTensor<uint32_t> sortedValuesU32 = outLocal[this->sortNum].ReinterpretCast<uint32_t>();
    Extract(sortedKeysFp32, sortedValuesU32, sortedLocal, this->sortNum / ONE_REPEAT_SORT_NUM);
    Muls(sortedKeysFp32, sortedKeysFp32, (float)-1, this->tileLength);

    LocalTensor<int32_t> sortedKeysInt32 = sortedKeysFp32.ReinterpretCast<int32_t>();
    Cast(sortedKeysInt32, sortedKeysFp32, RoundMode::CAST_ROUND, this->tileLength);

    copyOutQueue.EnQue<float>(outLocal);
    copyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void SortOneCore::CopyOut()
{
    LocalTensor<int32_t> outLocal = copyOutQueue.DeQue<int32_t>();
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = this->totalLength * sizeof(int32_t);
    DataCopyPad(sortedKeysGm, outLocal[0], intriParams);
    DataCopyPad(sortedValuesGm, outLocal[this->sortNum], intriParams);
    copyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void SortOneCore::Process()
{
    if (GetBlockIdx() == 0) {
        CopyIn();
        SortCompute();
        CopyOut();
    }
    if (coreNum > 1) {
        AscendC::SyncAll();
    }
}

}  // namespace KvSort
#endif
