/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_MRGSORT_OUT_H
#define SORT_MRGSORT_OUT_H

#include "sort_mrgsort.h"

namespace KvSort {
using namespace AscendC;

class KvMrgSortOut : public KvMrgSort
{
public:
    __aicore__ inline KvMrgSortOut() {};
    __aicore__ inline void Init(MrgSortParam* param, TPipe* tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void SetOutput(
        GlobalTensor<int32_t>& gmKeyOut, GlobalTensor<int32_t>& gmValOut,
        LocalTensor<float>& ubOutput1, LocalTensor<float>& ubOutput2);
    __aicore__ inline void SetBuffer(LocalTensor<float>& tempBuffer);

private:
    __aicore__ inline void MrgSortCompute();
    __aicore__ inline void ExtractAndRestore();
    __aicore__ inline void CopyOut();

private:
    GlobalTensor<int32_t> gmKeyOut;
    GlobalTensor<int32_t> gmValOut;

    LocalTensor<float> tempBuf;
    LocalTensor<float> ubKeyOut;
    LocalTensor<uint32_t> ubValOut;
    LocalTensor<int32_t> ubKeyOutInt;
    LocalTensor<int32_t> ubValOutInt;
};

__aicore__ inline void KvMrgSortOut::SetOutput(
    GlobalTensor<int32_t>& gmKeyOut, GlobalTensor<int32_t>& gmValOut,
    LocalTensor<float>& ubOutput1, LocalTensor<float>& ubOutput2)
{
    this->gmKeyOut = gmKeyOut;
    this->ubKeyOut = ubOutput1;
    this->ubKeyOutInt = ubOutput1.ReinterpretCast<int32_t>();

    this->gmValOut = gmValOut;
    this->ubValOut = ubOutput2.ReinterpretCast<uint32_t>();
    this->ubValOutInt = ubOutput2.ReinterpretCast<int32_t>();
}

__aicore__ inline void KvMrgSortOut::SetBuffer(LocalTensor<float>& tempBuffer)
{
    this->tempBuf = tempBuffer;
}

__aicore__ inline void KvMrgSortOut::MrgSortCompute()
{
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    if (this->remainListNum == MERGE_LIST_TWO) {
        MrgSortSrcList sortList = MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[0], tmpUbInputs[0]);
        MrgSort<float, true>(this->tempBuf, sortList, elementCountListTail, listSortedNums, validBitTail, 1);
    } else if (this->remainListNum == MERGE_LIST_THREE) {
        MrgSortSrcList sortList = MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[MERGE_LIST_IDX_TWO], tmpUbInputs[0]);
        MrgSort<float, true>(this->tempBuf, sortList, elementCountListTail, listSortedNums, validBitTail, 1);
    } else if (this->remainListNum == MERGE_LIST_FOUR) {
        MrgSortSrcList sortList = MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[MERGE_LIST_IDX_TWO], tmpUbInputs[MERGE_LIST_IDX_THREE]);
        MrgSort<float, true>(this->tempBuf, sortList, elementCountListTail, listSortedNums, validBitTail, 1);
    } else {
        DataCopy(this->tempBuf, this->tmpUbInputs[0],
                 AlignUp(GetSortLen<float>(elementCountListTail[0]), sizeof(float)));
        listSortedNums[0] = elementCountListTail[0];
    }
}

__aicore__ inline void KvMrgSortOut::ExtractAndRestore()
{
    AscendC::Extract(this->ubKeyOut, this->ubValOut, this->tempBuf,
                     CeilDiv(curLoopSortedNum, ONE_REPEAT_SORT_NUM));
    Muls(this->ubKeyOut, this->ubKeyOut, (float)-1, AlignUp(curLoopSortedNum, sizeof(float)));
    Cast(this->ubKeyOutInt, this->ubKeyOut, RoundMode::CAST_ROUND, AlignUp(curLoopSortedNum, sizeof(float)));
}

__aicore__ inline void KvMrgSortOut::CopyOut()
{
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = curLoopSortedNum * sizeof(int32_t);
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyPad(this->gmKeyOut[outOffset], this->ubKeyOutInt, intriParams);
    DataCopyPad(this->gmValOut[outOffset], this->ubValOutInt, intriParams);
    outOffset += curLoopSortedNum;
}

__aicore__ inline void KvMrgSortOut::Init(MrgSortParam* param, TPipe* tPipe)
{
    // Delegate to base-class Init which handles the common offsets/listRemainElements/allRemainElements loop.
    KvMrgSort::Init(param);
    // Reset remainListNum — the base-class Init sets remainListNum = listNum, but
    // for KvMrgSortOut, remainListNum is computed from scratch in CopyIn().
    this->remainListNum = 0;
}

__aicore__ inline void KvMrgSortOut::Process()
{
    for (; allRemainElements > 0;) {
        CopyIn();
        UpdateMrgParam();
        MrgSortCompute();
        UpdateSortInfo();
        ExtractAndRestore();
        CopyOut();
    }
    ClearCache();
}

}  // namespace KvSort
#endif
