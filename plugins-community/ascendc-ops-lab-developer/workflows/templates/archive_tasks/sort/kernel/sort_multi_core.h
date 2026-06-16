/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_MULTI_CORE_H
#define SORT_MULTI_CORE_H

#include "sort_common.h"
#include "sort_tiling.h"
#include "sort_mrgsort.h"
#include "sort_mrgsort_out.h"

namespace KvSort {
using namespace AscendC;

class SortMultiCore
{
public:
    __aicore__ inline SortMultiCore() {};
    __aicore__ inline void Init(
        GM_ADDR keys, GM_ADDR values, GM_ADDR sortedKeys, GM_ADDR sortedValues,
        GM_ADDR workspace, const SortKernelTiling* tiling, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void VBSProcess();
    __aicore__ inline void UBSortProcess(int64_t progress, int64_t size, int64_t sn);
    __aicore__ inline void VBSCopyIn(int64_t progress, int64_t size, int64_t sn);
    __aicore__ inline void UBSortCompute(int64_t progress, int64_t size, int64_t sn);
    __aicore__ inline void VBSCopyOut(int64_t progress, int64_t size, int64_t sn);
    __aicore__ inline void OneCoreVMSProcess(int64_t listNum, int64_t perListElem, int64_t lastListElem);
    __aicore__ inline void VMSProcess();
    __aicore__ inline void SortOutProcess();
    __aicore__ inline void InitMrgSort(KvMrgSort* sorter, int64_t ln, int64_t coreOff, int64_t loopOff);
    __aicore__ inline void InitMrgSortOut(KvMrgSortOut* sorter, int64_t ln, int64_t coreOff);

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> copyInQueue;
    TQue<QuePosition::VECOUT, 1> copyOutQueue;
    TBuf<TPosition::VECCALC> tempBuffer;
    TBuf<TPosition::VECCALC> sortedBuf;

    GlobalTensor<int32_t> keysGm;
    GlobalTensor<int32_t> valuesGm;
    GlobalTensor<int32_t> sortedKeysGm;
    GlobalTensor<int32_t> sortedValuesGm;
    GlobalTensor<float> workspaceGms[2];

    KvMrgSort mrgsorter;
    MrgSortParam mrgsortParam;

    int64_t totalLength;
    int64_t blockIdx;
    int64_t coreNum;
    int64_t needCoreNum;
    int64_t srcWsIndex = 0;

    int64_t listNum;
    int64_t perListElements;
    int64_t lastListElements;

    int64_t sortTotalLength;
    int64_t sortCoreLoops;
    int64_t sortCoreLoopElements;
    int64_t sortCoreLastLoopElements;

    int64_t perCoreElements;
    int64_t lastCoreElements;
    int64_t oneLoopMaxElements;
};

__aicore__ inline void SortMultiCore::VBSCopyIn(int64_t progress, int64_t size, int64_t sn)
{
    LocalTensor<int32_t> inLocal = copyInQueue.AllocTensor<int32_t>();
    int64_t inOffset = progress * sortCoreLoopElements;
    DataCopyExtParams dataCopyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(size * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(inLocal[0], keysGm[inOffset], dataCopyParams, dataCopyPadParams);
    DataCopyPad(inLocal[sn], valuesGm[inOffset], dataCopyParams, dataCopyPadParams);
    copyInQueue.EnQue(inLocal);
}

__aicore__ inline void SortMultiCore::UBSortCompute(int64_t progress, int64_t size, int64_t sn)
{
    LocalTensor<int32_t> inLocal = copyInQueue.DeQue<int32_t>();
    LocalTensor<int32_t> keysLocal = inLocal[0];
    LocalTensor<float> keysFp32 = keysLocal.ReinterpretCast<float>();
    Cast(keysFp32, keysLocal, RoundMode::CAST_ROUND, sn);
    Muls(keysFp32, keysFp32, (float)-1, sn);

    int64_t duplicateNum = size % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = size - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(keysFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }

    LocalTensor<float> concatLocal;
    LocalTensor<float> tempTensor = tempBuffer.Get<float>(GetSortLen<float>(sn));
    Concat(concatLocal, keysFp32, tempTensor, sn / ONE_REPEAT_SORT_NUM);

    LocalTensor<float> sortedLocal = sortedBuf.Get<float>(GetSortLen<float>(sn));
    LocalTensor<float> outLocal = copyOutQueue.AllocTensor<float>();
    LocalTensor<uint32_t> valuesLocal = inLocal[sn].ReinterpretCast<uint32_t>();
    Sort<float, true>(outLocal, concatLocal, valuesLocal, sortedLocal, sn / ONE_REPEAT_SORT_NUM);

    copyOutQueue.EnQue<float>(outLocal);
    copyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void SortMultiCore::VBSCopyOut(int64_t progress, int64_t size, int64_t sn)
{
    LocalTensor<float> outLocal = copyOutQueue.DeQue<float>();
    DataCopy(
        workspaceGms[0][this->blockIdx * GetSortLen<float>(this->perCoreElements) +
                        GetSortLen<float>(progress * sortCoreLoopElements)],
        outLocal, AlignUp(GetSortLen<float>(size), sizeof(float)));
    copyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void SortMultiCore::UBSortProcess(int64_t progress, int64_t size, int64_t sn)
{
    VBSCopyIn(progress, size, sn);
    UBSortCompute(progress, size, sn);
    VBSCopyOut(progress, size, sn);
}

__aicore__ inline void SortMultiCore::InitMrgSort(
    KvMrgSort* sorter, int64_t ln, int64_t coreOff, int64_t loopOff)
{
    GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex][blockIdx * coreOff + loopOff];
    LocalTensor<float> inLocal = copyInQueue.AllocTensor<float>();
    LocalTensor<float> outLocal = copyOutQueue.AllocTensor<float>();
    for (int64_t i = 0; i < ln; i++) {
        LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(this->oneLoopMaxElements) * i];
        sorter->SetInput(srcWsGm, inLocalT);
    }
    GlobalTensor<float> dstWsGm = workspaceGms[1 - srcWsIndex][blockIdx * coreOff + loopOff];
    sorter->SetOutput(dstWsGm, outLocal);
    copyInQueue.FreeTensor(inLocal);
    copyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void SortMultiCore::InitMrgSortOut(
    KvMrgSortOut* sorter, int64_t ln, int64_t coreOff)
{
    GlobalTensor<float> srcWsGm = workspaceGms[srcWsIndex];
    LocalTensor<float> inLocal = copyInQueue.AllocTensor<float>();
    LocalTensor<float> outLocal = copyOutQueue.AllocTensor<float>();

    for (int64_t i = 0; i < ln; i++) {
        LocalTensor<float> inLocalT = inLocal[GetSortLen<float>(this->oneLoopMaxElements) * i];
        sorter->SetInput(srcWsGm, inLocalT);
    }

    LocalTensor<float> outLocalV = outLocal[this->oneLoopMaxElements * MAX_MRGSORT_LIST];
    sorter->SetOutput(this->sortedKeysGm, this->sortedValuesGm, outLocal, outLocalV);

    LocalTensor<float> tb = sortedBuf.Get<float>(
        GetSortLen<float>(this->oneLoopMaxElements) * MAX_MRGSORT_LIST);
    sorter->SetBuffer(tb);
    copyInQueue.FreeTensor(inLocal);
    copyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void SortMultiCore::OneCoreVMSProcess(
    int64_t ln, int64_t perListElem, int64_t lastListElem)
{
    int64_t coreOffset = GetSortLen<float>(this->perCoreElements);
    mrgsortParam.oneLoopMaxElements = this->oneLoopMaxElements;

    for (int64_t i = 0; ln >= 1; i++) {
        int64_t loops = (ln + MAX_MRGSORT_LIST - 1) / MAX_MRGSORT_LIST;
        int64_t remainLn = ln - (loops - 1) * MAX_MRGSORT_LIST;

        mrgsortParam.perListElements = perListElem;
        mrgsortParam.lastListElements = perListElem;

        int64_t loopOffset = GetSortLen<float>(mrgsortParam.perListElements * MAX_MRGSORT_LIST);
        for (int64_t loop = 0; loop < loops - 1; loop++) {
            InitMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, loop * loopOffset);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        }

        mrgsortParam.perListElements = perListElem;
        mrgsortParam.lastListElements = lastListElem;
        InitMrgSort(&mrgsorter, remainLn, coreOffset, (loops - 1) * loopOffset);
        mrgsorter.Init(&mrgsortParam);
        mrgsorter.Process();

        ln = loops;
        lastListElem = perListElem * (remainLn - 1) + lastListElem;
        perListElem = perListElem * MAX_MRGSORT_LIST;
        srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;

        if (loops == 1) break;
    }
}

__aicore__ inline void SortMultiCore::VBSProcess()
{
    if (this->blockIdx < this->needCoreNum) {
        int64_t sn = CeilDiv(sortCoreLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        for (int64_t loop = 0; loop < sortCoreLoops - 1; loop++) {
            UBSortProcess(loop, sortCoreLoopElements, sn);
        }
        sn = CeilDiv(sortCoreLastLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        UBSortProcess(sortCoreLoops - 1, sortCoreLastLoopElements, sn);

        OneCoreVMSProcess(sortCoreLoops, sortCoreLoopElements, sortCoreLastLoopElements);
    }

    // After OneCoreVMSProcess, if the last core had a different number of loops
    // it may have ended with srcWsIndex pointing to a different buffer.
    // Fix: compute the expected srcWsIndex from core 0's perspective,
    // and if mismatched, copy the data to the correct buffer.
    int64_t perCoreLoopsCalc = CeilDiv(this->perCoreElements, sortCoreLoopElements);
    int64_t mainMergeRounds = 0;
    {
        int64_t ln = perCoreLoopsCalc;
        while (ln > 1) {
            ln = CeilDiv(ln, MAX_MRGSORT_LIST);
            mainMergeRounds++;
        }
    }
    int64_t expectedWsIndex = mainMergeRounds % WORK_GM_NUM;

    if (this->blockIdx < this->needCoreNum && srcWsIndex != expectedWsIndex) {
        int64_t coreOff = this->blockIdx * GetSortLen<float>(this->perCoreElements);
        int64_t dataLen = AlignUp(GetSortLen<float>(this->sortTotalLength), sizeof(float));

        LocalTensor<float> tmpBuf = copyInQueue.AllocTensor<float>();
        int64_t bufFloats = CeilDiv(
            MaxVal(this->oneLoopMaxElements * MAX_MRGSORT_LIST, sortCoreLoopElements),
            ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM * 2;

        for (int64_t off = 0; off < dataLen; off += bufFloats) {
            int64_t chunk = MinVal(bufFloats, dataLen - off);
            int64_t alignChunk = AlignUp(chunk, sizeof(float));

            event_t e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(e1);
            WaitFlag<HardEvent::MTE3_MTE2>(e1);
            DataCopy(tmpBuf, workspaceGms[srcWsIndex][coreOff + off], alignChunk);

            event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
            SetFlag<HardEvent::MTE2_MTE3>(e2);
            WaitFlag<HardEvent::MTE2_MTE3>(e2);
            DataCopy(workspaceGms[expectedWsIndex][coreOff + off], tmpBuf, alignChunk);
        }
        copyInQueue.FreeTensor(tmpBuf);
    }
    srcWsIndex = expectedWsIndex;

    AscendC::SyncAll();
}

__aicore__ inline void SortMultiCore::VMSProcess()
{
    int64_t currentStageNeedCoreNum = this->needCoreNum;
    perListElements = this->perCoreElements;
    lastListElements = this->lastCoreElements;
    listNum = this->needCoreNum;

    for (; listNum > MAX_MRGSORT_LIST;) {
        currentStageNeedCoreNum = CeilDiv(listNum, MAX_MRGSORT_LIST);
        int64_t coreOffset = GetSortLen<float>(perListElements * MAX_MRGSORT_LIST);
        int64_t remainLn = listNum - (currentStageNeedCoreNum - 1) * MAX_MRGSORT_LIST;

        if (this->blockIdx < currentStageNeedCoreNum - 1) {
            mrgsortParam.perListElements = perListElements;
            mrgsortParam.lastListElements = perListElements;
            mrgsortParam.oneLoopMaxElements = this->oneLoopMaxElements;
            InitMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, 0);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        } else if (this->blockIdx == currentStageNeedCoreNum - 1) {
            mrgsortParam.perListElements = perListElements;
            mrgsortParam.lastListElements = lastListElements;
            mrgsortParam.oneLoopMaxElements = this->oneLoopMaxElements;
            InitMrgSort(&mrgsorter, remainLn, coreOffset, 0);
            mrgsorter.Init(&mrgsortParam);
            mrgsorter.Process();
        }
        listNum = currentStageNeedCoreNum;
        srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;
        lastListElements = perListElements * (remainLn - 1) + lastListElements;
        perListElements = perListElements * MAX_MRGSORT_LIST;
        AscendC::SyncAll();
    }
}

__aicore__ inline void SortMultiCore::SortOutProcess()
{
    if (this->blockIdx < 1) {
        mrgsortParam.perListElements = perListElements;
        mrgsortParam.lastListElements = lastListElements;
        mrgsortParam.oneLoopMaxElements = this->oneLoopMaxElements;

        KvMrgSortOut sorter;
        InitMrgSortOut(&sorter, listNum, GetSortLen<float>(perListElements));
        sorter.Init(&mrgsortParam, pipe);
        sorter.Process();
    }
    AscendC::SyncAll();
}

__aicore__ inline void SortMultiCore::Init(
    GM_ADDR keys, GM_ADDR values, GM_ADDR sortedKeys, GM_ADDR sortedValues,
    GM_ADDR workspace, const SortKernelTiling* tiling, TPipe* tPipe)
{
    this->pipe = tPipe;
    this->totalLength = tiling->totalLength;
    this->coreNum = tiling->coreNum;
    this->needCoreNum = tiling->needCoreNum;
    this->blockIdx = GetBlockIdx();

    this->perCoreElements = tiling->perCoreElements;
    this->lastCoreElements = tiling->lastCoreElements;
    this->oneLoopMaxElements = tiling->oneLoopMaxElements;

    int64_t perCorePerLoopElem = tiling->perCorePerLoopElements;
    int64_t lastCorePerLoopElem = tiling->lastCorePerLoopElements;

    this->sortTotalLength = this->perCoreElements;
    if (this->blockIdx == this->needCoreNum - 1) {
        this->sortTotalLength = this->lastCoreElements;
        sortCoreLoops = tiling->lastCoreLoops;
        sortCoreLoopElements = tiling->lastCorePerLoopElements;
        sortCoreLastLoopElements = tiling->lastCoreLastLoopElements;
    } else {
        sortCoreLoops = tiling->perCoreLoops;
        sortCoreLoopElements = tiling->perCorePerLoopElements;
        sortCoreLastLoopElements = tiling->perCoreLastLoopElements;
    }

    keysGm.SetGlobalBuffer(
        (__gm__ int32_t*)keys + this->blockIdx * this->perCoreElements,
        this->sortTotalLength);
    valuesGm.SetGlobalBuffer(
        (__gm__ int32_t*)values + this->blockIdx * this->perCoreElements,
        this->sortTotalLength);
    sortedKeysGm.SetGlobalBuffer((__gm__ int32_t*)sortedKeys, this->totalLength);
    sortedValuesGm.SetGlobalBuffer((__gm__ int32_t*)sortedValues, AlignUp(this->totalLength, sizeof(int32_t)));

    int64_t kvFactor = 2;
    int64_t perCoreSortLen = GetSortLen<float>(this->perCoreElements);
    int64_t oneBufSize = this->needCoreNum * perCoreSortLen;
    workspaceGms[0].SetGlobalBuffer(
        (__gm__ float*)workspace,
        oneBufSize);
    workspaceGms[1].SetGlobalBuffer(
        (__gm__ float*)workspace + oneBufSize,
        oneBufSize);

    int64_t bufferSize = CeilDiv(
        MaxVal(this->oneLoopMaxElements * MAX_MRGSORT_LIST, sortCoreLoopElements),
        ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM * sizeof(int32_t) * kvFactor;
    pipe->InitBuffer(copyInQueue, 1, bufferSize);
    pipe->InitBuffer(copyOutQueue, 1, bufferSize);
    pipe->InitBuffer(tempBuffer, bufferSize);
    pipe->InitBuffer(sortedBuf, bufferSize);
}

__aicore__ inline void SortMultiCore::Process()
{
    VBSProcess();
    VMSProcess();
    SortOutProcess();
}

}  // namespace KvSort
#endif
