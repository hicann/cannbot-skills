// bad_sync_kernel.cpp — 故意含多种同步缺陷，用于验证 sync_audit.py
// 每处缺陷标注 [SYNC-xx]

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t EVENT_ID0 = 0;
constexpr int32_t EVENT_ID7 = 7;
constexpr int32_t EVENT_ID8 = 8;

class KernelBad {
public:
    __aicore__ inline KernelBad() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        xGm.SetGlobalBuffer((__gm__ half *)x + this->blockOffset * totalLength / this->blockNum, totalLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + this->blockOffset * totalLength / this->blockNum, totalLength);
        pipe.InitBuffer(inQueueX, 2, totalLength * sizeof(half));
        pipe.InitBuffer(outQueueY, 2, totalLength * sizeof(half));
    }

    // [SYNC-01] Wait 先于 Set（死等）
    __aicore__ inline void CaseWaitBeforeSet() {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    // [SYNC-02] 搬入后未同步即计算
    __aicore__ inline void CaseMissingSyncIn() {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        DataCopyPad(xLocal, xGm, copyParams, padParams);
        Adds(yLocal, xLocal, 1.0f, count);  // 缺 MTE2→V 同步
    }

    // [SYNC-02] 计算后未同步即搬出
    __aicore__ inline void CaseMissingSyncOut() {
        Adds(yLocal, xLocal, 1.0f, count);
        DataCopyPad(yGm, yLocal, copyParams);  // 缺 V→MTE3 同步
    }

    // [SYNC-04] Set/Wait 个数不一致 + EVENT_ID 不匹配
    __aicore__ inline void CaseUnpaired() {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);  // 多余 Set
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        SetFlag<HardEvent::V_MTE3>(EVENT_ID7);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID8);  // EVENT_ID 不匹配
    }

    // [SYNC-06] 跨 PIPE 误用 PipeBarrier<PIPE_V>
    __aicore__ inline void CaseCrossPipeMisuse() {
        Adds(midLocal, xLocal, 1.0f, count);
        PipeBarrier<PIPE_V>();                  // 不跨 PIPE
        DataCopyPad(wsGm, midLocal, copyParams); // V→MTE3 缺 Flag
    }

    // [SYNC-08] 提前 return 跳过 SetFlag
    __aicore__ inline void ComputeMm1(bool edgeCase, int32_t syncC1V1) {
        if (edgeCase) {
            return;                              // 跳过 SetFlag → 死锁
        }
        CrossCoreSetFlag<HardEvent::MTE2_V>(syncC1V1);
    }

    // [SYNC-09] PipeBarrier<PIPE_ALL> 过粗 + 连续过多
    __aicore__ inline void CaseCoarseBarrier() {
        PipeBarrier<PIPE_ALL>();
        Adds(a, b, 1.0f, n);
        PipeBarrier<PIPE_V>();
        Adds(a, b, 1.0f, n);
        PipeBarrier<PIPE_V>();
        Adds(a, b, 1.0f, n);
        PipeBarrier<PIPE_V>();
        Adds(a, b, 1.0f, n);
        PipeBarrier<PIPE_V>();  // 连续 >3
    }

    // [SYNC-10] 无符号下溢
    __aicore__ inline void CaseLoopUnderflow(uint32_t loop, uint32_t preLoadNum) {
        uint32_t inIdx = preLoadNum > 0 ? (loop - 1) % preLoadNum : 0;  // loop=0 下溢
        DataCopyPad(buf[inIdx], xGm, copyParams);
    }

    // [SYNC-03] 单核场景残留 CrossCoreWaitFlag（无人 SetFlag → 死等）
    __aicore__ inline void CaseSingleCoreResidual(int32_t syncX) {
        CrossCoreWaitFlag<HardEvent::MTE2_V>(syncX);  // 单核无人 Set
        Adds(yLocal, xLocal, 1.0f, count);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueX;
    TQue<QuePosition::VECOUT, 2> outQueueY;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    GlobalTensor<half> wsGm;
    LocalTensor<half> yLocal;
    LocalTensor<half> xLocal;
    LocalTensor<half> midLocal;
    LocalTensor<half> a;
    LocalTensor<half> b;
    LocalTensor<half> buf[2];
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<half> padParams;
    uint32_t count;
    uint32_t n;
    uint32_t blockOffset;
    uint32_t blockNum;
};
