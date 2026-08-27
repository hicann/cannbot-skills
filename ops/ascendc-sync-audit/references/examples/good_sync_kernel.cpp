// good_sync_kernel.cpp — 同步正确的参考实现
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t EVENT_ID0 = 0;

class KernelGood {
public:
    __aicore__ inline KernelGood() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        xGm.SetGlobalBuffer((__gm__ half *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ half *)y, totalLength);
        pipe.InitBuffer(inQueueX, 2, totalLength * sizeof(half));
        pipe.InitBuffer(outQueueY, 2, totalLength * sizeof(half));
    }

    // 标准 EnQue/DeQue 三段流水，自带同步，无冗余 Flag
    __aicore__ inline void Process(uint32_t tileIdx, uint32_t tileSize) {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        DataCopyPad(xLocal, xGm[tileIdx * tileSize], copyParams, padParams);
        inQueueX.EnQue(xLocal);                           // MTE2→V 同步

        LocalTensor<half> xIn = inQueueX.DeQue<half>();   // 阻塞等待 MTE2
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        Adds(yLocal, xIn, 1.0f, tileSize);
        outQueueY.EnQue(yLocal);                          // V→MTE3 同步
        inQueueX.FreeTensor(xIn);

        LocalTensor<half> yOut = outQueueY.DeQue<half>(); // 阻塞等待 V
        DataCopyPad(yGm[tileIdx * tileSize], yOut, copyParams);
        outQueueY.FreeTensor(yOut);
    }

    // 自定义多 PIPE：方向正确、配对一致、Set 先于 Wait
    __aicore__ inline void CustomPipe(uint32_t tileIdx) {
        DataCopyPad(midLocal, xGm[tileIdx], copyParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Adds(midLocal, midLocal, 1.0f, count);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(wsGm[tileIdx], midLocal, copyParams);
    }

    // 双 buffer 轮转：下一轮搬入前等上一轮 MTE3 搬完，首轮保护
    __aicore__ inline void DoubleBuffer(uint32_t nLoop, uint32_t preLoadNum) {
        for (uint32_t i = 0; i < nLoop; i++) {
            if (i > 0) {
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            uint32_t inIdx = preLoadNum > 0 ? i % preLoadNum : 0;  // 不用 (loop-1)，无下溢
            ComputeV(buf[inIdx]);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(wsGm[i], buf[inIdx], copyParams);
        }
    }

    __aicore__ inline void ComputeV(LocalTensor<half> b) {
        Adds(b, b, 1.0f, count);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueX;
    TQue<QuePosition::VECOUT, 2> outQueueY;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    GlobalTensor<half> wsGm;
    LocalTensor<half> midLocal;
    LocalTensor<half> buf[2];
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<half> padParams;
    uint32_t count;
};
