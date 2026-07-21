# Base Code

```cpp
class KernelSample {
public:
  __aicore__ inline KernelSample() {}
  __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* dstGm)
  {
    src0Global.SetGlobalBuffer((__gm__ float*)src0Gm);
    dstGlobal.SetGlobalBuffer((__gm__ float*)dstGm);
    pipe.InitBuffer(inQueueSrc0, 1, 1024 * sizeof(float));
    pipe.InitBuffer(outQueueDst, 1, 1024 * sizeof(float));
  }
  __aicore__ inline void Process()
  {
    CopyIn();
    Compute();
    CopyOut();
    CopyIn1();
    Compute1();
    CopyOut1();
  }

private:
  __aicore__ inline void CopyIn()
  {
    LocalTensor<float> src0Local = inQueueSrc0.AllocTensor<float>();
    DataCopy(src0Local, src0Global, 1024);
    inQueueSrc0.EnQue(src0Local);
  }
  __aicore__ inline void Compute()
  {
    LocalTensor<float> src0Local = inQueueSrc0.DeQue<float>();
    LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
    Exp(dstLocal, src0Local, 1024);
    outQueueDst.EnQue<float>(dstLocal);
    inQueueSrc0.FreeTensor(src0Local);
  }
  __aicore__ inline void CopyOut()
  {
    LocalTensor<float> dstLocal = outQueueDst.DeQue<float>();
    DataCopy(dstGlobal, dstLocal, 1024);
    outQueueDst.FreeTensor(dstLocal);
  }
  __aicore__ inline void CopyIn1()
  {
	PipeBarrier<PIPE_ALL>();
    LocalTensor<float> src0Local = inQueueSrc0.AllocTensor<float>();
    DataCopy(src0Local, dstGlobal, 1024);
    inQueueSrc0.EnQue(src0Local);
  }
  __aicore__ inline void Compute1()
  {
    LocalTensor<float> src0Local = inQueueSrc0.DeQue<float>();
    LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
    Abs(dstLocal, src0Local, 1024);
    outQueueDst.EnQue<float>(dstLocal);
    inQueueSrc0.FreeTensor(src0Local);
  }
  __aicore__ inline void CopyOut1()
  {
    LocalTensor<float> dstLocal = outQueueDst.DeQue<float>();
    DataCopy(dstGlobal, dstLocal, 1024);
    outQueueDst.FreeTensor(dstLocal);
  }

private:
  TPipe pipe;
  TQue<QuePosition::VECIN, 1> inQueueSrc0;
  TQue<QuePosition::VECOUT, 1> outQueueDst;
  GlobalTensor<float> src0Global, dstGlobal;
};
```