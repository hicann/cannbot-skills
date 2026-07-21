# Good Code

```cpp
...
// 该样例仅做示例说明，非完整代码，省略了部分同步控制代码
public:
  __aicore__ inline KernelSample()
  {
    aSize = m * k;
    bSize = k * n;
    cSize = m * n;
  }
  __aicore__ inline void Init(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *bias, __gm__ uint8_t *c)
  {
    aGM.SetGlobalBuffer((__gm__ half *)a);
    bGM.SetGlobalBuffer((__gm__ half *)b);
    cGM.SetGlobalBuffer((__gm__ float *)c);
    biasGM.SetGlobalBuffer((__gm__ float *)bias);
    pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(half));
    pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(half));
    pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(half));
    pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(half));
    pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(float));
    pipe.InitBuffer(inQueueC1, 1, n * sizeof(float));
    pipe.InitBuffer(outQueueC2, 1, n * sizeof(float));
  }
  __aicore__ inline void Process()
  {
    CopyIn();
    SplitA();
    SplitB();
    SplitBias();
    Compute();
    CopyOut();
  }
private:
  __aicore__ inline void CopyIn()
  {
    LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
    LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();
    LocalTensor<float> bias1Local = inQueueC1.AllocTensor<float>();

    Nd2NzParams dataCopyA1Params;
    dataCopyA1Params.ndNum = 1;
    dataCopyA1Params.nValue = m;
    dataCopyA1Params.dValue = k;
    dataCopyA1Params.srcNdMatrixStride = 0;
    dataCopyA1Params.srcDValue = k;
    dataCopyA1Params.dstNzC0Stride = m;
    dataCopyA1Params.dstNzNStride = 1;
    dataCopyA1Params.dstNzMatrixStride = 0;
    DataCopy(a1Local, aGM, dataCopyA1Params);

    Nd2NzParams dataCopyB1Params;
    dataCopyB1Params.ndNum = 1;
    dataCopyB1Params.nValue = k;
    dataCopyB1Params.dValue = n;
    dataCopyB1Params.srcNdMatrixStride = 0;
    dataCopyB1Params.srcDValue = n;
    dataCopyB1Params.dstNzC0Stride = k;
    dataCopyB1Params.dstNzNStride = 1;
    dataCopyB1Params.dstNzMatrixStride = 0;
    DataCopy(b1Local, bGM, dataCopyB1Params);
    // 将bias从GM搬运到L1
    DataCopy(bias1Local, biasGM, n);

    inQueueA1.EnQue(a1Local);
    inQueueB1.EnQue(b1Local);
    inQueueC1.EnQue(bias1Local);
  }
  __aicore__ inline void SplitA()
  {
    ...
  }
  __aicore__ inline void SplitB()
  {
    ...
  }
  __aicore__ inline void SplitBias()
  {
    LocalTensor<float> bias1Local = inQueueC1.DeQue<float>();
    LocalTensor<float> bias2Local = outQueueC2.AllocTensor<float>();
    // 将bias从L1搬运到BT
    DataCopy(bias2Local, bias1Local, { 1, (uint16_t)(n * sizeof(float) / 64), 0, 0 });
    outQueueC2.EnQue<float>(bias2Local);
    inQueueC1.FreeTensor(bias1Local);
  }
  __aicore__ inline void Compute()
  {
    LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
    LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
    LocalTensor<float> bias2Local = outQueueC2.DeQue<float>();
    LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
    MmadParams mmadParams;
    mmadParams.m = m;
    mmadParams.n = n;
    mmadParams.k = k;
    mmadParams.cmatrixInitVal = false;
    // 矩阵乘
    Mmad(c1Local, a2Local, b2Local, bias2Local, mmadParams);
    outQueueCO1.EnQue<float>(c1Local);
    inQueueA2.FreeTensor(a2Local);
    inQueueB2.FreeTensor(b2Local);
    outQueueC2.FreeTensor(bias2Local);
  }
  __aicore__ inline void CopyOut()
  {
    LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
    FixpipeParamsV220 fixpipeParams;
    fixpipeParams.nSize = n;
    fixpipeParams.mSize = m;
    fixpipeParams.srcStride = m;
    fixpipeParams.dstStride = n;

    fixpipeParams.ndNum = 1;
    fixpipeParams.srcNdStride = 0;
    fixpipeParams.dstNdStride = 0;
    Fixpipe(cGM, c1Local, fixpipeParams);
    outQueueCO1.FreeTensor(c1Local);
  }
private:
  TPipe pipe;
  TQue<QuePosition::A1, 1> inQueueA1;
  TQue<QuePosition::A2, 1> inQueueA2;
  TQue<QuePosition::B1, 1> inQueueB1;
  TQue<QuePosition::B2, 1> inQueueB2;
  TQue<QuePosition::CO1, 1> outQueueCO1;
  TQue<QuePosition::C1, 1> inQueueC1;
  TQue<QuePosition::C2, 1> outQueueC2;

  GlobalTensor<half> aGM;
  GlobalTensor<half> bGM;
  GlobalTensor<dst_T> cGM;
  GlobalTensor<float> biasGM;
  uint16_t m = 32, k = 32, n = 32;
  uint16_t aSize, bSize, cSize;
```