# Base Code

```cpp
反例1：
...
static constexpr uint32_t REP_LEN = 256;
TBuf<QuePosition::VECCALC> calcBuf;
pipe.InitBuffer(calcBuf, totalLength * sizeof(float));
AscendC::LocalTensor<float> tempTensor1 = calcBuf.Get<float>();
constexpr uint32_t repCount = REP_LEN / sizeof(float);
const uint32_t repNum0 = (totalLength + repCount - 1) / repCount;
AscendC::SetMaskCount();
AscendC::SetVectorMask<float>(0, totalLength);
AscendC::WholeReduceSum<float, false>(tempTensor1, xLocal, AscendC::MASK_PLACEHOLDER, 1,
  DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float>(0, repNum0);
AscendC::WholeReduceSum<float, false>(zLocal, tempTensor1, AscendC::MASK_PLACEHOLDER, 1,
  DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskNorm();
...
反例2
...
constexpr uint32_t c0Count = BLK_LEN / sizeof(DTYPE_X);
const uint32_t blockNum0 = (totalLength + c0Count - 1) / c0Count;
const uint32_t blockNum1 = (blockNum0 + c0Count - 1) / c0Count;
AscendC::SetMaskCount();
AscendC::SetVectorMask<DTYPE_X>(0, totalLength);
AscendC::BlockReduceSum<DTYPE_X, false>(tempTensor1, xLocal, AscendC::MASK_PLACEHOLDER, 1,
  DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<DTYPE_X>(0, blockNum0);
AscendC::BlockReduceSum<DTYPE_X, false>(tempTensor1, tempTensor1, AscendC::MASK_PLACEHOLDER, 1,
  DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<DTYPE_X>(0, blockNum1);
AscendC::BlockReduceSum<DTYPE_X, false>(zLocal, tempTensor1, AscendC::MASK_PLACEHOLDER, 1,
  DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskNorm();
...
```