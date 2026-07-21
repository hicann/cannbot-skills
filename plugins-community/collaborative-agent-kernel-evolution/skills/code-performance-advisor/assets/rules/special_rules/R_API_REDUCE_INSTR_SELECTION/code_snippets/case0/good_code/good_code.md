# Good Code

```cpp
...
static constexpr uint32_t BLK_LEN = 32;
TBuf<QuePosition::VECCALC> calcBuf;
pipe.InitBuffer(calcBuf, totalLength * sizeof(float));
AscendC::LocalTensor<float> tempTensor1 = calcBuf.Get<float>();
constexpr uint32_t c0Count = BLK_LEN / sizeof(float);
const uint32_t blockNum0 = (totalLength + c0Count - 1) / c0Count;
AscendC::SetMaskCount();
AscendC::SetVectorMask<float>(0, totalLength);
AscendC::BlockReduceSum<float, false>(tempTensor1, xLocal, AscendC::MASK_PLACEHOLDER, 1,
  DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<float>(0, blockNum0);
AscendC::WholeReduceSum<float, false>(zLocal, tempTensor1, AscendC::MASK_PLACEHOLDER, 1,
  DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskNorm();
...
```