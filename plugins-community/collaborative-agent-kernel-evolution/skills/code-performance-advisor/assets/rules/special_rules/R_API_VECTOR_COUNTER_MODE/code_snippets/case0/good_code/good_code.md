# Good Code

```cpp
uint32_t ELE_SIZE = 15000;
AscendC::BinaryRepeatParams binaryParams;
AscendC::SetMaskCount();
AscendC::SetVectorMask<DTYPE_X, AscendC::MaskMode::COUNTER>(ELE_SIZE);  // 设置counter模式mask，总共计算15000个数
AscendC::Add<DTYPE_X, false>(zLocal, xLocal, yLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);                // MASK_PLACEHOLDER值为0，此处为mask占位，实际mask值以SetVectorMask设置的为准
AscendC::ResetMask();  // 还原mask值
```