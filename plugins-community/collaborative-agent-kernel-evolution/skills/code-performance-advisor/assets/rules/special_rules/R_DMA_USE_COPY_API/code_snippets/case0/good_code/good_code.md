# Good Code

```cpp
LocalTensor<float> tensorIn;
GlobalTensor<float> tensorGM;
...
constexpr int32_t copyWidth = 2 * 1024 / sizeof(float);
constexpr int32_t imgWidth = 16 * 1024 / sizeof(float);
constexpr int32_t imgHeight = 16;
// 通过DataCopy包含srcStride/dstStride/blockLen/blockCount的接口一次搬完
DataCopyParams copyParams;
copyParams.blockCount = imgHeight;
copyParams.blockLen = copyWidth / 8;   // 搬运的单位为DataBlock(32Byte)，每个DataBlock内有8个float
copyParams.srcStride = (imgWidth  - copyWidth ) / 8;   // 表示两次搬运src之间的间隔，单位为DataBlock
copyParams.dstStride = 0;                              // 连续写，两次搬运之间dst的间隔为0，单位为DataBlock
DataCopy(tensorGM, tensorIn, copyParams);
```