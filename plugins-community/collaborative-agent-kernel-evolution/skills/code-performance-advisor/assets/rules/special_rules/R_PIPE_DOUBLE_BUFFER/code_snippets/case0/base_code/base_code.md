# Base Code

```cpp
__aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
{
  src0Global.SetGlobalBuffer((__gm__ half*)src0Gm);
  src1Global.SetGlobalBuffer((__gm__ half*)src1Gm);
  dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
  // 不使能double buffer,占用的物理空间是 1 * sizeSrc0 * sizeof(half)
  // 3个InitBuffer执行后总空间为1 * (sizeSrc0 * sizeof(half) + sizeSrc1 * sizeof(half) + sizeDst0 * sizeof(half)) 
  pipe.InitBuffer(inQueueSrc0, 1, sizeSrc0 * sizeof(half));
  pipe.InitBuffer(inQueueSrc1, 1, sizeSrc1 * sizeof(half));
  pipe.InitBuffer(outQueueDst, 1, sizeDst0 * sizeof(half));
  }
__aicore__ inline void Process()
{
  // 需要round*2次循环才能处理完数据
  for (uint32_t index = 0; index < round * 2; ++index) {
    CopyIn(index);
    Compute();
    CopyOut(index);
  }
}
```