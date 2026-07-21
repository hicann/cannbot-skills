# Good Code

```cpp
__aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
{
  src0Global.SetGlobalBuffer((__gm__ half*)src0Gm);
  src1Global.SetGlobalBuffer((__gm__ half*)src1Gm);
  dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
  // InitBuffer中使用2表示使能double buffer,占用的物理空间是 2 * sizeSrc0 * sizeof(half)
  // 3个InitBuffer执行后总空间为2 * (sizeSrc0 * sizeof(half) + sizeSrc1 * sizeof(half) + sizeDst0 * sizeof(half)) 
  pipe.InitBuffer(inQueueSrc0, 2, sizeSrc0 * sizeof(half));
  pipe.InitBuffer(inQueueSrc1, 2, sizeSrc1 * sizeof(half));
  pipe.InitBuffer(outQueueDst, 2, sizeDst0 * sizeof(half));
  }
__aicore__ inline void Process()
{
  // 开启double buffer的前提是循环次数 >= 2
  for (uint32_t index = 0; index < round; ++index) {
    CopyIn(index);
    Compute();
    CopyOut(index);
  }
}
```