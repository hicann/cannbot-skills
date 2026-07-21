# Good Code

```cpp
constexpr int32_t TOTAL_LENGTH = 384 * 1024 * 1024 / sizeof(half);
constexpr int32_t TILE_NUM = 2;
constexpr int32_t USE_CORE_NUM = 20;
constexpr int32_t TILE_LENGTH = TOTAL_LENGTH / TILE_NUM;
constexpr int32_t BLOCK_LENGTH = TILE_LENGTH / USE_CORE_NUM;

class KernelSample {
public:
  __aicore__ inline KernelSample() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int32_t index)
  {
    xGm.SetGlobalBuffer((__gm__ half*)x + BLOCK_LENGTH * GetBlockIdx() + index * TILE_LENGTH, BLOCK_LENGTH);
    yGm.SetGlobalBuffer((__gm__ half*)y + BLOCK_LENGTH  * GetBlockIdx() + index * TILE_LENGTH, BLOCK_LENGTH);
    pipe.InitBuffer(inQueueX, 1, BLOCK_LENGTH * sizeof(half));
    pipe.InitBuffer(inQueueY, 1, BLOCK_LENGTH * sizeof(half));
  }
  __aicore__ inline void Process()
  {
    // 示例演示对输入数据加2的运算
    constexpr int32_t loopCount = 2;
    for (int32_t i = 0; i < loopCount; i++) {
      // 每次循环对输入数据进行加1的运算
      CopyIn();
      Compute();
      CopyOut();
    }
  }
private:
  __aicore__ inline void CopyIn()
  {
    LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
    // 对于每个核，除了首次读取外，第二次读取可以命中L2Cache；
    // 每个核2次读取GM上的数据，2次访问L2Cache读数据
    DataCopy(xLocal, xGm, BLOCK_LENGTH );
    inQueueX.EnQue(xLocal);
  }
  __aicore__ inline void Compute()
  {
    LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
    LocalTensor<half> xLocal = inQueueX.DeQue<half>();
    Adds(yLocal, xLocal, 1, BLOCK_LENGTH);   
    inQueueY.EnQue<half>(yLocal);
    inQueueX.FreeTensor(xLocal);
  }
  __aicore__ inline void CopyOut()
  {
    LocalTensor<half> yLocal = inQueueY.DeQue<half>();
    DataCopy(yGm, yLocal, BLOCK_LENGTH);
    inQueueY.FreeTensor(yLocal);
  }
}
...

extern "C" __global__ __aicore__ void simple_kernel(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm)
{
  AscendC::KernelAdd op;
  // 输入数据均等切分成2份数据进行计算
  for (int32_t i = 0; i < TILE_NUM; i++) {
    op.Init(srcGm, dstGm, i);
    op.Process();
  }
}
...
```