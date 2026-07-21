# Base Code

```cpp
constexpr int32_t TOTAL_LENGTH = 384 * 1024 * 1024 / sizeof(half);
constexpr int32_t USE_CORE_NUM = 20;
constexpr int32_t TILE_NUM = 2;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM;

class KernelSample {
public:
  __aicore__ inline KernelSample() {}
  __aicore__ inline void Init(GM_ADDR x)
  {
    xGm.SetGlobalBuffer((__gm__ half*)x + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH);
    pipe.InitBuffer(inQueueX, 1, BLOCK_LENGTH * sizeof(half));
    pipe.InitBuffer(inQueueY, 1, BLOCK_LENGTH * sizeof(half));
  }
  __aicore__ inline void Process()
  {
    // 示例演示对输入数据加2的运算
    constexpr int32_t loopCount = 2;
    for (int32_t i = 0; i < loopCount; i++) {
      // 外层的每次循环对输入数据进行加1的运算
      for (int32_t j = 0; j < TILE_NUM; j++) {
        // 内层循环分别处理每个核第0块和第1块数据
        CopyIn(j);
        Compute();
        CopyOut(j);
      }
    }
  }
private:
  __aicore__ inline void CopyIn(int32_t process)
  {
    LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
    // 对于每个核，除了首次读取外，读取第0块数据时，L2Cache内缓存的是第1块数据；
    // 对于每个核，读取第1块数据时，L2Cache内缓存的是第0块数据；
    // 每个核需要4次读取GM上的数据
    DataCopy(xLocal, xGm[process * TILE_LENGTH], TILE_LENGTH );
    inQueueX.EnQue(xLocal);
  }
  __aicore__ inline void Compute()
  {
    LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
    LocalTensor<half> xLocal = inQueueX.DeQue<half>();
    Adds(yLocal, xLocal, 1, TILE_LENGTH);   
    inQueueY.EnQue<half>(yLocal);
    inQueueX.FreeTensor(xLocal);
  }
  __aicore__ inline void CopyOut(int32_t process)
  {
    LocalTensor<half> yLocal = inQueueY.DeQue<half>();
    DataCopy(yGm[process * TILE_LENGTH], yLocal, TILE_LENGTH);
    inQueueY.FreeTensor(yLocal);
  }
}
...
```