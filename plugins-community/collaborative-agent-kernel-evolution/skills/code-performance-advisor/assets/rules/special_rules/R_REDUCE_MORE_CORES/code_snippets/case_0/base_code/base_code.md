# Base Code

```cpp
// 原始代码（慢）：每 tile 标量累积
// N_CORES = 16，每核 12 个 tile
float partialSum = 0.0f;
for (uint32_t i = 0; i < this->innerLoops; i++) {
    CopyIn1(i);
    // Compute1 内部：
    AscendC::ReduceSum(sqLocal, sqLocal, sharedLocal, this->tileSize);
    float tileSum = sqLocal.GetValue(0);  // SCALAR BOTTLENECK
    partialSum = partialSum + tileSum;    // SCALAR ACCUMULATION
}
```
