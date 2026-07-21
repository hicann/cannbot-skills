# Good Code

```cpp
// 优化代码：增加核心数 (16->48) + 向量 Add 累积替换标量循环
// N_CORES = 48，每核 4 个 tile，accBuf 向量累积

// 1. 初始化向量累积缓冲区
AscendC::LocalTensor<float> accLocal = accBuf.Get<float>();
AscendC::Duplicate(accLocal, 0.0f, this->tileSize);

// 2. 内层循环：向量 Add 累积（无标量 GetValue）
for (uint32_t i = 0; i < this->innerLoops; i++) {
    CopyIn1(i);
    // Compute1 内部改为：
    AscendC::Add(accLocal, accLocal, sqLocal, this->tileSize);  // VECTOR ACC
}

// 3. 循环结束后一次性 ReduceSum
AscendC::LocalTensor<float> sharedLocal = sharedBuf.Get<float>();
AscendC::ReduceSum(accLocal, accLocal, sharedLocal, this->tileSize);
float partialSum = accLocal.GetValue(0);  // 只调用一次 GetValue

// 4. Phase 2：使用动态 nCores 而非硬编码 16
uint32_t loadElems = ((this->nCores + 7) / 8) * 8;  // 32B 对齐
AscendC::DataCopy(workspaceLocal, workspaceGm[0], loadElems);
AscendC::ReduceSum(sharedLocal, workspaceLocal, sharedLocal, loadElems);
```
