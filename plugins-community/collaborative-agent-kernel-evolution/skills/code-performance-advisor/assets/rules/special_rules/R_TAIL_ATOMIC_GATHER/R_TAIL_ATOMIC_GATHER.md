# 规则名称:非对齐尾块 AtomicAdd/GatherMask 处理

## 1. 需求场景 (Requirement)
- **业务背景**：数据长度非 32B 对齐时，简单的 DataCopy 可能导致越界访问或数据竞争，特别是多核并行写入同一输出区域时。
- **形状/数据类型上下文**：数据维度不是 32B（FP32 8个元素，FP16 16个元素）的整数倍，存在尾块数据需要特殊处理。

## 2. 模式描述 (Pattern)
- **优化思路**：针对非对齐尾块，使用 AtomicAdd 保证多核写入的原子性，或使用 GatherMask 指令重排数据后精确写入，避免数据竞争和越界。
- **目标**：保证非对齐数据的正确性，避免数据竞争导致的错误结果。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：AtomicAdd 操作有一定性能开销，但能避免数据竞争；GatherMask 需要额外的重排操作，但避免了 Atomic 开销。
- **事实桥接**：
  - 非对齐数据 -> 需要特殊处理 -> 避免越界和竞争
  - AtomicAdd -> 保证原子性 -> 牺牲少量性能
  - GatherMask -> 数据重排 -> 精确写入

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `Output Shapes`（输出形状，判断是否对齐）
  - `Data Type`（数据类型，影响对齐粒度）
  - `Block Dim`（多核并行，可能存在竞争）
  - `Correctness`（功能正确性，观察是否有数据错误）
- **如何解读（定性）**：
  - 输出维度不是 32B 对齐
  - 多核并行写入同一输出区域
  - 运行结果不正确或出现随机性错误
  - 使用标签 `C.Align.256B` 标注对齐约束

## 5. 动作实现 (Action)
- **参考代码位置**：`code_snippets/case1/` (adaptive_avg_pool3d AtomicAdd), `case2/` (adaptive_avg_pool3d GatherMask)
- **实施步骤（示例性）**：

  **方案 A: AtomicAdd 处理**
  1. 判断是否为尾块且可能存在竞争：
     ```cpp
     if ((validDataLen < numPerBlock) && (offset + validDataLen * atomicAddNum >= nextCoreAddrOffset))
     ```
  2. 使用 Duplicate 清零超出部分：
     ```cpp
     uint64_t mask0 = (1ul << numPerBlock) - (1ul << validDataLen);
     uint64_t mask[2] = {mask0, 0};
     Duplicate<T>(outputLocal, 0, mask, 1, 1, 1);
     ```
  3. 设置 AtomicAdd 模式执行 DataCopy：
     ```cpp
     SetAtomicAdd<T>();
     DataCopy(outputGlobal[offset], outputLocal, cTailAlign);
     SetAtomicNone();
     ```

  **方案 B: GatherMask 处理**
  1. 先写入对齐部分：
     ```cpp
     DataCopy(outputGlobal[offset], outputLocal, cTailAlign - numPerBlock);
     ```
  2. 使用 GatherMask 重排尾块数据：
     ```cpp
     LocalTensor<uint32_t> bufPattern = tmpPattern.Get<uint32_t>();
     int32_t preLeftShift = numPerBlock + lastLeftShift;
     bufPattern.SetValue(0, (1u << preLeftShift) - (1u << lastLeftShift));
     GatherMask(outputLocal[gatherOffset], outputLocal[gatherOffset],
                bufPattern, true, mask, {1, 1, 8, 8}, rsvdCnt);
     ```
  3. 写入重排后的尾块：
     ```cpp
     DataCopy(outputGlobal[nextCoreAddrOffset - numPerBlock],
              outputLocal[gatherOffset], numPerBlock);
     ```

## 6. 约束与副作用 (Constraints)
- **AtomicAdd 开销**：原子操作有性能损失（约 10-20%），但实现简单
- **GatherMask 复杂度**：需要额外的 pattern buffer 和复杂的重排逻辑
- **适用场景**：非对齐数据场景，多核并行写入
- **不适用场景**：数据已对齐或单核算子

## 7. 验证逻辑 (Verification)
- **验证原则**：功能正确性优先，性能损失可控
- **推荐验证项**：
  - `Correctness`：运行结果正确且稳定
  - `Task Duration(us)`：性能损失 < 15%（仅尾块有开销）
  - `Data Race`：消除数据竞争问题
- **验证方法**：
  - 对比有无尾块处理的运行结果
  - 使用 PyTorch Golden Reference 验证正确性
  - 多次运行确认结果稳定性

## 标签
- Domain: `U.Vector`, `U.Mix`
- Symptom: `S.LocalCopyRedundant`
- Context: `C.Align.256B`
