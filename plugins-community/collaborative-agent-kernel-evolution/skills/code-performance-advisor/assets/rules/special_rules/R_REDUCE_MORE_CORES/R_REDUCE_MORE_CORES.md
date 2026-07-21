# 规则名称：两阶段全局归约算子：增加核心数 + 向量累积替换标量循环

## 1. 需求场景 (Requirement)
- **业务背景**：两阶段全局归约算子（如 MSELoss、ReduceSum 等），Phase 1 各核心计算部分和，Phase 2 单核聚合。
- **形状/数据类型上下文**：输入数据量大（如 [4,128,768]=393216 元素），默认分配的核心数远少于硬件支持的最大核心数。当参考实现（如 torch.nn.MSELoss）使用更多核心时，若自定义算子使用过少核心，会造成显著的并行度损失。

## 2. 模式描述 (Pattern)
- **优化原理一：增加核心数**：将 Phase 1 核心数从少量（如 16）增加到接近参考实现使用的核心数（如 48），使每个核心处理更少数据，提高整体并行度。
- **优化原理二：向量累积替换标量循环**：在 Phase 1 内层 tile 循环中，将每个 tile 的 `ReduceSum + GetValue(0) + 标量加法` 模式替换为向量 `Add` 累积（维护 tileSize 大小的累积缓冲区），最后在所有 tile 完成后做一次 `ReduceSum` 得到每核部分和。
- **目标**：减少核心等待时间（更多并行度），消除每 tile 的标量 GetValue 依赖链（减少标量停顿）。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果一（核心数不足）**：
  - 核心数过少 → 每核数据量过大 → 总执行时间 = 串行时间 × 较少的并行因子
  - 对比：48 核 × 8192 elem/core = 393216；16 核 × 24576 elem/core = 393216（同等数据，16核串行 3 倍数据量）
- **因果二（标量循环开销）**：
  - 每 tile `ReduceSum + GetValue(0)` → 产生标量指令 → 阻塞下一 tile 的向量计算发射
  - 表现为 `aiv_scalar_ratio` 高（原始 0.292，优化前每 12 次 tile 循环各有一次标量等待）

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `Block Dim`（核心数是否显著少于参考实现）
  - `aiv_scalar_ratio`（标量占比是否高于预期，如 > 0.20）
  - `Task Duration(us)`（是否接近或超过参考实现时间）
  - `aiv_vec_ratio`（向量利用率是否偏低，如 < 0.20）
- **如何解读（定性）**：
  - 若 `Block Dim` 远小于参考实现使用的核心数，优先考虑增加核心数。
  - 若内层 tile 循环使用 `ReduceSum + GetValue + 标量累积` 模式，且 `aiv_scalar_ratio` > 0.20，考虑改为向量 Add 累积。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_REDUCE_MORE_CORES/code_snippets/`
- **实施步骤一（增加核心数）**：
  - 在 host tiling 中增加可用核心数（`N_CORES`），确保 `elemsPerCore = totalElems / nCores` 仍能被 `TILE_SIZE` 整除；
  - 在 tiling struct 中新增 `nCores` 字段，传递给 kernel；
  - 更新 workspace size：`nCores * sizeof(float)`（按 32 字节对齐）；
  - 更新 Phase 2 的 workspace 读取大小：从固定 16 改为 `((nCores + 7) / 8) * 8`。
- **实施步骤二（向量累积替换标量循环）**：
  - 新增 `TBuf<VECCALC> accBuf`，大小为 `tileSize * sizeof(float)`；
  - 在 `Process()` 开始时，`Duplicate(accLocal, 0.0f, tileSize)` 初始化为零；
  - Compute1 中将 `ReduceSum + GetValue + partialSum += tileSum` 替换为 `Add(accLocal, accLocal, sqLocal, tileSize)`；
  - 在 tile 循环结束后，调用一次 `ReduceSum(accLocal, accLocal, sharedLocal, tileSize)` 得到标量。

## 6. 约束与副作用 (Constraints)
- **核心数约束**：`totalElems` 必须能被 `nCores * TILE_SIZE` 整除，否则有尾块处理问题（需要 tail block 处理或保守选取 nCores）。
- **UB 内存开销**：新增 accBuf（tileSize * sizeof(float) = 8KB 对于 2048 float），需确认 UB 容量允许（910B UB = 256KB）。
- **Workspace 增大**：从 16 floats 扩展到 nCores floats（如 48 floats），需确保 workspace 申请对齐。
- **Phase 2 ReduceSum 要求**：加载的 workspace 元素数需为 32B 对齐（8 float），因此 `reduceElems = ((nCores+7)/8)*8`，多余的元素需初始化为 0（可通过 workspace 申请时 memset 或 padding 处理）。
- **适用场景**：`O.Loss`, `O.Reduce`, `U.Vector` 两阶段归约，且 `Block Dim` 有增长空间。

## 7. 验证逻辑 (Verification)
- **验证原则**：增大核心数 + 减少标量停顿的综合提升。
- **推荐验证项**：
  - `Block Dim`：期望从 N 增加到 M（如 16→48）；
  - `Task Duration(us)`：期望下降 ≥ 20%；
  - `aiv_vec_ratio`：可能略有变化（tile 数减少，每核 Add 次数减少）；
  - 精度验证：三路对比（Golden/Ref/Ans）所有 ratios ≤ 阈值，无 NaN/Inf 引入。
- **实测数据（MseLoss_cake_test）**：
  - Before: Block Dim=16, Duration=12.52us, scalar_ratio=0.292
  - After: Block Dim=48, Duration=9.224us (best), scalar_ratio=0.469 (更高但绝对时间减少)
  - 改善：+26.3%，达成 ≥20% 目标

## 标签
- Domain: `U.Vector`, `O.Loss`, `O.Reduce`
- Symptom: `S.ScalarBound`, `S.LowVecUtil`
- Context: `C.Arch.910B`, `C.Reduce.LastDim`
