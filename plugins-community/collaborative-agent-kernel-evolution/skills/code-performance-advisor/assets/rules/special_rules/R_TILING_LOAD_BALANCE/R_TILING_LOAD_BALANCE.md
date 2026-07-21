# 规则名称：多核负载均衡 Former/Tail 模式

## 1. 需求场景 (Requirement)
- **业务背景**：多核并行场景下，待处理数据量不能被核心数整除，导致部分核心负载较轻或空闲，整体并行效率下降。
- **形状/数据类型上下文**：输出点数/行数等可并行维度与 `Block Dim`（核心数）存在余数关系。

## 2. 模式描述 (Pattern)
- **优化思路**：采用 Former/Tail 双段分配策略，前 `formerNum` 个核心处理 `formerLength` 个单元，剩余 `tailNum` 个核心处理 `tailLength` 个单元（通常为 `formerLength - 1`），使得核间负载差异最多为 1 个处理单元。
- **目标**：最大化多核并行效率，避免核心空闲和负载严重不均。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：简单均分导致最后一个核心负载可能远小于其他核心，造成等待同步时的时间浪费。Former/Tail 模式使负载差异控制在 ±1 个单元。
- **事实桥接**：
  - 均匀负载 -> 减少核间同步等待 -> 降低整体耗时
  - 动态核数调整 -> 避免无效核心启动 -> 节省资源

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `Block Dim`（实际使用的核心数）
  - `Output Shapes`（输出数据量，判断是否能被核数整除）
  - `Task Duration(us)`（整体耗时）
- **如何解读（定性）**：
  - 如果 `输出点数 % Block Dim != 0`，存在负载不均问题
  - 观察最后一个核心的处理量是否明显小于其他核心
  - 检查是否有核心处于空闲状态（`usedCoreNum < Block Dim`）

## 5. 动作实现 (Action)
- **参考代码位置**：`code_snippets/case1/` (adaptive_avg_pool3d), `case2/` (deep_norm)
- **实施步骤（示例性）**：
  1. 计算总处理单元数 `totalNum`（如输出点数、行数等）
  2. 计算 `formerLength = ceil(totalNum / coreNum)`
  3. 计算 `formerNum = totalNum % coreNum`（余数个核心多处理一个单元）
  4. 计算 `tailNum = coreNum - formerNum`
  5. 计算 `tailLength = formerLength - 1`
  6. 特殊处理：当 `totalNum < coreNum` 时，设置 `usedCoreNum = totalNum`，`formerLength = 1`
  7. 在 Kernel 中根据 `block_idx` 判断当前核心属于 former 还是 tail 段

## 6. 约束与副作用 (Constraints)
- **Tiling 复杂度**：需要维护 `formerNum/tailNum/formerLength/tailLength` 等多个参数
- **适用场景**：多核并行场景，特别是输出数据量与核数存在明显余数关系时
- **不适用场景**：单核算子或数据量恰好能被核数整除的场景

## 7. 验证逻辑 (Verification)
- **验证原则**：多核利用率均衡，无明显的核间等待
- **推荐验证项**：
  - `Task Duration(us)`：期望呈下降趋势（特别是非整除场景）
  - `Core Utilization Distribution`：各核心负载差异控制在 ±1 个处理单元内
  - `Synchronization Overhead`：核间同步等待时间减少
- **验证方法**：对比 Former/Tail 模式与简单均分模式的性能数据，确认负载均衡效果

## 标签
- Domain: `U.Vector`, `U.Mix`, `U.Cube`
- Symptom: `S.LowComputeUtil`, `S.PipeStall`
- Context: `C.Batch.Small`, `C.Tile.Small`
