# 规则名称：通过FP Buffer存放量化参数实现高效随路量化

## 1. 需求场景 (Requirement)
- **业务背景**：算子需要对矩阵乘的结果进行量化计算（如量化、去量化、Re-Quant）。
- **形状/数据类型上下文**：通常存在多层数据溢出或精度转换（`T.INT8`, `T.FP16`, `T.FP32`）。

## 2. 模式描述 (Pattern)
- **优化原理**：将量化参数搬运到 Fixpipe 专用缓冲区（FP Buffer/C2PIPE2GM）上。利用 `DataCopy` 或 `Fixpipe` 接口自带的硬件量化能力进行“随路量化”。
- **目标**：消除额外的 `GM -> UB` 搬运以及后续的 Vector 指令量化开销（如 `Cast`, `Mul` 等）。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：软量化路径会导致数据在主存片上多次循环。随路硬件量化可以将操作集成在搬运流水中，实现零延迟量化。
- **事实桥接**：
  - 流水集成 -> 减少 `aiv_vec_ratio` 的压力。
  - 搬运效率 -> 减小 `S.MemoryBound` 导致的等待。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `aic_mte2_ratio` / `aic_mte3_ratio`（数据搬运与回写占比）
  - `aiv_vec_ratio`（评估 Vector 层级的量化指令占比）
  - `Task Duration(us)`（耗时走向）
- **如何解读（定性）**：
  - 确认甘特图中是否包含显式的量化计算段。
  - `aic_mte2_ratio` 与 `aiv_vec_ratio` 均呈现异常高峰，说明量化过程占用了核心资源。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_MEM_FP_BUFFER_QUANT_PARAMS/code_snippets/`
- **实施步骤**：
  - 将量化参数分配在 `QuePosition::C2PIPE2GM`；
  - 调用 `DataCopy` 时设置量化模式参数（如 `QuantMode_t`）；
  - 配套使用 `SetFixpipeNz2ndFlag` 等硬件标记位。

## 6. 约束与副作用 (Constraints)
- **架构约束**：仅适用于 Atlas A2 / 910B 系列硬件（`C.Arch.910B`）。
- **功能局限**：量化参数格式必须符合硬核预设要求。
- **适用场景**：`S.MemoryBound`, `O.Fused`。

## 7. 验证逻辑 (Verification)
- **验证原则**：Vector 指令段的显著精简。
- **推荐验证项**：
  - `aiv_vec_ratio`：期望呈显著下降趋势；
  - `Task Duration(us)`：性能期望整体提升。
- **验证方法**：检查 Profiling，确认原有的量化计算循环被内嵌到数据搬运中。

## 标签
- Domain: `U.Cube`, `O.Fused`
- Symptom: `S.MemoryBound`, `S.TransferDominated`
- Context: `C.Arch.910B`
