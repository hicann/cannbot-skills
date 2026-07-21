# 规则名称：TQueBind 复用 VECIN/VECOUT，避免冗余拷贝

## 1. 需求场景 (Requirement)
- **业务背景**：算子属于纯数据搬运类型（如 DataCopy, Pad, Reverse），并不参与任何 Vector 或 Cube 计算。
- **形状/数据类型上下文**：数据量较大，导致片上局部张量（LocalTensor）频繁换入换出的情形。

## 2. 模式描述 (Pattern)
- **优化原理**：使用 `TQueBind` 接口将 `VECIN` (搬入队列) 与 `VECOUT` (搬出队列) 进行逻辑绑定。
- **目标**：省去“从 VECIN 到 VECOUT 重复拷贝一次”的动作。数据搬入后直接由搬出端消费。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：片上张量间的拷贝（Local to Local）虽然不消耗 HBM 带宽，但仍会占用 Vector 执行单元和指令流水。
- **事实桥接**：
  - 路径捷径 -> 消除搬运路径上的无效同步点（WaitFlag/SetFlag）。
  - 带宽直达 -> 使搬运重心集中在 MTE2/MTE3 上而非内存储搬移。

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `aic_mte2_ratio` / `aic_mte3_ratio`（搬运占比）
  - `aiv_vec_ratio`（向量利用率，如果不计算却占用了 vector）
  - `Task Duration(us)`（耗时走向）
- **如何解读（定性）**：
  - 如果是一个纯搬运算子，但 `aiv_vec_ratio` 却有明显的脉冲波形。
  - 源代码中包含类似于 `DataCopy(dstLocal, srcLocal, ...)` 的片上拷贝代码。

## 5. 动作实现 (Action)
- **参考代码位置**：`assets/rules/special_rules/R_API_TQUEBIND_VECIN_VECOUT/code_snippets/`
- **实施步骤**：
  - 将原有的 `QuePosition::VECIN` 和 `QuePosition::VECOUT` 的声明替换为 `TQueBind<VECIN, VECOUT, ...>`；
  - 移除两队列间的 `AllocTensor` / `EnQue` / `DeQue` 冗余传递代码。

## 6. 约束与副作用 (Constraints)
- **内存共享**：绑定后两阶段共享同一物理空间，必须确保处理顺序不冲突。
- **适用场景**：`U.DMA`, `S.TransferDominated`, `S.MteBusy`。
- **不适用场景**：中间需要进行向量计算的情形。

## 7. 验证逻辑 (Verification)
- **验证原则**：Vector 指令执行脉冲的消失。
- **推荐验证项**：
  - `aiv_vec_ratio`：期望降至极低（接近 0）；
  - `Task Duration(us)`：性能期望整体提升。
- **验证方法**：检查 Profiling，确认指令流中不再包含额外的片上 DataCopy 记录。

## 标签
- Domain: `U.DMA`, `O.DataCopy`
- Symptom: `S.TransferDominated`, `S.MteBusy`
- Context: `C.Arch.910B`
