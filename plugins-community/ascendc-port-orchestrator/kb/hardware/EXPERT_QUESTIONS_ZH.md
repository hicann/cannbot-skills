# 硬件专家咨询问题 — 精简版（2026-04-21）

> 本文档是发给 **CANN / 硬件 / 编译器团队** 的问题清单。只保留公开文档 + 本地源码 + aog-hardware-probe 实测都无法回答的问题。
>
> 已答问题的处理方式：
> - aog-hardware-probe 实测 → `src/skills/references/hardware/probe_findings/`
> - CANN 源码扫描 → `src/skills/references/target/ascendc/patterns/` + `src/skills/references/hardware/target/ascend950pr.md`
> - 平台 bug → `src/skills/references/target/ascendc/PLATFORM_BUGS.md`

---

## 问题 1（CANN / bisheng 编译器团队）：PB-16 `DataCopy(UB, TBuf<TPosition::A1>)` 在纯 AIV kernel 静默 miscompile

**问题摘要**：在不含任何 Cube/Mmad 指令的纯 AIV kernel 中声明 `TBuf<TPosition::A1>` + 调用 `DataCopy(UB_tensor, L1_tensor)` 或反方向时，bisheng 静默编译通过（零 warning、零 error），运行时 100% 抛 `aivec error 259 "Illegal instruction"` (subErrType 0x4)。

**证据**：
- 实测：`probe_findings/2026-04-21_Q_l1_scratch_op11_kind2.md`（2 轮 iter，PipeBarrier 和 MTE3→MTE1 SetFlag 两种 sync 方案都是相同失败；不是 sync bug，是 opcode 层）
- CANN 源码扫描：`ops-transformer / ops-nn / opbase / catlass / graph-autofusion` 中所有 `TPosition::A1` 使用**全部**在 matmul / Cube 上下文。零纯 AIV 使用案例。低级 intrinsic `DataCopyUB2L1Impl` 只在 catlass 内部（cube 侧）调用。
- 受影响栈：bisheng clang 15.0.5 (build 2026-03-21)、CANN 9.0.0 V100R001C10SPC001B218、Ascend950PR_9589

**需要回答**（按优先级）：
1. 这是预期行为（A1/B1 本就是 Cube-only），还是 bisheng 缺少 reject 路径？若是预期行为，公开文档是否会明确加"TPosition::A1 禁用于纯 AIV kernel"的约束说明？
2. **AIV scope 下是否有 UB↔L1 的公开 intrinsic** 可使用 351x 架构引入的 AIV↔L1 硬通道？（如 `CopyUbufToL1` / `DataCopyUB2L1` / `LoadData` 变种）若有，API ref 页面 URL？
3. 若短期内无公开 AIV-scope UB↔L1 intrinsic，CANN 未来版本是否计划暴露？大致目标版本？
4. 针对 UB-budget-overflow 场景，CANN 官方推荐的优化路径（smaller tiles / fp16 intermediate / split kernel / mixed AIC+AIV task）在内部算子库中是否有 canonical 案例？

**业务动机**：DequantSwigluQuant 在 H=4992 下 UB 用量 194KB/192KB。能借 ~40KB L1 做 scratch 就可开 IN_QUE_DEPTH=2 获得 MTE2/VEC 流水 overlap（预期 2× 提升）。目前被 PB-16 完全堵死，只能走代价更大的 tile-split / fp16 intermediate 路径。

---

## 问题 2（硬件 / bisheng 团队）：UB 256KB 的 bank 数量与每 bank 宽度

**已知**（hiascend.com 351x 公开文档原文）：每个 bank group 有 **2 组读口 + 2 组写口**，支持 2R0W 或 1R1W 并发；冲突类型（RW / WW / RR）也有规则性描述。

**未知**（文字里没数字，只在架构图 3 里以图片呈现）：
1. UB 256KB 总共分多少个 bank？每个 bank 宽度（bytes）？
2. bank 之间的交织 granularity（interleave stride）？这决定 buffer 起始地址对齐值应选 64B / 128B / 256B / 其他哪个
3. 哪些 AscendC 原语在访问 UB 时有可预测的 bank-stride 模式（如 VEC 256B register load 是纯连续还是跨 bank 交织；DataCopyPad 多路并行访问同一 bank 的概率）？

**备注**：我们会同时尝试用 stride-sweep microbench + msprof bank_conflict 计数器实测推断，但实测只能告诉"哪些 stride 不冲突"，直接问到 bank 数量 + 宽度会快很多。

---

---

> 内部查询队列详情与历史：`src/skills/references/hardware/INTERNAL_QUERY_QUEUE.md`
>
> 本轮 aog-hardware-probe 实测已答的问题（不需要专家时间）：
> - Q_l1_scratch → ACCEPT_MISCOMPILE（升级为上面的问题 1）
> - Q_mte2_parallel → SHARED_CHANNEL（K2/K3 = 1.106×，DataCopy 在单 AIV 视角下基本共享通道）
> - Q_instruction_cycles → WholeReduceMax/BlockReduceMax ~100 cyc floor; MrgSort 线性于 output length (0.55 cyc × output_pairs + 42 cyc)
> - Q_scalar_broadcast → Brcb+Mul 路径 25.3× K_base（H=8 block-matched）;`Muls 灵活标量位置` 不 bypass Scalar pipe (argument-order variant)
