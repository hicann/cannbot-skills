---
type: CATLASS DSL Programming Concept
title: 多 kernel 合并为单 kernel
description: 多个串行 launch 的 kernel 拼成一个 @tla.kernel 单次 launch 时，段间同步的原语选择、边界判定、硬约束与容量限制。
tags: [catlass-dsl, fusion, cross-flag, sync, single-kernel]
status: stable
generated: {by: process:catlass-dsl-pattern-capture, at: '2026-11-15T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-kernel-run-audit, at: '2026-11-15T00:00:00Z'}
sources:
  - id: sync
    resource: https://gitcode.com/cann/catlass/blob/master/python/tla_dsl/docs/zh/kernel_development/core_concepts/sync.md
    title: CATLASS DSL 同步原语与 mode 表
  - id: sample
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/tla/kernel/matmul/fused_ffn/ffn/sync_hard.py
    title: 用 cross_flag 复刻 SyncAll<false>()
  - id: measure
    resource: project-evidence:topk-log-probs-backward-2k-to-1k-merge
    title: 合并前后逐位一致、同进程配对计时（Ascend950PR）
operator_families: [matmul, mixed]
dsl_version: ascend-catlass-dsl 2.2.0
arch: [c310]
---

# 适用条件

多个 `@tla.kernel` 依次 launch、后一段读前一段写在 GM 的结果，这种"stream 串行"在合成
一次 launch 后**不再存在**：MIX 1:2 执行体里所有 AIC/AIV 同时启动，段间先后必须逐条重建。

合并可行、且值得做的条件：

- 段间数据流是单向的（A → B → C，没有回边）；
- 每段的 tile 映射可以独立保留，不需要两段同时驻留同一份片上缓冲；
- 合并后仍只有一个 `@tla.kernel`、`run` 只 launch 一次。

各段原 `block_num` 不同也可以合并，但合并后整个 kernel 只保留一个 `block_num`：任务少的
段里，多出来的核走空 range 后仍须到达该段栅栏。

不满足时不要合并：例如两段各自需要一份完整 L0C 输出 tile（`BM·BN·4` 顶满后无法再放第二
份）。这类情形只能缩 tile 或把中间量落到 GM，吞吐通常会变差，**必须重测配对性能后再决定**，
不能默认合并值得做；确实不可行时如实记录"用户要求无法满足"及容量账，比强行拼更可靠。

# 原语能力边界

| 原语 | 作用范围 | 关键约束 |
|---|---|---|
| `tla.mutex(id=...)` | 仅单核内跨 pipe | 不能跨核；Cube / Vector 两侧 id 空间互相独立 |
| `cross_flag` mode 0 | 同类全核栅栏（所有 AIC 或所有 AIV） | 全 kernel 共 16 个 id，每 id 一次性计数 |
| `cross_flag` mode 1 | 同核内两个 AIV 互等 | 同上 |
| `cross_flag` mode 2（默认） | 同核 AIC ↔ 本核两个 AIV | AIC set 一次放行本核 2 个 AIV；2 个 AIV 都 set 才放行本核 AIC |
| `cross_flag` mode 4 | 同核 AIC ↔ 单个 AIV（`aiv_id=0/1`） | 仅 Ascend 950；必须给 aiv_id |
| `AscendC::SyncAll()` | 仅 AIV 全核 | 不等 AIC，不能当全局屏障 |
| `AscendC::SyncAll<false>()` | AIV-AIV + AIC-AIC + AIV↔AIC | 需 MIX 1:2 配对成立、两侧调用次数严格一致、`block_num ≤ 物理核数` |

证据：`python/tla_dsl/docs/zh/kernel_development/core_concepts/sync.md`（mode 表与 16 个 id
上限）[^sync]、`tla/kernel/matmul/fused_ffn/ffn/sync_hard.py`（用 cross_flag 复刻
`SyncAll<false>()`）[^sample]、`include/catlass/gemm/kernel/dual_level_quant_mx_batched_matmul_tla.hpp`
（`SyncAll<false>()` 一次完成三组同步）。

# 边界 → 原语判定

按"谁等谁"和"覆盖范围"两维判定。**本节在 implement 阶段使用**：design 只登记边界方向与
映射一致性，不预选原语，也不要为选型去调研同步 API；implement 按本节选型后回填 DESIGN 的
边界清单。

1. **AIV → AIV**：`SyncAll()`。
2. **AIC → AIC**：`cross_flag` mode 0，不要用 `SyncAll<false>()`（会强制 AIV 参与）。
3. **跨引擎且前后映射一致**（本核 AIC 只消费本核 AIV 的产出）：mode 2；要按 AIV 粒度细分
   时用 mode 4。
4. **跨引擎且前后映射不一致**（第 2 段的 AIC 可能读别的核 AIV 写的数据）：必须全局栅栏，
   即按 `SyncAll<false>()` 的组合语义实现——`aiv_arrive`（mode 4，每个 AIV 独立到达）+
   `aic_ffts`（mode 0，全 AIC 栅栏）+ `aic_done`（mode 2，放行本核 AIV）。只用 mode 2 不够。
5. **无依赖的两段**（都只读同一上游、写不同输出）：不设栅栏，用一张任务表并列
   （`tla.range(block_idx, total_a + total_b, block_num)`），每核按 `block_idx` 先领完前段
   tile 再领后段。

# 三条硬约束

1. **空核必须到达每个栅栏**：任务数少于核数时，空核走空 range 后仍调用同一栅栏；任何
   `if` 分支跳过栅栏都会死锁。栅栏写在 per-task 循环之外。
2. **pipe 必须对**：`set_flag` 用发布方数据真正落盘的那条管（AIV 写 GM → MTE3；AIC 写 GM
   → FIX）；`wait_flag` 用等待方真正开始搬运的那条管（读 GM → MTE2，或按实际等待的
   SCALAR/FIX）。选错不报错，只出脏数据或偶发挂死。
3. **每段前后 `pipe_barrier(ALL)`**：发布前排空本核各 pipe，消费前同样排空。

# 实施步骤

1. 先拿到一个可运行的多 kernel 版本作为语义与性能基线（已有就直接复用，没有才先写一版）。
2. 列出段清单与边界清单（依赖方向、前后段映射是否一致、覆盖核范围）。原语与 pipe 用上一节
   的判定表一次选型，不重新调研 API。
3. 把各段 kernel 体原样搬进同一个 `@tla.kernel`，保留各自的 tile 常量与缓冲；段间只加上一步
   选定的栅栏，不改数学语义与累加顺序。全局栅栏之后**不能沿用前一段的 tile 索引**：
   每段按自己的 `tla.range` 重新领 tile，这就是"映射不一致"的落地方式。
4. 无依赖的两段（第 5 类）合并成一张任务表，不额外加栅栏。
5. 逐位对比：同一 seed、同一输入，合并版与基线输出 `bitwise_equal`；不等先查栅栏与 pipe。
   前提是中间量的窄化点没变——基线把中间量落 GM 并窄化、合并后留在片上不窄化时，本来就
   不该逐位，按精度分档规则判"不劣于基线"即可，不得为了逐位强行插一次窄化。
6. 同进程配对计时（基线 / 合并版交替），确认回退在可接受范围内；profiling 确认每次 `run`
   只有一行目标 kernel。

# 实测数据

`topk_log_probs_backward`（M=41316/51350，H=2560，V=124160，K=128，Ascend950PR，28 cube）[^measure]：

- 2 kernel / 3 次 launch → 1 个 `@tla.kernel` / 1 次 launch；
- `grad_hidden`、`grad_weight` 与基线**逐位一致**（`bitwise_equal=true`、`max_abs_err=0`）；
- 端到端（同进程配对）：小 shape −0.7%、device −1.5%；最大 modelShape +1.85%、
  device +1.81%，均在「不慢于基线 5%」以内；
- 结构比对：段 A（cube/vector）、段 B 的 dh 部分与基线逐 token 相同，仅 dw 部分尾部多一个
  必要的 `pipe_barrier(ALL)`；
- 边界 B1 用第 4 类组合栅栏，B2 用任务表并列，均未挂死、未超时。

# 失败表现

- 用默认 `SyncAll()` 当全局屏障：AIC 不等 AIV，下游读到未写完的 GM，表现为"只累加最后
  一个 K 块"、多次运行结果漂移。
- `SyncAll<false>()` 只在一侧调用或两侧次数不等：直接死锁。
- 映射变化的边界上用 mode 2：只保证本核 AIV 写完，读到别的核未写完的数据。
- 用 `tla.mutex` 做跨引擎同步：无效（两套对象）。
- 栅栏写在 `if` 里：空核死锁。

[^sync]: 固定提交的同步原语文档，含 mode 表与 16 个 cross_flag id 上限。
[^sample]: 官方 MIX 样例，展示用 cross_flag 组合复刻 `SyncAll<false>()` 的语义。
[^measure]: 项目实测证据：2 kernel / 3 次 launch 合并为单 kernel / 单次 launch，逐位一致与配对计时记录。
