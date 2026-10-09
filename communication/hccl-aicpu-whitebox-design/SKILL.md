---
name: hccl-aicpu-whitebox-design
description: 从 HCCL AICPU 算法的真实源码与 Dataflow Spec 设计可追溯的分支、边界和数据流用例矩阵。触发：设计或审查 AICPU 白盒测试计划时使用；不执行测试、不修改生产源码或宣称覆盖率达标。
---

# AICPU 白盒测试设计

输入为已确认的算子与算法、具名分支和源码身份、改动文件、需求/支持矩阵，以及已有的 [Dataflow Spec](../hccl-aicpu-design/references/07-dataflow-spec.md)；没有 Spec 且改动数据流时先标明设计缺口。只从当前源码可到达的路径推导用例，不照搬另一算子的阈值、通信域或测试数据。

## 从实现推导用例

1. 标出本次改动的 selector/成本候选、executor 分块与资源消费、template 的数据搬运和同步分支，以及可能变化的公共路径。对每个分支记录源码位置、触发条件、预期算法或输出与反例。
2. 针对实际分支选边界：最小/目标 rank，count 与对齐边界，dtype/归约操作，拓扑/链路模式，`loopTimes > 1` 与 `repeatNum > 1`。两个循环是不同路径；大小须由目标 executor 公式及测试程序的 bytes/count 换算得出，参见 [HCCL-VM 测试计划](../hccl-test-tool-hvm/references/test-plan.md)。没有该分支的算法不强加用例。
3. 对每条数据流用独立期望值核对源/目标、rank、slice、repeat、stride、scratch 容量与不重叠；对线程路径核对 Notify 数量、索引及同步前后的读写依赖。规则与量化方法分别见 [Template 规则](../hccl-aicpu-best-practice/references/07-rules.md) 和 [五参数说明](../hccl-aicpu-best-practice/references/09-template-data-params.md)。
4. 将可执行用例映射到现有测试工具：静态检查证明结构或公式，Checker 用例证明功能；选路用定向命中和默认回归两种证据。UT/ST 只在目标仓存在适用入口且调用方明确纳入范围时设计，不把静态结果或 HCCL-VM dry-run 写为测试 PASS。

每项矩阵至少记录 `case_id`、源码分支/条件、输入与拓扑、预期输出或错误、预期算法、所用工具、原始证据位置及来源身份。不可执行的分支保留为 `UNVERIFIED` 并注明缺少的工具、环境或可观察信号；不凭用例数宣称覆盖率。供 `hccl-op-dev` 使用时，Architect 将矩阵写入其拥有的 `TEST_PLAN.md`，Reviewer 可独立核对，Verifier 按冻结计划执行；本 Skill 不拥有这些角色或流程状态。
