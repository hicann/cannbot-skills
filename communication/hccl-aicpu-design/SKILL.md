---
name: hccl-aicpu-design
description: 基于目标 HCCL 版本的源码和算子契约设计 AICPU 算法的层级改动、数据流、资源与选路方案。触发：新增或修改 AICPU 算法，需要形成设计与 Dataflow Spec 时使用。角色分工和阶段流转由 hccl-op-dev 定义。
---

# HCCL AICPU 设计能力

适用于新增或修改 AICPU 算法时的需求澄清、方案选择和数据流规格。本 Skill 的算法设计入口是[通信算法设计](references/08-algorithm-design.md)；目标版本的 HCCL 源码和架构文档优先于历史示例。不预设 HCCL 仓路径，调用方须提供已确认的仓根、分支和源码身份。

## 设计检查

1. 固定算子语义、目标拓扑、rank、数据类型、数据量、默认选路影响与功能验收范围。缺少会改变设计或验收的用户决策时，列出输入缺口，不猜测。
2. 按目标源码划分 template、executor、selector、TopoMatch 的实际改动；未受影响层明确标记不改。注册名、算法身份、资源申请与消费者保持一致。算法形态、阶段、片归属和正确性证明按[通信算法设计](references/08-algorithm-design.md)推导，不照搬历史实现。
3. 新增算法或改变 slice、scratch、同步点、收发模式时，用[Dataflow Spec 记法](references/07-dataflow-spec.md)、[规格模板](dataflow-spec-template.md)和[已核验示例](assets/dataflow-spec.example.md)描述每轮数据的源、目标、偏移、长度和同步；再调用 `scripts/check_spec.py`、`scripts/check_layout.py` 消除语义 TBD，才进入实现。
4. 只显式启用与改变默认选择是两种不同设计。依据目标版本实际 selector 和成本候选链固定影响范围、支持矩阵及验收矩阵；不凭空设阈值或假定某个配置入口存在。
5. 需要可追溯的分支和边界用例时，再加载 `hccl-aicpu-whitebox-design`；不要把用例设计等同于实际测试通过。

输出技术设计应能让实现者和独立检视者复核每一层的必要性、数据流、资源、选路及其源码依据。谁编写设计产物、何时冻结及如何流转，由上层 Agent 和 Plugin 决定。
