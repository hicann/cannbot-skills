# 需求清单（CANNBot-Insight 设计文档索引）

本目录收录 CANNBot-Insight 的所有需求设计。**每个需求一个子目录**,内部按三阶段组织:
`phase1 需求分析` → `phase2 需求设计` → `phase3 开发计划`。

> 说明:子目录里的设计文档**只描述设计意图,不记录实现进度**。「是否实现」这类执行状态统一在本清单跟踪。

## 清单

| 需求名称 | 目录 | 需求描述 | 类型 | 创建时间 | 是否实现 | 对应 issue |
|-|-|-|-|-|-|-|
| Session 对话级洞察（Turn/Skill/SubAgent 可观测） | [session-turn-insight](session-turn-insight/) | 查看 session 中每次对话(turn)的 token 消耗、时间、调用的工具、加载的 skill；查看每个 subagent 的同样信息；以及主 agent 和 subagent 之间的交互信息 | Feature | 2026-06-12 | ⬜ 未实现（设计完成,待开发） | —（待补） |

## 字段口径

- **创建时间**:取该需求 phase1 起草日期。
- **是否实现**取值:
  - ⬜ **未实现** —— 仅有设计,代码未动
  - 🟡 **实现中** —— 部分落地
  - ✅ **已实现** —— 全部落地并通过验收
- **对应 issue**:关联的跟踪 issue/工单链接;暂无则填「待补」。

## 新增需求时的约定

1. 在本目录下新建一个**短横线命名**的子目录(如 `xxx-yyy`)。
2. 子目录内放 `phase1-requirements-analysis.md` / `phase2-requirements-design.md` / `phase3-development-plan.md`。
3. **回到本清单追加一行**,填齐上表各列。
