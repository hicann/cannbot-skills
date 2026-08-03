# 状态结构

> 每个算子各有一份 `.cannbot/<算子名>/state.json`，支撑「状态可观测」「可恢复性」。每阶段（步骤或 CP）完成后由 PM 更新。无顶层注册表——PM 启动时扫描 `.cannbot/*/state.json` 识别已存在算子（详见末尾「多算子识别与恢复」）。

## 设计目标

- **可观测**：不依赖会话记忆，从磁盘即可读出每个算子的当前进度。
- **可恢复**：记录已完成阶段、当前阻塞点、已产出交付件、当前轮次，恢复时从状态点续跑，不重跑已通过阶段。
- **算子隔离**：每个算子独占一个 `.cannbot/<算子名>/` 子目录与一份 `state.json`，多算子并行不互相覆盖。

## 字段定义

```json
{
  "operator": "算子名",
  "chip": "目标芯片/架构",
  "current_stage": "流程表编号，如 3.1 / CP3",
  "completed_stages": ["0", "CP0", "1.1", "CP1"],
  "blocked": {
    "at": "CP3",
    "reason": "结构化问题摘要（详见对应 report 或 issue）",
    "round": 2,
    "loop": "acceptance"
  },
  "rounds": { "CP1": 1, "CP2.2": 2 },
  "pending_questionnaire": {
    "cp": "CP2.2",
    "path": ".cannbot/<算子名>/questionnaires/CP2.2-方案确认.json",
    "reply_path": ".cannbot/<算子名>/questionnaires/CP2.2-方案确认.reply.json",
    "status": "sent"
  },
  "pending_ci": {
    "pr": "PR 链接或编号",
    "status": "waiting"
  },
  "deliverables": {
    "env_info": ".cannbot/环境信息.md",
    "requirement": ".cannbot/<算子名>/1.1-需求分析.md",
    "test_plan": ".cannbot/<算子名>/2.1-测试方案设计.md",
    "dev_plan": ".cannbot/<算子名>/2.2-开发方案设计.md",
    "cp3_report": ".cannbot/<算子名>/CP3-功能验收报告.md"
  },
  "updated_at": "2026-07-09T00:00:00Z"
}
```

## 字段说明

| 字段 | 含义 |
|------|------|
| `operator` / `chip` | 算子名、目标芯片，贯穿全程 |
| `current_stage` | 当前所处流程表编号，恢复时的入口 |
| `completed_stages` | 已通过的阶段列表，恢复时跳过 |
| `blocked` | 暂停时填充；`at`=阻塞的 CP，`reason`=问题摘要，`round`=当前轮次，`loop`=所属循环（`design`/`acceptance`/`ci`，对应 [error-handling.md](error-handling.md) 的轮次表） |
| `rounds` | 各 CP 已用轮次，**按 CP 编号分槽计数**；跨 CP 回退（如 CP2.2 因归属需求回退 1.1）不清零发起方槽位，避免往返途中计数丢失导致循环失去边界。`blocked.round` 取当前阻塞 CP 的槽值 |
| `pending_questionnaire` | 有问卷待用户答复时填；`cp`=发出问卷的 CP，`path`=问卷 json 路径，`reply_path`=用户回复落盘路径（问卷同名加 `.reply` 后缀，如 `1.需求.json` → `1.需求.reply.json`），`status`=`sent`（已发未回）/ `answered`（结论已回未处理）。用户回复到达时 PM 先落盘到 `reply_path` 再处理，问卷与回复成对持久化。用户确认类 CP（CP0 / CP1 / CP2.2）中断恢复的依据，处理完毕后清空 |
| `pending_ci` | PR 已提交、等待外部 CI 结果时填；`pr`=PR 标识，`status`=`waiting`（等待线上结果，会话已结束）/ `reported`（用户已回传 CI 结果、待处理）。PM 在 6.2 提交 PR 后立即填写并结束会话，禁止本地轮询死等；用户回传结果后置 `reported`，按结果进 6.4/6.5 或 CP6 并清空。**未清空即表示仍处等待态，恢复时不得直接进入后续步骤** |
| `deliverables` | 已产出交付件路径（与 [data-flow.md](data-flow.md) 的落盘位置一致） |
| `updated_at` | 最后更新时间（ISO8601） |

## 更新时机

- 每步/每 CP 完成后，PM **立即**更新 `.cannbot/<算子名>/state.json` 的 `current_stage`、追加 `completed_stages`、登记 `deliverables`——子 Agent 回传后、调度下一步之前先落盘，禁止攒到算子开发完成后一次性补写；中断恢复与多算子识别都依赖 state.json 的实时性。
- 暂停/失败（含过程有界超限）时填 `blocked`，作为可恢复状态点；每次打回/返工递增 `rounds` 中对应 CP 的槽位。
- 用户确认类 CP 发出问卷后填 `pending_questionnaire`，用户结论回传并处理完毕后清空。**未清空即表示该 CP 尚未收口，恢复时不得跳过**——防止把已产出的问卷误判为验收已通过。
- 6.2 提交 PR 后填 `pending_ci`（`waiting`）并结束会话；用户回传 CI 结果后置 `reported`，按结果分流处理完毕后清空。
- 恢复：读 `current_stage` + `blocked` 定位入口，`completed_stages` 决定跳过范围。

> 字段结构可由 `scripts/validate_state.py`（预留）校验：编号合法性、completed_stages 与 current_stage 的顺序一致性。

## 多算子识别与恢复

无顶层注册表。PM 会话开始（通过 `.cannbot/permissions/` 启动检查后）按以下规则识别已存在算子并选定本次推进对象：

1. 扫描 `.cannbot/*/state.json`，按 `operator` 字段列出所有已存在算子及其 `current_stage`。
2. 多个算子时，向用户确认本次推进哪一个；用户未指定且仅一个算子时默认选它。
3. 新算子：PM 在 `.cannbot/<算子名>/` 下创建子目录并初始化 `state.json`。初始化时先检查共享件 `.cannbot/环境信息.md`：
   - **不存在或环境已变**：`current_stage="0"`，`completed_stages=[]`——从阶段 0 跑起，采集并落盘环境信息。
   - **已存在且环境未变**：复用该文档，跳过阶段 0 与 CP0，`current_stage="1.1"`，`completed_stages=["0","CP0"]`。
