# 状态结构

> `.cannbot/state.json` 的字段约定，支撑「状态可观测」「可恢复性」。每阶段（步骤或 CP）完成后由 PM 更新。

## 设计目标

- **可观测**：不依赖会话记忆，从磁盘即可读出当前进度。
- **可恢复**：记录已完成阶段、当前阻塞点、已产出交付件、当前轮次，恢复时从状态点续跑，不重跑已通过阶段。

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
  "deliverables": {
    "requirement": ".cannbot/1.1-需求分析.md",
    "test_plan": ".cannbot/2.1-测试方案设计.md",
    "dev_plan": ".cannbot/2.2-开发方案设计.md",
    "cp3_report": ".cannbot/CP3-功能验收报告.md"
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
| `deliverables` | 已产出交付件路径（与 [data-flow.md](data-flow.md) 的落盘位置一致） |
| `updated_at` | 最后更新时间（ISO8601） |

## 更新时机

- 每步/每 CP 完成后，PM 更新 `current_stage`、追加 `completed_stages`、登记 `deliverables`。
- 暂停/失败（含过程有界超限）时填 `blocked`，作为可恢复状态点。
- 恢复：读 `current_stage` + `blocked` 定位入口，`completed_stages` 决定跳过范围。

> 字段结构可由 `scripts/validate_state.py`（预留）校验：编号合法性、completed_stages 与 current_stage 的顺序一致性。
