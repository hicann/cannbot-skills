# 状态结构

> 每个算子各有一份 `.cannbot/<算子名>/state.json`，每阶段完成后由 PM 更新。

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
  "pending_user_review": [
    {
      "at": "CP4",
      "decision": "未达标项按已知限制收口",
      "scope": "具体未达标项与量化差距",
      "evidence_path": ".cannbot/<算子名>/CP4-性能验收报告.md",
      "decided_by": "PM",
      "decided_at": "2026-07-09T00:00:00Z"
    }
  ],
  "deliverables": {
    "env_info": ".cannbot/环境信息.md",
    "requirement": ".cannbot/<算子名>/1.1-需求分析.md",
    "test_plan": ".cannbot/<算子名>/2.1-测试方案设计.md",
    "dev_plan": ".cannbot/<算子名>/2.2-开发方案设计.md",
    "joint_debug_report": ".cannbot/<算子名>/3.4-联调报告.md",
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
| `completed_stages` | 已通过的阶段列表，恢复时跳过。可插拔流程插件的内部步骤编号亦记入本列表，编号含义见对应插件文档 |
| `blocked` | 暂停时填充；`at`=阻塞的环节（CP 或流程表步骤），`reason`=问题摘要，`round`=当前轮次，`loop`=所属循环（`design`/`joint_debug`/`acceptance` 对应 [error-handling.md](error-handling.md) 的轮次表；`ci` 属上库插件的轮次表，见对应插件文档） |
| `rounds` | 各环节已用轮次，**按流程表编号分槽计数**（含 3.4 联调槽位）；跨环节回退（如 CP2.2 因归属需求回退 1.1）不清零发起方槽位，避免往返途中计数丢失导致循环失去边界。`blocked.round` 取当前阻塞环节的槽值 |
| `pending_questionnaire` | 有问卷待用户答复时填；`cp`=发出问卷的 CP，`path`=问卷 json 路径，`reply_path`=用户回复落盘路径（问卷同名加 `.reply` 后缀，如 `1.需求.json` → `1.需求.reply.json`），`status`=`sent`（PM 派出问卷类 CP 时预填，已发未回）/ `answered`（QA 结论已回传、未处理）。问卷由 QA 用 question 工具直接发送用户：发出时落盘 `path`，收到回复后先落盘 `reply_path` 再回传，问卷与回复成对持久化。用户确认类 CP（CP0 / CP1 / CP2.2）中断恢复的依据，处理完毕后清空 |
| `pending_user_review` | **待用户复核的需求级决策**列表（放宽需求文档声明的性能 / 精度硬门槛等）。每条：`at`=作出决策的环节，`decision`=决策内容，`scope`=涉及的具体项与量化差距，`evidence_path`=完整依据所在交付件，`decided_by`/`decided_at`=作出方与时间。**只增不自行清除**——须随任务完成总结逐条上报，由用户裁定后才可移除并记录裁定结果。无此类决策时省略该字段或置空数组 |
| `deliverables` | 已产出交付件路径（与 [data-flow.md](data-flow.md) 的落盘位置一致） |
| `updated_at` | 最后更新时间（ISO8601） |

## 更新时机

- 每步/每 CP 完成后，PM **立即**更新 `.cannbot/<算子名>/state.json` 的 `current_stage`、追加 `completed_stages`、登记 `deliverables`——子 Agent 回传后、调度下一步之前先落盘，禁止攒到算子开发完成后一次性补写；中断恢复与多算子识别都依赖 state.json 的实时性。
- 暂停/失败（含过程有界超限）时填 `blocked`，作为可恢复状态点；每次打回/返工递增 `rounds` 中对应 CP 的槽位。
- 放宽需求声明的硬门槛（性能 / 精度标准未达成的收口）时，**先落盘 `pending_user_review` 条目再继续推进**；依据不全时不得收口，按阻断处理（见 [error-handling.md](error-handling.md)「需求级硬门槛放宽」）。
- 用户确认类 CP 派出 QA 后即填 `pending_questionnaire`（`sent`），QA 结论回传并处理完毕后清空。**未清空即表示该 CP 尚未收口，恢复时不得跳过**——防止把已产出的问卷误判为验收已通过。
- 可插拔流程插件可在 state.json 扩展自有字段（等待态等），字段语义与更新时机以对应插件文档为准。
- 恢复：读 `current_stage` + `blocked` 定位入口，`completed_stages` 决定跳过范围。

> 字段结构可由 `scripts/validate_state.py`（预留）校验：编号合法性、completed_stages 与 current_stage 的顺序一致性。

## 多算子识别与恢复

无顶层注册表。PM 会话开始（通过 `.cannbot/permissions/` 启动检查后）按以下规则识别已存在算子并选定本次推进对象：

1. 扫描 `.cannbot/*/state.json`，按 `operator` 字段列出所有已存在算子及其 `current_stage`。
2. 多个算子时，向用户确认本次推进哪一个；用户未指定且仅一个算子时默认选它。
3. 新算子：PM 在 `.cannbot/<算子名>/` 下创建子目录并初始化 `state.json`。初始化时先检查共享件 `.cannbot/环境信息.md`：
   - **不存在或环境已变**：`current_stage="0"`，`completed_stages=[]`——从阶段 0 跑起，采集并落盘环境信息。
   - **已存在且环境未变**：复用该文档，跳过阶段 0 与 CP0，`current_stage="1.1"`，`completed_stages=["0","CP0"]`。
