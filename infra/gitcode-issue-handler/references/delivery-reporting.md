# 交付：报告与完成条件

## 读取时机与产物

所有 Issue/分组到达终止或等待状态后、步骤 9 前完整读取。每次运行必须生成内容相同的：

- 历史：`.cannbot/gitcode-issue-handler/reports/<run_id>/summary.md`（不可覆盖其他 run）
- 最新：`.cannbot/gitcode-issue-handler/reports/latest.md`

即使无匹配、启动/同步失败、只有 `no_attention`、全部等待或发布失败也要生成，并说明结束原因。缺 Token 暂停时只用已有状态记录等待原因，不为报告继续 API、诊断或加载测试。顺序固定：

1. 按 `responsibility` 汇总：`handle` 进入逐项 `issues`/group；`list-only` 中按正常规则仍为 `need_attention` 的项规范化到顶层 `listed_issues`，其余只留聚合计数；`ignore` 不进入逐项报告；纯获取/分类只留聚合计数。
2. 回查回复及已批准执行的外部操作，记录授权、写入和 GET 结果；未批准只保留预览。
3. 按 `code-worktree.md` 安全清理并回写结果。
4. 未知 owner 按 `operator-owner-candidates.md` 逐算子给候选账号与贡献（每算子最多 5 人，整单可超过）；无候选写“待确认”，可留 `awaiting_offline_confirmation`，候选不等于已确认负责人/解决。临时指派成功时另写“已临时指派 @账号，请确认真正负责人”，保留逐算子候选，不冒充正式 owner。兜底指派按 `assignment_source: fallback_user` 标明“无候选、兜底接收人”，不将其加入候选表。
5. 按 [逐项响应材料](batch-analysis.md#逐项响应材料) 核对本轮每条 `need_attention` 的分析、适用草稿和 `response_artifacts`；关闭自动响应不免除落盘，受阻缺稿须有具体原因，未完成子任务须有恢复点。设置 `completed_at`/`overall_status`，写 `run_state.json`。
6. 在目标仓根执行：
   ```bash
   python "$ISSUE_HANDLER_SKILL_ROOT/scripts/generate_summary_report.py" \
     --state ".cannbot/gitcode-issue-handler/reports/<run_id>/run_state.json" --strict
   ```
7. 回读 `summary.md`：逐项只含 `handle`，数量等于 `issues_total`；`list-only` 只在“仅列举”中一行展示编号链接和 `responsibility_summary`，不展示候选、过程或外部操作；`ignore`、仅观察到的 `self_assigned` 和既有 `/assign` 不出现；本轮 response 阶段实际补分配的 `needs_pr_owner_handoff` 必须保留。失败先修状态重试，不用对话摘要替代报告，也不因单 Issue 失败跳过整批。
   同时核对每项结果和下一步是否具体、候选是否实际渲染；`--strict` 通过不代表内容合格，不能用全批相同的“已回复、维护侧承接”替代调查结论。
8. 写 `report_generated: true`、`report_path`；最终答复给简短结论和历史报告路径。

## 逐项契约与字段

`issues` 只保存本轮实际处理且新路由为 `responsibility: handle` 的 Issue。实际处理包括：进入 `need_attention` 后诊断/答疑/索要上下文/转交/等待落盘；处理新增评论或状态迁移回查；复现、根因、修改、测试、提交、推送、PR、CI；或发送非 `/assign` 有效评论并成功回查；也包括按本轮 response 计划准备、执行和回查 PR 作者分配，哪怕没有文字首响。仅获取/分类、发现已有负责人/PR/评论/跟进、仅观察到的 `self_assigned`、既有纯 `/assign` 都不算。本轮仅分配项保留处理前 `needs_pr_owner_handoff` 分类，记录 `handled_in_run: true`、分配结果、简短分析与稿件路径；自提豁免不计首响成功，分配不计问题解决。`list-only` 即使 `handled_in_run: true` 也移至 `listed_issues`；`ignore` 完全不入逐项。按 iid（缺失则 URL）去重；旧 `in_scope | out_of_scope | unknown` 字段兼容读取，不能删除历史。

`run.issues_scanned_total` 是扫描规模；`run.issues_total == len(issues)` 只计实际处理；`run.issues_listed_total == len(listed_issues)` 单列仅列举。其他无需处理最多留 category 聚合计数，不留标题、作者、assign、PR 逐项明细。`--strict` 对 `handle` 用完整契约，对 `list-only` 只校验 `iid`、`url`、非空 `responsibility_summary` 和 `responsibility: list-only`；缺摘要必须报错。

每个 `handle` 至少保留：iid、标题、URL、作者；`handling_status`、`resolution_status`、`result_summary`、`next_action`；实际动作和证据。实际等待/再回复才记录对话状态、双方最近 comment ID/时间和 follow-up SLA；有复现/变更/测试/发布/卡点才记录对应字段。`process_log` 格式：

```json
{"time":"2026-08-11T12:00:00+08:00","stage":"triage | diagnose | reproduce | implement | validate | authorize | deliver | comment","action":"执行了什么","result":"结果与状态变化","evidence":["命令、相对路径、commit/PR/Issue URL 或关键观测"]}
```

时间取不到写 `unknown`，不得删记录。创建未合入 PR、等待上下文、环境不匹配、转交和 CI 失败保持 `resolution_pending`/`unresolved`；不得为好看标 `resolved`。

## 对外摘要、指标和清理

生成器报告只含：运行概览（仓库、时间、总体状态、扫描/处理/列举数）；实际处理 Issue 的结果、状态、下一步和链接（根因、评论、变更、测试、PR 仅真实存在时追加）；仅列举一行摘要；有代码组才有变更与交付；仅被阻塞/部分完成或与实际 Issue 相关才有卡点/边界。禁止空章节、例行轨迹、空环境/测试/清理、`unknown` 表格和逐项 `no_attention`。Markdown 可含内部 blocker，禁止 Token、Authorization、密码等 secret；命令和产物优先相对路径，脚本仍会脱敏。

- 有效响应是维护者/责任人/授权助手的受理、判断、最小信息请求、明确转交或解决结论；系统消息和纯 `/assign` 不计。
- 解决时长是 `created_at` 到可核验 `resolved_at` 的自然日；解决率仅为本仓 `handle` 已解决/`handle` 总数。`list-only` 不计，`ignore` 不入集合；有证据的重复、非本仓、无效项可排除并注明理由。
- 未合入 PR、`delegated`、`waiting_*`、`intermittent_waiting`、`ci_blocked`、`comment_failed` 都不算解决；缺失数据用 `unknown` 或 `resolution_pending`，禁止推测。
- 内部卡点保留 `waiting_environment`（目标/当前 CANN、源码 revision、SoC、架构对比）、`ci_blocked`（PR、阶段、分类、轮次、建议）、权限/基础设施/工具缺失和验证边界，均不写外部评论。
- 报告列已清理 group/worktree；因 active、blocked、不干净或 manifest 失败而保留的项及下一步；不得 force 或直接删目录。清理不删分支、远程分支、commit、PR、证据或 manifest。附件结束后仅可删除本流程生成的 `.cannbot/gitcode-issue-handler/tmp/downloaded-attachments`。

完成条件：每个 `need_attention` 有真实状态、责任人或下一步；发布可核验；历史 summary、run_state、latest 均可读；`--strict` 成功；`issues`/`listed_issues` 数量与计数一致、覆盖所有 `handle`，且无 `ignore`、仅观察到的 `self_assigned`、既有纯 `/assign` 或观察到的 `no_attention` 明细。若实际处理为 0，终端和最终回复只写“本次未实际处理任何 Issue”及报告链接，不把扫描范围冒充整体指标。
