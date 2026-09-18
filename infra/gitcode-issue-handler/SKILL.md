---
name: gitcode-issue-handler
description: >-
  处理 GitCode 单个或批量 Issue：分诊、首响答疑、算子责任人转交、再次回复跟踪、
  环境核对、复现修复、PR 交付和结果报告。支持仅回复、不改代码，以及已答复咨询的自动闭环。
  触发：用户要求处理、回复、跟进仓库 Issue，从 Issue 修复并创建 PR，或初始化、调整本 Skill 的项目配置。
license: CANN-2.0
---

# GitCode Issue Handler

## 入口与边界

- **步骤 -1：区分 policy_query 与真实执行**。只询问规则时直接回答，不建运行树、不检查 Token/Git/CANN、不访问 API。实际处理或配置时才加载下表对应路径。
- `configure`：用户明确要求初始化或调整项目配置，按 [configuration-setup.md](references/configuration-setup.md) 补齐文件并引导常用设置，不拉取或处理 Issue；首次真实使用也检查配置引导状态，不以目录存在代替用户已配置。
- `single`：显式 Issue URL，只处理该项；不受批量时间窗或原 `no_attention` 限制，但仍先过滤核心 closed，再核查责任范围、已有解决证据和重复动作。
- `batch`：当前目标仓库批量分诊，只推进核心 open 且责任范围内的 `need_attention`。
- 批量响应先判断内容性质：纯路线图/规划汇总不进入响应。现任负责人或任一有效关联 PR 作者与 Issue 作者同账号，即按自提处理；需求与缺陷使用相同判定。已有实质回复和责任人、无新跟进时也不纳入本轮，不为历史状态/watch 补录重新激活。
- 正常 `single`/`batch` 不跟进 closed、不自动 reopen；获取元数据后先过滤，再做评论、PR 和责任调查。详细入口规则以 [issue-intake.md](references/issue-intake.md) 为准。
- `auto-close-stale`：用户明确要求已答复咨询闭环时使用独立维护路径。
- 仅回复/答疑时做到有证据的回应与回查，不进入代码修复或交付阶段；明确“只回复”也不隐含指派或状态变更授权。
- 规则冲突或遇到尚未明确的边界时，先向用户问清再执行依赖该结论的动作，不按默认假设决定；已明确的独立事项可继续。

## 跨阶段规则

1. 目标与范围先确定。配置 `repo` 非空直接使用；为空时推导并保存，歧义或显式目标冲突用问卷确认。`responsibility` 的 `handle` 正常处理、`list-only` 仅在按正常规则仍需关注时列举一行、`ignore` 忽略；未核实的 `pending` 不执行外部写入。950/A5 必须核查实际 arch35 故障实现，不能只凭报告硬件判断。“全部 Issue”指扫描全集，不覆盖责任配置；首次分类的 pending 不是发送清单，须核查后重跑分类器。详见初始化与 intake。
2. 首响先于转交和修复。自提 Issue 只补必要指派，不生成首次文字响应；已有负责人不重复分配。自提关系为“assignee 与 Issue 作者同账号”或“任一有效关联 PR 与 Issue 同作者”，任一条件满足即可；非自提无实质回复时补首响。历史自提且负责人已落实的失效 PR 项保留豁免和未闭环跟踪。已回复不重复，新追问走 follow-up。对外回复聚焦问题与处理方向；已有显式关联 PR 时概述其方案和状态，审查发现的测试缺口、实现不足留内部。没有显式关联时不搜索或展示潜在修复 PR，给出有依据的潜在方案即可。
   首响/新追问统一用 `post_triage_comment.py` 校验分类、发送并 GET 回查；有效修复 PR 的自动关联及临时指派 PR 作者按 [automation.md](references/automation.md) 执行；已有 PR 的分配属于 response 阶段和 auto-response，自提作者优先，否则按问题覆盖面选择，不受 auto-assign 限制。每条 `need_attention` 先按 [逐项响应材料](references/batch-analysis.md#逐项响应材料) 保存简短分析和适用的回复/分配稿；关闭自动响应或受阻也须落盘已知分析并说明未形成草稿的原因。
3. 算子问题默认转交已确认 owner，未知时不能自行修复。仍有维护动作且无已确认 owner/有效 assignee、也无有效关联 PR 时，起草首响并按涉及的每个算子分析核心贡献候选，每算子最多 5 人；summary 只写可指派账号及简述，供用户线下确认。
   候选不等于 owner，不公开候选列表 @，不延迟首响；用户对当前 Issue 明确 `direct` 才自行处理。
   纯答疑已完整解决且无剩余维护动作时不额外收集 owner。
4. 首响、安全 PR 关联与临时指派按 [automation.md](references/automation.md) 的两开关执行，默认均关闭；开启配置与当前处理请求共同授权对应操作，明确会话指令可单次覆盖。复用准确授权，不重复询问；Token、能力检查不是授权。
   `single`/`batch` 从 `interactive` 开始；回复与交付分开确认，只有精确批次交付批准才进入 `approved_batch`。direct push 在 commit 后按确切目标和 SHA 单独确认。
5. 确定需要认证写操作而缺 Token 时，只汇总询问一次并停止本轮，保存恢复点；等待期间不继续 API 探测、真实 Issue 拉取或测试框架读取。其他能力在首个依赖操作前检查，失败只阻断依赖它的路径；详见 `runtime-capability-checks.md`。
6. 修改前核对 CANN/源码/SoC，稳定复现并确认根因；只在 manifest 管理的独立 worktree 做最小修复，执行相关测试和 NPU 门禁，缺失验证如实记录。不覆盖用户工作区、不自动合并 PR，不用破坏性 Git 恢复/force push/强删 worktree，也不 `git add -A` 或 `git add .`。
7. 每个阶段把结论和证据追加到唯一运行状态，不凭记忆重建。评论/指派/状态回查失败不记成功；停止依赖该操作的后续动作，继续其他独立 Issue。外部回复不披露维护侧环境、CI/权限故障、内部重试或敏感信息。转交、等待、PR 未合入和 CI 阻塞不算解决。
8. 范围待核查时、分类后准备响应时，先按 [batch-analysis.md](references/batch-analysis.md) 检查分工；两个以上互不依赖且需实质调查或拟稿的任务组默认委派，不等全部 `pending` 核查完。轻量任务优先用轻量模型子 agent；无法选模型仍用独立子 agent，无此能力或单步即可完成时顺序执行并简记原因。
   传最小相关材料、隔离输出，由主会话审核和写入，就绪项不等全批。复杂问题升级分析，不给回复设固定字数上限；不清理共享运行根目录。

## Reference 读取路由

按当前动作加载对应规则真源，已读且未变化的不重读。无新输入、失败恢复或必要回查时，按检查点继续，不轮询或重扫整批；条件文档不在启动时全量加载。

| 当前动作 | 读取内容 |
| --- | --- |
| 首次配置、主动调整常用参数 | [configuration-setup.md](references/configuration-setup.md)，仅本地初始化与引导 |
| 初始化、目标选择 | [runtime-setup.md](references/runtime-setup.md)、[runtime-state.md](references/runtime-state.md)；schema 仅在补对应字段时读取 |
| 首响自动发送、候选临时指派或人工审核 | [automation.md](references/automation.md)；首次真实处理读取 |
| 首个 API/Git/tmp/author 操作或能力故障 | [runtime-capability-checks.md](references/runtime-capability-checks.md) 中相应检查点 |
| 首次知识检索 | [runtime-knowledge.md](references/runtime-knowledge.md)：复用已有快照和受审卡 |
| 显式知识维护 | [knowledge-maintenance.md](references/knowledge-maintenance.md)；普通首响不读 |
| 获取、分类排序 | [issue-intake.md](references/issue-intake.md)；分类字段表仅在解释分类结果时读取 |
| 核查责任范围 | [responsibility-scope.md](references/responsibility-scope.md)；只调查范围时不读完整 intake |
| 多项范围待核查、分类后调查与拟稿 | [batch-analysis.md](references/batch-analysis.md)；调查开始前检查分工，协调者统一合并和发布 |
| 文字诊断与处置选择 | [issue-routing.md](references/issue-routing.md)；责任人转交细节仅在需要转交时读取 |
| 草拟/审核正文 | [response-writing.md](references/response-writing.md)；正式评分或质量不足时才展开评分表 |
| 复用/发送回复及后续状态 | [issue-comment-workflow.md](references/issue-comment-workflow.md)；只拟稿的子 agent 不读 |
| 准备或复用外部写授权 | [authorization-contract.md](references/authorization-contract.md)；缺授权或形成交付时读 [delivery-confirmation.md](references/delivery-confirmation.md) 对应检查点 |
| 具体算子候选责任人 | [operator-owner-candidates.md](references/operator-owner-candidates.md) |
| 等待、再次回复、恢复状态 | [issue-followup.md](references/issue-followup.md) |
| 代码修复分组/并行/worktree | [code-worktree.md](references/code-worktree.md) |
| 复现与根因；需要追溯历史 | [code-root-cause.md](references/code-root-cause.md)；按需 [code-git-history.md](references/code-git-history.md) |
| 实施与验证 | [code-validation.md](references/code-validation.md) |
| 提交/PR/CI/发布 | [delivery-publish.md](references/delivery-publish.md) |
| 阶段故障、等待或降级 | [policy-error-handling.md](references/policy-error-handling.md) 对应条目；正常路径不加载整个错误目录 |
| 每轮报告、终态清理 | [delivery-reporting.md](references/delivery-reporting.md) |
| 自动咨询闭环 | [maintenance-stale-close.md](references/maintenance-stale-close.md)，默认 dry-run，不继承批次授权 |

GitCode API、Token、通用 POST/GET 和 PR 接口以同级 `gitcode-toolkit` 为真源；确定性操作调用已有脚本，不先读其全部源码。首次安装或依赖异常才读 [安装指南](docs/installation-guide.md)。脚本参数以 `--help` 为准。

## 工作顺序与完成条件

目标/能力就绪 → 获取与范围复核 → 分类和文字诊断 → 本轮回复及回查 → 转交、等待或代码组 → 代码组复现/根因/最小修复/验证 → 精确交付与回查 → 报告和清理。
仅执行当前任务需要的分支；分组冲突或独占资源不得并行。

每项实际处理有真实终态或明确等待对象与下一步，每条 `need_attention` 的分析及适用草稿可读、缺稿有原因，操作有授权及回查证据，运行状态完整，受管 worktree 安全处置后，用 `generate_summary_report.py --strict` 生成历史报告并回读。
实际处理与仅列举分开计数；不为纯扫描项目补大段“不适用”。提供可点击报告链接。
解决率目标 >90%、平均解决时长 <7 个自然日、1 个工作日内有效响应；指标只据真实证据，不把转交、等待或未合入 PR 计为解决。

## Skill 修改验证

运行 `python3 -m pytest -q infra/gitcode-issue-handler/tests infra/gitcode-toolkit/tests`、 `python3 tests/lib/skill_validator.py validate-skill infra/gitcode-issue-handler/SKILL.md`。
环境门禁变更需额外覆盖延迟预检、缺 Token 恢复及 toolkit 兼容；无 NPU 不声称上板通过。
