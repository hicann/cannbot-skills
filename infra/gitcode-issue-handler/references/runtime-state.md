# 运行时：最小授权与状态契约

## 读取时机

真实执行请求在步骤 -1 初始化唯一运行状态时读取本文件；`policy_query` 不建立状态，也不读取执行阶段 reference。完整字段、枚举和兼容字段见 [runtime-state-schema.md](runtime-state-schema.md)，只有当前阶段需要的字段才按需读取。

## 初始化

- `single`、`batch` 均从 `authorization_mode: interactive` 开始。用户说“处理 Issue”“自动执行”不产生批次交付批准；首响、关联与指派可按 [automation.md](automation.md) 的配置及会话授权执行。
- 创建一个 `run_id`，记录 `mode`、时间、`overall_status: running`、`capability_checks`、目标仓库和后续阶段待填字段；不为补齐 schema 执行环境探测；不得用虚假值补齐条件字段。
- 真实操作按阶段更新同一状态，不另建平行口径；条件字段不适用时省略。每次分类、诊断、复现、修改、验证、授权、发布、回查或状态转换后立即写入结果和证据。
- 各阶段只更新自己负责的字段。未进入的能力保持 `not_started`，终态可记 `not_required`；能力状态仅在对应真实操作紧前更新，能力就绪不等于业务写入授权。外部写入必须记录稳定 `operation_id`、授权证据、结果和回查证据。

## 授权与回复

授权模型、回复检查点、批次批准失效条件和 direct-push 独立确认统一以 [authorization-contract.md](authorization-contract.md) 为准；本文件不重复其操作表。

`response_status` 独立于首响 SLA 和解决状态。新增追问会使本轮响应重新待处理，旧评论不能自动通过门禁；`verified/reused` 必须有本轮适用的 GET 证据，`waived_by_user` 留用户原话。`exempt_self_authored_pr` 只用于已核验同作者 PR 的免首响路径，不计首响成功；由 assignee 与作者同账号确立的 `self_assigned/no_attention` 仅保留分类证据，不伪造评论结果或本轮处理记录。历史失效自提且已有负责人时保留豁免及未闭环状态。
每个后续 operation 的 `depends_on` 引用该回复或适用的历史/豁免证据。

## 能力失败与恢复

能力检查的选择、顺序和失败路由以 [runtime-capability-checks.md](runtime-capability-checks.md) 为准。一般失败只阻断依赖该能力的操作；缺少 Token 且后续已确定需要认证写操作时，按以下契约暂停整轮：

1. 保存元数据，将 `overall_status` 和 `capability_checks.api` 设为 `waiting_for_input`，创建或复用唯一 `pending_user_inputs` 项 `input_id: gitcode_token`，记录 `request_count: 1` 与准确 `resume_from`，不保存 Token 明文。
2. 只询问一次并停止 API、Issue 拉取、知识刷新、代码诊断和测试读取。未解决项存在时不重复询问或重新初始化。
3. 用户补充 Token 后标记输入 `resolved`，重跑 `api` 检查；通过后恢复 `running`/`ready`，从 `resume_from` 继续，不重复已完成阶段。

## 状态骨架

初始化至少包含 `run`、`issues`、`groups`、`external_operations`、`internal_blockers`；按需填充 schema 中字段。实际处理项在 `issues` 中标记 `handled_in_run`，仅列举项放入顶层 `listed_issues`，报告生成不得丢失后者。
