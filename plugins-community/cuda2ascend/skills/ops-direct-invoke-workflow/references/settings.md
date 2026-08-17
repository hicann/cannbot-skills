# 工作流配置（settings.json）

> `.cannbot/settings.json` 是工作流运行时配置的**唯一文件与唯一权威**：工作流模式、插件注册信息（挂载点/步骤/启用状态）、询问状态与元数据全部聚合于此。由 init Step 5.5 生成（`--mode` / `--plugin-enable` 参数写入），PM 在会话中可显式修改。本文件定义配置结构、语义与读取方式。

## 配置结构（v2）

```json
{
  "version": 2,
  "mode": "interactive",
  "surveyed": false,
  "plugins": {
    "plugin-experience-summary": {
      "hook": "before:7.1",
      "stages": ["plugin-experience-summary-1", "plugin-experience-summary-2", "plugin-experience-summary-3", "plugin-experience-summary-4"],
      "standalone": true,
      "enabled": true
    }
  },
  "updated_at": "2026-08-06T00:00:00Z"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `version` | int | 配置结构版本，当前为 2；结构演进时递增 |
| `mode` | string | 工作流模式：`interactive`（默认，交互式）\| `silent`（静默，完全无人值守） |
| `surveyed` | bool | 插件启用是否已询问过用户（`false` = 首次会话须逐项询问启用哪些插件，询问后置 `true`；新增插件时复位为 `false`） |
| `plugins` | object | 插件注册信息（`{"<插件名>": {"hook","stages","standalone","enabled"}}`），由 init 扫描 `skills/plugin-*/` frontmatter 生成；`enabled` 为启用状态唯一权威 |
| `updated_at` | string | 最近一次修改时间（UTC ISO 8601） |

## 读取方式（唯一权威）

- **工作流模式**：读 `mode` 字段；settings.json 缺失或字段非法时按 `interactive` 处理（不阻塞）。
- **插件环节启用**：读 `plugins.<name>.enabled`——`--plugin-enable` 与会话内调整都落在该字段；无插件机制时 `plugins` 为空对象。
- **首次询问**：`surveyed=false` 时，首次会话按 `plugins` 逐项询问用户启用哪些插件（on/off），询问结果写回各插件 `enabled` 并置 `surveyed=true`；非交互场景跳过询问并保持现状。

## 生成与修改

- **init 生成**（Step 5.5）：扫描 `skills/plugin-*/` 插件 frontmatter（同名插件以本地实现优先）的 `workflow-hook` / `workflow-stages` / `standalone` 生成 `plugins`——**每次重扫重写：保留各插件 `enabled`、并入新增（新增时顶层 `surveyed` 复位）、剔除失效插件**；`workflow-hook` 须通过格式与挂载点存在性（统一流程表）校验、`workflow-stages` 必填，不合法仅 warn 不注册。`--mode interactive|silent` 写入 `mode`（未传保留现有值，首次默认 `interactive`）。`--plugin-enable <name> on|off` 直接改对应插件 `enabled`（未注册仅 warn）。旧版 `.cannbot/plugin-registry.json` 存在时一次性迁移并入并删除，此后不再生成该文件。生成失败仅 warn 不 fail。
- **会话中显式修改**：用户直接指示「开启静默模式 / 关闭静默模式（进入交互模式）」时，PM 立即更新 `mode` 字段并刷新 `updated_at`，随后按新模式继续；不重启会话、不影响当前状态。
- **向后兼容**：字段只增不删；新增字段须提供默认值，旧 settings 缺失新字段时按默认值工作。

## 静默模式（mode=silent）

静默模式 = **完全无人值守**：工作流自动推进直到任务完成或遇阻断，期间不输出中间进度、不向用户询问。

### 静默下的行为

| 环节 | 静默行为 |
|------|----------|
| 中间进度 | 不向用户输出；todolist 与 `state.json` 照常更新（状态可观测不豁免） |
| ⛔ 用户确认点（CP0 / CP1 / CP2.2） | QA 不发送问卷，按下方「默认决策」执行，落盘 `.reply.json`（`{"mode":"silent","decision":"accepted"}`），保持状态机与中断恢复兼容 |
| 失败回退 | 按 [error-handling.md](error-handling.md) 自动回退至最大轮次；超上限 = 阻断 |
| 阻断 | 输出结构化问题清单与可恢复状态点（属中止性总结，静默下唯一允许的问题输出） |
| 插件内异步等待（如提交 PR 后的 CI 等待） | 属插件内部步骤（如 `plugin-pr-submit`）：照常落盘等待态结束会话、用户回传结果，其等待态告知归插件自身输出约定，不计入主工作流例外清单 |
| 需求级硬门槛放宽 | 按默认决策收口继续推进，同时落盘完整依据并标记 `pending_user_review`；不单独弹问卷，随任务完成总结上报 |
| 任务完成 | **必汇报**：输出完整总结——交付物清单、各 CP 结论、遗留问题，以及全部 `pending_user_review` 条目（逐条含决策内容与依据） |

### 静默下唯一允许的输出

1. **权限预检警告**（启动时，见 AGENTS.md「工作流配置」节）：opencode 检查工作区 opencode.json 未显式全量授权时输出一次提示；dsh 检查运行上下文声明的文件/审批策略未全量授权时输出一次提示。
2. **任务完成总结**（含阻断中止性总结）。

其余一切进度、结论、中间信息均不输出。

> **机制兜底**：`mode=silent` 时 permission-guard hook 在工具层拦截问卷发送（opencode `question`/`ask` / claude `AskUserQuestion`，按工具名子串匹配，任何角色都不得绕过）——「QA 不发送问卷」既是 prompt 约束，也有 hook 保证；`mode` 切回 `interactive` 后立即解除拦截（见 `workflow-agent-permissions` skill）。**dsh**：默认无项目级 hook、无机制兜底（仅 prompt 约束）；安装部署级守卫（`hooks/dsh/install.sh`，挂 `$DSH_HOME/cordis.patch.yml`）后，`tools/pre-execute` 门按同一语义拦截 `ask_user_question` 等问卷工具。**codex**：无 hook，仅 prompt 约束。

### 静默默认决策表（用户确认点替代）

| 确认点 | 默认决策 |
|--------|----------|
| CP0 环境确认 | 按环境信息文档通过（文档缺失或记录异常 = 阻断，交用户） |
| CP1 需求确认 | 需求文档核对无硬伤即通过；架构选型固定采用 SIMD，不评估 SIMT |
| CP2.2 方案检查 | QA 评审通过即视为用户确认（按方案文档决策执行） |
| 需求级硬门槛放宽（性能 / 精度标准未达成的收口） | 可收口继续推进，但**须落盘完整收口依据并标记 `pending_user_review`**（见 [state-schema.md](state-schema.md)），随任务完成总结逐条上报；依据不全时不得收口，按阻断处理 |

> **默认决策 ≠ 永久定案**：上表前三项是常规确认点的默认放行，静默下即视为已确认；最后一项性质不同——它放宽的是需求文档声明的硬门槛，属需求级决策，静默下只是"不阻塞地继续推进"，**必须以待复核状态留痕并上报**，由用户在下一次交互中裁定。放宽依据的构成见 [error-handling.md](error-handling.md)「需求级硬门槛放宽」。

### 与插件机制的关系

`plugins` 字段（注册信息 + `enabled`）由 init Step 5.5 扫描插件 frontmatter 生成，为插件启用判定唯一权威；插件内部步骤同样遵守静默行为（不询问、不输出，仅完成条件相关的阻断汇报）。插件含外部异步等待（如提交 PR 后的 CI 等待）时，落盘等待态结束会话、用户回传结果属流程必需，其等待态告知由插件自身约定承载（如 `plugin-pr-submit` 的 error-handling），不计入主工作流例外清单。

## 交互模式（mode=interactive，默认）

按工作流既有约定执行：⛔ 确认点由 QA 用会话问卷工具（opencode `question` / claude `AskUserQuestion` / dsh `ask_user_question`）直接发送用户并收集结论，PM 汇报中间进度。本模式为默认，settings.json 缺失时即此模式。
