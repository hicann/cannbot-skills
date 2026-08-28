# Audit 板块改造：skill audit 子 tab（设计方案）

> 本文档取代 `docs/sift-audit-design.md`（独立 Sift tab 方案，已废弃）。

## 背景与决策

cannbot-insight 已接入 sift 的 `audit` 能力（patch），目前入口散在 Skills tab（skill 对账按钮）+ Subagents tab（agent 对账按钮）。经讨论确认：

1. **去掉 agent 对账**：agent .md 不在 session 里，导入 session 多数 404，且要配 `AGENTS_ROOT`，净负担。sift 原生支持，lib 里保留 agent 分支休眠，将来 co-located 部署可复苏。
2. **skill 对账收进 Audit 板块**：不再散在 Skills tab。Audit tab 成为"所有审计"的家（workflow outcome + skill process）。
3. **Audit tab 拆两个子 tab**：当前 Audit 内容已过多（v1-v4 版本切换 + 生成/导入/导出 + provider 配置 + 进度 + renderer），再加 skill 对账会更乱。拆成 `Workflow audit`（现有 v1-v4）+ `Skill audit`（新）两个子 tab，各管各的。
4. **LLM 后端不变**：skill audit 仍用 sift 自带 `claude -p`，不透传 provider，不改 sift。

## 范围

**本期**：Audit tab 拆 sub-tab；skill audit 子 tab 实现 sift 对账（target 列表 + 跑 + iframe 报告 + sessionStorage 持久化）；移除 Skills/Subagents tab 的散落对账入口；移除 agent 对账端点。

**二期（不在本文档）**：skill audit findings 的 `#turn` → workflow audit v4 agent 树归属层（两层同 tab 才能做跳转，本期铺路）；in-place findings 薄 React renderer（替代 iframe）；流式进度；agent 对账复苏（条件成立时）。

## Audit tab 新结构

```
Audit tab（page.tsx key="workflowAnalyse"，label 不变）
└─ <AuditBoardTab taskId framework skillEvents onJumpToTurn />   （新容器）
    ├─ 子 tab bar：[Workflow audit]  [Skill audit]
    ├─ Workflow audit 子 tab：渲染现有 <WorkflowAnalyseTab .../>（不动，原样嵌进来）
    └─ Skill audit 子 tab：渲染新 <SkillAuditTab taskId framework skillEvents />
```

- 子 tab 切换不持久化（默认 Workflow；useState 即可）。WorkflowAnalyseTab 自身的状态（audit-job 模块级 + localStorage）已能跨 unmount 恢复，嵌进 sub-tab 不破坏其 resume。
- 主 tab bar 不变（"Audit" 仍是一个主 tab）。

## Skill audit 子 tab 布局

```
Skill audit 子 tab（SkillAuditTab）
├─ 信息 banner：本子 tab 用 sift 自带 LLM（claude -p），需装 sift + Claude Code CLI；
│              与 Workflow audit 的 OpenAI-compatible provider 无关。
├─ 左：Auditable skills 列表（来自 skillEvents，只列有 invoke/use 事件的 skill）
│   每行：skillName + [skill] 徽章 + 事件数 + 已审 summary 徽章（sessionStorage 命中）
│   源数据：page.tsx 已加载的 allSkillEvents，group by skillName，过滤 eventType∈{invoke,use}
└─ 右：选中 skill 的对账结果（状态机）
    ├─ idle（未选）：提示选 skill
    ├─ loading：POST in-flight（getInflightAudit 去重，key=`skill:taskId:skillName`）
    ├─ loaded：summary 徽章（total/pass/fail/na/unresolved/indeterminate）+ warnings
    │        + iframe 内嵌 audit-report.html（blob URL）+ [在新页打开 ↗] + [重跑]
    └─ error：中文错误（sift 未装 / 正文恢复不到 / 切片为空 等）
```

## 数据流

```
SkillAuditTab 选 skill + 点对账
  → POST /api/ai/audit-skillsift { taskId, skillName, framework }
  → 路由（已有，不改）：
      recoverSkillBody（session invoke ToolCall.resultJson 剥 <skill_content>）
      → 物化临时 SKILL.md（注入 frontmatter name=skillName）
      → framework=opencode → --db sourcePath；其余 → --transcript buildStructuredRecords
        （parent_tool_use_id=subagentSessionId，不带 resultJson）
      → execFileSync sift audit <dir> --kind skill -o <tmp>
      → 读 audit-report.json + .html → 回传（_html）
  → SkillAuditTab 右面板：summary + iframe(_html) + 新页链接
  → sessionStorage 持久化：`skill-audit-${taskId}-${skillName}` = {summary, warnings, _html?}
```

复用现有 `src/lib/sift-audit.ts`（bridge lib：recoverSkillBody / buildStructuredRecords / buildAuditArgs / buildTranscriptArgs / getInflightAudit）+ 现有 `src/app/api/ai/audit-skillsift/route.ts`（端点不改）。本期只新增 UI 容器 + 子 tab 内容组件 + 接线，不动后端。

## 文件清单

### 新增
- `src/components/observe/AuditBoardTab.tsx` — Audit tab 容器：子 tab 切换器（workflow | skill）+ 透传 props。轻量（~40 行）
- `src/components/observe/SkillAuditTab.tsx` — skill audit 子 tab 内容：左 target 列表 + 右结果面板 + sessionStorage 持久化 + iframe 内嵌 + inflight 去重

### 改
- `src/app/session/[taskId]/page.tsx` — `renderWorkflowAnalyse()` → `renderAudit()`，返回 `<AuditBoardTab taskId framework skillEvents={allSkillEvents} onJumpToTurn />`；import 换。Audit tab 主 tab 项不动
- `src/components/observe/SkillDetail.tsx` — **移除对账列**（按钮列 + 列头 + audit state + SiftAuditDialog 引用 + colSpan 14→13 回退）。Skills tab 不再做对账入口
- `src/components/observe/SubagentCards.tsx` — **移除 agent 对账**（taskId prop + onAuditAgent + 按钮 + dialog + useState import 回退）

### 删
- `src/app/api/ai/audit-agentsift/route.ts` + `tests/audit-agentsift-api.test.ts` — agent 对账端点及测试（agent 不做）
- `src/components/observe/SiftAuditDialog.tsx` — patch 的 dialog 组件，被 SkillAuditTab 内联面板取代

### 保留（不动）
- `src/lib/sift-audit.ts` — bridge lib（agent 分支休眠：auditKindsForEvents 的 agent 路径保留，无 UI 触发）
- `src/app/api/ai/audit-skillsift/route.ts` — skill 对账端点
- `tests/sift-audit.test.ts` — bridge lib 测试（纯函数，agent 用例保留无妨）
- `tests/audit-skillsift-api.test.ts` — 端点测试

### 小改
- `src/lib/version.ts` — +0.01
- `AGENTS.md` — 删 AGENTS_ROOT 段（agent 不做）；记 sift + claude CLI 依赖（Skill audit 子 tab）
- `README.md` / `README-zh.md` — Audit tab 描述补"含 workflow audit + skill audit 两个子 tab"

## 关键设计点

### 1. provider 分离
Workflow audit 子 tab 用 OpenAI-compatible provider（v4 LLM），显示 `AIProviderConfigPanel`。Skill audit 子 tab 用 sift 自带 `claude -p`，**不显示 provider 配置**，只显示信息 banner（"需装 sift + Claude Code CLI"）。两子 tab 的 LLM 后端独立、配置独立，不串。

### 2. 结果持久化
sessionStorage `skill-audit-${taskId}-${skillName}` = `{summary, warnings, _html?}`。跨子 tab 切换（workflow↔skill）+ 跨主 tab 切换（Audit↔Turns）都不丢。2MB size guard：`_html` 超大则降级只存 summary + warnings，iframe 区显示"重跑以查看完整报告"。不进 localStorage（不占额度，重启失无所谓——audit 可重跑）。

### 3. inflight 去重
复用 `getInflightAudit`，key=`skill:taskId:skillName`。dev HMR remount / 快速双击不堆叠不可中止的 execFileSync POST。

### 4. 报告呈现
iframe 内嵌 audit-report.html（blob URL）+ "在新页打开 ↗" 兜底（不可缩放时逃生）。不自建 React renderer——sift 的 HTML 已含五态 findings 表/聚合/noise 徽章/related 关联，重写是浪费；二期 in-place renderer 再说。`#turn` 跳回 cannbot-insight turn 本期做不了（HTML 在 iframe 里、无 hook），二期 React renderer 才能做。

### 5. target 列表来源
直接用 page.tsx 已加载的 `allSkillEvents`（内存数据，不查 DB），group by skillName，过滤 `eventType∈{invoke,use}`。与 Skills tab 一致的数据源。无 invoke 事件的 skill（纯 load / 纯 dispatch）不出现在列表（无可恢复正文）。

### 6. 跨子 tab 状态
子 tab 用 useState（默认 Workflow audit）。SkillAuditTab 内部 selected skill 用 useState；切走再回，selected 重置为空但 sessionStorage 结果仍在（列表行带"已审"徽章 + 点开即恢复 summary）。不持久化子 tab 选择（不增加复杂度）。

## 验收标准

1. session 详情页主 tab "Audit" 不变；点进去顶部出现子 tab bar：[Workflow audit] [Skill audit]。
2. 默认进 Workflow audit 子 tab = 原 Audit 内容（v1-v4 版本切换/生成/导入/导出/provider 配置/renderer），行为与改造前完全一致。
3. 切到 Skill audit 子 tab：左列出本 session 所有有 invoke/use 事件的 skill（名称 + [skill] 徽章 + 事件数）；无 skill 显示空状态。
4. 选 skill → 右面板"对账"按钮 → loading → 跑完显示 summary 徽章 + warnings + iframe 内嵌完整 HTML + "在新页打开 ↗" + "重跑"。
5. 跨主 tab 切换（Audit↔Turns）+ 跨子 tab 切换（workflow↔skill）再回 Skill audit，已审 skill 的 summary + HTML 仍在（sessionStorage）。
6. dev remount 不堆叠请求（inflight 去重）。
7. 未装 sift / 正文恢复不到 / 切片为空 → 各自清晰中文错误提示。
8. Skills tab 不再有"对账"列；Subagents tab 不再有 agent 对账按钮。
9. `npm run test` + `npm run test:cli` 全绿；`npm run lint` 通过。

## 不在本期范围（二期）

- **findings → workflow v4 agent 树归属层**：skill audit findings 的 `#turn` evidence → `buildAgentIO` 的 `ownerAgentId` → workflow audit v4 树的 quality 单元格。两层同在 Audit tab 才能做"点 finding 跳到 v4 对应 agent"。本期把 skill audit 收进 Audit tab 是为这步铺路，但归属层本身二期做。
- **in-place findings 薄 React renderer**：替代 iframe，板内看非-PASS findings + 跳 turn 按钮（替代 iframe）。
- **agent 对账复苏**：当部署机有 `.opencode/agents/` 且需审被 dispatch 子 agent 时，lib 的 agent 分支已休眠待命，UI 重新接通即可。
- **流式进度**：sift CLI 进度转 NDJSON（要 spawn + 解析 rich 输出，或引 sift 当 lib）。
- **sift 的 OpenAI-compatible 后端**：一期跑通后再评估是否给 sift 加 adapter（统一 LLM 后端）。

## 依赖

- 外部：`sift` CLI（在 PATH）、`claude` CLI（Claude Code，sift 的 LLM 后端）、可选 `opencode` CLI（仅 framework=opencode 的 --db 路）。
- env：无（agent 不做后，AGENTS_ROOT 不再需要）。
