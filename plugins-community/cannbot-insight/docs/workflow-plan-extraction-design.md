# 主 agent 编排声明提取：设计文档

> 当前实现在 `src/lib/sift-audit.ts` 的 `recoverWorkflowDeclaration`。
> root 对账（`--kind root`）审主 agent 编排规程遵循度，声明由本模块恢复。

## 背景

root 对账审"主 agent 编排 vs 声明"。按 sift 设计意图（`slice.py` 注释：
"root 声明常是 workflow 级 SKILL.md"），root 对账的声明应该是 **workflow skill 的编排规程**，
不是任务清单（STATE.md）或自写计划（todowrite）——那些是"任务完成度"视角，不是
"编排规程遵循度"。

三套对账都是"外部声明（规程）vs 执行段"模式：

| 对账 | 声明（规程） | 执行段 | 审什么 |
|---|---|---|---|
| skill | SKILL.md（功能规程） | skill 调用后的执行段 | 功能规程遵循度 |
| agent | agent .md（角色规程） | 子 agent 执行段 | 角色规程遵循度 |
| **root** | **workflow SKILL.md（编排规程）** | 主 agent 作用域 | **编排规程遵循度** |

### 旧方案的问题

旧方案用 `recoverPlanFileDeclaration`（D 混合方案）从计划文件（STATE.md/PLAN.md/…）+
todowrite + 内容特征 scan 提取声明。问题：

1. **STATE.md 是任务清单**（state-generator 生成的运行时任务），不是编排规程 → 审的是
   "任务完成度"不是"编排纪律"，偏离 sift 设计意图。
2. **todowrite 是主 agent 自写计划** → 自审自（声明=执行者自己写的），不是"外部声明 vs 执行"
   对账。
3. **PLAN.md 是产物**（主 agent"方案设计"步骤生成的迭代穿刺表格），从来不是 workflow 声明。

## 当前方案：workflow skill 编排规程

声明来源是 workflow skill 的编排规程，有两种加载方式：

### S2 型：主 agent Skill invoke

主 agent 调用 Skill 工具加载 workflow skill（如 `ops-registry-invoke-workflow`）。

```
1. resolveWorkflowSkillName
   查 SkillEvent where isSubagent=0 AND eventType=invoke, orderBy createdAt asc
   → 取首个 skillName = workflow skill name
   S2 → "ops-registry-invoke-workflow"

2a. recoverSkillResourceFile（优先级 1：具体编排模板）
   查 read ToolCall where argsJson 含 skillName
   → filePath 含 skillName 且含 "resources" 的 → 取最长 content
   S2 → task-prompts.md（每阶段 dispatch+prompt+验收标准）

2b. recoverSkillBody（优先级 2：编排规程骨架）
   找 invoke 的 <skill_content> → stripSkillContent 剥外壳
   S2 → SKILL.md body（核心原则/职责边界/阶段推进）
```

### S1 型：turn0 注入（无 Skill invoke）

workflow skill 正文注入首条 user turn（≥500 字），主 agent 不 invoke skill。

```
3. recoverMainAgentWorkflowBody（fallback）
   查 Turn where role=user, isSubagent=0, orderBy turnIndex asc
   → content.length ≥ 500 ? content : null
   S1 → turn0（1809 字编排规程："你是纯编排者，读取 STATE.md…"）
```

### resources/ 目录过滤

`recoverSkillResourceFile` 只匹配 filePath 含 `"resources"` 的文件（编排模板在 `resources/`
目录），排除 `templates/` 目录的文档模板（STATE.md.templ / DESIGN.md.templ）。

- S2：`.../ops-registry-invoke-workflow/resources/task-prompts.md` → 命中 ✓
- S1：`.../ops-registry-invoke-glacier/templates/STATE.md.templ` → 排除 ✓（退 SKILL.md body）

## 优先级

```
1. skill resources 文件（task-prompts.md 等，具体编排模板）
   ← 外部声明 + 具体（每阶段 dispatch+验收标准）→ 最好审
   ← source = 文件名（如 "task-prompts.md"）

2. SKILL.md body（编排规程骨架）
   ← 外部声明 + 抽象（核心原则/职责边界/阶段推进）
   ← recoverSkillBody 从 invoke 的 <skill_content> 恢复
   ← source = "skillName (SKILL.md)"

3. turn0 注入（编排规程，S1 型 fallback）
   ← 外部声明 + 注入（主 agent 不 invoke skill）
   ← recoverMainAgentWorkflowBody 取 turn0 ≥500 字
   ← source = "turn0 (注入编排规程)"

4. null（无任何编排规程 → 不显 root 目标）
```

**STATE.md / todowrite 不再用于 root 对账**——它们是"任务完成度"视角，不是"编排规程遵循度"。
旧函数 `recoverPlanFileDeclaration` + `recoverStateMdPlan` 保留（向后兼容），但不再被
root 路由调用。

## API

```ts
recoverWorkflowDeclaration(sessionId, prisma): Promise<PlanFileDeclaration | null>

interface PlanFileDeclaration {
  content: string;   // 编排规程正文（skill resources 剥行号 / SKILL.md body / turn0）
  source: string;    // 来源标识，供 UI 显示
}

// source 格式：
// "task-prompts.md"                     — skill resources 文件
// "ops-registry-invoke-workflow (SKILL.md)" — SKILL.md body
// "turn0 (注入编排规程)"                  — turn0 注入
```

辅助函数：
- `resolveWorkflowSkillName(sessionId, prisma)` — 主 agent Skill invoke → skillName
- `recoverSkillResourceFile(sessionId, skillName, prisma)` — resources/ 文件恢复（内部）
- `recoverMainAgentWorkflowBody(sessionId, prisma)` — turn0 ≥500（已有，复用）
- `recoverSkillBody(sessionId, skillName, prisma)` — SKILL.md body（已有，复用）

## 调用点

| 文件 | 用途 |
|---|---|
| `audit-skillsift/route.ts` | kind=root → `recoverWorkflowDeclaration` → body |
| `main-agent-workflow/route.ts` | available + name + source（`resolveWorkflowSkillName` 取 name） |
| `skill-content/route.ts` | 全文展示（MAIN_AGENT_WORKFLOW_NAME sentinel） |
| `SkillDetail.tsx` | root 行来源标签 + 全文按钮 + 对账按钮 |
| `page.tsx` | fetch main-agent-workflow 端点 → state → 传 SkillDetail |

## Skills tab subagent 行

Skills tab 不仅展示 invoke 的 skill，还展示 dispatch 的 subagent 行（恢复之前被误删的功能）：

| 行类型 | 全文来源 | 对账 kind |
|---|---|---|
| invoke skill | `fetchOne`（session 的 `<skill_content>`） | skill |
| dispatch subagent | `fetchAgent`（磁盘扫 agent .md，`agent-content` 端点） | agent |
| dual-nature（invoke+dispatch） | `fetchOne`（session） | skill + agent |
| 主 agent 编排 | `fetchOne`（workflow 编排规程） | root |

`agent-content` GET 端点用 `resolveAgentMd`（agent-md-scan）从磁盘扫 agent .md，
多插件按 session dispatch 覆盖率消歧。

## 三个 session 的实测

| session | Skill invoke | resources 文件 | turn0 | 命中 | source |
|---|---|---|---|---|---|
| ses_0515（S1） | `ops-registry-invoke-glacier` | 无 resources（templates 排除） | 337 字（短用户任务） | SKILL.md body | `ops-registry-invoke-glacier (SKILL.md)` |
| ses_076ca（S2） | `ops-registry-invoke-workflow` | task-prompts.md | 329 字（短） | resources | `task-prompts.md` |
| ses_0751（S1 旧） | 无 invoke | — | 1809 字（注入编排规程） | turn0 | `turn0 (注入编排规程)` |

## 边界

| 场景 | 预期 |
|---|---|
| 主 agent Skill invoke + resources 文件 | 优先 resources（task-prompts.md） |
| 主 agent Skill invoke + 无 resources | 退 SKILL.md body |
| 主 agent Skill invoke + templates 文件（非 resources） | 排除 templates → 退 SKILL.md body |
| 无 Skill invoke + turn0 ≥500 | 用 turn0（注入编排规程） |
| 无 Skill invoke + turn0 <500 | null（短用户任务不是编排规程） |
| 无 Skill invoke + 无 turn0 | null |
| resources 内容过短（<100 字） | 退 SKILL.md body |
| SKILL.md body 过短（<100 字） | 退 turn0 或 null |

## 范围外

- **STATE.md / todowrite 任务完成度审计**：旧 `recoverPlanFileDeclaration` 保留，将来
  可作为"任务完成度"对账目标的声明（与 root 编排规程对账不同的审计视角）。
- **skill-md-scan 磁盘反查**：旧方案用 turn0 body 反查磁盘 SKILL.md frontmatter name
  做"真名"显示。新方案用 `resolveWorkflowSkillName`（Skill invoke）直接取 skillName，
  更精准。`skill-md-scan` 仍用于 agent .md 扫描。
