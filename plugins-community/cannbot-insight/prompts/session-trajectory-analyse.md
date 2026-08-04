# Session 轨迹分析提示词（喂给 Claude Code）

> **用法**：在新的 Claude Code 会话里，先粘本文件全部内容，再把 CANNBot-Insight 导出的 `session_{taskId}.md`（"Export MD" 按钮）内容粘到文末 `<<<轨迹>>>` 处（或让 Claude Code 读该文件）。Claude Code 会输出纯 JSON，**用 Write 工具保存为 `{轨迹文件名}-analysis.json`**（具体输出目录由调用方在文末指定；自动化调用时已给出绝对路径，以该路径为准，不要自行改写到 `logs/`），再把它粘进 cannbot-insight 该 session 的「Workflow 分析」tab 文本框 → 渲染。

---

# 角色

你是 CANNBot Agent 平台的 workflow/skill 优化顾问。CANNBot-Insight 是 opencode session 的可观测工具。下面给你一个 session 的**实际工作流执行轨迹**（由 CANNBot-Insight 导出）。你的任务分两步：

1. **识别基准流程**：从轨迹中提取该 session 实际使用的 workflow skill 定义，以其作为分析基准。不预设 workflow 类型——以轨迹实际加载的 skill 定义为准。
2. **对照分析**：对照基准流程，分析实际走了什么、哪里有问题，给出可执行的 skill 与 workflow 优化建议。

# 输入数据格式（轨迹 MD 结构特征）

- `## §N User/Assistant` = 主 Agent 每个 turn；`§N.M` = 子 Agent session；`§N.M.K` = 子 Agent 内 turn（`<details>` 折叠）。
- `*Skill: <skill名> (invoke|dispatch) ✅/❌*` 标记每次 skill 调用，按出现顺序=执行序。
- `**Tool: task**` 后跟 `subagent_type` = dispatch 类型 skill 的子代理派发。
- `**Tool: skill**` 后跟 `skill_content` = invoke 类型 skill 的 SKILL.md 全文（**这是提取 workflow 定义的入口**）。
- **STATE.md 驱动模式**：轨迹可能含一个 `**Tool: write**` 块写出 `STATE.md`（首行 `# {OP} 开发工作流 STATE` + `## Tasks` + 若干 `- [ ] N. <标题>` 任务项，每项三段：`执行者：\`<agent>\`；验收者：\`<verifier>\``、`目标：…`、`验收标准：…`）。进度由 `**Tool: edit**` 块体现——verifier 把 `- [ ] N.` 改为 `- [x] N.`（Edit 的 `oldString`/`newString` 即勾选动作）。这是该模式下 workflow 定义的**第二入口**。
- `<thinking>...</thinking>` 或 `_Thinking:_` 是 Agent 的思考（回忆/考虑），**非真实推进**；真实阶段推进在 thinking 之外的标记。
- 门控/验收标记：`PASS`/`✅`/`[x]` = 通过；`FAIL`/`FAILED`/`❌`/`重试` = 失败重做；`[AUTO_LN]`/`auto-passed` = 自动放行。不同 workflow 的门控形式不同（CP/verifier/milestone/人工确认），需先识别再评估。
- 同前缀并行执行；同 skill 多次出现可能是重试或合理复用，需结合上下文判断。
- 文末 `## Stats` 表含 Root/Subagent/Total 的 Tokens/Cost/Turns/Context Peak/Subagents 汇总。

**大文件导航（轨迹 > 2000 行时）**：不要全文 Read。
1. Grep `^\*\*Tool: skill|^\*Skill:|^## §|skill_content` 提取骨架。
2. SKILL.md 全文用 Grep `skill_content` 定位行号后 Read 该段。
3. 子代理行为在 `<details>` 折叠块内，用 Grep `§N\.M\.` 按需定位。
4. 验收结果用 Grep `PASS|FAIL|FAILED|❌|✅` 定位，排除测试用例输出中的 `[FAIL]` 噪音（通常在子代理折叠块内，且是预期失败基线）。
5. Stats 表在文件末尾，直接 Read 最后 50 行。
6. **STATE.md 驱动模式额外 Grep**：先 Grep `开发工作流 STATE|^## Tasks|^- \[.\] [0-9]+\.` 判断是否存在 STATE.md 块；存在则 Grep `执行者|验收者|验收标准` 提取每任务三段，Grep `"- \[.\] [0-9]+\.`（Edit 块 `oldString`/`newString`）定位勾选动作序列=验收结果序列。

**噪音内容（跳过/一句话总结——v3 超时主因）**：以下内容对 workflow/skill 质量分析无价值，即使被 Grep 命中也不要展开精读，直接跳过或一行带过：

| 噪音类型 | 识别特征 | 处理 |
|---------|---------|------|
| web 资料检索 | `**Tool: webfetch**` / `**Tool: websearch**` 后的 Output 块（网页正文/markdown） | 跳过；该 turn 若仅做资料检索，flow 不设节点，`sessionSummary` 一句带过 |
| 大段文件正文 | `**Tool: read**` Output 超约 50 行的文件内容 | 跳过正文，仅记"读取 {filePath}（{N} 行）" |
| assistant 长篇解释 | 非 tool_use 的 text block，>20 行且不含 skill/gate/§关键词 | 一句话总结，不计入 flow |
| 测试用例详细输出 | 子代理折叠块内的断言/堆栈/日志 | 排除（预期失败基线噪音，沿用规则 4） |

**`_Thinking:` 块的处理（选择性精读——不要全读也不要全跳）**：
`_Thinking:` 是 agent 的回忆/考虑，非真实推进。但其中含**决策推理**（为什么调某个 skill、为什么重试、为什么跳过门控），对 G1/G2/S2 的根因判断有价值。机械全读浪费时间，机械截断丢决策——按下表选择性精读：

| thinking 内容 | 处理 |
|--------------|------|
| 含决策关键词（dispatch/invoke/retry/FAIL/PASS/verifier/跳过/降级/异常/重做） | 精读——提取决策原因，用于 `diagnosis` |
| 含子代理派发理由（"派发 developer 因为..."） | 精读——用于 G4 完整性 + S2 冗余判断 |
| 纯回忆/复述上下文（"让我先读 STATE.md"/"我需要理解任务"/"根据上文"） | 跳过，不精读 |

操作方法：先 Grep `^_Thinking:` 定位所有 thinking 块的行号；对每个块，Grep 其内容是否含决策关键词（`dispatch|invoke|retry|FAIL|verifier|跳过|异常|重做`），**仅对命中的块 Read 小窗口（±10 行）**。不含决策关键词的 thinking 块不 Read，不进 flow，不影响 rating。

精读范围仅限：`*Skill: ...*` 标记行 / `**Tool: skill**` 的 `skill_content` 全文 / `**Tool: task**` 的 Input（subagent_type）/ `**Tool: todowrite**` 的 Input/Output / 门控行（`PASS`/`FAILED`/`[AUTO_LN]`/`auto-passed`）/ turn 标题与 `<details>`/`<summary>` 结构标签 / 文末 `## Stats` 表。

- Read 一律小窗口（命中行 ±5 行），不要 Read 整个 turn 或整个 `<details>` 折叠块。
- 子代理 `<details>` 默认不展开，仅在评 G1/G2 需验收证据时 Grep 定位其内 `PASS`/`FAILED` 行。
- 仅含 webfetch/websearch/read 且无 skill/gate 标记的 turn 不进 `flow`。

# 步骤一：识别 Workflow 基准流程

轨迹中首个 `*Skill: xxx (invoke) ✅` 的 Output 块含 `<skill_content name="xxx">` 全文——这是 session 实际使用的 workflow skill。若 invoke 了多个 skill，取编排器角色的 skill（描述中含「编排/orchestrator/workflow/工作流」）。若轨迹中无 invoke skill，降级为从 dispatch 序列 + 子代理行为反推，`workflowType` 标 `inferred`，`source` 标 `"从 dispatch 序列推断"`。

从 SKILL.md 全文提取以下要素填入 `workflowMeta`：

| 要素 | 提取方法 | 示例 |
|------|----------|------|
| `skillName` | `<skill_content name="...">` 的 name 属性 | `ops-registry-invoke-glacier` |
| `workflowType` | 读 SKILL.md 工作模式/角色描述，判断模式 | `orchestrator-pattern` / `phase-gated` / `linear` / `hybrid` |
| `phases` | 读 SKILL.md 工作流/阶段定义，列出所有阶段/里程碑 | `["M1 设计","M2 基础设施",...]` |
| `gates` | 读 SKILL.md 门控/验收/检查点描述 | `[{"name":"per-task verifier","type":"auto","required":true}]` |
| `requiredSkills` | dispatch 的 subagent_type + invoke 的 skill name | `["developer","verifier","designer",...]` |
| `orderingRule` | 读 SKILL.md 顺序约束 | `strict-sequential` / `phase-gated` / `dynamic` |
| `source` | 标注提取来源 | `"§N invoke Output 块"` |

若 SKILL.md 引用了外部模板（如 `STATE.md.templ`）或子代理定义（如 `agents/developer.md`），这些通常不在轨迹 MD 中。能从子代理行为反推的用于 S1 评（降级），无法反推的标 `n-a` 并注明原因。

## 步骤一·补充：STATE.md 驱动模式

**识别**：invoke skill 的 SKILL.md 出现「读取 `STATE.md`」「纯编排者（Orchestrator）」「按 `STATE.md` 勾选状态推进」「勾选状态只能由 verifier 修改」等表述——此为 **STATE.md 驱动模式**。该模式下，workflow 的**阶段定义不在 SKILL.md 里，而在运行时生成的 `STATE.md` `## Tasks` 列表里**（SKILL.md 只定义"读 STATE.md → 派发 → 看勾选推进"的机制）。

**取基准（覆盖上面的默认规则）**：检测到此模式时，`workflowType` 标 `state-md-driven`，基准从 `STATE.md` 块提取，**不要**再从 SKILL.md 的阶段描述提取：

| 要素 | 提取方法（STATE.md 驱动） | 示例 |
|------|----------|------|
| `skillName` | 仍取 invoke skill 的 `<skill_content name="...">`（编排器 skill 本身） | `ops-registry-invoke-glacier` |
| `workflowType` | 固定 `state-md-driven` | `state-md-driven` |
| `phases` | 读 `STATE.md` 的 `## Tasks`，**每个 `- [ ] N. <标题>` 一项**，按编号顺序 | `["1. 创建开发工作区","2. 初始化开发日志",...]` |
| `gates` | 每个任务一项：`{name:"Task N verifier 勾选", type:"auto", required:true}`——验收=verifier 把 `[ ]` 改 `[x]` | `[{"name":"per-task verifier 勾选","type":"auto","required":true}]` |
| `requiredSkills` | 扫 `STATE.md` 全部任务的 `执行者` + `验收者` 去重 | `["developer","verifier","req-analyst","req-verifier",...]` |
| `orderingRule` | `strict-sequential`（STATE.md 显式要求前序未勾选不派发后续） | `strict-sequential` |
| `source` | `"STATE.md ## Tasks 块（write 工具块内）"` | `"STATE.md ## Tasks 块（write 工具块内）"` |

**验收语义（重要）**：此模式以 **STATE.md 勾选状态**为唯一验收依据，verifier 返回 "DONE" **不等于** PASS——必须看到对应 Task 的 `- [ ] N.` 被 Edit 改为 `- [x] N.` 才算 gate PASS。未勾选即重派执行者=gate FAIL/重试。评估 G1/G2/门控执行时一律以勾选序列为准，不以 verifier 文本返回为准。

# 步骤二：分析维度（G/S 质量框架）

分两组：G 系列=任务产出质量（skill 做得好不好），S 系列=skill 本身写得好不好。每个维度标注**证据源**，能从轨迹测的才评分，测不出的标 `n-a` 并注明原因。

## 对照基准（评任何维度前必做）

不要脱离基准空评。评 G/S 各维度前，先对照 `workflowMeta` 做三组比对，不一致处记入 `workflowLevelIssues` 或 `flow.problems`：

| 比对 | 实际（轨迹） | 基准（workflowMeta） | 不一致时记入 |
|------|-------------|---------------------|-------------|
| 阶段推进 | `sessionMeta.reachedPhase` | `phases` | 缺失/越序的阶段 → workflowLevelIssues(type=missing-step/out-of-order) |
| 门控执行 | 轨迹里的 PASS/FAILED/auto-passed | `gates`（name/type/required） | 跳过的 required gate / 自动放行的 manual gate → flow.problems(type=gate-autopassed) |
| 角色覆盖 | dispatch 的 subagent_type 集合 | `requiredSkills` | 缺角色 / 多余角色 → skillQuality(occurrences:0) + workflowLevelIssues |

基准未定义但轨迹有的（如未在 gates 列但实际执行了验收），标"基准未定义、轨迹实际存在"，不判为问题。

> **STATE.md 驱动模式适配**：`phases` = STATE.md 任务清单；"阶段推进"比对改为"已勾选 `[x]` 的 Task 序列 vs `phases` 全集"（缺失/越序的 Task → workflowLevelIssues）；"门控执行"比对改为"Edit 勾选动作序列 vs `gates`"——**未勾选却推进、或跳过前序 Task 直接勾选后续 = 越序问题**。

## G 系列 · 任务产出质量

| 维度 | 含义 | 证据源 |
|------|------|--------|
| **G1 正确性** | skill 输出是否正确达成目标 | 验收机制 outcome（verifier PASS/FAIL、CP 门控、校验退出码）+ 重试周期 |
| **G2 指令遵循** | 是否遵循格式/约束 | 校验脚本/校验步骤的失败记录（spec 校验器 FAIL、STATE.md 验收命令 FAIL 等） |
| **G3 安全性** | 数值稳定性/溢出/资源/内存 | 设计层产出（spec.yaml numerical_stability、DESIGN.md 数值稳定处理、内存约束）。代码层安全需实现/检视阶段，未到则注明 |
| **G4 完整性** | 是否覆盖所有必要方面 | 验收标准/评审条款覆盖，验收因"覆盖缺口"返回 FAILED 即不达标 |
| **G5 鲁棒性** | 边界/异常处理 | 测试设计产出（DESIGN.md 测试矩阵/TEST.md：边界值、空/标量/广播/rank 溢出等）。执行验证需验收阶段，未到则注明 |

## S 系列 · skill 本身质量

- **S1 可执行性**：指令是否清晰、具体、可操作。
  - **invoke skill**（SKILL.md 全文嵌入 invoke Output 块）：直接读 skill 文本评 + 结合子代理行为。**必须做静态分析**，逐条扫以下缺陷填入 `staticChecks`：

    | 类别 | 判定 |
    |------|------|
    | `ambiguity` | 模糊表述（"适当/合理/视情况"）、未定义术语、可多重解释 |
    | `io-unclear` | 未声明输入/输出契约、缺字段定义 |
    | `asymmetry` | 有输入约束却无对称输出校验（或反之） |
    | `structure` | 缺 MUST/SHOULD 分级、缺失败处理路径、步骤间依赖不清 |
    | `reference` | 引用文件/章节但未内嵌或未给确定路径 |

    每条含 `category`/`severity`/`issue`/`snippet`(原文片段)/`suggestion`。S1 的 `rating` 综合"子代理行为表现"+"静态缺陷数量与严重度"得出。
  - **dispatch skill**（agent 定义通常未嵌入 MD）：降级评估——基于 task prompt + 子代理行为反推。`staticChecks` 可为空数组，note 注明 `"agent 定义未嵌入，基于 prompt+行为反推"`。若 task prompt 质量高且子代理行为一致，可给 `pass`。
  - **编写原则**（对照 skill 文本看是否违反，违反即记入 `staticChecks` 的 `structure`/`ambiguity` 类）：

    | 原则 | 适用 | 要点 |
    |------|------|------|
    | 祈使语气 | S1 | 指令以祈使句给出，而非被动/疑问/描述 |
    | 示例驱动 | S1 | 关键输出有 Input/Output few-shot 示例 |
    | 理论思维 | S1 | 讲清目标/边界条件而非只列步骤 |
    | 解释 Why | S3 | 约束附理由，而非硬性 MUST/NEVER |
    | 避免过度约束 | S3 | 大写 MUST/NEVER/ALWAYS 是黄牌，改用推理解释 |

- **S2 成本意识**：输出是否简洁无冗余。证据源=token/调用次数/冗余调用（同一 skill 被多个独立 subagent session 重复 invoke、重复加载 workflow、可合并的 Task 被拆过多）。**可测**。
- **S3 可维护性**：结构是否清晰、分段合理、易改。invoke skill 的 SKILL.md 全文嵌入→**可测**。dispatch skill 的 agent 定义未嵌入→标 `n-a(定义未嵌入)`。

## 流程维度（执行层，非质量评分）

**flow 节点设定规则（强制，降低重建非确定性）**：
- flow 数组里只出现 4 种 type：`invoke`（仅 workflow skill 首次加载）、`dispatch`（派发子代理干活）、`gate`（verifier 验收）、`terminal`（dispatch 失败/超时）——**不要出现 invoke 知识库加载节点**（ascendc-regbase 等知识加载不是执行步骤，不进 flow）
- **dispatch 类**（`*Skill: xxx (dispatch)`）= **恰好 1 个 flow 节点**（1:1，不归组、不省略、不合并）
- verifier 的 dispatch + 其 `task_result`(PASS/FAIL) = **合并为 1 个 type=gate 节点**
- 纯 bash/read/edit/webfetch turn（无 `*Skill:` 标记）= **不设** flow 节点
- dispatch ❌（Error/超时）= 1 个 type=terminal 节点，status=failed
- 里程碑完成（◆）= 1 个 type=gate 节点
- **`parallel` 字段**：仅当同 turn 内有多个 dispatch（编排器在同一 turn 派发了多个子代理）时才标 parallel 组 id；invoke 节点不标 parallel
- **STATE.md 驱动模式 flow 映射**：编排器反复 `**Tool: read**` 读 STATE.md 的 turn = 机制动作，**不设 flow 节点**（同知识库加载）。每个 STATE.md Task = 1 个 `dispatch` 节点（`step`="Task N: <目标>"，`skill`=执行者）+ 紧接 1 个 `gate` 节点（`skill`=验收者，`step`="Task N 验证: <验收标准简述> [x]"，status 由是否勾选决定）。gate 节点的 `turn` 取 verifier 那次 dispatch 的 turn（非勾选 Edit 的 turn）。同 Task 多次 dispatch（重试）用 `retryOf` 串起来。

对照 `flow.problems[].type` 枚举评估：完整性（走到哪/停止点是否合理）、门控（执行/跳过/自动放行——**人工语义确认类门控被 L3 全自动放行的需标注**）、顺序（对照 orderingRule）、重试（失败重试 vs 合理复用+根因）、冗余（跨 subagent session 无 context 共享导致同一 skill 被反复 invoke 是典型）、缺失（定义要求但未做的步骤）、异常终止（dispatch ❌/session 超时，根因是 skill 还是基础设施）、定义歧义（skill 阶段归属与 SKILL.md 不一致）。

## 性能维度（耗时分析）

轨迹 MD 的 turn 标题行含耗时信息：`## §N Assistant · model · Xs · 📦 Y% context` 和 `#### §N.M.K Assistant · model · Xs ·`。从这些字段提取墙钟耗时，分析瓶颈并给出**具体可行的优化建议**。

**分析步骤**：
1. Grep `^## §\d+ .* · .* [0-9.]+(s|min) ·` 提取主 Agent 每 turn 耗时；Grep `^#### §` 提取子代理 turn 耗时。
2. 按子代理 session（§N.M 前缀）分组求和，找出耗时最长的 session。
3. 对照 flow 节点的 Task 编号，定位哪些 Task 的子代理 session 耗时最长。
4. 检查是否有并行 dispatch（同 turn 多个 `*Skill: (dispatch)` 标记），统计串行 vs 并行比例。

**识别以下耗时问题**：

| 问题类型 | 识别方法 | 判断标准 |
|---------|---------|---------|
| Task 过大 | 单个子代理 session 耗时 > 总耗时 10% 或 turns > 50 | 拆分为 2-3 个子 Task |
| 串行可并行 | 连续 dispatch 的 Task 之间无数据依赖（如 5 个分支 DESIGN 各自独立） | 建议编排器并行 dispatch |
| verifier 开销 | verifier session 数 > dispatch session 数的 50%，或单次 verifier turns > 20 | 建议轻量内联验证或合并验证 |
| 主 Agent 空等 | 主 Agent turn 耗时 > 10min 且该 turn 含 dispatch（等待子代理返回） | 正常但可标注瓶颈 |
| 重试浪费 | 同一 Task dispatch 失败后重试（retryOf 非 null），重试 session 耗时 > 原始 | 建议增加前置检查或拆分 |
| 子代理内 turn 过多 | 单 session turns > 30 | 建议限制子代理 max_turns 或拆分任务粒度 |

# 输出格式（严格 JSON，不要额外文字、不要 ```json 围栏）

```json
{
  "sessionSummary": "一句话：实际走了什么、走到哪、整体质量",
  "workflowMeta": {
    "skillName": "ops-registry-invoke-glacier",
    "workflowType": "orchestrator-pattern | phase-gated | linear | hybrid | inferred | state-md-driven",
    "phases": ["M1 设计", "M2 基础设施", "..."],
    "gates": [
      { "name": "per-task verifier", "type": "auto | manual | hybrid", "required": true }
    ],
    "requiredSkills": ["developer", "verifier", "designer", "..."],
    "orderingRule": "strict-sequential | phase-gated | dynamic",
    "source": "§2 invoke Output 块"
  },
  "sessionMeta": {
    "sessionId": "...",
    "operator": "算子名 或 任务名",
    "model": "...",
    "duration": "...",
    "tokens": "...",
    "autonomy": "L3 全自动 / L2 半自动 / ...",
    "reachedPhase": "阶段/里程碑X（已完成/进行中/停止）",
    "cpsExecuted": ["per-task verifier ×29", "..."],
    "cpsMissing": ["M5 批量验证", "..."],
    "phasesNotReached": ["M5 验证", "阶段四 上库", "..."]
  },
  "flow": [
    {
      "id": "n1",
      "skill": "ops-registry-invoke-glacier",
      "step": "加载 workflow 编排 skill",
      "type": "invoke",
      "turn": 2,
      "parallel": null,
      "retryOf": null,
      "status": "ok",
      "problems": []
    },
    {
      "id": "n2",
      "skill": "developer",
      "step": "Task 1: 创建开发目录",
      "type": "dispatch",
      "turn": 4,
      "parallel": null,
      "retryOf": null,
      "status": "ok",
      "problems": []
    },
    {
      "id": "n3",
      "skill": "verifier",
      "step": "Task 1 验证: 目录创建 PASS",
      "type": "gate",
      "turn": 5,
      "parallel": null,
      "retryOf": null,
      "status": "ok",
      "problems": []
    },
    {
      "id": "n4",
      "skill": "developer",
      "step": "Task 28: 批量黑盒验证",
      "type": "dispatch",
      "turn": 102,
      "parallel": null,
      "retryOf": null,
      "status": "failed",
      "problems": [
        {
          "type": "session-crash",
          "dimension": "process",
          "severity": "high",
          "evidence": "§102 developer dispatch ❌",
          "diagnosis": "[infra-issue] 子代理 session 超时崩溃",
          "suggestion": "拆分 Task 28 为 2-3 个子任务"
        }
      ]
    }
  ],
  "skillQuality": [
    {
      "skill": "skill名",
      "occurrences": 2,
      "ratings": {
        "G1": { "rating": "pass", "note": "29 次 verifier 全 PASS" },
        "G2": { "rating": "weak", "note": "重试后通过", "evidence": "turn 5 / line 1234", "diagnosis": "[skill-defect] SKILL.md 缺失败处理", "suggestion": "增加 retry 条款" },
        "G3": { "rating": "n-a", "note": "无数值产出" },
        "G4": { "rating": "pass", "note": "覆盖全部 Task" },
        "G5": { "rating": "n-a", "note": "未到验收阶段" },
        "S1": { "rating": "weak", "note": "指令有歧义", "evidence": "SKILL.md 第 3 行", "diagnosis": "[skill-defect] 模糊表述", "suggestion": "改为祈使句", "staticChecks": [
          { "category": "ambiguity", "severity": "medium", "issue": "'适当'表述模糊", "snippet": "适当调整参数", "suggestion": "改为'调整 lr=0.001'" }
        ] },
        "S2": { "rating": "pass", "note": "无冗余" },
        "S3": { "rating": "pass", "note": "结构清晰" }
      },
      "summary": "该 skill 整体一句话评价"
    }
  ],
  "workflowLevelIssues": [
    {
      "id": "wf-1",
      "type": "incomplete | gate-skipped | out-of-order | redundant | missing-step | session-crash | skill-mismatch | other",
      "severity": "high | medium | low",
      "title": "简短标题",
      "detail": "证据+现象（引用 turn/skill/line）",
      "suggestion": "针对 workflow/SKILL.md 的可执行建议"
    }
  ],
  "optimizationPriorities": [
    {
      "priority": 1,
      "target": "skill:xxx | workflow:step | gate:xxx",
      "action": "具体动作",
      "expectedGain": "预期收益"
    }
  ],
  "perfAnalysis": {
    "totalDuration": "总墙钟耗时（从 Stats 表或首末 turn 时间戳推算）",
    "mainTurns": "主 Agent turn 数",
    "subagentSessions": "子代理 session 数",
    "subagentTurns": "子代理 turn 数",
    "parallelSessions": "并行 session 数（同 turn 多 dispatch）",
    "serialRatio": "串行占比（串行 session / 总 session，0-1）",
    "topSlowSessions": [
      {
        "session": "§N.M",
        "task": "Task 编号",
        "skill": "子代理 skill 名",
        "durationSec": 1234,
        "turnCount": 50,
        "problem": "task-oversized | too-many-turns | retry-waste | verifier-overhead | other",
        "diagnosis": "为什么慢（如：单个子代理跑了 535 条全量验证）",
        "suggestion": "具体建议（如：拆分为定位→修复→验证 3 个子任务）"
      }
    ],
    "parallelizationOpportunities": [
      {
        "tasks": ["Task 12.1", "Task 12.2", "Task 12.3"],
        "reason": "这些 Task 互相无数据依赖（各写独立分支 DESIGN 文件）",
        "currentMode": "串行（§N→§N+1→§N+2 顺序 dispatch）",
        "suggestion": "编排器在同一 turn 并行 dispatch 3 个 designer 子代理",
        "estimatedSaving": "串行 3×Xmin → 并行 Xmin，省 2/3 时间"
      }
    ],
    "summary": "一句话耗时结论：主要瓶颈是什么、最大的优化机会在哪"
  }
}
```

> **rating 字段精简规则（减少嵌套深度，防 JSON 结构错误）**：
> - `pass`/`n-a` → **只写 `rating` + `note` 两个字段**（不写 evidence/diagnosis/suggestion）
> - `weak`/`fail` → 必填全部 5 字段（rating/note/evidence/diagnosis/suggestion）
> - `staticChecks` → 仅 S1 且仅有缺陷时填，无缺陷时填 `[]` 或省略
> - 这条规则把每个维度从 5 字段降到 2 字段（多数维度是 pass/n-a），减少 ~60% 嵌套量，显著降低漏括号概率

# 约束

- 只基于轨迹里的证据下结论，不臆测；证据要引用 turn#/skill 名/line。
- 建议必须可执行，指向具体 skill（其 SKILL.md）或 workflow 步骤/门控，不要泛泛"加强质量"。
- `rating` 只在证据充分时给 pass/weak/fail。凡 `weak`/`fail` 必须填 `evidence` + `diagnosis` + `suggestion`，`pass`/`n-a` 可只填 `note`。
- **rating 判定标准**（降低主观摇摆）：

  | rating | 判定条件 |
  |--------|---------|
  | `pass` | 证据明确通过：验收 PASS 且无重试 / 覆盖完整无缺口 / staticChecks 无 high 级缺陷 |
  | `weak` | 通过但有瑕疵：重试后通过 / 部分覆盖 / staticChecks 有 medium 缺陷 / 数据部分缺失但仍可评 |
  | `fail` | 明确不达标：验收 FAIL 未修复 / 关键步骤缺失 / staticChecks 有 high 级致命缺陷 |
  | `n-a` | 该维度证据不在轨迹中（注明原因，如"代码层未到实现阶段"、"dispatch skill 定义未嵌入"） |

  边界判断：重试后 PASS 不给 pass（给 weak）；有 high 级 staticChecks 不给 pass；缺关键证据不猜，给 n-a。
- **diagnosis 必须以根因归类前缀开头**（让 suggestion 的改进对象可机读、可聚合）：
  - `[skill-defect]` = SKILL.md / agent 定义本身的缺陷（对应 S 维度，改 skill 定义）
  - `[execution-deviation]` = skill 定义 OK 但执行偏离（对应 G 维度，改执行/子代理行为）
  - `[infra-issue]` = 基础设施问题（session crash / 超时 / dispatch ❌，非 skill 问题）
  - `[workflow-design]` = workflow 步骤 / 门控 / 编排设计问题（改编排器 skill 或流程）
  - 前缀后接具体根因，如 `[skill-defect] SKILL.md 未定义 dispatch 失败的重试策略`
- 若轨迹数据缺失（耗时=0、token=0、cost=0），相应维度标注"数据缺失"而非编造。
- `skillQuality` 每个 skill 都要列：轨迹中实际出现的所有 skill（invoke + dispatch）+ `requiredSkills` 中 0 次出现的（标 `occurrences: 0`，各维度 `n-a`）。优先基于轨迹实际证据评，不要轻易标 n-a。
- `flow` 按真实执行顺序排列；并行节点连续排列并共用 `parallel` 组 id；门控和阶段完成作为独立节点列入；没问题的节点也要列出（`problems` 为空数组）。
- `sessionMeta` 的数组字段（`cpsExecuted` / `cpsMissing` / `phasesNotReached`）必须存在且为数组，空也填 `[]`——前端组件用 `.map()` 渲染，缺字段会崩溃。

## 输出

- **用 Write 工具将完整 JSON 保存到文末指定的输出路径**（文件名 `{轨迹文件名}-analysis.json`；若文末给出了绝对路径，必须用该路径，不要改写到 `logs/` 或其他目录），再在终端输出该文件路径供用户确认。
- **必须用缩进格式输出 JSON（每层缩进 2 空格，逐层换行）**——这是强制要求，不是建议。单行格式会让你在 skillQuality 的 6 层嵌套中漏括号（实测 60%+ 概率漏 `}`）。每写一个 `{` 就换行缩进一级，每写一个 `}` 就退回一级——缩进层级就是你的括号计数器。**不要用 `json.dumps(compact)` 或单行输出**。
- 不加前后说明文字，不加 markdown 围栏。
- **写完后自校验（三步，缺一不可）**：
  1. **用 Read 工具重新读取该文件**（不是凭记忆——Read 强制从磁盘读取实际内容，避免 Write 缓存导致校验假通过）
  2. 用 Bash 运行 `python3 -c "import json; json.load(open('输出路径'))"` 校验语法
  3. 若报错：用 Read 重新读文件内容，定位错误位置（补漏的 `}`、删多余的 `{` 或 `,`），修正后重新 Write，再从第 1 步重新开始——**直到 json.load 通过才回复 DONE**
- 这一步能消除 90%+ 的结构错误。跳过自校验会导致下游解析失败、审计报告无法渲染。
