# Workflow 分析提示词（喂给 Claude Code）

> **用法**：在新的 Claude Code 会话里，先粘本文件全部内容，再把 CANNBot-Insight 导出的 `session_{taskId}.md`（"Export MD" 按钮）内容粘到文末 `<<<轨迹>>>` 处（或让 Claude Code 读该文件）。Claude Code 会输出纯 JSON，把它粘进 cannbot-insight 该 session 的「Workflow 分析」tab 文本框 → 渲染。
>
> **可选附加**：skill 文本通常已嵌在轨迹 MD 里（invoke 的 Output 块 / dispatch 的子代理 turn），S1/S3 可直接评；仅当某 skill 的 SKILL.md/agent 定义全文未包含在 MD 时，才需额外粘其 SKILL.md。
>
> 输出 JSON 结构：`sessionSummary` / `sessionMeta` / `flow[]`(执行节点, 每个含 `problems[]` 带 `dimension`) / `skillQuality[]`(G1-G5/S1-S3 评分) / `workflowLevelIssues[]` / `optimizationPriorities[]`。框图按 `flow` 画，节点徽章/侧栏按 `problems` 与 `workflowLevelIssues` 渲染，`skillQuality` 供后续质量看板。

---

# 角色
你是 CANNBot Agent 平台的 workflow/skill 优化顾问。CANNBot-Insight 是 opencode session 的可观测工具。下面给你一个 session 的**实际工作流执行轨迹**（由 CANNBot-Insight 导出），该 session 基于 `ops-registry-invoke-workflow` skill 运行。你的任务：对照规范流程，分析实际走了什么、哪里有问题，给出可执行的 skill 与 workflow 优化建议。

# 规范流程（ops-registry-invoke-workflow 定义，摘要）
- 阶段一 需求与设计：1.1 开发准备 → 1.2 需求分析 → ⛔CP1 → 1.2.5 spec 生成(ascendc-ops-architect) → 1.2.5R spec 评审(design-reviewer) → ⛔CP1.5 → 1.3a 设计准备(designer) → 1.3b 分段切片(主Agent) → 1.3c 并行分段生成(5× ascendc-ops-design-*) → 1.3d 组装校验 → 1.3R 方案评审(designer) → 1.4 测试设计(ascendc-ops-tester) → 1.4R 测试设计评审(design-reviewer) → ⛔CP2
- 阶段二 开发：动态迭代 Wave1(A1-Main+A1-P+B)/Wave2(A2+重试) → 测试工程师验收
- 阶段三 验收：3.1 精度验收(ST测试) → ⚪CP3 → 3.2 性能验收 → ⚪CP4
- 阶段四 上库：4.1 文档示例 → 4.2a 全量代码检视 → 4.2b 一致性检查 → ⚪CP5 → 4.3 总结
- 门控：CP1/CP1.5/CP2 必需确认；每阶段必须通过校验才进下一阶段；禁止跳阶段。
- 注意：不是每个 session 都走完完整 workflow；只到 阶段一 就停是常见情况，不算缺陷。
- 如需完整定义，可附加 `plugins-official/ops-registry-invoke/workflow/SKILL.md` 全文。

# 输入数据（实际执行轨迹 MD）
轨迹是 CANNBot-Insight 导出的 session 全文 MD，结构特征：
- `## §N User/Assistant` = 主 Agent 每个 turn；`§N.M` = 子 Agent session；`§N.M.K` = 子 Agent 内 turn。
- `*Skill: <skill名> (invoke|dispatch) ✅/❌*` 标记每次 skill 调用，按出现顺序=执行序。
- `<thinking>...</thinking>` 是 Agent 的思考（回忆/考虑），**非真实推进**；真实阶段推进在 thinking 之外的 `阶段X...完成/开始` 标记。
- `CP1/CP1.5/CP2...` + `[AUTO_LN]` = 门控自动放行；`状态=❌`/`重试`/`FAIL` = 失败重做。
- 同前缀并行执行；同 skill 多次出现可能是重试或合理复用，需结合上下文判断。

请基于下面给出的实际轨迹进行分析：

<<<在此粘贴 session_{taskId}.md 全文>>>

# 分析维度（G/S 质量框架，对照规范流程）
分两组：G 系列=任务产出质量（skill 做得好不好），S 系列=skill 本身写得好不好。每个维度标注**证据源**，能从轨迹测的才评分，测不出的标 `n-a` 并注明原因。

## G 系列 · 任务产出质量
- **G1 正确性**：skill 输出是否正确达成目标。证据源=评审 outcome（1.2.5R/1.3R/1.4R 的 ✅/❌）+ 重试周期。review-reject 即正确性不达标。**可测**。
- **G2 指令遵循**：是否遵循格式/约束。证据源=校验脚本失败：spec 9-stage FAIL、1.3d 组装校验 FAIL、checklist STATUS。**可测**。
- **G3 安全性**：生成内容是否安全无害（算子场景=数值稳定性/溢出/资源/内存等）。证据源=**轨迹 MD 里的 spec.yaml/DESIGN.md 设计层安全考虑**（数值稳定性处理、Workspace 内存占用、资源约束等，阶段一已产出）。**可测（设计层）**——评设计是否覆盖安全/数值稳定；代码层安全（漏洞）需阶段二/四，未到则在 note 注明"代码层未评估"。
- **G4 完整性**：是否覆盖所有必要方面。证据源=评审条款覆盖（13 SPEC-*、测试覆盖缺口），review-reject 因"覆盖缺口"。**可测**。
- **G5 鲁棒性**：边界/异常处理。证据源=**轨迹 MD 里的 TEST.md(1.4) 测试设计**（边界值处理、边界情况处理、L0/L1/L2 用例覆盖）。**可测（测试设计层）**——评测试设计是否覆盖边界/异常；执行验证（用例是否 PASS）需阶段三，未到则在 note 注明"执行层未验证"。

## S 系列 · skill 本身质量
- **S1 可执行性**：SKILL.md/Agent 定义指令是否清晰、具体、可操作。证据源=**轨迹 MD 里嵌入的 skill 文本**：invoke skill 的 Output 块含完整 SKILL.md；dispatch skill 的 agent 定义引用 + 子代理 turn 行为（是否理解执行到位）。**可测**——直接读嵌入的 skill 文本评指令清晰度，结合子代理行为看是否被准确执行。**额外做静态分析**：只读 skill 文本本身（不依赖执行轨迹），逐条扫以下缺陷并填入 `staticChecks`：
  - `ambiguity` 歧义：模糊表述（"适当/合理/视情况/尽量"）、未定义术语、可多重解释的指令。
  - `io-unclear` 输入输出不明确：未声明输入契约/输出契约/输出格式；缺字段定义。
  - `asymmetry` 不对称：有输入约束却无对称的输出校验（或反之）；要求做的事与要求自检的事不配对。
  - `structure` 结构：缺 MUST/SHOULD 分级、缺失败处理路径、步骤间依赖不清。
  - `reference` 引用：引用了文件/章节但未内嵌或未给确定路径，子代理可能找不到。
  每条含 `category`/`severity`/`issue`/`snippet`(原文片段)/`suggestion`。S1 的 `rating` 综合"子代理行为表现"+"静态缺陷数量与严重度"得出。
- **S1/S3 编写原则（评判依据，对照 skill 文本看是否违反）**：
  - 祈使语气：直接告诉模型做什么（"Prefer using the imperative form"）——指令是否以祈使句给出，而非被动/疑问/描述。
  - 解释 Why：给出原因而非硬性规则（"explain to the model why things are important in lieu of heavy-handed MUSTs"）——约束是否附理由。
  - 示例驱动：用 Input/Output 对展示（SKILL.md 示例章节）——关键输出是否有 few-shot 示例。
  - 避免过度约束：写大写 MUST/NEVER/ALWAYS 是黄牌，改用推理解释（"if you find yourself writing ALWAYS or NEVER in all caps...that's a yellow flag"）。
  - 理论思维：让模型理解任务而非死记步骤（"Use theory of mind"）——是否讲清目标/边界条件而非只列步骤。
  S1 评祈使语气/示例驱动/理论思维；S3 评解释 Why/避免过度约束。违反即记入对应维度的 `staticChecks`（`structure`/`ambiguity` 类）。
- **S2 成本意识**：输出是否简洁无冗余。证据源=token/调用次数/冗余调用（env-check×2、npu-arch×2、重复加载 workflow）。**可测**。
- **S3 可维护性**：SKILL.md/Agent 定义结构是否清晰、分段合理、易改。证据源=**轨迹 MD 里嵌入的 skill 文本**（同 S1）。**可测**——读 skill 文本的结构/分段/交叉引用评。若某 dispatch skill 的 agent 定义仅被引用而全文未嵌入，则该 skill 的 S3 标 `n-a(定义未嵌入)`。

## 流程维度（执行层，非质量评分）
- 完整性：session 走到哪个阶段？是否中途停止？停止点是否合理（用户指令 vs 异常）？
- 门控：CP1/CP1.5/CP2/CP3/CP5 是否执行？CP1.5 这种"必须人工语义确认"的是否被 L3 自动放行？
- 顺序：步骤顺序是否合规（1.3R 前不应进 1.4；spec 评审前不应设计）？
- 重试：哪些 skill 重复调用？失败重试 vs 合理复用？根因？给出重试周期。
- 冗余：重复工作、可并行却串行、token 异常高、同 session 重复调用支撑类 skill。
- 缺失：规范要求但实际没做的步骤/skill。
- 定义歧义：skill 阶段归属与 SKILL.md 不一致（如 st-design 在 阶段一跑但 SKILL.md 放阶段二/三）。

# 输出格式（严格 JSON，不要额外文字、不要 ```json 围栏）
```json
{
  "sessionSummary": "一句话：实际走了什么、走到哪、整体质量",
  "sessionMeta": {
    "sessionId": "...",
    "operator": "算子名",
    "model": "...",
    "duration": "...",
    "tokens": "...",
    "autonomy": "L3 全自动 / ...",
    "reachedPhase": "阶段X（已完成/进行中/停止）",
    "cpsExecuted": ["CP1", ...],
    "cpsMissing": ["CP3", ...],
    "phasesNotReached": ["阶段二 开发", ...]
  },
  "flow": [
    {
      "id": "n1",
      "skill": "skill名 或 CP1/阶段一完成",
      "step": "1.1 加载workflow / ⛔CP1 / 1.3c 并行分段 / ◆ 阶段一完成",
      "type": "invoke | dispatch | gate | terminal",
      "turn": "主 Agent turn 序号(整数)",
      "parallel": "p1 | null（同 turn 并行的节点共用一个 parallel 组 id）",
      "retryOf": "被重试的原节点 id | null",
      "status": "ok | auto-passed | failed",
      "problems": [
        {
          "type": "retry | failure | tool-error | slow | high-token | out-of-order | missing-prereq | redundant | gate-autopassed | review-reject",
          "dimension": "G1 | G2 | G3 | G4 | G5 | S1 | S2 | S3 | process",
          "severity": "high | medium | low",
          "evidence": "turn N / line / 现象",
          "diagnosis": "根因判断",
          "suggestion": "针对该 skill SKILL.md 或 workflow 步骤的可执行建议"
        }
      ]
    }
  ],
  "skillQuality": [
    {
      "skill": "ascendc-ops-design-implementation",
      "occurrences": 2,
      "ratings": {
        "G1": { "rating": "pass | weak | fail | n-a", "note": "一句话结论", "evidence": "证据：引用 turn#/skill/line/校验输出", "diagnosis": "根因判断", "suggestion": "针对该 skill SKILL.md/agent 定义的可执行建议" },
        "G2": { "rating": "fail", "note": "1.3d 组装校验失败——分段边界/接口不一致", "evidence": "n19 turn 17 首轮分段未通过 1.3d 组装校验：相邻段 section 边界重叠/留白、sibling 接口签名与 plan 段不一致 (line ~23000)", "diagnosis": "agent 定义要求按 section-map 切片但未强制输出前 sibling 接口自检；并行分段无共享接口契约，边界拼不上", "suggestion": "agent 定义新增 MUST 步骤：返回前对照 section-map 边界 + sibling 接口表自检，未过则就地修正" },
        "G3": { "rating": "weak", "note": "设计层覆盖数值稳定性/内存；代码层未评估(阶段二/四未执行)" },
        "G4": { "rating": "...", "note": "" },
        "G5": { "rating": "weak", "note": "TEST 设计含边界用例；执行层未验证(阶段三未执行)" },
        "S1": { "rating": "weak", "note": "SKILL.md 指令含模糊表述，子代理首次理解偏差", "evidence": "子代理 turn 17 直接产出未做接口自检即返回", "diagnosis": "agent 定义\"按 section-map 切片\"偏粗，缺强制自检步骤", "suggestion": "补 MUST 步骤：返回前 sibling 接口自检", "staticChecks": [
          { "category": "ambiguity", "severity": "high", "issue": "\"按 section-map 切片\"表述模糊，未定义边界重叠/留白判定规则", "snippet": "按 section-map.md 对应区段生成本段实现", "suggestion": "显式定义 [start,end] 与相邻段无交集无间隙" },
          { "category": "io-unclear", "severity": "high", "issue": "输出契约未声明：未规定返回须含起止 section id + sibling 接口签名表", "snippet": "（agent 定义未出现 output 契约字段）", "suggestion": "补 Output Contract：{sectionRange, inputs[], outputs[], siblingInterface}" },
          { "category": "asymmetry", "severity": "medium", "issue": "输入侧有切片约束但输出侧无对称自检", "suggestion": "补对称输出自检步骤" },
          { "category": "structure", "severity": "medium", "issue": "缺 MUST/SHOULD 分级与失败处理路径", "suggestion": "MUST 不过→就地修正重发；SHOULD→仅告警" },
          { "category": "reference", "severity": "low", "issue": "引用 section-map.md/sibling 接口表未内嵌或未给确定路径", "suggestion": "固定引用路径或内嵌接口表摘要" }
        ] },
        "S2": { "rating": "weak", "note": "2 次重炉+高 token" },
        "S3": { "rating": "pass", "note": "结构清晰、分段合理" }
      },
      "summary": "该 skill 整体一句话评价"
    }
  ],
  "workflowLevelIssues": [
    {
      "id": "wf-1",
      "type": "incomplete | gate-skipped | out-of-order | redundant | missing-step | other",
      "severity": "high | medium | low",
      "title": "简短标题",
      "detail": "证据+现象（引用 turn/skill）",
      "suggestion": "针对 workflow/SKILL.md 的可执行建议"
    }
  ],
  "optimizationPriorities": [
    {
      "priority": 1,
      "target": "skill:xxx | workflow:1.3c | gate:CP2",
      "action": "具体动作",
      "expectedGain": "预期收益"
    }
  ]
}
```

# 约束
- 只基于轨迹里的证据下结论，不臆测；证据要引用 turn#/skill 名/line。
- 建议必须可执行，指向具体 skill（其 SKILL.md）或 workflow 步骤/门控，不要泛泛"加强质量"。
- 没有问题的 skill 节点也要列出（occurrences 体现在 flow 里多次出现，problems 可为空数组），便于全覆盖渲染。
- `flow` 按真实执行顺序排列（按 turn/line 序）；并行节点连续排列并共用 `parallel` 组 id；重试节点用 `retryOf` 指向原节点。
- 门控(CP)和阶段完成(terminal)作为独立节点列入 flow，便于框图展示门控位置。
- 若轨迹数据缺失（如耗时=0、token=0、cost=0），相应维度标注"数据缺失"而非编造。
- `skillQuality` 每个 skill 都要列（含出现 0 次但规范要求的）；`rating` 只在证据充分时给 pass/weak/fail。凡 `weak`/`fail` 的维度必须填 `evidence`(引用 turn#/skill/line/校验输出) + `diagnosis`(根因) + `suggestion`(指向 SKILL.md/agent 定义的可执行动作)，`pass`/`n-a` 可只填 `note`。**S1 额外必须填 `staticChecks`**：只读 skill 文本扫歧义/输入输出不明确/不对称/结构/引用五类缺陷，每条含 category/severity/issue/snippet/suggestion；只要 skill 文本嵌入 MD 就要扫，不要省略。本 session MD 含 skill 文本(S1/S3)、spec/DESIGN(G3)、TEST 设计(G5)、评审 outcome(G1/G4)、校验失败(G2)、token(G2/S2)，8 维均有至少部分证据——优先基于这些评，不要轻易标 n-a。仅当某维度证据确实不在 MD（如某 dispatch skill 的 agent 定义未嵌入→S3、或代码层安全→G3 代码部分）才标 `n-a` 并在 note 注明原因。禁止无证据编造评分。
- 仅输出上述 JSON 对象，不加前后说明文字，不加 markdown 围栏。
