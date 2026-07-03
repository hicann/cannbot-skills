# Session 对话级洞察 — Phase1 需求分析

版本：v0.2  
最后更新：2026-06-12

> 文档类型：Phase1 需求分析 ｜ 关联项目：CANNBot-Insight  
> 复杂度：**Medium-High**（涉及数据模型重构、per-turn 粒度拆解、subagent 独立存储与交互链路追踪、LLM 上下文治理展示、UI 交互设计）  
> 独立项目：CANNBot-Insight，从零实现，无外部代码依赖

---

## 导读（工程师先看这段）

**这是什么** —— 一套面向「session 中每次对话 turn」的细粒度可观测系统。用户能看到：
1. 每个 turn 的 token 消耗（input/output/reasoning/cache 分项）、耗时、调用的工具、加载的 skill
2. 每个 subagent 独立存在一条可查的记录，同样有上述 turn 级信息
3. 主 agent → subagent 的交互链路：父 agent 发起 task() 的参数、子 agent 返回的结果、两者之间的时间衔接
4. 每个 LLM turn 的**输入上下文**（发给模型的完整 prompt 消息序列）和**输出内容**（模型返回的完整响应），用于上下文治理——理解 LLM 收到了什么、产出了什么、上下文是否合理

**为什么需要** —— 传统方案只在 **Execution 行级** 做聚合统计（总 token、总耗时、tool 调用总数），per-turn 数据被锁在 Session.interactions 的 JSON blob 里，必须全量解析才能回答"第 3 轮对话用了多少 token？调了哪个工具？花了多久？"。subagent 的 Execution 行虽有 schema 支撑但 **从未在写入路径实际创建**，所有 subagent 交互混在 root Session blob 里，无法独立查询。主 agent ↔ subagent 的交互是隐含在 JSON 工具调用中的，无结构化记录。

**CANNBot-Insight 怎么做** —— 不沿用"大 JSON blob + 聚合行"架构，而是以 **Turn** 为一等公民建表，每个 LLM turn / tool turn / skill load event / subagent dispatch 均独立成行。subagent 自动拆出独立 Execution 子树，写入路径落地。主-sub 交互显式建桥记录。

**这一轮做到哪**
1. Turn 级数据拆解：每次对话拆为独立可查行（token/时间/工具/skill）
2. Subagent 独立存储与查询：写入路径实际创建 subagent Execution + subagent Turn 行
3. 主-sub 交互链路表：显式记录 task() dispatch → subagent response 的衔接关系
4. Session 总览 + Turn 下钻 UI：列表看 session 聚合，点击下钻看每个 turn
5. LLM 上下文治理：每个 LLM turn 可查看发给模型的输入消息序列 + 模型返回的完整响应，默认折叠、点击展开

**明确不碰**（§3 边界）：不做 LLM 评估/评分、不做 skill 优化、不做 A/B 测试、不做 AgentDebug 诊断——CANNBot-Insight 只做可观测。

---

## §1 基本信息

### 1.1 项目背景

**需求价值**：让开发者/运维人员能直观理解一次 agent session 的内部运作细节——哪轮对话消耗最多 token、哪个工具调用最慢、哪个 skill 被加载/未加载、subagent 做了什么、主 agent 和 subagent 怎么交互的。同时能查看每轮 LLM 调用的输入上下文和输出内容，帮助进行上下文治理——判断上下文是否过长、是否包含冗余信息、模型是否被误导等。这对性能优化、成本控制、故障排查、skill 效果验证、上下文治理都是刚需。

**需求描述**：新建 CANNBot-Insight 项目，专注 session 可观测性。核心能力：
- **Turn 级拆解**：每个 LLM turn / user turn / tool turn 独立成行，含 token 分项、时间、工具调用明细、skill 加载事件
- **LLM 上下文治理**：每个 assistant turn 存储发给 LLM 的输入消息序列（inputMessagesJson）和模型返回的完整响应（contentJson），默认折叠展示摘要，点击可展开查看全文
- **Subagent 一等公民**：写入路径自动拆分 subagent trace 为独立 Execution + Turn 行树
- **主-sub 交互链路**：显式记录 dispatch（task 参数）→ response（子 agent 产出）的桥接关系
- **UI**：Session 总览列表 → Turn 下钻详情 → Subagent 独立视图

### 1.2 结构化信息

|维度|内容|
|-|-|
|Who|开发者、运维人员、AI 产品经理——需要理解 agent session 内部运作细节的人|
|When|session 完成后事后复盘查看；也支持实时跟进（数据边写边可见）|
|What|per-turn token/时间/工具/skill 洞察；per-subagent 同样洞察；主-sub 交互衔接；LLM 输入上下文 + 输出内容（上下文治理）|
|Why|聚合级统计无法回答细粒度问题（"第 5 轮为什么慢？""subagent 消耗了多少 token？""skill X 被加载了吗？""第 3 轮给 LLM 发了什么上下文？""上下文是否过长导致输出质量下降？"）|
|Where|独立 Web 应用（Next.js），cannbot-insight 目录下|
|How Much|支持单 session 100+ turn、5+ subagent 的查看性能；per-turn 查询无需全量解析 JSON blob|
|How|数据采集通过本地文件导入：(a) CLI/API 导入 opencode sessions.db 文件（用户从多 session 中选择要分析的 session）；(b) CLI/API 导入 Claude Code session JSONL 文件，离线分析|

### 1.3 已确认前提

- **P-01 数据源**：CANNBot-Insight 只支持本地文件导入——直接读取 opencode 的 sessions DB 文件（一个 DB 含多个 session，需用户选择）和 Claude Code 的 session JSONL 文件进行离线分析。**实时上报（POST upload）暂不实现**
- **P-02 技术栈**：Next.js + Prisma + SQLite（成熟生态，开发阶段轻量）
- **P-03 项目范围**：CANNBot-Insight 只做可观测，不做评估/评分/优化——评估相关功能不在本项目
- **P-04 代码归属**：所有代码和设计文档只能放在 cannbot-insight 目录下
- **P-05 离线分析定位**：本地文件导入是唯一的数据入口，用于事后复盘——不需要 agent 在运行时上报，只需指定本地文件路径即可导入分析
- **P-06 Opencode 多 session 选择**：opencode sessions.db 通常包含多个 session，导入时必须先展示 session 列表（含时间、第一个提示词等基本信息供用户判断），用户选择后再入库

---

## §2 核心能力

### 2.1 场景分析

**主成功场景**

```
用户打开 CANNBot-Insight → Session 列表页
  → 看到所有 session 的聚合信息（总 token、总耗时、turn 数、subagent 数、skill 数）
  → 点击某个 session → Session 详情页
    → 看到所有 turn 的列表（时间线视图）
      → 每个 turn 显示：角色(user/assistant/tool/subagent)、token 分项、耗时、工具调用(s)、skill 加载(s)
    → 看到所有 subagent 的卡片（独立视图入口）
    → 看到主-sub 交互链路图（哪个 turn dispatch 了哪个 subagent，subagent 做了什么，结果返回到哪个 turn）
  → 点击某个 subagent → Subagent 详情页
    → 看到该 subagent 的 turn 列表（与主 session 同样的粒度）
    → 看到该 subagent 调用的工具、加载的 skill、token 消耗
  → 点击某个 turn → Turn 详情
    → 看到完整内容（user input / assistant response / tool call input+output）
    → 看到精确 token 分项和时间
    → 看到 LLM 输入上下文区（默认折叠）：
      → 折叠态：显示"输入 N 条消息，共 X tokens"摘要行
      → 点击展开：逐条显示发给 LLM 的消息（system prompt / user messages / assistant history / tool results），每条消息也可独立折叠
    → 看到 LLM 输出内容区（默认折叠）：
      → 折叠态：显示内容摘要（前 200 字）+ "输出 X tokens"
      → 点击展开：完整 markdown 渲染 + thinking/reasoning 折叠区
```

|编号|路径|类别|触发|步骤|
|-|-|-|-|-|
|S-001|主成功|业务|用户查看 session|列表看聚合 → 下钻看 turn → 下钻看 subagent → 看交互链路|
|S-002|扩展|业务|用户查看 subagent 独立视图|直接跳转到某 subagent 的详情页，不看主 session|
|S-003|扩展|业务|用户对比多个 subagent|同一 session 下多个 subagent 的 token/耗时/工具对比|
|S-004|异常|数据|session 上报数据缺失 turn 信息|仍能展示已有 turn，缺失的 turn 标注"数据不完整"|
|S-005|异常|数据|subagent trace 未上报|主 session 仍可见，subagent 卡片标注"trace 缺失"|
|S-006|异常|性能|超大 session（500+ turns）|Turn 列表支持分页/虚拟滚动，不一次性渲染|
|S-007|扩展|业务|多层级 subagent（subagent 内再 dispatch subagent）|递归展示：主 → sub1 → sub2 的树形结构|
|S-008|异常|数据|turn 内 tool_calls 数据缺失|turn 仍可见，工具调用区域标注"无工具调用记录"|
|S-009|扩展|业务|用户查看 LLM 输入上下文（上下文治理）|点击 turn 的"上下文"区展开，逐条查看发给 LLM 的消息序列，理解模型收到了什么|
|S-010|扩展|业务|用户查看 LLM 输出内容|点击 turn 的"输出"区展开，查看模型完整响应（含 thinking/reasoning blocks），判断输出质量|
|S-011|扩展|业务|上下文长度分析|查看某 turn 的输入上下文总 token 数占比（占模型 context window 的百分比），判断上下文是否过长|
|S-012|主成功|业务|本地文件导入分析——Opencode DB|用户指定 opencode sessions.db 文件 → 系统读取 DB → 展示 session 列表（含时间、第一个提示词等基本信息）→ 用户选择要分析的 session → 入库拆解为 Turn/ToolCall/SkillEvent 等 → 可在 UI 中查看|
|S-013|主成功|业务|本地文件导入分析——Claude Code JSONL|用户指定 Claude Code JSONL 文件 → 系统读取并解析 → 入库拆解 → 可在 UI 中查看|
|S-014|异常|数据|本地文件格式不兼容|系统提示"文件格式不匹配，请确认是 opencode sessions.db 或 Claude Code JSONL"|
|S-015|扩展|业务|Opencode DB 批量选择|用户在 session 选择界面勾选多个 session → 批量入库|

### 2.2 业务规则

|编号|描述|原因|影响范围|
|-|-|-|-|
|BR-001|每个 LLM turn 必须独立成行，不得将 turn 数据锁在 JSON blob 里需要全量解析才能获取|per-turn 查询是核心需求，全量解析性能不可接受且无法索引|数据模型|
|BR-002|subagent 的 Execution + Turn 行必须在写入路径实际创建，不得只存在 schema 定义而从未写入|有 schema 无写入 = 功能不存在|入库逻辑|
|BR-003|主 agent dispatch subagent 的 task() 调用与 subagent 的第一轮输出之间必须显式建桥记录|交互链路是用户明确要求的核心信息，隐含在 JSON 中无法直观查看|数据模型|
|BR-004|Turn 的 token 分项（input/output/reasoning/cache_read/cache_write）必须独立存储，不得只存总量|用户需要区分"输入 token 多还是输出 token多""cache 命中了多少"来优化成本|数据模型|
|BR-005|Skill 加载事件必须独立记录为事件行（而非仅在 tool_call 中隐含）|用户明确要求看到"加载了哪些 skill"，且需区分 skill load vs skill invoke|数据模型|
|BR-006|CANNBot-Insight 只做可观测，不做评估/评分/优化|项目范围边界，避免功能膨胀|项目范围|
|BR-007|LLM 输入上下文和输出内容**默认折叠**，不自动展开全文|上下文内容可能极长（数千行），默认展开会严重影响页面加载和视觉噪音；用户按需点击展开|UI|
|BR-008|LLM 输入上下文中的每条消息也可独立折叠/展开|system prompt 可能 200 行、工具返回可能 500 行；全展开不现实，用户需要逐条按需查看|UI|
|BR-010|Opencode DB 导入必须先展示 session 列列供用户选择|一个 DB 含多个 session，不能默认全量导入；需展示每个 session 的创建时间、第一个用户提示词（query）等基本信息帮助用户判断要分析哪个|数据采集|
|BR-011|实时上报（POST upload）暂不实现|MVP 只做离线文件导入；实时上报留作后续迭代|项目范围|

### 2.3 数据约束

|编号|类别|名称|描述|
|-|-|-|-|
|DC-001|模型|Turn 行|每个 turn = 一条 DB 行，含 role、agent_identity、token 分项、时间、content_ref（内容引用而非全文存储以控制行宽）|
|DC-002|模型|InteractionBridge 行|主 agent 的 task() tool_call → subagent 的 session_id 映射，含 dispatch_args、response_summary、dispatch_time、response_time|
|DC-003|模型|SkillEvent 行|skill load / skill invoke 的独立事件行，含 skill_name、version、load_time、success/failure、调用方 agent|
|DC-004|约束|Turn 序号单调|同一 agent 内 turn_number 单调递增，不回退，用于排序和分页|
|DC-005|约束|Subagent 树深度|subagent 可递归嵌套（parentExecutionId 链），但 UI 默认只展开两层，深层可按需下钻|
|DC-006|约束|内容存储|turn 的完整 content（可能很长）单独存储在 Content 表或文件中，Turn 行只存引用 ID|
|DC-006a|约束|LLM 输入上下文存储|assistant turn 的 inputMessagesJson 存完整 prompt 消息序列（system/user/assistant/tool_result 消息数组），可能很长（10KB~100KB），存为独立字段或文件引用，Turn 行只存 inputMessagesCount + inputMessagesTokens|
|DC-006b|约束|LLM 输出内容存储|assistant turn 的 contentJson 存完整输出（含 thinking/reasoning blocks），可能很长，存为独立字段或文件引用，Turn 行只存 contentSummary（前 200 字截断）|
|DC-007|约束|时间精度|所有时间字段使用毫秒精度（ISO 8601 + ms），不丢失 sub-second 信息|

---

## §3 需求列表

### 3.1 功能性需求

|编号|类别|名称|描述|优先级|
|-|-|-|-|-|
|FR-001|Turn 拆解|Per-turn token 存储|每个 turn 独立存储 input_tokens、output_tokens、reasoning_tokens、cache_read_tokens、cache_write_tokens 五项|P0|
|FR-002|Turn 拆解|Per-turn 时间存储|每个 turn 独立存储 created_at、completed_at（毫秒精度），派生 duration_ms|P0|
|FR-003|Turn 拆解|Per-turn 工具调用明细|每个 turn 的 tool_calls 存为独立 ToolCall 行（或 turn 内的 JSON 字段），含 name、args_summary、output_summary、duration_ms、state、error_message|P0|
|FR-004|Turn 拆解|Per-turn skill 加载事件|每个 turn 的 skill load/invoke 存为 SkillEvent 行，含 skill_name、version、event_type(load/invoke)、duration_ms、success、agent_name|P0|
|FR-005|Turn 拆解|Turn 角色与 agent 身份|每个 turn 存储 role（user/assistant/tool/subagent/system）和 agent_name、subagent_session_id（如果是 subagent turn）|P0|
|FR-006|Turn 拆解|Turn 内容引用|每个 turn 存储 content_ref 指向 Content 表，而非在 Turn 行内存全文。Content 表存完整 content|P0|
|FR-006a|上下文治理|LLM 输入上下文存储|每个 assistant turn 存储 inputMessagesJson（发给 LLM 的完整消息序列：system prompt + 对话历史 + 工具结果），含每条消息的 role/content/tokenCount；Turn 行额外存 inputMessagesCount 和 inputMessagesTokens（用于折叠态摘要）|P0|
|FR-006b|上下文治理|LLM 输出内容存储|每个 assistant turn 存储 contentJson（模型完整响应，含 thinking/reasoning/text 结构化 blocks）；Turn 行额外存 contentSummary（前 200 字截断摘要，用于折叠态展示）|P0|
|FR-006c|上下文治理|上下文长度指标|每个 assistant turn 存储 contextWindowPct（inputMessagesTokens 占模型 context window 的百分比），用于判断上下文是否过长|P1|
|FR-007|Subagent 存储|Subagent Execution 行写入|写入路径自动从 root session 的 interactions 中识别 subagent 交互，为每个 subagent 创建独立 Execution 行（含 parentExecutionId、rootExecutionId、subagentType、subagentName）|P0|
|FR-008|Subagent 存储|Subagent Turn 行写入|为每个 subagent 的交互创建独立 Turn 行，关联到 subagent Execution|P0|
|FR-009|交互链路|InteractionBridge 行|主 agent 的 task() tool_call 与 subagent session 之间建桥记录：dispatch_turn_id、dispatch_tool_call_id、subagent_execution_id、subagent_first_turn_id、dispatch_args、response_summary、dispatch_time、response_time|P0|
|FR-010|交互链路|交互链路可视化|UI 展示主 agent → subagent 的交互链路图，标注 dispatch 参数、subagent 产出、时间衔接|P0|
|FR-011|Session 总览|Session 聚合信息|Session 列表页显示每个 session 的总 token、总耗时、turn 数、subagent 数、skill 数、模型名|P0|
|FR-012|Session 详情|Turn 时间线视图|Session 详情页展示所有 turn 的时间线（按序号排列），每个 turn 卡片显示 role + token + 耗时 + 工具数 + skill 数|P0|
|FR-013|Turn 详情|Turn 下钻面板|点击 turn 展开详情：完整 content、token 分项柱状图、工具调用列表、skill 加载列表|P0|
|FR-013a|上下文治理|LLM 输入上下文展示（默认折叠）|Turn 详情面板中增加"上下文"区：默认折叠，显示"输入 N 条消息，共 X tokens"；点击展开后逐条展示每条消息（system/user/assistant/tool_result），每条消息也可独立折叠/展开|P0|
|FR-013b|上下文治理|LLM 输出内容展示（默认折叠）|Turn 详情面板中增加"输出"区：默认折叠，显示摘要（前 200 字）+ "输出 X tokens"；点击展开后展示完整内容（markdown 渲染 + thinking 折叠区）|P0|
|FR-014|Subagent 详情|Subagent 独立视图|Subagent 详情页展示该 subagent 的所有 turn，与主 session 同样的粒度和布局|P0|
|FR-015|Subagent 详情|Subagent 与主 session 对比|同一 session 下多个 subagent 的 token/耗时/工具数对比表|P1|
|FR-016|数据采集|本地文件导入——唯一数据入口|CANNBot-Insight MVP 只支持本地文件导入，不支持实时 POST 上报。所有数据通过导入 opencode sessions.db 或 Claude Code JSONL 进入系统|P0|
|FR-017|数据采集|Opencode 格式适配|适配 opencode sessions.db 的表结构（sessions/messages/tools），提取 per-turn 数据入库|P0|
|FR-018|数据采集|本地文件导入——Opencode sessions DB（多 session 选择）|支持读取本地 opencode sessions.db（SQLite）文件；**先展示 session 列表**（含每个 session 的创建时间、第一个用户提示词、turn 数、模型等基本信息）供用户判断和选择；用户选择后再解析入库为 Turn/ToolCall/SkillEvent 等|P0|
|FR-019|数据采集|本地文件导入——Claude Code JSONL|支持读取本地 Claude Code session JSONL 文件（`~/.claude/projects/*/sessions/*.jsonl`），解析其中的消息和工具调用，自动拆解入库|P0|
|FR-020|数据采集|CLI 导入命令|提供 CLI 命令 `cannbot-insight import --source opencode-db --path /path/to/sessions.db`：先列出 DB 中所有 session 供用户选择；`cannbot-insight import --source claude-jsonl --path /path/to/session.jsonl`：直接入库|P0|
|FR-021|数据采集|UI 文件上传入口|UI 提供"导入本地文件"按钮，支持拖拽上传 opencode sessions.db 或 Claude Code JSONL 文件；上传 opencode DB 后弹出 session 选择界面|P1|
|FR-018|性能|Turn 分页查询|Turn 列表支持按 turn_number 分页/范围查询，不一次性加载全部 turn|P1|
|FR-019|搜索|Turn 内搜索|支持在 session 内搜索 turn 内容（关键词、工具名、skill 名）|P2|
### 3.2 非功能性需求

|编号|类别|名称|描述|优先级|
|-|-|-|-|-|
|NFR-001|性能|Turn 查询性能|单 session 100+ turn 时，Turn 列表加载 ≤ 500ms；单 turn 详情加载 ≤ 200ms|P0|
|NFR-002|性能|Subagent 查询性能|单 session 5+ subagent 时，subagent 列表加载 ≤ 300ms|P0|
|NFR-003|存储|DB 体积控制|500 session、平均 50 turn/session + 2 subagent/session 时，DB ≤ 500MB（content 全量存储时需评估）|P1|
|NFR-004|兼容|上报格式兼容|沿用 opencode 的上报格式，无需 agent 端改动|P0|
|NFR-004a|兼容|本地文件格式兼容|opencode sessions.db 格式和 Claude Code JSONL 格式需与当前主流版本兼容；格式变化时通过 adapter 层适配|P0|
|NFR-005|可维护|代码目录隔离|所有代码仅在 cannbot-insight 目录下，无外部代码依赖|P0|

---

## §4 设计决策分析

### 4.1 为何选择 Turn 级独立建表

传统方案将 per-turn 数据锁在 JSON blob 里，全量解析才能回答细粒度问题。CANNBot-Insight 以 Turn 为一等公民建表，每个 LLM turn / tool turn / skill load event / subagent dispatch 均独立成行，支持直接索引和查询。

### 4.2 关键改进点

|维度|传统做法|CANNBot-Insight 方案|改进点|
|-|-|-|-|
|Turn 数据存储|JSON blob（需全量解析）|Turn 独立 DB 行（可索引可查询）|无需全量解析 JSON；支持 per-turn 过滤和聚合|
|Per-turn token|仅在 JSON blob 中|Turn 行含 input/output/reasoning/cache 五项|可直接查"第3轮用了多少token"|
|Per-turn timing|timeInfo 在 JSON blob 中|Turn 行含 createdAt/completedAt/latencyMs/ttftMs|可直接查每轮耗时|
|Subagent Execution|schema 有字段但从未写入|写入路径实际创建 subagent Execution + Turn 行|每个 subagent 有独立聚合指标|
|主↔子交互链路|隐含在 task() tool_call JSON 中|InteractionBridge 显式建桥|可直观查看 dispatch→response 全链路|
|Cost 计算|读取时动态重算，历史 cost 随 pricing 漂移|写入时定格存储|历史 cost 不漂移|
|LLM 输入上下文|无独立存储|Turn 行存 inputMessagesJson + count/tokens/pct|可直接查"第5轮 LLM 收到了什么"|
|LLM 输出内容|存为 content 字段，无分层|Turn 行存 contentJson + contentSummary|默认折叠摘要，按需展开全文|

---

## §5 验收方案

### 5.1 验收准则

|编号|关联能力|维度|描述|验收标准|
|-|-|-|-|-|
|AC-001|FR-001~FR-006|功能|Turn 行拆解|上传一个含 10 turn 的 session 后，DB 中有 10 条 Turn 行，每条含 token 五项分项、时间、role、agent_identity、content_ref|
|AC-001a|FR-006a|功能|LLM 输入上下文存储|assistant turn 的 inputMessagesJson 非空且含完整消息序列；inputMessagesCount/inputMessagesTokens 正确|
|AC-001b|FR-006b|功能|LLM 输出内容存储|assistant turn 的 contentJson 非空且含完整响应；contentSummary 为前 200 字截断|
|AC-002|FR-007~FR-008|功能|Subagent 行拆解|上传一个含 2 subagent（各 5 turn）的 session 后，DB 中有 2 条 subagent Execution 行 + 10 条 subagent Turn 行，parentExecutionId 正确指向 root|
|AC-003|FR-009|功能|交互链路建桥|上述 session 的 InteractionBridge 行正确映射 task() dispatch → subagent response|
|AC-004|FR-010/FR-012|功能|UI 展示|Session 详情页正确显示 turn 时间线 + subagent 卡片 + 交互链路图|
|AC-004a|FR-013a/FR-013b|功能|上下文治理 UI|Turn 详情面板的"上下文"区和"输出"区默认折叠；点击可展开查看 LLM 输入消息序列和完整输出；每条输入消息可独立折叠|
|AC-005|FR-014|功能|Subagent 独立视图|点击 subagent 卡片进入 subagent 详情页，正确展示其 turn 列表|
|AC-006|FR-004|功能|Skill 加载事件|含 skill load 的 session 上传后，SkillEvent 行正确记录 skill_name/version/event_type/duration|
|AC-007|NFR-001|性能|Turn 查询|100 turn session 的 Turn 列表加载 ≤ 500ms|
|AC-008|NFR-002|性能|Subagent 查询|5 subagent session 的 subagent 列表加载 ≤ 300ms|

### 5.2 测试用例

|编号|关联准则|前置条件|操作步骤|预期结果|
|-|-|-|-|-|
|TC-001|AC-001|服务已启动|上传含 10 turn 的 session（无 subagent）|DB 10 条 Turn 行；Session 详情页 10 个 turn 卡片|
|TC-002|AC-002|服务已启动|上传含 1 root + 2 subagent 的 session|DB 3 条 Execution 行（1 root + 2 sub）+ 对应 Turn 行；InteractionBridge 行正确|
|TC-003|AC-003|同上|查看 InteractionBridge 行|task() dispatch 参数与 subagent 首轮内容正确对应|
|TC-004|AC-006|服务已启动|上传含 skill load 的 session|SkillEvent 行含 skill_name/version/load/duration/success|
|TC-005|AC-007|DB 含 100 turn session|打开 Session 详情页|Turn 列表 ≤ 500ms 加载|
|TC-006|AC-004|同 TC-002|查看交互链路图|主→sub 连线正确，标注 dispatch 参数和 response summary|
|TC-007|AC-004a|含 assistant turn 的 session|打开 Turn 详情面板|上下文区默认折叠，显示"输入 5 条消息，共 3000 tokens"；点击展开后逐条显示消息；输出区默认折叠，显示摘要；点击展开后显示完整响应|

### 5.3 交付物定义

|交付物|描述|
|-|-|
|数据模型|Prisma schema：Session、Execution、Turn、ToolCall、SkillEvent、InteractionBridge、Content|
|入库拆解逻辑|Upload route + turn-splitter + subagent-splitter + bridge-builder + inputMessages-reconstructor 服务函数|
|API 路由|Session 列表/详情、Turn 列表/详情（含 inputMessagesJson/contentJson）、Subagent 详情、InteractionBridge 查询|
|UI 页面|Session 列表页、Session 详情页（Turn 时间线 + Subagent 卡片 + 交互链路）、Subagent 详情页、Turn 下钻面板（含 LLM 上下文折叠展示 + LLM 输出折叠展示）|

---

## §6 附录

### 6.1 数据格式参考

Opencode sessions.db 中 RawInteraction 数据结构（CANNBot-Insight 入库拆解的源数据格式）：

```typescript
interface RawInteraction {
  role: string;                    // "user" | "assistant" | "subagent" | "system" | "opencode"
  content: string | object;        // 消息内容
  timestamp: string;               // ISO 8601
  timeInfo?: {
    created: string;               // 创建时间（ms 精度）
    completed: string;             // 完成时间（ms 精度）
  };
  agent?: string;                  // agent 名称
  subagent_name?: string;          // subagent 显示名
  subagent_session_id?: string;    // subagent 的 session ID
  tool_calls?: ToolCall[];         // 工具调用数组
  usage?: {                        // per-interaction token 使用
    total: number;
    input: number;
    output: number;
    reasoning: number;
    cache: { read: number; write: number };
  };
  model?: string;
  modelID?: string;
  providerID?: string;
  latency?: number;                // 秒
  finish_reason?: string;
}
```

每个 RawInteraction 即为 CANNBot-Insight 一个 Turn 行的源数据。tool_calls 内的 skill/load_skill/task 调用额外拆出 SkillEvent/InteractionBridge 行。
