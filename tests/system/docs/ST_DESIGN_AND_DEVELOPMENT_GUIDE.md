# ST 用例设计与开发规范

## 1. ST 架构设计

### 1.1 分层看护策略

cannbot-skills 项目的质量看护体系分为三层：

| 层级 | 看护类型 | 看护对象 | 方式 |
|------|---------|---------|------|
| **第一层：UT** | Plugin / Agent / Skill 规范看护 | 静态结构、格式校验 | 静态看护 |
| **第二层：ST** | Plugin / Agent / Skill 独立能力的正确性、稳定性看护 | 语义评测、行为验证 | 动态看护 |
| **第三层：冒烟** | Cannbot 整体算子生成能力看护 | 端到端业务验证（对接 bench） | 集成看护 |

ST（System Test）处于看护体系的核心位置，负责验证每个 Skill / Agent / Plugin 的独立功能正确性和稳定性。

### 1.2 看护流程

```
第一步：代码检查、Code Check
第二步：UT 检查 & ST 检查
第三步：代码上库
第四步：bench 冒烟任务 Daily 看护
```

ST 用例在代码合入前执行，由 `gate_check.sh` 在 CI 流水线中自动触发。

### 1.3 ST 用例看护维度

设计 ST 用例时需要从以下五个维度综合考虑：

| 看护维度 | 测试目标 | 当前能力 | 设计要点 |
|----------|---------|---------|---------|
| **正向看护** | 显式 / 隐式提示词 → 调用到目标 skill | **具备** | 在 Config 中配置 Distractor skills（干扰技能），验证即使存在多个类似 skill，AI 仍能正确选择和触发目标 skill |
| **负向看护** | 不该调用的提示词 → 不会误调用 skill | 不具备 | 设计边界场景 prompt 验证 skill 不会被错误触发 |
| **正确性看护** | 黑盒场景验证、确保结果正确 | **具备** | 设计典型用户场景，描述预期输出要点，验证 AI 回复语义覆盖 |
| **调用流程看护** | 关键工具被调用、交付件完整输出 | **具备** | 验证 skill 执行过程中关键工具是否被调用、关键文件是否生成 |
| **资源消耗看护** | Token 消耗监控、防止资源浪费 | **具备** | 评测 session 自动检查 Token 消耗合理性（占总分 10 分） |

### 1.4 ST 框架架构

```
tests/system/
├── README.md                        # 框架说明
├── config/
│   └── st-test.config            # skill 扫描路径与白名单配置
├── docs/
│   ├── USER_GUIDE.md                # 框架使用指南
│   └── ST_DESIGN_AND_DEVELOPMENT_GUIDE.md  # 本文档
├── cases/                           # 集中式评测用例（MD 格式）
│   └── <skill_or_team_name>_evals.md # Skill 用 skill_name，Team 用 team_name
├── sandboxes/                       # 用例执行隔离沙箱（自动创建）
│   └── <name>_eval_<id>/
│       ├── .opencode/
│       │   ├── skills/              # skill 符号链接（skill 测试）
│       │   └── opencode.json        # 安全权限配置
│       └── logs/                    # session 日志
├── results/                         # 测试报告输出
├── logs/                            # 运行日志
└── scripts/
    ├── main.py                      # CI 门禁主入口（支持 skill + team）
    ├── conftest.py                  # pytest 配置、skill/team 扫描与 HTML 报告渲染
    ├── test_skill_basic.py          # Phase 1: Skill 静态结构验证
    ├── test_team_basic.py           # Phase 1: Team 静态结构验证
    ├── test_skill_evals.py          # Phase 2: Skill AI 语义评测
    ├── test_team_evals.py           # Phase 2: Team AI 语义评测（复用 skill 验证逻辑）
    ├── evals_parser.py              # MD 格式评测用例解析器（支持 skill/team）
    ├── opencode_runner.py           # opencode CLI 流式封装
    ├── sandbox_manager.py           # 沙箱隔离管理（skill symlink + team init.sh）
    ├── session_stats.py             # Session 数据统计
    ├── run_eval.py                  # pytest 评测命令行启动脚本
    ├── test_opencode_runner.py      # opencode_runner 单元测试
    ├── opencode_runner_examples.py  # opencode_runner 使用示例
    ├── pytest.ini                   # pytest 渲染配置
    └── requirements.txt             # Python 依赖
```

### 1.5 测试执行流程

```
输入：repo_root + changed_files
    │
    ├─ 步骤1：识别受影响的 skills / teams
    │   └─ 从变更文件路径中提取 skill 或 team 名称
    │
    ├─ 步骤2：加载评测用例
    │   └─ 读取 tests/system/cases/<name>_evals.md
    │       （Skill 用 skill_name，Team 用 team_name frontmatter）
    │
    ├─ 步骤3：执行评测 — 逐个 target 进行
    │   ├─ Phase 1: 静态结构验证
    │   │   ├─ Skill: test_skill_basic.py
    │   │   │   ├─ evals.md 存在性、格式合法性、必填字段检查
    │   │   │   ├─ 用例 ID 唯一性、连续递增校验
    │   │   │   └─ SKILL.md 存在性、YAML frontmatter 格式校验
    │   │   ├─ Team: test_team_basic.py
    │   │   │   ├─ evals.md 存在性、格式合法性、必填字段检查
    │   │   │   ├─ 用例 ID 唯一性、连续递增校验
    │   │   │   ├─ AGENTS.md 存在性、frontmatter 格式校验
    │   │   │   ├─ plugin.json 存在性与合法性校验
    │   │   │   └─ init.sh 存在性校验
    │   │   耗时：秒级，无需 AI 调用
    │   │   ⚠️ Phase 1 失败的 target 不进入 Phase 2（需要先修复基础结构问题）
    │   │
    │   └─ Phase 2: AI 语义评测（仅通过 Phase 1 的 target 进入）
    │       ├─ 支持重试：EVAL_EXEC_RETRIES 控制重试次数（默认 1）
    │       ├─ Skill: test_skill_evals.py
    │       │   ├─ 沙箱部署: symlink skill 目录到 .opencode/skills/
    │       │   └─ (以下与 team 共享) ...
    │       ├─ Team: test_team_evals.py
    │       │   ├─ 沙箱部署: 执行 init.sh project opencode <sandbox>
    │       │   └─ (以下与 skill 共享) ...
    │       ├─ 执行 Session：opencode 加载 target，发送 prompt → 收集 AI 回复
    │       ├─ 评测 Session：独立 opencode session 评审回复质量
    │       │   ├─ 信息覆盖度（40 分，≥20 通过）：是否完整覆盖预期要点
    │       │   ├─ 技术准确性（30 分，≥15 通过）：技术信息是否正确
    │       │   ├─ 回复质量（20 分，≥10 通过）：结构清晰、表达简洁
    │       │   └─ Token 消耗（10 分，≥3 通过）：回复长度合理、工具调用高效
    │       │   总分 ≥ 60 且各维度均不低于阈值方为通过
    │       │   评审方式：Agent 通过 Write 工具填写 review-template.md 模板
    │       │   框架通过正则解析模板提取结构化的 Status/Score/维度得分
    │       └─ 模式匹配：检查 expectations 中的 contains/not_contains/file_exists/file_list/file_contains/skill_activated
    │       耗时：分钟级，需要 opencode CLI
    │
    ├─ 步骤4：保存结果
    │   ├─ results/basic_validation.html               # Skill Phase 1 报告
    │   ├─ results/team_basic_validation.html          # Team Phase 1 报告
    ├─ results/ST_validation_report_<ts>.html     # Skill+Team 统一报告（文件名含北京时间戳，类型列区分）
    │   └─ results/<name>_<timestamp>.json             # 结构化结果
    │
    └─ 返回：0（全部通过）/ 1（存在失败）
```

---

### 1.6 测试执行架构：双 Agent 协同

每个 ST 用例的执行涉及两个独立的 AI Agent（均为 opencode CLI 会话），与其各自的评测机制协同工作：

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                         ST 评测用例 (Eval Case)                          │
 │                                                                         │
 │  ┌──────────────┐  ┌──────────────────────────────┐  ┌──────────────┐  │
 │  │    Prompt    │  │       Expected Output         │  │ Expectations │  │
 │  │ (测试问题)    │  │    (语义预期要点描述)           │  │ (精确断言)    │  │
 │  └──────┬───────┘  └─────────────┬────────────────┘  └──────┬───────┘  │
 │         │                        │                          │          │
 └─────────┼────────────────────────┼──────────────────────────┼──────────┘
           │                        │                          │
           ▼                        │                          │
 ┌─────────────────────┐            │                          │
 │   执行 Agent         │            │                          │
 │  (Execution Session) │            │                          │
 │                     │            │                          │
 │  1. 加载目标 skill   │            │                          │
 │  2. 加载干扰 skills   │           │                          │
 │  3. 发送 Prompt      │            │                          │
 │  4. AI 理解 + 工具调用│           │                          │
 │  5. 生成回复/文件     │           │                          │
 │                     │            │                          │
 │  产出:               │            │                          │
 │  ├─ AI 回复文本      │            │                          │
 │  ├─ 工具调用记录      │           │                          │
 │  ├─ 生成的文件        │           │                          │
 │  └─ session 导出JSON │           │                          │
 └─────────┬───────────┘            │                          │
           │                        │                          │
           ▼                        ▼                          ▼
 ┌─────────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
 │  评测 Agent          │  │  模式匹配引擎      │  │ Token 预算检查        │
 │ (Review Session)    │  │                  │  │                      │
 │                     │  │  ├─ [contains]   │  │  读取 session JSON    │
 │  读取 Expected Output│  │  ├─ [not_contains]│  │  计算总 token 消耗    │
 │  读取 AI 回复 + 推理  │  │  ├─ [file_exists]│  │  对比 Max Tokens 上限 │
 │  读取生成的文件(如有)  │  │  ├─ [file_list]  │  │                      │
 │  ├─ [file_contains]│  │                      │
 │                     │  │  └─ [skill_acti- │  │                      │
 │  按评分标准打分:      │  │     vated]       │  │                      │
 │  ├─ 信息覆盖度 (40分) │  └────────┬─────────┘  └──────────┬───────────┘
 │  ├─ 技术准确性 (30分) │           │                        │
 │  ├─ 回复质量   (20分) │           │                        │
 │  ├─ Token消耗  (10分) │           │                        │
 │  └─ 总分 ≥ 60 通过    │           │                        │
 └─────────┬───────────┘            │                        │
           │                        │                        │
           ▼                        ▼                        ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                          最终判定 (Test Result)                          │
 │                                                                         │
 │  通过条件: 评测 Agent 总分 ≥ 60  AND  各维度均 ≥ 最低阈值  AND  所有 Expectations 通过  AND  Token 未超限 │
 │                                                                         │
 │  结果: Passed / Failed + 详细原因                                        │
 └─────────────────────────────────────────────────────────────────────────┘
```

**角色说明**：

| 角色 | 能力 | 职责 |
|------|------|------|
| **执行 Agent** | 通用 AI 能力 + skill 知识 | 模拟真实用户与 skill 交互，执行测试任务 |
| **评测 Agent** | 评审能力（另一个 AI 实例） | 以评分标准为框架，判断 AI 回复是否覆盖预期要点 |
| **模式匹配引擎** | 确定性字符串/文件匹配 | 精确断言：关键词、文件存在性、skill 激活检查 |
| **Token 预算检查** | 数值比较 | 防止资源滥用，保障测试经济性 |

**关键特性**：
- **双 Agent 解耦**：执行和评测使用独立的 opencode 会话，避免评审偏差污染执行过程
- **语义 + 确定性双通道**：Expected Output 走语义评测（灵活），Expectations 走精确匹配（严格），互补验证
- **正向看护**：`[skill_activated]` 从 session 导出 JSON 中提取工具调用记录，不依赖评审 Agent
- **沙箱隔离**：每个用例有独立沙箱，skill 通过软链接部署（team 通过 init.sh 部署），干扰 skill 同等待遇
- **评审模板化**：评审 Agent 通过 Write 工具填写 `review-template.md` 模板，框架正则解析提取状态/评分/各维度得分，不再依赖 JSON 输出格式
- **重试机制**：通过 `EVAL_EXEC_RETRIES` 环境变量控制评测用例的重试次数（默认 1，即不重试）
- **模型 Token 预算**：支持 `Max Tokens (<model>)` 语法按模型指定 Token 上限，与 `--eval-model` 参数或 `EVAL_MODEL` 环境变量配合使用

---

### 2.1 用例文件组织

ST 用例以 **MD 文件** 的形式存放在 `tests/system/cases/` 目录下，命名规则为：

```
tests/system/cases/<skill_name>_evals.md
```

> **注意**：用例文件不再放在各 skill 目录下的 `evals/evals.json`，而是集中管理在 `cases/` 目录，以便统一维护和 CI 检测。

### 2.2 用例文件格式

#### 2.2.1 基本结构

每个用例文件由 **YAML frontmatter** 和多个 **Markdown 用例块** 组成：

```markdown
---
skill_name: <skill名称>
eval_mode: text          # 评测模式，可选值：text（默认）/ file_based
---

# Case 1: <用例名称>

## Prompt

<发送给 AI 的测试问题>

## Expected Output

<对 AI 回复的语义预期，描述应覆盖的关键要点>

## Expectations

- [contains] <期望包含的内容>
- [not_contains] <不应包含的内容>
- [file_exists] <期望生成的文件路径>
- [file_contains] <文件路径或glob> : "<文本1>";"<文本2>"

---

# Case 2: <用例名称>

...
```

#### 2.2.2 Frontmatter 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `skill_name` | Skill 必填 | 目标 skill 名称，需与 SKILL.md 中的 `name` 字段一致。与 `team_name` 二选一 |
| `team_name` | Team 必填 | 目标 team 名称，需与 plugin.json 中的 `name` 字段一致。与 `skill_name` 二选一 |
| `eval_mode` | 否 | 评测模式，默认 `text`。`file_based` 用于需要验证沙箱中生成文件的场景 |

> **注意**：`skill_name` 和 `team_name` 只能设置一个。解析器会根据 frontmatter 中存在的字段自动确定 target 类型（`target_type: "skill"` 或 `"team"`）。

#### 2.2.3 用例字段说明

每个 `# Case N: 用例名称` 块包含以下字段：

| 字段 | 必填 | 说明 |
|------|------|------|
| `## Prompt` | **是** | 发送给 AI 的测试问题，应模拟真实用户场景 |
| `## Expected Output` | **是** | 对 AI 回复的语义预期。**不要求逐字匹配**，描述应覆盖的关键要点即可 |
| `## Config` | 否 | 用例级配置，支持覆盖 eval_mode（如 `- Eval Mode: file_based`）。可用字段见下方 Config 字段说明 |
| `## Expectations` | 否 | 模式匹配规则列表，用于精确断言 |

**Config 字段说明：**

| Config Key | 说明 |
|------------|------|
| `Eval Mode` | 覆盖用例级评测模式：`text`（默认）/ `file_based` |
| `Max Tokens` | Token 消耗硬上限，超过则测试失败 |
| `Max Tokens (<model>)` | 按模型指定 Token 上限，如 `Max Tokens (deepseek-v4-flash): 140000`。Phase 2 执行时自动从 session 导出数据检测模型名称并匹配对应预算。可通过 `--eval-model` 或 `EVAL_MODEL` 环境变量指定模型 |
| `Distractor skills` | 正向看护：分号分隔的干扰 skill 名称列表。这些 skill 会被部署到沙箱中，验证 AI 在多个 skill 同时可用时仍能正确选择目标 skill。示例: `cann-env-setup;ascendc-task-focus;npu-arch` |
| `Ascend Platform` | 用例适用的昇腾平台，分号分隔可多选，如 `A2;A5`。配合 `--ascend-platform` 参数按平台过滤。**未配置此字段的用例在任何平台下均不执行** |
| `Disabled` | 设为 `true` 则跳过该用例的执行（Phase 2 中显示为 SKIPPED）|
| `Timeout` | 用例执行超时时间（秒）。正整数，未配置时默认 600s。适用于需要更长执行时间的复杂场景，如 `Timeout: 900` |
| `Truncate Length` | AI 回复传递给评审 Agent 时截断长度（字符数）。默认 30000。当 AI 回复较长时（如包含大段代码），评审 Agent 可能因回复截断看不到完整内容而误判。可按需增大，如 `Truncate Length: 60000` |
| `覆盖度阈值` / `准确性阈值` / `质量阈值` / `Token阈值` | 按维度覆盖默认通过阈值。覆盖度默认 20/40，准确性 15/30，质量 10/20，Token 3/10。如 `覆盖度阈值: 25` |

### 2.3 Expectations 类型详解

#### 2.3.1 contains — 文本包含检查

检查 AI **最终回复**（`ai_text`）中是否包含指定字符串。

```markdown
## Expectations

- [contains] npu-smi info
- [contains] set_env.sh
```

#### 2.3.2 not_contains — 文本排除检查

检查 AI **最终回复文本**中是否**不**包含指定字符串。用于负向看护场景。

> **注意**：只检查 AI 对用户的最终回复（`ai_text`），不检查工具调用过程中的参考文档内容和中间输出。`full_output` 中包含 skill 加载时的参考文档内容，这些文档中的术语不应被视为 AI 的回复内容。

```markdown
## Expectations

- [not_contains] - [ ]
- [not_contains] ## 待办事项
```

#### 2.3.3 file_exists — 文件存在检查

检查指定文件是否被创建或修改。搜索顺序：`sandbox/<path>` → `sandbox/skill/<path>` → `skill_dir/<path>`。

```markdown
## Expectations

- [file_exists] todo.md
- [file_exists] src/main.cpp
```

#### 2.3.4 file_list — 文件列表匹配检查

检查沙箱中是否存在匹配 glob pattern 的文件。适用于需要验证生成文件但不确定确切路径的场景。

```markdown
## Expectations

- [file_list] *.md
- [file_list] src/*.py
```

#### 2.3.5 skill_activated — 程序化 skill 激活检查

**程序化**检查 AI 执行过程中是否加载了指定 skill。不依赖 AI 评审模型，直接从 tool_use 事件中精确匹配 skill 名称。用于正向看护场景。

```markdown
## Expectations

- [skill_activated] cann-env-setup
```

> **注意**：`skill_activated` 是确定性检查，不受评审模型主观判断影响。即使 AI 回复在技术上是正确的，如果它加载了错误的 skill（或没有加载任何 skill），此断言会直接导致测试失败。


#### 2.3.6 file_contains — 文件内容包含检查

检查沙箱中匹配 glob 路径的文件是否包含所有指定文本。支持 glob 通配符（`*`、`?`、`[]`），
匹配到多个文件时，只要有一个文件包含所有 pattern 即判定通过。

```markdown
## Expectations

- [file_contains] src/kernel/*.asc : "__global__";"LocalTensor";"DataCopy"
- [file_contains] docs/DESIGN.md : "数据流";"Buffer 规划"
```

**格式说明**：
- 路径部分：文件路径或 glob 模式（相对于沙箱根目录）
- 分隔符：` : `（空格-冒号-空格）
- 文本模式：双引号包裹，多个用英文分号 `;` 分隔
- 验证逻辑：所有列出的文本模式必须在同一个文件中全部出现才判定通过

### 2.4 评测模式

#### 2.4.1 text 模式（默认）

适用于大多数场景，AI 回复以文本方式输出。评测流程：
1. 执行 session：向 skill 发送 prompt，收集 AI 文本回复
2. 评测 session：独立 session 基于 Expected Output 评审回复质量
3. 模式匹配：检查 expectations 中的 contains/not_contains/file_contains 规则

#### 2.4.2 file_based 模式

适用于验证 AI 生成文件的场景（如生成代码、配置文件等）。评测流程：
1. 系统自动向 prompt 末尾追加 `FILE_BASED_HINT`（要求 AI 列出创建/修改的文件清单并说明用途，不输出完整文件内容）
2. 执行 session：AI 在沙箱中创建/修改文件
3. 评测 session：独立 session 读取沙箱中的生成文件，基于文件内容评审质量
4. 模式匹配：检查 expectations 中的 file_exists/file_list/file_contains 规则
5. `collect_generated_files()` 收集沙箱中新增的文件（排除 logs/ .opencode/ 和源 skill 中已存在的文件）

**使用 file_based 模式的用例示例：**

```markdown
# Case 1: 创建任务计划

## Config
- Eval Mode: file_based

## Prompt

我需要开发一个Add算子的ST测试，大概需要5个步骤：需求分析、API调研、方案设计、代码实现、编译测试。请帮我创建一个todo.md来管理这个任务。

## Expected Output

创建的todo.md文件应包含：任务标题、目标（1-2句话）、待办事项（5个步骤用- [ ]勾选框列出）、进度（0/5），内容结构完整

## Expectations

- [file_exists] todo.md
- [file_list] *.md
```

### 2.5 用例设计原则

#### 2.5.1 场景覆盖

设计用例时应覆盖以下场景类型：

| 场景类型 | 说明 | 示例 |
|---------|------|------|
| **典型场景** | 最常见的用户使用方式 | "我想检查NPU驱动是否已安装" |
| **边界场景** | 边界条件或特殊输入 | "我只需要快速查询命令，不需要创建任务计划" |
| **错误场景** | 用户描述不完整或有歧义 | "安装CANN需要哪些依赖？"（缺少平台/版本信息） |
| **复杂场景** | 涉及多步骤、多文件的操作 | "开发包含6个阶段的算子项目" |

#### 2.5.2 Prompt 编写原则

- **模拟真实用户**：使用自然语言，贴近用户实际提问方式
- **场景明确**：提供足够的上下文信息，让 AI 能理解任务意图
- **简短精炼**：1-3 句话描述清楚场景即可，无需冗长的技术背景

#### 2.5.3 Expected Output 编写原则

- **描述语义要点**：写"回复应说明使用 npu-smi info 命令检查驱动"，不写"回复必须包含 `npu-smi info`"
- **聚焦核心信息覆盖**：描述必须覆盖的关键信息点，让评测模型判断 AI 是否遗漏
- **避免过于精确的措辞约束**：AI 输出是非确定性的，语义等价即可
- **技术准确性是底线**：期望的要点必须与官方文档或社区标准一致

#### 2.5.4 Expectations 使用建议

- `contains` / `not_contains` 用于精确的关键词 / 结构断言，是对 expected_output 语义评测的**补充**，不是替代
- 不要用 `contains` 逐一验证 expected_output 中的每个要点——语义评测已经做了这件事
- `contains` 适用于验证关键 API 名称、命令、文件结构标记等
- `not_contains` 适用于负向验证（如验证不应创建任务计划的场景）

### 2.6 用例 ID 规范

- 用例 ID 从 1 开始，**连续递增**
- 不允许跳号、重复
- ID 顺序应与用例逻辑顺序一致（从简单到复杂、从核心到边缘）
- Phase 1 静态验证会自动检查 ID 唯一性和连续性

### 2.7 完整用例示例

以下是一个 Skill 的完整用例文件示例（`tests/system/cases/cann-env-setup_evals.md`）：

```markdown
---
skill_name: cann-env-setup
---

# Case 1: 检查NPU驱动安装命令

## Prompt

我有一台昇腾服务器，想检查NPU驱动是否已安装，应该用什么命令？

## Expected Output

回复应说明使用 npu-smi info 命令检查驱动，并解释如何根据命令输出判断驱动是否已安装

## Expectations

- [contains] npu-smi info

---

# Case 2: 配置环境变量永久生效

## Prompt

我已经用离线安装包安装完CANN Toolkit和Ops，现在需要配置环境变量使其永久生效，应该怎么做？

## Expected Output

回复应说明如何配置环境变量实现永久生效：通过 source set_env.sh 命令并将其写入 ~/.bashrc 文件

## Expectations

- [contains] source
- [contains] set_env.sh

---

# Case 3: 验证安装是否成功

## Prompt

CANN安装完成后，如何验证安装是否成功？

## Expected Output

回复应提供至少一种验证 CANN 安装是否成功的方法

## Expectations

- [contains] npu-smi

---

# Case 4: 离线安装顺序

## Prompt

我需要安装CANN，但服务器没有网络，只有离线安装包。请问安装顺序是什么？

## Expected Output

回复应明确说明安装顺序：先安装 Toolkit 包，再安装 Ops 包

## Expectations

- [contains] toolkit

---

# Case 5: Conda在线安装CANN

## Prompt

我想用conda方式在线安装CANN，需要Python 3.10环境，具体步骤是什么？

## Expected Output

回复应说明 conda 安装 CANN 的完整步骤：创建 conda 环境、添加昇腾 conda 源、使用 conda install 安装 cann-toolkit 和 ops 包

## Expectations

- [contains] conda

---

# Case 6: 安装前依赖检查

## Prompt

安装CANN之前需要检查哪些依赖？

## Expected Output

回复应说明安装 CANN 前需要检查的依赖项，至少包括 Python 和 pip 是否已安装

## Expectations

- [contains] Python
- [contains] pip
```

#### 正向看护示例

当需要验证 AI 在多个类似 skill 存在时仍能正确选择目标 skill 时，使用 `Distractor skills` 配置：

```markdown
# Case 7: 正向看护-多skill环境下正确触发目标skill

## Config
- Max Tokens: 100000
- Distractor skills: ascendc-runtime-debug;ascendc-task-focus;npu-arch;ascendc-docs-search

## Prompt

我有一台昇腾服务器，想检查NPU驱动是否已安装，应该用什么命令？

## Expected Output

回复应说明使用 npu-smi info 命令检查驱动，并解释如何根据命令输出判断驱动是否已安装。应成功激活并使用了 cann-env-setup skill。

## Expectations

- [skill_activated] cann-env-setup
- [contains] npu-smi info
```

### 2.8 用例开发工作流

```
1. 确定目标 Skill
       │
2. 分析 Skill 功能
   ├─ 阅读 SKILL.md 了解 skill 的核心功能
   ├─ 阅读 references/ 了解详细的领域知识
   └─ 确定 skill 的适用场景和边界
       │
3. 设计用例场景（参考 1.3 看护维度）
   ├─ 正向看护：典型使用场景
   ├─ 负向看护：不应触发的场景
   └─ 正确性看护：关键信息覆盖验证
       │
4. 编写 evals.md 文件
   ├─ 填写 frontmatter（skill_name）
   ├─ 编写各个 Case（Prompt + Expected Output + Expectations）
   └─ 检查 ID 连续递增、格式正确
       │
5. 本地验证（开发调试）
   ├─ Phase 1: python -m pytest tests/system/scripts/test_skill_basic.py -v -k "<skill>"
   └─ Phase 2: python -m pytest tests/system/scripts/test_skill_evals.py --skill <skill> -v
       │
6. 提交到 tests/system/cases/ 目录
       │
7. CI 自动验证（PR 时 gate_check.sh 自动触发）
```

### 2.9 配置新的 Skill / Team 扫描路径

如需为新 Skill 目录添加 ST 看护，在 `tests/system/config/st-test.config` 中配置：

```yaml
skill_dirs:
  - "ops"
  - "graph"
  - "model"
  - "ops-lab"              # 新增

skill_whitelist:            # 白名单：仅这些 skill 触发评测（为空表示全部生效）
  - "ascendc-task-focus"
  - "cann-env-setup"

team_dirs:                  # Team 扫描目录
  - "plugins-official"
  - "plugins-community"

team_whitelist:             # Team 白名单：仅这些 team 触发评测
  - "ops-direct-invoke"
```

> **注意**：`skill_whitelist` 非空时仅列出的 skill 会被评测。如果要新增 skill 的白名单支持，需要同时添加到此列表。Team 的白名单机制同理。

### 2.10 常见问题

**Q1: expected_output 检查持续失败？**

检查 `expected_output` 是否过于严格——不要描述"AI 应该说什么话"，描述"AI 回复应覆盖哪些要点"。

**Q2: 评测 session 返回"无法解析判定结果"？**

该问题已在 2026-06 版本解决：评审机制已从易出错的 JSON 解析改为 **review-template.md 模板化方案**。评审 Agent 通过 Write 工具填写沙箱中的 `review-template.md` 模板，框架通过正则提取结构化评审结果。

如仍有问题，检查 `logs/<skill>_case_X_review_ses.json` 中的原始评测输出，确认 `review-template.md` 是否被正确填写。

**Q3: 如何给一个 Skill 新增第一个 ST 用例？**

1. 在 `tests/system/cases/` 下创建 `<skill_name>_evals.md`
2. 确保 `config/st-test.config` 中的 `skill_dirs` 包含该 skill 所在目录
3. 如果启用了 `skill_whitelist`，将 skill 名称加入白名单
4. 运行 Phase 1 静态验证确认格式正确

**Q4: file_based 模式和 text 模式如何选择？**

- 需要验证 AI 生成的文件内容（代码、配置、文档等）→ 使用 `file_based`
- 只需要验证 AI 文本回复的正确性 → 使用 `text`（默认）

**Q5: 如何给 Team 新增第一个 ST 用例？**

与 Skill 类似，但注意：
1. evals.md 的 frontmatter 使用 `team_name` 而非 `skill_name`
2. Phase 1 额外校验 AGENTS.md、plugin.json、init.sh 的存在性
3. 沙箱部署方式不同：Team 通过执行 `init.sh project opencode <sandbox>` 部署
4. 确保 `config/st-test.config` 的 `team_dirs` 和 `team_whitelist` 包含该 team

**Q6: 偶发性评测执行失败如何排查？**

设置环境变量 `EVAL_EXEC_RETRIES` 启用重试（默认 1）：
```bash
EVAL_EXEC_RETRIES=3 python -m pytest tests/system/scripts/test_skill_evals.py --skill cann-env-setup -v
```
重试会重新执行 opencode session 并重新评审，适用于网络抖动或 AI 服务器偶发错误。

---

### 3.1 ops/ — Ascend C 算子开发（23 个）

| 序号 | Skill 名称 | 功能描述 | 路径 |
|------|-----------|---------|------|
| 1 | aiss-tiling-solver | 使用 AISS-TilingSolver 工具自动求解 Ascend C 算子最优 Tiling 参数 | `ops/aiss-tiling-solver` |
| 2 | ascendc-api-best-practices | Ascend C API 使用最佳实践，提供算术、归约、数据搬运、Buffer 管理等 API 正确用法 | `ops/ascendc-api-best-practices` |
| 3 | ascendc-blaze-best-practice | Matmul/Cube/GEMM/BMM 单算子直调生成（Blaze/tensor_api 路径），覆盖模板选型、改造、Tiling 及排错 | `ops/ascendc-blaze-best-practice` |
| 4 | ascendc-code-review | Ascend C 代码检视，基于假设检验方法论进行安全规范检视 | `ops/ascendc-code-review` |
| 5 | ascendc-crash-debug | Ascend C 算子卡死/崩溃调试，处理 hang/crash/deadlock/plog 解析 | `ops/ascendc-crash-debug` |
| 6 | ascendc-direct-invoke-template | Kernel 直调工程模板，用于创建 Ascend C Kernel 直调工程项目 | `ops/ascendc-direct-invoke-template` |
| 7 | ascendc-direct-invoke-to-registry-invoke | Kernel 直调形式改造为自定义算子工程 | `ops/ascendc-direct-invoke-to-registry-invoke` |
| 8 | ascendc-docs-gen | Ascend C 算子文档写作，提供需求分析、设计、接口文档等标准模板 | `ops/ascendc-docs-gen` |
| 9 | ascendc-docs-search | Ascend C 开发资源检索，优先查本地索引、缺失时查在线文档 | `ops/ascendc-docs-search` |
| 10 | ascendc-env-check | Ascend C 算子开发环境检查，查询 NPU 设备信息、CANN 环境配置 | `ops/ascendc-env-check` |
| 11 | ascendc-performance-best-practices | Ascend C 算子性能优化最佳实践库，按算子族组织优化经验 | `ops/ascendc-performance-best-practices` |
| 12 | ascendc-precision-debug | Ascend C 算子精度调试，输出异常、精度验证失败、FP16 精度等诊断 | `ops/ascendc-precision-debug` |
| 13 | ascendc-regbase-best-practice | DAV_3510 RegBase 算子 API 约束确认、实现结构、常见陷阱排查 | `ops/ascendc-regbase-best-practice` |
| 14 | ascendc-registry-invoke-template | 完整自定义算子工程模板，标准工程结构、UT/ST 样例、多芯片架构参考 | `ops/ascendc-registry-invoke-template` |
| 15 | ascendc-registry-invoke-to-direct-invoke | 自定义算子工程中的 kernel 模板改造为 `<<<>>>` 直调形式 | `ops/ascendc-registry-invoke-to-direct-invoke` |
| 16 | ascendc-runtime-debug | Ascend C 算子运行时错误调试，aclnn 错误码、plog 日志解析 | `ops/ascendc-runtime-debug` |
| 17 | ascendc-st-design | Ascend C 算子系统测试（ST）设计，参数定义、测试因子提取、用例生成 | `ops/ascendc-st-design` |
| 18 | ascendc-task-focus | 任务聚焦与注意力管理，通过 todo.md 保持长任务焦点 | `ops/ascendc-task-focus` |
| 19 | ascendc-tiling-design | Ascend C 算子 Tiling 设计指南，多核切分、UB 切分、Buffer 规划 | `ops/ascendc-tiling-design` |
| 20 | ascendc-ut-develop | Ascend C 算子 UT 开发与覆盖率增强，补充测试用例、生成覆盖率报告 | `ops/ascendc-ut-develop` |
| 21 | ascendc-whitebox-design | Ascend C 算子白盒测试用例生成，参数枚举、三档覆盖级别 | `ops/ascendc-whitebox-design` |
| 22 | cann-env-setup | 昇腾 NPU CANN 安装与环境配置指导 | `ops/cann-env-setup` |
| 23 | npu-arch | Ascend NPU 架构知识查询，芯片型号映射、架构代际、条件编译策略 | `ops/npu-arch` |

### 3.2 ops/ — 辅助工具（5 个）

| 序号 | Skill 名称 | 功能描述 | 路径 |
|------|-----------|---------|------|
| 24 | ops-precision-standard | 算子精度标准，各 dtype 精度比对标准（atol/rtol） | `ops/ops-precision-standard` |
| 25 | ops-profiling | NPU 性能采集与分析，采集算子性能数据、定位瓶颈 | `ops/ops-profiling` |
| 26 | ops-simulator | NPU 仿真器使用指导，精度仿真、性能仿真、流水线分析 | `ops/ops-simulator` |
| 27 | torch-ascendc-op-extension | 通过 TORCH_LIBRARY 将 Ascend C 直调工程对接 PyTorch | `ops/torch-ascendc-op-extension` |
| 28 | torch-ops-profiler | 使用 torch_npu.profiler 维护 JSONL 用例并输出性能对比报告 | `ops/torch-ops-profiler` |

### 3.3 ops/ — Catlass 算子开发（3 个）

| 序号 | Skill 名称 | 功能描述 | 路径 |
|------|-----------|---------|------|
| 29 | catlass-op-design | Catlass 算子方案设计 | `ops/catlass-op-design` |
| 30 | catlass-op-develop | Catlass 算子代码实现与测试 | `ops/catlass-op-develop` |
| 31 | catlass-op-perf-tune | Catlass 算子性能调优 | `ops/catlass-op-perf-tune` |

### 3.4 ops/ — PyPTO 算子开发（8 个）

| 序号 | Skill 名称 | 功能描述 | 路径 |
|------|-----------|---------|------|
| 32 | pypto-api-explore | 探索 PyPTO API，提供 API 映射、约束检查和 Tiling 需求分析 | `ops/pypto-api-explore` |
| 33 | pypto-golden-generate | 生成 PyTorch golden 参考实现作为精度验证基准 | `ops/pypto-golden-generate` |
| 34 | pypto-intent-understand | PyPTO 算子需求意图理解，将自然语言描述转化为结构化需求文档 | `ops/pypto-intent-understand` |
| 35 | pypto-op-design | 设计 PyPTO 算子实现方案，生成 DESIGN.md | `ops/pypto-op-design` |
| 36 | pypto-op-develop | 编写 PyPTO 算子实现，生成完整可运行代码与测试 | `ops/pypto-op-develop` |
| 37 | pypto-op-perf-tune | PyPTO 算子性能分析和自动调优，用例执行、数据采集、分步骤调优（含 4 个子技能：perf-analyzer / tune-frontend / tune-incore / tune-swimlane） | `ops/pypto-op-perf-tune` |
| 38 | pypto-precision-compare | PyPTO 算子精度对比，文件保存和二分对比两种方法 | `ops/pypto-precision-compare` |
| 39 | pypto-precision-debug | PyPTO 算子精度调试，用户代码层面语法逻辑检查和规避方法 | `ops/pypto-precision-debug` |

### 3.5 ops/ — Triton 算子开发（5 个）

| 序号 | Skill 名称 | 功能描述 | 路径 |
|------|-----------|---------|------|
| 40 | triton-latency-optimizer | Triton-Ascend 算子延迟优化 | `ops/triton-latency-optimizer` |
| 41 | triton-op-coding | Triton-Ascend 算子代码生成 | `ops/triton-op-coding` |
| 42 | triton-op-designer | Triton-Ascend 算子方案设计 | `ops/triton-op-designer` |
| 43 | triton-op-verifier | Triton-Ascend 算子验证 | `ops/triton-op-verifier` |
| 44 | triton-task-extractor | Triton-Ascend 任务提取 | `ops/triton-task-extractor` |

### 3.6 graph/ — PyTorch 图模式（6 个）

| 序号 | Skill 名称 | 功能描述 | 路径 |
|------|-----------|---------|------|
| 45 | torch-custom-ops-guide | 自定义算子入图完整指南，Eager 算子适配 npugraph_ex 图模式 | `graph/torch-custom-ops-guide` |
| 46 | torch-npugraph-ex-compile-error-diagnosis | npugraph_ex 编译期报错诊断，TorchDynamo/FX/AOTAutograd 阶段排查 | `graph/torch-npugraph-ex-compile-error-diagnosis` |
| 47 | torch-npugraph-ex-dfx-triage | npugraph_ex DFX 问题分诊入口，采集+分类+路由到专科 sub-skill | `graph/torch-npugraph-ex-dfx-triage` |
| 48 | torch-npugraph-ex-knowledge | npugraph_ex（aclgraph）模式使用指南，Capture & Replay 方式 | `graph/torch-npugraph-ex-knowledge` |
| 49 | torch-npugraph-ex-runtime-error-diagnosis | npugraph_ex 运行时报错诊断，aclnn/HCCL/stream/OOM 排查 | `graph/torch-npugraph-ex-runtime-error-diagnosis` |
| 50 | torch-npugraph-ex-template | npugraph_ex 模式 MRE 代码模板，标准编译模板和缓存编译模板 | `graph/torch-npugraph-ex-template` |

### 3.7 model/ — 模型推理优化（11 个）

| 序号 | Skill 名称 | 功能描述 | 路径 |
|------|-----------|---------|------|
| 51 | model-infer-fusion | 昇腾 NPU 模型推理融合算子优化，识别可替换为 torch_npu 融合算子的计算模式 | `model/model-infer-fusion` |
| 52 | model-infer-graph-mode | 昇腾 NPU 模型推理图模式适配，torch.compile/npugraph_ex/GE 图模式 | `model/model-infer-graph-mode` |
| 53 | model-infer-kvcache | 昇腾 NPU KVCache 优化，连续缓存、PagedAttention、MLA 压缩缓存 | `model/model-infer-kvcache` |
| 54 | model-infer-migrator | 昇腾 NPU 模型推理适配与部署基线，适配到 ModelRunner 推理框架 | `model/model-infer-migrator` |
| 55 | model-infer-multi-stream | 昇腾 NPU 多流整网优化，双流、stream overlap、TorchAir 多流改造 | `model/model-infer-multi-stream` |
| 56 | model-infer-parallel-analysis | 昇腾 NPU 推理并行策略分析，推荐最优 TP/EP/DP 并行配置 | `model/model-infer-parallel-analysis` |
| 57 | model-infer-parallel-impl | 昇腾 NPU 推理并行切分实施，并行线性层、MoE 并行、通信组创建 | `model/model-infer-parallel-impl` |
| 58 | model-infer-precision-debug | 昇腾 NPU 推理精度问题诊断，KVCache/FlashAttention 精度排查 | `model/model-infer-precision-debug` |
| 59 | model-infer-prefetch | 昇腾 NPU 推理权重预取优化，torch_npu.npu_prefetch 特性 | `model/model-infer-prefetch` |
| 60 | model-infer-runtime-debug | 昇腾 NPU 推理运行时错误诊断，aicore timeout/HCCL/OOM/算子约束 | `model/model-infer-runtime-debug` |
| 61 | model-infer-superkernel | 昇腾 NPU SuperKernel 算子二进制融合技术，ge_graph 模式/decode 阶段 | `model/model-infer-superkernel` |

### 3.8 infra/ — GitCode 协作工具（4 个）

| 序号 | Skill 名称 | 功能描述 | 路径 |
|------|-----------|---------|------|
| 62 | gitcode-issue-gen | 根据 PR 代码变更生成关联 Issue，自动选用模板并回写 PR 描述 | `infra/gitcode-issue-gen` |
| 63 | gitcode-issue-handler | GitCode Issue 端到端处置，自动判断 PR 路径或 Comment 路径 | `infra/gitcode-issue-handler` |
| 64 | gitcode-pr-handler | 重新生成符合约定式提交规范的 PR 标题与描述 | `infra/gitcode-pr-handler` |
| 65 | gitcode-toolkit | GitCode 协作通用基础参考（API/Token/URL/工作流等，内部使用） | `infra/gitcode-toolkit` |

### 3.9 ops-lab/ — 实验性模块（6 个）

| 序号 | Skill 名称 | 功能描述 | 路径 |
|------|-----------|---------|------|
| 66 | cuda2ascend-simt | CUDA 算子迁移到 Ascend C SIMT，支持 standalone sample / torch_npu / pybind 三类交付形态 | `ops-lab/cuda2ascend-simt` |
| 67 | tilelang-api-best-practices | TileLang Ascend API 使用最佳实践，内存分配、数据搬运、矩阵计算等 | `ops-lab/tilelang/skills/tilelang-api-best-practices` |
| 68 | tilelang-op-design | TileLang-Ascend 算子设计文档生成，编程模式选型、内存层级规划 | `ops-lab/tilelang/skills/tilelang-op-design` |
| 69 | tilelang-op-developer | 基于设计文档生成 TileLang-Ascend 算子实现代码与测试 | `ops-lab/tilelang/skills/tilelang-op-developer` |
| 70 | tilelang-programming-model-guide | TileLang Ascend Developer/Expert 模式选择与 pass_configs 配置指南 | `ops-lab/tilelang/skills/tilelang-programming-model-guide` |
| 71 | tilelang-review | TileLang NPU kernel 代码格式检查与自动修复（ruff/clang-format） | `ops-lab/tilelang/skills/tilelang-review` |

### 3.10 plugins-official/ — 插件内置 Skill（2 个）

| 序号 | Skill 名称 | 功能描述 | 路径 |
|------|-----------|---------|------|
| 73 | asc-api-ut-gen | Ascend C API 单元测试生成，分支覆盖分析、参数化测试设计 | `plugins-official/ops-registry-invoke/asc-devkit/.agent/skills/asc-api-ut-gen` |
| 74 | asc-npu-arch | Ascend NPU 架构知识，芯片型号、NpuArch、SocVersion、条件编译 | `plugins-official/ops-registry-invoke/asc-devkit/.agent/skills/asc-npu-arch` |

### 3.10 plugins-community/ — 社区贡献 Skill（2 个）

| 序号 | Skill 名称 | 功能描述 | 路径 |
|------|-----------|---------|------|
| 75 | ops-easyasc-dsl | EasyAsc DSL 到 AscendC 的工作流，编写/调试/验证 Ascend NPU kernel | `plugins-community/ops-easyasc-dsl/skill` |
| 76 | tilelang-op-orchestrator | TileLang-Ascend 算子开发编排器，集成设计/开发/Review 全流程 | `plugins-community/tilelang-op-orchestrator` |

### 3.11 plugins-official/ — 官方 Team 插件（7 个）

| 序号 | Team 名称 | 功能描述 | 路径 |
|------|-----------|---------|------|
| 77 | ops-direct-invoke | Ascend C Kernel 直调算子开发 Team，含完整工作流（设计→实现→审查→性能验收） | `plugins-official/ops-direct-invoke` |
| 78 | ops-registry-invoke | Ascend C 自定义算子开发（算子仓库模式） | `plugins-official/ops-registry-invoke` |
| 79 | pypto-op-orchestrator | PyPTO 算子端到端开发编排 Team | `plugins-official/pypto-op-orchestrator` |
| 80 | triton-op-generator | Triton-Ascend 算子代码生成与优化 Team | `plugins-official/triton-op-generator` |
| 81 | torch-compile | PyTorch torch.compile 图模式编排 Team | `plugins-official/torch-compile` |
| 82 | model-infer-optimize | NPU 模型推理端到端优化 Team | `plugins-official/model-infer-optimize` |
| 83 | ops-code-reviewer | Ascend C 代码审查 Team | `plugins-official/ops-code-reviewer` |

### 3.12 plugins-community/ — 社区 Team 插件（1 个）

| 序号 | Team 名称 | 功能描述 | 路径 |
|------|-----------|---------|------|
| 84 | tilelang-op-orchestrator | TileLang-Ascend 算子开发编排器，集成设计/开发/Review 全流程 | `plugins-community/tilelang-op-orchestrator` |

---

> **统计**：全仓共 **75 个 Skill** + **8 个 Team**，分布在 12 个目录域中。ST 框架支持对 Skill 和 Team 的统一评测看护。

## 参考文档

- [Skill Test Framework 使用指南](USER_GUIDE.md)
- [Skill Test Framework README](../README.md)
