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
| **正向看护** | 显式 / 隐式提示词 → 调用到目标 skill | **具备** | 在 `config.distractor_skills` 配置干扰技能，验证即使存在多个类似 skill，AI 仍能正确选择目标 skill |
| **负向看护** | 不该调用的提示词 → 不会误调用 skill | 不具备 | 设计边界场景 prompt 验证 skill 不会被错误触发 |
| **正确性看护** | 黑盒场景验证、确保结果正确 | **具备** | 设计典型用户场景，描述预期输出要点，验证 AI 回复语义覆盖 |
| **调用流程看护** | 关键工具被调用、交付件完整输出 | **具备** | 验证 skill 执行过程中关键工具是否被调用、关键文件是否生成 |
| **资源消耗看护** | Token 消耗监控、防止资源浪费 | **具备** | 评测 session 自动检查 Token 消耗合理性（占总分 10 分） |

### 1.4 测试执行流程

```
输入：repo_root + changed_files
    │
    ├─ 步骤1：识别受影响的 skills / teams
    │   └─ 从变更文件路径中提取 skill 或 team 名称
    │
    ├─ 步骤2：加载评测用例
    │   └─ 从 Skill/Team 本地目录读取 evals/evals.json
    │       （Skill: {skill_dir}/{skill_name}/evals/evals.json）
    │       （Team: {team_dir}/{team_name}/evals/evals.json）
    │
    ├─ 步骤3：执行评测 — 逐个 target 进行
    │   ├─ Phase 1: 静态结构验证
    │   │   ├─ Skill: test_skill_basic.py — evals.json 结构/格式/ID 校验 + SKILL.md frontmatter 校验
    │   │   └─ Team: test_team_basic.py — evals.json 校验 + AGENTS.md / plugin.json / init.sh 存在性校验
    │   │   耗时：秒级，无需 AI 调用
    │   │   ⚠️ Phase 1 失败的 target 不进入 Phase 2（需先修复基础结构问题）
    │   │
    │   └─ Phase 2: AI 语义评测（仅通过 Phase 1 的 target 进入）
    │       ├─ 支持重试：EVAL_EXEC_RETRIES 控制重试次数（默认 1）
    │       ├─ Skill: test_skill_evals.py — 沙箱部署 symlink skill 目录到 .opencode/skills/
    │       ├─ Team: test_team_evals.py — 沙箱部署执行 init.sh project opencode <sandbox>
    │       ├─ cann_bench 模式额外部署 cann-bench 仓库副本 + 任务定义文件 + 工程模板
    │       ├─ 执行 Session：opencode 加载 target，发送 prompt → 收集 AI 回复
    │       ├─ 评测（评分标准详见 1.6 节）：
    │       │   ├─ AI 评审模式（text/file_based）：独立 session 按四维标准评审（总分 ≥60 且各维度 ≥ 阈值）
    │       │   │   └─ 评审方式：Agent 通过 Write 工具填写 review-template.md 模板
    │       │   └─ cann_bench 模式：执行 run_evaluation.sh，通过条件为编译通过 + 精度达标 + 综合得分 > 50
    │       └─ 断言验证：contains / not_contains / file_exists / file_list / file_contains / skill_activated
    │       耗时：分钟级，需要 opencode CLI
    │
    ├─ 步骤4：保存结果
    │   └─ results/[<平台>_]ST_validation_report_<ts>.html   # Skill+Team 统一报告
    │       └─ results/<name>_<timestamp>.json                # 结构化结果
    │
    └─ 返回：0（全部通过）/ 1（存在失败）
```

### 1.5 测试执行架构：双 Agent 协同

每个 ST 用例的执行涉及两个独立的 AI Agent（均为 opencode CLI 会话），与其各自的评测机制协同工作：

```
ST 评测用例（Prompt + Expected Output + Expectations）
    │
    ▼
执行 Agent（Execution Session）
    ├─ 加载目标 skill + 干扰 skills → 发送 Prompt
    ├─ AI 理解 + 工具调用 → 生成回复/文件
    └─ 产出：AI 回复文本、工具调用记录、生成的文件
    │
    ▼
┌─ 评测 Agent（Review Session） ──── ┌─ 模式匹配引擎 ── ┌─ Token 预算检查 ─
│ 读取 Expected Output + AI 回复     │ contains/       │ 读取 session JSON
│ 按四维标准打分（40/30/20/10）       │ not_contains/   │ 计算总 token 消耗
│ 总分 ≥ 60 且各维度 ≥ 阈值           │ file_exists/    │ 对比 max_tokens
│                                   │ file_list/      │ 上限
│                                   │ file_contains/  │
│                                   │ skill_activated │
└────────────┬──────────────────────┴────────┬────────┴───────┬─────────┘
             │                               │                │
             ▼                               ▼                ▼
最终判定（Test Result）：总分 ≥ 60 AND 各维度 ≥ 阈值 AND 所有 Expectations 通过 AND Token 未超限
结果：Passed / Failed + 详细原因
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
- **正向看护**：`skill_activated` 从 session 导出 JSON 中提取工具调用记录，不依赖评审 Agent
- **沙箱隔离**：每个用例有独立沙箱，skill 通过软链接部署（team 通过 init.sh 部署），干扰 skill 同等待遇
- **评审模板化**：评审 Agent 通过 Write 工具填写 `review-template.md` 模板，框架正则解析提取状态/评分/各维度得分，不再依赖 JSON 输出格式
- **重试机制**：通过 `EVAL_EXEC_RETRIES` 环境变量控制评测用例的重试次数（默认 1，即不重试）
- **模型 Token 预算**：支持 `config.max_tokens_by_model` 按模型指定 Token 上限，与 `--eval-model` 参数或 `EVAL_MODEL` 环境变量配合使用
- **安全环境隔离**：opencode 子进程仅传递最小环境变量（PATH、HOME 及 LLM API 密钥），剥离继承的敏感信息，防止不可信 prompt 泄露令牌

### 1.6 评分维度与阈值

ST 评测支持两种评分体系，根据 `eval_mode` 自动选择：

#### 标准四维评分（text / file_based 模式）

| 维度 | 满分 | 最低阈值 | 说明 |
|------|:----:|:--------:|------|
| 信息覆盖度 | 40 | 20 | 是否完整覆盖预期回复中的关键要点 |
| 技术准确性 | 30 | 15 | 技术信息是否正确，无错误或误导 |
| 回复质量 | 20 | 10 | 结构清晰、逻辑连贯、简洁直接 |
| Token 消耗 | 10 | 3 | 回复长度合理，思考过程工具调用高效 |
| **总分** | **100** | **60** | 总分 ≥ 60 **且**各维度均不低于阈值方为通过 |

**维度名称归一化**：AI 评审输出中可能出现同义变体，框架自动映射为标准名。例如 `技术准确性` → `准确性`、`回复质量` → `质量`、`Token 消耗`/`token` → `Token`。

各维度阈值可在用例 `config.dim_thresholds` 中单独覆盖，如 `{"覆盖度": 25}`。

#### 确定性评测（cann_bench 模式）

cann_bench 模式使用 cann-bench 项目的确定性评测管道，替代 AI 评审打分：

| 维度 | 满分 | 说明 |
|------|:----:|------|
| 编译通过 | 1 | 二元判定：算子工程能否成功编译 |
| 精度达标 | 1 | 二元判定：算子输出精度是否满足要求 |
| 综合得分 | 100 | cann-bench 管道输出的综合性能评分（HAP 评分） |

**通过条件**：编译通过 = 1 且 精度达标 = 1 且 综合得分 > 50

**工作流程**：
1. AI Agent 在沙箱中根据 cann-bench 任务定义生成算子代码到 `output/` 目录
2. 框架自动执行 `run_evaluation.sh` 三阶段评测（编译/精度/性能）
3. 解析 cann-bench 生成的 JSON 报告，提取编译/精度/性能评分
4. 将 cann-bench 生成的 HTML 报告通过 iframe 嵌入 pytest-html 最终报告

**cann_bench 模式专属配置项**（配置在用例 `config` 中）：

| 配置项 | 类型 | 说明 | 默认值 |
|--------|------|------|--------|
| `cann_bench_operator` | string | 必填：目标算子名称（如 `mish`） | - |
| `cann_bench_level` | string | 任务难度等级（`level1`/`level2`/`level3`） | `level1` |
| `cann_bench_device` | string | NPU 设备 ID | `0` |
| `cann_bench_no_perf` | bool | 跳过性能评测，仅验证编译和精度 | `false` |
| `cann_bench_warmup` | int | 性能评测预热次数 | 框架默认 |
| `cann_bench_repeat` | int | 性能评测重复次数 | 框架默认 |

---

## 2. 用例设计

### 2.1 用例文件组织

ST 用例以 **JSON 文件**（`evals/evals.json`）的形式存放在每个 Skill/Team 目录下：

```
{skill_dir}/{skill_name}/evals/evals.json    # Skill 评测用例
{team_dir}/{team_name}/evals/evals.json      # Team 评测用例
```

支持的目录（由 `st-test.config` 配置）：`ops/`、`graph/`、`model/`、`infra/`、`runtime/`、`plugins-official/`、`plugins-community/`。

> **注意**：评测用例与 Skill/Team 代码同目录存放，便于维护和版本控制。测试框架会自动扫描配置的目录，发现包含 `evals/evals.json` 的实体。

### 2.2 用例文件格式（evals.json）

#### 2.2.1 基本结构

每个用例文件为 JSON 格式，由顶层元数据和 `evals[]` 数组组成，由 `evals_json_parser.py` 解析：

```json
{
  "skill_name": "cann-env-setup",
  "eval_mode": "text",
  "evals": [
    {
      "id": 1,
      "title": "检查NPU驱动安装命令",
      "config": {
        "max_tokens": 160000,
        "ascend_platforms": ["A2"],
        "eval_mode": "text"
      },
      "prompt": "我有一台昇腾服务器，想检查NPU驱动是否已安装，应该用什么命令？",
      "expected_output": "回复应说明使用 npu-smi info 命令检查驱动，并解释如何根据命令输出判断驱动是否已安装",
      "files": [],
      "expectations": [
        { "type": "contains", "pattern": "npu-smi info", "description": "回复中提到了 npu-smi info" }
      ]
    }
  ]
}
```

#### 2.2.2 顶层字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `skill_name` | Skill 必填 | 目标 skill 名称，需与 SKILL.md 中的 `name` 字段一致。与 `team_name` 二选一 |
| `team_name` | Team 必填 | 目标 team 名称，需与 plugin.json 中的 `name` 字段一致。与 `skill_name` 二选一 |
| `eval_mode` | 否 | 评测模式，可选值：`text`（默认，语义评审）、`file_based`（验证生成文件）、`cann_bench`（cann-bench 确定性评测）。详见 2.4 节 |

> **注意**：`skill_name` 和 `team_name` 只能设置一个。解析器根据顶层字段自动确定 target 类型（`target_type: "skill"` 或 `"team"`）。

#### 2.2.3 用例字段说明

每个 `evals[]` 中的对象包含以下字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | int | **是** | 用例编号，从 1 开始连续递增 |
| `title` | string | **是** | 用例标题 |
| `prompt` | string | **是** | 发送给 AI 的测试问题，应模拟真实用户场景 |
| `expected_output` | string | **是** | 对 AI 回复的语义预期。**不要求逐字匹配**，描述应覆盖的关键要点即可 |
| `config` | object | 是 | 用例级配置，可用字段见下方 |
| `expectations` | array | 否 | 断言列表，条目见下方 |
| `files` | string[] | 否 | 输入文件路径列表（skill-creator 使用，ST 框架忽略） |

**`config` 字段说明**：

| 键 | 类型 | 说明 | 默认值 |
|-----|------|------|--------|
| `max_tokens` | int | Token 消耗硬上限，超过则测试失败 | 无限制 |
| `max_tokens_by_model` | object | 按模型的 Token 上限，如 `{"deepseek-v4-flash": 140000}` | `{}` |
| `ascend_platforms` | string[] | 适用平台，如 `["A2"]`。配合 `--ascend-platform` 参数按平台过滤 | `[]` |
| `disabled` | bool | `true` 跳过该用例 | `false` |
| `distractor_skills` | string[] | 正向看护：干扰 skill 名称列表 | `[]` |
| `timeout` | int | 执行超时秒数 | `600` |
| `eval_mode` | string | 覆盖文件级评测模式 | 同文件级 |
| `dim_thresholds` | object | 维度阈值覆盖，如 `{"覆盖度": 25}` | `{}` |
| `truncate_len` | int | AI 回复截断长度（字符） | `30000` |
| `cann_bench_operator` | string | cann_bench 模式：目标算子名称 | - |
| `cann_bench_level` | string | cann_bench 模式：难度等级 | `level1` |
| `cann_bench_device` | string | cann_bench 模式：NPU 设备 ID | `0` |
| `cann_bench_no_perf` | bool | cann_bench 模式：跳过性能评测 | `false` |
| `cann_bench_warmup` | int | cann_bench 模式：性能预热次数 | 框架默认 |
| `cann_bench_repeat` | int | cann_bench 模式：性能重复次数 | 框架默认 |

### 2.3 Expectations 类型详解

`expectations` 为断言数组，每条含 `type`（断言类型）、`pattern`（匹配模式）与可选 `description`：

```json
"expectations": [
  { "type": "contains", "pattern": "npu-smi info", "description": "回复中提到了 npu-smi info" }
]
```

#### contains — 文本包含检查

检查 AI **最终回复**（`ai_text`）中是否包含指定字符串（**不区分大小写**）。

```json
{ "type": "contains", "pattern": "npu-smi info" }
```

> **注意**：`contains` 校验不区分大小写，可匹配 "npu-smi info"、"Npu-Smi Info" 或 "NPU-SMI INFO" 等方式。

#### not_contains — 文本排除检查

检查 AI **最终回复文本**中是否**不**包含指定字符串。用于负向看护场景。

```json
{ "type": "not_contains", "pattern": "- [ ]" }
```

> **注意**：只检查 AI 对用户的最终回复（`ai_text`），不检查工具调用过程中的参考文档内容和中间输出。

#### file_exists — 文件存在检查

检查指定文件是否被创建或修改。搜索顺序：`sandbox/<path>` → `sandbox/skill/<path>` → `skill_dir/<path>`。

```json
{ "type": "file_exists", "pattern": "todo.md", "description": "文件 todo.md 已生成" }
```

#### file_list — 文件列表匹配检查

检查沙箱中是否存在匹配 glob pattern 的文件。适用于需要验证生成文件但不确定确切路径的场景。

```json
{ "type": "file_list", "pattern": "*.md" }
```

#### skill_activated — 程序化 skill 激活检查

**程序化**检查 AI 执行过程中是否加载了指定 skill。不依赖 AI 评审模型，直接从 tool_use 事件中精确匹配 skill 名称。用于正向看护场景。

```json
{ "type": "skill_activated", "pattern": "cann-env-setup" }
```

> **注意**：`skill_activated` 是确定性检查，不受评审模型主观判断影响。即使 AI 回复在技术上是正确的，如果它加载了错误的 skill（或没有加载任何 skill），此断言会直接导致测试失败。

#### file_contains — 文件内容包含检查

检查沙箱中匹配 glob 路径的文件是否包含所有指定文本。支持 glob 通配符（`*`、`?`、`[]`），匹配到多个文件时，只要有一个文件包含所有 pattern 即判定通过。

```json
{ "type": "file_contains", "pattern": "src/kernel/*.asc : \"__global__\";\"LocalTensor\"" }
```

**pattern 格式说明**：
- 路径部分：文件路径或 glob 模式（相对于沙箱根目录）
- 分隔符：` : `（空格-冒号-空格）
- 文本模式：双引号包裹，多个用英文分号 `;` 分隔
- 验证逻辑：所有列出的文本模式必须在**同一个**文件中全部出现才判定通过

### 2.4 评测模式

#### text 模式（默认）

适用于大多数场景，AI 回复以文本方式输出。评测流程：
1. 执行 session：向 skill 发送 prompt，收集 AI 文本回复
2. 评测 session：独立 session 基于 Expected Output 评审回复质量
3. 模式匹配：检查 expectations 中的 contains/not_contains/file_contains 规则

#### file_based 模式

适用于验证 AI 生成文件的场景（如生成代码、配置文件等）。评测流程：
1. 系统自动向 prompt 末尾追加 `FILE_BASED_HINT`（要求 AI 列出创建/修改的文件清单并说明用途，不输出完整文件内容）
2. 执行 session：AI 在沙箱中创建/修改文件
3. 评测 session：独立 session 读取沙箱中的生成文件，基于文件内容评审质量
4. 模式匹配：检查 expectations 中的 file_exists/file_list/file_contains 规则

#### cann_bench 模式

适用于验证 AI 生成算子代码并通过 cann-bench 项目进行确定性评测的场景。评测流程：
1. 沙箱部署：额外部署 cann-bench 仓库、任务定义文件和算子工程模板
2. 系统自动向 prompt 末尾追加 `CANN_BENCH_HINT`（要求 AI 将生成的算子代码输出到 `output/` 目录）
3. 执行 session：AI 在沙箱中根据 cann-bench 任务定义生成算子代码
4. 合并式目录复制：将 AI 生成的代码合并到算子工程模板中（保留模板文件，覆盖同名文件）
5. 确定性评测：执行 cann-bench 评测管道（`run_evaluation.sh`），三阶段评测（编译检查 → 精度验证 → 性能评分）
6. 结果解析与报告嵌入：从 JSON 报告提取评分，HTML 报告通过 iframe 嵌入 pytest-html

**配置示例**：

```json
"config": {
  "eval_mode": "cann_bench",
  "max_tokens": 10000000,
  "timeout": 10800,
  "ascend_platforms": ["A5"],
  "cann_bench_operator": "mish",
  "cann_bench_level": "level1"
}
```

### 2.5 用例设计原则

#### 2.5.1 场景覆盖

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

以下是一个 Skill 的完整用例文件示例（`ops/cann-env-setup/evals/evals.json`）：

```json
{
  "skill_name": "cann-env-setup",
  "eval_mode": "text",
  "evals": [
    {
      "id": 1,
      "title": "检查NPU驱动安装命令",
      "config": { "max_tokens": 160000, "ascend_platforms": ["A2"] },
      "prompt": "我有一台昇腾服务器，想检查NPU驱动是否已安装，应该用什么命令？",
      "expected_output": "回复应说明使用 npu-smi info 命令检查驱动，并解释如何根据命令输出判断驱动是否已安装",
      "expectations": [{ "type": "contains", "pattern": "npu-smi info" }]
    },
    {
      "id": 2,
      "title": "配置环境变量永久生效",
      "config": { "max_tokens": 160000, "ascend_platforms": ["A2"] },
      "prompt": "我已经用离线安装包安装完CANN Toolkit和Ops，现在需要配置环境变量使其永久生效，应该怎么做？",
      "expected_output": "回复应说明如何配置环境变量实现永久生效：通过 source set_env.sh 命令并将其写入 ~/.bashrc 文件",
      "expectations": [
        { "type": "contains", "pattern": "source" },
        { "type": "contains", "pattern": "set_env.sh" }
      ]
    }
  ]
}
```

#### 正向看护示例

当需要验证 AI 在多个类似 skill 存在时仍能正确选择目标 skill 时，配置 `config.distractor_skills`：

```json
{
  "id": 3,
  "title": "正向看护-多skill环境下正确触发目标skill",
  "config": {
    "max_tokens": 160000,
    "ascend_platforms": ["A2"],
    "distractor_skills": ["ascendc-runtime-debug", "ascendc-task-focus", "npu-arch", "ascendc-docs-search"]
  },
  "prompt": "我有一台昇腾服务器，想检查NPU驱动是否已安装，应该用什么命令？",
  "expected_output": "回复应说明使用 npu-smi info 命令检查驱动，并解释如何根据命令输出判断驱动是否已安装。应成功激活并使用了 cann-env-setup skill。",
  "expectations": [
    { "type": "skill_activated", "pattern": "cann-env-setup" },
    { "type": "contains", "pattern": "npu-smi info" }
  ]
}
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
4. 编写 evals.json 文件
   ├─ 填写顶层字段（skill_name / team_name）
   ├─ 编写各个用例（prompt + expected_output + expectations）
   └─ 检查 ID 连续递增、格式正确
       │
5. 本地验证（开发调试）
   ├─ Phase 1: python -m pytest tests/system/scripts/test_skill_basic.py -v -k "<skill>"
   └─ Phase 2: python -m pytest tests/system/scripts/test_skill_evals.py --skill <skill> -v
       │
6. 提交到 {skill_dir}/{skill_name}/evals/ 目录
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

1. 在 `{skill_dir}/{skill_name}/evals/` 下创建 `evals.json`
2. 确保 `config/st-test.config` 中的 `skill_dirs` 包含该 skill 所在目录
3. 如果启用了 `skill_whitelist`，将 skill 名称加入白名单
4. 运行 Phase 1 静态验证确认格式正确

**Q4: file_based 模式和 text 模式如何选择？**

- 需要验证 AI 生成的文件内容（代码、配置、文档等）→ 使用 `file_based`
- 只需要验证 AI 文本回复的正确性 → 使用 `text`（默认）

**Q5: 如何给 Team 新增第一个 ST 用例？**

与 Skill 类似，但注意：
1. evals.json 的顶层字段使用 `team_name` 而非 `skill_name`
2. Phase 1 额外校验 AGENTS.md、plugin.json、init.sh 的存在性
3. 沙箱部署方式不同：Team 通过执行 `init.sh project opencode <sandbox>` 部署
4. 确保 `config/st-test.config` 的 `team_dirs` 和 `team_whitelist` 包含该 team

**Q6: 偶发性评测执行失败如何排查？**

设置环境变量 `EVAL_EXEC_RETRIES` 启用重试（默认 1）：
```bash
EVAL_EXEC_RETRIES=3 python -m pytest tests/system/scripts/test_skill_evals.py --skill cann-env-setup -v
```
重试会重新执行 opencode session 并重新评审，适用于网络抖动或 AI 服务器偶发错误。

**Q7: cann_bench 模式评测失败如何排查？**

1. **cann-bench 仓库不存在**：检查 `CANN_BENCH_PATH` 环境变量或默认路径 `<repo_root>/../cann-bench`
2. **编译失败**：检查 AI 生成的算子代码是否完整，特别是 CMakeLists.txt 配置
3. **精度不达标**：检查 kernel 实现的数值稳定性，特别是 fp16/bf16 的混合精度计算
4. **任务定义文件缺失**：检查 `cann-bench-task/` 目录下的 proto.yaml、cases.yaml 是否存在

查看 cann-bench 生成的 HTML 报告（在沙箱目录的 `cann_bench_report.html`）获取详细评测结果。

---

## 参考文档

- [Skill Test Framework README](../README.md)
