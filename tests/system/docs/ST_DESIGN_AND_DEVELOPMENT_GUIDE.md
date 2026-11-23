# ST 用例设计与开发规范

## 1. ST 架构设计

### 1.1 分层看护策略

| 层级 | 看护对象 | 方式 |
| --- | --- | --- |
| UT | Plugin、Team、Skill 的结构及题库格式 | 静态校验 |
| ST | 所选 Skill 或 Team 的场景行为及输出 | 语义评审、确定性断言或算子评测 |
| 冒烟 | Cannbot 的端到端算子生成能力 | 集成看护 |

### 1.2 看护流程

GitCode PR 的 `UT_cannbot-skills` 随现有 PR 流水线执行，`ST_test` 调用 RDV 运行符合条件的文本题。两者的范围见 [PR 校验与执行入口](#224-pr-校验与执行入口)。

### 1.3 ST 用例看护维度

| 目标 | 设计方法 |
| --- | --- |
| 正向触发 | 对本地 Skill 用例配置 `config.distractor_skills`，用 `skill_activated` 检查实际加载的 Skill。 |
| 负向行为 | 给出边界场景，用 `expected_output` 或 `not_contains` 验证回复不出现指定结果。现有断言没有专用的「未加载某个 Skill」类型。 |
| 内容正确性 | 在 `expected_output` 中写出关键语义要点。 |
| 交付文件 | 用本地文件断言检查文件名、路径或关键内容。现有断言没有通用的工具调用序列检查。 |
| 资源消耗 | 本地语义评审包含 Token 维度。可通过 `max_tokens` 或 `max_tokens_by_model` 加上硬上限。 |

### 1.4 本地测试执行流程

1. `tests/system/scripts/main.py` 根据 `--changed-files` 识别受影响的 Skill/Team，或用 `--all` 扫描配置目录中的题库。两种方式均受本地白名单限制。
2. 对所选对象检查 OpenCode 可用性，再执行基础检查。Skill 检查题库及 `SKILL.md`，Team 检查题库、`AGENTS.md`、`.claude-plugin/plugin.json` 和 `init.sh`。仅基础检查通过且有评测用例的对象进入评测。只做静态检查时可直接运行相应的 pytest 文件。
3. 为每题创建沙箱。Skill 放入沙箱的 `.opencode/skills/`，Team 执行 `init.sh project opencode <sandbox>` 安装。随后按评测模式运行。
4. HTML 报告写入 `tests/system/results/`，运行日志和导出的会话写入 `tests/system/logs/`。没有符合条件的对象时会产生跳过报告。查具体对象的基础检查和题目结果时同时查看日志。

### 1.5 本地执行与评审

`text` 和 `file_based` 先执行题目，再用独立 OpenCode 会话评审结果。评审 Agent 填写 [review-template.md](../config/review-template.md)，框架读取判定和各维度分数。两种模式还会执行配置的结构化断言。配置 Token 硬上限时，另按执行会话的实际消耗核对。`cann_bench` 在生成算子后调用 cann-bench 打分。

`EVAL_EXEC_RETRIES` 表示初次执行后的额外重试次数。默认 `1`，最多两次尝试。`0` 表示只执行一次。硬性 Token 上限失败直接退出当前用例。

### 1.6 评分维度与阈值

本地 ST 按有效的 `eval_mode` 选择评分方式。

#### 标准四维评分（text / file_based 模式）

| 维度 | 满分 | 最低阈值 | 说明 |
|------|:----:|:--------:|------|
| 信息覆盖度 | 40 | 20 | 是否完整覆盖预期回复中的关键要点 |
| 技术准确性 | 30 | 15 | 技术信息是否正确，无错误或误导 |
| 回复质量 | 20 | 10 | 结构清晰、逻辑连贯、简洁直接 |
| Token 消耗 | 10 | 3 | 回复长度合理，思考过程工具调用高效 |
| **总分** | **100** | **60** | 总分 ≥ 60 **且**各维度均不低于阈值方为通过 |

**维度名称归一化**：AI 评审输出中可能出现同义变体，框架自动映射为标准名。例如 `技术准确性` → `准确性`、`回复质量` → `质量`、`Token 消耗`/`token` → `Token`。

总分 60 是评审模板的判定口径。运行器检查评审为通过、附有理由且各维度达标。可在用例的 `config.dim_thresholds` 中覆盖维度阈值。

#### 确定性评测（cann_bench 模式）

本地运行器以 cann-bench 报告的综合分 `≥ 30` 为通过条件。报告另列编译和精度结果。执行失败或缺少可读评分报告时可得到零分。运行前准备 NPU 环境、cann-bench 仓库和题目所需的任务定义、工程模板。

## 2. 用例设计

### 2.1 用例文件组织

ST 用例以 **JSON 文件**（`evals/evals.json`）的形式存放在每个 Skill/Team 目录下：

```
{skill_dir}/{skill_name}/evals/evals.json    # Skill 评测用例
{team_dir}/{team_name}/evals/evals.json      # Team 评测用例
```

本地扫描目录和白名单以 [st-test.config](../config/st-test.config) 为准。当前 Skill 目录为 `ops/`、`community/ops/`、`graph/`、`model/`、`infra/`、`runtime/`、`tools/`，Team 目录为 `plugins-official/`、`plugins-community/`。

### 2.2 用例文件格式（evals.json）

#### 2.2.1 基本结构

题库由顶层元数据和 `evals` 数组组成，由 [evals_json_parser.py](../scripts/evals_json_parser.py) 解析。单题最小示例见 [PR 校验与执行入口](#224-pr-校验与执行入口)，包含本地断言的示例见 [完整用例示例](#27-完整用例示例)。

#### 2.2.2 顶层字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `skill_name` | Skill 必填 | 目标 skill 名称，需与 SKILL.md 中的 `name` 字段一致。与 `team_name` 二选一 |
| `team_name` | Team 必填 | 目标 team 名称，需与 `.claude-plugin/plugin.json` 中的 `name` 一致。与 `skill_name` 二选一 |
| `eval_mode` | 否 | 评测模式，可选值：`text`（默认，语义评审）、`file_based`（验证生成文件）、`cann_bench`（cann-bench 确定性评测）。详见 2.4 节 |

按对象类型只填写一个目标名称。本地解析器在两个字段均有值时优先使用 `team_name`。

#### 2.2.3 用例字段说明

每个 `evals[]` 中的对象包含以下字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | int | **是** | 文件内唯一的整数。新题集建议从 1 开始连续编号，已有题集保留稳定 ID |
| `title` | string | **是** | 用例标题 |
| `prompt` | string | **是** | 发送给 AI 的测试问题，应模拟真实用户场景 |
| `expected_output` | string | **是** | 对 AI 回复的语义预期。**不要求逐字匹配**，描述应覆盖的关键要点即可 |
| `config` | object | 否 | 用例级配置，可用字段见下方。可写 `{}`，省略时使用默认值 |
| `expectations` | array | 启用的 `file_based` 题必填 | 文件题至少写一条原始结构化断言。其他模式可省略，条目见下方 |
| `files` | string[] | 否 | 输入文件相对路径。平台是否准备附件取决于实际执行入口，见下方说明 |

**`config` 字段说明**：

| 键 | 类型 | 说明 | 默认值 |
|-----|------|------|--------|
| `max_tokens` | int | Token 消耗硬上限，超过则测试失败 | 无限制 |
| `max_tokens_by_model` | object | 按执行会话导出的实际模型名匹配 Token 上限，未匹配时用 `max_tokens` | `{}` |
| `ascend_platforms` | string[] | 适用平台，可用 `A2`、`A3`、`A5`、`Kirin`。本地通过 `--ascend-platform` 筛选 | 未声明平台限制 |
| `disabled` | bool | `true` 跳过该用例 | `false` |
| `distractor_skills` | string[] | 正向看护：干扰 skill 名称列表 | `[]` |
| `timeout` | int | 单题执行超时，填 1–2147483647 之间的整数秒数。题目层的 `timeout` 不生效 | `600` |
| `eval_mode` | string | 覆盖文件级评测模式 | 同文件级 |
| `dim_thresholds` | object | 维度阈值覆盖，如 `{"覆盖度": 25}` | `{}` |
| `truncate_len` | int | 传入本地评审的执行回复及推理各自的截断长度（字符） | `30000` |
| `cann_bench_operator` | string | cann_bench 模式：目标算子名称 | - |
| `cann_bench_level` | string | cann_bench 模式：难度等级 | `level1` |
| `cann_bench_device` | string | cann_bench 模式：NPU 设备 ID | `0` |
| `cann_bench_no_perf` | bool | cann_bench 模式：跳过性能评测 | `false` |
| `cann_bench_warmup` | int | 性能预热次数 | 未设置时由 cann-bench 决定 |
| `cann_bench_repeat` | int | 性能重复次数 | 未设置时由 cann-bench 决定 |

#### 2.2.4 PR 校验与执行入口

GitCode PR 现有流水线的 `UT_cannbot-skills` 通过 `ut.sh` 无参数调用 `tests/run-tests.sh`，运行默认 fast 测试集，其中包括 S-EVAL-01。本地可选的 `--incremental` 用于选择测试，和该流水线使用的 PR 文件清单是两套入口。

- 原有全仓基础检查扫描 `ops/`、`model/`、`graph/`、`runtime/`、`infra/`、`tools/` 下的一级 Skill，检查题库是否存在、能否读取 JSON 及是否声明 `skill_name`。原有变更题库 `expectations` 结构检查继续执行。
- 逐题格式检查根据 PR 文件清单和候选版本中的 `SKILL.md` 确定归属。新增或受修改的 Skill 检查其题库的全部题目。只改该 Skill 的 `SKILL.md`、脚本或参考资料时，也会查其题库。直接变更且仍存在的 `evals/evals.json` 同样纳入，包括 Team 题库及只做排版、重排或改名的题库。只改仓根文档不会把无关 Skill 纳入逐题检查。

- 新题所在题库填写 `skill_name` 或 `team_name`。每题使用 `title`、`prompt`、`expected_output`，将单题执行限时放在 `config.timeout`。同一文件的 ID 不重复。语义要求放在 `expected_output`，结构化断言写 `expectations`，使用下文列出的 `type` 和非空字符串 `pattern`。单题的 `name`、`assertions` 不作相应字段的别名。
- 本地 ST 可执行 `text`、`file_based` 和 `cann_bench`。使用对应模式的本地判定，`files` 不会将输入附件放入沙箱。单题执行默认 600 秒。`main.py` 还给整个 pytest 子进程设置时限。指定单题时取该题的 `config.timeout`，未配用 1200 秒。批量执行取已配最大值，至少 1200 秒。
- GitCode 的 `ST_test` 调用 runner 上的 RDV `ci-pr-verify.sh`。普通 PR 选择启用且平台匹配的 `text` 题，按 `expected_output` 做文本评审。要执行结构化断言、文件内容评审或算子评测，使用支持相应模式的入口。RDV 的单题执行限时读取 `config.timeout`，未配置时默认 600 秒。现有工作流给 `ci-pr-verify.sh` 传入 `--timeout 3600`，这是整次等待时限。
- RDV Core 正式采用固定 PR 文本题输入附件能力后，才能在这条普通 PR 链路用 `files`。填写插件相对路径，或带同一插件前缀的仓库相对路径。执行器按固定 PR 合并版本和原题 ID 读取同插件目录内的普通文件，以原文件名放入题目工作目录。路径越界、LFS 指针、缺少文件或目标文件名冲突会报错。

最小文本题示例：

```json
{
  "skill_name": "cann-env-setup",
  "evals": [
    {
      "id": 1,
      "title": "查看 NPU 信息",
      "prompt": "我想确认这台昇腾服务器能识别 NPU，应该用什么命令？",
      "expected_output": "说明使用 npu-smi info，并指出应查看的设备信息。",
      "config": {"timeout": 600}
    }
  ]
}
```

### 2.3 Expectations 类型详解

`expectations` 为断言数组，每条含 `type`（断言类型）、`pattern`（匹配模式）与可选 `description`：

```json
{
  "expectations": [
    { "type": "contains", "pattern": "npu-smi info", "description": "回复中提到了 npu-smi info" }
  ]
}
```

#### contains — 文本包含检查

检查本地执行会话的完整输出是否包含指定字符串（不区分大小写）。过程输出也可能满足该断言。若要求命令出现在最终回复，在 `expected_output` 中明确说明。

```json
{ "type": "contains", "pattern": "npu-smi info" }
```

> **注意**：`contains` 校验不区分大小写，可匹配 "npu-smi info"、"Npu-Smi Info" 或 "NPU-SMI INFO" 等方式。

#### not_contains — 文本排除检查

检查最终回复是否排除了指定字符串，区分大小写。

```json
{ "type": "not_contains", "pattern": "- [ ]" }
```

> **注意**：只检查 AI 对用户的最终回复（`ai_text`），不检查工具调用过程中的参考文档内容和中间输出。

#### file_exists — 文件存在检查

检查路径是否存在。依次检查沙箱、沙箱的 `.opencode/skills/<skill>`、原目标目录，必要时递归搜索沙箱。已有文件也能满足该断言。

```json
{ "type": "file_exists", "pattern": "todo.md", "description": "文件 todo.md 已生成" }
```

#### file_list — 文件列表匹配检查

在沙箱及其 `.opencode/skills/` 下各 Skill 子目录中按 glob 查找匹配项。

```json
{ "type": "file_list", "pattern": "*.md" }
```

#### skill_activated — 程序化 skill 激活检查

**程序化**检查 AI 执行过程中是否加载了指定 skill。不依赖 AI 评审模型，从导出的会话或执行事件流中精确匹配 Skill 名称。用于正向看护场景。

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

适合验证最终回复。执行 Agent 响应 `prompt`，本地评审 Agent 按 `expected_output` 评审。配置了 `expectations` 时，本地运行器也执行对应断言。

#### file_based 模式

适合验证代码、配置、文档等生成文件。本地执行器补充交付文件提示，评审会话从沙箱读取文件并依据 `expected_output` 评审。普通 GitCode PR ST 仍选取 `text` 题。

受 PR UT 检查的启用文件题必须在 `expectations` 声明至少一条原始结构化断言，包括继承题库顶层 `file_based` 的题。RDV 的文件评测入口需要这些断言才能执行，并支持 `file_exists`、`file_list`、`file_contains` 和 `contains`。例如可用 `{"type": "file_exists", "pattern": "output/diagram.svg"}` 检查输出文件，再用 `expected_output` 描述质量要求。[Expectations 类型详解](#23-expectations-类型详解)

#### cann_bench 模式

本地执行器将 cann-bench 任务定义和工程模板复制到沙箱，提示 Agent 把算子输出到 `output/`，再将生成内容合并到工程模板。运行 `cann-bench/scripts/run_evaluation.sh`，从 JSON 报告读取评分。有 HTML 报告时会带入 pytest 报告。可在 `config` 中指定算子、设备和评测选项。

```json
{
  "eval_mode": "cann_bench",
  "ascend_platforms": ["A5"],
  "timeout": 10800,
  "cann_bench_operator": "mish",
  "cann_bench_level": "level1"
}
```

以上对象放在单题的 `config` 中。先确认指定算子位于所选 cann-bench 版本的 `tasks/<level>/<operator>/`。示例中的 10800 秒用于作答和外层单题运行。cann-bench 评测子进程自身仍使用每次 600 秒的限时。

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

- 用本地 `contains` 或 `not_contains` 验证必要的关键词或输出结构，语义要点写进 `expected_output`
- 不要用 `contains` 逐一验证 expected_output 中的每个要点——语义评测已经做了这件事
- `contains` 适用于验证关键 API 名称、命令、文件结构标记等
- `not_contains` 适用于负向验证（如验证不应创建任务计划的场景）

### 2.6 用例 ID 规范

PR UT 要求受影响 Skill 或直接变更题库的每道题使用文件内唯一的有符号 32 位整数 ID。本地 Skill 和 Team 基础检查还要求文件里的 ID 按题序连续递增，起点不限。新题库建议从 1 开始，维护已有题库时保留现有 ID。

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
    "distractor_skills": ["ascendc-runtime-debug", "npu-arch", "ascendc-docs-search"]
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

1. 阅读目标 Skill 的 `SKILL.md` 或 Team 的 `AGENTS.md`，从实际使用场景写题。
2. 在目标目录的 `evals/evals.json` 编辑用例。用 `expected_output` 表达期望效果，需要本地验证关键文字或文件时追加 `expectations`。
3. 从仓库根目录安装依赖，再按目标类型运行 Skill 或 Team 基础检查。

```bash
python -m pip install -r tests/system/scripts/requirements.txt
python -m pytest tests/system/scripts/test_skill_basic.py -k cann-env-setup -v
python -m pytest tests/system/scripts/test_team_basic.py -k ops-registry-invoke -v
```

在 Linux 终端运行指定题的本地 ST 时，使用 `main.py` 先做基础检查，再执行评测。评测需要 PATH 中的 OpenCode 和可用的模型配置。评测沙箱依赖 Python 的 `fcntl`，原生 Windows 可单独运行上面的基础检查。

```bash
python tests/system/scripts/main.py --repo-root . --changed-files ops/cann-env-setup/evals/evals.json --eval-id 1 --ascend-platform A2
```

按变更集运行本地评测时，可以使用 `tests/gate_check.sh` 包装器。未显式选择且未设置 `ASCEND_PLATFORM` 时，该包装器默认使用 A2。GitCode PR 页面分别查看 UT 和 `ST_test` 结果。

### 2.9 配置新的 Skill / Team 扫描路径

在 [tests/system/config/st-test.config](../config/st-test.config) 中设置本地扫描范围。

- `skill_dirs` 和 `team_dirs` 填相对仓库根目录的扫描路径。
- `skill_whitelist` 和 `team_whitelist` 填对应对象名称。列表非空时仅运行列出的对象，空列表不限制名称。

GitCode 普通 PR ST 的题目选择由 RDV 执行。此配置用于仓内 ST。

### 2.10 常见问题

**本地评审无法解析或判定失败**

查看 `tests/system/logs/` 的评审会话及题目沙箱中填写后的 `review-template.md`。确认判定、理由和各维度分数齐全。预期结果应描述可以判断的要点。

**本地执行偶发失败**

初次执行后可按 `EVAL_EXEC_RETRIES` 额外重试。以 Bash 为例，下列命令允许最多三次尝试。

```bash
EVAL_EXEC_RETRIES=2 python tests/system/scripts/main.py --repo-root . --changed-files ops/cann-env-setup/evals/evals.json --eval-id 1 --ascend-platform A2
```

**cann_bench 找不到仓库、任务或报告**

设置 `CANN_BENCH_PATH` 指向实际安装的 cann-bench 仓库。未设置时，本地运行器默认查找 `<Skills 仓库>/../../cann-bench`。核对算子任务、工程模板和设备。结果 JSON 保存在沙箱的 `cann_bench_reports/`，生成的 HTML 会复制为 `cann_bench_report.html`。

## 参考文档

- [Skill Test Framework README](../../README.md)
