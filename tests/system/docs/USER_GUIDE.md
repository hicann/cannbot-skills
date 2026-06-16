# Skill Test Framework 使用指南

## 功能特性

Skill Test Framework 是 cannbot-skills 的技能质量看护框架，提供以下能力：

- **多目录扫描** — 通过 `config/st-test.config` 配置跨多个目录发现 skill，不限于单一 `skills/` 目录
- **变更驱动测试** — 基于 Git 变更文件自动识别受影响的 skill，按需执行评测
- **双层验证** — Phase 1 静态结构检查 + Phase 2 AI Eval 语义评测
- **独立评测 Session** — Phase 2 使用独立 opencode session 评审执行结果，避免自我检查偏差
- **HTML + JSON 报告** — 输出可视化和结构化测试报告，便于 CI/CD 集成和人工排查

## 架构概览

```
tests/system/
├── config/
│   ├── st-test.config       # skill/team 扫描路径配置
│   └── review-template.md      # 评测 session 模板
├── scripts/
│   ├── main.py                 # CI 门禁入口（一键执行）
│   ├── conftest.py             # pytest 配置 + skill/team 扫描逻辑
│   ├── test_skill_basic.py     # Phase 1: Skill 静态结构验证
│   ├── test_team_basic.py      # Phase 1: Team 静态结构验证
│   ├── test_skill_evals.py     # Phase 2: Skill AI 语义评测
│   ├── test_team_evals.py      # Phase 2: Team AI 语义评测
│   ├── opencode_runner.py      # opencode CLI 封装
│   ├── sandbox_manager.py      # 沙箱环境管理
│   ├── evals_parser.py         # evals.md 解析器（支持 skill/team）
│   ├── run_eval.py             # 评测执行入口
│   └── session_stats.py        # Session 统计分析
├── evals/                      # 框架自身的评测用例
├── results/                    # 测试报告输出
├── logs/                       # opencode session 日志 + 归档
├── sandboxes/                  # 隔离测试环境
├── cases/                      # 评测用例 _evals.md 文件
└── docs/                       # 文档
```

### 两阶段测试

```
输入: changed_files
  │
  ├─ Phase 1: 静态结构检查 (test_skill_basic.py / test_team_basic.py)
  │   ├─ evals.md 存在性、合法性、结构完整性
  │   ├─ SKILL.md 或 AGENTS.md 存在性、frontmatter 必填字段
  │   │   (Team 额外检查: plugin.json、init.sh)
  │   ├─ 用例 ID 唯一性、顺序正确性
  │   └─ ⚠️ 失败则跳过该 target 的 Phase 2
  │
  └─ Phase 2: AI 语义评测 (test_skill_evals.py / test_team_evals.py)
      ├─ 支持重试: EVAL_EXEC_RETRIES 控制（默认 1）
      ├─ 执行 Session: opencode 加载 skill/team 处理 prompt → AI 回复
      │   (Skill: symlink skill 目录到沙箱; Team: 通过 init.sh 安装完整环境)
      ├─ 评测 Session: 独立 opencode session 评审回复质量
      │   ├─ 输入: 原始问题 + AI 思考链 + AI 回复 + 预期要点
      │   ├─ 评审方式: Agent 填写 review-template.md 模板
      │   ├─ 框架解析: 正则提取 Status + Score + 维度得分
      │   └─ 输出: pass/fail + 判定依据
      └─ 模式匹配: expectations 中的 contains/not_contains/file_exists/file_list/file_contains/skill_activated 检查
```

## 配置

### st-test.config

```yaml
# 扫描 skill 的目录列表（相对于仓库根目录）
skill_dirs:
  - "ops"
  - "graph"
  - "model/skills"

# 排除的 skill 名称
exclude_skills:
  - "skill-test-framework"

# 扫描 team 的目录列表（相对于仓库根目录）
team_dirs:
  - "plugins-official"
  - "plugins-community"

# Team 白名单：仅这些 team 触发评测（空数组表示全量）
team_whitelist:
  - "ops-direct-invoke"
```

### _evals.md 用例文件格式

每个 skill 的评测用例定义在 `tests/system/cases/<skill_name>_evals.md`，由 **YAML frontmatter** 和多个 **Markdown 用例块** 组成：

```markdown
---
skill_name: cann-env-setup
eval_mode: text          # 可选，默认 text
---

# Case 1: 检查NPU驱动安装命令

## Config
- Max Tokens: 100000

## Prompt

我有一台昇腾服务器，想检查NPU驱动是否已安装，应该用什么命令？

## Expected Output

回复应说明使用 npu-smi info 命令检查驱动，并解释如何根据命令输出判断驱动是否已安装

## Expectations

- [contains] npu-smi info
---
```
```

**Frontmatter 字段说明**：

| 字段 | 必填 | 说明 |
|------|------|------|
| `skill_name` | 是（Skill） | 目标 skill 名称，需与 SKILL.md 中的 `name` 一致。与 `team_name` 二选一 |
| `team_name` | 是（Team） | 目标 team 名称，需与 plugin.json 中的 `name` 一致。与 `skill_name` 二选一 |
| `eval_mode` | 否 | 评测模式，默认 `text`。可选 `file_based`（用于验证生成文件的场景） |

> **注意**：`skill_name` 和 `team_name` 互斥，同一个 evals.md 文件中只能设置一个。解析器会根据 frontmatter 中的字段自动识别 target 类型。

**Team evals.md 示例**：

```markdown
---
team_name: ops-direct-invoke
eval_mode: text
---

# Case 1: 基本算子开发流程问答

## Config
- Max Tokens: 200000
- Timeout: 900

## Prompt

我想开发一个 Ascend C Kernel 直调算子，计算两个向量的逐元素加法。请描述开发这个算子的完整流程。

## Expected Output

回复应覆盖：环境检查、tiling 策略、host/device 代码结构、代码审查、性能验收

## Expectations

- [contains] kernel
- [contains] tiling
```

**用例字段说明**：

| 字段 | 必填 | 说明 |
|------|------|------|
| `# Case N` | 是 | 用例标题，N 从 1 开始连续递增 |
| `## Config` | 否 | 用例级配置，详见下方 Config 字段说明 |
| `## Prompt` | 是 | 发送给 AI 的测试问题 |
| `## Expected Output` | 是 | 对 AI 回复的语义预期，描述应覆盖的关键要点 |
| `## Expectations` | 否 | 模式匹配规则列表 |

**Config 字段说明**：

| Config Key | 说明 |
|------------|------|
| `Max Tokens` | Token 消耗硬上限，超过则测试失败 |
| `Max Tokens (<model>)` | 按模型指定 Token 上限，如 `Max Tokens (deepseek-v4-flash): 140000`。通过 `--eval-model` 或 `EVAL_MODEL` 环境变量指定模型 |
| `Eval Mode` | 覆盖用例级评测模式：`text` / `file_based` |
| `Distractor skills` | 正向看护：分号分隔的干扰 skill 列表，如 `skill-a;skill-b`。这些 skill 会被部署到沙箱中，验证 AI 在多个 skill 存在时仍能正确选择目标 skill |
| `Ascend Platform` | 用例适用的昇腾平台，分号分隔可多选，如 `A2;A5`。配合 `--ascend-platform` 参数在对应服务器上执行。**未配置此字段的用例在任何平台下均不执行** |
| `Disabled` | 设为 `true` 则跳过该用例（Phase 2 执行时显示为 SKIPPED）。适用于尚未调试完成的用例。默认不启用。有效值：`true`、`yes`、`1` |
| `Timeout` | 用例执行超时时间（秒）。正整数，未配置时默认 600s。用于需要更长执行时间的复杂场景 |
| `覆盖度阈值` / `准确性阈值` / `质量阈值` / `Token阈值` | 按维度覆盖默认通过阈值。覆盖度默认 20/40，准确性 15/30，质量 10/20，Token 3/10。如 `覆盖度阈值: 25` |
| `Truncate Length` | AI 回复传递给评审 Agent 时截断长度（字符数）。默认 30000。当 AI 回复较长时（如包含大段代码），评审 Agent 可能因回复截断看不到完整内容而误判。可按需增大，如 `Truncate Length: 60000` |

**expectations 类型**：

| type | 必填字段 | 说明 |
|------|---------|------|
| `contains` | `pattern` | AI **最终回复**（`ai_text`）中必须包含该字符串。不检查工具调用过程的中间输出 |
| `not_contains` | `pattern` | AI **最终回复**中不得包含该字符串（不检查工具调用过程中的参考文档内容） |
| `file_exists` | `path` | 指定文件必须存在（搜索顺序：`sandbox/<path>` → `sandbox/.opencode/skills/<skill>/<path>` → `skill_dir/<path>`） |
| `file_list` | `pattern` | 沙箱中存在匹配 glob pattern 的文件 |
| `file_contains` | `pattern` | 沙箱中匹配 glob 的文件至少有一个包含所有指定文本。格式：`path : "p1";"p2"`，路径支持 glob 通配符 |
| `skill_activated` | `pattern` | 程序化验证 AI 执行过程中加载了指定 skill（直接从 session 导出 JSON 的 tool_use 事件中精确匹配，不依赖评审模型。用于正向看护场景） |

**编写 expected_output 的建议**：

- 描述语义要点，不要求逐字匹配 — 写"回复应提供至少一种验证方法"，不写"回复必须包含 `acl.init()`"
- 聚焦核心信息覆盖 — 评测模型会判断 AI 是否遗漏关键信息
- 避免过于精确的措辞约束 — AI 模型输出是非确定性的

## 执行测试

所有命令在 `tests/system/scripts/` 目录下执行。

### 环境准备

```bash
pip install -r tests/system/scripts/requirements.txt
```

依赖项：`pytest`、`PyYAML`、`pytest-html`、`pytest-metadata` 等。

### 方式一：直接运行 pytest（开发调试）

**Phase 1 — 静态结构检查**（秒级，无需 opencode）：

```bash
cd tests/system/scripts

# 测试指定 skill
python -m pytest test_skill_basic.py -v -k "cann-env-setup"

# 测试所有含 evals.json 的 skill
python -m pytest test_skill_basic.py -v
```

**Phase 2 — AI 语义评测**（分钟级，需要 opencode CLI）：

```bash
cd tests/system/scripts

# 测试指定 skill 的全部 eval 用例
python -m pytest test_skill_evals.py --skill cann-env-setup -v --tb=short

# 测试指定 skill 的单个用例
python -m pytest test_skill_evals.py --skill cann-env-setup --eval-id 3 -v --tb=long

# 按平台过滤（仅执行 Ascend Platform 配置为 A2 的用例）
python -m pytest test_skill_evals.py --skill cann-env-setup --ascend-platform A2 -v --tb=short

# 多平台过滤（执行 A2 和 A5 的用例）
python -m pytest test_skill_evals.py --skill cann-env-setup --ascend-platform A2 --ascend-platform A5 -v

# 启用重试（默认 1=不重试），适用于偶发性失败
EVAL_EXEC_RETRIES=3 python -m pytest test_skill_evals.py --skill cann-env-setup -v
```

**Team Phase 1 — 静态结构检查**（秒级，无需 opencode）：

```bash
cd tests/system/scripts

# 测试指定 team
python -m pytest test_team_basic.py -v -k "ops-direct-invoke"

# 测试所有有 evals.md 的 team
python -m pytest test_team_basic.py -v
```

**Team Phase 2 — AI 语义评测**（分钟级，需要 opencode CLI）：

```bash
cd tests/system/scripts

# 测试指定 team 的全部 eval 用例
python -m pytest test_team_evals.py --team ops-direct-invoke -v --tb=short

# 测试指定 team 的单个用例
python -m pytest test_team_evals.py --team ops-direct-invoke --eval-id 1 -v --tb=long

# 按平台过滤
python -m pytest test_team_evals.py --team ops-direct-invoke --ascend-platform A2 -v

# 指定评估模型（匹配 Max Tokens (<model>) 预算）
EVAL_MODEL=claude-sonnet-4-20250514 python -m pytest test_team_evals.py --team ops-direct-invoke -v
```

### 方式二：main.py 一键执行（CI 门禁）

```bash
python tests/system/scripts/main.py \
    --repo-root /mnt/workspace/gitCode/cann/cannbot-skills \
    --changed-files ops/cann-env-setup/SKILL.md
```

`main.py` 自动完成：识别受影响的 skill/team → 逐个 Phase 1（失败的跳过 Phase 2）→ 合并 Phase 2 → 生成统一 HTML 报告 → 归档日志。

支持的参数：

| 参数 | 说明 |
|------|------|
| `--eval-model <model>` | 指定评测模型名称，用于按模型匹配 `Max Tokens (<model>)` 预算 |
| `--parallel` / `-p` | 并发数，`1` 顺序执行（默认），`auto` 自动取核数，最大 32 |
| `--ascend-platform A2 A3` | 按平台过滤，仅执行 `Ascend Platform` 匹配的用例。不指定则跳过评测 |
| `--report-only` | 仅重新生成 HTML 报告，不执行测试（从已有沙箱 JSON 文件读取数据） |
| `--eval-id <id>` | 仅执行指定 ID 的单个用例 |

### 方式三：gate_check.sh（完整 CI 流程）

```bash
# 自动检测变更文件，执行评测
./tests/gate_check.sh

# 指定平台（仅执行 A2 用例），多平台可重复
./tests/gate_check.sh --ascend-platform A2 --ascend-platform A5

# 通过环境变量指定平台（逗号或空格分隔）
ASCEND_PLATFORM="A2,A5" ./tests/gate_check.sh

# 重复执行多次（稳定性测试）
./tests/gate_check.sh --repeat 3
./tests/gate_check.sh --ascend-platform A2 --repeat 5
```

> **注意**：`gate_check.sh` 要求必须指定平台（`--ascend-platform` 或 `ASCEND_PLATFORM` 环境变量），否则直接 exit 0 跳过评测。平台值仅支持 `A2`、`A3`、`A5`。

`gate_check.sh` 支持以下参数：

| 参数 | 说明 |
|------|------|
| `--ascend-platform <A2\|A3\|A5>` | 指定目标昇腾平台，可多次指定多平台 |
| `--repeat <N>` | 重复执行 N 次门禁检查（默认 1），用于稳定性测试 |

## 结果解读

### 测试输出

Phase 2 通过时 stdout 显示：

```
--- AI Response (eval 1) ---
使用 `npu-smi info` 命令检查 NPU 驱动...
--- End AI Response ---

[REVIEW RESULT] {"type": "step_finish", ...}

test_skill_evals.py::test_eval_case[cann-env-setup::eval_1] PASSED
```

Phase 2 失败时显示具体原因：

```
AssertionError: Eval 3: expected_output check failed
Reviewer reason: 遗漏了预期要点：Python 导入 acl 模块验证
--- AI Response (by execution session) ---
CANN 安装完成后，可通过以下方式验证：
1. npu-smi info 查看 NPU 设备状态
2. cat /usr/local/Ascend/version.cfg 查看版本信息
--- End AI Response ---
```

### 结果文件

一次完整执行会产生以下文件：

#### results/ — 测试报告（查看结果用）

| 文件 | 来源 | 用途 |
|------|------|------|
| `basic_validation.html` | Phase 1 pytest-html | **静态结构检查报告**，浏览器打开可看 16 项测试（evals.md 格式、SKILL.md frontmatter 等）的通过/失败详情 |
| `team_basic_validation.html` | Team Phase 1 pytest-html | **Team 静态结构检查报告**，包含 AGENTS.md/plugin.json/init.sh 等 20 项测试 |
| `ST_validation_report_<YYYYMMDD_HHMMSS>.html` | Phase 2 统一 pytest-html | **Skill + Team 统一 AI 语义评测报告**，所有 target 合并展示，表中"类型"列区分 Skill/Team |
| `<skill>_<timestamp>.json` | `main.py` 的 `save_results()` | **结构化结果 JSON**，供脚本/CI 解析，含每个用例的 prompt、expected_output、实际输出、通过状态 |

#### logs/ — 运行时日志（排查问题用）

| 文件 | 数量 | 来源 | 用途 |
|------|------|------|------|
| `<skill>_case_X.json` | 每用例 1 个 | `opencode_runner._save_session_info()` | 执行 session 的 ID 记录，内含 opencode session ID 和时间戳 |
| `<skill>_case_X_review_ses.json` | 每用例 1 个 | `opencode_runner.export_session_data()` | **评测 session 完整导出**，含评审模型收到的 prompt、思考链、输出的 pass/fail 判定。当 expected_output 报"无法解析判定结果"时，查这个文件看评测模型实际输出 |
| `test_results_<timestamp>.zip` | 1 | `main.py` 的 `archive_logs_and_results()` | logs + results 的打包归档，供 CI 流水线下载 |

#### 文件生成关系

```
一次完整执行
  │
  ├─ Phase 1 ──→ results/basic_validation.html (Skill)
  │               results/team_basic_validation.html (Team)
  │
  ├─ Phase 2 ──→ 每个用例生成:
  │                logs/<target>_case_X.json              (执行 session ID)
  │                logs/<target>_case_X_review_ses.json   (评测 session 完整对话)
  │              ↓
  │              results/ST_validation_report_<ts>.html   (统一报告，含 Skill+Team)
  │
  └─ 归档 ────→ logs/test_results_<timestamp>.zip    (以上全部打包)
```

**日常使用建议**：看结果打开 `results/` 下的 HTML 报告，排查问题查 `logs/` 下对应的 `review_ses.json`。

## 常见问题

### pytest 报 `unrecognized arguments: --html`

缺少 `pytest-html` 插件，执行：
```bash
pip install pytest-html
```

### 测试全部 ERROR（NameError: SKILLS_DIR）

框架版本过旧，`SKILLS_DIR` 变量已移除。确保使用最新版本的 `test_skill_evals.py`。

### expected_output 检查持续失败

检查 `expected_output` 是否过于严格——不要描述"AI 应该说什么话"，描述"AI 回复应覆盖哪些要点"。

### 评测 session 返回"无法解析判定结果"

该问题已在 2026-06 版本解决：评审机制从易出错的 JSON 解析改为 **review-template.md 模板化方案**。评审 Agent 通过 Write 工具填写沙箱中的 `review-template.md` 模板，框架通过正则从模板中提取结构化评审结果（Status + Score + 各维度得分），不再依赖 JSON 输出格式。

如仍有问题，检查 `logs/<skill>_case_X_review_ses.json` 中的原始评测输出，确认 `review-template.md` 是否被正确填写。

### 用例偶发性失败需要重试

设置环境变量 `EVAL_EXEC_RETRIES` 控制重试次数（默认 1，即不重试）：
```bash
EVAL_EXEC_RETRIES=3 python -m pytest tests/system/scripts/test_skill_evals.py --skill cann-env-setup -v
```
重试会重新执行 opencode session 并重新评审，适用于网络抖动或 AI 服务器偶发错误。

### 添加新 skill 目录

在 `tests/system/config/st-test.config` 的 `skill_dirs` 中追加新路径即可，例：
```yaml
skill_dirs:
  - "ops"
  - "graph"
  - "model/skills"
  - "my-new-dir/skills"   # 新增
```

### 添加新 team 的 ST 看护

1. 在 `tests/system/cases/` 下创建 `<team_name>_evals.md`（使用 `team_name` frontmatter 字段）
2. 确保 `config/st-test.config` 中的 `team_dirs` 包含该 team 所在目录
3. 将 team 名称加入 `team_whitelist`（如已启用白名单）
4. 运行 Team Phase 1 验证格式正确

### Skill 和 Team 测试的区别

| 维度 | Skill 测试 | Team 测试 |
|------|-----------|----------|
| 源码目录 | `ops/`、`graph/`、`model/` 等 | `plugins-official/`、`plugins-community/` |
| 标识文件 | `SKILL.md` | `AGENTS.md` + `.claude-plugin/plugin.json` |
| evals.md frontmatter | `skill_name: <name>` | `team_name: <name>` |
| 沙箱部署方式 | symlink skill 目录到 `.opencode/skills/` | 执行 `init.sh project opencode <sandbox>` |
| Phase 1 检查项 | SKILL.md 格式、evals.md 结构 | AGENTS.md 格式、plugin.json 合法性、init.sh 存在性 |
| Phase 2 统一报告 | `ST_validation_report_<ts>.html`（表中"类型"列区分 Skill/Team） |
