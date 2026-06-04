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
  │   └─ 用例 ID 唯一性、顺序正确性
  │
  └─ Phase 2: AI 语义评测 (test_skill_evals.py / test_team_evals.py)
      ├─ 执行 Session: opencode 加载 skill/team 处理 prompt → AI 回复
      │   (Team: 通过 init.sh 在沙箱中安装完整 team 环境)
      ├─ 评测 Session: 独立 opencode session 评审回复质量
      │   ├─ 输入: 原始问题 + AI 思考链 + AI 回复 + 预期要点
      │   └─ 输出: pass/fail + 判定依据
      └─ 模式匹配: expectations 中的 contains/not_contains 检查
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
| `Max Tokens (<model>)` | 按模型指定 Token 上限，如 `Max Tokens (deepseek-v4-flash): 140000` |
| `Eval Mode` | 覆盖用例级评测模式：`text` / `file_based` |
| `Distractor skills` | 正向看护：分号分隔的干扰 skill 列表，验证 AI 在多个 skill 存在时仍能正确选择目标 skill |
| `Disabled` | 设为 `true` 则跳过该用例（Phase 2 执行时显示为 SKIPPED）。适用于尚未调试完成的用例。默认不启用 |
| `Timeout` | 用例执行超时时间（秒）。正整数，未配置时默认 600s。用于需要更长执行时间的复杂场景 |
| `覆盖度阈值` / `准确性阈值` / `质量阈值` / `token阈值` | 按维度覆盖默认通过阈值 |

**expectations 类型**：

| type | 必填字段 | 说明 |
|------|---------|------|
| `contains` | `pattern` | 执行 session 的原始输出中必须包含该字符串 |
| `not_contains` | `pattern` | AI 最终回复中不得包含该字符串 |
| `file_exists` | `path` | 指定文件必须存在（`path` 相对于 skill 目录或沙箱） |
| `file_list` | `pattern` | 沙箱中存在匹配 glob pattern 的文件 |
| `skill_activated` | `pattern` | 程序化验证 AI 执行过程中加载了指定 skill（用于正向看护） |

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
```

### 方式二：main.py 一键执行（CI 门禁）

```bash
python tests/system/scripts/main.py \
    --repo-root /mnt/workspace/gitCode/cann/cannbot-skills \
    --changed-files ops/cann-env-setup/SKILL.md
```

`main.py` 自动完成：识别受影响的 skill → Phase 1 → Phase 2 → 保存 JSON 结果 → 归档日志。

### 方式三：gate_check.sh（完整 CI 流程）

```bash
# 自动检测 HEAD 变更
./tests/gate_check.sh

# 指定变更文件
CHANGED_FILES="ops/cann-env-setup/SKILL.md" ./tests/gate_check.sh
```

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
| `evals_validation_<YYYYMMDD_HHMMSS>.html` | Phase 2 pytest-html | **Skill 统一 AI 语义评测报告**，所有技能合并展示，每行标注 skill 归属 |
| `team_evals_validation_<YYYYMMDD_HHMMSS>.html` | Team Phase 2 pytest-html | **Team AI 语义评测报告**，所有 team 合并展示，每行标注 team 归属 |
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
  ├─ Phase 1 ──→ results/basic_validation.html
  │
  ├─ Phase 2 ──→ 每个用例生成:
  │                logs/<skill>_case_X.json              (执行 session ID)
  │                logs/<skill>_case_X_review_ses.json   (评测 session 完整对话)
  │              ↓
  │              results/evals_validation.html   (统一报告)
  │
  └─ 归档 ────→ logs/test_results_<timestamp>.zip    (以上全部打包)
                results/<skill>_<timestamp>.json      (结构化摘要)
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

评测模型可能将 JSON 输出包裹在 markdown 代码块中（已兼容），如仍有问题，检查 `logs/<skill>_case_X_review_ses.json` 中的原始评测输出。

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
| Phase 2 报告文件名 | `evals_validation_<ts>.html` | `team_evals_validation_<ts>.html` |
