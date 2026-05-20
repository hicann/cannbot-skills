# Skill Test Framework 使用指南

## 功能特性

Skill Test Framework 是 cannbot-skills 的技能质量看护框架，提供以下能力：

- **多目录扫描** — 通过 `config/skill-test.config` 配置跨多个目录发现 skill，不限于单一 `skills/` 目录
- **变更驱动测试** — 基于 Git 变更文件自动识别受影响的 skill，按需执行评测
- **双层验证** — Phase 1 静态结构检查 + Phase 2 AI Eval 语义评测
- **独立评测 Session** — Phase 2 使用独立 opencode session 评审执行结果，避免自我检查偏差
- **HTML + JSON 报告** — 输出可视化和结构化测试报告，便于 CI/CD 集成和人工排查

## 架构概览

```
skill-test-framework/
├── config/
│   └── skill-test.config       # skill 扫描路径配置
├── scripts/
│   ├── main.py                 # CI 门禁入口（一键执行）
│   ├── conftest.py             # pytest 配置 + skill 扫描逻辑
│   ├── test_skill_basic.py     # Phase 1: 静态结构验证
│   ├── test_skill_evals.py     # Phase 2: AI 语义评测
│   └── opencode_runner.py      # opencode CLI 封装
├── evals/                      # 框架自身的评测用例
├── results/                    # 测试报告输出
├── logs/                       # opencode session 日志 + 归档
└── docs/                       # 文档
```

### 两阶段测试

```
输入: changed_files
  │
  ├─ Phase 1: 静态结构检查 (test_skill_basic.py)
  │   ├─ evals.json 存在性、合法性、结构完整性
  │   ├─ SKILL.md 存在性、frontmatter 必填字段
  │   └─ 用例 ID 唯一性、顺序正确性
  │
  └─ Phase 2: AI 语义评测 (test_skill_evals.py)
      ├─ 执行 Session: opencode 加载 skill 处理 prompt → AI 回复
      ├─ 评测 Session: 独立 opencode session 评审回复质量
      │   ├─ 输入: 原始问题 + AI 思考链 + AI 回复 + 预期要点
      │   └─ 输出: pass/fail + 判定依据
      └─ 模式匹配: expectations 中的 contains/not_contains 检查
```

## 配置

### skill-test.config

```yaml
# 扫描 skill 的目录列表（相对于仓库根目录）
skill_dirs:
  - "ops"
  - "graph"
  - "model/skills"

# 排除的 skill 名称
exclude_skills:
  - "skill-test-framework"
```

### evals.json 格式

每个 skill 的评测用例定义在 `<skill_dir>/evals/evals.json`：

```json
{
  "skill_name": "cann-env-setup",
  "evals": [
    {
      "id": 1,
      "prompt": "我有一台昇腾服务器，想检查NPU驱动是否已安装，应该用什么命令？",
      "expected_output": "回复应说明使用 npu-smi info 命令检查驱动，并解释如何根据命令输出判断驱动是否已安装",
      "files": [],
      "expectations": [
        {"type": "contains", "pattern": "npu-smi info"}
      ]
    }
  ]
}
```

**字段说明**：

| 字段 | 必填 | 说明 |
|------|------|------|
| `id` | 是 | 用例编号，需从 1 开始连续递增 |
| `prompt` | 是 | 发送给 AI 的测试问题 |
| `expected_output` | 是 | 对 AI 回复的语义预期，描述应覆盖的关键要点。独立评测 session 据此评判回复质量 |
| `files` | 是 | 文件列表（通常为空数组） |
| `expectations` | 否 | 模式匹配规则列表 |

**expectations 类型**：

| type | 必填字段 | 说明 |
|------|---------|------|
| `contains` | `pattern` | 执行 session 的原始输出中必须包含该字符串 |
| `not_contains` | `pattern` | 执行 session 的原始输出中不得包含该字符串 |
| `file_exists` | `path` | 指定文件必须存在（`path` 相对于 skill 目录） |

**编写 expected_output 的建议**：

- 描述语义要点，不要求逐字匹配 — 写"回复应提供至少一种验证方法"，不写"回复必须包含 `acl.init()`"
- 聚焦核心信息覆盖 — 评测模型会判断 AI 是否遗漏关键信息
- 避免过于精确的措辞约束 — AI 模型输出是非确定性的

## 执行测试

所有命令在 `ops/skill-test-framework/scripts/` 目录下执行。

### 环境准备

```bash
pip install -r ops/skill-test-framework/scripts/requirements.txt
```

依赖项：`pytest`、`PyYAML`、`pytest-html`、`pytest-metadata` 等。

### 方式一：直接运行 pytest（开发调试）

**Phase 1 — 静态结构检查**（秒级，无需 opencode）：

```bash
cd ops/skill-test-framework/scripts

# 测试指定 skill
python -m pytest test_skill_basic.py -v -k "cann-env-setup"

# 测试所有含 evals.json 的 skill
python -m pytest test_skill_basic.py -v
```

**Phase 2 — AI 语义评测**（分钟级，需要 opencode CLI）：

```bash
cd ops/skill-test-framework/scripts

# 测试指定 skill 的全部 eval 用例
python -m pytest test_skill_evals.py --skill cann-env-setup -v --tb=short

# 测试指定 skill 的单个用例
python -m pytest test_skill_evals.py --skill cann-env-setup --eval-id 3 -v --tb=long
```

### 方式二：main.py 一键执行（CI 门禁）

```bash
python ops/skill-test-framework/scripts/main.py \
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
| `basic_validation.html` | Phase 1 pytest-html | **静态结构检查报告**，浏览器打开可看 16 项测试（evals.json 格式、SKILL.md frontmatter 等）的通过/失败详情 |
| `<skill>_evals_validation.html` | Phase 2 pytest-html | **AI 语义评测报告**，浏览器打开可看每个 eval 用例的通过/失败、AI 回复原文、评测模型判定依据 |
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
  │              results/<skill>_evals_validation.html   (汇总报告)
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

在 `config/skill-test.config` 的 `skill_dirs` 中追加新路径即可，例：
```yaml
skill_dirs:
  - "ops"
  - "graph"
  - "model/skills"
  - "my-new-dir/skills"   # 新增
```
