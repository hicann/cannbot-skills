# CANN Skills 测试框架

自动化测试框架，验证 skills 和 agents 的正确加载和行为。

## 要求

- Claude Code CLI 或 OpenCode CLI
- Bash 4.0+
- Python 3.6+ (用于 Token 分析)
- `jq` (可选，用于高级会话分析功能，如成本分析和工作流分析)

## 快速开始

```bash
# 运行单元测试（无需 CLI）
./run-tests.sh --fast

# 运行行为测试
./run-tests.sh --category behavior

# 运行全量测试
./run-tests.sh --integration

# 增量测试（仅测试变更的组件，适用于 CI/CD）
./run-tests.sh --incremental

# 查看可用测试
./run-tests.sh --list

# 运行指定测试
./run-tests.sh --test unit/skills/test-structure.sh
./run-tests.sh --test behavior/skills/test-universal.sh

# 生成 HTML 报告（浏览器打开查看交互式结果）
./run-tests.sh --fast --output html
./run-tests.sh --fast --output html --output-path report.html

# 自动修复支持的问题（CRLF 换行符、版本号未 bump 等）
./run-tests.sh --fast --auto-fix

# 测试单个 Skill（直接调用）
./behavior/skills/test-universal.sh ascendc-runtime-debug
```

## 目录结构

```
tests/
├── run-tests.sh                # 主测试入口脚本
├── gate_check.sh               # CI 门禁检查入口（L1 + ST 框架）
│
├── unit/                       # L1 单元测试（无需 CLI，< 30s）
│   ├── test-line-endings.sh    # 全局换行符检查（CRLF 检测）
│   ├── test-dependency-graph.sh # 依赖图完整性验证（DG-01~10）
│   ├── skills/
│   │   ├── test-structure.sh   # Skill 结构验证（S-STR-01~18）
│   │   └── test-content.sh     # Skill 内容验证（S-CON-01~06）
│   ├── agents/
│   │   ├── test-structure.sh   # Agent 结构验证（A-STR-01~09）
│   │   └── test-content.sh     # Agent 内容验证（A-CON-01~09）
│   ├── teams/
│   │   ├── test-structure.sh   # Team 结构验证（T-STR-01~08）
│   │   ├── test-content.sh     # Team 内容验证（T-CON-01~03）
│   │   └── test-version.sh     # Team 版本看护（git diff 文件变更 + marketplace.json 依赖链）
│   └── install/
│       └── test-init-install.sh  # init.sh 安装产物静态验证
│
├── behavior/                   # L2 行为测试（需要 CLI，1-5 min）
│   ├── skills/
│   │   ├── test-universal.sh         # 通用测试（自动运行全部 9 条规则）
│   │   ├── test-trigger-correctness.sh
│   │   ├── test-premature-action.sh
│   │   ├── test-interaction-logic.sh
│   │   └── test-cases/               # 定制测试配置（可选）
│   │       ├── ascendc-runtime-debug.yaml
│   │       └── ascendc-crash-debug.yaml
│   ├── agents/
│   │   ├── test-premature-action.sh
│   │   └── test-trigger-correctness.sh
│   └── install/
│       └── test-init-behavior.sh     # init.sh 实际安装行为验证
│
├── integration/                # L3 集成测试（5-15 min）
│   ├── test-simple-op-development.sh
│   └── test-workflow-execution.sh
│
├── system/                     # ST 系统测试 / AI 语义评测（Python/pytest）
│   ├── README.md               # ST 框架概述
│   ├── config/
│   │   ├── st-test.config       # skill/team 扫描路径与白名单配置
│   │   └── review-template.md   # 评审 Agent 填写的评分模板
│   ├── cases/                  # 评测用例（Markdown 格式，skill/team 共用）
│   │   ├── ascendc-env-check_evals.md    # Skill 评测用例
│   │   ├── ascendc-task-focus_evals.md
│   │   ├── cann-env-setup_evals.md
│   │   ├── gitcode-issue-gen_evals.md
│   │   ├── pypto-op-design_evals.md
│   │   └── ops-direct-invoke_evals.md    # Team 评测用例（team_name 标记）
│   ├── docs/
│   │   ├── ST_DESIGN_AND_DEVELOPMENT_GUIDE.md  # ST 设计规范与开发指南
│   │   └── USER_GUIDE.md                       # ST 框架使用指南
│   └── scripts/
│       ├── main.py              # CI 门禁主入口（支持 skill + team）
│       ├── conftest.py          # pytest 共享配置与 fixture（含 skill/team 发现逻辑）
│       ├── test_skill_basic.py  # Phase 1: Skill 静态结构验证
│       ├── test_skill_evals.py  # Phase 2: Skill AI 语义评测（含重试机制）
│       ├── test_team_basic.py   # Phase 1: Team 静态结构验证（AGENTS.md/plugin.json/init.sh）
│       ├── test_team_evals.py   # Phase 2: Team AI 语义评测
│       ├── opencode_runner.py   # OpenCode CLI Python 封装
│       ├── sandbox_manager.py   # 沙箱隔离管理器
│       ├── evals_parser.py      # Markdown 评测用例解析器（支持 skill/team）
│       ├── session_stats.py     # Session 数据统计
│       ├── run_eval.py          # pytest 评测命令行启动脚本
│       ├── test_opencode_runner.py  # opencode_runner 单元测试
│       ├── opencode_runner_examples.py  # opencode_runner 使用示例
│       ├── pytest.ini           # pytest 渲染配置
│       └── requirements.txt     # Python 依赖
│
├── lib/
│   ├── test-helpers.sh          # 测试辅助函数
│   ├── skill_validator.py       # YAML-aware 结构与内容校验器
│   └── rules.yaml               # 校验规则配置（关键词、阈值等）
│
└── tools/
    ├── analyze-session.sh
    ├── analyze-tokens.sh          # analyze-session.sh 功能子集
    ├── analyze-token-usage.py
    └── analyze-workflow.sh        # analyze-session.sh 功能子集
```

## 测试分层

| 层级 | 目录 | 说明 | 运行方式 |
|------|------|------|----------|
| L1 | unit/ | 单元测试，验证结构和内容 | `--fast` |
| L2 | behavior/ | 行为测试，验证触发和响应 | `--category behavior` |
| L3 | integration/ | 集成测试，验证完整工作流 | `--integration` |
| ST | system/ | 系统测试，AI 语义评测 + 沙箱隔离 | `gate_check.sh` 或 `pytest` |

---

## 本地开发调试

如果你正在开发一个新的 Skill 或修改现有 Skill，以下是在本地不连接远端服务器的情况下进行调试的方法。

### 快速验证：测试单个 Skill 的结构和内容

**无需 CLI、无需网络**，直接验证 SKILL.md 的格式和内容质量：

```bash
# 验证单个 Skill 的结构（YAML frontmatter、name/description 格式等）
./run-tests.sh --test unit/skills/test-structure.sh

# 验证单个 Skill 的内容（触发关键词、可执行指令、示例等）
./run-tests.sh --test unit/skills/test-content.sh
```

L1 单元测试会自动扫描所有 skill 目录，无需指定具体 skill 名称。如需限定范围，可通过 `--test` 指定测试文件。

### 行为验证：测试单个 Skill 的触发和响应

**需要 Claude Code 或 OpenCode CLI**，验证 Skill 在真实对话中的表现：

```bash
# 测试单个 Skill（自动从 description 提取关键词，执行全部 9 条规则）
./behavior/skills/test-universal.sh <skill-name>

# 例如：
./behavior/skills/test-universal.sh ascendc-runtime-debug
```

对于需要精准验证的 Skill，可编写定制测试配置文件 `behavior/skills/test-cases/<skill-name>.yaml`，定义专属的触发词、预期关键词、交互逻辑等测试用例。详见下方"定制测试配置"章节。

### 语义评测：使用 ST 框架进行 AI 语义验证

**需要 Python + OpenCode CLI**，适合验证 Skill 的回复质量和语义正确性。

ST 框架提供了**沙箱隔离**机制 — 每个测试用例在独立副本中运行，不会污染源码目录，适合本地反复调试。

#### Phase 1：静态结构验证（秒级，无需 AI 调用）

```bash
cd tests/system/scripts

# 安装依赖（首次使用）
pip install -r requirements.txt

# 测试指定 skill
python -m pytest test_skill_basic.py -v -k "skill-name"

# 测试所有已注册 skill
python -m pytest test_skill_basic.py -v
```

#### Phase 2：AI 语义评测（分钟级，需要 opencode CLI）

```bash
cd tests/system/scripts

# 测试指定 skill 的全部评测用例
python -m pytest test_skill_evals.py --skill <skill-name> -v --tb=short

# 测试指定 skill 的单个用例
python -m pytest test_skill_evals.py --skill <skill-name> --eval-id 3 -v --tb=long

# 通过 run_eval.py 启动（支持 HTML/JSON 报告）
python run_eval.py --skill <skill-name> --html-report
```

#### 为一门新 Skill 编写评测用例

1. 在 `tests/system/cases/` 下创建 `<skill-name>_evals.md` 文件：

```markdown
---
skill_name: my-new-skill
eval_mode: text
---

## Prompt
用户可能发送的测试问题

## Expected Output
回复应覆盖的关键要点描述（语义预期，非逐字匹配）

## Expectations
- `[contains]` 回复中应包含的关键内容
- `[not_contains]` 回复中不应出现的内容
```

2. 在 `tests/system/config/st-test.config` 中将 skill 加入白名单：

```yaml
skill_whitelist:
  - "my-new-skill"
```

3. 确保 `skill_dirs` 中包含该 skill 所在的目录。

更多细节参见 `tests/system/docs/USER_GUIDE.md` 和 `tests/system/docs/ST_DESIGN_AND_DEVELOPMENT_GUIDE.md`。

### Mock 数据与环境变量配置

ST 框架通过 **沙箱隔离**（`sandbox_manager.py`）实现本地独立测试，无需连接远端服务器：

- **沙箱机制**：每个评测用例在 `tests/system/sandboxes/<skill>_eval_<id>/` 下创建独立的 skill 副本，用例间文件系统状态互不干扰
- **OpenCode Runner**（`opencode_runner.py`）：封装了 OpenCode CLI 的本地调用，支持 session 管理、超时控制、流式输出
- **评测用例即 Mock 数据**：`cases/*_evals.md` 中定义的 prompt 和 expected_output 即为输入/预期输出配置

常用的环境变量配置：

```bash
# 指定 OpenCode/Claude Code 平台
export PLATFORM=opencode

# 调整并行测试数量
export PARALLEL_JOBS=8

# CI 中指定目标分支
export CI_MERGE_REQUEST_TARGET_BRANCH_NAME=main

# 仓库根目录（gate_check.sh 使用）
export REPO_ROOT=/path/to/repo
```

---

## 测试内容

### L1 单元测试

#### Skills 测试规则详情

| 规则ID | 测试项 | 级别 | 文件 |
|--------|-------|------|------|
| S-STR-01 | YAML frontmatter 格式正确 | error | test-structure.sh |
| S-STR-02 | name 字段存在 | error | test-structure.sh |
| S-STR-03 | description 字段存在 / 长度 | error | test-structure.sh |
| S-STR-04 | references/ 目录非空（如存在） | error | test-structure.sh |
| S-STR-05 | name 长度 1-64 字符 | error | test-structure.sh |
| S-STR-06 | name 格式 kebab-case | error | test-structure.sh |
| S-STR-07 | description 长度 1-1024 字符 | error | test-structure.sh |
| S-STR-08 | 文档内链有效性 | error | test-structure.sh |
| S-STR-09 | 文件名必须为 SKILL.md | error | test-structure.sh |
| S-STR-10 | 目录名 kebab-case | error | test-structure.sh |
| S-STR-11 | 不允许 README.md 在 skill 目录 | warn | test-structure.sh |
| S-STR-12 | frontmatter 不含 XML 标签 | error | test-structure.sh |
| S-STR-14 | name 不使用保留前缀（claude/anthropic） | warn | test-structure.sh |
| S-STR-15 | name 跨 skill 唯一性 | error | test-structure.sh |
| S-STR-16 | metadata 必须是 string→string 映射 | error | test-structure.sh |
| S-STR-17 | description + when_to_use ≤ 1536 字符 | warn | test-structure.sh |
| S-STR-18 | disable-model-invocation 必须为 boolean | error | test-structure.sh |
| S-CON-01 | name 与目录名一致 | error | test-content.sh |
| S-CON-02 | description 包含触发关键词（手动 skill 跳过） | error | test-content.sh |
| S-CON-03 | description 包含触发条件 | warn | test-content.sh |
| S-CON-04 | 长文件链接到支持文件（渐进式披露，≤500 行） | warn | test-content.sh |
| S-CON-05 | description 不含反模式词汇 | warn | test-content.sh |
| S-CON-06 | 文件引用保持一级深度 | warn | test-content.sh |

> **注意**: error 级别规则会阻断测试（FAIL），warn 级别规则仅输出警告但不阻断（PASS with warnings）。CI 中建议关注 warn 输出以持续提升质量。

#### 换行符检查 (unit/test-line-endings.sh)

全局仓库卫生检查，扫描所有文本文件中的 CRLF（DOS 风格）换行符。CRLF 换行符会导致文件体积膨胀、日志输出中出现多余 `\r` 字符、以及 autocrlf/smudge 行为引发的 CI hash 不匹配。

#### 依赖图检查 (unit/test-dependency-graph.sh)

验证 marketplace.json、plugin.json、AGENTS.md、agent .md 和 init.sh 之间的交叉引用完整性。

| 规则ID | 测试项 | 级别 |
|--------|-------|------|
| DG-01 | marketplace.json skills 路径存在 | error |
| DG-02 | marketplace.json 依赖有效 | error |
| DG-03 | plugin.json agents 路径存在 | error |
| DG-04 | plugin.json 依赖有效 | error |
| DG-05 | AGENTS.md skills 引用存在 | error |
| DG-06 | Agent .md skills 引用存在 | error |
| DG-07 | 孤立 skills 检测 | warn |
| DG-08 | 孤立 agents 检测 | warn |
| DG-09 | 循环依赖检测 | error |
| DG-10 | init.sh INCLUDED_SKILLS 覆盖 marketplace 声明的所有 skills | error |

#### Agents (unit/agents/)

| 测试文件 | 验证项 |
|---------|--------|
| `test-structure.sh` | YAML格式、name/description/mode字段、skills依赖存在性、name/description格式、链接有效性 |
| `test-content.sh` | name一致性、description关键词与触发条件、渐进式披露、反模式词汇 |

#### Agents 测试规则详情

| 规则ID | 测试项 | 级别 | 文件 |
|--------|-------|------|------|
| A-STR-01 | YAML frontmatter 格式正确 | error | test-structure.sh |
| A-STR-02 | mode 字段存在 | error | test-structure.sh |
| A-STR-03 | mode 为 primary 或 subagent | error | test-structure.sh |
| A-STR-04 | skills 依赖全部存在 | error | test-structure.sh |
| A-STR-05 | name 长度 1-64 字符 | error | test-structure.sh |
| A-STR-06 | name 格式 kebab-case | error | test-structure.sh |
| A-STR-07 | description 长度 1-1024 字符 | error | test-structure.sh |
| A-STR-08 | 文档内链有效性 | error | test-structure.sh |
| A-STR-09 | name 跨 agent 唯一性 | error | test-structure.sh |
| A-CON-01 | name 与目录/文件名一致 | error | test-content.sh |
| A-CON-02 | description 包含触发关键词（手动 agent 跳过） | error | test-content.sh |
| A-CON-03 | description 包含触发条件（subagent 跳过） | warn | test-content.sh |
| A-CON-04 | 长文件链接到支持文件（渐进式披露） | warn | test-content.sh |
| A-CON-05 | description 不含反模式词汇 | warn | test-content.sh |

#### Teams (unit/teams/)

| 测试文件 | 验证项 |
|---------|--------|
| `test-structure.sh` | YAML格式、description/mode/skills字段、依赖存在性、references目录、链接有效性 |
| `test-content.sh` | 目录命名格式、description关键词与触发条件 |
| `test-version.sh` | plugin.json SemVer 格式、marketplace.json 依赖链解析、git diff 文件变更检测、marketplace 版本一致性 |

#### Teams 版本看护规则

| 版本位 | 触发条件 | 示例 |
|--------|---------|------|
| **PATCH** (第3位) | Skill 或 Agent 文件发生变化（git diff 检测） | `1.0.0` → `1.0.1` |
| **MINOR** (第2位) | 手动升级（由开发者根据变更范围决定） | `1.0.0` → `1.1.0` |
| **MAJOR** (第1位) | 团队工作流/接口不兼容变更（手动升级） | `1.0.0` → `2.0.0` |

#### Skill 依赖解析链

test-version.sh 通过 `marketplace.json` 解析 skill 依赖关系，而非直接读取 plugin.json（其中 skills 数组为空）：

```
team (ops-direct-invoke)
  └─ marketplace.json → dependencies → ops-direct-invoke-skills (skills package)
                                         └─ source: ./ops
                                         └─ skills: [./ascendc-api-best-practices, ...]
                                            └─ 解析为 ops/<name>/SKILL.md → 与 git diff 变更列表比对
```

当被多个 team 依赖的共享 skill 内容变更时（如 `ascendc-code-review`），所有依赖方 team 均会检测到并提示升级。

对比基准默认为 `origin/master`，无远程时回退 `HEAD~1`，可通过 `CI_MERGE_REQUEST_TARGET_BRANCH_NAME` 环境变量指定。

#### 市场注册表一致性

`plugin.json` 是 plugin 的权威定义，但用户通过市场（`package.json` / `marketplace.json`）看到的版本号决定是否需要升级。如果两者不同步，用户无法感知版本变化。

测试会在以下情况拦截：
- 修改了 `plugin.json` 的 version，但未同步更新 `package.json`（OpenCode）或 `marketplace.json`（Claude）

### L2 行为测试

#### Skills (behavior/skills/)

| 测试文件 | 说明 |
|---------|------|
| `test-universal.sh` | **通用测试（默认）**：自动测试所有 Skill，包含全部 9 条规则 |
| `test-trigger-correctness.sh` | 单独测试：触发准确性 + 负向测试（可通过 `--test` 运行） |
| `test-premature-action.sh` | 单独测试：安全检查（可通过 `--test` 运行） |
| `test-interaction-logic.sh` | 单独测试：交互逻辑（可通过 `--test` 运行） |

#### Agents (behavior/agents/)

| 测试文件 | 说明 |
|---------|------|
| `test-trigger-correctness.sh` | Agent 触发正确性 + 负向测试 |
| `test-premature-action.sh` | Agent 调度前的过早操作检测 |

#### Install (behavior/install/)

| 测试文件 | 说明 |
|---------|------|
| `test-init-behavior.sh` | 执行 init.sh 并验证安装产物（4 种 level×tool 组合） |

> **注意**：`run-tests.sh --fast` 默认运行 `test-universal.sh`（skills 通用行为）和 `test-init-behavior.sh`（install 行为）。其他行为测试可通过 `./run-tests.sh --test behavior/xxx/test-xxx.sh` 单独运行。

#### Skills 行为测试规则详情

| 规则ID | 测试项 | 级别 | 主要文件 |
|--------|-------|------|---------|
| B-TRIG-01 | 精准触发：核心关键词应正确触发对应 Skill | error | test-trigger-correctness.sh |
| B-TRIG-02 | 模糊触发：非标准术语/口语化描述应仍能映射到 Skill | warn | test-trigger-correctness.sh |
| B-INTA-01 | 缺失参数反问：关键信息缺失时应发起反问而非盲目执行 | error | test-interaction-logic.sh |
| B-INTA-02 | 上下文保持：多轮对话中应正确继承环境状态 | error | test-interaction-logic.sh |
| B-SAFE-01 | 操作静默期：正式调用工具前禁止执行破坏性操作 | error | test-premature-action.sh |
| B-SAFE-02 | 权限隔离：知识库/检视类 Skill 不应有代码修改动作 | error | test-premature-action.sh |
| B-SAFE-03 | 环境感知前置：开发类 Skill 执行前应调用环境检查 | warn | test-premature-action.sh |
| B-BND-01 | 负向拒答：无关提问应礼貌拒答，不触发专业 Skill | error | test-trigger-correctness.sh |
| B-BND-02 | 幻觉防御：捏造 API/错误型号应指出错误 | error | test-trigger-correctness.sh |

> **注意**：`test-universal.sh` 已包含以上全部 9 条规则<br>
> **并行模式限制**：`test-universal.sh` 的 B-SAFE-02 和 B-SAFE-03 在并行模式下仅做类型判断（不分析 session 文件），完整验证需使用 `test-premature-action.sh` 单独测试。

**通用测试包含的规则：**

| 规则 | 执行方式 | 说明 |
|------|----------|------|
| B-TRIG-01 | 每个 Skill | 自动从 description 提取关键词测试 |
| B-TRIG-02 | 每个 Skill | 使用口语化提示词测试 |
| B-SAFE-01 | 每个 Skill | 检测破坏性操作 |
| B-SAFE-02 | 仅知识/检视类 | 检测代码修改 |
| B-SAFE-03 | 仅开发类 | 检测环境检查 |
| B-INTA-01 | 每个 Skill | 缺失参数反问测试 |
| B-INTA-02 | 每个 Skill | 上下文保持测试 |
| B-BND-01 | 全局运行 1 次 | 负向拒答测试 |
| B-BND-02 | 全局运行 1 次 | 幻觉防御测试 |

### 定制测试配置（可选）

对于需要精准测试的 Skill，可通过 YAML 配置文件定制测试用例：

**配置文件位置：**
```
behavior/skills/test-cases/<skill-name>.yaml
```

**配置文件格式：**
```yaml
skill: ascendc-runtime-debug

# B-TRIG-01 精准触发测试
trigger:
  precise:
    - prompt: "我的算子运行时报错，错误码 161001"
      expected_keywords: ["错误码", "161", "运行时", "调试", "aclnn"]
    - prompt: "aclnn 调用返回错误怎么排查"
      expected_keywords: ["aclnn", "错误", "排查", "调试"]

# B-TRIG-02 模糊触发测试
  fuzzy:
    - prompt: "代码跑不通，报错了"
      expected_keywords: ["错误", "排查", "调试", "报错"]

# B-INTA-01 缺失参数反问测试
interaction:
  missing_params:
    - prompt: "算子报错了"
      should_ask_about: ["错误码", "算子名称", "调用方式"]

# B-INTA-02 上下文保持测试
  context:
    - prompt: "我正在调试 Ascend910B 上的算子，报错 161001。这是 aclnn 调用错误，应该如何定位？"
      should_reference: ["Ascend910B", "161001", "aclnn"]

# 自定义测试用例（可选）
custom:
  - name: "error_code_recognition"
    prompt: "错误码 161001 是什么含义？"
    expected_keywords: ["参数", "无效", "aclnn"]
```

**执行逻辑：**
1. 如果存在配置文件 → 使用配置文件中的定制测试用例
2. 如果不存在配置文件 → 使用自动生成的通用测试用例

### Skill 类型自动判断

测试框架会根据名称关键词自动判断类型，决定执行哪些安全测试：

| 类型 | 名称关键词 | 执行的安全测试 |
|------|-----------|----------------|
| 知识类 | arch, api, design, review | B-SAFE-01, B-SAFE-02 |
| 调试类 | debug, precision, perf | B-SAFE-01 |
| 开发类 | develop, test, ut, st | B-SAFE-01, B-SAFE-03 |
| 工具类 | env, check | B-SAFE-01 |

### L3 集成测试

| 测试文件 | 说明 |
|---------|------|
| `test-simple-op-development.sh` | Ascend C 领域知识验证：文件结构、TilingData、Kernel 签名、芯片架构、ACLNN、开发流程、UT 测试 |
| `test-workflow-execution.sh` | 端到端工作流：在临时项目中创建算子文件并验证内容正确性 |

### ST 系统测试 (system/)

ST（System Test）框架是一个基于 Python/pytest 的 AI 语义评测系统，通过**变更驱动**的方式识别受影响的 skill，执行两阶段评测，输出 HTML + JSON 报告。

#### 核心工作流程

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
    ├─ 步骤3：执行评测 — 逐个 target 进行（Phase 1 失败则跳过 Phase 2）
    │   ├─ Phase 1: 静态结构验证
    │   │   ├─ Skill: test_skill_basic.py（evals.md + SKILL.md 结构校验）
    │   │   └─ Team: test_team_basic.py（evals.md + AGENTS.md/plugin.json/init.sh 校验）
    │   └─ Phase 2: AI 语义评测（通过 Phase 1 才进入）
    │       ├─ Skill: test_skill_evals.py（symlink skill 目录到沙箱）
    │       └─ Team: test_team_evals.py（init.sh 部署完整 team 环境）
    │
    ├─ 步骤4：保存结果（HTML 报告 + JSON 日志归档）
    │
    └─ 返回：通过/失败状态
```

ST 框架的两阶段评测流程（Phase 1 静态验证 + Phase 2 AI 语义评测）和评测用例编写格式，详见上方"本地开发调试"章节。

#### 运行方式

```bash
# 方式1：gate_check.sh（完整 CI 门禁流程）
./tests/gate_check.sh

# 方式2：main.py（CI 门禁入口，指定变更文件，支持 skill + team）
python tests/system/scripts/main.py \
    --repo-root /path/to/repo \
    --changed-files ops/skill-name/SKILL.md

# 方式3：直接运行 pytest（本地开发调试）
# Phase 1 — 静态结构验证
cd tests/system/scripts
python -m pytest test_skill_basic.py -v -k "skill-name"      # Skill
python -m pytest test_team_basic.py -v -k "team-name"        # Team

# Phase 2 — AI 语义评测（支持 --eval-model 指定模型，对应 Max Tokens (<model>) 预算）
python -m pytest test_skill_evals.py --skill skill-name -v    # Skill
python -m pytest test_team_evals.py --team team-name -v       # Team

# 仅测试单个 eval 用例
python -m pytest test_skill_evals.py --skill skill-name --eval-id 3 -v --tb=long

# 方式4：run_eval.py（pytest 封装，支持报告生成）
python tests/system/scripts/run_eval.py --skill skill-name --html-report

# 并行执行：Phase 2 用例相互独立，可通过 --parallel 加速
python tests/system/scripts/main.py \
    --repo-root /path/to/repo \
    --changed-files ops/foo/SKILL.md \
    --parallel auto

# 重试机制：设置 EVAL_EXEC_RETRIES 环境变量（默认 1，即不重试）
EVAL_EXEC_RETRIES=3 python -m pytest test_skill_evals.py --skill skill-name -v

# 按模型指定 Token 预算：--eval-model 匹配 evals.md 中的 Max Tokens (<model>)
python tests/system/scripts/main.py \
    --repo-root /path/to/repo \
    --changed-files ops/foo/SKILL.md \
    --eval-model claude-sonnet-4-20250514

# 仅重新生成报告（不执行测试）
python tests/system/scripts/main.py \
    --report-only --repo-root . \
    --changed-files ops/cann-env-setup/SKILL.md
```

#### 输出结果

| 文件 | 说明 |
|------|------|
| `results/basic_validation.html` | Skill Phase 1 静态结构验证 HTML 报告 |
| `results/team_basic_validation.html` | Team Phase 1 静态结构验证 HTML 报告 |
| `results/evals_validation_<YYYYMMDD_HHMMSS>.html` | **Skill 统一** Phase 2 语义评测 HTML 报告（含时间戳，所有 skill 合并展示） |
| `results/team_evals_validation_<YYYYMMDD_HHMMSS>.html` | **Team 统一** Phase 2 语义评测 HTML 报告 |
| `results/<skill>_<timestamp>.json` | 结构化评测结果 JSON |
| `logs/<skill>_case_X.json` | 每个用例的执行 session ID |
| `logs/<skill>_case_X_review_ses.json` | 评测 session 完整对话导出 |
| `logs/test_results_<timestamp>.zip` | logs + results 打包归档 |

ST 框架的配置（`st-test.config`）、沙箱隔离机制、以及评测用例编写格式，详见上方"本地开发调试"章节。

## 运行参数

| 参数 | 说明 |
|------|------|
| `--fast`, `-f` | 仅运行单元测试 |
| `--integration`, `-i` | 包含集成测试 |
| `--all` | 运行所有测试 |
| `--category`, `-c CAT` | 运行指定类别（unit/behavior/integration/all） |
| `--platform PLATFORM` | 指定平台（claude/opencode/auto，默认: opencode） |
| `--test`, `-t NAME` | 运行指定测试 |
| `--timeout SECONDS` | 设置超时时间（默认: 600） |
| `--verbose`, `-v` | 显示详细输出 |
| `--output FORMAT` | 输出格式（text/json/**html**） |
| `--output-path PATH` | HTML 报告输出路径（默认: `tests/test-ut-report.html`） |
| `--auto-fix` | 自动修复支持的问题（CRLF、版本号 bump 等） |
| `--list`, `-l` | 列出所有可用测试 |
| `--help`, `-h` | 显示帮助信息 |

### 增量测试参数（CI/CD）

| 参数 | 说明 |
|------|------|
| `--incremental` | 启用增量测试模式，仅测试变更的 skill/agent/team |
| `--base-branch BRANCH` | 指定对比的基础分支（默认: master） |
| `--force-full` | 强制运行全量测试，即使启用了增量模式 |
| `--auto-fix` | 自动修复支持的问题（CRLF 换行符、版本号未 bump） |

### 评估结果检查参数

| 参数 | 说明 |
|------|------|
| `--eval-results` | 运行 Skill 评估结果检查（workspace benchmark 验证） |
| `--workspace PATH` | 指定评估工作区路径 |
| `--iteration N` | 指定迭代版本（默认: latest） |
| `--threshold RATE` | 覆盖通过率阈值（0.0-1.0） |
| `--detect-regression` | 启用迭代间回归检测 |

## 环境变量

| 变量 | 说明 |
|------|------|
| `NO_COLOR` | 设置后禁用彩色输出 |
| `FORCE_COLOR` | 设置为 `1` 强制启用彩色输出（适用于 CI/非 TTY 环境） |
| `PARALLEL_JOBS` | 并行测试数量（默认: 4） |
| `BASE_BRANCH` | 增量测试的基础分支（默认: master） |
| `CI_MERGE_REQUEST_TARGET_BRANCH_NAME` | CI 流水线目标分支名，用于版本检查对比基准 |
| `REPO_ROOT` | 仓库根目录路径（gate_check.sh 使用） |
| `CHANGED_FILES` | 手动指定变更文件列表（gate_check.sh 使用，空格分隔） |

## HTML 报告

运行测试时自动生成交互式 HTML 报告（`tests/test-ut-report.html`），支持：

| 特性 | 说明 |
|------|------|
| **测试统计面板** | 通过/失败/跳过/警告数量及耗时汇总 |
| **失败修复指南** | 存在失败时顶部显示 fix-guide（含失败类型速查表和可复制 AI 提示词） |
| **搜索过滤** | 按名称搜索、按状态过滤（全部/仅失败/仅通过/仅跳过） |
| **排序** | 默认排序、按耗时升序/降序 |
| **日志查看** | 带行号的语法高亮日志，支持折叠/展开 |
| **导航** | 「上一个失败」/「下一个失败」按钮及键盘快捷键 `j`/`k` |
| **Sticky Header** | Dashboard + Banner + Toolbar 整体固定，滚动时不丢失 |
| **响应式布局** | 内容居中显示，适配宽屏和小屏（`@media (max-width: 640px)`） |

```bash
# 默认自动生成 HTML 报告（无需额外参数）
./run-tests.sh --fast

# 指定自定义报告路径
./run-tests.sh --fast --output html --output-path my-report.html
```

## 自动修复

部分测试失败支持一键自动修复：

| 问题 | 修复方式 |
|------|---------|
| CRLF 换行符 | 自动转换为 LF |
| 版本号未 bump | 自动递增 plugin.json / marketplace.json 版本号 |

```bash
# 运行测试并尝试自动修复
./run-tests.sh --fast --auto-fix

# 自动修复后重跑确认
./run-tests.sh --fast
```

> **注意**：`--auto-fix` 只修改项目源文件（plugin.json、SKILL.md 等），不会修改 tests/ 目录下的任何文件。

## CI/CD 集成

### 增量测试

增量测试功能用于 PR 流水线，仅测试发生变更的组件：

```bash
# 在 CI 中运行增量测试
./run-tests.sh --incremental --platform none --output json
```

**工作原理：**

1. 通过 `git diff` 检测变更文件
2. 解析变更文件，识别受影响的 skill/agent/team
3. 只运行与变更相关的测试

**自动回退场景：**

当检测到以下变更时，自动运行全量测试：

| 变更类型 | 说明 |
|---------|------|
| 测试框架变更 | `tests/` 目录、`test-helpers.sh` 等 |
| 配置文件变更 | `package.json`、`.claude-plugin/` 等 |
| Git 不可用 | 非 Git 仓库或无法检测变更 |

### ST 框架门禁 (gate_check.sh)

`gate_check.sh` 是 ST 框架的 CI 门禁入口，自动检测变更文件并执行 Phase 1 + Phase 2 评测：

```bash
# 自动检测 HEAD 变更（对比 origin/master）
./tests/gate_check.sh

# 指定变更文件
CHANGED_FILES="ops/my-skill/SKILL.md" ./tests/gate_check.sh

# 指定目标分支
CI_MERGE_REQUEST_TARGET_BRANCH_NAME=main ./tests/gate_check.sh
```

### 变更检测示例

```
=== Incremental Test Analysis ===

Base branch: master

  [SKILL] ascendc-runtime-debug <- skills/ascendc-runtime-debug/SKILL.md
  [SKILL] ascendc-api-best-practices <- skills/ascendc-api-best-practices/references/api.md
  [AGENT] ops-debug-agent <- agents/ops-debug-agent/AGENT.md
  [TEAM] ops-direct-invoke <- teams/ops-direct-invoke/AGENTS.md

Changed components:
  Skills: ascendc-runtime-debug ascendc-api-best-practices
  Agents: ops-debug-agent
  Teams: ops-direct-invoke

Tests to run: 3
```

## 测试辅助库

### test-helpers.sh

```bash
# 平台检测
is_platform_available "claude"
get_platform_version "claude"
detect_platforms

# 执行函数
run_claude "prompt" [timeout] [allowed_tools]
run_opencode "prompt" [timeout]
run_ai "prompt" [timeout] [platform]

# 断言函数
assert_contains "output" "pattern" "test name"
assert_not_contains "output" "pattern" "test name"
assert_count "output" "pattern" expected "test name"
assert_order "output" "pattern_a" "pattern_b" "test name"
assert_file_exists "/path/to/file" "test name"

# 查询函数
get_all_skills
get_all_agents

# 增量测试函数
is_incremental_mode                           # 检查是否处于增量模式
should_test_skill "skill-name"                # 判断某个 skill 是否需要测试
should_test_agent "agent-name"                # 判断某个 agent 是否需要测试
should_test_team "team-name"                  # 判断某个 team 是否需要测试
get_skills_to_test                            # 获取需要测试的 skill 列表
get_agents_to_test                            # 获取需要测试的 agent 列表
get_teams_to_test                             # 获取需要测试的 team 列表

# 结构验证函数
validate_skill_structure "/path/to/SKILL.md"    # S-STR-01 ~ S-STR-18
validate_skill_content "/path/to/SKILL.md"      # S-CON-01 ~ S-CON-06
validate_agent_structure "/path/to/AGENT.md"    # A-STR-01 ~ A-STR-09
validate_agent_content "/path/to/AGENT.md"      # A-CON-01 ~ A-CON-05
validate_team_structure "/path/to/AGENTS.md"    # T-STR-01 ~ T-STR-08
validate_team_content "/path/to/AGENTS.md"      # T-CON-01 ~ T-CON-03
validate_global_uniqueness "skill|agent|team"   # 跨文件名称唯一性检查

# 版本管理函数
get_team_plugin_json "team-name"                # 获取 team 的 plugin.json 路径
extract_plugin_version "/path/to/plugin.json"   # 提取 version 字段
validate_semver "1.0.0"                         # 校验 SemVer 格式
compute_file_hash "/path/to/file"               # 计算 SHA256 前 16 位
recommend_version_bump "1.0.0" true false       # 根据变更推荐版本号
semver_compare "1.0.0" "1.0.1"                 # SemVer 比较（-1/0/1）

# Team 查询函数
get_all_teams                                    # 获取所有 team 列表
get_all_teams_with_paths                         # 获取所有 team:full_path
find_team_file "team-name"                       # 查找 team 的 AGENTS.md 路径

# Session 分析
find_recent_session [minutes_old]
verify_skill_invoked "$session_file" "skill-name"
verify_agent_dispatched "$session_file" "agent-name"
count_tool_invocations "$session_file" "ToolName"
check_premature_action "$session_file" "skill-name"
get_triggered_skills "$session_file"
analyze_workflow_sequence "$session_file"
analyze_tool_chain "$session_file"
analyze_cost_breakdown "$session_file"
extract_token_usage "$session_file"

# 测试项目管理
create_test_project [prefix]
cleanup_test_project "$test_dir"

# 结果跟踪
init_test_tracking
record_test "pass" "test_name" ["duration"]
print_test_summary
output_test_json
```

## 分析工具

```bash
# Session 分析
./tools/analyze-session.sh session.jsonl --brief
./tools/analyze-session.sh session.jsonl --full
./tools/analyze-session.sh session.jsonl --json
./tools/analyze-session.sh session.jsonl --tools
./tools/analyze-session.sh session.jsonl --cost
```

### analyze-session.sh 选项

| 选项 | 说明 |
|------|------|
| `--brief` | 显示简要摘要（默认） |
| `--full` | 显示完整分析报告 |
| `--json` | 以 JSON 格式输出 |
| `--tools` | 显示工具调用详情 |
| `--cost` | 显示成本分析（需要 jq） |

### 其他工具

```bash
# Python Token 分析（被 analyze-session.sh 和 test-simple-op-development.sh 内部调用）
python3 tools/analyze-token-usage.py session.jsonl
```

> **注意**: `tools/` 下的 `analyze-tokens.sh` 和 `analyze-workflow.sh` 是 `analyze-session.sh` 的功能子集，推荐直接使用 `analyze-session.sh`。

## 添加新测试

### 单元测试

在 `unit/skills/` 或 `unit/agents/` 创建测试文件：

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Your Test Name ==="

# 支持增量测试模式
if is_incremental_mode; then
    echo -e "${CYAN}[INCREMENTAL MODE]${NC} Testing only changed components"
fi

# 获取需要测试的组件列表（增量模式下已过滤）
COMPONENTS_TO_TEST=$(get_skills_to_test)

for skill in $COMPONENTS_TO_TEST; do
    # 在增量模式下检查是否应该测试此组件
    if is_incremental_mode && ! should_test_skill "$skill"; then
        print_skip "$skill: Not in changed list"
        continue
    fi

    # 执行测试...
    if [ condition ]; then
        echo "  [PASS] $skill: Test description"
    else
        echo "  [FAIL] $skill: Test description"
        exit 1
    fi
done
```

### 行为测试

在 `behavior/skills/` 或 `behavior/agents/` 创建测试文件：

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

TIMEOUT=30
output=$(run_claude "your prompt" $TIMEOUT)

if echo "$output" | grep -qiE "expected_pattern"; then
    echo "  [PASS] Test passed"
else
    echo "  [FAIL] Test failed"
fi
```

### ST 系统测试评测用例

在 `tests/system/cases/` 下创建 `<name>_evals.md` 文件（Skill 用 `skill_name`，Team 用 `team_name`）：

```markdown
---
skill_name: my-new-skill
eval_mode: text          # text（默认）或 file_based
---

# Case 1: 用例标题

## Config                            # 可选：用例级配置
- Max Tokens: 100000                  # Token 硬上限
- Max Tokens (deepseek-v4-flash): 140000  # 按模型指定 Token 上限
- Eval Mode: file_based               # 覆盖文件级评测模式
- Distractor skills: skill-a;skill-b  # 正向看护：干扰 skill 列表
- Disabled: true                      # 跳过该用例（默认不启用）
- Timeout: 900                        # 用例执行超时（秒，默认 600）
- 覆盖度阈值: 25                       # 按维度覆盖默认阈值

## Prompt
测试问题内容

## Expected Output
回复应覆盖的关键要点（语义描述，非逐字匹配）

## Expectations
- [contains] 必须包含的内容
- [not_contains] 不得包含的内容（只检查 AI 最终回复，不检查工具调用过程中的参考文档）
- [file_exists] todo.md              # 文件存在性检查
- [file_list] *.md                    # 匹配 glob 模式的文件列表
- [skill_activated] cann-env-setup    # 正向看护：程序化检查 skill 是否被加载
```

然后在 `tests/system/config/st-test.config` 中将该 skill/team 加入对应的白名单，确保 `skill_dirs` / `team_dirs` 包含其所在目录。

评测用例编写指南详见 `tests/system/docs/ST_DESIGN_AND_DEVELOPMENT_GUIDE.md`。

## 故障排查

### CLI 未找到

```bash
npm install -g @anthropic-ai/claude-code
```

### 测试超时

```bash
./run-tests.sh --timeout 900
```

### 无 CLI 环境

```bash
./run-tests.sh --fast
```

### 查看详细日志

```bash
./run-tests.sh --verbose --test unit/skills/test-structure.sh
```

### ST 框架依赖缺失

```bash
pip install -r tests/system/scripts/requirements.txt
```

### 评测 session 返回"无法解析判定结果"

该问题已在 2026-06 版本解决：评审机制从易出错的 JSON 解析改为 **review-template.md 模板化方案**。评审 Agent 通过 Write 工具填写沙箱中的 review-template.md 模板，框架通过正则从模板中提取结构化评审结果，不再依赖 JSON 输出。

如仍有问题，检查 `tests/system/logs/<skill>_case_X_review_ses.json` 中的原始评测输出，确认 review-template.md 是否被正确填写。

### 重试执行失败的评测用例

设置环境变量 `EVAL_EXEC_RETRIES` 控制评测用例的重试次数（默认 1，即不重试）。重试会重新执行 opencode session 并重新评审：

```bash
EVAL_EXEC_RETRIES=3 python -m pytest tests/system/scripts/test_skill_evals.py --skill skill-name -v
```
