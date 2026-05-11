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
./run-tests.sh --incremental-ci

# 查看可用测试
./run-tests.sh --list

# 运行指定测试
./run-tests.sh --test unit/skills/test-structure.sh
./run-tests.sh --test behavior/skills/test-universal.sh

# 测试单个 Skill（直接调用）
./behavior/skills/test-universal.sh ascendc-runtime-debug
```

## 目录结构

```
tests/
├── unit/                      # L1 单元测试（无需 CLI，< 30s）
│   ├── skills/
│   │   ├── test-structure.sh  # Skill 结构验证（S-STR-01~16）
│   │   └── test-content.sh    # Skill 内容验证（S-CON-01~09 + S-STR-13）
│   ├── agents/
│   │   ├── test-structure.sh  # Agent 结构验证（A-STR-01~09）
│   │   └── test-content.sh    # Agent 内容验证（A-CON-01~09）
│   ├── teams/
│   │   ├── test-structure.sh  # Team 结构验证（T-STR-01~08）
│   │   ├── test-content.sh    # Team 内容验证（T-CON-01~03）
│   │   └── test-version.sh    # Team 版本看护（git diff 文件变更 + marketplace.json 依赖链）
│   └── install/
│       └── test-init-install.sh  # init.sh 安装产物静态验证
│
├── behavior/                  # L2 行为测试（需要 CLI，1-5 min）
│   ├── skills/
│   │   ├── test-universal.sh        # 通用测试（自动运行全部 9 条规则）
│   │   ├── test-trigger-correctness.sh
│   │   ├── test-premature-action.sh
│   │   ├── test-interaction-logic.sh
│   │   └── test-cases/              # 定制测试配置（可选）
│   │       └── ascendc-runtime-debug.yaml
│   ├── agents/
│   │   ├── test-premature-action.sh
│   │   └── test-trigger-correctness.sh
│   └── install/
│       └── test-init-behavior.sh    # init.sh 实际安装行为验证
│
├── integration/               # L3 集成测试（5-15 min）
│   ├── test-simple-op-development.sh
│   └── test-workflow-execution.sh
│
├── lib/
│   ├── test-helpers.sh        # 测试辅助函数
│   ├── skill_validator.py     # YAML-aware 结构与内容校验器
│   └── rules.yaml             # 校验规则配置（关键词、阈值等）
│
├── tools/
│   ├── analyze-session.sh
│   ├── analyze-tokens.sh          # analyze-session.sh 功能子集
│   ├── analyze-token-usage.py
│   └── analyze-workflow.sh        # analyze-session.sh 功能子集
│
└── run-tests.sh
```

## 测试分层

| 层级 | 目录 | 说明 | 运行方式 |
|------|------|------|----------|
| L1 | unit/ | 单元测试，验证结构和内容 | `--fast` |
| L2 | behavior/ | 行为测试，验证触发和响应 | `--category behavior` |
| L3 | integration/ | 集成测试，验证完整工作流 | `--integration` |

## 测试内容

### L1 单元测试

#### Skills 测试规则详情

| 规则ID | 测试项 | 级别 | 文件 |
|--------|-------|------|------|
| S-STR-01 | YAML frontmatter 格式正确 | error | test-structure.sh |
| S-STR-02 | name 字段存在 / 长度 / 格式 | error | test-structure.sh |
| S-STR-03 | description 字段存在 / 长度 | error | test-structure.sh |
| S-STR-04 | references/ 目录非空（如存在） | error | test-structure.sh |
| S-STR-08 | 文档内链有效性 | error | test-structure.sh |
| S-STR-09 | 文件名必须为 SKILL.md | error | test-structure.sh |
| S-STR-10 | 目录名 kebab-case | error | test-structure.sh |
| S-STR-11 | 不允许 README.md 在 skill 目录 | error | test-structure.sh |
| S-STR-12 | frontmatter 不含 XML 标签 | error | test-structure.sh |
| S-STR-14 | name 不使用保留前缀（claude/anthropic） | error | test-structure.sh |
| S-STR-16 | metadata 必须是 string→string 映射 | error | test-structure.sh |
| S-CON-01 | name 与目录名一致 | error | test-content.sh |
| S-CON-02 | description 包含触发关键词（手动 skill 跳过） | error | test-content.sh |
| S-CON-03 | description 包含触发条件 | error | test-content.sh |
| S-CON-04 | 含可执行指令（代码块、步骤） | warn | test-content.sh |
| S-CON-05 | 含错误处理 / 故障排查章节 | warn | test-content.sh |
| S-CON-06 | 含示例 / 场景章节 | warn | test-content.sh |
| S-CON-07 | 长文件链接到 references/（渐进式披露） | warn | test-content.sh |
| S-CON-08 | description 遵循三段式结构 | warn | test-content.sh |
| S-CON-09 | description 不含反模式词汇 | warn | test-content.sh |
| S-STR-13 | 正文字数不超过上限 | warn | test-content.sh |

> **注意**: error 级别规则会阻断测试（FAIL），warn 级别规则仅输出警告但不阻断（PASS with warnings）。CI 中建议关注 warn 输出以持续提升质量。

#### Agents (unit/agents/)

| 测试文件 | 验证项 |
|---------|--------|
| `test-structure.sh` | YAML格式、name/description/mode字段、skills依赖存在性、name/description格式、链接有效性 |
| `test-content.sh` | name一致性、description关键词与触发条件、内容质量（可执行指令、错误处理、示例、渐进式披露等） |

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
| A-CON-03 | description 包含触发条件 | warn | test-content.sh |
| A-CON-04 | 含可执行指令（代码块、步骤） | warn | test-content.sh |
| A-CON-05 | 含错误处理 / 故障排查章节 | warn | test-content.sh |
| A-CON-06 | 含示例 / 场景章节 | warn | test-content.sh |
| A-CON-07 | 长文件链接到 references/（渐进式披露） | warn | test-content.sh |
| A-CON-08 | description 遵循三段式结构 | warn | test-content.sh |
| A-CON-09 | description 不含反模式词汇 | warn | test-content.sh |

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

### 通用测试使用说明

`test-universal.sh` 可以自动测试所有 Skill，无需手动配置测试用例：

```bash
# 测试所有 Skill（并行执行，默认4个并发）
./behavior/skills/test-universal.sh

# 指定并行数量
PARALLEL_JOBS=8 ./behavior/skills/test-universal.sh

# 测试单个 Skill（串行执行）
./behavior/skills/test-universal.sh ascendc-runtime-debug
```

**并行执行说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `PARALLEL_JOBS` | 并行测试数量 | 4 |

测试结果会写入临时文件，最后统一汇总输出。

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

## 运行参数

| 参数 | 说明 |
|------|------|
| `--fast`, `-f` | 仅运行单元测试 |
| `--integration`, `-i` | 包含集成测试 |
| `--all` | 运行所有测试 |
| `--category`, `-c CAT` | 运行指定类别（unit/behavior/integration/all） |
| `--platform PLATFORM` | 指定平台（claude/opencode/auto，默认: opencode） |
| `--test`, `-t NAME` | 运行指定测试 |
| `--timeout SECONDS` | 设置超时时间（默认: 300） |
| `--verbose`, `-v` | 显示详细输出 |
| `--output FORMAT` | 输出格式（text/json） |
| `--list`, `-l` | 列出所有可用测试 |
| `--help`, `-h` | 显示帮助信息 |

### 增量测试参数（CI/CD）

| 参数 | 说明 |
|------|------|
| `--incremental-ci` | 启用增量测试模式，仅测试变更的 skill/agent/team |
| `--base-branch BRANCH` | 指定对比的基础分支（默认: master） |
| `--force-full` | 强制运行全量测试，即使启用了增量模式 |

## 环境变量

| 变量 | 说明 |
|------|------|
| `NO_COLOR` | 设置后禁用彩色输出 |
| `FORCE_COLOR` | 设置为 `1` 强制启用彩色输出（适用于 CI/非 TTY 环境） |
| `PARALLEL_JOBS` | 并行测试数量（默认: 4） |
| `BASE_BRANCH` | 增量测试的基础分支（默认: master） |

## CI/CD 集成

### 增量测试

增量测试功能用于 PR 流水线，仅测试发生变更的组件：

```bash
# 在 CI 中运行增量测试
./run-tests.sh --incremental-ci --platform none --output json
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

### GitHub Actions 示例

```yaml
name: Test Skills

on:
  pull_request:
    branches: [master]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # 需要完整历史用于增量检测

      - name: Run Incremental Tests
        run: |
          cd cannbot-skills/tests
          ./run-tests.sh --incremental-ci --platform none --output json > results.json

      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: skills/tests/results.json
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
validate_skill_structure "/path/to/SKILL.md"    # S-STR-01 ~ S-STR-16
validate_skill_content "/path/to/SKILL.md"      # S-CON-01 ~ S-CON-09 + S-STR-13
validate_agent_structure "/path/to/AGENT.md"    # A-STR-01 ~ A-STR-09
validate_agent_content "/path/to/AGENT.md"      # A-CON-01 ~ A-CON-09
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

## 故障排查

### CLI 未找到

```bash
npm install -g @anthropic/claude-code
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
