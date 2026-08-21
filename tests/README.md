# CANN Skills 测试框架

本仓库自动化测试分为**两个独立入口**，职责不同、互不替代：

| 入口                   | 触发                            | 测试内容                                                          | 是否需要 AI CLI |
| ---------------------- | ------------------------------- | ----------------------------------------------------------------- | --------------- |
| **静态 UT 测试** | `./tests/run-tests.sh --fast` | L1 单元测试：skill/agent/team 结构、内容、依赖图、evals.json 门禁 | 否              |
| **ST 测试**      | `./tests/gate_check.sh`       | AI 语义评测：Phase 1 静态验证 + Phase 2 AI 语义评测               | 是（opencode）  |

---

## Part 1：静态 UT 测试（run-tests.sh）

### 1.1 概述

由 `tests/run-tests.sh` 驱动，**无需 AI CLI**，秒级完成，是 CI 的硬性门禁。验证 SKILL.md / AGENT.md / AGENTS.md / plugin.json / init.sh 等文件的结构、内容与跨组件依赖一致性。

### 1.2 覆盖范围

| 类别       | 测试文件                                      | 校验内容                                                                                       |
| ---------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Skill 结构 | `unit/skills/test-structure.sh`             | S-STR-01~21：frontmatter、name/description 格式、references/、内链、evals 基础校验、脚本可执行 |
| Skill 内容 | `unit/skills/test-content.sh`               | S-CON-01~09 + S-STR-13：触发关键词、触发条件、渐进式披露、三段式、反模式                       |
| 评测门禁   | `unit/skills/test-evals-required.sh`        | **S-EVAL-01**：每个 skill 必须提供合法 `evals/evals.json`                              |
| Agent      | `unit/agents/`                              | A-STR-01~09 / A-CON-01~09：结构、内容、skills 依赖                                            |
| Team       | `unit/teams/`                               | T-STR-01~08 / T-CON-01~03 / 版本看护（marketplace 依赖链）                                    |
| 依赖图     | `unit/test-dependency-graph.sh`             | DG-01~11：marketplace.json / plugin.json / AGENTS.md / init.sh 交叉引用                        |
| 换行符     | `unit/test-line-endings.sh`                 | 全局 CRLF 检测                                                                                 |
| 安装       | `unit/install/test-init-install.sh`         | init.sh 安装产物静态验证                                                                       |
| 基础设施   | `unit/infra/test-gitcode-issue-workflow.sh` | GitCode 工作流 pytest 单元测试                                                                 |

### 1.3 运行方式

```bash
# 运行全部 L1（CI 默认入口）
./tests/run-tests.sh --fast

# 运行单个测试
./tests/run-tests.sh --test unit/skills/test-structure.sh

# 增量模式（仅测试变更的 skill/agent/team）
./tests/run-tests.sh --incremental --base-branch master

# 自动修复可修复问题（CRLF、版本号未 bump）
./tests/run-tests.sh --fast --auto-fix
```

> 完整参数（`--parallel`、`--incremental-ci`、`--output json`、HTML 报告等）见 `tests/README.md`。

### 1.4 目录结构

```
tests/unit/
├── test-line-endings.sh            # CRLF 检查
├── test-dependency-graph.sh        # 依赖图 DG-01~11
├── infra/test-gitcode-issue-workflow.sh
├── skills/
│   ├── test-structure.sh           # S-STR-01~21
│   ├── test-content.sh             # S-CON-01~09 + S-STR-13
│   └── test-evals-required.sh      # S-EVAL-01
├── agents/                         # A-STR / A-CON
├── teams/                          # T-STR / T-CON / 版本看护
└── install/test-init-install.sh
```

---

## Part 2：ST 测试（gate_check.sh）

### 2.1 概述

ST（System Test）是基于 Python/pytest 的 **AI 语义评测**系统。`tests/gate_check.sh` 自动检测变更文件，识别受影响的 skill/team，执行两阶段评测并输出统一 HTML 报告，验证技能在实际对话中的回复质量与正确性。

### 2.2 核心工作流程

```
输入：repo_root + changed_files
    ├─ 识别受影响的 skills / teams（从变更文件路径提取）
    ├─ 加载评测用例（{skill}/evals/evals.json）
    ├─ Phase 1: 静态结构验证（秒级，无需 AI）
    │   ├─ Skill: test_skill_basic.py（evals.json + SKILL.md 结构校验）
    │   └─ Team: test_team_basic.py（evals.json + AGENTS.md/plugin.json/init.sh 校验）
    ├─ Phase 2: AI 语义评测（分钟级，需 opencode CLI）
    │   ├─ Skill: test_skill_evals.py（沙箱 symlink skill 目录）
    │   └─ Team: test_team_evals.py（沙箱部署完整 team 环境）
    │   ├─ 执行 Session：加载 target 发送 prompt → 收集 AI 回复
    │   └─ 评测 Session：评审回复质量（AI 评审 / cann_bench 确定性评测）
    │       └─ 断言验证：contains / not_contains / file_exists / skill_activated 等
    ├─ 保存结果（统一 HTML 报告 + JSON 归档）
    └─ 返回：通过（0）/ 失败（1）
```

> Phase 1 失败则该 target 不进入 Phase 2。

### 2.3 评测用例格式（evals.json）

存放在 `{skill}/evals/evals.json`（Team 为 `{team}/evals/evals.json`），由 `evals_json_parser.py` 解析：

```json
{
  "skill_name": "cann-env-setup",
  "eval_mode": "text",
  "evals": [
    {
      "id": 1,
      "title": "检查NPU驱动安装命令",
      "config": {
        "max_tokens": 200000,
        "ascend_platforms": ["A2"],
        "eval_mode": "text"
      },
      "prompt": "我有一台昇腾服务器，想检查NPU驱动是否已安装，应该用什么命令？",
      "expected_output": "回复应说明使用 npu-smi info 命令检查驱动",
      "expectations": [
        { "type": "contains", "pattern": "npu-smi info" }
      ],
      "files": []
    }
  ]
}
```

**顶层字段**：`skill_name` / `team_name`（二选一，须与目录名一致）、`eval_mode`（`text` 默认 / `file_based` / `cann_bench`）。

**`config` 关键字段**：

| 字段                    | 说明                                                               |
| ----------------------- | ------------------------------------------------------------------ |
| `max_tokens`          | Token 硬上限                                                       |
| `max_tokens_by_model` | 按模型 Token 上限，如`{"deepseek-v4-flash": 140000}`             |
| `ascend_platforms`    | 适用平台（A2/A3/A5）                                               |
| `distractor_skills`   | 正向看护：干扰 skill 列表                                          |
| `disabled`            | 跳过该用例                                                         |
| `timeout`             | 执行超时（秒，默认 600）                                           |
| `dim_thresholds`      | 维度阈值覆盖                                                       |
| `cann_bench_*`        | cann_bench 模式专用（operator/level/device/no_perf/warmup/repeat） |

**`expectations` 断言类型**：`contains`、`not_contains`、`file_exists`、`file_list`、`file_contains`、`skill_activated`。

### 2.4 沙箱隔离

每个用例在独立沙箱中执行，互不干扰：

| 类型       | 部署方式                                                                                  |
| ---------- | ----------------------------------------------------------------------------------------- |
| Skill      | 软链接 skill 目录到沙箱`.opencode/skills/`（默认）；`SKILL_SANDBOX_COPY=1` 切复制模式 |
| Team       | 沙箱中执行`init.sh project opencode <sandbox>`                                          |
| cann_bench | 额外部署 cann-bench 仓库副本、任务定义文件与工程模板                                      |

沙箱目录：`tests/system/sandboxes/<skill>_eval_<id>/`（含 `logs/` session 数据）。

### 2.5 运行方式

```bash
# CI 门禁（自动检测变更，对比 origin/master）
./tests/gate_check.sh

# 指定变更文件 / 平台 / 重复轮数 / 全量扫描
CHANGED_FILES="ops/ascendc-st-design/SKILL.md" ./tests/gate_check.sh
./tests/gate_check.sh --ascend-platform A2 --ascend-platform A3
./tests/gate_check.sh --repeat 3
./tests/gate_check.sh --all

# main.py 直接调用（支持并行、仅重生成报告）
python tests/system/scripts/main.py \
    --repo-root /path/to/repo \
    --changed-files ops/ascendc-st-design/SKILL.md \
    --parallel auto

# 直接运行 pytest（本地调试）
cd tests/system/scripts
python -m pytest test_skill_basic.py -v -k "skill-name"        # Phase 1
python -m pytest test_skill_evals.py --skill skill-name -v     # Phase 2
python -m pytest test_skill_evals.py --skill skill-name --eval-id 3 -v

# 重试机制（默认 1 = 不重试）
EVAL_EXEC_RETRIES=3 python -m pytest test_skill_evals.py --skill skill-name -v
```

### 2.6 配置（tests/system/config/st-test.config）

```yaml
skill_dirs: ["ops", "graph", "model", "infra", "runtime"]   # 扫描路径
skill_whitelist: ["cann-env-setup"]                          # 白名单（为空表示全部）
team_dirs: ["plugins-official", "plugins-community"]       # Team 扫描路径
team_whitelist: [...]                                        # Team 白名单
```

### 2.7 注意事项

1. 仅白名单内、且目录下有 `evals.json` 的 target 才执行 Phase 2。
2. Phase 1 失败则跳过 Phase 2。
3. 退出码：全部通过返回 0，任一失败返回 1。
4. 评审机制：评审 Agent 通过 Write 工具填写 `review-template.md` 模板，框架正则解析评分（不再依赖 JSON 输出）。
5. cann_bench 模式需 NPU 环境，通过条件：编译通过 + 精度达标 + 综合得分 > 50；依赖 cann-bench 仓库（`CANN_BENCH_PATH` 可覆盖默认路径）。
6. 依赖安装：`pip install -r tests/system/scripts/requirements.txt`。

### 目录结构（精简）

```
tests/system/
├── config/          # st-test.config + 评审模板（review-template*.md / skip-report-template.html）
├── docs/            # ST_DESIGN_AND_DEVELOPMENT_GUIDE.md / USER_GUIDE.md / 覆盖度文档
├── scripts/         # main.py / pytest 用例 / evals_json_parser.py / sandbox_manager.py / opencode_runner.py
├── results/         # HTML 报告输出
├── logs/            # 运行日志与归档
└── sandboxes/       # 沙箱隔离目录（<skill>_eval_<id>/）
```

> 脚本清单与目录详解见 `tests/README.md`；评测用例编写规范见 `tests/system/docs/ST_DESIGN_AND_DEVELOPMENT_GUIDE.md`。
