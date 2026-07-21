# Skill 测试框架

基于变更文件识别受影响的 skills，执行对应的评测用例，输出 HTML 测试报告。用于 CI/CD 门禁检查，确保 skills 代码变更质量。

## 核心工作流程

```
输入：repo_root + changed_files
    │
    ├─ 步骤1：识别受影响的 skills / teams
    │   └─ 从变更文件路径中提取 skill 或 team 名称
    │
    ├─ 步骤2：加载评测用例
    │   └─ 读取 {skill_dir}/{skill_name}/evals/evals.json
    │       （Skill 用 skill_name，Team 用 team_name frontmatter）
    │
    ├─ 步骤3：执行评测 — 逐个 target 进行（Phase 1 失败则跳过 Phase 2）
    │   ├─ Phase 1: 静态结构验证（秒级，无需 AI 调用）
    │   │   ├─ Skill: test_skill_basic.py
    │   │   │   ├─ evals.json 文件存在性、格式合法性、必填字段检查
    │   │   │   └─ SKILL.md 存在性、YAML frontmatter 格式校验
    │   │   ├─ Team: test_team_basic.py
    │   │   │   ├─ evals.json 文件存在性、格式合法性、必填字段检查
    │   │   │   ├─ AGENTS.md 存在性、frontmatter 校验
    │   │   │   ├─ plugin.json 存在性与合法性
    │   │   │   └─ init.sh 存在性校验
    │   │   注意：若 Phase 1 失败，该 target 不进入 Phase 2
    │   │
    │   └─ Phase 2: AI 语义评测（分钟级，需要 opencode CLI）
    │       ├─ Skill: test_skill_evals.py
    │       │   └─ 沙箱部署：symlink skill 目录到 .opencode/skills/
    │       ├─ Team: test_team_evals.py
    │       │   └─ 沙箱部署：执行 init.sh project opencode <sandbox>
    │       ├─ 重试机制：失败可通过 EVAL_EXEC_RETRIES 控制重试次数
    │       └─ (skill/team 后续流程共享) ...
    │           ├─ 执行 Session：opencode 加载 target，发送 prompt → 收集 AI 回复
    │           ├─ 评测 Session：独立 opencode session 评审回复质量
    │           │   ├─ AI 评审模式（text/file_based/code_gen）：
    │           │   │   └─ 评审方式：Agent 通过 Write 工具填写 review-template.md 模板
    │           │   │   └─ 框架解析：从模板中提取结构化评分（Status + Score + 各维度）
    │           │   │   └─ 评分标准：信息覆盖度(40) + 技术准确性(30) + 回复质量(20) + Token(10)
    │           │   └─ cann_bench 模式：
    │           │       └─ 确定性评测：执行 cann-bench 评测管道（编译/精度/性能）
    │           │       └─ 通过判定：综合得分 > 50 且编译+精度均通过
    │           └─ 模式匹配/Token检查：contains/not_contains/file_exists/skill_activated
    │
    ├─ 步骤4：保存结果（HTML 报告 + JSON 日志归档）
    │
    └─ 返回：通过/失败状态
```

## 输入参数

| 参数 | 必填 | 说明 | 默认值 |
|------|------|------|--------|
| `repo_root` | **必填** | 仓库根目录的绝对路径 | - |
| `changed_files` | **必填** | 变更文件列表（空格分隔，支持相对或绝对路径） | - |
| `--parallel` / `-p` | 否 | 并行 worker 数，`auto`=CPU 核数 - 1，或指定数字如 `4` | `1` (顺序执行) |
| `--report-only` | 否 | 仅从已有沙箱 JSON 文件重新生成 HTML 报告，不执行测试 | `false` |
| `--eval-id` | 否 | 仅运行指定 ID 的评测用例（传入 pytest） | 全部 |
| `--eval-model` | 否 | 指定评测模型名称（如 `claude-sonnet-4-20250514`），用于按模型匹配 `Max Tokens (<model>)` 预算 | 自动检测 |

## 输出结果

| 文件 | 说明 |
|------|------|
| `results/ST_validation_report_<YYYYMMDD_HHMMSS>.html` | **Skill + Team 统一** Phase 2 语义评测 HTML 报告（所有 target 合并展示，表中"类型"列区分 Skill/Team） |
| `logs/test_results_<timestamp>.zip` | logs/ 与 results/ 目录的归档压缩包，供流水线下载 |

### HTML 报告结构

报告包含每个评测用例的一行记录，展示：

- **Result** — 通过/失败徽章
- **Skill** — skill 名称
- **描述** — 用例中文描述（如"AI 语义评测"）
- **评测得分** — 百分制分数，颜色标识：>=80 绿色，>=60 黄色，<60 红色，未解析显示 `—`
- **TestId** — 参数化测试节点 ID，格式 `<skill>::eval_<N>`
- **Duration** — 执行耗时

点击行可展开查看交互详情，包含：

- 输入 Prompt
- 预期要点
- AI 思考过程的工具调用
- AI 最终回复
- 评审结果（含各维度得分说明）

## 测试阶段

### Phase 1：静态结构验证（test_skill_basic.py）

无需 AI 调用，快速验证 skill 的结构完整性：

- `evals.json` 文件存在性、格式合法性、必填字段检查
- 每个 eval case 的 id、prompt、expected_output 格式校验
- SKILL.md 存在性、YAML frontmatter 格式校验

### Phase 2：AI 语义评测（test_skill_evals.py）

使用 opencode CLI 执行评测用例，验证 skill 的实际表现：

- **执行 Session**：向 skill 发送 prompt，收集 AI 回复
- **评测 Session**：独立评测模型评审回复质量
- **断言验证**：检查 expectations 中的 `contains` / `not_contains` 模式

### 评分标准

评审模型按四个维度评分（总分 100，总分 ≥ 60 **且各维度均不低于最低阈值** 方为通过）：

| 维度 | 分值 | 最低阈值 | 说明 |
|------|:----:|:--------:|------|
| 信息覆盖度 | 0-40 | **20** | 是否完整覆盖预期回复中的关键要点 |
| 技术准确性 | 0-30 | **15** | 技术信息是否正确，无错误或误导 |
| 回复质量 | 0-20 | **10** | 结构清晰、逻辑连贯、简洁直接 |
| Token 消耗 | 0-10 | **3** | 回复长度合理，思考过程工具调用高效 |

> 各维度阈值可在每个用例的 `Config` 中单独覆盖（见下方配置说明），如 `覆盖度阈值: 25`。

#### cann_bench 模式评分标准

cann_bench 模式使用确定性评测（非 AI 评审），通过 cann-bench 评测管道进行编译检查、精度验证和性能评分：

| 维度 | 满分 | 说明 |
|------|:----:|------|
| 编译通过 | 1 | 二元判定：算子工程能否成功编译 |
| 精度达标 | 1 | 二元判定：算子输出精度是否满足要求 |
| 综合得分 | 100 | cann-bench 管道输出的综合性能评分 |

**通过条件**：编译通过 = 1 且 精度达标 = 1 且 综合得分 > 50

## 评测用例格式（evals.json）

评测用例文件统一使用 JSON 格式，存放在每个 Skill/Team 目录下的 `evals/evals.json`：

```
ops/{skill-name}/evals/evals.json               # Skill 评测用例
plugins-official/{team-name}/evals/evals.json    # Team 评测用例
```

文件由 `scripts/convert_evals.py` 从原有的 `evals.md` 转换生成，亦可直接编写。由 `evals_json_parser.py` 解析为 ST 框架内部 dict 结构（`evals_parser.py` 保留为遗留 MD 解析器参考）。

### 顶层结构

```json
{
  "skill_name": "cann-env-setup",
  "eval_mode": "text",
  "evals": [...]
}
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `skill_name` | Skill 必填 | 目标 skill 名称，与 SKILL.md 中 `name` 一致。与 `team_name` 二选一 |
| `team_name` | Team 必填 | 目标 team 名称，与 plugin.json 中 `name` 一致。与 `skill_name` 二选一 |
| `eval_mode` | 否 | 评测模式，默认 `text`。可选：`text`（语义评审）、`file_based`（文件验证）、`code_gen`（算子项目编译运行）、`cann_bench`（cann-bench 确定性评测） |

### 用例结构（evals[]）

```json
{
  "id": 1,
  "title": "检查NPU驱动安装命令",
  "config": {
    "max_tokens": 200000,
    "ascend_platforms": ["A2"],
    "eval_mode": "text"
  },
  "prompt": "我有一台昇腾服务器，想检查NPU驱动是否已安装，应该用什么命令？",
  "expected_output": "回复应说明使用 npu-smi info 命令检查驱动，并解释如何根据输出判断驱动是否已安装",
  "files": [],
  "expectations": [
    {
      "type": "contains",
      "pattern": "npu-smi info",
      "description": "回复中提到了 npu-smi info"
    }
  ]
}
```

#### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | int | **是** | 用例编号，从 1 开始连续递增 |
| `title` | string | **是** | 用例标题 |
| `config` | object | 是 | 用例级配置，见下方 |
| `prompt` | string | **是** | 发送给 AI 的测试问题 |
| `expected_output` | string | **是** | 对 AI 回复的语义预期，描述应覆盖的关键要点 |
| `files` | string[] | 否 | 输入文件路径列表（skill-creator 使用，ST 框架忽略） |
| `expectations` | array | 否 | 断言列表，见下方 |

#### config 字段

| 键 | 类型 | 说明 | 默认值 |
|-----|------|------|--------|
| `max_tokens` | int | Token 消耗硬上限 | 无限制 |
| `max_tokens_by_model` | object | 按模型的 Token 上限，如 `{"deepseek-v4-flash": 240000}` | `{}` |
| `ascend_platforms` | string[] | 适用平台列表，如 `["A2"]` | `[]` |
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

#### expectations 条目

每条包含三个字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 断言类型：`contains` / `not_contains` / `file_exists` / `file_list` / `file_contains` / `skill_activated` |
| `pattern` | string | 匹配模式 |
| `description` | string | 可读描述（供 skill-creator 使用，ST 框架仅用 type+pattern） |

各 type 含义：

| type | pattern 说明 |
|------|-------------|
| `contains` | AI **最终回复**中必须包含该字符串 |
| `not_contains` | AI **最终回复**中不得包含该字符串 |
| `file_exists` | 指定文件路径必须存在于沙箱中 |
| `file_list` | 沙箱中有匹配该 glob 模式的文件 |
| `file_contains` | 沙箱中匹配 glob 的文件包含指定文本 |
| `skill_activated` | 程序化检查 AI 是否加载了指定 skill |

### 完整示例（Team）

```json
{
  "team_name": "ops-direct-invoke",
  "eval_mode": "text",
  "evals": [
    {
      "id": 1,
      "title": "基本算子开发流程问答",
      "config": {
        "max_tokens": 200000,
        "timeout": 900,
        "ascend_platforms": ["A2"],
        "eval_mode": "text"
      },
      "prompt": "我想开发一个 Ascend C Kernel 直调算子，计算两个向量的逐元素加法。请描述完整流程。",
      "expected_output": "回复应覆盖：环境检查、tiling 策略、host/device 代码结构、代码审查、性能验收",
      "files": [],
      "expectations": [
        {"type": "contains", "pattern": "kernel", "description": "回复中提到了 kernel"},
        {"type": "contains", "pattern": "tiling", "description": "回复中提到了 tiling"}
      ]
    }
  ]
}
```

## 沙箱隔离机制

每个评测用例在独立的沙箱目录中执行，确保互不干扰。Skill 和 Team 的沙箱部署方式不同：

| 类型 | 沙箱部署方式 |
|------|-------------|
| **Skill 测试** | 将 skill 目录通过软链接部署到沙箱的 `.opencode/skills/` 下 |
| **Team 测试** | 在沙箱中执行 `init.sh project opencode <sandbox>` 安装完整 team 环境 |
| **cann_bench 模式** | 额外部署 cann-bench 仓库（复制模式）、任务定义文件和工程模板 |

```
tests/system/sandboxes/
├── <skill>_eval_1/
│   ├── .opencode/
│   │   ├── opencode.json                    # opencode 工具权限白名单配置
│   │   └── skills/
│   │       └── <skill>/                     # skill 目录（默认软链接，指向源目录）
│   ├── logs/                                # session 导出 JSON
│   │   ├── <skill>_case_1_ses.json          # 执行 session 数据
│   │   └── <skill>_case_1_review_ses.json   # 评审 session 数据
│   ├── cann-bench/                          # cann_bench 模式：cann-bench 仓库副本（只读）
│   ├── cann-bench-task/                     # cann_bench 模式：任务定义文件
│   │   ├── proto.yaml                       #   算子原型信息
│   │   ├── cases.yaml                       #   测试用例定义
│   │   ├── desc.md                          #   算子描述
│   │   └── golden.py                        #   标杆参考实现
│   ├── direct_launch_example/               # cann_bench 模式：算子工程模板
│   │   ├── csrc/                            #   扩展代码模板
│   │   ├── cann_bench/                      #   Python 接口模板
│   │   └── build.sh                         #   编译脚本
│   ├── output/                              # cann_bench 模式：AI 生成的算子代码
│   └── _cann_bench_meta.json                # cann_bench 模式：评测元数据（自动生成）
├── <skill>_eval_2/
│   └── ...
└── ...
```

Skill 目录默认使用**软链接**指向源目录，避免每个用例重复复制 skill 文件，节省磁盘空间和创建时间。
若需切回复制模式（Agent 可写源文件），设置环境变量 `SKILL_SANDBOX_COPY=1`：

```bash
SKILL_SANDBOX_COPY=1 python tests/system/scripts/main.py \
    --repo-root . \
    --changed-files ops/foo/SKILL.md
```

`logs/` 目录下的 JSON 文件可在测试后用于分析或重新生成报告。

### cann_bench 模式沙箱部署

当 `eval_mode` 为 `cann_bench` 时，框架会自动完成以下部署：

1. **cann-bench 仓库**：复制到 `<sandbox>/cann-bench/`（使用复制模式，避免污染源仓库）
2. **任务定义文件**：从 `cann-bench/tasks/<level>/<operator>/` 复制到 `<sandbox>/cann-bench-task/`
3. **工程模板**：从 `cann-bench/examples/direct_launch_example/` 复制到 `<sandbox>/direct_launch_example/`

AI Agent 在沙箱中根据任务定义生成算子代码到 `output/` 目录，框架自动执行 cann-bench 评测管道进行验证。

## 使用方式

### 完整执行评测

```bash
# Skill 测试
python tests/system/scripts/main.py \
    --repo-root /path/to/repo \
    --changed-files ops/ascendc-st-design/SKILL.md

# Team 测试（变更 team 目录时自动识别）
python tests/system/scripts/main.py \
    --repo-root /path/to/repo \
    --changed-files plugins-official/ops-direct-invoke/AGENTS.md

# 指定模型（匹配 Max Tokens (<model>) 预算）
python tests/system/scripts/main.py \
    --repo-root /path/to/repo \
    --changed-files ops/cann-env-setup/SKILL.md \
    --eval-model claude-sonnet-4-20250514
```

### 重试失败的评测用例

设置 `EVAL_EXEC_RETRIES` 环境变量控制重试（默认 1 = 不重试）：

```bash
EVAL_EXEC_RETRIES=3 python -m pytest test_skill_evals.py --skill cann-env-setup -v
```

### 仅重新生成报告（不执行测试）

当已有沙箱 JSON 文件时，跳过测试执行，仅重新生成 HTML 报告（用于验证报告展示效果或修复报告生成逻辑后）：

```bash
python tests/system/scripts/main.py \
    --report-only \
    --repo-root . \
    --changed-files ops/cann-env-setup/SKILL.md
```

### 通过 gate_check.sh 调用（CI 门禁）

```bash
# 方式1：手动指定变更文件
export CHANGED_FILES="ops/ascendc-st-design/SKILL.md"
export REPO_ROOT="/path/to/repo"
./tests/gate_check.sh

# 方式2：Git 自动检测（对比 origin/master HEAD 变更）
./tests/gate_check.sh

# 方式3：指定目标分支对比
export CI_MERGE_REQUEST_TARGET_BRANCH_NAME="main"
./tests/gate_check.sh
```

### 并行执行

Phase 2 的 eval 用例相互独立，可通过 `--parallel` 并行执行：

```bash
python tests/system/scripts/main.py \
    --repo-root /path/to/repo \
    --changed-files ops/foo/SKILL.md \
    --parallel auto
```

## 配置

编辑 `tests/system/config/st-test.config` 调整扫描路径：

```yaml
skill_dirs:
  - "ops"
  - "graph"
  - "model"

# Skill 白名单：仅这些 skill 触发评测（为空表示全部生效）
skill_whitelist:
  - "cann-env-setup"
```

## 依赖安装

```bash
pip install -r tests/system/scripts/requirements.txt
```

## 目录结构

```
tests/system/
├── README.md                    # 本文档
├── config/
│   ├── st-test.config            # skill/team 扫描路径与白名单配置
│   ├── review-template.md        # text/file_based 模式评审 Agent 模板
│   ├── review-template-code-gen.md # code_gen 模式评审 Agent 模板
│   └── skip-report-template.html # 无变更时的跳过报告模板
├── docs/
│   ├── ST_DESIGN_AND_DEVELOPMENT_GUIDE.md  # ST 设计规范与开发指南
│   ├── USER_GUIDE.md            # 详细使用指南
│   └── ST_COVERAGE_REPORT.md    # 覆盖率快照
├── results/                     # HTML 报告输出目录
├── logs/                        # 运行日志与归档压缩包
├── sandboxes/                   # 沙箱隔离目录（用例级别）
│   └── <skill>_eval_<id>/
│       ├── cann-bench/          # cann_bench 模式：cann-bench 仓库副本（只读）
│       ├── cann-bench-task/     # cann_bench 模式：任务定义文件（proto.yaml, cases.yaml 等）
│       ├── direct_launch_example/ # cann_bench 模式：算子工程模板
│       └── output/              # cann_bench 模式：AI 生成的算子代码输出目录
└── scripts/
    ├── main.py                  # CI 门禁主入口（支持 skill + team）
    ├── conftest.py              # pytest 共享配置、钩子、工具函数（含 skill/team 发现逻辑）
    ├── evals_json_parser.py     # evals.json JSON 解析器（当前使用）
    ├── opencode_runner.py       # opencode CLI 封装（流式输出、Session 导出）
    ├── sandbox_manager.py       # 沙箱创建与管理（含 cann-bench 部署）
    ├── subprocess_streamer.py   # 子进程流式输出封装（心跳防超时）
    ├── test_skill_basic.py      # Phase 1: Skill 静态结构验证
    ├── test_skill_evals.py      # Phase 2: Skill AI 语义评测（含重试机制、cann_bench 模式）
    ├── test_team_basic.py       # Phase 1: Team 静态结构验证
    ├── test_team_evals.py       # Phase 2: Team AI 语义评测
    ├── run_eval.py              # 命令行评测启动脚本
    ├── session_stats.py         # Session 数据统计工具
    ├── test_opencode_runner.py  # opencode_runner 单元测试
    ├── opencode_runner_examples.py  # opencode_runner 使用示例
    ├── pytest.ini               # pytest 渲染配置
    └── requirements.txt         # Python 依赖
```

## 注意事项

1. **变更识别**：只有配置的 `skill_dirs` / `team_dirs` 目录下的变更才会触发评测。支持白名单过滤（`skill_whitelist` / `team_whitelist`）。
2. **评测用例必需**：target 必须在 `{skill_dir}/{skill_name}/evals/` 下有 `evals.json` 文件才会执行 Phase 2 评测。
3. **Phase 1 拦截**：Phase 1 静态验证失败时，该 target 不进入 Phase 2（不生成 HTML 报告），需先修复基础结构问题。
4. **`--report-only` 前提**：沙箱目录中须有可用的 JSON 文件（来自之前的完整测试运行）。若数据不完整（如 `ses.json` 截断），会话详情区块可能缺失，但评分不受影响。
5. **超时设置**：批量评测超时 1200 秒，单个评测用例超时默认 600 秒（可通过 `Timeout` 配置覆盖）。
6. **退出码**：所有评测通过返回 0，任一失败返回 1。
7. **评审机制**：评审 Agent 通过 Write 工具填写 `review-template.md` 模板，框架正则解析提取状态/评分/各维度得分。不再依赖容易出错的 JSON 输出格式。
8. **重试机制**：通过 `EVAL_EXEC_RETRIES=N` 环境变量控制评测用例的重试次数（默认 1，即不重试），适用于偶发性执行失败。
9. **模型 Token 预算**：支持按模型名称匹配 `Max Tokens (<model>)` 配置，通过 `--eval-model` 或 `EVAL_MODEL` 环境变量指定。未指定时自动从 session 导出数据检测。
10. **正向看护**：`Distractor skills` 配置 + `[skill_activated]` 断言，程序化验证 AI 在多个相似 skill 存在时正确选择目标 skill。
11. **cann_bench 模式**：
    - 需要 cann-bench 仓库存在，默认路径为 `<repo_root>/../cann-bench`，可通过 `CANN_BENCH_PATH` 环境变量覆盖
    - 需要 NPU 硬件环境支持（Ascend 910B/950 等）
    - 通过条件：编译通过 + 精度达标 + 综合得分 > 50
    - cann-bench 生成的 HTML 报告会通过 iframe 嵌入 pytest-html 报告的 Details 展开区
