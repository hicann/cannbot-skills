# Skill 测试框架

基于变更文件识别受影响的 skills，执行对应的评测用例，输出 HTML 测试报告。用于 CI/CD 门禁检查，确保 skills 代码变更质量。

## 核心工作流程

```
输入：repo_root + changed_files
    │
    ├─ 步骤1：识别受影响的 skills
    │   └─ 从变更文件路径中提取 skill 目录名
    │
    ├─ 步骤2：加载评测用例
    │   └─ 读取 tests/system/cases/<skill>_evals.md
    │
    ├─ 步骤3：执行评测
    │   ├─ Phase 1: 静态结构验证（test_skill_basic.py）
    │   └─ Phase 2: AI 语义评测（test_skill_evals.py）
    │       ├─ 创建独立沙箱
    │       ├─ 通过 opencode CLI 执行 skill，收集回复
    │       ├─ 使用独立 AI 模型评审回复质量
    │       └─ 生成 HTML 报告（含评分和交互详情）
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

## 输出结果

| 文件 | 说明 |
|------|------|
| `results/evals_validation_<YYYYMMDD_HHMMSS>.html` | **统一** Phase 2 语义评测 HTML 报告（所有 skill 合并为一份，含评分和交互详情，每行标注 skill 归属，文件名含北京时间戳，每次运行生成新文件） |
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

- `_evals.md` 文件存在性、格式合法性、必填字段检查
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
| Token 消耗 | 0-10 | **5** | 回复长度合理，思考过程工具调用高效 |

> 各维度阈值可在每个用例的 `Config` 中单独覆盖（见下方配置说明）。

## 评测用例格式（evals.md）

评测用例文件存放在 `tests/system/cases/<skill>_evals.md`，使用 Markdown + YAML frontmatter 格式：

```yaml
---
skill_name: skill-name
eval_mode: text        # text（默认）或 file_based
---
```

每个用例以 `# Case <N>: <标题>` 开头，包含以下章节：

### Config（可选）

用例级配置，格式 `- Key: value`：

| 配置项 | 类型 | 说明 | 默认值 |
|--------|------|------|--------|
| `Eval Mode` | string | `text`（语义评审）或 `file_based`（文件产出验证） | `text` |
| `Max Tokens` | int | Token 消耗上限，超过则用例失败 | 无限制 |
| `覆盖度阈值` | int | 信息覆盖度维度的最低通过分数（0-40） | `20` |
| `准确性阈值` | int | 技术准确性维度的最低通过分数（0-30） | `15` |
| `质量阈值` | int | 回复质量维度的最低通过分数（0-20） | `10` |
| `Token阈值` | int | Token 消耗维度的最低通过分数（0-10） | `5` |

```markdown
## Config
- Eval Mode: file_based
- Max Tokens: 50000
```

### Prompt

发送给 skill 的用户问题。

### Expected Output

预期回答的要点描述，供评审模型评分时参考。

### Expectations

可选的断言列表，格式 `- [contains] 期望内容`，用于模式匹配验证。

```markdown
# Case 1: 检查NPU驱动安装命令

## Prompt

我有一台昇腾服务器，想检查NPU驱动是否已安装，应该用什么命令？

## Expected Output

回复应说明使用 npu-smi info 命令检查驱动

## Expectations

- [contains] npu-smi info
```

## 沙箱隔离机制

每个评测用例在独立的沙箱目录中执行，确保互不干扰：

```
tests/system/sandboxes/
├── <skill>_eval_1/
│   ├── .opencode/
│   │   ├── opencode.json                    # opencode 工具权限白名单配置
│   │   └── skills/
│   │       └── <skill>/                     # skill 目录（默认软链接，指向源目录）
│   └── logs/                                # session 导出 JSON
│       ├── <skill>_case_1_ses.json          # 执行 session 数据
│       └── <skill>_case_1_review_ses.json   # 评审 session 数据
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

## 使用方式

### 完整执行评测

```bash
python tests/system/scripts/main.py \
    --repo-root /path/to/repo \
    --changed-files ops/ascendc-st-design/SKILL.md
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
  - "model/skills"
exclude_skills:
  - "skill-test-framework"
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
│   └── st-test.config        # 扫描路径与白名单配置
├── docs/
│   └── USER_GUIDE.md            # 详细使用指南
├── cases/                       # 集中式评测用例定义
│   └── <skill>_evals.md
├── results/                     # HTML 报告输出目录
├── logs/                        # 运行日志与归档压缩包
├── sandboxes/                   # 沙箱隔离目录（用例级别）
└── scripts/
    ├── main.py                  # CI 门禁主入口
    ├── conftest.py              # pytest 共享配置、钩子、工具函数
    ├── evals_parser.py          # evals.md Markdown 解析器
    ├── opencode_runner.py       # opencode CLI 封装（流式输出、Session 导出）
    ├── sandbox_manager.py       # 沙箱创建与管理
    ├── test_skill_basic.py      # Phase 1: 静态结构验证
    ├── test_skill_evals.py      # Phase 2: AI 语义评测
    ├── run_eval.py              # 命令行评测启动脚本
    ├── session_stats.py         # Session 数据统计工具
    ├── pytest.ini               # pytest 渲染配置
    └── requirements.txt         # Python 依赖
```

## 注意事项

1. **变更识别**：只有配置的 `skill_dirs` 目录下的变更才会触发评测。支持白名单过滤（`skill_whitelist`）。
2. **评测用例必需**：skill 必须在 `tests/system/cases/` 下有 `<skill>_evals.md` 文件才会执行 Phase 2 评测。
3. **`--report-only` 前提**：沙箱目录中须有可用的 JSON 文件（来自之前的完整测试运行）。若数据不完整（如 `ses.json` 截断），会话详情区块可能缺失，但评分不受影响。
4. **超时设置**：批量评测超时 1200 秒，单个评测用例超时 300 秒。
5. **退出码**：所有评测通过返回 0，任一失败返回 1。
