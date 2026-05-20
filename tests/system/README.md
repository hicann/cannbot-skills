# Skill 测试框架

基于变更文件识别受影响的 skills，执行对应的评测用例，输出测试结果报告。用于 CI/CD 门禁检查，确保 skills 代码变更质量。

## 核心工作流程

```
输入：repo_root + changed_files
    │
    ├─ 步骤1：识别受影响的skills
    │   └─ 从变更文件路径中提取 skill 目录
    │
    ├─ 步骤2：加载评测用例
    │   └─ 读取 <skill>/evals/evals.json
    │
    ├─ 步骤3：执行评测
    │   ├─ Phase 1: 静态结构验证（test_skill_basic.py）
    │   └─ Phase 2: AI 语义评测（test_skill_evals.py）
    │
    ├─ 步骤4：保存结果
    │   └─ 输出到 tests/system/results/
    │
    └─ 返回：通过/失败状态
```

## 输入参数

| 参数 | 必填 | 说明 | 默认值 |
|------|------|------|--------|
| `repo_root` | **必填** | 仓库根目录的绝对路径 | - |
| `changed_files` | **必填** | 变更文件列表（空格分隔，支持相对或绝对路径） | - |
| `--parallel` / `-p` | 否 | 并行 worker 数，`auto`=全部CPU，或指定数字如 `4` | `1` (顺序执行) |

## 输出结果

| 文件 | 说明 |
|------|------|
| `results/<skill_name>_<timestamp>.json` | 每个受影响skill的测试结果报告 |
| `results/basic_validation.html` | Phase 1 静态结构验证报告 |
| `results/skill_evals.html` | Phase 2 语义评测报告 |

### 结果文件结构

```json
{
  "skill_name": "ascendc-aclnn-execute",
  "timestamp": "20260509_143000",
  "repo_root": "/path/to/repo",
  "changed_files": ["ops/ascendc-aclnn-execute/SKILL.md"],
  "total_evals": 10,
  "passed_evals": 8,
  "failed_evals": 2,
  "results": [
    {
      "eval_id": 1,
      "passed": true,
      "prompt": "...",
      "expected_output": "...",
      "actual_output": "...",
      "error": "",
      "expectations": ["..."],
      "expectations_met": ["..."],
      "expectations_failed": []
    }
  ]
}
```

## 测试阶段

### Phase 1：静态结构验证（test_skill_basic.py）

无需 AI 调用，快速验证 skill 的结构完整性：

- `evals.json` 存在性、JSON 合法性、必填字段检查
- 每个 eval case 的 id、prompt、expectations 格式校验
- SKILL.md 存在性、YAML frontmatter 格式校验

### Phase 2：AI 语义评测（test_skill_evals.py）

使用 opencode CLI 执行评测用例，验证 skill 的实际表现：

- **执行 Session**：向 skill 发送 prompt，收集 AI 回复
- **评测 Session**：独立评测模型评审回复质量（信息覆盖度 40 分、技术准确性 30 分、回复质量 20 分、Token 消耗 10 分，总分 >= 60 通过）
- **断言验证**：检查 expectations 中的 `contains` / `not_contains` / `file_exists` 模式

## evals.json 格式

```json
{
  "skill_name": "skill-name",
  "evals": [
    {
      "id": 1,
      "prompt": "测试场景描述",
      "expected_output": "预期输出描述",
      "files": [],
      "expectations": ["期望内容1", "期望内容2"]
    }
  ]
}
```

### expectations 类型

| 类型 | 说明 | 示例 |
|------|------|------|
| `contains` | AI 回复必须包含指定文本 | 普通的字符串 |
| `not_contains` | AI 回复不能包含指定文本 | `!not_contains:禁止出现的内容` |
| `file_exists` | 指定文件必须被创建或修改 | `file_exists:path/to/file` |

## 使用方式

### 命令行调用

```bash
python tests/system/scripts/main.py \
    --repo-root /path/to/repo \
    --changed-files ops/ascendc-st-design/SKILL.md ops/ascendc-st-design/evals/evals.json
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

### 配置

编辑 `tests/system/config/skill-test.config` 调整扫描路径：

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
│   └── skill-test.config        # 扫描路径配置
├── docs/
│   └── USER_GUIDE.md            # 详细使用指南
├── results/                     # 测试结果输出目录
├── logs/                        # 运行日志
└── scripts/
    ├── main.py                  # CI 门禁主入口
    ├── conftest.py              # pytest 共享配置与工具函数
    ├── opencode_runner.py       # opencode CLI 封装
    ├── test_skill_basic.py      # Phase 1: 静态结构验证
    ├── test_skill_evals.py      # Phase 2: AI 语义评测
    ├── run_eval.py              # 命令行评测启动脚本
    ├── session_stats.py         # Session 数据统计工具
    ├── pytest.ini               # pytest 渲染配置
    └── requirements.txt         # Python 依赖
```

## 注意事项

1. **变更识别**：只有配置的 `skill_dirs` 目录下的变更才会触发评测
2. **evals.json 必需**：skill 必须有 `evals/evals.json` 文件才会执行 Phase 2 评测
3. **超时设置**：批量评测超时 1200 秒，单个评测用例超时 300 秒
4. **退出码**：所有评测通过返回 0，任一失败返回 1
