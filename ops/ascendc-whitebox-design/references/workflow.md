# White-box Pytest Test Generation Workflow

---

## Step 1：输入收集

从用户消息中提取以下信息。缺失项**主动询问**（一次性问完）：

- **算子源码路径**（必选）
- **目标平台**（默认 ascend950）
- **覆盖档位**（默认 medium——Pairwise + 常见网络 shape，~100-300 个）
  - low——常见网络 shape，~10-30 个，快速冒烟
  - medium——Pairwise + 常见网络 shape，~100-300 个，简单自测
  - high——全笛卡尔积 + 常见网络 shape，~1000+ 个，全量覆盖
- **输出目录**（默认 `{算子源码路径}/tests/whitebox/`）

平台参数默认值：
| 平台 | 核数 | UB 大小 |
|------|------|---------|
| ascend910b | 48 | 192KB |
| ascend950 | 64 | 240KB |

step2前获取平台npuarch编号: 
使用skill /npu-arch 获取npuarch, 禁止猜测和推到

确认平台后声明：

> "目标平台 {platform} {npuarch}，将使用 {核数} 核、{UB}KB UB。结果输出到 {算子源码路径}/tests/whitebox/。如需修改核数、UB 大小，或有额外特殊条件需添加，请告知。确认后开始分析。"

```
IF 用户消息中包含算子路径 + 平台 + 档位:
    直接进入 Step 2
ELSE:
    询问缺失项（一次性问完）
    等用户回复后再进入 Step 2
```
Step1结束后, 进入Step 2
---

## Step 2：并行分析源码

**Phase 1**：读取提示词：`references/prompts/code-analyzer.md`和`references/prompts/param-derivation.md`，按照标准完成下面4个task任务：

- **task A（tiling + kernel）**：读 op_host/*_tiling*.cpp + op_kernel/arch35/*.h → 分支树 + 路径清单 + 源码约束表，task A 的路径清单持久化为 `path_list.json`。
- **task B（接口）**：读 _def.cpp 或 torch_ops_extension/csrc/ + proto.h → 合法输入空间 + 平台限制
- **task C（网络搜索）**：用 WebSearch 搜索该算子常见网络 shape → low_configs。搜不到则由主 Agent 推断
- **task D（路径分析）**：路径清单（disputed 已解决）+ 接口合法输入空间 → 可达路径 + 路径分组 + 参数定义 + 约束 + 一致性检查

**Phase 1 完成后：检查 disputed 路径**

主 Agent 检查 task A 的路径清单，如果存在接口层声明不支持但 kernel 有实现的路径（disputed），向用户提问（一次性问完）：

> "源码分析 Phase 1 完成，发现 {N} 条代码路径。
>
> 以下路径在接口层声明不支持，但 kernel 中有完整实现，需要您确认是否纳入测试：
> 1. {路径名}——{描述}——建议：{处理方式}
> 2. {路径名}——{描述}——建议：{处理方式}
>
> 请逐条回复'包含'或'排除'，或回复'全部接受建议'继续。"

用户确认后，将 disputed 路径标记为 reachable 或排除（排除的记入 test_design.md）。

**主 Agent 合并后输出**：`path_list.json` + `param_def.json` + `test_design.md`
其中test_design.md需要读取（test_design.md 模板）：`references/prompts/test-design-template.md`

## Step 3：交叉验证

**执行方式**：派 1 个独立子 agent（不复用 Step 2 的 agent，确保独立视角），读取提示词`references/prompts/test-design-checker.md`, 按照提示词工作。

输入：param_def.json + test_design.md + path_list.json + 源码路径
输出：`verification_report.json`

**处理验证结果**：
1. **fail 项**：必须修正，回 Step 2 重新分析，最多 3 轮
2. **warn 项**：逐条判断是否需要修正，将结论写入 test_design.md 的"验证结论"节
3. **pass 项**：无需处理

## Step 4 完成前面step1-3步后,停下来等用户确认

将验证结论更新到 test_design.md 后，**必须停下来**展示摘要并等待用户确认：

> "源码分析和交叉验证已完成，请检视 test_design.md：
> - {N} 个测试 group，预估 ~{M} 个参数组合
> - 覆盖档位：{level}（{描述}）
> - {K} 个未确认项需要您决定
> - 验证状态：{status}
>
> 确认后继续生成。"

**用户确认后才能进入 Step 5。** 如果用户要调整，修改后重新展示。

### Step 4 闸门（强制，防止跳过）

在收到用户确认之前，**禁止**执行以下动作：

| 禁止项 | 说明 |
|--------|------|
| 运行 `scripts/run.py` | 会生成 `cases.json`，属于 Step 5 |
| 写入或覆盖 `cases.json` / `coverage_report.json` | Step 5 产物 |
| 生成依赖 `cases.json` 的 `review_report.json`、最终 `test_{op}.py` | Step 6–7；占位草稿需标注「待 Step4 确认」 |

**允许的继续条件（满足其一即可）**：

1. 用户在对话中**明确确认**（例如「确认」「继续生成 cases」）。
2. 用户在同一次需求中写明 **「跳过 Step4 确认」** 或 **「一次跑完全流程」**（Agent 须在回复中声明已跳过闸门）。

可选工程化手段：在 `{output_dir}/` 下由用户创建空文件 `STEP3_APPROVED`（或写入 `yes`），表示已线下确认；自动化脚本可检查该文件后再调用 `run.py`（由团队自行接 CI）。

---

## Step 5：枚举参数组合

**执行方式**：主 Agent 直接调用 Bash 运行脚本。

```bash
python scripts/run.py \
  --param-def <output_dir>/param_def.json \
  --output_dir <output_dir>/ \
  --coverage {用户选择的档位} --seed 42
```

引擎读取 param_def + constraints + low_configs，展开后自动过滤不合法组合。

输出：`cases.json` + `coverage_report.json`（单因子和 pairwise 覆盖率报告）

---

## Step 6：审查参数组合

**执行方式**：派 1 个独立子 agent。

输入：cases.json + coverage_report.json + param_def.json + path_list.json + 源码路径
输出：`review_report.json`

检查：约束合规性、覆盖完整性、low_configs 包含、源码交叉验证。

fail 时检查 constraints 是否遗漏，回 Step 2 或 Step 4 修正。

读取提示词，按照提示词工作：`references/prompts/result-checker.md`

---

## Step 7：生成 pytest

**执行方式**：主 Agent 直接执行（需要完整上下文来写代码）。

输入：cases.json + reference 实现 + 接口信息
输出：`test_{op_name}.py`

生成后验证：
1. `python -m py_compile test_{op_name}.py` — 语法检查
2. `pytest --collect-only test_{op_name}.py` — 收集检查（区分语法错误 vs 环境缺依赖）

读取提示词，按照提示词工作：`references/prompts/pytest-generator.md`

---

## 最终产物

```
{算子源码路径}/tests/whitebox/
├── path_list.json           # Step 2 (task A + task D)
├── param_def.json          # Step 2
├── test_design.md           # Step 2 + Step 3 验证结论
├── verification_report.json # Step 3
├── cases.json               # Step 5（枚举）
├── coverage_report.json     # Step 5（覆盖率报告）
├── review_report.json       # Step 6
└── test_{op_name}.py        # Step 7
```

## 参考提示词索引

| Step | 提示词 | 执行方式 |
|------|--------|---------|
| 2 Phase 1 | `references/prompts/code-analyzer.md` | 子 agent 并行 (task A) |
| 2 Phase 2 | `references/prompts/param-derivation.md` | 子 agent 串行 (task D) |
| 2 模板 | `references/prompts/test-design-template.md` | 主 Agent 合并输出 |
| 3 | `references/prompts/test-design-checker.md` | 独立子 agent |
| 5 | `references/prompts/result-checker.md` | 独立子 agent |
| 6 | `references/prompts/pytest-generator.md` | 主 Agent |

