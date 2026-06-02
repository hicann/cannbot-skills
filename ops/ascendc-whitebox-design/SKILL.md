---
name: ascendc-whitebox-design
description: Ascend C 算子白盒测试用例生成系统。分析算子源码提取参数维度，自动枚举参数组合，生成可执行的白盒测试用例。自动两套输出：全量（路径覆盖+网络用例）与低覆盖（采样+网络用例）。触发场景：(1) "为 X 算子生成白盒测试用例" (2) "算子白盒用例生成" (3) "generate whitebox test cases for operator"。
metadata:
  category: testing
  workflow-steps: "6+TTK"
---

## 做什么

- 分析算子源码（tiling + kernel + 接口），提取分支路径和参数维度
- 自动枚举参数组合，生成白盒测试用例（S5_mapped_cases_low.json / S5_mapped_cases_high.json + pytest 脚本）
- 支持路径覆盖率分析、交叉验证、用例审查

## 何时使用

- 用户要求为某个算子生成白盒测试用例
- 关键词："白盒用例"、"whitebox test cases"、"路径覆盖"

**触发示例：**

```
为 add 算子生成白盒测试用例
```

## 输入参数

> **平台参数（芯片型号/核数/UB）由 `npu-arch` skill 自动检测，无需用户指定。**
>
> **参数收集时序**：下表参数统一在 workflow.md Step 1.1 中通过用户交互一次性收集。其中「算子路径」仅收集获取方式（自动查找/手动输入），实际路径定位在 Step 1.2 执行。**禁止在 Step 1.1 完成前执行任何路径搜索（glob/find）**。

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| 算子名称 | 是 | — | 如 `add`、`add_rms_norm` |
| 跳过 Step 4 闸门 | 否 | 不跳过 | 跳过则 Step 1-3 后自动进入 Step 5 |
| 算子路径 | 否 | 自动查找 | 选择「自动查找」或「手动输入」具体路径 |
| TTK CSV 生成 | 否 | 不生成 | 生成 ttk_ 前缀的 CSV 文件（可选模块） |

## 前置条件

- 算子源码存在于项目中（需包含 tiling 代码 + kernel 代码 + 接口定义）
- Python 3.7+ 环境
- 基础流程不需要额外 pip 包
- 产物会写入算子源码目录下的 `tests/whitebox/`，重复执行会覆盖已有文件
- TTK 模块（启用时）：需安装 `numpy`，项目需包含 `ops-test-kit/` 目录，TTK kernel 执行命令必须在 `ops-test-kit/` 目录下运行
- TTK 模块（启用时）：算子目录需能定位 aclnn API 文档（如 `docs/aclnn{OpName}.md`），用于生成 `golden_plugin.py` 的参考实现
- TTK 模块（启用时）：需使用支持 CSV kernel 模式、`--plugin` 加载和当前 golden 函数签名的 TTK 版本；若本地 TTK 行为不匹配，停止并报告版本不兼容

## 人工交互节点

| 阶段 | 交互内容 | 是否必须 |
|------|---------|---------|
| Step 1 | 收集输入参数，确认摘要 | 是 |
| Step 2 Phase 0 | Phase 0 校验失败且重试仍失败时，报告用户决策 | 仅当校验失败时 |
| Step 2 Phase 3a | 源码分析发现争议路径时，询问保留或排除 | 仅当存在争议路径时 |
| Step 4 | 安全闸门：确认分析和验证结果后再生成用例 | 是（除非提前选择跳过） |
| TTK 模块 | 确认 TTK CSV 生成结果 | 仅当输入选择"生成"时 |

## 执行流程

触发后，加载 `references/workflow.md` 按步骤执行。

| 步骤 | 目标 | 主要产物 |
|------|------|----------|
| Step 1 | 收集参数和定位算子路径 | 输入摘要 |
| Step 2 | 源码侦察、接口建模、参数推导 | S2P0/S2P1/S2P2/S2P3 产物 |
| Step 3 | 交叉验证测试设计 | S3_verification_report.json |
| Step 4 | 人工闸门确认 | 用户确认记录 |
| Step 5 | 映射抽象用例到 tensor 配置 | S5_mapped_cases_low/high.json |
| Step 6 | 生成并执行 pytest/ST 证据 | S6 pytest、结果和覆盖率 |
| TTK | 可选转换为 TTK CSV 并执行 | ttk_*.csv、golden_plugin.py |

预计用例规模：全量覆盖（路径覆盖+网络用例），低覆盖约 25 个（随机采样+网络用例）。
单 pytest 文件通过 --cases-file 切换数据源，两套 JSON 始终同时产出。

## 关键行为说明

- **自动重试**：Step 3 交叉验证失败时，会自动回退 Step 2 重新分析，最多 3 轮，无需人工干预。
- **精度兜底**：Step 6 中精度不达标的用例会自动标记为 XFAIL（保留偏差信息），而非删除或阻塞交付。
- **无 NPU 兼容**：Step 6 在无 NPU 环境下会自动跳过硬件执行，标记为 SKIPPED，不会报错。
- **精度标准锁定**：pytest 中的 rtol/atol 阈值禁止手动修改，这是质量保证的最后门槛。
- **TTK CSV 可选**：TTK 模块为可选扩展，默认不执行。仅在用户选择时，于 Step 6 完成后生成 TTK CSV 文件。
- **TilingKey 覆盖率**：Step 6c 自动从 plog 提取所有用例命中的 tiling key 值，与 `S2P2_param_def.json` 的顶层 `tiling_keys` 期望集合对比，生成 `S6_tilingkey_coverage.json`（含全局与 per_group 覆盖率）。覆盖率为诊断指标。路径B `tilingkey_single.py` 提供单用例调试入口。

## 最终产物

输出目录：`{算子路径}/tests/whitebox/`

| 文件 | 阶段 | 说明 |
|------|------|------|
| `S2P0_scout_t.md` | Step 2 Phase 0 | tiling 侦察报告（分支可达性 + 平台标注） |
| `S2P0_scout_k.md` | Step 2 Phase 0 | kernel 侦察报告（dispatch 模式 + key 数量） |
| `S2P0_file_manifest.json` | Step 2 Phase 0 | 源码文件侦察清单（tiling/kernel 文件优先级 + dispatch 模式 + key 计数） |
| `S2P1_path_list.json` | Step 2 | 代码路径清单 + 分支树 |
| `S2P2_param_def.json` | Step 2 | 参数定义 + 约束 + 分组 + `tiling_keys`（供 Step 6c 覆盖率计算） |
| `S2P1_operator_model.json` | Step 2 | 算子输入输出模型（dtype/shape/presence 规则）+ shape_mapping 节（Step 5 消费） |
| `S2P3_test_design.md` | Step 2-3 | 测试设计文档，Step 4 闸门的确认依据 |
| `S3_verification_report.json` | Step 3 | 交叉验证结果 |
| `S5_case_mapper.py` | Step 5 | 参数组合映射脚本（运行时加载 S2P1_operator_model.json） |
| `S5_verify_mapper.py` | Step 5 | mapper 4 层验证脚本 |
| `S5_mapped_cases_path.json` | Step 5a | 路径覆盖用例（mapped 格式，中间产物） |
| `S5_mapped_cases_network.json` | Step 5b | 网络用例（mapped 格式，中间产物） |
| `S5_mapped_cases_low.json` | Step 5d | low 档位基础映射集合（全路径+全网络 all normal） |
| `S5_mapped_cases_high.json` | Step 5d | high 档位 data_range 扩展映射集合（one-hot + 全统一） |
| `S6_test_{op_name}.py` | Step 6a | 可执行的 pytest 脚本。通过 --cases-file 切换数据源，-k 筛选单用例 |
| `S6_tilingkey_coverage.json` | Step 6c | tiling key 覆盖率报告（全局 + per_group 命中/未命中/覆盖率） |
| `tilingkey_logs/{op_name}_full.log` | Step 6c | 全量 plog 副本 |
| `tilingkey_logs/{op_name}_{case_id}.log` | 路径B 调试 | 单用例 plog 副本 |
| `ttk_extract_case_info.py` | TTK 模块（条件生成） | 单用例信息提取脚本（无 torch 依赖） |
| `ttk_{op_name}_cases_low.csv` | TTK 模块 | low 数据源 CSV（由 `S5_mapped_cases_low.json` 转换） |
| `ttk_{op_name}_cases_full.csv` | TTK 模块 | full 数据源 CSV（由 `S5_mapped_cases_high.json` 转换） |
| `golden_plugin.py` | TTK 模块（条件生成） | TTK 自定义 golden 函数（通过 `--plugin` 加载） |

## 使用指南

- **查看测试设计**：`S2P3_test_design.md` 包含所有测试分组的路径分析结果和验证结论，建议在 Step 4 闸门时检视。
- **运行测试**：
  - low 数据源：`cd {算子路径} && pytest tests/whitebox/S6_test_{op_name}.py --cases-file=S5_mapped_cases_low.json -v`
  - high 数据源：`cd {算子路径} && pytest tests/whitebox/S6_test_{op_name}.py --cases-file=S5_mapped_cases_high.json -v`
  - 单用例：`cd {算子路径} && pytest tests/whitebox/S6_test_{op_name}.py --cases-file=S5_mapped_cases_low.json -k case00001 -v`

## 执行约束（强制）

### 全局顺序约束（强制）

- **禁止跳步**：必须按 workflow.md 的 Step 编号顺序执行，完成当前步骤的全部子步骤后才能进入下一步骤。
- **禁止抢跑**：前置条件未全部满足时，禁止启动该步骤的任何操作（包括搜索、派发子 agent、读写文件、生成产物）。
- **前置条件检查**：执行每个步骤前，必须确认该步骤在 workflow.md 中标注的 `> **前置条件**` 已全部满足。不满足则禁止继续，不得跳过检查。
- **禁止合并/跳过子步骤**：每个 Step 内部的子步骤（如 Step 1 的 1.0→1.1→1.2→1.3→1.4）必须严格按编号顺序逐步执行，禁止合并执行或跳过任何子步骤。

### 子 agent 规则传递

- **子 agent 自读规则**：所有行为规范、约束条件、执行步骤必须由子 agent 通过 Read 工具直接从源文件读取获取。禁止主 Agent 读取源文件后转述、摘要、改写或拆分后再传入子 agent prompt。
- **主 Agent 仅传上下文参数**：主 Agent 派发子 agent 时，prompt 中只传递上下文参数（算子路径、平台参数、JSON 文件路径、产出写入路径等变量值），不传递任何规则定义。
- **无专用 prompt 文件时**：若某步骤无专用 prompt 文件（如 Step 1 路径查找、Step 2 Phase 1 Task C），应指示子 agent 直接 Read `references/workflow.md` 的对应小节获取规则，主 Agent 同样不做规则转述。
- **结构化数据例外**：默认情况下子 agent 应自行 Read 源文件获取数据。仅当源文件体积过大而子 agent 只需其中极少部分时，允许主 Agent 从中提取相关内容按固定格式模板拼接后内联传入（此时子 agent 读不到完整源文件，不会浪费上下文空间）。此例外**每次使用时必须在 workflow.md 中显式声明提取来源、提取范围和拼接格式**，禁止作为默认做法滥用。当前使用场景：Phase 0→1 过渡中从 `S2P0_file_manifest.json` 提取文件清单和读取范围，供 Phase 1 Task A 和 Phase 2 Task D 消费（见 Step 2 源码读取范围规则）。

### 子 agent 数据传递

- **有文件承载的数据**：主 Agent 派发子 agent 时，仅传文件路径，让子 agent 使用 Read 工具自行读取。禁止将文件内容复制粘贴到 prompt 中——这会挤占子 agent 上下文空间，导致输出截断或任务失败。

### 子 agent 按需读取

- **禁止提前读取**：子 agent 在执行过程中必须按步骤顺序逐步读取参考文档，仅当执行到某步骤时才能 Read 该步骤标注的文件。禁止在启动时或前期步骤中一次性 Read 所有参考文档——这会导致上下文拥塞和卡顿。
- **违规后果**：主 Agent 发现子 agent 提前读取后续步骤文件并导致卡顿时，应终止该子 agent 并重新派发（仅传当前步骤所需的最小文件路径集）。

### Step 1.2 路径查找约束

- **条件激活**：Step 1.2 仅在用户于 Step 1.1 选择「自动查找」时执行搜索逻辑；选择「手动输入」时仅验证路径存在性。
- **主 Agent 执行搜索**：搜索规则严格遵循 `references/workflow.md` Step 1.2 小节（限项目根目录，精确匹配 `ops-*/*/{op_name}/`，禁止子串匹配，禁止跨项目）。
- **禁止派发子 agent**：路径查找由主 Agent 直接执行，禁止派发子 agent（避免绕过搜索范围约束）。

### 安全闸门

- **Step 4 闸门**：完成 Step 3 后必须停下来，展示摘要并等待用户明确确认，不得自动进入 Step 5。
- **TTK 模块闸门**：TTK CSV + golden_plugin 生成并 Kernel 执行验收通过后必须停下来，展示摘要并等待用户确认。
- **禁止抢跑**：用户确认前，不得生成 `S5_mapped_cases_low.json` / `S5_mapped_cases_high.json` / `S6_test_*.py`。
- 详细禁止项与例外见 `references/workflow.md`「Step 4 闸门」小节。
