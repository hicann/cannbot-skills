# White-box Pytest Test Generation Workflow

> ⚠️ **全流程约束规则的唯一定义在 SKILL.md「执行约束（强制）」节。** 主 Agent 和子 agent 执行前必须核对。

---


## Step 1：输入收集

> **前置条件**：无（流程起始步）。

主 Agent 直接执行。按 1.0 → 1.1 → 1.2 → 1.3 → 1.4 严格逐步收集平台参数和用户输入，最后展示摘要并等待用户确认。详细规则见 `references/S1-input-collection.md`。

> **关键串行要求**：1.2（定位算子路径）必须在 1.1（收集用户输入）完成后才能执行。1.1 会询问用户选择"自动查找"还是"手动输入路径"，只有确认后才能决定是否执行 `find` 搜索。question 工具返回前，禁止执行任何路径搜索操作。

**输出数据**：

| 数据 | 来源 | 用途 |
|------|------|------|
| `npu_arch` / `soc_version` / `chip_model` | 1.0 平台检测 | 全流程平台判断 |
| `platform`（核数 / UB 大小） | 1.3 参数映射 | 子 agent 上下文 |
| `op_name` / `op_path` / `gate_status` / `ttk_status` | 1.1 用户输入，1.2 两阶段查找（目录名 + OP_ADD 反查） | 流程控制和产物路径 |

---

## Step 2：分析源码

> **前置条件**：Step 1.4 用户已确认摘要（"确认，开始分析"）；算子路径、平台参数均已确定。

**输入文件**：

| 文件 | 用途 |
|------|------|
| 算子源码路径 | Step 1.2 定位，tiling / kernel / 接口分析 |
| 平台参数（npu_arch / soc_version / core_count / ub_size） | Step 1.0-1.3，平台分支判断 |

**执行概要**：

主 Agent 按阶段调度子 agent 完成源码分析。执行顺序、各 Phase 的详细规则和文件索引见 `references/source-analysis/00-execution-order.md`。

**输出文件**：

| 文件 | 说明 |
|------|------|
| `S2P0_scout_t.md` | tiling 侦察报告（入口函数 + P0/P1/P2 文件清单） |
| `S2P0_scout_t.json` | tiling 侦察结构化数据（供 verify 消费） |
| `S2P0_scout_k.md` | kernel 侦察报告（dispatch 类型 + key 列表） |
| `S2P0_scout_k.json` | kernel 侦察结构化数据（供 verify 消费） |
| `S2P0_file_manifest.json` | 源码文件清单（tiling/kernel 优先级 + 排除列表） |
| `S2P0_source_scope.md` | 源码读取范围（供 Task A/D 直接读取） |
| `S2P1_path_list.json` | 代码路径清单 + 分支树 |
| `S2P1_tiling_glossary.md` | tiling 变量含义表（tiling 源码变量名 → 语义名映射） |
| `S2P1_operator_model.json` | 算子接口模型（inputs/outputs/attributes） |
| `S2P1_low_configs.json` | 常见网络 shape 配置（语义参数名） |
| `S2P2_analysis_data.json` | 紧凑格式分析数据（LLM 生成） |
| `S2P2_dim_spec.json` | 维度范围规格（由 assemble_dim_spec.py 生成） |
| `S2P2_param_def_groups.json` | 推导数据文件（由 pick_dims.py 自动生成） |
| `S2P2_reachability_data.json` | 可达性 + 分组中间数据（update_path_list.py 输入） |
| `S2P2_param_def.json` | 参数定义 + 约束 + 分组 + tiling_keys（由 builder 自动生成） |
| `S2P2_gen_cases.py` | 参数组合枚举脚本 |
| `S2P2_cases.json` | 参数组合枚举结果 |
| `S2P2_traceability.md` | 推导追溯文档 |
| `S2P3_test_design.md` | 测试设计文档 |

---

## Step 3：交叉验证

> **前置条件**：Step 2 全部 Phase（0/1/2/3）已完成；`S2P3_test_design.md` 已生成；Phase 3a 的 disputed 路径已由用户确认（无 disputed 则自动满足）。

派 1 个独立子 agent（不复用 Step 2 agent，确保独立视角）。主 Agent prompt：指示 Read `{skill_base}/references/design-verifier/00-execution-order.md`，按执行顺序约束表逐步执行。上下文参数：`op_name` / `op_path` / `skill_base` / `output_path`（写入路径 `{op_path}/tests/whitebox/S3_verification_report.md`）。

**输入文件**：每个步骤按需读取所需文件（详见 `design-verifier/00-execution-order.md` 执行顺序约束表），不提前读取后续步骤的文件。

输出：`S3_verification_report.md`。fail 项 → **主 Agent 独立核实**：对每个 fail 项 Read 验证报告中引用的源码位置，获取源码原文，与验证报告的「S2 声称」和「源码实际」比对，确认 fail 判定正确后才修改 S2 产物。若发现验证报告引用的源码与实际源码不一致（验证器误判），保留 S2 产物不修改，在报告中注明验证器误判。核实后 → 回 Step 2 Phase 1 重分析（Phase 0 不需重跑，最多 3 轮），仍 fail → 触发轮次耗尽协议。warn 项 → Step 3 子 agent 返回后，主 Agent 在 Step 4 用户确认前逐条判断是否修正，结论写入 S2P3_test_design.md 验证结论节；pass 项 → 无需处理。

---

## Step 4：用户确认

> **前置条件**：Step 3 已完成，`S3_verification_report.md` 已生成；验证结论已更新到 `S2P3_test_design.md`。

将验证结论更新到 S2P3_test_design.md 后停下来等待用户确认。提示语："源码分析和交叉验证已完成：{N} 个测试 group，S2P2 参数组合 ~{M} 条（最终用例数含网络和空 tensor 变体可能更多），验证状态 {status}。"

| 选项 | 说明 |
|------|------|
| 确认，继续 | 进入 Step 5（case mapper） |
| 需要调整 | 回 Step 2 修改 |

IF Step 1 选择「跳过 Step 4 闸门」→ 跳过此步直接进入 Step 5；ELSE 用户确认后才能进入 Step 5。

### Step 4 闸门（强制）

收到用户确认之前，**禁止**：运行 Step 5（case mapper）、生成 `S5_mapped_cases_low.json` / `S5_mapped_cases_high.json` 或 `S6_test_{op_name}.py`。

**允许继续条件（满足其一）**：用户在对话中明确确认（如「确认」「继续生成 cases」）或用户写明「跳过 Step4 确认」/「一次跑完全流程」。

---

## Step 5：映射参数组合（子 agent）

> **前置条件**：Step 4 用户已确认；`S2P2_cases.json`、`S2P1_operator_model.json`、`S2P2_param_def.json`、`S2P1_low_configs.json` 均已产出。

派 1 个独立子 agent。主 Agent prompt：

先 Read `{skill_base}/references/case-mapper/00-execution-order.md`，按其「执行顺序约束」表逐步 Read 子文件并执行。约束规则见 `case-mapper/05-constraints.md`。

**5a**：先按 `case-mapper/01-mapping-spec.md` 生成 `S5_mapping_spec.md`（自然语言映射规格），再基于 spec 以 `S2P2_cases.json` + `S2P1_operator_model.json` + `S2P2_param_def.json` 为输入生成 `S5_case_mapper.py` + `S5_verify_mapper.py`，运行产出 `S5_mapped_cases_path.json`（路径覆盖）。验证失败→修复（最多 3 轮），仍 fail → 触发轮次耗尽协议。

**5b**：读取 `S5_mapping_spec.md`（算子侧规格）+ `S2P1_low_configs.json`（网络侧结构），自行生成映射规则并写回 `S5_mapping_spec.md` §网络用例映射，将语义参数名映射为算子参数名，生成 `S2P2_network_cases.json`（与 `S2P2_cases.json` 格式一致），调用 `S5_case_mapper.main(network_cases_file, network_out_file, id_prefix="network")` 复用 5a 全部管道，附加 `_source`/`_reason` 后产出 `S5_mapped_cases_network.json`。

**5c**：生成 `S5_merge_expand.py`，运行默认入口 → `S5_mapped_cases_low.json`（过滤元素数 > 1 亿的 path/network case，合并过滤后的路径覆盖 + 网络用例 + 空 tensor 变体，所有输入置 `_data_range: "normal"`，从 `S2P1_operator_model.json` 读取 `value_domain` 附加为 `_value_domain`），随后运行 `python S5_merge_expand.py 5d` → `S5_mapped_cases_high.json`（基于 low 做 one-hot + 全统一展开，按 `_value_domain` 过滤不兼容 data_range，空 tensor 变体保留 normal）。

- low 用于门禁（6b）和 tilingkey 覆盖率（6c）
- high 用于全量 data_range 验证（信息性，非门禁）

---

## Step 6：生成 pytest 并执行

> **前置条件**：Step 5 已完成，`S5_mapped_cases_low.json` / `S5_mapped_cases_high.json` 已生成。

**输入文件**：

| 文件 | 用途 |
|------|------|
| `S5_mapped_cases_low.json` | low 档位用例（门禁 + tilingkey 覆盖率） |
| `S5_mapped_cases_high.json` | high 档位用例（data_range 展开） |
| `S2P2_param_def.json` | 参数定义（理解参数含义 + tilingkey 期望集合） |
| 算子 `docs/aclnn*.md`「计算公式」节 | Reference 实现的唯一依据（只读该节） |

**执行概要**：

主 Agent 直接执行。先 Read `{skill_base}/references/pytest-gen/00-execution-order.md`，按其「文件索引」表按需 Read 子文件并执行。约束规则见 `pytest-gen/02-constraints.md`（全程适用）。生成 pytest 测试文件，执行门禁验证，生成 tilingkey 覆盖率。验证流程、门禁规则、失败处理均见 `pytest-gen/02-constraints.md`。

修复后重新运行（最多 3 轮），仍不通过 → 触发轮次耗尽协议。

**输出文件**：

| 文件 | 说明 |
|------|------|
| `S6_test_{op_name}.py` | pytest 测试文件（通过 `--cases-file` 切换 low/high） |
| `S6_tilingkey_coverage.json` | tilingkey 覆盖率报告（信息性产出） |
| `tilingkey_logs/{op_name}_full.log` | plog 副本 |

**验证指令**：

```bash
ASCEND_GLOBAL_LOG_LEVEL=1 pytest S6_test_{op_name}.py --cases-file=S5_mapped_cases_low.json -q --tb=line
```

单用例 tilingkey 调试（按需）：`python {skill_scripts}/tilingkey_single.py --op-path {op_path} --case-id {case_id}`。

---

## 可选模块：TTK CSV 生成

> **注意**：当前 TTK 模块仅支持 `kernel` 模式（`ttk kernel`），不支持 `e2e` 和 `aclnn` 模式。执行验收时必须使用 `python3 -m ttk kernel` 命令。

### 启用条件

Step 1 输入 4 选择了「启用」。若未启用，跳过本模块。

### 调用方式

**方式 A（自动）**：主流程 Step 6 门禁全部通过后自动触发。

> **前置条件**：Step 6 门禁全部通过（0 FAILED / 0 ERROR / 0 RuntimeError）。

**方式 B（独立）**：用户可随时派发子 agent 单独生成，只需提供 `S5_mapped_cases_low.json` / `S5_mapped_cases_high.json` + `S2P1_operator_model.json` 路径和算子路径，不依赖其他 Step 产物或上下文。

### 执行

派 1 个独立子 agent。主 Agent prompt：指示 Read `{skill_base}/references/ttk-converter/00-execution-order.md`（优先执行入口顶部的执行顺序约束节），传入上下文参数：`S5_mapped_cases_low.json` / `S5_mapped_cases_high.json` / `S2P1_operator_model.json` / `S5_mapping_spec.md` 路径、`op_name`、算子源码路径（`*_def.cpp` / `*_infershape.cpp`，infershape 可能位于共享目录，需按 3 级回退定位实际路径后传入）、产出写入路径。

子 agent 在 ttk-converter/00-execution-order.md 的任务 1-5 完成后返回。

主 Agent **必须先 Read `{skill_base}/references/ttk-converter/05-kernel-acceptance.md`**，严格按其中 6a/6b 节的命令模板执行（禁止凭经验拼凑命令）。未读取该文件禁止执行任务 6a/6b/6c。

**6c（全量执行，可选）**：6b 门禁通过后，主 Agent 向用户确认是否执行全量 low CSV。默认不执行；1 分钟内未回复则跳过。用户确认后用 nohup 后台执行，提供进度查询指令，不阻塞流程。

### 产出（仅模块启用时）

产出文件列表见「最终产物」节 TTK 模块子树。

---

## 最终产物

```
{算子源码路径}/tests/whitebox/
├── S2P0_scout_t.md
├── S2P0_scout_t.json
├── S2P0_scout_k.md
├── S2P0_scout_k.json
├── S2P0_file_manifest.json
├── S2P0_source_scope.md
├── S2P1_path_list.json
├── S2P1_tiling_glossary.md
├── S2P1_low_configs.json
├── S2P1_operator_model.json
├── S2P2_analysis_data.json      ← LLM 生成的紧凑格式分析数据
├── S2P2_dim_spec.json            ← 由 assemble_dim_spec.py 生成的维度范围规格
├── S2P2_param_def_groups.json    ← 由 pick_dims.py 从 dim_spec 生成
├── S2P2_reachability_data.json  ← 可达性 + 分组中间数据（update_path_list.py 输入）
├── S2P2_param_def.json          ← 由 build_param_def.py 从 groups 文件生成
├── S2P2_gen_cases.py
├── S2P2_cases.json
├── S2P2_traceability.md
├── S2P3_test_design.md
├── S3_verification_report.md
├── S5_case_mapper.py
├── S5_verify_mapper.py
├── S5_merge_expand.py
├── S5_mapping_spec.md
├── S5_mapped_cases_path.json
├── S5_mapped_cases_network.json
├── S2P2_network_cases.json
├── S5_mapped_cases_low.json
├── S5_mapped_cases_high.json
├── conftest.py
├── S6_test_{op_name}.py
├── S6_tilingkey_coverage.json
├── ttk_extract_case_info.py
├── ttk_{op_name}_cases_low.csv
├── ttk_{op_name}_cases_full.csv
├── ttk_{op_name}_cases_low_sample_result.csv   ← 6b 产出
├── ttk_{op_name}_cases_low_result.csv          ← 6c 产出（可选）
├── golden_plugin.py          ← 条件产物：仅自生成时（已有 tests/assets/golden.py 时不生成）
├── tilingkey_logs/
│   ├── {op_name}_full.log
│   └── {op_name}_{case_id}.log
```

## 参考提示词索引

| Step | 提示词文件 | 执行方 | 执行顺序节 |
|------|-----------|--------|-----------|
| 1 | `S1-input-collection.md` | 主 Agent | — |
| 2 | `source-analysis/00-execution-order.md` | 主 Agent | `执行顺序约束（强制）` — 10 行 |
| 3 | `design-verifier/00-execution-order.md` + `design-verifier/05-output-schema.md` | 子 agent | `执行顺序约束（强制）` — 4 步（Task A→B→D 严格顺序 + 输出汇总） |
| 5 | `case-mapper/00-execution-order.md` + `case-mapper/05-constraints.md` | 子 agent | `执行顺序约束（强制）` — 4 步（5a-pre→5a→5b→5c 逐步按需读取） |
| 6 | `pytest-gen/00-execution-order.md` | 主 Agent | `执行顺序约束（强制）` — 3 步 |
| 6c | `scripts/compute_tilingkey_coverage.py` | 主 Agent | — |
| 6c 路径B | `scripts/tilingkey_single.py` | 主 Agent（按需） | — |
| TTK 模块 | `ttk-converter/00-execution-order.md` | 子 agent（可选模块，由 Step 1 输入6 控制） | `执行顺序约束（强制）` — 8 任务 |
