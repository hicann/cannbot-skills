# Source Analysis — 执行总纲

## 角色

分析算子源码（tiling + kernel + 接口），提取分支路径和参数维度，产出代码路径清单、算子接口模型、参数定义和测试设计文档，供下游 Step 5（case mapper）和 Step 6（pytest）消费。

## 输入

| 数据 | 来源 | 用途 |
|------|------|------|
| 算子源码路径 | Step 1.2 | tiling / kernel / 接口分析 |
| 平台参数（npu_arch / soc_version / core_count / ub_size） | Step 1.0-1.3 | 平台分支判断 |

## 源码读取范围规则（强制）

`S2P0_file_manifest.json` 体积较大，子 agent 只需文件清单和读取范围。主 Agent 在 Phase 0→1 过渡阶段提取 `tiling.file_list` + `kernel.file_list` + `tiling.excluded` + `kernel.excluded`，按固定格式拼接为文本块后内联传入 Task A 和 Task D。子 Agent 严格按此范围读取，不得自行添加。

### 格式化模板

Phase 0→1 过渡阶段，主 Agent 将 `S2P0_file_manifest.json` 提取为以下文本块，传入 Phase 1 Task A 和 Phase 2 Task D：

```
源码读取范围（严格遵守，禁止自行添加其他文件）：
【tiling】
- P0: {path} — {read_strategy 描述，附 symbols}
- P1: {path} — 仅读符号 {symbols}（read_strategy={value}）
【kernel】
- P0: {path} — 仅读 TILING_KEY_IS dispatch 块（{key_count}条, pattern={pattern}）
- 总计 {kernel.total_key_count} 条 key
【排除】以下文件禁止读取：
- {excluded path} — {reason}
```

## 输出文件

| 文件 | Phase | 说明 |
|------|-------|------|
| `S2P0_scout_t.md` | Phase 0 | tiling 侦察报告（分支可达性 + 平台标注） |
| `S2P0_scout_k.md` | Phase 0 | kernel 侦察报告（dispatch 模式 + key 数量） |
| `S2P0_file_manifest.json` | Phase 0 | 源码文件清单（tiling/kernel 优先级 + 排除列表） |
| `S2P1_path_list.json` | Phase 1 | 代码路径清单 + 分支树 |
| `S2P1_tiling_glossary.md` | Phase 1 | tiling 变量含义表（tiling 源码变量名 → 语义名映射） |
| `S2P1_operator_model.json` | Phase 1 | 算子接口模型（inputs/outputs/attributes，以 `_def.cpp` 为权威约束源） |
| `S2P1_low_configs.json` | Phase 1 | 常见网络 shape 配置（语义参数名） |
| `S2P2_analysis_data.json` | Phase 2 | 紧凑格式分析数据（LLM 生成，assemble_dim_spec.py 的输入） |
| `S2P2_dim_spec.json` | Phase 2 | 维度范围规格（由 assemble_dim_spec.py 从 S2P2_analysis_data.json 生成） |
| `S2P2_param_def_groups.json` | Phase 2 | 推导数据文件（由 pick_dims.py 自动生成） |
| `S2P2_reachability_data.json` | Phase 2 | 可达性 + 分组中间数据（update_path_list.py 输入） |
| `S2P2_param_def.json` | Phase 2 | 参数定义 + 约束 + 分组 + tiling_keys（由 builder 自动生成） |
| `S2P2_gen_cases.py` | Phase 2 | 参数组合枚举脚本 |
| `S2P2_cases.json` | Phase 2 | 参数组合枚举结果 |
| `S2P2_traceability.md` | Phase 2 | 推导追溯文档 |
| `S2P3_test_design.md` | Phase 3 | 测试设计文档 |

**可选产物**（不阻塞门禁）：

| 文件 | Phase | 说明 |
|------|-------|------|
| `S2P1_task_a_todo.md` | Phase 1 | Task A 子 agent 进度跟踪（命名固定） |
| `S2P2_task_d_todo.md` | Phase 2 | Task D 子 agent 进度跟踪（命名固定） |

## 执行顺序约束（强制）

以下 Phase 必须按编号顺序逐步执行，禁止跳步或抢跑。

| Phase | 任务 | 执行方 | 参考文档 | 输入 | 前置条件 | 状态判断 |
|-------|------|--------|---------|------|---------|---------|
| Phase 0 | Scout-T：tiling 侦察 | 子 agent | `01-scout-tiling.md` | 算子路径、平台参数 | 算子路径 + 平台参数已就绪 | `S2P0_scout_t.md` 已生成 |
| Phase 0 | Scout-K：kernel 侦察 | 子 agent | `02-scout-kernel.md` | 算子路径、平台参数 | 同上 | `S2P0_scout_k.md` 已生成 |
| Phase 0 | Scout-Verify：校验 + 清单 | 子 agent | `03-scout-verify.md` | 算子路径、`S2P0_scout_t.md`、`S2P0_scout_k.md` 路径 | Scout-T + Scout-K 均完成 | `S2P0_file_manifest.json` 已生成，`verification.status` = `pass` / `pass_with_fixes` |
| Phase 0→1 | 提取读取范围文本块 | 主 Agent | 本文件（格式化模板） | `S2P0_file_manifest.json` | Phase 0 全部完成 | 文本块已格式化 |
| Phase 1 | Task A：代码路径分析 | 子 agent | `04-code-analyzer.md`，prompt 模板见本文件 §子 Agent Prompt 模板 → Task A | 算子路径、平台参数、源码读取范围文本块、`S2P0_scout_t.md`、`S2P0_scout_k.md` 路径、产出写入路径 | Phase 0→1 过渡完成 | `S2P1_path_list.json` + `S2P1_tiling_glossary.md` 已生成 |
| Phase 1 | Task B：接口分析 | 子 agent | `05-interface-analyzer.md`，prompt 模板见本文件 §子 Agent Prompt 模板 → Task B | 算子路径、平台参数、产出写入路径 | 同上 | `S2P1_operator_model.json` 已生成 |
| Phase 1 | Task C：网络搜索 | 子 agent | `08-network-search.md`，prompt 模板见本文件 §子 Agent Prompt 模板 → Task C | 算子名称、算子路径、平台参数、产出写入路径 | 同上 | `S2P1_low_configs.json` 已生成 |
| Phase 2 | Task D：参数推导 | 子 agent | `06-param-derivation.md`，prompt 模板见本文件 §子 Agent Prompt 模板 → Task D | 算子路径、平台参数、源码读取范围文本块、`S2P1_path_list.json`、`S2P1_operator_model.json`、`S2P1_low_configs.json` 路径、产出写入路径 | Phase 1 全部完成 | `S2P2_analysis_data.json` 已写入，assemble_dim_spec.py 已运行生成 `S2P2_dim_spec.json`，pick_dims.py 已生成 `S2P2_param_def_groups.json` 并格式化，build_param_def.py 已运行产出 `S2P2_param_def.json`，`S2P2_gen_cases.py` + `S2P2_cases.json` + `S2P2_traceability.md` 已生成 |
| Phase 3a | 处理 disputed 路径 | 主 Agent | — | Task D 返回的 disputed 列表 | Phase 2 完成 | `S2P1_path_list.json` 已更新（reachability + group） |
| Phase 3b | 生成测试设计文档 | 主 Agent | `07-test-design-template.md` | `S2P2_param_def.json`、`S2P1_low_configs.json`、Phase 3a 确认结果 | Phase 3a 完成 | `S2P3_test_design.md` 已生成 |

**并行规则**：
- Phase 0：Scout-T 和 Scout-K **并行**派发，两者完成后派 Scout-Verify
- Phase 1：Task A、B、C **并行**派发（无依赖）
- Phase 2：Task D **串行**（依赖 Phase 1 全部完成）
- Phase 3：3a 和 3b **串行**（3b 依赖 3a 完成）

**校验失败处理**：

- Scout-Verify `verification.status` = `fail` → 重跑失败的 Scout（最多 1 次），仍 fail → 触发轮次耗尽协议
- Phase 0→1 过渡：`pass` / `pass_with_fixes` → 进入 Phase 1

**完成标志**：`S2P3_test_design.md` 已生成，Phase 3a 的 disputed 路径已由用户确认（无 disputed 则自动满足）。

## 子 Agent Prompt 模板

主 Agent 派发 Task A/B/C/D 子 agent 时，Read 本节对应模板，替换占位符后作为 prompt 发送。**禁止自行改写模板结构或措辞**，仅允许替换 `{占位符}`。

### 占位符定义

| 占位符 | 含义 | 填入来源 |
|--------|------|---------|
| `{op_name}` | 算子名称 | Step 1.1 |
| `{op_path}` | 算子源码路径 | Step 1.2 |
| `{npu_arch}` | NPU 架构 | Step 1.0 |
| `{soc_version}` | SOC 版本 | Step 1.0 |
| `{chip_model}` | 芯片型号 | Step 1.0 |
| `{core_count}` | VectorCore 核数 | Step 1.3 |
| `{ub_size}` | UB 大小(KB) | Step 1.3 |
| `{output_dir}` | 产物写入路径 | `{op_path}/tests/whitebox/` |
| `{skill_base}` | 技能根目录绝对路径 | skill 自身路径 |
| `{file_manifest_text}` | 源码读取范围文本块 | Phase 0→1 过渡提取 |
| `{scout_t_path}` | S2P0_scout_t.md 路径 | Phase 0 产出 |
| `{scout_k_path}` | S2P0_scout_k.md 路径 | Phase 0 产出 |
| `{path_list_path}` | S2P1_path_list.json 路径 | Phase 1 产出 |
| `{operator_model_path}` | S2P1_operator_model.json 路径 | Phase 1 产出 |
| `{low_configs_path}` | S2P1_low_configs.json 路径 | Phase 1 产出 |

---

### Task A：代码路径分析

```
你是代码路径分析专家。请执行 Phase 1 Task A：代码路径分析（tiling + kernel）。

**优先执行入口文件顶部的执行顺序约束节**。

请先 Read 参考文档 `{skill_base}/references/source-analysis/04-code-analyzer.md`，然后按照文档中的规则逐步执行代码路径分析。

上下文参数：
- 算子名称：{op_name}
- 算子路径：{op_path}
- platform：npu_arch={npu_arch}, soc_version={soc_version}, chip_model={chip_model}, core_count={core_count}, ub_size={ub_size}
- skill_base：{skill_base}
- 产出写入路径：{output_dir}
- S2P0_scout_t.md 路径：{scout_t_path}
- S2P0_scout_k.md 路径：{scout_k_path}

{file_manifest_text}

请生成 `S2P1_path_list.json` 和 `S2P1_tiling_glossary.md` 到产出路径。完成后返回产出的文件路径和关键发现摘要（路径总数、分组数、disputed 路径列表）。
```

---

### Task B：接口分析

```
你是接口分析专家。请执行 Phase 1 Task B：接口分析。

**优先执行入口文件顶部的执行顺序约束节**。

请先 Read 参考文档 `{skill_base}/references/source-analysis/05-interface-analyzer.md`，然后按照文档中的规则逐步执行接口分析。

上下文参数：
- 算子名称：{op_name}
- 算子路径：{op_path}
- platform：npu_arch={npu_arch}, soc_version={soc_version}, chip_model={chip_model}, core_count={core_count}, ub_size={ub_size}
- skill_base：{skill_base}
- 产出写入路径：{output_dir}

请生成 `S2P1_operator_model.json` 到产出路径。完成后返回产出的文件路径和接口模型摘要（inputs/outputs/attributes 列表）。
```

---

### Task C：网络搜索

```
你是网络搜索专家。请执行 Phase 1 Task C：常见网络 shape 配置搜索。

**优先执行入口文件顶部的执行顺序约束节**。

请先 Read 参考文档 `{skill_base}/references/source-analysis/08-network-search.md`，然后按照文档中的规则逐步执行网络搜索。

上下文参数：
- 算子名称：{op_name}
- 算子路径：{op_path}
- platform：npu_arch={npu_arch}, soc_version={soc_version}, chip_model={chip_model}, core_count={core_count}, ub_size={ub_size}
- skill_base：{skill_base}
- 产出写入路径：{output_dir}

请生成 `S2P1_low_configs.json` 到产出路径。完成后返回产出的文件路径和搜索到的典型配置数量。
```

---

### Task D：参数推导

```
你是参数推导工程师。请执行 Phase 2 Task D：路径枚举 + 参数推导。

**优先执行入口文件顶部的执行顺序约束节**。

请先 Read 参考文档 `{skill_base}/references/source-analysis/06-param-derivation.md`，然后按照文档中的规则逐步执行参数推导。

上下文参数：
- 算子名称：{op_name}
- 算子路径：{op_path}
- platform：npu_arch={npu_arch}, soc_version={soc_version}, chip_model={chip_model}, core_count={core_count}, ub_size={ub_size}
- skill_base：{skill_base}
- 产出写入路径：{output_dir}
- S2P1_path_list.json 路径：{path_list_path}
- S2P1_operator_model.json 路径：{operator_model_path}
- S2P1_low_configs.json 路径：{low_configs_path}

{file_manifest_text}

完成后返回产出的文件路径、参数分组数量和关键发现摘要。
```

---

## Phase 3a：disputed 路径处理规则

Task D 返回的 disputed 列表（路径可达性不确定的条目）由主 Agent 处理：

1. 向用户提问（一次性问完所有 disputed 项）
2. 选项：全部接受建议 / 逐条确认
3. 用户确认后，主 Agent 更新 `S2P1_path_list.json`：
   - accepted → reachable
   - excluded → dead 并记录原因
4. 无 disputed 则跳过此步骤

## 文件索引

| 文件 | 职责 | 读入时机 |
|------|------|---------|
| `01-scout-tiling.md` | tiling 入口定位、平台分支可达性标注 | Phase 0 Scout-T |
| `02-scout-kernel.md` | kernel dispatch 定位、key 统计、平台预过滤 | Phase 0 Scout-K |
| `03-scout-verify.md` | Scout 报告校验、文件清单生成 | Phase 0 Scout-Verify |
| `04-code-analyzer.md` | 代码路径提取、约束分析、分支树构建 | Phase 1 Task A |
| `05-interface-analyzer.md` | 接口签名分析、参数约束、API 暴露 | Phase 1 Task B |
| `06-param-derivation.md` | 参数推导、组合枚举、分组、tiling_keys | Phase 2 Task D |
| `07-test-design-template.md` | 测试设计文档模板、group 汇总 | Phase 3b |
| `08-network-search.md` | aclnn 文档参数类型提取、网络 shape 搜索、TensorList 约束保障 | Phase 1 Task C |
