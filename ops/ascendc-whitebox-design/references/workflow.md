# White-box Pytest Test Generation Workflow

> **通用约束：子 agent 规则自读原则（强制，全流程适用）**
> 1. 主 Agent 派发子 agent 时，必须在 prompt 中指示：
>    a. "先 Read 完整 prompt 文件 {path}"
>    b. "**优先执行「执行顺序（最高优先级）」节**（文件顶部，blockquote 标记）。该节是你的主执行路线，包含每步的编号、产出和前置条件。详细规则见 prompt 文件后续章节。"
> 2. **结构化数据例外**：从文件中确定性提取的元信息（因源文件过大、子 agent 只需极少部分），允许按固定格式模板内联传入，但必须显式声明提取来源、范围和拼接格式。
> 3. **按需载入参考文档（强制）**：prompt 文件包含分步参考文档时，子 agent 必须按执行步骤顺序逐步 Read，到达步骤 N 时才 Read 该步骤标注的参考文档。**禁止一次性读取全部参考文档。**
>
> **JSON 完整性校验（强制，全流程适用）**
> 主 Agent 在读取任何子 agent 产出的 JSON 文件后，必须立即用 `python3 -c "import json; json.load(open('{filepath}'))"` 验证 JSON 可解析。解析失败 → 立即要求对应子 agent 重新生成（重试最多 1 次）。二次失败 → 触发通用轮次耗尽协议。

> **轮次耗尽协议（通用，全流程适用）**
> 任何步骤/验证/重试达到最大轮次后仍未通过 → 主 Agent 向用户报告：
>
> ⚠️ {步骤名} 经过 {轮次} 轮仍未通过。剩余问题：
> {逐条列出}
>
> 选项：
> 1. 强制继续 — {步骤特定的回退描述}
> 2. 终止 — 停止流程，保留当前产物供人工处理
> 3. 手动修正 — 由用户指示修改方向后重试（额外 1 轮）
>
> 各步骤在自身描述中注明最大轮次和选项 1 的回退描述。选项 2/3 为统一行为，无需重复定义。

---

## Step 1：输入收集

> **Step 1 全局约束（强制）**
>
> **顺序**：1.0 → 1.1 → 1.2 → 1.3 → 1.4，严格逐步执行，禁止合并、跳过或重排。
>
> **禁止抢跑**：每步仅允许其自身描述的操作（如 1.0 仅允许平台检测，1.1 才能用 question，1.2 才能搜索路径），前一步未完成不得启动下一步。
>
> **question 工具规范（Step 1.1 专用）**：
> - 单次调用，所有未收集参数作为 `questions` 数组的独立元素逐项呈现（独立 header、独立 question、独立 options）
> - 禁止创建"全部使用默认值"或"自定义参数"等聚合选项
> - 已从用户消息或前置步骤获取的值，以文字回显，不再放入 questions 数组

### 1.0 平台检测

加载 `npu-arch` skill，自动检测当前 NPU 平台的 NpuArch 与 SocVersion，确定代表芯片型号。**禁止猜测和推导**——必须以硬件检测结果为准。

检测方式（按优先级降级）：
1. `npu-smi info 2>&1 | sed -n '/^[| ]*[0-9]\+\s\+[0-9A-Za-z]/p' | head -1 | awk '{print $2, $3}'` 获取 NPU 卡号和芯片名称（输出格式：`NPU_ID CHIP_NAME`）
2. 若 npu-smi 不可用 → 读取环境变量 `$ASCEND_SOC_VERSION`
3. 两者均不可用 → 报告「无法检测 NPU 平台，请手动指定」，等待用户输入

> **注意**：禁止使用 `npu-smi info -t board -i 0`（硬编码卡号 0，且 `-t board` 返回产品代号而非芯片型号）。

将检测结果交给 `npu-arch` skill 解析，得到 `npu_arch`、`soc_version`、`chip_model` 等平台字段。

产出：`npu_arch`、`soc_version`、`chip_model`。

### 1.1 收集用户输入

在 Step 1.0 完成后，使用 question 工具收集以下 4 项输入（已从用户消息或前置步骤获取的值以文字回显，仅将未收集项作为 questions 数组中的独立条目）。

| # | 项 | 提示语 | 必填 | 默认值/可选值 |
|---|-----|-------|------|-------------|
| 1 | 算子名称 | "请输入算子名称" | 是 | 自由文本 |
| 2 | Step 4 闸门 | "是否跳过 Step 4 闸门？" | 是 | 默认不跳过；可选跳过（一次跑完全流程） |
| 3 | 算子路径 | "请选择算子路径获取方式：自动查找 / 手动输入路径" | 是 | 默认「自动查找」 |
| 4 | TTK CSV 模块 | "是否启用 TTK CSV 模块？" | 是 | 默认不启用 |

### 1.2 定位算子路径

在 Step 1.1 完成后执行。由用户对「算子路径」的选择驱动：
- 选择「自动查找」→ 执行以下搜索逻辑，定位算子路径。
- 选择「手动输入」→ 仅验证用户提供的路径（确认 op_kernel/ 和 op_host/ 目录存在），验证通过则跳过搜索，直接进入 Step 1.3。

**「自动查找」分支**：

> **查找方式（强制）**：限定**项目根目录**，禁止跨项目。使用 bash `find {项目根目录} -maxdepth 4 -type d -path "*/ops-*/*/{op_name}"` 按目录名精确匹配（禁止子串匹配）。未找到→直接报告"未找到"，不得扩大搜索范围。禁止派发子 agent。

主 Agent 验证每个候选目录同时存在 op_kernel/ 和 op_host/，过滤不合格项。唯一结果→使用；多项→让用户选择。

**「手动输入」分支**：

验证用户提供的路径下存在 `op_kernel/` 和 `op_host/` 目录。验证失败→报错并要求用户修正路径。

### 1.3 查询技术参数

在 Step 1.2 完成后执行。读取 `npu-arch` skill 维护的平台资料，将 Step 1.0 的检测结果解析为完整 `platform` 对象并传递给子 agent。白盒 workflow 不维护平台映射表。

`platform` 至少包含：
- `npu_arch`：NpuArch 枚举名
- `soc_version`：SocVersion 名称
- `chip_model`：芯片型号
- `npu_arch_macro`：`__NPU_ARCH__` 数值
- `arch_dir`：算子目录中的 archXX 简写
- `core_count`：当前平台可用核数
- `ub_size`：当前平台 UB 容量
- `capabilities`：`npu-arch` 已确认的平台能力标签（如 Regbase 等）
- `source`：参数来源（硬件检测、环境变量、用户输入或 npu-arch 文档）

若 `npu-arch` 无法确认 `npu_arch_macro`、`arch_dir`、`core_count` 或 `ub_size`，向用户说明缺失项并补齐后再进入 1.4。禁止在白盒 workflow 中按芯片名硬编码推导。

### 1.4 确认摘要

在 Step 1.0-1.3 全部完成后执行。展示确认摘要，硬件参数（自动检测）与用户输入分隔显示：

```
── 硬件参数（自动检测）──
目标平台 {chip_model}（{npu_arch}），将使用 {core_count} 核、{ub_size}KB UB。
── 用户输入 ──
- 算子名称：{op_name}
- 算子路径：{op_path}
- Step 4 闸门：{gate_status}
- TTK CSV 模块：{ttk_status}

结果输出到 {算子源码路径}/tests/whitebox/。
如需修改核数、UB 大小，或有额外特殊条件需添加，请告知。
确认后开始分析。
```

占位符填入规则：
- `{chip_model}`/`{npu_arch}`/`{core_count}`/`{ub_size}` — 来自 Step 1.3 生成的 `platform` 对象
- `{op_name}`/`{op_path}`/`{gate_status}`/`{ttk_status}` — 由 Step 1.1 用户输入
- `{算子源码路径}` — 即 Step 1.2 确定的算子路径

展示后等待用户确认：

| 选项 | 说明 |
|------|------|
| 确认，开始分析 | 进入 Step 2 |
| 需要修改 | 核数/UB — 主 Agent 更新 `platform` 对象后回到 1.4 重新展示；其它参数 — 回到 1.1 重新收集 |

---

## Step 2：分析源码

> **前置条件**：Step 1.4 用户已确认摘要（"确认，开始分析"）；算子路径、平台参数均已确定。

> **源码读取范围规则（强制）**：`S2P0_file_manifest.json` 体积较大，子 agent 只需文件清单和读取范围。主 Agent 在 Phase 0→1 过渡阶段提取 `tiling.file_list` + `kernel.file_list` + `tiling.excluded` + `kernel.excluded`，按固定格式拼接为文本块后内联传入 Task A 和 Task D。子 Agent 严格按此范围读取，不得自行添加。

### Phase 0（文件侦察）— 并行

> **前置条件**：Step 2 已开始；算子路径和平台参数已就绪。

1. **并行派 2 个子 agent**：
   - **Scout-T**：Read `references/prompts/S2P0-scout-tiling.md`，Glob/Grep 定位 tiling 入口文件（P0），提取 #include 和外部函数引用，代入目标平台求值平台判断分支，标注分支可达性。产出 `S2P0_scout_t.md`。
   - **Scout-K**：Read `references/prompts/S2P0-scout-kernel.md`，Grep TILING_KEY_IS 定位 kernel dispatch 文件（P0），根据 npu_arch 预过滤平台不相关文件，识别 6 种 dispatch 模式并统计 key 数量。产出 `S2P0_scout_k.md`。

2. **等两个 Scout 完成后，派 1 个子 agent**：
   - **Scout-Verify**：Read `references/prompts/S2P0-scout-verify.md`，先 Read Scout 报告，执行 Phase A（独立 Grep 基准扫描）→ Phase B（交叉比对）→ Phase C（沿路径 Read 验证）→ Phase D（汇总生成）。产出 `S2P0_file_manifest.json`。

### Phase 0→1 过渡（主 Agent 执行）

> **前置条件**：Scout-T 和 Scout-K 均已完成，`S2P0_scout_t.md` 和 `S2P0_scout_k.md` 已生成；Scout-Verify 已完成，`S2P0_file_manifest.json` 已生成且 `verification.status` 为 `pass` 或 `pass_with_fixes`。

1. Read `S2P0_file_manifest.json`
2. 遍历 `tiling.file_list` + `kernel.file_list`，提取每个文件的 path/priority/read_strategy/symbols（tiling）或 pattern/key_count（kernel）；遍历 `tiling.excluded` + `kernel.excluded`

3. 格式化为以下文本块（作为 Task A / Task D prompt 的输入数据之一传递）：

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

**校验失败处理**：检查 `verification.status`，`pass`/`pass_with_fixes` → 进入 Phase 1；`fail` → 重跑失败的 Scout（最多 1 次），仍 fail → 触发通用轮次耗尽协议。

---

### Phase 1 — 并行派 3 个子 agent（A/B/C 无依赖，必须并行）

> **前置条件**：Phase 0→1 过渡已完成，源码读取范围文本块已格式化。

- **Task A（代码路径分析）**：`prompt` 含 2 部分：
  1. `先 Read 完整 prompt 文件 {skill_base}/references/prompts/S2P1-code-analyzer.md，然后严格按其 spec 执行。禁止一次性读取全部参考文档。`
  2. 输入数据：算子路径、平台参数、源码读取范围文本块、`S2P0_scout_t.md` 路径、`S2P0_scout_k.md` 路径、产出写入路径
  → 产出 `S2P1_path_list.json`（路径清单 + 源码约束表 + completeness_checklist）
  → 可选产出 `S2P1_task_a_todo.md`（子 agent 进度跟踪，命名固定）

- **Task B（接口分析）**：`prompt` 含 2 部分：
  1. `先 Read 完整 prompt 文件 {skill_base}/references/prompts/S2P1-interface-analyzer.md，然后严格按其 spec 执行`
  2. 输入数据：算子路径、平台参数、产出写入路径
  → 产出 `S2P1_operator_model.json`（接口签名/参数约束/平台限制/torch_npu_api_exposure）

- **Task C（网络搜索）**：无专用 prompt 文件。指示子 agent Read 本文件此 Task C 段落获取规则。搜索该算子常见网络 shape → 产出 `S2P1_low_configs.json`。使用语义参数名（如 `dim`、`dtype`），Step 5b 通过 `shape_mapping.role` 自动映射为算子参数名。搜不到则产出空数组 `[]`。格式：

```json
[
  {"config": {"dim": 4096, "dtype": "float16", ...}, "source": "来源URL或 null", "reason": "推断理由或 null"}
]
```

### Phase 2（串行）— Phase 1 全部完成后

> **前置条件**：Phase 1 的 Task A、B、C 全部完成；`S2P1_path_list.json`、`S2P1_operator_model.json`、`S2P1_low_configs.json` 均已生成。

- **Task D（路径分析）**：`prompt` 含 2 部分：
   1. `先 Read 完整 prompt 文件 {skill_base}/references/prompts/S2P2-param-derivation.md，然后严格按其 spec 执行。禁止一次性读取全部参考文档。`
    2. 输入数据：算子路径、平台参数、源码读取范围文本块、`S2P1_path_list.json` 路径、`S2P1_operator_model.json` 路径、`S2P1_low_configs.json` 路径、产出写入路径
   → 可选产出 `S2P2_task_d_todo.md`（子 agent 进度跟踪，命名固定）

### Phase 3（串行，主 Agent 执行）— Phase 2 全部完成后

> **前置条件**：Phase 2 的 Task D 已完成；`S2P2_param_def.json`、`S2P2_gen_cases.py`、`S2P2_cases.json`、`S2P2_traceability.md` 已生成，`S2P1_path_list.json` 已更新（含 reachability 和 group 字段）。

**Phase 3a：处理 disputed 路径**
Task D 返回的 disputed 列表向用户提问（一次性问完）。选项：全部接受建议 / 逐条确认。用户确认后主 Agent 更新 `S2P1_path_list.json`（accepted→reachable, excluded→dead 并记录原因）。无 disputed 则跳过。

**Phase 3b：生成 S2P3_test_design.md**
主 Agent 读取 `references/prompts/S2P3-test-design-template.md`，以 `S2P2_param_def.json` + low_configs + Phase 3a 确认结果为输入，生成 `S2P3_test_design.md`。必须在 S2P2_param_def.json 和 disputed 确认之后生成。

**Phase 3c：生成 shape_mapping**
派 1 个独立子 agent。`prompt` 含 2 部分：
1. `先 Read 完整 prompt 文件 {skill_base}/references/prompts/S2P3-shape-mapping.md，然后严格按其 spec 执行`
2. 输入数据：`S2P1_operator_model.json` 路径、`S2P2_param_def.json` 路径、`S2P1_path_list.json` 路径（可选）、算子路径、产出写入路径
→ Task E 生成 `shape_mapping` 节并写回 `S2P1_operator_model.json`。

---

## Step 3：交叉验证

> **前置条件**：Step 2 全部 Phase（0/1/2/3）已完成；`S2P3_test_design.md` 已生成；Phase 3a 的 disputed 路径已由用户确认（无 disputed 则自动满足）。

派 1 个独立子 agent（不复用 Step 2 agent，确保独立视角）。主 Agent prompt：指示 Read `{skill_base}/references/prompts/S3-test-design-checker.md`，传入上下文参数：`S2P2_param_def.json` / `S2P3_test_design.md` / `S2P1_path_list.json` 路径、算子路径、产出写入路径。

输出：`S3_verification_report.json`。fail 项 → 回 Step 2 Phase 1 重分析（Phase 0 不需重跑，最多 3 轮），仍 fail → 触发通用轮次耗尽协议。warn 项 → 逐条判断是否修正，结论写入 S2P3_test_design.md 验证结论节；pass 项 → 无需处理。

---

## Step 4：用户确认

> **前置条件**：Step 3 已完成，`S3_verification_report.json` 已生成；验证结论已更新到 `S2P3_test_design.md`。

将验证结论更新到 S2P3_test_design.md 后停下来等待用户确认。提示语："源码分析和交叉验证已完成：{N} 个测试 group，预估 ~{M} 个参数组合，验证状态 {status}。"

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

> **前置条件**：Step 4 用户已确认；`S2P2_cases.json` 已产出；`S2P1_low_configs.json` 已产出。

派 1 个独立子 agent。主 Agent prompt：

先 Read `{skill_base}/references/prompts/S5-case-mapper.md`，然后按顺序执行：

**5a**：以 `S2P2_cases.json` + `S2P1_operator_model.json` 为输入，生成 `S5_case_mapper.py` + `S5_verify_mapper.py`，运行产出 `S5_mapped_cases_path.json`（路径覆盖）。验证失败→修复（最多 3 轮），仍 fail → 触发通用轮次耗尽协议。

**5b**：读取 `S2P1_low_configs.json` / `S2P2_param_def.json` / `S2P1_operator_model.json`，通过 `shape_mapping.shape_params.role` 将语义参数名映射为算子参数名（`leading_product` 同 role 多值相乘），复用 mapper 的 `load_mapped_configs` 产出 `S5_mapped_cases_network.json`。

**5c**：合并 + 空 tensor 补全——
1. 过滤路径覆盖 case：剔除任意输入 tensor 元素数 > 1 亿的 case
2. 读取 `S5_mapped_cases_path.json` + `S5_mapped_cases_network.json`
3. 合并路径 + 网络 + 空 tensor 补全 → `S5_mapped_cases_low.json`（所有输入全 normal）

**shape_mapping 异常回退**：mapper 因缺少 shape_mapping 崩溃 → 诊断缺失字段 → 重新派发 Task E（Phase 3c）修正 → 重跑 Step 5（最多 2 轮回退），仍 fail → 触发通用轮次耗尽协议。

**5d：data_range 维度展开**

> **前置条件**：5c 合并已完成，`S5_mapped_cases_low.json` 已生成（路径+网络已合并）。

按 `S5-case-mapper.md` 的「data_range 维度展开」节（§7）拆分 low / high 两个档位：

1. **low**：全路径 case + 全网络 case，所有输入置 normal → `S5_mapped_cases_low.json`
2. **high**：全路径 case + 全网络 case，one-hot（N_inputs × 8）+ 全统一（9 种，含 normal）→ `S5_mapped_cases_high.json`

两个数据源均可通过 `--cases-file` 执行；调用方按目标场景选择需要执行的数据源。

---

## Step 6：生成 pytest 并执行

> **前置条件**：Step 5 已完成，`S5_mapped_cases_high.json` / `S5_mapped_cases_low.json` 已生成。

主 Agent 直接执行。输入：`S5_mapped_cases_high.json` / `S5_mapped_cases_low.json` + `S2P2_param_def.json`。Reference 实现的编写依据：算子 `docs/aclnn*.md` 文件中的「计算公式」节（只读该节，不读其他节）。

### Step 6a：生成 S6_test_{op_name}.py

Read `references/prompts/S6-pytest-generator.md` 按要求生成。语法检查 + 收集检查（全部通过才算完成）：
1. `python -m py_compile S6_test_{op_name}.py`
2. `pytest --collect-only S6_test_{op_name}.py --cases-file=S5_mapped_cases_high.json`

### Step 6b：执行 pytest

执行 pytest（同时产出 plog 用于 6c tilingkey 覆盖率）：
```bash
ASCEND_GLOBAL_LOG_LEVEL=1 pytest S6_test_{op_name}.py --cases-file=S5_mapped_cases_high.json -q --tb=line
```

> `--cases-file` 可切换 `S5_mapped_cases_low.json` / `S5_mapped_cases_high.json`；本步骤默认示例使用 high 数据源。

执行检查规则：

| 检查项 | 通过标准 | 不通过处理 |
|--------|---------|----------|
| FAILED | 0 个 | assertion 失败用 try/except + pytest.xfail 转为 XFAIL；RuntimeError/TypeError 等修复代码 |
| XFAIL | 允许存在 | 精度不达标合法结果，保留偏差信息 |
| ERROR | 0 个 | 修复 import/API 名称/代码错误 |
| RuntimeError | 0 个 | 修复 API 签名/shape 构造/tiling 约束 |
| AttributeError | 0 个 | 修复 API 名称/模块属性访问 |
| 假 PASS | 0 个 | NPU 不可用必须 `pytest.skip()` |

- 合法结果：PASSED / XFAIL / SKIPPED
- 精度标准（rtol/atol）禁止修改
- 修复后重新运行（最多 3 轮），仍不通过 → 触发通用轮次耗尽协议

### Step 6c：tilingkey 后处理

> 默认针对 high 数据源（`S5_mapped_cases_high.json`）生成 tilingkey 覆盖率。

> **plog 来源**：来自 6b 的 pytest 执行。data_range 不影响 tiling 路径选择，同一 key 在 plog 中出现次数与覆盖率无关（`set` 去重后结果一致）。

```bash
mkdir -p tests/whitebox/tilingkey_logs/
PLOG=$(ls -t ~/ascend/log/debug/plog/plog-*.log | head -1)
cp "$PLOG" tests/whitebox/tilingkey_logs/{op_name}_full.log
python {skill_scripts}/compute_tilingkey_coverage.py \
  --log-path tests/whitebox/tilingkey_logs/{op_name}_full.log \
  --param-def tests/whitebox/S2P2_param_def.json \
  --output-dir tests/whitebox/
```

输出 `S6_tilingkey_coverage.json`。无 plog 或 `tiling_keys` 字段时打印提示并继续。

### 路径B：单用例 tilingkey 调试（按需，非必须）

```bash
python {skill_scripts}/tilingkey_single.py --op-path {op_path} --case-id {case_id}
```

---

## 可选模块：TTK CSV 生成

> **注意**：当前 TTK 模块仅支持 `kernel` 模式（`ttk kernel`），不支持 `e2e` 和 `aclnn` 模式。执行验收时必须使用 `python3 -m ttk kernel` 命令。

### 启用条件

Step 1 输入 4 选择了「启用」。若未启用，跳过本模块。

### 调用方式

**方式 A（自动）**：主流程 Step 6 执行检查全部通过后自动触发。

> **前置条件**：Step 6 执行检查全部通过（0 FAILED / 0 ERROR / 0 RuntimeError）。

**方式 B（独立）**：用户可随时派发子 agent 单独生成，只需提供 `S5_mapped_cases_low.json` / `S5_mapped_cases_high.json` + `S2P1_operator_model.json` 路径和算子路径，不依赖其他 Step 产物或上下文。

### 执行

派 1 个独立子 agent。主 Agent prompt：指示 Read `{skill_base}/references/ttk-converter.md`，传入上下文参数：`S5_mapped_cases_low.json` / `S5_mapped_cases_high.json` / `S2P1_operator_model.json` 路径、`op_name`、算子源码路径（`*_def.cpp` / `*_tiling_check.cpp` / `*_infershape.cpp`）、产出写入路径。

子 agent 返回后，主 Agent 确认校验结果，停下来等待用户确认后结束流程。

子 agent 在 ttk-converter.md 的任务 1-4 完成后，**必须继续执行任务 5（生成 `golden_plugin.py`）和任务 6 初步验证**。任务 6 的 TTK Kernel 执行验收由主 Agent 执行，作为模块最终验收。

### 产出（仅模块启用时）

```
{算子源码路径}/tests/whitebox/
├── ttk_extract_case_info.py
├── ttk_{op_name}_cases_low.csv
├── ttk_{op_name}_cases_full.csv
└── golden_plugin.py
```

---

## 最终产物

```
{算子源码路径}/tests/whitebox/
├── S2P0_scout_t.md
├── S2P0_scout_k.md
├── S2P0_file_manifest.json
├── S2P1_path_list.json
├── S2P1_low_configs.json
├── S2P1_operator_model.json
├── S2P2_param_def.json
├── S2P2_gen_cases.py
├── S2P2_cases.json
├── S2P3_test_design.md
├── S3_verification_report.json
├── S5_case_mapper.py
├── S5_verify_mapper.py
├── S5_mapped_cases_path.json
├── S5_mapped_cases_network.json
├── S5_mapped_cases_high.json
├── S5_mapped_cases_low.json
├── S6_test_{op_name}.py
├── S6_tilingkey_coverage.json
├── ttk_extract_case_info.py
├── ttk_{op_name}_cases_low.csv
├── ttk_{op_name}_cases_full.csv
├── golden_plugin.py
├── tilingkey_logs/
│   ├── {op_name}_full.log
│   └── {op_name}_{case_id}.log

TTK CSV 模块产出（仅模块启用时）见「可选模块：TTK CSV 生成」节的产出列表。
```

## 参考提示词索引

| Step | 提示词文件 | 执行方 | 执行顺序节 |
|------|-----------|--------|-----------|
| 2 Phase 0 | `S2P0-scout-tiling.md` | Scout-T (子 agent) | `执行顺序（最高优先级）` — 6 步 |
| 2 Phase 0 | `S2P0-scout-kernel.md` | Scout-K (子 agent) | `执行顺序（最高优先级）` — 6 步 |
| 2 Phase 0 | `S2P0-scout-verify.md` | Scout-Verify (子 agent) | `执行顺序（最高优先级）` — 5 步 |
| 2 Phase 1 | `S2P1-code-analyzer.md` | Task A (子 agent) | `执行顺序（最高优先级）` — 8 步 |
| 2 Phase 1 | `S2P1-interface-analyzer.md` | Task B (子 agent) | `执行顺序（最高优先级）` — 7 步 |
| 2 Phase 2 | `prompts/S2P2-param-derivation.md` | Task D (子 agent) | `执行顺序（最高优先级）` — 6 步（逐步按需读取） |
| 2 Phase 3 | `S2P3-test-design-template.md` | 主 Agent | — |
| 2 Phase 3c | `S2P3-shape-mapping.md` | Task E (子 agent) | `执行顺序（最高优先级）` — 10 步 |
| 3 | `S3-test-design-checker.md` | 子 agent | `执行顺序（最高优先级）` — 5 步 |
| 5 | `S5-case-mapper.md` | 子 agent | `执行顺序（最高优先级）` — 3 步 |
| 6 | `S6-pytest-generator.md` | 主 Agent | — |
| 6c | `scripts/compute_tilingkey_coverage.py` | 主 Agent | — |
| 6c 路径B | `scripts/tilingkey_single.py` | 主 Agent（按需） | — |
| TTK 模块 | `ttk-converter.md` | 子 agent（可选模块，由 Step 1 输入6 控制） | — |
