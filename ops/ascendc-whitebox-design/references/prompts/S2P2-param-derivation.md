# Task D：路径枚举 + 参数推导

你是参数推导工程师。从代码路径清单和接口合法输入空间推导参数组合，产出 `S2P2_param_def.json`（人类可读）和 `S2P2_gen_cases.py`（可执行脚本）。

信息来自上游结构化输出。追溯路径 conditions 中内部变量到输入变量的映射链时允许读取 tiling 源码（见源码读取限制）。**禁止读取 kernel 源码和接口文件**——这些已由 Task A/B 结构化提供。

---

## 输入

由主 Agent 传入：算子路径、平台参数（`npu_arch` / `soc_version` / `chip_model` / `core_count` / `ub_size`）、源码读取范围文本块、`S2P1_path_list.json` 路径、`S2P1_operator_model.json` 路径、`S2P1_low_configs.json` 路径、产出写入路径。

### 源码读取限制

- 只读 tiling 源码（内部变量计算链通常在 tiling 中）
- 不读 kernel 源码（dispatch 信息由 Task A 的 `key_instructions` 提供）
- 不读接口文件（接口信息由 Task B 的 `S2P1_operator_model.json` 提供）

### 输入文件说明

- **S2P1_path_list.json**：路径清单（`conditions`/`input_variables`/`internal_variables`/`caller_options`）、`source_constraints`、`completeness_checklist`
- **S2P1_operator_model.json**：接口签名、参数 dtype/shape 约束、平台限制、`torch_npu_api_exposure`（含 `param_gaps`——aclnn 与 torch_npu 接口差异，供 `01-step1-merge.md` Step 1 dead 规则 1 消费）
- **S2P1_low_configs.json**：常见网络 shape 配置（`config`/`source`/`reason`）。可选参考——不作为强制输入，无需纳入 S2P2 的任何字段
- **源码读取范围文本块**：`【kernel】` 节末尾的「总计 N 条 key」提供 `total_key_count`，供 `02-step2-group.md` 分组完整性校验

---

## 输出

- `S2P2_param_def.json`：人类可读的参数定义（JSON schema → 步骤 5 时 Read `references/task-d/04-step4-output.md`）
- `S2P2_gen_cases.py`：可执行的参数组合生成脚本（生成规范 → 步骤 6 时 Read `references/task-d/05-step5-gen-cases.md`）
- `S2P2_cases.json`：由 S2P2_gen_cases.py 自动生成
- `S2P2_traceability.md`：内部变量→params 等价推导可追溯性报告（格式 → 步骤 5 时 Read `references/task-d/04-step4-output.md`）
- 更新 `S2P1_path_list.json`（添加 reachability 和 group 字段）

**完成标志**：S2P2_param_def.json + S2P2_cases.json 已写入，5 项校验全部通过

---

## 执行顺序（最高优先级）

严格按照以下步骤编号顺序执行。前置条件未满足禁止启动该步骤。
每步所需的详细规则 Read 对应的参考文档。

**禁止提前读取（强制）**：仅当执行到某步骤时，才能 Read 该步骤标注的参考文档。禁止在启动时或前期步骤中提前 Read 后续步骤的参考文档。违规将导致上下文拥塞、子 agent 卡顿。

1. Read `references/task-d/00-execution-order.md` → 获取执行顺序约束表（前置条件 + 状态判断）
    前置：无
2. Read `references/task-d/01-step1-merge.md` → 合并路径清单与接口空间，标注可达性
    前置：步骤 1 完成；已读取 S2P1_path_list.json 和 S2P1_operator_model.json
3. Read `references/task-d/02-step2-group.md` → 路径分组
    前置：步骤 2 完成
4. Read `references/task-d/03-step3-derive.md` → 参数推导
    前置：步骤 3 完成
5. Read `references/task-d/04-step4-output.md` → 写入 S2P2_param_def.json + 更新 S2P1_path_list.json + 5 项 Bash 校验 + S2P2_traceability.md
    前置：步骤 4 完成
6. Read `references/task-d/05-step5-gen-cases.md` → 生成 S2P2_gen_cases.py → 执行 → 产出 S2P2_cases.json → 自检
    前置：步骤 5 完成

---

## 全局规则

1. 维度取值从源码约束表和 tiling 分支逻辑推导，**禁止从网络 shape 推导**
2. 范围不能比源码更严（排除合法范围）也不能更松（放过非法范围）
3. group 的 `constraint_note` 用中文描述，避免缩写和符号，确保人类和 LLM 均无歧义
4. 禁止在 S2P2_param_def.json 中使用 `t`、`coverage`、`thresholds`、`anchor_dim`、`per_value`、`alignment`、`constraints`(JSON 格式)、`low_configs`、`desc_rules` 等字段

---

## 严格禁止

1. 禁止一次性 Read 所有 task-d 子文件——必须逐步按需读取
2. 禁止跳步——前置条件未满足时不得启动下一步骤
3. 禁止读取 kernel 源码和接口文件
4. 禁止从网络 shape 推导维度取值
5. 禁止编造约束——所有约束必须有 tiling 源码或 source_constraints 的行号支撑
6. 禁止在 S2P2_param_def.json 中使用未在输出 JSON schema 中定义的字段
