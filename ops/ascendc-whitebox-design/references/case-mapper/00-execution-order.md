# Case Mapper 执行总纲

> **路径约定**：`{skill_base}` = 技能根目录绝对路径，由主 Agent 在构建 prompt 或执行流程时作为上下文参数传入。文档中的 `{skill_base}/references/...` 需替换为实际路径后再 Read。

## 角色

将 S2P2_cases.json 的抽象参数组合映射为具体的 tensor 构造配置（shape + dtype），产出 S5_case_mapper.py 脚本 + S5_verify_mapper.py 验证脚本 + S5_merge_expand.py 合并展开脚本 + JSON / md 输出文件，供下游 pytest 和 TTK 模块消费。

## 输入

| 文件 | 必须？ | 用途 |
|------|--------|------|
| `S2P2_cases.json` | 是 | path 用例的参数组合，mapper 运行时 `json.load` 读取 |
| `S2P1_operator_model.json` | 是 | 约束框架：inputs/outputs 的 dtype、rank、shape.constraints、attributes |
| `S2P2_param_def.json` | 是 | dtype 参数名（`dtype_tensors[0].param`）、shape 参数 key 名与 groups 结构 |
| `S2P1_low_configs.json` | 是 | 网络用例的语义参数值，网络映射阶段读取 |
| `S5_mapping_spec.md` | 是 | 5b 读算子侧规格（§dtype ~ §验证规则）+ `（DYNAMIC）` 标注识别 DYNAMIC 输入；5b 生成 §网络用例映射 并写回 |
| `S2P0_source_scope.md` | 是 | tiling 源码文件路径清单，5a-pre 据此读取源码提取参数与维度的对应关系 |
| `infershape.cpp` 等源码 | 辅助 | 理解输出 shape 推导表达式（derived.expr） |
| `06-dynamic-tensorlist.md` | 按需 | DYNAMIC param_type 时读取，用于 TensorList 约束处理 |

### S2P2_cases.json 读取限制

子 agent 在验证映射逻辑时，**禁止**全文读取 S2P2_cases.json。必须使用 Read 工具的 `offset=1, limit=30` 参数仅读取前几条用例作为验证样本。若样本用例的映射结果正确，即可认为映射逻辑验证通过，全部用例由生成的 S5_case_mapper.py 脚本运行时通过 `json.load` 批量处理。子 agent 可自行把握样本数量，以覆盖主要 group 为准。

## 信息来源优先级（强制）

约束信息（ndim range、tensor_constraints 等）**必须以 `S2P1_operator_model.json` 的 `inputs[*].rank` 和 `inputs[*].shape.constraints` 为唯一来源**，禁止从 tiling/infershape 源码重新推断。源码仅用于理解 shape 分解规则（decompose 策略）和输出 shape 推导表达式（derived.expr）。

## 产出物概览

### 可执行脚本

| 文件 | 功能 |
|------|------|
| `S5_case_mapper.py` | `map_case()` 由 S5_mapping_spec.md 散文翻译生成（imperative Python），另有通用工具函数 |
| `S5_verify_mapper.py` | 4 层自动验证（验证通过才能进入后续） |
| `S5_merge_expand.py` | 合并 + 空 tensor 补全 + 元素数过滤 + data_range 展开 |

## ID 命名规范

| 类型 | 格式 | 示例 |
|------|------|------|
| path | `case{序号:05d}` | `case00000` |
| network | `network{序号:05d}` | `network00001` |
| 空 tensor（REQUIRED） | `case{path 最大 ID + 1:05d}_{input_name}_empty` | `case00081_x_empty` |
| 空 tensor（DYNAMIC） | `case{path 最大 ID + 1 + idx:05d}_{name}_{suffix}` | `case00081_x_first_empty`（suffix: first/middle/last/partial_empty 或 all_empty，示例见 case00081_x_all_empty） |
| all 变体 | `{原 ID}_all_{range}` | `case00001_all_zero` |
| one-hot 变体 | `{原 ID}_{input_name}_{range}` | `case00001_{input_name}_{range}` |

## 执行顺序约束（强制）

以下 Steps 必须按编号顺序逐步骤执行，禁止跳步或抢跑。

### Step 5a-pre：映射规格生成
- **指导文件**：`{skill_base}/references/case-mapper/01-mapping-spec.md`
- **前置条件**：S2P2_cases.json + S2P1_operator_model.json + S2P2_param_def.json + S2P0_source_scope.md 已就绪
- **职责**：分析 operator_model 中各 input/output 的 param_type，生成 S5_mapping_spec.md（§dtype ~ §验证规则，不含 §网络用例映射）
- **完成标志**：S5_mapping_spec.md 已写入

### Step 5a：生成 mapper + verifier
- **指导文件**：`{skill_base}/references/case-mapper/02-step5a-mapper.md`
- **前置条件**：Step 5a-pre 完成
- **职责**：根据 S5_mapping_spec.md 生成 S5_case_mapper.py 和 S5_verify_mapper.py；运行验证确保映射正确
- **完成标志**：
  - S5_case_mapper.py + S5_verify_mapper.py 已生成
  - 运行退出码 0
  - S5_mapped_cases_path.json 已写入
  - case 数量 == S2P2_cases.json
  - 0 validation errors

### Step 5b：映射网络用例
- **指导文件**：`{skill_base}/references/case-mapper/03-step5b-network.md`
- **前置条件**：Step 5a 完成
- **职责**：读 S5_mapping_spec.md（算子侧）+ S2P1_low_configs.json（网络侧），生成映射规则写回 S5_mapping_spec.md §网络用例映射；生成 S2P2_network_cases.json；复用 5a mapper 生成 S5_mapped_cases_network.json
- **完成标志**：
  - S2P2_network_cases.json 已写入（字段名全部为算子参数名，_group="network"）
  - S5_mapped_cases_network.json 已写入
  - 网络用例 tensor 构造合法

### Step 5c：合并 + 空 tensor 补全 + data_range 展开
- **指导文件**：`{skill_base}/references/case-mapper/04-step5c-merge-expand.md`
- **前置条件**：Step 5b 完成
- **职责**：合并 path + network 用例；补全空 tensor 变体；生成 low 档位（全 normal）和 high 档位（data_range 展开）
- **完成标志**：
  - S5_merge_expand.py 已生成
  - `python S5_merge_expand.py` 退出码 0
  - `python S5_merge_expand.py 5d` 退出码 0
  - S5_mapped_cases_low.json + S5_mapped_cases_high.json 已写入
  - 空 tensor 变体已补全

### 通用规则
- 前置条件未全部满足时，禁止启动该 Step
- 完成当前 Step 并确认完成标志满足后，才能进入下一 Step
- 自检失败 → 回到对应 Step 修正，修正完成后方可继续
