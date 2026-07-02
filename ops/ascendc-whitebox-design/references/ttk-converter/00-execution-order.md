# TTK Converter 执行总纲

> **路径约定**：`{skill_base}` = 技能根目录绝对路径，由主 Agent 在构建 prompt 或执行流程时作为上下文参数传入。文档中的 `{skill_base}/references/...` 需替换为实际路径后再 Read。
> - `{whitebox_dir}` = 白盒测试产出目录（通常为 `{算子路径}/tests/whitebox/`）
> - `{ops_test_kit_path}` = TTK 工具目录（`ops-test-kit/`）
> - `{plugin_path}` = golden plugin 路径（任务 5 确定：已有 golden → `{算子路径}/tests/assets/golden.py`，自生成 → `{whitebox_dir}/golden_plugin.py`）

## 角色

信息提取与格式转换器。从 `S5_mapped_cases_low.json` / `S5_mapped_cases_high.json`（Step 5 产出的已映射 tensor 配置）中直接提取字段，转换为 TTK CSV 格式。

**TTK 工具**：`ops-test-kit/` 目录是 TTK 调试工具的代码仓库，提供 kernel/aclnn/e2e 三种模式的编译、执行、精度比对能力。所有 `python3 -m ttk` 命令**必须在 `ops-test-kit/` 目录下执行**（TTK 通过 `__main__.py` 启动）。

**命名规则**：输出文件带 `ttk_` 前缀（`ttk_extract_case_info.py`、`ttk_{op_name}_cases_low.csv`、`ttk_{op_name}_cases_full.csv`），例外：`golden_plugin.py` 为**自生成时**的固定文件名（已有 `tests/assets/golden.py` 时不生成）。

## 模式支持

TTK 工具支持三种模式，基于 CSV 表头自动识别：

| 模式 | 识别条件 | 使用结构 | 当前状态 |
|------|---------|---------|---------|
| Kernel | 表头不含 `api_name` | `UniversalTestcaseStructure` | **已实现** |
| ACLNN | 表头含 `api_name` 且值以 `aclnn` 开头 | `ApiTestcaseStructure` | 预留（见 ACLNN/E2E 预留节） |
| E2E | 表头含 `api_name` 且值不以 `aclnn` 开头 | `FrameworkApiTestcaseStructure` | 预留（见 ACLNN/E2E 预留节） |

> **当前仅 Kernel 模式已实现。** 新增 ACLNN/E2E 时，新增 `0X-{mode}-fields.md` + `0X-{mode}-extraction.md` + `0X-{mode}-tasks.md` 三个文件，无需改动现有 Kernel 文件。

## 输入

| 文件 | 必须？ | 用途 |
|------|--------|------|
| `S5_mapped_cases_low.json` | 是 | Step 5 低档位输出（路径+网络+空 tensor，全 normal，门禁用） |
| `S5_mapped_cases_high.json` | 是 | Step 5 高档位输出（data_range 展开，信息性验证） |
| `S2P1_operator_model.json` | 是 | 算子模型（提取 `attributes` 节中的属性名列表，用于过滤 `attributes`） |
| `S5_mapping_spec.md` | 是 | 提取属性名列表 + 识别各 input/output 的 param_type（REQUIRED/DYNAMIC），用于派发提取逻辑 |
| `op_name` | 是 | 算子名称（小写字母+下划线，由主 Agent 在调用时提供） |
| `scripts/ttk_validate_csv.py` | 辅助 | CSV 格式校验脚本（任务 4 使用），由 skill scripts 目录提供 |
| 算子源码（校验用） | 辅助 | `*_def.cpp`（输入/输出/属性注册）、`*_tiling_check.cpp`/`*_tiling*.cpp`（约束检查）、`*_infershape.cpp`（输出 shape 推导，可能位于共享目录如 `*_utils/op_host/`） |

## 输出

| 文件 | 说明 |
|------|------|
| `ttk_extract_case_info.py` | 单用例信息提取脚本（直接从 `case["tensors"]` / `case["params"]` 提取，无 torch 依赖） |
| `ttk_{op_name}_cases_low.csv` | low 档位用例（`S5_mapped_cases_low.json`，全 normal） |
| `ttk_{op_name}_cases_full.csv` | high 档位用例（`S5_mapped_cases_high.json`，data_range 展开） |
| `golden_plugin.py` | 自生成的 TTK golden 函数（通过 `--plugin` 加载）。已有 `tests/assets/golden.py` 时 `--plugin` 直接指向该文件，不生成此文件 |

## 信息来源优先级（强制）

校验阶段（任务 2/4）以算子源码为权威源，S5 映射数据为待验证对象：

- **L0** `*_def.cpp`：输入/输出名称、顺序、dtype 注册（最高权威）
- **L1** `*_tiling_check.cpp` / `*_tiling*.cpp`：参数约束检查
- **L2** `*_infershape.cpp`：输出 shape 推导
- **L3** `S2P1_operator_model.json`：算子接口模型（由 _def.cpp 导出，次权威）
- **L4** `S5_mapped_cases_*.json`：映射数据（由 S5 mapper 生成，**可能有 bug**）

禁止直接从 S5 映射数据取值而不通过源码校验。禁止凭直觉推断 dtype/shape。

## 执行顺序约束（强制）

以下任务必须按编号顺序逐步执行，禁止跳步或抢跑。

| 任务 | 文件 | 前置条件 | 状态判断 |
|------|------|---------|---------|
| 任务 1 | `{skill_base}/references/ttk-converter/01-csv-common.md` + `{skill_base}/references/ttk-converter/02-kernel-fields.md` + `{skill_base}/references/ttk-converter/03-kernel-extraction.md` + `{skill_base}/references/ttk-converter/04-kernel-tasks.md`（任务 1 节） | 无 | ttk_extract_case_info.py 已生成，初步验证打印结果与 JSON 一致 |
| 任务 2 | `{skill_base}/references/ttk-converter/04-kernel-tasks.md`（任务 2 节） | 任务 1 完成 | 校验结果全部 PASS（发现 bug → 修复 → 重验证通过） |
| 任务 3 | `{skill_base}/references/ttk-converter/04-kernel-tasks.md`（任务 3 节） | 任务 2 完成 | ttk_{op_name}_cases_low.csv + ttk_{op_name}_cases_full.csv 已生成 |
| 任务 4 | `{skill_base}/references/ttk-converter/04-kernel-tasks.md`（任务 4 节） | 任务 3 完成 | `python scripts/ttk_validate_csv.py` 校验全部 PASS |
| 任务 5 | `{skill_base}/references/ttk-converter/04-kernel-tasks.md`（任务 5 节） | 任务 4 完成 | golden_plugin.py 已生成，初步验证通过 |
| 任务 6a | `{skill_base}/references/ttk-converter/05-kernel-acceptance.md`（6a 节） | 任务 5 完成 | 单用例 TTK kernel 执行成功 |
| 任务 6b | `{skill_base}/references/ttk-converter/05-kernel-acceptance.md`（6b 节） | 任务 6a 完成 | `--tc 10` 采样门禁全部 PASS |
| 任务 6c | `{skill_base}/references/ttk-converter/05-kernel-acceptance.md`（6c 节） | 任务 6b 完成 | 用户确认后执行，nohup 后台运行 |

**通用规则**：

- 前置条件表中标明的条件未全部满足时，禁止启动该任务
- 完成当前任务的全部子步骤并确认状态判断满足后，才能进入下一任务
- 自检失败 → 回到对应任务修正，修正完成后方可继续

## Kernel 模式约束

代码约束详见 `04-kernel-tasks.md` 任务 5「注意事项」表（6 条）。CSV 约束详见 `01-csv-common.md`「通用禁止」节（3 条）。

## ACLNN/E2E 预留

> TODO: 待实现。两种模式共享 `01-csv-common.md` 的 9 个公共字段，通过 CSV 表头含 `api_name` 区分。
> - **ACLNN**（`ApiTestcaseStructure`，24 字段 = 9 公共 + 15 专有）：`api_name` 以 `aclnn` 开头，新增 `tensor_view_*` 系列字段、`output_tensor_indexes`、`scalar_*` 字段等
> - **E2E**（`FrameworkApiTestcaseStructure`，19 字段 = 9 公共 + 10 专有）：`api_name` 为框架 API 路径（如 `torch.add`），新增 `tensor_view_*` 系列字段、`golden_api` 等
> 实现时新增 `0X-{mode}-fields.md` + `0X-{mode}-extraction.md` + `0X-{mode}-tasks.md` 三个文件。

## 文件索引

| 文件 | 职责 | 读入时机 |
|------|------|---------|
| `01-csv-common.md` | 公共字段定义（9 个，全模式通用）+ CSV 格式规则（9 条）+ 通用禁止（3 条） | 任务 1 |
| `02-kernel-fields.md` | Kernel 模式专有字段定义（17 个）+ CSV 列顺序（26 列） | 任务 1 |
| `03-kernel-extraction.md` | S5 JSON → CSV 字段提取规则 + data_range 映射表 + 返回值结构 | 任务 1 |
| `04-kernel-tasks.md` | 任务 1-5 详细执行指令 + golden 函数注意事项（6 条） | 各任务按需 |
| `05-kernel-acceptance.md` | 任务 6a/6b/6c TTK 执行验收标准 | 任务 5 完成后 |
