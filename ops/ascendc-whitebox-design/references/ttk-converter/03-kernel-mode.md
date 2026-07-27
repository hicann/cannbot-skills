# Kernel 模式 CSV 生成与验收

Kernel 模式使用固定脚本从 Step 5 final case 文件生成 TTK CSV，并通过 `python3 -m ttk kernel` 执行 low/high 少量用例验收。Kernel CSV 的 26 个字段全部由固定脚本生成，不要求 LLM 手写或复核字段提取脚本。

## 输入

| 输入 | 用途 |
|---|---|
| `S5_cases_low.json` | Step 5 final low cases，生成 low CSV。 |
| `S5_cases_high.json` | Step 5 final high cases，生成 high CSV。 |
| `S2P1_operator_model.json` | `_def.cpp` 未传入或无法提取 Attr 时的 attribute 名 fallback。 |
| `*_def.cpp` | 提取 Attr 注册名和 `.ValueDepend()` 输入名，用于生成 CSV `attributes` 列。 |
| `{op_name}` | 作为 CSV `op_name` 列。 |
| `{op_path}/tests/assets/golden.py` | 默认 TTK kernel golden 来源。 |

Step 5 final case 的 schema 由 Step 5 文档定义。固定脚本负责按该 schema 读取 tensor descriptor、params 和 meta 信息。

## 输出

| 输出 | 说明 |
|---|---|
| `ttk_{op_name}_cases_low.csv` | low 档位 Kernel CSV。 |
| `ttk_{op_name}_cases_high.csv` | high 档位 Kernel CSV。 |
| `ttk_{op_name}_cases_low_one_result.csv` | low 单用例 TTK kernel 验收结果。 |
| `ttk_{op_name}_cases_high_one_result.csv` | high 单用例 TTK kernel 验收结果。 |
| `ttk_low_one.log` | low 单用例 TTK kernel 日志。 |
| `ttk_high_one.log` | high 单用例 TTK kernel 日志。 |

## 任务 1：生成 Kernel CSV

使用固定脚本：

```bash
python {skill_base}/scripts/ttk_generate_kernel_csv.py \
  --op-name {op_name} \
  --operator-model {whitebox_dir}/S2P1_operator_model.json \
  --op-def-cpp {op_def_cpp_path} \
  --low-cases {whitebox_dir}/S5_cases_low.json \
  --high-cases {whitebox_dir}/S5_cases_high.json \
  --low-csv {whitebox_dir}/ttk_{op_name}_cases_low.csv \
  --high-csv {whitebox_dir}/ttk_{op_name}_cases_high.csv
```

`{op_def_cpp_path}` 必须由主流程或子 agent 传入实际定位到的 `*_def.cpp` 路径，不得假设文件名一定等于 `{op_name}_def.cpp`。

完成条件：

- 脚本返回 0。
- `ttk_{op_name}_cases_low.csv` 存在，且 CSV 数据行数等于 `S5_cases_low.json` 的 case 数。
- `ttk_{op_name}_cases_high.csv` 存在，且 CSV 数据行数等于 `S5_cases_high.json` 的 case 数。
- CSV 数据行数不包含 header 行。

## CSV 字段来源

CSV 列顺序必须与 `02-kernel-fields.md` 的 Kernel CSV 列顺序完全一致。

| CSV 列 | 来源类型 | 生成方式 |
|---|---|---|
| `testcase_name` | 脚本获取 | 生成稳定 `case00000` 格式名称；原始 case id 写入 `remark`。 |
| `network_name` | 脚本获取 | 从 case meta 中读取可用网络名；无则留空。 |
| `op_name` | 脚本参数 | 使用命令行 `--op-name`。 |
| `input_shapes` | 脚本获取 | 从 Step 5 final case 的 input descriptor 生成。 |
| `input_dtypes` | 脚本获取 | 从 Step 5 final case 的 input descriptor 生成。 |
| `input_formats` | 脚本获取 | 从 Step 5 final case input descriptor 的 `format` 字段生成。 |
| `output_shapes` | 脚本获取 | 从 Step 5 final case 的 output descriptor 生成。 |
| `output_dtypes` | 脚本获取 | 从 Step 5 final case 的 output descriptor 生成。 |
| `output_formats` | 脚本获取 | 从 Step 5 final case output descriptor 的 `format` 字段生成。 |
| `input_ori_shapes` | 脚本默认 | 留空，TTK 回退。 |
| `input_ori_formats` | 脚本默认 | 留空，TTK 回退。 |
| `output_ori_shapes` | 脚本默认 | 留空，TTK 回退。 |
| `output_ori_formats` | 脚本默认 | 留空，TTK 回退。 |
| `attributes` | 脚本获取 | 从 `_def.cpp` 提取 Attr 注册名和 `.ValueDepend()` 输入名，构造白名单并过滤 case params。 |
| `input_data_ranges` | 脚本获取 | 从 Step 5 final case 的 data range descriptor 生成，映射规则见下节。 |
| `precision_tolerances` | 脚本获取 | 根据 output dtype 生成。 |
| `absolute_precision` | 脚本默认 | 固定 `1e-8`。 |
| `output_inplace_indexes` | 脚本默认 | 固定 `()`。必要的同名 inplace 由 TTK op_info 自动推导。 |
| `output_shape_unknown_indexes` | 脚本默认 | 固定 `()`。 |
| `is_enabled` | 脚本默认 | 固定 `True`。 |
| `remark` | 脚本获取 | 写入原始 case id、source 和可审计 meta 信息。 |
| `soc_series` | 脚本默认 | 留空。 |
| `priority` | 脚本默认 | 固定 `0`。 |
| `dump_file_prefix` | 脚本默认 | 留空。 |
| `manual_input_binaries` | 脚本默认 | 固定 `()`。 |
| `manual_golden_binaries` | 脚本默认 | 固定 `()`。 |

## attributes 规则

CSV `attributes` 列是一个 dict 字段，写入形式必须为 key-value：

```text
{"attr_name": attr_value, "const_input_name": const_input_value}
```

`attributes` 的 key 白名单来自两类名称：

- operator attributes：来自 `_def.cpp` 的 Attr 注册名。
- const input values：来自 `_def.cpp` 中 `.ValueDepend()` 标记的输入名。

固定脚本构造 key 白名单：

```text
attribute_csv_keys = attr_names_from_def_cpp ∪ value_depend_input_names_from_def_cpp
```

随后从 case params 中按白名单提取实际值，生成 CSV `attributes` dict：

```text
attributes = {
  key: case.params[key]
  for key in attribute_csv_keys
  if key in case.params
}
```

因此，`_def.cpp` 只提供允许写入 `attributes` 的 key；对应 value 必须来自当前 case 的 `params`。如果某个白名单 key 不存在于当前 case params，则该 key 不写入该 case 的 CSV `attributes`。

case params 中的 shape 构造变量、tiling/router 信息、network 审计信息和 mapper 内部字段不得写入 CSV `attributes`。如果未传入 `_def.cpp` 或 `_def.cpp` 未提取到 Attr 注册名，脚本使用 `S2P1_operator_model.json["attributes"]` 作为 fallback。

## 固定默认值

以下字段不需要 LLM 判断，固定使用默认值：

| 字段 | 默认值 |
|---|---|
| `input_ori_shapes` | 空 |
| `input_ori_formats` | 空 |
| `output_ori_shapes` | 空 |
| `output_ori_formats` | 空 |
| `absolute_precision` | `1e-8` |
| `output_inplace_indexes` | `()` |
| `output_shape_unknown_indexes` | `()` |
| `is_enabled` | `True` |
| `soc_series` | 空 |
| `priority` | `0` |
| `dump_file_prefix` | 空 |
| `manual_input_binaries` | `()` |
| `manual_golden_binaries` | `()` |

Golden plugin 不在 Kernel CSV 生成阶段生成或配置。

## Format 规则

`input_formats` 和 `output_formats` 由固定脚本从 Step 5 final case descriptor 的 `format` 字段读取。TTK 阶段不重新解析 `_def.cpp`，不重新推导 format。

Step 5 默认写入 `ND`。若 path/config 明确存在 format 约束，Step 5 mapper 必须按当前 case 的 path/config 写入对应 format。缺失 `format` 或 `format` 不是非空字符串时，Kernel CSV 生成必须失败。

## TensorList 输出格式

TensorList 嵌套格式以 `02-kernel-fields.md` 为准。固定脚本负责将 Step 5 final case 中的 TensorList descriptor 转换为 TTK Kernel CSV 需要的嵌套 tuple：

- `input_shapes` / `output_shapes`：展开到每个子 tensor。
- `input_dtypes` / `output_dtypes`：使用压缩嵌套 dtype。
- `input_data_ranges` / `precision_tolerances`：展开到每个子 tensor。

## precision_tolerances 规则

固定脚本按输出 dtype 生成 tolerance：

| dtype | `(rtol, atol)` |
|---|---|
| `float16` | `(0.001, 0.001)` |
| `bfloat16` | `(0.001, 0.001)` |
| `float32` | `(0.0001, 0.0001)` |
| 其他 dtype | `(0.001, 0.001)` |

## data_range → input_data_ranges 映射表

固定脚本按每个输入 descriptor 的 data range 标签生成 TTK `input_data_ranges`。

| data_range | `input_data_ranges` 生成方式 | value_domain 约束 |
|---|---|---|
| `normal` | 固定 seed 随机范围 | positive → `(0.01, 10)`；non_negative → `(0, 10)`；non_zero → 排除 \|x\|<0.1；range → `(min, max)` |
| `zero` | `(0, 0)` | — |
| `extreme` | `(dtype_max * 0.9, dtype_max)` | — |
| `negative` | 固定 seed 负数随机范围 | range → 约束到 `(min, min(0,max))`，不满足则回退 normal |
| `tiny_pos` | `(1e-7, 1e-5)` | range → 约束到 `(max(min,1e-7), min(max,1e-5))`，不满足则回退 normal |
| `all_ones` | `(1, 1)` | — |
| `near_zero` | `(-0.01, 0.01)` | range → 约束到 `(max(min,-0.01), min(max,0.01))`，不满足则回退 normal |
| `with_inf` | `(1, float('inf'))` | — |
| `with_nan` | `(nan, nan)` | — |

固定 seed 为 `42`，保证 CSV 生成可复现。

## 任务 2：CSV 格式校验

生成 CSV 后必须执行格式校验：

```bash
python {skill_base}/scripts/ttk_validate_csv.py {whitebox_dir}/ttk_{op_name}_cases_low.csv
python {skill_base}/scripts/ttk_validate_csv.py {whitebox_dir}/ttk_{op_name}_cases_high.csv
```

完成条件：两条命令均返回 0。

## 任务 3：Golden 来源确认

默认使用已有 assets golden：

```text
{op_path}/tests/assets/golden.py
```

检查项：

- 文件存在。
- 包含模块级 `__golden__`。
- `__golden__["kernel"]` 中存在 `{op_name}`。

不默认生成 `golden_plugin.py`。如果 assets golden 不存在或未注册当前算子，跳过 TTK kernel 执行验收并报告缺少 golden。

## 任务 4：TTK Kernel 单用例验收

执行 TTK kernel 单用例验收前，必须先验证本地 CANN + TTK kernel 环境可启动。固定命令：

```bash
python {skill_base}/scripts/ttk_precheck_env.py \
  --whitebox-dir {whitebox_dir} \
  --ops-test-kit-path {ops_test_kit_path} \
  --golden-path {op_path}/tests/assets/golden.py \
  --op-name {op_name}
```

如果用户提供了非默认 CANN 环境脚本路径，可追加：

```bash
  --setenv-path {setenv_path}
```

随后读取 `{whitebox_dir}/ttk_precheck_report.json` 并按报告决定是否执行验收：

- 若 `kernel_gate.status != "passed"`：跳过任务 4 的 low/high TTK kernel 验收，不执行 `python3 -m ttk kernel`，报告 `kernel_gate.reason`。
- 若 `kernel_gate.status = "passed"` 且 `checks.cann_env.status = "passed"`：使用当前环境执行 low/high 验收命令。
- 若 `kernel_gate.status = "passed"` 且 `checks.cann_env.status = "passed_after_source"`：必须使用报告中的 `checks.cann_env.setenv_path`，source 后执行 low/high 验收命令。

low 单用例验收完成并检查通过后，再执行 high 单用例验收；禁止并行执行 low/high，避免 TTK profiling/profiler 目录争用。

low 单用例：

```bash
cd {ops_test_kit_path} && python3 -m ttk kernel \
  -i {whitebox_dir}/ttk_{op_name}_cases_low.csv \
  -o {whitebox_dir}/ttk_{op_name}_cases_low_one_result.csv \
  --plugin {op_path}/tests/assets/golden.py \
  -t {low_testcase_name} \
  --pc 1 \
  --seed 42 > {whitebox_dir}/ttk_low_one.log 2>&1
```

high 单用例：

```bash
cd {ops_test_kit_path} && python3 -m ttk kernel \
  -i {whitebox_dir}/ttk_{op_name}_cases_high.csv \
  -o {whitebox_dir}/ttk_{op_name}_cases_high_one_result.csv \
  --plugin {op_path}/tests/assets/golden.py \
  -t {high_testcase_name} \
  --pc 1 \
  --seed 42 > {whitebox_dir}/ttk_high_one.log 2>&1
```

若 `ttk_precheck_report.json` 显示必须 source CANN setenv，使用报告中的 `checks.cann_env.setenv_path` 执行等价命令：

```bash
bash -lc 'source "{setenv_path}" && cd "{ops_test_kit_path}" && python3 -m ttk kernel \
  -i {whitebox_dir}/ttk_{op_name}_cases_low.csv \
  -o {whitebox_dir}/ttk_{op_name}_cases_low_one_result.csv \
  --plugin {op_path}/tests/assets/golden.py \
  -t {low_testcase_name} \
  --pc 1 \
  --seed 42 > {whitebox_dir}/ttk_low_one.log 2>&1'

bash -lc 'source "{setenv_path}" && cd "{ops_test_kit_path}" && python3 -m ttk kernel \
  -i {whitebox_dir}/ttk_{op_name}_cases_high.csv \
  -o {whitebox_dir}/ttk_{op_name}_cases_high_one_result.csv \
  --plugin {op_path}/tests/assets/golden.py \
  -t {high_testcase_name} \
  --pc 1 \
  --seed 42 > {whitebox_dir}/ttk_high_one.log 2>&1'
```

用例选择：

- 避开 `case00000`。
- 优先选择 `case00001`。
- 如果 `case00001` 是特殊/empty case，则选择中间位置 case。

完成条件：

- 日志出现 `Loaded custom golden: kernel.{op_name}`。
- 日志出现 `Compilation Result: SUCC`。
- result CSV 中 `perf_status = PASS`。
- result CSV 中 `precision_status = PASS`。
- result CSV 中 `memory_oob_status = PASS` 或空。
- Golden Shape/Dtype 与 Output Shape/Dtype 匹配。

## 可选增强验收

默认不执行批量验收。用户明确要求或提交前需要增强验证时，可执行 low/high 抽样：

```bash
cd {ops_test_kit_path} && python3 -m ttk kernel \
  -i {whitebox_dir}/ttk_{op_name}_cases_low.csv \
  -o {whitebox_dir}/ttk_{op_name}_cases_low_sample_result.csv \
  --plugin {op_path}/tests/assets/golden.py \
  --tc 10 \
  --pc 1 \
  --seed 42 > {whitebox_dir}/ttk_low_sample.log 2>&1
```

high CSV 可按同样方式执行 `--tc 10`。批量执行可能存在 TTK 多进程环境干扰；如批量失败，应优先将失败 case 单独执行确认是否为真实用例问题。

## 失败处理

| 失败点 | 处理 |
|---|---|
| CSV validator 失败 | 回到任务 1，检查固定脚本和 Step 5 final case。 |
| assets golden 缺失或未注册 | 跳过 TTK kernel 执行验收，报告缺少 golden。 |
| `Compilation Result: FAIL` | 核查 case shape/dtype 是否违反算子约束。 |
| `precision_status != PASS` | 核查 golden 实现、data_range 或输出 dtype/shape。 |
| `memory_oob_status` 失败 | 单 case 复现并保留日志，必要时进入内存/崩溃调试流程。 |
| profiler 目录争用或 `Directory not empty` 等并发痕迹 | 按 low → high 顺序重跑单用例，不直接判为 case 失败。 |
