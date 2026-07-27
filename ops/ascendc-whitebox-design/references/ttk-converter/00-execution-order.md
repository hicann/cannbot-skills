# TTK Converter 执行入口

## 角色

将 Step 5 final case 文件转换为 TTK Kernel CSV，并按需执行 low/high 单用例 TTK kernel 验收。

常规流程只调用固定 wrapper 脚本，不再派发 TTK 子 agent，不再读取 `03-kernel-mode.md` 做逐步 LLM 分析。`01-csv-common.md`、`02-kernel-fields.md`、`03-kernel-mode.md` 暂保留为脚本规则来源和失败诊断参考。

## 输入

| 参数 | 说明 |
|------|------|
| `{skill_base}` | 技能根目录绝对路径 |
| `{whitebox_dir}` | 白盒测试产出目录，需包含 `S5_cases_low.json`、`S5_cases_high.json`、`S2P1_operator_model.json` |
| `{op_name}` | 规范算子名，如 `add_rms_norm` |
| `{op_path}` | 算子源码根目录，如 `.../ops-nn/norm/add_rms_norm` |
| `{op_def_cpp_path}` | 当前算子的 `*_def.cpp` 实际路径 |
| `{ops_test_kit_path}` | 主 Agent 已在当前工作目录 `$PWD` 下定位到的 `ops-test-kit/` 路径 |

## 常规执行命令

主 Agent 直接执行：

```bash
python3 {skill_base}/scripts/run_ttk_kernel_module.py \
  --op-name {op_name} \
  --whitebox-dir {whitebox_dir} \
  --op-path {op_path} \
  --op-def-cpp {op_def_cpp_path} \
  --ops-test-kit-path {ops_test_kit_path} \
  --skill-base {skill_base}
```

如用户提供了非默认 CANN 环境脚本，可追加：

```bash
  --setenv-path {setenv_path}
```

若只需生成/校验 CSV 和 precheck，不执行 kernel 单用例验收，可追加：

```bash
  --skip-kernel-run
```

## wrapper 内部流程

`run_ttk_kernel_module.py` 串行执行以下固定步骤：

1. 校验输入路径和 case 数量。
2. 调用 `scripts/ttk_generate_kernel_csv.py` 生成 low/high CSV。
3. 调用 `scripts/ttk_validate_csv.py` 校验 low/high CSV。
4. 调用 `scripts/ttk_precheck_env.py` 生成 `ttk_precheck_report.json`。
5. `kernel_gate.status == "passed"` 时，串行执行 low/high 各 1 个 `python3 -m ttk kernel` 用例。
6. 解析 result CSV 和日志关键状态，生成 `ttk_module_report.json`。

## 输出

| 文件 | 说明 |
|------|------|
| `ttk_{op_name}_cases_low.csv` | low 档位 TTK Kernel CSV |
| `ttk_{op_name}_cases_high.csv` | high 档位 TTK Kernel CSV |
| `ttk_precheck_report.json` | TTK 环境、assets golden 和 kernel gate 报告 |
| `ttk_{op_name}_cases_low_one_result.csv` | low 单用例验收结果 |
| `ttk_{op_name}_cases_high_one_result.csv` | high 单用例验收结果 |
| `ttk_low_one.log` | low 单用例日志 |
| `ttk_high_one.log` | high 单用例日志 |
| `ttk_module_report.json` | wrapper 统一结构化报告，含步骤状态和耗时 |

## 状态判断

常规流程只读取 `ttk_module_report.json` 判断模块结果：

| status | 含义 | 退出码 | 典型场景 |
|--------|------|--------|----------|
| `passed` | CSV 生成/校验通过，TTK 能消费 CSV，low/high 单用例编译成功并产出结果 | 0 | 正常路径 |
| `passed_with_warnings` | CSV 和执行链路通过，但 perf/precision/OOB 有非 PASS 项；数值正确性不作为 TTK 模块门禁 | 0 | 精度失败、OOB 失败、perf 非 PASS |
| `skipped` | CSV 生成/校验通过，但 kernel 验收被跳过 | 0 | `--skip-kernel-run`、golden 缺失/未注册、`ops-test-kit` 或 CANN/TTK 环境不可用 |
| `failed` | CSV 阶段失败，或 TTK 无法消费 CSV/完成基本执行链路 | 1 | 必需输入缺失、CSV 生成/校验失败、CSV 行数不匹配、kernel 编译失败或无 result/log |

### 输入分级

wrapper 将输入分为两类：

| 类别 | 路径 | 缺失处理 |
|------|------|----------|
| CSV 必需 | `S5_cases_low.json`、`S5_cases_high.json`、`S2P1_operator_model.json`、`*_def.cpp`、三个 `ttk_*.py` 固定脚本 | `status=failed`，无法生成 CSV |
| kernel 可选 | `ops-test-kit/`、`tests/assets/golden.py`、CANN 环境 | CSV 继续生成；precheck 记录原因；kernel 验收 `skipped` |

### 常见 reason

| reason | 说明 |
|--------|------|
| `missing_required_for_csv` | CSV 必需输入缺失 |
| `csv_row_count_mismatch` / `csv_validation_failed` | CSV 生成结果不满足要求 |
| `skip_kernel_run_requested` | 用户主动跳过 kernel 验收 |
| `golden_missing` / `golden_op_unregistered` | golden 缺失或未注册当前算子 |
| `env_unavailable` | `ops-test-kit`、TTK kernel 或 CANN 环境不可用 |
| `low_kernel_execution_failed` / `high_kernel_execution_failed` | kernel 编译失败，或未产出 result/log，TTK 未完成基本执行链路 |
| `precision_status_not_pass` / `memory_oob_status_not_pass` / `perf_status_not_pass` | TTK 已完成执行链路，仅作为 warning 输出 |

若 precheck 报告 `checks.cann_env.status == "passed_after_source"`，wrapper 会使用报告中的 `setenv_path` 执行 low/high TTK kernel 命令。

## 失败诊断

仅当 wrapper 失败或用户要求分析 TTK 规则时，再读取以下参考文档：

- `03-kernel-mode.md`：Kernel CSV 生成、字段来源、TTK kernel 验收规则。
- `02-kernel-fields.md`：Kernel CSV 字段定义。
- `01-csv-common.md`：公共 CSV 字段和格式规则。
