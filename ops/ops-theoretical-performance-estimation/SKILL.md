---
name: ops-theoretical-performance-estimation
description: 从算子设计或计算流估算性能、分析瓶颈并比较方案，适用于设计期性能目标制定及 OpModel 建模与优化任务。提供 Ascend950 RegBase 纯 Vector Roofline 估算，以及 OpModel 建模、性能评估、DAG 寻优、联合寻优、空泡与数据流分析和设计文档生成；OpModel 支持范围取决于实际 wheel、模型与硬件配置。
---

# 算子理论性能估算

用本 skill 把算子设计或计算流转成性能模型与用例报告，按任务选择以下路径。

## 路径选择

| 任务 | 入口与范围 |
| --- | --- |
| Ascend950 / DAV_3510 RegBase 纯 Vector 的理论上限与目标时延 | 本文下方的 Roofline 流程，使用内置标定数据，无需安装 OpModel |
| 明确使用 OpModel，或需要模型支持的 Cube / Vector / 混合计算、DAG 方案评估与搜索 | 读取 [OpModel 工作流入口](ops-opmodel-toolkit/SKILL.md)，按实际 wheel、模型、硬件配置与成本依据确认支持范围 |

OpModel 子目录整体随本 skill 安装，保留七项工作流、示例、模板、脚本和独立评测。
进入该路径时直接读取子目录的 `SKILL.md`，无需客户端将它单独发现为技能；将
`OPMODEL_SKILL_DIR` 设置为该子目录的绝对路径。安装、建模与运行约束均按子入口执行，
模型和结果保存在用户工作区。已有模型或指定方法的任务沿用其路径。

Roofline 结果是其建模假设下的时延下界；OpModel 结果是声明硬件、成本与调度下的模型耗时。
报告注明所用路径、完整 case 或局部范围及验证状态，不直接混用两者的排名或验收阈值。
模型软件运行无需 NPU；设备收益需要同口径实测。设备采集使用 `ops-profiling`，
CANN Simulator 仿真使用 `ops-simulator`，实现优化策略使用 `ascendc-perf-optimize`。

## Roofline 范围

下文的引擎、计算流、CLI 和标定约束均属于 Roofline 路径。引擎使用随 skill 提供的微基准标定数据。固定开销默认 0，可依据独立实测通过 `fixed_overhead_us` 显式给定；不提供预置固定开销与 HBM 爬坡修正。

- 硬件：Ascend950 / DAV_3510，纯 VectorCore kernel。
- 计算：RegBase 纯 vector 微指令流。
- 不支持：Cube kernel、cube-vector 混合 kernel、`references/vector_instruction_types.json` 里没有的隐式特殊数学模型。

## 架构

两层，严格分离：

- **共享标定引擎**（不要按算子重写）：`scripts/evaluate_a5_vector_roofline.py` + 两个参考 JSON。它掌管全部建模——指令 issue/cycle、overlap、带宽、tiling 修正——由 Ascend950 微基准标定，带回归测试（`tests/offline/test_overlap_model.py`）。
- **计算流定义**（`references/examples/<op>_flow.json`）：算子 = 一组**模版**，每个模版是一条完整的流：`copy_in`（GM→UB，→MTE2）/ `vf`（寄存器驻留的 vector 块，聚合若干 `op` 步骤，→Vector）/ `copy_out`（UB→GM，→MTE3）。`vf` 声明它的 `live` UB 张量；每 tile 的 UB footprint（峰值存活行 buffer）由此自动推出，用来定 tiling/驻留（行切分模版用显式 `tiling` 覆盖）。它是唯一真源，也是符合性金标准；对照真实设计文档评审。
- **计算流解释器**（`scripts/flow_runtime.py`）：把一个 case 路由到某个模版，并把其 flow 解释成引擎的标准 case dict。通用、算子无关。
- **通用 CLI**（`scripts/perf.py --op <name>`）：唯一算子无关入口——委托 `flow_runtime.build("<name>", ...)`。无 per-op Python；新算子只是一个 `<op>_flow.json` + `<op>_cases.json`。单一代码路径把可回溯/可诊断集中在一处（多模版路由在 flow 里，不在 CLI）。

## 工作流（推荐：flow + 模版）

1. 读算子设计，落成 `references/examples/<op>_flow.json`——一个或多个**模版**（每个：`name`、`select` 谓词、一条 copy_in / vf / copy_out 的 `flow`）。不同 shape/attr/驻留 → 不同模版（每个 pass = 一个 `vf`；非驻留 = 一个会重新 `copy_in` 数据的行切分模版）。`vf` 列出它的 `live` UB 张量（`extent: row|scalar`）→ UB footprint 自动推 tiling（chunk 切分则给显式 `tiling`）。`op` 步骤带 `on`（full=N / reduced=rows）；`compute_dtype:"fp32_for_lowp"` 对低精度输入按 fp32 算，`Cast` 显式写成 `when:"lowp"` 步骤（输入 dtype 用 `@in`）——注意每个 VF 各 cast 一次，所以 2-VF 的低精度流会对输入 cast 两次。
2. 写一份测试用例 JSON，由 `scripts/perf_testcase.py`（`validate()`，零依赖，唯一生效的契约）校验——shape/dtype/attr（同 TTK/ST 的点）；支持多输入/可选输入、动态个数（`repeat`）、`attrs`、`default_attrs`，以及可选的 `"template"` 强制指定模版。
3. 不加任何 Python——通用 CLI 按名字处理任何算子。
4. 跑 `python3 scripts/perf.py --op <op> --cases <op>_cases.json --show-case`（表格会显示路由到的模版），再跑 `python3 tests/offline/test_design_conformance.py` 确认解释出的 flow 与设计一致。
5. 把上板实测（TTK / profiling 的 Task Duration）与引擎输出的理论时延对照；若实测与理论对不上，只调白名单旋钮（`perf_testcase.TUNING_WHITELIST`）并记 `tuning_reason`，禁止改引擎标定常量。

样例：softmax/rms_norm/layer_norm/add/gelu/cast 的 `references/examples/<op>_flow.json` + `<op>_cases.json`（三个 norm 算子带 驻留 + 行切分 两种模版）。

## 工作流（底层：直接调引擎）

一次性检查可以跳过 flow/CLI，按 `references/perf_case_schema.md` 手写一份 JSONL case 文件，直接跑 `scripts/evaluate_a5_vector_roofline.py`（见下方「评估器」）。

## 参考

- 硬件数据：`references/hardware/ascend950pr.json`
- RegBase 指令执行单元与吞吐模型：`references/vector_instruction_types.json`；`source_documents` 记录 asc-devkit 公开性能/单双发表的固定版本，`performance_source` 与 `documented_performance_*` 标明逐指令核对范围。实测、兼容别名和复合模型单独注明依据。
- 标准 case dict（引擎输入）schema 与建模说明：`references/perf_case_schema.md`
- 算子测试用例 JSON（CLI 输入）契约：`scripts/perf_testcase.py`（`validate()`，零依赖）
- 算子计算流定义——模版，唯一真源 + 符合性金标准：`references/examples/<op>_flow.json`
- 计算流解释器（路由 + flow→case dict）：`scripts/flow_runtime.py`
- flow/模版 编写契约与步骤：`references/builder_template.md`
- 自验证（ST）：`tests/README.md`——三层：`tests/offline/` flow+引擎门禁（无需板子）；`tests/onboard/` 950 上理论 vs 实测（`tests/onboard/kernels/<op>/` 下自写的 `<<<>>>` kernel + TTK profiling）；`tests/calibration/` 把 JSON 常量绑定到 skill 内的上板 bench 并加门禁（`check_calibration.py`）。单指令吞吐 + 超越函数 overlap 由 skill 内仪器 `tests/calibration/instruments/regbase-microbench/` 标定（源码 + 入库板上原始数据作溯源）；GM 带宽由 `tests/onboard/kernels/membw` 标定。标定哲学（绝不为让某个算子通过而调常量）、`measured_cycles` 口径、`core_split_units` tiling 模型都写在 `tests/calibration/README.md` 和 `references/perf_case_schema.md`——动任何标定常量前先读它们。

## 评估器

```bash
python3 scripts/evaluate_a5_vector_roofline.py \
  --hardware-json references/hardware/ascend950pr.json \
  --instruction-json references/vector_instruction_types.json \
  --cases perf_cases.jsonl \
  --out-json roofline_report.json \
  --out-md roofline_report.md
```

默认建模选择（下面是要点；每个旋钮的完整参考见 `references/perf_case_schema.md`）：

- **Roofline**：`theoretical = max(mte2, vector, mte3)`，在 `overlap_model="max_pipe"`（稳态 overlap；`serial` 则相加）下，再加标定修正——`fill_drain`（有限 tile 的流水填充/排空）、`core_balance`（核间尾块负载，用 flow 的 `core_split_units`，**不是** UB 的 `tile_count`）、`launch_floor`（极小 shape 的设备下限，按 `max` 施加、非叠加）。
- **Vector 时间**：`issue_per_cycle = issue_slots / throughput_cycles`，先按执行单元、再按子管线聚合——一个单元的时间是 `max(发射口时间, 各子管线各自的 Σ)`，超越函数（`vector_transcendental`：Exp/Log/Div/Sqrt/Reciprocal/Rsqrt）与 1 拍 ALU 重叠。它是吞吐下限（需要 ILP）；`target_efficiency` / `dependency="dependent"` 覆盖低 ILP 的依赖链——两者不要叠加。
- **GM 带宽**：`chip_peak`（默认上限 roofline）；`l2_aware`（标定的 L2 驻留 copy 曲线，用于 norm 这类 1读1写 流式算子）；`gm_rw_contention`（读写共享 HBM，α≈0.3）。`gm_reread_bytes` 走 L2、与 HBM 流重叠。
- **Tiling / 模版**：`tiling` 由 UB 容量推 `tile_count`；`flow_runtime` 把 case 路由到第一个 `select` 命中且 tiling UB 可行的模版。驻留 1-pass vs 行切分（重读 x）由可行性自动落出——大 H 的重读是模版自己的 GM 流量，不是引擎规则。

如果 tile 太少、达不到稳态，就给 `tile_count`/`tiling` 让 fill/drain + 尾块均衡被建模；tile 稀少时不要把稳态 `max_pipe` 的 bound 当作可达目标上报。

说明低 ILP 的最终目标时保持口径一致：可保留吞吐模型并用 `target_efficiency` 表达
依赖损失；或用 `dependency="dependent"` 修正指令发射，并令 `target_efficiency=1`
避免再次计入同一损失。不能先说二者互斥，随后又把依赖修正后的时间除以默认 0.60。

理论与实测对照使用 `tests/onboard/compare.py` 的 3% 容差：实测低于
`theoretical × (1 - 0.03)` 时为 `BELOW_FLOOR`。例如 8 us < 9.7 us，需核对
计算流、实际工作量、硬件能力和测量／标定口径；不能据此称优化成功或为通过而改常量。
