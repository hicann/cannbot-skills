# 工作流索引

`ops-opmodel-toolkit` 提供以下七项工作流。根据任务选择对应指引，也可以在同一任务中
衔接建模、评估、寻优、分析与文档生成。

| 工作流 | 使用时机 | 主要产物 |
| --- | --- | --- |
| [新算子建模](new-operator.md) | 从公式/实现新建模型或扩展 stage | 可运行模型、完整 case、建模规格与验证记录 |
| [性能评估](perf-eval.md) | 评估固定 case 或比较配置 | 同口径周期、瓶颈与配置差异 |
| [DAG 方案寻优](agent-dag-opt.md) | 固定 DAG 配置搜索或探索结构、映射和实现族 | 候选模型、搜索覆盖与复评结果 |
| [联合寻优](joint-optimization/guide.md) | 结合模型、实现与设备实测迭代 | 算子实现、实测收益与模型反馈 |
| [空泡分析与优化](dag-bubble-opt.md) | 定位 trace 等待并验证原因 | 时间线观察、待验证原因与对照实验 |
| [数据流图](dag-dataflow-graph.md) | 查看数据通路、核对声明连接 | Mermaid/ASCII 图与连接诊断 |
| [设计文档生成](opmodel-design-gen.md) | 整理已有模型与评估结果 | 设计正文；多模板时含总览、各模板设计与 case 映射 |

## 调用示例

```text
使用 ops-opmodel-toolkit，把这份算子公式建立成可运行的模型。
先确认完整 shape、数值语义和成本来源，交付固定 case 与建模验证结果。
```

```text
使用 ops-opmodel-toolkit，比较 work/my_operator 的合法 tiling、阶段和缓冲方案。
保存基线，说明实际搜索覆盖，并将候选应用回完整 case 复评。本轮只做模型。
```

```text
使用 ops-opmodel-toolkit，分析这份 trace 的 M pipe 空泡并生成数据流图。
只分析已有产物，给出定位证据与尚未确认的原因。
```

```text
使用 ops-opmodel-toolkit，根据这个模型和寻优结果生成 DESIGN.md，
区分模型排名、数值验证与设备实测，缺少证据的地方注明未验证。
```

```text
使用 ops-opmodel-toolkit，为这批 case 和多个实现模板生成设计交接文档。
总览列出 case 到模板、候选和证据的映射；每种实现模板分别写设计，
共享该模板的 case 逐项记录参数和边界，明确分派、回退及未覆盖范围。
```

联合优化可补充算子工程、原正确性标准、设备环境、计时边界与预算，按
[联合迭代流程](joint-optimization/workflow.md)推进；这些信息已有时直接复用。

## 环境与支持范围

模型仿真和搜索使用已安装的 OpModel wheel，已有 profile 的选择和覆盖见
[硬件配置](hardware.md)。涉及算子编译、正确性验证和设备计时时，使用项目的设备工具链。

各工作流按任务所需范围执行；联合优化中的偏差分析与校准要求见
[联合迭代流程](joint-optimization/workflow.md)。当前版本的接口和已知限制见
[能力矩阵](capabilities.md)。

## Vec 方案与预测任务

```text
使用 ops-opmodel-toolkit，将这段向量计算表达为显式程序，比较融合与拆分计划。
保存 VF 分区、调用域和尾块、缓冲与流水安排，区分单次预测和图总周期。
本轮仅交付方案与接口使用方法，不进行编译或设备测量。
```

由建模／DAG 寻优工作流进入[Vec 参考](framework/vec-plans.md)，设计交接继续使用原工作流。
建模前确认硬件，评估与搜索继承保存配置；已有 trace 分析保留原身份。
