# 执行方式与支持范围

本页帮助选择输入和调用入口。以实际版本、命令帮助和返回状态为准。

| 输入与任务 | 调用入口 | 输出范围与限制 |
| --- | --- | --- |
| 普通固定配置 | `opmodel MODEL --output DIR` | 固定工作量，不自动等同完整 shape |
| 普通完整 case | `run_native_case` 或 `run_case.py` | 完整 case 与分核结果；CSV/trace 可能为关键核 |
| 普通模型检查 | `accel_model.model.validation.score_case` | `native_aggregate`；可选双后端，不认证物理分核，数值需执行用例 |
| Python Stage | `Stage.build(context)` 与 `STAGES` | 输出 DAG 声明；可用于普通/物理入口，具体变换另核对支持域 |
| 声明物理执行的模型 | `opmodel MODEL --case FILE --backend both --output DIR --trace` | 全物理 engine 的结果及各项检查 |
| 固定 DAG 实验 | `dag_run` | 参数/阶段实验结果，采纳后复评完整 case |
| 分核实验 | `split_run` | 分核方案、范围及软件复核状态 |
| 候选比较/图实验 | `graph_search` | 该实验类型支持的候选与验证状态 |

物理模型的后端可选 `simpy/replay/both`；分核评价的后端名使用
`replay/simpy_independent/simpy_shared`。这些选项不能跨入口照搬。
普通完整 case 不接受物理入口的任意 `backend/policies/gates` 组合。

[Python Stage](../new-operator/python-stages.md)、复杂存储/控制、VF 预测和数值检查分别有支持范围。
固定模型能运行，不表示每种搜索动作或数值模式都支持该模型。
遇到不支持组合时保存版本、输入和错误，不将跳过检查写成通过。

模型运行、后端一致性、数值验收、成本校准和设备验证是不同结果。
报告分别呈现 `passed/failed/not_run/unsupported` 等实际状态及缺口。
当前已知命令限制见[能力矩阵](../capabilities.md)。
