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

## 新接口的范围

| 入口 | 支持及限制 |
| --- | --- |
| `estimate_vf` | 显式有限程序的单 VF 分区预测，须有全部操作参数；不依赖编译产物 |
| `search_vec_plans` | 一次一个 Vec 区域的有限计划，基线／轴／存储限制见[Vec 接口](vec-plans.md) |
| 变换与跨步搬运 | 有限 ND/NZ、Load3D、显式动态 GM 读取；具体路径和数值限制见[搬运契约](transfer-contracts.md) |
| 编译反馈 | 校验用户实现的构建证据与身份；不启动编译或授予设备资格，见[编译反馈](compile-feedback.md) |

原生 DAG 和 Python Stage 是冻结输入下的有限构图；受限运行时分支／循环另见
[控制接口](runtime-control.md)，Vec 计划不自动支持该组合。原生固定周期和冻结 VF
的搜索路径也不会因为新增 Vec 接口而获得内部计划搜索能力。
