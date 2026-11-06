# 模型文件接口

新模型使用 `definition/` 保存固定定义，`optimize/` 保存搜索输入，结果写入用户项目的新目录。
完整可编辑模板见[矩阵乘示例](../../assets/examples/matmul/README.md)。

## 输入文件

| 文件 | 必要内容与使用范围 |
| --- | --- |
| `definition/semantic.yaml` | 张量、维度、数学操作、输入输出与数值容差 |
| `definition/workload.yaml` | `parameters.required/defaults`、`derive`、`assertions`、`semantic_bindings`、`hardware_fields`、`core_projection` |
| `definition/split.yaml` | `kind: split_spec`、任务空间、任务字段、约束与工作量 |
| `definition/operator.yaml` | `name/hardware/params/buffers/pipeline`、运行策略及可选物理执行声明 |
| `definition/dag.yaml` | `task_graph` 和 `kernels`，具体语法见[DAG 接口](dag-reference.md) |
| `definition/costs.yaml` | `source/coefficients/native/vf/sensitivity`，系数注明单位、来源、校准状态 |
| `optimize/search.yaml` | 参数空间及受支持的候选选项 |
| `case.yaml` | 一次运行的完整输入形状，字段须在 workload 中声明 |

完整模型 bundle 需提供六份固定定义和搜索声明。`semantic.yaml` 的操作应由
`dag.yaml:task_graph.implementation` 恰好覆盖一次；`task_graph.nodes` 与
`operator.yaml:pipeline.phases` 的 stage 集合必须相同。

普通固定配置运行和完整 bundle 检查不是同一入口；能运行一个 DAG 不代表其他文件已通过检查。
兼容根目录平铺的模型，但存在 `definition/` 后不能再在根目录放同名固定定义。

## 参数与表达式

完整 case 只填写公开输入，如 M/N/K 与模型允许的 tiling 参数。`derive` 按声明顺序
定义派生值；`num_tasks/num_groups` 等逐核字段不作为独立的完整 shape 输入。
`core_projection` 描述任务字段与逐核参数的对应关系，尾块须使用实际大小。
输入、表达式或断言不合法时，保留错误消息并修正模型。

`costs.yaml` 可为空系数表，但不能将它理解为所有操作免费。使用 `native` 成本绑定时，
不要在 `operator.params` 重复声明同名成本。可选的显式引用只使用 `costs: costs.yaml`。
VF 成本的常量、公式和外部预测配置见[VF 接口](vf-costs.md)。

## 调用与输出

普通完整 case 使用[run_case.py](../../scripts/run_case.py)或
`accel_model.app.case.native_case.run_native_case()`；参数和返回值见[Python API](../python-api.md)。
固定配置使用 `opmodel MODEL --output DIR`，其范围与完整 case 分开记录。
声明 `physical_execution` 的模型使用[物理执行接口](physical-execution.md)。

核对 `params.json`、`assignment.json`、`result.json` 与其中的 `artifacts`。
运行产物不是模型输入，编辑产物不会自动修改模型。搜索配置与固定定义各有用途，
不能把 `search.json` 当成下一轮的 `operator.yaml`。

## Python 扩展

| 文件 | 输入与返回契约 |
| --- | --- |
| `workload.py` | `derive(params, cfg=None) -> dict`，返回用户模型的派生参数 |
| `split_hooks.py` | 导出合法的 `SPLIT_SPEC`，见[分核接口](split-model.md) |
| `ms_hooks.py` | `SEARCH_DIMS` 及可选约束，见[DAG 搜索接口](dag-search.md) |
| `__init__.py` | 需要 Python Stage 时导出 `STAGES`，见[Stage 接口](../new-operator/python-stages.md) |

Python 文件属于用户模型，需随模型一起交付；模型目录名应为合法 Python 包名。
修改 hooks 后重新启动运行进程。纯 YAML 能表达的模型无需额外 Python 文件。
