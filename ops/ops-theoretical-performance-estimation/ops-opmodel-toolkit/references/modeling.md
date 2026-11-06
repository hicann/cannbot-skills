# 编写自己的模型

模型需要明确三个问题：算什么、如何把任务映射到执行资源、每条计算或搬运消耗什么。
YAML 模型文件和用户 Python 文件可以编辑；仿真和寻优能力由安装包提供。

建议从[矩阵乘示例](../assets/examples/matmul/README.md)复制一个完整目录，先运行，再逐项修改。
编写不同算子时，需要同时调整语义、任务空间和执行图，单独改模型名称不会改变计算语义。

## 目录结构

```text
my_matmul/
  __init__.py
  case.yaml
  hardware.yaml
  definition/
    semantic.yaml
    workload.yaml
    split.yaml
    operator.yaml
    dag.yaml
    costs.yaml
  optimize/
    search.yaml
    split.yaml
    dag.yaml
```

| 文件 | 用户需要定义什么 |
| --- | --- |
| `semantic.yaml` | 张量形状、数据类型、输入输出、数学操作及数值容差 |
| `workload.yaml` | 完整 case 参数、派生值、合法性约束和任务到逐核参数的映射 |
| `split.yaml` | 任务空间、任务字段、工作量和可拆分维度 |
| `operator.yaml` | 硬件 profile、缓冲区、流水阶段、固定参数和默认分核策略 |
| `dag.yaml` | 搬运、计算、循环、依赖和缓冲区使用 |
| `costs.yaml` | 成本来源、系数、单位和校准状态 |
| `case.yaml` | 一次运行的形状和参数 |
| `optimize/search.yaml` | 可搜索参数及其范围 |
| `optimize/split.yaml`、`dag.yaml` | 本次分核或 DAG 搜索实验 |

完整 case 入口读取 `case.yaml`；DAG 搜索读取实验中的 `shape/tiling`，分核搜索读取
实验中的 `shape`。对照同一组输入时，要同步修改这些文件。

## 第一个修改：改变输入形状

示例输入与分块如下：

```yaml
M: 256
N: 256
K: 128
Mb: 128
Nb: 128
Kb: 128
M_L0: 64
N_L0: 64
K_L0: 128
```

`M/N/K` 是完整矩阵形状。`Mb/Nb` 决定输出任务块，`Kb` 是 L1 的 K 面板，
`M_L0/N_L0/K_L0` 是内部计算分块。本例约束要求内部块整除对应外层块。

可以把 `M` 改为 `288` 后运行新的输出目录。`split.yaml` 会生成尾块，
`workload.yaml` 把实际的 `m_extent/n_extent` 传给执行图。确认任务覆盖完整输出，
不要用平均分块大小替代尾块。`num_groups/num_tasks` 等派生值无需写入完整 case。

## 描述计算和搬运

示例执行顺序是 GM → L1 → L0A/L0B → MMAD → L0C → UB → GM，按 K 面板循环累加。
数据类型、传输大小、pool 和 `use_pools` 共同决定工作量、容量及资源复用。

修改计算图时，需要一起检查：

- 计算的形状与数学语义一致；MMAD 输入的维度顺序为 M、K、N。
- 搬运的元素数、数据类型和实际访问区域一致。
- 缓冲区字节数足够，并且复用前已完成相关消费者的读取。
- 尾块和最终输出覆盖正确，K 方向累加结束后才发布输出。

字段和表达式写法见[模型配置参考](model-reference.md)。

## 自定义成本

示例的 Cube 和搬运成本使用安装包内的硬件 profile，`costs.yaml` 的系数表为空。
引入向量服务或外部测量成本时，系数应给出 `value`、`unit`、`source`、`calibrated`，
并通过 `native` 或 `vf` 中的命名绑定供模型使用。周期、字节和时间单位不能混用。
只有声明为已校准并不足以证明准确性，来源必须与当前操作、输入和硬件相符。

## Python 扩展 hooks

首先用 YAML 表达模型。需要扩展时，将 Python 文件放入自己的模型目录，保持
目录名称可作为 Python 包名，并保留 `__init__.py`。

| 扩展文件 | 契约 |
| --- | --- |
| `workload.py` | `derive(params, cfg=None) -> dict`；负责派生运行参数，需要维护与任务投影的一致性 |
| `split_hooks.py` | 导出 `SPLIT_SPEC`，类型为 `accel_model.model.split.model.spec.SplitSpec` |
| `ms_hooks.py` | 提供 `SEARCH_DIMS` 及可选的 `phase_filter(phases)` 等搜索约束 |

存在 `ms_hooks.py` 时要提供 `SEARCH_DIMS`，不能假定仍自动使用 YAML 的搜索空间。
随附[搜索 hook 示例](../assets/examples/hooks/ms_hooks.py)从当前模型的 `optimize/search.yaml`
读取参数声明，并限定单阶段分组。需要时将它复制为 `work/my_matmul/ms_hooks.py`，
重新启动 Python 进程后运行 DAG 搜索即可。它只是约束声明，搜索算法仍来自安装包。
`phase_filter` 用于 `--phases search` 的阶段划分枚举；固定 `C1` 时不调用该过滤器。

Python 派生函数和分核规格会承担原本由声明式配置完成的工作。仅为修改形状或分块，
无需引入这两个扩展。使用 YAML 还是 hooks 应保持一致，避免同一配置存在两个来源。

实现相关成本的 `implementation/scope` 和原生注入 `dtype_params` 见
[VF 成本接口](framework/vf-costs.md)。这些约束用于检查来源与适用域，不把已声明
`calibrated` 当作自动设备验收；成本更新只作用于匹配实现，不能用于修改硬件服务率。
