# 硬件配置

硬件 YAML 描述可用内存、流水线、搬运路由、数据类型、时钟和核数。模型必须使用
与目标硬件相符的配置，不能只修改设备名称来切换硬件行为。

教程的 `operator.yaml` 默认选择安装包内的 `matmul/ascend_950DT_streaming` profile。
另附[可编辑硬件示例](../assets/examples/matmul/hardware.yaml)，用于演示显式覆盖。
其中的共享服务参数为假设，不是设备实测标定。

| 字段 | 用途及单位 |
| --- | --- |
| `name` | 配置标识 |
| `clock_ghz` | 时钟 GHz，影响周期到时间的换算 |
| `usedCoreNum` | 配置核数；实际活跃核数还受任务数量影响 |
| `memories` | 存储层及 `size_kb` 容量 |
| `dtypes` | 每种元素占用字节数 |
| `pipelines` | 可用流水线 |
| `mov_routes` | `src/dst/pipe` 搬运路径，`bw` 为 byte/cycle |
| `compute.vec.aiv_per_aic` | 每个 AIC 对应的 AIV 数量 |
| `shared_services` | 共享带宽、容量及假设来源 |

共享带宽是显式参数；修改核数不会自动推导新的整机带宽。
改变数据类型时，也要检查元素字节数、缓冲区大小、搬运工作量和计算语义。

## 选择评估配置

复现已有运行时沿用其硬件与成本内容。独立性能评估未指定配置时可调用：

```python
from accel_model.model.case import resolve_eval_hardware

hardware_path = resolve_eval_hardware("work/my_matmul")
print(hardware_path)
```

该函数解析算子保存的 `hardware` 引用，显式 `hardware=名称或路径` 覆盖优先。
配置缺失时报告问题，不按本机设备、芯片前缀或名称改选普通／细化变体。
`theoretical` 不是可接受的硬件选择值；使用明确的名称或路径。

返回路径不自动改变模型。将选择一致地应用于本轮基线、搜索和复评；借用搜索工作流
时继承调用方的配置，不重新选型。普通完整 case 可在用户模型副本的 `operator.yaml`
中选择安装包内的该 profile，再统一运行各入口。自定义 YAML 的覆盖按下文的入口能力处理，
不要假定所有配置字段都接受文件路径。

新建模型先按[硬件准备](hardware-preparation.md)明确目标和来源，将配置引用保存到
`hardware`；独立评估和接续实验均沿用该配置。保留实际工作量、依赖、容量和生命周期。
未覆盖机制影响当前选择或用户明确要求时，再安排细化复核。
逐路由记录实际配置、物理绑定、来源和未覆盖范围。选择 `detailed` 不表示所有
路由均已细化或校准；配置错误、绑定缺失或后端不支持也不是静默降级的理由。
成本变更后重新评估所有参评模型，成本变化本身不计作实现收益。

## 计算 dtype

矩阵乘的输入 dtype 与累加器 dtype 分别声明；BF16 输入配 FP32 累加器不等于
FP32 输入计算。声明方式见[DAG 矩阵乘接口](framework/dag-reference.md#矩阵乘与累加)。
比较结果时保持输入及累加器 dtype、形状和硬件配置一致。
计算周期不包括完整搬运与执行耗时，也不表示对应芯片和 dtype 已完成设备实测校准。

## 当前支持的覆盖入口

固定配置仿真和 DAG 搜索接受 `--hardware`：

```bash
python -m accel_model.app.cli.dag_run work/my_matmul \
  --config work/my_matmul/optimize/dag.yaml \
  --hardware work/my_matmul/hardware.yaml --out work/dag_hardware
```

`--hardware` 可为安装包内 profile 名或存在的 YAML 文件路径。
物理完整 case 也支持硬件覆盖。普通完整 case 的 `run_native_case` 当前不支持
`hardware_path` 覆盖；分核 CLI 没有通用 `--hardware` 参数。
因此不能假定在某个入口覆盖后，其他运行或搜索也自动使用了相同硬件。

需要比较结果时，先确认各入口实际加载的 profile、核数、时钟和带宽假设一致。
