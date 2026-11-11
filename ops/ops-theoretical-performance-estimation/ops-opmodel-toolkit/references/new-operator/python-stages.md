# Python Stage 接口

默认使用 YAML DAG。需要 Python 根据输入参数组织有限的不规则声明时，在模型的
`__init__.py` 导出 `STAGES` 映射，类继承 `accel_model.kernel.stages_base.Stage`，
实现 `build(context) -> dict`。返回值沿用 [DAG 字段](../framework/dag-reference.md)，
包括 `body/signals/pools/dtypes`；它不是设备代码生成接口。

## 注册与输入输出

可编辑的 [CopyStage 示例](../../assets/examples/hooks/copy_stage.py) 使用
`ProgramBuilder` 声明 GM→UB→GM 搬运。复制到用户模型后注册：

```python
from .copy_stage import CopyStage

STAGES = {"Copy": CopyStage}
```

模型还需声明 `rows/cols` 参数、包含 `Copy` 的 `pipeline.phases`、有效硬件与
`scratch` buffer（UB，单槽至少 `rows * cols * 4` 字节）。示例按单 AIV 工作量
给出尺寸；完整 case 的任务划分与输出归属仍由用户模型声明。
同名 stage 已在 `dag.yaml` 中定义时优先使用 YAML，Python 只补齐缺失 stage。

| 上下文 | 含义 |
| --- | --- |
| `context.params` | 本次最终参数，用于构造尺寸及有限条件 |
| `context.hardware` | 本次硬件配置 |
| `context.operator` | 本次算子声明 |
| `context.owner` | 物理入口的当前 owner；普通入口为 `None` |
| `context.tasks()` | 本次有限任务访问序 |

上下文类型 `BuildContext` 与 `Stage/ProgramBuilder` 均可从
`accel_model.kernel.stages_base` 导入。构造须无副作用、可重复，返回普通可序列化数据；
不返回事件、资源、运行时 Tensor 或回调。

## 可选的声明辅助器

| 调用 | 输入与输出 |
| --- | --- |
| `ProgramBuilder(pools=..., dtypes=...)` | 可选 pool 与 dtype 映射 |
| `move(**fields)`、`vec(**fields)`、`mmad(**fields)` | 接受相应 DAG 指令字段，返回符号引用 |
| `emit(kind, after=(), **fields)` | 添加声明；`id` 在本 stage 内唯一 |
| `finish(signals=..., **metadata)` | 返回 stage 字典，信号可引用上述返回值 |

`after` 接受跨 stage 输入信号名，不接受符号引用或运行时完成事件。
局部数据依赖使用 tensor 和存储声明；跨 stage 连接使用 `signals` 与
`pipeline.dataflow`。容量、版本、全部消费者和覆盖要求见[存储接口](../framework/ub-storage.md)。

## 执行、搜索与限制

普通及物理入口均可使用该声明方式。物理模型按其 owner、投影参数和任务范围
构造，不能在每个 owner 中重复输出完整工作量。参数、硬件或 Python 文件变化后
重新运行；导出的候选快照只适用于绑定的输入与硬件，变更时从源模型重新生成。

物理结构搜索要求各 owner 的同名 stage 能用共享 DAG 模板表达；依赖 owner 或
任务列表而无法共享的模板可能支持执行或参数搜索，但不支持相应结构搜索。
Python 模型的 `partition/placement/accumulator_residency` 结构变换暂不自动应用。
其他动作仍以本模型的可用动作和实际验证状态为准。

运行时数据决定的新分支、无界循环及 Python 操作运行时事件/资源不受支持。
先验证固定 case，再覆盖尾块、多 task、多组及槽位回绕。后端一致性与
[执行数值验收](../framework/numeric-execution-validation.md)分别记录。

## 迁移有限任务与向量声明

`Stage.build(context)` 生成与 YAML 相同的 DAG，不操作运行时 token 或资源。
合法空 stage 不需要虚构占位 VEC；输出信号仍须对应实际产生的标签。
`ProgramBuilder.move` 使用端点、size/dtype 与当前搬运声明，不能套用旧 Stage 的 `bytes=`。
多 task、多 group、条件空输出和全部消费者按[并发指南](concurrency.md)核对。
需要 VF 内部步骤时显式声明[程序与计划](../framework/vec-plans.md)，固定周期指令不会自动补齐。
