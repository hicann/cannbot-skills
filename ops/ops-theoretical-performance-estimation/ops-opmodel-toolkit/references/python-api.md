# Python API

## 自定义模型：完整 case

```python
from pathlib import Path
import yaml
from accel_model.app.case.native_case import run_native_case

shape = yaml.safe_load(Path('work/my_matmul/case.yaml').read_text())
report = run_native_case(
    'work/my_matmul', shape,
    output_dir='work/from_python', derive=True, export_trace=True,
)
print(report['total_cycles'])
```

调用前应完成[快速入门](quickstart.md)的模型复制步骤，输出目录应为新目录。
完整调用签名为：

```text
run_native_case(operator_dir, shape=None, *, output_dir=None,
                strategy=None, trace_slots=False, derive=False, export_trace=True,
                hardware_path=None, backend='simpy', policies=None, gates=None,
                performance_expectations=None)
```

| 参数 | 说明 |
| --- | --- |
| `operator_dir` | 用户模型目录 |
| `shape` | 完整 case 参数映射，须符合 workload 输入契约 |
| `output_dir` | 结果目录，不能位于系统 `/tmp` 下 |
| `strategy` | 普通模型的分核策略；传入生成方案的 `split` 映射 |
| `derive` | 普通模型按输入重新派生工作量参数；教程设置为 `True` |
| `export_trace` | 是否导出完整 trace |
| `trace_slots` | 普通模型的 slot trace；物理模型不支持此选项 |
| `backend/policies/gates/hardware_path` | 除默认 SimPy 外的这些选项受物理执行模式限制 |
| `performance_expectations` | 独立的性能预期检查，须匹配模型和统计口径 |

返回值为字典。普通模型包含 `total_cycles/main_cycles/reduce_cycles/overhead_cycles`、
`summary/evaluation/artifacts` 等；物理模型返回结构不同，不能混用字段。
非法字段、形状、容量或不支持的选项通常抛出 `ValueError` 等异常；保留消息定位模型问题。

## 声明构造与模型验证

用户 Python stage 实现 `Stage.build(context)` 并导出 `STAGES`，输入输出及示例见
[Python Stage](new-operator/python-stages.md)。物理入口也支持此方式，但搜索变换有单独限制。

`accel_model.model.validation.score_case` 接受普通或物理模型，返回
`(model, report, streams)`；普通结果范围为 `native_aggregate`。
调用和数值证据解释见[数值验收](framework/numeric-execution-validation.md)。
需要完整 shape 与分核结果时继续使用本页的 `run_native_case`。

## 接口选择

使用本页列出的 API，调用前核对包版本和接口签名。
批量运行时为每个 case 分配独立输出目录；加载不同版本的 Python hooks 后重启进程，
避免模块缓存导致编辑未生效。
