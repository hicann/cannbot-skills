# 分析仿真结果

先阅读完整 case 的 `result.json`，确认输入、硬件和结果范围，再定位瓶颈。
下面的字段针对教程普通模型，不能套用到所有执行入口的返回值。

## 总体耗时与核分配

| 字段 | 含义 |
| --- | --- |
| `total_cycles` | 完整 case 的模型周期 |
| `main_cycles` | 主阶段耗时 |
| `reduce_cycles` | 跨核归约等后处理耗时 |
| `overhead_cycles` | 模型中声明的额外开销 |
| `summary.active_cores` / `ncores` | 有任务的核数与配置核数 |
| `summary.core_tasks` / `core_cycles` | 各核任务数与耗时 |
| `evaluation.bottleneck_pipe` | 评价器识别的瓶颈流水线 |

`host_runtime` 是仿真程序自身的运行时间，不能当作算子执行时间。
换算时 `duration_us = cycles / (clock_ghz × 1000)`，必须使用本次实际硬件时钟。

教程中四个输出块通常只占用四个核。扩大核数不会自动把一个任务拆成更多任务。
当尾块导致任务大小不同，应查看各核耗时和分配，不能仅用平均任务数判断均衡。

## 流水线与指令统计

`Pipe_statistic.csv` 的列为 `Pipe, Duration(us), Cycle, Size(B), Ops`，
`Instruction_statistic.csv` 按指令汇总相同的指标。

普通完整 case 导出的 CSV/trace 是关键核或各阶段关键核的执行证据。
它们不是所有核工作量的总和。不同流水线可以重叠，不能把各流水线的 `Cycle`
直接相加得到总周期；百分比相加也可能超过 100%。

建议按以下顺序分析：

1. 查看 `total_cycles` 和各核耗时，确定是单核执行还是负载不均衡占主导。
2. 查看主要流水线及搬运字节数，确认是计算、读入、写回还是等待造成耗时。
3. 对照 `params.json` 的生效参数与 `assignment.json` 的任务分配。
4. 在 trace 中观察计算和搬运重叠、空闲间隙以及缓冲区复用导致的等待。
5. 一次只修改一类变量，用相同形状和硬件重新运行。

`trace.json` 使用 trace-event 格式，可在支持该格式的时间线查看器中加载。
物理模型与普通模型的轨道和统计范围不同，以结果中的 `scope/artifacts` 为准。

## 比较两轮实验

完成[参数寻优](optimization.md)的 `work/tile64` 实验后执行：

```bash
python - <<'PY'
import json
from pathlib import Path
before = json.loads(Path('work/baseline/result.json').read_text())
after = json.loads(Path('work/tile64/result.json').read_text())
print('基线周期:', before['total_cycles'])
print('候选周期:', after['total_cycles'])
print('基线/候选:', before['total_cycles'] / after['total_cycles'])
PY
```

比值大于 1 表示这次模型仿真更快。比较前确认模型语义、输入形状、硬件、
缓存假设和结果范围一致。模型速度变化不等于真实设备一定获得相同比例的收益。

`performance_consistency_report.json` 等文件描述该次模型的额外检查状态。
未提供独立性能预期时，`not_run` 表示未执行该项检查，不等同于仿真启动失败或通过。

## 搬运与 Vec 诊断

逻辑输入、实际源读取、目的输出与物理分配分别核对，含重读、padding 和布局跨度时
不应相等。资源服务、延迟和等待可能重叠，见[搬运结果](framework/transfer-contracts.md)。
[Vec 预测](framework/vec-plans.md)的内部时间线与外部调用 trace 不重复累计；
单次周期、调用合计与包含搬运等待的图周期分别报告。模型成本不赋予编译或设备资格。
