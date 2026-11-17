# 阶段 5：采集所选 DAV_9201/DAV_9202 目标的兼容迁移基线

前置：完整的 `[GATE-4] PRECISION=N/N_PASS`。产物为 `${EVIDENCE_DIR}/perf_report.md` 及不可覆盖的 `perf/baseline/` 原始数据。本阶段采集与分析性能，不自动宣称达到加速目标。

## LOADED 检查点

读取 [feature-catalog.md](../references/feature-catalog.md)、[tiling-and-memory-guide.md](../references/tiling-and-memory-guide.md)，按计划加载命中的专题并记录 LOADED。有 `torch-ops-profiler` 或 `ops-profiling` 时按名称查询其实际版本；缺失时使用当前 torch_npu 官方 profiler 接口，不套用未支持的采集参数。

## Step 5.1：冻结比较条件

选取 `migration_plan.md` 中代表性 small/large/full/tail shape、主要 dtype/layout 与风险路径；不以无关平均随机 shape 替代。性能和正确性 case 可以不同，但性能输入同样必须合法并先验证结果。

记录所选目标（DAV_9201 或 DAV_9202）的同一设备、实际负载、输入/seed、同步与缓存策略、warmup/active 次数、profiler 配置、版本、代码/产物 hash。没有用户指定时，选择能稳定测量的采集次数并记录。

## Step 5.2：执行采集并按目标归因

1. 通过 Stage 4 的真实绑定执行 `torch_npu.profiler` 采集，按 case 单独保存原始输出。
2. 在实际版本导出的 `op_statistic.csv`、`kernel_details.csv` 或等价文件中，关联目标算子与其子 kernel、调用次数、设备耗时。多 kernel 算子的聚合口径先声明；不可把整张 CSV 求和当成目标算子耗时。
3. 总量按实际有效调用数归一化；保留原始值、归因行和统计方法，不能把 host wall/event 时间改名为 profiler device 时间。
4. 对主导成本记录计算/搬运/标量/同步等可取得指标；PMU/缓存指标按工具与设备能力采集，不支持时给出证据并记 N/A，不能捏造 pipe 占比。
5. 采集失败区分设备争用、工具能力和算子执行问题；配置降级的 case 标出实际配置，不与不同口径结果直接 A/B。

## Step 5.3：输出报告并保存基线

报告列出每个 case 的 shape/dtype/layout、目标调用/子 kernel、device 耗时、运行核数或可取得的调度信息、瓶颈与不确定项。跨芯片 3510 数据仅在输入和口径可比时另列；Stage 6 的基准始终是本次目标设备上的兼容实现，9201/9202 基线及报告分别保存。

## Gate：十项证据

| 序号 | 必要证据 |
|---|---|
| 1 | profiler 实际执行记录与退出状态 |
| 2 | 设备、版本、负载、代码/库 hash、采集配置 |
| 3 | 非空的原始输出目录与 CSV/等价 trace 文件 |
| 4 | 计划中代表性 case 的完整状态及输入定义 |
| 5 | 目标算子/自定义 kernel 的归因与调用数核对 |
| 6 | 逐 case 的 device 耗时与归一化公式 |
| 7 | 风险路径和源路径/回退路径的性能分析（不适用注明依据） |
| 8 | 瓶颈指标及采集能力边界；不支持项的 N/A 证据 |
| 9 | `perf_report.md` 的结论、问题与优化候选 |
| 10 | `perf/baseline/` 原始数据与基线产物清单，Stage 6 不覆盖 |

齐备后输出 `[GATE-5] PERF=COLLECTED`；任何必要采集或归因缺失输出 `[GATE-5] PERF=NOT_COLLECTED BLOCKED`。真实采集但性能无改善可以完成本阶段，不能将其写成性能达标。
