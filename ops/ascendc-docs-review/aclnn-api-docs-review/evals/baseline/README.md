# Bench 基线能力数据报告

本目录存放 **冻结的基线报告存档**，作为 skill 能力变更时命中率与耗时的对比锚点，后续 PR 不再提交运行产物（`evals/results/`、`evals/bench-hit-report.*` 已被 `evals/.gitignore` 忽略）。

## 基线数据（skill v1.1.0）

| 项目 | 内容 |
| --- | --- |
| 基线运行时间 | 2026-11-26 18:19:00 ~ 18:28:44（首次冷跑） |
| 基线刷新时间 | 2026-11-26 20:34:25（级别标定校准后重新执行命中比对） |
| 标定校准 | golden #22 级别=提示（修复字段错位）、#30 级别=一般；rules-1"芯片差异化描述规范"分档定级（键值对格式=一般，表格/代码段内=严重） |
| 数据集 | `evals/dataset/random/drop_out_v3/docs/aclnnDropoutV3.md` |
| golden | `evals/aclnn_bench.md`（52 个已知问题） |
| 质检结果 | `evals/results/aclnnDropoutV3-qc-result.json`（67 项） |
| 质检引擎 | DomainCheck_aclnn_API |

## 核心指标

| 指标 | 数值 |
| --- | --- |
| 命中率（查全） | 52/52 = **100%** |
| 精准率（查准） | 52/67 = 77.6% |
| 总耗时 | 584 秒（规范性 483s / 完备性 36s / 可读性 23s / 正确性 42s） |

> 附加观察：级别一致率 52/52 = **100%**（命中条目定级与 golden 严重程度一致的比例，不参与命中判定，不列入核心指标）。

- [bench-hit-report.md](./bench-hit-report.md)：人读命中报告（指标、误报清单、结论）
- [bench-hit-report.json](./bench-hit-report.json)：机器可读版本（含 52 条命中明细）

## 后续规则

后续能力变更时，跑完 bench 后请将本次结果与基线对比：**命中率不得低于基线、耗时不得显著劣化**方可合入；运行产物（`evals/results/`、`evals/bench-hit-report.*`）不入库；如需更新基线存档，须先与基线对比并经确认后复制至本目录。
