# Profiler 入库标准

适用于 `knowledge/profiler/`；先执行[公共入库标准](common-entry-standards.md)。

## 两类 concept

- **采集工作流**：环境、设备、命令、同步、warm-up/replay、输出和完整性检查。
- **指标解释**：字段、单位、分母、NA/estimate、产品差异、每核分布和源码归因。

## 强制内容

1. **工具合同**：工具/版本、SoC/Device、环境、采集模式和精确入口。
2. **采集边界**：kernel、launch/block、同步、迭代窗口及是否包含 host/L2/其他 kernel。
3. **artifact 地图**：原始文件、关键列、单位和缺失含义。
4. **样本有效性**：correctness、workload、Device/频率、block、metric、warm-up/replay 一致。
5. **解释顺序**：绝对总时长与最慢核 → pipe 时间 → 数据量/wait/stall → 源码/时间线。
6. **统计**：trials、中心值/离散度、每核分布、最慢 block 和原始 artifact。
7. **归因门禁**：瓶颈判断同时引用基础条件、绝对时间、相关指标、每核分布和源码证据。

## 拒绝条件

- 把高/低 ratio 直接解释为好坏，把 NA 当 0、estimate 当实测或混加不同核类型。
- 只保留截图/均值，或比较 correctness、频率、Device、block、输出语义不同的样本。
- 用官方字段定义证明当前算子的瓶颈或收益。

## 验收清单

- [ ] 命令、边界、artifact、单位和样本可比性可复现。
- [ ] 结论遵循绝对时间到源码的归因链，不由单个 ratio 得出。
