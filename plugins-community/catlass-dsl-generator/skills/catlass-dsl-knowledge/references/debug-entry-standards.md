# Debug 入库标准

适用于 `knowledge/debug/`；先执行[公共入库标准](common-entry-standards.md)。

## 失败阶段

先区分环境、Python/AST、TLAIR/lowering、编译/链接、launch/runtime、设备异常、同步/竞争、
数值正确性和 profiler 采集；不同阶段不得共用未经证实的根因。

## 强制内容

1. **症状指纹**：原始错误、返回码、artifact、首个 mismatch 或时间线现象。
2. **最小复现**：最小 shape/dtype/layout、代码/命令、环境和预期/实际结果。
3. **诊断树**：按低成本、非破坏顺序列命令、预期产物、分支和下一步。
4. **证据链**：源码/官方说明证明机制，日志/dump/IR/测试证明当前实例。
5. **根因**：区分失败位置、触发条件和原因；证据不足则保留候选。
6. **修复与回退**：最小修改、保持条件、副作用和撤回条件。
7. **验证闭环**：最小复现、相邻边界、完整 workload 和必要压力回归。

## 首查映射

- sentinel 未变：launch/grid/写回；稳定错位：layout/stride/tile。
- tail/dtype 错：mask/padding/cast；首 tile 对后续错：state/slot/依赖轴。
- 偶发旧数据、NaN、死锁：flag/event/barrier、buffer 所有权和并发写槽。

## 拒绝条件

- 未复现即声称唯一根因，或把重装、清缓存、加 barrier 当作无条件修复。
- 只给补丁而没有症状、诊断、触发条件和回归验证。
- correctness 未通过却进入性能归因或 learned 性能结论。

## 验收清单

- [ ] 症状、最小复现、诊断树、根因证据、最小修复和回归闭环完整。
