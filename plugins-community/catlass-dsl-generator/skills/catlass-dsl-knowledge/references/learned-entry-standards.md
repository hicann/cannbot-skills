# Learned 入库标准

适用于 `knowledge/learned/`；先执行[公共入库标准](common-entry-standards.md)。

## 唯一写入入口与字段合同

只能通过 `record_knowledge.py record` 写入。entry 必含：

- `operator_family/topic/arch/versions/applicability`；applicability 含 shape、dtype、layout 和集成条件。
- `hypothesis/actual_change/result/status`。
- `correctness_before/after`、`performance_before/after`、`profiling_observation`。
- 项目内 `evidence` 和最终 kernel 的 64 位小写 `kernel_sha256`。

measurement 使用 `status/summary/reason`，profiling 使用 `status/observation/reason`；未执行项填写
reason，不用空字符串。

## 准入门禁

- `correctness_after.status=passed`；编译或正确性失败留在 trace/debug，不进入 learned。
- 性能可 `not_run`，但不得声称速度变化；profiling=passed 必须有 profiling evidence。
- evidence 必须是项目内可访问的 test/profiling 普通文件，安全相对路径且不经过符号链接。
- `kernel_sha256` 必须绑定结论描述的最终 kernel。
- finish 一次批量提交全部候选；批次、同名和同日冲突先失败，不部分写入。

## 结论状态

- `有效`：证据在 applicability 内达到批准目标。
- `条件有效`：结果依赖明确 workload、SoC、launch、版本或集成条件。
- `无效`：正确性通过，但性能回退、未过噪声或假设被证伪。

## 证据与结果表述

- 正确性 evidence 覆盖批准的完整 cases、oracle、阈值和多输出。
- 性能 evidence 保留同配置 trials、统计、metric、设备/频率/launch、噪声门槛和 fresh best 复测。
- profiling 只陈述 artifact 直接支持的观测；`result` 不得越出 applicability。

## 追加与冲突

learned 只追加；新证据扩展或推翻旧结论时新增条件化 concept，不覆盖历史。

## 验收清单

- [ ] 字段、applicability、状态、证据和 SHA-256 完整且一致。
- [ ] correctness_after 通过；未测性能/profile 不产生对应结论。
- [ ] 使用 record 批量追加，结果不越出证据范围。
