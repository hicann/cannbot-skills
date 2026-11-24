# Optimization Log

## Optimization Policy

- mode: target | explore
- metric: performance.candidate.mean_ms（end2end 口径，含 `run` 内 host 成本）
- threshold: <value | none>
- early_stop: skeleton_ratio >= 0.80（baseline 之后按同一份 device profile 判定；命中则不做任何探索轮）
- device: npu:<id>
- dev workload: workload_dev.jsonl (<n> cases)

## Performance Rounds

- Before/After 均为同一 `end2end_ms` 口径；host 侧改动需在 Evidence 中附 `device_ms` /
  `host_gap_ms` 拆分。

| Round | Axis | Bottleneck | Before (ms) | After (ms) | Decision | Evidence |
|------:|------|------------|------------:|-----------:|----------|----------|

命中 `early_stop: skeleton_ratio >= 0.80` 时本表只保留表头（0 轮），Final Verdict 写明
依据与 develop 回测证据。

## Structural Delta

每轮 Analyze 先填本表：候选与主参考样例、原始 golden 的结构差异逐项列出。`Cost` 只能
用 fresh profile 的 pipe 占比或消融实测 ms；写 `not measured` 的差异项不得作为优化轴依据。

| Round | Item | Golden / 主参考样例 | Candidate | Cost (ms) | Source |
|------:|------|----------------------|-----------|----------:|--------|

## Diagnosis

每轮预登记至少 2 个候选根因；实测与预期方向相反时必须追加一条重新诊断记录，
不得直接 rejected 收工。

| Round | Candidate root causes (≥2) | Discriminative test | Expected | Actual | Conclusion |
|------:|----------------------------|---------------------|----------|--------|------------|

## Benchmark 劣化用例（benchmark 后必填）

合并结果里 `speedup < 1.0` 的用例逐条登记，缺项、只列部分用例或用整体结论代替逐条登记时
不得记 benchmark `passed`。结论取 `保留` / `已解释` / `tiny-shape-exempt`（单例
`end2end_ms < 1 ms` 且 `device_ms` 同量级）。

| uuid | speedup | device_ms | end2end_ms | 归因 | 结论 | Evidence |
|------|---------|-----------|------------|------|------|----------|

## Final Verdict

- 轮次数：
- best 前后耗时与加速比：
- 剩余瓶颈：
- 下一步建议：

## Notes

用于记录 benchmark 前的 `Expected:` 假设、被拒方向与死路。
