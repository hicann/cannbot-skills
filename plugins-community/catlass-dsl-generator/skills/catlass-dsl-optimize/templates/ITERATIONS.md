# Optimization Log

## Optimization Policy

- mode: target | explore
- metric: performance.candidate.mean_ms（end2end 口径，含 `run` 内 host 成本）
- threshold: <value | none>
- device: npu:<id>
- dev workload: workload_dev.jsonl (<n> cases)

## Performance Rounds

- Before/After 均为同一 `end2end_ms` 口径；host 侧改动需在 Evidence 中附 `device_ms` /
  `host_gap_ms` 拆分。

| Round | Axis | Bottleneck | Before (ms) | After (ms) | Decision | Evidence |
|------:|------|------------|------------:|-----------:|----------|----------|

## Diagnosis

每轮预登记至少 2 个候选根因；实测与预期方向相反时必须追加一条重新诊断记录，
不得直接 rejected 收工。

| Round | Candidate root causes (≥2) | Discriminative test | Expected | Actual | Conclusion |
|------:|----------------------------|---------------------|----------|--------|------------|

## Final Verdict

- 轮次数：
- best 前后耗时与加速比：
- 剩余瓶颈：
- 下一步建议：

## Notes

用于记录 benchmark 前的 `Expected:` 假设、被拒方向与死路。
