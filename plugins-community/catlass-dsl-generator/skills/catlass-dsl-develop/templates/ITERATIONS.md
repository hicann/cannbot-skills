# Iteration Log

## Milestones

| Iter | Stage | Kernel SHA256 | Command | Result | Evidence |
|------|-------|---------------|---------|--------|----------|

## Run Log（每次上板必填）

每一次在设备上执行（focused 用例、批量分片、benchmark、profile、探针）都记一行，记录实际
执行的那条命令与**实际设置的超时值**，不能只写"已运行"。

| Time | 用途 | Command | Timeout | Wall (s) | Exit | Outcome | Evidence |
|------|------|---------|---------|---------:|------|---------|----------|

- 单例运行 `Timeout` 默认 `timeout -k 5 60s`；分片命令填分片级实际超时（如
  `subprocess timeout=600`），并在备注写明片内逐例 60s 硬超时。填 `long`、`默认`、`继承`
  一律视为未设置超时；
- `Timeout` 填命令里真正生效的值（如 `timeout -k 5 60s`、`subprocess timeout=600`），
  用脚本跑的要写脚本内的逐例超时与分片级超时；
- 编译行同样要记 `Timeout` 与墙钟：编译不套 60s，但必须写清实际设置的上限（默认
  `timeout 600s`）；被编译上限杀掉时记 `killed by timeout`，先查编译日志再决定是否重编；
- 远端或容器模式填最终执行环境内生效的值，并在备注写明外层连接超时与 transport
  （如 `ssh 10s / docker exec <容器>`）；
- `Exit` 填真实退出码；被超时杀掉统一写 `killed by timeout`，`124` / `137` 也按这个口径记；
- `Wall` 填墙钟耗时，用于识别"每轮等 8 分钟"这类无效迭代；
- 被超时杀掉的运行不能作为性能结论；下一行单例缩小 shape 或加打印，分片缩小分片规模后
  再定位运行。


## Skeleton Performance（含 MMAD 时必填）

每个 matmul 阶段各占一行，在该 matmul 自己的最大 dev 用例上采集 fresh profile，evidence
按 `evidence/skeleton/<gemm-id>/profile/kernel_details.csv` 分目录。Decision 可写
`passed` / `passed (waived)` / `bandwidth-bound` / `failed` / `blocked`。不满足默认判据但
有可核对理由时写 `passed (waived)`，并在下方写明理由。

| GEMM (id / kernel) | Case (uuid / M,N,K,dtype) | mac_ratio | mte2_ratio | overlap | gemm_ms | torch_mm_ms | Decision | Evidence |
|------|------|-----------|------------|---------|---------|-------------|----------|----------|

块映射与带宽下界（每个 matmul 阶段一行，用于区分块映射重读与流水缺失）：

| GEMM | BM/BN/BK/SK | nm | nn | A 名义重读 | B 名义重读 | 理想字节 | 名义字节 | 带宽下界 (ms) |
|------|-------------|----|----|------------|------------|----------|----------|---------------|

融合回测（CV 融合算子 implement 后必填，偏差 >15% 必须写归因）：

| GEMM | 融合 kernel | GEMM 段耗时 (ms) | 有效算力 (TFLOPS) | 骨架同 shape (ms) | 偏差 | 归因 |
|------|-------------|------------------|-------------------|-------------------|------|------|

GEMM 段耗时取单轮 `kernel_details.csv` 的对应 kernel 行；多轮采集按 `Step Id` 分组，取
目标 kernel 齐全的一轮，不得对全部行求和。

豁免理由（`passed (waived)` 时必填）：

- 未达判据与实测值：
- 不适用 / 不可达原因（附 roofline 或对照）：
- 已做过的判断或改动与结果：
- optimize 阶段的补偿计划：

## Performance Rounds

optimize 阶段必填：模式 B 至少三轮，模式 A 至少一轮。每轮记录改动轴、瓶颈、前后耗时、决策与证据路径。

- Before/After 均为同一 `end2end_ms` 口径；host 侧改动需在 Evidence 中附 `device_ms` /
  `host_gap_ms` 拆分。

| Round | Axis | Bottleneck | Before (ms) | After (ms) | Decision | Evidence |
|------:|------|------------|------------:|-----------:|----------|----------|

## Notes

用于记录 benchmark 前的 `Expected:` 假设、死路和最终总结。
