# Iteration Log

## Milestones

| Iter | Stage | Kernel SHA256 | Command | Result | Evidence |
|------|-------|---------------|---------|--------|----------|

`Stage` 按主流程记录：`skeleton` → `implement` → `focused test` → `full test #1` →
`merge` → `full test #2` → `optimize` → `final review` → `full test #3` → `benchmark`。
`merge` 只有在 DESIGN 声明 `merge_required: true` 时才是真实融合记录（同时补一行融合版的
`focused test`）；否则追加一行 `Stage=merge`、`Result=passed`、Command 注明
`not_applicable`。`full test #3` 只在 optimize 采纳改动、kernel 变化时补跑。

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

先做解析预筛、再实跑不超过 3 组：容量与核数从设备读取（读取失败时写回退路径），被硬约束
淘汰的组合用一行汇总，不占实跑名额。

预筛与标定：

- 淘汰汇总：`<BM/BN/BK/SK × 映射 → 违反的硬约束>`
- 设备读取：L1/L0A/L0B/L0C/UB = `<...>`，核数 = `<...>`，L2 = `<...>`（来源：`<api | 回退>`
- 标定常数：峰值 = `<...>`，`BW_mte2` = `<...>`，`BW_mte1` = `<...>`，`BW_fix` = `<...>`
  （键：`<SoC 名, CANN 版本, dtype, 映射档>`；标定不成立时只用于排序）

每个 matmul 阶段每个实跑组各占一行，在该 matmul 自己的最大 dev 用例上采集 fresh profile，
evidence 按 `evidence/skeleton/<gemm-id>/profile/kernel_details.csv` 分目录。Decision 可写
`passed` / `passed (waived)` / `bandwidth-bound` / `failed` / `blocked`；`mac ≥ 0.99` 且
比值 ≤ `1.15` 记 `passed`（计算饱和，须写 `mac` 与两个峰值口径）。不满足默认判据但
有可核对理由时写 `passed (waived)`，并在下方写明理由。

| GEMM (id / kernel) | Case (uuid / M,N,K,dtype) | BM/BN/BK/SK × 映射 | 预测 ms | 预测比值 | mac_ratio | mte2_ratio | overlap | gemm_ms | torch_mm_ms | Decision | Evidence |
|------|------|------|------|------|-----------|------------|---------|---------|-------------|----------|----------|

块映射与名义搬运（每个 matmul 阶段一行，用于区分块映射重读与流水缺失）：

| GEMM | BM/BN/BK/SK | nm | nn | A 名义重读 | B 名义重读 | 理想字节 | 名义字节 | 名义折算 (ms) |
|------|-------------|----|----|------------|------------|----------|----------|---------------|

名义折算用标定的有效 MTE2 带宽作分母，不用 HBM 带宽；HBM 带宽只用于估算穿透到 HBM 的流量。

GEMM 回测（含 MMAD 的算子 implement 后必填；偏差 >15% 必须写归因；本轮未做时记
`not_run` 并写原因，optimize 入口会用 baseline profile 重算）：

| GEMM | 融合 kernel | GEMM 段耗时 (ms) | 有效算力 (TFLOPS) | 骨架同 shape (ms) | 偏差 | 归因 |
|------|-------------|------------------|-------------------|-------------------|------|------|

骨架达成率（分子计有实测的骨架组，`waived` 按实测计入并附饱和理由，`failed` 记 0；
分母为全部 GEMM 实现段耗时；分子分母逐段配对，各 GEMM 的最大 dev 用例不同时按各自用例
取骨架值与实现段耗时后再求和，不得跨用例混算）：

- `skeleton_ratio = <Σ骨架 gemm_ms ÷ Σ全部 GEMM 实现段耗时>`
- kernel SHA256：`<sha>`；证据：`<kernel_details.csv 路径>`

列式 A 的骨架（`layout_a=col`）需额外记录 K 相位：
`k_raw = <原始 K>`，`k_pad = <补到 BK 倍数后的 K>`，`k_phase_padded = <true/false>`。
约束为 `ceil(K/SK) % (BK/SK) == 0`，`skeleton_profile.py` 会自动补 K；实现侧同样要用
补后的 K，不能拿原始 K 直接跑列主序 A。

host 固定开销自检（implement 收敛后必填）：

| Case | device_ms | end2end_ms | host 占比 | top host 动作（占比 / 归类 a/b/c） | 处置 | 改后占比 |
|------|-----------|------------|-----------|-----------------------------------|------|----------|

- `host_selfcheck: <证据路径>`；处置前后 Σ：`<...>`；剩余项与原因：`<...>`；
  剩余项合计占 host 时间比例：`<...>`（单项都低但合计仍大时，必须写聚合评估结论）

GEMM 段耗时取单轮 `kernel_details.csv` 的对应 kernel 行；多轮采集按 `Step Id` 分组，取
目标 kernel 齐全的一轮，不得对全部行求和。

豁免理由（`passed (waived)` 时必填）：

- 未达判据与实测值：
- 不适用 / 不可达原因（附 roofline 或对照）：
- 已做过的判断或改动与结果：
- optimize 阶段的补偿计划：

## Benchmark 劣化用例（benchmark 后必填）

合并结果里 `speedup < 1.0` 的用例逐条登记，缺项、只列部分用例或用整体结论代替逐条登记时
不得记 benchmark `passed`。结论取 `保留` / `已解释` / `tiny-shape-exempt`（单例
`end2end_ms < 1 ms` 且 `device_ms` 同量级）。

| uuid | speedup | device_ms | end2end_ms | 归因 | 结论 | Evidence |
|------|---------|-----------|------------|------|------|----------|

## Performance Rounds

optimize 阶段必填：模式 B 2~3 轮（上限 3 轮），模式 A 至少一轮。每轮记录改动轴、瓶颈、前后耗时、决策与证据路径。

- Before/After 均为同一 `end2end_ms` 口径；host 侧改动需在 Evidence 中附 `device_ms` /
  `host_gap_ms` 拆分。

| Round | Axis | Bottleneck | Before (ms) | After (ms) | Decision | Evidence |
|------:|------|------------|------------:|-----------:|----------|----------|

## Notes

用于记录 benchmark 前的 `Expected:` 假设、死路和最终总结。
