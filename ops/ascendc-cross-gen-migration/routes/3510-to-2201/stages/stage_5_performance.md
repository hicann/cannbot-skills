# 阶段 5：性能验证与调优

**目标**：把"能跑"做到"A2 上高性能"，并用 arch22 实测证据支撑结论。

## 必读

- `[LOADED] performance-tuning-a2`
- `ops-profiling`（采集与指标解读的唯一入口）

## Step 5.1 口径确认

按 `input-contract.md` §3 明确：相对基线加速比 / 利用率 / 绝对时延 / 用户未指定。把口径写进报告首行。

## Step 5.2 基线

- 优先用 A2 上同语义既有实现（仓内参考、社区实现）。
- 无基线时，用"自建基线"：同一算子最朴素 A2 实现（如逐元素参考 kernel），并注明它是基线而非目标。
- 基线必须同 dtype、同 shape、同计时协议；记录版本。

## Step 5.3 采集

```bash
# 标准可比较计时（10 组 warmup→target，取中位数）
python3 <ops-profiling>/scripts/collect_perf_application.py ... --out-dir ./perf/a2

# 深度指标
bash <ops-profiling>/scripts/msprof_profile_run.sh --warm-up=3 --output=./msprof_a2 -- <可执行> <args>
python3 <ops-profiling>/scripts/msprof_perf_summary.py ./msprof_a2/PROF_GROUP_* <ops_dir>
```

- 保留 `PROF_GROUP_*` 原始目录。
- 记录 CANN/驱动/`NpuArch`/device/频率稳定性。
- 禁止混用 `msprof` 与 `msprof op` 的数据。

## Step 5.4 归因

按 `performance-tuning-a2.md` §4 的表，为每个代表用例填：主 Bound、关键指标、初步原因。

## Step 5.5 调优循环

1. 每次只改一件事（tiling / 搬运 / 计算 / 并行 / 微调）。
2. 记录 before/after 中位耗时与主指标。
3. 连续 3 轮改善 <3% → 换下一瓶颈或停止。
4. A2 恢复项（§5）逐项 A/B，失败完整回退并记录原因。
5. 单项时间盒：实现 + ≤5 轮构建-采集。

## Step 5.7 简单/数据搬运类算子的瓶颈自查

**硬性要求**：A5 侧若是通用模板/大 kernel（如 `XxxNormalSiso`），简单搬运类必须在 arch22 文件内**自行实现主循环**（新文件 + 独立 tilingKey），不得只新建文件再 include/调用 A5 通用类；验收以核时为准——只改通用分支或只包一层文件的，视为未完成。

对纯搬运、scatter/gather、cache 更新这类简单算子，正确性通过后，**在会话内对当前已验证的 case 做一次 profiling**，目的是定位瓶颈与实现缺陷，而不是做 A/B 或多 case 对比矩阵。

1. 用 `msprof` 采集一次 kernel 级数据，看 `Task Duration`、block 数、搬运/对齐是否合理；动态 shape 算子可能没有 `op_summary`，此时用 ACL event 计时并注明口径；
2. 重点判断有没有“本可避免”的开销：
   - 是否直接复用了 A5 的通用 kernel 类/模板（pipe 初始化、模板分支、冗余拷贝会带固定开销），简单算子优先写**专用 arch22 kernel**（模板见 `assets/simple-arch22-kernel.md`）；
   - 是否逐元素标量循环，改为 `DataCopyPad`/向量搬运并检查 512B 对齐；
   - block 数、tiling 切分是否与数据量匹配；
3. 只针对发现的具体缺陷改一版，改完再采一次确认；不做无关的第三方 case 扫描、不生成性能报告文档；
4. 若当前会话没有 profiling 工具或时间不足，记录“未做瓶颈自查”并说明原因，不要伪造数据。

## Step 5.6 证据齐备

对照 `performance-tuning-a2.md` §3 的 10 项清单逐项打勾；缺项即 BLOCKED。

## 完成判据

- 口径明确，基线可比。
- 代表用例有中位耗时、主 Bound、归因。
- 至少一轮有效优化或"已在上限"的证据。
- 10 项证据齐备。

## Gate

```
[GATE-5] PERF=COLLECTED TARGET=met EVIDENCE=10
[GATE-5] PERF=COLLECTED TARGET=not-met EVIDENCE=10   # 附原因与上限证据
[GATE-5] PERF=NOT_COLLECTED BLOCKED EVIDENCE=<n>
```

"not-met" 不等于失败：给出瓶颈与上限证据、整改项，可作为有条件交付；但不得写成 met。
