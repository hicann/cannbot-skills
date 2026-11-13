# A2 性能验证与调优

> 阶段 5 的手册。"可运行"靠阶段 3–4，"高性能"靠这里。性能结论只认 **arch22 实测**，且必须能解释为什么是这个数。

## 1. 验收口径

先按 `references/input-contract.md` §3 确认口径，再采集。没有口径时按 §7 的收敛规则判定，并在报告中标注 `TARGET=user-unspecified`。

**禁止**：拿 A5 的实测耗时当 A2 基线；拿 A5 的加速比直接套 A2；用单次计时（无 warmup、非中位数）下结论。

## 2. 采集协议（唯一入口：`ops-profiling`）

| 目的 | 方式 | 注意 |
|---|---|---|
| 可比较计时 | `msprof application` 标准计时：每例 10 组 `warmup → target`，取 target 中位数 | 不得改均值、不得丢前若干次、不得择优 |
| 瓶颈定位 | `msprof` 深度采集（`PipeUtilization` 等 7 组指标） | arch22 **支持**详细分析；按 `ops-profiling` 的 arch22 文档解读 |
| 基线对比 | `msprof_profile_run.sh --compare` | 基线与候选同 dtype、同 shape、同计时协议 |
| 快速摸底 | `--quick` | 只用于筛选，不能作为验收证据 |

- **禁止混用**：`msprof` 与 `msprof op` 的输出不能拼在一张表里；跨工具的数据只做趋势参考。
- **禁止用 A5 指标**：arch35 的 RegBase/VF/RVEC 指标在 arch22 不存在，缺失字段不按 0 处理，也不用于推断 A2 瓶颈。
- 采集环境要记录：CANN 版本、驱动、`NpuArch`（应为 2201）、device id、频率稳定性。

## 3. 证据清单（GATE-5 需要全部具备）

1. 用例清单：shape、dtype、layout、每例元素数。
2. 每例的中位耗时（含 warmup 说明）与极值/方差。
3. 基线数据（若口径是相对基线）：基线实现、版本、同环境同协议。
4. `PROF_GROUP_*` 原始归档目录（保留，不清理）。
5. `performance.json` + 报告（compare/quick 产物）。
6. 瓶颈归因表：每例的主 Bound 与依据（如 `aiv_mte2_ratio` > 50%）。
7. 加速比/利用率汇总（几何平均、按 dtype/shape 分组）。
8. 优化前后对比（至少一轮 A/B），含被回退的尝试与原因。
9. 复现命令。
10. 结论：达标 / 未达标原因 / 硬件受限证据。

缺任一项 → `[GATE-5] PERF=NOT_COLLECTED BLOCKED`。

## 4. 指标 → 瓶颈 → 手段

| 观测 | 主 Bound | 优先手段 |
|---|---|---|
| `aiv_vec_ratio` 高（>50%） | VEC | 减少无效 lane（tail padding）、合并小操作、换更高吞吐指令、减少 Cast 往返 |
| `aiv_scalar_ratio` 高（>30%） | SCALAR | 索引/地址计算外提、循环展开、减少分支、用 `Brcb`/`ArithProgression` 批量生成索引 |
| `aiv_mte2_ratio` / `aiv_mte3_ratio` 高（>50%） | MTE2 / MTE3 | 提高搬运粒度（≥512B）、对齐、双缓冲重叠、L2 复用、减少 GM 往返 |
| `aic_mte1_ratio` 高 | MTE1 | base 块与分形重排、L1 复用、`LoadData` 批量 |
| `aic_cube_ratio` 低而总时长高 | Cube 未打满 | baseM/baseN/baseK 调整、K 维展开、减少 fixpipe 等待 |
| `aic_fixpipe_ratio` 高（>15%） | Fixpipe | 检查输出地址对齐、`dstStride`、是否可批量回写 |
| `*_wait_ratio` 高（MTE/V 等待） | 同步/流水 | 双缓冲、事件合并、减少不必要 `PipeBarrier`、检查 flagId 冲突 |
| 各核 `aiv_time` 差异大 | 负载不均 | 重新切核（按 M/N/batch），尾核拿小片 |

> 指标名与判据以 `ops-profiling` 的 `references/csv_fields_reference.md` 为准；本文只列迁移场景最常见的组合。

## 5. A2 特有机会（来自"删除 A5 限制"）

A5 代码可能为规避 A5 的短板做了改写，迁移后**应逐项评估是否回退**（每项独立 A/B，不搭便车）：

| 机会 | 判定信号 | A/B 方法 | 回退条件 |
|---|---|---|---|
| 恢复 `TransDataTo5HD`（A2 高吞吐 `vnchwconv`） | 代码里有手写转置/Copy 重组替代 | 替换后比中位耗时与 `aiv_vec_ratio` | 精度失败或改善 <3% |
| 恢复常规归约 tiling（A5 为 R=16/24 做过特殊切分） | tiling 里有 16/24 特判 | 回退特殊分支，比 `aiv_scalar_ratio` 与耗时 | 同上 |
| 恢复 `BilinearInterpolation` API | 代码里有手写 Gather+乘加循环 | 替换后比耗时与精度 | 同上 |
| 恢复 int4 Mmad / 4:2 稀疏 | A5 侧 cast 成 int8、或删了稀疏逻辑 | 恢复后比耗时、带宽与精度 | 精度不达标（量化语义变化）即回退 |
| 恢复 L1→GM / GM→L0A/L0B 直连 | 代码里为绕开缺失多绕了 GM | 改直连后比 `aic_mte2_ratio` 与耗时 | 改善 <3% |
| 合并过碎 tiling（A2 L2 更大） | tiling 注释写着"为适配 950 L2" | 放大 tile 后比耗时与 L2 命中 | UB 超限或改善 <3% |

## 6. 调优顺序（先大后小）

1. **预算正确**：UB/L0C 不越界、tiling 不保守（先保证 tile 用满可用容量）。
2. **搬运**：对齐、粒度、双缓冲、减少 GM 往返（通常收益最大）。
3. **计算**：减少 Cast/无效 lane、合并 pass、换指令。
4. **并行**：核数切分、流水重叠、多阶段流水。
5. **微调**：循环展开、标量外提、flag 合并。

每一步只改一件事，记录 before/after 中位耗时与主指标。

## 7. 收敛与时间盒

- 单项优化：实现 + 最多 5 轮构建-采集；连续 3 轮改善 <3% 即停止并记录。
- 整体：主瓶颈消除后再判定"下一瓶颈"，最多 3 个瓶颈循环；仍不达标则输出"未达标 + 硬件/算法上限证据"。
- 上限估算：可用 `ops-theoretical-performance-estimation`（注意其当前仅支持 Ascend950 RegBase 级 Vector，**不能直接用于 A2**）；A2 的上限用带宽/算力手算并写明公式与取值来源。

## 8. 报告格式

```markdown
## 性能结果（arch22）
| case | shape | dtype | 中位耗时(us) | 基线(us) | 加速比 | 主 Bound | 关键指标 |
|---|---|---|---|---|---|---|---|

## 瓶颈归因
- case 1：MTE2 Bound（aiv_mte2_ratio=0.63），原因：…；已尝试：…；结论：…

## 优化记录
| 改动 | before(us) | after(us) | 改善 | 保留/回退 | 依据 |
|---|---|---|---|---|---|

## 结论
TARGET=met（口径：…）/ not-met（原因 + 上限证据）/ blocked（缺设备或环境）
```
