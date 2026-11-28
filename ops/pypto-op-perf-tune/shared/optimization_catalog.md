# PyPTO 算子性能优化点索引库

> 本文件是性能调优的**单一信息源**。编排器 ITER_START 阶段从这里选题，子技能 SKILL.md 只保留操作指南（怎么改），不重复优化点枚举。

## 使用方法

1. **编排器进入 PHASE 时**，按「一、按阶段分组」章节生成调优点清单
2. **ITER_START 选优化点时**，如已有性能数据，可先查「二、按症状索引」定位方向
3. **确定优化点编号后**，回到对应子技能 SKILL.md 的「调优方向」章节查看详细操作指南

**⛔ 全局验证与采纳规则（适用于全部 tile/unroll 类优化点：F-8/F-9/F-10/F-11/F-16/F-17/F-19/F-20/S-11/S-12 等）**：

1. **测量范围 = 精度集（via `run_cases.py`）**：`run_cases.py --cases <affected> --perf` 逐 case 测 E2E（使用 [perf-analyzer](../perf-analyzer/SKILL.md) 从各 case 的 `bubble_analysis.log` 提取），台账按 case 分列。精度集 + runner 由 INIT 确定/创建（见 tune-orchestrator/SKILL.md「covering set + runner 创建」）。
2. **分支验证（ITER_MODIFY 后、ITER_VERIFY 前）**：`run_cases.py --cases <affected> --compile-only`（纯编译路径）或 `run_cases.py --cases <affected>`（精度模式）。详见 tune-frontend/SKILL.md §5。
3. **分歧→分支细化**（非整体回退）：为回退 case 命中的分支条件增设/修改 if/elif/else 分支后重测。详见 basic-block-optimization.md §1b 采纳决策策略。
4. 单一信息源：§1b 采纳决策策略（adopt/revert/divergent 全表）+ tune-orchestrator/SKILL.md（covering set + runner 创建）+ tune-frontend/SKILL.md §5（分支验证流程）。

---

## 一、按阶段分组

### 开箱调优（tune-frontend）

| 编号 | 优化方向                                                    | 优先级 | 适用条件                                          | 详细指南                       |
| ---- | ----------------------------------------------------------- | ------ | ------------------------------------------------- | ------------------------------ |
| F-1  | 任务粒度检查                                                | ⭐⭐⭐ | 含 Matmul 算子                                    | tune-frontend SKILL.md §1.2   |
| F-2  | 循环体计算量                                                | ⭐⭐⭐ | 含内层 loop                                       | tune-frontend SKILL.md §1.2   |
| F-3  | 循环次数优化                                                | ⭐⭐⭐ | loop 次数 > 100                                   | tune-frontend SKILL.md §1.2   |
| F-4  | Reshape 全局优化（含 squeeze/unsqueeze → reshape inplace） | ⭐⭐⭐ | 有 reshape/squeeze/unsqueeze 或 shape 维度大于 2D | tune-frontend SKILL.md §2     |
| F-5  | 静态轴改 Python for                                         | ⭐⭐   | 有静态轴用 pypto.loop                             | tune-frontend SKILL.md §1.1   |
| F-6  | 合并独立 loop                                               | ⭐⭐   | 有多个独立 loop                                   | tune-frontend SKILL.md §1.3   |
| F-7  | 外层动态轴切块                                              | ⭐⭐   | 外层动态轴范围大                                  | tune-frontend SKILL.md §1.2.1 |
| F-8  | 内层 unroll                                                 | ⭐⭐⭐ | 内层动态轴范围大                                  | tune-frontend SKILL.md §1.2.2 |
| F-9  | Cube TileShape 设置                                         | ⭐⭐⭐ | 含 Matmul                                         | tune-frontend SKILL.md §3 → references/basic-block-optimization.md §2   |
| F-10 | Vector TileShape 设置                                       | ⭐⭐   | 含 Vector 计算                                    | tune-frontend SKILL.md §3 → references/basic-block-optimization.md §3   |
| F-11 | 常量配置调整                                                | ⭐     | 算子有 BLOCK_SIZE 等常量                          | tune-frontend SKILL.md §A2    |
| F-12 | 输入矩阵 NZ 格式                                            | ⭐     | 权重矩阵较大                                      | tune-frontend SKILL.md 局部§1 |
| F-13 | Transpose 优化                                              | ⭐     | 含 transpose+matmul                               | tune-frontend SKILL.md 局部§2 |
| F-14 | 冗余搬运消除                                                | ⭐     | 有 concat/assemble 搬运                           | tune-frontend SKILL.md 局部§3 |
| F-15 | 尾轴 Broadcast 合轴（combine_axis）                         | ⭐     | 存在尾轴为1的 broadcast 二元运算                  | tune-frontend SKILL.md 局部§4 |
| F-16 | 多 Matmul 差异化 Cube TileShape                              | ⭐⭐⭐ | 算子含2个及以上 Matmul（如 attention 的 QK^T + PV） | tune-frontend SKILL.md「多 Matmul 差异化 Cube TileShape」+ `scripts/list_tile_boundaries.py`（机械枚举 matmul×cube_tile_set 配对）   |
| F-17 | V 段内多 shape 分段 vec tile                                 | ⭐⭐⭐ | 同一 V 段内处理的 tensor shape 发生变化（如 [M,N]→[M,1]→[M,N]） | tune-frontend SKILL.md「V 段内多 shape 分段 vec tile」+ `scripts/list_tile_boundaries.py`（机械枚举 V 段 + cube→vector 边界，闭合清单防遗漏） |
| F-18 | 语义维度的静态循环保留                                       | ⭐⭐   | 有 n_kv/group 等语义维度的静态循环（即使迭代次数=1） | tune-frontend SKILL.md「语义维度的静态循环保留」 |
| F-19 | 内层批量化（循环 → 批量 tensor 操作 + 外提）                 | ⭐⭐⭐ | 循环体内 matmul M_current<8（通常=1）且可批量化进 M 轴；或共享输入可广播批量外提 | tune-frontend SKILL.md §1.2.4 |
| F-20 | 邻居算子感知的 TileShape 优化                                | ⭐⭐   | matmul→reduce / matmul→vector(无reduce) / 并发 matmul | tune-frontend SKILL.md「邻居算子感知的 TileShape 优化」+ `scripts/list_tile_boundaries.py`（机械枚举 matmul 邻居链） |
| F-21 | 操作链依赖深度外提                                          | ⭐⭐⭐ | 嵌套循环（≥2层）体内有依赖部分循环变量的操作链       | tune-frontend SKILL.md §2.1 → references/loop-hoisting.md                                |

### 深度调优（tune-swimlane）

| 编号 | 优化方向                                     | 优先级 | 适用条件                                                                            | 详细指南                                                 | 前置条件       |
| ---- | -------------------------------------------- | ------ | ----------------------------------------------------------------------------------- | -------------------------------------------------------- | -------------- |
| S-1  | 核使用率分析                                 | ⭐⭐⭐ | 所有算子                                                                            | tune-swimlane SKILL.md §3 → references/core-usage-load-balancing.md §1                             | 无（强制首选） |
| S-2  | 核填充（TileShape 调整增任务数）             | ⭐⭐⭐ | 核未满                                                                              | tune-swimlane SKILL.md §3 → references/core-usage-load-balancing.md §3                             | S-1 完成       |
| S-3  | 负载均衡分析                                 | ⭐⭐⭐ | 多子图算子                                                                          | tune-swimlane SKILL.md §3 → references/core-usage-load-balancing.md §4                             | S-2 完成       |
| S-4  | Vector 普通合图（sg_set_scope，只包裹 V 段） | ⭐⭐⭐ | 有连续 AIV 操作                                                                     | tune-swimlane SKILL.md §4 → references/merge-optimization.md §2.2                           | S-3 完成       |
| S-5  | Vector 自动合图（nbuffer）                   | ⭐⭐   | 短耗时 AIV 任务                                                                     | tune-swimlane SKILL.md §4 → references/merge-optimization.md §2.1                           | S-3 完成       |
| S-6  | Cube L1Reuse（消除重复搬运）                 | ⭐⭐   | AIC 核满、有重复搬运                                                                | tune-swimlane SKILL.md §4 → references/merge-optimization.md §3.1                           | S-3 完成       |
| S-7  | Cube CubeNBuffer（合并同构子图）             | ⭐⭐   | AIC 核满、短耗时                                                                    | tune-swimlane SKILL.md §4 → references/merge-optimization.md §3.2                           | S-3 完成       |
| S-8  | L1Reuse + CubeNBuffer 协同                   | ⭐     | 已有 S-6/S-7 基础                                                                   | tune-swimlane SKILL.md §4 → references/merge-optimization.md §3.3                           | S-6 或 S-7     |
| S-9  | Stitch 调优                                  | ⭐⭐   | 所有算子                                                                            | tune-swimlane SKILL.md §1                               | 无             |
| S-10 | 调度策略                                     | ⭐     | 依赖简单的上下游                                                                    | tune-swimlane SKILL.md §5                               | 无             |
| S-11 | Cube TileShape 深度调优                      | ⭐⭐   | Matmul 需减少重复载入/K 轴分核                                                      | tune-swimlane SKILL.md §2 → references/tileshape-deep-tuning.md §1                             | S-2 完成       |
| S-12 | Vector TileShape 深度调优                    | ⭐⭐   | Vector 计算需深度调优                                                               | tune-swimlane SKILL.md §2 → references/tileshape-deep-tuning.md §2                             | S-2 完成       |
| S-13 | Matmul 分核布局优化（L2 命中率优化）         | ⭐⭐   | 大 shape Matmul L2 命中率低、MTE2 带宽利用率不足                                    | tune-swimlane SKILL.md §6                               | S-2 完成       |
| S-14 | A5 Mix合图（CV 融合，消除 CV 间搬运）        | ⭐⭐⭐ | A5 平台（`npuarch == 'DAV_3510'`）+ 有 CV 间搬运的算子                            | tune-swimlane SKILL.md §4 + merge-optimization.md §4   | S-3 完成       |
| S-15 | ooo_sched_mode（CV 交替算子，非通用）        | ⭐⭐   | A5 平台（`npuarch == 'DAV_3510'`）+ 连续 Cube↔Vector 交替结构（如 attention 类） | tune-swimlane SKILL.md §7                               | S-3 完成       |
| S-16 | vf_options                                   | ⭐⭐   | A5 平台（`npuarch == 'DAV_3510'`）+ 含 Vector 计算的算子                          | tune-swimlane SKILL.md §8                               | S-3 完成       |
| S-17 | VF 融合编排原则                              | ⭐⭐   | A5 平台（`npuarch == 'DAV_3510'`）+ 含连续 vec 算子序列                           | tune-swimlane SKILL.md §8 + merge-optimization.md §5.2 | 无             |
| S-18 | sg_set_tunevf_mode（VF 调优 Pass 行为模式）  | ⭐⭐   | A5 平台（`npuarch == 'DAV_3510'`）+ 含 Vector 计算的算子                          | tune-swimlane SKILL.md §8 + merge-optimization.md §5.4 | 无             |
| S-19 | ready_on_host_tensors                        | ⭐⭐   | 算子有小 tensor 输入通过 AICPU gather 下发（如 block_table、actual_seq 等索引类 tensor） | tune-swimlane SKILL.md §9                               | 无             |
| S-20 | max_workspace_kb + host_options 完整性检查   | ⭐⭐⭐ | 所有算子（NPU 编译输出含 workspace 推荐值时强制）     | tune-swimlane SKILL.md §10 + 主 SKILL.md §2.1           | 无（S2阶段）   |
| S-21 | Mix合图多 scope 策略（Mix + 普通合图协同）   | ⭐⭐⭐ | A5 平台 + 最后一个 Cube 后的 V 段涉及跨迭代依赖 | tune-swimlane SKILL.md §11 + merge-optimization.md §4.1 Step 0b 规则3 | S-14 Step 0 完成 |

### 核内调优（tune-incore）

| 编号 | 优化方向                                 | 优先级 | 适用条件                  | 详细指南                 |
| ---- | ---------------------------------------- | ------ | ------------------------- | ------------------------ |
| I-1  | 小 Shape 矩阵乘                          | ⭐⭐⭐ | Matmul Shape 特殊         | tune-incore SKILL.md §1 |
| I-2  | L2 Cache 策略（权重矩阵 NONE_CACHEABLE） | ⭐⭐   | 含大型权重矩阵 / 融合算子 | tune-incore SKILL.md §2 |
| I-3  | 冗余计算消依赖                           | ⭐⭐   | 一对多子图依赖            | tune-incore SKILL.md §3 |
| I-4  | 尾轴长度优化                             | ⭐⭐   | 尾轴 < 32B 对齐           | tune-incore SKILL.md §4 |
| I-5  | TileOperation 实现检查                   | ⭐     | 上述优化无效时            | tune-incore SKILL.md §5 |
| I-6  | 操作数连续性检查                         | ⭐⭐   | TileOperation 输入非连续  | tune-incore SKILL.md §6 |
| I-7  | Gather/Scatter 数据搬运方向优化          | ⭐⭐   | 有 HBM↔L1 搬运瓶颈       | tune-incore SKILL.md §7 |
| I-8  | submit_before_loop 计算与搬运重叠        | ⭐⭐   | 子 loop 未正确提交        | tune-incore SKILL.md §8 |
| I-9  | valid_shape 尾块零填充避免               | ⭐⭐   | 尾块存在无效零填充计算    | tune-incore SKILL.md §9 |
| I-10 | 合并 gather 减少搬运次数                 | ⭐⭐⭐ | 多个 gather 取同一 source 不同段 | tune-incore SKILL.md §10 |
| I-11 | view 复用消除重复搬运                    | ⭐⭐⭐ | 已 gather 数据被独立 gather 再次取 | tune-incore SKILL.md §11 |

---

## 二、按症状索引

> 供 ITER_START 阶段根据性能指标快速定位优化方向。编号指向「一、按阶段分组」中的条目。

### 症状 A：气泡率 > 10%

| 优先级 | 优化点             | 所在阶段 | 操作速览                                                                       |
| ------ | ------------------ | -------- | ------------------------------------------------------------------------------ |
| ⭐⭐⭐ | F-2 循环体计算量   | 开箱     | 增大切块或 loop_unroll                                                         |
| ⭐⭐⭐ | F-3 循环次数优化   | 开箱     | 增大 tile size 或切块                                                          |
| ⭐⭐⭐ | F-8 内层 unroll    | 开箱     | `unroll_list=[64,16,4]`                                                      |
| ⭐⭐⭐ | F-21 操作链依赖深度外提 | 开箱     | 外提部分依赖操作链 → 减少循环内冗余计算 → 减少调度气泡（脚本 `list_tile_boundaries.py` 自动检测） |
| ⭐⭐⭐ | S-20 max_workspace_kb | 深度     | 从 NPU 输出提取推荐值设置 `max_workspace_kb`，激活 memory-driven mode         |
| ⭐⭐   | F-6 合并独立 loop  | 开箱     | 合并无数据依赖的独立 loop                                                      |
| ⭐⭐   | F-20 邻居感知 vector 和 cube TileShape | 开箱     | R1 增大 nL0 → 减少 matmul 任务数 → 减少依赖等待气泡；R3/R4 整shape tile → 减少 AIV 任务数 → 减少 CV 交接气泡 |
| ⭐⭐   | S-9 Stitch 调优    | 深度     | `stitch_function_max_num: 128`                                               |
| ⭐⭐⭐ | S-4 / S-5 普通合图 | 深度     | sg_set_scope 包裹 V 段 或 nbuffer                                              |
| ⭐⭐⭐ | S-14 A5 Mix合图    | 深度     | A5 平台：`pypto.experimental.auto_mix_partition(1)`（自动）和 `sg_set_scope`（手动）是同一功能的两种开关方式，优先尝试自动，当前方式充分调优后未达到预期目标性能或需要最优性能时再切换另一种（仅 `npuarch=='DAV_3510'`） |
| ⭐⭐⭐ | S-21 Mix多scope策略 | 深度     | A5 平台：涉及跨迭代依赖的 V 段从 Mix scope 放出，用独立 `sg_set_scope` 做普通合图；多段无数据依赖的 CV 段分独立 Mix scope                  |
| ⭐     | S-10 调度策略      | 深度     | `device_sched_mode` 调整                                                     |
| ⭐⭐⭐ | §4.3 A-D1~A-D3     | 算法     | ⚠️ S-14 配置级 Mix合图失败后，走算法级优化减少 DDR 往返（合并 gather / view 复用 / 消除中间 assemble），修复 CV 通路断点后重试 S-14。详见主 SKILL.md §4.3 |

### 症状 B：核心利用率 < 50%

| 优先级 | 优化点                                                          | 所在阶段 | 操作速览                              |
| ------ | --------------------------------------------------------------- | -------- | ------------------------------------- |
| ⭐⭐⭐ | F-1 任务粒度                                                    | 开箱     | 增大 Matmul M/N 轴                    |
| ⭐⭐⭐ | F-9 Cube TileShape                                              | 开箱     | 推荐配置                              |
| ⭐⭐⭐ | F-16 多 Matmul 差异化 Cube TileShape                            | 开箱     | C1/C2 按 M/N/K 独立配置 L1/K_L1       |
| ⭐⭐⭐ | F-17 V 段内多 shape 分段 vec tile                               | 开箱     | shape 变化时重设 vec tile              |
| ⭐⭐⭐ | F-19 内层批量化                                                 | 开箱     | matmul M 轴批量化（Mode A，自含 unroll）/ tile 扩展 + 操作外提（Mode B）——必做 |
| ⭐⭐⭐ | F-21 操作链依赖深度外提 | 开箱     | 外提部分依赖操作链至循环外 → 减少冗余计算 → 提升有效利用率（脚本 `list_tile_boundaries.py` 自动检测） |
| ⭐⭐⭐ | S-1 核使用率分析                                                | 深度     | `analyze_core_usage.py`             |
| ⭐⭐⭐ | S-2 核填充                                                      | 深度     | 减小 L0/L1 增加任务数                 |
| ⭐⭐⭐ | S-20 max_workspace_kb                                           | 深度     | 从 NPU 输出提取推荐值设置              |
| ⭐⭐   | F-4 Reshape 全局优化（含 squeeze/unsqueeze → reshape inplace） | 开箱     | `reshape(inplace=True)` 外提 + 合轴 |
| ⭐⭐   | F-7 外层动态轴切块                                              | 开箱     | 切块增加任务数                        |
| ⭐⭐   | F-10 Vector TileShape                                           | 开箱     | 设置 vec_tile_shapes                  |
| ⭐⭐   | F-20 邻居感知 vector 和 cube TileShape（边界 tile，区别于 F-9/F-10 基线） | 开箱     | R1 cube nL0 增大 / R2 L1 核数匹配 / R3 vec 跟随 cube L0 / R4 view·assemble 整shape |
| ⭐⭐   | S-12 Vector TileShape 深度调优                                  | 深度     | 对齐上下游 TileShape / 泳道图驱动     |
| ⭐     | S-10 调度策略                                                   | 深度     | `device_sched_mode`                 |
| ⭐⭐   | S-6 / S-7 Cube 合图                                             | 深度     | 核满后再启用                          |
| ⭐⭐⭐ | 主SKILL.md §4.3 A-D1~A-D3                                       | 算法     | ⚠️ 核利用率因 DDR 等待而偏低且配置调优无改善时，走算法级优化减少 DDR 往返（合并 gather / view 复用） |

### 症状 C：负载不均衡（AicoreTime 差异 > 20%）

| 优先级 | 优化点                                    | 所在阶段 | 操作速览                       |
| ------ | ----------------------------------------- | -------- | ------------------------------ |
| ⭐⭐⭐ | S-3 负载均衡分析                          | 深度     | 按 total(us) 排序 → 调整瓶颈  |
| ⭐⭐   | S-11 Cube TileShape 深度调优              | 深度     | 减小瓶颈子图 L0/L1             |
| ⭐⭐   | S-13 Matmul 分核布局优化（L2 命中率优化） | 深度     | 优化分核布局提升 L2 命中率     |
| ⭐⭐   | S-12 Vector TileShape 深度调优            | 深度     | 调整 Vector TileShape 均衡负载 |
| ⭐⭐⭐ | S-4 手动合图                              | 深度     | sg_set_scope 合并子图          |

### 症状 D：单 task 耗时过长

| 优先级 | 优化点                               | 所在阶段 | 操作速览                                  |
| ------ | ------------------------------------ | -------- | ----------------------------------------- |
| ⭐⭐⭐ | I-1 小 Shape 矩阵乘                  | 核内     | Vector 预处理 reshape                     |
| ⭐⭐   | I-2 L2 Cache（融合算子批量设置权重） | 核内     | 所有权重同时`NONE_CACHEABLE`            |
| ⭐⭐   | I-3 冗余计算消依赖                   | 核内     | 复制数据使分支独立                        |
| ⭐⭐   | I-4 尾轴长度优化                     | 核内     | concat/transpose 增大尾轴                 |
| ⭐⭐   | I-9 valid_shape 尾块零填充避免       | 核内     | `valid_shape` 标记有效数据范围          |
| ⭐⭐   | I-6 操作数连续性检查                 | 核内     | reshape/transpose 修复非连续输入          |
| ⭐⭐   | I-7 Gather/Scatter 搬运方向优化      | 核内     | HBM→L1 用 cube_tile，L1→HBM 用 assemble |
| ⭐⭐   | I-8 submit_before_loop 重叠          | 核内     | `submit_before_loop=True`               |
| ⭐⭐⭐ | I-10 合并 gather 减少搬运次数        | 核内/算法| 拼接 source 一次 gather 取多列段        |
| ⭐⭐⭐ | I-11 view 复用消除重复搬运           | 核内/算法| `view(已有tensor)` 替代独立 gather      |
| ⭐     | I-5 Operation 检查                   | 核内     | 与 Ascend C 对比                          |

---

## 三、优化点全表（供调优点清单生成）

> 每个优化点的完整信息，编排器据此生成调优点清单。
>
> **⚠️ 编号顺序说明**：F-16~F-21 为后期新增的开箱调优点，排列在 I-11 之后（本节末尾），但逻辑上属于 F 系列（开箱阶段），应在 F-15 之后查阅。S-20~S-21 同理，属于 S 系列（深度阶段），应在 S-19 之后查阅。

### [F-1] 任务粒度检查

- **阶段**: 开箱调优
- **优先级**: ⭐⭐⭐ P0
- **适用条件**: 算子包含 Matmul
- **检查方法**: 检查 Matmul 的 M/N/K 轴是否充分利用硬件，M 轴 < 8 是常见问题
- **操作指南**: tune-frontend SKILL.md §1.2（切块 + unroll）
- **典型收益**: 5-50%（取决于粒度差异）
- **约束**: 切块大小不应超过 shape 中该维度的大小
- **关联优化**: F-7（外层切块）、F-8（内层 unroll）

### [F-2] 循环体计算量

- **阶段**: 开箱调优
- **优先级**: ⭐⭐⭐ P0
- **适用条件**: 含内层 loop
- **检查方法**: 检查循环体内部的计算量是否太小，用不满算力
- **操作指南**: tune-frontend SKILL.md §1.2
- **典型收益**: 10-30%
- **解决方案**: 开启 loop_unroll 或增加切分块大小
- **关联优化**: F-1（任务粒度）、F-8（内层 unroll）

### [F-3] 循环次数优化

- **阶段**: 开箱调优
- **优先级**: ⭐⭐⭐ P0
- **适用条件**: loop 次数 > 100
- **检查方法**: 检查循环总次数，过多会导致调度开销大
- **操作指南**: tune-frontend SKILL.md §1.2
- **典型收益**: 5-20%
- **解决方案**: 切块减少循环次数
- **关联优化**: F-7（外层切块）、F-1（任务粒度）

### [F-4] Reshape 全局优化（含 squeeze/unsqueeze → reshape inplace）

- **阶段**: 开箱调优
- **优先级**: ⭐⭐⭐ P0
- **适用条件**: 算子中存在 `pypto.reshape`/`pypto.squeeze`/`pypto.unsqueeze` 调用，或循环体内参与计算的 tensor shape 维度超过 2D
- **检查方法**: 使用 `grep -nE "pypto\.(reshape|squeeze|unsqueeze)"` 获取所有相关调用，逐行分析每个操作是否必须出现在最内层循环体中（参考 references/reshape-global-optimization.md §1 分析表格）
- **操作指南**: tune-frontend SKILL.md §2
- **典型收益**: 5-30%
- **优化方式**:
  - **方式 1（原始输入 reshape 外提）**：对原始输入（函数参数）的 reshape，挪到算子入口（所有 loop 之前），使用 `inplace=True`，避免循环内重复数据拷贝
  - **方式 2（高维计算提前合轴）**：循环体内计算超过 2D 时，进入循环前对原始输入 `reshape inplace` 合轴为 2D，避免循环体内出现 reshape
  - **方式 3（冗余 reshape 删除）**：检查源 shape 是否等于目标 shape 的冗余 reshape（常见于分析阶段误操作的残留），直接删除无效 reshape 调用，消除不必要的数据搬运
  - **方式 4（squeeze/unsqueeze 替换为 reshape inplace 外提）**：`pypto.squeeze`/`pypto.unsqueeze` 不支持 `inplace=True`。对原始输入的 squeeze/unsqueeze 操作，替换为等价 `pypto.reshape(..., inplace=True)` 并挪到算子入口（所有 loop 之前），消除循环内的重复搬运（详见 references/reshape-global-optimization.md §2 方式4）
- **约束**: 只有原始输入（函数参数）可用 `reshape(inplace=True)`，中间结果和输出 tensor 不能 inplace reshape；输出 tensor inplace reshape 会导致切片写入索引断裂（输出全零）
- **关联优化**: F-9（Cube TileShape）、F-10（Vector TileShape）、F-21（通用操作链依赖深度外提——非 reshape/squeeze/unsqueeze 的部分依赖操作链由其处理）

### [F-5] 静态轴改 Python for

- **阶段**: 开箱调优
- **优先级**: ⭐⭐ P1
- **适用条件**: 静态轴使用了 `pypto.loop`
- **检查方法**: 搜索代码中 `pypto.loop` 调用，判断循环轴是否为静态（编译期可知的常量）
- **操作指南**: tune-frontend SKILL.md §1.1
- **典型收益**: 3-10%
- **代码示例**: `for i in pypto.loop(n, ...)` → `for i in range(n):`

### [F-6] 合并独立 loop

- **阶段**: 开箱调优
- **优先级**: ⭐⭐ P1
- **适用条件**: 有多个独立的 pypto.loop
- **检查方法**: 检查是否有循环体无数据依赖的独立 loop
- **操作指南**: tune-frontend SKILL.md §1.3
- **典型收益**: 3-15%
- **约束**: 合并的操作之间无数据依赖冲突

### [F-7] 外层动态轴切块

- **阶段**: 开箱调优
- **优先级**: ⭐⭐ P1
- **适用条件**: 外层动态轴范围较大
- **检查方法**: 检查外层 loop 的动态轴数值范围
- **操作指南**: tune-frontend SKILL.md §1.2.1
- **典型收益**: 10-40%
- **约束**: 切块大小从较大值开始尝试（如 64, 32, 16），不应超过 shape 中该维度的大小
- **代码示例**: `pypto.loop(b // b_block_size, ...)`

### [F-8] 内层 unroll

- **阶段**: 开箱调优
- **优先级**: ⭐⭐⭐ P1
- **适用条件**: 内层动态轴范围较大
- **检查方法**: 检查最内层 loop 的动态轴数值范围
- **操作指南**: tune-frontend SKILL.md §1.2.2
- **典型收益**: 5-30%
- **约束**: loop_unroll 必须放在最内层循环；unroll_list 最大值不要超过循环次数
- **⚠️ 副作用**: unroll 减小（如从 8 降到 1）会增加 task 数量和调度开销，可能导致核上 compute（AICore E2E Time）上升——更多 task 意味着更多核间同步等待。实测案例：unroll=8（1 task/batch）AICore E2E=102us，unroll=1（8 task/batch）AICore E2E=167us。选择 unroll 值时须以 AICore E2E Time 为准，不能仅看 wall time
- **⛔ unroll_list 编译爆炸防护（强制）**: `unroll_list` 的变体数必须随被批量化维度的大小自适应。当每迭代被批量化维度（不同算子小至 ≈4、大至 ≈128；非 matmul 循环则为每迭代 tensor 大小）已足够大（≥ CUBE_ALIGN = cube tile 最小对齐（§一），或 vector-only 循环的 tensor 已填满 UB），则完整 7 变体列表会产生过大的中间 tensor（如 unroll 64 × 每迭代维度 128 = 8192）→ 每个变体编译巨大 kernel tiling plan → 多变体导致 JIT 编译超时。规则：随被批量化维度增大而减少变体数：
  - 每迭代维度 < CUBE_ALIGN：`[64, 32, 16, 8, 4, 2, 1]`（完整批量化）
  - 每迭代维度 ≥ CUBE_ALIGN，大 loop_count（如 prefill 类 case）：`[large_unroll, 1]`（2 变体——批量主体迭代，单次处理尾块；具体 large_unroll 值须在目标 case 上实测以避免编译爆炸）
  - 每迭代维度 ≥ CUBE_ALIGN，小 loop_count（如 decode 类 case，loop_count=1）：`[1]`（单变体）
  - 非 matmul 循环：检查每迭代 tensor 大小是否已填满 UB——若是，则无 unroll 收益，使用 `[1]`
  此约束在 F-8 设定 `unroll_list` 和 F-19 Mode A Step 1 引入 `loop_unroll` 时均须检查。
- **⛔ 全 case 普遍性检查（unroll_list 选择强制）**: 设定 `unroll_list` 前必须枚举**本实现所辖 case 集合**的全部 case 的循环迭代数（loop_count——出现在 view shape 中的 kernel int 参数，如序列长度或序列长度×组数等语义维度，可用于 if 条件）和每迭代被批量化维度（M_current）。找出 MAX(loop_count) 跨全部 case。unroll_list 的最大值 × MAX(loop_count) 产生的 root function 数不得导致 JIT 编译超时（经验阈值：root function 数 > 512 时风险高；> 1024 几乎必然超时）。
- **⛔ fallback（if/elif/else on loop_count）**: unroll_list 不可全局硬编码单一值——必须用 if/elif/else 分支（分支条件用 loop_count，2 的幂边界点）为不同 loop_count 范围选择不同 unroll_list。大 loop_count → 小 unroll（限 root function 数）；小 loop_count（decode 类 case，loop_count=1）→ `[1]`。unroll_list 是 `pypto.loop` kwarg（非 `set_*_tile_shapes` 参数），lint 不检查——if/elif/else 分支 + list Assign 自由使用。

```python
# ✅ if/elif/else fallback（2 的幂边界点，per-case-range unroll_list）
if S1 >= 1024:
    ulist = [4, 2, 1]         # 1024/4=256 root funcs — bounded
elif S1 >= 256:
    ulist = [16, 8, 4, 2, 1]  # 256/16=16 — optimal for mid-range
elif S1 >= 64:
    ulist = [8, 4, 2, 1]
else:
    ulist = [1]               # decode S1=1 — no unroll
for idx, unroll in pypto.loop_unroll(0, t, 1, name="LOOP", unroll_list=ulist):
    ...
```

- **循环消除策略（当 loop_count 跨 case 变化时）**: 循环可全批量消除（单一 matmul M = loop_count，1 root function）当且仅当满足以下任一条件：
  1. **全部 case 的 loop_count 均有界小**——编译开销显然可控，无需测试
  2. **MAX(loop_count) 较大但循环体内操作链简短**——须用 MAX(loop_count) 值实际编译测试通过方可批准（编译爆炸风险取决于 M_max × 操作链长度 × tensor 规模，非 M_max 单独）
  不满足时：保留 `loop_unroll` + `unroll_list`（if/elif/else on loop_count 选择 `unroll_list` 使 M_new = unroll × M_current ≤ 16）。
  ⛔ for-range 全展开（O(loop_count) root function, each M=1）始终禁止——编译开销 O(loop_count) 不可控。
  tile 约束（UB/L1）由 P2 tile 优化按 M_new 推导覆盖。
  示例：M_current ∈ {4, 8, 128} × 15 ops → M_max=128 × 15 ≈ 可控 → 编译测试通过 → 全消除；loop_count ∈ {1, 256} × 12-op softmax 链 → M_max=256 × 12 = 爆炸风险 → 编译测试若失败 → 保留 loop_unroll + unroll_list。
- **代码示例**: `pypto.loop(n, unroll_list=[8, 4, 2, 1], ...)`

### [F-9] Cube TileShape 设置

- **阶段**: 开箱调优
- **优先级**: ⭐⭐⭐ P2
- **适用条件**: 含 Matmul（Cube 计算）
- **检查方法**: 检查是否设置了 `set_cube_tile_shapes`，配置是否为推荐值
- **操作指南**: tune-frontend SKILL.md §3 → references/basic-block-optimization.md §2（Cube TileShape 设置规范）
- **值输出形式**: cube tile 值按 basic-block-optimization.md §2「Cube Tile 公式推导方法」推导为分档分支值：buffer 容量参数与 basis 推导（fp32 [128,128,128]、fp16 [128,256,128]，贪心加倍）→ 8-case 分裂模板（逐轴 T=128 分类，M 永远是分支、禁止常量 CUBE_ALIGN）→ 归一化（CUBE_ALIGN 对齐 + floor-pow2，pow2 仅 cube）→ L0C clamp → 普遍性检查（本实现所辖 case 集合逐一代入）。禁止推荐配置字面值照抄。放置见 §1b。
- **典型收益**: 5-20%
- **推荐配置**: `[128, 128], [64, 256], [256, 256]` 或 `[256, 256], [64, 256], [128, 128]` 或 `[128, 128], [128, 128], [128, 128]`
- **约束**: L1 不超过实际轴长；`L0 <= L1` 且 `L1 % L0 == 0`；BF16 下 L0/L1 需 CUBE_ALIGN 元素对齐（§一）；多 Matmul 时每个独立设置
- **Decode M=1 特殊配置**: 使用 K 轴三维配置 `[kL0, kAL1, kBL1]`，让 A 矩阵完全驻留 L1（kAL1=K），B 矩阵分批加载（kBL1=256）
- **Double Buffer**: 推荐配置可满足 L0 buffer 约束，自动开启 Double Buffer

### [F-10] Vector TileShape 设置

- **阶段**: 开箱调优
- **优先级**: ⭐⭐ P2
- **适用条件**: 含 Vector 计算
- **检查方法**: 检查是否设置了 `set_vec_tile_shapes`
- **操作指南**: tune-frontend SKILL.md §3 → references/basic-block-optimization.md §3（Vector TileShape 设置规范）
- **值输出形式**: vec tile 按 basic-block-optimization.md §3「Vector Tile 公式推导方法」从本实现所辖 case 集合的 shape 量级分析推导为分档分支值（§1b 四步流程）：列维 = 归约轴/scatter_ 轴长度（不切分）或 512B 对齐切分，行维 = 「不超过 M」与「UB 预算除以（col × dtype_bytes）」中较小值、下限 1（dtype 随 cast 变化重算）；归一化**仅 CUBE_ALIGN 对齐/512B 对齐，不做 pow2**（pow2 为 cube 专属）；禁止单 case 字面值硬编码。放置见 §1b。
- **议程来源**: 阶段A「A4 分段优化议程表」（tune-frontend SKILL.md §A4）以结构性判定提供本优化点在各段/边界的适用性与优先级（附证据），本步骤的实际执行在阶段C ITER 中完成（实测+精度回归）。
- **典型收益**: 3-10%
- **推荐配置**: 根据实际数据维度设置，不可使用固定小值。设置方法：
  1. **第一维（行方向）**：取对应 tensor 的实际行数或其整数倍缩小（如 tensor 为 [128, 512]，第一维可取 128 或 64/32 等约数）
  2. **第二维（列方向/尾轴）**：优先用满尾轴（等于 tensor 的列数），尾轴过大时建议按 512B 对齐切分（最低要求 32B 对齐）；归约类计算不在归约轴上切分（第二维 = 实际归约轴长度）
  3. **shape 变化时重设**：V 段内 tensor shape 从 [M,N] 变为 [M,1] 或反向变化时，必须重新设置匹配当前 shape 的 vec tile（见 F-17）
  4. **reshape 前后分别设置**：reshape 前按源 shape 设置，reshape 后按目标 shape 重设
  5. **参考起始值**：常见最优配置如 [128, 512]、[128, 128]、[64, 128] 等，具体取决于 tensor 实际维度
- **约束**: 优先用满尾轴；尾轴过大建议按 512B 对齐切分（最低要求 32B 对齐）；归约类计算不在归约轴上切分
- **调优方向链（用满尾轴之后的调整顺序）**: 用满尾轴 → UB_size（§一）超限时优先调小第一维（无 reduce 操作时可尝试切分尾轴，建议 512B 对齐，最低 32B）→ 性能不优时调整第一维（增大减少循环 overhead / 减小增加 task 并行度）→ Mix合图场景须与 cube L0 衔接轴相等或整数倍 → 多 V 段共享 UB 时各段 vec tile 总并发占用不超 UB_size（§一），在段间平衡分配
- **尾轴切分规则**: 后续 vec 操作含 reduce 操作时，尾轴一般不切（第二维 = 实际归约轴长度，切分会产生跨子图 reduce 开销）；后续无 reduce 操作时，可尝试切分尾轴验证性能（建议按 512B 对齐，最低 32B）
- **reshape 前后重设规则**: reshape 前按源 shape 设置 vec_tile，reshape 后必须按目标 shape 重设 vec_tile，尤其 assemble 操作前必须重设，否则会出错
- **冗余设置检查**: 合并连续相同的 `set_vec_tile_shapes` 为一次调用（常见 copy-paste 残留），减少冗余配置指令；同时检查每个 vec_tile_shapes 是否与对应 tensor shape 匹配，不匹配的及时修正

### [F-11] 常量配置调整

- **阶段**: 开箱调优
- **优先级**: ⭐ P3
- **适用条件**: 算子中有 BLOCK_SIZE 等硬编码常量
- **检查方法**: 搜索算子代码中的常量定义（如 BLOCK_SIZE_KV、TILE_SIZE 等）
- **操作指南**: tune-frontend SKILL.md §A2
- **值输出形式**: BLOCK_SIZE 等常量同样按 basic-block-optimization.md §1b 四步流程以 case 标量分档（如某 case 标量达到 64 阈值时取 64、其余范围按更小边界分支 + fallback 兜底），禁止单一 case 字面值硬编码。放置见 §1b。
- **典型收益**: 3-15%
- **常见调整值**: BLOCK_SIZE 可尝试 16/32/64/128

### [F-12] 输入矩阵 NZ 格式

- **阶段**: 开箱调优
- **优先级**: ⭐ P4
- **适用条件**: 权重矩阵 Shape 较大
- **检查方法**: 检查输入矩阵是否可以提前以 NZ 格式存储
- **操作指南**: tune-frontend SKILL.md 局部§1
- **典型收益**: 5-15%
- **原理**: NZ 格式的数据搬运到 L1 的带宽更高

### [F-13] Transpose 优化

- **阶段**: 开箱调优
- **优先级**: ⭐ P4
- **适用条件**: 含 transpose + matmul 结构
- **检查方法**: 搜索 transpose 操作后紧跟 matmul 的模式
- **操作指南**: tune-frontend SKILL.md 局部§2
- **典型收益**: 3-10%
- **解决方案**: 通过 matmul 的 `a_trans` / `b_trans` 参数融合 transpose，当 M 轴较大 N 轴较小时更换左右矩阵并使用转置配置

### [F-14] 冗余搬运消除

- **阶段**: 开箱调优
- **优先级**: ⭐ P4
- **适用条件**: 有 concat / assemble 等数据搬运操作
- **检查方法**: 检查是否有不合理数据操作导致的冗余搬运
- **操作指南**: tune-frontend SKILL.md 局部§3
- **典型收益**: 3-10%
- **解决方案**: 更换 concat 为 assemble

### [F-15] 尾轴 Broadcast 合轴（combine_axis）

- **阶段**: 开箱调优
- **优先级**: ⭐ P4
- **适用条件**: 算子中存在尾轴为1的 tensor 参与 broadcast 二元运算（如 `[M,1] * [M,N]`）
- **检查方法**: 扫描算子中所有 tensor shape，标记 shape 尾轴为 1 的 tensor，检查其参与的所有二元运算（mul/add/sub/div）另一侧尾轴是否 >1
- **操作指南**: tune-frontend SKILL.md 局部§4
- **典型收益**: 0-5%（Cube 密集型算子无收益，Vector 密集型预期更高）
- **配置方式**: 在 JIT 函数体首行添加 `pypto.experimental.set_operation_options(combine_axis=True)`
- **约束**:
  - 尾轴 broadcast 输入尾轴**必须连续**，否则功能失效
  - `pypto.sum(keepdim=True)` / `pypto.amax(keepdim=True)` 输出保证连续，符合条件
  - 若前序是 COPY_IN，需在前端保证 GM 连续
  - 设置是**局部**的，只影响当前 jit/loop 作用域
- **典型案例**: Pangu 7B Fused Layer online softmax 中 `[4,128] * [4,1]` 和 `[4,128] / [4,1]` → combine_axis 启用 brcb inline，但 Cube 占主导时无显著收益

---

### [S-1] 核使用率分析

- **阶段**: 深度调优
- **优先级**: ⭐⭐⭐ P0（强制首选）
- **适用条件**: 所有算子
- **检查方法**: 运行 `analyze_core_usage.py` 统计每个 leafHash 占用的 core 数量
- **操作指南**: tune-swimlane SKILL.md §3 → references/core-usage-load-balancing.md §1
- **输出**: 每个 leafHash 的核使用率（used/total），判定 FULL / NOT FULL
- **后续路径**: 核未满 → S-2；核已满 → S-3
- **关联优化**: S-2（核填充）、S-11（Cube TileShape 深度调优）

### [S-2] 核填充（TileShape 调整增任务数）

- **阶段**: 深度调优
- **优先级**: ⭐⭐⭐ P0
- **适用条件**: S-1 分析后有 NOT FULL 子图
- **前置条件**: S-1 完成
- **检查方法**: 对 NOT FULL 子图，运行 `leafhash_to_code.py` 定位代码，减小 nL0/nL1 增加任务数
- **操作指南**: tune-swimlane SKILL.md §3 → references/core-usage-load-balancing.md §3
- **约束**: `nL0 <= nL1 && nL1 % nL0 == 0`；逐步减小（如 256→128→64），每步实测
- **值输出形式**: 调整后的 tile 值仍须满足 §1b 分档分支形式——修改分支 cap/边界参数并保留分支结构（退化见 §1b）
- **⛔ 禁止**: 以结构限制为由跳过核填充，必须尝试完所有轴的 TileShape 调整

### [S-3] 负载均衡分析

- **阶段**: 深度调优
- **优先级**: ⭐⭐⭐ P1（核填充后强制执行）
- **适用条件**: 多子图算子
- **前置条件**: S-2 完成
- **检查方法**: 按 total(us) 降序排列所有子图，识别瓶颈子图，量化差距（>20% 必须优化）
- **操作指南**: tune-swimlane SKILL.md §3 → references/core-usage-load-balancing.md §4
- **达标条件**: 瓶颈子图 total 与次大 total 差距 < 20%，或连续 3 轮调整无法改善
- **约束**: 每次只调整一个子图的 TileShape

### [S-4] Vector 手动合图（sg_set_scope）

- **阶段**: 深度调优
- **优先级**: ⭐⭐⭐ P2
- **适用条件**: 有连续 AIV 操作（有直接数据依赖、同循环层级、无 Cube 夹杂）
- **前置条件**: S-3 完成
- **检查方法**: 运行 `analyze_aiv_dep_chains.py` 分析 AIV 依赖链，用 `leafhash_to_code.py` 映射到代码行
- **操作指南**: tune-swimlane SKILL.md §4 → references/merge-optimization.md §2.2
- **约束**: 仅对有直接上下游数据依赖的 Vector 操作生效；不包裹 Cube 操作；不跨 loop 边界
- **代码连续性调整**: sg_set_scope 前需调整前端代码顺序，移除无关操作使待合并操作相邻；PyPTO 是声明式的，只要依赖关系不变可调整顺序
- **⛔ 最易跳过的优化项**: 如果跳过此项，必须说明具体原因

### [S-5] Vector 自动合图（nbuffer）

- **阶段**: 深度调优
- **优先级**: ⭐⭐ P3
- **适用条件**: 短耗时（<10us）AIV 任务
- **前置条件**: S-3 完成
- **检查方法**: 运行 `analyze_swimlane.py` 查看 `[AIV]` 部分，确认短耗时任务
- **操作指南**: tune-swimlane SKILL.md §4 → references/merge-optimization.md §2.1
- **配置示例**: `pass_options={"vec_nbuffer_setting": {-1: 2}}`
- **调优方法**: 可先用 `{-1: N}` 全局配置，再按 psgId 精细调优

### [S-6] Cube L1Reuse（消除重复搬运）

- **阶段**: 深度调优
- **优先级**: ⭐⭐ P3
- **适用条件**: AIC 核满、matmul 的 M 或 N 轴进行了切分（存在重复搬运）
- **前置条件**: S-3 完成
- **检查方法**: 运行 `analyze_swimlane.py` 查看 `[AIC]` 部分，优先对 total 耗时大且有重复搬运的子图调优
- **操作指南**: tune-swimlane SKILL.md §4 → references/merge-optimization.md §3.1
- **配置示例**: `pass_options={"cube_l1_reuse_setting": {-1: 2, 0: 8}}`
- **调优方法**: t/iter 越大，L1 复用收益越高，可设更大粒度

### [S-7] Cube CubeNBuffer（合并同构子图）

- **阶段**: 深度调优
- **优先级**: ⭐⭐ P3
- **适用条件**: AIC 核满、同构子图数量多且每个 task 执行耗时很短（<10us）
- **前置条件**: S-3 完成
- **检查方法**: 运行 `analyze_swimlane.py`，avg<10us 且 t/iter≥2 的组优先设置
- **操作指南**: tune-swimlane SKILL.md §4 → references/merge-optimization.md §3.2
- **配置示例**: `pass_options={"cube_nbuffer_setting": {-1: 2}}`
- **⛔ 风险**: 不要使用空字典 `{}` 自动模式，可能过度合图导致性能严重退化

### [S-8] L1Reuse + CubeNBuffer 协同

- **阶段**: 深度调优
- **优先级**: ⭐ P3
- **适用条件**: 已有 S-6 或 S-7 的配置基础
- **前置条件**: S-6 或 S-7 完成
- **操作指南**: tune-swimlane SKILL.md §4 → references/merge-optimization.md §3.3
- **约束**: 两者不宜同时设置过大，会导致单个子图过大、L1/UB 内存争用；优先调 cube_l1_reuse_setting，再调整 cube_nbuffer_setting

### [S-9] Stitch 调优

- **阶段**: 深度调优
- **优先级**: ⭐⭐ P4
- **适用条件**: 所有算子
- **检查方法**: 查看当前 `stitch_function_max_num` 配置
- **操作指南**: tune-swimlane SKILL.md §1
- **配置示例**: `runtime_options={"stitch_function_max_num": 128}`
- **调优方法**: 在内存资源允许的前提下逐步增大，结合泳道图和端到端耗时调整
- **与 max_workspace_kb 的关系**: `max_workspace_kb`（S-20）优先于 `stitch_function_max_num`。`max_workspace_kb` 激活 memory-driven mode 后，编译器自动管理 stitch 并行度，此时 `stitch_function_max_num` 通常无需额外设置。仅在 `max_workspace_kb` 未设置或设置后仍有 stitch 瓶颈时，才单独调优 `stitch_function_max_num`

### [S-10] 调度策略

- **阶段**: 深度调优
- **优先级**: ⭐ P4
- **适用条件**: 上下游子图之间依赖较为简单，或下游子图输入 Tensor 的 L2 命中率较为重要。**device_sched_mode 是公共场景的配置项**，非 Mix合图专属
- **操作指南**: tune-swimlane SKILL.md §5
- **配置示例**: `runtime_options={"device_sched_mode": 1}`
- **调优方法**: 尝试不同调度策略，值域范围 [0, 3]
- **与其他参数的关系**: `device_sched_mode` 是公共调度配置，与 Mix合图（S-14）/ stitch（S-9）/ max_workspace_kb（S-20）无互斥关系，可独立调优

### [S-11] Cube TileShape 深度调优

- **阶段**: 深度调优
- **优先级**: ⭐⭐ P2
- **适用条件**: Matmul 需减少重复载入 / K 轴分核
- **前置条件**: S-2 完成
- **操作指南**: tune-swimlane SKILL.md §2 → references/tileshape-deep-tuning.md §1
- **优化手段**:
  - **减少重复载入**（增大 L1 或用 `[kL0, kAL1, kBL1]` 让 A 矩阵驻留 L1）
  - **K 轴分核**（`enable_split_k=True` 自动切 K 但有非确定性计算问题；或前端手动 loop 切 K，无确定性计算问题，详见 references/tileshape-deep-tuning.md §1.2）
- **值输出形式**: 深度调优的 tile 修改仍遵循 F-9 分档分支形式——修改分支 cap/边界参数（退化见 §1b）

### [S-12] Vector TileShape 深度调优

- **阶段**: 深度调优
- **优先级**: ⭐⭐ P2
- **适用条件**: Vector 计算需深度调优 TileShape
- **前置条件**: S-2 完成
- **操作指南**: tune-swimlane SKILL.md §2 → references/tileshape-deep-tuning.md §2
- **优化原则**:
  - **上下游对齐**: 下游 Vector Operation 的 TileShape 尽可能使用上游 Operation 的输出 TileShape，减少子图边界
  - **泳道图驱动**: 并行核数较少（<一半 Vector 核）时减小 TileShape；子图耗时短、调度开销占比高时增大 TileShape
  - **Cube 协同**: 调整相邻 Cube 和 Vector Operation 的 TileShape，使依赖更简单
  - **归约轴不切**: 归约类计算尽可能不在归约轴上切分
- **值输出形式**: 深度调优的 vec tile 修改仍遵循 F-10 分档分支形式——修改分支 cap/边界参数（退化见 §1b）

### [S-13] Matmul 分核布局优化（L2 命中率优化）

- **阶段**: 深度调优
- **优先级**: ⭐⭐ P2
- **适用条件**: 大 shape Matmul（M、N、K 均较大），L2 命中率偏低、MTE2 带宽利用率不足
- **前置条件**: S-2 完成
- **操作指南**: tune-swimlane SKILL.md §6
- **优化原理**: L2 命中率由单轮次分核数 mDim、nDim 和 mL1、nL1 共同决定。最优条件为 `nDim·nL1 = mDim·mL1`，使 M、N 轴分核到 L1 的数据量相等，L2 复用最大化
- **优化方法**: 在 M、N 轴外层添加一层 loop，手动控制每轮 M 和 N 的计算范围，从而控制分核数（详见 swimlane §6）
- **典型收益**: 大 shape（M=N=K=6144）+31%（2.1ms→1.6ms）

### [S-14] A5 Mix合图（CV 融合，消除 CV 间搬运）

- **阶段**: 深度调优
- **优先级**: ⭐⭐⭐ P2+（A5 平台强制考虑）
- **适用条件**: A5 平台（`pypto.platform.npuarch == 'DAV_3510'`）+ 算子存在连续 Cube↔Vector 交替结构
- **前置条件**: S-3 完成
- **检查方法**: 确认平台为 A5；分析泳道图确认存在 CV 间搬运开销（MTE2/AIV 等待 AIC 或反之）；确认算子 CV 衔接形态/shape/tile/dtype 满足 §4.2 硬限制
- **操作指南**: tune-swimlane SKILL.md §4 + merge-optimization.md §4
- **原理**: 走 CV 通路（片上直连）替代 CV 间 GM 搬运，CV 最优配比 1:2（1 Cube : 2 Vector）。**Mix合图功能只在 A5（DAV_3510）平台上才有**；DAV_3003/3113 等 Lite 平台有独立 LiteNPU 路径，不可等同；只包裹 Vector 段的合图称为普通合图（S-4），在所有平台均可用
- **两种开关方式（功能完全一致，异常处理方式也完全一致）**:
  - **自动合图（优先尝试）**：在 jit kernel 函数体内调用 `pypto.experimental.auto_mix_partition(1)`，编译器自动决定 scope 范围。默认关闭（0），受代价/收益判断影响可能跳过部分段
  - **手动合图**：`sg_set_scope` 包裹 CV 交替段（开始 正整数，结束 -1），用户手动控制 scope 范围。scope 后的数字只是唯一标志，无功能差异，相同数字=同一段合图，不同数字=不同段合图
- **统一调优流程**: ⛔ 先完成 Step 0 数据流分析（CV 数据流表→scope 布局→配套参数）→ Step 1 基于分析结论开启 Mix合图（原子优化点）→ 达标则完成 → ⛔ 编译超时**禁止放弃 Mix合图**，按关键路径 Step 2 缩小 scope/TileShape/unroll 后重试（自动超时→切手动缩小 scope）→ ⛔ 退化**禁止跳过分析直接回退**，按关键路径 Step 3 分析原因并调参 → Step 4 调整 scope 框架 → 当前方式全部走完仍无收益 → 按关键路径 Step 5 切换另一种开关方式。详见 merge-optimization.md §4.1 关键路径
- **硬性限制条件（⛔ 不满足则不生效，详见 merge-optimization.md §4.2）**:
  - ⛔ CV 间数据传递与消费关系：方向只能 1:N 或 N:1（不支持 M:N）；一个 matmul 结果只喂一条 vector 链（不同需求消费者须拆分）；同一 L0C_COPY_UB 的 vector 消费者须在同一 AIV 核
  - CV 间 shape 变化须单调（禁止交叉大小，如 [64,128]→[128,64]）
  - ⛔ UB 并发使用上限 UB_size（§一）：单个 tensor 的ND+NZ的总大小须 < UB_size，超限则走 DDR
  - ⛔ 算子衔接形态：matmul 直接接 vector（中间无其他 Cube）；判断依据是最终 TileGraph 是否匹配直连识别模式（小块→大块 ASSEMBLE 汇聚 / 大块→小块 VIEW 拆分），框架自动插入或合法手写的 assemble/view 均可，非"是否手写"
  - ⛔ shape/tile 硬数值约束：衔接 tensor 须 2D；L0C→UB vec tile 两维 CUBE_ALIGN 对齐（§一）；cube tile 与 vec tile 衔接轴相等或整数倍；UB→L1 内轴切分 32B 对齐；assemble 场景输出 ≤ UB×0.35
- **调优建议（非硬性限制，详见 merge-optimization.md §4.2.5）**: Loop unroll 减少调度开销（第 8 条）、TileShape 同步调整——L0 调小 + L1 调大 + loop tile 调小三者作为一组（第 9 条）、cube_nbuffer 配合 Mix 消除 Cube 子图间调度开销（merge-optimization.md §4.1 说明）
- **⛔ nbuffer 调优流程（Mix合图场景必做，不可跳过）**:
  1. **初始值设为 1**：开启 Mix合图时，初始值设为1（`vec_nbuffer_setting: {"DEFAULT": 1}` + `cube_nbuffer_setting: {-1: 1}` + `cube_l1_reuse_setting: {-1: 1}`），因为 Mix合图已将 CV 段串行化，nbuffer>1 初始会增加 UB 占用导致 spill。但 nbuffer 最优值因算子而异（部分算子 cube_l1_reuse=8、cube_nbuffer=4 为最优），须逐值实测，不可仅试 1 后就放弃更大值
  2. **在 nbuffer=1 基线上完成其他参数调优**（TileShape、unroll、ooo_sched_mode 等）
  3. **逐值实测 vec_nbuffer**：从 1 开始按 2 的幂次递增（1→2→4→8→16→32…），每次实测 AICore E2E Time。1/2/4/8 是常用候选范围，不是硬性上限——若 8 仍有收益，继续试 16/32。若调大后劣化则回退至上一个最优值，不再继续增大（nbuffer 是 UB 压力和并行度的 tradeoff，劣化说明 UB 压力已超阈值，继续增大几乎不可能反转）
  4. **cube_nbuffer / cube_l1_reuse 同理**：在 vec_nbuffer 最优值基础上，cube_nbuffer 和 cube_l1_reuse 从 1 开始按 2 的幂次递增逐值尝试，劣化则回退
- **最优结果标准（CV 全合须同时满足四条）**: (1) 用满核无气泡 (2) 全走 CV 通路无 DDR 中转 (3) 无 spill (4) 核内计算流水排布紧密无空闲。调小 TileShape 是逼近最优的首要手段（提高核内并行度+增加任务数填满核+减少 UB 占用降低 spill）
- **验证方法（⛔ 调优必做，须追踪 CV 间数据流向）**: 解析 program.json，识别 CV 间应传递的数据（通过 semantic_label 定位），追踪每个数据的传递路径——走 CV 通路 opcode（`L0C_COPY_UB`/`UB_COPY_L1`/`UB_COPY_ND2NZ` ✅）还是 DDR 中转（`COPY_OUT` 后紧接 `COPY_IN` 无 CV_SYNC 邻居 ❌）。⛔ 不能只看 opcode 是否存在：存在 CV 通路 opcode 只能证明部分数据走了 CV 通路，须确认所有 CV 间数据都走 CV 通路。正常的 DDR 搬运（最终输出/原始输入/跨迭代依赖 mi/li/oi）不影响 Mix合图生效判断。**性能指标**：检查泳道图中 spill（`WorkspaceGm`）数量，复杂算子 ≤20 正常（经验阈值），超过则调小 TileShape或调小 nbuffer 减少 spill
- **典型收益**: 消除 CV 间 DDR 搬运开销，视算子 CV 搬运占比而定。A5 场景下 Mix合图最终一般能带来优化，首次劣化不可直接回退，需多次尝试（同步调小 TileShape、对比 nbuffer、排查 DDR 回退）。**⛔ CV 全合和 CV 不全合都是合法调优路径**——选定一个 scope 范围后先充分调参，所有参数都调完仍退化才切换 scope 范围。**⚠️ Mix合图可能降低 wall time 但增加核上 compute（AICore E2E Time），须以 AICore E2E Time 下降为准判断优化效果，详见 merge-optimization.md §4.2.5 第 6 条**
- **关联优化**: S-4（Vector 手动合图）、S-11/S-12（TileShape，Mix合图下可调小）

### [S-15] ooo_sched_mode（CV 交替算子，非通用）

- **阶段**: 深度调优
- **优先级**: ⭐⭐ P2（仅适用算子，非通用）
- **适用条件**: A5 平台（`pypto.platform.npuarch == 'DAV_3510'`）+ 算子存在连续 Cube↔Vector 交替结构（如 attention 类算子的 Q@K^T→softmax→P@V 模式）。**⚠️ 非通用优化点**：纯 Cube 或纯 Vector 算子无收益，仅对 CV 交替结构的算子有效
- **前置条件**: S-3 完成
- **检查方法**: 确认平台为 A5；确认算子有连续 CV 交替结构
- **操作指南**: tune-swimlane SKILL.md §7
- **参数说明**: `ooo_sched_mode` 取值范围为 `{"", "GAPMIN", "HLF"}`，配置在 `pass_options`。各取值语义：
  - `""`（默认）：基于拓扑序遍历和局部搜索的调度（GapMin 调度 + local-search）
  - `"GAPMIN"`：仅执行 GapMin 调度，跳过 local-search
  - `"HLF"`：Highest Level First 调度（按任务到汇点最长路径降序排列后做 EFT 插入调度）
  - 三个取值需逐个尝试验证是否有性能收益，无通用推荐值
- **配置示例**:
  ```python
  pass_options={
      "ooo_sched_mode": "HLF",
  }
  ```
- **适用场景**: host 下发开销主导、核上 compute 已接近最优时的补充优化。须以 AICore E2E Time 不恶化为前提
- **关联优化**: S-14（A5 Mix合图，两者可配合）、S-16（vf_options，通常配合使用）

### [S-16] vf_options

- **阶段**: 深度调优
- **优先级**: ⭐⭐ P2
- **适用条件**: A5 平台（`pypto.platform.npuarch == 'DAV_3510'`）+ 算子含 Vector 计算段。VF 融合在 A5 上默认开启，`vf_options` 是让其生成代码更优的编译选项，对所有含 Vector 计算的算子通用
- **前置条件**: S-3 完成
- **检查方法**: 确认平台为 A5；确认算子含 Vector 计算段
- **操作指南**: tune-swimlane SKILL.md §8
- **参数说明**: `vf_options` 为 vector fusion 代码生成选项，传入 LLVM 编译参数。推荐值 `-mllvm -cce-vf-enable-vloopv2-recognizer=true -mllvm -enable-pto-colop-fusion=true`。配置在 `codegen_options`。
- **配置示例**:
  ```python
  codegen_options={
      "vf_options": "-mllvm -cce-vf-enable-vloopv2-recognizer=true -mllvm -enable-pto-colop-fusion=true"
  }
  ```
- **适用场景**: 优化 Vector 指令融合的代码生成质量，减少指令开销。须以 AICore E2E Time 不恶化为前提
- **关联优化**: S-17（VF 融合编排原则，编排原则让融合候选更多、vf_options 让生成代码更优，互补可叠加）、S-14（A5 Mix合图，两者可配合）

### [S-17] VF 融合编排原则

- **阶段**: 深度调优
- **优先级**: ⭐⭐ P2
- **适用条件**: A5 平台（`pypto.platform.npuarch == 'DAV_3510'`）+ 算子含连续 vec 算子序列（如 softmax 段、elementwise 连续计算段、reduce→expand 组合段）
- **前置条件**: 无
- **检查方法**: 审视连续 vec 算子序列的代码编排，逐条对照三原则
- **操作指南**: tune-swimlane SKILL.md §8 + merge-optimization.md §5.2
- **三条原则**:
  1. **相同 dst shape**：Reduce/expand 只看目标输出 dst shape，dst shape 一致方可并入同一融合段
  2. **区间约束**：融合段内 tensor 区间只允许完全重叠（inplace 复用）或完全不重叠（独立 buffer），禁止部分重叠
  3. **Reduce 最后、expand 最前**：让尽可能多的 OP 落在相同的大 shape 上，最大化融合收益
- **典型收益**: 让 A5 默认开启的 VF 融合合并更多 vec 指令，减少指令开销
- **约束**: VF 融合在 A5 默认开启，本优化是让其效果更好的代码编排指导，独立于 mix合图
- **关联优化**: S-16（vf_options，编排原则让融合候选更多、vf_options 让生成代码更优，互补可叠加）

### [S-18] sg_set_tunevf_mode（VF 调优 Pass 行为模式）

- **阶段**: 深度调优
- **优先级**: ⭐⭐ P2
- **适用条件**: A5 平台（`pypto.platform.npuarch == 'DAV_3510'`）+ 算子含 Vector 计算段。控制 VF 调优 Pass 的行为模式，对所有含 Vector 计算的算子通用，独立于 mix合图
- **前置条件**: 无
- **检查方法**: 确认平台为 A5；确认算子含 Vector 计算段
- **操作指南**: tune-swimlane SKILL.md §8 + merge-optimization.md §5.4
- **参数说明**: `sg_set_tunevf_mode` 控制 VF 调优 Pass 在 OoO Pass 输出的 op 序列基础上是否调整 op 顺序、以及调整时偏向流水还是融合。默认值 `0`，取值范围 `{0, 1, 2}`：
  - `0`：均衡模式，自动调整 op 顺序，平衡 Pipeline 流水与 VF 融合的整体性能收益
  - `1`：指令流水优先模式，不改变 OoO 排好的 op 执行序
  - `2`：vf 融合优先模式，不考虑性能建模的收益评估，尽量调整 op 顺序以保证更大范围的 VF 融合
- **配置示例**:
  ```python
  if pypto.platform.npuarch == 'DAV_3510':
      pypto.set_pass_options(sg_set_tunevf_mode=2)  # vf 融合优先
  ```
- **调优建议**: 先以默认 `0` 采集基线；编排原则（[S-17]）就位但融合收益未达预期时尝试 `2`；流水退化（AICore E2E Time 上升）则回退 `1`。须以 AICore E2E Time 不恶化为前提
- **关联优化**: S-17（VF 融合编排原则，编排原则扩大融合候选、sg_set_tunevf_mode 控制 Pass 调整 op 序列的激进程度，协同——编排原则就位后 `=2` 收益更明显）、S-16（vf_options）

### [S-19] ready_on_host_tensors

- **阶段**: 深度调优
- **优先级**: ⭐⭐ P2
- **适用条件**: 算子有小 tensor 输入通过 AICPU gather 下发（如 paged KV cache 类算子的 block_table、actual_seq 等索引类 tensor），wall time 远大于 AICore E2E Time 时优先考虑
- **检查方法**: 对比 AICore E2E Time 与 wall time，若 wall time >> AICore E2E Time，检查算子输入中是否有小 tensor（如索引表、序列长度等）通过 AICPU gather 下发
- **操作指南**: tune-swimlane SKILL.md §9
- **参数说明**: `ready_on_host_tensors` 是 `runtime_options` 参数，值为 tensor 名称列表，标记这些 tensor 在 host 端 ready，避免 AICPU 等待 tensor 从 device 拷回 host 再下发
- **配置示例**:
  ```python
  @pypto.frontend.jit(
      runtime_options={
          "ready_on_host_tensors": ["block_table", "kv_act_seqs"],
      }
  )
  ```
- **原理**: 部分 tensor（如 block_table、kv_act_seqs）数据量小但需要 AICPU 读取后下发给核。默认流程中 AICPU 需等待这些 tensor 从 device 拷回 host 才能读取，标记为 host ready 后跳过该等待
- **典型收益**: 减少 wall time 中的 host 下发开销，不影响 AICore E2E Time
- **关联优化**: S-9（Stitch 调优）、S-10（调度策略）

---

### [I-1] 小 Shape 矩阵乘

- **阶段**: 核内调优
- **优先级**: ⭐⭐⭐ P0
- **适用条件**: Matmul 的 Shape 特殊（如 M 很大 N 很小）
- **检查方法**: 通过泳道图定位到耗时较长的 task，检查其 Matmul 的 M/N/K 是否特殊
- **操作指南**: tune-incore SKILL.md §1
- **解决方案**: 使用 Vector 操作提前处理输入矩阵，通过 concat/reshape 构造标准 Shape
- **典型案例**: 左右矩阵 (884736, 16) × (16, 16) → 从 500us 优化到 40us

### [I-2] L2 Cache 策略

- **阶段**: 核内调优
- **优先级**: ⭐⭐ P1（融合算子中效果显著）
- **适用条件**: 算子包含大型权重矩阵（matmul 权重），或有过大的输出 Tensor
- **API**: `tensor.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)`
- **API 文档**: `https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/tensor/pypto-Tensor-set_cache_policy.md`
- **检查方法**: 分析算子中所有权重矩阵的访问模式，识别只读一次且不复用的大 Tensor
- **操作指南**: tune-incore SKILL.md §2
- **调优策略**:
  - 简单算子：逐个对候选 Tensor 设置 NONE_CACHEABLE，每次实测对比
  - 融合算子（含多个大权重矩阵）：**同时对所有权重设置 NONE_CACHEABLE**，避免 L2 争用失衡
- **⛔ 不适用场景**:
  - 输入 Tensor（数据量小，硬件预取已足够，绕过反增延迟）
  - 输出 Tensor（增加写回延迟）
  - 融合算子中单独对某个权重设置（打破 L2 平衡，可能恶化）
- **典型收益**: 10-20%（融合算子中批量设置所有权重）
- **典型案例**: Pangu 7B Fused Layer，5 个权重矩阵同时设置 NONE_CACHEABLE → 437.28us→354us（-19.1%）
- **代码示例**:
  ```python
  # 融合算子中：对所有权重矩阵同时设置
  qkv_weight.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
  o_weight.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
  gate_weight.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
  up_weight.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
  down_weight.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
  ```

### [I-3] 冗余计算消依赖

- **阶段**: 核内调优
- **优先级**: ⭐⭐ P1
- **适用条件**: 一对多的子图依赖（一个 tensor 被多个下游消费）
- **检查方法**: 分析 `dyn_topo.txt` 中是否存在一个节点有多个不同 psgId 的后继
- **操作指南**: tune-incore SKILL.md §3
- **解决方案**: 增加冗余计算（复制数据），使每个分支独立，避免一对多的子图依赖
- **典型案例**: GLM MoE Fusion 中将 e_score_bias_2d 复制 tile_batch 份

### [I-4] 尾轴长度优化

- **阶段**: 核内调优
- **优先级**: ⭐⭐ P1
- **适用条件**: Operation 输入 Tensor 尾轴 < 32B 对齐
- **检查方法**: 检查参与计算的关键 tensor 的最后一个维度大小
- **操作指南**: tune-incore SKILL.md §4
- **解决方案**: 使用 concat 增大尾轴、transpose 调整轴顺序、reshape 调整 Shape
- **关联优化**: I-9（valid_shape 尾块零填充避免）

### [I-5] TileOperation 实现检查

- **阶段**: 核内调优
- **优先级**: ⭐ P2
- **适用条件**: 上述所有优化手段都已尝试，单 task 耗时仍然过长
- **检查方法**: 构造单独 Operation 的测试用例，与 Ascend C 小算子性能对比
- **操作指南**: tune-incore SKILL.md §5
- **解决方案**: 确认性能差距后检查是否使用了更优指令，或考虑使用其他 Operation 组合替代
- **关联优化**: I-6（操作数连续性）、I-7（搬运方向）、I-8（计算搬运重叠）

### [I-6] 操作数连续性检查

- **阶段**: 核内调优
- **优先级**: ⭐⭐ P2
- **适用条件**: TileOperation 输入 Tensor 内存不连续（非 contiguous）
- **检查方法**: 在 TileOperation 检查流程中，确认输入 tensor 是否内存连续；不连续会导致额外搬运或性能下降
- **操作指南**: tune-incore SKILL.md §6
- **解决方案**: 使用 `pypto.reshape` 或 `pypto.transpose` 调整非连续输入为连续布局
- **典型案例**: transpose 后的 tensor 作为下游输入前需确保连续性

### [I-7] Gather/Scatter 数据搬运方向优化

- **阶段**: 核内调优
- **优先级**: ⭐⭐ P2
- **适用条件**: 存在 HBM ↔ L1 数据搬运瓶颈
- **检查方法**: 分析 TileOperation 的数据流向，区分 Gather（HBM→L1）和 Scatter（L1→HBM）
- **操作指南**: tune-incore SKILL.md §7
- **解决方案**: Gather 方向使用 `set_cube_tile_shapes` 的 block size 控制搬运粒度；Scatter 方向使用 `pypto.assemble` 写回 HBM
- **典型案例**: 大矩阵分块加载时选择合适的 cube_tile block size 以匹配 L1 容量

### [I-8] submit_before_loop 计算与搬运重叠

- **阶段**: 核内调优
- **优先级**: ⭐⭐ P2
- **适用条件**: 子 loop 未正确提交，导致计算与搬运无法重叠执行
- **检查方法**: 检查子 loop 是否使用了 `submit_before_loop=True` 参数
- **操作指南**: tune-incore SKILL.md §8
- **解决方案**: 设置 `submit_before_loop=True` 使子 loop 正确提交，实现计算与数据搬运的时间重叠
- **典型案例**: 内层多个子 loop 串行执行 → 开启 submit_before_loop 后计算与搬运流水化

### [I-9] valid_shape 尾块零填充避免

- **阶段**: 核内调优
- **优先级**: ⭐⭐ P2
- **适用条件**: 切块后最后一块数据量不足一个完整 BLOCK_SIZE，存在无效零填充计算
- **检查方法**: 检查 `pypto.view` 切块时尾块是否存在零填充，对比实际数据量与 BLOCK_SIZE
- **操作指南**: tune-incore SKILL.md §9
- **解决方案**: 使用 `pypto.view` 的 `valid_shape` 参数标记有效数据范围，避免对零填充部分执行无效计算
- **代码示例**: `pypto.view(tensor, [BLOCK_SIZE, ...], valid_shape=[actual_last_size if i == last_tile else BLOCK_SIZE])`
- **典型案例**: 尾块仅 3 个有效元素但按 BLOCK_SIZE=16 计算 → valid_shape 标记后跳过无效计算
- **关联优化**: I-4（尾轴长度优化）

### [I-10] 合并 gather 减少搬运次数

- **阶段**: 核内调优 / 算法级优化
- **优先级**: ⭐⭐⭐ P1+（DDR 瓶颈场景强制）
- **适用条件**: 算子中存在多个 `gather_in_ub`/`gather_in_l1` 调用，且这些 gather 取的是同一 source tensor 的不同列段
- **检查方法**: `grep -nE "gather_in_ub|gather_in_l1"` 统计所有 gather 调用，检查是否有多个 gather 的 source tensor 相同但取不同列范围
- **操作指南**: tune-incore SKILL.md §10
- **典型收益**: 10-40%（减少一次完整 DDR→UB 往返，视 gather 数据量而定）
- **修改方式**: 在算子入口（loop 之前）将 source tensor 拼接为 `[rows, col1+col2+...]`，用一次 gather 取全部列段，后续用 view 切出各段
- **约束**: gather 结果须满足 UB_size（§一）限制（单 tensor ND+NZ < UB_size）；拼接须在 GM 完成（`pypto.concat`），gather 后数据在 UB 内通过 view 切分
- **典型案例**: sparse_flash_attention_quant 中 kn (512维) + kr (64维) 两次 gather → 拼接 key_2d=[kn,kr] (576维) 一次 gather，性能 1019→611us（-40%）
- **关联优化**: I-11（view 复用）、S-14（A5 Mix合图，合并 gather 后 CV 通路可闭合）

### [I-11] view 复用消除重复搬运

- **阶段**: 核内调优 / 算法级优化
- **优先级**: ⭐⭐⭐ P1+（DDR 瓶颈场景强制）
- **适用条件**: 已 gather 到 UB 的数据在后续阶段被独立 gather 再次取用（取的是已 gather 数据的子段或重叠段），或已 assemble 到 UB 的数据在后续 matmul 中被独立 view 重新取用
- **检查方法**: 逐个分析后续阶段的 gather/view 调用，判断其取的数据是否与 V0 阶段已 gather 的 tensor 或已 assemble 的 tensor 存在内容重叠
- **操作指南**: tune-incore SKILL.md §11
- **典型收益**: 10-30%（消除一次完整 DDR→UB 往返，或消除一次 UB 内重复搬运）
- **修改方式**: 用 `pypto.view(已有tensor, [shape], [offset])` 从已 gather 或已 assemble 的 tensor 切出所需数据，替代独立 gather 或独立 view
- **约束**: view 的 offset 和 shape 须在已有 tensor 的有效范围内；view 后数据布局须与下游计算期望一致（注意 NZ/ND 格式差异）
- **典型案例1**: sparse_flash_attention_quant 中 V2 阶段 vj 独立 gather kn 的部分段 → `view(kn, [s2_tile, dn])` 复用已 gather 的 kn
- **典型案例2**: attention 中 C1 已将 kn assemble 到 kj_view（`kj = [kn | kr]`），C2 的 V 矩阵（=kn）应从 kj_view 切出 `vj = view(kj_view, [s2_tile, dn], [0, 0])`，而非独立 `view(kn, ...)` 重新从原始数据切——kn 已在 kj_view 的前 dn 列中，复用避免 UB 内重复搬运
- **关联优化**: I-10（合并 gather）、S-14（A5 Mix合图，view 复用后 CV 通路可闭合）

---

### [F-16] 多 Matmul 差异化 Cube TileShape

- **阶段**: 开箱调优
- **优先级**: ⭐⭐⭐ P0
- **适用条件**: 算子含 2 个及以上 Matmul（如 attention 类的 QK^T + PV）
- **检查方法**: 逐个分析每个 Matmul 的 M/N/K 轴大小，判断是否需要不同的 L1/K_L1 配置
- **操作指南**: tune-frontend SKILL.md「多 Matmul 差异化 Cube TileShape」章节
- **值输出形式**: 每个 matmul 独立执行 basic-block-optimization.md §2「Cube Tile 公式推导方法」得到自己的分档分支实例（各自的 M/N/K case 标量代入，普遍性检查按本实现所辖 case 范围逐 matmul 进行），禁止共享一组字面常量；每个 tile 设置点使用自己的 tile 变量（§1b 步骤 3），可选配 trace-time assert（三重积 `mL0·kL0 ≤ C_A`、`kL0·nL0 ≤ C_B`、`mL0·nL0 ≤ C_C`，非法 case compile 期失败）。后续 refinement（F-20 等）只修改分支参数。放置/退化见 §1b。
- **典型收益**: 10-30%（减少重复搬运 + K 轴复用）
- **核心原则**:
  - **C1 (QK^T)**: K 轴较大时（如 576），L1 调大（如 256）让 A/B 矩阵驻留 L1，减少重复载入
  - **C2 (PV)**: K 轴与 C1 的 N 轴相同时（如 512），K_L1 可调大（如 256）复用 C1 已加载的 B 矩阵数据
  - **禁忌**: 所有 Matmul 使用完全相同的 cube tile 配置（除非 M/N/K 完全一致）
- **L1 推荐值推导方法**:
  - L1 应设为 ≥ K 轴实际值且能被 L0 整除的值；若 K 值本身 ≤ 256，直接用 K 值（如 K=128 → L1=128）；若 K 值 > 256，用 256（L1 上限通常 256）
  - C1 和 C2 的 K 轴 L1 应独立推导：C1 的 K 是 Q 的维度（如 dn+dr=576），C2 的 K 是 C1 的 N（如 s2_tile=512）
  - ⚠️ L1 过大会导致 L1 容量不足引发 spill，需实测验证
- **配置示例**:
  ```python
  # C1 (QK^T): M=128, N=s2_tile, K=576 → L1 调大让 K 驻留
  pypto.set_cube_tile_shapes([128, 128], [256, 256], [128, 128])
  # C2 (PV): M=128, N=512, K=s2_tile → K_L1 调大复用
  pypto.set_cube_tile_shapes([128, 128], [128, 128], [256, 256])
  ```
- **关联优化**: F-9（Cube TileShape 基础设置）、S-11（Cube TileShape 深度调优）

### [F-17] V 段内多 shape 分段 vec tile

- **阶段**: 开箱调优
- **优先级**: ⭐⭐⭐ P0
- **适用条件**: 同一 V 段内处理的 tensor shape 发生变化（如 online softmax 中 [M,N]→[M,1]→[M,N] 交替）
- **检查方法**: 逐行扫描 V 段内每个 operation 的输入/输出 tensor shape，标记所有 shape 变化点
- **操作指南**: tune-frontend SKILL.md「V 段内多 shape 分段 vec tile」章节
- **值输出形式**: 每个 shape 变化点的 tile 按 basic-block-optimization.md §3「Vector Tile 公式推导方法」独立推导（§1b 四步流程）：col=该点归约轴/scatter_ 轴长度，row=「不超过 M」与「UB 预算除以（col × dtype_bytes）」中较小值、下限 1；每个 shape 变化点用**自己的 tile 变量**插入独立的 `set_vec_tile_shapes` 调用（展平前提不变），禁止字面值硬编码。💡 插入更多 tile 调用成本可忽略。放置见 §1b。
- **议程来源**: 阶段A「A4 分段优化议程表」（tune-frontend SKILL.md §A4）以结构性判定提供本优化点在各段/边界的适用性与优先级（附证据），本步骤的实际执行在阶段C ITER 中完成（实测+精度回归）。
- **典型收益**: 5-20%（消除 padding 浪费 + 减少子图数量）
- **核心原则**:
  - **shape 变化时必须重设 vec tile**：每次 tensor shape 从 [M,N] 变为 [M,1] 或反向变化时，必须重新设置匹配的 vec tile。**⛔ 任务数爆炸机制（重设的性能依据）**：shape 从 [M,1] 变回 [M,N]（如 reduce 输出广播回全宽）时，若继承小列 tile，[M,N] tensor 被按 col=1 切成 **N 个子任务**（而非 1 个任务）——任务数爆炸、调度开销大；反向（[M,N]→[M,1]）继承大列 tile 则为 1 列数据分配整行宽 UB（浪费）但不爆炸。**正确重设 = 每个变化点 1 任务**，与 Rule 4/A4 task-count ratio 同一机制（装载/计算任务数 = 任务数判据）
  - **归约类 [M,1] ops**：用大行数小列数 tile（如 [128, 128]），匹配 reduction 输出 shape
  - **elementwise [M,N] ops**：用匹配 N 轴的 tile（如 [32, 512]），用满尾轴带宽
  - **⚠️ 分段前提：嵌套表达式必须展平**：若 V 段内使用嵌套表达式（如 `pypto.add(pypto.mul(a,b), pypto.mul(c,d))`），编译器将整个嵌套视为一个不可分割的 op，无法在中间插入 `set_vec_tile_shapes`。必须先将嵌套表达式展平为独立中间变量（`t1=mul(a,b); t2=mul(c,d); t3=add(t1,t2)`），才能在 shape 切换点插入 vec tile 分段。**展平是分段的必要前置条件，不展平则分段无法生效**。当 V 段内 [M,1] 归约 ops 和 [M,N] elementwise ops 交替时，嵌套表达式下编译器用开头设的 tile 统一处理所有 ops，导致 [M,N] ops 用 [M,1] 的 tile 产生严重 padding；展平后在切换点插入匹配 [M,N] 的 vec tile 可消除 padding
- **典型案例**: online softmax 等含 online update 结构的算子，V 段内归约 ops 输出 [M,1]、elementwise ops 输出 [M,N] 交替出现——前者用大行小列 tile（如 [128,128]），后者用匹配 N 轴的 tile（如 [32,512]）。前提是 V 段已展平为独立中间变量，否则嵌套表达式无法在两种 shape 之间插入 vec tile 切换
- **关联优化**: F-10（Vector TileShape 基础设置）、S-12（Vector TileShape 深度调优）

### [F-18] 语义维度的静态循环保留

- **阶段**: 开箱调优
- **优先级**: ⭐⭐ P1
- **适用条件**: 算子有 n_kv/group 等语义维度的静态循环（即使迭代次数=1）
- **检查方法**: 搜索代码中所有被省略的语义循环（如 n_kv=1 时直接展开为无循环）
- **操作指南**: tune-frontend SKILL.md「语义维度的静态循环保留」章节
- **典型收益**: 3-15%（影响编译器 root function 划分）
- **核心原则**:
  - **F-5 的例外**：F-5 说"静态轴改 Python for"，但 n_kv/group 等语义维度应保留为 `pypto.loop`，即使迭代1次
  - **原因**：保留 `pypto.loop` 让编译器能正确识别这些维度的语义边界，优化 root function 划分和数据流分析
  - **判断标准**：该循环变量是否在 tensor shape 或 view offset 中被使用？是 → 保留 pypto.loop；否 → 可改 Python for
- **代码示例**:
  ```python
  # 保留语义循环（即使 n_kv=1, group_loop=1）
  for n_kv_idx in pypto.loop(0, n_kv_sym, 1, name="LOOP_n_kv"):
      for group_idx in pypto.loop(0, g_loop_sym, 1, name="LOOP_group"):
          cur_offset = batch_idx * s1_n2_gsym + slc_idx * nq + n_kv_idx * group + group_idx * cur_group_tile
  ```
- **关联优化**: F-5（静态轴改 Python for）、F-1（任务粒度）

### [F-19] 内层批量化（循环 → 批量 tensor 操作 + 外提）

F-19 有两种模式：**Mode A**（matmul M 轴批量化，自含 unroll 机制，不依赖 F-8 先行应用）和 **Mode B**（通用 tile 扩展 + 操作外提，独立于 F-8）。两种模式均为必做项，须分别检查触发条件。

- **阶段**: 开箱调优
- **优先级**: ⭐⭐⭐ P1（loop 结构级批量化，先于 P2 tile 基线——P2 按 §2/§3 以 M_new 直接推导 tile 值，无需既有分支可更新）
- **强制性**: 满足任一模式触发条件时 F-19 为必做项，不可跳过（须注明具体原因方可标记"不适用"）
- **操作指南**: tune-frontend SKILL.md §1.2.4
- **值输出形式**: F-19 批量化后的 M_new = unroll × M_current 由 P2 tile 优化（F-9/F-10）按 §2/§3 直接代入 row 维 UB 预算除法 / cube 三重积推导，逐 case 普遍性检查。放置/退化见 §1b。

#### Mode A: matmul M 轴批量化（自含 unroll 机制）

- **前置条件**: 无（**自含 unroll**：目标循环为 `pypto.loop` 时，由本优化点自行插入 `unroll_list` 与 `for range(unroll)` 中间循环，即内联执行 F-8 变换；已应用 F-8 的循环直接复用其 unroll 结构）
- **触发条件**: 检查**任意循环**（`for range(unroll)` / `pypto.loop_unroll` / **`pypto.loop`**）体内的 matmul。若 M_current < 8（通常为 1），且循环维度可通过 view 重构批量化进 matmul M 轴（M_new = unroll × M_current）、各次迭代数据独立，则**必须**执行 Mode A。
- **不适用情况**: 循环维度无法映射到 matmul M 轴（如映射到 N/K/Vector 维度）；迭代间存在数据依赖；M_current ≥ 8（但须继续检查 Mode B）
- **检查方法**: 枚举全部循环（含 `pypto.loop`），对每层循环体内 matmul 记录其 M_current；追踪循环维度是否可通过 view 重构（相邻内存 / 可合并 assemble）映射到 M 轴
- **典型收益**: 30-60%（M 轴批量化提升 Cube 利用率；同时减少 matmul 调用次数 → 减少 cube→vector 与 vector→cube 交接次数 → **降低 AIV↔AIC 交接延迟**——每次交接 = GM 读 roundtrip + 调度等待，批量化后交接次数按 M 增大倍数缩减）
- **unroll_list 选择**（同 F-8，含编译爆炸防护）:
  - `[64, 32, 16, 8, 4, 2, 1]`: 默认起始（每迭代维度 < 16）
  - `[32, 16, 8, 4, 2, 1]`: 循环次数不被 64 整除时
  - `[16, 8, 4, 2, 1]`: 较小循环次数
  - `[1]`: 最小（无批量化，仅 reshape 无 transpose）
  - ⛔ **编译爆炸防护 + 全 case 普遍性（同 F-8）**：当 M_current ≥ CUBE_ALIGN（§一）时缩减 unroll_list 变体数（变体选择规则 + 全 case 普遍性检查 + fallback if/elif/else + 循环消除策略见 F-8 对应小节）。F-19 特有：M_new = unroll × M_current 随 case 变化——大 loop_count 用小 unroll → M_new 较小 → tile 也不同（P2 tile 分支须覆盖全部 case）；large_unroll 具体值须在目标 case 上实测。
  - 规则: 必须在最内层 loop_unroll；最大值不超过循环次数；1 必须在列表中
- **⛔ 渐进变换（每步精度门控）**——⛔ 禁止一次性大爆炸变换（历轮实测：一次性完成全部变换后精度失败无法定位根因，连续多轮 max_diff≈0.6 级失败且不可调试）。**执行纪律**：每步完成后先跑精度测试（**Steps 1-4 期间不测性能**），PASS 才进入下一步；**若某操作精度失败，仅回退该操作**（保留在 for range 内）；已批量化的操作保留：
  - **Step 1（结构改造，无批量化）**: `pypto.loop` → `pypto.loop_unroll(..., unroll_list=[U1,...,1])`（自含 unroll，不依赖 F-8；已应用 F-8 时跳过改写）+ 添加 `for range(unroll)` 中间循环——循环体内**保持原有逐迭代操作不变**（每迭代做与改造前相同的工作，建立代码结构）。⛔ 精度门控：FAIL → `loop_unroll`/`for range` API 本身问题，与批量化逻辑无关
  - **Step 2（外提，仍逐迭代）**: 将可外提的操作移出 `for range`——view 操作（若输入可批量重构：连续 → 单一批量 view；非连续 → assemble 收集后批量 view）移出为 Step 3 批量化做准备；静态操作（仅依赖部分循环变量，如阈值只依赖其中某一个）按 F-21 外提为一次性预计算。外提后按 F-20 Rule 4 Step 1 修正相关 view 的 vec tile。⛔ 精度门控：FAIL → 外提/重构 bug
  - **Step 3（逐操作批量化分析）**: 逐行分析 Step 1 添加的 `for range` 循环体内的每个操作。对每个操作：
    - **(a) 判定可批量化**：输入可重构为批量 tensor（连续 → 单一 view；非连续 → assemble 收集到独立 GM buffer 后批量 view）、操作支持批量执行、无阻碍统一的逐迭代控制流
    - **(b) 可批量化** → 替换为批量操作（M = unroll × M_current）+ 更新 tile + 移出 for range；批量输出被逐迭代消费时 assemble 到 GM buffer
    - **(c) 不可批量化** → 保留在 for range 内，若前后有可批量化操作则创建独立 for range 循环通过 GM-assemble 管线连接
    - **(d) 条件批量化**：若某操作在条件的一个分支可批量化但另一个不可——仅不可批量化分支保留 for range，仅对 gather/assembly 步骤分支，可批量化操作共享不复制，避免增加条件上下文和代码重复
    - **(e) 精度门控**：每个操作变更后运行精度测试，失败则仅回退该操作
    - ⛔ 禁止 for range 直接 view loop_unroll 体内 local tensor（scope hazard）
    - ⛔ `view(X)→assemble(...,X)` 是环（constraint 13），须用独立 `pypto.tensor` 作 assemble 目标
    - ⛔ 禁止一次性大爆炸变换——逐操作处理
  - **Step 4（M_new 确认）**: 确认 M_new = unroll × M_current（P2 tile 优化按此值经 §2/§3 推导 tile）+ 精度复验 + 普遍性检查（M_new 随 case class 变化，逐 case 代入）
  - **Step 5（性能测量——仅此步测性能）**: 全部精度 PASS 后才测性能（3 次取中位）
- **约束**:
  - 每次迭代必须**数据独立**（无跨迭代依赖）
  - 批量 tensor 须满足 UB/L1 内存约束
  - transpose 开销须被 matmul M 增大收益覆盖
  - 所有 tile 须更新以匹配新 tensor shape，不同 unroll 因子可能需要不同 tile（用条件分支）
  - **⛔ M_new tile 推导规则（P2 tile 优化必循）**：
    - **M_new** = unroll × M_current
    - **vec row 维** = 「不超过 M_new」与「UB_budget 除以（col × dtype_bytes）」中较小值、下限 1（col 不变——M 批量化不改列轴；UB 溢出时切行不切列；UB_budget 按 op 类型区分：view/assemble 站点 = 全部 UB_size（§一），普通计算 op = UB_size // 3）
    - **cube m 维** = 按 M_new 代入三重积推导（`mL0·kL0 ≤ C_A`、`kL0·nL0 ≤ C_B`、`mL0·nL0 ≤ C_C`；C_A=C_B=L0A_size//dtype_bytes、C_C=L0C_size//4）（§一）
    - **trace-time assert**（可选）：vec: UB 预算 assert；cube: 三重积 assert，非法 case compile 期失败
    - **普遍性检查**：M_new 随 case class 变化，将本实现所辖 case 集合逐一代入验证
    - 放置见 §1b
  - 本变换改变最内层循环体（**结构级**变更）→ 采纳后须在结构变更台账中标记 `结构级`（由编排器在阶段交接后触发 FRONTEND 重跑）
- **⚠️ 副作用**:
  - unroll 减小增加 task 数量和调度开销（同 F-8）；pre-nbuffer 阶段测得的 unroll 回归为**任务数体制相关**结论——S4_SWIMLANE 采纳 nbuffer（任务数大幅缩减）后须重新评估，不得永久 DO-NOT-RETRY
  - 需更新所有 set_vec_tile_shapes / set_cube_tile_shapes 以匹配新 tensor shape
  - 改多值 `unroll_list` 后若精度回归，立即回退该 loop
- **代码示例**:
  ```python
  # Step 1（自含 unroll）: 目标循环为 pypto.loop 时自行添加 unroll_list
  for idx, unroll in pypto.loop_unroll(0, t, 1, name="LOOP",
                                        unroll_list=[4, 2, 1]):

      # Step 2: for-range 中间状态（M=1，逐元素处理）
      for bu in range(unroll):
          x = pypto.view(src, [1, N, D], [bu, 0, 0])
          y = pypto.matmul(x, weight)  # M=1
          pypto.assemble(y, [idx + bu, 0, 0], out)

  # Step 3 pattern: line-by-line batching with for..range for unbatchable ops

  # Operation A (batchable): moved out of for..range
  x_all = pypto.view(src, [unroll, N, D], [0, 0, 0])       # contiguous → single view
  # (non-contiguous: gather via assemble to GM buffer, then batched view)
  y_all = pypto.matmul(x_all, weight, pypto.DT_FP32)        # M=unroll
  y_gm = pypto.tensor([unroll * N, H], pypto.DT_FP32)       # GM buffer
  pypto.assemble(y_all, [0, 0], y_gm)

  # Operation B (not batchable): stays in for..range
  for bu in range(unroll):
      y_i = pypto.view(y_gm, [N, H], [bu * N, 0])           # ✅ from GM buffer
      z_i = per_iteration_op(y_i)                            # not batchable
      pypto.assemble(z_i, [bu * N, 0], z_gm)                 # to GM buffer

  # Operation C (batchable): moved out of for..range
  z_all = pypto.view(z_gm, [unroll * N, H], [0, 0])
  out = pypto.matmul(z_all, weight2, pypto.DT_FP32)          # M=unroll
  ```

#### Mode B: 循环维度 tile 扩展 + 操作外提（独立于 F-8）

- **前置条件**: 无 — 不依赖 F-8，可独立检查和应用
- **适用循环类型**: `for range` 和 `pypto.loop_unroll`（**不含 `pypto.loop`** — 每次迭代为独立 root function，无法跨迭代扩展 tile）
- **tile 扩展目标**:
  - `for range(N)`: 扩展 tile 覆盖全部 N 次迭代（所有迭代在同一编译 scope 内）
  - `pypto.loop_unroll(N, unroll_list=[U])`: 扩展 tile 覆盖 unroll 块大小 U（每个 task 处理 U 个元素，tile 在 task scope 内扩展）
- **触发条件**（全部满足）:
  1. 循环体内存在操作（任意类型：matmul / elementwise / reduction 等），且该操作的 tile **未覆盖**当前循环的迭代单元
  2. 操作至少有一个输入为**共享输入**（不依赖循环变量 — 如共享权重、预计算查找表、归一化参数、常量等），或所有输入均可通过 reshape 将循环维度纳入 tile
  3. 循环变量仅通过 `pypto.view` 的**规律步长 offset** 选择数据切片（不转换数据，仅选择偏移）
  4. 源数据可通过 reshape 将循环维度纳入新维度（规律 stride 布局）
  5. 共享输入可**广播**到循环维度（或通过 reshape 匹配）
  6. 扩展后的 tile 不超出内存约束（UB_size / L1_size（§一））
- **不适用情况**:
  - 操作使用**逐迭代不同权重**且无可用的批量操作（如逐 head 不同 weight 的 matmul，且 `transposed_batchmatmul` 不支持当前 dtype）
  - 数据切片为**不规则 stride**（无法通过 reshape 视为单个 tensor）
  - 扩展 tile 超出内存约束
  - 循环维度**是归约轴**（对 reduction 操作而言，扩展会改变语义）
- **检查方法**: 遍历循环体内每个操作，检查：(1) 操作的 tile 是否已覆盖循环迭代单元；(2) 操作是否有共享输入；(3) 循环变量是否仅用于 view offset；(4) 源数据是否可 reshape 包含循环维度
- **典型收益**: 10-50%（将 N 次小操作合并为 1 次批量操作，消除调度开销 + 提升 tile 利用率）
- **变换步骤**:
  - **Step 1**: Reshape 源数据，将循环维度纳入新维度
    - `for range(N)`: `reshape(src, [dim0, N, dim2, ...])` — 覆盖全部迭代
    - `pypto.loop_unroll(N, unroll_list=[U])`: `reshape(src, [dim0, U, dim2, ...])` — 覆盖 unroll 块
  - **Step 2**: View 所有迭代的数据切片为单个 tensor
    - `view(src_reshaped, [dim0, N_or_U, slice_dim], [0, 0, offset])`
  - **Step 3**: 扩展操作的 tile 以覆盖循环维度
    - `set_vec_tile_shapes(dim0, N_or_U * slice_dim)` 或 `set_cube_tile_shapes(...)` 含扩展维度（概念描述——表达式值须按 F-9/F-10 值输出形式逐 case 预计算；放置见 §1b）
  - **Step 4**: 执行操作一次（共享输入自动广播到循环维度）
  - **Step 5**: Reshape 结果回 2D（如需要）并 assemble 到输出
  - **Step 6**: 将操作从循环体内**外提**到循环外（移到循环前或循环后，视数据依赖而定）
- **约束**:
  - 每次迭代必须**数据独立**（无跨迭代依赖）
  - 共享输入须支持广播到循环维度（否则需先 reshape/expand）
  - 批量 tensor 须满足 UB/L1 内存约束
  - 须更新所有 `set_vec_tile_shapes` / `set_cube_tile_shapes` 以匹配新 tensor shape
  - 外提后循环内剩余操作须保持正确数据依赖（外提的操作不依赖循环内其他操作的输出）
- **代码示例 — elementwise 批量外提（RoPE 模式）**:
  ```python
  # BEFORE: 128 次小 elementwise 操作在循环内
  for n in range(N_HEADS):
      q_r_n = pypto.view(qr, [tile_bs, Dr], [0, n * stride + D])
      x1 = pypto.view(q_r_n, [tile_bs, Dr // 2], [0, 0])
      x2 = pypto.view(q_r_n, [tile_bs, Dr // 2], [0, Dr // 2])
      neg_x2 = pypto.mul(x2, -1.0)           # 共享常量 -1.0
      x1_mat = pypto.add(x1, 0.0)            # 共享常量 0.0
      rot = pypto.concat([neg_x2, x1_mat], -1)
      y = pypto.add(pypto.mul(q_r_n, cos_fp32),    # 共享 cos_fp32
                    pypto.mul(rot, sin_fp32))       # 共享 sin_fp32
      pypto.assemble(y, [bs_offset, n * Dr], out)

  # AFTER: 1 次批量操作外提到循环外
  qr_3d = pypto.reshape(qr, [tile_bs, N_HEADS, stride])
  q_r_all = pypto.view(qr_3d, [tile_bs, N_HEADS, Dr], [0, 0, D])
  # ⚠️ 禁止 N_HEADS*Dr 等 BinOp 直接作 tile 参数——实际按 F-10 值输出形式
  # 以 if/elif/else 分支逐 case 取 int 字面量预计算（如 n_head_cols = 98304），再引用变量
  pypto.set_vec_tile_shapes(tile_bs, n_head_cols)
  x1_all = pypto.view(q_r_all, [tile_bs, N_HEADS, Dr // 2], [0, 0, 0])
  x2_all = pypto.view(q_r_all, [tile_bs, N_HEADS, Dr // 2], [0, 0, Dr // 2])
  neg_x2_all = pypto.mul(x2_all, -1.0)       # 广播
  x1_all_mat = pypto.add(x1_all, 0.0)        # 广播
  rot_all = pypto.concat([neg_x2_all, x1_all_mat], -1)
  y_all = pypto.add(pypto.mul(q_r_all, cos_fp32),    # cos_fp32 广播
                    pypto.mul(rot_all, sin_fp32))     # sin_fp32 广播
  y_2d = pypto.reshape(y_all, [tile_bs, N_HEADS * Dr])
  pypto.assemble(y_2d, [bs_offset, 0], out)

  # 循环内仅保留不可批量的操作（如逐 head 不同 weight 的 matmul）
  for n in range(N_HEADS):
      q_c_n = pypto.view(qr, [tile_bs, D_HEAD], [0, n * stride])
      w_uk_n = pypto.view(w_uk, [D_HEAD, HCKV], [n * D_HEAD, 0])
      q_n = pypto.matmul(q_c_n, w_uk_n)      # 逐 head 不同 weight → 不可批量
      pypto.assemble(q_n, [bs_offset, n * HCKV], query)
  ```
- **代码示例 — 共享权重 matmul 批量外提**:
  ```python
  # BEFORE: N 次小 matmul（M=1，共享权重 W）
  for i in range(N):
      x = pypto.view(src, [1, K], [i, 0])
      y = pypto.matmul(x, W)   # M=1, W 共享
      pypto.assemble(y, [i, 0], out)

  # AFTER: 1 次批量 matmul 外提（M=N）
  x_all = pypto.view(src, [N, K], [0, 0])
  # ⚠️ 示意值——实际 tile 值须按 F-9 值输出形式分档分支推导（M=N 随 case 变化，
  # 禁止字面值直接照抄；每分支 int 字面量）
  pypto.set_cube_tile_shapes([16, 16], [128, 128], [128, 128])
  y_all = pypto.matmul(x_all, W)   # M=N, W 共享
  pypto.assemble(y_all, [0, 0], out)
  ```

- **关联优化**: F-8（内层 unroll，Mode A 前置）、F-1（任务粒度）、F-4（Reshape 外提，Mode B 的 reshape 步骤与之相关）、F-21（依赖深度外提，Mode A Step 2 引用其外提静态操作）、F-9（Cube TileShape）、F-10（Vector TileShape）

### [F-20] 邻居算子感知的 TileShape 优化

- **阶段**: 开箱调优
- **优先级**: ⭐⭐ P2（在 F-9/F-10 之后执行）
- **适用条件**: matmul 后接 reduce / matmul 后接 vector(无reduce) / 多个并发 matmul 共享依赖
- **前置条件**: F-9（baseline cube tile）和 F-10（baseline vector tile）已完成
- **检查方法**: 追踪每个 matmul 的下游邻居操作链，判断是否存在 reduce / 纯 vector / 下一个 matmul / 计算结束；检查是否存在共享依赖的并发 matmul
- **操作指南**: tune-frontend SKILL.md「邻居算子感知的 TileShape 优化」
- **议程来源**: 阶段A「A4 分段优化议程表」（tune-frontend SKILL.md §A4）以结构性判定提供本优化点在各段/边界的适用性与优先级（附证据），本步骤的实际执行在阶段C ITER 中完成（实测+精度回归）。
- **典型收益**: 5-15%（取决于邻居模式和 tile 匹配程度）
- **值输出形式**: R1 变体空间由 F-9 分档分支实例化（各轴 cap、dtype 以该算子 case 标量代入为准），nL0 上限由 L0B 容量约束计算；R1 回退沿「变体阶梯」重指派——每档是对 if/elif/else 分支参数（分支内 int 字面量值、分支条件边界）的修改，非字面替换。R3 `vec_tile=[cube_mL0, cube_nL0]` 从 cube L1 tile 分支预计算。UB 溢出/回退沿变体阶梯（整shape→m/2→m/4，下限 1 行）重指派；R4 整shape tile 用 view/assemble 目标张量的标量维度计算。💡 R3/R4 需在 matmul 输出/view 站点插入新的 `set_vec_tile_shapes` 调用——成本可忽略，不要因调用数量犹豫。放置/退化见 §1b。
- **五条规则**:

  **Rule 1: Matmul→Reduce — 最大化 n，减小 k**
  - **触发**: matmul 输出在下一个 matmul 或计算结束前经过 reduce（sum/amax/amin/mean）
  - **策略**: 最大化 nL0/nL1（减少 N 轴切分 → 减少 partial reduce 汇聚开销），减小 kL0 腾出 L0B 空间
  - **变体搜索**: kL0 ∈ {64, 128, 256} × nL0 ∈ {128, 256}
  - **约束**: L0B: `kL0×nL0×dtype_bytes ≤ L0B_size`; L0C: `mL0×nL0×dtype_bytes ≤ L0C_size`; L1 ≤ L1_size（§一）; CUBE_ALIGN 对齐（§一）

  **Rule 2: 并发 Matmul — 核数匹配**
  - **触发**: 多个 matmul 共享相同依赖（同一 ready tensor / 同一 vec/cube 操作输出）
  - **计算**: `kernel_count = Σ ceil(M_i/mL1_i) × ceil(N_i/nL1_i)`（L1 tile 决定核分派粒度，见 matmul-l2-layout.md）
  - **Cube 核数**: cube_units（来自 npu-memory-arch.md §二）
  - **策略**:
    - `kernel_count < cube_units`: 减小 mL1 优先，然后 nL1（增加并行度填满核）
    - `cube_units ≤ kernel_count ≤ 2×cube_units`: 当前合理，微调
    - `kernel_count > 2×cube_units`: 增大 nL1 优先，然后 mL1（减少调度开销）
  - **⚠️ L0/L1 耦合约束**: Rule 2 调 L1，L0 仍须满足 `L0<=L1 && L1%L0==0`（compile-time；减小 mL1 下限=mL0）+ mL0≥64（S-14 CV，优先 128，禁 CUBE_ALIGN）+ L0B/L0C 预算；增大 nL1 须 `nL1%nL0==0` 且 L1 ≤ L1_size（§一）

  **Rule 3: Matmul→Vector(无reduce) — Vector tile 跟随 Cube tile**
  - **触发**: matmul 后只有 vector 操作（cast/mul/add 等），无 reduce，到下一个 matmul 或结束
  - **策略**: `vec_tile = [cube_mL0, cube_nL0]`；UB 溢出时**递归减 M**（`m/2→m/4→...`，下限 1，`cube_mL0%vec_m==0`）直至 fit UB 有效预算（`UB_effective = UB − Σ live intermediates`，非全部 UB）；M=1 仍溢出则**减 N**（`n/2→...`，下限 1）。⛔ 不是一步 `[m/2,n]`——L0C > UB 的平台（950PR，§一）带 intermediates 常需多次减行；L0C < UB 的平台（910B2）通常 0-1 次
  - **保留 N 原因**: N 是 CV 交接轴（L0C→UB），`vec_n = cube_nL0` = 1:1 对齐最安全；先减 M 保持 N 1:1 尽久，M 触底才切 N
  - **终止性**: vec_m=1, vec_n=1 → 单元素 ≤ 8B << UB，保证终止
  - **S-14 caveat**: Mix合图 场景 vec tile 两维 CUBE_ALIGN 对齐（§一）（`vec_m∈{1,16,32,...}`，1 为 floor 例外）；非 Mix 场景 vec_m 可取 8/4/2

  **Rule 4: View/Assemble/Reshape — 整shape tile（零搬移优先）+ 邻居 tile 传播回退**
  - **触发**: kernel 中存在任意 `pypto.view` / `pypto.assemble` / `pypto.reshape`（元数据/定位类操作，不产生计算）
  - **策略**: 在每个 view/assemble/reshape 前设 `set_vec_tile_shapes(*whole_shape)`（整shape、单 tile）。整shape tile 对任意 size 有效——核心判据是**链级无真实数据搬移**，而非容量
  - **链级前提（任一环节真实搬移则整链失效）**: view 恒为零搬移寻址；reshape 需 `inplace=True` 或尾轴对齐（整shape tile 下自动满足：tile 尾轴 == shape 尾轴）；assemble 需无冲突模式（无 offset 不完全重叠的多个 inplace 写入者、无跨 scope 中转，后者即 A-D3 反模式）；链内 offset 静态可追溯
  - **回退（邻居传播，不引入中间档）**: 链内存在真实搬移或设置报错时，前向继承最近前驱 op 的 vec_tile；无前驱则后向回传最近后继 op 的 vec_tile；继承后复检（每维 ≤ tensor 实际维度、reshape 链尾轴未切分、assemble 无冲突模式）
  - **Guard**: 链内 reshape 必须切分尾轴时真实搬移确定发生（编译器改写为 COPY_OUT→reshape→COPY_IN），tile 调整无法消除 → F-4 inplace 外提 / I-4 增大尾轴 / 调整 tile 对齐尾轴
  - **优先级**: view/assemble 直接消费 matmul 输出（CV 衔接）时 Rule 3 优先；Rule 4 适用于**所有** view/assemble/reshape 站点——含 matmul 输入侧 view（矩阵输入 A/B 操作数均按 view→matmul 对齐约束处理；scale_a/scale_b 等非矩阵操作数除外），不限于 vector 产生的 tensor 链
  - **判定依据**: view/assemble/reshape 在 IR 中均为 in-place 寻址节点，真实搬移仅出现在 reshape 尾轴与 raw 布局尾轴不一致、多写者 offset 冲突/动态 offset 不可追溯、跨 scope 经中间 buffer 三种情形
  - **⛔ view→matmul 链硬性要求**：当 view 结果作为 matmul 矩阵输入（A 或 B 操作数）时：
    - view 的 vec_tile 必须 == matmul 对应 L1 tile（A 按 a_trans 映射表、B 按 b_trans 映射表），必须在 view 前 set（`set → view → reset` 模式）
    - **不可豁免**——不可标记为 ⏳ 或跳过，即使 tuning case 精度通过也不行
    - **原因 (1) 性能**：不匹配增加 VEC-CUBE 交接延迟（AIV GM→UB 预取分块 ≠ AIC GM→L1B 装载分块 → 额外 GM 读 roundtrip）
    - **原因 (2) 精度稳定性**：未测试 case（decode 类：loop_count=1、batch 维 ≥16）上 GM 读调度冲突导致非确定性数据损坏
    - 满足条件 = view vec_tile == matmul L1 tile + trace-time assert（不是 tuning case 精度通过）
    - 详见 basic-block-optimization.md §3「view→matmul tile alignment 约束」（A/B 操作数完整映射表）
  - **多消费者 UB 约束**：当 view 同时被 matmul（A/B 操作数）和其他 vector op 消费时：
    - **matmul-only**：vec_tile = matmul L1 tile（硬约束，无需 UB 检查——matmul 经 L1 装载，不经 UB）
    - **matmul + vector**：vec_tile = matmul L1 tile，**且须验证 L1 tile 满足 UB 预算**（vector 消费者经 UB 装载，L1 tile 超 UB 则 vector 侧溢出）。L1 cube tile **不修改**（matmul 经 L1 装载，不受 UB 约束）
    - UB 预算 = 全部 UB_size（§一；view op 本身不占 UB）
    - L1 tile 超 UB → **仅减小 view vec_tile**（m/2 → m/4 → ...，同 Rule 3 递归减行）直至 fit UB；matmul L1 tile 保持不变。N 保持（CV 交接轴）至 M=1 再减 N。vec_tile 可 < L1 tile——可接受，因 vector 消费者需要 UB 适配而非 L1 对齐

- **S-14 Mix合图 兼容性**（7 项检查）:
  1. nL0 是 vec tile n 的整数倍（或通过 Rule 3 同步）
  2. mL0 ≥ 64（优先 128），禁止 CUBE_ALIGN
  3. nL0 同 Check 1 约束
  4. vec tile 两维 CUBE_ALIGN 对齐（§一）
  5. UB ≤ UB_size（§一），独立于 L0C 是否溢出
  6. UB→L1 内轴 32B 对齐（若 vector 回喂 cube）
  7. 若 S-14 可能使用（A5），优先标准值（~128）

- **F-20 内部执行顺序**: Rule 1&2（最终确定 cube tile）→ Rule 3（vec tile 跟随）→ Rule 4（view/assemble/reshape 整shape tile + 邻居传播回退）→ S-14 兼容性检查
- **⛔ view→matmul tile alignment 约束（强制，须 trace-time assert）**: 当 `view` 的结果作为 matmul 矩阵输入（A 或 B 操作数）时，view 的 vec_tile 各维度必须 == matmul 对应 L1 tile 维度。转置映射——**A 操作数（按 a_trans）**：`a_trans=False` → view dim0=M↔`m_l1`、view dim1=K↔`k_l1`；`a_trans=True` → view dim0=K↔`k_l1`、view dim1=M↔`m_l1`。**B 操作数（按 b_trans）**：`b_trans=True` → view dim0=N↔`n_l1`、view dim1=K↔`k_l1`；`b_trans=False` → view dim0=K↔`k_l1`、view dim1=N↔`n_l1`。同一 view 同时喂 A/B 两个操作数时，两个映射须同时满足（分别核对）。不匹配时在未测试 case 上产生非确定性精度错误（GM 读调度冲突）。规则：(1) view tile 从 L1 tile 派生（不独立设置）；(2) 改一处必须同步改另一处；(3) 必须有 trace-time assert（未测试 case 编译期失败）。详见 basic-block-optimization.md「view→matmul tile alignment 约束」（A/B 操作数完整映射表）。
- **关联优化**: F-9（Cube TileShape 基线）、F-10（Vector TileShape 基线，Rule 3 修改）、F-16（多 Matmul 差异化）、F-17（V 段 shape 变化）、F-2/F-3（循环体计算量/循环次数）、S-14（Mix合图，Rule 3 自然满足 §4.2.5b）、S-11（深度 Cube TileShape 调优）、F-14/I-11（assemble 零额外分配 / view 零搬运复用，Rule 4 判定依据）、I-9（valid_shape 尾块零填充）

### [F-21] 操作链依赖深度外提

- **阶段**: 开箱调优
- **优先级**: ⭐⭐⭐ P0
- **适用条件**: 嵌套循环（≥2 层）体内存在依赖部分（非全部）循环变量的操作链（reshape/squeeze/unsqueeze 由 F-4 方式 1-4 处理，不适用本优化点）
- **检查方法**: A3 依赖深度分析表（tune-frontend SKILL.md §A3）逐操作链统计依赖的循环变量数量，分类为完全静态/部分依赖/完全依赖
- **操作指南**: tune-frontend SKILL.md §2.1 → references/loop-hoisting.md
- **典型收益**: 5-20%（消除热循环内部分依赖操作链的冗余重复计算）
- **为什么**: F-4 方式 1-4 仅覆盖 reshape/squeeze/unsqueeze；F-5 改循环构造类型不外提操作链；F-19 Mode A 批量化 M 轴、Mode B 按 tile 可扩展性外提——均不按依赖深度分类。嵌套循环中部分依赖操作链（如因果掩码依赖 1/N 循环变量）的冗余计算无其他 F-step 消除
- **核心原则**:
  - 依赖 0 个循环变量（完全静态）→ 外提至所有循环之前，执行一次
  - 依赖少于全部层数（部分依赖）→ 独立 `pypto.loop` 预计算至 GM buffer，主循环内 view 消费
  - 依赖全部层数（完全依赖）→ 保留在循环体内
- **约束**:
  - 预计算 loop 须为 `pypto.loop`（非 Python `for range`，否则 trace-time 求值问题）
  - 预计算 buffer 在 GM（非 UB），不超过 GM 预算
  - 预计算 loop 须与主 loop 无数据依赖（或仅通过 GM buffer 传递）
  - 主循环内 view 的 vec tile 须覆盖 buffer 列维（F-20 Rule 4 Step 1a/1b）
- **关联优化**: F-4（reshape 外提，方式 1-4）、F-19（Mode A Step 2 引用本优化点外提静态操作）、F-5（静态轴改 Python for，不同机制）

---

### [S-20] max_workspace_kb + host_options 完整性检查

- **阶段**: 深度调优
- **优先级**: ⭐⭐⭐ P0（强制前置）
- **适用条件**: 所有算子（NPU 编译输出含 workspace 推荐值时强制）
- **检查方法**: 检查 NPU 编译输出中是否包含 "Recommended: set max_workspace_kb near XXX KB" 提示，若有则必须设置
- **操作指南**: tune-swimlane SKILL.md §10 + 主 SKILL.md §2.1
- **典型收益**: 5-30%（激活 memory-driven mode，优化调度）
- **核心原则**:
  - **max_workspace_kb**：NPU 编译输出会给出推荐值（如 "Recommended: set max_workspace_kb near 1607648KB"），必须按推荐值设置
  - **host_options**：`{"compile_monitor_enable": 0}` 减少编译监控开销
  - **检查时机**：S2_COLLECT 阶段首次运行时，检查 stdout 中的 workspace 推荐提示
- **配置示例**:
  ```python
  @pypto.frontend.jit(
      runtime_options={
          "max_workspace_kb": 1607648,  # 从 NPU 输出推荐值获取
      },
      host_options={"compile_monitor_enable": 0},
  )
  ```
- **关联优化**: S-9（Stitch 调优，workspace 影响 stitch 并行度）、S-14（Mix合图，workspace 影响 CV 调度）

### [S-21] Mix合图多 scope 策略（Mix + 普通合图协同）

- **阶段**: 深度调优
- **优先级**: ⭐⭐⭐ P2+（A5 平台 + 最后一个 Cube 后的 V 段涉及跨迭代依赖）
- **适用条件**: A5 平台 + loop 体内最后一个 Cube 之后的 V 段涉及跨迭代依赖（V 段中写入在 loop 外声明、loop 内读写的 tensor），或有多段无数据依赖的 CV 段可分别独立合图
- **前置条件**: S-14 已进入调优流程（Mix合图 Step 0 数据流分析已完成）
- **检查方法**: 按 §11 scope 划分条件诊断——识别跨迭代依赖 tensor → 判断最后一段 V 是否涉及 → 决定是否切 scope。分析多段 CV 交替段是否有数据依赖，无依赖的可分独立 Mix scope
- **操作指南**: tune-swimlane SKILL.md §11 + merge-optimization.md §4.1 Step 0b 规则3
- **典型收益**: 5-15%（消除放出 V 段的子图间调度开销，或多段 CV 独立合图提升并行度）
- **多 scope 策略**:
  - **策略1：Mix scope 放出的 V 段用独立 sg_set_scope 做普通合图**
    - CV交替段（V0→C1→V1→C2）用一个 scope 包裹
    - V2 update 段（涉及跨迭代依赖，需从 Mix scope 放出）用独立 scope 包裹
  - **策略2：多段无数据依赖的 CV 段分为多段独立 Mix scope**
    - 第1段独立的 CV 交替段（V10→C11→V11→C12...）用一个 scope 包裹
    - 第2段独立的 CV 交替段（V20→C21→V21→C22...）用一个 scope 包裹
    - 第1段和第2段生成数据的计算用独立 scope 包裹
- **核心原则**:
  - **scope ID 说明**：scope ID 无功能差异，仅作唯一标志，每个 scope 不重复即可；Mix 用大 ID、普通用小 ID 仅为便于阅读
  - **放出的 V 段特征**：写入在 loop 外声明、loop 内读写的 tensor（构成跨迭代依赖），不能走 CV 通路
- **代码示例**:
  ```python
  # === 策略1：Mix scope + 放出V段普通合图 ===
  # Mix scope: V0→C1→V1→C2
  pypto.set_pass_options(sg_set_scope=20001)
  # ... V0 gather + dequant + C1 matmul + V1 softmax + C2 matmul ...
  pypto.set_pass_options(sg_set_scope=-1)

  # 普通合图 scope: V2 online softmax update
  pypto.set_pass_options(sg_set_scope=1)
  # ... V2: max/sum/exp/oi_update operations ...
  pypto.set_pass_options(sg_set_scope=-1)

  # === 策略2：多段独立 Mix scope ===
  # 第1段 CV 交替段
  pypto.set_pass_options(sg_set_scope=10001)
  # ... V10→C11→V11→C12 ...
  pypto.set_pass_options(sg_set_scope=-1)

  # 第2段 CV 交替段（与第1段无数据依赖）
  pypto.set_pass_options(sg_set_scope=10002)
  # ... V20→C21→V21→C22 ...
  pypto.set_pass_options(sg_set_scope=-1)

  # 两段生成数据的后续计算
  pypto.set_pass_options(sg_set_scope=3)
  # ... 合并/归约操作 ...
  pypto.set_pass_options(sg_set_scope=-1)
  ```
- **关联优化**: S-14（A5 Mix合图）、S-4（Vector 手动合图）
