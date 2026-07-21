# CSV 系统化分析框架（Phase 1 轻量级分析）

**来源**：整合自 triton-ascend-dev-main/guides/csv-interpretation.md 的方法论
**适用场景**：AscendC 算子的 msprof op 性能分析 - **仅使用 op_summary.csv**
**数据来源**：`workspace/InputMessages/raw/{op}/profiling_data/profiling_csv/op_summary*.csv`
**分析成本**：低（秒级，无需额外 profiling）
**配合使用**：`op_summary_header_guide.md`（字段定义）

> **渐进式披露设计**：本框架用于 Phase 1 快速分析。当 Phase 1 建议无效时，再进入 Phase 2 使用 `deep_research` 子技能做流水图/指令级深度分析。

---

## 前置理解：block/sub_block 机制

从 `OpBasicInfo.csv` 或 `op_summary.csv` 中可以看到：

- **Block Dim**：该算子启动的 block 数量（并行工作组数）
- **Task Type**：执行模式
  - `MIX_AIC`：混合模式，同一 block 下有多个子执行器（cube + vector）
  - `AI_CORE`：纯 cube 模式
  - `AI_VECTOR_CORE`：纯 vector 模式

### CSV 中的 sub_block_id 对应关系

| sub_block_id | 执行单元 | 相关指标前缀 |
|--------------|----------|--------------|
| `cube0` | Cube/AICore（矩阵/张量计算） | `aic_*` |
| `vector0`, `vector1` | Vector/AIVector（向量/标量计算、传输） | `aiv_*` |

### 正确的对比方法

1. **同一 block_id 内**：对比 `cube0` vs `vector0/1`
   - 目的：检查子执行器间的瓶颈是否一致
   - 示例：cube 等待 vs vector 等待

2. **跨 block_id 统计**（mean/min/max）：
   - 目的：检查 block 间是否存在高方差
   - 高方差可能表示：访存不均、分支/mask 导致负载不均

---

## 8 维度系统化分析框架

### 维度 1：基本信息（OpBasicInfo）

**关键字段**：
- `Op Name`, `Op Type`, `Task Type`
- `Task Duration(us)`：**最重要的宏观指标**，用于版本间对比
- `Block Dim` / `Mix Block Dim`

**如何使用**：
- `Task Duration` 是算子整体时长，适合横向对比不同优化版本
- 其他维度的时间（`aic_time/aiv_time`）是子执行器分解，可能有并行/重叠，不一定加和到 Task Duration

---

### 维度 2：计算利用率（ArithmeticUtilization）

**关键字段**：

| 字段 | 含义 | 调优信号 |
|------|------|----------|
| `aic_cube_ratio` (或 `aic_mac_ratio`) | Cube 计算占比 | **低值** → 计算未饱和，可能受访存/同步/控制流影响 |
| `aic_cube_fp16_ratio` / `int8_ratio` | 数据类型分布 | 用于确认是否使用了预期的数据类型 |
| `aiv_vec_ratio` | Vector 计算占比 | **低值** → Vector 侧受等待/传输/分支影响 |
| `aiv_vec_misc_ratio` | Misc 指令占比 | **高值** → 标量/控制开销高 |

**判断逻辑**：
```
IF (aic_cube_ratio < 0.3 OR aiv_vec_ratio < 0.5) THEN
    → 不是"计算饱和"瓶颈
    → 检查访存、同步、控制流
ELSE
    → 可能是计算bound
    → 检查算法优化空间
END IF
```

---

### 维度 3：流水线分解（PipeUtilization）

**关键字段**：

#### Cube 侧
| 字段 | 含义 | 调优信号 |
|------|------|----------|
| `aic_cube_time/ratio` | Cube 计算时间/占比 | - |
| `aic_scalar_time/ratio` | 标量/控制流时间/占比 | **高值** → 减少标量阶段、合并小算子、避免频繁同步 |
| `aic_mte1/2/3_time/ratio` | 各传输引擎时间/占比 | **高 mte2_ratio** → 传输占比高，检查访存粒度/对齐/复用 |

#### Vector 侧
| 字段 | 含义 | 调优信号 |
|------|------|----------|
| `aiv_vec_time/ratio` | Vector 执行时间/占比 | - |
| `aiv_scalar_time/ratio` | Vector 标量时间/占比 | **高值** → 标量开销大 |
| `aiv_mte2/3_time/ratio` | Vector MTE 时间/占比 | **高值** → 传输/访存占用高 |

---

### 维度 4：资源冲突与等待（ResourceConflictRatio）

**关键字段**：

| 字段 | 含义 | 调优信号 |
|------|------|----------|
| `aic_cube_wait_ratio` | Cube 等待比例 | **高值** → 等待资源/数据/调度 |
| `aic_mte*_wait_ratio` | MTE 等待比例 | **高值** → 传输引擎等待 |
| `aiv_vec_wait_ratio` | Vector 等待比例 | **高值** → Vector 等待 |
| `aiv_vec_total_cflt_ratio` | Vector 总冲突比例 | **高值** → Bank 冲突 |
| `aiv_vec_bank_cflt_ratio` | Bank 冲突细节 | **高值** → 访存模式导致 bank 冲突 |

**关键组合判断**：
```
IF (wait_ratio > 0.4 AND bandwidth_usage < 0.2) THEN
    → 不是带宽饱和，而是延迟/停顿
    → 原因：依赖链、访存碎片、非连续突发
    → 优化方向：减少依赖链、增大访存粒度、改善对齐
END IF
```

---

### 维度 5：内存带宽（Memory）

**关键字段**：

| 字段类型 | 示例字段 | 调优信号 |
|----------|----------|----------|
| 带宽（GB/s） | `aic_main_mem_read_bw`, `aiv_gm_to_ub_bw` | 实际带宽数值 |
| 数据量（KB） | `read_main_memory_datas`, `GM_to_UB_datas` | 实际搬运数据量 |
| 使用率（%） | `GM_to_UB_bw_usage_rate(%)` | **低值**（< 10%）→ 不是带宽饱和 |

**关键判断**：
```
IF (GM_to_UB_bw_usage_rate < 10%) THEN
    → 不是"带宽瓶颈"
    → 结合 wait_ratio 看，更可能是：
       - 访存粒度小
       - 对齐不好
       - 依赖链长导致流水填不满
END IF
```

---

### 维度 6：L0 缓存（MemoryL0 - 主要针对 Cube）

**关键字段**：
- `aic_l0a_*_bw`, `aic_l0b_*_bw`, `aic_l0c_*_bw_cube`

**如何使用**：
- 主要用于**版本间对比**，看 tile/layout 改变对 L0 的影响
- 绝对数值难以判断好坏，关注异常抖动或突然归零

---

### 维度 7：UB 缓存（MemoryUB - 主要针对 Vector）

**关键字段**：
- `aiv_ub_read_bw_vector`, `aiv_ub_write_bw_vector`

**如何使用**：
- UB 带宽稳定 → UB 内部不是主要瓶颈
- 瓶颈更可能在 GM<->UB 或 等待/依赖

---

### 维度 8：L2 缓存命中率（L2Cache）

**关键字段**：

| 字段 | 含义 | 调优信号 |
|------|------|----------|
| `aic_write_hit_rate(%)` | Cube 写命中率 | - |
| `aic_read_hit_rate(%)` | Cube 读命中率 | - |
| `aic_total_hit_rate(%)` | Cube 总命中率 | **高方差** → 访存模式不均 |
| `aiv_*_hit_rate(%)` | Vector L2 命中率 | **高方差** → scatter/gather 或非连续访问 |

**判断逻辑**：
```
IF (total_hit_rate 整体高 BUT 个别 block 显著低) THEN
    → 访存模式不均
    → 可能原因：尾块、mask、对齐、stride
    → 优化：调整 tile 策略、padding
END IF
```

---

## 信号 → 行动决策表

| 信号模式 | 优先行动 | 次要行动 |
|---------|----------|----------|
| **低计算利用率 + 高 MTE2 比例** | 增加 tile 复用、减少中间写回、改 layout 使 load 连续、合并小粒度 load | 调整 BLOCK_SIZE 以获得更好对齐 |
| **高等待比例 + 低带宽使用** | 减少依赖链、减少同步点、减少小粒度访存、检查 Pipe 流水 | 关键 tensor 做 padding/对齐 |
| **高 scalar 比例** | 标量逻辑移到编译期、减少分支和 mask、合并零散 elementwise op、使用向量化接口 | - |
| **L2 命中率方差大** | 检查访问是否连续、是否有 scatter/gather、重新排序或调整 blocking 策略 | Padding 优化、改变数据布局 |
| **高 bank 冲突** | 调整数据访问模式、改变 stride、使用不同的 buffer 布局 | 增加 buffer 数量分散访问 |

---

## 典型瓶颈模式识别

### 模式 1：访存效率低
**信号组合**：
- 低计算利用率（`aic_cube_ratio < 0.3`）
- 高等待比例（`aic_cube_wait_ratio > 0.4`）
- 低带宽使用率（`GM_to_UB_bw_usage_rate < 10%`）

**根因**：
- 访存粒度小（非连续突发）
- 对齐不好
- 依赖链长

**优化方向**：
1. 增大 tile size，提高访存粒度
2. 检查数据对齐，必要时 padding
3. 使用 Pipe 流水隐藏访存延迟
4. 减少同步点

---

### 模式 2：标量/控制流开销高
**信号组合**：
- 高 scalar 比例（`aic_scalar_ratio > 0.3` 或 `aiv_scalar_ratio > 0.3`）
- 低向量/cube 利用率

**根因**：
- 显式 for 循环过多
- 分支和 mask 触发频繁
- 小算子碎片化

**优化方向**：
1. 使用硬件计数模式（如 `SetVectorMask<COUNTER>`）
2. 合并小算子
3. 减少分支，使用 select 替代 if-else
4. 向量化标量操作

---

### 模式 3：流水不平衡
**信号组合**：
- Cube 和 Vector 利用率差异大
- 某个 MTE 占比特别高

**根因**：
- CV 融合不当
- 数据搬运和计算没有重叠

**优化方向**：
1. 调整 CV 分工
2. 使用 Double Buffer / Ping-Pong
3. 调整 Pipe 流水策略

---

## 分析工作流（推荐顺序）

```
Step 1: 查看 OpBasicInfo
  ├─ 确认 Task Type、Block Dim、Task Duration
  └─ 建立整体认知

Step 2: 查看计算利用率（ArithmeticUtilization）
  ├─ 检查 cube_ratio、vec_ratio
  └─ 判断是否计算饱和

Step 3: 查看资源冲突（ResourceConflictRatio）
  ├─ 检查 wait_ratio
  └─ 判断是否在等待

Step 4: 结合内存带宽（Memory）
  ├─ 检查 bandwidth usage rate
  └─ 判断是带宽瓶颈还是延迟瓶颈

Step 5: 查看流水线分解（PipeUtilization）
  ├─ 检查各 stage 占比
  └─ 找出时间消耗最多的阶段

Step 6: 查看 L2 缓存（L2Cache）
  ├─ 检查命中率及方差
  └─ 判断 cache 行为是否异常

Step 7: 根据"信号→行动决策表"确定优化方向
```

---

## 与现有文档的配合使用

| 文档 | 用途 | 使用时机 |
|------|------|----------|
| `op_summary_header_guide.md` | 查询字段定义 | 遇到不认识的字段时 |
| 本文档 | 系统化分析框架 | 进行 CSV 数据分析时 |
| `profiling_interpretation_template.md` | 流水图解读模板 | 结合流水图分析时 |
| `msprof_usage_guide.md` | msprof 工具使用 | 需要生成 profiling 数据时 |

---

## 示例：组合判断

**场景**：某 GEMM 算子性能不达预期

**Step 1**：查看基本信息
- `Task Duration = 6.0 us`
- `Task Type = MIX_AIC`
- `Block Dim = 2`

**Step 2**：查看计算利用率
- `aic_cube_ratio = 0.137`（**低**）
- `aiv_vec_ratio = 0.450`（**中等偏低**）
- → 结论：不是计算饱和

**Step 3**：查看资源冲突
- `aic_cube_wait_ratio = 0.45`（**高**）
- `aiv_mte2_wait_ratio = 0.78`（**非常高**）
- → 结论：大量时间在等待

**Step 4**：查看带宽
- `GM_to_UB_bw_usage_rate = 5.95%`（**极低**）
- → 结论：不是带宽饱和，而是访存效率低

**综合判断**：
- **瓶颈类型**：访存效率低（延迟/停顿）
- **优化方向**：
  1. 增大 tile size（提高访存粒度）
  2. 检查数据对齐
  3. 使用 Pipe 流水隐藏访存延迟
  4. 减少同步点

---

**最后更新**：2026-02-24
**维护者**：code-performance-advisor skill
