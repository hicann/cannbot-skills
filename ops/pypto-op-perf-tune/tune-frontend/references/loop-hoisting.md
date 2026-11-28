#### 操作链依赖深度外提（F-21）

**核心目标**：消除嵌套循环内冗余计算——操作链依赖的循环变量数量少于全部嵌套层数时，按依赖深度外提（完全静态直接外提、部分依赖预计算至 GM buffer），热循环内仅保留完全依赖的操作。

##### 为什么与何时

**何时应用**：
- 阶段 A3（依赖深度分析，tune-frontend SKILL.md §A3）发现循环体内操作链依赖的循环变量数量少于全部嵌套层数时
- 必须在 P0 阶段完成（先于 P1 F-19 与 P2 tile 基线）——F-19 Mode A Step 2 引用本优化点外提静态操作；未完成则批量化后逐 unroll 块重复计算放大开销

**为什么不由其他 F-step 覆盖**：
- F-4 方式 1-4 仅处理 reshape/squeeze/unsqueeze 的外提，不覆盖 view/full/le/arithmetic 等通用操作链
- F-5 将静态轴的 `pypto.loop` 改为 Python `for range`（改变循环构造类型），不外提操作链
- F-19 Mode A 批量化 matmul M 轴、Mode B 按 tile 可扩展性 + 共享输入外提——两者均不按依赖深度分类
- F-6 合并独立 loop（合并而非外提）

**典型场景**：
- 因果掩码：`s1+offset → full → le` 依赖 1 个循环变量（s1），嵌套在 B×N2×S1 三层循环内 → 预计算 S1 行至 GM buffer，热循环内仅 view（消除 B×N2 倍冗余计算）
- 位置编码查找表：依赖 1 个循环变量 → 同上
- 阈值/常量计算：完全静态（依赖 0 个）→ 外提至所有循环之前

**不适用**：
- 单层循环（无嵌套 → 不存在部分依赖）
- 操作链依赖全部循环变量（完全依赖 → 保留在循环体内）
- 操作链是 reshape/squeeze/unsqueeze（F-4 方式 1-4 已覆盖）

##### 分类与外提方式

发现阶段的逐操作链依赖统计与分类由 A3「依赖深度分析表」（tune-frontend SKILL.md §A3）完成，本节给出分类到外提方式的映射：

| 依赖循环变量数量 | 分类 | 外提方式 |
|----------------|------|---------|
| 0（完全静态） | 循环不变量 | 外提至所有循环之前，执行一次 |
| 少于全部层数 | 部分依赖 | 移至仅迭代所依赖变量的独立 `pypto.loop`，结果 assemble 到 GM buffer；主循环内通过 `view(buffer, [1, ...], [loop_var, 0])` 消费 |
| 全部层数 | 完全依赖 | 保留在循环体内 |

##### 变换步骤

**Step 1**：逐操作链统计依赖的循环变量（跨全部外层循环）——A3 依赖深度分析表已产出，直接使用

**Step 2**：完全静态（依赖 0 个）→ 外提至所有循环之前

**Step 3**：部分依赖（少于全部层数）→ 在主循环之前创建独立 `pypto.loop`，仅迭代所依赖的变量；每次迭代执行操作链，结果 `assemble` 到 GM buffer `[loop_count, ...]`

**Step 4**：主循环内用 `view(buffer, [1, ...], [loop_var, 0])` 替代原操作链

**Step 5**：预计算 loop 中的 view 须设 F-20 Rule 4 Step 1b vec tile（`rows = UB_budget // (col × dtype_bytes)`）

##### 代码示例 — causal 预计算外提

```python
# BEFORE: causal 在热循环内每次重新计算（依赖 s1 1 个变量，嵌套 3 层 b→n2→s1）
for b in pypto.loop(B):
    for n2 in pypto.loop(N2):
        for s1 in pypto.loop(S1):
            thr = s1 + S2_minus_S1          # 依赖 s1
            thr_tensor = pypto.full([1, S2], thr, pypto.DT_FP32)
            s2_row = pypto.view(s2_range_2d, [1, S2], [0, 0])  # 完全静态
            causal_row = pypto.le(s2_row, thr_tensor)           # 依赖 s1
            mask_bool = pypto.logical_and(mask_bool, causal_row)

# AFTER: causal 预计算至 GM buffer，热循环内仅 view
# 预计算 loop（仅迭代 s1）
causal_2d = pypto.tensor([S1, S2], pypto.DT_BOOL)
s2_row = pypto.view(s2_range_2d, [1, S2], [0, 0])  # 外提至所有循环前
for s1_idx in pypto.loop(S1, name="LOOP_causal"):
    thr = s1_idx + S2_minus_S1
    thr_tensor = pypto.full([1, S2], thr, pypto.DT_FP32)
    causal_row = pypto.le(s2_row, thr_tensor)
    pypto.assemble(causal_row, [s1_idx, 0], causal_2d)

# 主循环内仅 view
for b in pypto.loop(B):
    for n2 in pypto.loop(N2):
        for s1 in pypto.loop(S1):
            causal_slice = pypto.view(causal_2d, [1, S2], [s1, 0])  # 替代 4 行操作
            mask_bool = pypto.logical_and(mask_bool, causal_slice)
```

##### 约束

- 预计算 loop 须为 `pypto.loop`（非 Python `for range`，否则 trace-time 求值问题）
- 预计算 buffer 在 GM（非 UB）；buffer 大小 `[N, col] × dtype_bytes` 须合理（不超过 GM 预算）
- 预计算 loop 须与主 loop 无数据依赖（或仅通过 GM buffer 传递）
- 主循环内 view 的 vec tile 须覆盖 buffer 列维（F-20 Rule 4 Step 1a/1b）
- 完全静态操作（如常量、不依赖循环变量的 `view`）直接外提至所有循环之前，无需独立 loop
