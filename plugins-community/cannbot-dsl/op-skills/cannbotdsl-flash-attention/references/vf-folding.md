# `vf` fusion 区域 — 折叠规律

`vf`（vector-fold）把多个 vec op 融合到同一个 AscendC vector loop 里，消掉中间 UB 写读。这是 FA 性能最大的旋钮——folding 减少 UB 占用 + 减少 vec 循环 overhead。

## 三条铁律

### 规则 1：`vf` 内不许 runtime 分支

`scf.if` 在 vf 区域内**不会**被折叠（lowering 撑不开分支的 join 点）。

后果：variant D 把 `softmax_round_b_finalize` 拆成 4 个直线函数：
- `first_has_second`
- `first_no_second`
- `more_has_second`
- `more_no_second`

而不是写一个带 `if has_second: ...` 的统一版本。分支在 vf 外面判（python 层），里面纯直线。

如果实在需要"条件不同"，要么从外面分发到不同 vf，要么把分支变成数据上的 mask（`vselect`）。

### 规则 2：`outputs=[...]` 必须列全 vf 之外被读的 buffer

```python
with vf(outputs=[qk_ub_a, p_ub_a, m_a_partial_ub, sum_a_partial_ub]):
    muls(qk_ub_a, qk_ub_a, scale)
    reduce_max(m_a_partial_ub, qk_ub_a)
    ...
    cast(tmp_tile_ub_fp8, qk_ub_a)
    mem_copy(p_ub_a, tmp_tile_ub_fp8, engine=nd2nz)
```

`outputs` 决定哪些 store 保留。**没列**的 store 被 vf 当作"区域内的临时量"消掉。漏一个（典型如 `sum_a_partial_ub`），它的 store 被丢，跨 macro 累加值静默变垃圾——**无编译报错**。

排查的简单原则：vf 区域之外**任何**会被读的 buffer，必须在 `outputs` 里。重复列不是性能代价，只是保证 store 不被吃。

### 规则 3：`vf` 区域内 program order 保留（RaW 可信）

同一个 vf 内：
- 先写 `qk_ub_a`
- 后读 `qk_ub_a`

lowering 会把它们装到同一个 AscendC vector loop 的同一 iteration 内，read-after-write 是单元素粒度安全的。不需要插同步。

但跨 vf 区域，RaW 就回到 PIPE 调度（vec_sync_intra）那一套了。

## 哪些 op 能折掉、哪些折不掉

### 容易折掉

- 同 shape / 同 align 的 elementwise（`add`, `sub`, `mul`, `muls`, `exp`, `cast`）
- `reduce_max` / `reduce_sum` → `expand` 给下一个 elementwise——同 inner-loop trip 时
- `vselect`

### 容易折不掉

- **跨 `cast` → `mem_copy(nd2nz)` 边界的中间 UB**：cast 段用 fp32 → fp8 的 store；mem_copy 段用 NZ 重排的 load。两段 inner-loop trip 不同（storealign / loadalign 不匹配），中间的 `tmp_tile_ub_fp8` 写存活、读存活——**它仍占 UB**。在 variant D 计划里这是 UB 预算大踩坑的点之一。
- `mem_copy(gm, ub)` 与 `mem_copy(ub, gm)`：这些走 MTE2 / MTE3 不走 V，根本不在 vf 调度范围内。
- 不同 shape 的 broadcast（`expand` 跨 row 重复）有时撑不开。

预算紧时**默认假设什么都不折**。折叠与否取决于 cast → `mem_copy(nd2nz)` 边界两侧的 storealign / loadalign 是否匹配（见本页规则），无法从源码一眼看出，因此不要把"应该会折掉"当预算依据。

## variant D 路径的 vf 布局参考

```python
# Round A（首半 K chunk）
with vf(outputs=[qk_ub_a, m_a_partial_ub, sum_a_partial_ub, p_ub_a]):
    if not has_atten_mask_already:
        vselect(qk_ub_a, ...)
    muls(qk_ub_a, qk_ub_a, scale)
    reduce_max(m_a_partial_ub, qk_ub_a)
    expand(tmp_tile_ub, m_a_partial_ub)
    sub(qk_ub_a, qk_ub_a, tmp_tile_ub)
    exp(qk_ub_a, qk_ub_a)
    muls(qk_ub_a, qk_ub_a, scale_p)            # 提前到 reduce_sum 之前
    reduce_sum(sum_a_partial_ub, qk_ub_a)
    cast(tmp_tile_ub_fp8, qk_ub_a)              # 可能不折叠
    mem_copy(p_ub_a, tmp_tile_ub_fp8, engine=nd2nz)

store_p(p_l1_chunk_a, p_ub_a)                   # vf 外，MTE3

# Round B 4 个变体之一同理
```

## 常见坑回顾

| 现象 | 元凶 |
| --- | --- |
| 跨 macro 的 partial sum 是垃圾 | `sum_a_partial_ub` 漏列 outputs |
| vf 编译报错 / lowering 没动 | 区域内有 `scf.if` |
| UB 实占比 tally 多一截 | cast→nd2nz 边界的 tmp fp8 没折 |
| 同 buffer 先写后读结果错 | 不是 RaW，是 align 问题或者两个 op 分到不同 vf 但你以为同一个 |
