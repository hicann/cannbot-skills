# vf 折叠规律详解

`vf`（vector-fold）把多个 vec op 折叠到同一个 AscendC vector for-loop 里，是 vec 性能最大的旋钮。本文解释**它是怎么折叠的、为什么有时折不动**。

## 折叠的底层机制

AscendC 把一段 vector 计算翻译成一个 for-loop：

```c
// 没折叠前：两个 vec op 是两段 for-loop
for (int v = 0; v < N; v += vec_width) { muls(qk_ub + v, qk_ub + v, scale); }
for (int v = 0; v < N; v += vec_width) { exp(qk_ub + v, qk_ub + v); }

// vf 折叠后：一段
for (int v = 0; v < N; v += vec_width) {
    tmp_reg = muls(qk_ub_load(v), scale);
    qk_ub_store(v, exp(tmp_reg));
}
```

中间值 `tmp_reg` 是寄存器，不是 UB——这就是"消掉中间 UB 写读"。折叠成立的条件：两段 for-loop 的**stride、trip count、对齐**都必须相同，lowering 才能把它们装到一个 iteration 内。

## 折得动的 op 类型

| 类型 | 例子 | 为什么折得动 |
| --- | --- | --- |
| 同 shape elementwise | `add`, `sub`, `mul`, `muls`, `exp`, `cast`, `vselect` | inner-loop 完全一致 |
| `reduce_max` / `reduce_sum` → `expand` | softmax 内核 | reduce 出 (M, 1)，expand 广播回 (M, N)；同 inner trip |
| 连续 elementwise + reduce + elementwise | 全套 softmax | 都在 V PIPE 上，inner trip 由 vec_m × tile_n_qk 决定 |
| 同 vf 内 RaW | 先 `muls(buf, buf, scale)` 再 `exp(buf, buf)` | lowering 装到同一 iteration 内，元素粒度安全 |

## 折不动的几种典型

### 1. cast → mem_copy(nd2nz) 边界

```python
cast(tmp_tile_ub_fp8, qk_ub_fp32)
mem_copy(p_ub, tmp_tile_ub_fp8, engine=nd2nz)
```

`cast` 段：fp32→fp8 在 V PIPE 上跑，inner-loop 按 fp32 align（4 元素一拍）。
`mem_copy(nd2nz)` 段：把 ND 格式 fp8 重排成 NZ 格式 fp8，inner-loop 按 NZ block align（通常 16 元素一拍）。

两段的 inner-loop trip count 与 align 不同——lowering 撑不开把它们装到同一 iteration。结果：`tmp_tile_ub_fp8` 的 store（cast 输出）保留、load（mem_copy 输入）保留，它**仍然占 UB**。

预算紧时，这是常见的"以为折了实际没折"的坑。验证方法：

```bash
grep "tmp_tile_ub_fp8" /tmp/asc
# 看见 storealign 指向它 = 没折
```

FA 蓝本里这个 tmp_fp8 一直是大头，~16 KB UB。variant D 实验里发现"假设折了"算出来的预算每次都对不上实测，最终是按"不折"算预算才稳。

### 2. mem_copy(gm, ub) / mem_copy(ub, gm)

```python
with vf(outputs=[...]):
    ...
    mem_copy(qk_ub, qk_gm)   # MTE2，不在 V 上跑
```

GM↔UB 的 mem_copy 走 MTE2 / MTE3 PIPE，**根本不在 V 上**。vf 调度的是 V PIPE 上的 op，它对 MTE op 没有控制。

意思是：这条 mem_copy 不会被 fused 进 vf，它在 vf 之外按 PIPE 调度跑。把它放在 vf 区域里只是"语法上能写"，没有性能收益。

### 3. scf.if 内的分支

```python
with vf(outputs=[...]):
    muls(qk_ub, qk_ub, scale)
    if has_atten_mask:        # ← scf.if，在 vf 内
        vselect(qk_ub, ...)
    exp(qk_ub, qk_ub)
```

`scf.if` 把 vf 区域劈成两段，分支汇合点 lowering 装不进同一 for-loop iteration。后果：vf 区域 lowering 失败（或部分折叠、不报错但不优化）。

修法：把分支提到 Python 层，写两个直线变体：

```python
def softmax_round_with_mask(qk_ub, ...):
    with vf(...):
        vselect(qk_ub, ...)
        muls(qk_ub, qk_ub, scale)
        exp(qk_ub, qk_ub)

def softmax_round_no_mask(qk_ub, ...):
    with vf(...):
        muls(qk_ub, qk_ub, scale)
        exp(qk_ub, qk_ub)

# Python 层判断
if has_atten_mask:
    softmax_round_with_mask(...)
else:
    softmax_round_no_mask(...)
```

FA variant D 的 4 个 `softmax_round_b_finalize_*` 就是这么拆的（first/more × has_second/no_second）。

### 4. 跨 vf 区域

```python
with vf(outputs=[qk_ub]):
    muls(qk_ub, qk_ub, scale)

# 这里 vf 已结束
with vf(outputs=[p_ub]):
    exp(qk_ub, qk_ub)      # 读上一个 vf 写完的 qk_ub
    cast(p_ub, qk_ub)
```

两个 vf 区域的 RaW 不再被 vf 保证——退回到 PIPE 调度。如果 lowering 把它们扔进同一个 V for-loop，可能 OK；但保险做法是要么把两段合到一个 vf 里、要么由 channel-first pass 处理跨 PIPE 同步。

## `outputs=[...]` 的精确语义

vf 区域里的 store 操作分两类：

- **被 outputs 列出的 buffer 的 store**：保留（vf 之外有人读，必须落到 UB）。
- **没被 outputs 列出的 buffer 的 store**：消掉（vf 当成区域内的临时量、走寄存器或 elide）。

漏列的代价：跨 vf 后读这个 buffer **读到的是 vf 之前的旧值**——因为 vf 内的更新被吃掉、没落到 UB 上。

没有编译错——只是数值乱。

排查规则：
1. 列出 vf 区域之外**任何会被读到的** UB buffer。
2. 包括 softmax 状态的 `m_a_partial_ub`、`sum_a_partial_ub` 这种小标量——它们经常是漏的。
3. 重复列没有代价，宁可多写。

FA 里 variant D 的 vf outputs 典型形态：

```python
with vf(outputs=[qk_ub_a, p_ub_a, m_a_partial_ub, sum_a_partial_ub]):
    if not has_atten_mask_already:
        vselect(qk_ub_a, ...)
    muls(qk_ub_a, qk_ub_a, scale)
    reduce_max(m_a_partial_ub, qk_ub_a)         # m_a_partial 写
    expand(tmp_tile_ub, m_a_partial_ub)
    sub(qk_ub_a, qk_ub_a, tmp_tile_ub)
    exp(qk_ub_a, qk_ub_a)
    muls(qk_ub_a, qk_ub_a, scale_p)
    reduce_sum(sum_a_partial_ub, qk_ub_a)       # sum_a_partial 写
    cast(tmp_tile_ub_fp8, qk_ub_a)              # 写不到 outputs—被吃，但 tmp_fp8 实际折不掉（见上）
    mem_copy(p_ub_a, tmp_tile_ub_fp8, engine=nd2nz)  # MTE3，不在 V 上调度
```

`tmp_tile_ub_fp8` **没列**在 outputs——但因为下面的 mem_copy 是 MTE3 不在 vf 调度范围内，cast 的 store 必须保留让 mem_copy 能读，所以**实际上**这个 store 不会被消（与一般规则反过来）。这是 cast→nd2nz 边界的副作用。

## 验证 vf 折叠效果

折叠效果无法从源码直接读出，只能靠 msprof 的实测反推：vec 侧耗时与 UB 占用若显著高于纸面「全折」估算，说明有 op 没折进来，要回去查 outputs 列表与 align 匹配。采集方法见 `../../../debug-skills/cannbotdsl-msprof-compare/SKILL.md`。

经验：FA variant D 的一个 round 期望 vf 融合成 1 段（理想）或 2 段（cast→nd2nz 边界分了）。

## 折叠失败的总流程

```
vf 区域写好了 → 编译没报错 → 但性能上不去？

1. 用 msprof 确认瓶颈确实在 vec 侧
2. 逐条核对下面 4 种常见成因
3. 如果是 cast→mem_copy 边界 → 接受现实，按"不折"算预算
4. 如果是 scf.if → 拆成多条直线函数
5. 如果是 outputs 漏列 → 补
6. 如果跨 vf → 把两段合并到同一 vf 里
```
