# 设计期 API 可行性检查 — 代码示例

R2.5 三张检查表对应的代码示例。所有示例自包含，不引用外部文件。

## 1. raw-VF 替代 expand（表 1）

`expand` op 在 VF 自动归组中无模板，任何位置使用都会编译失败。

**错误写法**（高层 expand）：

```python
from cannbotdsl.math import expand, sub, exp, reduce_max

# 编译失败：vf-transform found no template for operation 'cannir.expand'
reduce_max(m_ub, s_ub, axis=-1)
expand(exp_tmp, m_ub, axis=(1,))   # ← 失败
sub(s_ub, s_ub, exp_tmp)
exp(s_ub, s_ub)
```

**正确写法**（raw-VF `vload` + `vdup_lane0` 替代广播）：

```python
from cannbotdsl.raw_reg import (full_mask, vload, vstore, vdup_lane0,
                              vsub, vexp, vmuls, vreduce_max,
                              vreduce_sum, vmax, vadd, vmem_bar)
from cannbotdsl.vf import vf

@jit
def softmax_raw(qk_ch, m_ub, l_ub, p_ub):
    with vf(mode='raw'):
        m32 = full_mask(elem_bits=32)
        for row in range(64):
            b = row * 128
            s0 = vmuls(vload(qk_ch, b), 0.0884, mask=m32)
            s1 = vmuls(vload(qk_ch, b + 64), 0.0884, mask=m32)
            # rowmax — 替代 reduce_max + expand 广播
            rmax = vmax(vreduce_max(s0, mask=m32),
                        vreduce_max(s1, mask=m32), mask=m32)
            rmb = vdup_lane0(rmax, mask=m32)   # ← 替代 expand 广播
            # P = exp(S - rmax)
            p0 = vexp(vsub(s0, rmb, mask=m32), mask=m32)
            p1 = vexp(vsub(s1, rmb, mask=m32), mask=m32)
            vstore(p_ub, b, p0, m32)
            vstore(p_ub, b + 64, p1, m32)
        vmem_bar('vst_vld')
```

## 2. DMA 分隔不同 shape vec op（表 1）

不同 shape 的 vec op 被 VF grouping 自动合并后报 shape mismatch。在它们之间插入 DMA 操作可打断 grouping。

**错误写法**（不同 shape muls 被合并）：

```python
from cannbotdsl.math import muls

# 编译失败：vector elementwise operands require matching logical shapes
# muls(o_ub, o_ub, 0.0)  — shape (64, 128)
# muls(m_ub, m_ub, 0.0)  — shape (64, 1)
# 两者被 VF grouping 合并到同一 region → shape mismatch
muls(o_ub, o_ub, 0.0)
muls(m_ub, m_ub, 0.0)
muls(l_ub, l_ub, 0.0)
```

**正确写法**（用 DMA 分隔）：

```python
from cannbotdsl.math import muls
from cannbotdsl import dtypes
from cannbotdsl.tensor import mem_copy, make_copy_engine

nd2nz = make_copy_engine(format_transform="nd2nz", dtype=dtypes.float16, pad_value=0.0)

muls(o_ub, o_ub, 0.0)
# DMA 操作打断 VF grouping — 不同 shape 的 muls 不再被合并
mem_copy(a_l1, q_tile, engine=nd2nz)   # ← DMA 分隔
muls(m_ub, m_ub, 0.0)
muls(l_ub, l_ub, 0.0)
```

## 3. Channel 与 Buffer 的同步边界（表 2）

`Buffer` 只是无同步语义的单块 scratch。一条数据流如果跨 PIPE、跨核或需要 depth-N 缓冲区生命期，它的生产者/消费者边界必须由 Channel 表达，不能指望 Buffer 隐式生成 sync。

### 3.1 Buffer 计算结果 → GM

```python
# Buffer → Channel → GM：Channel 承担跨 PIPE 生命期。
muls(out_ub, o_ub, 1.0)              # o_ub 是 Buffer，out_ub 是 Channel
o_tile = tile_view(o_gm, (64, 128), (0, 0))
mem_copy(o_tile, out_ub)
```

### 3.2 跨核 Channel → Buffer

```python
# 让消费 op 直接读 Channel，并把结果写入 Buffer。
muls(s_ub, qk_ub, SCALE)             # qk_ub: 跨核 Channel; s_ub: Buffer
```

### 3.3 显式 `addr=` 别名 channel 无同步保护

两个 Channel 用 `addr=` 显式别名时，**地址重叠对同步层完全不可见**。单迭代内靠数据依赖可能侥幸正确；一旦别名操作数或其相邻操作数 `depth≥2` 让相邻迭代重叠，立刻静默串数据（编译无告警、运行无 fault）。**别名操作数自己 `depth=1` 不构成保护。** 决策顺序与实测见 `../SKILL.md` §2.0。

### 3.4 `const_expr(cond)` 守卫变负失效

形如 `if const_expr(NPAD > 0):` 的保护，当 `NPAD = VH - BMV` 因上游参数变化而变负时**静默跳过**，被保护的越界写读照常发生。API 可行性检查时，所有 `const_expr` 守卫须确认 cond 中的变量有下界保证（`assert VH >= BMV`）或参数在 host 侧推导。详见 `../../cannbotdsl-vf-fusion/SKILL.md` 陷阱 11。

## 4. Buffer 生命周期（表 2 补充）

Buffer 的声明位置决定其生命周期。

**错误写法**（running state 声明在 @jit 方法内 — 每次调用新建，不持久）：

```python
@jit
def softmax_step(qk_ch, p_ub):
    m_ub = Buffer(MemLoc.UB, (64, 8), dtypes.float32)   # ← 每次调用新建
    l_ub = Buffer(MemLoc.UB, (64, 8), dtypes.float32)   # ← running state 跨迭代不持久
    with vf(mode='raw'):
        ...
```

**正确写法**（running state 声明在 @kernel body，作为参数传给 @jit）：

```python
@kernel
def fa_kernel(q_gm, kt_gm, v_gm, o_gm):
    # Buffer 在 @kernel body 声明 — 跨循环迭代持久
    o_ub     = Buffer(MemLoc.UB, (64, 128), dtypes.float32)
    m_ub     = Buffer(MemLoc.UB, (64, 8), dtypes.float32)
    l_ub     = Buffer(MemLoc.UB, (64, 8), dtypes.float32)
    alpha_ub = Buffer(MemLoc.UB, (64, 8), dtypes.float32)

    for tidx in range(bid, total, bnum):
        muls(o_ub, o_ub, 0.0)

        for n_iter in range(1, num_n):
            # Buffer 作为参数传给 @jit raw-VF 函数
            softmax_step(qk_ub, p_nz_ub, m_ub, l_ub, alpha_ub)
            rescale_o(o_ub, alpha_ub)
            ...

@jit
def softmax_step(qk_ch, p_ub, m_ub, l_ub, alpha_ub):
    # m_ub/l_ub/alpha_ub 是外部传入的 Buffer — 同一物理 storage
    with vf(mode='raw'):
        ...
```

## 5. 4D tile_view 规避（表 3）

4D/3D `tile_view` 视图传播到 `matmul` 时触发 shape inference 失败。

**错误写法**（4D tile_view）：

```python
# 编译失败：typed-region writer has no registered access-shape rule
q_tile = tile_view(q_gm, (1, 1, 128, 128), (bi, hi, mi, 0))
mem_copy(a_l1, q_tile, engine=nd2nz)
mem_copy(l0a, a_l1)
matmul(l0c, l0a, l0b)               # ← 4D 视图传播到 matmul 失败
```

**正确写法**（host 侧 4D→2D 展平）：

```python
# Host 侧：4D → 2D 展平
q_flat = q.reshape(B * H_q * S, D)           # [B,H_q,S,D] → [B*H_q*S, D]
kt_flat = kt.reshape(B * H_kv * D, S)        # [B,H_kv,D,S] → [B*H_kv*D, S]
v_flat = v.reshape(B * H_kv * S, D)          # [B,H_kv,S,D] → [B*H_kv*S, D]
o_flat = o.reshape(B * H_q * S, D)

# Kernel 内：只用 2D tile_view
q_tile = tile_view(q_gm, (128, 128), (tidx, 0))
mem_copy(a_l1, q_tile, engine=nd2nz)
mem_copy(l0a, a_l1)
matmul(l0c, l0a, l0b)               # ← 2D 视图正常
```

### idx2crd 分解避免 runtime 除法

**错误写法**（runtime `//` 不被支持）：

```python
bi, hi, mi = idx2crd(tidx, [B, H_q, num_m])
ki = hi // g                        # ← SSA 整数除法不被支持
```

**正确写法**（分解为 `[H_kv, g, num_m]`）：

```python
# 分解为 [H_kv_total, g, num_m] — ki 直接从 idx2crd 获得，无需除法
ki, gi, mi = idx2crd(tidx, [H_kv_total, g, num_m])
hi = ki * g + gi                    # query head index = kv_head * g + group_idx
```

## 6. pre-loop 建立 channel producer（表 1）

Channel in-place 操作（如 `muls(ch, ch, scalar)`）在 channel 无 producer 时报错。

**错误写法**（循环内直接 in-place，首迭代无 producer）：

```python
for n_idx in range(num_n):
    muls(s_ub, qk_ub, SCALE)
    # ...
    if n_idx > 0:
        muls(o_ub, o_ub, 0.0)       # ← 首迭代 o_ub 无 producer（如果是 channel）
```

**正确写法**（pre-loop 首迭代在循环外建立 producer）：

```python
# Pre-loop: 首迭代（n_idx=0）在循环外
n_idx = 0
mem_copy(qk_ub, l0c, engine=fixpipe)
muls(s_ub, qk_ub, SCALE)
# ... softmax init: m=rowmax, l=rowsum, O=PV
muls(pv_ub, l0c, engine=fixpipe)
muls(pv_tmp, pv_ub, 1.0)            # ← 建立 pv_tmp 的 producer
add_pv(pv_tmp, o_ub)                # ← o_ub 首次写入

# Main loop: 剩余迭代
for n_iter in range(1, num_n):
    muls(s_ub, qk_ub, SCALE)
    # ... softmax step: rescale m/l/O
    muls(pv_tmp, pv_ub, 1.0)        # ← 后续迭代已有 producer
    add_pv(pv_tmp, o_ub)
```
