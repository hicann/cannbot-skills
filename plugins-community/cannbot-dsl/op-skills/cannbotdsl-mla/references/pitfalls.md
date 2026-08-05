# MLA 已知陷阱

按「排查代价」排序。前两条是本算子最贵的，都是**编译干净、静默算错**。

---

## §1 QK 与 PV 的 L0 操作数形状只在 `d_chunk == tile_n` 时偶然重合 ⚠️

**代价最高的一条**，排查耗时超过其他所有 bug 之和。

**现象**：`d_nope=448` 静默算错（MERE 8e-2），而 512/384/256 全对。四个隔离探针
（QK 尾块、零填充语义、PV+UB 存储、strided 视图）**全部精确通过**，kernel 照错不误。

**根因**：两个 matmul 的 L0 操作数形状本来就不同——

```
QK:  S[M,tile_n]  = Q[M,d_chunk] @ K^T[d_chunk,tile_n]
     L0A=(M, d_chunk)    L0B=(d_chunk, tile_n)

PV:  O[M,d_chunk] = P[M,tile_n]  @ V [tile_n,d_chunk]
     L0A=(M, tile_n)     L0B=(tile_n, d_chunk)
```

`d_chunk == tile_n == 128` 时两者**恰好重合**，共用一对 channel 毫无问题；
`d_chunk=64`（448 的除数切分）时形状不同，复用就喂给 mmad 一个错形状的操作数。
每个隔离探针单独测时用的都是同一种形状，测不出来。

**修法**：显式判断，不要默认能复用。

```python
if d_chunk == tile_n:
    self.l0a_pv, self.l0b_pv = self.l0a, self.l0b
else:
    self.l0a_pv = Channel(MemLoc.L0A, shape=(tile_cube_m, tile_n), dtype=dt, depth=1)
    self.l0b_pv = Channel(MemLoc.L0B, shape=(tile_n, d_chunk), dtype=dt, depth=1)
```

**方法论教训**：隔离探针全绿而整体仍错时，**立刻转向找组件间的共享假设**，
不要再往单个组件里挖。详见
`../../../core-skills/cannbotdsl-probe-debug/SKILL.md` §3。

---

## §2 split-M 的空分区 AIV 污染共享 softmax 状态

**现象**：query 长度 S ≤ `tile_vec_m` 时**所有行**都错，误差看起来**均匀**。
边界锐利：tile_vec_m=16 时 S=16 全错、S=17 全对。

**根因**：cube 的 M tile 由 2 个 AIV 分摊，S ≤ tile_vec_m 时一个 AIV 分不到行。
但它**不是什么都不做**——照样在 padding 上算 rowmax/rowsum 并写入**共享的**
running max/sum，把另一个 AIV 的行也带偏。这解释了为什么误差是「均匀」而非「一半对一半错」
（曾据此误判为 softmax 权重算错，走了弯路——见 probe-debug SKILL.md §5 的形态表）。

**同一个 bug 有两个触发面**：

| 触发面 | 条件 | 症状 |
|---|---|---|
| 整个张量太短 | `S ≤ tile_vec_m` | 全部行错 |
| **尾 m-tile 太短** | `S % tile_cube_m ∈ (0, tile_vec_m]` | 只错最后几行，如 S=65 错行=[64] |

第二个更隐蔽：`_pick_tile_m` 只看全局 S，看不到尾 tile。实测 S=65（尾 tile 1 行）、
S=112（尾 48 行）都错，而 S=127（尾 63 行）对。

**修法（两段）**：

```python
def _pick_tile_m(S):
    """缩小 M tile 直到两个 AIV 都有行，下限 16 = NZ fractal"""
    if S >= TILE_CUBE_M:
        return TILE_CUBE_M, TILE_VEC_M
    vec_m = TILE_VEC_M
    while vec_m > 16 and S <= vec_m:
        vec_m //= 2
    return vec_m * 2, vec_m

def _plan_m(S):
    """再把 S 补齐到 tile 整数倍，消灭尾 tile"""
    cube_m, vec_m = _pick_tile_m(S)
    padded = max(S_MIN, (S + cube_m - 1) // cube_m * cube_m)
    cube_m, vec_m = _pick_tile_m(padded)     # 变长后可能配得上更大的 tile
    return cube_m, vec_m, max(S_MIN, (S + cube_m - 1) // cube_m * cube_m)
```

`S_MIN = 17`（必须 > 最小的 tile_vec_m=16）。S ≤ 16 时在 host 侧补齐、算完切掉。

---

## §3 补 query 行必须**前置**，causal 下追加会系统性算错

**现象**：短序列修好后，非 causal 的 S=1/2 全绿，但 causal 的 MTP case 仍错。

**根因**：causal mask **右下角对齐**（`j <= i + (S_kv - S)`）。**追加** padding 时真实行
位置不变而 S 变大，可见上界从 `i + (S_kv - S)` 缩到 `i + (S_kv - S - n)`——
**每个真实行都少看 n 个 key**。

**前置**则真实行 i 移到 i+n，`(i+n) + (S_kv - (S+n)) == i + (S_kv - S)`，可见范围恒定。
输出侧相应取**尾部**：

```python
def _pad_seq(t, axis, n):
    shape = list(t.shape); shape[axis] = n
    return torch.cat([torch.zeros(shape, dtype=t.dtype, device=t.device), t], dim=axis)

# ... 算完之后
out = out[:, pad_rows:] if layout == "BSND" else out[:, :, pad_rows:]
```

**验证教训**：所有针对短序列的定向验证都跑了 `is_causal=False`，全绿——
这个 bug 躲过了全部定向验证，直到完整 sweep 才暴露。
**验证 padding 类修复必须带 causal 变体。**

---

## §4 L0C 累加器必须跟随实际 M

**现象**：decode（S=1）编译期报
`'cannir.matmul' op M dimension mismatch: dst[0]=64 vs lhs[0]=1`。

**根因**：GM 的 Q 视图是 tail-aware 的，M 尾块时 L0A 行数 < tile_cube_m，
而 L0C 缓冲区仍是声明的满尺寸。

**修法**：`compute_qk` / `compute_pv` 里把累加器切到实际 M：

```python
m_rows = qn_slice.shape[0]
acc = local_slice(self.qk_l0c, (m_rows, self.tile_n))   # ← 对齐 M
```

PV 的 band 同理用 `local_slice(self.pv_l0c, (m_rows, d_chunk), offset=...)`。

---

## §5 归约轴的零填充列污染 softmax 分母（已知限制）

**现象**：`S_kv % tile_n != 0` 时精度不达标。误差随**被 padding 的列数**增长，
不是随尾块大小：

| S_kv | 尾块列数 | 被 pad 列数 | 结果 |
|---|---|---|---|
| 255 | 127 | **1** | PASS |
| 192 | 64 | 64 | FAIL (MERE 2.3e-1) |
| 130 | 2 | 126 | FAIL |

**根因**：K tile 缺失列被 `nd2nz` 零填充 → score=0 → softmax 里
`exp(0 - rowmax)` 是**非零权重**，计进 rowsum 分母。V 对应行是 0，分子不受影响，
但分母被撑大，输出整体偏小。

**causal 只减轻不消除**（曾误以为能消除）：mask 跟的是因果对角线，不是 S_kv 边界，
越过 S_kv 的 padding 列在对角线以下时仍可见。实测 causal 下 S_kv=192 仍 MERE 1.63e-1。

**状态**：未修。修法要动 raw-VF 的 rowmax/rowsum 越界 lane 处理（无运行期分支地把越界
lane 压成 -inf），属最精密且已验证的一段。动前**先写只测该逻辑的探针**。

---

## §6 rope 段必须用独立且尺寸精确的 channel

`d_rope=64 ≠ d_chunk`。复用 nope 的宽 channel 装 rope tile 会让 mmad 的 K 长度不明确。
给 rope 单独一套 `qr_l1 / kr_l1 / l0a_r / l0b_r`。

---

## §7 通用 API 陷阱（非 MLA 特有，但 MLA 全踩过）

这些在
`../../../core-skills/cannbotdsl-probe-debug/references/silent-failure-catalog.md`
有完整说明，此处只列触发点：

| 陷阱 | MLA 里的触发点 |
|---|---|
| `tile_view` 的 coord 是 **tile 索引** | chunk 循环写 `(0, c)` 而非 `(0, c*CHUNK)` |
| PV 的 L1→L0B 需 `transpose=True`，QK 不需要 | 两个 chunk 循环 |
| N 个 mmad 写同一 L0C 区域编译期报错 | QK 5 个 chunk、PV 4 个 band 都要 `tuple(range(n))` 静态展开 + `local_slice` 切子区域 |
| `const_expr(range(n))` 不是循环包装器 | 静态展开写 `tuple(range(n))` |
| `@jit` 参数不能是 Python `bool` | 编译期开关放 `__init__` 属性 |
| 同 dtype golden 当精度基准会误判 | 验收必须用官方 checker |
| `const_expr(cond)` 守卫变负失效 | `NPAD = VH - BMV` 变负时 `if const_expr(NPAD > 0):` 静默跳过，加 `assert VH >= BMV` |
| 显式 `addr=` 别名 channel 无同步保护 | depth≥2 重叠即静默串数据，先确认不别名装不装得下 |
| benchmark 覆盖差集 | 规划器接受的域 − 用例集覆盖的域 = 边界测试目标，MLA 的 `G > 32` 就在差集里 |
