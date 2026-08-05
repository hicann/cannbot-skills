# FA 4-stage 软件流水实战指南

> 本文记录 FA pipeline 优化的关键技术，从 depth=1 串行（318.8 us）到
> 4-stage pipeline N=3（51.5 us, 6.2x）的完整优化路径。所有技术均经真机
> 验证（Ascend NPU, CANN 9.1.0, cannbotdsl dev260）。

## 1. 为什么需要 4-stage pipeline

FA 的一个 macro-iteration（一个 n-tile）包含 4 个串行阶段：

```
QK(n) → softmax(n) → PV(n) → update_o(n)
Cube    Vec          Cube    Vec
```

串行执行时 Cube 和 Vec 交替空等：

| 阶段 | 执行核 | 耗时（us） | 另一核状态 |
|------|--------|-----------|-----------|
| QK | Cube | ~11 | Vec 空闲 |
| softmax | Vec | ~22 | Cube 空闲 |
| PV | Cube | ~11 | Vec 空闲 |
| update_o | Vec | ~5 | Cube 空闲 |

4-stage pipeline 让每个 tick 推进所有 stage 各一步，不同 stage 处理不同 n 的数据：

```
tick:    0     1     2     3     4     5     6     7
─────────────────────────────────────────────────────────
QK:     QK0   QK1   QK2   QK3   QK4   QK5   QK6   QK7
SM:            SM0   SM1   SM2   SM3   SM4   SM5   SM6
PV:                  PV0   PV1   PV2   PV3   PV4   PV5
UPD:                       UPD0  UPD1  UPD2  UPD3  UPD4
```

稳态时 Cube 同时做 QK(n) + PV(n-2)，Vec 同时做 softmax(n-1) + update(n-3)。

**实测效果**：

| 指标 | 串行 (depth=1) | Pipeline N=3 | 提升 |
|------|---------------|-------------|------|
| Task Duration | 318.8 us | 51.5 us | **6.2x** |
| mac_ratio | 0.288 | 0.818 | 29% → 82% |
| vec_ratio | 0.555 | 0.790 | 56% → 79% |

## 2. 核心架构：@kernel class 三层拆分

FA pipeline 采用三层 class 结构，这不是代码风格，而是**编译器正确性的必要条件**：

```
flash_attn_kernel (@kernel class)
├── Matmul class     — Cube 侧所有 op
├── Vector class     — Vec 侧所有 op
└── __call__         — 顶层编排（DelayLineGroup + stage-gate）
```

### 为什么不能用 @kernel def

`@kernel def` 中所有代码被 trace 到一个 IR function 中。当 stage-gate 的
`if tick >= 1 and ...` 编译成 `scf.if` 时，`scf.if` 分支内的 CrossCore channel
`wait/release` 会被 框架复制到所有分支路径，
导致 ring buffer 游标错位 → L1 越界崩溃。

`@kernel class` 的 `@jit` 方法是独立的 IR function，channel 操作在方法内部
（不在 `__call__` 的 `scf.if` 分支中），框架能正确分析每个 stage 的 channel
使用模式。

### Matmul class 关键设计

```python
class Matmul:
    def __init__(self, tile_cube_m, tile_n, tile_d, dtype_16):
        # 分离 K/V 的 L1 channel（不能共享 — pipeline 中同时使用）
        self.q_l1 = Channel(MemLoc.L1, (tile_cube_m, tile_d), dtype_16, depth=2)
        self.k_l1 = Channel(MemLoc.L1, (tile_n, tile_d), dtype_16, depth=2)
        self.v_l1 = Channel(MemLoc.L1, (tile_n, tile_d), dtype_16, depth=2)
        # L0 全部 depth=2 double buffer
        self.l0a = Channel(MemLoc.L0A, (tile_cube_m, tmp_n), dtype_16, depth=2)
        self.l0b = Channel(MemLoc.L0B, (tile_d, tile_n), dtype_16, depth=2)
        self.l0c = Channel(MemLoc.L0C, (tile_cube_m, tmp_n), dtypes.float32, depth=2)
```

关键点：
- **K 和 V 用独立 channel**（`k_l1`/`v_l1`），不能共享一个 `b_l1`。pipeline 中
  QK stage 加载 K(n) 的同时 PV stage 加载 V(n-2)，共享 channel 会 slot 冲突。
- **L0A/L0B 也 depth=2**：QK 写 L0A slot 0 时 PV 读 L0A slot 1。
- **tmp_n = max(tile_n, tile_d)**：QK 时 L0A 是 (M, K)，PV 时 L0A 是 (M, N)，
  取最大值避免 shape 不匹配。

### Vector class 关键设计

```python
class Vector:
    def __init__(self, ...):
        # Triple-buffer softmax state — 槽与生产/消费同步由 Channel 管理
        self.sm_max_tb = Channel(MemLoc.UB, (tile_vec_m, 1), dtypes.float32,
                                 depth=preload_num)
        self.sm_sum_tb = Channel(MemLoc.UB, (tile_vec_m, 1), dtypes.float32,
                                 depth=preload_num)
        self.sm_exp_tb = Channel(MemLoc.UB, (tile_vec_m, 1), dtypes.float32,
                                 depth=preload_num)
        # 临时量（不跨 iteration 持久）
        self.tmp_new_max = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.tmp_sum = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        # O 累加器（跨 n-tile 持久，按 m-tile 重置）
        self.res_o = Buffer(MemLoc.UB, (tile_vec_m, tile_d), dtypes.float32)
        # P buffer — depth=2 让 softmax 写 P(n) 时 PV 读 P(n-2)
        self.p_ub = Channel(MemLoc.UB, (tile_vec_m, tile_n), dtype_16, depth=2,
                            data_format="nz", n1_pad=p_n1_pad)
```

关键点：
- **softmax state stride=1**：`Channel(MemLoc.UB, (tile_vec_m, 1), dtypes.float32, depth=preload_num)` 让 64 行的
  max/sum/exp 连续存储（256 bytes），`vload_brc` 能正确广播。
- **sm_max/sm_sum 与 sm_exp 的消费顺序必须是 Channel FIFO 可表达的顺序**。旧实现若要求
  `m_seq % preload_num` 与 `tick % preload_num` 两套手工随机槽索引，且无法改写为 FIFO 生产/消费，
  该深流水当前不受公开前端支持；不得用 Buffer 数组伪装成带同步的多槽 storage。
- **p_ub depth=2**：softmax Pass-B 写 P(n) 到 p_ub slot 0，`store_p` 把 p_ub
  拷到 p_l1 slot 0；同时 PV 读 p_l1 slot 1（来自 P(n-2)）。

## 3. DelayLineGroup + stage-gate 调度

### DelayLineGroup

```python
delay_slots = preload_num + 1  # = 4
dl = DelayLineGroup(delay_slots, 'tile', 'n', 'm_seq')
```

DelayLineGroup 是移位寄存器，`push` 写当前 tick 的参数，`tap(K)` 取 K 拍前的参数。
`advance()` 所有 slot 向前移 1 格。

### Stage-gate 条件

```python
for n_idx in range(n_end):
    dl.push(tile=tile_idx, n=n_idx, m_seq=m_seq)
    self._stage_qk(tile_idx, n_idx, ...)      # Stage 0: always fires
    issued = issued + 1

    if tick >= 1 and tick - 1 < issued:       # Stage 1: softmax(n-1)
        self._stage_softmax(tick, dl.tile.tap(1), dl.n.tap(1), dl.m_seq.tap(1))
    if tick >= 2 and tick - 2 < issued:       # Stage 2: PV(n-2)
        self._stage_pv(dl.tile.tap(2), dl.n.tap(2))
    if tick >= 3 and tick - 3 < issued:       # Stage 3: update(n-3)
        self._stage_update(tick, dl.tile.tap(3), dl.n.tap(3), dl.m_seq.tap(3))

    dl.advance()
    tick += 1
```

两个条件的作用：
- `tick >= K`：warmup 屏障 — 前 K 个 tick 不发射 stage K
- `tick - K < issued`：drain 屏障 — issued 停增长后逐步停止

### Drain 循环

```python
for _ in range(preload_num):  # = 3
    if tick >= 1 and tick - 1 < issued: self._stage_softmax(...)
    if tick >= 2 and tick - 2 < issued: self._stage_pv(...)
    if tick >= 3 and tick - 3 < issued: self._stage_update(...)
    dl.advance()
    tick += 1
```

drain 只发剩余 stage，不发 QK。一个循环体 + 两个条件 = 覆盖 warmup/steady/drain。

## 4. CrossCore channel depth 设计

| Channel | depth | kind | 依赖距离 | 原因 |
|---------|-------|------|---------|------|
| qk_ub | 2 | CrossCore | QK→SM = 1 | Cube 写 slot[t%2]，Vec 读 slot[(t-1)%2] |
| p_l1 | **3** | CrossCore | SM→PV = 2 | Vec 写 slot[t%3]，Cube 读 slot[(t-2)%3] |
| pv_ub | 2 | CrossCore | PV→UPD = 1 | Cube 写 slot[t%2]，Vec 读 slot[(t-1)%2] |

**p_l1 depth=3 的原因**：softmax 写 P(n) 到 p_l1，2 个 tick 后 PV 才读 P(n)。
depth=2 时 produce 堆积 2 个 slot 后才 consume，ring buffer 游标可能错位。
depth=3 提供 3 个 slot 缓冲，匹配 `scf.if` 分支展开后的 4 次 release（3 个
分支 × 1 次执行 + 1 次 drain）。

**CrossCore depth ≤ 8**（框架限制，见 `../cannbotdsl-channel/SKILL.md`）。
Σ depth = 2+3+2 = 7 ≤ 8 ✓。

## 5. 两遍 softmax + vmem_bar

softmax 分两遍，中间用 `vmem_bar("vst_vld")` 屏障：

```python
@jit
def softmax_first(self, qk_ch, scale, m_axis_triple):
    with vf(mode="raw"):
        # Pass A: scale + store in-place + rowmax → sm_max
        for row in range(rows):
            v0 = vmuls(vload(qk_ch, base), scale, mask=half0_mask)
            v1 = vmuls(vload(qk_ch, base + VL_T), scale, mask=half1_mask)
            vstore(qk_ch, base, v0, half0_mask)    # 写回 qk_ch
            vstore(qk_ch, base + VL_T, v1, half1_mask)
            rmax = vreduce_max(vmax(v0, v1, mask=full), mask=full)
            vstore_first(sm_max, row, rmax)
        vmem_bar("vst_vld")    # ← 屏障：等 Pass A 的 store 完成
        # Pass B: sub(max) + exp + cast fp16 NZ + reduce_sum → sm_sum
        for row in range(rows):
            mx = vload_brc(sm_max, row)
            ve, vo = vload_deinterleave(qk_ch, base, width="b32")
            ...
```

### 为什么需要两遍

- Pass A 用 `vload`（非 deinterleave）读 qk_ch，scale 后 `vstore` 写回。
  这修改了 qk_ch 的内容。
- Pass B 用 `vload_deinterleave` 读修改后的 qk_ch，做 exp + cast。
- 如果没有 `vmem_bar`，Pass B 可能读到 Pass A 还没写完的数据。

### 为什么 Pass A 用 vload 而非 vload_deinterleave

`vload` 读 64 个连续 float，`vstore` 写回 64 个连续 float — 数据排列不变。
`vload_deinterleave` 读 128 个 float 拆成 even/odd 各 64 — 如果 Pass A 用
deinterleave + store，数据排列会被重排，Pass B 的 deinterleave 会读到错误数据。

## 6. Fused O update（vmadd）

用 `vmadd` 融合 rescale + add：

```python
# 中间 tile: O = O * alpha + PV
o = vmadd(pre, exp_b, cur, mask=mask)    # fused: pre * exp_b + cur

# 末次 tile: O = (O * alpha + PV) / l  （融合除法）
o = vmadd(pre, exp_b, cur, mask=mask)
o = vdiv(o, safe_sum, mask=mask)
```

对比分离写法（`vmul` + `vadd`），`vmadd` 消掉一次 UB 读写：
- 分离：`load O → mul alpha → store tmp → load tmp → add PV → store O`（6 次 UB 访问）
- 融合：`load O → load PV → vmadd → store O`（4 次 UB 访问）

末次 tile 还融合了除法，消掉独立的 `finalize` pass。

## 7. store_p 的 channel-first 4 相协议

`store_p` 在 Vec 侧把 p_ub（UB）拷到 p_l1（L1）—— 这是 Vec→Cube 的跨核 handoff：

```python
def store_p(self, p_l1_ch, partition):
    piece = partition_view(p_l1_ch, partition, self.subblock_idx)
    mem_copy(piece, self.p_ub)
```

`partition_view` 把 L1 CrossCore channel 切成 per-AIV 的 M-tile 半块。
`mem_copy(piece, self.p_ub)` 是 UB→L1（MTE3 pipe），channel-first 下框架自动
合成 acquire/commit（Vec 侧 produce）和 wait/release（Cube 侧 consume）。

关键：`store_p` 不是 `@jit` 方法 — 它是普通方法，被 `_stage_softmax`（`@jit`）
内联调用。这样 channel 操作在 `_stage_softmax` 的 IR function 内，不在
`__call__` 的 `scf.if` 分支中。

## 8. finalize_o 的手动 4 相协议

`finalize_o` 用手动 acquire/commit/wait/release 来支持 M-tail：

```python
def finalize_o(self, o_tile_gm, sm_sum_buf, div_done=False):
    # ... div if needed ...
    o_slot = self.o_ub.acquire()
    o_full = local_slice(o_slot, (tile_vec_m, tile_d), stride=(tile_d, 1))
    cast(o_full, self.res_o)        # fp32 → fp16
    self.o_ub.commit(o_slot)

    o_slot_r = self.o_ub.wait()
    o_view_r = local_slice(o_slot_r, (actual_vec_m, tile_d), stride=(tile_d, 1))
    mem_copy(half, o_view_r)        # UB → GM
    self.o_ub.release(o_slot_r)
```

手动 4 相的原因：`local_slice` 需要根据 `actual_vec_m`（M-tail 的实际行数）
动态切视图，channel-first 的自动合成不支持这种动态 shape。

## 9. local_slice 的使用规则

`local_slice` 创建 channel slot 的零拷贝视图，用于：
1. **qk_ub → softmax**：`local_slice(self.qk_ub, (actual_vec_m, actual_n), stride=(tile_n, 1))`
   — 让 softmax 读到正确的 slot 和 shape
2. **pv_ub → update_o**：同上
3. **p_l1 → compute_pv**：`local_slice(self.p_l1, (tile_cube_m, actual_n))`
   — 让 PV matmul 读到正确的 P slot
4. **o_ub → finalize_o**：`local_slice(o_slot, ...)` — 动态 M-tail 支持

规则：
- `local_slice` 必须在 `@jit` 方法内调用（不在 `__call__` 的 `scf.if` 中）
- stride 参数必须匹配 channel 的 physical stride
- `actual_vec_m` / `actual_n` 从 `tile_view` 或 `partition_view` 的 `.shape` 获取

## 10. 已知限制与调试经验

### scf.if 内的 channel 操作问题

`__call__` 中的 `if/elif/else` 选择不同 `@jit` stage 方法时，编译器为每个分支
生成 `scf.if`。如果 `@jit` 方法内部有 CrossCore channel 的 `wait/release`，
框架可能在所有分支路径上复制 channel 操作，
导致 ring buffer 游标错位。

**规避**：把 channel 操作放在 `@jit` 方法内部，`__call__` 中的 `if/elif/else`
只选择调用哪个 `@jit` 方法。`_stage_update` 内部有 `if/elif/else` 分支
（first/mid/last），但这些分支直接调用 `self.vector.init_o`/`update_o`/
`update_o_last` — channel 操作在这些 `@jit` 子方法内部。

### 2D flattened vs 4D BNSD 输入

2D flattened `[B*H*S, D]` 的 `tile_view` stride 推断与 4D BNSD `[B, H, S, D]`
不同，可能影响 `asc_copy_gm2l1_nd2nz` 的参数。**建议用 4D BNSD**。

### NPU 设备状态污染

kernel 崩溃后 NPU 设备状态可能被污染，导致后续所有 kernel 都崩溃（即使代码正确）。
`torch.npu.empty_cache()` 不足以恢复，需要换环境或重启设备。

**调试建议**：每次崩溃后换一个干净环境重测，或用 `pytest` 的进程隔离。

## 11. 性能数据汇总

| 版本 | Task Duration | mac_ratio | vec_ratio | 说明 |
|------|--------------|-----------|-----------|------|
| depth=1 串行 | 318.8 us | 0.288 | 0.555 | 原始 baseline |
| depth=2 + 循环重排 | 271.1 us | 0.313 | 0.595 | QK/softmax 同 tick |
| **4-stage pipeline N=3** | **51.5 us** | **0.818** | **0.790** | DelayLineGroup + stage-gate |

6.2x 加速来自：
- Cube/Vec 重叠（mac 0.288 → 0.818）
- 搬运/计算重叠（mte2 0.305 → 0.767）
- Fixpipe 重叠（fix 0.155 → 0.752）
- Fused O update（减少 UB 访问）
- 两遍 softmax + vmem_bar（Pass A/B 并行度更好）
