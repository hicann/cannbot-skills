# FA kernel 的 buffer 预算

## 硬件上限（Ascend dav-3510）

| Region | 上限 | 在 FA 里用来装什么 |
| --- | --- | --- |
| UB    | 256 KB | vec 侧所有中间量：softmax max/sum/exp、P cast 暂存、atten_mask、res_o、pv_ub |
| L1    | 512 KB | Q/K/V tile、P tile、mxfp8 scale tile |
| L0A   | ~64 KB | Q tile（PV 阶段也是 P tile——mxfp8 时与 QK 共享 offset 0）|
| L0B   | ~64 KB | K tile（PV 阶段也是 V tile——同上）|
| L0C   | ~256 KB | QK 的 fp32 累加器（PV 写入 (M, D) 子区域）|
| L0A_MX / L0B_MX | 小 | mxfp8 的 e8m0 scale |

任何 region 超上限的表现都是**runtime NPU error 507015**（编译干净）。最常踩的是 UB。

## 预算 tally 模板

写 kernel 前，在文件 docstring 里填这张表：

```
| Buffer            | shape           | dtype     | slots | bytes |
| ---               | ---             | ---       | ---   | ---   |
| qk_ub             | (M, N_qk)       | fp32      | 2     | 64K   |
| p_ub              | (M+1, N_qk)     | fp8       | 2     | 32.5K |
| pv_ub             | (M, D)          | fp32      | 1     | 32K   |
| atten_mask_ub     | (M, N_qk)       | fp32      | 1     | 16K   |
| res_o + cast_fp16 | (M, D)          | fp32/fp16 | 1     | 32K   |
| tmp_*             | (shared)        | —         | —     | 16K   |
| softmax_*_ub_tb   | (M, 1)          | fp32      | 3×3   | 2.25K |
| TOTAL                                                  | ≈ 195K |
```

TOTAL 超上限时，按顺序拉以下手段：

## 减 UB 的手段

1. **`tile_n_qk`（或 `tile_k_pv_chunk`）减半**：所有沿 K 轴的 buffer 大小减半。影响 `qk_ub`（−N_qk×M×4）、`p_ub`（−N_qk×M）、`atten_mask_ub`（−N_qk×M×4）。只要 N 仍 ≥ cube 最小有效 N（~64），cube 利用率没有大塌方。

2. **`qk_ub` Channel 由 depth=2 降到 depth=1**（−`tile_n_qk × M × 4`）：失去 cube-QK / vec-P 重叠，但 FA 是 PV-bound 的，性能损失有限。QK_a 与 QK_b 都 drain 到同一个 slot，FIXPIPE ↔ V 的生命期锁由 Channel lowering 生成。**这是 507015 修复时用过的手段**（见 pitfall #2）。

3. **`p_ub` 双 buf 降单 buf**：失去 cast→store_p 在 round 之间的重叠。

4. **`res_o` 与 `cast_ub_fp16` 共享 region**：生命周期不重叠（res_o 在 accumulate 阶段、cast_ub_fp16 仅在 finalize），UB 同 offset alias 可省 32 KB。

5. **`tmp_*` 共享**：4 个逻辑 `tmp_tile_ub_*` 都 alias 到同一个 `(vec_m, tile_n_qk)` region。蓝本已经这样做。

## L1 预算（512 KB）

| Buffer | 典型 | 备注 |
| --- | --- | --- |
| q_l1 | (M, D) fp8，1 slot × 16 KB | Q 在整个 batch_head iter 内保留 |
| k_l1 | (N_qk, D) fp8，2 slot × 16 KB | DB 用于 cube prefetch |
| v_l1 | (K_pv_total, D) fp8，2 slot × 32 KB | DB |
| p_l1 | (M, K_pv_total) fp8，Channel depth=3 × 32 KB | macro preload 深度对齐的三槽 storage |
| sq_l1 | (M, ks_qk) e8m0，1 × 0.5 KB | mxfp8 only |
| sk_l1 | (ks_qk, N_qk) e8m0，2 × 1 KB | mxfp8 only |
| sv_l1 | (ks_pv, D) e8m0，2 × 1 KB | mxfp8 only |
| sp_l1 | (M, 2*ks_pv) e8m0，Channel depth=3 × 2 KB | mxfp8 + Channel preload（variant D）|

Channel depth=3 preload 参数下大约 250 KB，余量充裕。

## L0 预算

L0A 与 L0B 上限 ~64 KB。`(128, 128)` fp8 = 16 KB / slot，2 slot = 32 KB，舒服。

**关键**：mxfp8 的 L0A/L0B 数据通道需使 lowering 能推导隐式 L0A_MX/L0B_MX handle。旧的“手动让 PV/QK 的 MX buffer 共享 offset 0”方案需要已删除的 MX/NBuffer 前端 API，当前不可表达，不得伪造替代接口。可支持的 channel-first 方案应按物理 slot 容量核算 `max(QK_size, PV_size)` 或分别分配。

## L0C 预算

L0C 256 KB。一个 Channel slot `(M, N_qk) fp32` = 64 KB；depth=2 占 128 KB。PV 通过 `local_slice` 写 `(M, D)` 子视图。

## 已验证可行的 shape 组合

| 配置 | M | N_qk | D | K_pv_chunk | K_pv_total | UB 合计 |
| --- | --- | --- | --- | --- | --- | --- |
| Channel depth=3 preload | 128 | 128 | 128 | 128 | 256 | ~195 KB |
| gqa_mxfp8 baseline | 128 | 128 | 128 | —（单 PV）| — | ~150 KB |
| gqa | 128 | 128 | 128 | — | — | ~120 KB |

## 预算挂了的 debug 步骤

1. 按本页表格重新纸面核算每个 UB buffer 的 `depth × slot_bytes`，加总与 262144（= 256 KB）比较，找出溢出点。
2. 重点复核最近改过 depth 的 buffer——最常见的触发是把某个大 buffer 从单 buf 改 DB 之后没重算预算。
3. 复核 `vf` 区域内的 cast destination：它**不一定**被折叠掉（见 `vf-folding.md`），预算紧时按「不折」计。
4. 确认所有 UB 地址都来自 `Source.alloc_buffer_ub(...)` 机械计算，没有手写 offset。

有时你期望 `vf` 把某个 `tmp_*_fp8` cast destination 折叠掉，但实际没折（storealign / loadalign 不匹配）。这个 buffer 仍占 UB——即便你以为它不占。**预算先按"什么都不折"算，再去优化**。
