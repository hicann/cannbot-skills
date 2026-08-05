# FA kernel 已付学费的坑

这些都是在真实 FA 开发里踩过的 bug。绝大多数是**静默错误**——编译干净、运行结果错。**写代码前先读**，不要等调一天后才回来翻。

## 1. `asc_mmad_mx` fp8 在非零 L0 offset 时输出全 0

**症状**：PV matmul 写到 `pv_ub` 全是 bit-zero，但 QK 看起来正常。

**原因**：`asc_mmad_mx` 硬件特性——只要 L0A / L0B / L0A_MX / L0B_MX 中任何一个 buffer 的 offset 不是 0，fp8 mxfp8 mmad 就静默输出 0。普通的 fp16 `asc_mmad`（无 `_mx`）没这个问题。

**修法**：使用 channel-first matmul，让 lowering 从 L0A/L0B 数据通道推导隐式 MX handle：

```python
# Matmul.__init__ 里：
self.l0a = Channel(MemLoc.L0A, (tile_m, tile_d), Float8E4M3FN,
                     depth=2, data_format="nz")
self.l0b = Channel(MemLoc.L0B, (tile_n_qk, tile_d), Float8E4M3FN,
                     depth=2, data_format="nz")
# scale_a/scale_b 通过受支持的 matmul 路径传入；MX L0 handle 不在
# Python 前端手工构造。
```

`Buffer` 明确拒绝 L0A_MX/L0B_MX。因此，依赖手动 MX 地址共享的旧方案已不受公开前端支持；不要把上述 channel-first 示例扩写成不存在的 MX Buffer API。

来源：memory `project_fa_fp8_pv_zero`；蓝本的 `Matmul.__init__`。

## 2. UB 越界 → 设备错误

**症状**：UB 分配超 256 KB 上限。编译干净，运行时报设备错误。

**原因**：UB 分配超 256 KB 上限。最常见的触发是把某个大 buffer（例如 `qk_ub`）从单 buf 改 DB 之后没重新算预算。**`vf` 区域内的 cast destination 不一定会被折叠掉**（cast→mem_copy(nd2nz) 边界两侧 storealign / loadalign 不匹配），所以 fp8 暂存 buffer 仍可能实占 UB，即使你以为 vf 会消掉它。

**修法**：分配前先纸面算预算，详见 `buffer-budget.md`。预算紧时按顺序：
1. `qk_ub` 双 buf 降单 buf（失去 cube-QK / vec-P 重叠；FA 是 PV-bound 的，影响有限）。
2. `tile_n_qk` / `tile_k_pv_chunk` 减半（K 维所有 buffer 一起缩）。
3. 把生命周期不重叠的 UB region aliasing（`res_o` + `cast_ub_fp16`）。

来源：memory `project_fa_mxfp8_preload_ub_oob`。

## 3. softmax_max / softmax_sum UB 重叠 → 一半行全 0

**症状**：PV 输出上半行（0..63）正确，下半行（64..127）bit-zero。容易被误判成 2-AIV 竞态。

**原因**：你手工写 `softmax_max_ub` (M×4 字节) 与 `softmax_sum_ub` (同 M×4 字节) 的地址，它们重叠了。cannir lowering 把 `reduce_max` 与 `reduce_sum` fuse 到同一个外层循环，循环内 `asc_storealign_1st(softmax_sum_ub + i)` 会在循环还没跑完时就 clobber `softmax_max_ub` 的后半。被污染的 max 流入 `exp(qk - max)` → 下溢成 0 → 那些行的 P 全 0 → PV 也全 0。

**修法**：**永远不要手工写 UB offset**。通过 `Source.alloc_buffer_ub(dtype_size, num_elems)` 让地址机械计算。所有蓝本变体都已经这样做。

来源：memory `feedback_fp8_softmax_ub_overlap`。

## 4. 插 `mem_copy(ub→gm)` 调试回读读到陈旧数据 → 追了一天假 bug

**症状**：你想看某个中间结果，插一条回读，看到零或乱码，认定"上游 op 坏了"，然后追一个根本不存在的 bug。

**原因**：跨 PIPE 同步规则——`mem_copy(gm, ub)` 跑在 MTE3 上，所以想等 fixpipe 产出的 UB，需要等 MTE3 通道就绪——**不是** FIXPIPE（那会把 FIXPIPE 自己 block 住）、**也不是** V。

**修法**：调试回读用的对照表（源/目的侧只标注 pipe 语义即可）：

| 产回读 buffer 的源 | 源侧 pipe | 回读侧 pipe（永远 MTE3）|
| --- | --- | --- |
| cube fixpipe → UB | FIXPIPE | MTE3 |
| vec V → UB | V | MTE3 |
| GM→UB DMA → UB | MTE2 | MTE3 |

如果下游还要读这个 UB，记得在 `mem_copy` 之后补 MTE3 通道就绪。

来源：memory `feedback_debug_dump_pipe_sync`。

## 5. L0B QK shape 顺序——N 在前、K 在后

**症状**：QK mmad 只写满 L0C 的一半。最终输出部分为 0。

**原因**：matmul lowering 把 NZ 的 rhs 当成逻辑 `(N, K)` 读。如果你按"自然顺序"写成 `Channel(MemLoc.L0B, (tile_d, tile_n_qk), ...)`，N 就被读成 tile_d，少一半。

**修法**：QK 的 L0B 把 N 写在第一维：`shape=(tile_n_qk, tile_d)`。PV 用 ZN 时顺序是 `(K, N) = (tile_k_pv_chunk, tile_d)`。当前参数下两者都恰好是 `(128, 128)`——这是巧合，不是规律。

来源：memory `project_fa_mxfp8_preload_ub_oob` §"相关修复"。

## 6. PV fixpipe 漏切 slice → 写过 pv_ub 边界

**症状**：`pv_ub` 的 `(M, D)` 区域后面是垃圾；严重时撞上 UB 上限。

**原因**：L0C 是按 QK 的 `(M, N_qk=128)` 大小分配的。PV 写其中 `(M, D=128)` 子区域。如果你 fixpipe-drain 整个 L0C 缓冲区到 `(vec_m, D)` 大小的 `pv_ub`，fixpipe 会按父缓冲区的 N=N_qk 推 `n_size`，越界写。

**修法**：在 `_delayed_pv_and_update` 里给 `mm_pv_chunk` 和 `store_single_tile_to_ub` 都传一个 `local_slice(l0c, (M, D), 0)`。slice 自带正确的 `(M, D)` 形状。

来源：memory `project_fa_mxfp8_preload_ub_oob` §"相关修复" 第 2 项。

## 7. `vf` outputs 列表必须列全 vf 之外被读到的 buffer

**症状**：某个 buffer 在 vf 区域之后是垃圾。无编译报错。

**原因**：`vf` 的 `outputs=[...]` 声明哪些 store 要保留。没列进去的会被丢掉（vf 假设它只在区域内用）。例如忘了列 `sum_a_partial_ub`，跨 macro 的 partial 累加就静默丢失。

**修法**：vf 区域之外每个会被读到的 buffer 都列进 `outputs`。拿不准就列上——这不是性能代价，只是不丢 store 而已。

## 8. `softmax_max` 三 buffer 索引取错 → 拿到陈旧 max

**症状**：online softmax 在首轮之后累加错误的 rescale 因子。unit_scale 测试中"本该 ~0 的行"出现 `max|err| > 0.5`。

**原因**：softmax_max / sum / exp 用 macro_idx % 3 三 buffer 轮转。如果索引错了周期（例如本该读 previous 的拿了 current），rescale 拿到未初始化数据。

**修法**：可支持的新实现用 `Channel(..., depth=3)` 让生产/消费顺序驱动缓冲区轮转。旧蓝本的 `m_axis_triple = (macro_idx + 2) % 3` 随机索引访问若无法改写为 Channel 生产/消费顺序，当前前端不支持；不要复制旧 API 或自行发明索引接口。

## 9. K-chunk PV 用 init=True/False 时需要跨 chunk rescale（variant D）

**症状**：unit_scale 测试 ~7% NaN 散落在输出里。

**原因**：你天真地 softmax_round_a 处理 QK_a（max frame=m_a），cast P_a→fp8，softmax_round_b 处理 QK_b（frame=m_b），cast P_b→fp8，然后 `mm_pv_chunk(init=True)` 算 chunk_a、`mm_pv_chunk(init=False)` 算 chunk_b——两 chunk 在不一致 frame 下累加。P_a 应该被 `exp(m_a - m_macro)` rescale，但没做。

**修法**：variant D——把 `exp_a_correction = exp(m_a - m_macro)` 编码成 `sp_l1.chunk_a` 的 e8m0 scale，`sp_l1.chunk_b` 保持 0x7F（= 1.0）。mxfp8 PV mmad 把这两个当 per-block-of-32 scale 用，rescale 由硬件"免费"完成。详细数学看 variant D 计划文档或蓝本顶部 docstring 的 "Math (D 方案 — solved)" 段。

来源：蓝本顶部 docstring。
