---
name: chunk-linear-attn-bwd
description: Chunk 线性注意力反向（结合律重排 / WY）Triton-Ascend 经验。禁止套用 fused-recurrent 的 1D persistent 时间串行 L1。
metadata:
  type: reference
---

# Chunk 线性注意力反向（结合律重排）

写法分类：**四 线性类 · 结合律重排**（无 softmax；chunk 内 GEMM + 下三角 `A`；按 chunk 二维发核）。
**不要**套用 `linear-recurrent.md` 的 L1.1（强制 1D persistent + `for t in range(T)`）。

本卡 Layer 1 只卡 **骨架 / 比较链 dtype / `A` 布局**。`make_block_ptr`、tile 数、fused 写回约定、kernel 名是 **一次任务的默认姿势**，放在 Layer 2/3，禁止当成「换一种等价实现就不合法」。

证据（一次任务记录，不是类别 KPI）：`chunk_kda_bwd_wy_dqkg_fused`，Ascend950PR（DAV_3510），Q3 + FLA GPU 语义，6 pass shape。

| 项 | 值 |
|---|---|
| Phase 3 基线 `S_arith` | 0.3258（fp32 `tl.dot`，BK=BV=16） |
| Phase 4 `opt_iter_11`（裸指针+mask，`i_t` 仍 int64） | 0.8648 |
| 该任务 Q3 合法最佳 | **1.0981**（int32 + `make_block_ptr` + `dv2` 留在 cube V-loop；卡 6 相对拆开档 1.0535 为 +4.2%） |
| 精度 | verify 6/6 |

| | 机制 A（比较链位宽） | 机制 B（block_ptr） |
|---|---|---|
| 本任务消融 | `i_t` int64→int32：**86%**（0.8654→1.0245） | 再上 block_ptr：**14%**（几乎只在 HV=2） |

---

## Layer 1 设计约束（硬；进 precheck）

1. **禁止时间轴 persistent 串行。** 本类按 **chunk × (batch/头)** 二维（或等价多维）分派，chunk 内做 GEMM。禁止套 recurrent 的 `grid=(B*HV,)` + `for t in range(T)`。等价切分（例如 `(NT, B, HV)`）合法，**不要**把某一次的 `grid=(NT, B*HV)` 写成唯一合法公式，也 **不要**把 kernel 函数名写成硬约束。
2. **比较链 int32，地址链 int64。** chunk/token 下标 `i_t` 以及由其生成的 `o_t` / 越界 mask / 下三角 `m_A` 必须是 int32（varlen：`chunk_indices` 载入后 `.to(tl.int32)`；非 varlen：`i_pid.to(tl.int32)`）。禁止把已是 int32 的 `o_t` 加到 int64 的 `bos` 上再比较（会提升回 int64）。**保持 int64**：`bos` / `eos` / `i_tg`、`base + bos * stride`。仅当 `i_t * BT + BT < 2^31` 时截 int32。GPU FLA 的 `program_id().to(int64)` 不得流进比较链。
3. **`A` 是转置语义，不是「必须某种 API」。** 存储按时间步 × BT，读成 `(BT, seqlen或T)` 才能 `A @ dv`（或同类收缩）。草图必须写出这一布局；用 `make_block_ptr` 或 int32 的仿射指针均可，**不得**用会把比较链抬回 int64 的写法。
4. Ascend 通用：`@triton.jit` 内禁止 `return/break/continue`；mix 算子忽略 GPU `num_warps/num_stages`。

---

## Layer 2 算法骨架（参考方向，允许等价实现）

本任务落地的默认姿势，**不是**「不这么写就不合规」：

- 外层 program：`(i_t, i_bh)`；常见 `grid=(NT, B*HV)`；varlen 用 `chunk_indices` + `cu_seqlens`。
- 内层：`for i_k` × `for i_v` 切 K/V；`dq += do @ h`，`dk += v_new @ dh`，`dw += dv @ h`，再乘 `exp2(g)` / `scale`。
- **2D tile 默认 `make_block_ptr` + `boundary_check`**（转置 `A`、非 v-first 的 `h`、TND 的 q/k/v/do/dq/…）。offsets 用 int32，可顺带做完 L1.2。int32 裸指针+mask 在消融里也可以没有 4096 环；HV=2 上 block_ptr 另有约 14%。不要把「不用 block_ptr」判成架构错误。
  - 转置 A 示例：`make_block_ptr(A_ptr, (BT, seqlen), (1, HV*BT), (0, i_t*BT), (BT, BT), (0, 1))`
  - 非 v-first h：`make_block_ptr(h_ptr, (V, K), (1, V), (i_v*BV, i_k*BK), (BV, BK), (0, 1))`
- last-row `g`（一行）可留 `ptr + last_t*stride + arange` + mask；1D strided block_ptr 不保证更快。
- **本 fused 的写回约定**（换切分可改）：`i_k==0` 才写 dv2 / 累加 db / `dA += dv @ trans(v)`。默认 **留在 cube V-loop 内**，不要为点 18 再扫一轮 V。
- WY：`dA` 来自 `dv @ trans(v)` 与 `dw @ (k*exp2(g))ᵀ`，再 strict-lower。
- Tile 起点（950PR、`BT=K=V=64`）：`BK=64, BV=32`（`64,64` 会 UB 溢出）；其它 K/V 另记账。

---

## Layer 3 关键技巧（本任务验证，可替代）

1. **Cube 操作数可用输入 fp16/bf16，acc fp32**（优化点 23）。本任务 0.61→0.86 的主杠杆。不要把 `exp2(g)` 路径改成低精度。
2. **WY 末段两次收缩 `tl.dot` 用 fp32 `b_Af`。** 本任务 Q3 上改成 `b_A.dtype` 会 dA `max_abs≈200`。换标杆须重验，不要当成一切 gold 的定律。
3. **比较链 int32 是性能主项**；block_ptr 是本类默认写法 + 部分 HV 的余量。禁止把 +22% 写成 gather→DataCopy。点 4 不打 `pid*BT+arange`。`tl.trans(b_v)` 代替再 load `vᵀ`：本任务增益约 0%。
4. 不要用 1D persistent 救小 HV（本任务 t001/t002/t006 未抬 `S_arith`）。
5. **`dv2` 留在 cube V-loop 内（`i_k==0`）**，不要为点 18 再扫一轮 V。本任务从拆开并回 `S_arith` 1.0535→1.0981（+4.2%，6/6）。点 18 要的是 `atomic_add` scatter；本 fused 的 `dv2` 是独占地址的 plain store。
6. Simulator：改 `i_t` int32 后 Cube `FLOWCTRL` 占比可以仍高，Vector 总 cycles 下降。不要把 FLOWCTRL 高当成「改 int32 没用」。
7. 本任务 kernel 名曾为 `chunk_kda_bwd_kernel_wy_dqkg_fused`，仅作记录。

---

## 陷阱

| 陷阱 | 说明 |
|---|---|
| 套用 recurrent L1 | 会生成错误的 1D 时间串行草图 |
| **L1 写死 `i_t` int64** | last_pass `scf.for 0..64/4096`；点 4 不命中。比较链必须 int32 |
| 把 L2 的 block_ptr 当成 L1 | int32 裸指针也可以没有 4096 环；强制全部 TND 会挡合法等价实现 |
| 写死 kernel 名 / `grid=(NT,B*HV)` 唯一 | 等价 2D 分派应合法 |
| 把 22% 写成 gather→DataCopy | 仿射 mask 可以仍是 2d ubuf |
| `bos`/`i_tg` 也改 int32 | 地址溢出 |
| 残缺 attention-bench 参考 | 缺 db/dA/dw 会假通过 |
| 末段 dA 走 tile dtype | 本任务 Q3 挂；换标杆重验 |
| `(B*HV).to(int64)` | constexpr 没有 `.to` |
| 1D strided block_ptr 加速 last-row | 已证伪；last-row 可留 mask |
| 无 atomic 却按点 18 拆 dv2 | `dv2` 是独占地址的 plain store；拆开会再扫一轮 `dv`。本任务并回 +4.2% |
