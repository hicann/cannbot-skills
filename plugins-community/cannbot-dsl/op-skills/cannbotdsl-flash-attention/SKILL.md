---
name: cannbotdsl-flash-attention
description: 在 CANNBotDSL Python DSL（cannbot-dsl 仓库）里生成或修改 Flash Attention kernel。用户要求写新的 FA kernel、在 FA 变体之间移植特性（causal/GQA/mxfp8/Channel preload）、调整 tile 形状或 buffer 预算、修 FA 精度问题、或在已有 FA 上加 online-softmax / K-chunk PV streaming rescale 时，触发此 skill。即便用户只说"加个 nbuffer preload""改成 mxfp8"也要触发——前者是需要识别并改写成 Channel（或明确报不支持）的旧术语。非 FA 的 NPU kernel（matmul、conv、layernorm）跳过。
---

# 在 CANNBotDSL 里写 Flash Attention kernel

这个仓库（`cannbot-dsl`）是一套 Python DSL，JIT 编译到 AscendC 跑 Ascend NPU。有 5 个 FA 变体，从 FP16 dense 阶梯式演进到 FP8 mxfp8 + macro-iter preload，每一级是前一级的超集——你要做的是**挑最近的那个变体当蓝本，做 diff**，不是从零写。

FA 在这个 DSL 里最难的部分**不是数学**——online softmax 是死的。难的是：buffer 预算装得下、`vf` fusion 区域合法、cube/vec PIPE sync 正确、（mxfp8 情况下）e8m0 scale 格式 / copy engine 与 verifier 规则对得上。这个 skill 把这些约束都固化下来。

## 第 1 步 —— 把用户的需求定到一个变体

如果用户要的组合不存在（例如"causal + mxfp8"），按**最难加上的那个维度**选蓝本：在 mxfp8 基础上加 GQA / causal mask 很机械（head 索引、mask 在 softmax 前）；但**在 FP16 基础上加 mxfp8 不机械**——它会动 L0 buffer 类型、scale buffer、copy engine、matmul 签名。深度 preload 只能基于 Channel FIFO 可表达的蓝本扩展；旧 nbuffer-preload 蓝本不能作为可编写 API 范例。

> **sync 写法：channel-first。** depth-N 的 L1/L0/UB storage 用 `Channel(..., depth=N)`，单块无同步 scratch 用 `Buffer`；buf_id/sync_id 由框架自动管理，跨核 handoff 用 `kind=CrossCore`。旧蓝本中手工 NBuffer、外部指定 buf_id 或 MX 地址空间的公开前端 API 均已移除；`Buffer` 也不接受 L0A_MX/L0B_MX。如果一个旧 mxfp8/深流水方案必须手工多槽、精确 MX offset 且无法用 Channel 表达，当前应明确标为"前端不支持"，不得伪造替代 API。隐式 MX handle 由现有 `matmul` lowering 推导的 channel-first 路径可继续使用。

挑好蓝本后，**先和用户确认**，一句话即可："我准备以 `<file>` 为蓝本，做这些调整：`<diff 列表>`，对不对？"

## 第 2 步 —— 读懂蓝本的骨架

每个变体（从 `gqa` 起）都是同一套 5 个 class 的骨架：

| Class | 作用 |
| --- | --- |
| `Source` | L1 / L0A / L0B / L0C / L0A_MX / L0B_MX / UB 的累加式字节分配器，返回 offset。手工填 offset 容易踩坑（见 `feedback_fp8_softmax_ub_overlap`），**永远走 Source 分配**。 |
| `Matmul` | 包装 cube 侧所有 op：`load_q/k/v`、`load_scale_*`（仅 mxfp8）、`mm_qk`、`mm_pv[_chunk]`、`store_*`。持有 L0A/L0B/L0C Channel，用 depth 表示槽数。 |
| `Vector` | 包装 vec 侧 softmax：`atten_mask`、`online_Softmax[_first]`（preload 变体里是 4 个 `softmax_round_*_finalize_*`）、`store_p`、`init/update/finalize_o`。持有 UB buffer 与三 buffer 的 softmax_max/sum/exp 标量。 |
| `BlockInfo` | 算 `(n_min, n_max)` 行 tile 范围——处理 causal 稀疏。 |
| `FlashAttention` | 顶层编排。`__init__` 配置 tile 与 sparse_mode；`flash_attention_kernel` 是 `@kernel` 装饰的主循环；`run` 是 host 侧 `@jit` 入口。 |

简单变体从最近的、已迁移为 Buffer/Channel 的可编译蓝本做减法。不得复制仍依赖已删除多槽 API 的旧变体。

## 第 3 步 —— 写代码前先**算 buffer 预算**

Ascend 内存层级有硬上限：

| Region | 上限 | FA 典型用量 |
| --- | --- | --- |
| UB  | 256 KB | softmax 中间量、P cast 暂存、atten_mask、res_o |
| L1  | 512 KB | Q/K/V tile、P tile、scale tile（mxfp8） |
| L0A | ~64 KB | Q tile（PV 时也是 P tile） |
| L0B | ~64 KB | K tile（PV 时也是 V tile） |
| L0C | ~256 KB | QK 与 PV 的 fp32 累加器 |

超上限的表现是 **NPU error 507015**——编译干净，跑到 `torch.npu.synchronize()` 时报错。我们在这上面烧过好几天（见 `project_fa_mxfp8_preload_ub_oob`）。**在 `Matmul.__init__` / `Vector.__init__` 写第一行 alloc 之前，先在 docstring 里列出 buffer 预算表算一遍。**

预算模板与已知合法配置见 `references/buffer-budget.md`。

mxfp8 还有一个**关键坑**：`asc_mmad_mx` 跑 fp8 时，**只要 L0A / L0B / L0A_MX / L0B_MX 任何一个 offset 不是 0，输出就是全 0**。修法是让 L0_pv 与 L0_qk 共享 offset 0（细节见 `project_fa_fp8_pv_zero`）。所有 mxfp8 变体都已这么做。如果你写 `addr_l0a_pv = self.source.alloc_buffer_l0a(...)` 而不是 `addr_l0a_pv = addr_l0a_qk`，准备再烧一天。

## 第 4 步 —— 规划 cube/vec PIPE sync

CANNBotDSL 同步分两种通道（channel-first 下均由框架自动合成）：

- **4 相协议（acquire/commit/wait/release）**：ring-buffer slot 生命期锁。每个 Channel 的 `buf_id` 是一个通道，生产者/消费者要为整个 slot 生命周期持锁。
- **跨 PIPE 点对点同步**：两个 PIPE 之间在 `(PIPE, buf_id)` 通道上的显式 happens-before。

`*_sync_intra_*` 第一个参数 = **执行 wait/arrive 那一侧的 pipe**（规则见 `../../core-skills/cannbotdsl-cv-fusion/SKILL.md` §5 铁律 3）。搞反不报编译错，但会读到陈旧数据（见 `feedback_debug_dump_pipe_sync`）。

PIPE 含义（是动作类别，不是位置）：
| PIPE | 动作 |
| --- | --- |
| `MTE2` | GM → UB / GM → L1 |
| `MTE3` | UB → GM / UB → L1 |
| `MTE1` | L1 → L0 |
| `V` | Vector 计算 |
| `M` | Cube mmad |
| `FIXPIPE` | L0C → UB / L0C → GM |

FA 用到的 cube↔vec 同步模式（p_l1 交接、softmax 状态 triple-buffer、variant D 的 sp_l1 通道）走 channel-first：跨核 handoff Channel 用 `kind=CrossCore`，框架自动合成 arrive/wait；macro 级 triple-buffer 也用 `Channel(..., depth=3)` 表示存储与同步。如果旧 `[macro_idx % 3]` 手动多槽方案依赖 Channel 无法表达的自定义调度，则该方案当前不支持。

## 第 5 步 —— 放对 `vf` fusion 区域

`vf`（vector-fold）是把几个 vec op 融合到一个 AscendC vector 循环里、消掉中间 UB 写读的机制，是 FA 性能最大的旋钮。

三条规则：
1. **`vf` 内禁止 runtime 分支**。`scf.if` 在 vf 里不会折叠。所以 preload 变体把 `softmax_round_b_finalize` 拆成 4 个直线函数（`first_has_second` / `first_no_second` / `more_has_second` / `more_no_second`），而不是一个带 `if has_second` 的。
2. **`outputs=[...]` 必须列出所有 vf 区域之外被读到的 buffer**。漏一个（例如 `sum_a_partial_ub`），它的 store 会被丢掉，跨 macro 的累加值就静默变成垃圾。
3. **vf 在区域内保留 program order（RaW 是可信的）**。同一个 vf 内可以先写后读同一 buffer，lowering 会处理。

`vf` 并非什么都能折叠：跨 `cast` → `mem_copy(nd2nz)` 边界的中间 UB buffer，因为 storealign / loadalign 的 inner-loop trip 不同，**往往折不掉**。别假设它会折——预算紧的时候一律按「不折」计。

折叠规律的例子看 `references/vf-folding.md`。

## 第 6 步 —— mxfp8：copy engine 与 format 必须配对

`cannir.mem_copy` 的 verifier 强制 `make_copy_engine(format_transform=...)` 与 dst L1 buffer 的 `data_format` 一一对应。错了在编译期就挂 `requires dst physical format X, got Y`。

| `format_transform=` | 用于 | dst `data_format=` |
| --- | --- | --- |
| `nd2nz` | Q、K（data） | `nz` |
| `dn2nz` | V（data）—— 注意名字带 nz，verifier 实际要 `zn` | `zn` |
| `mx_scale_a` | sQ、sP（A 侧 scale） | `zn` |
| `mx_scale_b` | sK、sV（B 侧 scale） | `nz` |

## 第 7 步 —— 动手写文件

完全镜像蓝本结构，差异只在标记的几个点：

- `Source.__init__` / `alloc_buffer_*`：新增 arena 才扩展（例如为 mxfp8 加 L0A_MX）。
- `Matmul.__init__`：调整 L0/L1 Channel 的 shape 和 depth。**mxfp8 的 L0A/L0B 数据通道需满足 lowering 的隐式 MX handle 约束**；不得通过已删除的公开 API 手工构造 MX buffer。
- `Matmul.mm_qk` / `mm_pv[_chunk]`：调 matmul 签名（mxfp8 多 `scale_a` / `scale_b` 参数）。
- `Vector.__init__`：UB layout；确认 softmax_max / sum / exp 不重叠（虽然小，但 `reduce_sum` 的 storealign 会踩到 `reduce_max` 的 loadalign，见 `feedback_fp8_softmax_ub_overlap`）。
- `Vector.online_Softmax[_first]`（或 preload 变体的 4 个 `softmax_round_*`）：写实际的数学。
- `FlashAttention.flash_attention_kernel`：主循环。保留 Channel depth-N preload 时要维持 warmup / steady-state / drain 的形态。
- `test_*` host 函数：pytest parametrize 元组与 torch 参考。

**文件顶部的 license header 是必需的**——`check-headers` pre-commit hook 没它会失败。从蓝本原样复制。

## 第 8 步 —— 验证

```bash
# 先确认能编译通过（不需要 NPU）
pytest -m ascendc_toolchain <case> -q
```

报 507015 → buffer 预算溢出（UB 通常是元凶）。在纸上算一遍，找出超 256 KB 的那一笔，把双 buffer 降单 buffer 或缩 tile。

`unit_scale` 测试精度挂（输入全 1、K=1）→ 是数学 bug，不是量化。先查 softmax_max/sum 的三 buffer 索引，再查 `vf` 的 outputs 列表。能撑到精度挂这一步的疑案，大概率是 `references/pitfalls.md` 里的某条。

## References

| 文件 | 何时读 |
| --- | --- |
| `references/buffer-budget.md` | 第 3 步——预算模板、合法形状组合 |
| `references/vf-folding.md` | 第 5 步——什么折得了、什么折不了 |
| `references/pitfalls.md` | 始终——已经付过代价的坑 |
| `../cannbotdsl-perf-optimize/references/fa-pipeline-optimization.md` | 性能优化——4-stage pipeline 实战（DelayLineGroup + stage-gate + fused vmadd + 两遍 softmax + scf.if 限制等关键技术，从 318.8us 到 51.5us 的 6.2x 优化路径） |
