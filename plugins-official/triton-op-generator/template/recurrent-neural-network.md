---
name: recurrent-neural-network
description: 循环神经网络类算子（LSTM / GRU / RNN）的 Triton Ascend 优化经验合集，按算子分章节组织，含通用经验 + 各算子专属约束/骨架/kernel
metadata:
  type: reference
---

# 循环神经网络类算子优化经验

本文档合并该类别算子的优化经验。按以下结构组织：
- **§1 通用经验**：跨算子（LSTM/GRU/RNN）共有的递归工程约束，首次生成必须遵守
- **§2 LSTM**（首个归档算子）/ **§3 GRU**（待归档）/ **§4 RNN**（待归档）
- **§5 常见陷阱**（按算子分小节）

> ⚠️ **关键区分**：本类别的核心优化哲学是 **「host-loop 单步 kernel + 禁止 host 转置 + kernel 内 coalesced 加载」**。时间维递归必须在 host 侧串行；权重以原生布局传入，kernel 内通过 `tl.trans` 获得转置视图，禁止调用 `.t()`/`.transpose()`。
> 首次归档算子：`1_LSTM`（单层 LSTM 前向，60-case 多 shape，ascend910b1）；归档时最佳几何平均 **13.3673x**。
> 历史经验仅供启发，**禁止直接复制代码结构**；变量名/结构必须重新设计。
>
> **与 `linear-recurrent.md` 的分工**：本卡状态是 `[B,H]` 向量、无大 tile 可驻留，故用 host-loop 单步 kernel；那张卡负责**矩阵状态**（`[K,V]` fp32 ≤32KB 可整块驻留 UB）的线性递推算子（gated delta rule 等，attention_index 行 12），其正解是单融合 persistent kernel + device 内 t 循环（op92 实证 5.75×）。两卡 R1/R2/R3 的"时间维串行、program 持完整状态"结论一致，分歧仅在发射粒度。

---

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| LSTM | `recurrent-neural-network` | 4-gate(i/f/g/o) 递归，cell c + hidden h 双状态 | **host-loop 单步 kernel 保证冷启动稳定**，matmul + elementwise 解耦 |
| GRU | `recurrent-neural-network` | 3-gate(r/z) 递归，单 h 状态，无 cell | 待归档；可套用 §1 R1-R5 |
| RNN（vanilla） | `recurrent-neural-network` | 单 gate `h=tanh(x@W+h@U+b)` 递归 | 待归档；重点是 §1 R1-R4 + UB/tile |

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下 5 条约束是本类别算子**共有**的递归工程约束。其他文件已提取的通用约束（如 `tensor-transform.md` 的 G1 动态 num_cores / G2 pow2 BLOCK / G4 grid 不超核数 / G7 contiguous 等）此处不再重复，各算子章节引用时标注。

### R1 时间维必须串行（递归数据依赖）
- **必须** 在单个 program 内 `for t in range(S)` 串行推进时间步；**禁止**把时间维展平进 grid 跨 program 并行。
- **Why:** 递归依赖 `h_t = f(h_{t-1}, ...)` 使时间步间存在真数据依赖；Triton 缺跨 program 全局屏障，跨 program 并行时间维会 RAW 冒险。
- **典型应用**：所有 recurrent 算子（LSTM/GRU/RNN）的前向。
- 与 tensor-transform G4（grid 不超核数）的差异：R1 进一步约束 grid 维度只能落在 batch，**时间维绝对不可分**。

### R2 每个 program 必须持有完整隐维 H（递推归约全 H）
- **必须** 让每个 program 持有完整隐维 H；隐维 tile 只能在**同一 program 内**循环（`for ht in range(num_hidden_tiles)`），且每个 tile 的 W_hh 归约仍读全 H。**禁止**按隐维 tile 把不同 program 分到不同 H 段。
- **Why:** 递推项 `h_{t-1} @ W_hh` 的归约维是全 H（W_hh 为 [H, G·H]，G=gate 数：LSTM=4/GRU=3/RNN=1），每个输出 gate 维依赖全部输入隐维。跨 program 切隐维 → program A 只算了 h_{t-1} 半段，下一步 W_hh 却要读全段 → RAW。
- **How to apply:** 外层 `for ht`（输出隐维 tile），内层 `for k in range(0,H,BLOCK_K)` 读全 H 的 h_tile，同一 h_tile 复用于全部 G 个 gate 的 `tl.dot`（避免 G× 重复 load h）。
- **典型应用**：LSTM(G=4)/GRU(G=3)/RNN(G=1) 递推 kernel。

### R3 仅按 batch 块分核
- **必须** `grid=(num_cores,)`，每个 program 处理一个 `BLOCK_BATCH` 块并串行全部时间步 + 全部隐维 tile。**禁止**用隐维分核"提升并行度"（违反 R2）。
- **Why:** 时间维（R1）与隐维（R2）都不可跨 program 分，唯一可并行的维度是 batch。**小 batch case 并行度低是固有特性**（B=1 时 19/20 核空闲），非优化能解。
- **典型应用**：所有 recurrent；B=1 大 S case 是性能洼地，受限于递归时间屏障（host-loop 逐时间步全 GEMM grid 亦被时间屏障挡）。

### R4 (2,B,H) 双缓冲工作区规避递归 RAW
- **必须** 相邻时间步用乒乓读写 h/c（及 cell state）：`read_off=(t%2)*stride` / `write_off=((t+1)%2)*stride`。**禁止**单缓冲。
- **Why:** 单缓冲下 t 步写 h_t 与 t+1 步读 h_t 落在同一地址，UB/GM 间 RAW 冒险；双缓冲让读写落在不同物理页。
- **How to apply:**
  ```python
  read_off  = (t % 2)       * stride_ws_s
  write_off = ((t + 1) % 2) * stride_ws_s
  ```
- **典型应用**：LSTM（h+c 双状态需两组双缓冲）、GRU（h 单状态一组）。

### R5 recurrent kernel 瓶颈是 scalar/address overhead，不是 cube 吞吐
- **优化方向必须**是削减 K-loop 内的 load 条数与地址算术，**而非**放大 BLOCK_M/BLOCK_K 堆 cube。
- **Why:** profiling 实测传统 fused recur kernel PipeUtilization = scalar ~45% / mte(load) ~29% / **cube(macc) 仅 ~7%**。根因是每个隐维 tile 在 K-loop 内对每个 gate 各做一次 strided load + 小-N dot，地址算术与 load 指令数爆炸，cube 空转。堆 BLOCK 实测普遍退化（§5.1：BLOCK_K=256→0.594x、BLOCK_BATCH=32→UB 溢出）。
- **How to apply in host-loop design:** 把 recurrence 拆成独立 matmul + elementwise，让 matmul 走标准二维 coalesced load，elementwise 无 dot；避免在单个 kernel 内做 4-gate strided load 的复杂 K-loop。
- 与 math-compute M4（UB 决定 tile）的差异：R5 强调瓶颈在地址/load 而非 cube 或纯 UB，优化杠杆是"简化 load"而非"调 tile 形状"。

---

## §2 LSTM 算子（lstm）

**算子类别**: `recurrent-neural-network`
**典型特征**: 4-gate(i/f/g/o) 递归，cell state c + hidden h 双状态，单层前向；输入 x[S,B,I]、W_ih[4H,I]、W_hh[4H,H]；多 dtype（fp32/fp16/bf16）、多 shape（含 batch_first 翻转）。
**性能基准**: 60/60 pass，几何平均加速比 **13.3673x** vs torch（ascend910b1）

### §2.1 Layer 1: 设计约束（LSTM 专属，Agent 必须遵守）

#### L1.1 禁止 host 侧 `.t()` / `.transpose()`；权重按原生 `[4H,I]` / `[4H,H]` 布局传入 kernel，kernel 内 coalesced 加载 + `tl.trans`
- **必须** 以 nn.LSTM 原生布局 `[4H,I]` / `[4H,H]` 直接传入 kernel，**禁止** host 侧 `weight_ih_l0.t().contiguous()`、`weight_hh_l0.t().contiguous()` 或 `output.transpose(0,1).contiguous()`。
- **Why:** host 转置方法在需求上被禁止；同时在 triton-ascend 上，若直接按 `[BK,BN]` tile 交换 stride 读取（`offs_k*1 + offs_n*I`），tile 的快速轴 `BN` 会落在内存中非连续的 `4H` 维度上，产生非 coalesced gather load，实测几何平均从 7.2x 暴跌到 **0.12x**。
- **How to apply:** 在 kernel 内先以**原生朝向 `[BN,BK]`** 加载权重（此时 `BK=I/H` 轴连续，coalesced），再在寄存器中用 `tl.trans` 转成 `[BK,BN]` 供 `tl.dot` 使用：
  ```python
  w_native = tl.load(w_ptr + offs_n[:,None]*stride_w_n + offs_k[None,:]*stride_w_k, ...)
  acc += tl.dot(x_tile, tl.trans(w_native))   # w_native[n,k] = weight[n,k] == W^T[k,n]
  ```
  其中 `stride_w_k = weight.stride(1)`（I/H 轴，=1），`stride_w_n = weight.stride(0)`（4H 轴，=I/H）。`tl.trans` 是 tile/寄存器级操作，不是 tensor 方法，不在禁止之列。实测修复后几何平均回升并提升到 **13.37x**。
- **batch_first I/O:** 输出按目标布局直接 `torch.empty(B,S,H)` 或 `(S,B,H)`，将逻辑 S/B/H stride 传入 gate-update kernel，避免末尾 `transpose()`；batch_first 输入用每步仿射切片 `x[:, t, :]` 读取（融合 GEMM 无法以仿射 stride 读 `[B,S,I]`）。

#### L1.2 冷编译首次执行稳定性：禁止 kernel 内跨 timestep 循环，必须使用 host-loop 单步 kernel
- **必须** 将跨时间步的 recurrence 放在 **host 侧循环**（`for t in range(S):`），每个 timestep 启动独立的单步 kernel（输入投影 matmul / h@W_hh matmul / gate elementwise-update）。**禁止**在单个 `@triton.jit` kernel 内写 `for t in range(S)` 并配合 double-buffer workspace 完成多步递归。
- **Why:** triton-ascend 上带有内部 timestep 循环 + double-buffer workspace 的 recurrence kernel，在冷编译后首次 device 执行时存在不稳定的首启路径；表现为 gate 累加异常（`gate_g/o` dot 贡献缺失），`sigmoid(gate_o)=0` → `h_new` 全零，同 shape 第二次执行才正确。该现象与 kernel 内部 recurrence 结构强相关，单纯的 dot 模式修改、grid 调整、kernel 间/内同步均无法可靠消除。**偶发精度 fail 一律按真 bug 处理，硬标准：单次执行也要对。**
- **How to apply:** 采用三层稳定 kernel 结构：
  1. `lstm_input_proj_kernel`：一次性 `x_proj[S*B,4H] = x @ W_ih + bias`（稳定 matmul）。
  2. Host 侧 `for t in range(S)`：每步调用 `lstm_hh_gemm_kernel(h_cur, W_hh, gbuf)` 计算 `h_prev @ W_hh`（纯 matmul，无 recurrence）。
  3. 同一步调用 `lstm_gate_update_kernel(gbuf, x_proj[t], c_cur, h_nxt, c_nxt, output)` 完成激活与状态更新（纯 elementwise，无 dot）。
  这样没有任何 kernel 包含跨 timestep 的循环或 double-buffer workspace，从根本上避开首启不稳定结构。实测 case11(H=128 fp32) 冷启动 6/6 PASS，case13(H=256 fp32) 3/3 PASS，热缓存 verify 60/60 PASS，benchmark geomean 13.3673x。
- **与 R1/R4 的关系**：R1/R4 描述的是"若必须在 kernel 内做时间串行"时的约束；L1.2 是更高优先级的**替代方案**——直接把时间串行移出 kernel，用 host 循环保证正确性。当稳定性和 peak 性能冲突时，优先遵守 L1.2。

### §2.2 Layer 2: 算法骨架

当前稳定实现统一走 **host-loop 单步 kernel**（L1.1 + L1.2），所有 shape 共用同一路径，不做 I/H dispatch，也不在 kernel 内做 gate 合并。整体结构如下：

```
# (1) 一次性输入投影：x_proj[S*B,4H] = x_flat @ W_ih^T + (b_ih+b_hh)
lstm_input_proj_kernel[grid_proj](x_flat, W_ih, b_ih, b_hh, x_proj)
x_proj = x_proj.view(S, B, 4H)

# (2) host 循环：每 timestep 一个 matmul + 一个 elementwise
gbuf = empty(B, 4H)
for t in range(S):
    h_cur = h_ws[t % 2];      h_nxt = h_ws[(t+1) % 2]
    c_cur = c_ws[t % 2];      c_nxt = c_ws[(t+1) % 2]
    lstm_hh_gemm_kernel[grid_gemm](h_cur, W_hh, gbuf)          # gbuf = h_cur @ W_hh^T
    lstm_gate_update_kernel[grid_upd](gbuf, x_proj[t], c_cur, h_nxt, c_nxt, output)
```

**4-gate 处理**：`lstm_input_proj_kernel` 与 `lstm_hh_gemm_kernel` 都输出 4H 宽度的 gate buffer；`lstm_gate_update_kernel` 在 elementwise 阶段拆成 i/f/g/o，分别做 sigmoid/tanh，然后 `c_new = f·c_old + i·g`、`h_new = o·tanh(c_new)`。

**实测 tile 配置**：
- `lstm_input_proj_kernel`: `BLOCK_M=BLOCK_N=BLOCK_K=128`（大 matmul，一次性处理 S*B 行）。
- `lstm_hh_gemm_kernel`: `BLOCK_M=BLOCK_N=BLOCK_K=64`（小 matmul [B,4H]=[B,H]@[H,4H]，避免 UB 溢出）。
- `lstm_gate_update_kernel`: `BLOCK_B=32, BLOCK_H=64`（同时驻留 8 个 gate tile 时的 UB 上限）。

### §2.3 Layer 3: 关键技巧（可参考，不可照抄）

- **bias 合并到输入投影**：`b_ih + b_hh` 在 host 侧相加后一次性加到 `x_proj`，避免在每个 timestep 重复加 bias；bias 为 None 时创建 `torch.zeros(4H)` 占位，避免 kernel 分支。
- **工作区强制 fp32**：`h_ws/c_ws` 与 `x_proj/gbuf` 都用 fp32，只在最终写 output / 返回 h_n/c_n 时 `.to(x.dtype)`，减少递归中 fp16 累放误差。
- **gate buffer 的宽连续布局**：`gbuf[B,4H]` 与 `x_proj[t][B,4H]` 都按 `[batch, 4*H]` 连续存储，`lstm_gate_update_kernel` 通过 `g*H + h_offs` 取对应 gate 段，无需切片或 split。
- **UB/tile 上限**：`lstm_gate_update_kernel` 同时加载 8 个 tile（4 个来自 gbuf、4 个来自 x_proj），容易触发 UB 溢出。需根据 H 调整 `BLOCK_B × BLOCK_H`，实测 32×64 在 192KB UB 下安全。
- **测量必须用 benchmark.py（torch_npu.profiler 解析 op 时长）**，自写 harness（npu.Event/host loop）对 <0.3ms 的快 case 有 host-launch 开销下限，不可信。

### §2.4 LSTM 性能基准（ascend910b1，60-case 几何平均）

| 方案 | 几何平均 | 说明 |
| --- | --- | --- |
| **host-loop 单步 kernel 重实现 + 原生权重 `tl.trans`（L1.1 + L1.2）** | **13.3673** | **2026-08-07 最终稳定版本，移除 host `.t()`/`.transpose()`，kernel 内 coalesced load + `tl.trans`，60/60 verify + 冷启动稳定性验证** |

**host-loop + 原生权重布局**：将 recurrence 从 kernel 内部移到 host 循环，每 timestep 拆分为独立 matmul + elementwise kernel；同时以原生 `[4H,I]`/`[4H,H]` 权重布局传入，kernel 内做 `[BN,BK]` coalesced 加载 + `tl.trans` 得到 `W^T`。该结构既消除了冷编译首次执行的不稳定首启路径（case11 6/6 冷启动 PASS），又避免了 host 转置方法的禁止调用，几何平均达到 **13.37x**。

---

## §3 GRU 算子（gru）

**待归档**。3-gate(r/z) 递归、单 h 状态（无 cell），递推 `h_t = (1-z)·h_{t-1} + z·r̃`。可复用：§1 R1-R5 全部通用约束；无 cell state → R4 只需一组双缓冲。优化重点在于 host-loop 拆分后的单步 matmul + elementwise 的 tile 与 UB 适配。

---

## §4 RNN 算子（vanilla rnn）

**待归档**。单 gate `h_t = tanh(x_t@W_ih + h_{t-1}@W_hh + b)`。可复用：§1 R1-R4；单 gate 无 multi-load 问题，优化重点回到 UB/tile 形状与 R3 batch 分核的并行度挖掘。

---

## §5 常见陷阱与避免方法

### §5.1 LSTM 陷阱

| 陷阱 | 现象 | 正确做法 |
| --- | --- | --- |
| 跨 program 隐维分核 | verify 58/60，RAW 冒险 | 遵守 §1 R2，隐维 tile 留在同一 program 内 |
| `(B,4H)` 缓冲切片提 4 gate | `unsupported tensor index: slice(constexpr[0,64])` | 在 elementwise kernel 中通过 `g*H + h_offs` 索引读取，不切片 |
| BLOCK_K=256（输入投影 matmul） | 0.594x，UB 压力退化 | 输入投影 `BLOCK_K=128` |
| BLOCK_BATCH=32 + BLOCK_H_TILE=128 | AICore 异常 507015（UB 溢出） | 保持 16/64 |
| 循环不变量 bias 外提到时间环外 | ~14 case 精度错（impl 该 0 处非 0） | bias 并入输入投影一次性加完 |
| **verify 偶发报 fp32 Path-B case impl 全零、framework 正常**（典型 case12，forward 内实际 `S=1,B=2,H=128`，`batch_first` 翻转后看 S/B） | **kernel 内有 `for t in range(S)` 跨 timestep 循环 + double-buffer workspace 时，冷编译首次 device 执行出现 gate 累加异常**：`gate_g/gate_o` 的 dot 贡献缺失，`sigmoid(gate_o)=0` → `h_new` 全零；**同 shape 第 2 次执行即正确**。这是**真 bug**（偶发精度 fail 必须修，硬标准：单次执行也要对）。已排除 workspace 未初始化（output 预填 sentinel 仍被写成 0，证明 store 执行了）、host 同步、grid 大小、4 连续 dot 模式等假根因 | **真正修复 = 按功能重实现为 host-loop 单步 kernel**：把 `for t` 移到 host 侧，每 timestep 启动独立的 `lstm_hh_gemm_kernel`（纯 matmul）和 `lstm_gate_update_kernel`（纯 elementwise），没有任何 kernel 包含跨 timestep 循环或 double-buffer workspace，从而避开首启不稳定结构。`self-warmup` 只是用第二次执行掩盖问题，不被接受。实测 case11(H=128 fp32) 冷启动 6/6 PASS、case13(H=256 fp32) 3/3 PASS、热缓存 verify 60/60 PASS |
| 原生 `[4H,I]` 权重直接交换 stride 读 `[BK,BN]` tile | benchmark 几何平均从 7.2x 暴跌到 **0.12x**（实现平均延迟 0.056ms → 7.18ms） | 遵守 L1.1：以原生朝向 `[BN,BK]` coalesced 加载 + 寄存器 `tl.trans`，禁止让 tile 快速轴落在非连续维 |
