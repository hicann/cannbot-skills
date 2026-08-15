# A2/A3 线性 Attention Stage 设计经验

本文沉淀 GDN/KDA 等线性 Attention 算子在 A2/A3 上的 stage 设计经验。适用于同时包含 Cube/Vector 协作、chunk 内矩阵、scan/state 依赖、GQA/GVA head 映射、跨 stage workspace 或性能流水调优的 Catlass/AscendC 算子。

## 何时读取

命中线性 Attention 路由后，如果出现以下任一情况，设计和实现前必须读取本文：

- 算子内部存在多个 stage。
- AIC/Cube 的产物要给 AIV/Vector 使用，或反向依赖。
- 某个 chunk/head workspace 会跨 stage 复用。
- 覆盖 `HV > HK`、GQA/GVA、`V=256`、partial chunk 或 varlen。
- 性能优化涉及 L1 resident、L0 double buffer、UB double buffer、CrossCoreFlag 或 workspace slot 复用。

## 1. Stage 边界来自依赖关系

不要按公式顺序或“一个 matmul 一个 stage”机械切分。先列出每个中间张量的 producer、consumer、pipe、内存层级和同步边界。

推荐依赖字段：

| 字段 | 含义 |
|---|---|
| producer | Cube、Vector、host tiling 或原始输入 |
| consumer | 下一 stage 和消费 pipe |
| memory | L1 resident、L0、UB resident、GM workspace、最终 GM |
| scope | `(batch, chunk, hk/hv)` 或全局 |
| sync | 无同步、同 pipe event、CrossCoreFlag |
| lifetime | 单 tile、单 stage、window、整个 launch |

Stage 切分规则：

1. 同一 stage 内的计算，要么只依赖原始输入，要么依赖本 stage 入口已经 ready 的数据。
2. A2/A3 上 Cube/Vector 跨 stage 边默认走 GM workspace。
3. producer 只有在 GM workspace slot 完整写完后才能 set ready flag。
4. consumer 的 wait 放在 stage 入口，不能放在 row/tile 热循环里反复握手。
5. 逻辑 `(chunk, hv)` 或 `(chunk, hk)` 在所有 stage 中必须命中稳定 workspace slot。
6. 推荐按 stage-by-stage 推进 window，不推荐“单 head 跑完整 full-flow 再切下一个 head”。

## 2. GM Workspace 与 Flag 协议

A2/A3 上不要假设存在可用的物理 L0C 到 UB 直通路径，除非当前 Catlass header/example 和芯片文档明确证明。Cube/Vector 跨域数据默认按以下路径设计：

```text
producer pipe
  -> GM workspace 稳定 slot
  -> CrossCoreFlag ready
  -> consumer pipe 读取 slot
```

当 producer/consumer 的窗口深度固定且 window 深度 <= 2 时，可以使用小型 4-slot ring：

```text
window 0: slot 0 / slot 1
window 1: slot 2 / slot 3
window 2: slot 0 / slot 1
...
windowStartSlot = (windowIdx & 1) * 2
```

上述公式只适用于双 window ping/pong。若 window 深度 > 2，必须重新设计 slot 数、credit/free 或 reverse flag，不能继续复用该公式，否则 producer 会覆盖 consumer 尚未读取的 slot。

必须区分以下概念：

| 概念 | 作用域 | 复用规则 |
|---|---|---|
| GM workspace slot | 跨 stage、跨 pipe 数据 | 按 `(chunk, hv/hk)` 和 window 稳定映射 |
| UB ping/pong | Vector 本地 copy/compute/output 缓冲 | 只在本地 vector 例程内复用 |
| L1 scratch | 短生命周期 Cube operand tile | 当前 matmul 消费 |
| L1 resident | 跨 stage 复用的 Cube 输入 tile | 生命周期显式管理，不能和 scratch 混用 |

## 3. L1 Resident 策略

L1 resident 只用于跨 stage 或多个 matmul 重复使用的块状数据。常见候选包括 `K`、`A`、`DW`、`DU`、`Q/V/WY`，具体取决于算子。

常见符号含义：

| 符号 | 常见来源 | 含义 |
|---|---|---|
| `K` | GDN/KDA forward/backward | key 输入或 K-side cache 输入 |
| `A` | KDA/GDN backward stage | attention 或 gate/mask 后的中间矩阵，常被 `A.T @ X`、`X @ A` 复用 |
| `DW` / `DU` | delta rule / gated delta rule backward | backward 路径中的权重/状态梯度中间量 |
| `Q` / `V` | attention family | query/value 输入块 |
| `WY` | GDN/delta rule 同族实现 | 与 value/state 更新相关的投影或中间块，具体含义以 primary reference 和用户 contract 为准 |

retention、RWKV 或其他 state recurrence 算子应按自身数学公式替换上表符号，不得静默继承 GDN/KDA 中间量名称。

规则：

- resident 区和 scratch 区在地址布局上分开，并在 DESIGN 中写明地址划分。
- 每个 `(chunk, head/window)` 只在消费者前加载一次 resident。
- 不要通过延长 scratch 生命周期来冒充 resident。
- L1 容量不足时，允许重算或重新搬运，不要无界扩大 resident。
- DESIGN 里记录字节预算：resident bytes + scratch bytes <= 可用 L1。

推荐布局示例：

```text
L1 base
  +------------------------------+
  | resident region              |  # 跨 stage/window 复用，按 512B 或硬件要求对齐
  |  K/A/DW/DU/Q/V/WY candidates |
  +------------------------------+  resident_end = AlignUp(base + resident_bytes)
  | scratch region               |  # 当前 matmul / copy / compute 临时块
  |  L1 operand tile buffers      |
  +------------------------------+  scratch_end <= base + available_l1_bytes
```

伪代码：

```cpp
auto resident_base = l1_base;
auto resident_end = AlignUp(resident_base + resident_bytes, 512);
auto scratch_base = resident_end;
auto scratch_end = AlignUp(scratch_base + scratch_bytes, 512);
CHECK(scratch_end <= l1_base + available_l1_bytes);
```

若采用 scratch 从 L1 尾部向下分配，也必须写清 `resident_end <= scratch_base`、对齐粒度和可用 L1 字节数。

典型复用：

| 数据 | 复用原因 |
|---|---|
| `A` | 被 `A.T @ X`、`X @ A` 等多个 backward matmul 消费 |
| `K` | 被 KKT、Dkb/DK 或 HK cache 消费 |
| `DW/DU` | 在 stage0/stage1 backward 路径中复用 |

## 4. L0 与 Fixpipe 纪律

当 MTE1 到 L0 与 MMAD 可以重叠时，L0A/L0B double buffer 有价值。每个 slot 的事件必须配对：

```text
Wait M_MTE1 -> LoadData L1->L0 -> Set MTE1_M
Wait MTE1_M -> MMAD -> Set M_MTE1
```

A2/A3 基线设计里 L0C 通常是单累加区。若 `V=256` 或大 reduce 维度需要 split accumulation：

- 所有 K/V subtile 累加到同一个 L0C。
- 内层 split 循环中禁止提前 fixpipe/write final output。
- 只有最后一个 subtile 触发 Fixpipe/cast/writeback。
- 用例必须覆盖会激活该路径的大维度。

## 5. Vector UB Buffering

Vector 侧如果把输入、计算、cast、输出混用同一个 buffer 概念，吞吐通常会崩。建议拆成独立组：

| UB 组 | 用途 |
|---|---|
| matrix input ping/pong | GM 到 UB 的矩阵行或 tile |
| scalar/vector resident | beta、g、exp(g)、mask、scale、小元数据 |
| fp32 compute buffer | 本地算术、reduce、broadcast |
| output ping/pong | cast/writeback 路径 |

`CopyInRows` 只负责把一个 GM 源搬入一个 input slot 并返回 slot。cast 和 compute 显式消费该 slot。输出 cast 和输出 copy 使用独立 output slot，避免 MTE3 覆盖 Vector 仍在使用的数据。

对 beta/g 类输入：

- 每个 active window/head slot 搬入完整 chunk。
- 后续 stage 需要 fp32 时先 cast 到 fp32 resident。
- `exp(g)` 或变换后的 gate 如果多 stage 复用，应放短生命周期 resident。

## 6. 物理布局优先于 Scatter

如果后续 stage 会重复消费转置矩阵，优先写一次 layout-converted workspace。不要在高频 Vector stage 中用 scatter/gather 模拟转置。

示例：

```text
Cube 写 DA6_T = DA6.T
Vector 连续读取 DA6_T[p, q]
Vector 计算 D[p, q] 和 triangular/gate mask
```

这通常优于生产 `DA6` 后让 Vector 每行散读或散写。

## 7. HK/HV/GQA Cache 作用域

当 `HV > HK` 时，只依赖 `(chunk, hk)` 的 K-side 中间量必须按 HK 缓存，不能按 HV 重算。

```text
hk = hv / (HV / HK)
slot_k_side = f(batch, chunk, hk)
```

规则：

- DESIGN 中写明 `HV % HK == 0` 约束。
- 除非有 overwrite 保护，否则不要跨 window 共享 HK cache。
- final Vector stage 必须能把每个 `hv` 映射到正确的 `hk` cache slot。
- shape 覆盖必须包含 `HK == HV` 和 `HV > HK`。

## 8. Task 分配

GVA/GQA backward 类 stage 常见的简单正确策略是按 `B * chunkNum` 分核，`HV` 在 core/window 内串行：

```text
for taskIdx = coreIdx; taskIdx < B * chunkNum; taskIdx += coreNum:
  for hvBase in range(0, HV, 2):
    run stages for the current 2-head window
```

这样可以避免 `dk` 等 HK 聚合输出上的跨 core atomic。若输出需要从多个 `hv` 聚合到 `hk`，使用确定性 ownership：

- group 内第一个 `hv` 写初值。
- 后续 `hv` 读已有值、累加、写回。
- 一旦引入跨 core ownership，必须改成明确的 reduction stage。

小 `B * chunkNum` 会导致核利用率不足，这是调度设计问题，不是单纯调 TileShape 能解决的问题。

## 9. TilingKey 与 Shape 覆盖

模板参数应覆盖会改变热路径或 buffer 预算的条件：

| 条件 | 常见取值 |
|---|---|
| dtype | fp16、bf16、支持时的 fp32 gate/accumulator |
| `V_DIM` / `K_DIM` | 64、128、256 |
| `CHUNK_SIZE` / `BT` | 64、128、partial tail |
| schedule mode | normal、small-task、split-K/split-V |
| head mapping | `HK == HV`、GQA/GVA |

shape 覆盖必须由这些类别推导。若保留历史固定 shape，必须注明它覆盖的分支或 bug 类型。

## 10. Checklist

- [ ] 每个中间张量都有 producer、consumer、pipe、内存层级和生命周期。
- [ ] Stage 边界只出现在真实数据依赖处。
- [ ] 跨 pipe 数据写完 GM 后才 set ready flag。
- [ ] consumer 在 stage 入口 wait，不在 row/tile 内 wait。
- [ ] `(chunk, hv/hk)` 的 workspace slot 公式跨 stage 稳定。
- [ ] GM slot 和 UB ping/pong 没有混用概念。
- [ ] L1 resident 与 scratch 生命周期分离。
- [ ] L0A/L0B 的 `SetFlag`/`WaitFlag` 成对。
- [ ] split accumulation 只在末 subtile 后做 L0C fixpipe。
- [ ] 覆盖 `V=256` 和 partial chunk。
- [ ] GQA/GVA 的 K-side cache 按 `hk` 作用域管理。
- [ ] 改变 window 深度时，重新设计 flag/credit 协议。
- [ ] 性能报告区分 scheduling underfill 与 tile 级瓶颈。
