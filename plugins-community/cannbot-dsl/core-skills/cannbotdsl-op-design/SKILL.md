---
name: cannbotdsl-op-design
description: "在 CANNBotDSL 中实现新算子前做系统性设计时使用。用户要写一个新算子、或需要在编码前规划 API 映射/Buffer 层次/Tiling 策略/流水方案时触发。提供算子分类路由（Elementwise/Reduction/Matmul-based/Attention/Sort-Select/Composite）、4 轮迭代设计法（计算图+dtype 路由 → Buffer 预算+Tiling → 流水方案+伪代码 → 交叉验证）、Buffer 预算计算器（UB 256KB/L1 512KB/L0A/B 64KB/L0C 128KB 硬限制）、TILING.md+DESIGN.md 文档模板、验证策略决策（简单/中等/复杂）。设计前先查已知框架限制避免设计出无法实现的方案。Triggers: cannbotdsl 算子设计, DESIGN.md, TILING.md, Buffer 预算, 算子分类, 4 轮设计, 流水编排, double buffer, 核内流水, 验证策略。Architect sub-agent 在 Stage 2 调用。"
---

# cannbotdsl-op-design

CANNBotDSL 算子设计方法论。Architect sub-agent 在 Stage 2 使用；输出 `TILING.md` + `DESIGN.md` + 验证策略决策，交给 Developer 在 Stage 3 实现。

**只产出设计文档，不写 kernel 代码。**

## 使用前提

- 设计前先用 `cannbotdsl-api-reference` 查已知框架限制，避免设计出无法实现的方案。
- 涉及 Buffer 层次先读 `../cannbotdsl-tiling-design/SKILL.md`；涉及 VF 先读 `../cannbotdsl-vf-fusion/SKILL.md`。
- **真实 API 以源码为准**：真实 buffer 分配 API 与容量上限以源码为准。

## 1. 算子分类路由

拿到需求后先归类，类别决定硬件路径和后续 skill：

| 类别 | 主导原语 | Buffer 层级 | 需要 Cube? |
|------|----------|-------------|:---:|
| **Elementwise** | `add/sub/mul/div/exp/cast/muls`（结构化 vec）或 raw `vadd/vmul/vcast/vexp_sub` | 仅 UB | 否 |
| **Reduction** | `reduce_max/reduce_sum(dst, src, axis=)`；跨 chunk 用 `vmax/vadd` 在 lane0 合并 | 仅 UB | 否 |
| **Matmul-based** | `matmul(c_l0c, a_l0a, b_l0b, init=...)` + `make_copy_engine(format_transform="nd2nz")` + fixpipe engine | L1→L0A/L0B→L0C→GM | **是** |
| **Attention** | Cube(QK^T, PV) + Vec(online softmax) 混合 + 跨核 sync | UB(vec) + L1/L0(cube) | 是（混合） |
| **Sort-Select** | topk/argmax 类，随框架能力而定，先查 `cannbotdsl-api-reference` 已知限制 | 通常 UB | 否 |
| **Composite (C/V Mix)** | Cube pipeline + vec tail（fixpipe→cv_ub→out_ub），跨核 token 传递 | UB + L1 + L0* + 跨核 | 是 |

**核心路由信号**：只要有矩阵乘累加就必须走 Cube（L0A×L0B→L0C 的 `matmul`/mmad）；其余纯逐元素/归约留在 UB 上用 vec/vf。

## 2. Buffer 预算硬限制

容量上限（"容量是硬上限，没有自动管理"）：

```
GM   HBM, GB 级
L1   ~512 KB  共享 scratchpad
L0A  ~64 KB   矩阵左乘缓冲
L0B  ~64 KB   矩阵右乘缓冲
L0C  ~128 KB  矩阵累加结果
UB   ~256 KB  vector 工作区
```

> **L0C 实测为 256 KB**（`_onchip.py:_MEM_LOC_CAPACITY_BYTES`），上表的 128 KB 偏保守。

Buffer 无同步语义。

### 2.0 L0A/L0B 紧张时：显式别名是**下策**，先确认真的装不下

attention 类算子有两个串行 matmul（QK 与 PV），左操作数形状不同（`(BM,D)` 与 `(BM,BN)`）。各自独立分配时，D=256 需 64 KB + 32 KB > 64 KB 上限。一个自然想法是让它们共用同一块 L0A：

```python
a_q = Channel(MemLoc.L0A, shape=(BM, D),  dtype=dtypes.float16, depth=1, addr=0)
a_p = Channel(MemLoc.L0A, shape=(BM, BN), dtype=dtypes.float16, depth=1, addr=0)  # 同址
```

显式 `addr=` 让别名成为**有意声明**而非分配顺序的偶然（框架 docstring 明确认可："Explicit addresses may intentionally alias an earlier allocation"）。

> ⚠ **但"框架认可"只到地址层为止：别名没有任何同步保护。先读完本节再决定用不用。**
>
> 两个别名 channel 的地址重叠对同步层完全不可见，没有任何 pass 会因为二者同址而串行化它们。
>
> **跨迭代竞态**（真实失效路径，不是理论担忧）：
> ```
> MTE1: l0a_q(n) l0b_k(n) l0a_p(n) l0b_v(n) │ l0a_q(n+1) ...
> M   : QK(n)                       PV(n)   │ QK(n+1)   ...
> ```
> `MTE1(l0a_p)(n)` 阻塞在 P 的跨核到达上；一旦 P 到达，`M(PV)(n)`（lhs 刚就绪）与 `MTE1(l0a_q)(n+1)`（唯一门禁是 `l0a_q` 自己的 WAR 锁，早在 `QK(n)` 后就释放）**同时可发射，且写同一批字节**。
>
> **关键**：自证第 1 条"活跃区间不重叠"若只看**单次迭代内**的数据依赖，会得出安全的错误结论 —— 而 `k_l1`/`v_l1` 取 `depth=2` 的全部意义就是让**相邻迭代重叠**，重叠一生效别名就破。**别名操作数自己 `depth=1` 不构成保护。**
>
> **实测**：去掉数据依赖后别名立刻静默算错（QK max_abs **78.45**）；四通道分开则 1.5e-5 / 9.5e-6 正常。编译无告警、运行无 fault。

**决策顺序**：

1. **先算不别名装不装得下** —— 四个独立通道 `l0a_q (BM,D)` + `l0a_p (BM,BN)`、`l0b_k (BN,D)` + `l0b_v (D,BN)`，逐个 (mode, BM, BN, D) 组合核对。实测某 GQA 全量配置**最坏恰好 100% 用满 64 KB、零溢出** —— 当初逼出别名的是自设的「L0 ≤ 半容量」余量规则，不是硬件。**别为一条自设的余量规则去换一个静默竞态。**
2. 真的装不下 → 先调 tile（降 BM/BN）。
3. 仍不行才考虑别名，且**必须自证三条**（写进 DESIGN.md）：
   1. 两个操作数的**活跃区间确实不重叠**，且论证必须覆盖**跨迭代**（不能只看单次迭代内的数据依赖）；
   2. 别名块大小取 `max(两者字节)`，且仍在容量内；
   3. **无 double buffer** —— 一旦某操作数 `depth≥2`，多级缓冲轮转会打破不重叠假设。
4. 用了别名就**记录一条硬约束**：后续给 L0 加 DB 必须先撤销别名。

**代价对照**：别名把 L0 占用减半，代价是一个同步层看不见、只在流水重叠时才暴露的静默竞态。绝大多数情况下，**不别名 + 合适的 tile 是更好的交易**。

**真实分配 API**：

```python
# 单块、无同步语义的片上临时存储
Buffer(mem_loc, shape, dtype, *, addr=None, stride=None,
       data_format=None, n1_pad=0)

# depth-N / double buffer / 同步存储
Channel(mem_loc, shape, dtype, *, depth,
        addr=None, stride=None, data_format=None, n1_pad=0)
```

> Buffer 与 Channel 共用地址 arena；`addr=None` 自动 bump 分配，显式 `addr=` 可声明有意 alias。Buffer 始终只有一块，无同步语义。旧 NBuffer、`make_*`、`make_buf/make_nbuf` 已移除；无法由 Channel 表达的旧手动多级方案当前不受支持，不要伪造替代 API。

**字节计算**：

```
bytes_per_buffer = prod(shape) * (dtype.width // 8)
total_bytes    = num_buffers * aligned(bytes_per_buffer, align)
```

dtype 宽度：`dtypes.float16/bfloat16 = 16`，`dtypes.float32/int32 = 32`，`dtypes.int64 = 64` bit。Buffer 占一块；Channel 总占用按 `depth` 级核算。

**对齐要求**：vector UB 访问需 32B 对齐 —— per-row 标量要 pad 到 32B stride（fp16 → 16 elems/row，fp32 → 8 elems/row，见 §5 的 `SM_MAX_STRIDE`）。矩阵 tile 的 NZ 物理形状：`m0=16, n0=32//dtype_bytes`，logical (M,N) → 4D `(n1, m1, m0, n0)`。

## 3. 4 轮迭代设计法

| 轮次 | 步骤 | 产出 | 锁定的决策 |
|------|------|------|-----------|
| **R1 计算图 + dtype 路由** | 画 op DAG，标注每条边 dtype 与精度提升点（如 softmax：fp16 in → fp32 做 exp/reduce → cast 出）；判断是否含 matmul 定硬件路径 | 带 dtype 标注的计算图 + 类别归属（§1） | 硬件路径（纯 Vec/纯 Cube/C-V Mix）、每个中间量的存储 dtype |
| **R2 Buffer 预算 + Tiling** | 选 tile 尺寸；枚举所有 on-chip storage，用 §2 公式算字节，逐层对硬限制加和 | storage 清单（Buffer/Channel、mem_loc/shape/dtype/depth/字节）+ 各层占用合计 | tile 大小、Channel depth、对齐 |
| **R3 流水方案 + 伪代码** | 定 preload 深度与 prologue/main/epilogue；写 channel-first `mem_copy/matmul/muls` + 计算 op 伪代码，标 PIPE（MTE2 load / MTE1 L1→L0 / M mmad / FIXPIPE 输出 / V 向量） | kernel 体伪代码（含 pipe 标注与 depth 决策） | 同步序列、pipe 分配、cursor vs 显式 `[i%N]` 索引 |
| **R2.5 设计期 API 可行性检查** | 逐表确认设计中每个 buffer 的数据流方向、vec 实现路径、tile_view 维度是否在框架能力范围内 | 三张检查表（表 1/2/3）的逐条确认记录 | vec 实现路径（高层 API vs raw-VF）、Buffer 中转方案、tile_view 维度策略 |
| **R4 交叉验证** | 逐 tile 推演缓冲不冲突；核对生产/消费顺序与数据流一致（防死锁）；核对精度契约（golden 与 device fp16 round 对齐） | 缓冲冲突推演表 + 死锁检查 + golden 对齐说明 | 可编译无死锁、数值精度容差 |

### R2.5 设计期 API 可行性检查

> **关键原则：vec 计算优先使用 `vf(mode='raw')` 模式。** 在 CV 融合算子（含跨核 channel + 动态循环）中，高层结构化 vec API（`expand`/`muls`/`cast`/`reduce_max`/`reduce_sum` 等）的 VF 自动归组脆弱，raw-VF 是最可靠的 vec 实现路径。R3 伪代码应以 raw-VF 为基准编写 vec 段。

在 R2 Buffer 预算完成、R3 伪代码编写之前，必须逐表确认以下三项。任一未通过则需调整设计。代码示例见 `references/api-feasibility-checklist.md`。

**表 1 — 高层 vec API 可行性**

| 检查项 | 触发条件 | 失败现象 | 规避 |
|--------|---------|---------|------|
| `expand` op | 任何位置使用 | `vf-transform found no template for operation 'cannir.expand'` | raw-VF `vload` + `vdup_lane0` 替代广播 |
| 不同 shape vec op 相邻 | 如 `(64,128)` 和 `(64,1)` 的 `muls` 被 VF grouping 合并 | `vector elementwise operands require matching logical shapes` | 在不同 shape 的 vec op 之间插入 DMA 操作打断 grouping |
| 动态循环内跨核 channel | 动态 `range` + 跨核 channel 操作数 | Channel shape 变为 `(-1,N)`，VF grouping 失败 | vec 段全部用 `vf(mode='raw')` |
| 高层 API in-place channel | `muls(ch, ch, scalar)` 在 channel 无 producer 时 | `in-place use of a channel with no producer, buffer would never be filled` | pre-loop 首迭代在循环外建立 producer |

**表 2 — Channel 与 Buffer 数据流约束**

| 数据流方向 | 是否可行 | 正确中转方式 |
|-----------|:-------:|------------|
| GM → Buffer (`mem_copy(buffer, gm)`) | 可行 | 直接 `mem_copy` |
| Buffer → GM (`mem_copy(gm, buffer)`) | **不可行，输出全零** | `muls(channel, buffer, 1.0)` → `mem_copy(gm, channel)` |
| 跨核 channel → Buffer (`mem_copy(buffer, channel)`) | **不可行，输出全零** | `muls(buffer, channel, scalar)` |
| Buffer → channel (`mem_copy(channel, buffer)`) | 可行 | 直接 `mem_copy` |
| channel → channel (`muls(dst_ch, src_ch, scalar)`) | 可行 | 直接 `muls` |

**表 3 — tile_view 维度限制**

| 检查项 | 失败现象 | 规避 |
|--------|---------|------|
| 4D/3D `tile_view` 视图传播到 `matmul` | `typed-region writer has no registered access-shape rule` | host 侧 4D→2D 展平，kernel 内只用 2D `tile_view` |
| `idx2crd` + runtime 整数除法 | SSA 除法不被支持 | 分解为 `[H_kv, g, num_m]` 避免 `//` 运算 |

## 4. 验证策略决策（为 Stage 3 Step 3.1 做准备）

按复杂度分级，决定 Developer 是否需要模块分解验证：

- **简单**（纯 elementwise / 单归约，单 vf 段，UB-only，无跨核）→ 跳过模块验证，直接完整实现 + translate 断言 + 一个 NPU precision test。
- **中等**（online softmax 类：多 vf 段 + 跨 tile 状态 running max/sum，或纯 Cube tiled matmul）→ 验 **2–3 个模块**：(1) 单段 vf 数值正确，(2) 跨 tile 状态校正，(3) buffer layout/32B 对齐。translate + CPU + NPU 三重 test。
- **复杂**（C/V Mix、attention、动态 shape、多核尾核分发）→ 验 **4–6 个模块**：tiling/尾核分发、L1 缓冲推演、跨核 sync token、Cube 累加、vec tail、动态 shape（`Dim` + `TensorSpec` + staged `.compile()`，并覆盖约束与维度关系的缓存身份）。

## 5. 真实最小 kernel 骨架

### 5.0 默认范式：channel-first

**新算子默认这样写**：声明 `Channel`（只给 `depth`），直接把它当操作数传给 `mem_copy/matmul/muls/…`。

跨核 C/V mix 最小骨架：

```python
from cannbotdsl.channel import Channel
from cannbotdsl import dtypes
from cannbotdsl.math import matmul, muls
from cannbotdsl.tensor import tile_view, make_copy_engine, mem_copy
from cannbotdsl.typing.types import MemLoc

a_l1  = Channel(MemLoc.L1,  shape=(tile_m, tile_k), dtype=dtypes.float16, depth=2)  # depth≥2 → DB
l0a   = Channel(MemLoc.L0A, shape=(tile_m, tile_k), dtype=dtypes.float16, depth=2)
l0c   = Channel(MemLoc.L0C, shape=(tile_m, tile_n), dtype=dtypes.float32, depth=1)  # K-loop 同级累加
cv_ub = Channel(MemLoc.UB,  shape=(tile_vm, tile_n), dtype=dtypes.float32, depth=2)  # 跨核 handoff
out_ub = Channel(MemLoc.UB, shape=(tile_vm, tile_n), dtype=dtypes.float32, depth=2)
nd2nz   = make_copy_engine(format_transform="nd2nz", dtype=dtypes.float16, pad_value=0.0)
fixpipe = make_copy_engine(dtype=dtypes.float32, dual_dst_ctl=1)

mem_copy(a_l1, a_gm, engine=nd2nz)   # Channel 直接做 dst
mem_copy(l0a, a_l1)                   # 既做 dst 又做 src
matmul(l0c, l0a, l0b)                 # K-loop 累加器由框架识别（仅 matmul）
mem_copy(cv_ub, l0c, engine=fixpipe)  # 跨核 CUBE→VEC
muls(out_ub, cv_ub, scale)            # vec 侧原地事务
mem_copy(tile_view(out_gm, (tile_vm, tile_n), (subblock_idx, 0)), out_ub)
```

> `depth` 与软件流水结构（prologue/steady/epilogue、跨核反馈依赖的 lag-N）是算法决策，框架推不出，须显式写。

## 6. 同步方案选型（强制产出，进 DESIGN.md）

含跨核数据交接（Cube↔Vec）或深流水的算子，**必须显式做一次同步方案选型并写明理由**，不许默认继承基线而不记录。**channel-first 是唯一推荐路径**：

| 方案 | 机制 | 优点 | 代价 / 上限 |
|------|------|------|------------|
| **Channel (channel-first)** | depth-N 多级缓冲抽象，`Channel(mem_loc,...,depth=N)` 直接传给 `mem_copy`/`matmul`/… | 大幅减少手动 sync 记账、加深流水只改 `depth` | 跨核 Channel 过多时需拆分（见 `cannbotdsl-cv-fusion` §4.2） |

**决策信号**：

- **默认 channel-first**（含纯核内单生产单消费 DoubleBuffer）：一个 Channel + `depth=2` 就拿到 DB。
- 跨核 Cube↔Vec 多通道 handoff、preload_num ≥ 3 的宏流水、sync 通道数 ≥ 4 → channel-first 收益最大。
- **跨核 handoff 过多** → 拆 kernel（中间量落 GM），拆分决策见 `cannbotdsl-cv-fusion` §4.2。

**产出要求**：DESIGN.md 的同步通道表前，写一句选型结论 + 理由。关键是**决策被记录、理由可追溯**。示例：

- 选 **channel-first**："跨核通道 3 条、preload_num=2，无跨兄弟循环累加器 → channel-first，只显式声明 depth。"

## 7. 流水编排设计（强制产出，进 DESIGN.md）

Tiling 定"切多大"、Channel depth 定"多少级在途"，本节定"怎么让搬运与计算重叠"。
**double buffer 不是纯性能旋钮**：它同时改地址预算（`depth × bytes`）和多级同步。事后再加 ≈ 重做 §2 / §4，故必须在**设计阶段决策、初版实现即落地**，不能整体推给 Perf-Tune。

### 7.1 两类流水，两种时机

| 类别 | 内容 | 时机 |
|------|------|------|
| **常规 double buffer / 核内流水** | Cube 核内四 PIPE 重叠 + L0A/L0B/L1 tile DB；Vec 核内三 PIPE 重叠 + UB DB；Cube↔Vec 核间 `cv_ub` DB(depth≥2) | **设计阶段决策 + 初版实现即落地**（`../cannbotdsl-perf-optimize/SKILL.md` 自述"L0A/L0B DB 是常规手段"） |
| **macro 级深流水 preload** | preload_num≥3，cube QK / vec softmax / cube PV 三段跨 N 个 macro 错位并行 | 可留独立 Perf-Tune |

初版单缓冲（`depth=1`）**不是默认**，是需要逐条论证的例外（buffer 过大超预算、无相邻 tile 可重叠、强数据依赖串行）。

### 7.2 三类流水的重叠策略（逐条产出）

1. **Cube 核内**（Matmul / Composite 必产）：`MTE2(GM→L1)→MTE1(L1→L0)→M(mmad)→FIXPIPE(L0C→UB)` 四 PIPE 在相邻 tile / K-step 上重叠。L0A/L0B 上 `depth=2` 让"载入 k+1"与"mmad k"并行；K-loop 累加同一 L0C 缓冲区。详见 `../cannbotdsl-perf-optimize/SKILL.md`。
2. **Vec 核内**（含 Vec epilogue 必产）：`MTE2(GM→UB)→V(compute)→MTE3(UB→GM)` 三 PIPE 重叠；输入/输出 UB 上 `depth=2` 让"载入 tile n+1"、"算 tile n"、"存 tile n-1"并行。详见 `../cannbotdsl-perf-optimize/SKILL.md`。
3. **Cube↔Vec 核间**（C/V Mix / Attention 必产）：Cube fixpipe 写 `cv_ub[缓冲 b]` 的同时 Vec 消费 `cv_ub[缓冲 a]`，靠跨核 Channel `depth≥2` 解耦；`depth=1` 则 Cube 与 Vec 严格串行、跨核流水失效。

### 7.3 depth 决策回填（硬要求）

每个 on-chip buffer 给一行 depth 决策，并把结果**回填**：
- §2 Buffer 预算表：字节按 `depth × buffer_bytes` 重算，再对硬限制（L1 512KB / L0A/B 64KB / L0C 128KB / UB 256KB）校验。

| buffer | 建议 depth | 理由 |
|--------|:---------:|------|
| L0A / L0B tile | 2 | 载入与 mmad 重叠（常规 DB） |
| cv_ub (Cube→Vec) | 2 | 解耦跨核，Cube 不等 Vec |
| Vec 输入/输出 UB | 2 | 载入/计算/写回三段重叠 |
| L0C 累加器 | 1 | K-loop 同级累加，无 DB 收益 |
| 单发不复用的常驻量 | 1 | 无相邻 tile 重叠收益 |

### 7.4 与 Perf-Tune 的边界

常规 DB（7.1 上行）属**设计 + 初版实现**范围，不得整体推给 Perf-Tune。留给独立 Perf-Tune 的只有：macro 级深流水（preload_num≥3，`../cannbotdsl-perf-optimize/SKILL.md`）、tiling 尺寸重选、系统级（多核负载 / GM 带宽）优化。

## 输出模板

**TILING.md**：多级 tiling（GM/L1/L0/UB tile shape）、Buffer/Channel 预算表（每项字节 + 各层合计 ≤ 硬限制）、depth 配置、Tail block 策略、验证策略（简单/中等/复杂）。

**DESIGN.md**：架构概览、Kernel 类结构、Buffer 分配表、**同步方案选型结论 + 理由（§6）**、**流水编排（§7：Cube 核内 / Vec 核内 / Cube↔Vec 核间三类流水重叠策略 + 每 buffer 的 DB 深度决策，回填预算）**、同步通道表、VF 融合区域、多核 dispatch、伪代码（含 shape/dtype 标注）、AOT 动态 shape 方案（如适用）。

## 门禁

- 所有 mem_loc 的 Total bytes ≤ 硬限制才能输出 TILING.md。
- **含跨核 handoff / 深流水的算子已显式做同步方案选型（§6）并在 DESIGN.md 记录理由。**
- **流水编排设计已产出（§7）**：Cube 核内 / Vec 核内 / Cube↔Vec 核间三类流水各自的重叠策略（按算子类别必产项）+ 每个 buffer 的 DB 深度决策；depth 默认 2，选 `depth=1` 须逐条论证；深度已回填 §2 预算表。常规 double buffer 属设计+实现范围，**不得整体推给 Perf-Tune**，仅 macro 级深流水(preload_num≥3)可延后。
- 同步通道表完整列出所有 Channel 的生产/消费配对。
- VF 区域声明完整 outputs 列表。
- 必须输出验证策略决策。
- **R2.5 三张检查表（表 1/2/3）逐条确认通过后才可进入 R3。**
- 若 Stage 3 返回 `DESIGN_ERROR`，回到本阶段修改设计。

## 参考

- `references/single-kernel-fusion-lessons.md`（多 kernel/host loop/多阶段收敛为单 kernel 的经验：职责边界、buffer 生命周期、GM 中间量清理、split launch 验收口径）
- `references/api-feasibility-checklist.md`（R2.5 检查表的代码示例：raw-VF 替代 expand、Buffer 中转、4D tile_view 规避）
