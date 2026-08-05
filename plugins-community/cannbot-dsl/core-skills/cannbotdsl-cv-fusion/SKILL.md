---
name: cannbotdsl-cv-fusion
description: "在 CANNBotDSL 中设计 Cube-Vector（CV）融合算子的顶层架构时使用。CV 融合算子 = 既有矩阵乘累加（必上 Cube）又有 vec 后处理（必上 Vec），因而必然跨 AIC/AIV 核。这类算子不是单跳 Cube→Vec，而常是多级交替链（CV / CVC / CVCV / VCVCV，含跨迭代反馈），跨核 handoff 拓扑、核间分工用错会编译期报错或跨核流水失效。当需要：判定算子是不是 CV 融合并分流原型、把算子建成 stage-graph（阶段节点 + 边表）、决定 handoff 拓扑与 depth、决定融成一个 kernel 还是拆分、或规划 N 段稳态重叠时触发。含跨核铁律权威表述（prime-drain 方向 / split-M / dual_dst_ctl）。Triggers: cv 融合, Cube-Vector 融合, 混合算子, CVCV, cvcv, cvc, vcvcv, MIX_AIC_1_2, 跨核 handoff, stage graph, 核间分工, 融合拆分。注意：VF 折叠机制归 cannbotdsl-vf-fusion；Cube 核内 4-PIPE 归 cannbotdsl-cube-pipeline；Vec 核内 3-PIPE 归 cannbotdsl-vec-pipeline；macro 级深流水（preload_num≥3）实现归 cannbotdsl-perf-optimize；完整 FA 变体移植归 cannbotdsl-flash-attention；通用四轮法/字节预算归 cannbotdsl-op-design（与本 skill 并行调用）。Architect sub-agent 在 Stage 2 调用。"
---

# cannbotdsl-cv-fusion

CANNBotDSL Cube-Vector 融合算子的**架构决策层**。Architect 在 Stage 2 判定算子为 CV 融合（Attention / Composite / 多阶段融合）后、切 tiling 与字节预算之前,先定"谁在哪个核上算、多级链怎么交接、跨核预算装不装得下"。

**只产出架构决策(进 DESIGN.md),不写 kernel 代码,也不下沉到核内流水/深流水的实现细节(那些见 §注意)。** 与 `../cannbotdsl-op-design/SKILL.md` 并行调用:op-design 是通用四轮法 + 字节预算,本 skill 是 CV 专属的跨核架构前置。

## 使用前提

- 先用 `../cannbotdsl-op-design/SKILL.md` §1 归类;归到 **Attention** 或 **Composite (C/V Mix)** 才进本 skill。
- 源码为准。

## 1. 判定 + 把算子建成 stage-graph

**判定**:有矩阵乘累加(L0A×L0B→L0C 的 `matmul`/mmad)**且**有 vec 后处理(逐元素/归约/mask/select)→ 必然跨 AIC/AIV → 是 CV 融合。纯 Vec 或纯 Cube 不进本 skill。

**建模**:CV 融合算子不是"一个 C、一个 V",而是 **C / V 节点交替的链**,用链式速记描述拓扑:`CV`(cube QK + vec tail)、`CVC`、`CVCV`、`VCVCV`…。链上每一跳 C↔V 都是一次跨核 handoff。

三件套记法(进 DESIGN.md),**预算直接落在边表最后一列**:

**链式速记**(反馈边单独标注)——以 channel-first Flash Attention 为例,它是 **CVCV-with-feedback**:

```
      ┌──── 反馈: m/l 三缓冲 (V→V, 同核) ────┐
      │     ┌──── 反馈: res_o 累加 (V→V, 同核) ──┐
      ▼     ▼                                      │
  [C1 QK^T]═►[V1 softmax]═►[C2 P@V]═►[V2 update_o] ─┘ ─► (epilogue V: norm)
       └qk_ub┘      └p_l1┘      └pv_ub┘
```

`═►` = 前向跨核 handoff(**吃 sync 预算**);`─┘`(反馈) = 同核跨迭代(**不吃 sync 预算**)。对应源码:C1 `mm_qk` → V1 `softmax_first`/`softmax_rest` → C2 `mm_pv` → V2 `update_o`(`res_o = res_o*exp(Δmax) + P@V` 校正累加)→ epilogue `_finalize_div_vf`。**尾部那个 V2 就是"online"的本体**——去掉它就只是 attention,不是 flash attention。

**阶段节点表**:

| 阶段 | 核 | 主导 PIPE | 计算 | 产出 buffer |
|------|:--:|-----------|------|-------------|
| C1 | Cube | M / FIXPIPE | QK^T | qk_ub |
| V1 | Vec | V | mask + online softmax | p_l1 |
| C2 | Cube | M / FIXPIPE | P@V | pv_ub |
| V2 | Vec | V | res_o 校正累加 | res_o(驻留) |
| ep | Vec | V / MTE3 | 归一化 + cast | out |

**边表**:

| 边 | 类型 | 载体 | mem_loc | depth |
|----|------|------|:-------:|:-----:|
| C1→V1 | 前向跨核 | qk_ub | UB | 2 |
| V1→C2 | 前向跨核 | p_l1 | L1 | 3 |
| C2→V2 | 前向跨核 | pv_ub | UB | 2 |
| V1⟲V1 | 同核反馈 | m/l 三缓冲 | UB | 3 |
| V2⟲V2 | 同核反馈 | res_o | UB | 1 |

边表把 §2 分工、§4.2 拆分缝在一张表里——改分工(增删一个阶段)时跨核 handoff 变化立刻在表尾可见。

## 2. 核间分工(AIC : AIV)—— 原则:跨核边数最少

- **拓扑**:Atlas A2 是 `MIX_AIC_1_2`(1 AIC : 2 AIV)。切分轴优选 **split-M**:M 维切两半分给两个 AIV,单个 AIC 服务两个 vec 核。
- **分工判据**:有 matmul 累加必留 Cube;逐元素/归约/mask/select 下 Vec。
- **灰色地带原则**:`cast`/`muls`/`scale` 这类轻算子,能在 **cube 的 fixpipe drain 时顺带做**的,就别单独开一个 V 阶段。

  > 每多一次 C↔V 往返 = 边表里多两条前向跨核边 = 多两次同步延迟。所以 §2 的核心不是"哪步能放 vec",而是"**在精度允许下,把阶段数压到最少**"。

**因果链**:分工决定阶段数 → 阶段数决定跨核跳数。两者不是独立小节。

## 3. 单跳 handoff 契约(双向,原子单位)

多级链 = 单跳 handoff × N 次。每一跳只有两种方向,契约固定:

| 方向 | 载体典型 | 说明 |
|------|----------|------|
| **Cube→Vec** | cv_ub / qk_ub(fixpipe drain L0C→UB,fp32) | vec 收 cube 算完的 tile |
| **Vec→Cube** | p_l1(vec 写 P→L1 供 PV) | cube 收 vec 算完的 tile |

`depth≥2` 解耦:生产者写缓冲 b 时消费者消费缓冲 a、互不等。Channel 选型见 `../cannbotdsl-op-design/SKILL.md` §6;跨核 handoff 过多逼退拆分见 §4.2。

## 4. 多级链专章(本 skill 的核心)

单跳契约(§3)组合成长链后,冒出三个单跳看不见的问题:中间量要跨跳存活(§4.1)、跨核边多了可能要拆 kernel(§4.2)、多段能否稳态重叠(§4.3)。

### 4.1 中间量跨跳存活表

stage1 的产物若到 stage3 才消费,必须跨 stage2 一直占着缓冲区——单跳模型里 buffer 用完即释放,长链里不行。给一张"中间量 × 活到第几跳 × 占哪个缓冲区"表,防止把还活着的缓冲区复用掉。

### 4.2 融 vs 拆决策

跨核 handoff 过多或片上预算吃紧 → **把长链拆成两个 kernel**,中间量落 GM 再起第二个 kernel。取舍:融合省一次 GM 往返但吃片上预算;拆分放开预算但多一次 HBM 读写。每加一对 C→V + V→C 逼近预算墙,加两对就该拆。

### 4.3 N 段稳态重叠(决策,非实现)

- **重叠可行性**:相邻 C 段与 V 段能否跨 tile 错位(稳态时 cube 算 tile n+1 的 C1,vec 同时算 tile n 的 V1)。
- **气泡诊断**:`MIX_AIC_1_2` 下若不重叠,cube 在 V 段空转、两个 AIV 在 C 段空转,利用率上限 ≈ `max(ΣC_time, ΣV_time) / (ΣC + ΣV)`。这个数决定值不值得上深流水。
- **下沉信号**:需要 `preload_num ≥ 3` 的深度错位(如 FA 的 delayed-PV)→ 下沉 `../cannbotdsl-perf-optimize/SKILL.md`，以 `Channel(..., depth=N)` 实现生产/消费顺序驱动的生命期并设计 warmup/steady/drain。`preload_num=2` 的常规 DB 重叠留在本 skill + cube/vec-pipeline。若调度必须随机访问 `macro_idx % N` 索引且 Channel 生产/消费顺序无法表达，当前前端不支持。**本 skill 只画"哪几段重叠、深度定几、利用率上限多少",不伪造索引 API。**

## 5. 跨核铁律(canonical home,其他 skill 引用这里)

CV 融合跨核交接必守的规则,权威表述在此;`vf-fusion`/`cube-pipeline`/`vec-pipeline`/`flash-attention`/`perf-optimize` 引用本节而非各自复述。

1. **跨核 channel depth ≤ 8**：跨核 Channel 的全局 depth 总和有上限（编译期强制，超过 raise）。超过时降低 depth 或拆 kernel（§4.2）。

2. **跨核 channel 无条件 split-M**：编译器把**每个 AIV 的视图收窄到 `M/2` 行**，并**拒绝**声明成全高的 vec 侧 buffer（报 `'cannir.reduce_max' op dst shape mismatch at dim 0: expected 64, got 128`）。因此所有纯 vec 侧 buffer（reduce 输出、广播中转、cast 目标）都必须按 `M/SUBBLOCKS` 声明。**这不是可选优化，是必须遵守的形状约束**。副作用是 vec 侧 UB 占用直接砍半——核预算时别按全高算。

3. **per-core 存储 `tile_view(gm, ..., (subblock_id, 0))` 的正确性来自"数据源是跨核 channel"**：源是跨核 channel 时，两个 AIV 各自写对自己的半区；源是**普通 UB buffer** 时，**只有 half0 被写、half1 从未被触碰**（实测：全高 tile 的后半区哨兵值 100% 留存）。跨核 channel 携带 per-core 的 row-range 绑定，普通 UB tile 没有。**同一个 `tile_view(sub,0)` 写法在两种上下文下行为完全不同** —— 见到"输出只有上半对、下半是垃圾/未写"时，查数据源是不是 channel。

## 6. 真实蓝本索引

> CVCV 之外更长的链(VCVCV 等)当前仓库尚无可跑蓝本,§4 相应部分是**约束驱动 + 前瞻**——照 §4.2 拆分决策决定融/拆,不套现成蓝本。

## 注意(反触发 / 下沉边界)

- **跨核架构设计时就要考虑分发轴顺序**：causal 下 tile 代价沿 m-block 轴变化，`idx2crd` 轴排列不当会导致每核工作量恒定不均。这不是"性能调优阶段的事"——架构师画 stage-graph 时就该把分发轴顺序纳入设计。详见 `../cannbotdsl-perf-optimize/SKILL.md` 第 0 步。
- VF(vector-fold)折叠**机制** → `../cannbotdsl-vf-fusion/SKILL.md`。
- Cube 核内 4-PIPE(MTE2/MTE1/M/FIXPIPE)重叠 → `../cannbotdsl-cube-pipeline/SKILL.md`。
- Vec 核内 3-PIPE(MTE2/V/MTE3)重叠 + vf 折叠 → `../cannbotdsl-vec-pipeline/SKILL.md`。
- macro 级深流水(`preload_num≥3`)的 warmup/steady/drain **实现** → `../cannbotdsl-perf-optimize/SKILL.md`；Channel 生产/消费顺序不能表达的旧手动多级模型标记为不支持。
- 本 skill 的 stage-graph 定好后，把它**落成三层类骨架 + Channel 归属 + 多 stage 派发循环** → `../cannbotdsl-kernel-structure/SKILL.md`（本 skill 的**下游**：Developer 在 Stage 3 骨架步调用）。
- 完整 FA 变体移植/mxfp8 → `../cannbotdsl-flash-attention/SKILL.md`。
- 通用四轮设计法、字节预算、同步方案选型(Channel vs 手动)→ `../cannbotdsl-op-design/SKILL.md`(与本 skill 并行调用)。
