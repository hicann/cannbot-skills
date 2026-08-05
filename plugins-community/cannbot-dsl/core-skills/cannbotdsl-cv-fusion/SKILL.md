---
name: cannbotdsl-cv-fusion
description: "在 CANNBotDSL 中设计 Cube-Vector（CV）融合算子的顶层架构时使用。CV 融合算子 = 既有矩阵乘累加（必上 Cube）又有 vec 后处理（必上 Vec），因而必然跨 AIC/AIV 核。这类算子不是单跳 Cube→Vec，而常是多级交替链（CV / CVC / CVCV / VCVCV，含跨迭代反馈），跨核 handoff 拓扑、全局 sync 预算、核间分工用错会编译期报错（sync_id 耗尽）或跨核流水失效。当需要：判定算子是不是 CV 融合并分流原型、把算子建成 stage-graph（阶段节点 + 边表）、核算全局跨核 sync 预算（CrossCore ≤8 slot/func）与双核 buf_id 预算、决定 handoff 拓扑与 depth、决定融成一个 kernel 还是拆分、或规划 N 段稳态重叠时触发。含跨核铁律权威表述（depth≤8 / CUBE +16 双发 / prime-drain 方向）。Triggers: cv 融合, Cube-Vector 融合, 混合算子, CVCV, cvcv, cvc, vcvcv, MIX_AIC_1_2, 跨核 handoff, sync 预算, sync_id 耗尽, stage graph, 核间分工, 融合拆分。注意：VF 折叠机制归 cannbotdsl-vf-fusion；Channel 4 相协议机制归 cannbotdsl-channel；Cube 核内 4-PIPE 归 cannbotdsl-cube-pipeline；Vec 核内 3-PIPE 归 cannbotdsl-vec-pipeline；macro 级深流水（preload_num≥3）实现归 cannbotdsl-channel 与 cannbotdsl-perf-optimize；完整 FA 变体移植归 cannbotdsl-flash-attention；通用四轮法/字节预算/buf_id 表归 cannbotdsl-op-design（与本 skill 并行调用）。Architect sub-agent 在 Stage 2 调用。"
---

# cannbotdsl-cv-fusion

CANNBotDSL Cube-Vector 融合算子的**架构决策层**。Architect 在 Stage 2 判定算子为 CV 融合（Attention / Composite / 多阶段融合）后、切 tiling 与字节预算之前,先定"谁在哪个核上算、多级链怎么交接、跨核预算装不装得下"。

**只产出架构决策(进 DESIGN.md),不写 kernel 代码,也不下沉到核内流水/深流水的实现细节(那些见 §注意)。** 与 `../cannbotdsl-op-design/SKILL.md` 并行调用:op-design 是通用四轮法 + 字节/buf_id 预算,本 skill 是 CV 专属的跨核架构前置。

## 使用前提

- 先用 `../cannbotdsl-op-design/SKILL.md` §1 归类;归到 **Attention** 或 **Composite (C/V Mix)** 才进本 skill。
- 跨核机制(Channel 4 相协议、SameCore/CrossCore API)先读 `../cannbotdsl-channel/SKILL.md`;本 skill 只讲**架构约束**,不重复原语用法。
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

**边表(预算落在最后一列)**:

| 边 | 类型 | 载体 | mem_loc | depth | 吃 sync 预算? |
|----|------|------|:-------:|:-----:|:-------------:|
| C1→V1 | 前向跨核 | qk_ub | UB | 2 | ✓ 2 |
| V1→C2 | 前向跨核 | p_l1 | L1 | 3 | ✓ 3 |
| C2→V2 | 前向跨核 | pv_ub | UB | 2 | ✓ 2 |
| V1⟲V1 | 同核反馈 | m/l 三缓冲 | UB | 3 | ✗(仅占 UB) |
| V2⟲V2 | 同核反馈 | res_o | UB | 1 | ✗(仅占 UB) |
| | | | | **Σ 跨核** | **7 ≤ 8** |

边表最后一列一加 = 7,对上 channel-first 版三个 `CrossCore` Channel 的 depth 之和(qk_ub depth=2 + pv_ub depth=2 + p_l1 depth=3 = 7）——**每个 cross-core slot 花一个 sync_id,声明里 depth 直接可数**。**"吃预算?"这一列把 §2 分工、§4.1 预算、§4.4 拆分缝在一张表里**——改分工(增删一个阶段)时预算变化立刻在表尾可见。

## 2. 核间分工(AIC : AIV)—— 原则:预算允许下阶段数最少

- **拓扑**:Atlas A2 是 `MIX_AIC_1_2`(1 AIC : 2 AIV)。切分轴优选 **split-M**:M 维切两半分给两个 AIV,单个 AIC 服务两个 vec 核。
- **分工判据**:有 matmul 累加必留 Cube;逐元素/归约/mask/select 下 Vec。
- **灰色地带原则(联动 §4.1)**:`cast`/`muls`/`scale` 这类轻算子,能在 **cube 的 fixpipe drain 时顺带做**的,就别单独开一个 V 阶段。

  > 每多一次 C↔V 往返 = 边表里多两条前向跨核边 = 多烧 sync 预算 + 多两次同步延迟。所以 §2 的核心不是"哪步能放 vec",而是"**在预算和精度允许下,把阶段数压到最少**"。

**因果链**:分工决定阶段数 → 阶段数决定跨核跳数 → 跳数决定 §4.1 预算。三者不是独立小节。

## 3. 单跳 handoff 契约(双向,原子单位)

多级链 = 单跳 handoff × N 次。每一跳只有两种方向,契约固定:

| 方向 | 载体典型 | 生产者 | 消费者 | 说明 |
|------|----------|----------------|--------------|------|
| **Cube→Vec** | cv_ub / qk_ub(fixpipe drain L0C→UB,fp32) | Cube 侧 fixpipe 通道 arrive | Vec 侧 V 通道 wait | vec 收 cube 算完的 tile |
| **Vec→Cube** | p_l1(vec 写 P→L1 供 PV) | Vec 侧 MTE3 通道 arrive | Cube 侧 MTE1 通道 wait | cube 收 vec 算完的 tile |

两侧共享同一 `buf_id` 建立屏障(channel-first 下由框架自动合成 arrive/wait,源码不出现原语)。`depth≥2` 解耦:生产者写 slot b 时消费者消费 slot a、互不等。**方向写反不报编译错但读到陈旧数据**——见 §5 铁律 4。Channel 选型见 `../cannbotdsl-op-design/SKILL.md` §6;跨核 >8 slot 逼退拆分见 §4.1。

## 4. 多级链专章(本 skill 的核心)

单跳契约(§3)组合成长链后,冒出四个单跳看不见的问题。**前两个是编译期硬墙**(能不能跑),不是性能问题。

### 4.1 全局跨核 sync 预算(第一硬墙)

**关键事实**:跨核 sync 预算是**全局 per-kernel** 的,不是 per-channel。`alloc_sync_ids` 把 `_sync_id` **累加到所有 channel 上**,上限 `MAX_SYNC_ID=16`,注释明写"at most 8 cross-core slots/func"。

- Channel 路径:`Σ(每个 CrossCore channel 的 depth) ≤ 8`,超了编译期 `raise ValueError`。
- 手动路径:裸 `sync_intra` 直接占 `[0,31]`,其中 `[16,31]` 被 CUBE 的 `+16` 双发(§5 铁律 2)保留 → 有效 base id ≈ 13。

**免费规则(FA 结构给出)**:**只有前向跨核 handoff 吃预算;同核跨迭代反馈状态(softmax m/l、输出累加器)只占 UB,不吃 sync 预算。** 所以 CVCV-*with*-feedback 在这条硬墙上并不比裸 CVCV 贵——FA 的"online"难点(尾部 V 校正)不威胁 sync 墙。

**核算方法**:把 §1 边表"吃预算?"列求和。FA = 7,离墙(8/13)还有余量,可再加约一跳往返;VCVCV 若跳数把和顶过墙 → §4.4 拆分。

### 4.2 双核 buf_id 预算

cube / vec 各自独立计数(`_cube_buf_id` / `_vec_buf_id`),各自逼近 `MAX_BUF_ID`。长链 buffer 多时两侧分别核算,回填 `../cannbotdsl-op-design/SKILL.md` §2.1 的 buf_id 表。

### 4.3 中间量跨跳存活表

stage1 的产物若到 stage3 才消费,必须跨 stage2 一直占着 slot——单跳模型里 buffer 用完即释放,长链里不行。给一张"中间量 × 活到第几跳 × 占哪个 slot"表,防止把还活着的 slot 复用掉。

### 4.4 融 vs 拆决策(阈值量化)

`Σ(前向跨核边 × depth)` 超过 §4.1 预算(Channel 8 / 手动 ~13),或双核 buf_id 超限 → **把长链拆成两个 kernel**,中间量落 GM 再起第二个 kernel。取舍:融合省一次 GM 往返但吃片上预算;拆分放开预算但多一次 HBM 读写。FA 的 7 尚有余量不必拆;每加一对 C→V + V→C(≈+4)逼近墙,加两对就该拆。

### 4.5 N 段稳态重叠(决策,非实现)

- **重叠可行性**:相邻 C 段与 V 段能否跨 tile 错位(稳态时 cube 算 tile n+1 的 C1,vec 同时算 tile n 的 V1)。
- **气泡诊断**:`MIX_AIC_1_2` 下若不重叠,cube 在 V 段空转、两个 AIV 在 C 段空转,利用率上限 ≈ `max(ΣC_time, ΣV_time) / (ΣC + ΣV)`。这个数决定值不值得上深流水。
- **下沉信号**:需要 `preload_num ≥ 3` 的深度错位(如 FA 的 delayed-PV)→ 下沉 `../cannbotdsl-channel/SKILL.md` 与 `../cannbotdsl-perf-optimize/SKILL.md`，以 `Channel(..., depth=N)` 实现 FIFO 槽生命期并设计 warmup/steady/drain。`preload_num=2` 的常规 DB 重叠留在本 skill + cube/vec-pipeline。若调度必须随机访问 `macro_idx % N` 槽且 Channel FIFO 无法表达，当前前端不支持。**本 skill 只画"哪几段重叠、深度定几、利用率上限多少",不伪造槽索引 API。**

## 5. 跨核铁律(canonical home,其他 skill 引用这里)

CV 融合跨核交接必守的九条,权威表述在此;`vf-fusion`/`channel`/`cube-pipeline`/`vec-pipeline`/`flash-attention`/`perf-optimize` 引用本节而非各自复述。

1. **CrossCore depth ≤ 8**(`if is_cross_core and depth > 8: raise ValueError`)。根因:16 base sync_id,CUBE 侧 `+16` 双发把 `[16,31]` 占掉(见铁律 2),每个 cross-core slot 用 2 个硬件 counter → 至多 8 slot/func。这是**全局 per-func 预算**(§4.1),不是 per-channel。

2. **CUBE 侧 sync `+16` 双发**:cross-core 握手时 CUBE 核对每个 arrive/wait 同时发 `id` 和 `id+16`(`id+16` routes to VECCORE1 / AIV1),VEC 侧单发。channel-first 下 `+16` 由框架自动合成、源码不可见(所以数预算靠 §4.1 的 CrossCore depth 求和)。

3. **跨核同步由框架自动合成**:channel-first 下用户不写原语,框架按 Channel 的 Write/Read 操作数 + 数据依赖自动选对 PIPE、合成 arrive/wait。典型:vec 写 p_l1 是 MTE3 写、cube 读是 MTE1 读 → 框架在 vec 端合成 MTE3 arrive、cube 端合成 MTE1 wait;想等 fixpipe 产出的 UB 再 `mem_copy(gm,ub)` dump,因 mem_copy 跑在 MTE3,要等 MTE3 通道——不是 FIXPIPE、也不是 V。框架推不出这些场景(见 `cannbotdsl-channel` §2 边界)需退回显式 acquire/commit/wait/release。

4. **Channel prime/drain 方向**:入口 prime **必须 consumer 端发起 ×slot**,出口 drain **必须 producer 端发起 ×slot**。写反 → 死锁。channel-first 下 prime/drain 由框架合成。(注:此 prime/drain 指 Channel 入口/出口,与"FIXPIPE 把 L0C drain 到 UB"是不同概念,勿混。)

5. **Cube→Vec 的 UB handoff 必须*同时*满足两个条件**：(a) UB Channel 声明 `kind=ChannelKind.CrossCore`；(b) fixpipe engine 带 `dual_dst_ctl=1`。**少任一个都不报错**，但 AIV 读到的是**自己那份从未被写过的 UB** → 输出全 0。MIX 模式下 cube 与 vec 的 UB 是物理分开的两份，`dual_dst_ctl=1` 才让 fixpipe 同时写两侧。这是**静默失败**，编译与执行均无异常，唯一症状是结果全 0 —— 排查"跨核拿到全 0"时先查这两项。

6. **CrossCore channel 无条件 split-M**：编译器把**每个 AIV 的视图收窄到 `M/2` 行**，并**拒绝**声明成全高的 vec 侧 buffer（报 `'cannir.reduce_max' op dst shape mismatch at dim 0: expected 64, got 128`）。因此所有纯 vec 侧 buffer（reduce 输出、广播中转、cast 目标）都必须按 `M/SUBBLOCKS` 声明。**这不是可选优化，是必须遵守的形状约束**。副作用是 vec 侧 UB 占用直接砍半——核预算时别按全高算。

7. **per-core 存储 `tile_view(gm, ..., (subblock_id, 0))` 的正确性来自"数据源是 CrossCore channel"**：源是 CrossCore channel 时，两个 AIV 各自写对自己的半区；源是**普通 UB buffer** 时，**只有 half0 被写、half1 从未被触碰**（实测：全高 tile 的后半区哨兵值 100% 留存）。跨核 channel 携带 per-core 的 row-range 绑定，普通 UB tile 没有。**同一个 `tile_view(sub,0)` 写法在两种上下文下行为完全不同** —— 见到"输出只有上半对、下半是垃圾/未写"时，查数据源是不是 channel。

8. **CrossCore channel 必须由 cube 侧生产者填充；直接从 GM 灌会真机 fault**：`mem_copy(crosscore_ch, gm_tile)` 编译通过，但运行时 `device error type 3, error code 507035`。原因是跨核 channel 的 4 相协议由框架按"cube 产 / vec 消"的形态合成，没有 cube 侧 arrive 时 vec 侧的 wait 永远等不到对端。**写 probe 时尤其容易踩**：想造一个"已知输入"直接喂给 vec 段做数值验证，就会不自觉地把 CrossCore channel 当普通 buffer 用。**规避**：probe 里用**普通 channel**（只验 AIV0 那一半，对纯算术检查足够）；真 kernel 里 CrossCore channel 的上游必须是 `mem_copy(ch, l0c, engine=fixpipe)`。

9. **同一 kernel 里并列多条 CrossCore 边、但只有部分有真实 cube 生产者 → aivec timeout（507014）**：`Σdepth ≤ 8` 只约束**数量**，不保证**每条边都能被框架正确配对**。实测：一个诊断 kernel 里声明 3 个 CrossCore channel（1 个接 fixpipe、2 个只被 vec 侧读写用于 dump），编译通过但真机 `aicore timeout, retCode=0x25`、`aivec error exception, core id 1` —— vec 侧在等一个永远不会到来的 arrive。**这是铁律 8 的推论**。**实践含义**：想同时 dump 多个中间量做对比时，**只保留真实的那条跨核边，其余用普通 channel**。

> 铁律 5/6/7/8/9 都是**只在真机或编译期才暴露**的约束，且 5 与 7 是静默失败（无报错、结果错），8/9 是真机 fault（507035 / 507014）。设计跨核 handoff 时逐条对照，比事后调试便宜得多。

## 6. 真实蓝本索引

> CVCV 之外更长的链(VCVCV 等)当前仓库尚无可跑蓝本,§4 相应部分是**约束驱动 + 前瞻**——照 §4.1 预算核算决定融/拆,不套现成蓝本。

## 注意(反触发 / 下沉边界)

- VF(vector-fold)折叠**机制** → `../cannbotdsl-vf-fusion/SKILL.md`;Channel 4 相协议**机制**(acquire/commit/wait/release、SameCore/CrossCore API)→ `../cannbotdsl-channel/SKILL.md`。
- Cube 核内 4-PIPE(MTE2/MTE1/M/FIXPIPE)重叠 → `../cannbotdsl-cube-pipeline/SKILL.md`。
- Vec 核内 3-PIPE(MTE2/V/MTE3)重叠 + vf 折叠 → `../cannbotdsl-vec-pipeline/SKILL.md`。
- macro 级深流水(`preload_num≥3`)的 warmup/steady/drain **实现** → `../cannbotdsl-channel/SKILL.md` + `../cannbotdsl-perf-optimize/SKILL.md`；Channel FIFO 不能表达的旧手动多槽模型标记为不支持。
- 本 skill 的 stage-graph 定好后，把它**落成三层类骨架 + Channel 归属 + 多 stage 派发循环** → `../cannbotdsl-kernel-structure/SKILL.md`（本 skill 的**下游**：Developer 在 Stage 3 骨架步调用）。
- 完整 FA 变体移植/mxfp8 → `../cannbotdsl-flash-attention/SKILL.md`。
- 通用四轮设计法、字节预算、buf_id 表、同步方案选型(Channel vs 手动)→ `../cannbotdsl-op-design/SKILL.md`(与本 skill 并行调用)。
