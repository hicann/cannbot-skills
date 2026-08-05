---
name: cannbotdsl-channel
description: "设计或调试 CANNBotDSL 的 Channel 跨迭代/跨核通信时使用。Channel 是 CANNBotDSL 的 ring-buffer 抽象，**唯一推荐写法是 channel-first**：把 Channel 直接传给 mem_copy/matmul/muls 等数据 op，acquire/commit/wait/release 4 相协议由框架自动合成，零手写；用户只声明 depth 和 kind。当需要：用 Channel 建 ring buffer / 跨迭代状态 / 跨核 handoff、选 SameCore 还是 CrossCore、核算 CrossCore depth≤8 预算、或判断该用 Channel 还是单块 Buffer 时触发。含 channel-first 规则与边界、构造 API、4 相协议语义表、Channel vs Buffer 选择指南、真实调用点。CrossCore 跨核铁律（+16 双发 / prime-drain 方向 / 全局 sync 预算）权威表述在 cannbotdsl-cv-fusion §5，本 skill 引用不复述。注意：channel-first 是同步去样板器，不是流水调度器——跨核环形依赖（如 FA）的软件流水 lag-N/prologue/epilogue 仍需手写。Triggers: cannbotdsl Channel, channel-first, ring buffer, acquire commit wait release, 4 相协议, 四相协议, SameCore, CrossCore, channel_rewind, depth 预算, 跨核 handoff, Channel vs Buffer。Developer Stage 3 实现/调试调用。"
---

# cannbotdsl-channel

CANNBotDSL Channel（ring-buffer 通信抽象）的用法与设计。Developer 在 Stage 3 用 Channel 实现跨迭代状态、ring buffer、跨核 handoff。**唯一推荐写法是 channel-first（§2）**：把 Channel 直接传给数据 op，4 相协议（acquire/commit/wait/release）由框架自动合成。

> **边界**：本 skill 讲 Channel **怎么用**（channel-first、构造、SameCore/CrossCore、vs Buffer）。CrossCore **跨核架构铁律**（`+16` 双发 / prime-drain 方向 / 全局 sync 预算核算）权威表述在 `../cannbotdsl-cv-fusion/SKILL.md` §5，本 skill 引用而不复述。VF 计算折叠见 `../cannbotdsl-vf-fusion/SKILL.md`。

## 1. 创建 API

```python
def __init__(self, mem_loc, shape=None, dtype=None, *, depth,
             kind=ChannelKind.SameCore, addr=None,
             stride=None, data_format=None, n1_pad=0):
```

```python
Channel(MemLoc.L1, (128,128), dtypes.float16, depth=3)                              # same-core
Channel(MemLoc.UB, (64,128), dtypes.float32, depth=2, kind=ChannelKind.CrossCore)   # cross-core
```

- `data_format` 自动推断（L1/L0X→`nz`，UB→`nd`），可显式覆盖（`data_format="nz"`）。
- buf_id / sync_id 由框架自动分配，用户**不指定**；`addr=` 可保留有意的静态物理布局。
- `channel_rewind()` 重置 Channel 的硬件 ID 游标和 Channel/Buffer 共享的地址 bump pointer；之后创建的 Buffer 也会复用低地址。

## 2. channel-first（推荐写法）

**默认这样写**：把 Channel 对象直接当操作数传给数据 op（`mem_copy`/`matmul`/`muls`/…），**不写任何 acquire/commit/wait/release**。框架从 Write/Read 操作数 + 循环规则自动合成 4 相协议。

**最小跨核 CV-mix**（零手写 4 相原语）：

```python
a_l1  = Channel(MemLoc.L1, (tile_m, tile_k), dtypes.float16, depth=1)
l0c   = Channel(MemLoc.L0C, (tile_m, tile_n), dtypes.float32, depth=1)
cv_ub = Channel(MemLoc.UB, (tile_vec_m, tile_n), dtypes.float32, depth=1, kind=ChannelKind.CrossCore)
out_ub = Channel(MemLoc.UB, (tile_vec_m, tile_n), dtypes.float32, depth=1)
...
mem_copy(a_l1, a_gm, engine=nd2nz)   # Channel 直接做 dst
mem_copy(l0a, a_l1)                   # Channel 既做 dst 又做 src
matmul(l0c, l0a, l0b)
mem_copy(cv_ub, l0c, engine=fixpipe) # 跨核 handoff（kind=CrossCore 仍显式声明）
muls(out_ub, cv_ub, scale)           # vec 侧消费，原地事务也自动合成（§9.3）
mem_copy(out_half, out_ub)
```

**用户仍需显式声明**（框架推不出，属算法决策）：`depth`（流水调度）、`kind`（sync_id 分配需全局信息）、软件流水结构（prologue/steady/epilogue、DelayLineGroup）。框架只合成同步样板，**不做流水调度**。

**自动覆盖的事务形态**：per-op、单 K-loop 累加器（`matmul(l0c,...)` loop-scoped 写）、Read-Many（Q 复用）、消费者原地事务（softmax `muls(slot,slot,·)` 读改写，§9.3）、split-M 子块生产（`tile_view`-on-Channel，§9.5）。已验证全集见设计文档 §15.3。

> **自动合成能力经真机 probe 确认覆盖到以下场景**（SageAttention channel-first 化，probe 在 `skills/probes/`（已验证 probe），均真机 PASS、与显式基线逐位同级）：
> 1. **channel 只读操作数落在 vf region 内** → 框架**下探 vf** 为其合成 wait(vf 前)/release(vf 后)。与"原地读改写（slot 是 vf output）"不同：**只读**消费（非 vf output、非原地）同样能 channel-first——不受"vf output 须 make_memref-rooted 才逼显式"约束。（`p_cf_readonly_vf_operand.py`，max_abs=0.0；SageAttention FW-8 后 qk_ub/pv_ub 消费即此例）
> 2. **跨 if-else 兄弟分支的 channel wait**（wait/release 在分支外、slot 在某分支的 vf 内被读）。（`p_cf_crossbranch_wait.py`）
> 3. **两个 AIV 共写同一 split-M slot、commit 一次**（各写一半 row，非单核写子块）→ 合成正确、不串数/不死锁。（`p_cf_splitm_shared_slot.py`）

> **channel-first 是同步去样板器，不是流水调度器**。`depth` 只加缓冲容量、**不改跨迭代发射顺序**：单向无环流水下 depth 可取代手写 lag-N；含**跨核环形依赖**（FA 的 `QK→softmax→(p_l1)PV→(pv_ub)update`）时，重叠必须把 lag 编码进程序序——lag-N/prologue/epilogue **仍需手写**（FA 旗舰用例逐字保留软件流水，性能持平 793 vs 798us）。

## 3. SameCore vs CrossCore

`kind=ChannelKind.SameCore`（默认）/ `ChannelKind.CrossCore` 决定底层用哪套硬件同步：

| | SameCore | CrossCore |
|---|----------|-----------|
| 底层 sync | `asc_lock`/`asc_unlock`（pipe mutex，`mutex_id = buf_id ∈ [0,31]`） | per-core counter（`+16` 双发由框架自动合成，源码不可见） |
| 边界处理 | 自平衡，无 prime/drain | 入口 **prime** / 出口 **drain**（方向写反 → 死锁） |
| depth 下限 | `depth ≥ 1` | `depth ≥ 1`，且 **≤ 8**（见 §4） |
| 用途 | 单核内 ring buffer / 跨迭代状态 | 跨 AIC/AIV 核 handoff |

**CrossCore 跨核铁律**（`+16` 双发、prime/drain 方向、全局 sync 预算）写反不报编译错但死锁或读陈旧数据——权威表述与推导见 `../cannbotdsl-cv-fusion/SKILL.md` §5，此处不复述。

## 4. depth 限制 —— CrossCore ≤ 8

`if is_cross_core and depth > 8: raise ValueError`。

这是**全局 per-func sync 预算**（不是 per-channel）：一个 func 内所有 CrossCore channel 的 `Σ depth ≤ 8`。根因（16 base sync_id / CUBE `+16` 双发 / 每 slot 2 counter = 8 slot）与逐项核算见 `../cannbotdsl-cv-fusion/SKILL.md` §4.1 / §5。超预算编译期直接 `raise`，需拆 kernel（中间量落 GM）——拆分决策见 cv-fusion §4.4。

## 5. Channel vs Buffer

两者可以在同一 kernel 中共存，并共享同一地址分配器，因而自动分配不会重叠。Buffer 不消耗 Channel 的 buf_id/sync_id，也不具有 ring/事务语义。

| | 用什么 | 何时用 |
|---|--------|--------|
| **Channel** | `Channel(...)`，channel-first 传给数据 op（§2） | ring buffer / 跨迭代状态 / 跨核 handoff；需要 buf_id + sync 语义 |
| **单块临时存储** | `Buffer(MemLoc.*, shape, dtype, ...)` | VF/kernel 内部临时量；无 buf_id、无 ring 语义、无 sync |

旧 NBuffer 与 `make_ub/make_l1/make_l0a/b/c` 已移除。depth-N、double buffer 或任何生产者/消费者同步必须用 Channel；不能由 Channel 表达的旧手动 NBuffer 流水当前不受支持。

## 6. 常见陷阱

1. **raw 模式 for 循环藏进普通 helper**：被 AST 全展开 → softmax 514 exp vs 滚动 10 exp，**~3.5× 退化且不报错**。raw-vf 的 for 必须在 `@jit` 函数体内（设计文档 §15.5）。
2. **跨核 prime/drain 方向写反 → 死锁**：prime 必须 consumer 端发起、drain 必须 producer 端发起。方向规则见 `../cannbotdsl-cv-fusion/SKILL.md` §5（channel-first 下 prime/drain 由框架合成）。
3. **CrossCore `Σ depth > 8`** → 编译期 `raise ValueError`。
4. **用 Buffer 模拟 double buffer** → Buffer 没有 slot 光标或同步；改为 `Channel(..., depth=N)`。旧手动 NBuffer 方案不受支持。
5. **忘记 `channel_rewind()`** → 跨 kernel 复用同一 arena 时游标不重置，读到上个 kernel 的残留。
6. **想靠大 depth 替代跨核软件流水** → ping-pong 空转无重叠。depth 只加缓冲、不改发射顺序（§2 末）。

## 参考

- 跨核架构铁律 / sync 预算核算 → `../cannbotdsl-cv-fusion/SKILL.md` §4-§5；VF 计算折叠 → `../cannbotdsl-vf-fusion/SKILL.md`
