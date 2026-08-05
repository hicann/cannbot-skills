# Channel 方式的 cube 核内流水（channel-first）

> 本文是稳定提炼：channel-first 的 cube 流水范式，内联骨架与数据，不指向易变的示例文件行号。
> 范式来源是 cookbook 的 channel-first tiled-matmul 与 CV-mix 算子（真机验证过），此处为快照。

## 1. 核心思想：depth 声明式

让 4 个 cube PIPE（MTE2/MTE1/M/FIXPIPE）在相邻 tile / K-step 上错位并行，写法是把每级 buffer 声明成 `Channel(mem_loc, ..., depth=N)`，`depth` 就是预取/流水深度旋钮，然后写**最朴素的 fused K-loop**。acquire→fill→commit（生产）与 wait→use→release（消费）4 相协议、buf_id、sync_id 全部由框架合成；尾块也由框架在使用点自动插 `local_slice` 视图。**加深流水只改 depth，不动循环体。**

Channel 是双游标环形缓冲：写游标与读游标各自推进，depth 个 slot 让生产/消费错位在途。

## 2. 声明骨架（cube 四级 buffer 全 Channel）

```python
from cannbotdsl.channel import Channel
from cannbotdsl import dtypes
from cannbotdsl.typing.types import MemLoc, ChannelKind

P_NUM = 2   # 预取深度：L1 携带 P_NUM+1 个 slot

a_l1 = Channel(MemLoc.L1,  shape=(tile_m, tile_k), dtype=dtypes.float16, depth=P_NUM + 1)   # depth=3: MTE2 预取 2 个未来 K-tile
b_l1 = Channel(MemLoc.L1,  shape=(tile_k, tile_n), dtype=dtypes.float16, depth=P_NUM + 1)
l0a  = Channel(MemLoc.L0A, shape=(tile_m, tile_k), dtype=dtypes.float16, depth=2)           # L0 tile double buffer
l0b  = Channel(MemLoc.L0B, shape=(tile_k, tile_n), dtype=dtypes.float16, depth=2)
l0c  = Channel(MemLoc.L0C, shape=(tile_m, tile_n), dtype=dtypes.float32, depth=2)           # K-loop 累加器
```

- 一律用 `shape=(...), dtype=...` 声明完整整块 tile；尾块由框架在使用点自动插 `local_slice` 视图，一份编译产物仍跑多组动态 M/K/N。
- 跨核 handoff（Cube→Vec）用 `Channel(MemLoc.UB, ..., depth>=2, kind=ChannelKind.CrossCore)`。

## 3. 朴素 K-loop（无 prologue/epilogue/drain）

```python
for k in range(K_TILES):
    a_tile = tile_view(a_gm, (tile_m, tile_k), (m_idx, k))
    mem_copy(a_l1, a_tile, engine=nd2nz)         # GM→L1 (MTE2)
    mem_copy(l0a, a_l1)                           # L1→L0A (MTE1)
    b_tile = tile_view(b_gm, (tile_k, tile_n), (k, n_idx))
    mem_copy(b_l1, b_tile, engine=nd2nz)
    mem_copy(l0b, b_l1, transpose=True)
    matmul(l0c, l0a, l0b, init=(k == 0))   # mmad (M)，k==0 清零累加
mem_copy(out_tile, l0c, engine=fixpipe)           # L0C→GM (FIXPIPE)
```

channel-first 下用户不写 prologue/epilogue/drain、不写 4 相原语、不维护 buf_id——全部由框架合成。

## 4. depth 语义（每级 buffer 该开多深）

| buffer | depth | 作用 |
| --- | --- | --- |
| L1（K/V tile） | `P_NUM+1` | 双游标环 = 预取窗口：depth=3 时 MTE2 可载入 k+1、k+2 而 matmul 还在消费 k |
| L0A / L0B | 2 | tile double buffer：载入 k+1 与 mmad k 重叠 |
| L0C | 2 | K-loop 累加器；单 K-loop 单累加器，acquire/commit 由框架提升到循环外 |
| cv_ub（Cube→Vec, CrossCore） | ≥2 | Cube fixpipe 写 slot b 时 Vec 消费 slot a、不互等 |

CrossCore Channel 的 depth 上限是 8（超过编译期 raise；超过预算需拆 kernel，见 `../../cannbotdsl-cv-fusion/SKILL.md` §4.4）。

## 5. 预算换算（depth 进 op-design §2 预算表）

Channel 字节 = `depth × slot_bytes`，按 depth 对硬限制校验（L1 512KB / L0A/B 64KB / L0C 128KB / UB 256KB）。

例（fp16、tile=128）：
- L1 各 `3 × 32KB = 96KB`
- L0A/L0B 各 `2 × 16KB = 32KB`
- L0C `2 × 64KB = 128KB`
- 均在限内。

## 6. 性能基线

数据来自 cookbook 的 tiled-matmul（M×N×K = 1664×1664×1152、tile=128、L1 depth=3、L0* depth=2、32 blocks）在真机上的 msprof PipeUtilization：

| 维度 | Channel-first |
| --- | --- |
| 单 kernel avg time | 29.40 us |
| kernel 方差（range） | 2.11 us |
| 总 cycles | 1,369,890 |
| MAC 吞吐 | 18.63 us |
| Scalar 开销 | 5.27 us |
| MTE2（L1→L0） | 25.03 us |
| Fixpipe | 3.93 us |
| Cube 利用率 | 78.5% |
| DSL 源码行数 | 159 |
| prologue/epilogue | 无（自动） |

约 78% cube 利用率，无显式软件流水结构。与 `../../cannbotdsl-op-design/SKILL.md §6`（同步方案选型）、`§7`（流水编排设计）一致。
