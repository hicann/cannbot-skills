# ScatterUpdate DSL：确定路径三段式段尾覆盖落地

> Phase1 线性化 / Phase2 嵌入排序（over-launch safe）/ Phase3 段尾覆盖（last-write-wins，(b,k,s) 分解）/ lookback 与 excl 发布 / U9 规避。det 程序族的实现真值源（编排模型 = [../../generic/conflict-domain.md](../../generic/conflict-domain.md) + AscendC 层 [../ascendc/deterministic.md](../ascendc/deterministic.md) §1-3，本文只载 DSL 落地差异）。

---

## 1 三段式 DSL 全景

```
Phase1 线性化（SIMT grid-stride）: pos → 冲突 key + 越界检查 + srcPos(SUBSET)
   排序前的 k 组原子发布：本核各 tile 的 es 累积 → 循环外一次性原子加（§2.3）
Phase2 嵌入排序（SortDSL 复用，kernel 内联非独立发射）:
   排序 tile = 单核容量定宽；多 tile 经 lookback 域衔接
Phase3 段尾覆盖（SIMT grid-stride）: (b,k,s) 任务分解 + 段尾单写者（O(1) 自判）+ 段尾任务直写
```

## 2 Phase1/Phase2 编排细节

### 2.1 冲突 key 合成

= generic/conflict-domain.md §2（覆盖全部维 = 逐维反解 + dim 替换；覆盖单维 = j 直用；nd = Σ idx·stride）——DSL 侧以**整除**实现逐维反解（被除数源 kernel 级参数；固定除数统一使用 `//`；禁 ceil_div——SIMT vf 内验证失败，用 `(n+d-1)//d`）。

### 2.2 嵌入排序（over-launch safe 模式）

宿主算子发射核数 > 排序所需核数时，库内全核屏障凑不齐而死锁。DSL 落地：

- **排序分块域 = active_cores 运行期参数**（非发射数）：`b >= active_cores` 的超发核 tile 循环空跑（`n_my = ceil((tc-b)/nb)` 钳 0），**屏障无条件全核到达**
- 宿主 tiling：`blockDim = max(phase3_cores, phase1_cores, sort_active)`；`sort_active = min(core_count, tile_count)`
- **单核路径**：`b == 0` 守卫——仅核 0 执行排序，其余核直接到屏障（消除冗余功；全核冗余为 ASC 形态，守卫为已裁定偏离）

### 2.3 excl/全局计数原子发布（位置铁律）

**发布在 tile 循环外**（每核每轮一次，= ASC `GetGlobalExclusiveSum` 尾部 `SimtGlobalOffset`）：

```
A 相 for i in range(n_my): prescan 累积 es 进 excl_acc（不发布）
循环外：slot2 ← excl_acc → 通道中转 → 原子加 excl_bins（一次/核/轮）
```

**循环内发布的失效模式**：每 tile 发布"累积至今"值 → 每核多 tile（多波：tc > 核数；单核；强制小 tile）时同核重复累加 → excl 虚增 → 位置飞出 [0,N) → 越界写硬错或静默错序。单波（tc ≤ 核数）时每核恰一次，两种写法等价；多波规模为必测项。
- u32 计数域：MTE 原子（通道中转 + `copy_options(atomic_add=dtypes.bool(True))`）；i64 计数域：SIMT 标量原子（MTE 原子不支持 i64 目的）——值须 canonical 构造（`dtypes.int64(...)`）

### 2.4 U9 规避：kernel 级预计算

任意相同线程无关子式出现在 ≥2 个 vf 作用域 → CSE 跨域合并 → 编译崩溃（无诊断）。**规则**：凡多 vf 域共享子式（坐标基址/步进类）一律 kernel 级预计算为显式参数传入各相位：

```
b64 = b + dtypes.int64(0)        # canonical 锚（kernel 级预计算）
pos_base = b64 * THREAD_NUM;  step = (nb + i64(0)) * THREAD_NUM
```

预计算链同时是 **canonical 源**（严格守卫 API 的被除数来源）。

## 3 Phase3：段尾覆盖（last-write-wins）

### 3.1 任务分解 (b,k,s)（整除）

= AscendC deterministic.md §3 三场景通用（覆盖全部维 b=s=1；覆盖单维轴首/轴末/轴中三特化）。被除数 `task` 锚定 kernel 级预计算参数链（canonical）；固定除数统一使用 `//`：

```python
q = task // s_dec                              # 整除
su = task - q * s_dec
b = q // n_idx                                 # i64 域整除（无 64 位乘高原语）
k = q - b * n_idx
```

- 段尾判据：`k < n_idx - 1 and sorted[k+1] == sorted[k]` → 非段尾跳过（读走普通 gm[]——barrier 稳定数据）；**写条件 = 全局末位或与下一索引不同**（O(1) 判据，无扫描、无跨线程协调）
- **段尾任务直写自己**：`perm[k]` 即段内原始行号最大（稳定排序保证）= last-write-wins 胜者，单元素字节搬运写入 var（纯搬运 bit-exact，无算术）；判据在 s 维冗余执行幂等无害
- 哨兵任务前置跳过：`key == sentinel` 先于段尾判据（error/skip 双路径越界语义一致，§5）
- 覆盖顺序（最后赢 / 第一个赢）是语义决策点——第一个赢判据反转为段首向（`k == 0 or sorted[k-1] != sorted[k]`），取段首 `perm[k]`，机制同构
- 源寻址 `src[(b·strideK + perm[k])·sNum + s]`；flat 任务布局相邻线程相邻 s → coalesced 访存

## 4 lookback 与 volatile 语义映射（读写两侧对表）

排序 lookback（多 tile 衔接）的跨核协议——**逐处对表 ASC volatile 清单**（ASC 仅 sort CopyOutGm exclBins 读 + doubleBuffer 写为 volatile，其余普通）：

| 访问点 | DSL 形态 | ⇔ ASC |
|---|---|---|
| gh 轮询读（他核在写的就绪标志） | `simt.l2cache_load` | volatile 读（ASC 经 MTE 引擎轮询实现同义新鲜度） |
| gh 聚合发布（早）/ 前缀发布（晚） | `simt.l2cache_store` | volatile 写（ASC 经 MTE 直写） |
| excl_bins 读（屏障后稳定） | `simt.l2cache_load` | volatile 读（ASC CopyOutGather 对应点 volatile） |
| 段尾判据/排序后表/src 基值读 | **复合单 kernel：`simt.l2cache_load`**；普通 `gm[k]` 仅限 ASC 分离 kernel + MTE 写出形态 | ASC 分离 kernel 无 volatile（屏障后稳定）——复合 kernel 下 sorted/perm 由 SIMT stcg 写出，普通读会命中他核 DCache 陈旧行 |
| 散射落点/dst 写（DB 跨轮 MTE 回读） | `simt.l2cache_store` | volatile 写 |
| var 终态写（host 唯一读者） | **普通 `gm[t] =`**（tracked 写） | 无 volatile（kernel 边界保证可见） |
| Phase1 键/srcPos 写（他核 MTE 回读） | `simt.l2cache_store` | ASC 分离 kernel 普通写——**复合单 kernel 的必需偏离**（写侧对表时以"读方路径"判定，非 ASC 字面形态） |

**自旋防外提**：轮询循环内 `l2cache_load` 会被 LICM 外提成死自旋——每轮 `simt.dcci_single(ptr)` 反外提锚（1 指令/轮，= volatile 指令级替身）+ 自旋上限安全网。

### 4.1 与 AscendC SortLib 的差异对照（维护用真值源）

| # | 事项 | AscendC（sort_radix_more_core.h / det_simt.h） | 本适配层（tsnu_sort / kernel） | 状态 |
|---|---|---|---|---|
| 1 | Phase1 读 index / 写 lin | 普通 `__gm__` 读写 + `SetFlag<V_S>+WaitFlag<V_S>+SyncAll` | tracked 读 + `l2cache_store`（stcg）写 + `global_sync_all` | 等价（写侧更强） |
| 2 | Phase1→Phase2 同步 | `SetFlag<V_S>(0)+WaitFlag<V_S>(0)+SyncAll()` | `global_sync_all()`（= `asc_sync` + MTE3 跨核同步 + `block_wait(S)`） | 等价 |
| 3 | Phase2 边界 | `pipe.Reset()`×2 + `SyncAll()` | `_barrier()` | 等价 |
| 4 | Phase3 读 sorted/perm | 普通读（前提：SortInvoke 以 MTE 写出，不经过他核 DCache） | 普通读（前提不成立：内嵌 sort 以 SIMT stcg 写出） | 结构差异→已修（改 `l2cache_load`） |
| 5 | A 相 per-tile hist/cumsum/字节行 | CopyUbToGm 落 GM workspace，B 相 MTE 回读 | `mem_copy` 写 ws_b8/ws_hist/ws_cumsum 行，B 相 `mem_copy` 回读 | 等价 |
| 6 | excl 发布 | `SetAtomicAdd<int32>+CopyUbToGm`（u32）/ SIMT 标量原子（i64） | `mem_copy(copy_options(atomic_add))`（u32）/ SIMT 标量原子（i64） | 一致 |
| 7 | 发布原始直方图 | `EncodeHistVF(hist\|bit30)` + `SyncEvent<V_MTE3>/<MTE3_V>` | SIMD `publish_fn(hist+2^30)` + `mem_copy`（MTE） | 编码等价；缺逐处事件对（依赖 DSL 自动同步） |
| 8 | lookback 轮询 | MTE `CopyGmToUb` 反复读前驱行到 UB + VF 全 256-bin 三态计数 + `SyncEvent<V_S>`；无超时 | SIMT `l2cache_load` 逐 lane 仅对 v==0 `dcci` 短自旋；`spin_limit` 上限 | 结构差异（最大）；已改位域三态（bit30\|hist=就绪 / bit31\|value=前缀 / 0=未发布）+ 仅对 v==0 dcci 短自旋 |
| 9 | 前缀编码 | `value\|bit31`（正，值域 0–29 位）覆写同一行 | `acc-2^30`（负）覆写同一行 | 语义等价、编码不同；定案保留 `acc-2^30` 负编码（三态解码同构） |
| 10 | 发布/lookback 屏障 | A→SyncAll→B（发布+lookback 交织）→SyncAll | 原版交织无中间屏障；已改 B1(发布)→barrier→B2(lookback) | 已修 |
| 11 | 尾行发布 guard | `tileId < tileCount-1` 才发布 | 无条件发布（多写一行） | 无害，保留 |
| 12 | 超发核 | `blockIdx >= activeCores` 直接 return | `n_my=0` 空跑 + 无条件 barrier | 已标注的同构偏差 |
| 13 | 输入/前驱流水 | `inQueueX_`/`inQueueGlobalHist_` MTE 双缓冲队列 | `RingBuffer` depth=1 串行 | 性能差异，非精度 |

## 5 越界与错误上报

- error 形态：Phase1 检查后 `simt.atomic_add(gm_err.ptr(0), 1)` 计数 → wrapper 同步后读 flag 上抛（= ASC `__trap` 语义等价）
- skip 形态：det 经哨兵 key（generic/bounds.md §4——键值域最大值，排序沉底跳过）；nondet 直跳；双路径语义一致
- 检查在原值域（禁先截断再比较）
