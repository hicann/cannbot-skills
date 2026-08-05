# MLA 的 buffer 预算

## 硬件上限（dav-3510）

| Region | 上限 | MLA 里装什么 |
|---|---|---|
| UB | 256 KB | qk_ub、pv_ub、p_ub(NZ)、o_ub、res_o、mask、softmax 标量 |
| L1 | 512 KB | Q/K/V 的 d_chunk 切片、rope 切片、p_l1（V→C handoff） |
| L0A | 64 KB | Q chunk（PV 时是 P）、rope 的 Q |
| L0B | **64 KB** | K chunk（PV 时是 V）、rope 的 K ← **最紧** |
| L0C | 256 KB | qk 累加器 (M,tile_n) f32、pv 累加器 (M,d_nope) f32 |

超限在 `Channel` 构造期就抛
`ValueError: <REGION> allocation exceeds the N-byte capacity`，不用等运行时设备错误。

## 已验证配置（板上实测，全部在限内）

`tile_n=128, d_rope=64, causal` 下：

| d_nope | d_chunk | tile (cube,vec) | L0 共用 | L0A | L0B | L0C | L1 | UB |
|---|---|---|---|---|---|---|---|---|
| 512 | 128 | (64, 32) | 是 | 40 K | 48 K | 192 K | 232 K | **225 K** |
| 512 | 128 | (32, 16) | 是 | 20 K | 48 K | 96 K | 188 K | 113 K |
| 448 | 64 | (64, 32) | **否** | 40 K | 48 K | 176 K | 152 K | 205 K |
| 448 | 64 | (32, 16) | **否** | 20 K | 48 K | 88 K | 116 K | 103 K |

「L0 共用」= `d_chunk == tile_n`，此时 QK 与 PV 的 L0 操作数形状重合可共用一对 channel；
否则必须各分配（见 `pitfalls.md` §1）。

最紧的两个：**L0B 恒为 48/64 K**（与 tile 无关），**UB 在 d_nope=512 满 tile 下 225/256 K**。

## 逐 buffer 明细（d_nope=512, tile=(64,32)）

### L0A（64 K）

| buffer | shape | dtype | depth | bytes |
|---|---|---|---|---|
| `l0a`（QK-Q / PV-P 共用） | (64, 128) | f16 | 2 | 32 K |
| `l0a_r`（rope Q） | (64, 64) | f16 | 1 | 8 K |
| | | | | **40 K** ✅ |

### L0B（64 K）— 最紧

| buffer | shape | dtype | depth | bytes |
|---|---|---|---|---|
| `l0b`（QK-K / PV-V 共用） | (128, 128) | f16 | **1** | 32 K |
| `l0b_r`（rope K） | (64, 128) | f16 | 1 | 16 K |
| | | | | **48 K** ✅ |

**L0B depth 只能是 1**：`(128,128) f16 = 32 K`，depth=2 就 64 K 打满，再加 rope 立刻超限。
代价是 K/V 的 L1→L0B 无法双缓冲，靠 L1 侧 depth=2 缓解。

### L0C（256 K）

| buffer | shape | dtype | depth | bytes |
|---|---|---|---|---|
| `qk_l0c` | (64, 128) | f32 | 2 | 64 K |
| `pv_l0c` | (64, 512) | f32 | **1** | 128 K |
| | | | | **192 K** ✅ |

`pv_l0c` 只能单缓冲：双缓冲 256 K + qk 的 64 K 超限。

### L1（512 K，余量充裕）

| buffer | shape | dtype | depth | bytes |
|---|---|---|---|---|
| `q_l1` | (64, 128) | f16 | 2 | 32 K |
| `k_l1` | (128, 128) | f16 | 2 | 64 K |
| `v_l1` | (128, 128) | f16 | 2 | 64 K |
| `qr_l1` | (64, 64) | f16 | 1 | 8 K |
| `kr_l1` | (128, 64) | f16 | 1 | 16 K |
| `p_l1`（跨核 V→C） | (64, 128) | f16 | 3 | 48 K |
| | | | | **232 K** ✅ |

### UB（256 K）— 按「什么都不折叠」保守计

| buffer | shape | dtype | depth | bytes |
|---|---|---|---|---|
| `qk_ub`（跨核 C→V） | (32, 128) | f32 | 2 | 32 K |
| `pv_ub`（跨核 C→V） | (32, 512) | f32 | **1** | 64 K |
| `p_ub`（NZ, n1_pad=16） | (32, 128) | f16 | 2 | 16 K |
| `o_ub` | (32, 512) | f16 | 1 | 32 K |
| `res_o` | (32, 512) | f32 | 1 | 64 K |
| `mask_ch`（causal） | (32, 128) | f32 | 1 | 16 K |
| `sm_max/sum/exp` 三缓冲 + tmp | (32, 1) | f32 | 11 | 1.4 K |
| | | | | **225 K** ✅ |

**`pv_ub` 必须单缓冲**：depth=2 会让 UB 到 289 K 超限。代价是 PV 与 update 无法重叠
（这也是 vec bound 的来源之一，见 `perf.md`）。

若要腾 UB：`o_ub` 与 `res_o` 生命周期不重叠（res_o 在累加期、o_ub 仅在 finalize 期），
alias 同一 region 可省 32 K。

## 跨核 sync 预算

跨核 Channel 每级占 2 个硬件计数器，**上限 8 级/func**。

| Channel | 方向 | depth |
|---|---|---|
| `qk_ub` | Cube → Vec | 2 |
| `pv_ub` | Cube → Vec | 1 |
| `p_l1` | Vec → Cube | 3 |
| | | **6 ≤ 8** ✅ |

## 预算核算

改 tile 或 d_nope 前先核算一遍，比等编译期报错快。按本页上述表格逐项计算各 region 占用，校验：
- 各 region 总量 ≤ 硬件上限
- `tile_vec_m ≥ 16`（NZ fractal 下限）
- 跨核级 ≤ 8

逻辑与本页表格同源，改预算规则时改表格。

> **除了 sync 预算，还要核算每核 tile 代价是否均衡**。MLA 的 causal 变体里 tile 代价沿 m-block 轴变化（`nkb = mb+1`）。若 `idx2crd` 把 m-block 放最内层且 extent 整除 GRID，该轴取值每核恒定 → 最贵的核永远最贵。把 m-block 轴挪到 `idx2crd` 维度表最外层即可均衡，这是纯排列、数值逐位不变。先用 host 侧算术算每核负载（`max(load)/mean(load) > 1.2` → 分发问题），见 `../../../core-skills/cannbotdsl-perf-optimize/SKILL.md` 第 0 步。

> **显式 `addr=` 别名的地址预算要单独标注**：别名 channel 共享同一物理地址，容量端按 `max(两者字节)` 而非 `两者之和` 核算。但同步层不可见地址重叠（`depth≥2` 重叠即静默串数据）—— 地址预算过了 ≠ 同步安全。详见 `../../../core-skills/cannbotdsl-op-design/SKILL.md` §2.0。

**先按「什么都不折叠」算**，再去优化。`vf` 不保证折掉中间 buffer——
跨 `cast` → `mem_copy(nd2nz)` 边界的暂存量往往折不掉，但它仍占 UB。
预算紧时一律按「不折」计。

## 减 UB 的手段（按优先级）

1. `pv_ub` / `qk_ub` 降 depth —— 失去对应的重叠，但 FA 类算子往往 PV-bound，损失有限
2. `o_ub` 与 `res_o` alias —— 生命周期不重叠，省 32 K，无性能损失
3. 缩小 `tile_vec_m` —— 所有 vec 侧 buffer 等比缩小，但会减少 M 方向并行度
4. 缩小 `tile_n` —— 影响 `qk_ub`/`p_ub`/`mask_ch`；注意 tile_n≠128 会让 vec 侧不能原样
   复用 FA 蓝本，代价大
