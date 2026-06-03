# MatMul 族 — 通用 Tiling 流程

> MatMul 族所有变体（a16w16 / mxfp4 / mxfp8 / batch_matmul / group_matmul）共享的 Tiling 推导流程。
>
> 核心流程：**初始分块 → 边缘优化 → 尾块处理 → L0C 缓冲 → L1 深度 → 缓冲数 → 组装**，七步递进。
>
> 变体差异见 [tiling-variants.md](tiling-variants.md)，跨变体字段见 [tiling-fields.md](tiling-fields.md)。
>
> **算法来源**：CANN `conv_api_tiling_*.cpp`（arch35），关键常量定义见 `conv_api_tiling_util.h` / `conv_template_utils.h`。

---

## 1. 算法三元组

MatMul Tiling 有三种算法策略，互斥选择：

| 算法 | 核心思想 | 适用条件 |
|------|---------|---------|
| **SWAT**（默认基线） | K 轴流式迭代，数据沿 K 分段载入 L1，每次只驻留一小段，计算完即滚动 | 所有 Shape，为回退保障 |
| **FullLoad** | A 或 B 全量驻留 L1，消除 K 轴重复搬运 | A 或 B 全载于 L1 内，对侧循环 ≥ 2 |
| **StreamK** | K 轴切分给多核，每核负责一段 K，workspace 归约 | K ≥ 32768 且 B=1 且无 group |

**互斥关系**：StreamK ⊥ FullLoad（StreamK 下 K 分散到多核，驻留语义失效）。

---

## 2. 关键常量

| 常量 | 值 | 含义 |
|------|----|----|
| `CUBE_BLOCK` | 16 | Cube 单元最小粒度，所有分块尺寸对齐到 16 |
| `DB_SIZE` | 2 | Double buffer 深度，L1 中 A/B 各保留 2 份实现 pingpong |
| `WINDOW_LEN` | 4 | 边缘合并滑动窗口，一次最多合并 4 个基本块 |
| `BASEM_BASEN_RATIO` | 2 | baseM/baseN 最大比例，防止维度失衡 |
| `STEPKA_THRESHOLD` | 4 | StreamK 下 stepK 截断上限 |

---

## 3. SWAT 七步推导

SWAT 是默认基线，也是 FullLoad 和 StreamK 的推导基础。

```
DoOpTiling():
  ├── _calc_basic_block()              # §3.1 初始分块 + 核利用率调整
  ├── _optimize_edge_basic_block()     # §3.2 M 方向边缘合并 (SWAT 机制 B)
  ├── _calc_tail_basic_block()         # §3.3 尾块拆分
  ├── _init_l0c_buffer_mode()          # §3.4 L0C 双缓冲判定
  ├── _calc_path_specific_l1()         # §3.5 L1 深度 + stepK
  ├── _calculate_n_buffer_num()        # §3.6 L1 缓冲数 (4 or 2)
  └── _build_tiling_data()             # §3.7 组装 TilingData
```

### 3.1 初始分块

确定 Cube 单元一次计算多大的块（baseM × baseN × baseK）。

```
1. 候选值:
   baseM = min(Align(M, 16), 256)
   baseN = min(Align(N, 16), 256)
   baseK = min(Align(K, 16), L0A容量 / (DB_SIZE × sizeof(dtype) × max(baseM, baseN)))
```

**核利用率检查**：用 baseM/baseN 算出总块数（mBlockCnt × nBlockCnt）。若总块数 < aicNum，说明分块太粗、核用不满，需要调小 baseM 或 baseN。

```
2. 核利用率检查:
   mBlockCnt = CeilDiv(M, baseM)
   nBlockCnt = CeilDiv(N, baseN)
   totalBlockCnt = mBlockCnt × nBlockCnt
   若 totalBlockCnt < aicNum → _adjust_basic_block() 重平衡
```

**_adjust_basic_block()**：优先调 tile 数较少的方向（更需要拆分），保持 baseM/baseN ≤ 2:1。调整后重算 baseK。

### 3.2 边缘合并（SWAT 机制 B）

M 不能被 baseM 整除时，末尾有不完整的"尾块"。边缘合并把最后几个完整块与尾块合并，重新划分为大小均匀的块，避免超小尾块导致 Cube 利用率骤降。

```
当 M % baseM > 0 且 K 内轴满足 cacheline 对齐时:

滑动窗口 (WINDOW_LEN=4) 搜索最优合并方案
若干 baseM 块与尾块合并为统一大小 mTailMain
目标: 最小化窗口总计算量

输出: mBaseTailSplitCnt, mTailMain, nBaseTailSplitCnt, nTailMain
```

### 3.3 尾块拆分

边缘合并后仍可能存在尾块。将大尾块切成多个小块，分给不同核并行处理。交替增长 mTailTile 和 nTailTile，优先拆尾块更大的方向。

```
当 tailBlockCnt > 0:
  优先拆分 M/N 中尾块更大的方向
  交替增长 mTailTile, nTailTile
  约束: mTailTile ≤ CeilDiv(baseM, 16), nTailTile ≤ CeilDiv(baseN, 16)
```

### 3.4 L0C 双缓冲

L0C 是 Cube 累加器输出缓冲区。双缓冲（=2）允许一份被 Cube 写入时另一份被 DMA 读出到 UB，实现计算和搬移 overlap。

```
dbL0c = (baseM × baseN × sizeof(FP32) × DB_SIZE ≤ l0cSize) ? 2 : 1
```

### 3.5 L1 深度 + stepK

决定 K 轴方向分段策略。A 和 B 的容量分配决定每次能载入多长的 K 段。

```
1. 每块占用:
   base_a_size = baseM × baseK × sizeof(dtype)
   base_b_size = baseN × baseK × sizeof(dtype)

2. 对称深度搜索（倍增搜索）:
   depth_init = _get_depth_a1b1(l1Size, base_l1_size)
   从 depth=1 逐次翻倍，直到 A+B 超过 L1 容量

3. stepK:
   stepKa = depthA1 / DB_SIZE
   stepKb = depthB1 / DB_SIZE
   互为倍数对齐; 上限 4; 不超过 K/baseK
   stepK ∈ {1, 2, 4}（2 的幂约束）
```

> **2 的幂约束**：`_get_depth_a1b1()` 倍增搜索使 depth 只能为 2 的幂，stepK = depth / 2。

### 3.6 L1 缓冲数

nBufferNum 控制 L1 中 pingpong 缓冲数量。4 缓冲比 2 缓冲能更好地隐藏 MTE2 搬移延迟。

```
kl1 = min(stepKa, stepKb) × baseK
used_4buf = baseN × kl1 × 4 + baseM × kl1 × 4
nBufferNum = (used_4buf < l1Size) ? 4 : 2
```

### 3.7 组装 TilingData

将 RunInfo（内部中间态）组装为 TilingData（下发 Kernel 的最终字段）。

```
usedCoreNum = (totalBlockCnt > 1 || tailBlockCnt == 0) ? aicNum
            : tailBlockCnt × mTailTile × nTailTile
kL1 = baseK × min(stepKa, stepKb)
```

### 3.8 SWAT 机制 A: BLOCK_TABLE 负载均衡（a16w16 专属）

当默认 baseM/baseN 导致各核任务量不均衡时（均衡度 < 0.88），触发查表重选：

```
触发条件:
  singleBlockNum = mBlockCnt × nBlockCnt / aicNum ∈ [1.0, 10.0]
  defaultBalance = CalcMultiCoreBalance(M, N, aicNum, baseM, baseN) < 0.88

若触发 → 在 BLOCK_TABLE 中查找预制评分更高的 (baseM, baseN) 组合
```

---

## 4. FullLoad 驻留策略

A 或 B 全量驻留 L1，消除 K 轴重复载入。小矩阵一次载入后驻留不动，K 轴迭代时只滚动对面的大矩阵。

### 4.1 门禁条件（五条，任一不过即回退 SWAT）

```
1. 策略条件: 未走 StreamK 分支
   └─ StreamK 下 K 已切给多核，"全载"语义失效

2. 小侧矩阵容量（至少一侧通过）:
   Bytes_A = baseM × Align(K, c0) × sizeof(dtype)
   Bytes_B = Align(K, c0) × baseN × sizeof(dtype)
   条件: min(Bytes_A, Bytes_B) × 2 ≤ L1_SIZE

3. 对侧循环次数 T ≥ 2:
   T_A = N / (usedCoreNum × baseN)
   T_B = M / (usedCoreNum × baseM)
   └─ T = 1 → 收益为 0，不开

4. 多核排布 (以 A-Full-Load 为例):
   mBlockCnt ≤ WINDOW_LEN(=4)
   aicNum % mBlockCnt == 0
   totalBlockCnt > aicNum

5. 流水健康度:
   Bytes_opp = baseX × kL1 × sizeof(dtype) ≥ 20 KB
   └─ < 20 KB 时 MTE2 带宽效率低，全载收益被反噬
```

两侧都通过时，选 min(Bytes_A, Bytes_B) 更小的那侧做全载。

### 4.2 优化目标

```
ΔBytes = (T - 1) × baseM × K × |dtype|    # A-Full-Load
```

典型 T=5 时，小侧 MTE2 字节下降 80%，Task 时间降低 5%~15%。

### 4.3 字段差量（相对 SWAT）

| 字段 | A-Full-Load | B-Full-Load |
|------|------------|------------|
| `stepKa` | CeilDiv(K, baseK) | SWAT 基线 |
| `stepKb` | 由剩余 L1 反推 | CeilDiv(K, baseK) |
| `isAFullLoad` | true | false |
| `isBFullLoad` | false | true |
| mTailTile（全载侧） | 强制 1 | SWAT 基线 |
| nTailTile（全载侧） | SWAT 基线 | 强制 1 |
| `nBufferNum` | 通常回退到 2 | 同上 |

**isAFullLoad 与 isBFullLoad 互斥，严禁同时为 true。**

### 4.4 L1 预算不等式

```
A_full_load + B_streaming_pingpong ≤ L1_SIZE

baseM × Align(K, c0) × sizeof(dtype)                           # A 全量驻留
+ (baseN × stepKb × baseK × sizeof(dtype)) × nBufferNum        # B 流式 pingpong
≤ L1_SIZE
```

如果溢出，自动收缩 B 侧 stepKb。缩到 1 仍溢出 → fallback SWAT。

### 4.5 选型决策

| 场景 | 策略 | 原因 |
|------|------|------|
| 两侧 > L1/2 | SWAT | 物理不可行 |
| 小侧 ≤ L1/2, T=1 | SWAT | 无重复搬移可消除 |
| 小侧 ≤ L1/2, T≥2, **真 MTE2 bound** (带宽 ≥ 85%) | **FullLoad** | MTE2 是瓶颈 |
| 小侧 ≤ L1/2, T≥2, 假 MTE2 bound (带宽 < 70%) | SWAT | 瓶颈不在搬移 |
| 已走 StreamK | **不叠加** | K 已切散 |

---

## 5. StreamK 策略

K 轴切分给多核并行，突破 MN 欠并行瓶颈。当 M 和 N 都很小但 K 很长时，MN 二维切分 tile 数不足，把 K 也切出来分给多核。

### 5.1 子模式门禁

**SK（纯 Stream-K）**：MN tile 数严重不足。mCnt×nCnt ≤ aicNum/2。

**DP+SK**：稳态走纯 DP（每核独立完整 K），仅末轮 tile 沿 K 切分。

| 子模式 | 判定条件 |
|--------|---------|
| **SK** | `Align(K, 256) ≥ max(8192, aicNum×256) / sizeof(FP16)` 且 `mCnt×nCnt ≤ aicNum/2` |
| **DP+SK** | `M%256==0 ∧ N%256==0 ∧ K ≥ max(8192, aicNum×128) / sizeof(FP16)` ∧ `mCnt×nCnt ≥ aicNum` ∧ `(mCnt×nCnt) % aicNum ∈ (0, aicNum/2]` |

### 5.2 函数链

```
DoOpTiling():
  ├── IsCapable()                # 门禁判断
  ├── ResetBase()                # baseM=baseN=256
  ├── FormulateBasicBlock()      # 确定每核 K 段长度
  ├── CalBaseK()                 # L0A 容量约束
  ├── CalL1Tiling()              # depth + stepK
  ├── AdjustL1Tiling()           # 对称化修正
  └── BuildTilingData()          # 组装
```

### 5.3 字段差量（相对 SWAT）

| 字段 | StreamK | SWAT |
|------|---------|------|
| `skSingleCoreK` | 每核 K 段长度 | **不存在** |
| `tailInfo.kCnt` | K 方向分段数 | **不存在** |
| `kL1` | baseK × min(stepKa, stepKb, **4**) | 无 4 截断 |
| `l1BufferNum` | **2** 固定 | 2 或 4 |
| `l0cDB` | **1** 固定 | 1 或 2 |
| `usedCoreNum` | **aicNum** 固定 | 可 < aicNum |
| `mBaseTailSplitCnt` | **1** 固定 | SWAT 机制 B |
| `mTailMain` | **0** 固定 | 边缘合并产出 |

**强制常量**：`l1BufferNum=2`、`l0cDB=1`、`mBaseTailSplitCnt=nBaseTailSplitCnt=1`、`mTailMain=nTailMain=0`、`usedCoreNum=aicNum`。

### 5.4 Workspace

StreamK 需要 Workspace 做跨核 partial sum 归约：

```
GetWorkSpace() = aicNum × 256² × sizeof(FP32)    # 部分和区
               + RPC_WORKSIZE × MB_SIZE           # 跨核通信区
```

---

## 6. 算法选择决策树

```
给定变体 + Shape (M, K, N, [B/g])

Step 1 — StreamK 判定
  ├─ 通过 → StreamK
  │   强制：l1BufferNum=2, l0cDB=1
  │   batch_matmul (B≥2) 和 group_matmul 默认不启用
  └─ 不通过 → Step 2

Step 2 — FullLoad 判定
  ├─ 通过 → FullLoad
  │   A-Full 与 B-Full 互斥，选小侧
  └─ 不通过 → SWAT（默认回退）
```

### 典型示例

**M=128, N=81920, K=4096 (FP16, aicNum=32) → A-FullLoad**:
```
Bytes_A = 128 × 4096 × 2B = 1 MB > L1/2(=256 KB) ✘ → 不开
```

Wait, that example from the mxfp4 doc won't work for FP16. Let me use correct numbers.

**M=512, N=512, K=16384 (FP16, aicNum=24) → SK**:
```
mCnt=2, nCnt=2, totalMNCnt=4 ≤ 12 → SK
kCnt = 24/4 = 6, skSingleCoreK = 2732
baseK=64, stepKa=min(4,4)=4, kL1=256
Workspace: 24 × 256² × 4 ≈ 6.3 MB
```

**M=N=1024, K=4096**:
```
Bytes_A = Bytes_B = 1024 × 4096 × 2B = 8 MB > L1/2 ✘
→ 两侧均放不下，回退 SWAT
```

---

## 7. 跨变体公共约束

| 约束 | 值 | 适用范围 |
|------|----|---------|
| `CUBE_BLOCK` | 16 | 所有变体 |
| `DB_SIZE` | 2 | 所有变体 |
| `WINDOW_LEN` | 4 | SWAT 机制 B |
| `BASEM_BASEN_RATIO` | 2 | 所有变体，baseM/baseN 最大比例 |
| stepK 取值 | {1, 2, 4} | SWAT/FullLoad（StreamK 有例外 stepK=3） |
| L1 缓冲数 | 4 或 2（不存在 1） | SWAT/FullLoad |
