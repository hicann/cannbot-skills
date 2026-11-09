# ArgReduce 模式映射决策树（AscendC，arch35）

> **层级**：AscendC 语言适配层。本文档回答一个问题：**给定一个算子的 shape，该走 8 个模式中的哪一个？** 它是通用层 [../../generic/splitting.md](../../generic/splitting.md)（切分维度）在 arch35 的全量落地，与工业实现（ops-math `arg_common_base_tiling_arch35.cpp`）的 8 模式一一对应——按本树的判定走，能复现工业实现的模式选择（468 白盒用例实测 0 分歧）。
>
> **怎么读这棵树**：从上往下按顺序判断，**后面的判定会覆盖前面的结果**（特化优先）。每一步只需要几个数：shape 折叠出的 A1/R/A0、元素字节数 eb、平台的 VL/vRegSize/coreNum/ubSize。常量记号：`eb` = 元素字节数，`VL` = `GetVLSize()`（一条向量能装几个元素），`vRegSize` = 向量寄存器字节数，`coreNum` = 动态核数（每 4KB 输入 1 核，上限物理核），`originCoreNum` = 物理核数，`ubSize` = UB 容量。
>
> **8 个模式一句话**（细节见 §2-§4）：
>
> | 模式 | 一句话 |
> |---|---|
> | COPY_ONLY | 每场只有 1 个选手，不用比，位置写 0 |
> | AR_CUT_A | 选手排成一排，按行分给多核各算各的 |
> | ARA_CUT_A | 选手隔着站，按列分给多核，跨步处理 |
> | AR_GATHER / ARA_GATHER | R 很短：先在片上把"一排"转成"一列"，一次比较完 |
> | RA_CUT_A | 只有一场比赛且选手跨步：把连续方向铺到向量通道上算 |
> | ARA_CUT_A_AND_NEXT_A | 行和列都很大：两个方向同时分给多核 |
> | GROUP_REDUCE | 一场比赛太长单核算不完、又没别的比赛：把这一场拆给多核，再合并 |

## 1 预处理

```
1. 负 dimension 归一化：dim < 0 → dim += rank
2. 空 tensor 短路（先于轴折叠）：
   outSize == 0（任一 A 维含 0）→ 输出为空：realCoreNum=0, SetBlockDim(1)，kernel 全核早退
   R == 0 且 A 非空            → 按算子级空约定输出（常见 indices 全 0）
3. 轴折叠：A1, R, A0 = ∏前 / dims[dim] / ∏后
4. 动态核数：coreNum = min(max(1, CeilDiv(dataBytes, 4096)), originCoreNum)
5. 位置精度核查：块内偏移 ≤ 位置元素容量（等宽选型：fp16→uint16 即 65536；UB 反解天然满足，仍须显式核对）；全局位置块基址拼接 → int32 输出上限 2^31
```

## 2 第一级：主场景判定

```cpp
if (R == 1)                          → COPY_ONLY (10003)
else if (dim == rank-1 || A0 == 1)   → AR_CUT_A (10001)      // R 连续
else                                  → ARA_CUT_A (10002)    // R 跨步
```

## 3 第二级：性能特化覆盖（按序执行，后判定覆盖前值）

记 `allA = A1 × A0`，`arSize = R × A0`。

```cpp
// ── 特化 1：RA 向量化（A1==1 时 A0 铺 lane）──
if (A1 == 1 && (A0 >= coreNum * VL || (A0 > 1 && R < VL)))
    mode = RA_CUT_A;                                    // 20003

// ── 特化 2：AR Gather（R 一条向量放得下 + A 够并行 + 无 bank 冲突）──
if (A0 == 1 && R * eb < vRegSize && A1 * eb >= coreNum * vRegSize
    && (R * eb) % 32 != 0)                              // 32 倍数排除：bank 冲突
    mode = AR_GATHER;                                   // 20001

// ── 特化 3：ARA Gather（R 小或 R×A0 超 UB 半区）──
if (allA * eb >= coreNum * vRegSize
    && (R * eb < vRegSize || (arSize * eb >= ubSize / 2 && R < maxSize)))
    mode = ARA_GATHER;                                  // 20002

// ── 特化 4：GROUP_REDUCE（R 算力型长轴 + A 方向无并行度）──
if (R * eb >= originCoreNum * VL && R < INT32_MAX
    && ((A0 == 1 && A1 < originCoreNum) || (A0 > 1 && allA < originCoreNum * 2)))
    mode = GROUP_REDUCE;                                // 30001

// ── 特化 5：ARA 双轴切核（A1 与 A0 都大）──
if (mode == ARA_CUT_A && allA * eb >= coreNum * 128) {
    uint64_t minNextABytes = (A1 > coreNum && indiceDtype != INT64) ? 64 : 128;
    if (A0 * eb >= minNextABytes)
        mode = ARA_CUT_A_AND_NEXT_A;                    // 10012
}
```

**特化 4 的回退修正**（栅栏不值得时）：末维归约且 A1 充足 → 回退普通切核：

```cpp
if ((dim == rank-1 || A0 == 1) && mode == GROUP_REDUCE
    && A1 > 0.85 * originCoreNum)
    mode = AR_CUT_A;
```

**GROUP_REDUCE 配套**（host 侧必设）：`SetScheduleMode(1)`；workspace 用户部分按 [capability-mapping.md](capability-mapping.md) `WORKSPACE_MODEL` 布局申请。

**特化 5（双轴切核）补充**：
- `useFlatPath` Host 判定门（对齐工业实现 L482-489）：仅 `cutNextASize == nextASize` 且整行 UB 核算（`ubNeededFlat ≤ ubSize − MAX_EXTRA_BLOCK×BLOCK_SIZE`）通过时置 1，否则恒 0 走 normal 兜底路径——禁止只写字段不写判定（闭环缺口）
- 双轴门槛 `allA×eb ≥ coreNum×128` 的实证口径为 64 核 UT 配置（物理核 > 125 时典型场景退化至 10002，正确性不受影响）；trace 应按实证口径钉死假设值并注明退化路径

## 3a 模式收敛与裁剪

**正确性完备集 = {COPY_ONLY, AR_CUT_A, ARA_CUT_A}**：任意 shape 必然路由到三者之一（AR/ARA 内含切 R 循环覆盖超大 R），其余五个模式（RA/GATHER×2/双轴/GROUP_REDUCE）均为性能特化。

裁剪决策规则：
1. 无硬性性能指标、或典型场景已命中基础路径最优时，可裁剪特化模式，缩减 TilingKey 数与迭代数
2. 裁剪**必须记录**：不纳入清单 + 依据（性能指标/典型场景命中）+ 扩展路径（"扩展时按本决策树补齐；GROUP_REDUCE 扩展须同时补 SetWorkspaceSize 布局 + SyncAll + SetScheduleMode(1)"）
3. 裁剪只影响性能覆盖，不影响正确性——设计评审按此核对裁剪记录的完备性

## 3b 已知覆盖洞与增强判定（参考实现自身的洞，实测证伪）

**洞域**（决策树在以下 shape 域退化单核，实测 0.067x 灾难）：

```
A1 == 1（RA 布局）且 laneCapacity(≈vRegSize/eb) < A0 < coreNum×VL 且 R ≥ VL
  → RA_CUT_A 不命中（A0 < coreNum×VL 且 R ≥ VL）
  → GROUP_REDUCE 不命中（A0 > laneCapacity，超出 lane 铺设容量）
  → 落默认 AR_CUT_A，其多核切分沿 A1（aSize==1）→ SetBlockDim(1) 单核
```

**来源**：本洞为参考实现（arg_max_with_value）自身固有——其决策树即如此；官方 aclnnArgMin（另一套实现）在该域 56 核并行。例证：shape (8192,1024) dim=0 → 单核 680us vs 官方 45.7us。

**增强判定**（超出参考实现的改进，建议新算子采纳）：

```cpp
// RA 判定扩展：aSize==1 且 A0>1 时无条件走 RA 路径（A0 不足整核×VL 时按 A0 切核）
if (aSize == 1 && A0 > 1)
    mode = RA_CUT_A;   // 覆盖原洞域：A0 ∈ (laneCapacity, coreNum×VL) 按 A0 切核并行
```

预期收益 ~10x（洞域内）；采纳此增强须在 DESIGN 裁剪记录中注明"决策树增强点（超出参考实现）"并在 UT 增加洞域 shape 的路由断言。

**警示条款**：凡 `aSize == 1` 的 shape 落入 AR_CUT_A 即为单核退化——模式选择的 UT 必须包含 `aSize==1` 的路由断言以防回归。

## 4 第三级：ARA 载入子模式（aRaMode 101~104）

`mode == ARA_CUT_A` 时按载入能力细分：

```cpp
if (A1 >= coreNum && arSize <= ubElement)          // 切 A1，R×A0 全载 UB
    aRaMode = (A0 * eb < 64) ? ARA_MODE1 : ARA_MODE2;   // 101/102（MODE2 走 NDDMA 重排）
else if (A1 >= coreNum && A0 * eb < 32)            // nextA 极小，放不下 → 切 R
    aRaMode = ARA_MODE3;                                 // 103
else                                               // 其余：切 R + A0
    aRaMode = ARA_MODE4;                                 // 104
```

**MODE4 改道**（二次修正）：nextA 切核块 ≥ 64B → 升 `ARA_CUT_A_AND_NEXT_A`；否则 `R/originCoreNum × A0 × eb ≥ 4KB` → 改道 `GROUP_REDUCE`。

## 5 完整决策树（汇总）

```
输入 shape, dim, dtype, coreNum, originCoreNum, ubSize, VL, vRegSize
  │
  ① 空 tensor？ → 早退（BlockDim=1）
  ② R == 1？ → COPY_ONLY
  ③ dim 末维或 A0==1？ → 基础=AR ： 基础=ARA
  ④ 特化覆盖（序）：RA(A1==1) → AR_GATHER(R小+A够+非32倍) → ARA_GATHER(R小或R×A0超UB半区)
     → GROUP_REDUCE(R大+A无并行) → ARA双轴(A1与A0都大)
  ⑤ 修正：GROUP_REDUCE 且末维且 A1>0.85×物理核 → 回退 AR
  ⑥ ARA 基础 → aRaMode 101~104 细分（+ MODE4 二次改道）
  ⑦ 输出：mode 枚举 + aRaMode + 判定 trace（每步条件记录）
```

## 6 模式 ↔ 通用层概念对照

| 工业模式 | 通用层分解 | 说明 |
|---|---|---|
| COPY_ONLY | 场景 COPY | 退化透传 |
| AR_CUT_A | 场景 AR × A1 切核（× R 切块当 UB 不足） | 主路径 |
| AR_GATHER | 场景 AR × Gather 特化（[highlevel-api.md](highlevel-api.md) 手法 C） | R 小优化 |
| ARA_CUT_A | 场景 ARA × A1 切核 × aRaMode 载入细分 | 手法 D/E |
| ARA_GATHER | 场景 ARA × Gather 特化 | R 小优化 |
| ARA_CUT_A_AND_NEXT_A | 场景 ARA × 双轴切核（splitting §3 A0 切核） | 增强 |
| RA_CUT_A | 场景 ARA(A1==1) × RA 向量化（手法 F） | lane 化 |
| GROUP_REDUCE | 正交维度 R 切核（splitting §4 GroupReduce） | 跨核归约 |

## 7 UT 实证表（64 核 / UB 256KB / VL=256）

| 输入 | 期望模式 | 判定依据 |
|---|---|---|
| (4,1) dim=1 | COPY_ONLY | R==1 |
| (4,2) dim=1 | AR_CUT_A | 末维 |
| (64,60000) dim=1 | AR_CUT_A（切 R） | 末维 + R 超预算 |
| (8192,100) dim=1 | AR_GATHER | R<VL、A 够、非 32 倍 |
| (2,3,4,5) dim=1 | ARA_CUT_A | 中间轴 |
| (256,3,256) dim=1 | ARA_GATHER | R 小 |
| (2,100,4000) dim=1 | ARA_CUT_A_AND_NEXT_A | A 与 A0 都大 |
| (50,100000) dim=0 | RA_CUT_A | A1==1、A0 大 |
| (50,8192) dim=1 | GROUP_REDUCE | R 大、A 无并行 |
| (1,8192,64) dim=1 | GROUP_REDUCE（RA 分支） | R 大、allA 小 |

## 8 使用规则

1. 模式选择**必须走本决策树**并输出判定 trace（每步条件），禁止按模式名直觉选择
2. 阈值常量（4KB/0.85/64B/128B/32 倍数）为 arch35 工业实现实测值，**换平台须重新标定**——标定方法：按第 6 节的通用层分解重新推导（场景/切分语义不变，只调比例常量）
3. 决策树输出的 mode + aRaMode 直接对应 kernel 分发（[kernel-entrance.md](kernel-entrance.md)）与 TilingData 填充（[tiling-preprocess.md](tiling-preprocess.md)）
