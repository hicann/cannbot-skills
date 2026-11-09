# ARA 场景模板 — Tiling（AscendC）

> **层级**：AscendC 适配层模板。适用模式：ARA_CUT_A（10002，含 aRaMode 101~104）/ ARA_GATHER（20002）/ RA_CUT_A（20003）/ ARA_CUT_A_AND_NEXT_A（10012）。共享部分见 [../ar/tiling.md](../ar/tiling.md)（A1 切核、R 精度三查）；本文档只讲 ARA 增量。

## 1 aRaMode 载入子模式判定

```cpp
// 记 arSize = R × A0（单个 A1 行的候选块总元素数）
if (A1 >= coreNum && arSize <= ubElement)
    aRaMode = (A0 * eb < 64) ? ARA_MODE1 : ARA_MODE2;   // 101: Gather 直取 / 102: NDDMA 重排
else if (A1 >= coreNum && A0 * eb < 32)
    aRaMode = ARA_MODE3;                                 // 103: nextA 极小 → 切 R
else
    aRaMode = ARA_MODE4;                                 // 104: 切 R + nextA（兜底）
```

## 2 各子模式切分要点

| 子模式 | 数据流 | 切分计算 |
|--------|--------|---------|
| MODE1 | [R,A0] 块 Gather 直取 → RA 式向量比较 | A0 < 64B：A0 铺 lane 粒度小，Gather 起始索引 = i×dimR 构造 |
| MODE2 | NDDMA 重排 [A1,R,A0] → [A1,A0,R]，重排后走 AR 手法 B | 重排目标 buffer 计入 UB 预算（重排后 R 连续） |
| MODE3 | 切 R（跨步载入，逐块选择 + 基址偏移合并） | cutRSize 按 UB 反解（跨步载入按 A0 stride） |
| MODE4 | 切 R + 切 A0（双维切块） | cutRSize × cutNextASize 联合预算 |

## 3 特化覆盖增量

- **RA_CUT_A**（A1==1）：A0 铺 lane（A0 ≥ coreNum×VL 或 R < VL 触发），无需 A1 切核；切核按 A0
- **ARA_GATHER**（R*eb < vRegSize 或 arSize*eb ≥ ubSize/2）：Gather 转置；内部按 `cutASize==1` 分"切 nextA 的 RA 式"与"NDDMA 重排 + A·A0 合轴 RA 式"两条子路径
- **ARA_CUT_A_AND_NEXT_A**（双轴切核）：`realCoreNum = blkNumA × blkNumANext`；`blkFactor2nd/blkTailFactor2nd/blkNum2nd` 按 A0 切核计算；`useFlatPath`（A0 扁平化子路径）按 A1 范围判定

## 4 TilingData 填充增量

在 AR 基础上追加：`nextASize`、`cutNextASize`（MODE4/双轴）、`blkFactor2nd/blkTailFactor2nd/blkNum2nd`（双轴）、`aRaMode`（101~104）、`gatherBlockSize`（GATHER）、`useFlatPath`。

## 5 产出校验

| # | 校验 |
|---|------|
| 1 | aRaMode 与 UB 预算一致（MODE1/2 要求 arSize 全载） |
| 2 | 双轴切核 blkNumA × blkNumANext == realCoreNum ≤ 物理核数 |
| 3 | 跨步载入地址 = 基址 + r×A0×eb（stride 核对，错 stride = 静默错误） |
| 4 | golden UT：各 MODE 的 (tilingKey, aRaMode, cutRSize, cutNextASize) 对拍 |
