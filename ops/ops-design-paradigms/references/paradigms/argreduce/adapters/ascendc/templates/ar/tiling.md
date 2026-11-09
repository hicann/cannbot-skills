# AR 场景模板 — Tiling（AscendC）

> **层级**：AscendC 适配层模板。适用模式：AR_CUT_A（10001）主路径与 AR_GATHER（20001）特化。本文档只讲 tiling 侧；kernel 骨架见 [kernel-template.md](kernel-template.md)。完整可编译代码见 [../example/](../../example/)。

## 1 输入

轴折叠结果 (A1, R, A0=1)、dtype（eb 字节）、平台量（coreNum / originCoreNum / ubSize / vRegSize / VL）。模式判定已完成（[../../mode-mapping.md](../../mode-mapping.md) §2 §3），本模板承接 `mode == AR_CUT_A`。

## 2 切分计算

### 2.1 A1 切核（均衡）

```cpp
realCoreNum_ = std::min(A1, coreNum_);              // 输出外层并行度上限
blkFactor_     = A1 / realCoreNum_;                 // 每核行数
blkTailFactor_ = A1 % realCoreNum_;                     // 余数摊派（工业口径）：前 blkTailFactor 核各 +1
loopANum_      = CeilDiv(本核行数, cutAPerLoop);    // 外层循环编排
```

### 2.2 R 切块（UB 双通道预算）

```cpp
// 全载条件：R × (eb + 4B位置通道) ≤ UB 可用（AR 手法 B 位置流在寄存器，UB 侧只算值流 + tmp）
// 工业实现按行整载（LoadUnAlign），R 超出时切 R：
cutRSize_ = min(rSize, ubRibbonLimit);              // ubRibbonLimit 按 ubSize 反解
// 跨块合并：块基址 startR 逐块累加，ArgMaxV1(..., startR) 生成侧一次施加
```

**R 上界三查**：UB 三通道预算 / 块内偏移 ≤ 位置元素容量（fp16→uint16 即 65536，UB 反解天然满足仍须核对）/ int32 输出 2^31（块基址拼接全域）——块内超界才需分段（块基址宽整数）。

### 2.3 GATHER 特化判定（覆盖为 AR_GATHER 时）

```cpp
// 触发：R*eb < vRegSize && A1*eb >= coreNum*vRegSize && (R*eb) % 32 != 0
// 此时无需切 R（R 单向量放得下），切核逻辑同 2.1；kernel 侧走 Gather 转置路径
```

## 3 TilingData 填充

按 [../../overview.md](../../overview.md) §4 struct 逐字段填充：aSize/rSize/nextASize(=1)、cutASize/cutRSize、blkFactor/blkTailFactor、realCoreNum、tilingKey(=10001 或 20001)、loopANum/cutAPerLoop。`workSpaceSize = 0`（AR 无 workspace）。

## 4 产出校验

| # | 校验 |
|---|------|
| 1 | blkFactor × (coreNum−1) + blkTailFactor == A1（无空块） |
| 2 | cutRSize ≤ 65536（uint16 块内偏移容量；UB 反解 ≤ ~29152@216KB 天然满足）且块基址累加 < 2^31 |
| 3 | 固定输入 golden 对拍：tilingKey / cutRSize / blkFactor 与期望一致（UT） |
