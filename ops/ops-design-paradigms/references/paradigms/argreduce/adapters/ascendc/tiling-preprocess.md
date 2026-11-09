# ArgReduce Host 侧 Tiling 流程（AscendC）

> **层级**：AscendC 语言适配层。Host 侧（`_tiling.cpp`）的处理顺序与产物。模式判定逻辑见 [mode-mapping.md](mode-mapping.md)；此处只讲编排与落地。

## 1 处理顺序（固定流水线）

```
1. 平台信息：coreNum(物理)、ubSize、vRegSize（GetPlatformInfo 系）
2. 预处理：负轴归一化 → 空 tensor 短路 → 轴折叠 (A1, R, A0) → 动态核数（每 4KB 输入 1 核）
3. 主场景判定（mode-mapping §2）
4. 性能特化覆盖 + 修正（mode-mapping §3 §4，含 aRaMode 细分）
5. 切分计算：按模式计算 blkFactor/blkTailFactor（A1 切核）、
   双轴时 blkFactor2nd/blkNum2nd（A0 切核）、cutRSize（R 切块）、cutAPerLoop/loopANum
6. GROUP_REDUCE 配套：isRaSplit 分支判定、workspace 布局与 workSpaceSize、SetScheduleMode(1)
7. 填充 TilingData：逐字段写入（§4 来源表）+ tilingKey 双写
8. SetTilingKey / SetBlockDim / workspace 总申请（workSpaceSize + 16MB）
9. 判定 trace 落日志（每步条件），供设计评审复现
```

## 2 关键计算

### 2.1 动态核数与物理核数

```cpp
coreNum_ = std::min(std::max(1UL, CeilDiv(dataBytes, 4096)), originCoreNum_);
originCoreNum_ = coreNum;   // 保存物理核数：GROUP_REDUCE 全链路（判定+SyncAll）使用
```

### 2.2 A1 切核（均衡，禁止截断式）

```cpp
blkFactor_     = A1 / realCoreNum;                       // 每核块大小
blkTailFactor_ = A1 % realCoreNum;                       // 余数摊派（工业口径）：前 blkTailFactor 核各 +1，负载差 ≤ 1
```

### 2.3 R 切块（UB 预算，双通道）

```cpp
// 值流 + 位置流(float32) 双通道 + tmp（INDEX_TRACKING_REDUCE scratch，如适用）
cutRSize = min(R, UB 预算反解上限);   // 跨块合并时块基址传入 startR（生成侧偏移）
```

### 2.4 GROUP_REDUCE 分支与 workspace

```cpp
isRaSplit = (A0 * eb >= 128 || blkFactor * eb < VL);   // RA 分支（全核聚合）；AR 分支 UB 不足时 fallback
// workspace：
//   AR 分支：outAAlign = CeilAlign(A1×A0, 256/eb)，大小 = outAAlign × 2 × 4B × originCoreNum
//   RA 分支：[value: A1×core×CeilAlign(A0,128/eb)][indice: A1×core×CeilAlign(A0,128/4B)]
```

## 3 值依赖三处契约

tiling 计算值（cutRSize 等）跨 Host/Kernel 传递遵循公共知识 `common/value-depend-process.md`：TilingData 字段是唯一真值源，kernel 侧禁止重算或猜测 Host 意图。

## 4 TilingData 字段来源表

| 字段 | 来源 | 消费方 |
|------|------|--------|
| aSize / rSize / nextASize | 轴折叠 | 全部模式 |
| cutASize / blkFactor / blkTailFactor / loopANum / cutAPerLoop | A1 切核计算 | AR / ARA / Gather |
| cutRSize | R 切块（UB 预算） | AR Split / ARA MODE3/4 / GROUP_REDUCE |
| cutNextASize | A0 切块（RA 批宽反解：`blkFactor_ = A1×CeilDiv(A0, cutNextASize)/realCoreNum`；A0 切核块大小） | **RA_CUT_A（20003）**、GROUP_REDUCE RA 分支（isRaSplit=1）、双轴切核（10012）；ARA MODE4 切 A0 |
| blkFactor2nd / blkTailFactor2nd / blkNum2nd | A0 切核（双轴专用因子） | 双轴切核（10012） |
| realCoreNum | 动态核数（GROUP_REDUCE 下为物理核口径） | 全部 |
| tilingKey / aRaMode | 模式判定 | 框架分发 / ARA 细分 |
| isRaSplit / gatherBlockSize / useFlatPath | GROUP_REDUCE 分支 / Gather 分块 / flat 子路径 | 对应模式 |
| workSpaceSize | GROUP_REDUCE 布局计算 | 框架 workspace |

## 5 校验

tiling 产出须通过独立 UT（golden：固定输入的 cutRSize/blkFactor/tilingKey 期望值），参考 example 包 `tests/ut/op_host` 结构。
