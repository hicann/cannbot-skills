> **层级**：AscendC 语言适配层（adapters/ascendc/）。指令的语义、参数限制与数据流是语言中立的，见 [../../generic/compute-flow.md](../../generic/compute-flow.md) §4；本文件只收录**与 AscendC 语言/工具链强耦合**的落地知识。参考实现见 [example/conv2d/](example/conv2d/)。

# 指令 API 落地要点

## 1 硬件指令 ↔ AscendC API 对照表

| # | 硬件指令 | AscendC 调用形式 | 参数结构体 |
|---|---------|-----------------|-----------|
| 1 | DN2NZ（GM→L1 布局转换） | `DataCopy(a1, gm[off], Dn2NzParams)` | `Dn2NzParams{dnNum, nValue, dValue, srcDnMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride}` |
| 2 | LOAD3D（L1→L0A，滑窗展开） | `LoadData<TPosition::A2, TPosition::A1, T>(a2, a1, Load3DBitModeParam)`<br>全局配置：`SetFmatrix(l1H, l1W, padList, mode)` 或 `Load3DSetFMatrixCal`；`SetLoadDataRepeatWithStride(rp)` | `Load3DBitModeParam`（Set 系列方法填充位域） |
| 3 | LOAD2D（L1→L0B，子块抽取） | `LoadData<TPosition::B2, TPosition::B1, T>(b2, b1, Load2DBitModeParam)` | `Load2DBitModeParam{MStartPosition, MStep, KStartPosition, KStep, SrcStride, DstStride, IfTranspose}` |
| 4 | MMAD（L0A × L0B → L0C） | `Mmad<L0cT, T, T>(c, a, b, MmadParams)` | `MmadParams{m, n, k, cmatrixInitVal, cmatrixSource, unitFlag}` |
| 5 | FIXPIPE（L0C → GM） | `Fixpipe<OutT, L0cT, CFG_COLUMN_MAJOR>(gm[off], c, params)` | `FixpipeParamsArch3510<COLUMN_MAJOR>{nSize, mSize, srcStride, dstStride, params.*, quantPre, deqScalar, unitFlag}` |
| 6 | L1→BT（bias 装载） | `DataCopy(c2, a1[off], DataCopyParams)` | `DataCopyParams(blockCount, burstLen(32B 块), 0, 0)` |

## 2 位域类参数的填充方式

LOAD3D / LOAD2D 的参数结构体为位域类（BitModeParam），两种填充方式：

| 方式 | 写法 | 适用场景 |
|------|------|---------|
| **Set 系列方法**（推荐） | `a3.SetStrideW(...); a3.SetFilterH(...); ...` | 逐字段填值，可读性好；与参考实现对齐时优先 |
| SetConfig0 / SetConfig1 整值下发 | `a3.SetConfig0(0x12345678);` | 性能敏感路径（少一次寄存器写入），需查位域定义 |

## 3 使用要点

- 字段的**语义、单位、限制**见 [compute-flow.md](../../generic/compute-flow.md) §4 各指令小节（语言中立层），本文件不重复；
- FIXPIPE 的列主序 / 行主序选择由输出 format 决定：NCHW 输出走 `CFG_COLUMN_MAJOR`；
- Load3D 的 FMatrix 全局配置（`SetFmatrix`）与块定位参数（`Load3DBitModeParam`）**分开下发**——前者在装载 AL1 后设置，后者随每次 LoadData 调用传入。
