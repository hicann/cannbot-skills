# ARA 场景模板 — Kernel 骨架（AscendC）

> **层级**：AscendC 适配层模板。完整实现见 [../example/](../../example/) 的 `arg_max_with_value_ara.h` / `ara_gather.h` / `ra.h` / `ara_cut_a_and_next_a.h`。

## 1 类骨架与子模式分发

```cpp
template <typename TX, typename TIdxMid, typename TI, bool withValue, bool isMin>
class ArgReduceAra {
public:
    __aicore__ inline void Process() {
        switch (tiling_->aRaMode) {
            case ARA_MODE1: ProcessGatherARAWithCutA(); break;   // 101
            case ARA_MODE2: ProcessWithCutA();          break;   // 102: NDDMA 重排 → AR 手法
            case ARA_MODE3: ProcessGatherARAWithCutAR();break;   // 103: 切 R
            case ARA_MODE4: ProcessWithCutRA();         break;   // 104: 切 R + nextA
        }
    }
};
```

## 2 手法 D：逐行迭代（MODE1/3 核心，RA 布局）

```cpp
// UB 布局 [R, A0]：沿 R 逐行与累积胜者比较；A0 铺 lane
Duplicate(winVal, identity, mask);
Duplicate(winPos, 0, mask);
for (uint16_t r = 0; r < dimR; r++) {
    // curRow = 第 r 行（跨步载入：GM 基址 + r × A0 × eb，或重排后连续）
    Compare<CMPMODE::LE>(cmpMask, winVal, curRow, mask);      // argmax 方向（isMin 反转）
    Compare<CMPMODE::NE>(nanMask, curRow, curRow, mask);      // NaN 检测
    Or(cmpMask, cmpMask, nanMask, b8Mask);
    Select(winVal, winVal, curRow, cmpMask);
    Duplicate(level1, r + startR, mask);                      // 当前行位置（含块基址）
    Select(winPos, winPos, level1, cmpMask);
}
// 位置语义：winPos 为 R 行号（局部 + startR 基址）；输出时 Cast float→TI
```

## 3 手法 E：NDDMA 重排（MODE2）

```cpp
// [A1, R, A0] --NDDMA--> [A1, A0, R]：重排后每行 R 连续，复用 AR 手法 B（ArgMaxV1）
DataCopyPad(rearrangeUb, xGm_[offset], copyParams /*blockCount=A1×A0, blockLen=R*/);
ArgMaxV1(valUb, idxUb, rearrangeUb, dimA /*=A1×A0*/, dimR, 0);
```

重排 buffer 与值/位置通道共同计入 UB 预算（tiling 侧已算）。

## 4 手法 F：RA 向量化（RA_CUT_A / Gather 子路径）

```cpp
Duplicate(laneIdx, 0, fullMask); Add(laneIdx, laneIdx, one, addMask);   // 0..VL-1 lane 列
for (uint16_t r = 0; r < dimR; r++) {
    Compare<GT>(cmpMask, curRow, winVal, mask);   // + NaN OR
    Select(winVal, winVal, curRow, cmpMask);
    Select(winPos, laneIdx, winPos, cmpMask);     // 位置 = lane 号（+ r×A0 行基址，输出侧换算）
}
```

**位置换算注意**：RA 布局下 winPos 的语义是"A0 lane 内位置"，全局位置 = 输出块基址 + lane 号（MODE1）或行号（手法 D），按子模式核对一次。

## 5 双轴切核（ARA_CUT_A_AND_NEXT_A）

- 核坐标 = (blockIdx / blkNum2nd, blockIdx % blkNum2nd)，分别对应 A1 块与 A0 块
- `useFlatPath`：A0 全体在单核内扁平处理时的轻量路径（免二维索引解码）
- 每核只产自己 (A1 块 × A0 块) 的输出，核间无依赖、无同步

## 6 Buffer / DAG 分析

| Buffer | 通道 | 说明 |
|--------|------|------|
| 值流（跨步或重排） | TX | MODE2 重排目标 buffer 独立计预算 |
| winVal / winPos | TX / 等宽整数（fp16→uint16） | 累积胜者双通道，选择段常驻（掩码共享，同 lane 宽） |
| 搬出（值+位置） | TX / TI | 位置 Cast 后落 GM |

**同步**：核内一条流水；**无跨核同步**（全部 ARA 路径核间独立）。

## 7 移植对照表

| example | 本范式 |
|---|---|
| `ArgMaxWithValueArA` | ArgReduceAra |
| `ProcessGatherARAWithCutA / WithCutA / GatherARAWithCutAR / WithCutRA` | MODE1~4（§2 §3） |
| `ArgMaxAra`（base.h） | 手法 D（§2） |
| `ArgMaxGatherRa`（base.h，AR→RA 构造） | 手法 C/F（§4） |
| `arg_max_with_value_ara_cut_a_and_next_a.h` | §5 双轴切核 |
