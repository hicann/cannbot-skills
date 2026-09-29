# Broadcast 范式 shape 归一化（补 1 / 去 1 / 归一 / 合轴）

> **层级**：通用知识层（generic/），语言中立。
>
> 来源：adapters/ascendc/tiling-preprocess.md §2/§3（原 broadcast-standard-tiling-preprocess.md）。C++ 参考实现（PadAndSqueeze / BroadcastMergeAxis / CheckBroadcastShape）与错误上报接口在适配层。

## 1 目标与产出

输入预处理是每个 Broadcast 算子都要做的通用步骤。目标：将不同 shape 的入参和出参归一化，拉到同一个参照系 `maximumBroShape` 下。

预处理产出三类 shape：

- `maximumBroShape` — 坐标系，所有输入/输出的 broadcast 上界
- `normalInputShapes` — 归一化后各输入 shape，与 maximumBroShape 同 rank
- `normalOutputShapes` — 归一化后各输出 shape，与 maximumBroShape 同 rank，输出可能比坐标系小

## 2 四步算法

> **顺序：补 1 → 去 1 → 归一 → 合轴，顺序不可换**

1. **补 1**：低 rank 入参/输出在 shape **最前面**补 1，拉齐到 maxRank（含入参+出参）。只能在 shape 最前面补 1，不允许在中间或末尾维度补 1
2. **去 1**：对齐后，所有入参+出参在某轴都是 1 → squeeze 掉，消除废维度
3. **归一**：去 1 后若 maximumBroShape 为空（全部均为标量），填 `(1,)`
4. **合轴**：归一后从最后一维向前贪心合并相邻维。对候选组 [d..groupHighDim]（闭区间），每个入参/出参在组内的维度乘积必须为 **1**（整组广播）或等于**坐标系 maximumBroShape 在组内的乘积**（整组稠密）；组内部分维为 1、部分维稠密的混合模式不能合并。扩展失败时封闭当前组，从当前维重新开组

**合轴的判定本质**：组内每个张量的乘积只有两种合法取值——`1`（该张量整组都是 1，合并后整维广播，stride=0）或 `坐标系组乘积`（该张量整组无 1，合并后整维稠密连续）。组内混有 1 和非 1（如某张量组内乘积为 `3×1×1=3`）说明广播边界落在组内部，合并后会破坏元素映射，必须拒绝。

**合轴的收益**：rank 与循环维数下降、内层连续段变长——单次搬运量从"最内维 × 少数几维"提升为"整组连续元素"，且 rank 超过搬运维数上限（`TRANSFER_MAX_DIMS`）时，split 轴更容易进入搬运单元的维数窗口，避免退化成逐次碎片化搬运。

**broadcast 兼容校验**：补 1 后逐维检查——对每个维度，若某入参/输出在该维的大小不为 1 且与其他入参/输出同维不为 1 的大小不同，则**报错并返回失败**。校验失败直接返回，不进入合轴——合轴必须在兼容校验通过之后执行（合轴的乘积判等依赖"每维取值 ∈ {1, 该维最大值}"这一前提）。

## 3 示例

例 1（正常广播，补 1 个 1）：
```
入参 A: shape = (4, 8)
入参 B: shape = (1, 8)
入参 C: shape = (8,)       // 补 1 → (1,8)

补 1 → A=(4,8), B=(1,8), C=(1,8)
去 1 → dim0: A=4, B=1, C=1 → 不全为 1，保留
        dim1: A=8, B=8, C=8 → 全为 8≠1，保留
maximumBroShape = (4, 8)
校验 → dim0: 4,1,1 → ref=4 ✓
        dim1: 8,8,8 → ref=8 ✓

合轴（从最后一维 dim1 开始，向左尝试扩展）：
  尝试 [dim0,dim1]: A=4×8=32, B=1×8=8, C=1×8=8, 坐标系组乘积 = 32
                     → B/C 的 8 既非 1 也非 32 ✗ → 封闭组 (8)，从 dim0 开新组
  尝试 [dim0]: 单维成组

合轴后: maximumBroShape = (4, 8)   — 无可合并，保持不变
        normal_A = (4, 8)
        normal_B = (1, 8)
        normal_C = (1, 8)
```

例 2（正常广播，补多个 1）：
```
入参 A: shape = (2, 4, 8)
入参 B: shape = (8,)       // 补 1 → (1,1,8)

补 1 → A=(2,4,8), B=(1,1,8)
去 1 → dim0: A=2, B=1 → 不全为 1，保留
        dim1: A=4, B=1 → 不全为 1，保留
        dim2: A=8, B=8 → 全为 8≠1，保留
maximumBroShape = (2, 4, 8)
校验 → dim0: 2,1 → ref=2 ✓
        dim1: 4,1 → ref=4 ✓
        dim2: 8,8 → ref=8 ✓

合轴（从最后一维 dim2 开始，向左尝试扩展）：
  尝试 [dim1,dim2]: A=4×8=32, B=1×8=8, 坐标系组乘积 = 32
                     → B 的 8 既非 1 也非 32 ✗ → 封闭组 (8)，从 dim1 开新组
  尝试 [dim0,dim1]: A=2×4=8, B=1×1=1, 坐标系组乘积 = 8
                     → A 稠密(8=maxProd)、B 整组广播(1) ✓ → 合并，当前组 = (8)

合轴后: maximumBroShape = (8, 8)
        normal_A = (8, 8)   — 原 dim0,dim2 合并为单维 8
        normal_B = (1, 8)   — 原前两维整组广播，合并后该维为 1
```

例 3（broadcast 不兼容报错）：
```
入参 A: shape = (4, 8)
入参 B: shape = (4, 3)

补 1 → A=(4,8), B=(4,3)     // 同 rank，无需补 1
去 1 → dim0: 4,4 → 不全为 1，保留
        dim1: 8,3 → 不全为 1，保留
maximumBroShape = (4, 8)
校验 → dim0: 4,4 → ref=4 ✓
        dim1: 8,3 → 两者均非 1 且 8≠3 → 报错！
→ 报错内容：dim 1、input[1] size 3 与 ref 8、broadcast incompatible: input sizes must be equal or 1
→ return false
（校验失败直接返回，不进入合轴）
```

例 4（标量）：
```
入参 A: shape = ()
入参 B: shape = ()

补 1 → A=(), B=()     // rank=0，无轴可补
去 1 → maximumBroShape = ()
归一 → maximumBroShape = (1,)
normal_A = (1,)
normal_B = (1,)
校验 → dim0: 1,1 → 全为 1 ✓
合轴 → rank=1，单维无可合并，保持 (1,)
```

例 5（合轴，从最后一维向前贪心合并）：
```
入参 A: shape = (2, 3, 4, 5)
入参 B: shape = (2, 3, 1, 1)
输出 C: shape = (2, 3, 4, 5)

补 1 → rank 已对齐，无需补 1
去 1 → 每维均非全 1，无废维
坐标系 maximumBroShape = (2, 3, 4, 5)

合轴（从最后一维 dim3 开始，向左尝试扩展）：
  尝试 [dim2,dim3]: A=4×5=20, B=1×1=1, C=4×5=20
                     → 乘积均为 20 或 1 ✓ → 合并，当前组 = (20)
  尝试 [dim1,dim2,dim3]: A=3×20=60, B=3×1=3, C=3×20=60
                     → 3 既非 1 也非 60 ✗ → 封闭组 (20)，从 dim1 开新组
  尝试 [dim0,dim1]: A=2×3=6, B=2×3=6, C=2×3=6
                     → 乘积全等 ✓ → 合并，当前组 = (6)

合轴后: maximumBroShape = (6, 20)
        normal_A = (6, 20)
        normal_B = (6, 1)     // 整组广播 → 合并后该维为 1
        normal_C = (6, 20)
校验 → dim0: 6,6,6 → ref=6 ✓
        dim1: 20,1,20 → ref=20 ✓
```

## 4 产出校验 golden

**对范式开发者的要求**：
1. 给一个复杂输入的样例，以合轴后的结果作为 golden，需要"人"保证正确性
2. 删除 golden，agent 读取本文档可以产出同样的合轴后的输出；产出不了则不断改进范式质量，直到成功
3. 样例随本文档上库

**golden 样例（Adam: 5 输入 3 输出, 输入 shape 全部一致）**：

```
输入:
  IN0 (input0):           (256, 128, 64)
  IN1 (input1):           (256, 128, 64)
  IN2 (input2):           (256, 128, 64)
  IN3 (input3):           (256, 128, 64)
  IN4 (input4):           (256, 128, 64)
  mul0_x/mul1_x/mul2_x/mul3_x/add2_y: (256, 128, 64)
输出:
  OUT0 (out0):            (256, 128, 64)
  OUT1 (out1):            (256, 128, 64)
  OUT2 (out2):            (256, 128, 64)

补 1 → 所有 tensor rank=3，无需补 1
去 1 → dim0: 256（非 1，保留）; dim1: 128（非 1，保留）; dim2: 64（非 1，保留）
归一 → 坐标系 maximumBroShape = (256, 128, 64)
校验 → 所有维度一致 ✓

合轴（从最后一维 dim2 开始，向左尝试扩展）：
  尝试 [dim1,dim2]: 各张量 = 128×64 = 8192，坐标系组乘积 = 8192
                     → 乘积均为 8192 = maxProd ✓ → 合并，当前组 = (8192)
  尝试 [dim0,dim1,dim2]: 各张量 = 256×8192 = 2097152，坐标系组乘积 = 2097152
                     → 乘积均为 2097152 = maxProd ✓ → 合并，当前组 = (2097152)

合轴后: maximumBroShape = (2097152,)
        normalInputShapes[0..9]  = (2097152,)  — 全部稠密连续
        normalOutputShapes[0..2] = (2097152,)
        inputStrides/outputStrides = (1,)
rank = 1 → 落入低档 RANK（rank ≤ 4）
```

**shape 不一致的样例（假设 IN1 shape=(128,64)）**：
```
补 1 → IN1=(1, 128, 64)，其他=(256, 128, 64)
去 1 → dim0: 256,1 → 不全为 1，保留
归一 → maximumBroShape = (256, 128, 64)
校验 → dim0: 256,1 → ref=256 ✓
        dim1/dim2: 全等 ✓

合轴（从最后一维 dim2 开始，向左尝试扩展）：
  尝试 [dim1,dim2]: IN1 = 128×64 = 8192，其他 = 8192，坐标系组乘积 = 8192
                     → IN1 整组稠密（8192 = maxProd）✓ → 合并，当前组 = (8192)
  尝试 [dim0,dim1,dim2]: IN1 = 1×8192 = 8192，坐标系组乘积 = 256×8192 = 2097152
                     → 8192 既非 1 也非 2097152 ✗ → 封闭组 (8192)，从 dim0 开新组
  尝试 [dim0]: 单维成组

合轴后: maximumBroShape = (256, 8192)
        normalInputShapes[1] = (1, 8192)      — dim0 broadcast，dim1 整组稠密
        inputStrides[1] = (0, 1)              — dim0 stride=0（broadcast）
        其他输入/输出 = (256, 8192)，strides = (8192, 1)
rank = 2 → 落入低档 RANK（rank ≤ 4）
校验（合轴后坐标系） → dim0: 256,1 → ref=256 ✓
                         dim1: 8192,8192 → ref=8192 ✓
```

## 5 适配层落地

| 事项 | 位置 |
|------|------|
| C++ 参考实现（PadAndSqueeze / BroadcastMergeAxis / CheckBroadcastShape） | adapters/ascendc/tiling-preprocess.md §2 |
| 错误上报接口与错误日志分层 | 各语言适配层定义（AscendC: `OP_LOGE_FOR_INVALID_*` 系列） |
| 平台/算子信息获取（值依赖、可选输入、attr 默认值） | 各语言适配层 tiling-preprocess 文档 |
