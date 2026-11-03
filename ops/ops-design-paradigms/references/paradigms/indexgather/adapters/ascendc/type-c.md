# IndexGather C 型：逐元素对齐（gather 型）

> index 与 output 同 shape，dim 单维坐标被 index 值替换；支持广播。代表算子：gather（torch.gather）。

---

## 1 语义

```
input:  (P_0..P_{R-1})      index: (I_0..I_{R-1})，d != dim 时 I_d <= P_d
output.shape = index.shape
out[c_0..c_{dim-1}, c_dim, c_{dim+1}..] = input[c_0.., index[...c...], ..c_{dim+1}..]
```

dim 维坐标被 index 值替换，其余维与输出位置对齐（广播场景下对齐坐标天然落在前缀区域）。

## 2 TilingData 字段（C 型，数组定长 MAX_RANK，kernel 按 rank 读前缀）

```cpp
int32_t dim;                        // 归一化收集轴（合轴后视图值）
int32_t inputRank;                  // 合轴后段数 mergeRank（数组有效长度 = mergeRank-1）
int64_t outSize;                    // 输出元素总数
int64_t gatherDimSize;              // input.shape[dim]（越界上界）
int64_t inStrideDim;                // input 的 dim 维 stride
int64_t outStrides[MAX_RANK];       // index 累积步长（坐标解码除数），有效 mergeRank-1 项，尾填 1
int64_t inStrides[MAX_RANK];        // input 累积步长（偏移累加），有效 mergeRank-1 项，尾填 1
```

> 数组定长声明（overview.md §2.1 原则 2）：outStrides/inStrides 按 MAX_RANK=8 定长声明（可按需调整），kernel 只读前 mergeRank-1 项，尾部填 1（安全值：步长 1 为恒等因子，即使被误乘不改变偏移）。合轴后 mergeRank 通常为 2~4，死空间为 8-(mergeRank-1) 个 int64，换取定长结构体与序列化确定性。

## 3 Kernel：合轴后 rank'-1 次解码

通用解码骨架即 [decode-model.md](../../generic/decode-model.md) §1（逐维 UintDiv + `if (d != dim)` 跳过 + 最后一维 + `inOff += jU * inStrideDim`）。本型要点：

- **tiling 侧先合轴**（[tiling-preprocess.md](tiling-preprocess.md) §2）：kernel 收到的 inputRank/dim/strides 均为合并后视图，解码次数自动降为 `rank'-1`
- **Process 按 inputRank switch 特化**（合并后 rank' 越小，实例越少；合轴收益直接体现为解码次数下降——实测混合广播大 shape 场景 -1.6%~-38.5%，rank 8→2 场景 -59%）
- **不可合轴场景**（全广播型/相等维孤岛型）：mergeRank = rank 原样，走 rank-1 次全维解码，无退化

## 4 广播正确性的根基：两套步长分离

```
解码（i → 坐标）：outStrides[d] = prod(index.shape[d+1:])   ← index 空间
偏移累加：        inStrides[d]  = prod(input.shape[d+1:])   ← input 空间
```

广播使 `outStrides[d] != inStrides[d]`（d < dim 时含 dim 维因子差异）——kernel 必须分离消费两套数组，禁止合并优化。合轴的段头广播豁免正是建立在该分离上（段 prod 两空间分别计算）。

## 5 退化场景核对表

| 场景 | 行为 | 核对点 |
|------|------|--------|
| 非广播全等 | 高段+低段各并 1 根 → rank'=2~3 | inputRank 合轴后值 |
| 段头广播 | 段头维并入合并轴（豁免） | 合并轴两空间 prod 可不同 |
| 孤岛型（相等维被广播隔断） | 全单根，rank'=rank | 无段长≥2 |
| rank=1（dim=0） | 无段可合；解码 0 次除法 | inOff = j 直达 |
| 空输出（index 前导维含 0） | outSize=0 早退 | 早退先于快除预计算 |
