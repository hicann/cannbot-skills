# IndexGather B 型：N 元组索引（gather_nd 型）

> indices 末 K 维为坐标元组，索引 params 前 K 维；支持 batch_dims 扩展。代表算子：gather_nd。

---

## 1 语义

```
params:  (P1..Pb, D_1..D_K, S...)      indices: (B..., N_result, K)
K = indices.shape[-1]，合法范围由 spec/需求确定；语义约束 0 <= K <= rank(params) - batch_dims
batch_dims = b（默认 0）
output.shape = indices.shape[:b] + indices.shape[b:-1] + params.shape[b+K:]
out[batch, n, s...] = params[batch, index[n][0..K-1], s...]
```

K 的合法范围由 `spec`/需求确定，范式不设固定上限；下文用 `Kmax` 表示需求给定的编译期特化上限，它不是范式常量。`K=0` 是否支持由需求决定：支持时走 copy 特化，不支持时由接口层拒绝。

无广播（batch_dims 除外：要求 `indices.shape[:b] == params.shape[:b]`）。batch_dims 为 A/B 型附加结构轴（定义见 overview §1.1）：B 型×batch 与 A 型×batch（[type-a.md](type-a.md) §1）是该结构轴的两个实例，机制差异 = batchIdx 解码后接 **K 元组逐分量解码** vs 接**整切片两段快除**。

## 2 TilingData 差异字段（数组定长 MAX_RANK，kernel 按 K 读前缀）

```cpp
int32_t k;                    // 索引深度 K = indices.shape[-1]，数组有效长度
int32_t batchDim;             // batch_dims b
int64_t innerTotal;           // prod(indices.shape[b:-1]) * sliceSize（batch_idx 解码除数）
int64_t paramsBatchStride;    // prod(params.shape[b:])（params batch 基址步长）
int64_t sliceSize;            // prod(params.shape[b+K:])（索引元组指向的切片大小）
int64_t dims[MAX_RANK];       // params.shape[b:b+K] 前 K 维尺寸（兼逐维越界上界），尾填 0
int64_t strides[MAX_RANK];    // params.shape[b:b+K] 前 K 维步长（偏移累加），尾填 0
```

> 数组定长声明（overview.md §2.1 原则 2）：dims/strides 按 MAX_RANK 定长声明；MAX_RANK 的值由需求的 rank/K 上限确定（示例取 8），不是范式上限。有效长度 = K，kernel 只读前 K 项，尾部填 0（安全值）——定长结构体免变长序列化，越界读不发生（前缀截断）。

## 3 Kernel：双级分派（batchDim × K）+ 四 kernel + batch_dims 三连解码

Process 内**双级分派**：第一级按 `batchDim > 0` 分派（batch 场景多一次 batchIdx 解码），第二级按 `k` 编译期特化（`asc_vf_call` 运行期选编译期实例）。两级组合出**四类 kernel 模板**：

| kernel | batchDim | K | 说明 |
|--------|:-------:|:---:|------|
| `GatherNdCopyKernel` | b=0 | K=0（需求允许时） | 整 params 按 indices 前导形状复制（无索引读、无越界） |
| `GatherNdSliceKernel<K>` | b=0 | K∈[1,Kmax]（Kmax 由需求确定；示例覆盖 1..7） | 通用 gather_nd（batchIdx ≡ 0） |
| `BatchGatherNdCopyKernel` | b>0 | K=0（需求允许时） | batch 版整块复制 |
| `BatchGatherNdSliceKernel<K>` | b>0 | K∈[1,Kmax]（Kmax 由需求确定；示例覆盖 1..7） | batch 版三连解码 |

```cpp
// Process 内双级分派（idxMode 双重载之上）
if (tilingData->batchDim > 0) {
    switch (tilingData->k) {
        case 0:  asc_vf_call<BatchGatherNdCopyKernel<T, IDX_T>>(...); break;
        case 1:  asc_vf_call<BatchGatherNdSliceKernel<T, Index, IDX_T, 1>>(...); break;
        // ... case Kmax（示例到 7；上限由需求确定）
    }
} else {
    switch (tilingData->k) {   // b=0 分支与无 batch 版逐字一致（零性能退化）
        case 0:  asc_vf_call<GatherNdCopyKernel<T, IDX_T>>(...); break;
        case 1:  asc_vf_call<GatherNdSliceKernel<T, Index, IDX_T, 1>>(...); break;
        // ... case Kmax（示例到 7；上限由需求确定）
    }
}

// GatherNdSliceKernel<K> 主体（b=0；Batch 版多一步 batchIdx 解码，见下）
IDX_T loc = AscendC::Simt::UintDiv<IDX_T>(i, magic, shift);    // i / sliceSize
IDX_T m = i - loc * sliceSize;                                  // 切片内偏移

__gm__ Index* indexRow = indices + loc * K;
bool outOfBounds = false;
IDX_T offset = 0;                                               // 偏移算术统一 IDX_T（窄化截断最后）
#pragma unroll
for (int32_t j = 0; j < K; ++j) {                               // 逐分量解码坐标元组
    Index indexJ = indexRow[j];                                  // 原值读入
    // 本段为 none + fill 预设：Step 2 fill 统一标记；其他 Step 1/Step 2 组合见 §4
    outOfBounds |= (indexJ < 0 || static_cast<int64_t>(indexJ) >= dims[j]);
    offset += static_cast<IDX_T>(strides[j]) * static_cast<IDX_T>(indexJ);
}
if (outOfBounds) {
    outputBytes[i] = fillBytes;                    // fill 位模式字节写（来源/写法见 decode-bounds §2）
} else {
    outputBytes[i] = inputBytes[offset + m];     // PackT 字节搬移（inputBytes/outputBytes 按 PackT 重解释 GM）
}

// Batch 版在 loc 解码前先拆 batchIdx：
IDX_T batchIdx = AscendC::Simt::UintDiv<IDX_T>(i, magic2, shift2);  // i / innerTotal
IDX_T inner = i - batchIdx * innerTotal;
// loc = inner / sliceSize；m = inner - loc * sliceSize
// indexRow = indices + (batchIdx * nInner + loc) * K
// params 基址 = batchIdx * paramsBatchStride
```

**batch_dims 三连解码（batchIdx/loc/m）即合轴闭式解**——batch 维全折叠进高段（innerTotal）+ K 维逐维隔离 + params 尾维全折叠进低段（sliceSize），已是合轴最优折叠（无广播 ⟹ 段结构固定 ⟹ 可闭式），无需通用 MergeAxis 切段算法；若未来出现带广播语义的 B 型变种，回退 [tiling-preprocess.md](tiling-preprocess.md) §2 通用规则。

**关键点**：

- **K 编译期特化**：`#pragma unroll for (j < K)` 需要编译期 K；K=0 为"整 params 按 indices 前导形状复制"特化（无索引读、无越界，需求允许时保留）；K>=1 走 slice 特化，`Kmax` 由需求确定
- **batch_dims 三连解码**：batchIdx（innerTotal 快除）→ loc（sliceSize 快除）→ m；indices 行定位 `(batchIdx*nInner + loc)*K`，params 基址 `batchIdx*paramsBatchStride`
- **dims/strides 按 `params.shape[b:]` 视角填充**（tiling 侧），kernel 越界检查代码与 b=0 逐字一致
- `b=0` 退化为普通 gather_nd（batchIdx ≡ 0）

## 4 越界处理（逐分量写法）

越界按两步模型决策（策略与业界证据见 [decode-bounds.md](../../generic/decode-bounds.md) §1/§2，写法见 [decode-bounds-impl.md](decode-bounds-impl.md) §3）。B 型为**逐分量判定**——每分量上界各异（`dims[j]`），Step 1 与 Step 2 都须逐分量执行；`err` 可逐分量立即失败，`fill` / `skip` 结果累积后统一分支：

```cpp
// 逐分量两步处理（+dim + err 预设示例）
Index indexJ = indexRow[j];
if (indexJ < 0) { indexJ += static_cast<int64_t>(dims[j]); }             // Step 1: +dim
if (indexJ < 0 || static_cast<int64_t>(indexJ) >= dims[j]) { __trap(); } // Step 2: err
offset += static_cast<IDX_T>(strides[j]) * static_cast<IDX_T>(indexJ);   // 变换后窄化截断，偏移累加用 IDX_T
```

要点：

- Step 1 / Step 2 均逐分量进行（各维上界不同，不能整体判一次）
- 同一分量内先 Step 1 负索引处理，再 Step 2 越界处理；Step 1 / Step 2 策略可按 spec 独立组合
- Step 2 为 `fill` 时改为 `outOfBounds = true`，循环后统一写 `fillBytes`；为 `skip` 时同样先标记，循环后跳过整个 update（不能只在分量循环内 `continue`，否则只跳过单个分量）
- 残留越界的 offset 为废值，被统一分支丢弃，无害
- 原值域变换、窄化截断最后（强制规则 3）
