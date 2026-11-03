# IndexGather A 型：单坐标整维切片（index_select 型）

> 每个索引值为单一 dim 坐标，收集整切片（index 任意 rank，寻址展平为 M）。代表算子：index_select / take（flatten 变体）。

---

## 1 语义

```
input:  (..., D_dim, ...)        index: 任意 rank（M = prod），值域 [0, D_dim)
output: dim 维替换为 index.shape：input.shape 去掉 dim 插入 index.shape
out[..., m, ...] = input[..., index[m], ...]     （m 为 index 空间坐标）
```

无广播（batch_dims 组合除外：前缀相等，见下）；**两段快除（o/j/inner）即合轴闭式解**——dim 隔离 + 高段/低段各自折叠到极致的解析形式（无广播 ⟹ 段结构固定 ⟹ 可闭式），无需通用 MergeAxis 切段算法；若未来出现带广播语义的 A 型变种，回退 [tiling-preprocess.md](tiling-preprocess.md) §2 通用规则。

**index 多维推广**（`tf.gather` / `ONNX Gather` 语义）：indices 为任意 rank 时

```
Gather(p, idx, axis) ≡ reshape(index_select(p, axis, idx.reshape(-1)),
                                p.shape[:axis] + idx.shape + p.shape[axis+1:])
```

**kernel 零改动**——两段快除中的 `M := prod(index.shape)` 即覆盖多维 index；标量 index（rank-0）为 M=1 特例。outSize = prod(output.shape) 公式不变。

**边界**：index dtype 仅 int32/int64；**bool index 拒收**（gather 系语义要求整数索引，布尔掩码走 masked_select / 高级索引 decompose 路径）。

**batch_dims 组合**（附加结构轴，b = batch_dims ≥ 0，分型不变仍为 A 型；定义见 overview §1.1）：

```
input: (B..., ..., D_axis, ...)    index: (B..., M...)    前缀相等：index.shape[:b] == input.shape[:b]
output: input.shape[:axis] + index.shape[b:] + input.shape[axis+1:]    （axis ≥ b，负值归一化后）
out[batch, ..., m, ...] = input[batch, ..., index[batch, m], ...]
```

官方等价式（TF `manually_batched_gather`）：**逐 batch 视图上的非批 gather**——input[batch] 视图上以 index[batch]（任意 rank M-shape）做标准 A 型插入收集；axis 可为 ≥ b 的任意值（不限于 axis == b）。约束：前缀逐维相等（相等非广播，双端校验）；axis 归一化后 ∈ [b, rank)。

## 2 TilingData 字段（A 型零数组设计）

A 型两段快除闭式结构不消费任何 strides/dims 数组，**TilingData 不定义数组字段**：

```cpp
#pragma pack(push, 1)
struct XxxTilingData {
    int32_t dim;               // 归一化收集轴 ∈ [0, rank(data))
    int64_t outSize;           // 输出元素总数 = outer × M × inner
    int64_t gatherDimSize;     // data.shape[dim]（越界判定上界 bound）
    int64_t gatherSize;        // M = prod(index.shape[b:])（展平索引数；标量 → 1）
    int64_t innerSize;         // prod(data.shape[axis+1:])（切片内元素数；axis 末维 → 1）
    // batch_dims 组合追加（b > 0 时，语义含 batch_dims 才加）：
    int32_t batchDim;          // b = batch_dims
    int64_t paramsBatchStride; // prod(data.shape[b:])（batch 视图基址步长）
    int64_t innerTotal;        // outSize / batchNum（batch 段解码除数，直传免 Process 除法）
};
#pragma pack(pop)
```

> 设计原则（overview.md §2.1）：字段按分型需要设计，A 型不定义 `inputRank`（不消费 rank）、不定义 `outStrides/inStrides`（不消费数组）。禁止为 TTK 长度统一保留死重字段。batch 组合三字段全标量，**零数组设计保持**；输出秩界不变（batch 前缀在 input/index 两侧共享，rank(data)+rank(index)-1 不变）。A 型输出秩可达 15，超出 MAX_RANK=8 数组容量——零数组设计天然规避此边界。

## 3 Kernel：按 axis 拆轴 + 越界 + 字节搬移

A 型按 axis 位置将输出线性下标 i 分解为 outer / M / inner 三段（axis 前缀折叠、indices 展平、axis 后缀折叠），**快除次数 = 拆轴段数 - 1**（每段边界一次 UintDiv），由拆轴结果自然决定。

```cpp
// 每输出元素 i 的坐标：o = i / (M * innerSize)，j_idx = (i / innerSize) % M，inner = i % innerSize
IDX_T q = AscendC::Simt::UintDiv<IDX_T>(i, mInner, shiftInner);      // q = i / innerSize
IDX_T o = AscendC::Simt::UintDiv<IDX_T>(q, mGather, shiftGather); // o = q / M
IDX_T jIdx = q - o * gatherSize;                                     // j_idx = q % M
IDX_T inner = i - q * innerSize;                                     // inner = i % innerSize

INDEX_T k = indexGm[jIdx];                                           // 读 index（原值读入）
// 越界处理：此处为默认 none + err 预设示例；Step 1/Step 2 策略见 decode-bounds §1/§2，写法见 decode-bounds-impl §3
if (k < 0 || k >= gatherDimSize) { __trap(); }                        // 原值域比较（decode-bounds §2）
IDX_T kU = static_cast<IDX_T>(k);                                     // 检查通过后截断（安全）

IDX_T src = (o * gatherDimSize + kU) * innerSize + inner;
outputBytes[i] = inputBytes[src];                                    // 字节级 PackT 搬移
```

- 快除次数由拆轴决定（上例 3 段 = 2 次 UintDiv）；magic/shift 全部 Process 标量预计算直传，见 [common/uintdiv-rules.md](../../../common/uintdiv-rules.md) §2.2
- **特例退化**：take（flatten 索引）时 `innerSize = 1`、`M = inputNum`，拆轴退化为单段；dim=0 时 `o ≡ 0`，`src = kU * innerSize + inner`
- idxMode 双路径同通用模型（uint32 可全展开特化；uint64 高 M 场景保留运行期形态，见 [kernel-entrance.md](kernel-entrance.md) §3）
- **batch_dims 组合**：解码加 batchIdx 段（4 段 3 除）——`batchIdx = i / innerTotal`，`iB = i - batchIdx * innerTotal` 在 batch 视图内跑上述标准两段快除；index 行定位 `indexGm[batchIdx * M + jIdx]`（index 后缀即 M-shape 连续）；params 基址 `batchIdx * paramsBatchStride`。分派：Process 内 `batchDim > 0` 运行期分支（双级分派同 B 型，b=0 路径与基础结构逐字一致零退化）
- **合法值映射**（通用机制，非越界形态，见 [decode-bounds.md](../../generic/decode-bounds.md) §3）：特定合法索引值（∈ [0, dim)）→ 固定输出（默认零值；逐元素写零，整行即零向量）；语义声明时才加 TilingData 标量字段（按语义添加，非固定），置越界检查之后、寻址之前（合法值不与 trap 冲突）：

```cpp
if (legalIdx >= 0 && k == legalIdx) {       // -1 = 无该语义（host 归一化短路）
    outputBytes[i] = PackT(0);              // 当前输出元素写零；整行零向量由各元素各自置零组成
    continue;                               // 跳过寻址与搬移
}
```
