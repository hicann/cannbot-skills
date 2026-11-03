# IndexGather 坐标解码与越界检查：AscendC 写法

> **层级**：AscendC 语言适配层。语言中立的决策方法论（优先级 / 两步模型 / 竞品证据）见 [../../generic/decode-bounds.md](../../generic/decode-bounds.md)；坐标解码模型与合轴算法见 [../../generic/decode-model.md](../../generic/decode-model.md)；本文件只承载 AscendC 的 kernel 写法。设计文档 §4.2 的真值源。

---

## 1 坐标解码（C 型通用，AscendC）

每输出元素一线程，行主序前向逐维解码：

```cpp
// VF 端（__simt_vf__），IDX_T 为 uint32/uint64
IDX_T stride = gridDim.x * blockDim.x;              // 全局 Grid-Stride
for (IDX_T i = blockIdx.x * blockDim.x + threadIdx.x; i < outSize; i += stride) {
    INDEX_T j = indexGm[i];                          // 1. 读 index 值
    /* 2. 越界检查（见 §2） */
    IDX_T jU = static_cast<IDX_T>(j);
    IDX_T rem = i;
    IDX_T inOff = 0;
    #pragma unroll
    for (int32_t d = 0; d < RANK - 1; ++d) {         // 3. 逐维解码（编译期 RANK）
        IDX_T c = AscendC::Simt::UintDiv<IDX_T>(rem, outMagic[d], outShift[d]);
        rem -= c * static_cast<IDX_T>(outStrides[d]);
        if (d != dim) {                              // dim 维坐标被 j 替代，跳过
            inOff += c * static_cast<IDX_T>(inStrides[d]);
        }
    }
    if (RANK - 1 != dim) {
        inOff += rem;                                // 4. 最后一维（stride 恒 1）
    }
    inOff += jU * static_cast<IDX_T>(inStrideDim);   // 5. dim 维贡献用 j
    outputBytes[i] = inputBytes[inOff];              // 6. 字节级搬移
}
```

**关键点**：

- 解码按 **index 空间**（outStrides/outMagic），偏移累加按 **input 空间**（inStrides）——两套步长分离是广播/合轴正确性的根基（[../../generic/decode-model.md](../../generic/decode-model.md) §1）
- `outMagic/outShift` 由 Process（scalar 端）逐维预计算经 UB 中转，`__ubuf__` 指针作 VF 参数（见 [../../../common/uintdiv-rules.md](../../../common/uintdiv-rules.md) §2.1）
- 字节级搬移：`inputBytes = reinterpret_cast<__gm__ PackT*>(inputGm)`，PackT 按元素宽 ∈ {uint8, uint16, uint32, uint64}，bit-exact

## 2 magic/shift 预计算（Process 内，scalar 端）

```cpp
for (int32_t d = 0; d < td->inputRank - 1; ++d) {
    outStridesPtr[d] = td->outStrides[d];
    IDX_T magic = 0;                                 // Local Memory 标量中转
    IDX_T shift = 0;                                 // （禁止直接 __ubuf__ 左值出参）
    AscendC::GetUintDivMagicAndShift<IDX_T>(
        magic, shift, static_cast<IDX_T>(td->outStrides[d]));
    outMagicPtr[d] = magic;
    outShiftPtr[d] = shift;
}
DataSyncBarrier<MemDsbT::UB>();                      // scalar 写 UB 后同步
```

## 3 两步模型 kernel 写法

统一前提（两步一致）：**j 的一切变换（Step 1 回绕/取模、Step 2 取模/钳位/截断）必须在 INDEX_T 原值域进行**，窄化截断（cast 到 IDX_T）永远最后（[../../generic/decode-bounds.md](../../generic/decode-bounds.md) §2）。

```cpp
INDEX_T j = indexGm[i];                                // 原值读入

// error（预设：Step 1 none + Step 2 err）
if (j < 0 || j >= bound) { __trap(); }

// neg-wrap（预设：Step 1 +dim + Step 2 err）
if (j < 0) { j += bound; }                             // Step 1：单次回绕
if (j < 0 || j >= bound) { __trap(); }                 // Step 2：回绕后仍越界 → 报错

// mod-wrap（预设：Step 1 none + Step 2 %dim）
j = j % bound;                                         // Step 2：真模（性能注记见 §5）
if (j < 0) { j += bound; }                             // C++ 负余数修正

// clamp（预设：Step 1 none + Step 2 clamp）
j = (j < 0) ? 0 : ((j >= bound) ? bound - 1 : j);      // Step 2：钳位到 [0, dim-1]

// fill（预设：Step 1 none + Step 2 fill；fillValue 默认 0）
// fillValue 位模式按字节搬运（PackT 字节宽），禁止 static_cast/数值转换：
//   - 标量 attr 来源：host 用 union/memcpy 位级打包进 TilingData 字节字段，device 写该字段
//   - tensor 来源：device 从 GM 按 PackT 字节宽读 fill 元素直写（与 copy 同构）
if (j < 0 || j >= bound) {
    outputBytes[i] = fillBytes;      // fillBytes：PackT 位模式（上述两来源之一产出）
    continue;
}

// undefined（预设：Step 1 none + Step 2 undefined）：不做逐元素检查
// 静态校验（dim 范围/广播约束）仍必须有；须 spec 显式声明风险（读越界内存行为自负）

IDX_T jU = static_cast<IDX_T>(j);                      // 变换/检查完成后截断（安全性见 §4）
```

其中 `bound` 即 `gatherDimSize`。Step 1 的 `%dim` 也可单独搭配 Step 2 `err`（负索引真模、上界报错）；Step 2 的 `skip` 用于 scatter，A/C 型把越界分支改为 `continue` 即可，B 型需在分量循环后跳过整个 update（见 [type-b.md](type-b.md) §4）。

## 4 cast 差异与截断安全性

| 重载 | 上界写法 | 依据 |
|------|---------|------|
| IDX32 特化 | `static_cast<INDEX_T>(bound)` | `use64Bit=false ⟹ inputNum/outSize/indexNum/taskNum` 均 ≤ INT32_MAX，且 `bound ≤ inputNum`，落在 int32 值域，cast 无损；INDEX_T=int64 时 cast 为 identity |
| IDX64 运行期 | **禁止 cast** | `use64Bit=true` 时 bound 可超 int32 值域（如 input=(2^32,2) dim=0），`static_cast<int32_t>(2^32)=0` 会令 "j >= 0" 恒真 → **合法 index 全部误报** |

截断安全性（IDX32 前提）：检查通过后 `j ∈ [0, bound)`，则 `jU·inStrideDim ≤ (input.shape[dim]-1)·prod(shape[dim+1:]) < inputNum ≤ INT32_MAX`——uint32 全程无溢出（`+dim` / `%dim` / `clamp` 变换后 j 同样落入 `[0, bound)`，结论一致）。

## 5 性能注记（AscendC 特有）

- **Step 2 `%dim`**：int64 取模一条（除法级成本，且负值域使 magic 快除不直接适用）——策略选型时考量
- **Step 2 `fill`**：越界分支内写标量 + continue，合法路径零开销
