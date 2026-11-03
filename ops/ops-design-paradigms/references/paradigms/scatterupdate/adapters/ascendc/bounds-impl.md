# ScatterUpdate 越界三形态 AscendC 写法

> **层级**：AscendC 语言适配层。语言中立的决策方法论（优先级 / 二维独立 / 竞品证据 / 哨兵契约 / 双模板一致性）见 [../../generic/bounds.md](../../generic/bounds.md)；本文件只承载 AscendC 的 kernel 写法。
>
> 统一前提：**j 的一切变换（归一化）必须在 IdxT 原值域进行，窄化截断永远最后**（防漏判原则，generic/bounds.md §2）。

---

## 1 三种形态 kernel 写法

```cpp
Index j = indexGm[pos];                                   // 原值读入

// error（默认，torch 系）：原值比较，硬失败
if (j < 0 || j >= varDims[d]) { __trap(); }              // det/nondet 同语义

// skip（TF/Ascend tensor_scatter 系）：标记哨兵，Phase3 跳过不写入
if (j < 0 || j >= varDims[d]) { linearIdxOut[pos] = SENTINEL_KEY; continue; }

// neg-norm（负索引归一化 + 上界报错，按竞品声明）
if (j < 0) { j += varDims[d]; }                          // 原值域归一化
if (j < 0 || j >= varDims[d]) { __trap(); }              // 归一化后仍负 → 报错
```

## 2 nd 形态逐维判定写法

nd 形态每 update 有 K 维坐标，**逐维检查、任一维越界即整行失效**：

```cpp
bool valid = true;
int64_t lin = 0;
for (int32_t d = 0; d < K; ++d) {
    int64_t j = static_cast<int64_t>(indices[pos * K + d]);
    if (j < 0 || j >= varDims[d]) { valid = false; break; }   // error 形态此处 __trap()
    lin += j * varStrides[d];
}
linearIdxOut[pos] = valid ? static_cast<KeyT>(lin) : SENTINEL_KEY;
```

neg-norm 形态在 nd 下**逐维独立归一化**（不存在「整体归一化」场景，generic/bounds.md §2）：

```cpp
for (int32_t d = 0; d < K; ++d) {
    int64_t j = static_cast<int64_t>(indices[pos * K + d]);
    if (j < 0) { j += varDims[d]; }                    // 每维独立归一化（原值域）
    if (j < 0 || j >= varDims[d]) { valid = false; break; }   // 归一化后仍越界 → error 形态 __trap()
    lin += j * varStrides[d];
}
```

## 3 双模板实现的 AscendC 细节

1. det 有排序用**哨兵沉底**跳过；nondet 无排序用 `continue` 直跳——行为一致非逐字（[../../generic/bounds.md](../../generic/bounds.md) §5）。
2. error 形态：运行时 `__trap()`（`utils/debug/asc_assert.h`），禁止 assert 宏（release 静默通过）。
3. 哨兵 `SENTINEL_KEY = std::numeric_limits<KeyT>::max()`；keyDtype 按 totalVarElems 自适应（≤32767→int16 / ≤2^31-1→int32 / int64），`AscendC_TPL_UINT_DECL(keyDtype, ...)` 场景位承载。
