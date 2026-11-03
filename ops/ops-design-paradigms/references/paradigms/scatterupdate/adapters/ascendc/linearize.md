# ScatterUpdate Phase1 线性化：冲突 key 生成（AscendC 写法）

> **层级**：AscendC 语言适配层。语言中立的冲突域模型（key/b/s 统一视角 / 排序量原则 / key 位宽自适应 / 空 tensor 早退）见 [../../generic/conflict-domain.md](../../generic/conflict-domain.md)；本文件只承载各 index 形态的 AscendC Phase1 写法与坐标计算域门控。设计文档 Phase1 部分的真值源。
>
> **与 Phase3 正交**：Phase1 只决定「冲突域」，不感知 Phase3 的并行分解（(b,k,s) 三场景见 [deterministic.md](deterministic.md) §3）；Phase3 只消费 `sortedKey + perm`，不感知 key 的合成方式。

---

## 1 index 覆盖全部维（dim 形态·逐元素子场景）：逐维坐标反解合成

index 与 var/src 同 rank，每散点一个独立 index 元素（示例：torch scatter(reduce=none) / ScatterElements），冲突域 = 目标**元素**：

```cpp
// Phase1 VF（__simt_vf__），pos 为散点编号 ∈ [0, N)，N = index.numel()
// IndexT = 坐标计算域（useInt32Index 五项计数门控，§6）；IdxT = 索引原始 dtype
IndexT rem = pos;
IndexT lin = 0;
for (int32_t d = 0; d < rank; ++d) {
    IndexT c = AscendC::Simt::UintDiv<IndexUT>(rem, idxDivMagic[d], idxDivShift[d]); // c = rem / indexStrides[d]
    rem -= c * static_cast<IndexT>(indexStrides[d]);   // rem %= indexStrides[d]
    if (d != dimNormalized) {
        lin += c * static_cast<IndexT>(varStrides[d]);  // 非 dim 维：坐标 × var 步长（IndexT 域）
    }
}
IdxT jRaw = indexGm[pos];               // 原值读入（原始 dtype，不参与降级）
if (jRaw < 0 || static_cast<int64_t>(jRaw) >= varDims[dimNormalized]) { __trap(); }  // 越界原值域判断（bounds-impl.md §1）
lin += static_cast<IndexT>(jRaw) * static_cast<IndexT>(varStrides[dimNormalized]);   // dim 维被 j 替换（检查通过后窄化）
linearIdxOut[pos] = static_cast<KeyT>(lin);
```

- **坐标计算全链路 IndexT 域**（rem/c/lin + 快除 + 坐标乘积）：useInt32Index 只看 var/src/out/index/task 五项元素计数是否均 ≤ INT32_MAX（见 §6）；**读索引与越界检查始终在 IdxT 原始 dtype 原值域**（[bounds-impl.md](bounds-impl.md)「原值域比较、窄化截断最后」），检查通过后才窄化进 IndexT。
- 两套步长分离：**反解用 index 空间步长（indexStrides），偏移累加用 var 空间步长（varStrides）**——前段子集（index.size(d) ≤ src.size(d)）时两空间不同构，与读收集范式 C 型「两套步长分离」同源；src 侧寻址见 §4（srcPos 双输出）。
- magic/shift 由 Process（scalar 端）预计算经 UB 中转，出参必须 Local 标量中转（禁止直写 `__ubuf__` 左值，见 [../../../common/uintdiv-rules.md](../../../common/uintdiv-rules.md) §2.1）。
- b=1、s=1（每散点单元素负载）→ Phase3 走 (k) 单维（(k,s) 的 s=1 特化再特化）。
- **规模路径风险提示**：本子场景逐维反解含 rank≤7 次 int64 快除（uint64 域时），若全部编译期展开，uint64 乘法多指令模拟在寄存器压力下存在溢出风险——生成 DESIGN 时须对快除循环做规模路径考量（窄域特化 / 运行期截断），本范式不展开具体方案。
- 形态验证覆盖（历史实证来自写散射族开发，机制同构）：覆盖全部维反解 + srcStrides 前段子集寻址（§4）+ useInt32Index 五项计数门控（var/src/out/index/task），rank 1~3 全编译通过。

## 2 index 覆盖单维（dim 形态·沿 dim 轴共享子场景）：index 值即 key

index 为 1D（长度 M）沿 axis，对所有非 axis 维共享（示例：index_copy_ / index_fill_ 类）——冲突域 = **仅 dim 维 j**，key 合成零成本：

```cpp
// Phase1 VF：归一化 + 越界检查 + 直写 key（零除法）
for (int64_t pos ∈ [0, M) Grid-Stride):
    int64_t j = static_cast<int64_t>(indices[pos]);   // 原值读入
    if (j < 0) { j += gatherDimSize; }                 // 负索引归一化（原值域，bounds-impl.md §1）
    if (j < 0 || j >= gatherDimSize) { __trap(); }     // 归一化后越界硬失败
    keyOut[pos] = static_cast<KeyT>(j);                // key = j（index 值本身）
```

- **排序量 = M**（不 × batch）——同一组 i→j 冲突对所有 (b, s) 复用，排序一次、Phase3 复用到全部 batch×slice。
- 独立并行维：`b = prod(shape[:dim])`、`s = prod(shape[dim+1:])`，Phase3 按 (b,k,s)/(k,s)/(b,k) 三场景展开（[deterministic.md](deterministic.md) §3）。
- TilingData 仅需四元标量 `gatherDimSize / indexSize / batchSize / sliceSize`——**零数组、零 dimNormalized 入 kernel**（host 算完四元标量后 dim 信息耗尽；dim 轴位置被 b/s 有无自然吸收）。

## 3 nd 形态：K 维坐标合成 1 维索引

indices (N, K) 末维 K 连续，每个 update 的坐标元组合成 1 维 lin——**合成后即 1 维索引，后续全程按 1 维处理**（排序 N 个 lin，s 是 lin 对应的连续块不进排序）：

```cpp
// Phase1 VF（__simt_vf__），pos ∈ [0, N)，N = prod(indices.shape[:-1])
int64_t base = pos * K;                 // indices 末维 K 连续，简单乘法无除法（GM 读偏移）
IndexT lin = 0;                         // 坐标计算域（useInt32Index 五项计数门控）
bool valid = true;
for (int32_t d = 0; d < K; ++d) {
    IdxT jRaw = indicesGm[base + d];    // 原值读入（原始 dtype，不参与降级）
    if (jRaw < 0 || static_cast<int64_t>(jRaw) >= varDims[d]) { valid = false; break; }  // 越界原值域判断（bounds 形态）
    lin += static_cast<IndexT>(jRaw) * static_cast<IndexT>(varStrides[d]);   // 坐标累加 IndexT 域
}
linearIdxOut[pos] = valid ? static_cast<KeyT>(lin) : SENTINEL_KEY;
```

- **零快除**（indices 末维连续直读，对照 index 覆盖全部维的 rank-1 次反解）。
- **坐标计算 IndexT 域**（lin 累加 + 坐标乘积，值域 totalVarElems）；**读索引与越界判断保持 IdxT 原始 dtype 原值域**（[bounds-impl.md](bounds-impl.md)，检查通过后窄化）——与 §1 同一纪律。
- `lin` 即目标 slice 在 var 中的起始偏移，`var.shape[K:]` 为最内层连续区域（s = prod(shape[K:]) 个连续元素）。
- b=1（坐标元组覆盖前 K 维，无独立前缀维）→ Phase3 走 (k, s) 二维。

### 展平 1D 退化（nd 的 rank=1 极限，put_/take 写侧）

var 视作 flatten 1D、index 为全局下标时（rank=1 / dim=0 / s=1），key = index 值本身（lin 的退化）：

```cpp
Index j = indexGm[pos];                      // 全局下标，原值读入
if (j < 0 || j >= totalVarElems) { /* 越界按 bounds 形态 */ }
linearIdxOut[pos] = static_cast<KeyT>(j);    // key = index 值本身
```

**零快除零反解**；KeyT 上界为 totalVarElems。

## 4 subset 前段子集寻址（srcPos 双输出）

**场景定义**：index 与 src 同 rank，但某维 `index.size(d) < src.size(d)`（前段子集，ScatterElements 类语义）——src 空间与 index 空间**不同构**，src 元素无法按 `src[perm[k]·sliceSize + s]` 平铺直读，必须经 src 空间坐标反解寻址（`srcStrides` 数组，见 [type-assign.md](type-assign.md) §3）。

`srcPos` 仅在此 SUBSET 场景生成；SAME / 单维 / nd 不产生 srcPos、不分配 workspace 段5。

**双输出加速机制**（Phase1 顺带产出 srcPos，Phase3 零反解直读）：

- Phase1 逐维反解本就要算 index 空间坐标 `c_d`（覆盖全部维子场景，§1）——**顺带用 srcStrides 累加** `srcPos = Σ_d c_d·srcStrides[d]`，与 lin 合成**共用坐标 c、零额外除法**，双输出写 workspace 段5（N×8 int64，[tiling-preprocess.md](tiling-preprocess.md) §5）。
- Phase3 段尾任务直接 `src[srcPos[perm[k]]]`——**免去每写任务 rank 次反解**（若不预计算，任务只有 `perm[k]` 原始散点号，需重新反解成坐标再算 srcPos；写任务数量大时反解开销随 rank 放大）。
- 代价：workspace 增段5（N×8）+ Phase1 多写一个数组。
- srcPos 为坐标计算，值域受 `srcNum` 计数覆盖；src 元素数计入 useInt32Index 门控（§6）。

> 非 subset 场景（index 与 src 同构平铺 / index 覆盖单维 / nd）src 可平铺直读，无本节机制。

## 5 空 tensor 早退（顺序约束）

```cpp
if (totalIndexNum == 0 || totalVarElems == 0) {
    return;   // 必须先于快除预计算与 SortLib tiling：N==0 时 strides 可能含 0 除数
}
```

（早退原则的语言中立表述见 [../../generic/conflict-domain.md](../../generic/conflict-domain.md) §4。）

## 6 uint32 坐标计算域降级（det/nondet 统一门控）

- **门控判据（det/nondet 统一）**：`useInt32Index` 只看 var/src/out/index/task 五项元素计数，五项均 ≤ INT32_MAX 才启用；任一超限即走 int64 域。写散射不再按合轴后的 strides/lin/srcPos 逐项重算。
- **显式判定**：

  ```cpp
  int64_t varNum   = var.GetShapeSize();
  int64_t srcNum   = src.GetShapeSize();              // 标量源填 1
  int64_t outNum   = varOut.GetShapeSize();
  int64_t indexNum = indexOrIndices.GetShapeSize();   // nd 形态为 N×K
  int64_t taskNum  = ...;                             // 见下表
  bool useInt64 = (varNum > INT32_MAX) || (srcNum > INT32_MAX) ||
                  (outNum > INT32_MAX) || (indexNum > INT32_MAX) ||
                  (taskNum > INT32_MAX);
  bool useInt32Index = !useInt64;
  ```

- **taskNum 按形态**：

  | 形态 | taskNum |
  |---|---|
  | index 覆盖全部维 / 展平 1D | `totalIndexNum` |
  | index 覆盖单维 | `batchSize * totalIndexNum * sliceSize` |
  | nd | `totalIndexNum * sliceSize` |

  `taskNum` 的乘积用 int64 计算。
- **读索引与越界检查不参与降级**：index 值读取与越界判断始终在 IdxT 原始 dtype **原值域**进行（[bounds](../../generic/bounds.md) §2「原值域比较、窄化截断最后」），检查通过后才窄化进 IndexT 参与坐标计算。
- 实现：kernel 加 `IndexT` 模板参数（rem/coord/lin/srcPos 全 IndexT 域，UB 槽 cast 取回）；Process 双宽度 `GetUintDivMagicAndShift<uint32_t/uint64_t>` + 运行期 if 分派；TilingData 加 1 个 int32 字段（det/nondet 结构共用）。
- det 消费点：Phase1 坐标反解/线性化与 Phase3 三场景分解的 `IndexT` 模板参数（[deterministic](deterministic.md) §3）；nondet 消费点：任务反解与目标寻址（[type-assign](type-assign.md) §4.1）——**同一门控、同一下发字段，两侧一致分派**。
- 实测收益：rank3/4 大 N 低竞争 **1.10~1.29x**（两算子独立验证）；原生 32 位乘法/累加替代 64 位多指令模拟链（收益覆盖整条坐标计算链）。
- 与读收集范式「use64Bit 不看 indexDtype，仅由四项计数决定」同源。
