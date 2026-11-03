# ScatterUpdate 冲突域模型（语言中立）

> Phase1 的数学本质：把任意 index 形态合成「**1 维冲突 key**」。冲突域统一视角（key/b/s 三轴）+ 排序量原则 + key 位宽自适应 + 空 tensor 早退原则。三段式编排中 Phase1 的语义真值源。
>
> **与 Phase3 正交**：Phase1 只决定「冲突域」（key 怎么合成、哪些散点归并），不感知 Phase3 的并行分解；Phase3 只消费 `sortedKey + perm`，不感知 key 的合成方式。两者独立组合。具体语言的 Phase1 写法（如 AscendC 的逐维反解/坐标合成代码）见对应语言适配层（AscendC：[adapters/ascendc/linearize.md](../adapters/ascendc/linearize.md)）。

---

## 1 核心原则：排序 key = 冲突域标识，slice 永不进排序

```
Phase1（产生冲突 key）      Phase2（排序）      Phase3（分段处理，并行展开）
────────────────────      ─────────────      ──────────────────────────
任意 index 形态              1 维 key            只感知 (b, k, s) 三维数值
→ 1 维冲突 key           → sortedKey + perm  → (b,k,s)/(k,s)/(b,k) 三场景
（决定冲突域）                                  （决定并行度，与形态无关）
```

## 2 统一合成视角

把 var 的维分成两类：

- **index 覆盖维** → 合成**冲突 key**（两个散点 key 相同 ⟹ 同一冲突段，段内处理）
- **index 不覆盖维** → 独立并行维：前缀维合轴为 **batch（b）**，后缀维合轴为 **slice（s）**（连续负载，Phase3 线程级展开——一维化映射公式见 [thread-layout.md](thread-layout.md)，永不进排序）

| 形态 | 冲突 key（Phase1 产出） | b（前缀独立维） | s（后缀独立维/负载） | 排序量 |
|------|------------------------|----------------|---------------------|--------|
| **index 覆盖全部维**（dim 形态，逐元素；示例：torch scatter / scatter_add_ / ScatterElements） | 逐维反解 + dim 替换 → 元素 lin | 1（index 覆盖全部非 dim 维） | 1（每散点单元素） | N = index.numel() |
| **index 覆盖单维**（dim 形态，沿 dim 轴共享；示例：tf scatter_add / index_add / index_max） | **j**（index 值本身，零合成成本） | prod(shape[:dim]) | prod(shape[dim+1:]) | M |
| nd（indices (N,K) 坐标元组，覆盖前 K 维；示例：scatter_nd_* / tensor_scatter_nd_update） | **Σ idx[d]·stride[d] = lin**（K 维坐标合成 1 维；合成后即 1 维索引，后续按 1 维处理） | 1（坐标元组覆盖前 K 维，无独立前缀） | prod(shape[K:]) | N = prod(indices.shape[:-1]) |
| 展平 1D（put_/take 写侧，flatten 视图，覆盖全部维的 rank=1 极限） | j（== lin 的极限退化） | 1 | 1 | M |

> **排序量永远 = 冲突单元数（索引个数），不含 b、不含 s。** slice 是 key 对应的连续块，在 Phase3 内线程级并行展开，不进 Phase2 排序——把 batch/slice 混进排序 key 是已实测否决的冗余（排序量膨胀 b 倍）。

## 3 key 位宽自适应

| 条件 | key 位宽 |
|------|---------|
| key 上界 ≤ 32767 | 16 位 |
| key 上界 ≤ 2^31-1 | 32 位 |
| 否则 | 64 位 |

- key 上界 = 冲突域值域上界：index 覆盖全部维为 totalVarElems（元素 lin）；index 覆盖单维/nd 为冲突域最大偏移 + 1（覆盖单维按 `D_dim·slice`、nd 按 totalVarElems 收口）；error 形态无哨兵，上界即值域。
- 位宽选择由 tiling 侧自适应下发（如 AscendC 的 TilingKey `keyDtype` 场景位）。

## 4 空 tensor 早退原则（顺序约束）

`totalIndexNum == 0 || totalVarElems == 0` 时短路返回，且**必须先于依赖 shape 的预计算**（快除参数 / 排序 tiling 等）——N==0 时 strides 可能含 0 除数。

- `totalIndexNum == 0`：无散点（indices 外层为空）。
- `totalVarElems == 0`：var 含 0 维（此时 sliceSize 亦为 0）。
- inplace 语义下输出 = var 本身直接返回；函数式语义走 var→output 全量拷贝路径。
