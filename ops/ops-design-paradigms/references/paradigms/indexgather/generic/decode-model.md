# IndexGather 坐标解码模型与合轴算法（语言中立）

> 行主序坐标解码模型（两套步长分离）/ 快除中转原则 / MergeAxis 合轴算法（宽定义切段 reshape 合并）与不变量。读收集侧 Phase 解码的数学真值源。具体语言的写法（如 AscendC 的 magic/shift 预计算与 UintDiv）见对应语言适配层；能力依赖见 [capability-contract.md](capability-contract.md)。

---

## 1 坐标解码模型（两套步长分离）

每输出元素一线程（排布模型见 [thread-layout.md](thread-layout.md)），行主序前向逐维解码：

```
i（输出线性下标）→ 逐维坐标 c_d = i ÷ outStrides[d]（依次取余）
  → dim 维坐标被 j（index 值）替代
  → 输入偏移 inOff = Σ_{d≠dim} c_d·inStrides[d] + j·inStrideDim
```

**两套步长分离**：解码按 **index 空间**（outStrides），偏移累加按 **input 空间**（inStrides）——广播/合轴场景两空间不同构，分离是正确性的根基。

**快除中转原则**（`UINTDIV_MAGIC_SHIFT` + `UINTDIV_VF`，见 [capability-contract.md](capability-contract.md)）：逐维除法经快除参数（magic/shift）中转，由 scalar 端逐维预计算、以参数形式传入计算域；出参须先落 Local 标量再转写（禁止直写参数缓冲左值——工具链报错）。

**字节级搬移**：读收集为纯数据搬运，按元素字节宽打包视图（1/2/4/8 字节）bit-exact 搬移（NaN/Inf/±0 原样透传）。

## 2 MergeAxis 合轴（宽定义切段 reshape 合并）

**目的**：把 dim 之外可合并的连续维折叠成单根 prod 轴，等效解码除法次数 `rank-1 → mergeRank-1`，对所有场景无退化（无段可合时 mergeRank = rank，行为不变）。

### 2.1 切段规则

- **dim 维单独保留**（其坐标被 index 值 j 替代，两空间步长不同构，禁止折叠）
- 其余维切成**极大可合并段**：段 `[a..b]` 可合并 ⟺ 段内除**段头 a**（行主序编号最小维）外 `index.size[d] == input.size[d]`（段身必须逐维相等）
- **段头广播豁免**：`index.size[a] < input.size[a]` 允许（段头 size 不进任何步长，只影响坐标范围）
- 段长 ≥ 2 才合并，单根段原样保留

**不可合轴判定**：除 dim 外不存在任何相邻维对 `(d, d+1)` 同时满足 `index.size==input.size` 且 `d+1 != dim`（全广播型 / 相等维被广播维隔断的孤岛型）。

### 2.2 不变量（合轴后必须保持）

| 量 | 不变原因 |
|----|---------|
| `outSize = prod(index.shape)` | 每维恰入一段/单根，prod 重组合不变 |
| `gatherDimSize = input.shape[dim]` | dim 单独保留 |
| `inStrideDim = prod(input.shape[dim+1:])` | 低段合并轴 prod = 同值 |
