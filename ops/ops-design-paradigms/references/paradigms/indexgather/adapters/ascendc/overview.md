# IndexGather 模板划分总览

> IndexGather 范式的三型模板 / TilingKey / TilingData 划分及对应关系。设计文档 §1/§2 的真值源。

---

## 1 三型模板与维度组合

IndexGather 的"模板"由四个正交维度组合而成，**不存在互斥的多 TilingKey 场景模板**：

| 维度 | 取值 | 载体 | 是否进 TilingKey |
|------|------|------|-----------------|
| 解码结构 | A 整维切片 / B N 元组 / C 逐元素对齐 | 不同的 kernel 内部结构（type-a/b/c.md） | **否**（同一算子只有一型） |
| 越界语义 | 负索引处理（none/+dim/%dim）× 越界处理（err/fill/%dim/clamp/undefined） | kernel 检查分支写法 | **否**（spec 声明，编译期语义固定） |
| 规模路径 | IDX32 / IDX64 | TilingKey idxMode 场景位 | **是** |
| tiling 预处理 | 合轴开/关（可合并段存在与否） | MergeAxis 运行期判定 | **否**（合并后视图值直接下发） |

**核心结论**：TilingKey 只编码 idxMode（2 值）；分型/越界/合轴全部在 tiling 数据与 kernel 逻辑层承载。这与"模板划分"的传统理解不同——IndexGather 的模板分派发生在 **kernel Process 内的 switch（运行期选编译期实例）**，而非 TilingKey 层。

### 1.1 batch_dims（附加结构轴）

> 全范式统一定义，各 type-*/semantics 文档引用本节，不再各自表述。

batch_dims（记作 b）是 A/B 型共有的 prefix batch 段属性：`index.shape[:b] == input.shape[:b]`（前缀逐维相等，非广播）。它**不改变解码结构分型**（A 仍 A、B 仍 B）、**不进 TilingKey**，但会实质改变实现：

- TilingData 增三字段：`batchDim` / `paramsBatchStride` / `innerTotal`
- kernel 增一次 batchIdx 快除：A 型两段→四段三除，B 型三连→四连解码
- Process 内触发 `batchDim > 0` 的二级运行期分派（双级分派，b=0 路径与基础结构逐字一致零退化）

C 型无此概念（前缀相等对齐是其广播约束的特例，天然涵盖）。

### 1.2 命名与术语映射

| 概念 | A 型（index_select） | B 型（gather_nd） | C 型（gather） |
|------|---------------------|-------------------|----------------|
| 数据张量 | input / data | params | input |
| 索引张量 | index / indices | indices | index |
| 收集轴 | axis / dim | —（无 dim，K 由 indices 末维深度决定） | dim |
| batch 属性 | batch_dims（b） | batch_dims（b） | —（广播约束天然涵盖） |
| 索引深度 | — | K = indices.shape[-1] | — |

### 1.3 范式参数常量（集中声明，各处引用）

| 常量 | 值 | 说明 | 可调 |
|------|-----|------|------|
| `MAX_RANK` | 8（示例值，按需求调整） | strides/dims 数组定长上限（C 型合轴后段数 ≤ rank；B 型 K 范围由需求确定，语义上 K ≤ rank(params) - batch_dims；A 型零数组） | 是（按需求的 rank/K 上限） |
| `THREAD_NUM` | 1024 | kernel 编译期线程常量（`__launch_bounds__` 与 `dim3` 共用，不入 TilingData） | 是（按平台） |
| `PER_CORE_MIN` | 1024 | 核数两步法 blockFactor 下限保护 | 是 |
| `DCACHE_MIN` / `DCACHE_MAX` | 32KB / 128KB | DCache 预留范围：越大性能可能越好，按剩余 UB 决定（见 tiling-preprocess §5） | 是（按剩余 UB） |

---

## 2 TilingData 分型设计（按需字段，不强制统一基型）

### 2.1 两条设计原则

1. **字段按分型需要设计**：基型仅保留真正全族必需的字段（dim / outSize / gatherDimSize / inStrideDim），strides / dims / rank 等字段归入分型设计——A 型不需要就不定义，禁止死重字段
2. **数组定长声明，MAX_RANK 按需求确定（示例 8）**：TilingData 为定长打包结构，数组字段按 MAX_RANK 定长声明；实际有效长度由 rank/K 决定——kernel 只读前缀，尾部填**安全值**（B 型 dims/strides 尾填 0、C 型 outStrides/inStrides 尾填 1）保证无效位既不越界也不污染解码。MAX_RANK 是示例值 8，需覆盖需求的 C 型段数与 B 型 K 上限；它不是范式上限；A 型零数组

### 2.2 各型 TilingData 实际字段表

| 字段 | A 型 | B 型 | C 型 | 说明 |
|------|:---:|:---:|:---:|------|
| `dim` | ✅ | — | ✅ | 归一化收集轴（A/C 型） |
| `batchDim` | ✅（batch 组合） | ✅ | — | batch_dims b（A/B 型附加结构轴） |
| `inputRank` | ❌ | ✅（k，索引深度） | ✅（合轴后 mergeRank） | A 型不消费 rank；B 型为 K；C 型为合轴后段数 |
| `outSize` | ✅ | ✅ | ✅ | 输出元素总数（全族必需） |
| `gatherDimSize` | ✅ | ❌ | ✅ | 越界上界（A/C 型全族必需）；B 型越界逐维判定（上界 = dims[d]，无单一 gatherDimSize） |
| `inStrideDim` | ✅（=innerSize） | — | ✅ | dim 维 stride（A 型与 innerSize 等值，可合并） |
| `outStrides[N]` | ❌ | — | ✅（N = mergeRank-1） | C 型坐标解码除数，长度按合轴后段数 |
| `inStrides[N]` | ❌ | — | ✅（N = mergeRank-1） | C 型偏移累加步长 |
| `dims[K]` | ❌ | ✅（K = 索引深度） | ❌ | B 型 params 前 K 维尺寸（兼逐维越界上界） |
| `strides[K]` | ❌ | ✅（K = 索引深度） | ❌ | B 型 params 前 K 维步长 |
| `sliceSize` | — | ✅ | ❌ | B 型 params.shape[K:] 连续后缀积（索引元组指向的切片大小，src 偏移乘法因子/快除除数） |
| `gatherSize` | ✅ | — | ❌ | A 型 M = prod(indices.shape) |
| `innerSize` | ✅ | — | ❌ | A 型 prod(data.shape[axis+1:]) |
| `innerTotal` | ✅（batch 组合） | ✅ | ❌ | batch 段解码除数（直传，A/B 型 batch 组合） |
| `paramsBatchStride` | ✅（batch 组合） | ✅ | ❌ | batch 视图基址步长（A/B 型 batch 组合） |

> 设计原则：**每个分型独立设计最简 TilingData**，禁止为「TTK 长度统一」而保留死重字段。TTK TilingData 长度校验按各算子实际结构体计算，不同算子长度不同是正常的。

---

## 3 双重载分派模型（kernel 侧）

| | IDX32 特化重载 | IDX64 运行期重载 |
|---|---------------|-----------------|
| 模板参数 | `<T, INDEX_T, IDX_T, RANK>`（4 参） | `<T, INDEX_T, IDX_T>`（3 参） |
| rank 来源 | 编译期模板参数，`switch(inputRank)` 8 实例 | 运行期函数参数 |
| 解码循环 | `for (d < RANK-1)` 固定次数全展开 | 展开 MAX_RANK-1 次 + `if (d < rank-1)` 截断 |
| 适用 | uint32（native 乘法单指令），全展开纯受益 | uint64（乘高多指令模拟），截断分支作调度屏障防 spill |
| 越界上界 | `static_cast<INDEX_T>(gatherDimSize)`（cast 无损前提：`use64Bit=false` ⟹ `inputNum/outSize/indexNum/taskNum` 均 ≤ INT32_MAX，故 `gatherDimSize ≤ inputNum ≤ INT32_MAX`） | **禁止 cast**（gatherDimSize 可超 int32 值域），保持 64 位原值比较 |

B 型的 K 特化（K 范围由需求确定）与 C 型的 rank 特化均复用该模型：分派维度（K 或 rank）进 Process 内 switch，与 idxMode 正交。

---

## 4 TilingKey 编码

```cpp
#define XXX_IDX32 0
#define XXX_IDX64 1

ASCENDC_TPL_ARGS_DECL(
    Xxx,
    ASCENDC_TPL_UINT_DECL(idxMode, 1, ASCENDC_TPL_UI_LIST, XXX_IDX32, XXX_IDX64));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(idxMode, ASCENDC_TPL_UI_LIST, XXX_IDX32),
        ASCENDC_TPL_TILING_STRUCT_SEL(XxxTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(idxMode, ASCENDC_TPL_UI_LIST, XXX_IDX64),
        ASCENDC_TPL_TILING_STRUCT_SEL(XxxTilingData)));
```

- 仅 idxMode 的 2 个二进制实例；dtype 组合由 spec 决定（示例 input 6 × index 2 = 12 组），经 DTYPE_INPUT/DTYPE_INDEX 宏在实例内编译期展开
- 由 def.cpp 的 `DynamicCompileStaticFlag(true)` 激活宏实例化，TilingKey 不枚举 dtype
