# ScatterUpdate 语义与变体推导

> 覆盖写语义 / src 表示（张量源/标量源）/ index 形态（dim 三轴分解 + nd 坐标元组 + 展平 1D 退化）/ scatter 覆盖写变体推导。场景判定的语义真值源（语言中立；文中出现的具体 API 名为 AscendC 适配层示例标注，其他语言按各自适配层映射——实现在 [adapters/ascendc/](../adapters/ascendc/index.md)）。

---

## 1 覆盖写语义（本范式承载范围）

**写语义**：`out[idx] = src`（纯赋值，无算术、无类型提升、无舍入——NaN/Inf/±0 bit-exact 原样写入，全 dtype binary 比对 rtol=atol=0）。

- **重复索引**：last-write-wins（靠后覆盖靠前，须 spec/竞品确认；第一个赢为同构变体）；nondet 路径下为任意赢
- **写冲突**：无归约冲突——多写者任意赢即覆盖语义正确；仅需对齐单条 store 原子防撕裂（对齐字的全部字节同属一个总线事务，全部 dtype 1B/2B/4B/8B 对齐写均原子安全，**无 dtype 白名单**）
- **确定性**：默认 nondet 直写（任意赢）；det 排序段尾覆盖（last-write-wins）为条件路径（张量源：重复度高/确定性请求；标量源值同：仅重复度高——性能）
- **精度**：纯数据搬运 bit-exact，binary 比对
- **越界常见形态**：跳过 skip（哨兵 key，TF/Ascend 对标）——均为 spec 决策，见 [bounds](bounds.md)

**划界**：reduce 归约写（`out[idx] = out[idx] op src`，op ∈ {add, mul, max, min, div, mean}）为归约写范式，不在本知识库范围。

### 1.1 src 表示（型内形态决策，不单列判定 Step）

| src 表示 | 语义 | kernel 形态 |
|---|---|---|
| **张量源**（主体） | 每散点搬一个 src 元素 / slice | GM 逐元素 / slice 读 |
| **标量 value**（边界子场景） | 所有目标写同一常量（填充类语义） | **值来源二选一**：标量属性（tiling `GetAttr` → host 位级打包字节字段）/ 标量张量输入（rank-0 / shape=[1]，kernel 运行时 GM 读），kernel 写常量（见 [type-assign](../adapters/ascendc/type-assign.md) §5 标量源模式） |

- 覆盖写分型可配标量源；标量源无满秩 src 寻址（每散点不再读独立 src 元素/slice），散点数 N 仅由 index/indices 决定；标量张量输入时仍存在 src 张量输入，仅其 shape ∈ {(), (1,)}，满秩 shape 校验（前段子集 / updates.shape 拼接）豁免。
- 标量值的位模式承载遵循字节搬移原则：属性来源由 host 位级打包进 TilingData 固定字节字段（禁数值转换）；标量张量输入由 kernel 运行时按字节读 GM——与读收集范式 fill 值「来源二选一」纪律同构。

## 2 index 形态（决定 Phase1 线性化）

### 2.1 dim 形态（单维 + dim，两个并列子场景）

dim 形态按 **index 覆盖维**划分两个并列子场景（冲突域即本质差异，见 [linearize](../adapters/ascendc/linearize.md) §1/§2）：

**子场景一：index 覆盖全部维（逐元素）**——index 与 var/src 同 rank，每散点独立 index 元素（示例：torch scatter(reduce=none) / ScatterElements）：

```
var/index/src 同 rank（rank ≤ 8）；index 与 src 前段子集（index.size(d) <= src.size(d)，不广播）；
dim 归一化 ∈ [0, rank)；index.size(d) <= var.size(d)（d != dim）
y = var.copy()
for pos in [0, N):                      # N = index.numel()
    coord = unravel(pos, index.shape)   # 坐标反解（UintDiv 快除）
    j = index[pos]                       # dim 维坐标被 j 替换
    y[coord 替换 dim 为 j] = src[pos]
```

**子场景二：index 覆盖单维（沿 dim 轴共享）**——index 为 1D（长度 n）沿 axis，对所有非 axis 维共享：

```
index 1D 沿 axis；冲突域仅 dim 维 j（排序 key = j，n 个，不升全局 lin）；
独立并行维 batch = prod(var.shape[:axis])、slice = prod(var.shape[axis+1:])
（Phase3 (b,k,s) 三场景展开，见 [deterministic](../adapters/ascendc/deterministic.md) §3）
```

**展平 1D 退化**：var 被视作 flatten 1D 视图，index 为**全局下标**（无 dim 概念）——即 index 覆盖全部维的 rank=1 / slice=1 极限退化，`target = index 值本身`，零快除零反解（读收集 take 的写侧镜像，锚点 put_ / 展平填充类）。

### 2.2 nd 形态（坐标元组，scatter_nd_update / tensor_scatter_nd_update 族）

```
indices: (N, K)，K = indices.shape[-1] ∈ [1, rank(var)]
updates.shape = indices.shape[:-1] + var.shape[K:]
y = var.copy()
for i in [0, N):
    idx = indices[i]                    # shape (K,)，var 的前 K 维被索引
    if all(0 <= idx[d] < var.shape[d]): # 越界/负索引按 bounds 形态决策
        y[tuple(idx)] = updates[i]      # updates[i] 覆盖 var 的一个 slice
```

**线性化无除法**：`lin = Σ_{d<K} idx[d]·varStrides[d]`（indices 末维 K 连续，直接 `indices[pos*K+d]` 读取）——K 维坐标**合成 1 维索引后全程按 1 维处理**（排序 N 个 lin，`slice = prod(var.shape[K:])` 是 lin 对应的连续块不进排序；b=1 → Phase3 走 (k,s) 场景），见 [linearize](../adapters/ascendc/linearize.md) §3。

### 2.3 dim 轴位置与三轴分解（index 覆盖单维专属，Phase3 并行场景来源）

index 覆盖单维子场景的目标偏移按 dim 轴位置三轴分解（index 覆盖全部维为逐维反解，见 linearize §1）：

```
target = batch·(D_dim·slice) + index·slice + s
batch = prod(var.shape[:dim])，D_dim = var.shape[dim]，slice = prod(var.shape[dim+1:])
```

| dim 位置 | 分解 | Phase3 场景（deterministic §3.2） |
|---------|------|----------------------------------|
| 轴首 dim=0 | 两轴 (index, slice)：batch=1 | (k, s)（÷s 一次快除） |
| 轴末 dim=rank-1 | 两轴 (batch, index)：slice=1 | (b, k)（÷k 一次快除） |
| 轴中 | 三轴 (batch, index, slice) | (b, k, s)（÷s、÷k 两次快除） |

> 与读收集范式 A 型「按 axis 拆轴 + 快除次数 = 段数-1」互为写侧镜像。nondet 路径与 det Phase1 共用该分解。

## 3 变体推导（scatter 覆盖写变体 → 维度组合映射）

> 变体表仅提供**结构性映射**（index 形态 × src 表示）；语义属性（越界形态、覆盖顺序）由 spec/竞品声明，范式不预判。**算子名仅作检索锚点，机制文档不绑定算子**。

| 变体（锚点） | index 形态 | 说明 |
|------|-----------|------|
| scatter（无 reduce）/ ScatterElements(reduce=none) | dim（覆盖全部维） | 覆盖写 + 逐元素索引，src 前段子集 |
| scatter 标量 value 模式 / 标量填充类 | dim（覆盖全部维） | **标量源子场景**：src 为标量值（值来源二选一：标量属性 / 标量张量输入 rank-0 或 shape=[1]），见 §1.1 与 [type-assign](../adapters/ascendc/type-assign.md) §5 标量源模式 |
| tensor_scatter_nd_update / ScatterNdUpdate / scatter_nd_update | nd | 覆盖写 + 坐标元组 |
| 展平 1D 索引写（put_(accumulate=False) / 展平填充类） | dim（**展平 1D 退化**） | flatten 视图全局下标，rank=1/slice=1，`target = index 值本身`（take 的写侧镜像） |
| index_put_ 单张量整数坐标退化（accumulate=False） | dim / nd | 高级索引写的**单张量退化**归本范式；accumulate=True 为归约写，属归约写范式；全形态见下方划界行 |
| **布尔掩码索引**（masked_scatter / masked_fill 类） | **划界** | bool mask 作索引，需 nonzero→坐标预处理，属 mask_predicate 范式，不属本范式整数坐标轴 |
| **高级索引组合**（index_put_ 全形态） | **划界** | 多张量逐维 + None 维 + broadcast，属动态高级索引复合范式；本范式仅覆盖其单张量整数坐标退化 |
| **分段组号索引**（unsorted_segment_* 类） | **划界** | segment id 组号语义 + 零初始化 base + 输出 shape 由分段数输入决定（值依赖），属 DynamicShape segment |
| reduce 归约写变体（scatter_add / scatter_max / index_add / scatter_nd_add / put_(accumulate=True) / embedding 梯度散播 等） | **划界** | 归约写分型，属归约写范式 |
| gather（读侧） | **不属于本范式** | 读收集，见读收集范式（共享越界/UintDiv 方法论） |

**扩展规则**：变体无法映射到 dim/nd × src 表示任一组合时（如新增索引形态），在 patterns.md 场景判定 Step 2 登记新 index 形态并补充 [linearize](../adapters/ascendc/linearize.md) 对应章节；属划界场景的（布尔掩码 / 高级索引组合 / 分段组号 / reduce 归约写）不在本范式扩轴，引用对应姊妹范式承载。
