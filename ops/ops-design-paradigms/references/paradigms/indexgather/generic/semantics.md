# IndexGather 语义与变体推导

> 三型语义 / 越界两步模型 / 广播约束 / gather 族变体的维度组合映射。场景判定（patterns.md Step 1-2）的真值源。

---

## 1 三型语义对照

| | A 型 整维切片 | B 型 N 元组 | C 型 逐元素对齐 |
|---|--------------|-------------|----------------|
| 代表算子 | index_select | gather_nd | gather |
| index 形态 | 任意 rank（M = prod，寻址展平；output 插入 index.shape） | 任意 shape，末 K 维为坐标元组 | 与 output 同 shape（任意维） |
| 收集方式 | 沿单一 dim 收集 M 个整切片 | 每个坐标元组索引 params 前 K 维 | 每个输出位置的 dim 维坐标被 index 值替换 |
| output shape | input.shape 去掉 dim 维插入 index.shape | indices.shape[:-1] + params.shape[K:] | **= index.shape** |
| 广播 | 无（batch_dims 前缀相等除外） | 无（batch_dims 前缀相等除外） | 有：`d != dim` 时 `index.size(d) <= input.size(d)` |
| dim 负值 | 支持，归一化 +rank | 支持（batch_dims 计入） | 支持 |
| 典型结构 | 两段快除（o/j/inner，即合轴闭式解） | K 编译期特化（K 范围由需求确定）+ batch_dims 三连解码（即合轴闭式解） | 合轴（通用切段算法）+ rank'-1 次逐维解码 |

> **分型本质 = 索引表示**：A 型单坐标（整维切片）/ B 型坐标元组（K 维联合索引）/ C 型逐元素对齐。**batch_dims 为 A/B 型附加结构轴**（prefix batch 段，前缀相等约束），定义见 [overview.md](../adapters/ascendc/overview.md) §1.1——不改变分型、不进 TilingKey，但改变 TilingData 三字段与 kernel 多一次 batchIdx 快除；C 型无此概念（前缀相等对齐是其广播约束的特例，天然涵盖）。

## 2 越界处理（两步模型，spec 必须显式记录）

越界处理拆成两步：**Step 1 负索引处理（none / +dim / %dim）→ Step 2 越界处理（err / fill / %dim / clamp / undefined）**。按优先级决策：① spec/用户声明 → ② 对标算子 → ③ **默认 none + err**。完整策略与业界证据详见 [decode-bounds.md](decode-bounds.md) §1/§2；`skip` 为写散射策略，不在本范式范围内。

| 预设 | Step 1 负索引 | Step 2 越界 | 业界证据 |
|------|---------------|-------------|---------|
| **error**（默认） | none | err | torch 全系 |
| neg-wrap | +dim 单次回绕 | err | **ONNX 全系**、NumPy 高级索引 |
| mod-wrap | none | %dim 真模 | `np.take(mode='wrap')` |
| clamp | none | 钳位 | `np.take(mode='clip')` |
| fill | none | 填充值 | JAX `take(mode='fill')` |
| undefined | none | 未定义 | torch GPU 未定义路径、JAX `promise_in_bounds` |

**fill 策略的 fillValue 来源（须由 spec 声明）**：fillValue 是标量 attr 还是 tensor 输入，由 spec 决定，默认 0；两者 device 侧均只做字节写、无重解释，写法见 [decode-bounds.md](decode-bounds.md) §2。tensor 来源时 fill 值只影响 kernel 计算、不影响 tiling/outshape，无需 `TilingInputsDataDependency`，但须在 def.cpp 注册为输入。

**易混淆注记**：padding_idx（embedding 族特定合法索引→零向量）是输出值映射而非越界形态；bool index 拒收（走 masked_select）；`+dim`（单次，`[-dim,dim)` 之外交给 Step 2）≠ `%dim`（真模，任意 j 合法）。**负索引一律不做 Python 式隐式回绕**——回绕仅当 Step 1 选择 `+dim` / `%dim`，或 Step 2 选择 `%dim`。

## 3 广播约束（仅 C 型）

- `index.rank == input.rank`，且 `d != dim` 时 `index.size(d) <= input.size(d)`
- dim 维大小自由（index.size(dim) 即输出 dim 维大小，与 input.shape[dim] 无关）
- 校验双端：infershape 静态维校验（动态维跳过），tiling 运行期逐维精确校验兜底

## 4 变体推导（gather 族变体 → 维度组合映射）

新 gather 族算子先过 patterns.md 四维判定，常见变体的映射：

> **变体表使用原则**：本表仅提供结构性映射（索引表示 → 分型）；语义属性（值依赖、越界形态）由 spec/用户声明，范式不预判；算子名仅作检索锚点，机制文档（type-*.md）不绑定算子。

| 变体 | 映射 | 说明 |
|------|------|------|
| embedding 查表 | A 型 dim=0（index 任意 rank 单坐标整行收集） | weight 即 params |
| embedding_bag | 复合（A 型 gather + segment 归约） | 超单范式，需拆解两算子或复合范式承载 |
| take / flatten 索引 | A 型 flatten 变体 | input 视作 1-D，两段快除退化为单次（innerSize=1） |
| gather × batch_dims（A 型侧） | A 型 × batch_dims 组合 | kernel = batchIdx 段 + A 型两段快除（4 段 3 除），见 [type-a.md](../adapters/ascendc/type-a.md)；tf.batch_gather 为 b=1 特例 |
| masked_select | 越界语义扩展预告 | 布尔掩码收集（输出 shape 数据依赖），输出非固定——属 VariableOutput 复合范式 |
| 隐式 index（index 不以张量出现，j 由标量参数/坐标闭式推导） | A 型结构照常（无 index GM 读，M 与 j 闭式计算） | 是否值依赖由 spec 声明，范式不预判（消费规则见 [patterns.md](../patterns.md) 消费规则） |
| sparse 索引读 | B 型变体 | indices 为稀疏坐标表，kernel 结构同 B 型 |
| scatter / scatter_nd（写侧） | **写散射范式，不属于本范式** | 写收集，方向相反；其解码/越界/合轴知识可直接复用本范式 references |

**扩展规则**：变体无法映射到 A/B/C 任一型时（如新增索引形态），在 patterns.md 场景判定 Step 1 登记新 type，并补充 `index-gather-type-x.md`（承载该型的 tiling/kernel 差异与内联代码），本范式文档结构即为此预留。
