# ScatterUpdate：覆盖写场景（type-assign）

> `out[idx] = src` 纯赋值覆盖，无归约。代表算子：tensor_scatter_nd_update / ScatterNdUpdate / scatter(reduce=none)。**默认 nondet 直写（任意赢）+ 条件 det 排序（段尾覆盖 last-write-wins）双路径**。

---

## 1 语义

```
out = var.copy()                       # 函数式（copy-on-write）或 inplace
for i in [0, N):                       # 顺序遍历
    if 合法（按 bounds 形态）:
        out[indices[i]] = updates[i]   # 重复索引 last-write-wins（靠后覆盖靠前）
```

- **纯数据搬运**：无算术、无类型提升、无舍入——NaN/Inf/±0 bit-exact 原样写入，全 dtype binary 比对（rtol=atol=0）。
- 重复索引的覆盖顺序（最后赢 / 第一个赢）是**语义决策点**（按 spec/竞品，默认 last-write-wins 对齐 TF CPU 确定性行为；nondet 路径下为任意赢，见 §1.1）。
- dtype 覆盖面 = **除 complex 外全量标量**（32 组 dtype 编译矩阵覆盖）：bool / float8_e5m2 / float8_e4m3fn / hifloat8 等 1 字节非常规 dtype、**double**（fp64，SIMT 无原生标量）均按 `BytePackType<T>` 字节视角（sizeof 1→uint8 / 2→uint16 / 4→uint32 / 8→uint64）bit-exact 搬运——**含 W==1 标量路径也走字节视角，禁止 `T` 原生赋值**（SIMT 原生标量仅到 float/int，`T` 原生赋值是官方早期版本未纳入 double/fp8 的根因）；complex 排除（复数语义非字节搬运可表达 + SIMT 无原生类型）；float8_e8m0 为芯片 registry 不支持的特例（950PR/DT，仅 kernel 层字节搬运能力实证）。

### 1.1 覆盖写的竞争语义（nondet 默认的依据，tensor_scatter_nd_update v2.0 实证）

- **多写者"任意赢"即覆盖语义本身正确**——"丢失"其他写者正是覆盖写语义（与 reduce 分型"丢更新即错误"形成对照，划界见归约写范式）。
- **仅需对齐单条 store 原子（防撕裂）**：对齐字的全部字节同属一个总线事务，并发写同一位置最终生效值必为某一写者的完整 bit 模式（无新旧 bit 混合）——全部 dtype（1B/2B/4B/8B）对齐写均原子安全，**无 dtype 白名单**（reduce 分型的原子 RMW 白名单属归约写范式）。
- **golden 兼容性**（nondet 任意赢的可判定关系，按竞争结构三层分层）：
  - 无竞争（每 target 唯一贡献）/ 同值贡献（同 target 所有 update 值相同）→ **bit 全等**（任意赢与 last-write-wins 数学等价）；
  - 多样值重复 → **元素级候选集合成员判定** `out[e] ∈ cand[e]`（cand[e] = 所有指向 e 的 update 值；粒度 = 元素级——nondet 胜者粒度是 pack，同 slice 不同 pack 可来自不同 update 行，slice 级"整片同源"不成立；pack 内 W 元素必然同源）；
  - **值域隔离纪律**：var 初始化值域与 update 值域隔离（如 var 0x55 模式填充）——否则写丢失（残值 var[e] ∈ cand）漏检；
  - **禁止断言**：多样值用例不得断言跨次一致与 nondet == det（胜者随调度漂移）；**非确定性与 dtype 无关，整数不豁免**。
  完整验收方法（成员判定实现/错误模式覆盖/det 互证）见 [deterministic](deterministic.md) §5。

## 2 双路径分派（默认 nondet 直写）

| 路径 | 触发条件 | 结构 |
|------|---------|------|
| **nondet 直写**（**默认**） | 张量源：`GetDeterministic()==0 && dupRatio ≤ THRESHOLD`；标量源（值同）：`dupRatio ≤ THRESHOLD`（**确定性开关不触发**——值同天然 bit 确定） | 单 VF 直接写覆盖：无排序/无段表/无用户 ws/无 SyncAll |
| **det 排序**（条件路径） | 张量源：`dupRatio > THRESHOLD \|\| GetDeterministic()!=0`；标量源（值同）：`dupRatio > THRESHOLD`（纯性能） | 三段式段尾覆盖（last-write-wins，[deterministic](deterministic.md) §3.4） |

- **重复度口径**：鸽笼下界 `dupRatio = M / D_dim`（M = 散点数，D_dim = 冲突域尺寸；host 静态可见，无值依赖——下界高 ⟹ 必然高重复，下界低仍有值集中风险，阈值留裕量），阈值 `DUP_RATIO_THRESHOLD` 经验值（待 perf 标定，鸽笼下界口径与归约写范式的门控思路同源）。
- **高重复走排序的动机是性能（不是正确性）**：同地址并发写慢一个数量级（[deterministic](deterministic.md) §6 硬件特征库）——排序段尾单写者把 M 次冲突写降为 unique(key) 次；张量源另有确定性动机（last-write-wins 固定顺序），标量源值同**无确定性问题**。
- Host 注册：张量源双模板（NondetTiling priority=0 + DetTiling priority=10 永兜底，骨架同 [overview](overview.md) §2）；标量源（值同）单模板 + TilingKey `dedup` 场景位（门控仅 `dupRatio`，无注册分叉）。

## 3 TilingData 字段（assign 型，nd 形态示例）

```cpp
#pragma pack(push, 1)
struct XxxTilingData {                 // det 结构（排序段尾覆盖）
    int32_t  needCoreNum;              // SortLib 决定
    int64_t  totalIndexNum;            // N = prod(indices.shape[:-1])
    int64_t  totalVarElems;            // var 总元素数（空 tensor 判据）
    int32_t  varRank;                  // 1~8
    int32_t  indexDimNum;              // K = indices.shape[-1]
    int64_t  varDims[8];               // 越界检查（前 K 维有效）
    int64_t  varStrides[8];            // 线性化（前 K 维有效）
    int64_t  sliceSize;                // = prod(var.shape[K:])，Phase3 写回长度
    int64_t  srcStrides[8];            // 仅 SUBSET：src 坐标反解（Phase1 计算 srcPos）
    // SortLib: numTileData / tileCount / activeCores / tmpUbSize / isSingleCore
    // workspace 偏移: sortWsBytes / wsLinearIdxOff / wsSortedOff / wsPermOff / wsSrcPosOff（仅 SUBSET）
};
struct XxxNondetTilingData {           // nondet 结构（直写覆盖，子集）
    // totalIndexNum / totalVarElems / varRank / indexDimNum
    // varDims / varStrides / sliceSize（VF 内联坐标线性化，与 det Phase1 同源参数）
    // + useInt32Index（uint32 坐标计算域降级位，det/nondet 共用，见 [linearize](linearize.md) §6）
};
#pragma pack(pop)
```

- **无归约字段**（无 AccT/精度标志——dtype 由 DTYPE_ 宏编译期实例化）；nd 形态无 magic/shift（Phase1 无除法）。
- **src 寻址按 index 形态分两式**（dim 形态两子场景不同，[semantics](../../generic/semantics.md) §2.1）：
  - **index 覆盖单维 / nd 形态**：updates 直接 flat slice 访问 `updates[perm·sliceSize + s]`——src 与散点一一平铺，无前段子集反解。
  - **index 覆盖全部维**：index 与 src 前段子集（`index.size(d) ≤ src.size(d)`），**仅 SUBSET 场景**需要 src 坐标反解。Phase1 用 TilingData 的 `srcStrides[8]` 计算 `srcPos`；启用 srcPos 双输出时，再追加 workspace 段5 与 `wsSrcPosOff`（SAME/单维/nd 不分配）。与归约写范式的 `shapeMode=SUBSET` 分支写侧同构。
- TilingKey `countMode + keyDtype + deterministic` 三场景位（张量源；nondet 不消费 countMode/keyDtype，SEL 用占位值）；**标量源（值同）为 `countMode + keyDtype + dedup`**（dedup 位即路径位，仅 `dupRatio` 驱动）；无 shapeMode——**限「index 覆盖单维 / nd 形态」**（覆盖全部维的前段子集寻址经 srcStrides 增补承载，不进 TilingKey）。

## 4 Kernel：双路径

### 4.1 nondet 单 VF 直写覆盖（默认）

```
任务反解（÷sliceVec 快除，magic/shift 经 UB 传 VF；slice==1 快路径无除法）
  → 内联坐标线性化（与 det Phase1 同款 UB 数组参数：varDims/varStrides，indices 直读）
  → 越界/负索引 __trap 或 continue 跳过（与 det 同语义，[bounds](../../generic/bounds.md) §2）
  → 直接写（无原子、无段判据、无 perm）：varVec[key/W + sw] = updVec[k*sliceVec + sw]
grid-stride 兜底：blockDim 不足时每线程多任务；无排序/无段表/无用户 ws/无 SyncAll；
KERNEL_TYPE_AIV_ONLY；末尾 VF 后无 V_S join（AIV task 结束协议排空向量管线）。
```

- 字节宽打包（W∈{1,2,4,8}）与 det 共用一份（common 单一事实源，防两路径 W 计算漂移）；nondet 竞争表现为 **pack 粒度任意赢**（≤8B 完整 bit 模式，元素级候选集合验证仍成立）。
- assign·nd 完整形态：全 dtype PackT 字节搬移（含 double/fp8/hifloat8 的 W==1 路径）+ 越界 continue skip（与 det 哨兵行为一致，bounds §5.1）+ `useInt32Index` 五项计数门控（var/src/out/index/task；nd 任务域 N×slice 计入 taskNum）。
- IndexT 域模板化（uint32 坐标计算域降级）与 det Phase3 同款。

### 4.2 det 三段式（段尾覆盖，条件路径）

骨架与 [deterministic](deterministic.md) §3 一致：

- **Phase1 线性化**：按 index 形态（dim 反解 / nd 直读，[linearize](linearize.md)）；越界 skip 形态标哨兵（bounds §4）。
- **Phase2 稳定排序**：同 key 连续成段，段内原始顺序唯一。
- **Phase3 段尾覆盖**：(b,k,s) 全量任务 O(1) 段尾自判（`k == kNum-1 || sortedKey[k+1] != key`），段尾任务写自己——`perm[k]` 即段内原始行号最大（稳定排序保证）= 胜者；元素级（W=1）通用兜底，sliceSize 可整除时字节宽打包（W∈{2,4,8}，受 `W·sizeof(T) ≤ 8` 限制）加速。
- **无原子操作**，结果 bit-exact，与线程调度/核数无关。

## 5 标量源模式（src 表示的边界子场景）

src 为标量 value 而非张量（**语义场景**：所有目标写同一常量；锚点：scatter 标量 value 模式 / 标量填充类）。与张量源的差异：

| 差异点 | 张量源 | 标量源 |
|--------|--------|--------|
| 算子输入 | src 张量（REQUIRED，满秩） | **src 为标量值，值来源二选一**：标量属性（无 src 张量输入）/ 标量张量输入（rank-0 或 shape=[1] 的 input tensor）——**优先单元素 tensor**（attr 类型有限：仅 int64/float32/bool/string/enum，int8/uint8/bool/低精度无法承载） |
| tiling | src.shape 校验（nd 形态 updates.shape 拼接校验） | 满秩 src.shape 校验豁免（标量张量输入时仅校验 shape ∈ {(), (1,)}）；N 仅由 index/indices 决定 |
| 标量下发 | — | **属性来源**：tiling `GetAttr` → host 位级打包进 TilingData 固定字节字段（禁数值转换）；**标量张量输入**：值在 GM，tiling 不读值（无值依赖），kernel 运行时按字节读——与读收集范式 fill 值「来源二选一」纪律同构 |
| 写回 | 段尾覆盖读 `updates[perm[k]·sliceSize + s]` / 直写读 `updVec[...]` | 段尾覆盖写常量：`varBytes[key + s] = valueBytes`；直写 `var[key] = value[0]`（GM 读单元素） |
| 确定性 | 重复索引 winner 不定（nondet 任意赢 / det last-write-wins） | **值同无确定性问题**——任意赢 = 同值 = 天然 bit 确定；确定性开关不触发排序 |

### 5.1 标量源双路径门控（仅重复度驱动）

> 标量源值同：**排序仅为性能**（高重复规避同地址并发写慢一个数量级），无确定性动机——确定性开关不触发排序（与张量源门控的差异点）。

| 路径 | 触发条件 | 结构 |
|------|---------|------|
| **直写**（默认） | `dupRatio = M / D_dim ≤ THRESHOLD`（鸽笼下界，host 静态可见） | Phase1 越界检查 + 直写常量（**无排序无段判据**） |
| **排序去重** | `dupRatio > THRESHOLD` | Phase1 key=j → Phase2 排序 → Phase3 段尾单写者写一次 value |

- **门控编码**：TilingKey `dedup` 场景位（两路径 kernel 结构差异大走独立实例）；`THRESHOLD` 经验值初始（待 perf 标定）。
- **鸽笼下界只看 shape**（`M / D_dim`），不看 index 值——host 无值依赖。

## 6 退化场景核对表

| 场景 | 行为 | 核对点 |
|------|------|--------|
| K = rank（全维索引） | sliceSize = 1，单元素覆盖 | Phase3 循环 1 次 / 直写无除法快路径 |
| K = 1（单维切片替换） | slice = var.shape[1:]，整行/片替换 | sliceSize 正确性 |
| 空 tensor（N==0 或 varElems==0） | 入口短路 return | 早退先于排序 tiling |
| 重复索引（det） | last-write-wins | golden 顺序遍历后写覆盖 |
| 重复索引（nondet，默认） | 任意赢（结果 ∈ 候选集合） | 无竞争/同值用例 bit 确定；多样值用例元素级候选集合验证（§1.1） |
| 越界/负索引 | skip 形态：det 哨兵沉底跳过 / nondet continue；error 形态双路径一致 `__trap` | bounds §4 / §2；双模板越界语义必须一致 |
| 标量源 | 直写常量 / 段尾写一次常量；值同无确定性问题 | 值来源二选一（优先单元素 tensor）；双路径仅 `dupRatio` 门控（§5.1） |
| 展平 1D 索引 | rank=1/dim=0/slice=1，`target = index 值本身` | 零快除零反解 |
| inplace（var_out == var 同名） | 无 Stage A 拷贝 | 非更新区域天然保留 |
| 函数式（y 独立输出） | Phase3 前置 Stage A 全量拷贝 | `asc_syncthreads()` 分隔 |
