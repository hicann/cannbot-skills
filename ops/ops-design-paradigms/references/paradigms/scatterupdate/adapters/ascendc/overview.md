# ScatterUpdate 模板划分总览

> ScatterUpdate（assign 覆盖写分型；reduce 归约写不在本知识库范围）的 index 形态 / src 表示 / 确定性 / TilingKey / TilingData 划分及对应关系。设计文档 §1/§2 的真值源。
>
> 范式定位：**index 张量驱动的写散射·覆盖写**——`output[index[i]] = src[i]`，纯搬运 bit-exact；与读收集范式互为读/写镜像；解码/越界决策方法论同源，实现独立承载。

---

## 1 正交维度模型与模板划分

ScatterUpdate 算子 = 五个正交形态维度的组合（写语义固定为 assign 覆盖写；reduce 归约写不在本知识库范围）。index 形态、src 表示、越界、确定性、SortLib/寻址场景为正交维度。场景判定流程（patterns.md）走写语义划界 / index 形态 / 越界 / 确定性；**src 表示为型内形态决策，不单列判定 Step**：

| 维度 | 取值 | 载体 | 进 TilingKey |
|------|------|------|--------------|
| **写语义**（划界轴） | assign 覆盖写（本范式）；reduce 归约写不在本知识库范围 | kernel 内部结构（[type-assign](type-assign.md)） | 否（同一算子只有一种写语义，spec 声明，编译期固定） |
| **index 形态** | dim 单维+dim（含**展平 1D 退化**：flatten 视图全局下标，rank=1 / slice=1）/ nd 坐标元组 | Phase1 线性化结构（[linearize](linearize.md)） | 否（host 侧判定，三段式骨架不变） |
| **src 表示** | 张量源（主体）/ 标量 value（边界子场景） | kernel src 读取形态：张量源逐元素读 GM；标量源写固定常量，**值来源二选一**——标量属性（tiling `GetAttr` → host 位级打包进 TilingData 字节字段）/ 标量张量输入（rank-0 / shape=[1]，值在 GM、kernel 运行时读），见 [type-assign](type-assign.md) §5 标量源模式。**优先单元素 tensor**：attr 类型有限（仅 int64/float32/bool/string/enum），int8/uint8/低精度位模式无法承载 | 否（型内形态决策） |
| **确定性** | nondet 直写（**默认**，任意赢）/ det 排序段尾覆盖（张量源：重复度高或确定性；标量源值同：仅重复度高——性能） | `deterministic` 场景位 + 双 TilingData（标量源值同为 `dedup` 位，仅重复度驱动） | **是**（1 bit） |
| **SortLib/寻址场景** | countMode（计数位宽）/ keyDtype（key 自适应） | TilingKey 场景位 | **是** |

**核心结论**：
- TilingKey 场景位编码——**assign 型**：`countMode + keyDtype + deterministic`（无 shapeMode；nondet 不消费的维度用占位值；标量源值同子场景 deterministic 位即 `dedup` 位——仅重复度驱动，确定性开关不触发）。
- 写语义 / index 形态 / src 表示不进 TilingKey，全部在 kernel 结构与编译期模板承载——与读收集范式「分型不进 key」的原则一致。
- assign 型**默认 nondet 直写**——多写者任意赢即覆盖语义正确，仅需对齐单条 store 原子防撕裂（全 dtype 无白名单，tensor_scatter_nd_update v2.0 实证）；det 排序为条件路径：张量源 `重复度高（规避同地址并发写慢一个数量级）|| 确定性（last-write-wins 固定顺序）`，标量源值同仅高重复（纯性能，值同无确定性问题）。
- **index 形态的边界（划界，不属本范式整数坐标轴）**：布尔掩码索引（bool mask，需 nonzero→坐标预处理，划界 mask_predicate 范式）、高级索引组合（多张量逐维 + None 维 + broadcast，划界动态高级索引复合）、分段组号索引（segment id 组号语义 + 零初始化 base + 输出 shape 由分段数输入决定，划界 DynamicShape segment）。三者仅在 [semantics](../../generic/semantics.md) §3 变体表登记。
- **Phase1 × Phase3 正交**（det 三段式的内层结构原则，见 [conflict-domain](../../generic/conflict-domain.md) / [deterministic](deterministic.md) §3.1）：Phase1 决定**冲突域**（任意 index 形态 → 1 维冲突 key，排序量 = 索引个数，不含 b/s）；Phase3 决定**并行展开**（(b,k,s)/(k,s)/(b,k) 三场景，不感知 nd/1D/dim 形态）。两者独立组合——新算子从「key 合成方式 × 并行场景」两个正交决策拼装，而非为每形态写独立 Phase3。

## 2 双模板分派模型（assign 张量源）

Host 侧多模板注册（TilingRegistry，priority 升序遍历）：

```
Tiling4Xxx → TilingRegistry::DoTilingImpl
  ├─ priority=0:  NondetTiling   IsCapable：
  │                  (GetDeterministic()==0) && (dupRatio ≤ THRESHOLD)   ← 无 dtype 白名单
  │                               ├─ true  → 单 VF 路径（NondetTilingData，无排序无用户 ws；直接写覆盖）
  │                               └─ false → GRAPH_PARAM_INVALID，registry 自动降级 ↓
  └─ priority=10: DetTiling      IsCapable ≡ true（兜底）→ 三段式排序段尾覆盖路径
```

- **默认方向 nondet**：assign 默认 nondet 直写（IsCapable 拒绝 = 高重复 / 确定性请求，自然落 det 排序）。
- `GetDeterministic()` 异常场景返回 `INT32_MAX`——**必须严格 `==0` 判断**，任何非 0 值自然落入确定排序路径。
- assign 标量源（值同）单模板 + TilingKey `dedup` 场景位（门控仅 `dupRatio`，无注册分叉——值同天然 bit 确定，确定性开关无意义）。

## 3 TilingData 分型设计（按需字段，禁止死重）

### 3.1 设计原则

1. **字段按路径（det/nondet）与 index 形态需要设计**：公共字段仅保留全族必需；dim 形态数组（varDims/varStrides/indexDims/indexStrides）、nd 形态字段（indexDimNum/sliceSize）、nondet 专属字段（useInt32Index 等）归入分型设计。
2. **数组定长声明，默认 MAX_RANK=8**：kernel 只读前缀，尾部填安全值（dims/strides 尾填 0）。
3. **不含 threadNum**（编译期常量 THREAD_NUM=1024）；**不含 magic/shift**（device 端 Process 预计算）；**不含 needCoreNum 之外的多核参数**（核数由 SortLib tiling 决定并直接下发）。
4. **SUBSET 专属字段**：`srcStrides[8]` / `wsSrcPosOff` 仅 det + index 覆盖全部维且 `index.size(d) < src.size(d)` 时有效；SAME/单维/nd 填 0 或不分配段5。

### 3.2 各型 TilingData 字段表

| 字段 | 公共 | dim 形态 | nd 形态 | nondet | 说明 |
|------|:---:|:---:|:---:|:---:|------|
| `needCoreNum` | ✅（仅 det） | | | ❌ | 实际核数（det：SortLib 核数传递；nondet 不携带——kernel 用 VF 内建 gridDim×blockDim 自解析，SetBlockDim 已下发，冗余携带徒增 TTK 长度） |
| `totalIndexNum` | ✅ | | | | 散点数 N（dim=index.numel()；nd=prod(indices.shape[:-1])） |
| `totalVarElems` | ✅ | | | | var 总元素数（空 tensor 判据 + Stage A 拷贝量） |
| `varRank` / `indexDimNum` | ✅ | | | | dim 形态为 rank；nd 形态为 K=indices.shape[-1] |
| `dimNormalized` | | ✅ | ❌ | | dim 归一化后收集轴（nd 无 dim，K 承载） |
| `varDims[8]` / `varStrides[8]` | | ✅ | ✅ | | dim：越界检查+反解；nd：线性化 Σ idx[d]·varStrides[d] |
| `indexDims[8]` / `indexStrides[8]` | | ✅ | ❌ | | 坐标反解用（nd 的 indices 末维连续，无除法） |
| `sliceSize` | | ✅（=prod(shape[dim+1:])） | ✅（=prod(var.shape[K:])） | | 每 update 覆盖的 slice 元素数 |
| `srcStrides[8]` | | ✅（仅 SUBSET） | ❌ | | 仅 index 覆盖全部维且 `index.size(d) < src.size(d)` 时使用；Phase1 据此计算 srcPos |
| `numTileData/tileCount/activeCores/tmpUbSize/isSingleCore` | ✅（det） | | | ❌ | SortLib tiling 输出直传 |
| `useInt32Index` | ✅（det/nondet 共用） | | | | uint32 坐标计算域降级位：var/src/out/index/task 五项元素计数均 ≤ INT32_MAX 时启用（见 [linearize](linearize.md) §6） |
| workspace 偏移（sortWsBytes/wsLinearIdxOff/wsSortedOff/wsPermOff + SUBSET: wsSrcPosOff） | ✅（det） | | | ❌ | 基础四段；SUBSET 追加段5 srcPosBuffer（SAME/单维/nd 不分配，`wsSrcPosOff=0`；nondet 用户 ws=0） |

> TTK 长度校验按各算子实际结构体计算，不同型长度不同是正常的。

## 4 TilingKey 编码（场景位，禁止枚举 dtype）

```cpp
// assign 型三场景位（张量源；nondet 不消费的维度用占位值；
// 标量源（值同）场景 deterministic 位即 dedup 位，仅重复度驱动）
ASCENDC_TPL_ARGS_DECL(
    Xxx,
    ASCENDC_TPL_UINT_DECL(countMode, 1, ASCENDC_TPL_UI_LIST, XXX_COUNT_U32, XXX_COUNT_I64),
    ASCENDC_TPL_UINT_DECL(keyDtype, 2, ASCENDC_TPL_UI_LIST, XXX_KEY_I16, XXX_KEY_I32, XXX_KEY_I64),
    ASCENDC_TPL_UINT_DECL(deterministic, 1, ASCENDC_TPL_UI_LIST, XXX_MODE_NONDET, XXX_MODE_DET));
```

- **SEL 只枚举有效组合**：det 的全组合（绑定 det TilingData）+ nondet 的最小组合（绑定 NondetTilingData；nondet 不消费的维度用占位值）。
- `REGISTER_TILING_DEFAULT` 取**较大者**（det 结构 ≥ Nondet 结构即覆盖安全）。
- dtype 组合由 spec 决定，经 DTYPE_VAR/DTYPE_INDEX 宏编译期展开，TilingKey 不枚举 dtype。
- `keyDtype` 按目标元素数自适应：`≤32767 → int16`、`≤2^31-1 → int32`、否则 int64（哨兵 key 需占 KeyT 最大值，见 [bounds](../../generic/bounds.md) §4）。

## 5 术语映射

| 概念 | dim 形态（scatter(reduce=none) 等） | nd 形态（scatter_nd_update / tensor_scatter_nd_update） |
|------|-------------------------------------|---------------------------------------------------|
| 被写张量 | var / self | var |
| 索引张量 | index | indices |
| 源张量 | src / source | updates |
| 收集轴 | dim / axis（负值归一化） | —（K = indices.shape[-1]，索引维数） |
| 散点数 N | index.numel() | prod(indices.shape[:-1]) |
| 目标 slice | prod(var.shape[dim+1:]) | prod(var.shape[K:]) |
| 写语义 | assign（覆盖 =） | 同左 |
| src 表示 | src / source（张量源）；标量 value 子场景值来源二选一（标量属性 / 标量张量输入 rank-0 或 shape=[1]） | updates（张量源）；标量 value 同左 |

## 6 范式参数常量（集中声明，各处引用）

| 常量 | 值 | 说明 | 可调 |
|------|-----|------|------|
| `THREAD_NUM` | 1024 | kernel 编译期线程常量（`__launch_bounds__` 与 `dim3` 共用，不入 TilingData） | 是（按平台） |
| `MAX_RANK` | 8 | dims/strides 数组定长上限（var rank ≤ 8；nd 的 K ∈ [1, rank]） | 是 |
| `DCACHE_MIN` / `DCACHE_MAX` | 32KB / 128KB | DCache 预留范围：越大性能可能越好，按剩余 UB 决定（SortLib 依赖） | 是（按剩余 UB） |
| `SENTINEL_KEY` | KeyT 最大值 | 越界跳过形态的哨兵（要求 totalVarElems ≤ KeyT_max - 1） | 否 |
