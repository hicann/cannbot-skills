# argreduce 模板总览（AscendC）

> 描述本范式 kernel 模式划分、TilingKey 编码、TilingData 划分及三者之间的联系。与工业实现的 8 模式体系对齐（映射与决策树见 [mode-mapping.md](mode-mapping.md)）。

| 概念 | 说明 |
|------|------|
| **模板** | Kernel 的编译期特化实例。同一份 kernel 代码根据 TilingKey 值的不同组合，编译出不同的执行路径 |
| **TilingKey** | Host 端 tiling 阶段算出的编译期开关，决定运行时框架加载哪个编译实例 |
| **TilingData** | Host 端 tiling 阶段算出的运行时参数（struct）。一个模式对应一份 TilingData |

**三者协作流程**：TilingKey 决定"走哪条路"，TilingData 提供"路上的参数"。

## 1 术语表（白话版）

> 先看懂这张表，后面的模式名就只是这些概念的组合。

| 术语 | 白话解释 | 例子 |
|------|---------|------|
| **A1 / R / A0** | 把任意 shape 按归约轴切成三段：归约轴**前面**的维度乘积（要做多少场选择）/ 归约轴本身长度（每场几个选手）/ 归约轴**后面**的维度乘积（相邻选手隔多远） | [2,100,4000] dim=1 → A1=2, R=100, A0=4000 |
| **AR 场景** | 选手**排成一排**（R 元素在内存里连续），一行一行处理 | [64,60000] dim=1 |
| **ARA 场景** | 选手**隔着站**（R 元素跨步排列，间隔 A0），要重排或跨步取 | [2,100,4000] dim=1 |
| **RA 特化** | ARA 的特例：只有一场比赛（A1==1），选手跨步但"下一名选手"连续——可以把连续方向铺到向量通道上加速 | [50,100000] dim=0 |
| **COPY 场景** | 退化：每场只有 1 个选手（R==1），不用比，位置直接写 0 | [4,1] dim=1 |
| **EMPTY** | 空 tensor（shape 含 0），直接返回 | [0,5] |
| **GroupReduce** | R 太长单核算不完、但行数又太少开不满核——把**一行拆给多个核**各算一段，再合并（需要跨核同步） | [50,8192] dim=1 |
| **Gather 特化** | R 很短（一条向量装得下）但行数很多——用 Gather 指令在片上把"一排"转成"一列"再一次性比较 | [8192,100] dim=1 |
| **双轴切核** | ARA 场景里，A1 和 A0 都很大——两个方向同时切给多核 | [2,100,4000] dim=1 |
| **isMin / withValue** | 模板参数：找最小还是找最大 / 是否同时输出值。**不进 TilingKey**（编译期固定） | argmin 的 isMin=true |
| **indice_dtype** | 位置输出的整数类型 int32/int64。**不进 TilingKey** | int32 |
| **D_T** | 数据的 dtype（框架按输入自动生成编译实例，开发者不用管） | float16 |

**为什么要分这些场景/模式**：① 场景（AR/ARA）决定数据怎么搬、能不能用现成 API（连续可以用整行归约，跨步不行）；② 性能特化（Gather/双轴/RA/GroupReduce）只是"在某些 shape 下更快的走法"，不影响正确性。

## 2 模式分类

模式 = **主场景 × 性能特化**（两级，详见 [mode-mapping.md](mode-mapping.md)）：

### 2.1 第一级：主场景（看归约轴在哪、数据连不连续）——决定"必须怎么写"

| 模式 | 判定 | 白话 |
|------|------|------|
| COPY_ONLY | R == 1 | 每场 1 个选手，不用比：值原样传，位置全写 0 |
| AR_CUT_A | A0 == 1（含 dim 为末维） | 选手排成一排：一行一行搬进来直接归约；行太长放不下就切成几段 |
| ARA_CUT_A | A0 > 1 | 选手隔着站：要么重排成连续再算，要么逐行跨步比较（比 AR 麻烦得多） |

### 2.2 第二级：性能特化（同样正确，只是更快）——按 shape 的相对大小选择

| 特化 | 触发本质（白话） | 覆盖 |
|------|---------|------|
| RA_CUT_A | A1==1，nextA 铺 lane 向量化 | ARA |
| AR_GATHER / ARA_GATHER | R < 一条向量寄存器宽度（R 很小），Gather 转置后一次向量比较 | AR / ARA |
| ARA_CUT_A_AND_NEXT_A | A1 与 A0 都大，双轴切核 | ARA |
| GROUP_REDUCE | R 大 + A 方向无并行度，R 切多核 | 任意 |

## 3 TilingKey

### 3.1 核心原则与两种落地机制

**原则**（与机制无关）：**dtype 不是 tiling 决策维**——由框架按输入 dtype 自动实例化；**模式（场景×特化）是 tiling 决策维**——由 Host 侧 tiling 现算选择。

模式用**分段枚举值**承载，数值分段有语义：1xxxx = 基础路径，2xxxx = 向量化优化，3xxxx = 跨核归约：

```cpp
enum class ArgReduceTilingMode : uint64_t {
    AR_CUT_A             = 10001,   // AR 基础
    ARA_CUT_A            = 10002,   // ARA 基础
    ARA_CUT_A_AND_NEXT_A = 10012,   // ARA 双轴切核
    COPY_ONLY            = 10003,   // R==1 透传
    AR_GATHER            = 20001,   // AR Gather 特化
    ARA_GATHER           = 20002,   // ARA Gather 特化
    RA_CUT_A             = 20003,   // RA 向量化特化
    GROUP_REDUCE         = 30001,   // 跨核分组归约
};
```

**机制 A（新式，当前工程标准，新开发算子必选）**——ASCENDC_TPL 模板参数：

```cpp
// 依据 ascendc-registry-invoke-template（禁止废弃的 TILING_KEY_IS，必须 ASCENDC_TPL_ARGS_DECL）
ASCENDC_TPL_ARGS_DECL(ArgReduceCustom,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT16, C_DT_FLOAT, C_DT_BF16, ASCENDC_TPL_INPUT(0)),  // dtype 维：框架绑定输入0的 dtype（DTYPE_X 宏语义的模板载体，非 tiling 决策）
    ASCENDC_TPL_UINT_DECL(TILING_MODE, 8, ASCENDC_TPL_UI_LIST, 10001, 10002, 10003, 10012, 20001, 20002, 20003, 30001)  // 模式维：tiling 决策
);
// SEL 按 dtype × 模式展开组合；Host 侧 ASCENDC_TPL_SEL_PARAM(context, 输入dtype, mode) 选择；
// kernel 侧以模板参数 + if constexpr 编译期分发
```

**机制 B（旧式，开源正式版实现采用，example 包代码形态）**——`SetTilingKey(枚举值)` + kernel 侧 `TILING_KEY_IS(...)` if/else 链（见 [example/](example/) `arg_max_with_value_apt.cpp`）。阅读参考实现时按此机制理解；新算子开发禁用（工程标准已废弃）。

### 3.2 设计决策

- **dtype 不进 tiling 决策**：机制 A 经 `ASCENDC_TPL_INPUT(0)` 框架绑定；机制 B 经 `DTYPE_<INPUT_NAME>` 宏——两种机制语义等价，dtype 组合的编译实例由框架管理
- **isMin / withValue 不进模式维**：模板参数 `<DTYPE_X, T_INDEX_MID, DTYPE_INDICE, withValue, isMin>` 承载，一套实现支撑 argmax/argmin/双输出四算子
- **索引中间类型按输入 dtype 编译期选择**：`ORIG_DTYPE_X == DT_INT64` 时中间索引选 int64，否则 int32 路径（位置寄存器按值通道等宽选型，见 capability-mapping）
- **模式收敛与裁剪**：{COPY_ONLY, AR_CUT_A, ARA_CUT_A} 是**正确性完备集**——任意 shape 必然路由到其中之一（AR/ARA 内含切 R 循环，覆盖超大 R）；RA_CUT_A / AR_GATHER / ARA_GATHER / ARA_CUT_A_AND_NEXT_A / GROUP_REDUCE 均为**性能特化**，可按算子范围裁剪。裁剪决策须记录于设计文档（不纳入依据 + 典型场景命中路径 + 扩展路径），禁止无记录地缺省
- **TilingData 模式字段双写**：模式枚举值同时写入 TilingData 字段（机制 A 命名建议 `tilingMode`，机制 B 命名 `tilingKey`；语义相同——框架分发 + 调试双通道）

### 3.3 模式 → 模板类对应

| TilingKey | 模板类（物理分文件） | 数据流 |
|---|---|---|
| AR_CUT_A | `ArgReduceAr` | 按 A1 循环：整行载入 → ArgMaxV1（cutRSize != rSize 时切 R 循环） |
| ARA_CUT_A | `ArgReduceAra` | 按 aRaMode 101~104 分发（GatherARA / NDDMA 重排 / 切R / 切R+nextA） |
| ARA_CUT_A_AND_NEXT_A | `ArgReduceAraCutAAndNextA` | 双轴切核 + flat/normal 两条子路径 |
| AR_GATHER / ARA_GATHER | `ArgReduceGather` | Gather 转置 → RA 式向量比较 |
| RA_CUT_A | `ArgReduceRa` | A0 铺 lane，R 逐行比较 |
| COPY_ONLY | `ArgReduceCopy` | values DataCopy 透传，indices Duplicate(0) |
| GROUP_REDUCE | `ArgReduceGroupReduce` | Phase1 局部(值,位置)→workspace→SyncAll→Phase2 聚合（AR 分支 core0 聚合 / RA 分支全核聚合） |

分发落地：机制 A 用模板参数 + `if constexpr` 编译期分发；机制 B（example）用 `TILING_KEY_IS` if/else 链（见 [kernel-entrance.md](kernel-entrance.md)）。

## 4 TilingData

单份 struct（对齐工业实现 `ArgMaxWithValueTilingData`，全部模式共用，未用字段按模式忽略）：

```cpp
BEGIN_TILING_DATA_DEF(ArgReduceTilingData)
    // ─── 轴折叠 ───
    TILING_DATA_FIELD_DEF(uint64_t, aSize);        // A1
    TILING_DATA_FIELD_DEF(uint64_t, rSize);        // R
    TILING_DATA_FIELD_DEF(uint64_t, nextASize);    // A0
    // ─── 切分（valid 值，对齐在搬运侧处理）───
    TILING_DATA_FIELD_DEF(uint16_t, cutASize);     // A1 切核块大小
    TILING_DATA_FIELD_DEF(uint16_t, cutRSize);     // R 切块大小（Split / GROUP_REDUCE）
    TILING_DATA_FIELD_DEF(uint16_t, cutNextASize); // A0 切块大小（RA 批宽 / 双轴切核 / GROUP_REDUCE RA 分支共用）
    // ─── 多核 ───
    TILING_DATA_FIELD_DEF(uint64_t, realCoreNum);  // 实际使用核数
    TILING_DATA_FIELD_DEF(uint64_t, blkFactor);    // A1 切核因子
    TILING_DATA_FIELD_DEF(uint64_t, blkTailFactor);// A1 尾块因子
    TILING_DATA_FIELD_DEF(uint64_t, blkFactor2nd); // A0 切核因子（双轴）
    TILING_DATA_FIELD_DEF(uint64_t, blkTailFactor2nd);
    TILING_DATA_FIELD_DEF(uint64_t, blkNum2nd);    // A0 切核块数
    // ─── 模式 ───
    TILING_DATA_FIELD_DEF(uint64_t, tilingKey);    // 模式枚举值（与 SetTilingKey 双写）
    TILING_DATA_FIELD_DEF(uint64_t, aRaMode);      // ARA 载入子模式（101~104）
    // ─── 循环编排 ───
    TILING_DATA_FIELD_DEF(uint16_t, loopANum);     // A1 外层循环次数
    TILING_DATA_FIELD_DEF(uint16_t, cutAPerLoop);  // 每循环 A1 处理数
    TILING_DATA_FIELD_DEF(uint8_t,  isRaSplit);    // GROUP_REDUCE 内部分支（RA/AR）
    TILING_DATA_FIELD_DEF(uint16_t, gatherBlockSize); // Gather/聚合分块
    TILING_DATA_FIELD_DEF(uint8_t,  useFlatPath);  // 双轴切核 flat 子路径开关
    // ─── workspace ───
    TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize);// GROUP_REDUCE 用户 workspace
END_TILING_DATA_DEF;
```

**对比 Reduction 范式**：Reduction 按模板分 2 份 struct（Generic/Empty）；ArgReduce 因空 tensor 完全短路（无独立 Empty 模板，见 mode-mapping §1），单份 struct + 模式枚举承载全部路径——与工业实现结构一致。

## 5 总结

| 模式 | 场景 | 特化 | TilingKey | kernel 模板类 |
|---|---|---|---|---|
| COPY_ONLY | R==1 | — | 10003 | ArgReduceCopy |
| AR_CUT_A | AR | — | 10001 | ArgReduceAr |
| AR_GATHER | AR | R 很小 Gather 转置 | 20001 | ArgReduceGather |
| ARA_CUT_A | ARA | — | 10002 | ArgReduceAra（aRaMode 细分） |
| ARA_GATHER | ARA | R 很小 Gather 转置 | 20002 | ArgReduceGather |
| ARA_CUT_A_AND_NEXT_A | ARA | 双轴切核 | 10012 | ArgReduceAraCutAAndNextA |
| RA_CUT_A | ARA(A1==1) | RA 向量化 | 20003 | ArgReduceRa |
| GROUP_REDUCE | 任意 | R 切多核 | 30001 | ArgReduceGroupReduce |

## 6 产出

### 6.1 产出结果

结合算子规格，逐步产出：

1. **输出主场景**：按 [mode-mapping.md](mode-mapping.md) §2 第一级判定（EMPTY 短路 → R==1 → A0==1/A0>1）
2. **输出特化覆盖**：按 §3 第二级逐项判定（RA / GATHER / 双轴切核 / GROUP_REDUCE），输出最终模式枚举与判定 trace
3. **输出 TilingData**：按 §4 struct 填充字段来源（Host 侧计算见 [tiling-preprocess.md](tiling-preprocess.md)）

### 6.2 产出校验

| # | 校验项 | 规则 |
|---|--------|------|
| 1 | dtype / isMin / withValue 不进 TilingKey | 走 DTYPE 宏与模板参数 |
| 2 | tilingKey 字段与 SetTilingKey 双写一致 | 框架分发 + 调试双通道 |
| 3 | GROUP_REDUCE 核数口径 = 物理核数 | 栅栏全核到齐约束 |
| 4 | 模式判定 trace 可复现 | 每次覆盖有判定条件记录 |
| 5 | 位置精度边界已核查 | R 总长 vs INDEX_STORAGE_TYPE 上界；int32 输出另核 2^31 |
