# ArgReduce 范式参考实现：arg_max_with_value 包

> **来源**：ops-math 仓 `math/arg_max_with_value`（arch35 正式版）**原样移植**，未改一行代码。许可遵循本仓 LICENSE（CANN Open Software License Version 2.0，与源仓同源）。
>
> **角色**：ArgReduce 范式的**唯一代码级参考实现**。范式知识（怎么推导、约束是什么）以 [../../../..](../../../../) 文档族为准；本包提供"推导结果在 AscendC 上长什么样"的代码事实。**实现参考用于核对 API 与组织方式；名称或类型匹配不能证明可以直接复制或融合，算法与参数需依据当前规格判断。**

---

## 0 先回答一个必然会被问到的问题：范式用三层结构，为什么 example 保持 8 模式？两者一致吗？

**一致，而且是刻意设计的一致性。** 关键要区分两件事：

| | 范式文档 | example |
|---|---|---|
| 回答的问题 | **怎么推导**：给定一个算子的规格，该走哪条路、为什么、有什么约束 | **结果长什么样**：推导出的模式在 AscendC 上具体怎么写成代码 |
| 组织方式 | 三层（选择语义 / 场景分类 / 正交维度） | 8 个模式（工业实现的工程命名） |
| 关系 | 8 模式是三层选择的**组合产物** | 每个模式都能被范式决策树**推导出来** |

**为什么不让 example 也按三层重写**：example 是工业实现的逐字节移植，它的价值恰恰在于**未经改写**——保留了实测阈值（如 R<VL 走 Gather）、家族复用结构（一套实现支撑 argmax/argmin/双输出四个算子）、UT golden。改写会丢掉这些事实，并与 ops-math 的后续演进脱节。范式文档已经明确：**知识按原因组织，example 保结果基线**。

**"一致"的可验证定义**（这不是口头保证，有实证）：把 example 的 tiling 判定代码（`op_host/arch35/arg_common_base_tiling_arch35.cpp` 的 `SetShapeInfo` / `SetShapeInfoHighPerf`）当作标准答案，用范式 `mode-mapping.md` 的决策树对 468 个白盒用例做预测——**实际路由与决策树预测 0 分歧**。也就是说：**example 里任何一个 shape 走哪个模式，范式都能推出来**。

**为什么 example 不能换成"三层结构版本"**：范式是知识组织方式（怎么想），不是代码组织方式（怎么写）。把 example 按三层重构成 3 个场景文件会丢掉 8 模式里的性能特化路径（GATHER/双轴/RA），这些正是工业实现的价值所在。正确做法是保持 example 原样 + 用本文档的导航表把"范式概念 → example 文件/行号"显式映射起来——这就是本 README 第 2 节做的事。

---

## 1 为什么选它（选型依据）

| 依据 | 事实 |
|---|---|
| **家族底座** | `arg_max_v2` / `arg_min` / `arg_min_with_value` 的 `_apt.cpp` 全部 include 本包的 arch35 头文件——**一套实现支撑四个算子**（模板参数 `<T_x, T_idx, T_y, isMin, withValue>`），是"算子族一套实现"的工业实证 |
| **模式最全** | 8 种 tiling 模式（AR / ARA / RA / COPY / AR_GATHER / ARA_GATHER / ARA_CUT_A_AND_NEXT_A / GROUP_REDUCE）+ ARA 子模式 101~104 全部落地 |
| **面向最新架构** | ascend350 / ascend950（arch35） |
| **可验证** | 已在 Ascend950PR 上编译出包并通过 aclnn 真机执行（`execute samples success`）；tiling UT / infershape UT 齐备 |

**适用边界（如实声明）**：

- 仅 arch35（ascend350/950）。A2/A3 平台请参考 ops-math 的 `experimental/math/arg_max_with_value`（按轴位置分 LAST/NLAST 的另一套实现）。
- 家族语义为 **max 方向 + 双输出（values+indices）**。派生 argmin / 单输出 argmax 见 §4。
- 不含"谓词型"变体（arg_first / find_index / top-k）——这类变体的选择段替换方法见 §5。

---

## 2 范式概念 → example 文件/行号 精确导航

> 用法：在范式文档里读到某个概念，用下表定位到本包的**具体文件和行号**，核对手法。行号基于本包当前版本。

### 2.1 模式判定（范式 mode-mapping.md ↔ example tiling）

| 范式概念 | 包内位置（文件:行号） | 看什么 |
|---|---|---|
| 模式枚举（8 模式分段值） | `op_host/arch35/arg_common_base_tiling_arch35.h:131-149` | `enum class ArgMaxWithValueTilingMode`：AR_CUT_A=10001 … GROUP_REDUCE=30001 |
| **第一级判定**（COPY / AR / ARA） | `op_host/arch35/arg_common_base_tiling_arch35.cpp:217-265`（`SetShapeInfo`） | R==1→COPY；末维或 A0==1→AR；否则 ARA —— 对应范式 mode-mapping.md §2 |
| **第二级特化**（RA / Gather / GROUP_REDUCE / 双轴） | 同文件 `:279-338`（`SetShapeInfoHighPerf`） | 每个特化的触发条件（R×eb vs vRegSize、A1 vs coreNum、0.85 回退）——对应 mode-mapping.md §3 |
| ARA 载入子模式 aRaMode 101~104 | 同文件 `:397-410`（`CalcSplitInfoForArA`，经 `CalcSplitInfoForArAAndTryMode4` 分发） | 对应 mode-mapping.md §4 |
| 切分计算（blkFactor/blkTailFactor/cutRSize） | 同文件 `:339` 起的 `CalcSplitInfo` 链 | 对应范式 splitting.md §2/§3（余数摊派） |
| TilingData 字段定义 | `op_host/arch35/arg_common_base_tiling_arch35.h:27-48` | 20 字段，对应 overview.md §4 |

### 2.2 选择段手法（范式 highlevel-api.md ↔ example kernel）

| 范式手法 | 包内位置（文件:行号） | 看什么 |
|---|---|---|
| **手法 B：ArgMaxV1 寄存器级**（AR 主路径） | `op_kernel/arch35/arg_max_with_value_base.h:80`（声明）/ `:586`（实现） | 逐行 LoadUnAlign + GE 比较（first-occurrence）+ NE 自比较（NaN）+ Or 掩码 + 值/位置双 Select |
| int64 / R<64 变体 | 同文件 `:118`、`:121` | `ArgMaxV1Int64` / `ArgMaxV1Int64ForRLessThanVL64` |
| 位置寄存器等宽选型 | 同文件 `:586` 实现内的 T5 模板参数 | fp16→uint16（128 lane），fp32→uint32 |
| **手法 D：ARA 逐行迭代** | `op_kernel/arch35/arg_max_with_value_ara.h` | 跨步行处理；MODE1~4 分发 |
| **手法 C/F：Gather / RA** | `op_kernel/arch35/arg_max_with_value_ar_gather.h`、`ara_gather.h`、`ra.h` | Gather 转置与 A0 铺 lane 的向量化 |
| **手法 E：NDDMA 重排** | `op_kernel/arch35/arg_max_with_value_ara.h`（MODE2 子路径） | 重排 [A1,R,A0]→[A1,A0,R] 后走 AR |
| **GROUP_REDUCE 两阶段** | `op_kernel/arch35/arg_max_with_value_group_reduce.h` | Phase1 局部(值,位置)→workspace→`SyncAll`→Phase2 聚合（AR/RA 两分支） |
| **COPY 场景** | `op_kernel/arch35/arg_max_with_value_copy.h` | 透传 + indices 全 0（注意：本包 COPY 不读输入，arg_first 等谓词型需改为读入判谓词） |

### 2.3 入口与分发（范式 kernel-entrance.md / overview.md §3 ↔ example）

| 范式概念 | 包内位置（文件:行号） | 看什么 |
|---|---|---|
| kernel 入口与分发 | `op_kernel/arg_max_with_value_apt.cpp:35`（入口）/ `:46-113`（`TILING_KEY_IS` 链） | **注意：这是机制 B（旧式）**。新算子按范式要求用机制 A（`ASCENDC_TPL_ARGS_DECL/SEL` + `if constexpr`），见 overview.md §3.1 |
| 模板参数约定 | `op_kernel/arg_max_with_value_apt.cpp` 各分支实例化 | `<DTYPE_X, TIdxMid, DTYPE_INDICE, withValue, isMin>` |
| 算子定义注册 | `op_host/arg_max_with_value.cpp` | dtype 组合与属性声明 |
| infershape（keep_dims / indice_dtype） | `op_host/arg_max_with_value_infershape.cpp`、`arg_common_base_infershape.*` | 输出 shape 与类型推导 |

### 2.4 验证资产（范式 Validation ↔ example tests）

| 范式验证要求 | 包内位置 | 看什么 |
|---|---|---|
| 模式判定 golden 对拍 | `op_host/arch35/test_arg_max_with_value_tiling.cpp` | 每个 shape → 期望 tilingKey 的 UT 结构（新算子的 tiling UT 照此写） |
| infershape UT | `op_host/test_arg_max_with_value_infershape.cpp` | shape 规则用例 |
| 精度 golden | `tests/assets/golden.py` | CPU 参考实现写法（**新算子必须显式 NaN-aware，禁用裸 np.argmin/np.argmax 默认 NaN 语义**） |
| ST 用例与执行器 | `tests/st/ttk_kernel_arg_max_with_value_st.csv`、`tests/st/aclnnMaxDim/executor_aclnn_max_dim.py` | 用例 CSV 结构与执行器 |
| aclnn 两段式接口 | `op_api/aclnn_max_dim.cpp`、`op_api/argmax_with_value.*` | 接口封装 |

---

## 3 阅读路线（按角色）

| 角色 | 建议顺序 |
|---|---|
| **设计者**（写 DESIGN.md） | 先读范式 `mode-mapping.md` 决策树 → 用 §2.1 对照 example tiling 验证推导 → 再读 `splitting.md` 预算公式，对照 §2.1 切分计算 |
| **开发者**（写 kernel） | 先读范式 `highlevel-api.md` 手法 A-F → 用 §2.2 找到对应模式的 kernel 文件 → 只参考**当前模式**的文件（禁止跨模式混用） |
| **评审者** | 用范式 `design-spec.md` 一致性检查表 + §2 导航表核对设计是否与 example 手法一致 |
| **测试开发者** | 用 §2.4 的 UT/golden/ST 结构为模板 |

---

## 4 派生 argmin / 单输出 argmax（模板参数）

本包天然支持四个算子（模板参数控制，同一份代码）：

| 目标算子 | 模板参数 | 改动 |
|---|---|---|
| ArgMaxWithValue（本包） | `isMin=false, withValue=true` | 无 |
| ArgMinWithValue | `isMin=true, withValue=true` | 换 `_apt.cpp` 实例化参数；比较方向自动切为 LE |
| ArgMax（单输出 indices） | `isMin=false, withValue=false` | 关闭 values 输出分支（`if constexpr (withValue)`） |
| ArgMin（单输出 indices） | `isMin=true, withValue=false` | 同上 + isMin |

参考同族实现：`ops-math/math/arg_max_v2`、`arg_min`、`arg_min_with_value` 的 `_apt.cpp` 就是三种实例化。


## 5 派生谓词型变体（arg_first / find_index / top-k）

本包**不包含**谓词型变体的实现——它们的差异不在场景/切分，而在**选择段的比较逻辑**。按范式 [variants.md](../../../../generic/variants.md) 的挂载原则替换即可：

| 变体 | 替换点 | 复用部分 |
|---|---|---|
| arg_first（首个非零） | 选择段主键从"值比较"换成"非零掩码"：`mask=(v!=0)`，按 mask 做首达合并（命中优先、命中取最小 position） | 场景判定、切分、GroupReduce、位置流、精度约束全部复用 |
| find_index(target) | 同上，谓词换成 `v==target`；无命中哨兵值算子级声明 | 同上 |
| arg_topk | 选择算子从"二选一"换成"两个有序 top-k 列表归并"（仍满足结合律，跨块/跨核结构不变）；buffer ×k | 同上 |

**注意两处与 argmax 不同的细节**（实战踩坑点）：
1. **COPY 场景**：argmax 系 COPY 不读输入直接写 0；谓词型必须**读入判谓词**（非零 → 0，为零 → -1）
2. **NaN 语义**：谓词型按 IEEE 754 判定 `NaN != 0` 为真（NaN 算命中），需在算子规格中显式声明

---

## 6 本包的已验证事实

| 项 | 证据 |
|---|---|
| 编译 | Ascend950PR 工具链编译出 run 包（本机实测） |
| 真机执行 | aclnn 两段式调用 `execute samples success`（本机 950PR 实测） |
| 模式判定一致性 | 白盒 468 用例：实际 tiling 路由与范式决策树预测 **0 分歧** |
| 家族复用 | arg_max_v2 / arg_min / arg_min_with_value 三个算子的 arch35 实现均 include 本包头文件（ops-math 源仓事实） |

---

## 7 与范式文档的引用关系

- 本包**不重复**范式知识——模式判定、语义约束、资源预算全部以 [../../../..](../../../../) 文档族为准
- 本 README 的 §2 是两者之间的**唯一桥接**：左列是范式概念，右列是代码事实
- 修改本包代码时须同步 §2 的行号锚点（本包预期保持"原样移植"，除非上游 ops-math 更新）
