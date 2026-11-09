# ArgReduce Kernel 入口与模式分发（AscendC）

> **层级**：AscendC 语言适配层。kernel 入口注册、模板参数、模式分发结构。

## 1 入口注册与模板参数

```cpp
extern "C" __global__ __aicore__ void arg_reduce_custom(GM_ADDR x, GM_ADDR indices,
                                                        GM_ADDR values, GM_ADDR workspace,
                                                        GM_TILING_PARAM(tilingData) /* 按算子裁剪 */)
{
    if (ASCEND_GLOBAL_UUID_FORMAT_WORKSPACE == workspace) { return; }   // 框架约定
    // 索引中间类型按输入 dtype 编译期选择（int64 值 → int64 中间索引）
    using TIdxMid = std::conditional_t<ORIG_DTYPE_X == ge::DT_INT64, int64_t, int32_t>;
    ArgReduceDispatch<DTYPE_X, TIdxMid, DTYPE_INDICE, WITH_VALUE, IS_MIN>(
        x, indices, values, workspace, &tilingData);
}
```

**模板参数约定**（算子族一套实现）：

| 参数 | 承载 | 来源 |
|------|------|------|
| `DTYPE_X` / `DTYPE_INDICE` | dtype 实例 | 框架 `DTYPE_<INPUT_NAME>` 宏（不进 TilingKey） |
| `TIdxMid` | 索引中间类型 | 编译期按 `ORIG_DTYPE_X` 推导 |
| `WITH_VALUE` | 是否输出值 | 算子规格（argmax=false / maxdim=true） |
| `IS_MIN` | min/max 方向 | 算子规格 |

## 2 模式分发

**机制 A（新式，新开发算子必选）**——ASCENDC_TPL 模板参数 + `if constexpr` 编译期分发：

```cpp
// tiling_key.h：ASCENDC_TPL_ARGS_DECL 声明 dtype 维（框架绑定）+ 模式维（tiling 决策），见 overview.md §3.1
// kernel 侧：模板参数分发，编译期展开
template <typename TX, typename TIdxMid, typename TI, uint64_t MODE, bool withValue, bool isMin>
__aicore__ inline void ArgReduceDispatch(GM_ADDR x, GM_ADDR indices, GM_ADDR values,
                                         GM_ADDR workspace, const ArgReduceTilingData* tiling)
{
    if constexpr (MODE == TILING_MODE_AR)               { ArgReduceAr<TX, ...> op; ... }
    else if constexpr (MODE == TILING_MODE_ARA)         { ArgReduceAra<TX, ...> op; ... }   // 内部 switch aRaMode 101~104
    else if constexpr (MODE == TILING_MODE_ARA_CUT_A_AND_NEXT_A) { /* 双轴切核 */ }
    else if constexpr (MODE == TILING_MODE_AR_GATHER)   { /* Gather 转置 + 向量比较 */ }
    else if constexpr (MODE == TILING_MODE_ARA_GATHER)  { /* cutASize==1 分支 + NDDMA 路径 */ }
    else if constexpr (MODE == TILING_MODE_RA)          { /* A0 铺 lane */ }
    else if constexpr (MODE == TILING_MODE_COPY)        { /* 透传 + Duplicate(0) */ }
    else if constexpr (MODE == TILING_MODE_GROUP_REDUCE){ /* 两阶段：Phase1 + SyncAll + Phase2 */ }
}
```

**机制 B（旧式，仅 example 开源参考实现采用）**——`TILING_KEY_IS(...)` if/else 链（example `arg_max_with_value_apt.cpp`）。工程标准已废弃该宏，新算子禁用；阅读参考实现时按此理解。

**要点**：

1. 每个分支实例化**不同的模板类**（编译期展开，每模式一份 binary；dtype × 模式组合由框架管理）
2. 模式内二级分支为运行时 if（如 AR 按 `cutRSize != rSize` 分 WithCutR / WithoutCutR；GROUP_REDUCE 按 `isRaSplit` 分 AR/RA 聚合分支）
3. EMPTY 不占分发分支：`realCoreNum == 0` 时 kernel 全核早退（Host 侧 `SetBlockDim(1)`）
4. `if constexpr` 只用于类内 dtype 子分支（如 bf16→fp32 计算路径），**不用于模式级分发**

## 3 模板类骨架（统一约定）

每个模式模板类遵循统一骨架（完整代码见 templates/ 与 example/）：

```cpp
template <typename TX, typename TIdxMid, typename TI, bool withValue, bool isMin>
class ArgReduceAr {
public:
    __aicore__ inline ArgReduceAr& Init(GM_ADDR x, GM_ADDR indices, GM_ADDR values,
                                        GM_ADDR workspace, const ArgReduceTilingData* tiling);
    __aicore__ inline void Process();          // 主循环：A1 外层 → 行载入 → ArgMaxV1 → 写出
private:
    // 值/位置双输出、NaN 掩码、平局 Compare 方向由模板参数 isMin 在 ArgMaxV1 内部落地
};
```

## 4 分发完整性自查清单

设计评审时手工核对：TilingKey 全模式覆盖（无遗漏模式）、模板参数（dtype/isMin/withValue）不进 TilingKey、EMPTY 早退路径存在、GROUP_REDUCE 分支含全核栅栏与 Phase 2 聚合。
