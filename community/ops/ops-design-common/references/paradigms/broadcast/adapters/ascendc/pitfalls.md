# Broadcast 范式 AscendC 陷阱与问题定位（聚合视图）

> **层级**：AscendC 语言适配层。本文件聚合各适配层文档中的编译约束与实战错题，便于一站式排查；详细上下文见各源文档（[kernel-entrance.md](kernel-entrance.md)、[kernel-template.md](kernel-template.md) §8、[tiling.md](tiling.md) §6）。

## 1 编译系统关键约束

| 错误 | 阶段 | 原因 |
|------|------|------|
| `unexpected type name` | INFERCHANNEL | `GET_TILING_DATA_WITH_STRUCT` 缺少 `REGISTER_NONE_TILING` |
| `no matching function for call to 'func_X_tilingkey'` | FATBIN | kernel.h 缺 `#include struct.h`，构建系统走了 `#define` 包装器 |
| `cast not allowed in aicore` | INFERCHANNEL | 试图 `reinterpret_cast` 跨地址空间 |
| `undeclared identifier 'GetBlockIdx'` | INFERCHANNEL | 缺 `AscendC::` 命名空间前缀 |

**交叉编译器限制**：宏参数里不能有 `<>`（模板尖括号被误解析为比较运算符，用 `using TilingData4 = ...` 别名规避）；`reinterpret_cast` 不允许（`__gm__` 到非 `__gm__`）；`__gm__` 到 `__gm__` 的 C 风格转换允许。

## 2 多 TilingData 关键坑汇总

| 坑 | 说明 |
|----|------|
| `<>` 在宏里 | 交叉编译器把 `BroadcastTilingData<4>` 的 `<` 当小于号。用 `using TilingData4 = ...` 别名 |
| `REGISTER_NONE_TILING` | 必须写在 `GET_TILING_DATA_WITH_STRUCT` 之前，否则宏不认类型名 |
| `KERNEL_TASK_TYPE_DEFAULT` | 声明核类型（AIV），模板化算子必须 |
| kernel.h 必须 include struct.h | 构建系统读到 `ASCENDC_TPL_ARGS_DECL` 才知道这是模板化算子，走模板编译路径而不是 `#define` 包装器 |
| 模板函数与 `#define` 不兼容 | 构建系统对非模板函数生成 `#define func func_0_tilingkey` 包装器。模板化算子不触发这个 |

## 3 实战错题集

| 案例 | 症状 | 根因 | 修复 |
|------|------|------|------|
| count=0 | [256,128,64] 全同形，device 结果全错 | RANK=4 但实际 rank=3，`maxBroShape[3]=0`（后补 0）乘进 innerCount | TilingData 改前补 1（padding 进 NDDMA 最外维无害） |
| NDDMA UB 溢出 | [17,7,13,24,29,6] 6 维，RANK=8 | ND=5 写死，NDDMA 覆盖范围与 flat loop / outer loop 三层重叠 | `nddmaDims = min(RANK-k, ND)`，三层不重叠 |
| CalcOffset bytes vs elements | Tiling 正确，device 仍错 | `CalcOffset` 末尾 `* sizeof(T)` 返回 bytes，`gmIn_[bytes]` 当元素索引用 | CalcOffset 返回元素数 |
| loopLpSize/loopRpSize 未初始化 | device 结果随机错误，同输入多次跑结果不同 | `NdDmaLoopInfo` 的 `loopLpSize`/`loopRpSize` 是随机值 | Init 循环内加 `loopLpSize[nd]=0; loopRpSize[nd]=0` |
| 缺 V_MTE2 | 精度失败，怀疑同步 | S5c Vector 读 UB2 毕，S7a MTE2 写 UB2，中间 S6 无 V_MTE2 保护 | S6 后加 `SetFlag V_MTE2 + WaitFlag V_MTE2` |

## 4 问题定位 4 步法

1. **Tiling 日志 × 设计文档**：逐字段对照设计文档预期值，从 `tiling->` struct 数组打印（R 维全量含 padding），禁止从中间 vector 打印
2. **Python 交叉验证**：计算流序列 / 循环 count / 地址偏移 / NDDMA 参数，4 块纯逻辑各出 Python 对比，随机 case 100 次（harness 见 [../../generic/verification.md](../../generic/verification.md)）
3. **API 接口契约检查**：每个 `DataCopy`/`DataCopyPad`/NDDMA 调用点，逐项对照约束速查表（方向、count 约束、入参单位、结构体字段，见 [kernel-template.md](kernel-template.md) §5.3）
4. **持有法则 sync 推导**：按持有法则格式标注每步三态持有 → 推导 RAW/WAR → 和代码 SetFlag/WaitFlag 逐项对照

## 5 Tiling 维测关键字段（日志对照用）

| 日志字段 | 常见 bug |
|---------|----------|
| `maxBroShape` | padding 方向/值（前补=1 vs 后补=0），broadcast 轴是否被 squeeze |
| `input[i].stride` | 播轴 stride=0，padding stride=0 |
| `split(ubSplitIdx, ubFactor, ubOuter)` | ubSplitIdx 是否因 padding 错位 |
| `rank→R` | 实际 rank vs 模板 RANK 映射 |
