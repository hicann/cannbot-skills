# SortSelect 范式 AscendC 语言适配层

> **层级**：AscendC 语言适配层（adapters/ascendc/）。与 [通用层](../../generic/index.md) 组合消费，语言路由见 [范式入口](../../patterns.md)。

## 定位与能力映射

本层承载实际 API、模板参数、代码组织、编译器约束。排序能力及资源限制以 [capability-mapping.md](capability-mapping.md) 的实现和适用域为准。

## 阅读路径

| 任务阶段 | 入口 | 内容 |
|---|---|---|
| 方案设计、实现替换与验收 | [design-spec.md](design-spec.md) | 输入输出、逐路线复用基线、替换证据及独立验收约束 |
| 整体模型与模板导航 | [overview.md](overview.md#按模板阅读) | 五类模板的 Tiling / Buffer / Kernel，以及公共符号 |
| 输出适配 | [公共映射](../../generic/output-mapping.md) → [AscendC 表达](output-mapping.md) | 一组输出参数生成 Sort、Argsort、TopK、KthValue，统一布局、批次复用与写回 |
| Host Tiling | [tiling-preprocess.md](tiling-preprocess.md) → [tiling.md](tiling.md) → 命中模板的 `tiling.md` | 全局选路后推导批次、块长、核数及尾块，含伪码和演算 |
| 参数对应 | [tiling-parameters.md](tiling-parameters.md) | 各 schedule 的字段含义、单位及 Host/Kernel 配对 |
| API 与布局包装 | [API 与非末轴转置包装](kernel-entrance.md#api-与非末轴转置包装) | L2/L0 分工、末轴规范化与转置、两个输出还原 |
| Kernel | [kernel-entrance.md](kernel-entrance.md) → [模板计算索引](kernel-template.md#按模板读取) | 命中模板的阶段伪码、GM/UB 地址、索引含义与同步；特殊值另查 helpers |
| Buffer | [预算索引](dag-buffers.md#按模板读取) → 命中模板的 `dag-buffers.md` | 逐项物理字节、峰值、别名条件和跨批复用 |
| 陷阱 | [pitfalls.md](pitfalls.md) | 类型、索引、数据布局与同步约束 |
| 实现核对 | [Sort 示例](example/sort/README.md) | 模板机制的具体实现、API caller 与 SDK 绑定，不限定生成工程的代码组织 |

## 消费规则

只读命中的本层与通用层。编码前按目标规格核实参考实现的 dtype、轴长、稳定性和平台。

先用全局选路确定候选，按需展开当前模板及其必要依赖。单条路线的设计可组合该模板的 Tiling、Buffer、Kernel 三份文档，再按问题读取输出适配或辅助约束；跨路线比较时按需展开相关模板。既有算法与同步协议的保留要求仍按 `design-spec.md` 执行。
