# SortSelect 范式 AscendC 语言适配层

> **层级**：AscendC 语言适配层（adapters/ascendc/）。与 [通用层](../../generic/index.md) 组合消费，语言路由见 [范式入口](../../patterns.md)。

## 定位与能力映射

本层承载实际 API、模板参数、代码组织、编译器约束。排序能力及资源限制以 [capability-mapping.md](capability-mapping.md) 的实现和适用域为准。

## 阅读路径

| 任务阶段 | 入口 | 内容 |
|---|---|---|
| 所有路线的共同前置 | [公共布局包装模板](layout-wrapper.md) | 原始轴规范化、连续尾轴交接、输出还原及白盒输入契约 |
| 方案设计、实现替换与验收 | [design-spec.md](design-spec.md) | 输入输出、逐路线复用基线、替换证据及独立验收约束 |
| 整体模型与模板导航 | [overview.md](overview.md#按模板阅读) | 五类模板的 Tiling / Buffer / Kernel，以及公共符号 |
| 输出适配 | [公共映射](../../generic/output-mapping.md) → [AscendC 表达](output-mapping.md) | 一组输出参数生成 Sort、Argsort、TopK、KthValue，统一布局、批次复用与写回 |
| Host Tiling | [tiling-preprocess.md](tiling-preprocess.md) → [tiling.md](tiling.md) → 命中模板的 `tiling.md` | 全局选路后推导批次、块长、核数及尾块，含伪码和演算 |
| 参数对应 | [tiling-parameters.md](tiling-parameters.md) | 各 schedule 的字段含义、单位及 Host/Kernel 配对 |
| API 实现参考 | [example 包装参考及差异](kernel-entrance.md#example-包装参考及差异) | L2/L0 分工、Transpose 用法及示例适用边界 |
| Kernel | [kernel-entrance.md](kernel-entrance.md) → [模板计算索引](kernel-template.md#按模板读取) | 命中模板的阶段伪码、GM/UB 地址、索引含义与同步；特殊值另查 helpers |
| 节点实现核对 | [实现导航](../../source-map.md#ascendc) → 对应的 [Sort 示例计算体](example/sort/README.md) | 随当前模板读取调用壳层及必要 helper，将描述中的节点对应到实际执行方式和目标适配 |
| Buffer | [预算索引](dag-buffers.md#按模板读取) → 命中模板的 `dag-buffers.md` | 逐项物理字节、峰值、别名条件和跨批复用 |
| 陷阱 | [pitfalls.md](pitfalls.md) | 类型、索引、数据布局与同步约束 |

## 消费规则

只读命中的本层与通用层。编码前按目标规格核实参考实现的 dtype、轴长、稳定性和平台。

先实例化公共布局包装，明确调用层与主体的交接，再用全局选路确定候选，按需展开当前模板及其必要依赖。单条路线的设计在公共包装之内组合该模板的 Tiling、Buffer、Kernel 三份文档，再按问题读取输出适配或辅助约束；跨路线比较时按需展开相关模板。既有算法与同步协议的保留要求仍按 `design-spec.md` 执行。
