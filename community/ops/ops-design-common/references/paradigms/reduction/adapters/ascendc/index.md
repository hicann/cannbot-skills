# Reduction 范式 AscendC 语言适配层

> **层级**：AscendC 语言适配层（adapters/ascendc/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。

## 1 定位

本目录承载 Reduction 范式中**强耦合 AscendC 语言**的知识：API 实体、代码模板、参考实现、编译器约束、设计约束。开发语言为 AscendC（由调用方明确提供）时，与通用知识层组合消费：

```
通用算法/模型（generic/）× AscendC 实现细节（本目录）→ AscendC 设计与代码产出
```

## 2 能力映射

本语言对通用层能力谓词的真值与 API 实体见 [capability-mapping.md](capability-mapping.md)。通用层算法中引用的谓词（对齐粒度、树缓存容量、归约指令、融合链长等）一律以该文件填值为准。

## 3 阅读路径

| 任务阶段 | 入口 | 内容 |
|------|------|------|
| 方案设计 | [design-spec.md](design-spec.md) | 输入输出、领域约束与一致性检查 |
| 全局心智模型 | [overview.md](overview.md) | 模板划分 / TilingKey（ASCENDC_TPL 宏）/ TilingData struct / 三者对应关系 |
| Host 侧 Tiling | [tiling-preprocess.md](tiling-preprocess.md) → [templates/binary-base/tiling.md](templates/binary-base/tiling.md) | 平台信息获取、值依赖三处契约、合轴 C++ 实现、UB 双切分三步 / 多核切分 / workspace 公式 |
| Kernel 侧计算 | [kernel-entrance.md](kernel-entrance.md) → [templates/binary-base/kernel-template.md](templates/binary-base/kernel-template.md) | 入口注册与模板分发、骨架/偏移/CopyIn/Compute/CopyOut/同步 |
| 归约核心算法 | [binary-tree.md](binary-tree.md) → [highlevel-api.md](highlevel-api.md) | 二分缓存树 kernel 侧落地、ReduceXxx 高阶 API 约束 |
| Buffer / 存活节点 / 融合落地 | [survival-node.md](survival-node.md) + [vf-fusion-rules.md](vf-fusion-rules.md) + [templates/*/dag-buffers.md](templates/binary-base/dag-buffers.md) | L/P 分析、VF 融合判据、各模板 DAG 分析与 Buffer 分类 |
| 变体场景 | [tuple-reduce.md](tuple-reduce.md)、[templates/binary-group/](templates/binary-group/)、[templates/empty/](templates/empty/) | Tuple Reduce 循环调用、Group 两阶段、空 tensor 模板 |
| 参考实现 | [example/](example/) | euclidean_norm 完整包（Kernel/Tiling/数据结构/测试） |

## 4 公共知识依赖

以下跨范式 AscendC 公共知识位于范式外，按需展开：

| 主题 | 位置 |
|------|------|
| Cast 规则 | [../../../common/cast-rules.md](../../../common/cast-rules.md) |
| VF 编程规则 | [../../../common/vf-programming-rules.md](../../../common/vf-programming-rules.md) |
| DataCopyPad 规则 | [../../../common/datacopypad-rules.md](../../../common/datacopypad-rules.md) |
| 同步与一致性 | [../../../common/sync-and-consistency.md](../../../common/sync-and-consistency.md) |
| 值依赖处理 | [../../../common/value-depend-process.md](../../../common/value-depend-process.md) |
| API 白名单（存在性唯一判据） | [../../../../knowledge/api/regbase_api_whitelist.md](../../../../knowledge/api/regbase_api_whitelist.md) |

## 5 消费规则

1. 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载其他语言适配层
2. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题
3. 按 [设计约束](design-spec.md) 确认当前问题的输入与输出，再读取相关主题资料核实 API 和资源事实。
4. 编码前必须检视至少一个现有参考实现（见 [example/](example/)，选型参考开源算子表 `references/reference-ops/open_source_operator_table.md`）
