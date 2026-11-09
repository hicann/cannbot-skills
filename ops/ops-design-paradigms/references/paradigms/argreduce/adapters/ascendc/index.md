# ArgReduce 范式 AscendC 语言适配层

> **层级**：AscendC 语言适配层（adapters/ascendc/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。

## 1 定位

本目录承载 ArgReduce 范式中**强耦合 AscendC 语言**的知识：API 实体、模式映射、代码模板、参考实现、编译器约束、设计约束。开发语言为 AscendC（由调用方明确提供）时，与通用知识层组合消费：

```
通用算法/模型（generic/）× AscendC 实现细节（本目录）→ AscendC 设计与代码产出
```

参考基线：ops-math argmax 家族工业实现（`arg_max_with_value` / `arg_max_v2` / `arg_min` / `arg_min_with_value`，arch35），本适配层知识与其 tiling 模式体系对齐（映射表见 [mode-mapping.md](mode-mapping.md)）。

## 2 能力映射

本语言对通用层能力谓词的真值与 API 实体见 [capability-mapping.md](capability-mapping.md)。通用层算法中引用的谓词（索引跟踪指令、向量比较、位置存储类型、通道宽度等）一律以该文件填值为准。

## 3 阅读路径

| 任务阶段 | 入口 | 内容 |
|------|------|------|
| 方案设计 | [design-spec.md](design-spec.md) | 输入输出、领域约束与一致性检查 |
| 全局心智模型 | [overview.md](overview.md) | 场景/模式划分、TilingKey 编码、TilingData struct、三者对应关系 |
| 模式选择 | [mode-mapping.md](mode-mapping.md) | 场景 × shape 比例 → 具体模式的完整映射（含工业实现 8 模式决策树与 UT 实证） |
| 归约 API | [highlevel-api.md](highlevel-api.md) | AR/ARA 场景的两条实现路径（Level-2 API / 寄存器级 Compare+Select）、NaN/平局落地、索引精度 |
| Host 侧 Tiling | [tiling-preprocess.md](tiling-preprocess.md) | 平台信息获取、轴折叠 C++ 实现、切分计算、TilingKey/TilingData 填充、workspace |
| Kernel 侧计算 | [kernel-entrance.md](kernel-entrance.md) | 入口注册与模式分发、各模式骨架 |
| 变体场景 | [../../generic/variants.md](../../generic/variants.md)、templates 各场景子目录 | 谓词首达 / top-k / 多输出挂载 |
| 参考实现 | [example/](example/) | arg_max_with_value 参考包（Kernel/Tiling/数据结构/测试） |

## 4 公共知识依赖

以下跨范式 AscendC 公共知识位于范式外，按需展开：

| 主题 | 位置 |
|------|------|
| Cast 规则 | [../../../common/cast-rules.md](../../../common/cast-rules.md) |
| VF 编程规则 | [../../../common/vf-programming-rules.md](../../../common/vf-programming-rules.md) |
| DataCopyPad 规则 | [../../../common/datacopypad-rules.md](../../../common/datacopypad-rules.md) |
| 同步与一致性 | [../../../common/sync-and-consistency.md](../../../common/sync-and-consistency.md) |
| 值依赖处理 | [../../../common/value-depend-process.md](../../../common/value-depend-process.md) |
| NDDMA 规则 | [../../../common/nddma-rules.md](../../../common/nddma-rules.md) |
| API 白名单（存在性唯一判据） | [../../../../knowledge/api/regbase_api_whitelist.md](../../../../knowledge/api/regbase_api_whitelist.md) |

## 5 消费规则

1. 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载其他语言适配层
2. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题
3. 按 [设计约束](design-spec.md) 确认当前问题的输入与输出，再读取相关主题资料核实 API 和资源事实
4. 编码前必须检视至少一个现有参考实现（见 [example/](example/)，选型参考开源算子表 `references/reference-ops/open_source_operator_table.md`）
5. 模式选择必须经 [mode-mapping.md](mode-mapping.md) 决策树推导并留 trace；禁止凭模式名直觉选择
