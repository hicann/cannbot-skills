# SortSelect 范式 CANNBotDSL 语言适配层

> **层级**：CANNBotDSL 语言适配层（adapters/dsl/）。与 [通用层](../../generic/index.md) 组合消费，语言路由见 [范式入口](../../patterns.md)。

## 定位与能力映射

本层承载 DSL 实体、能力支持范围、代码组织与编译器约束。公共能力与资源核对项目在 [capability-mapping.md](capability-mapping.md) 逐项映射，未知项保持待验证。

## 阅读路径

| 任务阶段 | 入口 | 内容 |
|---|---|---|
| 方案设计 | [generic/index](../../generic/index.md) → [capability-mapping](capability-mapping.md) | 通用算法叠加当前 DSL 能力与限制 |
| Host 选路与资源 | [tiling.md](tiling.md) → 命中模板的 `tiling.md` | 全局分派、逐路线容量公式、候选搜索与尾块演算 |
| 计算与调试 | [implementation-guide.md](implementation-guide.md) → 命中模板的 `kernel-template.md` | 实际数据流、原位置/名次含义、地址与阶段同步 |
| 共用 Radix 原语 | [local-radix.md](local-radix.md) → [toolkit 阶段映射](toolkit-radix-port.md) | 保序键、8-bit/2-bit 粒度、局部 LSD、各模板复用层级及 64 位处理 |
| 非末轴物理转置 | [Transpose 映射](toolkit-transpose-port.md) | NDDMA/SIMT 选路、GM/UB 地址与尾块、紧凑写回、64 位 Native ABI 和验证范围 |
| 模板选择 | [路线与布局对照](implementation-guide.md#路线命名与非末轴布局) → 下表 | 四类已适配路线；非末轴复用排序主体，归并尚未适配 |
| 输出适配 | [公共映射](../../generic/output-mapping.md) → [DSL 表达](implementation-guide.md#统一输出适配) | 同一排序主体、名次区间写回与批次复用 |
| 公开入口与启动 | [entry.py](example/sort/entry.py) → [sort_host.py](example/sort/sort_host.py) | 输入契约、Tiling、编译导出与 Kernel 分派 |

## 按模板读取

| 模板 | Tiling | Buffer | Kernel |
|---|---|---|---|
| 插入 | [切分](templates/insertion/tiling.md) | [内存](templates/insertion/dag-buffers.md) | [计算](templates/insertion/kernel-template.md) |
| 两阶段 | [切分](templates/two-stage/tiling.md) | [内存](templates/two-stage/dag-buffers.md) | [计算](templates/two-stage/kernel-template.md) |
| 单核 Radix | [切分](templates/radix-one-core/tiling.md) | [内存](templates/radix-one-core/dag-buffers.md) | [计算](templates/radix-one-core/kernel-template.md) |
| 多核 Radix | [切分](templates/radix-more-core/tiling.md) | [内存](templates/radix-more-core/dag-buffers.md) | [计算](templates/radix-more-core/kernel-template.md) |

模板 Markdown 给出可推导的设计和当前实现边界；核实语言 API 签名及编译绑定时再定位 example。按命中模板及其必要依赖读取，共用 Radix 和输出说明按需补读；跨路线比较时按需展开相关模板。

## 补充输入与消费规则

仅加载 DSL 本层与通用层；能力不足时报告具体缺口。API、资源和同步约束按实际接口确认。

example 的公开入口演示 Sort 配置；生成其他接口只适配统一输出参数、shape、最终写回和分配/寻址，不增加算子专用排序模板。

指定归并路线时报告当前适配缺口，不静默改算法。未限定路线的任务默认落实 [选路与容量的路线覆盖规则](tiling.md#路线覆盖与验收)，不能将“可选择已适配模板”解释为任意删除具有适用输入域的路线。设计和验收均须读取该规则。
