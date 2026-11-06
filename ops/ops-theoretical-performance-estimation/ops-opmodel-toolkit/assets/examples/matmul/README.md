# 可编辑矩阵乘模型

本例定义 `C = A @ B`：A、B 为 bf16，累加为 fp32，输出 C 为 bf16。
`definition/` 提供完整的语义、任务、资源、执行和成本定义，`case.yaml` 提供一次输入，
`optimize/` 提供小规模分核和 DAG 搜索实验。

从用户项目目录按[快速入门](../../../references/quickstart.md)复制为 `work/my_matmul` 后运行。
模型默认使用 wheel 内的硬件 profile，附带的 `hardware.yaml` 可用于支持覆盖的入口。
这些配置是建模示例，不能当作设备性能保证。

阅读顺序：[编写模型](../../../references/modeling.md) → [分析结果](../../../references/analysis.md)
→ [参数寻优](../../../references/optimization.md)。

本页链接相对于 `ops-opmodel-toolkit` 目录。将模型复制到项目后，可从
`$OPMODEL_SKILL_DIR/references/index.md` 继续查阅使用指南。
