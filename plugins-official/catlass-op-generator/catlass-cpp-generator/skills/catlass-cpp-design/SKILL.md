---
name: catlass-cpp-design
description: CATLASS C++ 已注册算法族五阶段工作流的 03 方案设计。用于按完整参考完成设计推演，再交付两章 docs/design.md，并应用 R01-R21 与 Stage/同步规则。触发条件：算子工作流处于 design 阶段，operator contract 与 golden contract 均已冻结。
---

# 03 方案设计

进入条件：两个 contract 均为 `frozen` 且 workflow 状态为 `design`。

必须通过 `catlass-cpp-knowledge` 完整读取：

- `workflow/solution-design-reference.md`
- `workflow/stage-design-rules.md`
- `workflow/kernel-stage-sync-patterns.md`

仅当当前公式包含单位下三角矩阵求逆时，读取
`operator/linear-attention/matrix-inverse-patterns.md`；不涉及该算法时不得把其 Stage 或布局带入设计。

`sparse_flash_mla` 查询并读取家族索引中的 `computation.md`、`interface.md`、
`metadata.md`、`development.md` 和 `validation.md`；普通 SWA 还完整读取
`pipeline.md`、`performance.md`，并核对接口中的一般合法域、开发中的工程与实现门槛。
长 prefill 同时应用 `performance.md` 的连续窗口推导及 case2 机制。设计主核和 metadata 的生产/消费、独立入口、动态调度与资源同步，
不能把 case2 常量当一般 SWA 合法域。

先读取 `docs/workflow.json` 的 `target_architecture`，并在 `docs/design.md` 中写入相同值、
CATLASS 编译标识、ArchTag、Vector 执行模型以及 Cube→Vector、Vector→Cube 的完整数据路径。
按所需组件查询 `catlass/`，再回到当前工作区固定 CATLASS 源码、文档和 example 核对真实签名。
先严格按照 `solution-design-reference.md` 的完整内容完成 Stage 划分和具体详设，不得因最终交付件
只有两章而跳过资源、精度、性能、测试、风险或 R01-R21 检查。完整设计推演可以保存为
`docs/design.full.md` 中间产物；该文件用于保留推导和评审过程，不替代最终交付件。

再将已完成的设计结论投影到 `templates/design.md.template`，生成仅含两章的 `docs/design.md`；
将模板 `workflow_id` 替换为当前 `docs/workflow.json` 中的值。
写清公式、shape、地址和生命周期、AIC/AIV owner、CrossCore/HardEvent 配对、tiling、
workspace 总量、精度观察点和性能目标；每个 L1、UB、L0、GM workspace 资源条目必须用 `tensor[N]` 显式写出物理预留份数，单份对象也写 `[1]`；同一条目分别写 dtype、shape/layout、实际激活份数或条件、每份大小、总大小、首次写入或搬入、最后消费者以及释放/复用条件。多份预留但按 leader、`valid_r` 或 ping/pong 选择性使用时，必须区分物理预留份数与当前 work 实际激活份数，不得只在“份数依据”中笼统描述。

R01-R21 作为完整推演与评审规则使用，不在最终文档中
生成独立检查表章节。测试、profiling、sanitizer 和实测结果进入 `docs/validation.md`。

运行 `scripts/validate_design.py --design <design> --workflow <workflow>`；有 kernel 时同时传
`--kernel`。校验器必须确认设计与 workflow 的架构一致：A2/A3 使用 `2201/AtlasA2/MemBase`
和 GM 中转路径；A5 使用 `3510/Ascend950`，并显式选择 MemBase 或 RegBase/VF 及数据路径。
结构校验和整体设计评审都通过后才设置 `stage=implementation`。
