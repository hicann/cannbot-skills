# 输入梯度与转置卷积：DSL 适配

本分支把 [input 通用模型](../../../generic/input/index.md) 落到 CANNBotDSL 的视图、搬运和 Cube 指令，适配 standard、small-shape、l1-resident 三种 Tiling 模板。类型、格式与场景准入以 [scene](../../../generic/input/scene.md) 为准。

Conv3DTranspose 先完成自己的目标形状、output_padding 和 bias 绑定，再复用本分支。普通 dX 不创建 bias 资源。

| 要解决的问题 | 阅读位置 |
|---|---|
| DSL 版本、公共接口、资源和视图单位 | [能力映射](../capability-mapping.md)、[Buffer 与类型](../buffer-and-types.md) |
| 三模板如何绑定到编译实例和运行时参数 | [Tiling 绑定](tiling.md) |
| dY 插零搬入、weight 三种变换、Load3D 与输出格式 | [指令映射](instruction-mapping.md) |
| 如何组织任务、K 循环及首次累加 | [Kernel 组织](kernel-template.md) |
| ping-pong、权重驻留、无贡献输出及初始化 | [同步与初始化](sync-implementation.md) |
| Transpose 的 FP32 bias 如何进入 BT | [bias 通路](bias-bt-path.md)，仅带 bias 时读取 |
| Host、编译与调用 | [公共入口](../kernel-entrance.md) |
| 可运行 dX 与 Transpose 共用样例 | [dx.py](../example/dx.py)、[运行说明](../example/README.md) |
| 精度与边界覆盖 | [通用验证](../../../generic/input/verification.md) |

按问题读取通用层和本语言适配层，单个子问题最多展开 5 份叶子文档。下文代码是明确前提下的接口片段；完整算子仍需实现通用层的地址、任务和边界公式。接口存在、指令生成正确和完整算子通过设备验证是三个不同结论，验收时分别记录。
