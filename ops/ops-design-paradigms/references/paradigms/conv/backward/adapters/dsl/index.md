# 卷积反向 CANNBotDSL 适配

本目录把 [generic](../../generic/index.md) 已有的 dX、dW 设计落实为 CANNBotDSL 接口。先确定求导对象和模板，再读对应分支；公式、选块算法和准入条件仍以 generic 为准；example 中明确标注的跨核 split-K 扩展单独说明适用范围，不改变 generic 模板。

## 阅读顺序

| 要实现的算子 | 现有模板 | 阅读入口 |
|---|---|---|
| dX / Conv3DBackpropInput | standard、small-shape、l1-resident | [input/index.md](input/index.md) |
| Conv3DTranspose | 沿用 input 模板，增加形状绑定和可选 bias | [input/index.md](input/index.md) |
| dW / Conv3DBackpropFilter | MN_StreamK，含已有窗口、布局和写回分支 | [filter/index.md](filter/index.md) |

类型、格式组合和卷积属性的准入条件分别由 [input scene](../../generic/input/scene.md) 与 [filter scene](../../generic/filter/scene.md) 定义；1D/2D 归一化见 [preprocessing](../../generic/preprocessing.md)。本 adapter 说明所选场景和模板如何用 DSL 实现，指令参数、存储类型及转换方式随该场景绑定。

| 公共文档 | 解决的问题 | 何时读 |
|---|---|---|
| [capability-mapping.md](capability-mapping.md) | 哪个 DSL 接口承接搬运、计算、写回和同步；如何核对生成指令 | 开始适配或检查能力时 |
| [buffer-and-types.md](buffer-and-types.md) | 类型、逻辑/物理布局、字节偏移、buffer 槽和容量 | 分配片上存储、构造视图时 |
| [load3d.md](load3d.md) | NC1HWC0 打包、多 C1 窗口、L0B 落点与独立验证 | 使用 Load3D 或定位多通道错误时 |
| [kernel-entrance.md](kernel-entrance.md) | Python 入口、运行时 tiling、编译配置、核数、launch 和验证 | 搭入口及交付时 |
| [DSL 性能基线](performance-baseline.md) | 设备计时、完整计算路径与 Host 开销的区分 | 设置性能目标或比较 ASC 时 |
| [example/README.md](example/README.md) | 可运行的 dW/dX、Transpose 复用入口、布局与性能实验 | 需要把接口片段串成完整计算链时 |
| [initialization.md](initialization.md) | 融合 AIV 清零与 MTE3→FIXPIPE 跨核同步 | 需要输出预清零时 |

一次只展开当前分支和当前问题需要的文档；一个子问题最多读 5 份叶子文档。使用 DSL 时读 generic 和本 adapter 即可，不必再把 Ascend C 的队列、结构体和注册代码翻译一遍。

## 接口依据

本适配的 DSL 版本为 `cannbot-dsl 0.8.2`，编译目标为 `dav-3510`（950），开发环境为 CANN 9.3.0。其他版本先检查同名接口的签名和生成结果。

其他范式的历史 DSL 文档可能使用 `Channel/make_channel`、`produce/consume`。本文使用已核对版本的 `RingBuffer/make_ring_buffer`、`next_write/next_read`，以及 `mem_copy(dst, src, options)`；不混用两套调用形式。

这里提供接口片段和实现步骤，不提供完整可直接发布的卷积工程。片段前会说明所需变量及布局。API 源码核对、编译检查和完整算子设备验证是三种不同结论；交付时仍须按各分支的 generic verification 验证实际生成的算子。
