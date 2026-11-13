# 卷积反向范式入口（语言中立）

本文件是非量化浮点三维卷积反向算子生成的统一入口，负责场景判定、求导分支选择、语言路由、知识索引和消费指引。通用层建立数学、规格与设计模型，语言适配层落实 API、平台能力及调用接口。

## 1 场景判定

`op.paradigms` 包含 `ConvBackward` 时，经 [范式路由](../../routes.yaml) 进入本页。按语义检索或使用自定义算子名时，也按下面的求导对象判定场景。

本范式接受两种规格来源：给定正向卷积参数后生成对应梯度算子；直接给定反向算子参数后生成算子。两种来源统一为明确的输入、输出与卷积属性，再按求导对象选择分支。

| 场景 | 数值输入 → 输出 | 目标形状来源 | 分支 |
|---|---|---|---|
| Conv3DBackpropInput / Conv3DDX | weight、dedy → dedx | input_size 或正向 X 规格 | input |
| Conv3DTranspose | x、filter、可选 bias → y；按 x→dedy、filter→weight、y→dedx 绑定 | 属性推导或显式 input_size | input |
| Conv3DBackpropFilter / Conv3DDW | x、dedy → dedw | filter_size 或正向 weight 规格 | filter |

准入需要：张量角色、shape、dtype、format 与卷积属性完整；批次、通道、组和正向空间形状关系一致；目标平台与 SDK 满足命中分支的能力与最终参数约束。具体支持组合如下：

| 分支 | 类型与布局 | 场景契约 |
|---|---|---|
| input | weight/dedy/dedx 使用相同 FP16 或 BF16；dedy/dedx 同为 NCDHW 或 NDHWC，weight 独立选择 NCDHW 或 NDHWC；Transpose bias 为 FP32 | 从 [input 索引](generic/input/index.md) 读取场景与 Tiling 契约 |
| filter | x/dedy 使用相同 FP16 或 BF16、同为 NCDHW 或 NDHWC；dedw 为 FP32，独立选择 NCDHW、NDHWC 或 DHWCN | 从 [filter 索引](generic/filter/index.md) 读取场景；普通正向权重梯度，output_padding=0，MN_StreamK 最终运行时准入 |

逻辑轴、属性及低维升维从公共知识进入；两分支各自建立 M/N/K、数据流、切分和初始化协议。目录名 filter 表示权重梯度分支，张量/API 中的 weight、dedw、filter_size 保留各自含义。

## 2 语言路由

语言由调用方明确给出或读取 `op.language`，随后加载通用知识和相应适配层。

| 开发语言 | 加载内容 | 入口 |
|---|---|---|
| ascendc | 公共规格 + 命中分支知识 + Ascend C 适配 | [adapters/ascendc/index.md](adapters/ascendc/index.md) |

通用知识可用于独立的数学与方案讨论。实现时由适配层获取 platformInfo，核对资源、API 和执行能力，按实际 SDK 确定参数。

## 3 相关知识索引

面向人的目录说明、算子示例与支持范围见 [范式档案](readme.md)。设计知识按下表进入。

| 层次 | 公共入口 | input 私有入口 | filter 私有入口 |
|---|---|---|---|
| 通用知识 | [generic/index.md](generic/index.md)：统一术语、规格归一 | [generic/input/index.md](generic/input/index.md)：输入梯度与转置卷积 | [generic/filter/index.md](generic/filter/index.md)：权重梯度 |
| Ascend C | [adapters/ascendc/index.md](adapters/ascendc/index.md)：平台能力与分支选择 | [input/index.md](adapters/ascendc/input/index.md)：API、Kernel、同步与示例 | [filter/index.md](adapters/ascendc/filter/index.md)：API、Kernel、同步与示例 |

分支索引维护自己的场景、计算流、Tiling 模板、验证及实现文档；公共入口不逐项展开分支叶子知识。

## 4 设计与消费指引

按 D1→D7 逐步加载，每步使用所选分支索引中的对应文档。

| 步骤 | 形成的结论 | 知识入口 |
|---|---|---|
| D1 场景与规格 | 求导对象、来源绑定、统一形状、属性与派生量 | 公共 preprocessing + 分支 scene |
| D2 接口 | 输入/输出/属性、dtype×format 和调用要求 | 公共 language + 分支适配索引 |
| D3 推导 | 目标形状值依赖、InferShape / InferDataType | 公共 preprocessing 中的分支绑定 |
| D4 Tiling | 模板、基本块、父段、任务、资源预算与传参 | 分支 tiling 及其模板 |
| D5 Kernel | 地址、数据搬运、布局转换、乘加和写回 | 分支 compute-flow、instruction-mapping、kernel-template |
| D6 同步 | buffer 生命周期、初始化、流水与启动契约 | 分支 sync-implementation；Transpose 按需加载 bias 通路 |
| D7 验证 | 独立 Golden、任务/归约/分支覆盖、正确性与执行记录 | 分支 verification |

1. 先读公共索引，再进入一个求导分支和一个语言适配层；单个子问题最多展开 5 个叶子文档。
2. D4 与 D5 联动核算。基本块、buffer 份数或流水改变后，重新检查容量、实际地址和生命周期。
3. 编码前从分支适配索引选择对应调用形态的示例；示例解析器返回的是候选目录，按求导对象和调用方式选取。
4. input 沿用已验证模板的切分约定；filter 的 MN_StreamK 在 Host 按形状和平台资源求解 baseM/baseN/baseK，通过 TilingData 发布，Kernel 按容量处理有效尾块。
5. 缺少平台或 API 依据时记录受影响的具体结论。数学推导、Host 复算、编译和设备结果分别保存。
6. 示例审查或运行中的修正回写所属规则，再从独立目录重新生成与验证。
