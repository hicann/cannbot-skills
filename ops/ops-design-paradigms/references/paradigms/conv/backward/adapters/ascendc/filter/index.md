# filter 分支 AscendC 语言适配层

本目录将 [filter 计算模型](../../../generic/filter/scene.md) 映射到 Ascend C。入口与语言选择见 [卷积反向入口](../../../patterns.md)，资源能力复用上级共享文档，同步与初始化由本目录定义。

## 1 定位

通用层定义数学、布局、任务和容量；本目录给出 API 参数、实例绑定、Kernel 骨架和执行协议。A 路始终为 dedy，B 路始终为 x，输出为 FP32 dedw。

```text
        +-------------------------------------------------+
        | generic/filter: scene -> compute-flow -> tiling |
        +------------------------+------------------------+
                                 |
                                 |
                                 v
        +------------------------+------------------------+
        | shared capability baseline                      |
        +------------------------+------------------------+
                                 |
                                 |
                                 v
       +-------------------------+-------------------------+
       | filter: API + Tiling binding + synchronization    |
       +-------------------------+-------------------------+
                                 |
                                 |
                                 v
+--------------------------------+--------------------------------+
| filter: kernel-template -> generated direct / registry project  |
+--------------------------------+--------------------------------+
                                 |
                                 |
                                 v
         +-----------------------+-----------------------+
         | filter: verification -> inspect -> regenerate |
         +-----------------------------------------------+
```

## 2 阅读路径

| 阶段 | 文档 | 产出 |
|---|---|---|
| 资源与类型 | [buffer-and-types](../buffer-and-types.md) | platformInfo、容量与公共资源能力 |
| 场景和计算 | [scene](../../../generic/filter/scene.md)、[compute-flow](../../../generic/filter/compute-flow.md) | 权重梯度模型、两路布局与窗口、DAG |
| 切分 | [tiling](../../../generic/filter/tiling/tiling.md)、[mn-streamk](../../../generic/filter/tiling/templates/mn-streamk/tiling.md) | 计算块、父段、任务和最终字段 |
| 指令 | [instruction-mapping](instruction-mapping.md) | 参数表、单位、构造片段和数值示例 |
| 同步 | [sync-implementation](sync-implementation.md)、Kernel 第 7～9 节 | 初始化、队列与固定事件 |
| 字段与 key | [tilingkey-templates](tilingkey-templates.md) | 共享结构、编译实例、启动契约 |
| Kernel | [kernel-template](kernel-template.md) | 入口、任务循环、地址和各阶段实现 |
| 验证 | [verification](../../../generic/filter/verification.md) | 输入、Golden、比对、覆盖与回写 |

## 3 文档索引

| 文档 | 覆盖内容 |
|---|---|
| instruction-mapping.md | 两路 ND2NZ/DN2NZ、单 co 的计算 M 扩展、宽片与父行带裁剪、普通/拆核 Load3D、三格式 Fixpipe 与非整窗状态递推 |
| sync-implementation.md | UpdateMNIdx 首次与 M/N 切换条件、两路父段预取与归还、L0/CO1 交接、AIV 清零通知、片上零操作数与调用接入 |
| kernel-template.md | 数据流图、代码骨架、核间/核内偏移、Buffer 划分、MTE2/MTE1/Compute/CopyOut、初始化和流水同步、产出检查 |
| tilingkey-templates.md | MN_StreamK 编译参数、MIX 核类型、字段类型与单位、Host 发布和两种调用接入 |

## 4 公共知识依赖

共享文档提供资源与类型能力；filter 的同步协议、对象绑定、队列实例、单份 Bound、N 窗口与输出格式以本目录为准。比如共享层的输入梯度具体 buffer 尺寸不能直接用作权重梯度的父段尺寸。

平台能力通过 platformInfo 与当前 SDK 确认；核类型、AIC/AIV 配比、初始化事件和指令重载形成同一份能力记录。具体设备型号写入验证记录，不用名称列表代替能力检查。

## 5 示例与生成产物

| 交付形态 | 参考实现 |
|---|---|
| Kernel 直调 | [conv3ddw_direct](../example/conv3ddw_direct/README.md)：Host 求解器、共享字段、两种输入类型与 MIX 入口 |
| ACLNN 注册 | [conv3ddw_aclnn](../example/conv3ddw_aclnn/README.md)：同一求解器与 Kernel，增加 OpDef、值依赖推导、TilingFunc、L2/L0 和包构建 |

| 示例位置 | 用途 |
|---|---|
| compute-flow 第 3～5 节 | 排布编号、三维窗口、地址与格式转换 |
| mn-streamk 第 6 节 | 基本块、L1 预算、任务分发和归约分段复算 |
| instruction-mapping 各阶段 | API 参数构造及数值算例 |
| kernel-template 第 2、6～8 节 | 入口、任务、乘加、初始化和原子写回骨架 |
| verification 第 3～5 节 | 输入量化、独立权重梯度 Golden 与比对代码 |

消费这些知识生成直调或注册工程时，在工程内定义求解器和全部阶段函数，记录编译、运行、字段与正确性证据。工程构建、注册和部署复用调用方所选工作流的工程技能。

## 6 消费规则

1. 先确定 filter 分支，再按问题加载本目录与对应通用文档，单个子问题最多展开 5 份叶子文档。
2. 先写明数学轴和矩阵朝向，再设置 API 参数；元素、字节和分形分别换算。
3. singleCoreN、baseN 与 baseUseN_ 分别表示任务、计算块和当前有效范围；非整窗通路按实际计划启用。
4. 普通/拆核窗口、原始组、streamkType 和原子写回协议在 Host 与 Kernel 一致。
5. 示例复算、SDK 编译和设备验证分别记录；人工修复回写对应知识，再独立重新生成验证。
