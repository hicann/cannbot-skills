# 卷积反向 input 分支 AscendC 适配索引

> **层级**：input 分支适配层（adapters/ascendc/input/）。语言中立的算法与模型见 [通用知识索引](../../../generic/index.md)；范式入口与语言路由见 [范式入口](../../../patterns.md)。

## 1 定位

本目录承载卷积反向范式中**强耦合 AscendC 语言/工具链**的知识：API 实体与参数结构体、类型与 buffer 基线、同步写法、Tiling 数据与 key 编码。开发语言为 AscendC（由调用方明确提供）时，与通用知识层组合消费：

```
通用算法/模型（generic/）× AscendC 实现细节（本目录）→ AscendC 设计与代码产出
```

指令的**语义、单位与数据流**（搬运矩阵、窗口几何、K 配对、迭代与生命周期）属语言中立知识，在 [compute-flow.md](../../../generic/input/compute-flow.md) 与 [tiling/tiling.md](../../../generic/input/tiling/tiling.md)，本目录不重复。

## 2 阅读路径

```
+--------------------------------+--------------------------------+------------------------------+
| 任务流程                       | 入口文档                       | 辅助参考实现                 |
+--------------------------------+--------------------------------+------------------------------+
| 骨架搭建（类型/槽/能力）       | ../buffer-and-types.md         | example/conv3ddx/            |
|   |                            | （platformInfo/TQue 选型）     | example/conv3dtranspose/     |
|   v                            |                                | （tiling + kernel 参考实现） |
| 字段与 key 发布                | tilingkey-templates.md         |                              |
|   |                            | （模板/字段/编译实例绑定）     |                              |
|   v                            |                                |                              |
| 逐条指令编码                   | instruction-mapping.md         |                              |
|   |                            | （API 参数表/位域/SDK 检查）   |                              |
|   v                            |                                |                              |
| 同步插入                       | sync-implementation.md         |                              |
|   |                            | （L1 队列/L0 事件/初始化）     |                              |
|   v                            |                                |                              |
| bias 功能（仅 Transpose）      | bias-bt-path.md                |                              |
|   |                            | （BT 装载/Mmad 初值）          |                              |
|   v                            |                                |                              |
| Kernel 阶段组织                | kernel-template.md             |                              |
|   |                            | （入口/任务循环/写回）         |                              |
|   +-- VecTranspose/C04/enlarge | instruction-mapping §3.4–§3.8  |                              |
|   +-- kernel-split 模板        | kernel-split-implementation.md |                              |
|   `-- AIC/AIV 协同             | cv-collaboration.md            |                              |
+--------------------------------+--------------------------------+------------------------------+
```

通用模型与 AscendC 实现的职责分工（链接均从本目录解析，准入和实现细节只在所属文档维护）：

| 设计问题 | 通用模型 | Ascend C 实现 |
|---|---|---|
| 分组合并、四通道 K 排列、源布局变化 | [scene.md：数据准备](../../../generic/input/scene.md#3-数据准备与分组合并) | [instruction-mapping.md §3.4–§3.8](instruction-mapping.md) |
| 完整 N/K 驻留与 M 分核 | [l1-resident Tiling](../../../generic/input/tiling/templates/l1-resident/tiling.md#41-nk-全载与-m-分核) | [kernel-template.md：完整驻留](kernel-template.md#44-完整-nk-驻留分支) |
| 子核窗口与 M/N/K、资源和任务 | [kernel-split Tiling](../../../generic/input/tiling/templates/kernel-split/tiling.md) | [kernel-split-implementation.md](kernel-split-implementation.md) |
| key、字段和入口实例绑定 | [tiling/tiling.md](../../../generic/input/tiling/tiling.md) | [tilingkey-templates.md](tilingkey-templates.md) |
| 跨核参与者与通知生命周期 | [compute-flow.md](../../../generic/input/compute-flow.md) | [cv-collaboration.md](cv-collaboration.md) |

示例按输入梯度或转置卷积从下一节选择，直调与 ACLNN 工程复用同一份计算实现。

## 3 文档索引

| 文档 | 覆盖 | 典型消费时机 |
|------|------|-------------|
| [buffer-and-types.md](../buffer-and-types.md) | platformInfo 获取、类型与 TQue/TBuf 选型、L1/L0 物理槽划分、能力判定 | 骨架搭建 |
| [instruction-mapping.md](instruction-mapping.md) | API 实体与参数结构体：Nd2Nz/Dn2Nz 参数表、Load3D/FMatrix 位域、三种 weight 变换的 Load2D 参数、MmadParams/Fixpipe 构造、SDK 适配检查（语义与地址公式在 generic/input/compute-flow 第 5 节）；三种数据准备分支 VecTranspose/C04/group enlarge 的准入、字段身份、排布线稿与 UB/L1/workspace 搬运（§3.4–§3.8） | 逐条指令编码；数据准备分支的实现设计 |
| [sync-implementation.md](sync-implementation.md) | L1 队列、L0 事件、L0C 交接、L1B 驻留和输出初始化 | 同步插入 |
| [bias-bt-path.md](bias-bt-path.md) | BT 装载、Mmad 初值参数、ComputeForBias 与生命周期（仅转置卷积） | bias 功能 |
| [tilingkey-templates.md](tilingkey-templates.md) | 当前执行配置、模板与求解器对应、loadB2Condition 判定、数据准备分支的最终编译绑定与发布、共享字段、Host 发布 | 字段与 key |
| [kernel-split-implementation.md](kernel-split-implementation.md) | kernel-split 的入口分发、任务解码、子核状态机（四相位轮转/单子核刷新）、子核 pad 映射、输出重排（UB/workspace 中转 + AIV 交织） | kernel-split 模板落地 |
| [cv-collaboration.md](cv-collaboration.md) | AIC/AIV 参与角色与核型（与目标执行模式配对）、CvCrossCoreSet/Wait 旗语协议、kernel-split 重排与 B 路权重整理（前置转置两级通知、C04/group enlarge 段级握手）的协同时序、workspace 布局与入口前置检查 | 跨核协同 |
| [kernel-template.md](kernel-template.md) | 设备入口、任务循环骨架、地址、各阶段代码组织与转置绑定 | Kernel 阶段组织 |
| [example/conv3ddx/](../example/conv3ddx/README.md) | 输入梯度 Tiling、Kernel 和 TilingData：三种 Tiling 策略、三种 loadB2 变换与输出写回 | 编码前检视；直调与 ACLNN 共用 |
| [example/conv3dtranspose/](../example/conv3dtranspose/README.md) | 转置卷积 Tiling、Kernel 和 TilingData：q/rho、目标形状、可选 bias BT 通路与输出写回 | 编码前检视；直调与 ACLNN 共用 |

## 4 公共知识依赖

以下跨范式 AscendC 公共知识位于范式外，按需展开：

| 主题 | 位置 |
|------|------|
| 同步与一致性 | [../../../../common/sync-and-consistency.md](../../../../../common/sync-and-consistency.md) |
| DataCopyPad 规则 | [../../../../common/datacopypad-rules.md](../../../../../common/datacopypad-rules.md) |
| API 白名单（存在性唯一判据） | [../../../../../knowledge/api/regbase_api_whitelist.md](../../../../../../knowledge/api/regbase_api_whitelist.md) |

## 5 消费规则

| # | 规则 |
|---|------|
| 1 | 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载其他语言适配层 |
| 2 | 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题 |
| 3 | 编码前按算子场景检视 [conv3ddx](../example/conv3ddx/README.md) 或 [conv3dtranspose](../example/conv3dtranspose/README.md)；直调与 ACLNN 工程使用同一份计算实现，并按当前规格重新核定算法与参数 |
| 4 | **Cube kernel 走 aclnn/registry 编译时必须声明核类型**，取值与初始化协议配对（独立初始化=AIC-only，融合 AIV 清零=MIX 混合类型；配对错误会挂起或空跑 AIV 份，见 [tilingkey-templates.md](tilingkey-templates.md)）|
| 5 | 通用层引用的能力真值（容量、指令限制）以所用头文件树实际定义为准；出现版本差异按 [buffer-and-types.md](../buffer-and-types.md) 的能力判定处理并声明假设 |

## 6 正式算子工程封装

本范式覆盖卷积反向算法与验证知识，不承载工程能力（编译构建、算子包注册、性能采集、目录规范、构建脚本、安装部署）。工程骨架见仓内技能 `ops/ascendc-registry-invoke-template`（注册形态）与 `ops/ascendc-direct-invoke-template`（直调形态）；性能采集见仓内技能 `ops/ops-profiling`。

各示例 README 说明源码的具体范围。生成时按算子场景选择底本，新增分支按上述专用文档实现并验证。
