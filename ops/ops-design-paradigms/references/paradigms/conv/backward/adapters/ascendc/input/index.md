# 卷积反向 input 分支 AscendC 适配索引

> **层级**：input 分支适配层（adapters/ascendc/input/）。语言中立的算法与模型见 [通用知识索引](../../../generic/index.md)；范式入口与语言路由见 [范式入口](../../../patterns.md)。

## 1 定位

本目录承载卷积反向范式中**强耦合 AscendC 语言/工具链**的知识：API 实体与参数结构体、类型与 buffer 基线、同步写法、Tiling 数据与 key 编码。开发语言为 AscendC（由调用方明确提供）时，与通用知识层组合消费：

```
通用算法/模型（generic/）× AscendC 实现细节（本目录）→ AscendC 设计与代码产出
```

指令的**语义、单位与数据流**（搬运矩阵、窗口几何、K 配对、迭代与生命周期）属语言中立知识，在 [../../generic/input/compute-flow.md](../../../generic/input/compute-flow.md) 与 [../../generic/input/tiling/tiling.md](../../../generic/input/tiling/tiling.md)，本目录不重复。

## 2 阅读路径

| 阶段 | 文档 | 重点 |
|---|---|---|
| 资源 | [buffer-and-types](../buffer-and-types.md) | platformInfo、类型与物理槽 |
| 指令 | [instruction-mapping](instruction-mapping.md) | API 参数与 SDK 检查 |
| 同步 | [sync-implementation](sync-implementation.md) | L1 队列、L0 事件与初始化 |
| bias | [bias-bt-path](bias-bt-path.md) | Transpose 的 BT 通路 |
| 字段与 key | [tilingkey-templates](tilingkey-templates.md) | 模板参数、字段与 Host 发布 |
| Kernel | [kernel-template](kernel-template.md) | 入口、阶段与写回 |

示例按输入梯度或转置卷积、直调或 ACLNN，从下一节索引选择。

## 3 文档索引

| 文档 | 覆盖 | 典型消费时机 |
|------|------|-------------|
| [buffer-and-types.md](../buffer-and-types.md) | platformInfo 获取、类型与 TQue/TBuf 选型、L1/L0 物理槽划分、能力判定 | 骨架搭建 |
| [instruction-mapping.md](instruction-mapping.md) | API 实体与参数结构体：Nd2Nz/Dn2Nz 参数表、Load3D/FMatrix 位域、三种 weight 变换的 Load2D 参数、MmadParams/Fixpipe 构造、SDK 适配检查（语义与地址公式在 generic/input/compute-flow 第 5 节） | 逐条指令编码 |
| [sync-implementation.md](sync-implementation.md) | L1 队列、L0 事件、L0C 交接、L1B 驻留和输出初始化 | 同步插入 |
| [bias-bt-path.md](bias-bt-path.md) | BT 装载、Mmad 初值参数、ComputeForBias 与生命周期（仅转置卷积） | bias 功能 |
| [tilingkey-templates.md](tilingkey-templates.md) | 当前执行配置、三模板与求解器对应、loadB2Condition 判定、共享字段、Host 发布 | 字段与 key |
| [kernel-template.md](kernel-template.md) | 设备入口、任务循环骨架、地址、各阶段代码组织与转置绑定 | Kernel 阶段组织 |
| [example/conv3ddx_direct/](../example/conv3ddx_direct/README.md) | conv3ddx 直调算子参考实现（op_kernel 四件：Host 求解器三模板 + 设备 kernel 三 loadB2 变换 + 直调 ABI）；README 含编译配方、测试口径、代码段↔范式章节映射与 20 例实测记录 | 编码前检视；实现与接入参考 |
| [example/conv3ddx_aclnn/](../example/conv3ddx_aclnn/README.md) | conv3ddx aclnn 注册算子参考实现（GEIR 原型 + 图推导 + OpDef 24 组合 + 值依赖推导 + arch35 TilingFunc + registry 入口 + 手写 ACLNN L2 契约，与 direct 共享求解器与 kernel 主体）；README 含与 direct 的迁移要点、构建/测试/实测说明与已知缺陷登记 | 编码前检视；注册交付参考 |
| [example/conv3dtranspose_direct/](../example/conv3dtranspose_direct/README.md) | conv3dtranspose 直调算子参考实现（转置卷积变体：与 conv3ddx_direct 同核主体 + 转置几何 q/rho/显式 input_size + bias BT 通路 + ComputeZero；op_kernel 四件）；README 含与 conv3ddx_direct 的迁移要点、功能验证与已知边界 | 编码前检视；转置卷积交付参考 |
| [example/conv3dtranspose_aclnn/](../example/conv3dtranspose_aclnn/README.md) | conv3dtranspose aclnn 注册算子参考实现（转置卷积变体完整注册包：GEIR 原型 + 图推导 + OpDef 72 组合 + bias/offset_w 五槽 + 值依赖双模式 + arch35 TilingFunc + 手写 ACLNN L2 + fix-3 72→74 布线 + R4 保留槽安装树脚本，与 conv3dtranspose_direct 共享求解器与 kernel 主体）；README 含与 conv3ddx_aclnn 的迁移要点、功能验证与已知边界 | 编码前检视；转置卷积注册交付参考 |

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
| 3 | 编码前必须检视参考实现——按交付形态与算子场景选 [example/conv3ddx_direct/](../example/conv3ddx_direct/README.md)（kernel 直调算子）、[example/conv3ddx_aclnn/](../example/conv3ddx_aclnn/README.md)（aclnn 注册算子）、[example/conv3dtranspose_direct/](../example/conv3dtranspose_direct/README.md)（转置卷积变体直调算子）或 [example/conv3dtranspose_aclnn/](../example/conv3dtranspose_aclnn/README.md)（转置卷积变体 aclnn 注册算子），并按当前算子规格重新核定算法与参数 |
| 4 | **Cube kernel 走 aclnn/registry 编译时必须声明核类型**，取值与初始化协议配对（独立初始化=AIC-only，融合 AIV 清零=MIX 混合类型；配对错误会挂起或空跑 AIV 份，见 [tilingkey-templates.md](tilingkey-templates.md)）|
| 5 | 通用层引用的能力真值（容量、指令限制）以所用头文件树实际定义为准；出现版本差异按 [buffer-and-types.md](../buffer-and-types.md) 的能力判定处理并声明假设 |

## 6 正式算子工程封装

本范式覆盖卷积反向算法与验证知识，不承载工程能力（编译构建、算子包注册、性能采集、目录规范、构建脚本、安装部署）。工程骨架见仓内技能 `ops/ascendc-registry-invoke-template`（注册形态）与 `ops/ascendc-direct-invoke-template`（直调形态）；性能采集见仓内技能 `ops/ops-profiling`。
