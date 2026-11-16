---
name: ge-custom-op-development
description: >-
  使用 GE BaseCustomOp 或 Python ge.custom_op，将已有或待开发的 kernel 接入图编译与执行。
  适用于能力接口选型、C++/Python 交付件生成、OM 部署和加载/回调排障；
  当用户需要把自定义 kernel 接入 GE 图编译、或定位算子回调不生效问题时使用。
  不负责 kernel 本体实现、IMPL_OP + .Tiling 的托管算子工程、单纯 ONNX 节点映射，
  或仅通过 torch.library + npugraph_ex 注册的 PyTorch 算子。
---

# GE 自定义算子接入

kernel 语言、交付件语言、构图入口分别选择；先按执行需求选能力，再适配产物加载。
本 Skill 面向 `BaseCustomOp` / `ge.custom_op` 的自组织交付件，通常走 `custom_engine`。

## 1. 确认任务边界

用户已明确使用本接口或已有对应工程时，直接进入工作流程，不重复分诊。
信息不足时先检查已有源码、构建脚本和产物，只询问仍影响方案的缺口。

| 已有材料或需求 | 处理方式 |
|---|---|
| BaseCustomOp 类、ge.custom_op 回调、自备 kernel 产物或 Host 计算 | 在本 Skill 内选型、生成或排障 |
| Ascend C 算子工程，期望框架执行 `.Tiling()` | 转 `plugins-official/ops-registry-invoke` 或 `ops/ascendc-registry-invoke-template` |
| ONNX 节点映射到已有算子 | 转 `graph/ge-onnx-plugin-development` |
| 只做 torch.library + npugraph_ex 注册，不交付 GE 实现 | 转 `graph/torch-custom-ops-guide` |
| 只写 kernel 本体或只做 kernel 直调 | 使用对应 `ops/` 域 Skill；本 Skill 可先输出入图契约 |

`IMPL_OP` 本身不是转出依据：它也可为 BaseCustomOp 补充推导或输入值依赖声明。
`REG_OP` 注册的是托管与 BaseCustomOp **两条路径共用**的算子原型，不是实现注册；分诊判据是有无 `.Tiling()`/`.TilingParse()` 与是否按算子工程组织，不是要不要注册。
混合注册或机制不明确时读取 [机制分诊](references/mechanism-triage.md)。

## 2. 按任务推进

### 接口咨询或 kernel 尚未完成

1. 从已有信息确定后端、编译时机、静态/动态图、输出 shape、部署目标和产物形态。
2. 用下表选最小能力组合；复杂情况读取 [场景选型](references/scenarios.md)。
3. 按 [kernel 契约](references/kernel-contract.md) 记录参数布局、产物、依赖和待验证项。kernel 未就绪时可先给暂定方案；产物格式与入口符号在最终接入前核实。

### 新建或改造工程

1. 确认 CANN/GE 安装版本、目标 SoC、主机架构及所需 Python bridge；能力以目标环境为准。来源及核对方法见 [版本与来源](references/sources.md)。
2. 选接口和交付语言，按 [模板选择](references/build-verify.md#模板选择) 只复制需要的文件。模板是待适配骨架，核对目标头文件签名后补齐加载、计算、推导和错误处理。
3. 读取对应 [C++ 接口](references/interfaces.md) 或 [Python 接口](references/python-track.md)；需要加载产物时读取 [产物契约](references/language-launch.md)。
4. 按 [构建与验证](references/build-verify.md) 完成构建、加载、数值及部署验证。缺少设备或依赖时完成可做的检查，明确列出未执行项，不将语法通过报告为入图成功。

### 已有工程排障

先读取 [故障诊断](references/troubleshooting.md)，从症状收集证据；不重建整个工程，也不因单条正常告警修改已正确的机制。

## 3. 能力选择速查

| 需求 | 选择 |
|---|---|
| 自行加载并 launch、unknown graph 或运行期决策 | `EagerExecuteOp` / Python `execute` |
| 静态图由 GE 生成 launch 和地址刷新 | `AnnotatedArgsOp` / `declare_launch_args` |
| 同一算子兼容两种图形态 | 同时提供声明式与命令式实现，分别验证两条路径 |
| 图编译期需要编译或准备特化产物 | `CompilableOp` / `compile`；预编译产物不因出 OM 而自动需要它 |
| 需要随 OM 恢复自定义状态 | 评估 `PortableOp`；声明式 bin、同库产物与独立状态分别确认携带通道 |
| 推导输出 shape/dtype | 默认用 `ShapeInferOp`；Python 用 `register_op` 的 infer-meta |
| Host CPU 计算 | `HostCpuExecuteOp` / `register_kernel(backend=OpBackend.HOST)` |
| C++ 回调式地址刷新 | `ArgsUpdater`；与声明式刷新路线择一 |

对加载属性、MIX/FFTS 或特殊参数布局不作仅凭语言、文件后缀或**存在节点属性**的可行性判断；入口、参数 ABI、FFTS 与目标运行时加载仍需逐项验证。
当前模板未验证的产物，先核对实际 metadata 与加载链路，再给推荐方案。

在线命令式由 `EagerExecuteOp::Execute` 加载预编译产物，并用 GE 提供的 stream 执行。
接入已有或构建期预编译的 kernel 时，默认将 binary 作为独立文件随交付件分发，运行时从明确路径读取。
将 binary **字节内嵌** `.so` 是备选方案，适用于有单文件交付等需求时，代价是更换 kernel 需重新链接；这与 Ascend C **源码同库编译**是两种不同机制，不要合并为一个选项，也不要把「预编译」默认解释为字节内嵌。
其他部署需求及选择依据见 [产物分发](references/language-launch.md#缓存和分发)。

## 4. 关键约束

- 原型、实现注册与构图侧的 **op type 一致**。`REG_AUTO_MAPPING_OP(cls)` 使用类名；保留独立类名时用 `REG_OP_BACKEND` 显式指定 op type。
- `ArgsUpdater` 优先于 `AnnotatedArgsOp`；同时实现会使声明式刷新不被选中。这不妨碍 `EagerExecuteOp` 与 `AnnotatedArgsOp` 的双模组合。
- 声明式任务当前限静态图；unknown graph 需要命令式实现。静态模型重放不保证每轮回调 `Execute`，按轮 Host 决策、JIT 和状态更新要按实际执行路径设计。
- 用户说的「动态 shape」或用了 TorchAir，都**不等于** GE 的 unknown graph：按实际图有无 unknown rank/dim 判断。即使不是 unknown graph，自行加载或运行期决策等需求也可能需要命令式实现。
- 判为双模时同时提供声明式与命令式实现，并**分别验证两条路径**，不以其中一条推断另一条。
- 离线编译再区分可枚举档位与 unknown 维：两者的 ATC 参数、OM 体积与执行序列都不同；V1 与 RT2 的 `Execute` 调用时机也不同，按实际运行路径核实，不从单侧源码推普遍结论。
- 命令式使用 GE 提供的 stream；辅流需要双向依赖及与重放兼容的 event 生命周期。第三方框架能否接受外部 stream，按其实际接口与版本核对；不把某一个框架的限制推广到全部 Python 算子。
- 回调的 ctx/Tensor 是借用对象，不缓存到回调之外；缓存仅保存自有数据，key 覆盖所有特化条件。
- 反序列化先完整解析并校验到临时状态，再合并或隔离，不先 clear 共享缓存。
- 同 key 不同产物（版本、长度或内容不一致）必须明确冲突策略：拒绝覆盖、保留旧值或隔离。
- 交付时给出坏 payload、不同 shape、同 key 不同产物三类测试场景与预期；未实际运行就如实标注。
- Python 实现类须可零参构造；Portable 方法在类自身成对定义，payload 为非空 `bytes`。OM 携带 Python 源码不等于目标机无需 Python、bridge 和第三方依赖。
- `AddLaunch` 的两参重载返回状态码，三参重载返回 token；按目标头文件检查返回值。
- 诊断注册或版本门控问题时，先只读核对**目标机实际安装**的头文件、运行库版本与 `GetGeVersion` 门控；更新源码仓库不会自动更新目标运行库。
- 未实现的计算不得返回成功。交付时分别报告构建、加载、数值和部署验证结果及适用版本。

## 5. 按需读取

不默认通读全部参考文件。下表是读取入口，文件内目录用于定位当前问题。

| 条件 | 文件 |
|---|---|
| 路径或注册机制不明确 | [mechanism-triage.md](references/mechanism-triage.md) |
| 选能力、离线携带状态、双模、动态 shape、V1/RT2 差异、辅流、重放 | [scenarios.md](references/scenarios.md) |
| kernel 未就绪，或需要确认双方接口 | [kernel-contract.md](references/kernel-contract.md) |
| C++ 签名、注册、上下文和地址刷新 | [interfaces.md](references/interfaces.md) |
| Python 装饰器、签名、能力和生命周期 | [python-track.md](references/python-track.md) |
| 产物格式、二进制提取、加载及 args | [language-launch.md](references/language-launch.md) |
| 模板、OPP、构建、离线交付链接、OM 执行序列与验收 | [build-verify.md](references/build-verify.md) |
| 加载失败、未回调、结果错误或重放问题 | [troubleshooting.md](references/troubleshooting.md) |
| 环境能力、源码版本、官方样例及复核依据 | [sources.md](references/sources.md) |
| 按误解关键词查找对应章节 | [misconceptions.md](references/misconceptions.md) |

可用的本地检查：`python3 scripts/check_env.py`（环境报告）、`python3 scripts/inspect_kernel.py <device-ELF>`（产物格式与符号）。
这两项都不代替真实加载和数值验证；完整操作见构建与验证文档。

维护与行为评测方法见 [evals/validation.md](evals/validation.md)。
