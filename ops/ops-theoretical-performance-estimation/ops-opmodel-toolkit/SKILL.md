---
name: ops-opmodel-toolkit
description: 使用 OpModel wheel 创建算子模型、评估性能、搜索 DAG 与流水方案、分析空泡和数据流，并生成设计文档。适用于用户要求 OpModel 建模、仿真、寻优或模型与实现联合优化。
---

# OpModel 算子建模与优化工具集

用已安装的 `opmodel` wheel 和用户项目完成任务。此目录提供使用文档、七项工作流、
可编辑模型、调用示例及接口验证脚本；仿真、模型执行和搜索算法由 wheel 提供。
框架说明以模型输入、调用方式、输出含义和使用限制为范围。编写模型使用公开的
YAML 与 hooks 接口，解释结果以用户模型和本次运行产物为依据。

## 开始工作

1. 从当前项目恢复数学语义、完整 shape、dtype/layout、目标硬件、已有模型和结果。
   只补问影响正确性、比较口径或执行范围且无法从资料确定的信息。
2. 首次使用或环境异常时读[安装说明](references/install.md)，核对 wheel 版本、
   Python/系统兼容性。安装或升级时默认获取兼容当前环境的最新版本（含预发布版本），
   读取并记录实际版本，核对本次使用的接口。设置指向本目录的 `OPMODEL_SKILL_DIR`，在用户项目中运行，
   模型与结果写入 `OPMODEL_WORKSPACE`；skill 目录和安装包可以只读。
3. 按下表选当前需要的工作流。已有模型先恢复固定 case；只请求分析、文档或模型
   仿真时按该范围交付。联合优化涉及真实实现时，再使用用户已有的设备工具链。
   只请求说明或规划时，也按对应工作流的交付要点解释验证和结论边界；文件名或
   参考链接不能代替关键约束，不将尚未执行的计划写成已完成结果。

## 硬件与交付范围

新建模型先按[硬件准备](references/hardware-preparation.md)确认目标芯片／SKU、配置与来源，
保存硬件引用；已有模型和搜索沿用保存配置，显式覆盖才变更，并同口径重评。
仅要求 Vec 方案或预测接口时，按[Vec 计划](references/framework/vec-plans.md)交付程序、
分区、调用域、缓冲与流水安排及实际验证状态；无编译产物也可使用该入口。
说明计划搜索入口时，明确一次只处理一个 Vec 区域，基线须未分片且无预建存储、
控制或发射协议；仅做方案说明也须交代这些前提，未核对的条件保留为待确认。
Vec 搜索将超容或非法构造记为失败后继续后续计划，失败候选也消耗评估预算。
若实际提前退出，先核对当前包版本、异常、预算及已有 `ledger/complete`，再决定是否重跑；
声明了计划不等于已评估，更多 slots 不保证更快。
需要实现收益时再衔接编译、设备正确性和计时，各项资格分别报告。

## 七项工作流

| 用户需求 | 读取 | 主要交付 |
| --- | --- | --- |
| 从公式、实现或已有模型创建/扩展算子 | [新算子建模](references/new-operator.md) | 固定定义、case、语义契约、建模规格及适用验证 |
| 评估理论耗时、分核和 tiling | [性能评估](references/perf-eval.md) | 同口径基线/候选耗时、覆盖与瓶颈 |
| 搜索固定 DAG 配置或不同结构、映射方案 | [DAG 方案寻优](references/agent-dag-opt.md) | 方案族、实际搜索、候选及复评证据 |
| 联合模型、DSL/AscendC 实现和实测 | [联合寻优](references/joint-optimization/guide.md) | 实现收益、模型反馈与可复现记录 |
| 定位 trace 中的流水空泡和等待 | [空泡分析与优化](references/dag-bubble-opt.md) | 时间线观察、待验证原因及对照实验 |
| 画计算流程、检查 stage 数据连接 | [数据流图](references/dag-dataflow-graph.md) | Mermaid/ASCII 图与声明连接核对 |
| 将模型或寻优结果交给开发者实现 | [设计文档生成](references/opmodel-design-gen.md) | 设计文档；多实现模板时含总览、各模板设计与逐 case 映射 |

通过 `ops-opmodel-toolkit` 按任务选择工作流，调用示例见[工作流索引](references/workflows.md)。
设计交接按实现方案组织：同一实现模板的多个 case 共用设计正文，结构不同的模板分别成文；
总览关联 case、分派规则、模板、候选与证据，具体规则见设计文档生成工作流。

## 首次运行与参考路由

先按[快速入门](references/quickstart.md)复制矩阵乘示例，运行完整 case，检查
`result.json`，再搜索分核并复评。该教程无需设备；模型成本不代表实测性能。

- 查模型输入、调用接口或支持域：按需读[框架参考](references/framework/index.md)。
- 表达 Vec/VF 方案、预测单次调用或搜索显式计划：读[Vec 计划与预测](references/framework/vec-plans.md)。
- 声明变换搬运、核对读写量和释放边界：读[搬运契约](references/framework/transfer-contracts.md)。
  说明工作量与空泡时，分列逻辑字节、实际读写、物理容量、命令数和内部服务单元；
  内部服务单元不增加命令数，重叠的服务与等待不能相加作总耗时。
- 交接真实实现的构建结果：读[编译反馈](references/framework/compile-feedback.md)。
  解释更新流程时包含失败分支：编译、资源检查、设备正确性或计时失败，均保留
  实际退出码、原始日志及已有报告，记录失败范围；模型、编译和设备资格分别报告。
- 编写定义或 hooks：读[建模指南](references/modeling.md)与[字段参考](references/model-reference.md)。
- 选择执行模式和校验：读[仿真说明](references/simulation.md)。
- 解读周期、统计和 trace：读[结果分析](references/analysis.md)。
- 接入搜索：读[统一搜索接入](references/agent-dag-opt/search-integration.md)；具体命令见[参数寻优](references/optimization.md)和[wheel 命令](references/joint-optimization/wheel-commands.md)。
- 查调用参数：读[CLI](references/cli.md)或[Python API](references/python-api.md)。
- 选择或覆盖已有 profile：读[硬件配置](references/hardware.md)。
- 处理版本缺口与失败：读[能力矩阵](references/capabilities.md)和[故障排查](references/troubleshooting.md)。

只加载本次任务需要的参考。示例、模板和脚本均在本目录内；完整阅读索引见
[使用指南](references/index.md)，版本要求和验证方法见[版本与验证](references/distribution.md)。

## 结果与运行约束

- 模型、hooks、case 和实验配置保存在用户工作区，安装目录保持只读。
  能力不足时保存最小复现、影响范围与可行替代方案。
- 分清完整输入和逐核工作量；保留实际硬件、成本、执行入口与结果 scope。
  搜索局部周期与完整 case 周期分别呈现；候选应用后用相同输入、硬件和成本复评。
  说明比较流程或报告收益时，同时列出完整 case 输入、基线/候选实际生效的执行参数
  （tile、核数、切分/映射、slots/phases）及搜索覆盖；缺配置或产物的项标为待确认，
  不编造参数值或覆盖结论，具体核对项见[比较证据](references/agent-dag-opt/evidence.md#局部评分与完整-case)。
- 沿用用户目标及修订；仅换目标时按[重评分接口](references/framework/objectives.md)
  复用有效观察。完整方案排名／并列、逐 case 耗时诊断与资格分别呈现。
- 记录真实搜索空间、固定维度、淘汰和预算未覆盖项。固定点、会话登记或 Top-K
  列表本身不能证明完成搜索，也不能证明全局最优。
- 模型数值检查、后端一致性、成本资格和设备正确性分别报告。VF 实测、系数/公式和
  可选预测须注明来源与适用域；缺 VF 实测可先估算并作探索性搜索，保留未校准状态
  与不确定性。确实未知的成本不补零值，探索性排名不等于设备收益。
- 运行结果使用新目录，完整 case 工作区避开系统 `/tmp`；检查失败保留原因，
  不通过删掉依赖、容量或正确性约束来获得周期数。
- 交付有效模型/候选路径、复现命令、实际完成的验证及未解决限制。仅做模型工作时，
  结论限定为声明成本下的模型结果；设备收益需要同口径的真实实现验证。
