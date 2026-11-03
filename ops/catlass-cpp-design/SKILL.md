---
name: catlass-cpp-design
description: CATLASS C++ 算子五阶段工作流的 03 方案设计。用于根据已验收接口、标杆和目标架构规则生成可实现的 Stage 设计。触发：workflow 处于 design 时。
---

# 03 方案设计

## 阶段说明

| 输入 | 本阶段处理 | 输出 |
|---|---|---|
| 已验收接口和标杆、精度策略、性能目标、目标 SoC 能力；性能优化返回时还包括当前代码、`docs/design.md` 和 `docs/validation.md` | 确定 Stage、任务映射、tiling、数据类型、片上空间、数据通路和同步 | 通过校验和评审的 `docs/design.md` |

## 准入规则

- 仅 `stage=design`，且 `issue_type` 为 `null` 或 `performance_optimize` 时进入本阶段。精度策略、性能目标和目标架构必须齐备。
- 设计的计算语义和芯片规则必须分别与 `docs/workflow.json` 中的 `algorithm_family` 和 `target_architecture` 一致，不能因 Stage 划分或任务映射改变而切换家族或架构。

## 执行规则

- 通过 `npu-arch` 获取目标 SoC 能力；通过 `catlass-cpp-knowledge` 按 `target_architecture`
  查询并完整读取唯一对应的芯片规则文件，完整读完后再推导，不用其它架构兜底。
  专门家族查询 Operator 卡；`general` 不查询 Operator 卡。两者都按实际组件查询 API 卡，
  并核对目标版本 CATLASS/CANN 组件源码中的接口、能力与约束。
- 将所选芯片规则文件中的全部 `Rxx` 逐条写入“芯片规则应用检查”：每条规则恰好出现一次，
  结论只能是“已应用”或“不适用”。“已应用”必须指向当前设计中的具体落点；“不适用”必须说明
  与当前算子不匹配的原因。存在缺项、重复项、空证据或空原因时不得定稿。
- 按公式、数据依赖、计算类型、生命周期和精度观察点划分逻辑 Stage，再确定 AIC/AIV 任务映射、tiling、同步和边界处理。
  同一 AIV 可以顺序执行多个逻辑 Stage，Stage 不与核或 pipe 一一对应。任务映射必须从当前接口、完整支持域、计算图和目标芯片资源推导，不能直接沿用参考代码的分工；输入不足时说明待确认条件，不臆造计算语义。
- 在进入实现前完成理论性能上限分析。按 AIC/Cube、AIV/Vector、GM/MTE 和不可重叠依赖链分别推导理论最低耗时，
  用关键路径而不是各 pipe 时间简单求和，并按当前 case 的工作量换算理论最大吞吐。设计阶段判断目标是否理论可达，不要求初版实现已经达到目标；只有目标低于理论最低耗时，
  或并行度、资源和依赖证据不足时，才判定设计不可行。
- 正式设备方案必须通过跨 work 并行和 Stage 流水表达；不得把逐 work 串行生产、计算、消费作为正式设计。串行路径只能用于 CPU
  reference 或临时诊断，不能作为实现准出或性能 baseline。
- 跨 Stage 同步契约必须分别写清数据边的 Producer、Consumer、ready、free/复用、本地 pipe 完成以及 slot 回绕、空任务和退出 drain；
  明确完整数据范围、参与方和通知次数；已进入协同流程的空任务保持通知配对，复用通知 ID 或退出前排空上一轮通知，不能只写事件编号或“等待完成”。
  同一 pipe 按实际依赖使用程序顺序或必要的 PipeBarrier，本地跨 pipe 使用目标版本支持的 HardEvent、Queue 或等价机制；跨 AIC/AIV 使用 CrossCore 通知，并单独闭合本地 pipe 完成条件。不存在异步依赖时不额外增加同步。
- 当 `issue_type=performance_optimize` 时，读取 `docs/validation.md` 中最新一轮性能结果、流水分析和已选优化候选。以当前代码和当前 `docs/design.md` 为 baseline，本轮只设计该候选；候选验证完成后，以保留或回退后的当前代码作为下一轮 baseline，并将最终采用的设计同步收敛到 `docs/design.md`。
- 按 `templates/design.md.template` 生成 `docs/design.md`。
- 候选推导可写入 `docs/design.full.md`，`docs/design.md` 只保留最终方案。

## 准出规则

### 验收

执行设计校验：

```bash
python scripts/validate_design.py \
  --design <docs/design.md> --workflow <docs/workflow.json>
```

芯片规则应用检查完整、设计校验和整体评审必须全部通过。

设计校验或评审未通过时，根据失败结果修正 `docs/design.md` 后重新验收。

### 状态更新

| 检查结果 | 转移事件 | 下一状态 |
|---|---|---|
| 发现接口问题 | `interface_problem` | `interface / null` |
| 发现公式或标杆问题 | `reference_problem` | `reference / null` |
| 设计验收通过 | `design_accepted` | `implementation / null` |

通过 `catlass-cpp-interface/scripts/transition_workflow.py` 执行表中的转移；状态更新成功后才进入 04。
