---
name: catlass-cpp-develop
description: CATLASS C++ 五阶段工作流的 04 算子开发。用于按已评审设计逐 Stage 实现、构建和快速精度验证。触发：workflow 处于 implementation 且设计校验通过时。
---

# 04 算子开发

## 阶段说明

| 输入 | 本阶段处理 | 输出 |
|---|---|---|
| 已评审 `docs/design.md`、已验收的接口/标杆/精度策略、`stage=implementation` | 按 Stage 实现、构建、设计一致性检查和快速精度验证 | 代码和构建产物、更新后的设计/验证记录 |

## 准入规则

- 仅 `stage=implementation`，且 `issue_type` 为 `null` 或 `precision_debug`、设计校验已通过时进入本阶段。
- 确认 workflow、设计文档、编译参数和设备架构一致，并按实际组件查询 API；架构冲突返回 03。

## 执行规则

- 按 Stage 执行“实现 → 构建 → 设计校验 → 运行 → 快速精度比较”。device、host tiling、workspace、AIC/AIV 分工、同步和数据生命周期必须与设计一致。
- 正式实现必须保留设计阶段确定的跨 work 并行和 Stage 流水；不得用逐 work 串行生产、计算、消费替代正式方案。
- 理论性能可达不等于初版实测达标。实现阶段优先验证高性能数据通路和生命周期，性能实测与优化留在 validation 阶段。
- Stage、公式、地址、数据类型、任务映射、缓冲、生产消费关系、同步或 workspace 发生变化时，先更新 `docs/design.md`，再继续实现。
- 快速精度使用本轮构建产物，只运行失败用例（如有）、受影响 Stage 用例和最小边界精度用例，重叠用例可去重；跨 AIC/AIV 通知和本地 pipe 完成条件必须分别闭合。连续 Vector 操作使用相同 mask 时只设置一次，mask 或有效 lane 改变后重新设置。
- 首个 Stage 开发、同步调试或精度定位时读取 [开发与定位方法](references/development.md)；临时检查点定位完成后删除。
- 编译、链接或代码实现失败且根因属于当前实现时，在本阶段修复，并从失败步骤继续执行。

## 准出规则

### 验收

保持 `stage=implementation`，确认设计一致性、构建和快速精度均通过后执行：

```bash
python scripts/validate_implementation.py \
  --design <docs/design.md> --workflow <docs/workflow.json> \
  --validation <docs/validation.md>
```

该校验统一检查 workflow 状态、设计文档和快速精度记录。

### 状态更新

| 检查结果 | 转移事件 | 下一状态 |
|---|---|---|
| 需要定向精度定位 | `precision_problem` | `implementation / precision_debug` |
| 发现设计问题 | `design_problem` | `design / null` |
| 发现公式或标杆问题 | `reference_problem` | `reference / null` |
| 发现接口问题 | `interface_problem` | `interface / null` |
| 构建和快速精度通过 | `implementation_accepted` | `validation / null` |

通过 `catlass-cpp-interface/scripts/transition_workflow.py` 执行表中的转移；状态更新成功后才进入对应阶段。
