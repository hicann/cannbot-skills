---
name: catlass-cpp-reference
description: CATLASS C++ 五阶段工作流的 02 标杆生成。用于生成唯一 CPU 标杆并校准精度策略。触发：workflow 处于 reference 时。
---

# 02 标杆生成

## 阶段说明

| 输入 | 本阶段处理 | 输出 |
|---|---|---|
| 已验收的 `docs/api.md`、参考资料和预期结果 | 生成并运行 CPU 标杆，确定精度判定口径 | `reference/reference.py`、`reference/definition.json`、`reference/precision-policy.json`、标杆运行证据 |

## 准入规则

- 仅 `stage=reference`、`issue_type=null` 且 `target_architecture` 已确定时进入本阶段。

## 执行规则

- 读取 [CPU 标杆生成与对齐](references/reference_generation.md)，并通过 `catlass-cpp-knowledge`
  完整读取 `guides/precision_policy.md`。
- `reference/reference.py` 是唯一可编辑源码；`definition.json` 必须由脚本生成，不得单独编辑。
- 实现可由测试代码直接调用的纯 CPU 标杆；从 `docs/api.md` 推导适用用例，先用可人工检查的小规模输入对齐计算语义，再执行正常、边界和非法输入。
- 固定输入、随机种子、初始化、状态和计算顺序；存在近似算法或顺序差异时，记录双方计算顺序、结果差异和阈值依据。
- 按实际输出 dtype、值域、shape 和有效区域校准 `reference/precision-policy.json`，确认公式、状态更新和比较区域。
- 实际执行调用 `reference.py` 的 CPU 测试；在 `docs/validation.md` 的“02 标杆运行”记录执行命令、用例范围、结果和证据。缺少该文件时按 `catlass-cpp-test/templates/validation.md.template` 创建。
- 精度阈值依据冻结的标杆语义和精度策略确定，不得为使 DUT 通过而放宽。

## 准出规则

### 验收

保持 `stage=reference`。先执行当前算子工程的 CPU 标杆测试并记录运行证据，再执行：

```bash
python scripts/generate_definition.py --source <reference.py> --template <definition.template.json> --output <definition.json>
python scripts/validate_reference.py --source <reference.py> --definition <definition.json>
python scripts/validate_precision_policy.py --policy <precision-policy.json>
python ../catlass-cpp-test/scripts/validate_validation.py --validation <docs/validation.md> --workflow <docs/workflow.json>
```

用户确认 golden 的计算语义和精度策略。

标杆或精度策略未通过验收时，根据失败证据修正 `reference/reference.py`，重新生成
`definition.json`，必要时修正精度策略后重新验收。

### 状态更新

| 检查结果 | 转移事件 | 下一状态 |
|---|---|---|
| 发现接口问题 | `interface_problem` | `interface / null` |
| 标杆验收通过 | `reference_accepted` | `design / null` |

通过 `catlass-cpp-interface/scripts/transition_workflow.py` 执行表中的转移；状态更新成功后才进入 03。
