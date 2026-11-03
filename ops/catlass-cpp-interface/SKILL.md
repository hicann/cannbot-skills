---
name: catlass-cpp-interface
description: CATLASS C++ 五阶段工作流的 01 接口确认。用于确认公开接口、目标架构和计算语义，并初始化或恢复 workflow。触发：开始 CATLASS C++ 算子，或 workflow 处于 interface 时。
---

# 01 接口确认

## 阶段说明

| 输入 | 本阶段处理 | 输出 |
|---|---|---|
| 用户需求、接口资料、测试或参考实现、目标产品和性能目标 | 确认公开接口、支持范围和计算语义 | `docs/api.md`、已确认的 `algorithm_family` 和 `target_architecture` |

## 准入规则

- 新工程运行 `scripts/init_operator_project.sh`；已有工程仅在 `stage=interface`、`issue_type=null` 时进入本阶段。进入后先校验 `docs/workflow.json`。

## 执行规则

- 按数学语义选择 `algorithm_family`；普通 CATLASS 算子使用 `general`，已登记家族通过 `catlass-cpp-knowledge` 查询 Operator 卡。
- 在 `docs/api.md` 的“接口签名”“支持范围”“边界行为”三个章节记录对应内容；公开文档、host、TilingData、kernel 入口和测试的签名必须一致。
- 同一公开接口在不同 SoC 上保持一致；预留参数需明确类型、当前行为、host 校验和兼容方式。
- 资料冲突按“用户明确说明和公开接口 → 测试与示例 → 参考实现 → 其他资料”裁决，未裁决内容不得冻结。
- 所有 `algorithm_family` 均支持 `atlas_a2_a3`（Atlas A2系列产品、Atlas A3系列产品）和 `ascend950`（Ascend 950PR&950DT系列产品）。用户使用的 A2、A3 归一为 `atlas_a2_a3`，A5、950 归一为 `ascend950`。

## 准出规则

### 验收

先校验接口文档已完成：

```bash
python scripts/validate_interface.py --api <operator>/docs/api.md
```

校验通过后，由用户确认 `docs/api.md` 的接口、支持范围和计算语义，且
`algorithm_family` 和 `target_architecture` 已确定。

接口校验或评审未通过时，修正 `docs/api.md` 或目标架构后重新验收。

### 状态更新

| 检查结果 | 转移事件 | 下一状态 |
|---|---|---|
| 接口验收通过 | `interface_accepted` | `reference / null` |

接口验收通过后执行：

```bash
python scripts/transition_workflow.py \
  --workflow <docs/workflow.json> --event interface_accepted \
  --algorithm-family <algorithm_family> \
  --target-architecture <target_architecture>
```

该事件同时写入已确认的算法家族和目标架构。状态更新成功后才进入 02。
