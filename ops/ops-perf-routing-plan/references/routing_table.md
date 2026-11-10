# 路由表

## 产物

`route_report.py` 从冻结的 `plan.json` 生成：

- `routes[]`：分类模板，按 `priority` first-match；
- `case_to_template`：每个 case 的实际分类模板；
- `tiling_table.csv`：逐 case 的 tiling_key、核数、tile、尾块策略和解析状态；
- `coverage_report.json`：覆盖检查、未路由 case 和错误。

## 规则

1. 模板创建顺序就是优先级。
2. `predicate` 只针对创建时的 `remaining` 设计。
3. 已覆盖 shape 可以被后续 predicate 命中，但 first-match 会保留旧模板。
4. `original` 只是默认兜底；冻结计划必须覆盖全部 shape，不能靠兜底掩盖遗漏。
5. 每个模板必须有 `representative_case_id`，且它必须是创建时 `remaining[0]`。
6. `is_single_case_group=true` 表示未形成有效分组，不得把该结果外推到不同特征的 shape。

## 示例

```json
{
  "stage": "frozen",
  "routes": [
    {
      "priority": 1,
      "template": "large_n_ub_full",
      "tiling_key": 1,
      "representative_case_id": "case_0042",
      "is_single_case_group": false,
      "predicate": "total_elems > 1048576 and align_ok",
      "cases": ["case_0042", "case_0051"]
    }
  ],
  "default": {"template": "original", "tiling_key": 0},
  "case_to_template": {
    "case_0042": "large_n_ub_full",
    "case_0051": "large_n_ub_full"
  }
}
```
