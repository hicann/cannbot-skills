# completeness-checklist：完整性自查

对应 Task A 执行步骤 7：完整性自查。

---

输出路径清单后，必须输出 `completeness_checklist` 字段（写入 S2P1_path_list.json）。

## JSON Schema

```json
{
  "completeness_checklist": {
    "api_variants": {"status": "covered|missing|na", "evidence": ["P1: Tensor API"]},
    "format_variants": {"status": "covered|missing|na", "evidence": []},
    "mode_variants": {"status": "covered|missing|na", "evidence": []},
    "quant_variants": {"status": "covered|missing|na", "evidence": []},
    "optional_input_combos": {"status": "covered|missing|na", "evidence": []},
    "dispatch_coverage": {"status": "covered|missing|na", "evidence": []},
    "tiling_analysis": {"status": "covered|delegated|na", "evidence": [], "note": "仅在 status=delegated 时填写，说明框架名称"}
  }
}
```

## 检查项说明

| 检查项 | 说明 |
|--------|------|
| api_variants | 算子是否有多种调用方式（Tensor vs Scalar、inplace vs outplace）？ |
| format_variants | 是否支持多种数据格式（NCHW/NHWC/ND/5D）？ |
| mode_variants | 是否有 static/dynamic、training/inference 等模式切换？ |
| quant_variants | 是否覆盖了所有量化类型（per-tensor/per-channel/per-token）？ |
| optional_input_combos | 每个可选输入的 present/absent 是否都在某条路径中出现？ |
| dispatch_coverage | 所有 kernel dispatch 分支是否都有对应路径覆盖？ |
| tiling_analysis | tiling 逻辑是否被完整分析？若委托给通用框架则 status=delegated |

## 状态值

- `covered`：已覆盖
- `missing`：存在但未体现
- `na`：不涉及
- `delegated`：仅 tiling_analysis，委托给通用框架
