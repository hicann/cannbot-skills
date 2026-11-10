# tiling_plan.json schema

```json
{
  "core_split": {
    "units": "total_elems",
    "num_cores": "min(aiv_core_num, ceil(total_elems / 1024))"
  },
  "tile": {
    "axis": "flat",
    "elems_per_tile": "min(total_elems, max(1, floor(ub_size * 0.8 / (4 * elem_bytes))))",
    "double_buffer": 2,
    "ub_bytes_per_elem": "2 * elem_bytes"
  },
  "tail_policy": "mask",
  "alignment": {
    "ub_block_bytes": 32,
    "require_inner_align": false
  }
}
```

示例中 `4 = 2（输入/输出各一份） × 2（double buffer）`。若修改 `double_buffer`，需同步调整分母，确保 `tile_elems * ub_bytes_per_elem * double_buffer <= 0.8 * ub_size`。

## 校验

`greedy_route.py commit` 会逐 shape 求值：

- `num_cores` 在 `[1, aiv_core_num]`
- `elems_per_tile >= 1`
- UB 占用 ≤ `0.8 * ub_size`
- 末维对齐（若 `require_inner_align=true`）
- 尾块：计算 `tail_elems = total_elems % elems_per_tile`；`tail_elems > 0` 时 `tail_policy` 必须为显式尾块处理策略（如 `mask` / `pad` / `handle`），禁止 `none` / `forbid` / 空值
- 表达式可安全求值

失败时不会静默通过；需修改 proposal 或模板 tiling。
