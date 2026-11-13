# 脚本说明

两个脚本都是**纯标准库 Python 3**，无第三方依赖，可在无 NPU 的机器上跑。

## scan_a5_only.py

静态扫描 Ascend C 算子工程中的 A5（DAV_3510 / arch35）专有构造，给出 A2 处置提示与迁移等级初判。

```bash
# 人读报告
python3 scripts/scan_a5_only.py --root /path/to/operator

# 报告落盘（md + json）
python3 scripts/scan_a5_only.py --root . --format both --out migration_scan

# 阶段 2 自检：仍有 blocking/verify 时返回 1
python3 scripts/scan_a5_only.py --root . --fail-on blocking --quiet
```

- `--root`：算子工程根目录。
- `--format`：`md`（默认）/ `json` / `both`；`--out` 为文件名，`both` 时自动加 `.md`/`.json` 后缀。
- `--exclude`：按相对路径正则排除（可重复）。
- `--fail-on`：`none`（默认）/ `blocking`（blocking+verify 命中即退出 1）/ `decision`（再加 decision）。
- 扫描范围：源码/构建文件（含 `.inc/.inl` 等 C/C++ 片段扩展名）；被跳过或超限的文件会在 Markdown/JSON 的 `skipped` 清单中列出，GATE-2 需人工确认该清单里没有 A5 构造。
- `--quiet`：不打印报告正文，只打印落盘路径与 SUMMARY。

**等级初判规则**：命中 RegBase/SIMT → D3；命中通路/同步/dtype/精度 → D2；仅工程接线 → D1；无命中 → D1（仍需人工确认工程接线）。

**规则维护**：规则表在 `scan_a5_only.py` 的 `RULES` 中，每条含 `rule_id/category/severity/level/pattern/message/a2_action`。新增规则时：

1. 正则尽量锚定字面 token（`\b` 边界），避免匹配注释里的普通单词。
2. `severity` 与 `level` 必须和 `references/a5-only-inventory.md` 对应行一致。
3. 加一条最小样例验证（正例命中、反例不命中）。
4. 不确定是否 A2 专有的，用 `verify` 而不是 `blocking`。

**能识别**：字面量 token、目录名、构建脚本字符串。
**不能识别**：宏拼接、运行时字符串、语义依赖（例如"用 GM 中转模拟 L0C→UB"）。这些靠阶段 1 的人工补盲。

## a2_ub_budget.py

按 A2（DAV_2201，典型 192KB/AIV）核算 per-AIV UB 预算，或把 A5 tile 缩放到 A2。

```bash
# 预算核算：name:dtype:elements:copies
python3 scripts/a2_ub_budget.py \
  --buffer in_x:float16:32768:2 \
  --buffer out:float32:32768:2 \
  --reserve-kb 2

# A5 -> A2 tile 缩放（默认 A5 248KB-40KB=208KB，A2 192KB）
python3 scripts/a2_ub_budget.py --scale-from-a5 --tile-elements 32768 --dtype float16 --align-bytes 32

# 机器可读
python3 scripts/a2_ub_budget.py --buffer a:bf16:16384:2 --json
```

- dtype 支持：`float32/fp32`、`float16/fp16/half`、`bfloat16/bf16`、`int8`、`int16`、`int32`、`int64`、`fp8_e4m3`、`fp8_e5m2`、`hifloat8`、`int4`。
- 退出码：`0` 预算通过；`1` 超预算；`2` 参数错误。
- 只做算术：padding、对齐、访问模式、双缓冲策略仍需人判断；PASS 后必须在阶段 3/4 真机确认无 `UB out-of-range`。

**校验口径**：`--ub-kb` 默认 192 是 DAV_2201 典型用户可用值；实际以 `GetCoreMemSize(CoreMemType::UB)` 为准，拿到真值后用 `--ub-kb` 传入。

## 测试建议

- 扫描器：构造一个含 `__VEC_SCOPE__` + `ascend950` 的最小工程，确认输出 D3 且 `--fail-on blocking` 返回 1。
- 预算器：`--buffer a:float32:16384:2`（131072 B）应 PASS；改成 `:65536:` 应 OVER 且退出 1。
