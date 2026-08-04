# tip: dump 前后变化 + 日志关键字清单

> 📎 导航落点：`references/fusion-troubleshooting.md` §7（图与输出是否正确）。本文件仍是 dump 变化对照与日志关键字清单的**唯一权威定义源**，正文未改动。迁移映射见 `tips/MIGRATION.md`。

**读者**：③ 分析。

## dump 文件对应关系

> 完整 dump 文件名与注册阶段、CANN 版本的对应关系（含 `kAfterOriginGraphOptimize` 和 CANN < 8.3.RC1 的函数式 graph pass 分支）见 SKILL.md 阶段三"验证顺序与证据"。

- 优化前：`ge_onnx_*_PreRunBegin.pbtxt`
- 自定义 pass 阶段：
  - `ge_onnx_*_RunCustomPassBeforeInferShape.pbtxt` → `kBeforeInferShape`
  - `ge_onnx_*_RunCustomPass_AfterInferShape.pbtxt` → `kAfterInferShape`

## 典型 dump 前后变化对照

| pass 类型 | 期望在 dump 中看到的变化 |
|---|---|
| MatMul+Add → GEMM | MatMul + Add 被单个 GEMM 替换 |
| move Relu before Concat | ConcatV2 → Relu 变为多路 Relu → ConcatV2 |
| modify Conv data_format | Conv 的 `data_format` 变为目标值（如 NHWC），冗余 Transpose 被删 |
| Add(x, 0) 消除 | Add(x, 0) 被删除，下游消费者改连 x |
| 嵌套 AddCustom | AddCustom(AddCustom(x, 0), y) 变为单层 AddCustom(x, y) |
| grouped Conv decompose | grouped Conv 被拆为 Split + 多个 Conv2D + Concat |

## 日志关键字清单

日志中应能搜到：

- `pass begin` / `pass end`
- `pattern defined`
- `meet requirements (true/false, reason=...)`
- `replacement created` / `replacement succeeded`
- `skip reason=...`
- `InferShape success` / `InferShape failure`

## 排错顺序

优先读过滤后的 GE 日志 `*-ge.log`（如 `*_decompose_python_grouped_conv-ge.log`），再看全量 stdout/slog，避免被噪声日志误导。

## 与诊断的衔接

- dump 里目标结构没变但没报错 → 多半是匹配没命中，回 `dump-first-op-type.md`。
- `Failed to select engine` → `es-all-no-version-rename.md`（背景 `kernel-registration-mismatch.md`）。
- `E50002` format → `format-sensitive-nchw.md`。
- `attribute order has changed` → `compliant-node-builder-ir-order.md`。
