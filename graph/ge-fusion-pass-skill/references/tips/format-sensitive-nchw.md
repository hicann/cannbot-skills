# tip: format-sensitive 算子输入必须显式设 NCHW/NHWC

> 📎 导航落点：`references/fusion-troubleshooting.md` §5/§6（replacement 是否成功、InferShape/format/engine）。本文件仍是 format 设置硬性做法的**唯一权威定义源**，正文未改动。迁移映射见 `tips/MIGRATION.md`。

**读者**：② 开发 / ③ 分析。

## 症状

decompose/替换里把 tensor 喂给 `Conv2D`/`Conv`/`Pool` 时报：

```
Not_Supported_Format(E50002): The format of input [x] of [Conv2D_x] op needs to be [NCHW or NHWC], but incoming format is [ND]
→ Call InferShapeAndType for node:Conv2D_x failed
→ Got null replacement graph
→ 整个 ATC 失败
```

这是"换对了 op type 仍挂"的常见根因（典型：grouped Conv 拆 SplitD→Conv2D→ConcatD）。

## 根因

ES API（`es::SplitD`/`es::Split`/`es::Const` 等）产出的 TensorHolder **默认 `format=ND`**；而 `Conv2D`/`Conv` 的 InferShape 会 `CheckConv2dFormat`，**要求输入 format 为 NCHW 或 NHWC**。两者不匹配即 E50002。

## 硬性做法

**ES 路径（最简）**：落 Conv 前对每个中间 TensorHolder 显式设 format（`SetFormat` 是链式接口，见 `es_tensor_holder.h`）：

```cpp
auto splits        = es::SplitD(x,      groups, /*split_dim=*/1, groups);  // 默认 ND
auto filter_splits = es::SplitD(filter, groups, /*split_dim=*/0, groups);
for (auto &s : splits)        s.SetFormat(ge::FORMAT_NCHW);   // ← 必须，否则 Conv2D InferShape 报 E50002
for (auto &s : filter_splits) s.SetFormat(ge::FORMAT_NCHW);
auto conv = es::Conv2D(splits[g], filter_splits[g], bias, /*...*/);
```

**CompliantNodeBuilder 路径**：InferShape 前对每个输入 `UpdateInputDesc`，并 `SetOriginFormat`/`SetOriginShape`：

```cpp
TensorDesc x_desc(in_shape, ge::FORMAT_NCHW, in_dtype);
x_desc.SetOriginFormat(ge::FORMAT_NCHW); x_desc.SetOriginShape(ge::Shape(in_shape));
conv_node.UpdateInputDesc(0, x_desc);     // filter 同理 UpdateInputDesc(1, f_desc)
```

## 自查

- dump `ge_onnx_*_AfterInferShape.pbtxt`，确认拆分后每个 Conv2D 输入的 `format:` 是 `NCHW` 而非 `ND`。
- 见到 E50002 一律回到本条补 format，**不要**靠重试或"仍然返回图（returning-graph-anyway）"绕过。
