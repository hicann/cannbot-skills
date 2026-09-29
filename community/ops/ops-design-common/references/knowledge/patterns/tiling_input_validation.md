---
name: tiling_input_validation
description: Tiling 入口异常值校验模式，覆盖 dtype/format/维度/attr/shape 五类校验，适用于所有范式。
title: Tiling Input Validation Patterns
purpose: Provide reusable validation functions that reject illegal inputs at tiling entry, before FindSplitAxis.
read_when:
  - The tiling function needs to reject unsupported dtype, format, or dimension.
  - The operator has attrs that require value-range checking.
  - The operator needs shape consistency or broadcast compatibility validation.
  - L2 exception test cases need a corresponding tiling-side rejection path.
not_for:
  - Kernel-side runtime validation (this is host-side tiling only)
  - API constraint lookup (use api/ subtree instead)
keywords:
  - validation
  - dtype check
  - format check
  - dimension check
  - attr range
  - shape consistency
  - exception
  - L2
next_reads:
  - tiling_patterns.md
  - regbase_operator_patterns.md
  - ../dev-experience/tiling_review_notes.md
depth: foundation
topic_type: pattern
type: knowledge_card
platform: common
verified: false
patterns: [validation, dtype, format, dimension, attr, shape]
---

# Tiling Input Validation Patterns

> 异常值校验是 Tiling 入口的硬门禁，在输入预处理（如 `PadAndSqueeze`）之后、`FindSplitAxis` 之前执行。
> 每个校验函数返回 `bool`，通过 `OP_CHECK_IF` 拦截不合法输入并返回 `GRAPH_FAILED`。
>
> **校验顺序**：dtype → format → 维度上限 → attr 值域 → shape（broadcast 兼容 or 一致性）。
>
> **GEIR cast 说明**：GEIR 图通路中编译器可能插入 Cast 节点将 dtype 转换为算子支持的类型。
> Tiling 函数从 `TilingContext` 获取的 dtype **已经是 cast 后的结果**，必定落入 spec.yaml
> §2.2 支持列表——因此 tiling 侧只需校验入口实际 dtype 是否在支持列表中，不需要额外容忍
> 未 cast 的中间 dtype。


## 1. 数据类型校验

每个输入有独立的 dtype 支持范围（来自 spec.yaml §2.2，按 input 逐行列出）。

```c++
bool CheckDtypeSupport(
    gert::TilingContext* ctx, int64_t numInputs,
    const std::vector<std::set<ge::DataType>>& supportedDtypesPerInput)
{
    for (int64_t i = 0; i < numInputs; i++) {
        auto desc = ctx->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, desc);
        ge::DataType dt = desc->GetDataType();
        if (supportedDtypesPerInput[i].find(dt) == supportedDtypesPerInput[i].end()) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ctx->GetNodeName(), "input[" + std::to_string(i) + "]",
                                                      ge::TypeUtils::DataTypeToSerialString(dt).c_str(),
                                                      "dtype not in the supported list");
            return false;
        }
    }
    return true;
}
```

## 2. 数据格式校验

校验每个 input/output 的 format 在算子支持列表中。

> **CANN API 注意**：`CompileTimeTensorDesc::GetFormat()` 返回 `const StorageFormat&`，
> 不是 `ge::Format`。需用 `GetStorageFormat()` 取运行时 format，或 `GetOriginFormat()`
> 取原始 format。Tiling 侧校验运行时 format 即可。

```c++
bool CheckFormatSupport(
    gert::TilingContext* ctx, int64_t numInputs, int64_t numOutputs,
    const std::set<ge::Format>& supportedFormats)
{
    for (int64_t i = 0; i < numInputs; i++) {
        auto desc = ctx->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, desc);
        ge::Format fmt = desc->GetStorageFormat();
        if (supportedFormats.find(fmt) == supportedFormats.end()) {
            OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(ctx->GetNodeName(), "input",
                ("[" + std::to_string(i) + "] " + ge::TypeUtils::FormatToSerialString(fmt)).c_str(),
                "format not in the supported list");
            return false;
        }
    }
    for (int64_t i = 0; i < numOutputs; i++) {
        auto desc = ctx->GetOutputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, desc);
        ge::Format fmt = desc->GetStorageFormat();
        if (supportedFormats.find(fmt) == supportedFormats.end()) {
            OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(ctx->GetNodeName(), "output",
                ("[" + std::to_string(i) + "] " + ge::TypeUtils::FormatToSerialString(fmt)).c_str(),
                "format not in the supported list");
            return false;
        }
    }
    return true;
}
```

## 3. 超出维度的 tensor 校验

校验每个 input/output 的 rank 不超过 `maxRank`（与 TilingKey 模板划分一致，Broadcast 范式 `maxRank=8`）。

```c++
bool CheckMaxDimensions(
    const std::vector<std::vector<int64_t>>& inputShapes,
    const std::vector<std::vector<int64_t>>& outputShapes,
    int64_t maxRank)
{
    for (size_t i = 0; i < inputShapes.size(); i++) {
        if ((int64_t)inputShapes[i].size() > maxRank) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("CheckMaxDimensions", "input",
                ("[" + std::to_string(i) + "] rank " + std::to_string(inputShapes[i].size())).c_str(),
                ("rank must not exceed " + std::to_string(maxRank)).c_str());
            return false;
        }
    }
    for (size_t i = 0; i < outputShapes.size(); i++) {
        if ((int64_t)outputShapes[i].size() > maxRank) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("CheckMaxDimensions", "output",
                ("[" + std::to_string(i) + "] rank " + std::to_string(outputShapes[i].size())).c_str(),
                ("rank must not exceed " + std::to_string(maxRank)).c_str());
            return false;
        }
    }
    return true;
}
```

## 4. 算子属性字段值域校验

按算子 DESIGN §2.1 定义的 attr 取值范围逐 attr 校验。有 attr 的算子（如 `lp_norm_v2` 的 `p`/`axes`/`keepdim`）必须实现；无 attr 的算子（如 `adam`/`xdivy`）跳过。

以下为 int 型 attr 的参考实现（bool/float 型同理，替换 getter 为 `GetBool`/`GetFloat`）：

```c++
struct AttrRange {
    const char* name;
    int64_t     index;
    int64_t     defaultVal;
    int64_t     minVal;
    int64_t     maxVal;
};

bool CheckAttrValueRange(
    gert::TilingContext* ctx,
    const std::vector<AttrRange>& intAttrRanges)
{
    auto attrs = ctx->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, attrs);
    for (const auto& r : intAttrRanges) {
        const int64_t* p = attrs->GetInt(r.index);
        int64_t val = (p != nullptr) ? *p : r.defaultVal;
        if (val < r.minVal || val > r.maxVal) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx->GetNodeName(), r.name,
                std::to_string((long)val).c_str(),
                ("value must be in range [" + std::to_string(r.minVal) + ", " + std::to_string(r.maxVal) + "]").c_str());
            return false;
        }
    }
    return true;
}
```

枚举型 attr（如 `reduction` 取值 `{0,1,2}`）可改用 `std::vector<int64_t> enumValues` + `std::find` 校验。

## 5. shape 一致性校验

**非 broadcast 算子**（输入 shape 必须完全一致）使用此函数替代 `CheckBroadcastShape`。
**Broadcast 算子**两者都保留：先 `CheckShapeConsistency`（若通过则无需广播），再 `CheckBroadcastShape`（处理需要广播的情况）。

```c++
bool CheckShapeConsistency(
    const std::vector<std::vector<int64_t>>& inputShapes)
{
    if (inputShapes.size() < 2) return true;
    const auto& ref = inputShapes[0];
    for (size_t i = 1; i < inputShapes.size(); i++) {
        if (inputShapes[i].size() != ref.size()) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("CheckShapeConsistency", "input",
                ("[" + std::to_string(i) + "] rank " + std::to_string(inputShapes[i].size())).c_str(),
                ("rank must equal input[0] rank " + std::to_string(ref.size())).c_str());
            return false;
        }
        for (size_t d = 0; d < ref.size(); d++) {
            if (inputShapes[i][d] != ref[d]) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("CheckShapeConsistency", "input",
                        ("[" + std::to_string(i) + "] dim[" + std::to_string(d) + "] " + std::to_string((long)inputShapes[i][d])).c_str(),
                        ("dim must equal input[0] dim " + std::to_string((long)ref[d])).c_str());
                return false;
            }
        }
    }
    return true;
}
```

## 调用约定

> **头文件**：使用 `ge::TypeUtils::DataTypeToSerialString` / `FormatToSerialString` 需 `#include "graph/utils/type_utils.h"`。

在 `GetShapeInfo()` 中输入预处理（`PadAndSqueeze`）之后、`CheckBroadcastShape` 之前，按固定顺序调用：

```c++
// 算子支持的 dtype/format/maxRank — 来自 spec.yaml §2.2/§2.3 和 DESIGN §3.1
static const std::set<ge::DataType> SUPPORTED_DTYPES  = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};
static const std::set<ge::Format>   SUPPORTED_FORMATS = {ge::FORMAT_ND};
constexpr int64_t MAX_RANK = 8;

OP_CHECK_IF(
    !CheckDtypeSupportAndCombination(ctx_, (int64_t)rawInputShapes_.size(), SUPPORTED_DTYPES),
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ctx_->GetNodeName(), "inputs", "unsupported", "dtype check failed"),
    return ge::GRAPH_FAILED);

OP_CHECK_IF(
    !CheckFormatSupport(ctx_, (int64_t)rawInputShapes_.size(),
                        (int64_t)rawOutputShapes_.size(), SUPPORTED_FORMATS),
    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(ctx_->GetNodeName(), "inputs/outputs", "unsupported", "format check failed"),
    return ge::GRAPH_FAILED);

OP_CHECK_IF(
    !CheckMaxDimensions(rawInputShapes_, rawOutputShapes_, MAX_RANK),
    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ctx_->GetNodeName(), "inputs/outputs", "exceeded", "dimension check failed"),
    return ge::GRAPH_FAILED);

// attr 校验（若有 attr）:
// OP_CHECK_IF(
//     !CheckAttrValueRange(ctx_, {{...}}),
//     OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_->GetNodeName(), "attr", "out of range", "attr check failed"),
//     return ge::GRAPH_FAILED);

// shape 校验 — broadcast 型用 CheckBroadcastShape（范式特有）；
//          非 broadcast 型用 CheckShapeConsistency。
OP_CHECK_IF(
    !CheckBroadcastShape(normalInputShapes_, normalOutputShapes_, rank_),
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ctx_->GetNodeName(), "input/output shapes",
        "incompatible", "check broadcast shape failed, shapes must be broadcast-compatible"),
    return ge::GRAPH_FAILED);
```

## Checklist

- [ ] dtype 校验：不支持类型 + 组合一致性（≥2 输入时）
- [ ] format 校验：输入 + 输出
- [ ] 维度上限校验：输入 + 输出，maxRank 来自模板划分
- [ ] attr 值域校验（若有 attr）：每个 attr 逐一校验
- [ ] shape 校验：broadcast 型 → CheckBroadcastShape；非 broadcast 型 → CheckShapeConsistency
- [ ] 校验顺序：dtype → format → 维度 → attr → shape
- [ ] GEIR cast：tiling 侧只校验入口实际 dtype，不额外容忍未 cast 的 dtype

## Related Documents

- [[tiling_patterns]]
- [[regbase_operator_patterns]]
- [[../api/index]]
- [[../dev-experience/tiling_review_notes]]
