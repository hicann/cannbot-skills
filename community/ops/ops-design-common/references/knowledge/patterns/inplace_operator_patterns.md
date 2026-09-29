---
name: inplace_operator_patterns
description: Inplace 算子的 OpDef 同名规则、kernel 参数命名、冒烟测试与 TTK 用例的 inplace 知识，覆盖从接口定义到测试验证的完整链路。
title: Inplace Operator Patterns
purpose: 判定算子是否为 inplace，并按 OpDef Input/Output 同名规则生成正确的接口桩与测试。
read_when:
  - 算子名含 Inplace，或需求文档声明输出复用输入地址。
  - 需要为 inplace 算子生成 OpDef / kernel 签名 / proto.md / 冒烟测试。
  - inplace 算子的 TTK CSV 需要填写 output_inplace_indexes。
not_for:
  - 非 inplace 算子的常规开发流程。
  - 范式特定的 Tiling / Kernel 计算链设计（inplace 可与 EleWise / Broadcast 等范式组合，参见对应范式 patterns.md）。
keywords:
  - inplace
  - OpDef 同名
  - Output Input same name
  - var_out
  - output_inplace_indexes
  - 原地写回
next_reads:
  - regbase_operator_patterns.md
  - ../dev-experience/regbase_kernel_case_notes.md
  - ../../reference-ops/open_source_operator_table.md
depth: foundation
topic_type: pattern
type: knowledge_card
platform: 950-regbase
verified: true
patterns: [inplace, opdef, kernel_entry, smoke_test, ttk_csv]
---

# Inplace Operator Patterns

Inplace 算子的核心特征是**输出复用输入的 GM 地址**（原地写回），而非分配独立输出内存。CANN 框架的 inplace 机制通过 **OpDef 中 `Output` 与 `Input` 同名**触发——框架据此自动把同一 GM 地址绑定给该 Input 和 Output，运行时 kernel 收到的两个参数指向同一块 device 内存。

本卡片覆盖从判定到测试的完整链路。inplace 可与 EleWise、Broadcast 等任意范式组合，不改变范式本身的 Tiling / Kernel 设计，只影响接口定义与测试方式。

---

## 1. 判定条件

满足以下**任一**条件即判定为 inplace 算子：

1. **算子名含 `Inplace`**（如 `InplaceApplyAdaMax`、`InplaceIndexAdd`）——最直接的命名信号。
2. **需求文档明确声明**「输出复用输入地址」「原地写回」「inplace」等语义。
3. **`spec.yaml`** 某 output 的 `aliasing` 字段值为 `inplace_with(<input_name>)`（显式声明；注意 `generate_spec.py` 默认生成 `aliasing: none`，此值需 spec-generator 手动改写，不一定存在）。

> 判定依据不唯一依赖 `aliasing` 字段——算子名和需求文档是更可靠的信号。

---

## 2. OpDef 规则：Input/Output 同名

inplace 输出的 `Output` 名称**必须等于**所 alias 的 `Input` 名称。框架据此同名关系绑定同一 GM 地址。

```cpp
// ✅ 正确：inplace 算子，Output 与 Input 同名
this->Input("var").ParamType(REQUIRED).DataType({...}).Format({...});
this->Output("var").ParamType(REQUIRED).DataType({...}).Format({...});  // 同名 "var"

// ❌ 错误：Output 改名为 "var_out"，框架视为普通 out-of-place 输出
//        分配独立内存，inplace 语义丢失
this->Output("var_out").ParamType(REQUIRED).DataType({...}).Format({...});
```

**禁止**把 inplace 输出命名为 `<input_name>_out`、`<input_name>Out` 等变体。

### 真实算子验证

| 算子 | OpDef Input | OpDef Output | 来源 |
|------|-------------|--------------|------|
| `MoeInplaceIndexAdd` | `Input("var")` | `Output("var")` | `ops-transformer/moe/3rd/moe_inplace_index_add` |
| `QuantGroupedMatmulInplaceAdd` | `Input("y")` | `Output("y")` | `ops-transformer/gmm/quant_grouped_matmul_inplace_add` |

---

## 3. Kernel 入口参数命名

OpDef 中 Input/Output 同名后，kernel 入口函数的 C++ 参数列表会出现同名参数——C++ 不允许同名参数，因此 **inplace Output 对应的 kernel 参数在 Input 参数名后加 `_out` 后缀**以区分。

```
OpDef:  Input("var")  +  Output("var")       // 同名 = inplace
Kernel: GM_ADDR var, ..., GM_ADDR var_out     // Output 参数加 _out 后缀避 C++ 冲突
```

- **`_out` 后缀仅是 C++ 命名层面的避让**，不改变 inplace 语义：运行时框架仍把同一 GM 地址传给 `var` 和 `var_out`。
- **非 inplace 的 Output 参数名**直接用 OpDef 中的 Output 名，不加后缀。
- 也有真实算子采用在 Input 侧加 `In` 后缀的策略（如 `yIn` + `y`），效果相同——`_out` 后缀是更常见的约定。

### 真实算子验证

| 算子 | kernel 入口参数 | 策略 |
|------|----------------|------|
| `moe_inplace_index_add` | `GM_ADDR var, ..., GM_ADDR var_out` | Output 加 `_out` |
| `quant_grouped_matmul_inplace_add` | `GM_ADDR yIn, ..., GM_ADDR y` | Input 加 `In` |

```cpp
// moe_inplace_index_add — OpDef Input("var") + Output("var") → kernel 参数 var + var_out
extern "C" __global__ __aicore__ void moe_inplace_index_add(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR alpha,
    GM_ADDR var_out, GM_ADDR workspace, GM_ADDR tiling)

// quant_grouped_matmul_inplace_add — OpDef Input("y") + Output("y") → kernel 参数 yIn + y
__global__ __aicore__ void quant_grouped_matmul_inplace_add(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR scale2, GM_ADDR groupList,
    GM_ADDR yIn, GM_ADDR scale1, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
```

---

## 4. Kernel 实现注意事项

inplace 算子的 `<name>` 与 `<name>_out` 运行时指向**同一块 GM 地址**：

1. **CopyOut 直接写回**：计算结果写回 `<name>_out` 的 GM 地址即可（原地写回），无需为输出单独分配 buffer。

2. **避免 alias 冲突**：计算链中不要出现「同一时刻既读 `<name>` 做 source、又写 `<name>_out` 做 destination」的指令——两者地址相同会产生 read-after-write hazard。中间结果须暂存 UB buffer，计算完成后再写回 GM。

3. **多输出 inplace**：一个算子可有多组 inplace 关系（如 `InplaceApplyAdaMax` 的 `var`/`m`/`v` 三个输入均原地更新），每组独立遵循同名 + `_out` 后缀规则。

4. **同步时序**：由于输入输出同地址，CopyIn 读取 `<name>` 与 CopyOut 写回 `<name>_out` 指向同一块 GM 内存。若 CopyOut（MTE3）与下一轮 CopyIn（MTE2）之间未正确同步，MTE2 可能在 MTE3 写回完成前读到旧数据。须在流水线轮次之间插入 `SetFlag<HardEvent::MTE3_MTE2>` / `WaitFlag<HardEvent::MTE3_MTE2>` 同步对，确保上一轮写回完成后再读取下一轮数据。同步策略选择详见 `regbase_sync_patterns.md`。

---

## 5. proto.md 生成规则

inplace 算子须遵循：

1. **§1 OpDef**：inplace 输出的 `Output` 名称与 Input 同名（无 `_out` 后缀）。
2. **§2 dtype 组合列表**：表头列名 = OpDef 中的 IO 名（inplace 输出列名与对应 Input 列名相同）。
3. **§3 kernel 函数签名**：inplace Output 参数加 `_out` 后缀以避 C++ 同名冲突。

---

## 6. GEIR 原型规则

### REG_OP（`_proto.h`）

GEIR 图模式原型中 inplace 的表现与 OpDef 完全对齐：

1. **`OUTPUT` 与 `INPUT` 同名**：inplace 输出的 `OUTPUT` 名称必须与所 alias 的 `INPUT` 名称相同。GE 框架据此识别 inplace 语义，绑定同一 GM 地址。

```cpp
// ✅ 正确：inplace 算子，OUTPUT 与 INPUT 同名
REG_OP(InplaceIndexAdd)
    .INPUT(var, TensorType({DT_FLOAT32, DT_FLOAT16, ...}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(updates, TensorType({DT_FLOAT32, DT_FLOAT16, ...}))
    .OUTPUT(var, TensorType({DT_FLOAT32, DT_FLOAT16, ...}))  // 同名 "var"
    .REQUIRED_ATTR(axis, Int)
    .OP_END_FACTORY_REG(InplaceIndexAdd)

// ❌ 错误：OUTPUT 改名为 "var_out"，GE 框架视为普通 out-of-place 输出
REG_OP(InplaceIndexAdd)
    .INPUT(var, ...)
    .OUTPUT(var_out, ...)  // 破坏 inplace 语义
    ...
```

2. **隐式 inplace（无 OUTPUT 声明）**：部分 foreach inplace 算子（如 `ForeachZeroInplace`、`ForeachAddListInplace`）完全不声明 `OUTPUT`，输入被原地修改。这是另一种 inplace 模式，不使用同名规则。

### 真实算子验证

| 算子 | REG_OP INPUT | REG_OP OUTPUT | 模式 |
|------|-------------|---------------|------|
| `InplaceIndexAdd` | `.INPUT(var)` | `.OUTPUT(var)` 同名 | 显式同名 |
| `InplaceMatmulAllReduceAddRmsNorm` | `.INPUT(residual)` | `.OUTPUT(residual)` 同名 | 显式同名 |
| `ForeachZeroInplace` | `.DYNAMIC_INPUT(x)` | 无 OUTPUT | 隐式 inplace |
| `ForeachAddListInplace` | `.DYNAMIC_INPUT(x1)` | 无 OUTPUT | 隐式 inplace |

### InferShape / InferDataType

inplace 只约束输入输出**共用同一 GM 地址**，不约束 shape/dtype 的推导关系——输出 shape/dtype 由算子本身的数学语义决定，据 `REQUIREMENTS.md` / `spec.yaml` 推导，与是否 inplace 无关。按 DESIGN.md §3（Shape 与 DataType 推导）的通用规则实现即可。

---

## Related Documents

- [[regbase_operator_patterns]]
- [[regbase_kernel_dataflow_patterns]]
- [[../dev-experience/regbase_kernel_case_notes]]
- [[../../reference-ops/open_source_operator_table]]
