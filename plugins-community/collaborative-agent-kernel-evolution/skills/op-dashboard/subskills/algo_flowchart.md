# Subskill: 算子计算流图生成（algo_flowchart）

## 功能说明

本 subskill 指导 Claude 阅读算子源码，生成 `algo_flow.json`。
`gen_dashboard.py` 通过 `--algo-flow algo_flow.json` 读取此文件渲染 Tab 1。

**适用范围**：所有 AscendC 算子（Vector、Cube、混合）。
**不适用**：无 kernel 源码的纯参考实现。

---

## 触发时机

当用户要求生成看板且：
- 存在 `*_custom.cpp` 但规则提取效果差（算子非标准 softmax/layernorm 等）
- 算子为 Cube 类（Matmul、GEMV、Attention 等）
- 用户指定 `--algo-flow` 路径时手动生成

---

## 生成步骤

**Step 1** — 阅读以下文件：
```
output/<op_name>/<Op>_op_desc.json           # 算子描述、输入输出形状
output/<op_name>/<Op>Custom/op_kernel/<op>_custom.cpp  # AscendC kernel
output/<op_name>/<Op>_dsl.py                # DSL（可选，有助于理解算法）
```

**Step 2** — 按以下规则提取计算节点：

### 提取规则

1. **只提取 Compute() 方法中的 AscendC API 调用**（排除 CopyIn/CopyOut）
2. **不包含 DataCopy**（内存搬运不属于计算流图）
3. **AscendC Vector API** → 归类为 `vector` 单元
4. **AscendC Cube API**（Matmul、BatchMatMul）→ 归类为 `cube` 单元
5. **保持执行顺序**（与源码顺序一致）
6. **公式使用数学符号**（Unicode：−、×、Σ、√、ε、μ、σ）

### API → 单元映射表

| AscendC API | 单元 | 数学含义 |
|-------------|------|---------|
| ReduceMax | vector | m = max(x, dim=−1) |
| ReduceMin | vector | m = min(x, dim=−1) |
| ReduceSum | vector | s = Σⱼ xⱼ |
| Adds | vector | y = x + scalar |
| Sub / Muls | vector | 元素级减法/乘标量 |
| Mul / Div | vector | 元素级乘法/除法 |
| Exp | vector | y = exp(x) |
| Log | vector | y = ln(x) |
| Sqrt / Rsqrt | vector | y = √x / y = 1/√x |
| Cast | vector | 类型转换 |
| Abs / Max / Min | vector | 元素级 |
| Sigmoid / Tanh | vector | 激活函数 |
| Matmul | cube | C = A × B |
| BatchMatMul | cube | 批量矩阵乘 |

**Shape 模板规则**：

| 输入维度 | 模板变量 |
|---------|---------|
| 1D `[N]` | `{N}` |
| 2D `[N, C]` | `{N}, {C}` |
| 3D `[B, N, C]` | `{B}, {N}, {C}` |
| 4D `[B, H, N, C]` | `{B}, {H}, {N}, {C}` |

输出 shape 模板：根据 API 语义推断（如 ReduceMax 沿最后维 → `{N}, 1`）。

**Step 3** — 输出 `algo_flow.json`：

```json
{
  "op_name": "<OpName>",
  "description": "<一句话算法描述>",
  "shape_vars": ["N", "C"],
  "units": [
    {
      "id": "vector",
      "label": "Vector Core  Compute",
      "bg": "#f5f0ff",
      "accent": "#8250df",
      "nodes": [
        {
          "api": "<AscendC函数名>",
          "formula": "<数学公式>",
          "in_tmpl": "{N}, {C}",
          "out_tmpl": "{N}, {C}"
        }
      ]
    }
  ]
}
```

若有 Cube 单元，追加：
```json
{
  "id": "cube",
  "label": "Cube Core  MAC",
  "bg": "#fffbf0",
  "accent": "#bf8700",
  "nodes": [...]
}
```

**Step 4** — 调用 gen_dashboard.py：
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/gen_dashboard.py \
    --op-dir output/<op_name>/ \
    --algo-flow output/<op_name>/algo_flow.json \
    --output output/<op_name>/dashboard.html
```

---

## 示例：Softmax algo_flow.json
> 见 `examples/algo_flow_softmax.json`

---

## 注意事项

- `shape_cases` 字段**不需要填写**（gen_dashboard.py 从 test_cases.csv 自动注入）
- `inp_name` / `out_name` / `inp_tmpl` / `out_tmpl` / `inp_dtype` / `out_dtype`
  **不需要在 algo_flow.json 中提供**（gen_dashboard.py 从 op_desc.json 自动读取）
- 若一个 Pass 对应多个 API（如先 Sub 再 Exp），每个 API 单独一个 node
- Cube 和 Vector 混合算子：两个 unit 按执行顺序排列（Vector → Cube → Vector 均可）
