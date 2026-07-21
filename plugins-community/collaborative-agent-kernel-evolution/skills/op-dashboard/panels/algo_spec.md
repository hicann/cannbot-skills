# Panel: 算法图示（Tab 1）— 设计规范

## 定位与原则

**展示什么**：算子的**纯计算逻辑**，按硬件单元分组，体现数学语义。
**不展示什么**：DataCopy / 内存搬运 / 性能指标 / 百分比 / 带宽数字。

核心设计理念：让工程师一眼看懂算子「在做什么数学」，而不是「数据怎么搬」。

---

## 数据来源

优先读取 `algo_flow.json`（Claude 生成），缺失时回退到规则提取。
详见 `subskills/algo_flowchart.md` 了解如何生成 `algo_flow.json`。

### algo_flow.json Schema

```json
{
  "op_name": "Softmax",
  "description": "Row-wise softmax: normalize exp scores to probability distribution",
  "shape_vars": ["N", "C"],
  "units": [
    {
      "id": "vector",
      "label": "Vector Core  Compute",
      "bg": "#f5f0ff",
      "accent": "#8250df",
      "nodes": [
        {
          "api": "ReduceMax",
          "formula": "m = max(x, dim=−1)",
          "in_tmpl": "{N}, {C}",
          "out_tmpl": "{N}, 1"
        }
      ]
    }
  ]
}
```

### Unit id 规范

| id | 标签 | bg | accent | 用于 |
|----|------|----|--------|------|
| `vector` | Vector Core  Compute | `#f5f0ff` | `#8250df` | 向量计算 API |
| `cube`   | Cube Core  MAC       | `#fffbf0` | `#bf8700` | 矩阵乘 API |

**不包含** `mte2` / `mte3`（DataCopy 不展示在计算流图中）。

---

## 视觉设计规则

1. **Shape 联动选择器**：顶部 `<select>` 显示所有测试 Case 的 shape，切换时所有 `{N}`, `{C}` 占位符实时更新。
2. **IO 节点**：顶部蓝色框（输入），底部绿色框（输出），显示张量名 + shape 模板 + dtype。
3. **硬件单元组**：圆角卡片，浅色背景，左上角标注单元名（小字全大写）。
4. **API 节点**：白色小卡片，左侧 3px accent 色竖条，API 名加粗 accent 色，公式 mono 小字，shape 灰色极小字。
5. **节点连接**：单元内用 `→` 横向连接，单元之间用 `▼` 纵向箭头。
6. **底部步骤列表**：原始 Pass 注释编号列表，供开发者核对源码。

---

## 示例参考函数
> **注意：以下为风格示例，仅供 Claude 理解渲染模式，不是生产代码。**
> Claude 在生成 algo tab HTML 时应遵循此风格，但可根据算子特性调整布局。

```python
# ── 示例：render_algo_tab() ── 风格参考，非生产代码 ──────────────────────────
def render_algo_tab_example(op_graph: dict, cases: list) -> str:
    """
    将 algo_flow.json 数据渲染为 Tab 1 HTML 片段。

    参数:
        op_graph: build_op_graph() 或 algo_flow.json 的输出
        cases:    [{id, shape, ...}] 用于填充 shape selector
    返回:
        HTML string（嵌入到 <div id="tab-algo"> 中）
    """
    shape_vars  = op_graph.get("shape_vars", ["N", "C"])
    shape_cases = op_graph.get("shape_cases", [])
    units       = op_graph.get("units", [])
    inp_name    = op_graph.get("inp_name", "x")
    out_name    = op_graph.get("out_name", "y")
    inp_tmpl    = op_graph.get("inp_tmpl", "{N}, {C}")
    out_tmpl    = op_graph.get("out_tmpl", "{N}, {C}")
    inp_dtype   = op_graph.get("inp_dtype", "float32")
    out_dtype   = op_graph.get("out_dtype", "float32")

    # ① Shape selector 选项
    options_html = "\n".join(
        f'<option value="{i}">{c["label"]}</option>'
        for i, c in enumerate(shape_cases)
    )

    # ② IO 节点（上方输入）
    io_input_html = f"""
    <div style="display:flex;justify-content:center;width:100%">
      <div class="io-node">
        <div class="io-lbl">输入 Input</div>
        <div class="io-name">{inp_name}</div>
        <div class="io-shape"><span data-sv="{inp_tmpl}">{inp_tmpl}</span></div>
        <div class="io-dtype">{inp_dtype}</div>
      </div>
    </div>"""

    # ③ 硬件单元组（每个 unit 一个 .unit-group 块）
    units_html = ""
    for unit in units:
        nodes_html = ""
        for i, node in enumerate(unit["nodes"]):
            sep = '<div class="unit-sep">→</div>' if i > 0 else ""
            same_shape = node["in_tmpl"] == node["out_tmpl"]
            shape_part = (
                f'<span data-sv="{node["in_tmpl"]}">{node["in_tmpl"]}</span>'
                if same_shape else
                f'<span data-sv="{node["in_tmpl"]}">{node["in_tmpl"]}</span>'
                f' → <span data-sv="{node["out_tmpl"]}">{node["out_tmpl"]}</span>'
            )
            nodes_html += f"""
            {sep}
            <div class="api-node" style="border-left:3px solid {unit['accent']}">
              <div class="api-name" style="color:{unit['accent']}">{node['api']}</div>
              <div class="api-formula">{node['formula']}</div>
              <div class="api-shape">{shape_part}</div>
            </div>"""

        units_html += f"""
        <div class="flow-arrow">▼</div>
        <div class="unit-group" style="background:{unit['bg']};border:2px solid {unit['accent']}40">
          <div class="unit-hdr" style="color:{unit['accent']}">{unit['label']}</div>
          <div class="unit-nodes">{nodes_html}</div>
        </div>"""

    # ④ IO 节点（下方输出）
    io_output_html = f"""
    <div style="display:flex;justify-content:center;width:100%">
      <div class="io-node" style="border-color:var(--gn)">
        <div class="io-lbl">输出 Output</div>
        <div class="io-name" style="color:var(--gn)">{out_name}</div>
        <div class="io-shape"><span data-sv="{out_tmpl}">{out_tmpl}</span></div>
        <div class="io-dtype">{out_dtype}</div>
      </div>
    </div>"""

    return f"""
    <!-- shape selector bar -->
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:20px">
      <label style="font-size:12px;font-weight:600;color:var(--tx2)">测试 Shape：</label>
      <select id="shape-sel" onchange="updateShapeVars()"
              style="background:var(--sf);color:var(--tx);border:1px solid var(--bd);
                     border-radius:8px;padding:6px 12px;font-family:var(--mono)">
        {options_html}
      </select>
    </div>
    <!-- flow -->
    <div class="flow-col" id="algo-flow">
      {io_input_html}
      {units_html}
      <div class="flow-arrow">▼</div>
      {io_output_html}
    </div>"""
# ── 结束示例 ─────────────────────────────────────────────────────────────────
```

---

## Claude 生成该 Tab 时的检查清单

- [ ] 无 DataCopy 节点
- [ ] 无性能数字（无 μs / 加速比）
- [ ] API 名与 kernel 源码一致
- [ ] shape 使用模板变量，不写死数字
- [ ] 按 OP_RULES 正确排序（执行顺序，不是字母顺序）
- [ ] 多 unit 时各 unit 颜色区分（vector=紫，cube=金）
