# Panel: 精度分析（Tab 3）— 设计规范

## 定位与原则

**展示什么**：多 Case 精度 PASS/FAIL 汇总 + 关键误差指标 + 跨 Case 对比图。
**不展示什么**：时延、加速比、UB 内存、算法步骤。

核心设计理念：工程师能快速看到哪些 Shape 精度有问题，及具体误差大小 vs 阈值。

---

## 数据来源

| 数据 | 来源文件 |
|------|---------|
| 多 Case 精度 | `precision_results.json`（v2 schema，含 ratios per case） |
| 单 Case fallback | `evaluation_results.json` 中的 `correctness_message` |
| Case 列表 | `test_cases.csv` / `multi_case_report.csv` |

### precision_results.json 关键字段

```json
{
  "cases": [{
    "id": 0,
    "params": {"var0_shape": "[16, 4096]"},
    "forward": {
      "output": {
        "passed": true,
        "ratios": {
          "max_re": 0.23, "mean_re": 0.08, "rmse": 0.15, "svec": 0.99
        },
        "ans_vs_golden": {
          "ae_max": 0.0001, "re_max": 0.23, "mismatch_rate": 0.0
        }
      }
    }
  }]
}
```

### 精度阈值参考（Ascend 910B2 典型值）

| 指标 | 绿色（好） | 橙色（注意） | 红色（超标） |
|------|-----------|-------------|-------------|
| max_re ratio | ≤ 0.5 | 0.5 – 10.0 | > 10.0 |
| mean_re ratio | ≤ 0.5 | 0.5 – 2.0 | > 2.0 |
| RMSE ratio | ≤ 0.5 | 0.5 – 2.0 | > 2.0 |

---

## 视觉设计规则

1. **Cases 总览表**：点击行选中，高亮显示，更新详细指标卡片。
2. **PASS/FAIL 徽章**：绿色 `✓ PASS` / 红色 `✗ FAIL`，不使用其他颜色表达状态。
3. **指标卡片（Metric Cards）**：
   - 大字显示数值（mono 字体）
   - 颜色根据 ratio/threshold 动态着色（绿/橙/红）
   - 底部细线进度条（bar 宽度 = min(ratio/limit, 1) × 100%）
   - 显示阈值 `≤ X` 参考
4. **跨 Case 对比 SVG 条形图**：max_re 和 mean_re 两组并排柱，X 轴为 shape 标签。
5. **数字格式化**：小于 0.001 时用科学计数法 `.toExponential(2)`，否则 `.toFixed(4)`。

---

## 示例参考函数
> **注意：以下为风格示例，仅供 Claude 理解渲染模式，不是生产代码。**

```python
# ── 示例：render_precision_tab() ── 风格参考，非生产代码 ───────────────────
def render_precision_tab_example(cases: list) -> str:
    """
    渲染 Tab 3 精度分析 HTML 片段。

    参数:
        cases: [{id, shape, passed, precision:{passed, max_re, mean_re, rmse,
                 ae_max, re_max, mismatch_rate}}]
    """
    # ① 总览表头
    thead = """<thead><tr>
      <th>Case</th><th>Shape</th><th>精度</th>
      <th>max_re ratio</th><th>mean_re ratio</th><th>RMSE ratio</th>
    </tr></thead>"""

    # ② 总览表体
    tbody_rows = ""
    for c in cases:
        p = c.get("precision", {})
        badge = (
            '<span class="badge bp">✓ PASS</span>' if p.get("passed")
            else '<span class="badge bf">✗ FAIL</span>'
        )
        def fmt(v):
            if v is None: return "—"
            return f"{v:.2e}" if abs(v) < 0.001 else f"{v:.4f}"
        tbody_rows += f"""<tr onclick="selectCase({c['id']})">
          <td class="val" style="color:var(--tx2)">Case {c['id']}</td>
          <td class="val" style="font-size:12px">{c['shape']}</td>
          <td>{badge}</td>
          <td class="val">{fmt(p.get('max_re'))}</td>
          <td class="val">{fmt(p.get('mean_re'))}</td>
          <td class="val">{fmt(p.get('rmse'))}</td>
        </tr>"""

    # ③ 详细指标卡片（4个指标）
    # 每个 metric card 结构：
    #   .mc-lbl   标签（全大写）
    #   .mc-val   数值（大字，颜色根据阈值）
    #   .mc-lim   阈值说明
    #   .mc-bar → .mc-fill  进度条
    metric_card_template = """
    <!-- 示例单个指标卡片结构 -->
    <div class="mc">
      <div class="mc-lbl">{label}</div>
      <div class="mc-val {cls}">{value}</div>
      <div class="mc-lim">阈值 ≤ {limit}</div>
      <div class="mc-bar">
        <div class="mc-fill" style="width:{pct}%;background:{color}"></div>
      </div>
    </div>"""
    # 实际 JS 中由 refreshPrecDetail() 动态填充，Python 仅展示结构

    # ④ SVG 条形图说明（实际由 JS svgBar() 渲染）
    svg_placeholder = '<svg class="svg-chart" id="prec-svg" height="200"></svg>'

    return f"""
    <div class="card">
      <div class="ct"><span class="ico">📊</span>所有 Cases 精度总览</div>
      <table class="poc">
        {thead}
        <tbody>{tbody_rows}</tbody>
      </table>
    </div>
    <div class="card" id="prec-detail-card">
      <div class="ct"><span class="ico">🔍</span>详细指标
        <span id="prec-sel-label" class="badge bi" style="margin-left:8px"></span>
      </div>
      <!-- 指标卡片网格，由 JS refreshPrecDetail() 填充 -->
      <div class="mg" id="prec-metrics"></div>
      <!-- 跨 Case 对比条形图 -->
      <div>
        <div class="chart-title">各 Case 精度指标对比（max_re / mean_re）</div>
        {svg_placeholder}
      </div>
    </div>"""
# ── 结束示例 ─────────────────────────────────────────────────────────────────
```

---

## Claude 生成该 Tab 时的检查清单

- [ ] 所有 Case 均显示在总览表中（包括 PASS 和 FAIL）
- [ ] 点击表格行能切换详情卡片（需 JS `selectedCase` 状态）
- [ ] 指标卡片颜色与阈值对应（不硬编码颜色，根据 ratio 动态判断）
- [ ] SVG 条形图 X 轴标签为 shape 字符串（如 `[16, 4096]`），超长截断
- [ ] 无时延、无加速比数字
- [ ] `mismatch_rate = 0.0` 时显示 `0.0`（不显示 `—`）
