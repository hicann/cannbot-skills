# Panel: 性能报告（Tab 4）— 设计规范

## 定位与原则

**展示什么**：各 Case 的 msprof 硬件计时加速比 + 时延对比图 + 数据源说明。
**不展示什么**：精度指标、UB 内存、算法步骤、CPU 计时（torch.npu.Event 与 msprof 不可混用）。

核心设计理念：用最大字号展示加速比，让评审者一眼得出结论；时延柱状图给出 Ref vs Custom 直观对比。

---

## 数据来源

优先级（从高到低）：

1. `profiling/multi_case_report.csv` — 多 Case msprof 硬件计时（最准确）
2. `profiling/report.txt` — 单 Case msprof 计时（msprof_fair_bench.py 输出）
3. `evaluation_results.json` — `base_time_ms` / `gen_time_ms`（torch.npu.Event，仅 fallback）

### multi_case_report.csv 字段

```
case_id,var0_shape,passed,ref_time_us,custom_time_us,speedup
0,"[16, 4096]",True,7.056,4.520,1.561
```

### report.txt 字段（msprof_fair_bench.py 输出）

```
Reference time: 7.056 us
Custom time: 4.520 us
Speedup: 1.56x
```

---

## 视觉设计规则

1. **加速比卡片（sp-card）**：
   - 大字 `X.XXx`（28px，mono，800 weight）
   - 颜色编码：≥ 2x → `--gn`（绿），≥ 1.2x → `--or`（橙），< 1.2x → `--rd`（红）
   - 子标题显示 Case id + shape
   - **每次切换到此 Tab 必须先清空容器**（防止重复累积）
2. **时延对比 SVG 条形图**：
   - 两组并排：Reference（灰 `#8b949e`）/ Custom（绿 `#1a7f37`）
   - Y 轴单位 μs，X 轴为 shape 标签
   - 柱子内/上显示具体数值
3. **数据来源标注**：
   - msprof → "msprof 硬件级别（op_summary 区间统计，排除 warmup）"
   - npu_event → "torch.npu.Event（含 inter-kernel gap，比 msprof 偏大）"
4. **性能解读文字**：
   - ≥ 2x → "✓ 平均加速比达到 2x+ 优秀目标"
   - ≥ 1.2x → "△ 加速有效但未达 2x，可进一步优化"
   - < 1.2x → "✗ 加速偏低，建议检查 GM-UB 搬运瓶颈"

---

## 示例参考函数
> **注意：以下为风格示例，仅供 Claude 理解渲染模式，不是生产代码。**

```python
# ── 示例：render_perf_tab() ── 风格参考，非生产代码 ────────────────────────
def render_perf_tab_example(cases: list, avg_speedup: float) -> str:
    """
    渲染 Tab 4 性能报告 HTML 片段。

    参数:
        cases:        [{id, shape, passed, performance:{ref_time_us, custom_time_us,
                        speedup, source}}]
        avg_speedup:  平均加速比（float 或 None）

    关键约束：
        - 调用方（JS buildPerf）必须先清空 #perf-cards / #perf-svg / #perf-note
        - 加速比颜色由 spCls(sp) 决定，不可硬编码
    """
    perf_cases = [c for c in cases if c.get("performance", {}).get("speedup")]
    if not perf_cases:
        return """
        <div class="card">
          <span style="color:var(--tx3)">（无性能数据，请先运行 msprof 评测）</span>
        </div>"""

    # ① 加速比卡片（JS 动态生成，此处展示结构）
    cards_html = ""
    for c in perf_cases:
        sp = c["performance"]["speedup"]
        # spCls: sp>=2 → sp-good, sp>=1.2 → sp-ok, else sp-slow
        cls = "sp-good" if sp >= 2 else ("sp-ok" if sp >= 1.2 else "sp-slow")
        cards_html += f"""
        <div class="sp-card">
          <div class="sp-val {cls}">{sp:.2f}<span style="font-size:16px">x</span></div>
          <div class="sp-unit">Speedup</div>
          <div class="sp-shape">Case {c['id']}  {c['shape']}</div>
        </div>"""

    # ② 时延对比 SVG 条形图（由 svgBar() 渲染，此处仅占位）
    svg_html = '<svg class="svg-chart" id="perf-svg" height="260"></svg>'

    # ③ 性能解读
    min_sp = min(c["performance"]["speedup"] for c in perf_cases)
    max_sp = max(c["performance"]["speedup"] for c in perf_cases)
    src    = perf_cases[0]["performance"].get("source", "msprof")
    src_note = (
        "msprof 硬件级别（op_summary 区间统计，排除 warmup）" if src == "msprof"
        else "torch.npu.Event（含 inter-kernel gap，比 msprof 偏大）"
    )
    if avg_speedup and avg_speedup >= 2:
        conclusion = f"✓ 平均加速比达到 <b>{avg_speedup:.2f}x</b>，达成 2x+ 优秀目标。"
    elif avg_speedup and avg_speedup >= 1.2:
        conclusion = f"△ 加速有效（{avg_speedup:.2f}x）但未达 2x，可通过双缓冲或向量化 API 进一步优化。"
    else:
        conclusion = "✗ 加速偏低，建议检查 GM-UB 搬运瓶颈或 SCALAR 指令过多。"

    note_html = f"""
    <div class="pnote">
      {len(perf_cases)} 个 Shape 评测完成。
      加速比范围 <b>{min_sp:.2f}x — {max_sp:.2f}x</b>，
      平均 <b>{avg_speedup:.2f}x</b>。<br>
      计时来源：<b>{src_note}</b>。<br>
      {conclusion}
    </div>"""

    return f"""
    <div class="card">
      <div class="ct"><span class="ico">⚡</span>加速比总览</div>
      <div class="perf-summary" id="perf-cards">{cards_html}</div>
    </div>
    <div class="card">
      <div class="ct"><span class="ico">📈</span>时延对比（msprof 硬件计时，单位 μs）</div>
      <div class="chart-wrap">{svg_html}</div>
    </div>
    <div class="card">
      <div class="ct"><span class="ico">💡</span>性能解读</div>
      <div id="perf-note">{note_html}</div>
    </div>"""
# ── 结束示例 ─────────────────────────────────────────────────────────────────
```

---

## Claude 生成该 Tab 时的检查清单

- [ ] `buildPerf()` 开头清空三个容器（防重复渲染）
- [ ] 加速比颜色由阈值动态判断（≥2x绿，≥1.2x橙，<1.2x红）
- [ ] 数据来源标注（msprof / npu_event）
- [ ] 无精度指标（无 max_re / rmse 等）
- [ ] SVG 图 X 轴标签超长时截断（> 12 字符 → 前 11 + `…`）
- [ ] 无性能数据时显示提示文字，不留空白
