# Panel: 内存 & Tiling（Tab 2）— 设计规范

## 定位与原则

**展示什么**：UB 内存分配可视化 + Buffer 详情表 + Tiling 策略。
**不展示什么**：性能数字、精度指标、时延。

核心设计理念：让工程师直观看到 `{ub_total_kb}` KB UB（默认 910B2=192KB，从 design_tokens.json chip.ub_kb 读取）里有什么、每个 buffer 占多大、Tiling 怎么切。

---

## 数据来源

自动从 `*_custom.cpp` 正则提取：

| 字段 | 来源 |
|------|------|
| `BUFFER_NUM` | `BUFFER_NUM = 2` |
| TQue buffers | `AscendC::TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;` |
| TBuf buffers | `AscendC::TBuf<TPosition::VECCALC> calcBuf;` |
| tiling_params | `tiling_data->fieldName` 引用 |
| tile_length | 取 input_shape 最后一维 |
| dtype_bytes | 从 input dtype 推导 |

Buffer 大小计算（近似值）：
```
TQue:  size_KB = BUFFER_NUM × tileLength × dtype_bytes / 1024
TBuf:  size_KB = 1          × tileLength × dtype_bytes / 1024
```
> 注意：TPipe 实际分配有对齐开销，此为近似值。

---

## 视觉设计规则

1. **UB 内存条**：线性 0–`{ub_total_kb}`KB 横向色块条，每段对应一个 buffer，颜色编码：
   - VECIN → `#0969da`（蓝）
   - VECOUT → `#1a7f37`（绿）
   - VECCALC → `#cf222e`（红/橙）
   - L1 → `#8250df`（紫）
2. **刻度尺**：0 / 64 / 128 / `{ub_total_kb}` KB 刻度（当 ub_total_kb=192 时为 0/64/128/192；256KB 芯片时刻度相应扩展）。
3. **图例**：色块 + buffer 名 + 类型 + 大小。
4. **Buffer 详情表**：6 列（缓冲区 / 类型 / 位置 / 倍数 / 单份 / 合计）。
5. **Tiling 策略表**：总行数 / 列数(tileLength) / Tile内存 / AIC数 / Tiling参数。

---

## 示例参考函数
> **注意：以下为风格示例，仅供 Claude 理解渲染模式，不是生产代码。**

```python
# ── 示例：render_memory_tab() ── 风格参考，非生产代码 ──────────────────────
def render_memory_tab_example(ub_buffers: list, ub_used_kb: float,
                               ub_total_kb: float, tiling: dict, chip: dict) -> str:
    """
    渲染 Tab 2 内存 & Tiling HTML 片段。

    参数:
        ub_buffers:  [{name, kind, position, multiplier, size_kb, color, label}]
        ub_used_kb:  已用 KB 合计
        ub_total_kb: float,  # from design_tokens.json chip.ub_kb (e.g. 192 for 910B2, 256 for 910/910A)
        tiling:      {tile_length, tile_size_kb, dtype_bytes, tiling_params}
        chip:        {name, ub_kb, aic}
    """
    # ① UB 内存条：每个 buffer 按 size_kb / total 计算宽度百分比
    bar_segments = ""
    for buf in ub_buffers:
        pct = buf["size_kb"] / ub_total_kb * 100
        label = buf["name"][:7] + "…" if len(buf["name"]) > 9 else buf["name"]
        text  = label if pct > 5 else ""
        bar_segments += (
            f'<div class="ub-seg" style="width:{pct:.2f}%;background:{buf["color"]}" '
            f'title="{buf["name"]}  {buf["size_kb"]:.1f}KB  ({pct:.1f}%)">'
            f'{text}</div>'
        )
    free_pct = (ub_total_kb - ub_used_kb) / ub_total_kb * 100
    bar_segments += (
        f'<div class="ub-seg ub-free" style="width:{free_pct:.2f}%" '
        f'title="空闲  {ub_total_kb - ub_used_kb:.1f}KB">空闲</div>'
    )

    # ② Ruler 刻度（末端刻度始终包含 ub_total_kb，确保任何芯片 UB 均显示边界）
    _base_ticks = [k for k in [0, 64, 128, 192, 256] if k <= ub_total_kb]
    _tick_set   = sorted(set(_base_ticks) | {ub_total_kb})
    ruler_ticks = "".join(
        f'<div class="ub-tick" style="left:{kb/ub_total_kb*100}%">{kb}KB</div>'
        for kb in _tick_set
    )

    # ③ Buffer 详情表行
    table_rows = ""
    for buf in ub_buffers:
        single = buf["size_kb"] / buf["multiplier"]
        table_rows += f"""<tr>
          <td><span class="val">{buf['name']}</span></td>
          <td><span class="pipe-tag" style="background:{buf['color']}20;
              color:{buf['color']}">{buf['kind']}</span></td>
          <td><span class="pipe-tag" style="background:{buf['color']}15;
              color:{buf['color']}">{buf['position']}</span></td>
          <td class="val">{buf['multiplier']}</td>
          <td class="val">{single:.1f} KB</td>
          <td><b class="val">{buf['size_kb']:.1f} KB</b></td>
        </tr>"""

    # ④ Tiling 策略表行（列举典型行）
    # 实际实现中从 tiling / input_shapes 中取值
    tiling_rows = f"""
      <tr><td>tileLength</td>
          <td class="val">{tiling['tile_length']}</td>
          <td style="color:var(--tx3)">每次处理一行的元素数</td></tr>
      <tr><td>Tile 内存(单份)</td>
          <td class="val">{tiling['tile_size_kb']:.2f} KB</td>
          <td style="color:var(--tx3)">{tiling['tile_length']} × {tiling['dtype_bytes']} B</td></tr>
      <tr><td>硬件 AIC 数</td>
          <td class="val">{chip['aic']}</td>
          <td style="color:var(--tx3)">910B2 单片 AI Core 数</td></tr>"""

    return f"""
    <div class="card">
      <div class="ct"><span class="ico">📦</span>UB 内存分配图
        <span class="badge bi" style="margin-left:auto">
          {ub_used_kb:.1f} KB / {ub_total_kb:.0f} KB — {ub_used_kb/ub_total_kb*100:.1f}% 占用
        </span>
      </div>
      <div class="lbl">Unified Buffer — {ub_total_kb} KB 地址空间（单核视图）</div>
      <div class="ub-bar">{bar_segments}</div>
      <div class="ub-ruler">{ruler_ticks}</div>
    </div>
    <div class="card">
      <div class="ct"><span class="ico">🔪</span>Tiling 策略</div>
      <table class="dt">
        <thead><tr><th>参数</th><th>值</th><th>说明</th></tr></thead>
        <tbody>{tiling_rows}</tbody>
      </table>
    </div>"""
# ── 结束示例 ─────────────────────────────────────────────────────────────────
```

---

## Claude 生成该 Tab 时的检查清单

- [ ] UB 条宽度总计 ≤ 100%（包含空闲段）
- [ ] 颜色与位置类型一致（VECIN=蓝，VECOUT=绿，VECCALC=红）
- [ ] Buffer 大小为近似值，注明「TPipe 实际有对齐开销」
- [ ] Tiling 表展示 tileLength / Tile内存 / AIC数 / tiling_params
- [ ] 无精度/性能数字
