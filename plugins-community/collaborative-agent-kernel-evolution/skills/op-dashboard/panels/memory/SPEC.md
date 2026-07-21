# Panel: 内存 & Tiling（Tab 2）— 规范 v4

---

## 职责分工（核心原则）

```
gen_dashboard.py  → 尽力解析 InitBuffer 表达式并计算 `ub_buffers[].size_kb`、`ub_used_kb`
                    但结果不保证准确，可能为 0 或估算值（见 `ub_estimated`）
                    ↓
              data.json（骨架 + 初步估算，Claude 仍需以源码推导为准）
                    ↓
Claude            → 直接读 op_kernel/*.cpp + op_host/*.cpp
                    理解语义，校正并确认每个 buffer 的实际字节数
                    → 生成 ub_viz.html（SVG donut + 图例，含正确 size_kb）
                    → 写 analysis.md（六项文字分析）
```

**为什么 Claude 直接读源码更优**：
- script 的 regex 只能处理能静态求值的简单表达式（`tileSize * 4`），对运行时变量、多级常量引用（`CHUNK_M = L1Shape::M / VEC_NUM`）容易出错
- Claude 能理解语义：看到 `pipe.InitBuffer(q, 1, tileSize * sizeof(float))` 且 `tileSize = TILE_SIZE = 4096`，直接推导 `4096 × 4 = 16 KB`
- 避免重复工作：script 做复杂解析 → 写 data.json → Claude 再读 data.json，等于做了两遍

---

## 数据来源（按优先级，路径兼容性：从算子根目录 rglob）

```
1. panels/memory/data.json          ← 脚本提取（骨架：buffer 名/position/color，tiling 常量名）
2. op_kernel/*_custom.cpp           ← Claude 直接读，推导 InitBuffer 实际字节数
3. op_host/*_custom.cpp             ← Claude 直接读，获取 TILE_SIZE 等常量定义
```

### data.json 结构（脚本生成，size_kb 为 0 或估算值，Claude 以源码为准）

```json
{
  "tiling_consts": {
    "L1_M": 128, "L1_N": 256, "L1_K": 256,
    "EPILOGUE_TILE_M": 64, "WORKSPACE_STAGES": 1
  },
  "ub_buffers": [
    {"name": "inQueueC",  "kind": "TQue", "position": "VECIN",   "color": "#0969da", "size_kb": 0},
    {"name": "outQueueD", "kind": "TQue", "position": "VECOUT",  "color": "#e36209", "size_kb": 0},
    {"name": "accBuf",    "kind": "TBuf", "position": "VECCALC", "color": "#1a7f37", "size_kb": 0}
  ],
  "ub_used_kb": 0,
  "ub_total_kb": 192.0
}
```

> `size_kb` 字段由 script 尽力解析，失败时为 0。**Claude 以读源码得到的值为准**，不信任 data.json 的 size_kb（除非 `ub_estimated: false` 且值 > 0）。

---

## ub_viz.html 生成协议

### 步骤

1. 读 `data.json` 获取 buffer 列表（名称、position、color）
2. 读 `op_kernel/*_custom.cpp`，找到每个 `pipe.InitBuffer(bufName, ...)` 调用
3. 读 `op_host/*_custom.cpp`，找到相关常量定义（`TILE_SIZE`、`L1_M` 等）
4. **自行推导**每个 buffer 的字节数：`size_bytes = count × size_expr（带单位求值）`
5. 以推导值生成 SVG donut + 图例

### 推导示例

```cpp
// op_host: static const uint32_t TILE_SIZE = 4096;
// op_kernel: pipe.InitBuffer(xInQueue, 1, tileSize * sizeof(float));
//            tileSize = tilingData.tileSize  （= TILE_SIZE = 4096）
// → size_bytes = 1 × 4096 × 4 = 16384 B = 16 KB
```

---

## 分析协议（analysis.md）

CC 读取源码后，必须回答六个问题，写入 `panels/memory/analysis.md`：

1. **切分方案**：各维度如何切分？tile 尺寸的约束来源（UB 容量/对齐/核数）？
2. **K 方向循环**：K 维度是否分块？若有，外层 K-loop 与内层 BlockMmad 如何配合？
3. **尾块处理**：哪些 case 有尾块？内核采用什么策略？
4. **流水线设计**：双缓冲几级？AIC↔AIV overlap？关键同步原语？
5. **UB 空间利用**：每个 buffer 的用途、生命周期、利用率；host 的 TILE_SIZE 计算是否正确？有无 buffer 可复用？
6. **负载均衡**：最差均衡 case；原因；改进建议？

### UB 空间利用分析要点（第 5 项细化）

回答这四个子问题：

- **实际字节数**：每个 InitBuffer 的实际分配是多少（TILE_SIZE 单位 × sizeof(dtype)）？
- **TILE_SIZE 正确性**：host 计算 TILE_SIZE 时假设的 buffer 数量与内核实际 buffer 数是否一致？若不一致，最优 TILE_SIZE 应为多少？
- **生命周期重叠**：哪些 buffer 不同时活跃（不同 Phase）？是否存在复用机会？
- **设计建议**：当前利用率，以及改为最优配置后预期利用率和 tileSize 提升幅度。

---

## 输出格式（analysis.md 文字分析）

```markdown
## Tiling 策略分析

### 切分方案
（2-4 句说明）

### K 方向循环
（有无分块；若有，说明结构）

### 尾块处理
（列出含尾块 case；说明处理策略）

### 流水线设计
（双缓冲级数；AIC↔AIV overlap；关键同步点）

### UB 空间利用
（见 ub_viz.html 可视化；回答四个子问题：实际字节数、TILE_SIZE 正确性、生命周期重叠、设计建议）

### 负载均衡诊断
（最差均衡 case；原因；建议）
```

---

## ub_viz.html 可视化规范

### 可用 CSS 类

| 类名 | 用途 |
|------|------|
| `ub-alloc` | 整体容器（flex 行：饼图左 + 图例右） |
| `ub-alloc-legend` | 右侧图例列表 |
| `ub-alloc-row` | 单个 buffer 图例行 |
| `ub-alloc-dot` | 颜色圆点 |
| `ub-alloc-label` | buffer 名称 |
| `ub-alloc-size` | 大小数字 |
| `ub-alloc-pct` | position 标签 |
| `ub-free-bar` | 空闲区域标注行 |

### SVG 饼图规范
- 纯 SVG，内联在 HTML 中，无 `<script>`
- viewBox="0 0 120 120"，圆心 (60,60)，stroke-width=20（r=44 → donut 效果）
- 每个 buffer 一段弧，颜色取自 `data.json` 中该 buffer 的 `color` 字段
- 空闲空间（`ub_total_kb - ub_used_kb`）用 `#e5e7eb` 浅灰色底圆表示
- 圆心显示利用率百分比（两行：`X.X%` + `UB`，`font-family="monospace"`）

### 角度与路径计算
- `角度 = size_kb / ub_total_kb × 360°`，从 12 点方向（-90°）顺时针累积
- SVG arc: `M x1,y1 A 44,44 0 {large-arc},1 x2,y2`（large-arc=1 当弧度>180°）
- 点坐标: `x = 60 + 44×cos(θ_rad)`, `y = 60 + 44×sin(θ_rad)`

### 规则
1. 禁止 `<script>`、外部 URL、外部 CDN
2. **颜色取 data.json 的 `color` 字段**（与 bar chart 一致），不按 position 统一着色（同 position 多 buffer 需可区分）

---

## 验收规则

- [ ] `analysis.md` 包含全部 6 个 `###` 小节
- [ ] UB 空间利用节回答了四个子问题
- [ ] `ub_viz.html` 存在且长度 > 300 字符
- [ ] `ub_viz.html` 包含 `<svg` 标签
- [ ] `ub_viz.html` 不含 `<script` 或外部 URL
- [ ] `ub_viz.html` 中的 size_kb 来自 Claude 读源码推导，非直接复制 data.json 的 0 值
