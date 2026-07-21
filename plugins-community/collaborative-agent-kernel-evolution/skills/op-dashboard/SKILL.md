---
name: op-dashboard
description: "Generate self-contained interactive HTML dashboard (4 tabs: algo flow, UB tiling, precision, performance) from an AscendC operator output directory. Use when asked to visualize or report operator results. Works for both precision-pass and precision-fail states."
---

# op-dashboard：AscendC 算子结果汇报看板

## 一句话说明

给定算子输出目录，**两阶段**生成自包含 HTML 看板，包含：计算逻辑流图 + UB 内存/Tiling + 精度分析 + 性能报告。

**编译通过即可生成**，无需等待精度全通过 — 精度未通过时，看板会自动显示 FAIL Case 诊断横幅和性能警告注释。

---

## 标准生成流程（两阶段）

### 阶段一：提取数据（脚本执行）

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/gen_dashboard.py \
    --op-dir output/<op_name>/
```

脚本完成：数据提取 → `panels/*/data.json` → `panels/*/analysis.md`（客观数据表）→ HTML 初步嵌入

> 输出默认为 `output/<op_name>/dashboard.html`（与算子数据同目录），可用 `--output` 显式指定。

### 阶段二：Claude 分析 + 可视化（必须执行）

⚠ **脚本只提取客观数据，不生成分析文字和可视化 HTML 片段。**
当 analysis.md 中含 `NEEDS_CLAUDE_ANALYSIS` 标记时，Claude **必须**按
`skills/op-dashboard/subskills/analyze_panels.md` 完成全部任务：

**2a — 写三个可视化 HTML 片段**（供 gen_dashboard.py 拼装）：

| 输出文件 | 内容 | 规范 |
|----------|------|------|
| `panels/algo/flow.html` | 计算流程图（AIC/AIV 分支 + 数据流） | `panels/algo/SPEC.md` |
| `panels/algo/steps.html` | 算法步骤列表（Pass 注释 + 关键 API） | `panels/algo/SPEC.md` |
| `panels/memory/ub_viz.html` | UB 分配饼图（SVG donut + 图例） | `panels/memory/SPEC.md` |

风格参照对应 `examples/` 目录下的示例文件（只看布局和 class 命名，内容必须反映真实内核逻辑）。

**2b — 写三个面板的文字分析**：

| 输出文件 | 内容 |
|----------|------|
| `panels/memory/analysis.md` | 切分方案 + 尾块处理 + 流水线设计 + UB 空间利用 + 负载均衡 |
| `panels/precision/analysis.md` | 总体结论(PASS/FAIL) + 误差类型 + 分布规律 + 来源推断 + 风险评估 |
| `panels/perf/analysis.md` | 基准说明 + 性能规律 + 瓶颈推断 + 优化建议 |

**2c — 重新生成并验证**：

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/gen_dashboard.py \
    --op-dir output/<op_name>/

python3 ${CLAUDE_SKILL_DIR}/scripts/check_dashboard.py output/<op_name>/dashboard.html
```

目标：`✅ 质量检测通过`（0 FAIL）。

---

## 架构说明：gen_dashboard.py 是纯拼装器

```
┌─────────────────────────────────────┐
│  Claude 读 kernel.cpp / data.json   │
│  → 写 flow.html / steps.html        │   可视化片段
│  → 写 ub_viz.html                   │
│  → 写 analysis.md × 3               │   文字分析
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  gen_dashboard.py（纯拼装器）        │
│  load_panels() 读所有 HTML 片段      │
│  → 嵌入 const D = {...}             │
│  → 输出自包含 dashboard.html         │
└─────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  check_dashboard.py（质量验证）      │
│  检查 flow_html / steps_html /      │
│  ub_viz_html + analysis 章节完整性  │
└─────────────────────────────────────┘
```

---

## 兼容的数据源（gen_dashboard.py 自动发现）

| 数据 | 自动发现的文件 |
|------|-------------|
| **精度结果** | `evaluation_results.json` / `results_precision.json` |
| **性能 profiling** | `profiling/multi_case_report.csv` / `profiling/op_summary_<case>.csv` |
| **算子描述** | `*_op_desc.json` |
| **UB buffers** | `**/op_kernel/*_custom.cpp`（TQue/TBuf 声明） |
| **Tiling 常量** | `**/op_host/*_custom.cpp`（constexpr/const/局部变量） |
| **Test cases** | `test_cases.py`（shape 参数自动解析，支持 mat1/mat2 及 MoE 等自定义字段） |
| **可视化片段** | `panels/algo/flow.html`, `panels/algo/steps.html`, `panels/memory/ub_viz.html` |
| **文字分析** | `panels/*/analysis.md`（md→HTML 自动转换） |

---

## 四个面板说明

| Tab | 内容 | 谁写 |
|-----|------|------|
| **计算逻辑** | 流程图 + 步骤列表 | **Claude**（`flow.html` + `steps.html`） |
| **UB 内存** | buffer 饼图 + 条形图 + tiling 常量 + 文字分析 | 脚本提取数据，**Claude** 写 `ub_viz.html` + `analysis.md` |
| **精度分析** | 通过率 + 误差表 + 文字分析 | 脚本提取数据，**Claude** 写 `analysis.md` |
| **性能报告** | speedup 图表 + profiling 明细 + 文字分析 | 脚本提取数据，**Claude** 写 `analysis.md` |

---

## 目录结构

```
output/<op_name>/
├── evaluation_results.json       ← 或 results_precision.json
├── profiling/                    ← 路径不固定，脚本全局 rglob 搜索
│   └── op_summary_<case>.csv     ← 可在任意子目录，gen_dashboard 自动发现
├── panels/
│   ├── algo/
│   │   ├── data.json             ← 脚本提取（pass_comments, compute_apis）
│   │   ├── flow.html             ← Claude 写（计算流程图 HTML 片段）
│   │   └── steps.html            ← Claude 写（算法步骤 HTML 片段）
│   ├── memory/
│   │   ├── data.json             ← 脚本提取（ub_buffers, tiling_consts）
│   │   ├── ub_viz.html           ← Claude 写（UB 饼图 SVG HTML 片段）
│   │   └── analysis.md           ← Claude 写（Tiling 策略文字分析）
│   ├── precision/
│   │   ├── data.json             ← 脚本提取
│   │   └── analysis.md           ← Claude 写
│   └── perf/
│       ├── data.json             ← 脚本提取
│       └── analysis.md           ← Claude 写
└── dashboard.html                ← 最终产物（gen_dashboard.py 拼装）
```

---

## 注意事项

- HTML 完全自包含（无 CDN 依赖），可离线打开、直接发送
- 自动适配亮色/暗色主题
- `flow.html` / `steps.html` / `ub_viz.html` 不得包含 `<script>` 或外部 URL
- UB buffer 大小为近似值（TPipe 实际分配有对齐开销）
- TQue 声明格式支持 `TQue<..., BUFFER_NUM>` 和 `TQue<..., 1>` 两种
- op_host 常量支持 `constexpr`、`const` 和 TilingFunc 内局部变量
