# 子技能：分析面板 + 写可视化 HTML 片段

> 本子技能在 `gen_dashboard.py` 写完客观数据后由 Claude 执行。
> 任务分两部分：
> - **Part A**：写三个可视化 HTML 片段（流程图、步骤、UB 饼图）
> - **Part B**：写三个面板的文字分析（analysis.md）

---

## 触发条件

当 `panels/*/analysis.md` 中存在 `NEEDS_CLAUDE_ANALYSIS` 标记时，
Claude **必须**执行本子技能，不能留空占位直接生成 HTML。

---

## Part A：写可视化 HTML 片段

### A-Step 1：收集原始材料

并行读取：

```
{op_dir}/panels/algo/data.json          ← pass_comments, compute_apis, inputs, outputs
{op_dir}/panels/memory/data.json        ← ub_buffers, ub_used_kb, ub_total_kb
{op_dir}/**/op_kernel/*_custom.cpp      ← 内核实现（AIC/AIV 分支、BlockMmad、CrossCoreSetFlag）
{op_dir}/**/*_op_desc.json              ← 输入输出名称、shape、dtype
```

同时读取规范和示例：
```
skills/op-dashboard/panels/algo/SPEC.md
skills/op-dashboard/panels/algo/examples/flow.html    ← 布局和 class 命名参考
skills/op-dashboard/panels/algo/examples/steps.html   ← 布局和 class 命名参考
skills/op-dashboard/panels/memory/SPEC.md
skills/op-dashboard/panels/memory/examples/ub_viz.html ← SVG donut 格式参考
```

### A-Step 2：写 `panels/algo/flow.html`

按 `panels/algo/SPEC.md` 解析规则，自由发挥写真实流程图：

- 识别 AIC/AIV 分支（`if ASCEND_IS_AIC` / `else`），分别用 `data-core="AIC"` / `data-core="AIV"` 的 `algo-phase` 块
- 若 `KernelVector` 已定义但 entry 未调用 → `algo-phase[data-core="AIV"]` 加 `style="opacity:0.45"` 并加 `algo-gate` 警告说明
- 输入节点用 `algo-node input`，输出用 `algo-node output`，中间计算用 `algo-node compute`，workspace 用 `algo-node workspace`，同步点用 `algo-node sync`
- 节点间用 `algo-arrow` 连接
- **不得包含 `<script>` 或外部 URL（http/https）**

写入 `{op_dir}/panels/algo/flow.html`。

### A-Step 3：写 `panels/algo/steps.html`

从 `data.json` 的 `pass_comments` 和 `compute_apis` 提炼算法步骤：

- 每步用 `algo-step` 块，步骤编号用 `algo-step-num`，说明用 `algo-step-body`，关键 API 用 `algo-step-code`
- 步骤数量跟随真实 Pass/Phase 注释，不要随意增删
- **不得包含 `<script>` 或外部 URL**

写入 `{op_dir}/panels/algo/steps.html`。

### A-Step 4：写 `panels/memory/ub_viz.html`

从 `panels/memory/data.json` 的 `ub_buffers` / `ub_used_kb` / `ub_total_kb` 生成 SVG donut 饼图：

- `viewBox="0 0 120 120"` cx/cy=60，r_outer=54，r_inner=34（stroke-width=20 donut 风格）
- 角度公式：`size_kb / ub_total_kb × 360°`，从 -90° 顺时针累加
- 颜色优先使用 `ub_buffers[].color`；仅当 `color` 缺失、为空或不可用时，才按 `position` fallback：VECIN=#0969da, VECOUT=#e36209, VECCALC=#1a7f37；剩余空间=#e5e7eb
- 图例用 `.ub-alloc-legend`，每行显示 buffer 名 + 大小 + 百分比，并与 donut / bar chart 的 buffer 颜色保持一致
- **不得包含 `<script>` 或外部 URL**

写入 `{op_dir}/panels/memory/ub_viz.html`。

---

## Part B：写文字分析（analysis.md）

### B-Step 1：收集补充材料

在 Part A 材料基础上，补充读取：

```
{op_dir}/panels/precision/data.json    ← per-case 精度数据
{op_dir}/panels/perf/data.json         ← per-case speedup + msprof 数据
{op_dir}/**/op_host/*_custom.cpp       ← tiling 函数（TILE_SIZE、分核逻辑等）
```

同时读取：
```
skills/op-dashboard/panels/memory/SPEC.md
skills/op-dashboard/panels/precision/SPEC.md
skills/op-dashboard/panels/perf/SPEC.md
```

### B-Step 2：撰写 `panels/memory/analysis.md`

按 `panels/memory/SPEC.md` 的分析协议，回答六个问题，写入：

```markdown
## Tiling 策略分析

### UB Buffer 分配（客观数据，由脚本提取）
[保留脚本写的数据表，不要修改]

### 切分方案
[从 op_host 的 TILE_SIZE 和分核公式推断]
[说明 tileSize 选择依据：对齐约束？UB 容量限制？]

### 尾块处理
[从 op_kernel 识别 myRemainder / CopyIn1Rem / Compute1Rem 等尾块路径]
[列出 data.json 中 has_tail=true 的 case]

### 流水线设计
[描述 GM↔UB 搬运模式：单 buffer / double buffer / pipeline]
[说明 SyncAll / CrossCoreSetFlag 等同步点的作用]

### UB 空间利用
[说明 ub_used_kb / ub_total_kb 利用率；Cube 算子低利用率属正常（主计算在 L1/L0）]
[可指向 UB 分配饼图（ub_viz.html）说明详细分布]

### 负载均衡诊断
[从 data.json 中找 balance_pct 最差 case，分析原因]
[结合 active cores / totalElems / tileSize 判断空转情况]
```

删除文件顶部的 `<!-- NEEDS_CLAUDE_ANALYSIS ... -->` 标记行。

### B-Step 3：撰写 `panels/precision/analysis.md`

按 `panels/precision/SPEC.md` 的分析协议，回答五个问题。

**首先检查精度是否全通过（data.json 中 n_pass === n_total）：**

#### 精度全通过时：

```markdown
### 总体结论
[PASS] — X/N 个 case 通过。最大误差 case：name（max_re=X）。

### 误差类型解读
[说明 max_re 的物理含义；FP32 单精度 ULP 误差量级]

### 误差分布规律
[列出 max_re 最大/最小 case；分析是否与 shape 规模相关]

### 误差来源推断
[从算子数学结构分析：精度损失路径]

### 风险评估
[当前误差是否满足推理/训练场景需求；是否有 edge case 需关注]
```

#### 精度未通过时（含 FAIL Case 诊断）：

```markdown
### 总体结论
[FAIL] — X/N 个 case 通过（FAIL：case_a, case_b, ...）。

### 误差类型解读
[分别说明 PASS case 和 FAIL case 使用的指标；FAIL case 若 max_re=None 表示运行时崩溃]

### 误差分布规律
[明确哪些 case PASS/FAIL；分析 shape/dtype 规律：
 - 小 shape PASS，大 shape FAIL → 典型 tiling 越界
 - 全部 FAIL → 典型地址映射或 kernel 未执行
 - 随机分布 → 竞态或内存覆盖]

### 误差来源推断
[针对 FAIL case 分析根因；说明为什么该 case 会触发该类型失败]

### 风险评估
[当前状态不可发布；优先修复 FAIL case；推荐修复方向]
```

删除 `<!-- NEEDS_CLAUDE_ANALYSIS ... -->` 标记行。

### B-Step 4：撰写 `panels/perf/analysis.md`

**首先检查精度状态（data.json 中 n_pass < n_total）**，若精度未通过，在"基准说明"小节首行加入警告。

```markdown
### 基准说明
<!-- 精度未通过时额外加此行 -->
⚠ **精度未通过（X/N PASS），以下性能数据仅供调试参考，不代表算子正确性。**

参考实现为 PyTorch NPU reference（msprof 硬件级计时）。
当前平均 speedup = Xx（几何均值）。

### 性能规律
[按数据量（小/中/大）分组分析 speedup 趋势]
[举出 speedup 最高和最低的 case，说明规律]

### 瓶颈推断
[结合 vec_ratio / mte2_ratio / scalar_ratio 定量分析]

### 优化建议
1. [具体建议]
2. [具体建议]
```

删除 `<!-- NEEDS_CLAUDE_ANALYSIS ... -->` 标记行。

### B-Step 5：重新生成并验证

Part A 和 Part B **全部写完后**，执行（不要每写一个文件就重新生成）：

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/gen_dashboard.py \
    --op-dir {op_dir} \
    --output {op_dir}/dashboard.html

python3 ${CLAUDE_SKILL_DIR}/scripts/check_dashboard.py {op_dir}/dashboard.html
```

目标：`✅ 质量检测通过`（0 FAIL）。

---

## 质量要求

| 文件 | 必须满足 |
|------|---------|
| `flow.html` | >200 字符，含 `algo-node` 或 `algo-phase`，无 `<script>` 无外部 URL |
| `steps.html` | >100 字符，无 `<script>` 无外部 URL |
| `ub_viz.html` | 含 `<svg>`，>300 字符，无 `<script>` 无外部 URL |
| `memory/analysis.md` | 含：切分方案、尾块处理、流水线设计、UB 空间利用、负载均衡 |
| `precision/analysis.md` | 含：`[PASS]` 或 `[FAIL]`、具体 case 名称、max_re 数值 |
| `perf/analysis.md` | 含：基准说明、speedup 数值、瓶颈推断、优化建议 |

---

## 注意事项

- **不要凭空捏造数值**：所有定量数据必须来自 data.json 或源码
- **分析与数据分离**：客观数据表（脚本写的部分）保留不修改，只写分析节
- **不要在 memory 面板中写精度数字，不要在 precision 面板中写性能数字**
- HTML 片段风格参照 `examples/` 示例，但内容必须反映**真实**内核逻辑，不得照抄示例内容
