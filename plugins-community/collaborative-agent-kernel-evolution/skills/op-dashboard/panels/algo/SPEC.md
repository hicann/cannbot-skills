# Panel: 算法图示（Tab 1）— 规范 v4

CC 读完内核源码后，写两个独立 HTML 片段文件：

| 输出文件 | 内容 |
|----------|------|
| `panels/algo/flow.html` | 计算流程图（AIC/AIV 分支 + 数据流箭头） |
| `panels/algo/steps.html` | 算法步骤列表（Pass 注释 + 关键 API） |

风格参照 `panels/algo/examples/flow.html` 和 `examples/steps.html`（只参照布局和 class 命名，内容必须反映真实内核逻辑）。

---

## 数据源（按优先级）

脚本将提取结果写入 `panels/algo/data.json`，CC 优先读此文件；
如字段缺失，再用以下 glob 路径直接读源码。

```
1. panels/algo/data.json          ← 脚本提取（pass_comments, compute_apis, tiling_consts）
2. op_kernel/*_custom.cpp         ← glob: **/op_kernel/*.cpp  或  **/*_custom.cpp
3. op_host/*_custom.cpp           ← glob: **/op_host/*.cpp    或  **/*_custom.cpp
4. *_op_desc.json                 ← glob: **/*_op_desc.json
```

路径兼容性：所有路径从算子根目录（`panels/` 的上级目录）向下 rglob，不假设固定层级。

---

## 解析规则

### AIC / AIV 分支识别
```
if ASCEND_IS_AIC { ... }    → Cube Core (AIC) phase
else { ... }                 → Vector Core (AIV) phase（若无 else → fixpipe 模式，无 AIV）
class KernelCube { ... }    → AIC 逻辑
class KernelVector { ... }  → AIV 逻辑（若 entry 未调用 → 标注「已定义但未启用」）
```

### 关键模式
```
BlockMmad(...)              → GEMM 计算节点
CrossCoreSetFlag(...)       → AIC 完成信号，画 sync 节点
CrossCoreWaitFlag(...)      → AIV 等待信号，画 sync 节点
pipe.InitBuffer(Q, N, sz)   → UB buffer 分配（memory 面板用）
// Pass N: / // Phase N:    → 步骤节点标题
```

### IO 信息
从 `*_op_desc.json` 的 `shape_info.input_shapes[]` / `output_shapes[]` 提取名称、shape、dtype。
若文件不存在，从内核 `extern "C" __global__` 函数签名参数名推断。

---

## CSS 类（dashboard 内置，不得引入外部资源）

| 类名 | 用途 |
|------|------|
| `algo-flow` | 流程图根容器（flex 列） |
| `algo-phase[data-core="AIC"]` | AIC 阶段框（金色边框） |
| `algo-phase[data-core="AIV"]` | AIV 阶段框（紫色边框） |
| `algo-phase-lbl` | 阶段标题行 |
| `algo-node input` | 输入张量节点（蓝色） |
| `algo-node output` | 输出张量节点（绿色） |
| `algo-node compute` | 纯计算节点（黄色） |
| `algo-node workspace` | 中间 workspace（灰色虚线框） |
| `algo-node sync` | 同步节点（红色虚线） |
| `algo-arrow` | 垂直箭头 ↓ |
| `algo-arrow-lbl` | 箭头旁标注小字 |
| `algo-gate` | 条件分支标注（蓝色框） |
| `algo-steps` | 步骤列表容器（flex 列） |
| `algo-step` | 单步（数字圆圈 + 文字） |
| `algo-step-num` | 步骤编号圆圈 |
| `algo-step-body` | 步骤描述 |
| `algo-step-code` | 行内代码片段 |

---

## 审美规范（必须遵守，与功能规则同等优先级）

### 节点文字格式

**输入/输出节点**：`名称 [shape] dtype`，一行一个概念，禁止括号内追加隐晦缩写。

- ✅ `mask [S0, S1] bool`
- ✅ `x [S0, S1] float32`
- ❌ `mask [S0, S1] float32 (0/1)`  — "(0/1)" 读者无法理解
- ❌ `xInQueue [tileSize] ← x[tileOffset]`  — 不是张量格式，是代码片段

若需补充语义，在节点内换行用小字 `<small style="opacity:.7">` 描述，而非塞入标题行：
```html
<div class="algo-node input">
  mask [S0, S1] bool
  <br><small style="opacity:.7">True = 替换为 updates 值，False = 保留 x 原值</small>
</div>
```

**UB Buffer / workspace 节点**：统一格式 `队列名 [容量] ← 来源`
- ✅ `maskScanQueue [tileSize] ← mask[GM]`
- ❌ `maskInQueue [tileSize] ← mask[tileOffset]`  — tileOffset 是运行时变量，节点里不写

**计算节点**：写操作名 + 输出变量，禁止写 C++ 表达式：
- ✅ `统计 tileTrue — 遍历 tile 计 True 数量`
- ❌ `GetValue 统计 tileTrue（本 tile True 数量）`  — API 名不属于语义描述

### 并行输入的布局规则

**禁止**：两个节点并排（flex row）后接一个单独箭头 — 读者不知道箭头从哪个框出发。

```
❌ 错误布局：
┌──────────┐  ┌────────────┐
│ xInQueue │  │ maskInQueue│
└──────────┘  └────────────┘
        ↓   ← 箭头从中间出，歧义！
```

**正确方案 A**（同相 CopyIn 合并为一个多行 workspace 节点）：

```html
<div class="algo-node workspace">
  <ul style="margin:4px 0 0 16px;padding:0;text-align:left">
    <li>xInQueue [tileSize] ← x[GM]</li>
    <li>maskInQueue [tileSize] ← mask[GM]</li>
  </ul>
</div>
<div class="algo-arrow">↓</div>
```

**正确方案 B**（分步展示，每步一个节点 + 箭头）：
```html
<div class="algo-node workspace">xInQueue [tileSize] ← x[GM]</div>
<div class="algo-arrow" style="opacity:.4">↓</div>
<div class="algo-node workspace">maskInQueue [tileSize] ← mask[GM]</div>
<div class="algo-arrow">↓</div>
```

### 箭头语义

- 每个 `algo-arrow` 必须连接**上方单一节点**到**下方单一节点**
- 如需汇聚多路输入，先用方案 A/B 合并成单节点，再接箭头
- 箭头标注 `algo-arrow-lbl` 只描述数据流向，不描述操作

### 多行描述文字

计算节点内有多个逻辑点时，用无序列表，不用逗号分隔长句：

```html
<div class="algo-node compute">
  <ul style="margin:4px 0 0 16px;padding:0;text-align:left">
    <li>mask=1 → 取 updates[localIdx]，localIdx++</li>
    <li>mask=0 → 保留 x[j] 原值</li>
  </ul>
</div>
```

### Phase 标题格式

统一：`阶段名 — 核心类型 · 核数 · 一句话职责`

- ✅ `Phase 1 — AIV · 48 核 · 前序 True 计数`
- ❌ `Phase 1: scalar GetValue 逐元素计数（绕过 VEC→Scalar pipeline sync 问题）。`  — 太长，实现细节属于 steps

---

## 规则（必须遵守）

1. **只用 inline style + 上表 class，禁止引入外部 JS / CDN / `<script>` 标签**
2. `algo-node.compute` 只放纯计算语义；DataCopy / CopyIn / CopyOut 不出现
3. 同步原语（CrossCoreSetFlag/WaitFlag）用 `algo-node.sync`
4. 如内核实际无 AIV 分支（fixpipe 模式），**不要虚构 AIV phase**，在流程图末尾加 `algo-gate` 注明
5. `steps.html` 中每步对应一个有意义的 Pass，不是每行代码
6. 禁止 `http://` / `https://` 外部 URL
7. **并排输入必须用方案 A 或 B 合并，禁止并排框 + 单箭头**
8. **节点标题行只写 `名称 [shape] dtype`，语义补充用 `<small>` 换行**

---

## 验收（check_dashboard.py 检验）

- `panels/algo/flow.html` 存在且长度 > 200 字符
- `panels/algo/steps.html` 存在且长度 > 100 字符
- `flow.html` 包含 `algo-node` 或 `algo-phase`
- 不含 `<script` 或外部 URL
