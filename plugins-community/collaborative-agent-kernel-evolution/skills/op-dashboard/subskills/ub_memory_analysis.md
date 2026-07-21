# UB 内存分配分析规范

> 本文档指导 Claude 对 AscendC 算子的片上 UB（Unified Buffer）内存进行正确分析。
> 无论哪种算子类型（Vector/Cube/CV 融合），均可按本规范推算实际分配。

---

## 1. AscendC 片上内存层级

| 层级 | 容量（910B） | 访问单元 | 用途 |
|------|------------|---------|------|
| L1   | 1 MB / 核  | AIC（Cube）| A/B 矩阵预取 |
| L0A  | 64 KB / 核 | AIC | Cube A 矩阵 |
| L0B  | 64 KB / 核 | AIC | Cube B 矩阵 |
| L0C  | 128 KB / 核 | AIC | Cube 累加输出 |
| **UB**  | **192 KB / vector_core（910B 系列）** | **AIV（Vector）** | **GM↔UB 搬运、向量计算** |

**关键原则**：UB 仅属于 AIV（向量核），Cube 核使用 L1/L0/L0C，不占用 UB。

> **注意**：不同芯片 UB 大小不同（910B 系列 = 192 KB，910/910A = 256 KB）。
> `design_tokens.json` 的 `chip.ub_kb` 是配置真源；gen_dashboard.py 应读取该字段
> 并将其写入 dashboard JSON 的 `ub_total_kb`，check_dashboard.py 则从 dashboard JSON
> 中读取 `ub_total_kb`。若 `ub_total_kb` 缺失，check_dashboard.py 回退到 **192 KB**（910B 基准）——
> 若目标芯片为 910/910A（256 KB UB），请确保 gen_dashboard.py 正确写入 `ub_total_kb: 256`。

---

## 2. UB Buffer 分配的数据来源

### 2.1 `op_kernel/*.cpp` — 实际分配语句

```cpp
// TQue（带 slot 数）
pipe.InitBuffer(inputQueue,    1, tileSize * sizeof(float));   // 1 × tileSize × 4 bytes
pipe.InitBuffer(workspaceInQueue, 1, wsInBytes);               // 间接引用局部变量

// TBuf（无 slot）
pipe.InitBuffer(diffBuf, tileSize * sizeof(float));
```

**dashboard 提取规则**（gen_dashboard.py 实现）：
1. 正则匹配 `pipe.InitBuffer(bufName, count, sizeExpr)` 或 `pipe.InitBuffer(bufName, sizeExpr)`
2. 在 `sizeExpr` 中替换已知常量（`tileSize`、`TILE_SIZE` 等）
3. 若 `sizeExpr` 是局部变量（如 `wsInBytes`），先从 kernel 源码提取其赋值表达式再代入

### 2.2 `op_host/*.cpp` — 常量来源

```cpp
// 示例：MSELoss
const uint32_t TILE_SIZE = 4096;           // ← 提取为 TILE_SIZE=4096
tiling.set_tileSize(TILE_SIZE);            // ← 同时提取 tileSize=4096
```

**关键映射**：`tiling.set_<field>(<expr>)` 调用将 op_host 常量名与 kernel 使用的 tiling 字段名绑定，
使 `tileSize * sizeof(float)` 能被正确解析（否则因名称不同导致求值失败）。

### 2.3 运行时变量的保守上界

当 `sizeExpr` 含有运行时变量（`nUsed`、`coreNum` 等），用芯片 **AIV（向量核）数**作为上界：

> **chip.aic vs chip.aiv**：`chip.aic` 代表 AI Core（AIC/Cube）总数；`chip.aiv` 代表 AIV（Vector）核总数。
> 910B 硬件：24 个 ai_core，每个含 2 个 vector_core → `chip.aiv = 48`，`chip.aic = 24`。
> UB 仅属于 AIV，故此处应以 `chip.aiv` 为上界，而非 `chip.aic`。

| 变量名 | 含义 | 上界估算 |
|--------|------|---------|
| `nUsed` / `usedCoreNum` | 实际使用核数 | `chip.aiv`（如 48） |
| `coreNum` | 总核数 | `chip.aiv` |
| `blockDim` | 启动 block 维度 | `chip.aiv` |

示例：`wsInBytes = ((nUsed * sizeof(float) + 31) / 32) * 32`
→ `nUsed=chip.aiv=48`: `((48*4+31)/32)*32 = 192 bytes ≈ 0.19 KB`

---

## 3. 各算子类型的 UB 分配模式

### 3.1 纯 Vector 算子（Element-wise / Reduction）

典型模式（MSELoss 为例）：

```
inputQueue:       1 slot × tileSize × dtype_bytes
targetQueue:      1 slot × tileSize × dtype_bytes
workspaceOutQueue: 1 slot × tileSize × dtype_bytes   (写 partial sum)
workspaceInQueue:  1 slot × ⌈nUsed×4/32⌉×32 bytes   (读 nUsed 个 partial sum)
outputQueue:      1 slot × tileSize × dtype_bytes
diffBuf:          tileSize × dtype_bytes             (TBuf, 中间计算)
sqBuf:            tileSize × dtype_bytes
sharedBuf:        tileSize × dtype_bytes             (ReduceSum workspace)
```

总 UB ≈ `(n_tile_bufs × tileSize × dtype_bytes + n_tbuf × tileSize × dtype_bytes) / 1024`
MSELoss: `(5 × 16 KB) + (3 × 16 KB) + 0.19 KB ≈ 128 KB = 66.7% UB`

### 3.2 Cube 算子（GEMM / Conv）

Cube 核（AIC）的主要矩阵在 L1/L0 中，**不占 UB**。
UB 仅用于 AIV 执行的向量 Epilogue（bias add、activation、量化输出等）。

典型 UB 用量：
- 输出 tile C: `L0Shape_M × L0Shape_N × dtype_bytes`（视 shape 和数据类型）
- Bias buffer: `N × dtype_bytes`
- 合计通常 < 10 KB（远低于 192 KB）

> ⚠ 当 UB 利用率 < 5%，通常属于正常（cube 算子以 L1/L0 为主）。check_dashboard.py 对 Cube/CV 算子的低利用率直接判 PASS，不产生 WARN。

### 3.3 CV 融合算子

- Cube 侧（AIC）：L1/L0/L0C 按 GEMM tile 分配，不计入 UB
- Vector 侧（AIV）：UB 按向量 Epilogue 分配
- workspace（GM）：跨核通信用，不属于 UB

UB 分析时只统计 `TQue<VECIN/VECOUT>` 和 `TBuf<VECCALC>` 中的实际用量。

---

## 4. 逐 Case 分析方法

当有多个测试 case 时，从 **profiling CSV**（`op_summary_*.csv`）直接读取实际数据：

| 字段 | 含义 |
|------|------|
| `Input Shapes` | 实际输入 shape（第一个 `;` 前为第一输入） |
| `Block Dim` | 实际启动核数 |
| `aiv_vec_ratio` | AIV 向量计算时间占比 |
| `aiv_mte2_ratio` | AIV 数据搬运（GM→UB）时间占比 |
| `aiv_scalar_ratio` | AIV 标量/同步时间占比 |

**计算 active cores（各 case 真实工作核数）**：

> `active` 的含义：实际参与有效计算的核数，即 `launched` 核中真正处理了至少一个 tile 的核数。
> - `max_by_tile`：若按"每核处理一块 tile"均匀分块，最多能利用的核数（与元素数无直接关系，取决于 tileSize）。
> - 注意 `totalElems`（元素总数，维度不同于核数）不应直接与核数放入同一 min()，否则维度不一致。

```python
totalElems  = product(shape dims)
tileCount   = ceil(totalElems / tileSize)          # 有效 tile 数 = 最大可并行核数上界
launched    = Block_Dim from profiling CSV          # 实际启动核数
active      = min(tileCount, launched)              # 真正有工作可做的核数（不超过 tile 数）
```

各 case 的 UB 用量 **与 shape 无关**（tileSize 恒定），差别只在 `workspaceInQueue`（随 nUsed 变化，极小）。

---

## 5. 正确性验证

当算子已成功运行（所有 case PASS），UB 分配必然在硬件限制内。
若静态分析显示"超出"，优先检查：

1. **名称不匹配**：op_host 用 `TILE_SIZE`，kernel 用 `tileSize` → 需解析 `tiling.set_<field>()` 建立映射
2. **局部变量未展开**：`pipe.InitBuffer(q, wsInBytes)` 中 `wsInBytes` 需从上文赋值语句提取
3. **回退逻辑误用**：当 size=0 时回退到 `tile_length × dtype_bytes`，而 `tile_length` 取 shape 最后一维可能异常大

---

## 6. 典型错误案例

| 错误现象 | 根因 | 修复方式 |
|---------|------|---------|
| UB 显示 800%+ 溢出 | tileSize 名称不匹配，回退用 shape 尾维 (50000×4B=195KB) | 解析 `tiling.set_tileSize(TILE_SIZE)` 建立映射 |
| workspaceQueue 异常大 | `wsInBytes` 局部变量未展开，回退用 tile_length | 扫描 kernel 局部变量赋值语句 |
| Cube 算子 UB 利用率异常低 | 正常现象（主计算在 L1/L0/L0C） | check_dashboard 自动 PASS，无需处理 |
