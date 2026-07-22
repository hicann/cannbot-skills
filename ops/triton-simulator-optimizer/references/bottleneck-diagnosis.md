# 瓶颈诊断（pipe 占比规则）

## 核心方法

解析 `core0.cubecore0`（Cube）和 `core0.veccore0`（Vector）的 `instr_exe.csv`，按 pipe 分类求 cycles 占比。**不要只看端到端时间猜**——同一总时间可能由完全不同的瓶颈导致。

## 910B 核心结构回顾

- 每物理核 = 1 CubeCore + 2 VectorCore（Cube:Vec=1:2）
- Cube 执行 `tl.dot`（MMAD 指令）+ 数据搬运（MTE2 GM→L1，FIXP L0C→UB）
- Vector 执行 `tl.exp`/`tl.where`/`tl.sum`/标量运算（VEXP/VCMPV/SCALAR）
- Cube 与 Vector 经 `WAIT_FLAG_DEVI`（FLOWCTRL）同步——**一方算完等另一方**
- UB 192KB，所有 Vector 操作数须在 UB

## 诊断规则

### 规则 1：Cube 空等 Vector（最常见、最易误诊）

**信号**（cubecore0 CSV）：
- `WAIT_FLAG_DEVI` 占比 > 50%（实测常见 95%）
- `MMAD` 占比 < 5%（实测常见 0.5%）

**含义**：Cube 几乎全程空等 Vector。真正的 dot 算力（MMAD）闲置。

**根因**：循环内存在 `dot → Vector 计算 → dot` 的依赖链。第二个 dot 依赖第一个 dot 经 Vector 处理的结果。一般形式：
```python
for tile in reduce_axis:
    s = dot(A, B)            # Cube
    v = transform(s, stats)  # Vector (依赖 s 的任何逐元素/归约变换)
    acc += dot(v, B)         # Cube, 依赖 v → Cube 等 Vector
```
每轮 Cube 算完 `s` 后必须等 Vector 算完 `v` 才能做第二个 dot。

**反例（不 stall）**：
```python
for tile in reduce_axis:
    s = dot(A, B)            # Cube
    stats = update_online(stats, s)  # Vector, 但下一轮 s 不依赖 stats
# 下一轮 s = dot(A, next_B) 不依赖上一轮 stats → Cube 可连续做 s 的 dot
```

**误诊陷阱**：曾有人见端到端慢 + fp32 输入，凭直觉判"fp32 dot 不走 Cube，30x 硬件瓶颈"。simulator 采集后 MMAD 仅 0.5%，真因是 Cube 空等 Vector。**永远先看 MMAD 占比再下 dot 瓶颈结论**。

**修复方向**（latency-optimizer 优化点，技术细节载其 `references/`，本 skill 不重复）：优化点 **19（Cube/MTE3 分阶段批量解耦）** / **21（Workspace 物化解耦）**——把消费 Vector 输出的 dot 从生产 stats 的循环里拆出独立 pass，消除 `dot→Vector→dot` 依赖链，让 Cube 连续流水。

### 规则 2：i32 比较标量降级

**信号**（veccore0 CSV）：
- `SCALAR` pipe 占比 > 30%
- `CMPN`/`ADD`/`LD`/`ST` 的 `call_count` = tile 元素数（如 BM×BN = 4096，或跨 tile 累计 8192）
- 而 `VCMPV`/`VCMAX` 等 VECTOR pipe calls 极少（如 128）

**含义**：i32 LT/GT/LE/GE 比较被编译为标量循环（每元素一次标量 op），而非向量化。checklist 规范2 要求转 f32，但大 tile 下 f32 cast 会 UB overflow（见 msprof-simulator.md 关键行为认知）。

**根因**：
```python
mask = i_idx[None, :] > j_idx[:, None]   # i32 GT → 标量降级, 4096 次标量比较
```

**修复方向**（latency-optimizer 优化点）：优化点 **6（避免向量 API 标量降级）** / **5（Scalar 转 Vector）** / **17（消除冗余边界运算）**——i32 比较转 f32 向量化；配合块级跳过（block-skip）让 f32 cast 只在对角/边界 tile 触发，以控制 UB 占用。技术细节见 latency-optimizer 对应参考文档。

### 规则 3：访存 bound

**信号**：
- `MTE2`（GM→UB load）或 `MTE3`（UB→GM store）占比 dominant
- Cube `MMAD` 与 Vector compute 都不高

**含义**：瓶颈在 GM 带宽，非计算。

**修复方向**（latency-optimizer 优化点）：
- 减少 GM 往返：优化点 **7（Pass 消除合并）**（融合多 pass）或 **21（Workspace 物化解耦）**（物化中间结果避免重算 dot——仅当 dot 比 GM 读回贵时）
- 增大 tile 减少 GM 访存次数（受 UB 限制）
- 检查是否重复 load 同一数据 → 优化点 **10（循环不变量外提）**

### 规则 4：真·计算 bound（Cube MMAD dominant）

**信号**：cubecore0 `MMAD` 占比 > 50%，`WAIT_FLAG_DEVI` 低。

**含义**：Cube 真的在算 dot，且算满了。这才是"dot 是瓶颈"的**唯一**合法判据。

**此时方可考虑**（均非 latency-optimizer 现成优化点，属硬件极限判据）：
- 增大 tile（受 UB 限制，需权衡）
- bf16/fp16 dot（Cube 原生 bf16/fp16，吞吐远高于 fp32；但需精度验证——fp32 参考实现下 bf16 dot 误差 ~3% 常超阈值）
- 减少 dot 数量（算法级，如块级跳过避免无效 tile 的 dot——配合优化点 6/17 的 block-skip 同时减少 mask 与无效 dot）

**结论**：若增大 tile / bf16 化均不可行 → 真·硬件极限，回 Phase 4.6 终局判定（无对应优化点可应用）。

**严禁**：在未看到 MMAD > 50% 前下"dot 是瓶颈"结论。

### 规则 5：barrier 同步开销

**信号**：`BAR`（barrier）占比高，分散在多 pipe。

**含义**：跨 pipe / 跨核同步频繁。常因循环内依赖链长、每轮都需 barrier。

**修复方向**（latency-optimizer 优化点）：优化点 **19（Cube/MTE3 解耦，减少循环内跨 pipe 依赖链）** / 增大 tile 减少循环次数 / 重组数据流让同 pipe 连续工作。

## 诊断输出格式

每次诊断给出 Cube/Vector 各自的 pipe 占比 + 规则结论 + top-3 热源码行（行号、cyc、calls、构造判读），最后综合主瓶颈/位置/**修复方向（latency-optimizer 优化点编号）**。示例：
```
[Cube] total=Xcyc  WAIT_FLAG_DEVI:A% MMAD:B% MTE2:C% BAR:D%  → 规则N
[Vec]  total=Ycyc  SCALAR:E% VECTOR:F% MTE3:G%             → 规则N
code_exe top: L67 <cyc> calls=<n> → fixp_l0c2ub → 规则1 → 修复方向=优化点19 ; ...
[综合] 主瓶颈=..., 位置=L67/L58(循环内 n 次), 修复方向=优化点 N
```

`code_exe` 行号指向 `kernel_meta/*_kernel.cpp`（需回溯 `.py`）或 line info 开启时直接 `.py`。定位：看 `code` 列路径后缀/目录判文件类型 → 读该行 ±5 行构造 → 查构造→规则表（msprof-simulator.md）→ 与 `instr_exe` 规则交叉验证 → 按文件类型回溯或直接编辑。**修复落地不在本 skill 完成**——诊断报告（含优化点编号 + 采集证据）交回编排器，由 latency-optimizer 命中该优化点产出代码。
