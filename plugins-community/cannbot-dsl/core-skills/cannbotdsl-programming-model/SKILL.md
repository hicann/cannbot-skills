---
name: cannbotdsl-programming-model
description: "理解 CANNBotDSL 独特编程模型时使用。CANNBotDSL 的 @jit/@kernel 双层装饰器、trace-time 执行、AST 预处理改写控制流与 TileLang/Triton/AscendC 差异大，容易混淆。讲清 @jit vs @kernel 语义和调用约定、trace-time/compile-time/runtime 三阶段执行、Buffer 单块静态分配、Channel depth-N 存储、VF 三种模式选择、多核 dispatch、典型 kernel 骨架。Triggers: cannbotdsl 编程模型, @jit, @kernel, trace-time, AST 预处理, Buffer, Channel, Channel depth, 多核 dispatch, kernel 骨架。工作流 Stage 1/3 参考。"
---

# cannbotdsl-programming-model

CANNBotDSL 编程模型指南。CANNBotDSL 是 Python 嵌入式编译器：Python 函数体在 trace 期执行、发射 IR，再编译到 NPU。理解"trace 期跑的是构造 IR 的 Python，不是数据计算"是掌握本框架的关键。

**真实来源以源码为准**：API 语义、签名、约束以源码为准。

## 触发条件

- 首次使用 CANNBotDSL、需要理解 `@jit`/`@kernel` 语义
- 需要理解 AST 预处理改写规则（为什么某些 Python 写法被禁）
- 需要理解 Buffer / Channel / VF 编程模型

## 1. `@jit` vs `@kernel` 与调用约定

- **`@jit`**：DSL runtime 入口，也是 host launcher。从 Python 调用会触发完整编译 + 执行。
- **`@kernel`**：device kernel。通过 `op[block_dim](*args)` 发射 `cannir.kernel_launch`，在多核上跑。**launch 实参按位置绑定、不按名字**：host wrapper 里传参顺序与 `@kernel` signature 槽位错位会静默错路由张量 → NaN，且编译/launch 均成功、无 device fault，极难定位。逐槽核对顺序。详见 `../cannbotdsl-api-reference/SKILL.md` §5 #17。

调用约定矩阵（权威表，caller → callee）：

| caller \ callee | `@jit` | `@kernel` | Python |
|-----------------|:------:|:---------:|:------:|
| **Python** | OK 全编译+执行 | **ERR** | — |
| **`@jit`** | OK 编译期内联（nested） | OK 发 `kernel_launch` → NPU | OK trace 期内联 |
| **`@kernel`** | OK 编译期内联 | **ERR** kernel-in-kernel | OK trace 期内联 |

两个 ERR：Python 直接调 `@kernel` 在 `kernel_materialize._materialize_kernel_func` 处 raise（无活跃 `JitLoweringState`）；kernel 里调 kernel 在 `kernel_launcher.py` 入口 raise。**嵌套语义**：外层已开 `ir.InsertionPoint` 时，内层 `@jit` 直接发 IR 而非另起编译流水线。

典型分层：`driver(@jit host)` → `op[grid](@kernel device)` → 内部调 `@jit` 计算段（如 raw-vf 融合块）。

## 2. 三阶段执行模型

| 阶段 | 何时 | 跑什么 | 例子 |
|------|------|--------|------|
| **trace-time** | `@jit`/`@kernel` 被调用、构造 IR 时 | Python 函数体本身 —— 每条 `mem_copy`/`matmul` 调用**发射一个 IR op** | `for i in range(4)` 静态展开成 4 段 IR |
| **compile-time** | IR build 后 | CANNIR 编译 → translate → bisheng | 框架融合 vf、插 sync |
| **runtime** | `.so` 加载后 | NPU 上真正算数据 | kernel 在 device 执行 |

**核心心智模型**：trace 期 Python 变量持有的是 `ir.Value`（SSA 句柄），不是数值。`a1 = a_l1` 拿到的是 Channel/Buffer 句柄；Python 层的 `if`/`for` 决定**发射哪些 op**，除非被 AST 预处理改写成 device 侧动态控制流（见 §3）。

## 3. AST frontend lowering

`@jit`/`@kernel`（默认 `enable_preprocessor=True`）先经无状态
`FrontendCompiler` 分析和 lowering，再执行 trace：

- **动态控制流**：普通 `for` / `if` / `while` outline 为显式 callback，
  typed runtime 生成 `scf.for` / `scf.if` / `scf.while`。跨 region 活跃且被写入的
  DSL value 由数据流分析确定为 state；只读值作为 capture。
- **Python 级保留**：`range_constexpr(...)`、`const_expr(...)` 和
  `target_version(...)` 保持 Python meta-stage 语义。相同 meta guard 下的条件赋值和读取
  由 guard-aware definite-assignment 关联。
- **Python 条件表达式**：`a if predicate else b` 在静态 predicate 下只求值被选分支；
  动态 predicate 下 lower 为惰性 `scf.if`。两个分支在 trace 时各构图一次，设备运行时
  只执行被选 region。分支 ValueTree 结构和 IR leaf 类型必须兼容。
- **early-exit 禁令**：动态 region 内禁止跨边界的 `return` / `raise` / `break` /
  `continue`，前端以稳定的 `FE102_UNSAFE_EARLY_EXIT` 诊断。
- **`range` 约束**：builtin `range` 与 `cannbotdsl.range` 接受 1～3 个位置参数并生成
  动态 SCF；`range_constexpr` 使用 Python 静态迭代。动态循环 step 必须严格为正。
- **闭包**：只读 closure/global 被安全捕获，动态 region 中的 `nonlocal`/`global`
  写入被拒绝；函数物化保留原始 closure cell、defaults 和 metadata。

> 想要"编译期就决定"的循环/分支用 `range_constexpr` / `const_expr`；想要"device 上按数据决定"的用普通 `for`/`if`/`while`，但不能在里面 early-exit。

> **`if <Python 常量>:` 里赋值供区域外使用 → `FE202_REGION_VALUE_ESCAPE`**。判据不是"predicate 是不是常量"，而是**分支体里有没有 runtime 值**：predicate 即使是模块级 Python `bool`，只要体内表达式引用了循环归纳变量之类的 runtime 值，整个 `if` 仍被 lower 成动态 `scf.if`，其局部变量无法逃逸区域。典型踩法（本仓 GQA causal 分块实测）：
> ```python
> if CAUSAL:                       # CAUSAL 是模块级 bool，但 mb 是 runtime 值
>     nkb = (mb * BM + BM - 1 + DIAG) // BN + 1
> else:
>     nkb = NKB
> for j in range(nkb):             # FE202: 'nkb' cannot escape a dynamic if region
> ```
> **规避**：predicate 已是编译期常量时，用 `const_expr(...)` 包一层，分支就留在 Python meta-stage：
> ```python
> if const_expr(CAUSAL):           # Python 级分支，只有被选中的那支进 trace
>     nkb = (mb * BM + BM - 1 + DIAG) // BN + 1
> else:
>     nkb = NKB
> ```
> 若 predicate 本身就是 runtime 值，则改为"在区域外先初始化、区域内只写"，或按诊断 hint 把该值完全在区域内消费掉。

> **`range_constexpr` 与 Channel shape 交互陷阱**：`range_constexpr` 做编译期展开时，循环内的 Channel 操作数的 shape 会变为动态 `(-1, N)`，导致 VF region shape mismatch（`vector elementwise operands require matching logical shapes`）。**规避**：在需要 Channel 操作数的循环中使用动态 `range(bid, bid+N, bnum)` + `n_idx = n_raw - bid`，保持 Channel shape 静态。

## 4. Buffer / Channel 分配模型

- `Buffer(MemLoc.*, shape, dtype, ...)` 表示**一块**片上临时存储。它要求 trace-time 静态正整数 shape/stride 和静态 `addr`，无同步语义，也没有缓冲区游标。
- `Channel(..., depth=N)` 表示 depth-N 同步存储。
- 两者共享同一地址 bump allocator；自动地址不会重叠。显式 `addr=` 可有意 alias，并推进高水位，但调用者必须自行保证生命周期不重叠。
- 旧 `BufferArena`、NBuffer、`make_*`、`make_buf/make_nbuf` 和 `.current()/.advance()` 前端接口已移除。double/depth-N storage 必须用 Channel；无法由 Channel 表达的旧手动 NBuffer 方案当前不受支持。

> **三层类不适用的情况**：当 tile 尺寸等常量随 shape 变化、又参与 `range_constexpr` / `const_expr` 静态展开时，实例属性 `self.BMV` 做不到 trace 期就是 Python 值 —— 常量只能放模块级全局、由 host 在 launch 前改写。此时三层类退化为纯命名空间，**扁平的模块级 `@jit` 函数 + 全局常量更直接**（`@jit`/`@kernel` 必须是模块级 `def`，动态合成一律 `FE002_TARGET_NOT_FOUND`）。变的只是"代码怎么摆"，Channel 归属与 `@jit` 铁律不变。详见 `../cannbotdsl-kernel-structure/SKILL.md` §1.1。

### 4.1 Buffer 生命周期规则

Buffer 的声明位置决定其 trace-time 分配次数：

- **`@kernel` body 内声明的 Buffer**：跨设备循环迭代持久（同一物理地址）。适合 running state（m/l/O 等跨迭代累加器）。
- **被多次 trace 调用的 `@jit` 方法内声明的 Buffer**：每次 trace 调用创建独立物理块。适合临时量，不适合跨迭代共享状态。

```python
@kernel
def fa_kernel(q_gm, kt_gm, v_gm, o_gm):
    # @kernel body 内声明 — 跨循环迭代持久
    o_ub = Buffer(MemLoc.UB, (64, 128), dtypes.float32)
    m_ub = Buffer(MemLoc.UB, (64, 8), dtypes.float32)
    l_ub = Buffer(MemLoc.UB, (64, 8), dtypes.float32)

    for tidx in range(bid, total, bnum):
        for n_iter in range(1, num_n):
            # Buffer 作为参数传给 @jit raw-VF 函数 — 操作同一物理 buffer
            softmax_step(qk_ub, p_nz_ub, m_ub, l_ub, alpha_ub)
            rescale_o(o_ub, alpha_ub)

@jit
def softmax_step(qk_ch, p_ub, m_ub, l_ub, alpha_ub):
    # m_ub/l_ub 是外部传入的 — 同一物理 buffer，跨迭代持久
    with vf(mode='raw'):
        ...
```

**错误写法**：running state 声明在 `@jit` 方法内 → 每次调用新建，跨迭代不持久，rescale 值丢失。

## 5. Channel（`channel.py`）

Channel 表示 depth-N 的同步存储：用户只声明 `depth`。跨核 handoff channel 全局 depth 总和有上限（≤8，超出编译期 raise）。写单跑 Channel op 需在 `@jit`/`@kernel` 上下文内。

## 6. VF 三种模式（`vf.py`）

- **default**（`with vf(unroll=U, outputs=[...]):`）：标记区域，编译器 `ascvec-fuse-vf` 自动融合结构化 vec op。`unroll=U` 静态展开内层。
- **register**（`with vf(mode='register'):`）：SSA RawVReg 数据流，手动 vload/vstore。
- **raw**（`with vf(mode='raw'):`）：原始寄存器操作，70+ 底层 `raw_reg` API。

互斥：`register`/`raw` 模式禁 `unroll=` / `outputs=`（无自动融合、无 transit buffer）。选择指南见 `../cannbotdsl-vf-fusion/SKILL.md`。

## 7. 多核 dispatch（`kernel_launcher.py` + `arch.py`）

- 启动：`op[block_dim](*args)` 或 `op[block_dim].method(*args)`；缺省 `block_dim=1`。`[grid]` 设 block 数。
- kernel 内取核 id：`get_block_idx()` / `get_block_num()` / `get_subblock_id()`。
- 标准 tile 分发 idiom（channel-first CV-mix 范式）：

```python
block_idx = get_block_idx(); block_num = get_block_num()
for tile_idx in range(block_idx, total_tile_num, block_num):   # stride = block_num
    ...
# subblock split-M 输出：
out_half = tile_view(out_tile, (tile_vec_m, tile_n), (subblock_idx, 0))
```

> **dispatch 轴顺序影响每核负载**：`idx2crd` 的维度表排列决定哪个轴变化最快。tile 代价沿某轴变化时（causal 的 `nkb = mb+1`），把该轴放最内层且 extent 整除 GRID 会导致每核工作量恒定不均。把代价沿轴变化的轴挪到最外层即可均衡。详见 `../cannbotdsl-perf-optimize/SKILL.md` 第 0 步。

## 8. 典型 kernel 骨架

见 `../cannbotdsl-op-design/SKILL.md §5`：单块 scratch 用 `Buffer(MemLoc.*, ...)`，流水/同步 storage 用 `Channel(..., depth=N)`；Cube 计算仍通过 L0A/L0B/L0C Channel + `matmul`。

## 参考

- `../cannbotdsl-vf-fusion/SKILL.md`（VF 细节）、`../cannbotdsl-op-design/SKILL.md`（kernel 骨架）
