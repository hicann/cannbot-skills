---
name: cannbotdsl-vf-fusion
description: "设计或调试 CANNBotDSL 的 VF（Vector-Fold）向量计算折叠时使用。VF 是 CANNBotDSL 独有的核心优化，把多个 vec op 折叠成一个 __simd_vf__ 函数，规则复杂（3 条铁律 + 允许列表 + 不可折叠边界），用错会性能退化或编译失败。当需要用 with vf() 折叠 vec 计算、选择 VF 三种模式（default 标记区域 / register SSA RawVReg / raw 原始寄存器）、用 unroll 静态展开、或写 fusable reduce 时触发。含 7 项常见 VF 陷阱。Channel 跨迭代/跨核通信见 ../cannbotdsl-channel。Triggers: cannbotdsl VF, vector-fold, with vf, __simd_vf__, VF 铁律, vf outputs, unroll, fusable reduce, RawVReg, raw register。Architect Stage 2 / Developer Stage 3 调用。"
---

# cannbotdsl-vf-fusion

CANNBotDSL VF（Vector-Fold）向量计算折叠机制。Architect 在 Stage 2 规划 VF 区域，Developer 在 Stage 3 实现/调试。Channel 跨迭代/跨核通信见 `../cannbotdsl-channel/SKILL.md`。

## 1. VF 三种模式

入口是 `vf` 上下文管理器：

```python
def __init__(self, *, mode: Optional[str] = None, unroll: int = 1, outputs: Optional[Sequence] = None):
```

`mode` 只能是 `None` / `"register"` / `"raw"`。导入：`from cannbotdsl.vf import vf`。

| 模式 | 进入方式 | 用户写什么 | lowering | 何时用 |
|------|----------|-----------|----------|--------|
| **default** (`mode=None`) | `with vf():` 或 `with vf(unroll=N, outputs=[...]):` | memref-tile 结构化 op（`add/sub/mul/exp/reduce_max/reduce_sum/cast/expand/muls` from `cannbotdsl.math`/`.tensor`） | 打 tag → 一个 `cannir.vf`；`ascvec-fuse-vf` 做 row-major 融合 + DSE + 自动 `mem_bar` | elementwise/reduce 链，想要自动 transit 消除 |
| **register** (`mode='register'`) | `with vf(mode='register'):` | `RawVReg` SSA 数据流（`vload/vstore/vstore_first`，`+ - * /`，`vmax/exp/reduce_sum/reduce_max` from `cannbotdsl.raw_reg`） | `cannir.vreg_*` → `LowerRegisterVf`（自动 unroll `U=N/VL_T`，mask 隐藏）；**跳过 FuseVf** | 精确 vreg 复用，避开脆弱的自动融合启发 |
| **raw** (`mode='raw'`) | `with vf(mode='raw'):` | 底层 intrinsic（`vload/vload_unpack/vcast/vreduce_max/vexp_sub/vstore_pack/full_mask/update_mask` from `cannbotdsl.raw_reg`） | 按手写形态 lower，mask/loop 全暴露 | 手调 kernel、packed/unpacked dtype 路径 |

**互斥规则**（`__init__` 强制）：`mode in ('register','raw')` 时禁止传 `unroll=`（报 "does not support unroll="）和 `outputs=`（报 "has no transit buffers"）。

## 2. VF 3 条铁律（`vf_region_design.md`）

`cannir.vf` / `ascvec.vf` 的 IR 级不变量：

1. **`IsolatedFromAbove`** —— 每个 `ascvec.vf` 翻成独立 C 函数；operand 列表 = C 形参列表，1:1 映射。区域内不能隐式引用外部 SSA。**违反**：emit 无法产出参数列表。
2. **Allowlist verifier** —— 区域内只允许约 15 个 op：cannir vector compute（`add/sub/mul/div/max/exp/reduce_sum/reduce_max/muls/cast/expand/ub_copy/ub_format_convert`）+ `scf.for/if/yield` + `arith.*`/`index.*` + `cannir.vf_yield`。**违反**：verifier 拒绝（kernel op / 跨 PIPE sync op（Channel acquire/wait 类）/ `matmul` / GM↔UB `mem_copy` 进 region 都被拒）。
3. **禁止嵌套 `cannir.vf`** —— 一个 flat region = 一个 C 函数；Python 侧 `RuntimeError: nested cannbotdsl.vf() is not allowed`。同理 **VF 不能跨 AIC/AIV 边界**（allowlist 只允许 vector op，天然不跨核）。

其他 verifier 事实（`VfOp::verify()`）：恰好一个 block；entry-block arg 数/类型与 operand 1:1；terminator 必须是 `cannir.vf_yield`。

## 3. 可折叠 / 不可折叠

**允许进 `with vf():`**（default 模式）：`add/sub/mul/div/vmax/exp/muls/cast`、`reduce_sum/reduce_max`、`expand`（广播）、UB↔UB `mem_copy`（被 `decompose-mem-copy` 展成 `ub_copy`/`ub_format_convert`）。

**不允许**（编译期报错）：
- 非纯的非 vector op：跨 PIPE sync op（Channel acquire/wait 类）/ `matmul` / GM↔UB `mem_copy`。报 `'foo' op cannir.vf_group N is non-contiguous within its block; a non-tagged op breaks the run`，或 `non-pure op between cannir.vf_group ops cannot be hoisted out of the run`（框架内部检查，wheel 不可见）→ 移出块外。SageAttention FW-5 即此：k_scale 的 MTE2 load 误入 softmax vf → 报错，移出后 PASS。另见 `../cannbotdsl-api-reference/SKILL.md` §5 #20。
- 嵌套 `with vf():` → Python `RuntimeError`。

> 纯 op（`.tensor`/`.buf_id`）落进块内会被框架自动上提，不用预先 materialize。**同步/DMA op 显式放在 `with vf():` 外**。

## 4. unroll + fusable reduce

**静态 unroll**：`with vf(unroll=2):`，transit 成 `cannir.vf_unroll` attr（默认 1）。`unroll=U` 意为每行由 U 个并行 vreg 处理；inner-for trip = `ceil(N/(U·VL_T))`，body 静态展开 U 份。fp32 时 `VL_T=64`，`N=128 → U=2`。

**fusable reduce**：`LowerReduce` 仅当全部满足才走 `fusableReduceShape`（约束 8）：user-VF + tail-axis（沿 inner==1 末轴 reduce）+ `unroll>1` + `axis>VL_T` + `axis % (U·VL_T) == 0`。否则保持原 lowering 不融合。目的是让 reduce 与上游 elemwise 的 inner-for shape 对齐，融进同一 outer row-for。约束 `unroll·VL_T ≤ cols`。

## 5. 真实 VF 代码片段

**default 模式**（online softmax first tile）：

```python
with vf(unroll=2):
    reduce_max(softmax_max_ub, qk_ub, axis=1)
    expand(self.tmp_tile_ub, softmax_max_ub, axis=1)
    sub(qk_ub, qk_ub, self.tmp_tile_ub)
    exp(qk_ub, qk_ub)
    reduce_sum(softmax_sum_ub, qk_ub, axis=1)
    cast(self.tmp_tile_ub_fp16, qk_ub)             # f32 → f16
    mem_copy(p_ub, self.tmp_tile_ub_fp16, engine=self.nd2nz_engine)
# sync/DMA 在 vf 外
```

**raw 模式**（online-softmax raw-`vf` 范式；完整可跑文件见文末「真实用例」）：

```python
with vf(mode="raw"):
    mask32 = full_mask(elem_bits=32)
    lane0, _ = update_mask(1, elem_bits=32)
    for row in range(SM_M):
        row_base = row * SM_N
        max_s = _row_max_fp32(ub_s, row_base, mask32, lane0)
        sum_s = _exp_tile_store_p(ub_s, ub_p, row_base, max_s, mask32, lane0)
        max_f16 = vcast(max_s, dtypes.float16, mask=lane0)
        vstore_pack(ub_max, row * SM_MAX_STRIDE, max_f16, lane0, mode=PackMode.B32_TO_B16)
        vstore(ub_sum, row * SM_SUM_STRIDE, sum_s, lane0)
```

> **raw 模式的一个关键用途——绕高层 cast int8 反交织（FW-1）**：高层 `cast(int8_dst, fp16_src)` 输出按 2 反交织（`res[k]==ref[2k]`）、编译不报错但结果静默错（SageAttention FW-1，见 `../cannbotdsl-api-reference/SKILL.md` §5 #21）。规避是退到 raw vf 用 `vstore_pack(mode=PackMode.B32_TO_B8)` 产连续 int8——这正是 raw 模式"手调 kernel、packed/unpacked dtype 路径"的典型场景。

**register 模式**：

```python
with vf(mode='register'):
    v_x   = vload(qk_ub.tensor)
    v_max = reduce_max(v_x, axis=-1)               # (M,N)→(M,1)，rank 保留
    vstore_first(max_ub.tensor, v_max)             # 只写 lane 0
    v_sub = v_x - v_max                            # 自动广播
    v_exp = exp(v_sub)
    vstore(qk_ub.tensor, v_exp)
    v_sum = reduce_sum(v_exp, axis=-1)
    vstore_first(sum_ub.tensor, v_sum)
```

## 6. 常见 VF 陷阱

1. **块内混入非纯/非 vector op** → `cannir.vf_group N is non-contiguous ...`。sync/DMA/matmul 放 `with vf():` 外。
2. **嵌套 vf** → `RuntimeError: nested cannbotdsl.vf() is not allowed`。
3. **(M,1) elemwise UB 未对齐**（已修）：单元素 step 非 32B 对齐 → NPU `UB address is not aligned`（507015/error 340）。需 `cols % VL_T == 0`，否则降级 `(1, totalElements)`。raw 模式仍需手动 32B 行 pad（cookbook 里 `SM_MAX_STRIDE=16` / `SM_SUM_STRIDE=8`）。
4. **bridging 选错 store（RAW hazard）**（已修）：融合时 forward-scan storeList 可能桥到陈旧 store（`asc_mul` 读到 `sub` 而非 `exp` 结果）。
5. **广播/brc load 不参与 bridging**（约束 3）：`loadalign_brc`/`storealign_1st` 永不桥；`storealign` + 同 buffer `brc/unpack` 读是硬融合屏障（RAW 保护）。
6. **跨 outer-for 死 store 不消除**（P1）：DSE 只在单 inner-for body 内工作；transit buffer 需显式 `vf(outputs=[...])` 声明，否则死 `storealign` 保留（仅影响性能，不影响正确性）。
7. **register / raw 模式 RAW hazard，无自动 mem_bar**：写后读同 buffer（`vstore(buf,v); v2=vload(buf)`）不保证 store-before-read，值可能错。**raw 模式必须自己在 region 末尾加 `vmem_bar('vst_vld')`** —— 只要该 region 写的任何 buffer 会被**后续代码**读（下一个 `@jit` vec 段、`mem_copy` DMA、高层 `cast`、或下一轮迭代），就需要。
   ```python
   with vf(mode="raw"):
       ...  # 写 m_ub / l_ub / o_acc / p_nz_ub
       vmem_bar('vst_vld')      # ← 缺了它，下游可能读到写入前的内容
   ```
   §5 的 raw 范例与 `references/raw-vf-channel-nz-patterns.md` §1 末尾都有这一句，**照抄时极易漏掉**（它长得像可有可无的收尾）。本仓 GQA 通用化改造中，4 个 raw-VF 函数（init / softmax / update_o / finalize）**全部漏加**，每个都存在跨段"写后读"。register 模式的自动 mem_bar 插入尚未实现，同样要手加。
8. **标量行（per-row max/sum/scale）的 lane 语义：存要 lane0 掩码，读要 lane0 广播 —— 两侧都会静默算错**。
   **写侧**：`vstore(buf, row*STRIDE, v, mask)` 写入的元素个数由 **mask** 决定，不由 buffer 声明的行宽决定。全掩码一次写 `VL_T` 个（fp32 = 64 个），而这类标量 buffer 的行距通常只有 8 或 16（32B 对齐 pad），于是**每写一行就覆盖掉后续 7~8 行**。
   **读侧（对偶陷阱）**：`vload(buf, row*STRIDE)` 也**总是读满一个寄存器**，lane1..63 装的是**后续 7 行**的内容；直接参与逐 lane 运算（如 `vmax(m_old, blk_max)`）就会跨行污染。同理 `vreduce_max/vreduce_sum` 的结果**只有 lane0 有效**，参与逐 lane 运算前必须先广播。
   ```python
   lane0, _ = update_mask(1, elem_bits=32)
   vstore(m_ub, r * ROWPAD, m_new, lane0)                        # ✅ 只写 1 个
   m_old_b = vdup_lane0(vload(m_ub, r * ROWPAD), mask=m32)       # ✅ 读后立即广播
   blk_max = vdup_lane0(vreduce_max(v, mask=m32), mask=m32)      # ✅ reduce 亦然
   m_new   = vmax(m_old_b, blk_max, mask=m32)
   # ❌ vstore(..., m32)：写 64 个，踩掉后面 7 行
   # ❌ vmax(vload(...), vreduce_max(...))：两侧 lane1.. 都是垃圾/邻行
   ```
   两者**编译均通过、无越界报错、结果静默错**。实测（本仓 GQA online-softmax）：写侧漏 lane0 → MERE 从 6e-4 恶化到 53；读侧漏广播 → running max 真值 0.26 被邻行读成 4.89。§5 raw 范例里的 `vstore(ub_sum, row * SM_SUM_STRIDE, sum_s, lane0)` 正是这个原因，照抄时别把 `lane0` 换成全掩码。
   **一句话记法**：raw-VF 里"标量"不是标量，是**一个寄存器里只有 lane0 有意义**。

9. **Vector 方法缺 `@jit` → VF 折叠严重退化（~7× vec 性能损失）**：包含 `with vf(mode="raw"):` + `for` 循环的 Vector 方法**必须**直接标注 `@jit`。若作为普通方法从 `@kernel` 内联调用，AST 预处理器会将其展开到 kernel 主循环体中，VF 编译器只能在庞大的 kernel 上下文中做有限折叠——实测 aiv_vec_time 从 420 us 暴增到 2,935 us（int8-QK FA 场景，shape 1×16×4096×4096×128，910B NPU）。标注 `@jit` 后该方法成为独立编译单元，VF 折叠深度显著增强。**这是 VF 性能的第一优先级检查项**——排查 vec 慢的 kernel，先确认所有含 raw-vf `for` 的方法是否都有 `@jit`。结构细则见 `../cannbotdsl-kernel-structure/SKILL.md` §2.2。

## 7. VF grouping 限制与 raw-VF 优先策略

> **选型不是"raw-VF 一律优先"，而是"两边都有代价，按约束选"。** 高层 API 的代价是 VF 自动归组脆弱（本节表格 4 条，**编译期报错**）；raw-VF 的代价是 lane 语义陷阱密集（§6 陷阱 7/8，**静默算错**）。二者性质不同：前者会拦住你，后者会放你过去然后给错数。
>
> **实测校准**（本仓两次 GQA 开发，同一算子两种写法都上过真机）：
> - **高层 API 在静态上下文里非常可靠** —— fp32 softmax（`reduce_max`/`expand`/`sub`/`exp`/`reduce_sum`/`div`）实测与 torch **bit-exact**（max_abs=0.0，行和恰为 1.000000），且 `expand` 无模板、混 shape grouping 这两条预期风险**均未出现**。
> - **raw-VF 改写同一段 softmax，连踩 3 个 lane 语义 bug**（标量存漏 lane0 掩码 → 覆盖后 7 行；标量读漏 lane0 广播 → 跨行污染；region 末尾漏 `vmem_bar`）。全部**编译通过、无 device fault、结果静默错**，逐个需要真机 probe 才能定位。
>
> **结论**：只有当下方决策树判定"**必须**"时才用 raw-VF（真正强制的只有两处：需要写真 NZ fractal 见 §8 规则 6，以及动态循环 + CrossCore channel 操作数）。其余情况**优先高层 API**，它错了会报错，而 raw-VF 错了不会。若必须用 raw-VF，把 §6 陷阱 7/8 当检查清单逐条过一遍再上板。

高层 vec API（`expand`/`muls`/`cast`/`reduce_max`/`reduce_sum` 等）在 `mode=None`（default）下被框架自动归组为 `cannir.vf` region 并下译。以下 4 种场景下该流程会失败：

| # | 限制 | 触发条件 | 错误信息 | 规避 |
|---|------|---------|---------|------|
| 1 | **`expand` op 无 vf-transform 模板** | 任何位置使用 `expand` | `vf-transform found no template for operation 'cannir.expand'` | raw-VF `vload` + `vdup_lane0` 替代广播 |
| 2 | **不同 shape vec op 自动 grouping mismatch** | 如 `(64,128)` 和 `(64,1)` 的 `muls` 被 VF grouping 合并到同一 region | `vector elementwise operands require matching logical shapes` | 在不同 shape 的 vec op 之间插入 DMA 操作打断 grouping |
| 3 | **CrossCore channel 动态循环 shape 动态化** | 动态 `range` 循环内，CrossCore channel 作为 vec op 操作数 | Channel shape 变为 `(-1,N)`，VF grouping 失败 | vec 段全部用 `vf(mode='raw')`，raw-VF 可直接读写 Channel slot |
| 4 | **高层 API in-place channel 无 producer** | `muls(ch, ch, scalar)` 在 channel 尚未被写入时执行 | `in-place use of a channel with no producer, slot would never be filled` | pre-loop 首迭代在循环外建立 producer |

**raw-VF 决策树**：

```
需要 vec 侧写出真 NZ fractal（P 回喂 cube）？
├─ 是 → 必须 raw-VF vstore_strided（§8 规则 6：唯一通路，另三条被硬守卫关闭）
└─ 否 → 在动态循环内且有 CrossCore channel 操作数？
   ├─ 是 → 必须 raw-VF（#3：channel shape 动态化会让 grouping 失败）
   └─ 否 → 优先高层 API
            · 用到 expand 且真的报 #1 → 再退 raw-VF（先实测，本仓未复现）
            · 混 shape 报 #2 → 先试插 DMA / 拆多个 vf 区，仍不行再退
```

> 决策树从"必须"往"可选"排，是因为 raw-VF 的失败模式是**静默算错**：走到这一步的代价远高于高层 API 报个编译错。表格里的 #1/#2 标注为"预期风险"而非"必然发生"——本仓实测未复现，遇到再退，不要预防性地全段改 raw-VF。

代码示例见 `references/raw-vf-channel-nz-patterns.md`。

## 8. raw-VF 操作 Channel 与 NZ 格式

raw-VF 模式下可直接读写 Channel slot，包括 CrossCore channel。以下是 raw-VF 与 Channel/NZ 格式交互的 5 条规则：

| # | 规则 | 说明 |
|---|------|------|
| 1 | **`vstore_strided` 目标必须在 UB** | bisheng 编译器报 `pointer arguments to simd_vf functions must reside in '__ubuf__' address space`；L1 不可作为 `vstore_strided` 目标。需写入 UB NZ Channel，再 `mem_copy` 到 L1 |
| 2 | **NZ strides 必须动态推导** | 硬编码 stride 在 `n1_pad≠0` 时错误。应从 `channel.physical_stride` 读取并计算 `block_stride = s_n1 // n0` |
| 3 | **`vcast` reg_layout 只接受 RegLayout 枚举** | 字符串 `"even"`/`"odd"` 报 `TypeError: reg_layout must be a RegLayout`；须用 `RegLayout.ZERO`/`RegLayout.ONE` |
| 4 | **raw-VF 可直接读写 CrossCore Channel slot** | `vload_deinterleave(channel, offset)` 读、`vstore_strided(channel, offset, ...)` 写均可行。raw-VF 函数接收 Channel 作为参数，channel-first 下框架自动合成 wait/release |
| 5 | **UB(NZ) → L1(NZ) 直接拷贝** | `mem_copy(p_l1, p_nz_ub)` 无需 engine（NZ→NZ 不需要 nd2nz 转换）。p_l1 和 p_nz_ub 需使用相同的 `data_format="nz"` 和 `n1_pad`。**成立前提见 #6** |
| 6 | **规则 5 的前提：UB 内容必须*真的*是 NZ，不是"声明为 NZ"** | `data_format="nz"` 只改变框架对该 buffer 的**元数据认知**，**不改变高层 vec op 的写入模式**。`cast`/`muls` 等高层 op 一律按 ND 连续写满目的地；cube 侧随后把同一批字节**当作 NZ fractal 解释** → 纯几何置换（数值不坏、位置全错、编译不报错）。实测解码：`O[r,c] == P_flat[16r + c]`，行跨度 16 正是 fp16 的 `n0 = 32/dtype_bytes`。**要让 UB 真是 NZ，只能用 raw-VF `vstore_strided` 手工排布** |
| 7 | **raw-VF region 内禁止整数除法** | NZ 几何推导（`s_m1 // s_m0` 等）若写在 `with vf(mode='raw'):` **内**，发射 `arith.divsi`，被 verifier 拒：`'arith.divsi' op non-pure op between cannir.vf_group ops cannot be hoisted out of the run`。**规避**：所有 stride/offset 在 region **之外**算成 Python int |
| 8 | **逐行 raw-VF 循环必须用 `range_constexpr`** | 普通 `for row in range(N)` 被前端改写成动态 `scf.for`，其 induction variable 是 runtime SSA 值；用它索引预计算的 Python offset 列表报 `TypeError: Dynamic list indexing requires all elements to be Tensor, but element 0 is int` |

代码示例见 `references/raw-vf-channel-nz-patterns.md`。

## 参考

- Channel 跨迭代/跨核通信 → `../cannbotdsl-channel/SKILL.md`
- `references/vf-lifecycle-dtype.md`（VF 边界、buffer owner、混合 AIC/AIV handoff、BF16/FP16 GM 累加与最终 dtype 门禁）
- `references/raw-vf-channel-nz-patterns.md`（raw-VF softmax 完整模式、NZ stride 动态推导、Channel slot 读写、UB(NZ)→L1(NZ) 拷贝）
