#### 基本块优化

对算子中所有 operation 逐个审查 shape 与基本块（TileShape）设置，识别维度不匹配、设置不合理等问题，产出审查表格后再进行优化。

##### 1. 逐 Operation Shape 与基本块审查

**分析方法**：
1. 逐行阅读算子 kernel 代码，记录每个 operation（matmul、cast、reshape、view、concat、add、mul、sum 等）
2. 统计每个 operation 的：输入 shape、输出 shape、TileShape 设置（cube_tile / vec_tile）
3. 检查每个 TileShape 是否与实际 shape 匹配、是否合理
4. 汇总问题清单，按优先级排序

**审查表格模板**：

| # | Operation | 输入 Shape | 输出 Shape | TileShape 设置 | 合理? | 问题与建议 |
|---|-----------|-----------|-----------|---------------|------|-----------|
| 1 | `matmul(A, B, b_trans=True)` | A:`[1,4096]` B:`[6144,4096]` → `[1,6144]` FP32 | `cube([16,16],[64,256],[256,256])` | ⚠️ | M=1, K 轴未驻留，建议 `[16,16],[128,4096,256],[256,256]` |

**关键检查项**（对每个 operation 逐一检查）：
1. **TileShape 维度与 Shape 维度是否匹配**：vec_tile_shapes 的每个维度不应超过对应 tensor 维度
2. **Cube TileShape 对齐约束**：L0 各维度必须 CUBE_ALIGN 元素对齐（§一；BF16/FP16 场景）
3. **L1 是否超过实际轴长**：mL1/kL1/nL1 超过对应维度实际大小是无意义的
4. **大轴的 L1 是否过小**：导致过多的重复载入次数
5. **Reduce 轴是否被不必要地切分**：归约计算尽量不对归约轴切分
6. **多个 matmul 是否共用同一 TileShape**：不同 shape 的 matmul 必须独立配置
7. **每个 view/assemble/reshape 站点是否显式设置了 vec_tile**：
   - 站点前按该站点目标 shape **显式**设（F-20 Rule 4 Step 1a/1b 公式取值），站点完成后按下游生效 shape 重设 ambient tile（`set → view → reset` 模式）
   - ⛔ **无效判定为措辞无关模式**：任何**无 ITER 前后实测 µs 数字**支撑的"无需设置/已覆盖/不适用/继承即可"结论（无论措辞）一律**无效 = 站点未评估**——台账须有每站点三元组（公式 set 值 + ITER 前后 µs + 采纳/回退），且全部站点 set+ITER **先于任何结构级变换尝试**执行
   - **vec tile 与 cube tile 是两条独立机制，任何一方不能替代另一方**：vec tile 控制的 GM→UB 装载粒度对**任意 consumer 类型**（matmul/vector/reduce/assemble）普遍成立；cube tile 仅管辖计算切分（发生在装载之后）
   - **set 后未 reset 的残留污染同样要查**（夹缝中小 shape view 继承失配 tile）
   - **判据 = consumer GM→UB 装载任务数（非 "ambient 覆盖"）**——large view [大行数,cols] fp16：ambient (小行,大列)→大量任务 vs F-20 Rule 4 Step 1b (UB 填充,cols)→少量任务（百倍级）；reshape 尾轴失配/assemble 冲突 = 真实 COPY 搬移
   - ⛔ **view→matmul 链（view 结果作为 matmul A/B 矩阵操作数）**：view 的 vec_tile 必须 == matmul 对应 L1 tile（A 按 a_trans、B 按 b_trans 映射表）+ trace-time assert，不可因 tuning case 精度通过而跳过——跳过 = 站点未评估（本条 ⛔ 无效判定条款触发）。完整映射表、原因（性能 + 精度稳定性：decode 类 loop_count=1、batch 维 ≥16 等未测试 case 上非确定性数据损坏）与代码示例见 §3「view→matmul tile alignment 约束」

##### 1b. Tile 值通用化规范（分档 tile 值：条件分支 + fallback；四步流程）

**tile 值必须按 case 分档自适应（if/elif/else 条件分支），禁止只写针对单一 tuning case 的字面常量**。每个 tile 设置点按以下**四步流程**执行：

**步骤 1：识别分块目标 tensor 的 shape 及其中的变量**（如 `[G, S2]` → 变量 `G`、`S2`）。

**步骤 2：确认变量非 dynamic**：变量必须可用于 `pypto.view(tensor, [var, ...], ...)` 的 **shape** 参数（= 传入 kernel 的 plain Python int，trace 期为具体值），且不得是 SymbolicScalar（`pypto.loop`/`pypto.loop_unroll` 循环变量——出现在 view **offsets** 而非 shape，trace 期动态）。只有这样的 int 变量可用于分支条件。

**步骤 3：每个 tile 设置点使用自己的 tile 变量**（如 `n_l0_m3`、`vec_row_m5`）——禁止在不同 `set_vec_tile_shapes`/`set_cube_tile_shapes` 调用之间共享同一变量，避免调优一处影响另一处。

**步骤 4：从最后一个维度开始，为每个维度建条件分支**（靠前的维度可能依赖靠后的维度，如 row 预算 = UB 预算 // (col × dtype_bytes)）。**条件中可使用任意函数/表达式**（`S2 >= 512`、`topK >= 64` 等）——lint 只检查 tile 调用参数，不检查条件。枚举本实现所辖 case 集合中该参数的全部取值来放置分支边界（边界推荐 2 的幂：16/32/64/128/256/512/1024/2048/4096/8192），每个分支内 tile 参数为 **int 字面量 Assign**（设计期预计算，非运行期表达式）；每条分支链必须有 **fallback 常量**（else 分支——对其余所有 case 安全的值）。

**可选（推荐）trace 期 assert**：硬件硬限（L0B/UB）可写为 assert——非法 case 在 compile 期直接失败，而非静默运行。

**放置规则**：所有 if/elif/else tile 条件块统一放置在 **kernel 函数体开头（所有 loop 之前）**；`set_vec_tile_shapes`/`set_cube_tile_shapes` 调用保留在各自 tile 站点（loop 内、shape 切换点等），引用预计算的 tile 变量。原因：(1) tile 条件变量均为 kernel int 参数（步骤 2 已验证非 dynamic），函数入口即可用；(2) 集中放置 = 单一 tile 决策块，便于审计；(3) 避免 loop 内 Assign 的作用域问题（loop 体内 Assign 的变量作用域限于该迭代体）。禁止把条件块散落进 loop 体内计算——条件块在函数体开头一次求值，set 调用按站点引用变量。

**💡 新增 tile 调用成本可忽略**：tile 值正确时按站点/shape 段插入调用是安全的，不要因"少设调用"放弃某站点的分档。

**Static（✅ 允许作 tile 参数）:** Python `int` 字面量（`256`），或 local/module Assign 其 RHS 为 `int` 字面量（`CUBE_K = 64`）。lint 的 `_resolve_to_const_int` 接受 `ast.Constant(int)` 和解析到 `ast.Constant` 的 `ast.Name` Assign 链。
**Dynamic（❌ 禁止作 tile 参数）:** func 参数（`S2`/`Dk`/`G`——plain int 但 lint 无法在 AST 解析为字面量）、SymbolicScalar（loop counter from `pypto.loop`）、及**任何 function of either**（`min(S2, 256)`、`S2 * 2`、`192*1024//(Dk*2)`——BinOp/Call 节点，lint 拒绝）。

| 写法 | 类型 | lint |
|---|---|---|
| `set_vec_tile_shapes(256, 64)` | static (int 字面量) | ✅ |
| `CUBE_K=64; set_vec_tile_shapes(CUBE_K, 64)` | static (Assign→int) | ✅ |
| `set_vec_tile_shapes(S2, 64)` | dynamic (func param) | ❌ |
| `set_vec_tile_shapes(min(S2,256), 64)` | dynamic (Call) | ❌ |
| `set_vec_tile_shapes(192*1024//(Dk*2), Dk)` | dynamic (BinOp+param) | ❌ |
| `if Dk>=128: k_l1=128; elif Dk>=64: k_l1=64; else: k_l1=32; set_vec_tile_shapes(768, k_l1)` | static (per-branch Assign→int) | ✅ |

```python
# 分档 tile 值（if/elif/else 条件分支；§1b 四步流程）
# ⚠️ 放置规则：条件块统一位于 kernel 函数体开头（所有 loop 之前）
# (1) 条件分支：条件中可用任意函数，边界推荐 2 的幂（枚举全部 case 取值放置）
# (2) fallback 常量 = else 分支（对其余所有 case 安全）
# (3) trace 期 assert = 可选硬限校验
# (4) UB/3 预算（普通计算 op，input+output+intermediates 共享 UB，≈64KB）
#     n_l0_m3 上限 = 64KB // (k_l0_m3 × dtype_bytes) = 64KB // (16 × 2) = 2048
if S2 >= 2048:
    n_l0_m3 = 2048       # UB/3 cap: 64KB//(16*2)=2048
elif S2 >= 1024:
    n_l0_m3 = 1024
elif S2 >= 512:
    n_l0_m3 = 512
else:
    n_l0_m3 = 256        # fallback
k_l0_m3 = 16             # int 常量（CUBE_ALIGN 对齐下限（§一），static ✅）
assert n_l0_m3 * k_l0_m3 * 2 <= 64 * 1024  # UB/3 硬限（普通计算 op）

# ...（函数体：loop 与操作）……tile 调用站点引用上述预计算变量（站点在 loop 内亦可）：
pypto.set_vec_tile_shapes(n_l0_m3, k_l0_m3)  # 两参数均为 static int ✅
# ❌ 不可写：pypto.set_vec_tile_shapes(min(S2, 512), 64)  — min() 是 Call ❌
# ❌ 不可写：pypto.set_vec_tile_shapes(S2, 64)             — S2 是 func param ❌
```

**硬约束（候选值必须全部满足；违反者禁止设置——即使实测"更快"也不得采纳，应报告框架排查静默溢出）**：
- **L0B**：`kL0 × nL0 × dtype_bytes ≤ L0B_size`（§一）
- **UB**：单任务 buffer（tile_rows × tile_cols × dtype_bytes）建议 16~64KB；input+output+intermediates 共享 UB（UB_size（§一））
- **reduce 段**：归约轴（尾轴）**不可切分**（col = 归约轴长度）——空间不足时**切行**（row-cut），不得切列
- **scatter_ 类不可切片轴**：col ≥ 该轴长度，且**仅约束包含该 op 的段**，不外溢到后续段（实测确认）

**禁止事项**：
- ⛔ 禁止以"我认为不优/收益低"等未实测论据跳过公式分支或缩短候选清单（性能论据只能影响优先级，见 SKILL.md §A4）
- ⛔ 禁止设置超出 UB/L0B 硬限的 tile（框架可能静默吸收溢出，实测结果不可信）
- ⛔ 禁止在 tuning case 上调出字面值后直接硬编码（必须还原为公式 + fallback）
- ⛔ 禁止假设某约束"全局生效"而不验证其作用段范围（约束随 op 走，不随段持久）
- ⛔ **普遍性检查**：禁止在分支中假设某标量"总是"某范围而不对照**本实现所辖 case 集合**验证——范围 = 路由到**本实现**的全部 case（按 case class 分实现时只看本 class 的 case 集 + 本实现的 L0/L1 测试 case；单实现覆盖多 case class 时则须覆盖全部）。分支产出逐一代入（维度相同但取值不同，如 `S2` 与 `N2` 混淆是典型错误——务必按 layout 正确读取轴语义），任一 case 使 tile 超限/除零/失配即须补充分支或修正 fallback
- ⛔ **禁止 refinement 把分支退化为全局字面常量**：F-16/F-17/F-20/S-11/S-12 等后续优化步骤对本方法产出的分支链做 refinement 时，只能修改 **if/elif/else 分支参数**（分支内 int 字面量值、分支条件边界（2 的幂）、新增 case 分支——如把 `elif S2 >= 256: n_l0 = 256` 改为 `n_l0 = 512`）

**采纳决策策略（每个 tile 设置点的 ITER_VERIFY 后必循）**：

> **测量范围（精度集，强制）**：tile/unroll 类 ITER 的性能实测不只测 tuning case——通过 `run_cases.py --cases <affected> --perf` 逐 case 测 AICore E2E（精度集 = `tuning_case_set`，2-3 case 覆盖结构极端值；INIT 确定，见 tune-orchestrator/SKILL.md「covering set + runner 创建」），台账按 case 分列。

**分支验证（tile/unroll 类 ITER_MODIFY 之后、ITER_VERIFY 之前，强制）**：见 tune-frontend/SKILL.md §5「分支验证」——使用 `run_cases.py --cases <affected> --compile-only`（纯编译路径变化）或 `run_cases.py --cases <affected>`（精度模式，tile 切分变化）。任一 case 失败即回退本次 ITER_MODIFY。本节不重复展开，以 SKILL.md §5 为准。

| ITER_VERIFY 实测结果（精度集 case） | 行动（强制） |
|---------------------|-------------|
| 改善（精度集全部 case > +2%） | 采纳 |
| **分歧（部分 case 改善、部分回退）** | **禁止整体回退**——分歧说明该分支参数对回退 case 非最优：定位回退 case 命中的分支条件（哪个参数值触发），为该参数段**增设/修改 if/elif/else 分支**（给回退 case 单独取值）→ 重新走分支验证 + ITER_VERIFY 精度集。仅精度集全部 case 回退时才走「回退」行 |
| 中性（精度集全部 case ±2% 内） | 若该公式可使**任何其他 case class** 受益（逐 case 代入公式验证 tile 值是否改善）→ 采纳。⛔ 禁止仅因 tuning case 中性就回滚——tuning case 中性 ≠ 公式无用 |
| 回退（精度集全部或多数 case < −2%） | **先变体重指派，后放弃**：沿适用 F-step 的变体阶梯（见 SKILL.md 对应 F-step「变体阶梯」小节）尝试下一个变体（每个 site ≤2-3 个变体，全部满足硬约束），逐个实测；首个中性或更优即采纳；全部回退才 Decline 并记录 DO-NOT-RETRY（附变体清单+实测数字）。MEMORY.md 已有充分实测证据的搜索空间可凭既有证据直接 Decline |
| 前置条件 | 变体重指派改变了 tensor shape → 重新运行枚举脚本并重新生成 A4 议程（stage C 规则 4）；新增/修改分支后 → 分支验证的 case 选择相应扩展 |

##### 2. Cube TileShape 设置规范

> **Buffer 容量约束**：cube tile 的 L0/L1 切分受 L1/L0A/L0B/L0C 容量限制（各平台具体数值见 [shared/npu-memory-arch.md](../../shared/npu-memory-arch.md) §一）。kernel 代码中获取精确值见该文「在 kernel 代码中获取 Buffer 容量」。

**函数原型**：
```python
pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kL1], [nL0, nL1])
# 高级用法：A/B 矩阵独立设置 K 轴切分
pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kAL1, kBL1], [nL0, nL1])
# K 轴分核
pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kL1], [nL0, nL1], enable_split_k=True)
```

**参数说明**：
- `mL0/mL1`：M 维度在 L0/L1 上的切分大小
- `kL0/kL1`：K 维度在 L0/L1 上的切分大小；三维 `[kL0, kAL1, kBL1]` 可分别设置 A/B 矩阵 K 轴切分
- `nL0/nL1`：N 维度在 L0/L1 上的切分大小
- `enable_split_k`：是否启用 K 轴分核，让不同核并行计算 K 轴的不同分块（默认不启用）

**对齐约束**（BF16/FP16 场景）：
- L0 各维度必须 CUBE_ALIGN 元素对齐（§一；即 L0_M, L0_K, L0_N 均为 CUBE_ALIGN 的倍数）
- kL0, kL1, nL0, nL1 需满足 32 字节对齐
- `L0 <= L1` 且 `L1 % L0 == 0`

**训练场景（M/N 较大）推荐初始配置**（A2/A3 平台）：
```python
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])
pypto.set_cube_tile_shapes([256, 256], [64, 256], [128, 128])
pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128])
```

**Cube Tile 公式推导方法（逐 matmul 执行；F-9 基线与 F-16 差异化共用此方法）**：

推荐配置仅为量级示意，**禁止直接照抄字面值**——每个算子的 tile 必须按下述步骤从 case 标量推导为分档分支值（§1b 四步流程）。**M 轴永远是 if/elif/else 分支**（每分支 int 字面量，边界推荐 2 的幂），禁止退化为全局常量。如 `if G >= 128: cube_m = 128; elif G >= 64: cube_m = 64; elif G >= 32: cube_m = 32; else: cube_m = 16`——M 大的 case class（如 G=N1//N2 可达 128）依赖分支取整轴；G < CUBE_ALIGN 时 cube_m = CUBE_ALIGN（CUBE_ALIGN 对齐下限（§一），编译器对不足部分 padding）。

**步骤 0：Buffer 容量参数（按 dtype/平台代入）**：
- `C_A = C_B = L0A_size // dtype_bytes`（§一；L0A/L0B 元素上限）
- `C_C = L0C_size // 4`（§一；L0C 累加器恒 fp32）
- 三重积硬约束：`mL0·kL0 ≤ C_A`，`kL0·nL0 ≤ C_B`，`mL0·nL0 ≤ C_C`

**步骤 1：basis 推导（全大轴 fallback tile）**：
1. `T` = 不超过「C_A、C_B、C_C 三者中较小者的平方根」的最大 2 的幂（fp32→128，fp16/bf16→128）
2. basis = `[T, T, T]`，然后**逐轴贪心加倍**（顺序 K → M → N；每步加倍后须重验三重积全部满足，不满足则该轴保持）：
   - fp32: [128,128,128] → 加倍 K 得 m·k=128·256·4B=128KB > 64KB ✗ → 加倍 M/N 同理 ✗ → basis **[128,128,128]**
   - fp16/bf16: [128,128,128] → 加倍 K: 128·256·2B=64KB ✓, 256·128·2B=64KB ✓, 128·128·4B=64KB ≤ 128KB ✓ → 再加倍 M/N 均 ✗ → basis **[128, 256, 128]**
3. 加倍顺序理由：K 在前（K 切分数直接倍乘 reduce 汇聚流量，且 L0B 通常是最紧约束）；实现按此推导，不按 dtype 硬编码。

**步骤 2：枚举 case 集合**：从 eval cases（`tasks/<op>/cases.csv`）与 L0/L1 测试用例提取该 matmul 的 M/K/N 全部取值——M/K/N 是 case 标量的函数（如 attention 类 M=G=N1//N2、K=Dk、N=S2），必须按 layout 语义正确读取（BSND/BNSD 轴序不同）。**范围 = 路由到本实现的 case class 集合**（§1b 普遍性检查同此范围；单实现多 class 时覆盖全部）。

**步骤 3：case 分裂模板（逐轴按 T=128 分类 <T / ≥T，8 case）**：

| # | M | K | N | 分档规则（概念描述——代码实现为 if/elif/else 分支，见 §1b 步骤 4） |
|---|---|---|---|---|
| 1 | <T | <T | <T | 三轴整取（附 CUBE_ALIGN 对齐下限分支（§一）） |
| 2 | ≥T | <T | <T | m 维 cap 分支：m 不超过 C 除以（K、N 中较大者）；K/N 整取 |
| 3 | <T | ≥T | <T | k 维 cap 分支：k 不超过 C 除以（M、N 中较大者）；M/N 整取（与 case 2 同构） |
| 4 | <T | <T | ≥T | n 维 cap 分支：n 不超过 C 除以（K、M 中较大者）；M/K 整取（与 case 2 同构） |
| 5 | ≥T | ≥T | <T | m、k 双 cap 分支：m 不超过 C 除以（N 与本行已算出的 k 值中较大者）；k 不超过 C 除以 N；N 整取 |
| 6 | ≥T | <T | ≥T | m、n 双 cap 分支：m 不超过「C 除以 K」与本行已算出的 n 值中较小者（m·n 还须过 L0C clamp）；n 不超过 C 除以 K；K 整取 |
| 7 | <T | ≥T | ≥T | k、n 双 cap 分支：k 不超过 C 除以（M 与本行已算出的 n 值中较大者）；n 不超过 C 除以 K；M 整取（与 case 5 同构） |
| 8 | ≥T | ≥T | ≥T | 全轴用 basis（步骤 1 结果） |

> 表中 `C` 泛指步骤 0 三重积约束中的相关容量（`C_A`/`C_B`/`C_C`），按受约束轴组合取值——m 轴 cap 受 `mL0·kL0 ≤ C_A` 与 `mL0·nL0 ≤ C_C` 双重约束、k 轴 cap 受 `mL0·kL0 ≤ C_A` 与 `kL0·nL0 ≤ C_B`、n 轴 cap 受 `kL0·nL0 ≤ C_B` 与 `mL0·nL0 ≤ C_C`；实现时以具体三重积为准，用对应容量代入。

**实现方式**：表中分档规则为概念描述——代码实现为 **if/elif/else 分支**（§1b 步骤 4），tile 参数必须解析为 int 字面量（lint 只检查 `set_cube_tile_shapes` 调用参数，不检查条件）。

**步骤 4：归一化与收尾（顺序固定）**：
1. **逐轴不超过轴长**：tile 不得超过轴实际长度（整取分支天然满足，cap 分支显式封顶）——分支实现（§1b 步骤 4）
2. **CUBE_ALIGN 对齐向下取整**（下限 CUBE_ALIGN（§一））：tile 向下取到 CUBE_ALIGN 的倍数、且不小于 CUBE_ALIGN——分支实现（§1b 步骤 4）
3. **floor-pow2 归一化（仅 cube）**：对由容量 cap 推导出的值向下取最近 2 的幂（整轴取值保留原值——整取优先于 pow2，避免多出切分；若整取值触发联合约束则对 cap 侧取 floor-pow2）。收益：地址计算 shift 化、切分数干净、容量整除
4. **L0C clamp**：`mL0·nL0 ≤ C_C` 不满足时优先减 mL0（M 切分对任务粒度影响最小），再减 nL0
5. **跨轴再平衡校验**：某轴 cap 松弛（小轴）时其余轴 cap 自动放大（cap 分支天然含「C 除以小轴 tile」的联动）——验证小轴整取后大轴 tile 是否达容量上限（如 N<512 时 kL0 可增大至 C_B/nL0），未达则按 F-20 R1/R3 邻居偏向调 cap（refinement 修改分支参数，禁止字面替换，见 §1b）

**步骤 5：普遍性检查（⛔ 强制）**：将各分支对本实现所辖 case 集合（步骤 2 范围）逐一代入，验证 (a) 每个产出 tile 满足 CUBE_ALIGN 对齐（§一）与三重积/L1 硬限 (b) 无 case 触发除零/负值/零 tile (c) fallback 分支覆盖超大 case。任一不满足即修正分支边界/值或补充分支。

**步骤 6：实例化写入代码**：把模板中的 M/K/N 替换为**该算子的真实参数名**（如 M=G、K=Dk、N=S2），按 §1b 四步流程写入 JIT 体（条件分支 + fallback 常量；trace 期 assert 可选），禁止在 tuning case 上调出字面值后硬编码。

优点：在满足 L0 Buffer 约束下达到较大算数强度，可开启 Double Buffer。

**推理/Decode 场景（M=1 或 M 较小）**：

M 轴较小时无法通过 M/N 切分提高算数强度，优化思路是**减少重复载入**：
- 使用 K 轴三维配置 `[kL0, kAL1, kBL1]`，让 A 矩阵一次搬入 L1 驻留
- 尽量增大 nL1，减少 N 轴切分次数

```python
# Decode 场景 matmul 示例：A=[1,K] × B^T=[N,K]
# kAL1 按 K 分档取大值（K 不超过档位值时整轴驻留 L1）→ 条件块在函数体开头（放置规则），站点引用预计算变量
# --- 函数体开头 ---
if K >= 256:
    ka_l1 = 256          # int 字面量 per branch ✅
elif K >= 128:
    ka_l1 = 128
else:
    ka_l1 = 64           # fallback
# --- matmul 站点（loop 内）---
pypto.set_cube_tile_shapes([16, 16], [16, ka_l1, 256], [256, 256])  # kAL1=K 的分档封顶值, kBL1=256
# A 矩阵 K 轴按 kAL1 驻留 L1（K≤档位值时整轴驻留，更大 K 时大幅减少载入批次）, B 矩阵 K 轴按 256 分批载入(kBL1=256)
# ❌ pypto.set_cube_tile_shapes([16, 16], [16, K, 256], [256, 256])  — K 是 func param
```

**配置要点**：
1. **L1 不超过实际轴长**：mL1/kL1/nL1 不应超过对应维度实际大小，超过无意义且浪费 L1 空间
2. **小轴不切**：某轴本身较小（如 K=128），L0=L1 取「实际值」与「最大对齐值」中较小者（分支实现，§1b 步骤 4），不切分
3. **大轴大 tile**：某轴较大（如 N=2048 或 K=4096），用较大 L1 减少切分次数

**⚠️ 独立设置原则**：同一算子中多个不同 shape 的 matmul，**必须在每个 matmul 前分别调用 `set_cube_tile_shapes`**，不要用统一值。

**🔥 案例**：[多 Matmul 独立 TileShape 优化](../cases/per-matmul-tile-shapes.md)（3 个 matmul 独立设 tile，-46.1%）

##### 3. Vector TileShape 设置规范

> **UB 容量约束**：vec tile 的输入+输出必须同时装入 UB（UB_size 每 AIV（§一）），详见 [shared/npu-memory-arch.md](../../shared/npu-memory-arch.md) §一。

**配置原则**：
1. 满足特定 Operation 对 TileShape 的规格约束
2. 保证 Operation 的输入与输出 Tensor 可以在 UB 中分配内存
3. TileShape 不能过大也不能过小（数据块大小在 16 到 64KB 之间）；例外：`view`/`assemble`/`reshape` op 本身不产生计算，但 vec tile 控制 consumer GM→UB 装载任务数（ineffective tile = 更多装载任务 = 性能差；reshape 尾轴失配/assemble 冲突 = 真实 COPY），按 F-20 Rule 4 设整shape/UB 填充 tile（见 tune-frontend SKILL.md「邻居算子感知的 TileShape 优化」Rule 4）
4. **优先用满尾轴**，即尾轴 TileShape 设为与实际 Shape 尾轴相同；尾轴过大必须切分时，按 **512B 对齐**切分
5. 归约类计算尽可能不要在归约轴上进行切分

**Vector Tile 公式推导方法（逐 tile 站点执行；F-10 基线与 F-17 分段共用此方法）**：

vec tile 同样必须按 case 分档自适应（§1b 四步流程），步骤：

1. **逐 tile 站点枚举 shape**：对每个 `set_vec_tile_shapes` 站点，列出该点 tensor shape 的全部 case 取值（case 标量函数，如 `[G, S2]`，范围=路由到本实现的 case 集合），按 layout 语义正确读轴。
2. **列维（col）**：归约轴 / scatter_ 不可切片轴 → col = 该轴长度（或 fallback 整轴封顶值），不切分；非归约段优先用满尾轴，尾轴超 UB 承载时按 512B 对齐切分。
3. **行维（row）UB 预算分支**：row 取「不超过 M」与「UB 预算字节除以（col × dtype_bytes）」两者中的较小值（下限 1）——此规则描述计算逻辑；**代码实现为 if/elif/else 分支**（§1b 步骤 4），不可将表达式直接作为 `set_vec_tile_shapes` 参数（lint 拒绝 BinOp/Call/func-param 节点）。UB 由 input+output+intermediates 共享，预算按单任务实际可用值取（保守 16~64KB）；dtype 按该点 tensor 实际类型（fp32=4B、fp16/bf16=2B），**同一 tensor 链上 dtype 随 cast 变化时预算随之重算**。
4. **shape 变化点重推**（F-17）：段内 shape 交替（[M,N]↔[M,1] 等）时，每个变化点独立代入本方法——[M,1] 归约输出点行维可按其预算放大，[M,N] 点按步骤 3 重算。
5. **归一化（仅对齐，⛔ 不做 pow2）**：row/col 向下取 CUBE_ALIGN 对齐（§一；row 下限 1；col 归约轴=轴长不取整）；尾轴切分按 512B 对齐。**vec 侧不做 floor-pow2**——pow2 收益（shift 地址计算）是 cube L0 专属，vec 强制 pow2 只会浪费 UB 填充（如 row cap 170 → pow2 128 丢弃 42 行，CUBE_ALIGN 对齐 160 保留）。
6. **refinement 只调分支参数**：F-17/F-20 R3/S-12 对本方法产出的分支做 refinement 时只能调 cap/分支边界与分支内字面值（同 §1b 禁止退化条款），采纳后必须仍是分档分支形式。
7. **普遍性检查 + 写入代码**：同 Cube 侧步骤 5-6。

```python
pypto.set_vec_tile_shapes(64, 512)
```

**⚠️ 维度匹配检查**：vec_tile_shapes 每个维度的值不应超过对应 tensor 的实际维度大小。例如 tensor shape 为 `[1, 1024]`，不应设置 `vec(8, 128)`（第 1 维 8 > 实际 1）。

**归约轴切分问题**：
- ❌ 对 reduce 轴切分：多个子图的输出需要在同一个子图进行 reduce 操作，产生 GM 搬运和调度开销
- ✅ 不对 reduce 轴切分：上下游子图合并，没有额外开销

**⚠️ reshape 前后的 vec_tile_shapes 设置规则**：

`reshape` / `view` 操作会改变 tensor 的 shape，但 vec_tile_shapes 需要与当前操作的实际 tensor shape 匹配。因此必须遵循：

1. **reshape 前**：vec_tile_shapes 按源 tensor（reshape 前）的 shape 设置
2. **reshape 后**：在操作 reshape 后的 tensor 之前，重新设置 vec_tile_shapes 按目标 tensor（reshape 后）的 shape

```python
# ✅ 正确：reshape 前按源 shape 设，reshape 后按目标 shape 重新设
pypto.set_vec_tile_shapes(8, 128)          # 源 tensor [8, 128]
k_embed_3d = pypto.reshape(k_embed, [8, 1, 128])
pypto.set_vec_tile_shapes(8, 1, 128)      # 目标 tensor [8, 1, 128]
pypto.assemble(k_embed_3d, [0, pos, 0], cache)

# ❌ 错误：reshape 前就按目标 shape 设了 vec_tile
pypto.set_vec_tile_shapes(8, 1, 128)      # ❌ 此时 tensor 还是 [8, 128]
k_embed_3d = pypto.reshape(k_embed, [8, 1, 128])
pypto.assemble(k_embed_3d, [0, pos, 0], cache)

# ❌ 错误：reshape 后未重新设 vec_tile，沿用旧的设置
v_cur = pypto.reshape(v_bf16, [8, 1, 128])  # 源 [1, 1024] → 目标 [8, 1, 128]
# ❌ 缺少 set_vec_tile_shapes(8, 1, 128)
pypto.assemble(v_cur, [0, pos, 0], cache)
```

**常见遗漏场景**：
- reshape 后紧跟 `assemble`：assemble 消费的是目标 shape，必须在 reshape 后、assemble 前设置匹配目标 shape 的 vec_tile
- reshape/view 后紧跟 `matmul`：view 的 vec_tile 各维度**必须 ==** matmul 对应 L1 tile 维度（见下「view→matmul tile alignment 约束」），否则在未测试 case 上产生非确定性精度错误
- 多个连续 reshape：每次 reshape 后都需确认 vec_tile 是否匹配

**⚠️ view→matmul tile alignment 约束（⛔ 强制，须 trace-time assert）**：

当 `view` 的结果作为 `matmul(A, B, *, a_trans, b_trans)` 的**矩阵操作数**（A=`input` 或 B=`mat2`）输入时，view 的 vec_tile 各维度必须 == matmul 对应轴的 **L1** tile 维度。转置映射按操作数区分：

| 操作数 | trans flag | 存储（view）形状 | view dim0（行）↔ matmul 轴 | view dim1（列）↔ matmul 轴 |
|---|---|---|---|---|
| B (mat2) | b_trans=True | [N, K] | N ↔ `n_l1` | K ↔ `k_l1` |
| B (mat2) | b_trans=False | [K, N] | K ↔ `k_l1` | N ↔ `n_l1` |
| A (input) | a_trans=False | [M, K] | M ↔ `m_l1` | K ↔ `k_l1` |
| A (input) | a_trans=True | [K, M] | K ↔ `k_l1` | M ↔ `m_l1` |

> `transposed_batchmatmul(A, B, out_dtype)` 为批量算子，操作数朝向由算子固定（A 存储 (M,B,K)、B 存储 (B,K,N)），上表 2D 映射不适用。`scaled_mm` 与 `matmul` 相同（A/B 亦为前两个位置参数）。同一 view 变量同时喂 A 与 B 两个操作数时，两个操作数各自按映射核对。

**原因（双重——性能 + 精度稳定性）**：
1. **性能（VEC-CUBE 延迟）**：view 的 vec_tile 控制 AIV 侧 GM→UB 预取的分块大小；matmul 的 L1 tile 控制 AIC 侧 GM→L1B 装载的分块大小。两者读同一 GM 区域但用不同分块大小时，AIV 预取粒度与 AIC 装载粒度不匹配，产生额外 GM 读 roundtrip（VEC 完成后 CUBE 需重新按自己的分块读同一数据），增加 VEC→CUBE 交接延迟。view vec_tile == matmul L1 tile 时两者读同一分块，无额外 roundtrip。
2. **精度稳定性（非确定性损坏）**：在大量小迭代场景（如 decode 类 loop_count=1、batch 维=16 → 128 个迭代）下，GM 读调度冲突导致**非确定性数据损坏**——仅影响未在调优中测试的 case，难以复现。

**⛔ 不可豁免**：即使 tuning case 精度通过（all_close=true），此约束仍**必须满足**。tuning case 精度通过 ≠ 约束满足——未测试 case（不同 case 标量/shape 组合）可能因 GM 读调度冲突产生非确定性精度错误。此约束的满足条件是 view vec_tile == matmul L1 tile（按操作数映射：B 按 b_trans、A 按 a_trans）+ trace-time assert，**不是** tuning case 精度通过。

**多消费者场景**：view 仅有 matmul 消费者时，vec_tile = L1 tile（无需 UB 检查——matmul 经 L1 装载）。view 同时被其他 vector op 消费时，view 的 vec_tile 须验证 UB 预算（vector 消费者经 UB 装载，L1 tile 超 UB 则溢出）。UB 预算 = 全部 UB_size（§一）。L1 tile 超 UB 时，**仅减小 view 的 vec_tile**（m/2 → m/4 → ...，同 Rule 3 递归减行）直至 fit UB——matmul L1 cube tile **不修改**（matmul 经 L1 装载，不受 UB 约束）。N 保持（CV 交接轴）至 M=1 再减 N。vec_tile 可 < L1 tile——可接受，因 vector 消费者需要 UB 适配而非 L1 对齐。

**规则**：
1. **view tile 从 L1 tile 派生**（不独立设置）：先确定 matmul 的 L1 tile 公式，再令 view 的 vec_tile == L1 tile（按上表映射）
2. **改一处必须同步改另一处**：修改 view tile 或 matmul L1 tile 时，必须同步修改对应侧
3. **必须有 trace-time assert**：未测试 case 在编译期失败，而非静默精度错误

```python
# ✅ 正确：view tile 从 L1 tile 派生（if/elif/else 分支）+ trace-time assert
# M3: matmul(q, k_row, b_trans=True), k_row = view [S2, Dk]
#   b_trans=True → view dim0(S2)=N↔n_l1, view dim1(Dk)=K↔k_l1
if Dk >= 64:
    k_l1 = 128          # cap value for the Dk>=64 range (precomputed)
elif Dk >= 32:
    k_l1 = 64
elif Dk >= 16:
    k_l1 = 32
else:
    k_l1 = 16
# n_l1 受 UB 预算约束：n_l1 <= UB_size // (k_l1 * 2)；k_l1 大时 n_l1 须相应减小
if S2 >= 1024 and k_l1 <= 64:
    n_l1 = 1024         # k_l1 ≤ 64: 1024*64*2=128KB ≤ UB_size ✅
elif S2 >= 512 and k_l1 <= 128:
    n_l1 = 512          # k_l1 ≤ 128: 512*128*2=128KB ≤ UB_size ✅
else:
    n_l1 = 256          # fallback: 256*k_l1*2 ≤ 256*128*2=128KB ≤ UB_size ✅
assert n_l1 * k_l1 * 2 <= UB_size, \
    "view→M3 tile UB overflow: n_l1(%d)*k_l1(%d)*2=%d > UB_size" % (
        n_l1, k_l1, n_l1 * k_l1 * 2)
pypto.set_vec_tile_shapes(n_l1, k_l1)  # view tile == L1 tiles, both int ✅
k_row = pypto.view(k_bnsd_input, [S2, Dk], [k_off, 0])  # view shape 用 S2/Dk (plain int, lint 不管 view)

# ❌ Call/BinOp/func-param 直接作 tile 参数：
# cube_k_l1 = min(Dk, 128)          # Assign RHS = Call → rejects when used as tile arg
# pypto.set_vec_tile_shapes(192 * 1024 // (Dk * 2), Dk)  # BinOp + func param → rejects
```

**冗余配置检查**：
检查是否存在连续多次 `set_vec_tile_shapes` 调用且参数相同的情况（常见于 copy-paste 残留），合并为一次调用；同时确认每个 vec_tile_shapes 是否与当前 tensor shape 匹配，不匹配的及时修正。此检查应在阶段 A4（基本块审查表）的"冗余操作一并检查"中完成。

##### 4. 常见基本块问题检查清单

| # | 检查项 | 症状 | 修复方法 |
|---|--------|------|---------|
| 1 | vec_tile 维度超过 tensor 实际维度 | 运行时异常或性能异常 | vec_tile 每维 ≤ 对应 tensor 维度 |
| 2 | cube L1 超过实际轴长 | L1 空间浪费，其他轴可用空间减少 | L1 取「推荐值」与「实际轴长」中较小者（分支实现，§1b 步骤 4），并满足对齐 |
| 3 | K 轴 L1 过小导致重复载入 | GM 搬运开销大（尤其 K≥4096 时） | 增大 kL1，或用 `[kL0, kAL1, kBL1]` 独立配置 |
| 4 | N 轴 L1 过小导致切分次数多 | 任务数过多，调度开销大 | 增大 nL1 减少切分 |
| 5 | 多个 matmul 共用同一 TileShape | 不同 shape 的 matmul 用统一 tile 导致次优 | 每个 matmul 前独立设置 |
| 6 | reduce 轴被切分 | 额外 GM 搬运和调度开销 | 归约轴 tile 设为全长 |
| 7 | Decode M=1 未利用 K 轴驻留 | A 矩阵重复载入 | 使用 `[kL0, kAL1, kBL1]` 让 A 矩阵驻留 L1 |
| 8 | reshape/view 前按站点 shape 设 vec_tile、站点后重设 ambient | reshape/view 操作使用了不匹配的 tile 配置或 set 后未重设 | reshape/view 前按站点目标 shape 设（F-20 Rule 4 Step 1a/1b），站点后按下游生效 shape 重设 ambient tile（`set → view → reset`） |
| 9 | reshape 后 assemble 前缺少 vec_tile | assemble 使用了不匹配的 tile 配置 | reshape 后、assemble 前显式设置匹配目标 shape 的 vec_tile |
| 10 | view→matmul tile 不匹配 | 非确定性精度错误（仅未测试 case 上出现，如 decode 类 loop_count=1、batch 维大） | view vec_tile 各维度 == matmul 对应 L1 tile 维度 + trace-time assert（A/B 操作数完整映射表见 §3「view→matmul tile alignment 约束」） |
| 11 | unroll_list 变体数过多 + 每迭代维度大 | JIT 编译超时（如 MLA G=128 × 7 变体 → M=8192 per variant） | 每迭代维度 ≥ CUBE_ALIGN（§一）且大 loop_count（如 prefill 类）→ `[large_unroll, 1]`（2 变体）；≥ CUBE_ALIGN 且小 loop_count（如 decode 类）→ `[1]`；< CUBE_ALIGN → 完整列表。具体 large_unroll 须实测。详见 F-8 编译爆炸防护 |
