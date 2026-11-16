# ScatterUpdate 范式 CANNBotDSL 语言规则与能力谓词

> **层级**：DSL 语言适配层（adapters/dsl/）私有语言知识。范式入口与语言路由见 [../../patterns.md](../../patterns.md)。全部结论自持（机制 + 规避写法内联完整），不引用外部文档。

## 0 能力谓词真值表

本范式 DSL 实现依赖的语言/硬件能力，以谓词组织（各节为谓词的展开说明）：

| 谓词 | 真值 | DSL 实体 | 展开节 |
|---|---|---|---|
| `SIMT_LOOP_FORM` | 计数型主循环恒三参 grid-stride range | `cannbotdsl.range(tid, bound, stride)` | §3 |
| `SCALAR_CANONICAL_GUARD` | 严格守卫 API 须 canonical 类型值；`dtypes.X()` 构造 canonical、`simt.cast` 部分路径产裸载体 | `dtypes.X(x)` / `simt.mulhi` / `simt.atomic_add` 值参 | §4 |
| `RANGE_ITER_BARE` | range 循环变量（iter_arg）恒裸载体，不可 canonical 化 | —（限制事实） | §4 |
| `SAME_TYPE_CAST_REJECT` | 同型 cast 拒绝（刻意防呆） | — | §4 |
| `DIM_SPEC` | Dim-backed TensorSpec 支撑张量规模运行期流动 | `Dim("N", min=1)` + `TensorSpec((nd,), dt)` | §2 |
| `META_DIRECT_READ` | 张量 shape/stride 只在 VF 外静态索引读取（编译期 int）；8 维预留（1-padding）+ `range_constexpr(8)` 展开 = 单程序覆盖全 rank（零重编译） | `gm.shape[d]` / `gm.stride[d]` / `gm.view(...)` | §2 |
| `HOST_DERIVE` | `@host` 在入队期从 shape 闭式派生标量（n/s_sz/var_sz/launch/active_cores），经 kernel 标量参数寄存器传入（零 GM/UB 访存） | `@host run` 内 `gm.shape[...]` | §2 |
| `COMPILED_PROGRAM_DISPATCH` | 生产使用 compiled-program 派发；dim3 JIT 直调仅用于调试，宿主开销高 | `cannbotdsl.compile` 产物 + `[blocks]` 发射 | §2 |
| `L2CACHE_VOLATILE_RW` | `l2cache_load/store` ⇔ volatile 读/写（L2 直达绕 DCache） | `simt.l2cache_load` / `simt.l2cache_store` | §5 |
| `SPIN_ANTI_HOIST` | 自旋内 `l2cache_load` 被外提为循环不变量 → 死自旋；dcci 每轮锚阻止 | `simt.dcci_single(ptr)`（1 指令/轮） | §5 |
| `ATOMIC_DTYPE_WHITELIST` | GM 原子 add 直发仅 i32/u32/f32/i64/u64/f16/bf16；白名单外永久回退 det | `simt.atomic_add` | §4 |
| `REDUCE_ADD_SUPPORT` | `simt.reduce_add` 支持 i32/f32/f16（warp 域语义）；acc 须支持集内 dtype | `simt.reduce_add` | §4 |
| `MTE_ATOMIC_I64_UNSUPPORTED` | MTE 原子不支持 i64 目的 → i64 计数域走 SIMT 标量原子 | `simt.atomic_add`（值须 canonical 构造） | §4 |
| `UNSIGNED_LITERAL_BITCAST_BUG` | 无符号 dtype 字面量经 codegen 位转换产出 `*(uint*)&0`（rvalue 取址，编译错） | 规避 = 同 dtype 零元张量读 | §6 |
| `EMPTY_TENSOR_EARLY` | 0 长 dim 的动态下标被 IR 构建器拒绝 | `N == 0` 宿主早退（先于排序 tiling 预计算） | §6 |
| `VF_SERIAL_EXEC` | vf = 一次调度单位；多 vf 串行执行；跨 vf 访存天然有序 | `vf(mode="simt")` / `vf(mode="simd")` 区域边界 | §0a |
| `VF_START_COST` | VF 启动有固定成本；纯计算循环应合入单 vf | — | §0a |
| `UB_CAPACITY_QUERY` | UB 容量 `get_mem_size("ub")` 返回 253952（248KB，8KB 预留已扣，**不含** SIMT DCache 32KB——须自留） | `cannbotdsl.get_mem_size("ub")` | §0a |
| `DCACHE_SIMT_SIZE` | SIMT 侧 DCache 32KB，无查询接口；**从 UB 划走（影响预算）** | — | §0a |
| `SIMT_REG_PER_THREAD` | 寄存器总容量固定，线程数翻倍→每线程减半；溢出到 GM 栈 | — | §0a |
| `I64_VEC_SOFT_DECOMP` | 无原生 i64 向量运算单元；i64 "向量运算" = 32 位运算 + interleave + reinterpret_lanes 组合 | `reg.vinterleave` / `reg.vreinterpret_lanes` | §4 补充 |
| `GLOBAL_SYNC_TRIPLE` | 跨核全局同步 = vec_sync_all + arrive + wait 三件套；DSL `global_sync_all` 单调用即三件套 | `simt.global_sync_all` | §0b |
| `SIMT_SYNC_PRIMITIVES` | syncthreads（块内栅栏）/ threadfence（线程间写可见性） | `simt.syncthreads` / `simt.threadfence` | §0b |

---

## 0a 硬件执行模型（引用公共层，本文仅载 DSL 映射）

芯片级行为事实的真值源在 [common/simt-hardware-model.md](../../../common/simt-hardware-model.md)（语言中立）——存储层级图 / VF 调度模型 / DCache 一致性与 volatile 三分法 / DCache 预留 / SIMT 线程代价 / 超发核屏障 / Bank 冲突。本文仅补充 DSL 侧的 API 映射与语言特有形态。

| 公共层硬件事实 | DSL 映射 |
|---|---|
| VF 调度单元 = `vf(mode="simt")` / `vf(mode="simd")` 区域 | 每个区域一次调度；多区域串行 |
| volatile 读 = L2 直读绕 DCache | `simt.l2cache_load`（ldcg） |
| volatile 写 = L2 直写 | `simt.l2cache_store`（stcg） |
| DCache 行失效 | `simt.dcci_single` |
| 普通读写（经 DCache） | tracked `gm[k]` 读 / `gm[k] = v` 写 |
| UB 容量查询 | `cannbotdsl.get_mem_size("ub")` 返回 253952（248KB，8KB 预留已扣，**不含** SIMT DCache 32KB） |
| SIMT 线程数 | `vf(mode="simt", thread=N)` 参数 N |

**grid-stride 铁律的推导根基**：[common/simt-hardware-model.md](../../../common/simt-hardware-model.md) §2。见 §3。

## 0b SIMT 域同步与屏障模型（引用公共层，本文仅载 DSL 映射）

同步原则的真值源在 [common/simt-hardware-model.md](../../../common/simt-hardware-model.md) §6（超发核屏障参与）。DSL 映射：

| 公共层同步概念 | DSL 实体 |
|---|---|
| 全核屏障三件套 | `simt.global_sync_all`（单调用 = vec_sync_all + arrive + wait） |
| 块内栅栏 | `simt.syncthreads` |
| 线程间写可见性 | `simt.threadfence` |
| 超发核计算分块域 | `active_cores` 运行期参数（非发射数） |
| 单核守卫 | `if b == 0` 分支 |

三件套缺失时 MTE/SIMT 写越过屏障对他核不可见 → 跨核数据非确定性空洞（首错位置漂移）。det 三段式的 Phase1/Phase2/Phase3 编排全依赖此模型。

---

## 1 三域 import 前缀纪律

scalar / reg（SIMD 向量寄存器）/ simt 三域存在**十余个同名接口**（cast / atomic / clz / popc / reduce 等）。规则：

```python
# 正确：模块前缀式
from cannbotdsl.ops import simt
from cannbotdsl.ops import reg
j = simt.cast(j, dtype=dtypes.int64)      # 域归属清晰
reg.vstore(buf, 0, v, m)

# 错误：裸名 from-import（丢失域上下文，同名接口无法区分）
from cannbotdsl.ops.simt import cast       # 禁止
cast(...)                                   # 域不可辨
```

别名式（`from cannbotdsl.ops.simt import cast as simt_cast`）为合规备选，适合个别歧义名；规模场景一律模块前缀式。

## 2 动态 shape 程序设计（零重编译范式）

### 2.1 结构键 + 运行期参数分层

**编译期（结构键，进 kernel `__init__`）**：dtype 组合 / 写语义 / KeyT 位宽 / index 形态衍生 / 计数域 / det 细分 / 确定性（det·nondet 双程序族）——**零 shape 数值**。
**运行期（shape 数值）**：两通道承载——

| 通道 | 形态 | 用途 |
|---|---|---|
| 结构化 8 维 TensorSpec（现行） | 全部张量按 `MAXD = 8` 维声明（真实维在前缀、尾部 1-padding） | metadata 直读：shape/stride 供 VF 外静态索引；rank 变化零重编译 |
| 标量入参（现行） | `@host` 从 shape 闭式派生后经 kernel 参数传入（寄存器零访存） | n / s_sz / var_sz / total_task / launch / active_cores |

同结构键所有 shape 共享一份程序——**shape 变更零重编译**（验证法：同键异 shape 二次调用不触发重编译且程序缓存恰 1 份）。dim/axis 与 rank 形态**进结构键**（metadata 静态索引需要），shape 数值零进键。

### 2.2 结构化 8 维 metadata 直读（现行交付形态）

**声明**：全部张量按 `MAXD = 8` 维声明（`t.view(*t.shape, *[1] * (8 - t.dim()))`，1-padding 零拷贝）。尾部 1 在乘积/步长/axis 中全部中性，真实维占据前缀 `[0, rank)`。

```python
pv = [Dim(f"V{d}", min=1) for d in range(MAXD)]     # 8 维预留
program = cannbotdsl.compile(run, TensorSpec(tuple(pv), vdt), ...)

# kernel 内：metadata 只在 VF 外读（编译期静态索引）
n_var = gm_var.shape[0]
for d in cannbotdsl.range_constexpr(MAXD - 1):
    n_var = n_var * gm_var.shape[d + 1]             # 全维乘积（view(-1) 替代）
var_f = gm_var.view(n_var)                          # vf 外扁平视图（元素访问用）

# 真实 rank 相关循环：8 块展开 + 运行期卫（d < rank）
for d in cannbotdsl.range_constexpr(MAXD):
    if d < rank:
        ...
```

要点：
- 维度循环 `range_constexpr(8)` 展开（metadata 须静态索引）；运行期卫 `d < rank` 处理真实 rank 截止
- `axis/dim` 进结构键（编译期 int 才能静态索引 metadata）；`view(-1)` 不支持——扁平视图用显式 8 维乘积
- 8 维定长 = 单编译产物覆盖全 rank（形状变化零重编译）；`@host` 派生见 §2.3

### 2.3 派发路径与 @host 派生：compiled-program（禁止 dim3 JIT 直调于生产）

dim3 JIT 直调仅用于调试，生产使用 compiled-program（`cannbotdsl.compile` 产物）派发。`@host run` 同时承担 **shape 派生 + 闭式 launch**（入队期 Python 计算，零设备访存）：

```python
@host
def run(gm_var, gm_index, gm_src, ...):
    n = gm_index.shape[0]                      # 入队期派生
    var_sz = gm_var.shape[axis]                   # axis = 结构键常量
    s_sz = 1
    for d in cannbotdsl.range_constexpr(MAXD - 1 - axis):
        s_sz = s_sz * gm_var.shape[axis + 1 + d]
    cores = get_platform_info().vector_core_num
    launch = min(cores, max(1, (n + THREAD_NUM - 1) // THREAD_NUM))
    kern[launch](gm_var, gm_index, gm_src, n, s_sz, var_sz)     # 标量经寄存器
```

## 3 SIMT 循环形态铁律：三参 grid-stride

> **推导根基**：VF 启动固定成本 + 串行调度（§0a）→ 纯计算循环合入单 vf → 每线程步进须以 range 表达而非手动步进（手动步进阻止编译器判界与优化）。

```python
# 正确（= AscendC for(i=tid; i<n; i+=stride) 逐形同构）
for i in cannbotdsl.range(tid, out_size, stride):
    ...

# 禁止：while 循环 / 手动步进 / 独立计数器（宿主不可优化的调度劣化形态）
i = tid
while i < out_size:
    ...
    i = i + stride        # 禁止形态
```

数据依赖退出的段扫描（段尾由数据决定）不属本铁律管辖——用前置条件 while（首元素无条件处理后 while 判段尾），禁 `while True + break`（触发 FE yield 错误）。本范式 Phase3 段内扫描的 k 步进循环即此类。

## 4 标量类型纪律（canonical vs 裸载体）

严格守卫 API（`simt.mulhi` / `simt.atomic_add` 值参 / l2cache_store 值参等）要求 **canonical 类型值**；裸载体（signless ir.Value）被拒：

| 构造法 | 产物 | 守卫结果 |
|---|---|---|
| `dtypes.X(x)` 构造器（x 为字面量或 canonical 值） | canonical | ✅ 过 |
| `simt.cast(x, X)` | **部分路径裸载体** | ❌ 严格守卫拒绝 |
| 字面量直接传参 | 部分接口可 / 严格守卫拒绝 | 视守卫级别 |

**域锚定**（canonical 算术链保持）：`b64 = b + dtypes.int64(0)` 后整条算术链保持 canonical——kernel 级预计算参数由此构成 canonical 源（写散射范式的 K68 式 pos_base/step 等即此）。

**range 循环变量（iter_arg）恒裸载体**——cast 包装 / dtypes 构造 / 双操作数序算术锚 / 锚定起点**均不可 canonical 化**（5 种形态均被拒绝）。后果：
- range 变量派生值**不可入 mulhi 类严格守卫 API**——被除数须来自 kernel 级参数链（canonical）
- 固定除数统一使用 `//`（不设快除域分支）。

**同型 cast 拒绝**（刻意防呆）：`cast(v, f32)` 在 v 已是 f32 时报错——同型直通须 const_expr 条件免 cast 守卫（AccType 三态：f32 直通 / f16·bf16 cast f32 / 整数原生）。

**原子值参**：`simt.atomic_add(ptr, value)` 值须 canonical 构造（`dtypes.int64(...)`）；i64 计数域的 excl 发布即此（MTE 原子不支持 i64 目的，走 SIMT 标量原子）。

**i64 向量软分解**（`I64_VEC_SOFT_DECOMP` 谓词）：硬件无原生 i64 向量运算单元（向量寄存器 256B 定宽 + 双寄存器 TraitNumTwo 软分解）——i64 "向量运算" = **32 位运算 + `vinterleave` + `vreinterpret_lanes` 组合**。例：零扩展加宽 = `vinterleave(源寄存器, 零寄存器)` 取其一 + `vreinterpret_lanes(int64)`。性能排序：原生 32 位 > 双寄存器软分解 > 便利封装（含寄存器重排开销）。**加宽转换优先用此组合而非 cast**（cast 指令支持集窄且存在前端接受但降低语义损坏的形态）。

## 5 volatile 语义映射（l2cache 家族）

> **推导根基**：[common/simt-hardware-model.md](../../../common/simt-hardware-model.md) §3（DCache 无跨核失效协议）→ 他核写入对本核 DCache 不可见 → 跨核新鲜度走 ldcg/stcg 绕 DCache 直达 L2；barrier 后稳定数据走普通 gm[k]（经 DCache，访问成本低于绕过 DCache 的 ldcg/stcg）。

DSL 无 volatile 限定符；`l2cache_load/store`（L2 直达，绕 DCache）承担跨核可见性：

| DSL 接口 | ⇔ AscendC | 适用 |
|---|---|---|
| `l2cache_load`（ldcg） | volatile **读** | 跨核轮询他核在写的标志（lookback gh 轮询 / excl_bins 读） |
| 普通 `gm[k]` 读（经 DCache） | 无 volatile | barrier 后稳定数据 / 同核热数据（段尾判据/排序后表/src 基值） |
| `l2cache_store`（stcg） | volatile **写** | 跨核可见直写（冲突 key 发布 / 散射落点 / 嵌入排序 dst / 复合 kernel 键写） |

**复合单 kernel 判定**：ASC 的多 kernel 边界可见性在 DSL 复合 kernel 内不成立——Phase1 写的键/srcPos 被他核 MTE 回读（线程跨步写 × tile 对齐读跨界）时，**写侧必须 stcg**（分离 kernel 的 ASC 用普通写即可，形态差异源自架构而非误用）。
**读侧对表**：逐处核对 ASC 源码 volatile 修饰清单再决定 load 形态——l2cache_load 绕 DCache，访问成本高于普通 DCache 读，把 volatile 语义泛化到全部跨核访问是误用模式（按访问点逐处对表源参照实现的 volatile 修饰）。
**var 终态写**（host 唯一读者）：普通 tracked 写 `gm[t] = v`（kernel 边界保证可见性），勿用 stcg。

**自旋防外提**：跨核轮询自旋中的 `l2cache_load` 被编译器外提为循环不变量（LICM）→ 死自旋。每轮 `simt.dcci_single(ptr)` 反外提锚（1 指令/轮 = volatile 指令级替身）+ 自旋上限安全网。

## 6 codegen/ABI 缺陷规避

| 缺陷 | 症状 | 规避 |
|---|---|---|
| 无符号 dtype 字面量位转换 | `*(uint32_t*)&0`（rvalue 取址，bisheng 编译错） | 字面量入 kernel 改由 **storage 类型值**承载（同 dtype 零元张量读 `gm_zero[z]`） |
| 空张量动态下标 | 0 长 dim 的下标被 IR 构建器拒绝 | `N == 0` 宿主早退（先于排序 tiling 预计算——N==0 时 strides 可能含 0 除数） |
| for-region 定赋值严于 while-region | 循环体内 region 中赋值的局部值不能再在嵌套 region 读 | 区域外初始化模式或内联消除局部 |
| ceil_div 域禁用 | ceil_div 在 SIMT vf 内触发验证失败 | SIMT 循环界用普通算术 `(n + d - 1) // d` |

## 7 越界与防漏判（DSL 落地）

- 越界检查在 index **原值域**进行，窄化截断永远最后（`j=2^32+7` 截断后绕过检查是必测 case）
- skip 形态：det 经哨兵 key（键值域最大值，排序沉底跳过）；nondet 直跳；双路径语义必须一致
- 越界形态 DSL 侧统一 **skip**：det 哨兵沉底 / nondet 直跳（原 err flag + 同步上报路线已弃用）

## 8 DSL 前端行为模型（语言特有，不进 generic）

### 构图期与设备执行期两阶段

同一份 DSL 代码经历**构图期**（Python 解释器执行函数体以收集控制流和接口调用）和**设备执行期**（跑编译产物）两个阶段。

- `print()` 仅构图期有效；设备侧用 `cbd.print()`（目前仅保证 UB/L1/L0C）。
- `__init__`、普通 Python 算术、编译期 `if` 的结果固化进设备代码。
- 列表推导、`lambda`、`try`、嵌套函数、`yield`、`match` 可出现在 `__init__` 或编译期展开循环，**不能写在设备循环体里**。

### 设备控制流硬约束

- 设备 `if`/`for`/`while` 内**禁** `break` / `continue` / `return` / `raise`——det Phase3 段扫描用前置条件 while 即此约束的合规形态。
- 循环变量不能直接带出循环。
- 分支内创建的 `Buffer` / `RingBuffer` / Tensor 不能带出分支。

### 三参 range 五形态对照

| 写法 | 设备执行时处理 | 生成设备代码时处理 |
|---|:---:|:---:|
| `if const_expr(...)` | ✗ | ✓ |
| `if` 运行时条件 | ✓ | ✗ |
| `for i in range_constexpr(...)` | ✗ | ✓ |
| `for i in 固定 tuple/list` | ✗ | ✓ |
| `for i in cannbotdsl.range(...)` | ✓ | ✗ |

`unroll=N` / `unroll_full=True` 是编译时展开**设置**，作用于**设备循环**，不会把 `cannbotdsl.range()` 变成编译期循环。

### 启动形态二分

| 形态 | 启动写法 | 线程指定 |
|---|---|---|
| 混合 SIMD + SIMT | `kernel[block_count](...)`（整数） | VF 内 `thread=N` |
| 纯 SIMT | `kernel[dim3(block), dim3(thread), dyn_ub_buf](...)` | Host `dim3(thread)` |
| compiled-program | `program(tensor..., scalar...)` | 同上（由 @host 内选择） |

两形态**不可互换**；编译器校验。`dyn_ub_buf` 可省略（默认 0）。
