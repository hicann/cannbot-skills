---
name: cannbotdsl-api-reference
description: "查询 CANNBotDSL API 用法/签名/约束，或确认某功能是否被框架支持时使用。CANNBotDSL API 面广（100+ 公开 API），本 skill 是结构化查询入口。当需要知道用什么 API 完成什么任务、某 API 的签名与约束与典型用法、某 API 在哪个上下文可用（@jit vs @kernel vs Python 的可用性矩阵）、调用约定矩阵、数据类型支持表（bool/int/float、dtypes.*、device-only Float8*）、或硬件参数（UB 256KB/L1 512KB/L0 容量、dav-3510）时触发。含已知框架限制清单（从代码提取，关联 skills/probes/ 验证用例）。Triggers: cannbotdsl API, API 签名, API 可用性, 数据类型支持, 已知限制, 框架限制, 硬件参数, @jit @kernel 可用性。工作流 Stage 1/2/3 查询。"
---

# cannbotdsl-api-reference

CANNBotDSL API 结构化查询入口，含已知框架限制清单。API 面广、真实签名以源码为准。

**真实来源以源码为准**：签名、约束、示例以源码为准。

## 触发条件

- 需要查询某个 API 的用法 / 签名 / 约束
- 需要确认某个功能是否被框架支持（先查本页"已知限制"，再查 probes/）
- 设计算子时需要了解可用 API 面

## 1. API 分类索引（按 `__init__.py` 导入分组）

| 类别 | 模块 | 主要 API |
|------|------|----------|
| **装饰器** | `jit_runner` / `kernel_launcher` | `jit`、`kernel` |
| **Arch/多核** | `arch` | `get_block_idx`、`get_block_num`、`get_subblock_id`、`get_subblock_dim` |
| **Buffer** | `buffer` | `Buffer`（单块片上临时存储）、`Buffer.reinterpret`（同址 typed alias） |
| **Tensor/搬运** | `tensor` | `mem_copy`、`make_copy_engine`、`tile_view`、`local_slice`（均需 `from cannbotdsl.tensor import`，未顶层导出） |
| **Layout / 索引** | `tensor` | `ceil_div`、`crd2idx`、`idx2crd`、`size`（均顶层导出） |
| **Compute（结构化 vec/cube）** | `math` | `add`、`sub`、`mul`、`div`、`vmax`、`vselect`、`exp`、`log`、`cumsum`、`reduce_sum`、`reduce_max`、`expand`、`muls`、`cast`、`matmul` |
| **Compute（raw 寄存器）** | `raw_reg` | `vadd/vsub/vmul/vdiv/vmax`、`vexp/vln/vsqrt`、`vexp_sub`、`vcast`、`vreduce_max/sum/min`、`vload/vload_unpack/vload_brc`、`vstore/vstore_pack`、`vcmp_*`、`vdup_scalar/vdup_lane0`、`vmuls/vadds`、`full_mask`、`update_mask`、`Mask`、`PackMode`、`UnpackMode`、`RawVReg`（83 个顶层导出）；`vmin`/`vrelu`/`vstore_strided`/`vor` 等另 11 个需 `from cannbotdsl.raw_reg import`（见 `raw_reg.py`） |
| **Sync** | `channel` | Channel 4 相协议（acquire/commit/wait/release）由框架自动合成，源码层不暴露原语；用户只声明 `Channel(depth, kind)` |
| **VF** | `vf` / `raw_reg` | `vf`、`RawVReg` |
| **Channel/流水** | `channel` / `delay_line` | `Channel`、`ChannelKind`、`DelayLine`、`DelayLineGroup` |
| **Debug** | `debug` | `print_tensor`、`print_scalar` |
| **Staged/AOT** | `spec` / `jit_runner` / `core.compiler` | `jit`（→`JitRunner`，需 `from cannbotdsl.jit_runner import JitRunner`）、`TensorSpec`、`Dim`、`Constexpr`、`CompiledFunction`、`load`、`clear_compile_cache` |
| **Runtime** | `runtime` | `from_dlpack`（顶层）、`from_torch_npu`（`from cannbotdsl.runtime import`）、`_RuntimeTensor` |
| **常量表达式** | `constexpr` | `const_expr`、`range_constexpr`、`target_version` |
| **类型** | `typing.types` | `Tensor`、`MemLoc`、`ChannelKind`、`RoundingMode`、`RegLayout`、`Coord`、`Shape`、`Stride`、`Tile`、`Tiler`（均顶层导出）；`PIPE`（`from cannbotdsl.typing.types import`） |
| **dtype/标量** | `dtypes` / 内建类型 | `bool`、`int`、`float`；`dtypes.bool/int*/uint*/float16/bfloat16/float32`；device-only `Float8E4M3FN/E5M2/E8M0` |

## 2. 常用 API 签名（源码核对）

```python
# Buffer（buffer.py）：只表示单块临时存储，不分配 buf_id/sync_id
Buffer(mem_loc, shape, dtype, *, addr=None, stride=None,
       data_format=None, n1_pad=0)
Buffer.reinterpret(dtype, shape=None) -> Buffer

# depth-N / 同步存储统一用 Channel
Channel(mem_loc, shape, dtype, *, depth, kind=ChannelKind.SameCore,
        addr=None, stride=None, data_format=None, n1_pad=0)

# Compute（math.py）
matmul(dst_l0c, lhs_l0a, rhs_l0b, *, init=...)      # :35
reduce_max(dst, src, axis=-1)  reduce_sum(dst, src, axis=-1)   # :164/:169
expand(dst, src, axis=None)  muls(dst, src, scalar)  cast(dst, src, *, rounding=RN)
cumsum(dst, src, axis=0)                              # :133 —— 仅 axis 语义见 §3

# VF（vf.py）
vf(*, mode=None, unroll=1, outputs=None)             # mode='register'/'raw' 禁 unroll/outputs

# staged AOT（@jit 返回的 JitRunner）
jitted_fn.compile(*specs, **specs) -> CompiledFunction
TensorSpec(shape, dtype, *, stride=None)
Dim(name, *, min=1, max=None, multiple_of=1)
load(so_path) -> CompiledFunction
clear_compile_cache(*, clear_disk=False)
```

## 3. 数据类型支持表（`dtypes.py`）

| dtype | 位宽 | 源码 | 备注 |
|-------|:---:|------|------|
| `dtypes.float16` | 16 | `dtypes.py` | vec 计算常用输入 |
| `dtypes.bfloat16` | 16 | `dtypes.py` | |
| `dtypes.float32` | 32 | `dtypes.py` | 累加/reduce/exp 精度提升目标 |
| `Float8E4M3FN` | 8 | `float.py` | MXFP8 量化 |
| `Float8E5M2` | 8 | `float.py` | MXFP8 量化 |
| `Float8E8M0` | 8 | `float.py` | **build 未暴露 `Float8E8M0FNUType`，用 signless i8 代替**（源码注释 154-158），部分操作受限 |
| `dtypes.int8/int16/int32/int64` | 8/16/32/64 | `dtypes.py` | 精确有符号宽度 |
| `dtypes.uint8/uint16/uint32` | 8/16/32 | `dtypes.py` | 精确无符号宽度 |
| `dtypes.bool` | 1 | `dtypes.py` | bool/i1 |

默认 runtime 标量注解和 AOT descriptor 使用 `bool`、`int`、`float`，分别映射到
i1/i64/f32；精确宽度和所有存储 dtype 使用 `dtypes.*`。旧 value wrapper 已从
`cannbotdsl` 顶层下线，也不能用于注解、AOT 或 dtype；只有
`cannbotdsl.integer.Int32(value)`、`cannbotdsl.float.Float32(value)` 这类动态显式
数值转换仍保留。FP8 因暂无 host descriptor，继续使用上表的 device-only class API。

对齐：vector UB 访问需 32B 对齐（fp16→16 elems/row，fp32→8 elems/row）。矩阵 NZ：`m0=16, n0=32//dtype_bytes`。

## 4. 调用约定矩阵

见 `../cannbotdsl-programming-model/SKILL.md §1`（调用约定矩阵权威表）：Python→`@kernel` 与 `@kernel`→`@kernel` 是仅有的两个 ERR。

## 5. 已知限制清单（源码守卫，逐条可证）

> 每条给出源码 raise 位置。⭐ = 已有 probe 验证（见 `skills/probes/`）；其余为源码可证的构造/参数守卫（无需 NPU）或运行时行为（需 NPU 实测）。

| # | 限制 | 源码守卫 | probe |
|---|------|----------|:---:|
| 1 | **CrossCore Channel depth ≤ 8** | raise `"cross-core Channel depth must be <= 8"`（sync_id 预算：16 base + CUBE +16 → 每 slot 2 计数器） | ⭐ `sync/probe_crosscore_channel_depth.py` |
| 2 | **Channel depth ≥ 1** | raise `"Channel depth must be >= 1"` | — |
| 3 | **`vf(mode='register'/'raw')` 禁 `unroll=`/`outputs=`** | 编译期守卫 | ⭐ `vf/probe_vf_mode_exclusion.py` |
| 4 | **`vf(unroll=)` 必须正整数** | raise `"unroll must be a positive int"` | — |
| 5 | **动态 for/if/while region 禁 early-exit（return/raise/顶层 break/continue）** | raise `"Early exit (...) is not allowed in dynamic ... region"` | — |
| 6 | **DSL for-loop `range` 只接受 1/2/3 参数** | raise `"Unsupported number of range arguments"` | — |
| 7 | **`Float8E8M0` 无原生类型，用 signless i8 代替**，部分操作未实现 | `float.py`（注释） | — |
| 8 | **`vstore_pack(mode='b64_to_b32')` 不支持**（arch3510 无 b64 mask update） | raise `NotImplementedError` | — |
| 9 | **`vreinterpret` 要求位宽相等** | raise `"vreinterpret: bit-width must match"` | — |
| 10 | **`vcast` rounding 仅 rn/rna/rd/ru/rz/ro** | raise | — |
| 11 | **`print_tensor` 仅支持 UB/L1/L0C tensor** | raise `NotImplementedError` | — |
| 12 | **`print_scalar` 仅支持 local/GM/UB 标量** | raise `NotImplementedError` | — |
| 13 | **`Dim` 要求非空 name、`min≥0`、`max≥min`、`multiple_of≥1`；`//` 除数必须是正整数；每个 Dim 至少在 shape/stride 槽裸出现一次** | `core/contracts/normalize.py` raise | — |
| 14 | **`DelayLineGroup.depth ≥ 2`** | raise | — |
| 15 | **bisheng 硬编码 `--npu-arch=dav-3510`** | `core/toolchain/ascendc.py` | — |
| 16 | **`local_slice` 静态 offset 在 vector lowering / UB→UB mem_copy 中可能丢失；不支持动态 SSA offset** | 见 `../../debug-skills/cannbotdsl-precision-debug/references/mixed-kernel-debug-lessons.md` | 可写 probe 验证 |
| 17 | **`make_copy_engine` fixpipe DEQ 反量化与 `dual_dst_ctl=1` 互斥**（运行时行为，需 NPU）：int32→fp16 DEQF16 模式（非单位 `deq_scale`）叠加 `dual_dst_ctl=1` → 真机 errcode 169「fixpipe parameter invalid, subErrType 0x4」（全 cube core）。而 split-M 的 CrossCore L0C→UB 强制要求 `dual_dst_ctl=1`，二者不能同时满足。规避：L0C int32 走 NoQuant passthrough 到 UB（`dtype=dtypes.int32, dual_dst=1`），把 int32→fp32 dequant 挪到 vec 侧首个 op | C++ 框架内部实现（wheel 不可见）（CrossCore L0C→UB 处 `dual_dst_ctl!=1` emitError） | ⭐ `skills/probes/p_qk_i32_ub.py`（真机 max_abs=0.0；SageAttention FW-8） |
| 18 | **`@kernel` launch `op[grid](*args)` 按位置绑定，不按名字**（运行时行为，需 NPU）：实参严格按 `@kernel` signature 槽位顺序绑定；host wrapper（`run()`）里传参顺序与 signature 错位会**静默错路由张量**（如 `q_scale` 落进 `key` 槽）→ NaN，且编译/launch 均成功、无 device fault，极难定位。逐槽核对 launch 调用顺序与 signature；看似关键字的对应关系不生效 | `jit_runner.py`（launch 发射 `cannir.kernel_launch`，实参按位置） | ⭐ SageAttention FW-10 真机实证（错序 → NaN，正序 → PASS） |
| 19 | **`Buffer` 只接受静态、非空、扁平正整数 shape，且只支持 UB/L1/L0A/L0B/L0C**；`addr` 只能是非负 Python `int`/`None`，dtype 必须整字节可寻址。Buffer 与 Channel **可以共存**并共享地址 bump allocator；Buffer 不发射 `cannir.make_buf`，也不消耗 buf_id/sync_id。需要 depth-N 或同步语义时必须用 Channel；旧 NBuffer/`make_*`/`make_buf`/`make_nbuf` 前端 API 已不存在 | `buffer.py` Python 构造守卫；IR 层由 `make_pointer + make_layout + make_memref` 构成（内部实现，非用户 API） | — |
| 20 | **`with vf` region 内禁非纯 op（DMA/sync/matmul）**（编译期守卫）：GM↔UB `mem_copy`、跨 PIPE 同步 op（Channel acquire/wait 类）、`matmul` 落进 vf region → 报 `non-pure op between cannir.vf_group ops cannot be hoisted out of the run`。所有 GM→UB load、sync、matmul 必须放在 `with vf():` 之外（vf 内只允许结构化 vector compute + scf/arith）。详见 `../cannbotdsl-vf-fusion/SKILL.md`（allowlist） | C++ 框架内部实现（wheel 不可见） | ⭐ SageAttention FW-5 真机实证（k_scale MTE2 load 误入 softmax vf → 报错；移出 vf 后 PASS） |
| 21 | **高层 `cast(int8_dst, fp16_src)` 反交织（stride-2）**（运行时行为，需 NPU，**非源码守卫**）：高层 `cast` fp16→int8 输出按 2 反交织（`res[k]==ref[2k]`），编译不报错、结果**静默错**。规避：退到 raw vf 用 `vstore_pack(mode=B32_TO_B8)` 产连续 int8（见 `../cannbotdsl-vf-fusion/SKILL.md` raw 模式）。证据为 probe 实测，无 raise 位置 | 无源码 raise（数值行为） | ⭐ `skills/probes/`（已验证 probe） / `m2c_pack_min.py`（真机；SageAttention FW-1） |

| 22 | **`matmul(dst, lhs, rhs)` 计算 `dst = lhs @ rhsᵀ`**（调用约定，非限制，但极易误读）：右操作数按**转置**参与，即 rhs 需声明成 `(N, K)` 而非 `(K, N)`。这是 Ascend cube 的 nZ 右操作数原生约定。**误读的症状**：结果数值范围正常但全错；令 `A=I` 可解码出 `C[i,j]==B[j,i]`。实测非方阵 `M=64,K=32,N=48`、rhs 声明 `(N,K)`、dst `(M,N)`：max_abs=**3.8e-6** ✅。**对 attention 极其有利**：`S = Q@Kᵀ` 正是这个形状，K 以 `(S_kv, D)` 原样喂入、零转置开销；反之 `O = P@V` 需要 rhs=Vᵀ，实测三条路都可行（host 预转置 / L1+L0B 声明 `data_format="zn"` 走 dn2nz / L1→L0B `transpose=True`，max_abs 均 9.5e-6），**首选 `zn`** | 无 raise（调用约定） | ⭐ 真机实测 |
| 23 | **fixpipe engine 的 `dtype` 必须匹配 *source*（L0C 累加器），不能顺带做窄化** | `mem_copy` 报 `copy engine dtype mismatch: engine='f16', src='f32'` | — |
| 24 | **`UB → L0A` 直通不支持**：必须经 L1 | `'cannir.mem_copy' op unsupported copy capability: src memloc=1, dst memloc=3, transform=identity` | ⭐ 真机实测 |
| 25 | **AST 前端重读函数真实源码 → `@jit`/`@kernel` 必须是模块级 `def`**：动态合成函数（`exec`、闭包工厂、包装临时函数）一律报错。参数化扫描要用模块级全局变量 + 改写全局 | `FrontendError: FE002_TARGET_NOT_FOUND` | ⭐ 真机实测（16 组合扫描全失败 → 改模块级+全局后跑通） |

> 清单基于当前源码。新增限制时补 raise 位置；能写成 probe 的优先补 `skills/probes/`（见 （cannbotdsl-framework-probe 在本仓不可用））。运行时行为类限制需 NPU 实测确认，不要凭空标 PASS。
> #17/#18 是**运行时行为类**（源码守卫锚在 C++ / 运行期，非 Python `raise`），已有真机 probe/实证；#19/#20 是 C++ 编译期守卫；#21 是纯运行时数值行为（无 raise、编译静默通过、结果错），只能靠 probe 实测发现。引用时注意区分证据类型（Python 静态 raise vs C++ guard vs 纯运行时数值行为）。

## 门禁

- 引用 API 前必须核对源码签名，不要凭记忆写参数。
- 声称"框架不支持某功能"必须给出源码 raise 位置或 probe 结论；查不到就标"未确认"而非"不支持"。

## 参考

- 各模块源码：`buffer.py`、`math.py`、`raw_reg.py`、`vf.py`、`channel.py`、`dtypes.py`、`float.py`、`spec.py`、`core/contracts/`、`debug.py`
- `skills/probes/README.md`（已验证的能力/限制）
- `../cannbotdsl-op-design/SKILL.md §2`（Buffer 预算硬限制）、（cannbotdsl-framework-probe 在本仓不可用）（如何验证一条限制）
