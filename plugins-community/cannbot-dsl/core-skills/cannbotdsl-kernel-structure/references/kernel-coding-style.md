# CANNBotDSL Kernel 代码风格

> 本清单以 **channel-first** 为基准：核内 staging / 跨核 handoff 一律用 `Channel`。

## 三层职责分离

每个 kernel 文件按三层组织，每层只做一件事：

```
顶层 kernel 类 (@kernel) — 声明跨核 handoff channel，实例化 cube/vec 模块，编排多 stage 派发循环与循环坐标推导
中间 Matmul/Cube 类     — 声明核内 staging channel(L1/L0A/L0B/L0C) 与 copy engine，暴露 load_/compute_/store_
中间 Vector 类          — 声明 vec 内部单块 Buffer scratch 与同核产出 Channel，VF region 封装在 @jit 方法内
```

**反例**：VF 代码散落在顶层循环里做 per-row cast→add→cast；跨核交接自己写一个 `C2VUbHandoff` 同步类在读写代码之外传递 token。

**正例**：`Matmul.compute_qk` / `Vector.softmax_first` 各自封装一个 PIPE 相干单步，顶层 `__call__` 只按 stage 顺序调用它们；跨核交接就是一条 `Channel`，无独立同步类。

## Channel 声明位置由「连接谁」决定

这是最易错的一条。判据看这个 Channel 连接哪两方：

| Channel 用途 | 声明在 | 例（参考代码） |
|--------------|:------:|-----|
| 核内 staging（GM→L1→L0→L0C 中转） | **对应模块类** `__init__` | `Matmul.q_l1 / k_l1 / v_l1 / l0a / l0b / l0c` |
| vec 产出的同核 channel | **vec 模块类** `__init__` | `Vector.o_ub` |
| vec 内部跨迭代状态 / scratch | **vec 模块类**，用 `Buffer(MemLoc.UB, ...)` | `Vector.res_o / sm_max_tb / sm_sum_tb / sm_exp_tb` |
| 跨核 handoff | **顶层 kernel 类** `__init__`，作参数传进模块方法 | `flash_attention_kernel.qk_ub / pv_ub / p_l1` |

**反例**（把核内 staging 藏进模块类是对的，但把跨核 handoff 也塞进模块类是错的）：
```python
class Matmul:
    def __init__(self):
        self.qk_ub = Channel(MemLoc.UB, ..., depth=2)  # 错：跨核边归任一模块破坏封装
```

**正例**：核内 staging 归模块类，跨核 handoff 归顶层、传参进模块方法：
```python
class Matmul:
    def __init__(self, tile_cube_m, tile_n, tile_d):
        self.q_l1 = Channel(MemLoc.L1, shape=(tile_cube_m, tile_d), dtype=dtypes.float16, depth=2)   # 核内 staging，只本模块读写
        self.l0c  = Channel(MemLoc.L0C, shape=(tile_cube_m, tile_n), dtype=dtypes.float32, depth=2)
    def store_s(self, ub_ch):                               # 跨核 handoff 由编排层传入
        mem_copy(ub_ch, self.l0c, engine=self.fixpipe)

@kernel
class flash_attention_kernel:
    def __init__(self, ...):
        self.qk_ub = Channel(MemLoc.UB, ..., depth=2)  # 跨核边归顶层
        self.matmul = Matmul(...)
    def __call__(self, ...):
        self.matmul.store_s(self.qk_ub)                     # 编排层持有并传入
```

原因：① 跨核边连接两个模块，归属任一方都破坏「模块只管自己核内」；② 跨核 channel 的 `Σ depth ≤ 8` 是 per-kernel **全局**预算，必须在能看到所有 handoff 的编排层集中声明才数得清。

> **Buffer vs Channel**：跨迭代要多级缓冲 / sync 语义 → `Channel`；只是驻留累加器 / 临时 lane 向量、由同一段 vf 读改写 → `Buffer`（无 sync 开销）。参考代码里 `sm_max_tb`/`res_o` 是 Buffer，`p_ub`/`o_ub` 是 Channel。

## Matmul 不切分 D 维度

L0 缓冲按 `max(K, N)` 放宽尺寸，单次 `matmul` 处理完整的 M×K × K×N。

**反例**：
```python
for tile_idx in range(head_dim // CHUNK_SIZE):   # D=128 拆成两个 64
    ...matmul(l0c_tile, l0a_tile, l0b_tile)...
```

**正例**（参考代码 `compute_qk`：一次 matmul 完成全 tile）：
```python
def compute_qk(self):
    mem_copy(self.l0a, self.q_l1)        # Q L1->L0A
    mem_copy(self.l0b, self.k_l1)        # K L1->L0B
    matmul(self.l0c, self.l0a, self.l0b, init=True)
```
L0 channel 的 size 按 `max(tile_n, tile_d)` 放宽（参考代码 `tmp_n = max(tile_n, tile_d)`），避免为切 D 而拆多次 matmul。

## VF 操作数：owning UB 或直接读写 Channel，不进 tile_view 视图

raw vf 的操作数是 owning `Buffer`，或直接读写 `Channel`（`rr.vload(ch, off)` / `rr.vstore(ch, off, ...)`）。**不把 `tile_view` 切出的普通 Tensor 视图当作 Buffer 传递**。

**反例**：`tile_view` 切出一个 (1,D) 视图作为 raw vf 的 rhs。

**正例**（参考代码 `update_o`：raw vf 直接读写 owning UB `res_o` 与 channel `pv_ch`）：
```python
@jit
def update_o(self, pv_ch, sm_exp_buf):
    VL_T = 2048 // 32
    with vf(mode="raw"):
        for row in range(self.tile_vec_m):
            exp_b = rr.vload_brc(sm_exp_buf, row)
            base = row * self.tile_d
            for col in tuple(range(0, self.tile_d, VL_T)):
                off = base + col
                mask, _ = rr.update_mask(self.tile_d - col, elem_bits=32)
                o = rr.vmul(rr.vload(self.res_o, off), exp_b, mask=mask)  # res_o: owning UB
                o = rr.vadd(o, rr.vload(pv_ch, off), mask=mask)            # pv_ch: channel 直读
                rr.vstore(self.res_o, off, o, mask)
```

`tile_view` 只用于 MTEx `mem_copy` 的 GM 侧偏移（如 `finalize_o` 里 `tile_view(o_tile_gm, ..., (subblock_idx, 0))`），不进 VF。

## raw-vf 的 for 必须在 @jit 方法体内

只要方法体内有 **raw-vf 的 `for` 循环**，该方法必须直接是 `@jit` 函数体——raw-vf 的 `for` 不能藏进普通 helper，否则被 AST 全展开、静默 ~3.5× 退化。

- 参考代码 `softmax_first` / `softmax_rest` / `update_o` / `_finalize_div_vf` 都是 `@jit`。
- 无 vf `for` 的 per-row helper（如 `_pass_a_row` / `_softmax_fold_row`）保持普通函数，由 `@jit` 方法体内的 `for row` 循环调用——**循环本身留在 `@jit` 体里**，helper 只做单行、不含循环。
- 纯 channel-first 的 `mem_copy`/`matmul` 步（`load_`/`compute_`/`store_`）不加 `@jit`。
- `finalize_*` 常拆两块：内层 vf 除法单独 `@jit`（`_finalize_div_vf`），外层普通包装做 `cast` + 写 channel/GM（`finalize_o`）。

## host 侧保持精简

- 不手工拼 `sys.path`，交给 `conftest.py`。
- host wrapper 就是一个薄函数/薄类：shape 校验 → 布局归一 → launch kernel → sync。参考代码的 `FlashAttention.run` 只做 layout 归一（`_bsnd_to_bnsd`）+ 取 tile config + launch。
- 不把 `_check_inputs`、`allocate_workspace`、`_launch_single` 拆成独立方法再包一层 class。

**反例**：
```python
class KdaStage3:
    def allocate_workspace(self, K): ...
    def _check_inputs(self, ...): ...
    def _launch_single(self, ...): ...
    def run(self, ...):
        ...self._check_inputs(...); self._launch_single(...)
```

**正例**：
```python
@jit
def run(self, attn_out, query, key, value, scale, attn_mask, ...):
    D = query.shape[3]
    tile_cube_m, tile_vec_m, tile_n, tile_d = get_tile_config(D)
    op = flash_attention_kernel(tile_cube_m=tile_cube_m, ...)
    op[36](attn_out, query, key, value, scale, attn_mask, ...)
```

## 循环剥离会制造 channel 第二读作用域

想让"多数迭代走便宜路径、少数走贵路径"时，**用循环体内的运行时分支，不要剥离循环**。把循环拆成"便宜的 n-1 轮 + 尾部一次贵的调用"，会让常驻 channel（如 read-many 的 `q_l1`）变成一写 + 两个读作用域，`cannir-resolve-channel-operands` 直接拒绝。**而在单个循环体内部放运行时 `scf.if` 选择两条路径是合法的** —— channel 仍是一写、一个包围读作用域。这个报错是循环嵌套的问题，不是分支的问题 —— 看到它别去掉分支，去看谁被拆出了第二个读作用域。附带好处：分支两侧各自是直线 vf region，正是 VF 折叠喜欢的形状。详见 `../SKILL.md` 陷阱 8。

## 验证顺序

```bash
python -m py_compile <file.py>
pytest <case> -q
pytest <npu_case> -q
```
