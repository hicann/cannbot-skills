---
name: cannbotdsl-tiling-design
description: "设计 CANNBotDSL 多级 tiling 策略时使用。CANNBotDSL 的 tiling 用 tile_view + local_slice + ceil_div 循环，与其他框架不同。当需要设计 GM→L1→L0→UB 多级切分、处理 tail block（非对齐残差）、或用 tile_view/local_slice 做分层切分时触发。含按算子类型的 tiling 模式库（Matmul M/N/K、Reduction axis 切分、Attention 三维、Elementwise 平铺）、以及用 Dim/TensorSpec/multiple_of 设计动态 shape tiling。Triggers: cannbotdsl tiling, tile_view, local_slice, tail block, 多级 tiling, ceil_div, Dim tiling。Architect 在 Stage 2 调用。"
---

# cannbotdsl-tiling-design

CANNBotDSL Tiling 策略设计指南。Architect 在 Stage 2 使用。所有符号/签名摘自源码。

## 1. `tile_view` + tail block

**真实签名**：

```python
def tile_view(input: Tensor, tiler: Tiler, coord: Coord) -> Tensor:
```

- `input`：源 memref（主用 GM），也接受 `Channel`（返回子 tile view channel）。
- `tiler`：tile 形状，`Tiler = Union[Shape, Tile]`（int / int-tuple / 嵌套 / 含 SSA 的混合 tuple）。
- `coord`：各维块坐标；常量直接写 Python `int`，动态 i64 SSA 直接传入。只有从其他位宽或 raw/index SSA 显式转换到 i64 时才使用 `Int64(value)`。

真实调用点（dense FA 范式；完整可跑文件见文末「真实调用点」）：

```python
q_tile_gm = tile_view(query_slice, (self.tile_cube_m, self.tile_d), (m_idx, 0))
k_tile_gm = tile_view(key_slice,   (self.tile_n,      self.tile_d), (n_idx, 0))
# 常量坐标保持为普通整数，嵌套 tile 也成立：
w_chunk   = tile_view(gm_w, (CHUNK_SIZE, head_dim), (linear, 0))
gamma_col = tile_view(gamma_col_full, (CHUNK_SIZE, 1), (subblock_idx, 0))
```

**尾块自动处理**：`LocalTileOp::inferReturnTypes` 对每个 tile leaf 计算 actual extent，不镜像 nominal tile：

```
pDim=父维; tDim=tile维; cVal=坐标
if any dynamic:          actualDim = kDynamic
elif cVal*tDim >= pDim:  return failure()          # 静态越界 verify 拒绝
else:                    actualDim = min(tDim, pDim - cVal*tDim)
```

- 静态尾块：父 `(100,50)` / tile `(16,16)` / coord `(6,3)` → result `<(4,2),(50,1)>`（不是 `(16,16)`）。
- 动态 coord + 静态父 → shape `kDynamic`，DMA codegen emit `arith.minui` SSA 链；静态可整除时 folder 折成常量，零开销。
- **约束**：NZ/ZN-formatted memref 上，tile leaf 必须整除 M0/N0，否则 verifier 报错；跨 fractal 边界改用 `local_slice`。

## 2. `local_slice`（核内切分）

真实可跑代码用 `local_slice`对**核内 buffer**（UB/L1/L0X，拒绝 GM）做静态 offset view：

```python
local_slice(buffer, tiler, stride=None, offset=0)
```

- `offset`：相对父 base 的静态字节偏移（须 ≥0、elemBytes 整数倍；L1/L0X 还须 512B fractal 整数倍）。
- 对齐语义：UB innermost `ceil_to(32/elemBytes)`；L1/L0X outermost `ceil_to(m0=16)`、innermost `ceil_to(n0=32/elemBytes)`。
- result 的 logical shape = 用户 tiler 真实 extent（不对齐）；physical shape 才带 padding。

真实的行/列 view + stride/offset（gated delta rule 范式；完整可跑文件见文末「真实调用点」）：

```python
src_row    = local_slice(g_col, (1, C), offset=0)
src_as_col = local_slice(src_row, (C, 1), stride=(0, 1), offset=(C - 1) * ELEM_F32)
tmp_col    = local_slice(tmp_mat, (C, 1), stride=(1, 1), offset=0)
```

> `buf_slice`（签名 `buf_slice(self, tiler, coord=None)`）是 `local_slice` 的计划继任统一抽象（stride packed 重算、coord 可选），**当前 whl 未提供**。写新代码用 `local_slice`。

**与 `tile_view` 区别**：`tile_view` 作用于 GM memref layout、继承父 trailing stride、coord 必填；`local_slice` 作用于核内 buffer、可 packed 重算 stride、coord/offset 可缺省。

## 3. 分层切分

活跃分层切分通过 `tile_view` + `ceil_div` 循环实现。以下三种 divide（`zipped_divide`/`tiled_divide`/`flat_divide`）及底层 layout 代数原语（`coalesce`/`composition`/`complement`/`logical_divide`/`logical_product`）**当前 whl 未提供 Python 入口**；如需等价语义，用 `tile_view` + `ceil_div` + `idx2crd` 循环替代：

| 切分模式 | 等价实现 | 用途 |
|---------|---------|------|
| 保留 tile 块结构 | `tile_view` + 外层单循环 `for m in range(ceil_div(M, tile_m))` | 外层单循环 |
| 按 rest 各维独立 | `tile_view` + `idx2crd(idx, [dim0, dim1, ...])` 分解多维坐标 | 外层多维独立循环 |
| 纯 flat 线性遍历 | `tile_view` + `ceil_div(total, tile)` 单层循环 | 线性遍历 |

> **替代原则**：`tile_view` 尾块感知保证非整除边界只算 actual extent，覆盖 divide 系列的尾块语义。`ceil_div` 计算总 tile 数，`idx2crd` 分解多维坐标。

## 4. 多级 tiling 模式库（源自真实 kernel）

- **Matmul（M/N/K split）**：tile `(tile_m, tile_k)` / `(tile_k, tile_n)`。映射 `a_gm →(nd2nz) a_l1 → l0a`，`b_gm →(nd2nz, transpose) b_l1 → l0b`，`matmul(l0c, l0a, l0b)`，`l0c →(fixpipe) UB`。输出用 `tile_view(out_gm, (tile_vec_m, tile_n), (subblock_idx, 0))` 按 vector subblock 二次切 M。
- **Attention（3-dim）**：外层 `idx2crd(tile_idx, [batch, head, seq_tile_num])` 解三维块坐标；Q 常驻 L1 切 `(tile_cube_m, tile_d)`，K/V 按 `(tile_n, tile_d)` 逐 tile 加载。cube tile 与 vector tile 分离（`tile_cube_m` vs `tile_vec_m`）。
- **Reduction（axis split）**：用 `local_slice` 沿单轴取行/列 view（`(1,C)` 行、`(C,1)` 列）配 stride 做逐轴 reduce/scan。
- **Elementwise（flat）**：`tile_view` + `ceil_div(total, tile)` 线性遍历；`tile_view` 尾块感知保证非整除边界只算 actual extent。

storage 容量按 physical layout 决定，逻辑 tiling 按 logical shape；Buffer 只有一个物理 slot，Channel 的每槽间距按 physical addressable extent（ND stride 下为 `stride[0]*shape[0]`）核算。

### 4.1 tile_view 维度限制

**4D/3D `tile_view` 视图传播到 `matmul` 时触发 shape inference 失败**：编译期报 `typed-region writer has no registered access-shape rule`。根因是多维 tile_view 切出的视图（3D/4D）经 `nd2nz` → 2D channel 后破坏 matmul 的 shape 推断。

**规避**：host 侧 4D→2D 展平，kernel 内只用 2D `tile_view`。

```python
# 错误：4D tile_view 传播到 matmul 失败
q_tile = tile_view(q_gm, (1, 1, 128, 128), (bi, hi, mi, 0))
mem_copy(a_l1, q_tile, engine=nd2nz)
mem_copy(l0a, a_l1)
matmul(l0c, l0a, l0b)   # ← shape inference 失败

# 正确：host 侧展平为 2D，kernel 内只用 2D tile_view
# host: q_flat = q.reshape(B * H_q * S, D)
q_tile = tile_view(q_gm, (128, 128), (tidx, 0))
mem_copy(a_l1, q_tile, engine=nd2nz)
mem_copy(l0a, a_l1)
matmul(l0c, l0a, l0b)   # ← 2D 正常
```

**`idx2crd` + runtime 整数除法限制**：SSA 整数除法（`//`）不被 `idx2crd` 下游支持。分解维度时应避免运行时除法。

```python
# 错误：runtime // 不被支持
bi, hi, mi = idx2crd(tidx, [B, H_q, num_m])
ki = hi // g   # ← SSA 除法失败

# 正确：分解为 [H_kv, g, num_m]，ki 直接获得
ki, gi, mi = idx2crd(tidx, [H_kv_total, g, num_m])
hi = ki * g + gi   # 乘法+加法，无除法
```

## 5. 从单 case 扩到全 case 矩阵：先判「补丁还是重写」

把一个跑通的单 shape 算子扩到 benchmark 全量 case（十几到几十个 shape × dtype × 开关组合）时，最贵的错误是**逐 case 打补丁**——每加一个 case 就在 kernel 里加一个分支，最后代码没人能改。

**先做特性矩阵，再决定形态。** 用脚本把每个 case 的关键量算出来（不要手算，也不要凭 shape 目测）：

```python
for case in cases:                          # 从 cases.csv / cases.yaml 直接解析
    scores_bytes = SV * SKV * 4             # 半高 fp32 分数块（split-M 后）
    kv_bytes     = 2 * SKV * D * dtype_bytes    # K+V 常驻 L1
    tags = []
    if scores_bytes > UB_CAP * 0.5: tags.append("必须 M 分块")
    if kv_bytes     > L1_CAP:       tags.append("必须 KV 分块")
    ...                                     # dtype / causal / D / S!=Skv 各一列
```

**判据：数「必须」类标签的数量与耦合关系。**

- 各 case 的需求**彼此独立**（如只差 dtype、只差 causal 开关）→ 参数化即可，不必重写。
- 出现**互相咬合的片上约束** → 必须重写成通用形态。本仓 GQA 实例：`scores 2 MB > UB 256 KB`（case 8）⇒ 必须 M 分块；`K+V 1 MB > L1 512 KB`（case 11-13/15）⇒ 必须 KV 分块；而 **KV 分块一旦引入，行归一化就跨块，softmax 必须改成 online 修正**。三条锁死，逐 case 打补丁做不到。

**通用形态的一个反直觉好处**：online softmax **包含**不分块的情形（`NKB == 1` 时 `alpha = exp(-BIG - m) = 0`，递推退化为一次性 softmax）。所以写一份通用 kernel 往往比「通用路径 + 小 shape 特例路径」更省事、也更少出错——**不要为小 shape 单开一条分支**。

**动手前先用脚本核算分块可行性**，一次覆盖全部 case：

```
case  D   BM  SV |    UB     L1  L0A   L0C | 结论
  5  256  64  32 |  90KB  176KB 32KB  96KB | OK      # D=256 时 BM 自动降到 64
  8  128 128  64 | 116KB  128KB 32KB 128KB | OK
```

`BM` 随 `D` 自适应（`BM*D*dtype_bytes` 要装进 L0A），`BN` 固定即可。**全部 OK 才动手写码**——本轮 20 个 case 一次核算全过，避免了写完才发现某个 shape 装不下。若某行 OVER，先调 BM/BN 或考虑 L0 别名复用（见 `../cannbotdsl-op-design/SKILL.md` §2.0）。

> **一句提醒**：可行性核算过了 ≠ 数值就对。通用化重写会同时引入 online 递推、跨块状态、raw-VF 等多个新面，**每一面都要单独验证**（见 `../../debug-skills/cannbotdsl-precision-debug/SKILL.md` 调试原则 8/9）。本轮实测：分块策略一次核算全过，但数值调试仍花了远超预期的时间。

## 6. 动态 shape tiling

用 `Dim` 描述动态维身份和约束，用
`TensorSpec` 描述 AOT tensor 契约：

```python
M = cannbotdsl.Dim("M", min=64, multiple_of=64)
a_spec = cannbotdsl.TensorSpec((M, 64), dtypes.float16)
c_spec = cannbotdsl.TensorSpec((M, 128), dtypes.float32)
```

- `Dim(name, min=1, max=None, multiple_of=1)` 的同名声明在一次 `.compile()` 中统一；
  约束不一致会拒绝编译。
- shape/stride 可使用 `+`、`-`、`*` 和正整数除数的 `//`，例如 `B * S`、
  `S // 64`。每个 `Dim` 必须至少在一个槽位裸出现，以便从实参直接绑定；框架不反解方程。
- `Dim` 及其派生表达式是调用契约，不是 kernel 内的 IR 值，**不能直接用于
  `range()` 或控制流**。

**关键模式**：kernel 体内用 `ceil_div(tensor.shape[i], tile)` 从 tensor shape 动态算 tile 数：

```python
@kernel
def matmul_kernel(gm_a, gm_b, gm_c):
    M_TILES = ceil_div(gm_a.shape[0], 64)     # 动态 tile 数，不能用 self.M_TILES
    for m in range(M_TILES): ...

M = cannbotdsl.Dim("M", multiple_of=64)
a_spec = cannbotdsl.TensorSpec((M, 64), dtypes.float16)
b_spec = cannbotdsl.TensorSpec((64, 128), dtypes.float16)
c_spec = cannbotdsl.TensorSpec((M, 128), dtypes.float32)
fn = matmul.compile(a_spec, b_spec, c_spec)               # 一编多执行
```

尾块靠 §1 的 `min` SSA 兜底；只有契约声明 `multiple_of=tile_size` 时，才可假定
动态维整除 tile。

## 7. 真实 tiling 代码片段

Matmul 全链路 tiling + buffer 映射：

```python
nd2nz   = make_copy_engine(format_transform="nd2nz", dtype=dtypes.float16, pad_value=0.0)
fixpipe = make_copy_engine(dtype=dtypes.float32, dual_dst_ctl=1)
subblock_idx = get_subblock_id()

mem_copy(a_l1, a_gm, engine=nd2nz)          # GM → L1 (ND→NZ)
mem_copy(l0a, a_l1)                          # L1 → L0A
mem_copy(l0b, b_l1, transpose=True)          # L1 → L0B (transpose → ZN)
matmul(l0c, l0a, l0b)                        # L0A × L0B → L0C
mem_copy(cv_ub, l0c, engine=fixpipe)         # L0C → UB (fixpipe NZ→ND)
out_half = tile_view(out_gm, (self.tile_vec_m, self.tile_n), (subblock_idx, 0))
mem_copy(out_half, out_ub)                   # UB → GM (subblock 二次切 M)
```

FA 三维块坐标 + 分层 tile_view：

```python
batch_idx, head_idx, m_idx = idx2crd(tile_idx, [batch_size, head_num, seq_tile_num])
query_slice = query[batch_idx, head_idx, None, None]         # (1,1,S1,D)
q_tile_gm = tile_view(query_slice, (self.tile_cube_m, self.tile_d), (m_idx, 0))
for n_idx in range(ceil_div(seqlen_k, self.tile_n)):
    k_tile_gm = tile_view(key_slice, (self.tile_n, self.tile_d), (n_idx, 0))
```
