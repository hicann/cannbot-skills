---
name: index-computation
description: 索引计算类算子（Index / IndexPut / Gather / Scatter / EmbeddingDenseBackward / Sort / TopK / LightningIndexer）的 Triton Ascend 优化经验合集，按算子分章节组织，含通用经验 + 各算子专属约束/骨架/kernel
metadata:
  type: reference
---

# 索引计算类算子优化经验

本文档合并索引计算类算子的优化经验。按以下结构组织：
- **§1 通用经验**：跨算子重复的工程约束（已提取，各算子章节不再重复）
- **§2 Index / Gather**（index-read，按索引读取）
- **§3 IndexPut / Scatter**（index-write，按索引写入/累加）
- **§4 EmbeddingDenseBackward**（index-accumulate，按索引累加）
- **§5 Sort / TopK / 选位**（index-sort / topk-select，排序/选择/top-K 选位；含 LightningIndexer）
- **§6 常见陷阱**（按算子分小节）
- **§7 参考算子列表**

---

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| Index / Gather | `index-read` | 根据索引从输入中收集元素，核心计算量低于访存/重排 | 维度特化 + UB 预载 + `tl.gather`；非 last-dim 评估转置 |
| IndexPut / Scatter | `index-write` | 根据索引向输出写入或原子累加，存在写冲突风险 | 并行轴与连续访问对齐 + 向量 `tl.atomic_add` + 冲突处理 |
| EmbeddingDenseBackward | `index-accumulate` | 按索引将梯度累加到 embedding 权重，尾维度宽且重复索引多 | 三阶段 kernel + 连续维向量 atomic + fp32 中间累加 |
| Sort / TopK | `index-sort` | 基于比较重排并返回索引 | 排序网络/分桶 + 向量比较原语 + 注意索引 dtype 对齐 |
| LightningIndexer | `topk-select` | 轻打分（score = Σ w·ReLU(q·k)）+ 重 top-K 选位 + gather 有序索引/值；select 成本 ≥ 打分成本 | 选位结构优先（sort-compaction 替代逐元素前缀和）+ `npu_sort_v2` 双角色 + score 链融合 |

> ⚠️ **关键区分**：本类别的核心优化哲学是 **按维度特化分派 + 向量化读写/原子**。Gather/Index 读取类走 **UB 预载 + `tl.gather`**，Scatter/IndexPut/EmbeddingDenseBackward 写入累加类走 **连续维向量 `tl.atomic_add`**，Sort/TopK 排序选择类走 **排序网络/分桶**。生成时**禁止混用**其他类别经验（如不要在 Scatter 里套用 Gather 的 UB preload 技巧）。

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下约束是索引计算类算子**共有**且**未在其他 template 文件覆盖**的工程约束。其他文件已提取的通用约束（如 `tensor-transform.md` 的 G1 动态 num_cores / G2 pow2 BLOCK / G4 grid 不超核数 / G7 contiguous 等）此处不再重复，各算子章节引用时标注。

### G1 并行轴选择：与连续访问对齐，而非与最大 numel 对齐

- **必须**按算子语义选择并行轴，使内层访问沿 trailing/E/D 连续维度。
- **Gather/读取类**：优先沿 `outer`（输出张量的外层维度）和 `index_len` 并行；每个 program 负责一条或多条 `(o, k)`，沿 `trailing` 连续读取/写入。
- **Scatter/写入类**：
  - 若 `trailing`/`E` 较短且索引重复可控：按 `outer * index_len` 并行，每个 program 写入一段连续输出位置（输出位置并行）。
  - 若 `trailing`/`E` 很宽（如 `E ≥ 128`）：按 `index`/`N` 分块并行，每个 program 对连续 `E` 维做**向量 atomic_add**。
- 当 `index_len` 很小但 `outer * trailing` 很大时，启用 **tile-based** 扩展，用 2D grid 占满 vector cores。
- **Why:** 索引算子核心计算量低于访存/重排，并行轴若不与连续访问对齐，会引入离散全局访问和原子竞争，性能通常跌至 0.1x 以下。

### G2 向量化读写与 Coalesced Memory Access

- **必须**对 `trailing`/`E`/`D` 连续维度，使用 `tl.arange(0, BLOCK)` + mask 进行向量 `tl.load`/`tl.store`。
- **必须**对索引张量本身也按 `BLOCK_M` 向量加载：`tl.load(index_ptr + offs, mask=mask).to(tl.int32)`。
- **Gather 中**，若索引沿连续维度排列，优先一次加载一段索引并配合硬件 gather；否则确保 indirect load 的目标地址尽可能连续。
- **Why:** 标量加载索引或标量 atomic 会破坏 SIMD，原子操作数量与元素数量相同，典型加速比从 0.3x~0.6x 跌至 0.05x~0.15x。

### G3 维度特化与 `tl.constexpr` 分发

- **必须**在 `ndim ≤ 4` 时，在 **host 侧**按 `dim=0/1/2/3` 启动独立 kernel 实例，将 `DIM`/`NDIM` 作为 `tl.constexpr` 传入。
- **禁止**在 kernel 内用运行时 `for j in range(NDIM)` 解码坐标，或在 kernel 内保留运行时 `if dim == X` 分支。
- **Why:** 运行时坐标解码会引入大量标量除法/取模、非合并访存，且无法使用 UB gather，典型加速比仅 0.05x~0.2x。
- **How to apply:** 见 §2.1 L1.1、§3.1 L1.3、§4.1 L1.2。

### G4 索引/偏移数据类型

- **必须**将索引值尽快转换为 `tl.int32`：`.to(tl.int32)`。
- **必须**在计算最终全局地址时，对可能溢出的扁平偏移使用 `tl.int64`（如 `base + batch_offset * stride`）。
- **禁止**在标量上做 `.to(tl.int64)` 或在 int64 上做 `%`；需要取模时改用 `idx - (idx // M) * M`。
- **Why:** int64 标量会触发地址计算降级；NPU 上 int32 索引更高效，但扁平偏移可能溢出需用 int64。

### G5 原子操作粒度

- **必须**在连续输出段 `E ≥ 64/128` 时使用向量 `tl.atomic_add(ptr + arange, vals, mask=...)`。
- **禁止**把 `tl.atomic_add` 放在 `for j in range(BLOCK_E)` 标量循环里，或用 `tl.get_element(..., [j])` 拆成标量后再 atomic_add。
- **Why:** 向量 atomic 一次覆盖多个连续元素，原子数下降 1~2 个数量级；拆成标量会破坏 SIMD 并爆炸原子数。
- **How to apply:** 见 §3.1 L1.2、§4.1 L1.1。

### G6 精度保持

- **必须**在 fp16/bf16 累加场景，分配 `torch.float32` 中间 buffer 进行 `atomic_add`，最后 cast 回原始 dtype。
- **必须**在涉及 `scale_grad_by_freq` 等加权累加时，先用独立 O(N) 频数 kernel 统计，主 kernel 查表缩放，禁止二次扫描。
- **Why:** fp16/bf16 `atomic_add` 直接累加会导致精度下降；频数统计与主累加分拆避免热循环内分支。
- **How to apply:** 见 §4.1 L1.3。

### G7 Host 侧与拷贝

- **必须**避免 `x.contiguous()`、`permute().reshape()` 等隐式拷贝；如必须转置，估算收益 `gather_gain > 2 × transpose_cost` 才执行。
- **必须**将所有传入 Triton kernel 的可选张量实例化为合法 dummy tensor，禁止传 `None`。
- **必须**将排除位置（如 `padding_idx`）清零、边界修正等作为**独立轻量 kernel**，不在热 scatter 循环内分支。
- **Why:** 隐式拷贝可能抵消 gather 收益；`None` 指针会导致 JIT 签名错误；热循环内分支破坏向量化。

### G8 GPU-source 迁移到 Ascend NPU 的 memory/index 改造检查清单

当参考代码来自 GPU/CUDA Triton 实现时，必须在生成 Ascend NPU 版本前完成以下迁移检查。这些项是**强制前置条件**，不应在 Phase 3/Phase 4 中才被修复。

| 检查项 | 迁移规则 | 反例（必须避免） |
|--------|----------|------------------|
| 设备与启动配置 | `cuda` → `npu`；移除 `num_warps`、`num_stages`、`cache_modifier`、`evict_last` 等 GPU-only 启动选项。 | `triton.Config({}, num_warps=4, num_stages=2)`、`tl.load(..., cache_modifier='.cg')` |
| `tl.load`/`tl.store` 参数 | 禁用 GPU-only 参数；明确指定 `care_padding=False` 当且仅当访问不越界，避免默认 padding 带来额外拷贝。 | 保留 `boundary_check`、`padding_option` 等 CUDA 参数 |
| 索引数据类型 | 索引张量进入 Vector 路径前统一转为 `tl.int32`；全局扁平偏移按需使用 `tl.int64`。禁用 `fp64`、`uint` 类型。 | 用 `int64` 索引做数组下标、在 kernel 内保留 `torch.int64` |
| 累加数据类型 | fp16/bf16 scatter/accumulate 必须分配 `torch.float32` 中间 buffer，最后 cast 回原始 dtype。 | 直接用 `tl.atomic_add` 累加 fp16/bf16 输出 |
| Block Pointer / 转置 | **禁止**通过交换 stride 实现 transpose；使用 `order=(...)` 或显式 host 侧 `permute().contiguous()`。 | `x_ptr + col * stride[1] + row * stride[0]` 假装二维连续 |
| Memory 布局决策 | 转置收益必须大于 2× 拷贝成本：`gather_gain ≈ saved_strided_loads × memory_latency`，`transpose_cost ≈ 2 × tensor.numel() × elem_bytes / copy_bw`。 | 无条件 `.contiguous()` 或 `.movedim(...).contiguous()` |
| Atomic 操作迁移 | 连续输出段优先向量 `tl.atomic_add(ptr + tl.arange(0, BLOCK_E), vals, mask=...)`；标量 atomic 仅作为不连续/冲突密集 fallback。 | `for e in range(BLOCK_E): tl.atomic_add(...)` |
| UB 容量与 tiling | UB preload 尺寸 `ROW_SIZE` 必须小于 UB 容量上限（fp32 约 32768 元素）；过大时分块为 `BLOCK_*` + `BLOCK_*_SUB`。 | 一次性预载整行但 `dim_size > UB_LIMIT` 导致 spilling |

> **判定口诀**：GPU 代码迁移后，若 kernel 内仍出现 `num_warps`、`cache_modifier`、int64 索引下标、stride 欺骗 transpose、或标量 atomic 内层循环，则迁移未完成。

### G9 语义正确性

- **必须**在 `accumulate=False` 且存在重复索引时，实现 **last-write-wins**（单 block 串行或两阶段）。
- **必须**覆盖边界检查：`index < 0`、`index >= dim_size`、`padding_idx` 等情况；越界读返回 0，越界写加 mask。
- **必须**将排除位置（padding_idx 等）通过独立后置 kernel 清零，不在热循环内判断。
- **Why:** 缺少写冲突处理会导致结果不稳定；越界访问触发 ACL 错误；热循环内判断破坏性能。

### G10 Autotune 与调试

- **必须**在累加/原地修改算子 autotune 时，每次重新 clone/初始化输出 buffer，或关闭 autotune。
- **必须**以目标维度（如 `K`、`E`、`D`）作为 autotune key，避免无关维度导致 config 错配。
- **出现 ACL 507035 时**优先检查是否未使用 UB/gather 或运行时 dim 分支。
- **Why:** 原地修改算子在 autotune 时重复累加同一 buffer，会导致结果错误且性能失真。

### G11 Grid 与并行规模

- **必须**根据实际 tile 数量计算 grid，配合 kernel 内 `for b in tl.range(pid, total, num_programs)` 做 strided 遍历。
- **必须**保证 1D grid 硬上限 **65535**；超出时使用 2D grid 或增大 BLOCK。
- **必须**让 BLOCK 大小取 `next_pow2_geq(dim)` 且不超过实际维度；避免 `BLOCK=256` 等固定大 block 在单 program 中循环过多而超时。
- **Why:** `grid = (min(num_cores, total_blocks),)` 后直接退出会浪费并行度；1D grid 超限导致启动失败；BLOCK 过大触发 Vector Core timeout（507034）。

---

## §2 Index / Gather 算子（index-read）

**算子类别**: `index-read`（按索引从输入中收集元素；`index_select` / `gather` / `take` / `embedding` 等）
**典型特征**: 索引张量决定数据读取位置，多为 `int32`/`int64`；数据张量多为 `fp16`/`bf16`/`fp32`，维度 1D~4D；核心风险是离散全局内存访问与非连续维度访问
**性能基准**: Level1 索引计算类算子旧版本均能达到 ≥0.6x 几何平均加速比 vs torch

### §2.0 首次生成必读：为什么必须把主要框架写对

Index / Gather 是**访存模式高度依赖 dim 位置**的算子：last-dim 可走 UB 预载 + `tl.gather`，非 last-dim 需评估转置收益，通用 ND kernel 性能极差。**首次生成如果把框架写偏（例如用通用 ND kernel 统所有路径、未将 gather dim 转置到 last-dim、循环边界用输入 shape），后续迭代很难通过局部修 bug 把性能救回来**——典型加速比会从 0.6x 跌至 0.05x~0.2x。

本章按 Layer 1→3 组织：
- **L1 是硬性约束**，首次生成必须全部满足；
- **L2 是 host 分派骨架 + 辅助函数**，必须一次写对；
- **L3 是关键 kernel 实现**，贴出优化的重点代码和那些不容易首次生成就生成出来的代码。

### §2.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 Host 侧必须按 `dim=0/1/2/3` 启动 `tl.constexpr` 特化 kernel

- **必须**在 host 侧将 `dim` 归一化，并按 `DIM=0/1/2/3` 启动独立 kernel 实例，将 `DIM`/`NDIM` 作为 `tl.constexpr` 传入。
- **禁止**在 kernel 内用运行时 `for j in range(NDIM)` 解码坐标，或用运行时 `if dim == X` 分支。
- **Why:** 运行时坐标解码会引入大量标量除法/取模、非合并访存，且无法使用 UB gather，典型加速比仅 0.05x~0.2x。
- **How to apply:** 见 §2.2 L2.1 分派决策树。

#### L1.2 Gather 必须优先走 UB 预载 + `tl.gather`

- **必须**在 `dim` 对应 last-dim 且 `dim_size ≤ UB_LIMIT` 时，走 **UB 预载整行 + `tl.gather(row, idx, axis=0)`**。
- **必须**在非 last-dim gather 且 `dim_size ≤ UB_LIMIT` 且转置收益 > 2× 拷贝成本时，host 侧 `movedim(dim, -1).contiguous()` 将 gather 维转置到最后，再走 UB 预载 + `tl.gather`，最后转置回。
- **禁止**将所有 gather 都实现为 `x_ptr + idx * stride` 的 direct-indirect 路径。
- **Why:** UB 预载 + `tl.gather` 是 Ascend 上的硬件 gather 路径；direct-indirect fallback 离散全局读取，典型加速比仅 0.05x~0.2x。
- **How to apply:** 见 §2.3 L3.1 `gather_ub_kernel`。

#### L1.3 循环边界必须基于输出/索引张量的 shape

- **必须**使用 `M = index.shape[:dim].prod()` 等基于输出/索引张量的 shape 计算循环边界。
- **禁止**使用 `M = x.shape[:dim].prod()` 等基于输入 shape 的边界。
- **Why:** gather 的输出位置由索引决定，用输入 shape 会导致越界或漏算。

#### L1.4 UB 预载尺寸必须校验 UB 容量上限

- **必须**保证 `tl.gather` 的 `ROW_SIZE` 小于 UB 容量上限（fp32 约 32768 元素）。
- **必须**在 `dim_size > UB_LIMIT` 时分块为 `BLOCK_*` + `BLOCK_*_SUB`。
- **Why:** 一次性预载整行但 `dim_size > UB_LIMIT` 会导致 spilling，反而更慢。

#### L1.5 转置收益必须大于 2× 拷贝成本

- **必须**先估算转置收益，仅当 `gather_gain > 2 × transpose_cost` 时才 host 侧转置。
- **Why:** 无条件 `.contiguous()` 或 `.movedim(...).contiguous()` 会引入额外 memcpy，可能抵消 gather 收益。
- **How to apply:** 见 §2.2 L2.2 转置收益估算。

### §2.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧分派决策树

1. Host 侧将 `dim` 归一化，并按 `DIM=0/1/2/3` 启动 `tl.constexpr` 特化 kernel。
2. 若 `dim` 为 last-dim 且 `dim_size ≤ UB_LIMIT`：走 **UB 预载 + `tl.gather`**。
3. 若 `dim != last` 且 `dim_size ≤ UB_LIMIT` 且转置收益 > 拷贝成本：先转置到 last-dim，再走 UB 预载 + `tl.gather`，最后转置回。
4. 仅当 UB 路径不可行且 shape 较小时，使用 **direct-indirect fallback**。

#### L2.2 转置收益估算

对非 last-dim gather，转置收益估算：

```text
transpose_cost ≈ 2 * tensor.numel() * elem_bytes / copy_bw
gather_gain    ≈ saved_strided_loads * memory_latency
仅当 gather_gain > 2 * transpose_cost 时转置
```

### §2.3 Layer 3: 关键 kernel 实现（优化重点与易错代码）

#### L3.1 `gather_ub_kernel`（UB 预载 + `tl.gather`）

适用条件：`dim` 对应 last-dim，且 `dim_size ≤ UB_LIMIT`。

```python
@triton.jit
def gather_ub_kernel(
    x_ptr, idx_ptr, out_ptr,
    M, K, T,
    UB_LIMIT: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    if pid_m >= M or pid_k >= K:
        return

    # UB 预载一行输入（last-dim）
    row = tl.load(x_ptr + pid_m * T + tl.arange(0, UB_LIMIT), mask=tl.arange(0, UB_LIMIT) < T)
    idx = tl.load(idx_ptr + pid_m * K + pid_k).to(tl.int32)

    # 沿 trailing 维度分块 gather
    for t_start in range(0, T, BLOCK_T):
        t_offs = t_start + tl.arange(0, BLOCK_T)
        mask = t_offs < T
        gathered = tl.gather(row, idx * T + t_offs, mask=mask)
        tl.store(out_ptr + (pid_m * K + pid_k) * T + t_offs, gathered, mask=mask)
```

> 非 last-dim 场景：先评估转置收益，必要时 host 侧转置到 last-dim 后再调用本模板。

### §2.4 Index / Gather 性能基准

| 维度 | cases | 加速比区间 | 备注 |
|------|-------|-----------|------|
| 参考 Level1 索引计算类算子 | 18_Index / 20_Gather | ≥0.6x | 旧版本几何平均加速比 vs torch；新版本若丢失 UB gather/维度特化普遍跌至 0.05x~0.4x |

**关键结论**:
1. last-dim gather 必须走 UB 预载 + `tl.gather`，direct-indirect 仅作为 fallback。
2. `ndim ≤ 4` 时必须编译期维度特化，禁止运行时 ND 坐标解码。
3. 循环边界必须基于输出/索引张量 shape，而非输入 shape。

---

## §3 IndexPut / Scatter 算子（index-write）

**算子类别**: `index-write`（按索引向输出写入或原子累加；`index_put_` / `scatter` / `scatter_add` 等）
**典型特征**: 输出张量形状固定，写入位置由索引映射；存在写冲突、原子操作粒度、坐标解码开销
**性能基准**: Level1 索引计算类算子旧版本均能达到 ≥0.6x 几何平均加速比 vs torch

### §3.0 首次生成必读：为什么必须把主要框架写对

IndexPut / Scatter 是**并行策略和原子粒度强相关**的算子：连续宽度 `E/D` 较大时必须用向量 `tl.atomic_add`，`ndim ≤ 4` 时必须编译期维度特化，重复索引时需要冲突处理。**首次生成如果把框架写偏（例如用标量 atomic 内层循环、通用 ND kernel 运行时解码、未处理 accumulate=False 的重复索引），后续迭代很难把性能救回来**——典型加速比会跌至 0.05x~0.15x。

### §3.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 连续宽度 `E/D ≥ 64` 时必须使用向量 `tl.atomic_add`

- **必须**在连续输出段 `E ≥ 64/128` 时使用向量 `tl.atomic_add(ptr + arange(0, BLOCK_E), vals, mask=...)`。
- **禁止**先对索引/值做向量 `tl.load` 得到 `idx_tile`/`val_tile`，再用 `tl.get_element(..., [j])` 拆成标量，然后对每个元素单独调用 `tl.atomic_add(output_ptr + idx * E + e, val)`。
- **Why:** 向量 atomic 一次覆盖多个连续元素，原子数下降 1~2 个数量级；拆成标量会破坏 SIMD 并爆炸原子数，典型加速比跌至 0.05x~0.15x。
- **How to apply:** 见 §3.3 L3.1 `index_put_1d_kernel`。

#### L1.2 根据 `E/D` 宽度选择并行轴

- **必须**根据 shape 选择并行轴：
  - `E/D` 较宽（≥128/256）：**index-并行 + E 维向量 atomic_add**。
  - `E/D` 较小或冲突密集：**E-并行 + 向量 atomic_add**，每个 core 串行扫描 indices。
  - 索引基本唯一且 `accumulate=False`：可走 **输出位置并行 + 向量 store**。
- **Why:** 并行轴若不与连续访问对齐，会引入标量原子竞争和离散写入。

#### L1.3 `ndim ≤ 4` 必须编译期维度特化

- **必须**在 host 侧 `dim` 归一化并 `tl.constexpr` 特化。
- **禁止**在 kernel 内运行时从 global buffer 读取 shape/stride 并做 `%`/`//` 解码。
- **Why:** 运行时坐标解码引入标量开销，且无法合并访存。
- **How to apply:** 见 §3.3 L3.2 `scatter_add_nd_kernel`。

#### L1.4 `accumulate=False` 且存在重复索引时必须实现 last-write-wins

- **必须**在 `accumulate=False` 且存在重复索引时，实现 last-write-wins（单 block 串行或两阶段）。
- **Why:** 缺少写冲突处理会导致结果不稳定。

#### L1.5 边界检查与 padding 清零必须外置

- **必须**覆盖 `index < 0`、`index >= dim_size` 等边界检查，越界写加 mask。
- **必须**将 `padding_idx` 等排除位置清零作为独立轻量 kernel，不在热 scatter 循环内分支。
- **Why:** 热循环内分支破坏向量化；越界写触发 ACL 错误。

### §3.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧分派决策树

1. Host 侧 `dim` 归一化并 `tl.constexpr` 特化。
2. 若 `accumulate=True` 且输出连续宽度 `E` 很大（`≥ 128/256`）：
   - 索引重复概率低或冲突可接受：走 **index-并行 + E 维向量 atomic_add**。
   - 冲突不可接受：走 **E-并行 + 串行扫描 indices + E 维向量 atomic_add**。
3. 若 `accumulate=True` 但 `E` 较小或通用 ND：走 **E-并行 + 标量 atomic**。
4. 若 `accumulate=False` 且索引唯一：可走输出位置并行 + 向量 store。
5. 若 `accumulate=False` 且存在重复索引：实现 last-write-wins（单 block 串行或两阶段）。

### §3.3 Layer 3: 关键 kernel 实现（优化重点与易错代码）

#### L3.1 `index_put_1d_kernel`（1D Output-Position-Parallel IndexPut）

适用条件：索引写入 1D 输出，`accumulate` 开启时冲突可控或索引基本唯一。

```python
@triton.jit
def index_put_1d_kernel(
    index_ptr, values_ptr, out_ptr,
    M, N,
    accumulate: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pos_start = pid * BLOCK_M
    pos_end = (pid + 1) * BLOCK_M
    offs = tl.arange(0, BLOCK_M)
    mask = offs < M

    # 每个 program 负责一段连续输出位置
    for idx_pos in range(0, M, BLOCK_M):
        target_idx = tl.load(index_ptr + idx_pos + offs, mask=mask).to(tl.int32)
        val = tl.load(values_ptr + idx_pos + offs, mask=mask)
        in_range = (target_idx >= pos_start) & (target_idx < pos_end) & mask
        if accumulate:
            tl.atomic_add(out_ptr + target_idx, val, mask=in_range)
        else:
            tl.store(out_ptr + target_idx, val, mask=in_range)
```

#### L3.2 `scatter_add_nd_kernel`（ND Scatter-Add with Coordinate Decomposition）

适用条件：通用 ND scatter_add，`ndim` 在编译期确定。

```python
@triton.jit
def scatter_add_nd_kernel(
    src_ptr, index_ptr, out_ptr,
    shape_ptr, flat_stride_ptr, out_stride_ptr, src_stride_ptr, index_stride_ptr,
    M, E,
    DIM: tl.constexpr, NDIM: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid_e = tl.program_id(0)
    e_start = pid_e * BLOCK_E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E

    # 将扁平 e 解码为 ND 坐标（NDIM 为 constexpr，循环会被展开）
    e = e_start + offs_e
    out_base = out_ptr
    src_base = src_ptr
    idx_base = index_ptr
    for j in range(NDIM):
        size = tl.load(shape_ptr + j)
        flat_stride = tl.load(flat_stride_ptr + j)
        coord = (e // flat_stride) % size
        out_base = out_base + coord * tl.load(out_stride_ptr + j)
        src_base = src_base + coord * tl.load(src_stride_ptr + j)
        if j != DIM:
            idx_base = idx_base + coord * tl.load(index_stride_ptr + j)

    # 沿索引维度 m 串行累加
    for m in range(0, M):
        idx = tl.load(idx_base + m * tl.load(index_stride_ptr + DIM)).to(tl.int32)
        val = tl.load(src_base + m * tl.load(src_stride_ptr + DIM)).to(tl.float32)
        tl.atomic_add(out_base + idx * tl.load(out_stride_ptr + DIM), val, mask=mask_e)
```

> 注意：此为通用 fallback。`ndim ≤ 4` 时，应改用 `DIM=0/1/2/3` 的特化版本，避免运行时循环解码。

### §3.4 IndexPut / Scatter 性能基准

| 维度 | cases | 加速比区间 | 备注 |
|------|-------|-----------|------|
| 参考 Level1 索引计算类算子 | 19_IndexPut / 21_Scatter | ≥0.6x | 旧版本几何平均加速比 vs torch；新版本若丢失向量 atomic/维度特化普遍跌至 0.05x~0.4x |

**关键结论**:
1. 连续宽度 `E/D ≥ 64` 时必须使用向量 `tl.atomic_add`。
2. `ndim ≤ 4` 时必须编译期维度特化。
3. `accumulate=False` 且存在重复索引时必须实现 last-write-wins。

---

## §4 EmbeddingDenseBackward 算子（index-accumulate）

**算子类别**: `index-accumulate`（按索引将梯度累加到 embedding 权重；`embedding_dense_backward`）
**典型特征**: 尾维度 `E`/`D` 通常较宽，索引重复概率高，fp16/bf16 累加精度敏感
**性能基准**: Level1 索引计算类算子旧版本均能达到 ≥0.6x 几何平均加速比 vs torch

### §4.0 首次生成必读：为什么必须把主要框架写对

EmbeddingDenseBackward 是**三阶段结构必须一次写对**的算子：频数统计、连续维向量 scatter_add、padding 清零。若试图用单个 kernel 内双重循环实现，原子数会随 `N×E` 爆炸，性能极差且易触发 Vector Core timeout。**首次生成如果把框架写偏（例如未拆分为 freq/scatter-add/padding-zero、未用 fp32 workspace、未对连续 E 维做向量 atomic），后续迭代很难救回**。

### §4.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 必须拆分为三阶段 kernel

- **必须**拆分为 `freq_kernel`、`scatter_add_kernel`、`padding_zero_kernel` 三个 kernel。
- **禁止**把所有可选逻辑塞进同一张 JIT 签名，或用单个 kernel 内双重标量循环实现。
- **Why:** 单个 kernel 内对 `N` 和 `E` 做双重标量循环，每个 `(pos, e)` 单独 `tl.atomic_add`，原子数随 `N×E` 爆炸，性能极差且易触发 Vector Core timeout。
- **How to apply:** 见 §4.2 L2.1 三阶段调用骨架。

#### L1.2 必须对连续 `E`/`D` 维做向量 `tl.atomic_add`

- **必须**按 embedding_dim `E` 分核，每个 program 负责一段连续 `E`，对 `grad_output` 做向量加载后执行向量 `tl.atomic_add`。
- **禁止**用 `tl.get_element(..., [j])` 拆成标量后再 atomic_add。
- **Why:** 向量 atomic 原子数下降 1~2 个数量级；标量提取破坏 SIMD。
- **How to apply:** 见 §4.3 L3.2 `embedding_grad_kernel`。

#### L1.3 fp16/bf16 累加必须使用 fp32 workspace

- **必须**分配 `torch.float32` 中间 buffer 进行 `atomic_add`，最后 cast 回原始 dtype。
- **Why:** fp16/bf16 `atomic_add` 直接累加会导致精度下降。

#### L1.4 `scale_grad_by_freq` 必须查表，禁止二次扫描

- **必须**先用 `freq_kernel` 统计每个 index 出现次数，主 kernel 查 `1/freq` 缩放。
- **禁止**在 `scatter_add_kernel` 内二次扫描 indices 统计频数。
- **Why:** 频数统计与主累加分拆避免热循环内分支和二次遍历。

#### L1.5 `padding_idx` 清零必须使用独立轻量 kernel

- **必须**用 `padding_zero_kernel` 独立轻量 kernel 清零 `padding_idx` 行。
- **禁止**在热 scatter 循环内判断 `if idx != padding_idx`。
- **Why:** 热循环内分支破坏向量化。

### §4.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 三阶段调用骨架

推荐拆分为三个 kernel，避免把所有可选逻辑塞进同一张 JIT 签名：

1. **`freq_kernel`**：N-并行 `tl.atomic_add(freq_ptr + idx, 1.0)`，统计每个输出位置出现次数。
2. **`scatter_add_kernel`**：N 分块，连续 D 维向量 `tl.atomic_add`，按 `1/freq` 缩放，fp32 累加。
3. **`padding_zero_kernel`**：清零 `padding_idx` 对应的行。

### §4.3 Layer 3: 关键 kernel 实现（优化重点与易错代码）

#### L3.1 `freq_kernel`

```python
@triton.jit
def freq_kernel(indices_ptr, freq_ptr, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    idx = tl.load(indices_ptr + offs, mask=mask).to(tl.int32)
    tl.atomic_add(freq_ptr + idx, 1.0, mask=mask)
```

#### L3.2 `embedding_grad_kernel`

```python
@triton.jit
def embedding_grad_kernel(
    grad_ptr, indices_ptr, freq_ptr, out_ptr,
    N, D, scale_grad_by_freq: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    idx = tl.load(indices_ptr + n_offs, mask=n_mask).to(tl.int32)
    scale = 1.0
    if scale_grad_by_freq:
        freq = tl.load(freq_ptr + idx, mask=n_mask)
        scale = 1.0 / freq

    for n in range(BLOCK_N):
        if n_start + n >= N:
            break
        row_idx = idx[n]
        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D
            val = tl.load(grad_ptr + (n_start + n) * D + d_offs, mask=d_mask) * scale
            tl.atomic_add(out_ptr + row_idx * D + d_offs, val, mask=d_mask)
```

#### L3.3 `padding_zero_kernel`

```python
@triton.jit
def padding_zero_kernel(out_ptr, padding_idx, D, BLOCK_D: tl.constexpr):
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    tl.store(out_ptr + padding_idx * D + d_offs, 0.0, mask=d_mask)
```

### §4.4 EmbeddingDenseBackward 性能基准

| 维度 | cases | 加速比区间 | 备注 |
|------|-------|-----------|------|
| 参考 Level1 索引计算类算子 | 24_EmbeddingDenseBackward | ≥0.6x | 旧版本几何平均加速比 vs torch；新版本若未拆分三阶段/未用 fp32 workspace 普遍跌至 0.05x~0.4x |

**关键结论**:
1. 必须拆分为 freq / scatter-add / padding-zero 三个 kernel。
2. 必须对连续 `E`/`D` 维做向量 `tl.atomic_add`。
3. fp16/bf16 累加必须使用 fp32 workspace。

---

## §5 Sort / TopK / 选位算子（index-sort / topk-select）

**算子类别**: `index-sort`（基于比较重排并返回索引；`sort` / `topk` / `argsort`）＋ `topk-select`（轻打分 + 重 top-K 选位 + gather 有序索引/值；LightningIndexer）
**典型特征**: 比较-选择模式，与 Gather/Scatter 的索引映射模式差异较大；需注意输出索引 `int64` 与数据类型的对齐；topk-select 形态含 score 打分（GEMM/点积）→ 选位（top-K）→ 索引输出，select 成本常 ≥ 打分成本
**性能基准**: Level1 索引计算类算子中 8_Sort 当前 memory 目录无专项文档，建议走 sort-topk 专项模板；LightningIndexer 54/54 pass，几何平均 **3.9558x** vs torch（§5.5）

### §5.0 首次生成必读：为什么必须把主要框架写对

Sort / TopK 是**比较-选择**模式，与 Gather/Scatter 的索引映射模式差异较大。问题规模差异大：小 `K` 可用 bitonic/odd-even 排序网络，大 `K` 需 reduce-split 或分桶 + 局部排序。**首次生成如果把框架写偏（例如大 K 用全排序网络、未注意 int64 索引对齐），性能和精度都很难救回**。

> ⚠️ **topk-select 形态（LightningIndexer 类）首次生成必读**：主墙是 **top-K 选位**（torch 参考的 CPU sort 占 ~97%），不是打分。**首次生成就要用 sort-compaction 结构（§5.5 L1.1）**，不要先写一个能跑的 `tl.cumsum` 前缀链再指望 Phase 4 优化——串行 cumsum 在 S2=32768 实测 ~10ms（aiv_scalar=0.999），**参数扫描救不了标量链，只能换结构**（§5.4 S1）。

### §5.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 根据问题规模选择排序策略

- **必须**根据 `K` 大小选择策略：小 `K` 用 bitonic/odd-even 排序网络；大 `K` 用 reduce-split 或分桶 + 局部排序。
- **Why:** 全排序网络复杂度随 K 增长过快，大 K 时性能极差。

#### L1.2 输出索引 `int64` 必须与数据类型对齐

- **必须**注意输出索引的 `int64` 与数据类型的对齐。
- **Why:** 索引 dtype 不匹配会导致下游算子错误。

#### L1.3 优先使用向量比较原语

- **必须**在 Ascend 上比较算子优先使用向量比较原语。
- **Why:** 标量比较会破坏 SIMD，无法发挥 Vector Core 并行能力。

### §5.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 策略选择

- 小 `K` 用 bitonic/odd-even 排序网络。
- 大 `K` 用 reduce-split 或分桶 + 局部排序。
- 当前 memory 目录中无 Sort 专项经验，建议在生成时单独走 `sort-topk` 模板或做针对性设计。

### §5.3 Sort / TopK 性能基准

| 维度 | cases | 加速比区间 | 备注 |
|------|-------|-----------|------|
| 8_Sort | 待补充 | 待补充 | 当前 memory 目录无专项文档，建议走 sort-topk 专项模板 |

**关键结论**:
1. Sort/TopK 与 Gather/Scatter 的索引映射模式差异较大，不要套用 Gather/Scatter 经验。
2. 大 K 时不要使用全排序网络。
3. 注意输出索引 `int64` 与数据类型的对齐。

---

### §5.4 选位家族通用经验（迁移自 cv-fusion.md §1 CV10/CV11/CV12）

以下三条是检索/select 类（`topk-select`）**跨算子通用**经验，适用于所有"打分 → 选位 → 索引"形态的算子（LightningIndexer / MoeGatingTopKSoftmax / 各类 topk）。

#### S1 ★ `tl.cumsum` 在 Ascend 上标量化（aiv_scalar≈0.99），select/前缀定位必须绕开

- **必须**凡出现"选 K 位之前的计数→定位"（`tl.cumsum` 前缀链）且 profiling `aiv_scalar_ratio≈0.99`，立即**换结构**：排序压缩（§5.5 L1.1）或矩阵乘前缀和；**不要扫参，参数救不了标量链**。
- **禁止**用 `torch.cumsum` host 前缀表（validator AST 白名单禁）。
- **Why（实测）**：LightningIndexer 串行 chunk-scan + `tl.cumsum` 前缀在 S2=32768 时 **10.1ms**（aiv_scalar=0.999）；并行 scatter + 两次 int32 cumsum 仍 **4.2ms**（int32 走 vector 但跨 chunk 前缀仍串行，aiv_scalar=0.997）；改 sort-compaction 后 **0.112ms**——**同一 select 问题，结构选择差 90x+**。
- **判别信号**：代码里 `tl.cumsum` 用于选位 / 紧凑化 ⇒ 预检直接标红，换结构而不是调 BLOCK。

#### S2 ★ `npu_sort_v2` 走 NPU 侧，是 `aclnnSort`（AiCpu）的 40x+ 替代，且可复用双角色

- **必须**需要"第 K 大的值" / "前 K 有序"时优先 `torch_npu.npu_sort_v2`（合法 `torch_npu.*` 命名空间，validator 放行），**喂 2D `[rows, N]` view**（1D 布局不可靠，踩坑）。
- **必须**复用同一个 sort 调两用：① 升序取 `tau=sorted[N-K]` 当阈值 oracle（省全量 top-k）；② 对 masked 数组升序压缩（把"选中的 rank 位 + 未选中 LARGE2"紧凑到前 K 位）——**排序同时完成"选 K + 排序 + 去重"**。
- **禁止**用 `torch.topk`（validator 禁，且是唯一已知快路径 ~17µs 但不可用）；**禁止** 纯 Triton 排序网络做 top-k（tl.sort 编译崩 / gather-bitonic ~1ms 地板）。
- **Why**：torch 参考的 `aclnnSort_SortAiCpu_Sort` 单次 **1466µs**（CPU sort）是参考主墙；实现侧 `npu_sort_v2` 两次仅 **38µs**——结构差 40x。

#### S3 ★ 检索/select 类算子：主墙在 select 不在打分；天花板估算先归零"可消除结构"

- **必须**先做 profiling `operators` 分解，确定打分 vs select 各自占比，再定优化方向——本类打分核常已 cube-bound 很轻，**select 才是墙**（LightningIndexer：打分 38µs 占 impl 22%，select 相关两次 sort 38µs + build_masked 68µs 占 61%）。
- **必须**天花板估算把**可消除的算法结构**（串行前缀和 / 逐元素定位）**先归零再算**——R1 曾推算 select ~1ms 上限，被 sort-compaction 证伪（select 侧几乎免费，只剩 sort 是真墙）。
- **禁止**用打分核的优化套路（tile 放大、流水）去攻 select——select 是**结构问题**，不是参数问题。

---

### §5.5 LightningIndexer 算子（`topk-select`，检索/select 融合：轻打分 + 重 top-K 选位 + gather 有序索引/值）

> 📌 本节迁移自 `cv-fusion.md §4`。op5 的运算形态是「score 打分 + top-K 选位 + 索引输出」，属索引计算家族的 `index-sort` / `topk-select`，故归位于本文件 §5。原 `cv-fusion.md §4` 已留指针。

**算子类别**: `topk-select`（轻打分 GEMM + 重 top-K 选位 + gather 有序索引/值）
**典型特征**: `q` bf16/fp16 `[B,S1,N1,D]` @ `k` `[B,S2,D]` + relu + `w` `[B,S1,N1]` head-sum → `score[B,S1,S2]`；`sparse_mode=3`（`j ≤ i + (as2-as1)`），`sparse_count=K=min(2048,S2)` **升序** top-K；输出 `out_idx[b,i,1,K]` int32 + `out_val` bf16（可选）
**性能基准**: 54/54 verify pass，几何平均 **3.9558x** vs torch（相对 baseline **12.02x**，target 5x 达成）；历史审计 0.1603x → R1 0.3178x → **R2 3.9558x**

#### L1.1 ★★★ top-K 选位禁用逐元素前缀和，用 sort-compaction（本算子最大的一笔，首次生成就要写对）

- **必须**把"选 K 个并升序"变成 4 步：**count kernel**（`grid=(rows,nchunks)` 每 chunk 并行 gt/eq 计数）→ **build_masked**（score>tau 写全局 rank 位、score==tau 按 chunk 前缀补位、未选中写 LARGE2=2^21）→ **第二次 `npu_sort_v2` 升序压缩**（前 K 个即升序 top-K 索引）→ **finalize**（清无效行 + value 位提取）。
- **禁止** `tl.cumsum` 前缀定位（AIV scalar 链，10.1ms）；**禁止** 并行 scatter + int32 cumsum（仍标量，4.2ms）；**禁止** `torch.cumsum` host 前缀表（validator 禁）。
- **Why（实测）**：route-1 串行 chunk-scan 10.1ms（S2=32768，aiv_scalar=0.999）；route-3 并行 scatter + 两次 int32 cumsum 4.2ms（aiv_scalar=0.997）；sort-compaction **0.112ms**——**同一 select 三个方案差 90x**。排序同时完成"选 K + 排序 + 去重"，`aiv_scalar` 回到正常。
- **LARGE2 取值**：必须大于 score 可达的 rank 上界（实测 `1<<21` 安全），未选中填 LARGE2 后升序压缩自然把选中项排到前 K；build_masked 的 rank 编码决定 tie-breaking 语义，必须与参考的 eq 前缀补位一致（见 L3.1）。

#### L1.2 `npu_sort_v2` 必须喂 2D `[rows, S2]` view（见 §5.4 S2）

- **必须** `score_buf.view(rows, S2)` 再调 `npu_sort_v2`；**禁止** 1D 布局（行为不确定）。
- **必须**复用两次：K2 升序取 `tau = sorted[:, S2-K]`（阈值 oracle，省全量 top-k）；K3c 对 masked 升序压缩（选位 + 排序合并）。两次共 38µs，是 select 变快路径的代价而非墙。

#### L1.3 score 打分链融合成单 kernel（轻但必要的地基）

- **必须** `qk = tl.dot(q, tl.trans(kt))`（bf16/fp32-acc）+ `s = tl.sum(tl.maximum(qk, 0.0) * w[:, None], axis=0)` + `tl.where(valid, s, -inf)` **一核完成**（dot + relu + w head-sum + mask 全融合）。
- **禁止**拆多 kernel / 逐 token 标量点积（融合是 0.2262 → 0.3178 的基础）。
- `BLOCK_J=256`：**1024 编译失败**（UB hivm-plan-memory）——tile 上限先探再定（§1 G11）。`num_stages=2` 让 S2 分块的 MTE2 预取与 cube 重叠。

#### L1.4 sparse_mode=3 mask 语义

- `j <= i + (as2-as1)` 的 OOB 列注入 `-inf`；54/54 验证此语义（**别按完整三角写**）。

#### L1.5 bf16 value 输出走 RNE 位模拟

- `out_val` 用 `(vb + 32767 + lsb) & -65536`（bf16 round-to-nearest-even 位运算）与 CANN 参考对齐；`return_value=True` 时另出 `out_val`，否则只出 `out_idx`。

#### L1.6 validator 白名单与命名空间

- **禁止** `torch.cumsum` / `torch.topk` / `torch.cat` / `self.xxx`（AST 白名单禁）；`torch_npu.npu_sort_v2` 属合法 `torch_npu.*` 命名空间放行。
- forward 内零 PyTorch 计算算子（仅 `empty`/`zeros`/`view` + kernel 发射 + `npu_sort_v2`）；host 侧无循环、无前缀表（count 并行算出）。

#### L1.7 count kernel 并行度

- **必须** `grid=(rows, nchunks)` 并行每 chunk 的 gt/eq 计数，**禁止** host 前缀表；rows×nchunks 超过核数时核内步长（§1 G11）。

#### L2.1 Host 侧发射流程（K1 → K2 → K3a → K3b → K3c → K4）

```python
def forward(self, q, k, w, topk_val, sparse_count, sparse_mode, return_value, as1, as2):
    B, S1, N1, D = q.shape; S2 = k.shape[1]
    rows = B * S1; K = sparse_count
    score_buf = torch.empty((B, S1, S2), dtype=torch.float32)     # 或 fp32 buffer
    lightning_score_kernel[(rows,)](q, k, w, score_buf, ...,
                                    BLOCK_J=256, num_stages=2, ...)          # K1 打分链（L1.3）
    sorted_vals, _ = torch_npu.npu_sort_v2(score_buf.view(rows, S2))         # K2 阈值 oracle（2D view 必须）
    tau = sorted_vals[:, S2 - K]                                             # 第 K 大
    gt_cnt = torch.zeros((rows, NCHUNK), dtype=torch.int32)
    eq_cnt = torch.zeros((rows, NCHUNK), dtype=torch.int32)
    lightning_count_kernel[(rows, NCHUNK)](score_buf, tau, gt_cnt, eq_cnt, ...)   # K3a 并行 count（L1.7）
    masked = torch.full((rows, S2), LARGE2, dtype=torch.int32)
    lightning_build_masked_kernel[(rows,)](score_buf, gt_cnt, eq_cnt, tau, masked, ...)  # K3b rank 位 + LARGE2
    sorted_masked, _ = torch_npu.npu_sort_v2(masked)                        # K3c 压缩排序（升序）
    idx = sorted_masked[:, :K]                                              # 前 K 列 = 升序 top-K 索引
    out_idx = torch.empty((B, S1, 1, K), dtype=torch.int32)
    out_val = torch.empty((B, S1, 1, K), dtype=torch.bfloat16)
    lightning_finalize_kernel[(rows,)](idx, score_buf, out_idx, out_val, ...)  # K4 清无效行 + value 提取
    return out_idx, out_val
```

#### L2.2 K3b build_masked 骨架（次墙，首次生成就要写对）

```python
@triton.jit
def lightning_build_masked_kernel(score_ptr, gt_cnt_ptr, eq_cnt_ptr, tau_ptr, masked_ptr,
                                  S2, NCHUNK, BLOCK_J: tl.constexpr, ...):
    pid = tl.program_id(0)                     # 每行一个 program
    offs_j = tl.arange(0, BLOCK_J)
    # 跨 chunk 的 gt/eq 前缀来自 K3a 的并行 count（无 host 前缀表）
    # 选中（score > tau）：rank = 全行 gt 前缀 + chunk 内 gt 排名 → 写 rank 位
    # 选中（score == tau）：rank = gt 前缀 + eq 前缀（chunk 间按序补位）
    # 未选中：写 LARGE2（升序压缩后自动沉到尾部）
```

#### L3.1 sort-compaction 为什么能绕开 cumsum（本算子核心机理）

把"选中的 j 的 rank 位" + "未选中填超大数"写成 masked 数组，**第二次升序排序后前 K 个天然就是升序 top-K 索引**——排序同时完成"选 K + 排序 + 去重"，不再需要逐元素前缀和。与"并行 cumsum"（route-3）的本质差异：cumsum 的**跨 chunk 前缀仍是串行标量**，排序是库级并行。这是"选位结构选择"比任何参数优化都值钱的最硬证据。

#### L3.2 排序压缩 vs 并行 cumsum 的 37 倍实测对比

route-2（sort-compaction）**0.112ms** vs route-3（并行 chunk-scatter + 两次 int32 cumsum）**4.2ms**——**同样绕过串行的两个方案差 37 倍**，区别只在 route-3 的 cumsum 仍是标量。

#### L3.3 打分核与 select 的"双 sort"复用

同一个 `npu_sort_v2` 调两次（阈值 oracle 22µs + 压缩 16µs）共 38µs，占 impl 22%——与参考的 CPU sort 1466µs 形成 40x 差距。**别想省第二次 sort**：它是把 select 从性能墙变快路径的代价。

#### L3.4 剩余墙（留给后续优化）

build_masked **68µs 占 impl 39%**（vector-bound 写入）是次墙；大 S2（16384/32768）tail 0.3–1.8ms 受第二次 sort 随 S2 线性增长拖累。若继续，可尝试 build_masked 位处理向量化、或对 S2 分块做 split-sort。

#### L4.1 性能演进

| 版本 | cases | geomean | 说明 |
|------|-------|---------|------|
| 历史审计（0811/5 镜像） | 54/54 | 0.1603x | 纯 Triton bitonic-topk（topk kernel ~3814µs 是墙） |
| R1（fusion-search route-2） | 54/54 | 0.3178x | `npu_sort_v2` 阈值 oracle + score 链融合 |
| **R2（sort-compaction 冠军）** | **54/54** | **3.9558x** | **sort-compaction select 消除串行 cumsum，impl 1.348ms → 0.1122ms（12.4x）** |

---

## §6 常见陷阱与避免方法

### §6.1 Index / Gather 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| Kernel 内运行时判断 `dim` | `if dim == 0:` 运行时变量 | Host 侧 `DIM=0` 作为 `tl.constexpr` 启动（§2.1 L1.1） |
| Gather 用输入 shape 做循环边界 | `M = x.shape[:dim].prod()` | `M = index.shape[:dim].prod()`（§2.1 L1.3） |
| 直接间接 gather 当作默认路径 | 所有 gather 都用 `x_ptr + idx_vals * stride` | 优先 UB preload + `tl.gather`，direct-indirect 仅 fallback（§2.1 L1.2） |
| UB 预载尺寸超过容量 | 预载整行 `tl.load(x_ptr + row * dim_size + arange(0, dim_size))` 且 `dim_size > UB_LIMIT` | 先校验 `ROW_SIZE ≤ UB_LIMIT`，过大则分 `BLOCK_*` + `BLOCK_*_SUB`（§2.1 L1.4） |
| 无条件 host 侧转置 | 未估算收益就 `.contiguous()` | 仅在 `gather_gain > 2 × transpose_cost` 时转置（§2.1 L1.5） |

### §6.2 IndexPut / Scatter 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 标量 atomic 内层循环 | 所有 scatter 都用标量 | 连续宽维度用向量 atomic（§3.1 L1.1） |
| 运行时 ND 坐标解码 | `for j in range(NDIM)` 从 global buffer 读 shape/stride | `ndim ≤ 4` 编译期维度特化（§3.1 L1.3） |
| `accumulate=False` 重复索引结果不稳定 | 多个 source 写入同一 target，顺序随机 | 实现 last-write-wins（§3.1 L1.4） |
| 在热 scatter 循环内判断 padding | `if idx != padding_idx:` 每个元素 | 后置独立 zero kernel（§3.1 L1.5） |
| 忽略索引越界 | 直接 `tl.load(x_ptr + idx * stride)` | 加 mask 与 `other=0`，必要时过滤（§3.1 L1.5） |

### §6.3 EmbeddingDenseBackward 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 单 kernel 双重循环 | 对 `N` 和 `E` 做双重标量循环 | 拆分为 freq / scatter-add / padding-zero 三个 kernel（§4.1 L1.1） |
| 向量 load 后拆标量 atomic | 用 `tl.get_element(..., [j])` 拆后再 `tl.atomic_add` | 直接构造向量地址并调用向量 `tl.atomic_add`（§4.1 L1.2） |
| fp16/bf16 直接累加 | `tl.atomic_add` 累加 fp16/bf16 输出 | 分配 fp32 workspace，最后 cast 回原始 dtype（§4.1 L1.3） |
| `padding_idx` 在热循环内清零 | 每个元素判断 `idx != padding_idx` | 独立 `padding_zero_kernel`（§4.1 L1.5） |

### §6.4 通用跨算子陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 向 `tl.atomic_add` 传 `None` | `freq_ptr=None` | 传合法 dummy tensor（§1 G7） |
| Autotune 复用累加 buffer | 同一 buffer 被多 config 累加 | 每次重新初始化或禁用 autotune（§1 G10） |
| GPU 源代码保留 cuda-only 启动参数 | 保留 `num_warps`/`num_stages`/`cache_modifier` | 完全移除；Ascend 只保留 `BLOCK_*`（§1 G8） |
| GPU 源通过 stride 交换实现 transpose | `ptr + col * stride[1] + row * stride[0]` | 使用 `order=(...)` 或显式 host 侧 `permute().contiguous()`（§1 G8） |
| `int64` 索引未降级为 `int32` | 用 `int64` 做下标或保留 `torch.int64` 进 kernel | Vector 路径前 `.to(tl.int32)`；扁平偏移按需用 `int64`（§1 G4） |
| 1D grid 超过 65535 | `grid = (total_blocks,)` | 使用 2D grid 或增大 BLOCK（§1 G11） |

### §6.5 LightningIndexer / topk-select 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| top-K 选位用 `tl.cumsum` 前缀链（S2 大时） | 标量链（aiv_scalar=0.999），参数扫描救不了 | §5.4 S1 + §5.5 L1.1：sort-compaction 结构 |
| `npu_sort_v2` 喂 1D 布局 | 行为不确定，结果错 | §5.5 L1.2：2D `[rows,S2]` view |
| 用 `torch.topk` / `torch.cumsum` 加速 select | validator AST 禁用 | 用 `npu_sort_v2` + count/build_masked 并行（§5.5 L1.1/L1.7） |
| 打分核调参（tile/流水）攻 select 墙 | select 是结构问题不是参数问题 | §5.4 S3 + §5.5 L1.1：先归零"可消除结构" |
| 大 K 用全排序网络 | 复杂度随 K 增长过快 | 按 K 规模选策略（§5.2 L1.1）；topk-select 用 sort-compaction |
| 忽略输出索引 dtype 对齐 | 索引 dtype 不匹配导致下游错误 | int64 索引对齐 + `.to(tl.int32)`（§5.2 L1.2 / §1 G4） |
