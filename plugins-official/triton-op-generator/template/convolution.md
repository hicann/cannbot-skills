---
name: convolution
description: 卷积类算子（ConvStandard1d / ConvDepthwise2d / ConvTranspose2d / ConvStandard3d / ConvStandard2d）的 Triton Ascend 优化经验合集，按算子分章节组织，含通用经验 + 各算子专属约束/骨架/kernel
metadata:
  type: reference
---

# 卷积类算子优化经验

本文档合并卷积类算子的优化经验，按以下结构组织：

- **§1 通用经验**：跨算子共有的工程约束（已提取，各算子章节不再重复）+ 通用算法骨架
- **§2 ConvStandard1d**（1D 标准卷积）
- **§3 ConvDepthwise2d**（2D depthwise）
- **§4 ConvTranspose2d**（2D 转置卷积）
- **§5 ConvStandard3d**（3D 标准 / grouped，`kernel_size=(k,k,1)`）
- **§6 ConvStandard2d**（2D 标准 + grouped + depthwise-like）
- **§7 常见陷阱**（按算子分小节）
- **§8 已验证无效方向**

> ⚠️ **关键区分**：卷积类的核心优化哲学是 **"按 stride/groups/kernel 特化多路径分派 + K-tap→1×1 分解 + reduction 维对齐 Cube C0=16"**。生成时先读 §1 通用经验（硬规则），再读本算子章节；**禁止混用**非卷积类别的经验。

> ⚠️ 本文件只记录**设计约束、算法骨架与关键技巧片段**，不保留可直接复制的完整 kernel；生成时应根据约束重新设计变量名与实现结构。

---

## §0 适用范围与算子分类

| 算子 | 计算特征 | 核心优化策略 | 性能基准 (geomean vs torch) |
|------|---------|-------------|---------------------------|
| ConvStandard1d | 1D 标准卷积，K-tap | stride==1→shifted 1×1 GEMM（groups>1 用 general shifted）；stride>1→Triton im2col+GEMM 直写+bias 融合；padding 用 `torch.zeros`+`copy_`（禁止 unfold/permute）；动态 BLOCK 分档；im2col BLOCK_L 自适应 | **~0.84x**（target 0.8x，达标） |
| ConvDepthwise2d | 2D depthwise，无跨通道规约 | element-wise mul + 3D/5D block_ptr；k=5/7 → split-KHW+atomic | **~0.90x**（达标） |
| ConvTranspose2d | 2D 转置卷积 | dilate+pad → flipped-weight stride-1 Conv2d 等价转换；确定性 BLOCK_C 启发式；连续块划分 + 最内层 ow_blk 解码复用 w_blk 缓存 | **0.8297x**（target 0.8x，**达标**） |
| ConvStandard3d | 3D 标准 + grouped（源 benchmark 实际 `kernel_size=(k,k,1)`） | NC1DHW0 + 1×1 分解（主）；**P3 no-permute 布局 + HW 展平 dot**（主力）；Cout_pg<16 → groups-batched M=16 | **1.1332x**（**达标** target 0.8x，60/60 pass） |
| ConvStandard2d | 2D 标准 + grouped + depthwise-like，stride/pad/dilation/groups/bias 全动态（forward 入参） | **多路径 hybrid 分派 + host 预零填充消除 kernel 内边界逻辑**：1×1 纯 GEMM / depthwise 通道块向量乘加 / skinny channel-last 打包单 dot / S=2 奇偶平面拆分 / S=1 大 kc 逐平面无 mask 大 dot / 逐窗口 tap dot 兜底 / im2col+GEMM（profiling 双门控）；自适应 tile + 运行期乘加量门控 + pad kernel 2D 平面 tile（T2d-15） | **0.6677x**（target 0.8x，未达） |


**归类规则**：算子名含 `Conv1d`/`ConvStandard1d` → §2；`ConvDepthwise2d` → §3；`ConvTranspose2d` → §4；`ConvStandard3d` → §5；`Conv2d`/`ConvStandard2d` → §6。

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下约束是卷积类算子**共有**的工程规则。各算子章节只补充**特化**部分（fan_in 口径、路径选择、门控阈值等）。

### Layer 1：通用设计约束

#### G1 手工 CPU 初始化权重/偏置
- **必须**：按 `nn.ConvNd.reset_parameters` 原理生成；CPU 上 `empty + uniform_` 后再 `.to(device)`；per-config 种子 `torch.manual_seed(hash(key) & 0xFFFFFFFF)`；顺序**先 weight 全量，再 bias 全量**。
- **禁止**：调用 `nn.ConvNd`；在 NPU 上 `uniform_`；在 `__init__` 中做权重预计算（必须移到 `forward()` 内）。
- **Why**：NPU RNG 与 CPU 同种子序列不同 → 精度失配。
- **fan_in 口径（各算子不同，见对应章节）**：
  - 标准卷积（1d/2d/3d）：`fan_in = (C_in // groups) * Π(kernel_size)`
  - depthwise（groups==C_in，即 C_in//groups=1）：`fan_in = Π(kernel_size)`
  - ConvTranspose2d：`fan_in = out_channels * Π(kernel_size)`（用 **out_channels**，与标准 Conv 相反）
  - 统一：`bound = 1 / sqrt(fan_in)`，`uniform_(-bound, bound)`。

#### G2 一维 grid + 核数上限
- `grid = (min(total_tiles, num_cores),)`；kernel 内 `for tile_idx in range(pid, total_tiles, NUM_CORES)` 覆盖。
- **禁止** 3D grid；**禁止**交织划分 `range(core_id, total, NUM_CORES)`（小 shape 延迟偏高，改连续块划分）。
- **Why**：Ascend 仅支持一维 grid；任务数超核数必须程序内循环覆盖。

#### G3 禁止 `%`（减法代模）
- tile / oh / ow / kk / group 解码全部用 `a - (a // b) * b`；比较用 int32。

#### G4 block_ptr 向量化访存 >> pointer-tensor gather
- 输入/输出加载优先 `tl.make_block_ptr` + `boundary_check`。
- **Why**：`tl.load(ptr + computed_offsets)` 走 gather（实测 ~0.8 GB/s）；`tl.make_block_ptr` 触发 `DataCopy`/`DataCopyPad`（实测 ~40 GB/s）。

#### G5 block_ptr 负 offset 必须 host 预补零
- pad>0 时先用 Triton pad kernel 把输入补成 `x_padded`，使 kernel 内 offset 恒 ≥ 0。
- **Why**：`boundary_check` 对负 offset 返回**整个 block 全零**（非逐元素零）→ 边界区域精度失败。

#### G6 forward() 禁止 torch/F 计算接口
- **禁止**：`F.pad`、`F.conv*`、`torch.unfold`（计算用）、`permute`（计算/拷贝重排）、切片赋值、`min()/max()`（validator 视为计算操作）。
- **允许**（纯布局/元信息）：`view / reshape / permute(纯布局) / contiguous / squeeze / .to`。
- padding / layout 转换 / 转置必须封装进 `@triton.jit` kernel。
- **Why**：validator Type3 退化检测；host 侧 torch 计算引入大拷贝与额外 launch。

#### G7 大 K_dim 保留 ieee 精度
- `tl.dot(..., input_precision="ieee")` 在 Ascend 触发原生 f32 Cube 路径。
- **Why**：大 reduction 维 tf32 精度不足。

#### G8 block_ptr 连续轴必须是单位 stride（⚠️强约束）
- `order` 中连续轴（stride=1）若实际 stride≠1，**K≥3 起读到数据错位**（K=1 因只取单 slice 不触发）。
- **解法**：device 转置使 reduction 轴 stride=1（如 weight 物理布局 `[C_OUT, C_IN, K]` 中 C_IN stride=K≠1，须转置为 `[C_OUT, K, C_IN]`）。
- **Why**：Ascend block_ptr 对非单位 stride 连续轴 lowering 有误；`boundary_check` 不救（只截 shape 边界，不截 stride 语义）。

#### G9 device 侧权重变换缓存键必须随源权重唯一（⚠️强约束）
- 任何对源权重做 device 变换（transpose/pack/reshape）后的缓存，键必须随源权重唯一。
- **推荐键**：`weight.data_ptr()`（源权重由 `_CONV_CACHE` 持久持有、跨 config 不释放，data_ptr 稳定唯一）或完整 conv_key `(in,out,K,stride,pad,dil,groups,bias) + variant tag`。
- **Why**：两个 config 共享 `(cin,cout,K)` 但权重张量不同 → 命中陈旧变换 → 该 config 全 shape 精度失配（rel err 1~2x，非精度问题）。

#### G10 reduction 维对齐 Cube C0=16
- `K_dim` pad 到 16 倍数；`BLOCK_K ≥ 16`（小 K case 不退化到 8）。
- **Why**：Cube 原生宽度 C0=16，未对齐走非 Cube 路径或精度异常。

#### G11 输出并行 tile + 禁止无保护 atomic_add
- 每个 program 负责一个输出 tile，内部规约后一次 `tl.store`。
- atomic_add 仅在受控 split 优化中使用：输出先 zero、全部 chunk 用 atomic_add、bias 仅 `k_split==0` 时加；**禁止** chunk 0 用 `tl.store` + 后续 atomic_add（race → err 2~3）。

#### G12 groups > 1 按组处理
- `g = co // C_out_per_group`；输入用组内相对索引 + 组起始 `g * C_in_per_group`；权重用 co_global + ci_local。
- **禁止** 用统一 GEMM 导致输出通道混组（精度全错）。

#### G13 Triton-Ascend 平台硬限制
- **禁止** kernel 内嵌函数定义（全部内联）。
- **禁止** 3D grid（见 G2）。
- **禁止** 对已加载的局部张量做切片 `tensor[a:b, c:d]` → `unsupported tensor index: slice`，编译期失败。
- **禁止** 把 `reshape`/`trans` 的输出直接喂 `tl.dot`（布局元数据被忽略，store 取值对但 dot 全错）；dot 两操作数必须来自 genuine 2D `block_ptr`/`tl.load`。
- **禁止** 3D batched `tl.dot`（编译/执行挂死 >10min）。
- **禁止** 用运行时谓词包绕 `tl.dot`（`if runtime_cond: acc = tl.dot(...)`）：Ascend `linalg_to_bin` / bishengir-compile pass 崩溃，全 case `MLIRCompilationError`（ConvTranspose2d opt_iter_0 "2-oh w_blk hoist" 实测 50/50 编译失败）。需用**编译期 `tl.constexpr` 分支**或**双命名累加器 + 无条件 dot**规避；运行时边界只能靠 `tl.load` 的 `mask`/`boundary_check` 处理，不能用 `if` 包 dot。

---

### Layer 2：通用算法骨架

#### A1 K-tap → K 个 shifted 1×1 block_ptr dot 分解（跨 1d/2d/3d 标准卷积的算法突破）
- 把 K-tap 卷积拆为 **K 个 shifted 1×1 卷积**，reduction 维从 `C_in·K` 降到 `C0=16`（Cube 原生宽度），K 次 `tl.dot` 累加进同一 accumulator。
- stride==1 时每个 tap 的输入列连续，**一次 block_ptr load 取整块**，无需 host im2col 物化；dilation 仅改变 tap 起点 `l_start + k*DIL`。
- 权重需 device 转置使 reduction 轴 stride=1（见 G8）。
- **适用**：groups≥1 标准卷积 + stride==1（Conv1d 主路径 / ConvStandard3d T7 NC1DHW0 / NCHWc+5D）。
- **收益**：单 shape 1.5–8x；reduction 维对齐 Cube C0=16 是关键（见 [[feedback_nc1hwc0_breakthrough]]）。

#### A2 NC1xC0 打包（通道内层 C0=16）
- pack x 为 `[N, groups, C1_pg, <spatial>, C0]`（2D=NCHWc / 3D=NC1DHW0 / 1D=NC1LC0），让 16 通道占 1 cacheline，利用率 100%。
- 权重按 `[groups, Cout_pg, C1_pg, <K>, C0]` 打包；配合 A1，每个 tap 读 `w[...k...]` + `x[...c0...]` 做 Cube dot。
- stride==1 可保留 `[N,g,C1,C0,<spatial>]`（W 内层连续，no-permute）；stride>1 必须 permute 到 C0 内层，否则 block_ptr 非法/gather 回退。

#### A3 im2col + GEMM（带 K_dim·M 门控）
- 两阶段：im2col kernel 把 x scatter-gather 为连续 `x_col`（输出端 block_ptr 连续 store），再 `W @ x_col` block_ptr Cube dot。
- **必须** host 门控（按 `K_dim * M` 阈值，需按数据集校准）：小 case Python/内存开销抵消收益。
- 仅当 A1 不适用（stride>1 列不连续）时使用。
- `x_col` reduction 顺序须与 weight 展平顺序一致（如 `(C, kd, kh)`）。

#### A4 depthwise element-wise vector 路径（无跨通道规约）
- depthwise 每个输出通道只依赖 1 输入通道（groups==C_in），**无跨通道规约**，KHW≤49。
- **禁止** `tl.dot`（Cube 16³ 阵列浪费 15/16 行；ascend910B1 cube:vec=1:2 下 Vector 路径更优）。
- 采用 Python 常量 KHW 循环 + `[BLOCK_M, BLOCK_N]` 向量化 element-wise mul + accumulate。

#### A5 多路径 hybrid 分派
- host 按 `(stride, padding, groups, kernel_size, 通道数)` 选路径；每条路径独立准备 pack 与 kernel，避免单 kernel fuse 导致 UB 放不下常驻累加器。

---

## §2 ConvStandard1d 算子（1D 标准卷积）

**算子类别**：`convolution-1d`　**典型特征**：`(N, C_in, L)` × weight `[C_out, C_in//groups, K]` → `(N, C_out, L_out)`
**性能基准**：50/50 pass，geomean **~0.84x** vs torch（target 0.8x，达标）。约束前上限 ~1.06x（使用 `unfold`/`permute`）；合规后 Path B/D 改 Triton gather，降至 ~0.84x。

### §2.1 Layer 1：特化约束

- **fan_in**（G1）：`fan_in = (C_in // groups) * K`。
- **C1d-NO-LAYOUT-OPS ⚠️ `forward()` 禁止 `torch.unfold` / `torch.permute` / `F.pad` 等布局接口**：padding、im2col、weight 重排必须改用 AST 白名单纯内存操作（`zeros`/`fill_`/`copy_`/`view`/`reshape`）或专用 Triton kernel。违反即使更快也不得保留（1.06x 的非合规实现已因此回退）。
- **G14 ⚠️ `tl.dot` 的 b 操作数必须连续加载**：stride>1 的 strided 2D `tl.load` 喂 `tl.dot` 在 Cube 上 mis-lower（stride 2/3 全错，stride 1 正常；同 b 喂 `tl.sum` 却正确）。**stride>1 输入必须先物化连续 im2col 再 GEMM**，不可 strided 直喂 Cube（隔离脚本 `repro_dot.py`）。
- **C1d-PAD ⚠️ 1D padding 用 `torch.zeros` + `copy_`**：独立 Triton pad kernel 即使 1D grid 全 tile 并行仍占实现耗时 ~80%；`x_pad = torch.zeros((N, C_in, L_pad), ...); x_pad[:, :, padding:padding+L_in].copy_(x_contig)` 在 AST 白名单内（纯内存/形状操作），是跨越 target 的核心杠杆。
- 其余通用约束见 §1 G1–G13。

### §2.2 Layer 2：路径选择

| 条件 | 路径 | 说明 |
|------|------|------|
| `groups==1 && stride==1` | **shifted 1×1 block_ptr GEMM**（A1，首选） | K-tap 分解为 K 个 shifted 1×1 dot，reduction 仅 C_in；block_ptr 输入；消除 host im2col |
| `stride==1 && groups>1` | **general shifted kernel**（opt_iter_2 新增） | scalar 连续输入加载（stride==1 连续，可喂 dot）+ block_ptr group-major 权重；消除 host im2col |
| `stride>1 && groups==1` | im2col + GEMM **直写 + bias 融合**（A3，opt_iter_4） | stride>1 输入非连续（G14），物化 im2col；GEMM 直接写 `out[N,Cout,Lout]` 连续布局并融合 bias，消除 out_2d/reshape/permute/copy/bias-kernel |
| `stride>1 && groups>1` | grouped im2col + GEMM **直写 + bias 融合**（opt_iter_4） | 同上，tile 解码 `(n_batch, g, co_b, l_b)`，按 group 独立 GEMM 直写 `out` |

**统一归约维**（im2col 路径）：`K_dim = (C_in // groups) * K`；权重按 `(C_out_per_group, K_dim)` 扁平连续；`x_col` reduction 顺序 `(C_in_per_group, K)` 须与 weight 展开一致。

### §2.3 Layer 3：关键技巧

- **T1/T8 禁布局接口后的 device 侧重排**（C1d-NO-LAYOUT-OPS）：
  - **im2col（stride>1）**：Triton gather kernel 把 strided 输入列收成连续 workspace；每 program 处理一个 `(n, ci, k)`，对 `l=0..L_out-1` 取 `x_pad[n, ci, l*stride + k*dilation]` 连续 store。groups==1 布局 `(C_in, K, N, L_out)`；groups>1 布局 `(groups, C_in_pg, K, N, L_out)`。
  - **group weight 打包**：`weight_pack_grouped_kernel` 逐元素重排 `[C_out, C_in_pg, K]` → `[GROUPS, C_out_pg, K, C_in_pg]`，缓存 key=weight `data_ptr()`（G9）。
  - ⚠️ device gather 明显慢于原 host `unfold/permute`（受约束不得不采用）；stride==1 路径仍优先 shifted block_ptr GEMM。gather 是**指令吞吐 bound（非带宽 bound）**——见 T9 放大 BLOCK_L。
- **T5 shifted 1×1 分解**（groups==1 && stride==1 主力）：K-tap 拆成 K 个 shifted 1×1 dot，reduction 仅 C_in。输入每 tap 一次 block_ptr load 取整块 `[l+k*D, l+k*D+BN)`；权重 device 转置为 `[C_out, K, C_in]`（G8）后 per-tap 读 `w_ptr + k*C_in`，`strides=(K*C_in, 1)`。**动态 BLOCK 分档**：`_gemm_blocks(c_out, c_in_red, n_dim)` 按 `BLOCK_N`(L_out: 128/64/32) / `BLOCK_M`(C_out: 16/32/64) / `BLOCK_K`(reduction: 16/32/64) 分档，受寄存器上限约束（acc[BM,BN]≤2048、b[BK,BN]≤4096、a[BM,BK]≤2048），block_ptr+boundary_check 天然 tile-size 无关、分档无精度风险。
- **T6 general shifted kernel**（stride==1 && groups>1）：T5 推广到 groups，tile 解码 `(n, g, co_b, l_b)`，scalar 加载 `x_pad[n, g*Ci_pg+ci, l+k*DIL]`（stride==1 连续，可喂 dot），权重 block_ptr 读 group-major 打包 `[GROUPS, Cout_pg, K, Ci_pg]`，tap-k base `w_ptr + g*(Cout_pg*K*Ci_pg) + k*Ci_pg`、`strides=(K*Ci_pg, 1)`（单位 stride reduction 轴，G8）。⚠️ **per-k 偏移不可漏**：布局 `[g,co,k,ci]` 下 base 必须含 `+k*Ci_pg`，否则所有 k 读同一 slice（首版全错）。
- **T7 im2col GEMM 直写最终布局 + bias 融合**（stride>1 两路径）：tile 解码增加 `n_batch`（groups 路径再加 `g`，`g = tile_idx // (m_blocks*n_blocks)` 仅读本组 w/x_col），GEMM 直接按 `out[N,Cout,Lout]` 连续布局写入（`out_ptr + nb*(Cout*Lout)`、`strides=(Lout,1)`），store 前融合 `bias[co]`。消除 `out_2d`/reshape/permute/copy/独立 bias kernel。边界：越界 l_tile 读到下一 batch x_col（合法内存）被 store 的 `boundary_check` 丢弃。
- **T2 K_dim 对齐 C0=16**（G10）：`x_col` 与 weight 的 `K_dim` pad 到 16 倍数。
- **T9 im2col BLOCK_L 自适应**（opt_iter_16 验证，stride>1 路径）：im2col gather 的 `tl.load(x_base + l*STRIDE)` 是指令吞吐 bound（3MB 流量有效仅 ~2GB/s，远低于 HBM 1.2TB/s），故更宽的 BLOCK_L 减指令数有效。`_im2col_block_l(L_out)` 取最小覆盖 L_out 的 2 的幂（≤256，floor 64），大 L_out 单次 masked 迭代。实测 case 25（L_out=256）im2col 1.39→1.09ms（−22%），全量 geomean +1.4%。
  - ⚠️ **已证伪的同类方向**（勿重试）：(1) `tl.make_block_ptr(strides=(STRIDE,))` strided load 与 pointer-tensor gather **同速**（0.98x，maxerr=0），Ascend 不把 1D strided block_ptr lower 成更快的 strided DMA；(2) 大 GEMM tile 调优（放宽 acc 上限到 8192）对 case 25 **无效**——分量 profile 显示 im2col gather 占 impl 97%，GEMM <3%，GEMM tile 不是瓶颈。

### §2.4 性能基准

- **里程碑**：scalar gather 0.016x → im2col+GEMM + shifted 1×1 分解（0.43x）→ general shifted kernel 覆盖 groups>1 stride==1（+9%）→ im2col GEMM 直写+bias 融合（+3%）→ **padding 改 `torch.zeros`+`copy_` 跨越 target（0.73→1.06x，核心杠杆）** → 移除 `unfold`/`permute` 合规后 0.83x → opt_iter_15 大 GEMM tile（证伪，GEMM 占比 <3%）→ **opt_iter_16 im2col BLOCK_L 自适应 0.84x**（case 25 gather −22%）。

---

### §2.5 近期验证记录（2026-08-04）

- **任务**: `6_ConvStandard1d` 50-case 多 shape 评测，架构 `ascend910B1`，target_speedup=0.8。
- **结果**: 50/50 verify pass，geomean **0.8348x** vs torch，平均实现延迟 0.0434 ms，达到目标。
- **Phase 3 关键修复（从 iter_0 38/50 → iter_1 50/50）**:  
  1. Path A/B `shifted_gemm_kernel` / `general_shifted_kernel` 内补充 `for ci0 in range(0, C_IN, BLOCK_K)`，保证 `BLOCK_K < C_IN` 时完整规约。  
  2. Path C/D `im2col_gemm_kernel` / `grouped_im2col_gemm_kernel` 将 `while k0 < K_DIM_PAD:` 改为 `for k0 in range(0, K_DIM_PAD, BLOCK_K):`，规避 Ascend MLIR 编译器 segfault。  
  3. `x_col` buffer 改为扁平零填充布局 `(K_DIM_PAD, N*L_OUT)` / `(groups, K_DIM_PAD, N*L_OUT)`，避免 K_DIM_PAD 越界。
- **Phase 4**: `triton-latency-optimizer` 未给出可落地的结构性优化点；本轮复测与 Phase 3 基线一致（0.8294x → 0.8348x，测量噪声范围）。

---

## §3 ConvDepthwise2d 算子（2D depthwise）

**算子类别**：`convolution-depthwise-2d`　**典型特征**：`(N, C, H, W)` × weight `[C, 1, kH, kW]`（groups==C），无跨通道规约。
**性能基准**：50/50 pass，geomean **~0.90x** vs torch（target 0.8x，达标）。

### §3.1 Layer 1：特化约束

- **fan_in**（G1）：depthwise `groups==C_in`，故 `fan_in = kH * kW`（**禁止** 写成 `out_channels * kH * kW`）。
- **UB 限制**：`BLOCK_M = 16`；`BLOCK_N ≤ 128`（如 OH_TILE=8, OW_TILE=16）。5D block_ptr 单次 load 会把整个 receptive field `[BLOCK_M, OH_TILE, OW_TILE, kH, kW]` 驻留 RF，总占用约 `BM·OH·OW·KHW·4B`，上限约 50KB；tile 须按 KHW 自适应。
- depthwise **禁止** `tl.dot`（A4）。

### §3.2 Layer 2：三路径 hybrid

| 条件 | 路径 | tile |
|------|------|------|
| stride==1 | 预 pad + **3D block_ptr KHW-loop** | 大空间 16×16，否则 8×16 |
| stride≥2 | **NCHWc pack + 5D block_ptr conv**（A2） | 按 KHW 自适应（见下） |
| k=5/k=7 大空间且 UB 受限 | **split-KHW + atomic_add**（G11） | 放大 spatial tile |

- **3-way pad 分派**（避免统一大 alloc）：stride=1&pad=0 → 透传 x（零开销）；stride=1&pad>0 → halo 分配；stride≥2 → 全 alloc，`H_alloc = max(H+2p, stride*H_out_aligned + kH - 1)`。
- **5D block_ptr tile（按 KHW 自适应）**：KHW≤9(k=3) → OH=4,OW=16（W_out≥32）或 4×8；KHW≤25(k=5) → 4×8；KHW≤49(k=7) → 2×4。
- **NCHWc pack+conv 流水线**：conv 前用 pack kernel 把 `x[N,C,H,W]` 转置为 `x_packed[N, C_BLK, H, W, 16]`（NCHWc），conv 用 5D block_ptr 读 `[OH_TILE, OW_TILE, kH, kW, 16]`，每个 x 元素只读 1 次；conv 内部 `tl.trans` 后**直接写 NCHW 输出**，**禁止**加独立 unpack kernel；**禁止** fuse pack+conv 进单 kernel（实测 geomean 0.90x → 0.59x）。
- **split-KHW + atomic_add 启用阈值**（经验值）：
  - k=5 stride≥2 且 H_out·W_out ≥ 256
  - k=5 stride=1 且 C ≤ 16 且 H_out·W_out ≥ 256
  - k=7 stride=1 且 H_out·W_out ≥ 400
  - k=7 stride=2 不受益；小 case 不受益。

### §3.3 性能基准

50/50 pass，geomean **~0.90x**。收益链：blockptr 向量化 0.0455→0.32x → 5D 单 load 0.50→0.58x → NCHWc pack+conv 0.58→0.83x → split-KHW+atomic 0.83→0.90x。

---

## §4 ConvTranspose2d 算子（2D 转置卷积）

**算子类别**：`convolution-transpose-2d`　**典型特征**：weight `(in_channels, out_channels, KH, KW)`，输出放大。
**性能基准**：50/50 pass，**当前最佳 geomean 0.8297x**（target 0.8x **达标**）。Phase 3 基线 0.7783x → Phase 4 连续块划分优化 0.8297x（+6.6%）。历史最佳 ~0.63x（2026-08 前，基线未做连续块划分）。

### §4.1 Layer 1：特化约束（硬规则）

- **等价转换正确性**：ConvTranspose2d ≡ 输入按 stride·dilation 插零 → 按 `pad = K - 1 - P` 补零 → 与**空间翻转**权重做 stride-1 Conv2d。输出 `H_out = (H_in - 1)*S - 2*P + K`，`W_out = (W_in - 1)*S - 2*P + K`。
- **fan_in**（G1，**与标准 Conv 相反**）：`fan_in = out_channels * KH * KW`；weight shape `(in_channels, out_channels, KH, KW)`。
- **权重空间翻转必须在 Triton kernel 内**（按 `KH-1-kh`/`KW-1-kw` 读），**禁止** host 侧 `torch.flip`。
- **host 侧禁止任何 torch 计算/布局操作**：对 weight 禁 `permute/flip/pad/reshape/contiguous`；对输出禁 `permute/reshape`（禁止为 NHWC→NCHW 转换做 host 侧布局操作）。NHWC workspace → NCHW 必须由 Triton kernel 在 device 侧完成。输出通道对齐由 C4d-EXACT-OUTPUT + C4d-STORE-NCHW-DIRECT 处理，**禁止** host 侧切片（见下）。
- **NHWC workspace 连续对齐**：`x_nhwc` shape `(N, H_pad, W_pad, C_in_pad)`，`C_in_pad = ceil(C_in/16)*16`；`H_pad = H_out + KH - 1`，`W_pad = W_out + KW - 1`。
- **weight 转换结果必须缓存**（G9/C5）：首次 forward 调 `weight_prep_kernel` 生成 `(KH*KW*C_in_pad, C_out_pad)` 并缓存；后续复用。**不缓存** geomean 从 ~0.60x 跌至 ~0.24x。
- **C4d-NO-SELF-HELPERS-IN-FORWARD**：`forward()` 内**禁止**调用 `self.xxx(...)` 形式的 helper（`validate_triton_impl.py` 判为 Type3 退化）。workspace cache / weight cache 必须**内联**在 `forward()` 内，通过 `self._workspace_cache` / `self._weight_cache` dict 直接读写。
- **C4d-WEIGHT-INIT-MODULE-LEVEL**：权重/偏置创建必须放在模块级 helper（如 `_create_weight_bias`）。**禁止实例化 `nn.ConvTranspose2d(...)` / `nn.Conv2d(...)` 来"获取权重"**——必须按 `reset_parameters()` 原理手工复刻（同 §1 G1 / 第 48 行）。**禁止**在 `ModelNew.__init__` / `forward()` 内调用 `nn.ConvTranspose2d`。
- **C4d-EXACT-OUTPUT**：输出张量**直接按精确 `out_channels` 分配** `out = torch.empty(N, out_channels, H_out, W_out, dtype=x.dtype, device=x.device)`，conv kernel 内 3D block_ptr store 用 `shape=(C_out, H_out, W_out)` + `boundary_check=(0,1,2)` 掩码最后一个通道 tile 的 padded 列（`C_out_pad` 仍用于 weight 形状/bias pad）。**禁止**分配 padded `out_workspace[N, C_out_pad, ...]` 再 host 侧 `out_workspace[:, :out_channels, ...].contiguous()` 切片——那次全输出张量拷贝是纯开销，大输出时显著拖累延迟。
- **C4d-STORE-NCHW-DIRECT**：conv kernel 输出必须通过 UB 内 `tl.trans(acc, 1, 0)` + 3D `tl.make_block_ptr` 直接写入**精确 `out_channels`** 的输出张量（与 C4d-EXACT-OUTPUT 配套），block_shape `(BLOCK_N, 1, OW_TILE)`，order `(2,1,0)`，`boundary_check=(0,1,2)`。**禁止**先写 NHWC 再调独立 transpose kernel、禁止 host `permute`、禁止 `out.view(...)` + pointer+mask store、禁止 padded workspace + host 切片。
- **C4d-DETERMINISTIC-CONV-CONFIG**：`conv_im2col_kernel` **禁止** `@triton.autotune`，`BLOCK_C` 由 host 侧确定性给出：`BLOCK_C = min(≤C_in_pad 的最大 2 的幂, 64)`（`C_in_pad≥64→64`、`≥32→32`、否则 16），`OW_TILE` 固定 64，二者作为 `tl.constexpr` 传入。`C_in_pad` 恒为 16 的倍数 → 2 的幂 tile 整除无碎片。autotune 对每个 shape key 做 timing 探测会引入运行间不可复现噪声，确定性启发式既稳定又更快。`weight_prep_kernel` 的 autotune 可保留（仅 cache-miss 跑一次，稳态无影响）。

### §4.2 Layer 2：三阶段流水线

1. **weight_prep_kernel**：device 上把 raw weight `(C_in, C_out, KH, KW)` → `(KH*KW*C_in_pad, C_out_pad)`，同时完成空间翻转 + channel padding。加 `@triton.autotune` 覆盖 `(BLOCK_C_IN, BLOCK_C_OUT)`，默认 configs 至少 `{16,32,64}×{16,32}`，`key=["C_in_pad", "C_out", "KH", "KW"]`（此 kernel 仅 cache-miss 跑一次，autotune 探测开销摊销到 warmup、稳态无影响，故保留）。
2. **prep_row_s1_kernel / prep_row_stride_kernel**：NCHW → dilated+zero-padded NHWC workspace。stride=1 与 stride>1 分两个 kernel，均用 `tl.make_block_ptr` 连续 load/store；workspace 通过 `self._workspace_cache` 按 shape 缓存复用（避免每次 forward 重新零填充）。
3. **conv_im2col_kernel**：在 NHWC workspace 上做显式 KH×KW `tl.static_range` 循环，内部 `tl.dot`。**禁止** `@triton.autotune`——`BLOCK_C`/`OW_TILE` 由 host 侧确定性给出（C4d-DETERMINISTIC-CONV-CONFIG），作为 `tl.constexpr` 传入。Host 侧 `BLOCK_N` 按 `out_channels` 选 16/32/64 并 pad `C_out_pad`（weight/bias 维度对齐）；输出直接写入**精确 `out_channels`** 张量（C4d-EXACT-OUTPUT），store 用 §4.3 的 UB transpose + 3D block_ptr + `boundary_check`（C4d-STORE-NCHW-DIRECT）。

### §4.3 Layer 3：关键技巧片段

**weight_prep_kernel 行映射**（raw `(C_in,C_out,KH,KW)` → out `(KH*KW*C_in_pad, C_out_pad)`）：
```python
kh_flip, kw_flip = KH - 1 - kh, KW - 1 - kw
out_row = (kh * KW + kw) * C_in_pad + c_in
# load raw block at (c_in, c_out) with spatial (kh_flip, kw_flip); store out at (out_row, c_out)
```

**UB transpose + NCHW store**（conv kernel 末尾，写入精确 `out_channels` 张量，C4d-STORE-NCHW-DIRECT）：
```python
acc_trans = tl.trans(acc, 1, 0)                 # (BLOCK_N, OW_TILE)
o_ptr = tl.make_block_ptr(
    base=out_ptr + n * stride_o_n,
    shape=(C_out, H_out, W_out),                # C_out = 精确输出通道，非 C_out_pad
    strides=(stride_o_c, stride_o_h, stride_o_w),
    offsets=(c0, oh, ow0),
    block_shape=(BLOCK_N, 1, OW_TILE),
    order=(2, 1, 0),
)
tl.store(
    o_ptr,
    tl.reshape(acc_trans, (BLOCK_N, 1, OW_TILE)).to(out_ptr.dtype.element_ty),
    boundary_check=(0, 1, 2),                    # 掩码最后一个通道 tile 的 padded 列
)
```

### §4.4 性能基准与经验教训

- **当前最佳**：50/50 pass，geomean **0.8297x**（target 0.8x **达标**）。Phase 3 基线 0.7783x（已超旧 ~0.63x 天花板）→ Phase 4 连续块划分 0.8297x（+6.6%）。
- **历史改动链**：① `conv_im2col_kernel` 去 `@triton.autotune` → host 侧确定性 `BLOCK_C = min(pow2≤C_in_pad, 64)`（C4d-DETERMINISTIC-CONV-CONFIG）；② 输出直接分配精确 `out_channels` + store `boundary_check`（C4d-EXACT-OUTPUT）→ 旧基线 ~0.593x 提到 ~0.63x；③ 三阶段 device 流水线（weight_prep / dilate+pad NHWC workspace / fused conv GEMM 直写 NCHW）+ 全缓存（conv_key 缓存 weight/Wmat、data_ptr+shape 缓存 workspace）→ Phase 3 基线 **0.7783x**；④ **连续块划分 + 最内层 ow_blk 解码**（见 §4.3 trick）→ **0.8297x**。
- **Phase 3 基线风险**：若生成时未遵守 C4d-NO-SELF-HELPERS-IN-FORWARD / C4d-STORE-NCHW-DIRECT / C4d-EXACT-OUTPUT / C4d-DETERMINISTIC-CONV-CONFIG，会退化到 pointer+mask + 无 workspace cache + autotune 路线，geomean 仅 **~0.4479x**。
- **Phase 4 已验证无效方向（勿重试）**：每个 program 处理 2 行 oh 并用运行时 `if valid1:` 包绕第二个 `tl.dot` 来复用 `w_blk`（loop-invariant hoisting 思路）→ **MLIRCompilationError 全 case 失败**（见 G13 新增条目）。w_blk 跨 oh 复用本身是对的，但**禁止用运行时谓词包 dot**；若要做跨行复用，须用编译期 `tl.constexpr` 分支或无条件双累加器，且需评估 UB 容量。

### §4.3a Layer 3 补充技巧：连续块划分 + 最内层 ow_blk 解码（Phase 4 验证 +6.6%）

**问题**：交错划分 `for tid in range(pid, total, NUM_CORES)` 把"相同 (n,co_blk,oh) 的相邻输出列 tile"打散到不同核，`w_blk`（仅依赖 kh,kw,ci0,co_blk，与 ow 无关）跨 tile 的 L1 复用被破坏。

**解法**（G2 连续块划分 + 解码顺序）：
```python
# 连续块划分：每个核拥有连续 tile 区间 [start, start+cnt)
tpc_full = total // NUM_CORES
rem = total - tpc_full * NUM_CORES
if pid < rem:
    start = pid * (tpc_full + 1); cnt = tpc_full + 1
else:
    start = rem * (tpc_full + 1) + (pid - rem) * tpc_full; cnt = tpc_full
for off in range(0, cnt):
    tid = start + off
    # 解码顺序：n 外、co_blk、oh、ow_blk 最内层
    n = tid // (CO_BLOCKS * HOUT * OW_BLOCKS); ...
    ow_blk = <最内层余数>
```
- `ow_blk` 作为**最内层**解码维 + 连续块划分 → 一个核连续扫描相同 (n,co_blk,oh) 的相邻 ow_blk → `w_blk` 从 L1 命中复用，减少 Wmat 重读。
- ConvTranspose2d 50-case 实测 geomean **0.7783 → 0.8297**（+6.6%），最差 case 改善显著（case45 0.50→0.61、case34 0.45→0.52）。
- ⚠️ `start/cnt` 计算的 `if pid < rem` 在循环**之前**，不包绕 dot；循环内 `for off in range(0, cnt)` 无运行时谓词包 dot（否则触发 G13 编译崩溃）。
- grid 仍为 `(min(total, NUM_CORES),)`（一维，G2）；`total < NUM_CORES` 时 `tpc_full=0, rem=total`，每核恰好 1 tile。

### §4.5 近期验证记录（2026-08-06）

- **任务**: `10_ConvTranspose2d` 50-case 多 shape 评测，架构 `ascend910B1`，target_speedup=0.8。
- **结果**: 50/50 verify pass，geomean **0.8297x** vs torch，平均实现延迟 0.0108 ms，框架延迟 0.0082 ms，**达到目标**（`target_reached=true`）。
- **Phase 3**（iter_0 一次通过）: 三阶段 device 流水线 + 全缓存 + 确定性 BLOCK_C + 精确 Cout 直写 NCHW，geomean **0.7783x**（已超旧 ~0.63x 天花板）。
- **Phase 4**: opt_iter_0（2-oh w_blk hoist，运行时 if 包 dot）编译失败回退；opt_iter_1（连续块划分 + 最内层 ow_blk 解码，见 §4.3a）成功，geomean **0.8297x**（+6.6%）。
- **瓶颈 case**: 17/3/24/16/34 等为小通道 + 大 K（in/out ≤8、K∈{4,5}），受 Cube C0=16 通道对齐 padding 浪费约束（Cin=1/Cout=1 时最高 ~94% 规约为 padding 零），属该范式天花板；element-wise direct conv 按 §8.1 已验证无效/aicore-timeout 风险，未尝试。

---

## §5 ConvStandard3d 算子（3D 标准 / grouped，源 benchmark 实际 `kernel_size=(k,k,1)`）

**算子类别**：`convolution-3d-standard-grouped`	**典型特征**：`(N, C, D, H, W)` × weight `[C_out, C_in//groups, Kd, Kh, 1]`；源 benchmark 固定 W 维 kernel 为 1，因此等价于在 D-H 平面做 2D 卷积并对 W 切片独立计算。
**性能基准**：60/60 pass，geomean **1.1332x** vs torch（target 0.8x，**达标**）。主力提升来自 P3 改为 no-permute 布局 + HW 展平 dot（opt_iter_26，见 §5.3 T20 / §5.5）。

### §5.1 Layer 1：特化约束

- **fan_in**（G1）：`fan_in = (C_in // groups) * Kd * Kh`（`Kw=1`，不显式乘）。
- **kernel_size 退化维显式处理**：任一维为 1 时坐标直接按 `ox*stride - pad` 计算，**禁止**写该维 `for k in range(1)` 循环。
- **batch 偏移必须用每 batch 平面体积**（⚠️致命）：`base = x_ptr + n * CDHW + ...`，其中 `CDHW = C*D*H*W`。**禁止**用 `n * NCDHW`（会跳到第 N 个 batch 之后，batch>1 时后续 batch 全零/越界）。
- groups>1 按组处理（G12）：`g = co // C_out_per_group`，权重 shape `[C_out, C_in//groups, Kd, Kh, 1]`。
- **forward() 禁止 torch/F 布局/计算接口**：padding、pack、permute、copy 切片赋值等必须封装进 `@triton.jit` kernel（G6）。

### §5.2 Layer 2：路径选择优先级

1. `C_in_per_group==1 and C_out_per_group==1` → **scalar depthwise fallback**（4D block_ptr depthwise 在该 workload 上已证伪，必须回退）。
2. `Cout_pg < 16` 且 groups 足够大、`GROUPS_PER_M * Cin_pg ≤ 16` → **groups-batched M=16 path**。
3. `stride==1`、`C_in_per_group≤16`、`C_out_per_group≤64`、`K≤5`、`H_out≥2` → **P3 no-permute 布局 + HW 展平 dot**（主力，详见 §5.3 T20）。
4. `stride==1` → **no-permute NC1DHW0 path**（保留 `[N,g,C1,C0,D,H,W]`，W 内层连续）。
5. 其余 → **标准 NC1DHW0 pack + 1×1 卷积分解**（A1+A2，主路径；x 打包 `[N,g,C1,D,H,W,C0]`）。

### §5.3 Layer 3：关键技巧

- **T7 NC1DHW0 + 1×1 分解**（A1 算法级突破，主路径）：`Kd·Kh` 卷积拆为 `Kd·Kh` 个 shifted 1×1 dot，reduction 维降到 C0=16。x 打包 `[N, groups, C1_pg, D, H, W, C0]`；weight 打包 `[groups, Cout_pg, C1_pg, Kd, Kh, C0]`；padding 做进 x，kernel 内用 `boundary_check` 处理输出边界。
  - **`c1_inner` 循环必须用 `tl.static_range`**（普通 `range` 导致结果错误）。
  - **weight c1_inner stride = `Kd*Kh*C0`**（**非** `C1_pg*Kd*Kh*C0`；C1_pg=1 会掩盖 bug，C1_pg>1 全失败）。
- **T14 no-permute（stride==1）**：当不进入 HW-flat 门控时，可直接从 padded `[N,C,D,H,W]` 以 strides `(D_pad*H_pad*W_pad, 1)` load `(C0, W)` block，与 packed weight 做 `tl.dot([BLOCK_COUT,C0], [C0,BLOCK_W])`，跳过 `pack_input` transpose。stride>1 必须 permute 到 `[N,g,C1,D,H,W,C0]`，否则 block_ptr 非法 / gather 回退（-26%~-50%）。
- **T9 groups-batched M=16 path**（`Cout_pg ≤ 8`）：多个 groups×Cout_pg 批量进 M=16 做饱满 Cube dot；x 重打包 `[N, M_BATCH, D, H, W, C0]`，weight 重打包 block-diagonal `[M_BATCH, 16, Kd, Kh, C0]`；`M_FLAT = GROUPS_PER_M * Cout_pg` 必须作为 `tl.constexpr` kernel 参数；**gp=4 不启用**（Python packing 开销高反而退化）。
- **T15 depthwise 路径**：4D block_ptr depthwise 在该 60-case 集合上 geomean 从 0.2976x 跌至 ~0.2745x，必须回退到 scalar direct-conv fallback；用 `BLOCK_W=32/16` 让 W 单块化，`BLOCK_C=min(16, pow2_ceil(C))`，`BLOCK_D/H=4`，K 循环 `tl.static_range`。
- **T20 P3 no-permute 布局 + HW 展平 dot（stride==1 主力路径，达标 1.1332x）**：仅当 §5.2 条件启用（必须严格门控，否则 aicore timeout / 编译极慢）。
  - **输入布局 no-permute** `[N, g, C1_pg, C0, D_pad, H_pad, W_pad]`（C0 stride=`Dp·Hp·Wp`，**W stride=1** 最内层）。由 `_pack_x_nc1dhw0_grouped` 产出：`torch.zeros` 预置 padded workspace（spatial pad 区 + channel 对齐到 C0=16 的多余通道天然为零）+ `pad_copy_4d_kernel` 连续 4D block-copy（只写 interior `[BLOCK_H, BLOCK_W]`，无 F.pad / permute / slice 赋值，G6）。pack 是纯连续拷贝（非 gather），效率高于旧 C0-HW 交错布局。
  - **HW 展平 dot**：`Kw=1 且 stride==1` ⇒ `W_out==W_pad`，(kd,kh) halo 在展平 H·W 轴上是常数（`id_pad·HW + kh·dil·W_pad`，其中 `id_pad=od·stride+kd·dil`），整个 (oh,ow) tiling 折叠为 1D flat HW 块 `(od, hw_block)`。每 tap：block_ptr load `x_tile[C0, BLOCK_HW]`（`shape=(C0, HW_out)`, `strides=(DHW, 1)`, `order=(1,0)`, `boundary_check=(0,1)`）+ `w_tile[BLOCK_COUT, C0]`（wpk `[groups,Cout_pg,C1,K,K,C0]`，c0 stride=1，G8）+ `tl.dot(ieee)` 累加；末尾融合 bias + flat-HW store。
  - **关键（二者缺一不可）**：① no-permute 物理布局（pack 连续拷贝，C0 大 stride 但 W/H 连续）；② flat-HW dot（保留大 `BLOCK_HW` ≤256 连续 Cube load，tile 数少）。**仅做 no-permute 但改用分离 W 分块**（`BLOCK_W≤64`、tile 数 `Hout×` 增加、丢失大块 load）→ 0.4632x 回退（opt_iter_25 已证伪，见 §8.1）。
  - **超参**：`BLOCK_COUT` = Cout_pg≤16→16 / ≤32→32 / 否则 64；`BLOCK_HW = max(16, min(256, next_pow2(H_out·W_out)))`（`=512` 必 aicore timeout）；`hw_blocks=ceil(HW_out/BLOCK_HW)`；`total_tiles = N·groups·co_blocks·D_out·hw_blocks`；`NUM_CORES = min(total_tiles, CUBE_CORE_NUM)`（Cube dot 路径）。
  - **FULL_COUT store 分支**（跨 group co 溢出）：store 用绝对地址 `shape=(C_out, HW_out)`（截总 C_out 不截 group 边界），故仅当 `BLOCK_COUT==Cout_pg`（`FULL_COUT=True`）才 block store；否则 masked scalar store（block store 会写零污染下一 group）。masked store 不加 `tl.where` 钳位 pointer（依赖 mask 语义）。
  - **stride 不可省**：kernel 形参含 `stride: tl.constexpr`（`id_pad = od·stride + kd·dil`），但 P3 门控恒 stride==1，host **必须显式传 `1`**（旧 hwflat kernel 隐式假设 stride==1 不接 stride 参数；迁移到带 stride 的 kernel 时漏传 → `NameError: name 'stride' is not defined`，34 case 全 fail）。
  - **被取代的旧方案**：旧 P3 用 C0-HW 交错布局 `[N,g,C1,Dp,C0,HWpad]`（`pack_input_hwflat_kernel` + `conv3d_hwflat_kernel`），geomean 0.6745x；no-permute+HW-flat dot 把 pack 从 gather 改为连续拷贝且 compute 保留大块 load，P3 per-path geomean 0.7585x→1.9182x，整体 0.6745x→1.1332x 达标。
- **block_ptr store 跨 group 溢出**（groups>1 且 `C_out_per_group < BLOCK_COUT`）：`boundary_check` 只截 C_out 不截 group 边界，masked 行越过 group 边界会写零污染下一组。**解法**：constexpr 分支——满 tile（`BLOCK_COUT==C_out_per_group`）才用 block store，否则 masked scalar store；**masked scalar store 不要额外加 `tl.where` 钳位 pointer**（groups 小 Cout case 严重衰退 0.2x 级），依赖 mask 语义。
- **Pack + Padding 融合**：对标准 / HW-flat / M=16 路径，用 `torch.zeros` 初始化 packed workspace，`pack_input_kernel` 只读取原始 `x` 的有效 interior 并写到 `x_packed` 的 `[P:P+W]` 列；padding 位置天然保持零。这样省掉 `x_pad = torch.zeros(...) → copy_(x)` 的完整 memory pass。
- **T21 Pack + Padding 融合的进一步源侧迭代**：在 Pack+Padding 融合基础上，将 pack kernel 的迭代空间从 padded 输出空间 `N·groups·C1_pg·D_pad·H_pad` 改为**源侧有效区域** `N·groups·C1_pg·D·H`，目标 padded 坐标直接由源坐标推导：`pd = sd + padding`、`ph = sh + padding`（无需边界判断）。W 维因 stride 映射不可逆，仍保留目标侧 `W_out` 循环，通过 `sw = ow*stride - padding` + `sw_ok` mask 处理。效果：消除 D/H 维度的 `d_ok/h_ok` 分支与坐标钳位，grid 缩小，对 pad>0 case 有额外 3-8% 提升。
  - **P3 no-permute 路径的标准实例**：`_pack_x_nc1dhw0_grouped` + `pad_copy_4d_kernel`（见 §5.3 T20）即本技巧的 4D block-copy 版本——源侧 tile `(ng, c, sd, sh_blk, w_blk)`，目标 `out[ng, c, sd+pad, sh+pad, sw+pad]`，pad 区由 `torch.zeros` 预置。
- **T13 weight packing 缓存**（G9）：key 含完整 conv_key + variant tag；仅对计算重路径（mgrouped / NC1DHW0 / im2col weight_pad）缓存，cheap 路径缓存得不偿失。

### §5.4 性能基准与优化轨迹

- **当前最佳（应复现方案）**：opt_iter_26，60/60 pass，geomean **1.1332x**（**达标** target 0.8x），平均实现延迟 0.0597 ms，框架延迟 0.0464 ms，vs Phase 3 基线 0.0531x = **21.34×**。
- **Phase 3 baseline**：im2col + 分组 GEMM，60/60 pass，geomean **0.0531x**。
- **关键里程碑**：
  - opt_iter_1：NC1DHW0 pack + 1×1 分解（T7 主路径骨架）。
  - opt_iter_5/6/7：Pack+Padding 融合（T21）+ 各路径 packed block_ptr load 补齐 → 五路径完整（旧基线下 0.7789x）。
  - opt_iter_19（2026-08-07，0.6745x）：新验证基线下补齐 P2/P3/P4/P5 packed 路径 + 调参，target 0.8x 未达。P3 当时用 **C0-HW 交错布局** `[N,g,C1,Dp,C0,HWpad]`（已被取代）。
  - **opt_iter_26（2026-08-08，1.1332x，达标）**：P3 改 **no-permute 布局 + HW 展平 dot**（§5.3 T20），P3 per-path geomean 0.7585x→1.9182x，整体达标。
- **沿用至今的有效改动**（opt_iter_19 起，opt_iter_26 未变）：
  1. **depthwise 并入 P2**（`Cin_pg==Cout_pg==1` 走 P2 而非 P1 scalar）：6 个 depthwise case geomean ~0.28x→~0.76x；P1 门控置 `False`。
  2. **P2 `M_FLAT=32`（`Cout_pg==4`）**：`GROUPS_PER_M=8`，case 20/45 等提升。
  3. **P4 no-permute `BLOCK_COUT` 上限 128**：`Cout_pg>64` 取 128。

### §5.5 近期验证记录（2026-08-08）

#### opt_iter_26（2026-08-08，P3 改 no-permute + HW 展平 dot → 达标 1.1332x，当前最佳）

- **任务**: `8_ConvStandard3d` 60-case 多 shape，架构 ascend910B1，target_speedup=0.8。
- **结果**: 60/60 verify pass，geomean **1.1332x** vs torch（**达标** `target_reached=true`），平均实现延迟 0.0597 ms，框架延迟 0.0464 ms，vs Phase 3 基线 0.0531x = 21.34×，vs opt_iter_19 0.6745x = +68%。
- **per-path geomean**: P3 **1.9182x**（34 case，主力）/ P2 mgrouped **0.7541x**（12 case）/ P5 NC1DHW0 **0.5285x**（9 case）/ P4 no-permute **0.3315x**（5 case）。
- **变更（P3 重构）**：
  - packer：`_pack_x_nc1dhw0_grouped` → `[N,g,C1,C0,Dp,Hp,Wp]`（torch.zeros + `pad_copy_4d_kernel` 连续 4D block-copy，T21 实例）。
  - compute：`conv3d_nc1dhw0_hwflat_kernel`（读 no-permute 布局 + flat-HW dot，详见 §5.3 T20）。
  - 旧 `pack_input_hwflat_kernel` + `conv3d_hwflat_kernel`（C0-HW 交错布局）已移除。
- **最终门控**（沿用 opt_iter_19，仅 P3 内部实现替换）：
  ```python
  if False and Cin_pg == 1 and Cout_pg == 1:        # P1 disabled (depthwise 并入 P2)
      _launch_depthwise(...)
  elif Cout_pg < 16 and (16 // Cout_pg) * Cin_pg <= 16 and groups != 4 and stride == 1:
      _launch_mgrouped(...)                         # P2
  elif stride == 1 and Cin_pg <= 16 and Cout_pg <= 64 and K <= 5 and Hout >= 2:
      _launch_hwflat(...)                           # P3 = no-permute + HW-flat dot
  elif stride == 1:
      _launch_nopermute(...)                        # P4
  else:
      _launch_nc1dhw0(...)                          # P5
  ```
- **关键超参（必须按此取值才能复现 1.1332x）**：
  - P2 `M_FLAT`：`Cout_pg==4 → GROUPS_PER_M=8, M_FLAT=32`；否则 `GROUPS_PER_M=16//Cout_pg, M_FLAT=GROUPS_PER_M*Cout_pg`。
  - P3 `BLOCK_COUT`：Cout_pg≤16→16 / ≤32→32 / 否则 64；`BLOCK_HW=max(16,min(256,next_pow2(H_out·W_out)))`；`FULL_COUT=(BLOCK_COUT==Cout_pg)`；`NUM_CORES=min(total_tiles, CUBE_CORE_NUM)`；host **传 stride=1**（门控恒为 1）。
  - P4 `BLOCK_COUT`：≤16→16 / ≤32→32 / ≤64→64 / >64→128。
  - P5 `BLOCK_COUT`：≤16→16 / ≤32→32 / 否则 64。
- **迁移要点**：
  - P3 kernel 形参含 `stride`/`dilation`（tl.constexpr），但 `_launch_hwflat` 不接 stride（P3 门控恒 stride==1）→ host 须传 `1`，否则 `NameError`（34 case 全 fail）。
  - wpk 布局 `[groups,Cout_pg,C1,K,K,C0]` 复用 `pack_weight_kernel`（不变）；x 用新 packer；out 标准 `[N,Cout,Dout,Hout,Wout]`。

---

## §6 ConvStandard2d（2D 标准 + grouped + depthwise-like）

**算子类别**：`convolution-2d`　**典型特征**：`(N, C_in, H, W)` × weight `[C_out, C_in//groups, K, K]` → `(N, C_out, H_out, W_out)`；`stride/padding/dilation/groups/bias` 全部为 forward 运行期入参（非构造期固定），覆盖标准 / grouped / 1×1 / depthwise-like（`Cin_pg==1 && Cout_pg==1`）变体。
**性能基准**：50/50 verify pass，geomean **0.6677x** vs torch aclnnConvolution（target 0.8x，**未达**）。多路径混合分派实现（§6.2），最终代码 = 工作目录 `7_ConvStandard2d_generated.py`。

### §6.1 Layer 1：特化约束

- **fan_in**（G1）：`fan_in = (C_in // groups) * K * K`；权重缓存键必须是**完整 conv_key** `(in_channels, out_channels, K, stride, padding, dilation, groups, bias)`。
- **C2d-WEIGHT-CACHE-INSTANCE-RESET ⚠️ 权重/布局缓存必须实例隔离**：每个实例代表一次全新评测上下文，缓存放实例内字典（键 = 完整 conv_key）；若用模块级缓存，必须在 `ModelNew.__init__` 清空，否则前 case 的权重/派生布局污染后 case，出现**序列相关的 AccuracyError**（实测 case 49）。
- **C2d-DOT-OPERAND-CONTIG ⚠️ tl.dot 操作数不得来自 strided/computed-offset gather**：非连续（stride>1 滑窗 / 跨步采样）输入喂 Cube 前，先在 host 侧重排成仿射可达布局（奇偶平面拆分 T2d-2 / channel-last 打包 T2d-7 / 预补零 T2d-1），或物化连续 im2col（load 侧 gather、store 侧连续写）再 GEMM（T2d-9）。**禁止** computed-offset pointer load 直喂 dot。
- **C2d-UNIFIED-GRID**：所有路径统一一维 grid = tile 总数与 vector 核数取小（`vector_core_num`，兜底 40）+ kernel 内 grid-stride 循环（每核从自身序号按总核数步进），dot 路径与向量路径共用同一核数模型。
- **C2d-NO-MIN-MAX**：host 侧禁 `min()/max()`（validator 视为计算操作 → Type3 退化判定）；grid 裁剪用条件表达式：`v = cap if total >= cap else total; return v if v >= 1 else 1`。
- **C2d-BM-GROUP-CLIP ⚠️ BLOCK_M 超过 Cout_pg 时必须按组裁剪**：Cube M 维下限 16，`Cout_pg<16` 时 BLOCK_M 强制补到 16 会多出越组行——这些行用**本组输入**算了**别组输出通道**并写穿。store/bias 的 mask 必须含 `(pid_co*BLOCK_M + tl.arange(0, BLOCK_M)) < Cout_pg`（G12 强化版）。
- depthwise（`Cin_pg==Cout_pg==1`）**禁止** `tl.dot`（A4：无跨通道规约，Cube 16³ 阵列浪费 15/16 行）。
- 权重 dtype 需与输入对齐（`weight.to(x.dtype)`）；`bias is None` 时传 x 本体当 dummy 指针 + `has_bias=False` constexpr（该组合在本算子各 kernel 上实测可正常运行）。
- **C2d-NO-CLAMP-IN-LOAD-ADDR ⚠️ 钳位（最小/最大）不得进入 load 地址表达式**：逐元素钳位作用在指针偏移上后，编译器无法证明地址仿射性，向量化 load 退化为逐元素标量 load（实测 10–70x 回退）。边界处理首选 host 预零填充使地址恒界内（T2d-1）；确需 kernel 内防御时，钳位只用于把地址钳到合法区间防崩溃、数值置零仍由 mask 负责（masked OOB 地址在部分后端触发地址对齐崩溃，S=2 实测）。
- **C2d-S2-VECTOR-LOAD-FORBIDDEN ⚠️ stride≥2 的向量化输入 load 一律禁止**（masked/unmasked、块指针/计算指针所有写法均中招）：本后端统一误编译（aicore 507015 崩溃或数值错）。stride-2 采样的唯一安全解 = host 把补零输入按列奇偶拆成两个连续平面，平面内 stride-1 仿射访存（T2d-2）。
- **C2d-HOST-COPY-PATHOLOGY ⚠️ host 侧矩形切片拷贝 / stride 视图拷贝 / 通道重排是隐形大瓶颈**：这些 torch 张量操作在 NPU 上落到 AiCpu 慢速拷贝路径（实测单次 92–176µs，可占端到端 70%+）。替代：行连续 load + 行连续 store 的专用拷贝核（已验证的唯一高效原语；列方向 stride-2 访存即触发标量化）；多段 host 预处理应融合为单核（补零 + 平面拆分融合后再提速约 2x）。
- **C2d-MACS-GATE-HOST-OVERHEAD ⚠️ 带 host 预处理（补零/重排/拆分）的路径必须按算力量门控**：host 固定开销约 40–60µs，小算力 case 无法摊薄（实测劣化至 0.67–0.75x）。forward 内按「输出元素数 × 输出通道数 × 规约维」估算乘加量设下限阈值，不达标回退无预处理路径。
- **C2d-NO-CONDITIONAL-BASE-POINTER**：运行期标量 if 包裹的双 load 路径（如「内部 tile 无 mask 快路径」）、以及编译期 if/else 切换双基址指针，在本后端均误编译（前者 dot 输出全零、后者误差 ~125 量级）。等价安全写法：把「选哪个基址」折成「平面序号 × 平面步长」的仿射偏移加到单一基址上。
- **C2d-PARALLELISM-COLLAPSE-GUARD**：行块化/放大 tile 使 tile 总数下降后，必须校验 tile 总数与核数同量级（实测 tile 数 8 < 24 核时并行度塌陷，回退 0.83x）；不足时回退小块配置。
- **C2d-SPLIT-RESULT-MATERIALIZE ⚠️ `tl.dot` 操作数必须 make_block_ptr 产物；`tl.split` 产物喂 dot 前必须 `tl.where(mask, x, 0)` 物化**（实测）：`reshape → split` 的直接输出作为 dot 操作数触发 `getIndexingMaps/ConvertLinalgRToBinary` 编译 abort；经 `tl.where` 折叠成普通 buffer 后稳定。**强化（终局验证）**：ptr-arith 地址、masked load、`tl.where` 物化、clamp 地址任何一种形式喂 `tl.dot` 一律编译期 SIGSEGV——dot 输入**唯一合法形态是 make_block_ptr**（含其经 `tl.where` 物化的稳定变体）。因此「mask 读原图省 pad」类融合在 dot 路径不可实现（§8.1 #18）；W 边界行折叠 mask 无法用 block_ptr 表达（行坐标折叠在 dim1 offset 中）。
- **C2d-NO-VECTOR-INTDIV ⚠️ 向量 `//`（整除）触发标量降级，memory kernel 实测 22x 劣化**：2D tile 行重排用 `r_off // H` → 20637µs（同 kernel program_id 分解版 937µs 的 22 倍）。行/列重排一律用 `program_id` 维度分解（如 `program_id(0)=channel` 免除法）或 `tid - (tid // t)*t` 减法代模；标量 if 包绕向量 load/store（762µs）与 `mask & m_scalar` 折叠（4242µs）同样劣化。
- **C2d-ODD-DIM0-BASE-UNRELIABLE ⚠️ make_block_ptr dim0 基址为奇数时 load 不可靠（含纯拷贝路径）**（实测）：dim0 基址 = `通道序号 × 单平面行数`，行数为奇（如 H=7）时基址奇 → load **间歇**取错数据。两个已证实的中招场景：① `[N,C,2,H_alloc,W2]` 交错平面 buffer 用 `r0=(nc*2+p)` 取第二奇偶平面（奇 dim0 基址）时，setitem/CPU 写入的数据读回全零、arange 写入的数据读回正确（与数据创建方式相关，ptr-arith 读同错）；② pad kernel 读原始输入 `base = x_ptr + nc*H_IN*W_IN`，奇 `H_IN*W_IN` 时间歇读错。**通用规避：无 dot 的 kernel（pad/拷贝/重排）一律 ptr-arith + masked load（实测零性能代价）；有 dot 的 kernel 用独立平面 tensor（dim0=通道，无奇基址）或维持 dim0=纯通道寻址**。另注：dim1 offset 含 `ih·W2` 行平面偏移时 COLS_TOTAL 必须传 `H_alloc·W2`（行平面展开长度），传 W2 即越界读零。
- **C2d-SKETCH-MANDATORY-PATHS ⚠️ sketch.txt 必须一次性列出 §6.2 全部 7 条路径**：`triton-op-designer` 生成 `sketch.txt` 时，**禁止**以“先实现 P1/P2/P5 三条路径保底，其余在 Phase 4 增量加入”为由裁剪路径。Phase 2 草图必须显式包含 P1 1×1-GEMM、P2 depthwise、P3 skinny-chlast、P4 S2-plane、P5 tap-window、P6 plane-dot、P7 im2col+kblock-GEMM 的 kernel 名、host 门控条件、输入/权重数据布局与输出 store 方式；Phase 3 的 3.2b 架构符合性检查门将逐项核对「kernel 列表完整性」「路径分派完整性」「禁用模式未出现」。任何路径缺失视为 `A-SketchDeviation`，不得进入精度验证。
- 其余通用约束见 §1 G1–G13。

### §6.2 Layer 2：多路径混合分派（A5）

Host 先按卷积配置（每组输入/输出通道、kc、K、stride、dilation、groups）静态选定主路径并预变换权重，forward 内再按 shape 与算力量动态回退；每条路径独立准备数据布局与启动配置：

| 优先级 | 门控条件 | 路径 | 核心并行策略 |
|---|---|---|---|
| 1 | K==1 且 stride==1 且无 pad/dilation 且 groups==1 | 1×1 纯 GEMM | 输入视作（批, 通道, 空间展平）的标准 GEMM 分块（T2d-8） |
| 2 | 每组输入/输出通道==1（depthwise-like） | 通道块二维 tile 向量乘加 | 权重按 tap-通道连续布局 + 整行宽 tile（T2d-6）；大算力叠加 host 预补零 + 无 mask |
| 3 | 每组输入通道 ≤6 且 kc ≤200 | skinny channel-last 单 dot | 打包规约维 + 行块化权重复用（T2d-7）；乘加量 ≥1.9M 门控，不足回退优先级 5 |
| 4 | stride==2 且 dilation==1 且 groups==1 | W 奇偶平面拆分 | host 融合补零+解交织，kernel 内 stride-1 仿射（T2d-2） |
| 5 | kc ≤200（其余） | 逐窗口 tap dot | 输出 tile（通道块 × 宽块），tap 全展开 + 通道分块 dot（规约块自适应 `pow2_ceil(cin_pg,16,64)`，T2d-12）；stride>1 用「钳位地址 + mask 置零」变体 |
| 6 | stride==1 且 16 ≤ 每组输入通道 ≤128（kc>200） | 零填充 + 逐平面无 mask 大 dot | host 预补零（右侧多补尾 tile 列）→ 全程无 mask 仿射 load，KC 分块 = 单 (kh,kw) 平面（T2d-1）；Wout<20 或小算力回退优先级 5（门槛经实测校准，T2d-12） |
| 旁路 | stride==1 且 K∈{3,5,7} 且 kc≥2000 且 kc×输出元素 ≥20万 | im2col 物化 + K 分块 GEMM | 抢占优先级 5/6 的大 kc case（T2d-9） |

**各路径骨架详述**：

- **路径 1（1×1 纯 GEMM）**：输入零拷贝改视角为（批, 通道, 空间展平），权重按（输出通道, 输入通道）行主序；tile = 输出通道块 64 × 空间块 64，通道规约块 64 循环，ieee dot + fp32 累加，偏置尾加后按 NCHW 偏移掩码直写。
- **路径 2（depthwise 通道块二维 tile）**：权重预重排「kh × kw × 通道」。tile = 通道块（组数 ≥16 取 16，否则取 ≥4 的 2 的幂）× 行块 4 × 宽块（输出宽 ≥128 取 128，否则取 ≥16 的 2 的幂，尽量整行单块）。逐 tap：输入（通道块, 宽块）仿射二维 load（行步长 = 通道步长、列单位步长）× 通道连续权重向量，累加；偏置融合后直写。大算力（批×组×输出元素 ≥32768）时先 host 预补零（T2d-1），kernel 内 load 全程无 mask；小算力保留逐 tap 边界 mask 版本，免 host 开销。
- **路径 3（skinny channel-last 单 dot）**：host 预补零（缓冲高宽按最大窗口坐标构造）后重排为「批 × 组 × 高 × 宽 × 组内通道」（组内通道最内连续）；权重零填充为「组 × 每组输出通道 × K × 对齐打包维」。tile = 通道块（≤64 自适应）× 宽块（≤64 自适应）× 行块（K≤5 且 tile 总数 ≥16 取 4，否则 2）。kernel 内按 kh 循环：权重 tile（通道块, 对齐打包维）只 load 一次供行块内各行复用；每行输入（宽块, 对齐打包维）一次仿射 load（宽方向步长 = stride×组内通道、打包维单位连续）转置后单 dot 累加；奇数尾行按「行号 < 输出高」mask 兜底。
- **路径 4（W 奇偶平面拆分，stride==2）**：host 预零填充「(2, 批, 通道, 高, 平面宽)」五维缓冲（平面宽 = 偶数化补零宽 ÷2），融合核逐行读原始行、按列奇偶拆两半分别连续写（pad 折入列偏移，按 pad 奇偶两种映射）。tile = 通道块 × 宽块；K² tap 展开 × 通道分块循环，行坐标 = 2h+kh 恒界内，平面选择 =「(kw 奇偶) × 平面步长」仿射偏移，平面内列 = w + kw//2 连续——全程无边界 mask（仅通道/宽尾块 mask）。
- **路径 5（逐窗口 tap dot，kc≤200 与兜底）**：tile = 通道块 × 宽块，tile 序号按「批→组→行→宽块→通道块」解嵌套（通道块最内，T2d-4）；K² tap 全展开 × 通道规约块循环（规约块自适应 `pow2_ceil(cin_pg,16,64)`，使 cin=32/64 的 case 每 tap 单次 dot 而非 2/4 次窄 dot，T2d-12），每（tap, 通道块）一次 dot；权重用「组 × kh × kw × 每组输出通道 × 组内通道」转置布局（规约轴 = 组内通道连续）；输入/权重均 masked load。stride=2 变体把越界坐标钳位到界内防地址对齐崩溃、mask 仍负责置零（C2d-NO-CLAMP-IN-LOAD-ADDR）。
- **路径 6（零填充 + 逐平面无 mask 大 dot，stride==1 大 kc）**：host 预补零并右侧多补「步长 × 宽块」列 → 所有窗口地址天然界内且 pad 区为 0（与卷积零填充语义一致）。KC 分块 = 单 (kh,kw) 平面（块大小 = 每组输入通道，整除无尾块）：每平面输入（通道, 宽块）仿射 strided load（行步长 = 通道步长、列步长 = 步长×宽步长）**无 mask**；权重（通道块, 每组输入通道）行主序连续 load；逐平面 dot 累加。Wout<32 时宽块尾块浪费严重，小算力回退路径 5。
- **旁路（im2col 物化 + K 分块 GEMM）**：host 补零（行连续拷贝核，T2d-5）→ 物化核按（批×组, tap, 行, 通道块）划分，源仿射无 mask load、目标行连续写「批×组, 对齐规约维, 输出空间」矩阵（规约维补到 128 倍数；缓冲未初始化分配，尾块垃圾由权重侧零填充对冲）→ GEMM：M = 输出通道块（保持 128 最小化物化矩阵重读）、N = 空间块（空间元素 <64 取 32 否则 64）、K 分块 128；累加器+双操作数受统一缓冲上限约束（通道块 × 空间块 ≤ 128×128）。

**统一要点**：

- 所有 kernel 一维 grid（启动数 = tile 总数与核数取小，核数取 vector 核数、兜底 40，C2d-UNIFIED-GRID）+ kernel 内 grid-stride 循环：每核从自身序号出发按总核数步进处理 tile；tile 解码一律减法代模（G3），输出通道块放最内层（T2d-4）。
- padding 只在需要「kernel 内仿射无 mask」的路径上支付，且补零/重排/解交织全部由专用原语核完成；**通用 pad kernel（`pad_input_kernel`）必须按 2D 平面 tile 实现（~16KB tile，实测带宽 353GB/s；行粒度 1KB tile 仅 48GB/s，T2d-15），读侧 ptr-arith + masked load（C2d-ODD-DIM0-BASE-UNRELIABLE）**；逐窗口兜底路径与 1×1 路径直接读原始输入，零预处理。
- 全路径**直写最终 NCHW 输出 + 偏置融合**（bias 为空时传任一张量当 dummy 指针 + constexpr 关闭），无中间输出重排 / 独立 bias 核。
- 矩阵乘一律 ieee 精度输入 + fp32 累加输出（G7）。
- tile 自适应公式（T2d-3）：通道块 = 128（每组输出通道 ≥128）否则 max(2 的幂, 16)；宽块 = 64（输出宽 ≥64）否则 max(2 的幂, 16)；通道规约块 = 每组输入通道的 2 的幂因子（上限 128），小通道直接取 2 的幂。
- 运行期门控三件套：乘加量下限（摊薄 host 固定开销，C2d-MACS-GATE-HOST-OVERHEAD）、输出宽/输出元素下限（宽块尾块浪费）、tile 总数 ≥ 核数量级（并行度守卫，C2d-PARALLELISM-COLLAPSE-GUARD）。

### §6.3 Layer 3：关键技巧（文字描述）

- **T2d-1 host 预零填充消除 kernel 内边界逻辑（核心范式）**：把 padding 语义全部折进 host 侧预零填充缓冲区——目标缓冲按「最大窗口坐标」构造高宽（保证最深 tap 落在界内），pad 区与尾列天然为零，右侧再额外多补「步长 × 宽块」列覆盖尾 tile 越界 lane。此后 kernel 内所有输入地址恒界内，边界 mask、安全索引、钳位全部消失，load 变成纯仿射无 mask 形式（每条 load 不再背负边界谓词链）。大 kc 的 stride==1 路径、stride==2 路径、depthwise 大算力路径、skinny 小通道路径全部采用。可替代方向：形状小到补零开销占比过高时保留 kernel 内 mask，但须按乘加量门控（C2d-MACS-GATE-HOST-OVERHEAD）。
- **T2d-2 W 奇偶平面拆分（stride==2 的 stride-1 化）**：stride-2 滑窗的本质是「输出列 w 对应输入列 2w+kw」——采样列的奇偶性只由核内列号 kw 决定（编译期常量）。host 侧把补零后的输入按列奇偶性解交织成两个等宽连续平面（合并为带前导平面维的单张量，纯 stride-1 读写）；kernel 内以「(kw 奇偶) × 平面步长」仿射偏移选平面，平面内列号 = w + kw//2 对输出列连续 → 全部 load 退化为 stride-1 仿射；行坐标因缓冲按最大窗口坐标构造而恒界内。补零与解交织融合为单核：逐行读原始行、按列奇偶拆两半分别连续写、pad 区保持零（列偏移按 pad 奇偶分两种映射）。实测把 stride=2 组从 ~0.10x 量级提到 0.3x+。
- **T2d-3 自适应 tile（按实际维度取 2 的幂）**：输出通道块 / 宽块不按阈值固定分档，而是按每组实际输出通道数与实际输出宽「向上取 2 的幂」，夹在下限 16、上限（通道 128 / 宽 64）之间；每组输入通道 <16 时通道规约块直接取通道数的 2 的幂，避免对齐填充浪费。参数扫描实测相对固定分档 +77.6%（geomean 0.1551→0.2755）。要点：小 shape 下固定 tile 的 lane 浪费是首要损失源。
- **T2d-4 tile 序号解嵌套：输出通道块放最内层**：一维 tile 序号按「批 → 组 → 输出行 → 宽块 → 输出通道块」从外到内解码（减法代模），使相邻核拿到**相同空间位置、不同输出通道块**——同一输入 tile 在 L2 中被多个输出通道块复用。输出通道块放最内层是关键（空间维放最内时输入复用消失）。
- **T2d-5 host 拷贝核化（行连续原语 + 病态形状门控）**：host 矩形切片拷贝/解交织等预处理触发 AiCpu 慢速路径时（C2d-HOST-COPY-PATHOLOGY），改用专用拷贝核：行基址用「行号 ÷ 输入高」解码批×通道、余数为源行，每行按输入宽连续 load、目标行基址连续 store。已验证边界：**列方向 stride-2 访存是标量化触发条件，行方向连续则始终编译正确**。门控：仅病态形状（通道多 × 输入窄，如通道 ≥128 且输入宽 ≤32）启用——普通形状 host 拷贝 38–60µs 已快于拷贝核 ~41µs 地板，盲目替换反而回退。
- **T2d-6 depthwise 通道块二维 tile（权重按 tap-通道连续布局）**：tile 为「通道块 × 输出行块 × 整行宽」：权重预重排为「kh × kw × 通道」布局，使每个 tap 的全部通道权重是一次连续向量 load；输入按（通道块, 宽块）二维仿射 load（行步长 = 通道步长，列单位步长）；逐 tap 向量乘加，无矩阵乘。相比「每 tile 单通道单行 + 逐 tap 标量权重 load」实测 8.3–12.4x。大算力时叠加 host 预零填充（T2d-1），kernel 内 load 全程无 mask（实测 275→101µs）。
- **T2d-7 skinny 小通道：channel-last 打包规约 + 行块化权重复用**：每组输入通道 ≤6 时，host 把补零输入重排为「批 × 组 × 补零高 × 补零宽 × 组内通道」布局（组内通道最内连续），使打包序号 r = kw×组内通道+c 成为对宽方向步长 = stride×组内通道 的仿射坐标、且 r 自身单位连续——kernel 内无需在统一缓冲区逐行组装，直接一次（宽块, 对齐规约维）load 后转置做单 dot。规约维 = 打包维对齐到 ≥16 的 2 的幂（lane 浪费从 5.3x 降到 ~1.5x）。行块化：每 tile 处理 2 个（K≤5 时可 4 个）输出行，每个 kh 的权重 tile 只 load 一次被多行复用（实测 280→230µs；行块 4 再 103→95µs）；奇数行由「行号 < 输出高」mask 兜底。可扩展到 grouped（组维进 tile 解嵌套），但需乘加量门控（≥1.9M，不足回退逐窗口路径）+ tile 总数守卫（C2d-PARALLELISM-COLLAPSE-GUARD）。
- **T2d-8 1×1 卷积纯 GEMM 快路径**：K=1、stride=1、无 pad、无 dilation、groups=1 时卷积严格等价于「(输出通道 × 输入通道) 权重 × (输入通道 × 空间展平) 输入」的标准 GEMM——输入零拷贝只改视角，按标准 GEMM 分块（空间维作 N 维），无任何卷积专用逻辑。实测该子集 1.87–3.03x。
- **T2d-9 大 kc 物化 + 分块 GEMM 的 profiling 双门控**：规约维极大（kc ≥2000）**且** GEMM 规模足够（kc × 输出元素 ≥20 万）时，物化 im2col 矩阵（规约维零填充到 128 倍数）再走「K 分块 × 空间 N 维」大 GEMM，优于逐窗口 tap dot——大 K 下 Cube 效率收益超过物化的额外访存往返。两个门控缺一不可：kc 达标但 GEMM 规模不足的 case 物化开销无法回收（实测 0.60x）；kc 不达标时 0.41–0.79x 劣化。省一次清零 launch 的技巧：物化缓冲用未初始化分配，规约尾块的垃圾 lane 由权重侧零填充对冲（垃圾 × 0 = 0）。
- **T2d-10 权重单级缓存 + 派生布局同条目**：按完整卷积配置键建立实例级缓存，一次查表取出该配置的全部派生物——主路径权重布局、逐窗口回退布局、零填充 GEMM 布局（规约维补到 128 倍数）、偏置、每组输入/输出通道数、kc；forward 内不再做任何权重变换。缓存随实例销毁，天然规避跨实例污染（若改用模块级缓存必须在使用前清空，否则序列相关 AccuracyError，C2d-WEIGHT-CACHE-INSTANCE-RESET）。
- **T2d-11 S=2 pad+解交错单步直写 kernel（pair-load + reshape+split）**：C<16 的 S=2 skinny case，把「补零 pad」与「奇偶平面拆分」合并为单 kernel：`make_block_ptr` 一次连续 `(BLOCK_R, 2·BLOCK_J)` pair load → `reshape (R, J, 2)` → `tl.split` → 两条 unit-stride store 直写目标平面布局。实测 7.1µs，替代「pad_input + pad_deinterleave 两步链」173µs（24x）。行解码用 `program_id(0)=channel`（免向量除法，C2d-NO-VECTOR-INTDIV）；写侧行号须加 pad 偏移 `nc·H_alloc + h + P`（目标 buffer 是四周 pad 布局，漏 `+P` 即 max_abs ~4 量级错）；W 相位常量化 `p0=P&1, j0=m+P//2`。
- **T2d-12 分派门槛数据驱动校准**：多路径分派的门槛值（如 P6 的 `Wo`、P5 的规约块 `BCI`）不是语义边界而是性能手段，须用 quickbench A/B 逐 case 强制分派对照校准。已校准结论：P5 `BCI` 由写死 16 改为 `pow2_ceil(cin_pg, 16, 64)` 自适应（cin=32/64 的 case 每 tap 从 2/4 次窄 dot 降为 1 次）；P6 门槛 `Wo>=20`（原 32，放宽覆盖 +4 case）。反向校准同样必要：P5 BCO 上限 64→128 无效（噪声带内）。
- **T2d-13 msprof simulator 终局诊断（瓶颈类型实证）**：fp32 ieee 直接卷积**不是** Cube 计算瓶颈——5 个瓶颈 kernel 实测 CUBE pipe 仅 12–30%，真因 MTE 搬运 53–69%（P6 plane_dot 有效带宽 ~310GB/s ≈ 910B 峰值 78%）+ FLOWCTRL 34%（fused 路径 latency-bound）。环境配方：必须 `export ASCEND_OPP_PATH=$CANN_HOME/opp`（否则 aclnn ZerosLike 报 EZ1013、ParseDynamicKernels 失败无 CSV）；被采集脚本 monkey-patch `torch.zeros/empty` 为 CPU 构造 + `.to('npu')`（H2D copy 可仿真，aclnn ZerosLike 不可）；`code_exe.csv` 行号可能全为 `internal:0`，诊断以 `instr_exe.csv` pipe 占比为准。在宣称「fp32 dot 硬件极限」前必须先采 pipe 占比。
- **T2d-14 小 HW 1×1 GEMM 的 CO tile 放大（BLOCK_CO 16→64）**：P1 1×1-GEMM 路径输出通道 tile 由 16 放大到 64（受益 case 9 kernel -35%/加速比 +52.7%、case 8 +1.4%）。机理：HW 小（如 196）时 sp_blocks 切多块，原 BLOCK_CO=16 导致 co_blocks 也多块，`w_tile`（权重）被每个 sp_block 重复加载 co_blocks 次（base co_blocks=4×sp_blocks=4=16 programs，dot M=16 仅半个 Cube 微块）；放大后 co_blocks=1，权重只读一遍 + dot M=64 吃满 Cube 微块。适用条件：小 HW 1×1 卷积/GEMM、cin≤128（BLOCK_K 上限）、cout 越接近 64 倍数收益越大（cout<64 有 padding 浪费但不劣）。警示：timeit 冒烟（含 dispatch+pack ~0.09ms 固定开销）对该收益完全失明，判定一律以全量 benchmark profiler 为准（方差方法论见 §8.1 #12）。
- **T2d-15 pad/拷贝 kernel 2D 平面 tile 提带宽（48→353GB/s）**：`pad_input_kernel` 按**行**（1KB tile）搬时实测仅 48GB/s，占 (pad+conv) ≥50% 的 case 有 11/50。重写为 2D 平面 tile：`BLOCK_W = pow2_ceil(W, 64, 1024)`、`BLOCK_R = 4096//BLOCK_W`（~16KB tile），grid=(N*C, r_blocks 分核)，stride 循环 `for tid in range(pid, r_blocks, NUM_CORES)`。带宽 48→**353GB/s**（case 28 pad 135.7→18.7µs），套件 geomean +20.79%（case 29 +100%、case 2 +89%）。**读侧必须 ptr-arith + masked load，双侧都不可用 make_block_ptr**：dim0 基址 = `nc*H_IN`，奇 H_IN（如 7）时基址奇 → load 间歇取错数据（C2d-ODD-DIM0-BASE-UNRELIABLE）；pad kernel 无 dot，ptr-arith 安全且零性能代价。**「消除 pad pass 改 mask 读原图」在本后端不可行**：dot 输入必须 make_block_ptr（C2d-SPLIT-RESULT-MATERIALIZE），W 边界行折叠 mask 无法用 block_ptr 表达；P2 depthwise mask 化则收益反转（pad 提速后只剩 3-11µs launch 地板，mask 使 conv 本体 +8~99%）。pad 降本唯一可行路线 = 提 pad kernel 自身带宽。

### §6.4 性能基准（最终态）

- **最终基准（7 路分派 + 2D tile pad，§6.2 全表 + T2d-15）**：50/50 verify pass，geomean **0.6677x** vs torch aclnnConvolution（实现平均延迟 0.0411ms / 框架平均 0.0274ms / 50 case 实现总延迟 2.8475ms；target 0.8x，**未达**）。复现本数字的代码 = 工作目录 `7_ConvStandard2d_generated.py`（7 路分派 P1 1×1-GEMM / P2 depthwise / P3 skinny-chlast / P4 S2-plane+fused / P5 tap-window / P6 plane-dot / P7 im2col+kblock-GEMM + 2D tile pad kernel）。**⚠️ 本数字是 7 路分派完整实现的产物。**
- **残余瓶颈（msprof 终局诊断实证，T2d-13）**：MTE 搬运主导 + 有效带宽已达 910B 峰值 ~78%，优化点 7/21/10/22 可行变体穷尽（9-tap 权重 UB 常驻 147KB+acc 超 192KB；tf32/bf16 被 MERE 1.221e-4 容限禁用，实测 tf32 相对误差 ~5e-4 超 4 倍）；小 shape 的 host 固定开销地板（40–60µs 无法摊薄）+ launch/发射开销主导几何平均值。非单点 kernel 效率问题，属该套件（小 shape 为主 vs 厂商闭源 aclnn）的结构性天花板。

---

## §7 常见陷阱与避免方法

> 仅列出 §1 G1–G13 之外、带具体失败症状的陷阱；通用陷阱（NPU 权重初始化、`%`、3D grid、负 offset 未补零、autotune symbol 重复传入、kernel 内嵌函数定义、`reshape/trans` 喂 dot）见对应 G 编号。

### §7.1 ConvStandard1d

| 陷阱 | 症状 | 避免方法 |
|------|------|---------|
| 对非连续输出 `out.view(C_out, N*L_out)` | 精度全错、数值巨大 | 用 T7 直写 `out[N,Cout,Lout]`；确需中间 buffer 先 `empty` 计算再 reshape/copy_ |
| group weight tap-k base 漏 `+k*Ci_pg` | 所有 k 读同一 slice，全错 | 打包布局 `[g,co,k,ci]` 下 base 必须含 per-k 偏移（T6） |
| autotune 引入错误配置 | 验证大量失败 | 小范围手动调参；autotune 后必须全量验证 |

### §7.2 ConvDepthwise2d

| 陷阱 | 症状 | 避免方法 |
|------|------|---------|
| 全场景用 5D block_ptr | stride=1 被迫小 tile，整体退步 | stride=1 保留 3D KHW-loop block_ptr |
| stride≥2 用 3D block_ptr + H/W stride trick | 仍有 KHW 次冗余 DRAM 读 | 升级到 5D block_ptr 单次 load 整个 rf |
| stride=1 走 NCHWc | tile 小，并行度下降 | 仅 stride≥2 用 NCHWc |
| k=5/k=7 UB-limited 不做 split | tile 被迫很小 | 按 §3.2 阈值启用 split-KHW + atomic_add |
| chunk 0 用 `tl.store` 替代 atomic_add | race 覆盖，err 2~3 | 全部 chunk atomic_add，输出先 zero（G11） |

### §7.3 ConvTranspose2d

| 陷阱 | 症状 | 避免方法 |
|------|------|---------|
| 权重未 flip | 等价转换错误 | `weight_prep_kernel` 内按 `KH-1-kh`/`KW-1-kw` 读 |
| pad 公式用 Conv2d 口径 | shape/精度错 | `pad = K - 1 - P` |
| weight bound 用 in_channels | fan_in 错 | `bound = 1/sqrt(out_channels * KH * KW)` |
| weight shape 维度搞反 | (in,out,KH,KW) 搞错 | ConvTranspose2d weight 为 `(in_channels, out_channels, KH, KW)` |
| conv kernel 直接写 NCHW 不做 UB transpose | 单 case 0.01ms→0.18ms | 先 UB 内 `tl.trans` accumulator，再 3D block_ptr 写 NCHW |
| 对 UB 中加载的 rf 张量切片 | `unsupported tensor index: slice` | 改用全局内存侧按 (kh,kw) 分别 block_ptr load |
| conv kernel 用 `@triton.autotune` 选 `BLOCK_C` | 运行间 geomean ±0.02~0.03、单 case ±0.27x，结论不可复现 | host 侧确定性启发式 `BLOCK_C = min(pow2≤C_in_pad, 64)`（C4d-DETERMINISTIC-CONV-CONFIG） |
| padded `out_workspace` + host `[:, :out_channels].contiguous()` 切片 | 全输出张量拷贝纯开销，大输出拖累延迟 | 直接分配精确 `out_channels` + store `boundary_check` 掩码（C4d-EXACT-OUTPUT） |
| 直接从原始 NCHW 做动态 clamp block_ptr load | aicore timeout / scalar 降级 | 先 prep kernel 生成 padded NHWC workspace 再读 |
| 返回 `out_workspace` 未切片 | 输出 shape 为 `(N, C_out_pad, H_out, W_out)`，verify 报 shape mismatch | 返回 `out_workspace[:, :out_channels, :, :].contiguous()` |
| 未缓存 device 转换后 weight | 0.60x→0.24x | `_weight_cache` 按 G9 键复用 |
| 未缓存 NHWC workspace | 0.60x→~0.45x，prep kernel 重复 alloc | `_workspace_cache` 按 shape/dtype/device 键复用 |
| 交错划分 `range(pid,total,NUM_CORES)` + ow_blk 非最内层解码 | 小/中 shape geomean ~0.78x，w_blk 跨 tile 无 L1 复用 | 连续块划分（start/cnt 形式）+ ow_blk 作最内层解码维（§4.3a，+6.6%） |
| 运行时 `if cond: tl.dot(...)`（如 2-oh hoist 的 `if valid1:`） | 全 case `MLIRCompilationError`（bishengir-compile 崩溃） | 运行时谓词不得包绕 dot；用 `mask`/`boundary_check` 处理边界，或编译期 constexpr 分支（G13） |

### §7.4 ConvStandard3d

| 陷阱 | 症状 | 避免方法 |
|------|------|---------|
| kernel_size 维度顺序错 | 输出 shape 错 | 元组顺序 (D, H, W) |
| `for c1_inner in range(...)` | 非 static_range 结果错 | 用 `tl.static_range`（T7） |
| weight c1_inner stride 写错 | C1_pg=1 掩盖，>1 全失败 | stride=`Kd*Kh*C0`（T7） |
| block_ptr order 与 strides 不匹配 | ow=0 偶然正确，ow>0 全错 | order 按 stride 从小到大排列 |
| Cout_pg<16 降 BLOCK_COUT 到 4/8 | Cube 16×16 下限，收益噪声 | 改 groups-batched M=16（T9） |
| `tl.arange(GROUPS_PER_M*COUT_PG)` 非 constexpr 报错 | 编译错 | 显式 `M_FLAT` 作为 constexpr kernel 参数（T9） |
| NC1DHW0 stride>1 尝试 no-permute | gather 回退 -26%~-50% | stride>1 必须 permute 到 C0 内层（T14） |
| block_ptr store 跨 group 溢出 | store 用绝对地址 `shape=(C_out, HW_out)` 只截总 C_out 不截 group 边界，`Cout_pg<BLOCK_COUT` 时写零污染下一组 | constexpr `FULL_COUT=(BLOCK_COUT==Cout_pg)` 分支：满 tile 才 block store，否则 masked scalar store；不加 `tl.where` 钳位（T20） |
| im2col kernel 用 `n * NCDHW` 当 batch 偏移 | batch>1 后续 batch 全零/越界 | 用 `n * CDHW`（§5.1） |
| HW-flat 无门控全量 dispatch | K≥7 或 C 超限时 timeout / impl 慢 10x | 仅 `C_in_pg≤16、C_out_pg≤64、K≤5` stride==1 启用（T20） |
| per-kd 合并 dot（reshape 3D load 扩大 K-dim） | dot 消费 reshape/trans 数值全错 | 保持 per-(kd,kh) K-dim=C0 小 dot（G13） |
| block_ptr `order=(0,1)`（最内轴非 stride=1） | **静默错值**：impl 全零 / 常数（不报错，36 case 全 fail） | Ascend 强约束：block_ptr 永远 `order=(1,0)`，布局让最后一维 stride=1 连续。改 pack 布局使最内轴 stride=1：P3 no-permute `[N,g,C1,C0,Dp,Hp,Wp]` 使 W stride=1、flat HW 轴连续（T20）；旧 C0-HW 交错 `[N,g,C1,Dp,C0,HWpad]` 同理使 HW 展平轴 stride=1 |
| gather+static_range kernel 的 `HAVE_BIAS=False` 特化 | **挂死 aicore**（507014 timeout / 507057 SUSPECT REMOTE ERROR）；同参数 bias=True 正常 | 该 constexpr 取值组合触发编译 bug；bias=None 时建零向量 `torch.zeros(Cout,...)` + 强制 `HAVE_BIAS=True` 绕过（零偏置不影响精度）。判据：两 case 仅一 constexpr 不同、一个挂死一个正常 → 嫌疑特化 bug |
| 隔离诊断 harness 忘 `x.to('npu')` | NPU kernel 读 CPU 内存报 507057（伪故障，误判 kernel 死） | `get_input_groups()` 返回的 tensor 常在 CPU；自写 harness 必须 `x.to('npu:0')` + `torch.npu.set_device`；verify.py 内部已搬 NPU 故其结果有效 |
| **P2 `xm` block_ptr order 与 strides 不匹配（packed block_ptr）** | 全 case 静默错值（ow=0 可能偶然对，ow>0 错） | `xm` 内存为 `[..., Wp, C0]`，读 `shape=(C0,Wp)` 时须 `strides=(1,C0), order=(0,1)`；若误用 `(C0,1), order=(1,0)` 则 W 维 stride 实际为 C0 而非 1，导致跨 W 错位（§5.3 T9 / §5.5 历史里程碑 opt_iter_7） |
| **P2 M-block group 偏移未局部化** | groups>GROUPS_PER_M（case19/44 g=8）时大块数值错误，g≤GROUPS_PER_M 时可能碰巧对 | `repack_weight_m_kernel` 必须用 `g_in_block = g - (g//GROUPS_PER_M)*GROUPS_PER_M` 计算 c0 偏移，与 `pack_input_m_kernel` 每 M-block 重置对齐（§5.3 T9） |
| **缓存分支内定义布局常量** | verify 单轮通过，benchmark 多轮调用报 `UnboundLocalError`（如 `WM_M_STRIDE`） | 由 tensor 布局决定的 strides（`WM_M_STRIDE`、`WM_KK_STRIDE` 等）必须在 `if cache is None:` 分支之前赋值，保证 cache 命中分支也能访问（§5.3 T9） |
| **P3 kernel 形参含 `stride` 但 `_launch_hwflat` 不传** | `NameError: name 'stride' is not defined`（34 case 全 fail，仅 P3 路径） | P3 门控恒 stride==1，host 显式传 `1`。旧 hwflat kernel 隐式假设 stride==1 不接 stride 参数；迁移到带 `stride: tl.constexpr` 形参的 kernel 时易漏（T20，opt_iter_26 首轮踩坑） |
| **P3 no-permute 布局但改用分离 W 分块** | 0.4632x 回退（`BLOCK_W≤64`、tile 数 `Hout×` 增加、丢失大块连续 Cube load） | no-permute 必须**叠加 flat-HW dot**（保留大 `BLOCK_HW≤256`），二者缺一不可。仅换布局不换 tile 策略会退化（opt_iter_25 已证伪，见 §8.1） |
| **P3 packer gather 直读（旧 C0-HW 布局）** | pack pass 偏慢，整体 0.6745x 未达标 | 改连续 4D block-copy packer（`_pack_x_nc1dhw0_grouped` + `pad_copy_4d_kernel`，T21 实例），pack 纯连续拷贝 → 1.1332x 达标 |

### §7.5 ConvStandard2d

| 陷阱 | 症状 | 避免方法 |
|------|------|---------|
| dot 路径 `BLOCK_M > Cout_pg`（Cube 下限 16 补齐）未按组裁剪 | 越组 co 行用本组输入计算并写穿到下一组输出通道 | store/bias mask 加 `(pid_co*BLOCK_M + arange) < Cout_pg`（C2d-BM-GROUP-CLIP） |
| 模块级权重/布局缓存跨 ModelNew 实例复用 | **序列相关** AccuracyError（部分 case 依赖运行顺序才失败） | 缓存放实例内字典；若必须模块级，`ModelNew.__init__` 清空（C2d-WEIGHT-CACHE-INSTANCE-RESET） |
| host 侧用 `min()/max()` 裁 grid / 计算块数 | `validate_triton_impl.py` 判 Type3 退化（PyTorchFallback） | 条件表达式实现 `_clip`（C2d-NO-MIN-MAX） |
| stride>1 输入用 computed-offset gather 直喂 `tl.dot` | Cube mis-lower，精度错 | 三选一：物化 im2col / NCHWc C0 内层 pack / gather 进 UB tile 再单 dot（C2d-DOT-OPERAND-CONTIG） |
| 小通道（每组输入通道 ≤6）仍走逐窗口 tap dot 路径 | K² 次 K=16 小 dot（仅少数行有效），矩阵乘发射开销主导 | 门控切 skinny channel-last 单 dot 路径（T2d-7） |
| im2col 规约顺序与 weight 展平不一致 | GEMM 数值全错（看似精度问题） | 统一规约序 =（kh×K+kw）×每组输入通道 + 通道序号（tap 外层、通道内层），weight 展平同序 |
| stride≥2 向量化输入 load（masked/unmasked、块指针/计算指针任一写法） | aicore 507015 崩溃或数值错 | host 奇偶平面拆分，平面内 stride-1 仿射（T2d-2，C2d-S2-VECTOR-LOAD-FORBIDDEN） |
| 钳位（min/max）进入 load 地址 | 仿射性破坏 → load 标量化，10–70x 回退 | host 预零填充消除边界（T2d-1）；kernel 内仅用「钳位保地址 + mask 置零」防崩溃（C2d-NO-CLAMP-IN-LOAD-ADDR） |
| host 矩形切片拷贝 / stride 视图拷贝 / 通道重排 | 落 AiCpu 慢速路径，单次 92–176µs，占端到端 70%+ | 行连续 load/store 专用拷贝核，多段预处理融合单核；仅病态形状启用（T2d-5，C2d-HOST-COPY-PATHOLOGY） |
| 运行期标量 if 双 load 快路径 | 误编译，dot 输出全零 | 删除条件双路径；选择逻辑折成「序号 × 步长」仿射偏移加单基址（C2d-NO-CONDITIONAL-BASE-POINTER） |
| 编译期 if/else 切换双基址指针 | 误编译，误差 ~125 量级 | 同上：单基址 + 仿射平面偏移 |
| 多段式「逐 tap 写入统一缓冲区组装」链嵌在 runtime 循环内 | MLIR 编译错误（cbuf to cbuf） | 退化为逐平面独立 dot；或 host 侧重排使组装坐标仿射（T2d-7） |
| 行块化/大 tile 后 tile 总数 < 核数 | 并行度塌陷（实测 0.83x 回退） | tile 总数 ≥ 核数量级校验，不足回退小块（C2d-PARALLELISM-COLLAPSE-GUARD） |
| 小算力 case 走带 host 预处理路径 | 固定开销 40–60µs 无法摊薄，0.67–0.75x | forward 按乘加量门控回退无预处理路径（C2d-MACS-GATE-HOST-OVERHEAD） |
| im2col+GEMM 用于 kc<2000 或 GEMM 规模小的 case | 物化开销无法回收，0.41–0.79x | 双门控 kc≥2000 且 kc×输出元素≥20万（T2d-9） |
| ptr-arith / masked load / `tl.where` 物化 / clamp 地址直接喂 `tl.dot` | 编译期 SIGSEGV（ConvertLinalgRToBinary） | dot 输入唯一合法形态 = make_block_ptr（含经 `tl.where` 物化的稳定变体，C2d-SPLIT-RESULT-MATERIALIZE） |
| make_block_ptr 读 `nc*H_IN` 为奇基址的输入（pad/拷贝 kernel） | 间歇取错数据（H_IN 奇如 7 的 case） | 无 dot 的 kernel 一律 ptr-arith + masked load（零性能代价，C2d-ODD-DIM0-BASE-UNRELIABLE） |
| pad/拷贝 kernel 按行（1KB tile）搬运 | 带宽仅 48GB/s（HBM 峰值 12%），pad 可占 pad+conv 的 50%+ | 2D 平面 tile ~16KB（`BLOCK_W=pow2_ceil(W,64,1024)`、`BLOCK_R=4096//BLOCK_W`）→ 353GB/s（T2d-15） |
| sketch.txt 仅规划 P1/P2/P5 三条路径，其余 case 走通用 fallback | Phase 3 3.2b 架构检查报 `A-SketchDeviation`；若强行绕过，非 1×1 case 全走 im2col，geomean 跌至 0.025x | sketch 必须一次性列出 §6.2 全部 P1–P7 路径、kernel 列表、门控与数据布局（C2d-SKETCH-MANDATORY-PATHS） |
| P2 depthwise host 预补零后 kernel 内仍 `-PAD_TOP/-PAD_LEFT` | `padding>0` 的 depthwise/grouped-like case 全部 AccuracyError（impl 输出接近 0，实测 10/50 fail） | host 补零后 kernel 内坐标直接用 `ih0=oh0*STRIDE+kh*DILATION`、`iw0=ow0*STRIDE+kw*DILATION`（C2d-DEPTHWISE-PAD-OFFSET） |
| 无门控地把所有非 1×1 case fallback 到 im2col+GEMM | `im2col_kernel` 标量 gather 占 95%+，geomean 仅 0.025x | P7 im2col 仅在 `kc≥2000` 且 `kc×输出元素≥20万` 时启用，其余走 P3/P4/P5/P6（C2d-IM2COL-GATE） |

---

## §8 已验证无效方向 / 未验证方向

### §8.1 已验证无效 / 需避免方向

**ConvStandard3d（ascend910b1，60 cases，kernel_size 实际生效为 (k,k,1)）**：
1. 4D block_ptr depthwise fast path：在该 60-case 集合上 geomean 从 0.2976x 跌至 ~0.2745x，必须回退 scalar direct-conv fallback。
2. `CHANNELS_PER_BLOCK = groups_per_m * C_in_pg` 作为 `tl.constexpr`：非 16 的 channel block 被编译器 scalarize，整体下降 ~10–15%。
3. 受控 im2col + GEMM（groups==1，S>1/Dil>1，K_dim≥64，Co≥32）：geomean 从 0.3109x 崩塌至 0.2310x，必须关闭。
4. no-permute 对 stride==1 中小 C case 有稳定增益，但无法弥补整体与 aclnn 的差距。
5. per-kd 合并 dot（reshape 3D load 扩大 K-dim）：dot 消费 reshape/trans 数值全错。
6. 单 dot 加载完整 `K_dim > 128`：触发 `MLIRCompilationError`。
7. 专用 1×1 纯 matmul：K 极小，launch/overhead 主导，geomean 下降。
8. 两阶段路径加大 `BLOCK_KK=64` 或 `BLOCK_HW=128`：UB overflow。
9. **stride==1 no-permute 路径的 host copy 融合**（`torch.zeros + copy_` → device `pad_input_kernel`）：2026-08-05 验证，no-permute case latency 下降 35–60%，但整体 geomean 从 0.7789× 微跌至 0.7694×，未成为有效全局优化方向。
10. **P4 `BLOCK_COUT=256` for Cout_pg>128**：单 case latency 可能微降，但 60-case geomean 从 0.6745x 降至 0.6366x，整体回退。
11. **P3 `BLOCK_HW=512`**：aicore timeout / 编译极慢，必须保持上限 256。
12. **§4.3a 连续块划分 + 最内层 tile 维度复用 `w_blk`**：当前 tile 已内嵌完整 `ow`/`HW` 循环，w_blk 已复用；改为连续块划分后 geomean 从 0.6745x 降至 0.6299x，不适用。
13. **收紧 P2 门控**（如要求 `groups*Cin_pg==16` 或排除更多 depthwise）：会把 case 20/45 等挤到 P3，整体 geomean 下降。
14. **P4 K=7 / P5 stride=2 K=7 路由到更大 BLOCK 或回退 P5**：case 48 等 bottleneck 无改善，整体 regress。
15. **opt_iter_24：depthwise s1d1 专用 scalar kernel（无 tl.dot）**：把 `stride==1 && dilation==1` 的 depthwise case（`Cin_pg==Cout_pg==1`）从 P2 mgrouped 改走专用 scalar depthwise kernel。geomean 0.6745x→**0.5610x** 回退。原因：depthwise 无跨通道规约，scalar kernel 无法饱满 Cube；P2 mgrouped（block-diagonal weight + Cube dot）对 depthwise 更优。结论：**depthwise 留在 P2，不要单开 scalar 路径**。
16. **opt_iter_25：P3 no-permute 布局 + 分离 W 分块**：把 P3 改 no-permute 布局但 compute 用 per-`(od,oh)` tile + 内层 `ow` 循环（`BLOCK_W≤64`），而非 flat-HW dot。geomean 0.6745x→**0.4632x** 回退。原因：丢失 flat-HW 的大 `BLOCK_HW` 连续 Cube load，且 tile 数 `Hout×` 增加。结论：**no-permute 布局必须叠加 flat-HW dot**（opt_iter_26 正解，见 §5.3 T20），二者缺一不可。
17. **（有效方向，勿与 4/16 混淆）P3 no-permute 布局 + flat-HW dot（opt_iter_26）**：no-permute 物理布局 `[N,g,C1,C0,Dp,Hp,Wp]`（连续 4D block-copy packer）**叠加** flat-HW dot（大 `BLOCK_HW≤256`），geomean 0.6745x→**1.1332x 达标**。与失效方向 4（P4 standalone no-permute ~0.33x）、16（no-permute+分离 W）的区别在于：本方案把 no-permute 当**物理布局**用，compute 仍走 flat-HW 大块 dot。
18. **旧 C0-HW 交错布局 P3（`[N,g,C1,Dp,C0,HWpad]` + `pack_input_hwflat_kernel`/`conv3d_hwflat_kernel`）**：0.6745x，**已被 opt_iter_26 取代**（pack 是 gather 非连续、compute 未利用 no-permute 的连续拷贝优势）。除非复现历史，新实现请用 §5.3 T20 方案。

**ConvTranspose2d**：
- RF block load + UB 内切片：不支持对加载后局部张量切片，编译期失败。
- K=4/5 stride=1 直接 NCHW scalar-load：触发 aicore timeout 或严重 scalar 降级。
- device-side im2col + 单大 GEMM：额外 global-memory round-trip，geomean 降至 ~0.36x。
- Winograd F(2×2, 3×3)：实现复杂、kernel 输出 NaN、触发 NPU 设备级异常（507057），风险不可控。
- 单 kernel fuse prep+conv：UB 放不下常驻累加器，性能显著退步。
- 对 K=4/5 继续调 BLOCK 参数：瓶颈在 KH×KW 循环调用次数与规约维对齐，调参收益已耗尽。
- 2-oh w_blk hoist（每 program 处理 2 行 oh，用 `if valid1:` 包第二个 `tl.dot` 复用 w_blk）：全 case `MLIRCompilationError`（运行时谓词包 dot，见 G13）。w_blk 跨行复用方向本身合理，但禁止运行时谓词包 dot；若要复用须改编译期 constexpr 分支 + 无条件双累加器。

**ConvStandard1d**（最终 geomean **0.8382x**，达标 0.8x；约束 C1d-NO-LAYOUT-OPS 内已无有效方向，有效技巧见 §2.3 T5–T9）：
1. **大 GEMM tile 调优**（opt_iter_15，放宽 acc 上限到 8192，dot 调用 −75%）：idx25 分量 profile 显示 im2col gather 占 impl 97%、GEMM <3%，tile 改了仅 −2%，geomean 持平。
2. **`make_block_ptr(strides=(STRIDE,))` strided load 加速 gather**：与 pointer-tensor `tl.load(ptr+arange*STRIDE)` 同速（0.98x，maxerr=0），Ascend 不把 1D strided block_ptr lower 成更快 DMA。
3. **im2col 按 K 维 2D tile load `[K_PAD,BLOCK_L]`**（想降 load 指令数 N·C·K→N·C）：A/B 实测 case25/36/6/42 慢 2-3×，BL=256 时 UB 溢出（multi-buffer 需 232KB>192KB）；Ascend 把 2D strided 计算指针 load 分解成逐元素，有效指令数不降反升。
4. **Path A（shifted GEMM）tile 调优**：case24 全 tile ±5%，且大 GEMM(case24) 与 tiny(case2) 核时间相近 → launch/计算地板 bound，非 GEMM 效率 bound。
5. **groups>1 / stride>1 直接 Vector element-wise conv（无 im2col）**：per-output gather `Cin_pg·K` 是 im2col 的 ~160×（case36），gather 已是地板，省下的 workspace 往返无法抵消。
- **结论**：stride>1 gather 是 Ascend 指令吞吐地板（case 25 ~1.1ms 占 impl 97%）；任何形式 strided 2D 访存（block_ptr strided / 显式 2D 计算指针）都被分解为逐元素，**约束内唯一有效手段是 1D BLOCK_L 放大（T9）**。唯一能进一步破地板的是放宽约束回 host `unfold/permute`（约束前 1.06x），需用户显式同意。

**ConvStandard2d**（50-case 套件实测，最终 0.6677x）：
1. **stride-2 向量化 load（所有写法）**：masked/unmasked、块指针/计算指针均触发 aicore 507015 崩溃或误编译——唯一出路是奇偶平面拆分（T2d-2）。
2. **钳位进 load 地址**：仿射性破坏 → 标量化 10–70x 回退；「钳位保地址 + mask 置零」只可作 S=2 防崩溃手段，不可当性能手段。
3. **运行期标量 if 双 load 快路径（内部 tile 无 mask）**：误编译，dot 输出全零。
4. **编译期 if/else 双基址指针**：误编译，误差 ~125 量级。
5. **多段式统一缓冲区组装链嵌 runtime 循环**：MLIR cbuf-to-cbuf 编译错误；多平面组装退化为逐平面独立 dot。
6. **stride-2 gather（expand_shape 视图方案）**：MLIR expand_shape 错误，方向废弃。
7. **行×宽×规约 3D 单 load 的 skinny 变体**：编译失败（at load）。
8. **im2col+GEMM 用于小 kc（<2000）**：物化开销无法回收，0.41–0.79x 劣化；kc 达标但 kc×输出元素 <20万（GEMM 规模小）同样劣化（0.60x）——双门控缺一不可（T2d-9）。
9. **拷贝核盲目替换一切 host 拷贝**：普通形状 host 拷贝 38–60µs 已快于拷贝核 ~41µs 地板，替换反而回退（0.16x case 实测）——仅「通道≥128 且输入宽≤32」病态形状启用（T2d-5）。
10. **skinny 行块 4 用于 K=7**：4 份累加器 ~64KB 寄存器压力，无收益（231≈228µs），K=7 保留行块 2。
11. **P4 fused 去掉 split→where 物化链（平面仿射偏移）**：奇偶平面拆成独立 `[N*C,H_alloc,W2]` buffer、两个半宽 `(BLOCK_CI,BLOCK_W)` block_ptr 直读喂 dot（err=0 全对，含 P 奇偶平面交换修正）。A/B 交替 3 轮取 min：row41 (K=5,S=2,P=2) **-21.3%**、row2 (K=3,S=2,P=1) **-19.1%**。原因：split/where 是廉价 Vector 物化，pair 宽载 `(BLOCK_CI, 2·BLOCK_W)` 的 MTE 连续性才是收益来源（与 simulator MTE 53-69% 诊断一致）；拆两个半宽 load 使 MTE 事务数翻倍。**结论：C2d-SPLIT-RESULT-MATERIALIZE 的 where 物化链保留，勿再尝试平面化消除**。
12. **P5/P6 门控与 tile 上限的盲目调参**：P6 Wo 门槛再放宽、P5 BCO 上限 64→128 均在 ±0.1% 噪声带内无效。**方法论要点：benchmark 单 case run-to-run 方差实测 ±4%**（与 base 完全同路径的 case 也会显示 -3.6%~-4.1% "差异"，表观 +18.7% 复跑归零）——任何 <4% 的单 case 改进在 50-case 套件不可分辨，勿据单轮 profiler 差异下结论，须同路径对照或复跑。根因：S=1 中小 shape case 的 impl 时延被固定开销主导（xpad 每次重算 + weight pack + 多 launch），P5/P6 配置调优已触地板。
13. **行块化后不校验 tile 总数**：tile 数 8 < 24 核时并行度塌陷，0.83x 回退（C2d-PARALLELISM-COLLAPSE-GUARD）。
14. **小算力 case 走带 host 预处理路径**：固定开销无法摊薄，0.67–0.75x；单 case 实测 0.342→0.292（C2d-MACS-GATE-HOST-OVERHEAD）。
15. **load 传 care_padding=False**：API 不存在，试误回退。
16. **移位 1×1 分解（K² 个「移位的纯通道规约」小 GEMM）作为 stride==1 主力**：并非报错，而是被证明次优——小 dot 发射开销主导，已被 skinny channel-last 单 dot（T2d-7）+ 零填充逐平面无 mask 大 dot（T2d-1）取代。
17. **NCHWc（通道 C0=16 内包）重排用于 stride>1**：其输入 load 涉及 stride-2 向量化，触发误编译（见第 1 条）；已被 W 奇偶平面拆分（T2d-2）取代。
18. **「conv kernel mask 读原图消除独立 pad pass」（两条子路线均证伪）**：(a) mask/ptr-arith/`tl.where` 物化/clamp 地址任何一种喂 `tl.dot` 都在编译期 SIGSEGV（ConvertLinalgRToBinary，C2d-SPLIT-RESULT-MATERIALIZE 强化），而 W 边界行折叠 mask 无法用 make_block_ptr 表达 → dot 路径（P4/P5/P6/P7）消除 pad 不可实现；(b) P2 depthwise mask 化可行但收益反转——pad 2D tile 化（T2d-15）后 pad 只剩 3-11µs launch 地板，mask 使 conv 本体 +8~99%（case 44 净亏 44%）。**pad 降本唯一可行路线 = pad kernel 自身提带宽（T2d-15）**。
19. **P4/P6/P7 的 xpad block_ptr 间歇分配地址敏感 bug（平台既有）**：同一代码不同进程 verify 偶发单 case AccuracyError（max_abs_diff 2~7.6 量级、违例数恰为 2 的幂如 1024/16384、连续区坏），重跑即过——xpad buffer 分配地址变化触发 block_ptr 间歇取错。**排查方法**：单模块确定性 stress（同进程多 trial vs F.conv2d 独立参考），err 稳定在 fp32 噪声级（3-7e-4）即证明无真回归。**⚠️ A/B 双模块对比法是伪影来源**：加载两份模块副本本身就改变 buffer 分配地址，base vs base（同文件双副本）也会出现 err 2.3-6.3——精度判定必须以 verify.py 单模块为权威。