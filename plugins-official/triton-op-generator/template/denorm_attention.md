# 去归一化 attention（denorm / silu-attention）模板卡片

> **适用范围**：无 softmax 的 attention 变体——逐元素激活（silu/tanh/gelu 等）替代归一化，
> 保留 QK dot + PV dot 骨架；jagged 布局 `[sum_B, num_heads, head_dim]`；
> 常见 `1/max_seq_len` 全局缩放与因果下三角掩码。代表算子：HSTU attention（Meta）。
> 实证数据来自 `hstu_attention_fwd/bwd`（Ascend950PR，op_0/op_1，2026-09-18~20，20 case 全流程）
> 与 Q3TritonKernel 手写版对照实验（2026-09-20）。

## Layer 1（硬约束，Phase 2 Step 3 合规门管辖）

| # | 约束 | 依据 |
|---|---|---|
| D1 | **掩码方向以 golden 为准**：golden 用 `torch.tril`（`n <= m` 有效）；GPU 参考内核可能用 triu 表达（`offs_m_minus_n > 0` 判无效）。两者数学对称但**禁止照抄 GPU 核的掩码方向** | fwd 实测 golden 与 GPU 核相反 |
| D2 | **标量序锁定与 golden 逐位同序**：`qk*alpha → silu → /max_seq_len → 掩码`。任意交换（如先除后激活）产生 ulp 级偏差 | fwd sketch N2 |
| D3 | **无归一化 ⇒ 无 online softmax 状态量**（m/l），KV 块间无跨块数据依赖，acc 纯累加。softmax 家族的 F1 幽灵列 / F2 `-inf` 替代**不适用**——掩码就是精确置 0，padding 列 silu(0)=0 自零（显式掩码仍保留兜底） | 与 FA 卡的差异点 |
| D4 | **jagged 偏移设备侧标量加载**（offsets 仅两个标量 load），禁止 host 读 offsets 回传构造稠密索引或 pack/unpack | L1.6 精神 + fwd N4 |
| D5 | `BLOCK_D = ceil16(D)`，**非 next_pow2**，禁止 BLOCK_D 进入 autotune 搜索空间 | F3 |
| D6 | UB 面积预算必须是 host 侧**与输入无关的常数公式**（950PR 248KB/AIV 预留余量 200KB；fp32 元素 ≤51200，fp16 按字节预算）。该公式是**保证可行的保守界，不是真实上界**（忽略编译器 buffer 复用，系统性高估占用）；tile 超出公式时必须以探针实证背书（编译通过 + 边界 case verify 通过），见 Layer 2 第 5 条 | F4 + iter_6 实测 |
| D7 | 两处 dot 均**原生 dtype 操作数 + fp32 累加器**，操作数同 dtype，禁止逐维外积 | L1.1/L1.2 |
| D8 | `grid = min(NUM_CORES, M_TOT)` + 核内步长循环领任务；NUM_CORES 必须 constexpr；task 解码用 constexpr 除/模改写（`a - (a//b)*b`），无运行期索引赋值 | L1.9/L1.10 |

## Layer 2（强推荐，Phase 3/4 起效）

1. **`/max_seq_len` 除法后移出 KV 循环**（dot 线性性，消除逐块冗余除法）——fwd iter_3 实测有效。
2. **因果 tril ⇒ KV 扫描区间收缩**到 `[0, min((m_blk+1)*BLOCK_M, L_b))`，对角块之上不扫。
3. **jagged 尾部空任务早退**：`if m_blk * BLOCK_M >= L_b: continue`（L_b ∈ [N/2, N]）。
4. **性能路线（对齐 GPU TF32 精度级，hf32 免 cast，iter_7 交付版）**：fp32 张量直进 kernel（不 cast）+ 两处 dot `input_precision="hf32"` + fp32 累加；`w` 经 silu（fp32 精确路径）后直进 PV dot（to(fp32) 为 no-op）。全量认证（2026-09-23，iter_7_hf32）：verify 20/20 全为 AccuracyError、MERE 带 5.27e-4~6.85e-4 与 fp16 路线**逐位一致**（同 10-bit 尾数档）；benchmark 定版 geomean **0.9345**（route bench 同窗 0.9100；≥0.92 目标首次达成）。**fp16+cast（iter_6 旧交付版 0.8420）降为 fallback**：hf32 dot 速率不足的计算重载 shape 才选——依据与实测见 Layer 3 注 2。
5. **分档 tiling**：两档均可用 BM=128/BN=128（D<128 档受元素预算 51200 约束）。⚠️ D≥128 档早期被 fp16 字节预算公式（6·BM·D + 4·D·BN + 6·BM·BN ≤ 204800B）封顶在 BM=64，**该封顶已证伪**（iter_6，vs 手写对照）：公式按所有中间张量最坏叠加估算、忽略编译器 buffer 复用，系统性高估占用——放开 BM=128 后 D=128 五 shape **+7.8~12.8%**（同窗交错 A/B，3 轮中位），jagged 尾部/seq=129 跨块/非 2 幂边界精度全过。**tile 上界标定方法**：公式值是保证可行的保守起点（可行下界），真实上界以探针为准——放大 tile 后须编译通过 + 边界 case verify 通过，二者缺一即回退公式值；**禁止把公式当硬上限封顶，也禁止无探针证据直接超公式取 tile**。
6. **grid 收缩到 cube 核数**（950PR=28）：mix 算子 grid > 28 劣化；运行时查询 `num_aicore` 后固化为 constexpr，查询前需先初始化 NPU（空张量上栈）。

## Layer 3（实测数据与硬结论，Ascend950PR，10 大 shape geomean speedup_vs_gpu）

> 本表为 2026-09-22 **六路线同窗同骨架复测**：全部基于 iter_6 终态骨架构造变体（只改 dot 精度路线与 host cast，tile/调度一致），10 shape 同进程交错 3 轮取中位；每变体先过正确性抽检——ieee 与 fp32 参考**零误差**、hf32/fp16 同为 3.1e-4、bf16 2.5e-3，档位符合预期。

| dot 路线 | speedup_vs_gpu | fp32 MERE 门（<1.22e-4） |
|---|---|---|
| ieee fp32 全链路 | 0.1856 | **20/20 过（精度合规回退版）** |
| **hf32（`input_precision="hf32"`，免 cast，iter_7 交付版）** | **0.9100** | 0/20（MERE 5.27e-4~6.85e-4，与 fp16 路线逐位一致，终态骨架抽检 3.1e-4 同档）；全量认证后 benchmark 定版 geomean 0.9345（2026-09-23 定版窗口，20/20 跑通） |
| bf16 host 预转换 | 0.8053 | 0/20（MERE ~4e-3） |
| fp16 分档 + host 预转换（iter_5，D≥128 档 BM=64） | 0.8014 | 0/20（MERE 5.3e-4~6.9e-4） |
| fp16 分档 + BM=128（iter_6，降为 fallback） | 0.8420 | 边界 case 全过（jagged 尾部/seq=129 跨块/非 2 幂）；全量 20 case 门未复测，量级参照 iter_5（0/20） |
| fp16 免 cast（诊断上限，不可交付） | 1.0785 | — |

注：
1. **本表取代生成期路线数字**（ieee 0.1945 / hf32 0.6137 / bf16 0.8997 / iter_5 0.7997 / iter_6 0.8000 / 免 cast 1.0373）——历史数字测于**不同迭代骨架 + 不同 bench 窗口**，同表混排会误导路线裁决，仅在生成过程报告中留档。
2. **受控复测修正了两处路线排序**：① hf32 实为 **0.9100**（历史 0.6137 测于 ieee 时代面积预算 tiling——D≥128 档被封在 BM=32，终态为 128；差距主项是 tile 不是路线）——免 cast 税 + dot 速率仅次于 fp16，**反超 fp16+cast 交付版（0.8420）约 +8%，且精度档位相同（10-bit 尾数）**；② bf16 与 fp16 同速（0.8053 vs 0.8014，历史 0.8997 的领先同为 tiling/骨架混淆）。hf32 免 cast 路线已于 2026-09-23 **全量认证并切换为交付版**（verify 20/20 全为 AccuracyError、MERE 带与 fp16 路线逐位一致；benchmark 定版 geomean 0.9345，≥0.92 目标首次达成）——后续同类算子应优先评估；fp16+cast 降为 fallback，仅当 hf32 dot 速率不足（计算重载 shape）时选用。
3. BM=128 增益：同表内 fp16 BM64→BM128 = +5.1%；四次同窗 A/B（跨日）geomean +4.3~5.1%、D=128 五 shape +7.4~13.7%、D=64 档 ±0.5% 零回归；跨窗口 geomean 绝对值漂移（base 0.76→0.81），**同窗交错 A/B 是唯一可信口径**。
4. cast 架构税今日实测 **+28%**（0.8420→1.0785）；hf32 免 cast 的本质是把"cast 一读一写"换成"K/V 直接 fp32 读"，总流量反而更低（2x vs cast 路线 3x）。

**核心硬结论**：
- fp32 级 MERE 门与 ≥0.9 级 speedup_vs_gpu 在本硬件**数学互斥**：唯一过精度门的 ieee fp32 仅 0.1856；hf32 虽近 0.9（0.9100）但 MERE 5.27e-4 同样过不了 fp32 门。**GPU 参考内核本身 TF32（10-bit 尾数），同样过不了 fp32 门**——精度口径对齐 GPU 时该门对两者一视同仁不可过。
- **cast 架构税 ~24-28%**：fp32→fp16 预转换是 Cube 只吃 16-bit 操作数的架构税（GPU TF32 直接消费 fp32 无此开销）。cast 优化全谱系已证伪：融合单 launch（带宽瓶颈非 launch 瓶颈）、K/V 预转换+Q 核内 cast、in-kernel cast（K/V tile 每 query block 重复 cast，大幅劣化至 0.42）、尺寸感知分档 cast——均无稳定增益。**根治路径是不 cast：hf32 免 cast 路线，已全量认证并切换为交付版（route bench 同窗 0.9100 / benchmark 定版 geomean 0.9345，2026-09-23）**。
- **路线裁决必须在终态骨架上同窗受控复测**：生成期逐迭代换骨架测出的路线数字不可横向比较（本次 hf32/bf16 排序修正即由此发现）。**主要混淆源是换路线不重标 tiling**——hf32 的 0.6137 直接继承了 ieee 时代面积预算封顶的 BM=32，从未被重新标定。
- 其他证伪方向：**算力（mac）瓶颈时** BM 加倍（每迭代成本增速超过迭代数减少）、grid > 28 核、任务 (b,h) 重排（负载不均）、num_stages 软件流水、K 转置加载。⚠️ 分界修正（iter_6）：**load 供数瓶颈（aic_mte2_ratio 高，如本卡 fwd 大 D 档）时 BM 加倍有效**（+7.4~13.7%）——动 tile 前先看 kernel_details 的 mac/mte2 ratio 定瓶颈类型。

## 陷阱表（fwd 实测踩坑，bwd/后续同类算子必读）

| 陷阱 | 正确做法 |
|---|---|
| Triton driver 查询设备属性在 NPU 初始化前失败 | 先 `_ = torch.zeros(1, device=dev)` 再查询 |
| kernel 内访问非 constexpr 全局变量报错 | 常量定义为 `tl.constexpr`（如 `LOG2E`） |
| autograd 类标杆（bwd）forward 含 `.backward()` | golden 为 autograd 反传，多输出 `(out, dq, dk, dv)` 逐项比对 |

## 反向（bwd）扩展要点

golden 为 autograd 反传（`out.backward(dout)` → dq/dk/dv）。数学分解（w = silu(alpha·qk)/max_seq_len·M）：
- `dv = wᵀ @ dout`；`dw = dout @ vᵀ`
- `ds = dw ∘ silu'(s) / max_seq_len · M`，其中 `silu'(s) = sig·(1 + s·(1−sig))`
- `dq = ds @ k · alpha`；`dk = dsᵀ @ q · alpha`

GPU 参考（`_hstu_attn_bwd`）：KV 列块并行（每 program 一列 BLOCK_N），核内重算 `qk_trans → silu`，4 个 dot；dq 经转置布局 + 自旋锁读改写累加（SEQUENCE_PARALLEL 时多 program 分 n 块）。NPU 适配注意：自旋锁 `atomic_cas` 在 Ascend Triton 上不可用/高风险，dq 累加需改为**独立 dq kernel 或单 program 串行 n 块**（无 SEQUENCE_PARALLEL）结构；标量序与掩码方向仍按 D1/D2 以 golden 为准。

### bwd 双 kernel 是结构性必然（fwd/bwd 全流程 + Q3 手写对照实证，2026-09）

- **根因：梯度归约方向对偶**。`dq_i = Σ_{j≤i} dS_ij·K_j`（行向，沿因果前缀）与
  `dk_j = Σ_{i≥j} dS_ij·Q_i`、`dv_j = Σ_{i≥j} W_ij·dOut_i`（列向，沿后缀）不可在同一种
  循环序内同时寄存器累加——**一个 kernel 只能选一种主序，另一侧必然跨 program 归约**。
- **单 kernel 在 NPU 三条出路全部走不通**：① 自旋锁（`atomic_cas`，Q3 SEQUENCE_PARALLEL
  路线）死锁风险，直接剔除；② GM 读-改-写无锁（Q3 autotune 实选路线）：每循环一轮 dq tile
  load/store，`aiv_vec_ratio` 0.71 挤占流水、`aic_mac_ratio` 仅 0.17、cube_util 77%，对比
  双 kernel（mac 0.51、cube_util 95~97%）**慢 1.30~2.90x**；③ 单 program 串行整个序列：
  并行度 = B×H，小 batch 喂不满 28 核。
- **双 kernel（K1: dk/dv kv 主序 + K2: out 重算 + dq query 主序）付出 2x 核心计算 +
  2x 输入读 + 双 launch，仍净赚**——重复计算换零原子、纯寄存器累加、全量直写在 NPU 上
  是正收益；小 shape（case11）优势收窄到 1.30x 即拆分固定开销占比升高的场景。
- **谱系佐证**：官方 FBGEMM CUDA 也拆两 kernel（Ampere `compute_dq_dk_dv` n-grid +
  `convert_dq` m-grid，dq 走 `atomicAdd` 进 fp32 GM buffer；Hopper 主 kernel +
  postprocess + `dq_semaphore`）。单 kernel 是 GPU 快原子红利下的 Triton 特化，
  **不可移植到无高效原子的硬件**；与 CV 融合证伪教训互为印证（合与分都须实测硬件同步/原子成本）。
