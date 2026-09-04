---
name: linear-recurrent
description: 四 线性类·状态递推算子（Gated Delta Rule / fused recurrent linear attention）的 Triton Ascend 优化经验
metadata:
  type: reference
---

# 线性类·状态递推算子优化经验（四 线性类）

> 锚定算子: `92_FusedRecurrentGatedDeltaRule`（naive/default，fla 系 fused_recurrent gated delta rule）
> 探索期: 2026-08-21 ~ 2026-08-24
> 硬件: Ascend910_9372（DAV_2201，910B 家族；与 `linear.md` 同 SKU）
> 结果: **50/50 cases 精度通过（全部 dtype × flag 组合；2026-08-24 独立复验确认）**。~~geomean 5.7527× vs torch~~ **加速比声明已撤回**：同环境配方（卡 1 + `TORCH_DEVICE_BACKEND_AUTOLOAD=0`）独立重测仅 **0.143×**——原测量中 torch 基线比所有其他口径慢 ~12 倍（case 5: 0.49ms vs 0.038~0.040ms），加速比来自被拉慢的基线而非更快的实现（详见 §5.1）
> 任务约束（本算子用户指定，同类任务可沿用）: forward() 计算路径禁止调用 torch_npu 专有接口，全部计算由 Triton kernel 完成（torch 仅限布局/视图/空分配等宿主操作）
> 头号教训: **masked 2D state store 在 Triton-Ascend 上会静默丢弃单元格**——output 全对（~9.5e-7）而 final_state 错，20/25 失败 case 源于此（详见 L1.3）

---

## §0 适用范围与算子分类

attention_index 行 12：**无 softmax**，改为状态递推或结合律重排。本卡先落**状态递推**细分；**结合律重排**细分尚无实证，待归档。

| 算子 | 子类标签 | 计算特征 | 优化哲学 |
|------|---------|---------|---------|
| FusedRecurrentGatedDeltaRule | `linear-recurrent` | 门控衰减（g/gk/gv 三种 gate 组合）+ delta rule rank-1 状态修正 `S = decay∘S + k⊗((v−Sᵀk)·β)`，每 `(b, vh)` 一套 `[K,V]` fp32 状态，T 步串行递推 | 单融合 persistent kernel：状态驻留 UB + flag constexpr 特化 + device 内串行 t 循环 |
| Mamba/SSM、RetNet 等状态递推变体 | `linear-recurrent`（待验证） | 状态递推同构（`S = a·S + k⊗v` 族） | 可复用 §1；首验先小规模 |
| 结合律重排（无递推） | 待归档 | `(QKᵀ)V → Q(KᵀV)` 型重排 | 本卡递推约束不适用，勿混用 |

> ⚠️ **与 `recurrent-neural-network.md`（LSTM/GRU/RNN）的分工**：那张卡的哲学是 **host-loop 单步 kernel**——其状态是 `[B,H]` 向量、无大 tile 可驻留；本卡状态是 `[BK,BV]` 矩阵（fp32 ≤32KB）可整块驻留 UB，因此**单融合 persistent kernel + device 内 t 循环**更优（5.75× 实证）。两卡「时间维串行、program 持完整状态」结论一致，分歧仅在发射粒度。
> ⚠️ **禁止套用 `flash_attention.md` 的 online softmax / KV 分块**——本类无 softmax，状态递推是唯一主链，FA 的滚动 `m/l` 约束整套不适用。

---

## Layer 1: 设计约束（硬性规则）

### L1.1 状态矩阵驻留 UB，单融合 persistent kernel（强制）
- **必须**: 1D `grid = (B*VH,)`，每 program 负责一个 `(batch, value_head)`（GQA: `h = vh // GQA_R`，q/k 用 h、v 用 vh），状态 `[BK, BV]` fp32 驻留寄存器/UB，`for t in range(T)` 在 kernel 内串行递推，一次发射完成全部计算。
- **Why**: 状态 ≤ `128*64*4B = 32KB ≪ 192KB UB`，驻留免每步读写 GM；host-loop 逐步发射被 launch 开销淹没。host-loop 仅当状态是向量、无 tile 可驻留时才是正解（RNN 卡场景）。
- **禁止**: 把时间维展平进 grid 跨 program 并行（递推真数据依赖，RAW）；循环内 `break`（与 trip-count 相关的编译约束）。

### L1.2 运行时 flag 全部 `tl.constexpr` 特化（强制）
- **必须**: 所有运行时开关（gate_mode 位 1/2/4、use_gate_in_kernel、use_bias、use_initial_state、output_final_state、use_qk_l2norm、use_beta_sigmoid、allow_neg_eigval、state_v_first 等）以 constexpr 参数传入，编译期消歧。Triton 按 flag 组合各自编译缓存；单 case 内组合固定，无重复编译开销。
- **禁止**: kernel 内用运行时标量 `if` 处理 flag —— 标量分支阻碍向量化，且不同 flag 组合各有最优代码路径。

### L1.3 ⚠️ masked 2D state store 静默丢单元格（头号精度陷阱，强制）
- **症状**: `tl.store(ptr + 2D偏移, tile, mask=2D_mask)` 写 `[BK, BV]` 状态时，**被 mask 掉的单元格不写且不报错**，读回呈**非确定性脏值**（用 `torch.zeros` 预填充后错误单元格精确为 0，可据此确诊"未写"）。典型表现：output 全对（~9.5e-7）而 `final_state` 错——实测 20/25 失败 case 与 `output_final_state=True` 相关，极隐蔽。
- **必须（修复，两案均已验证）**:
  1. **1D 连续 flat 布局**（终版采用）：kernel 内按 `[BK*BV]` 写（flat index = `k*BV + v`），host 侧 `view(B,VH,BK,BV)` → 裁剪 `[:, :, :K, :V]` → 按需 `transpose`；
  2. **计算出的行指针逐行 store**（Variant B2）：row-wise store with computed row pointers。
- **确诊手法**: 输出 buffer 预填 `torch.zeros`，错误格精确为 0 ⇒ 未写（本陷阱）；非 0 ⇒ 写错值（另有根因）。

### L1.4 边界 mask 用 fp32 向量比较（强制）
- `m_k = offs_k.to(tl.float32) < K.to(tl.float32)` —— 整数比较触发标量降级（scalar fallback），fp32 向量比较保向量核。

### L1.5 数值细节与 fp32 累加（强制）
- 状态、门控衰减、归约全程 fp32；输入读入即 `.to(tl.float32)`，仅最终 store cast 回目标 dtype。
- softplus 对齐 `F.softplus(threshold=20)` 的稳定形式：`tl.where(x > 20.0, x, tl.log(1 + tl.exp(x)))`。
- `F.normalize(p=2, eps=1e-6)` 对齐：`x / max(||x||₂, 1e-6)`（`tl.maximum` 防 0 除）。

### L1.6 kernel 返回前 host 侧同步（多 case 场景强制）
- host 侧 `torch.npu.synchronize()`：verify/benchmark 连续跑多 case 时，防止上一 case 的 kernel 异步写回与下一 case 的内存分配/复用竞争。单 case 无害，可常开。

### L1.7 可选输出指针不可传 None（强制）
- `output_final_state=False` 时 `ht_ptr` 仍需实参：传 1 元素占位张量（Triton 不接受 None 作指针），并用 constexpr 分支（L1.2）保证 kernel 不访问该指针。

### L1.8 BK=64 触发 PlanMemory 编译失败（UB 红线）
- 状态 + 多输入 tile 的 UB 需求超限时 bishengir 直接 `PlanMemory Failed`；需对 K 维分块（子 tile 递推）或降 BK。`[BK,BV]` 驻留与 BK 上限的边界本探索未完整扫描——遇编译失败先做 UB 记账再选 tile，不要盲目重试。

### L1.9 ⚠️ verify/benchmark 必须设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`（环境红线，强制）
- **症状**: 未设该变量时 torch 双 npu 后端注册冲突，报 `RuntimeError: Two accelerators cannot be used at the same time in PyTorch: npu and npu`。若发生在 impl/框架加载或运行期，会产生**大面积伪失败**——包括看似真实的数值错位（3/616 违例、1e3 量级脏值）、NaN、乃至整组 case 报错，极易被误判为代码 bug（2026-08-24 实例：同一份 50/50 正确的代码被误判为 41/50 失败）。
- **必须**: 所有 verify/benchmark 运行环境显式 `export TORCH_DEVICE_BACKEND_AUTOLOAD=0`（本模板 L1.6 的 `torch.npu.synchronize()` 路径同样依赖它）；配套 `LD_LIBRARY_PATH` 指向实际使用的 torch/torch_npu 构建。
- **教训**: 排查"数值失败"前先确认环境变量，再做代码归因；跨会话采纳/复核代码时双方必须对齐环境配方。

---

## Layer 2: 算法骨架（单融合 persistent 递推 kernel）

```
grid = (B * VH,)                         # 每 program: (b, vh)；GQA: h = vh // GQA_R
状态 S[BK, BV] fp32 = zeros / load(h0)   # 驻留 UB；h0 按 state_v_first 决定加载索引交换
for t in range(T):                       # 串行数据依赖，禁 break
    1) 门控衰减（flag 组合）:
         S *= exp(-exp(A_log)·softplus(g + dt_bias))     # gate_mode & 1 且 use_gate_in_kernel
       或 S *= exp(g)                                     # 标量 gate 路径
       或 S *= exp(gk)[:,None] · exp(gv)[None,:]          # 逐维 gate（&2 / &4）
    2) 读 q/k/v/beta（GQA: q/k 用 h，v/beta 用 vh）
       可选: l2norm(q), l2norm(k), sigmoid(beta)（allow_neg_eigval 时 ×2）
    3) delta rule rank-1 修正:
         kv    = Sᵀ k                    # [BV] = sum_k k[k]·S[k,v]
         resid = (v − kv) · β
         S    += k ⊗ resid
    4) 读出: o_t = (q·scale)ᵀ S          # [BV]，store 时 cast 回目标 dtype
末状态: S 按 1D flat [BK*BV] 连续写回（L1.3），host 侧 reshape/transpose 还原
```

host 侧要点：非 contiguous 输入防御性 `.contiguous()`；flag 全部转 bool 以 constexpr 传入（L1.2）；可选输出用占位张量（L1.7）；返回前 `synchronize()`（L1.6）；每 program 常量（如 `A_log[vh]`、`dt_bias[vh]`）在 t 循环外预取（LICM）。

---

## Layer 3: 关键代码片段（技巧可参考，变量名/结构须重设计）

**L1.3 修复：状态 1D flat 写回 + host 侧还原**
```python
# kernel 内：[BK,BV] 状态按 1D 连续布局写，元素 [k,v] 的 flat index = k*BV + v
flat_offs = offs_k[:, None] * BV + offs_v[None, :]
tl.store(ht_ptr + (b * VH + vh) * BK * BV + flat_offs, s_tile, mask=bound_mask)
```
```python
# host 侧：view -> 裁剪 -> 按目标布局转置
final = buf.view(B, VH, BK, BV)[:, :, :K, :V]
if state_v_first:
    final = final.transpose(-2, -1).contiguous()   # 目标 [B, VH, V, K]
```

**L1.2 constexpr flag 特化（调用侧）**
```python
kernel[(B * VH,)](
    q, k, v, ..., output, final_state,
    T, K, V, VH, H, GQA_R, scale,
    BK=next_pow2(K), BV=next_pow2(V),
    HAS_G=bool(gate_mode & 1), HAS_GK=bool(gate_mode & 2), HAS_GV=bool(gate_mode & 4),
    USE_INITIAL_STATE=bool(...), OUTPUT_FINAL_STATE=bool(...), ...,
)
```

**L1.4 fp32 边界 mask / L1.5 稳定 softplus**
```python
m_k = offs_k.to(tl.float32) < K.to(tl.float32)
sp  = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))   # F.softplus(threshold=20)
```

---

## 性能可达性

- ~~5.7527× vs torch（目标 5.0×，已达标）~~ **撤回**。2026-08-24 独立复测（同 impl md5 `5454f179`、同参考 sha `321a37eb`、同环境配方：卡 1 + `TORCH_DEVICE_BACKEND_AUTOLOAD=0` + 系统 torch 栈，warmup=5/repeats=50）：**geomean 0.143×**（50/50 精度通过）——比 torch 基线慢约 7 倍，未达任何加速目标。
- 原声明不可复现的根因：其测量的 framework 基线逐 case 慢 ~3-12×（case 5 报 0.4897ms，其余所有口径均为 0.038~0.040ms）；impl 耗时复测一致（case 1: 0.0079 vs 0.0070ms）。**加速比数字必须用 framework 绝对延迟做口径交叉校验**（见 §5.1）。
- 单融合 persistent kernel（状态驻留 UB + flag constexpr 特化 + device 内串行 t 循环）仍是本类形态的正确**架构方向**——精度 50/50 可达、结构合理；但架构正确 ≠ 性能达标：本实现 kernel 体（逐 token 标量 load + rank-1 更新）在大多数 case 上仍慢于优化过的 torch eager 参考实现，性能优化空间仍大（对照：同一 benchmark 上另一会话 13 轮迭代的最优实现为 1.3231×）。
- host 侧清理类微调（输入 contiguous 化精简等）无额外收益（见 §5），不值得做。

---

## §4 常见陷阱与避免方法

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| masked 2D store 写状态后 final_state 错、output 却对 | mask 掉的单元格静默不写，读回为脏值 | L1.3 flat 1D store + host reshape；zeros 预填确诊 |
| BK=64 编译失败 | UB 需求超限 → PlanMemory Failed | L1.8 先 UB 记账，K 维分块或降 BK |
| 整数边界比较 | 标量降级 | L1.4 fp32 向量比较 |
| softplus 大输入溢出 / 与参考不齐 | 未对齐 F.softplus threshold | L1.5 稳定形式 |
| 多 case 连续 verify 结果随机错 | kernel 异步写回与下 case 分配竞争 | L1.6 synchronize |
| 可选输出传 None | Triton 不接受 None 指针 | L1.7 占位张量 + constexpr 分支 |
| 时间维进 grid 并行 | 递推 RAW | L1.1 串行 t 循环 |
| forward 调 torch_npu 融合接口 | 本类任务用户约束 + AST 门禁 | 全部计算入 kernel |
| state_v_first 初态/末态错位 | 布局 `[V,K]` vs `[K,V]` 索引交换遗漏 | 加载与写回两侧都按 flag 交换（Layer 2 骨架注） |
| "Two accelerators: npu and npu" 大面积伪失败（含数值脏值/NaN） | 未设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，双 npu 后端注册冲突 | L1.9 环境红线；归因代码前先查环境 |
| speedup 声明虚高（如 5.75×） | framework 基线测量被拉慢（频率锁定异常/卡忙/计时段污染），impl 实际更慢 | framework 绝对延迟与历史口径交叉校验（§5.1）；同口径复测后再声明 |

---

## §5 已验证无效/排除的尝试（别重复）

### §5.1 ⚠️ 5.7527× 加速比声明撤回（2026-08-24，测量口径事故）
- **现象**: 本卡曾声明 geomean 5.7527×（复验 5.8651×）达标。独立复测（同代码/同参考/同环境配方/warmup=5/repeats=50）仅 **0.143×**，且 impl 耗时与原测量一致——分歧全在 framework 基线（原测量慢 ~3-12×）。
- **疑似根因**: 复验所用新版 benchmark.py 的 NPU 频率锁定（lock_npu_frequency）在 framework 计时段失效或降频；或 framework 计时段卡忙/环境残留（如 `ASCEND_LAUNCH_BLOCKING=1` 调试变量泄漏会把逐算子 launch 变同步，对多算子 torch 参考惩罚最大）。
- **规避**: ① 任何 speedup 声明前，将 per-case framework 绝对延迟与历史/跨会话口径对照（本算子 case 1/5 应为 ~0.012/~0.040ms 量级）；② benchmark 环境与调试环境严格分离，`unset ASCEND_LAUNCH_BLOCKING`；③ 频率锁定失败时按 error 处理而非 warn。

- **标量头基准指针的状态写回**（Variant A）：无效。
- **指针行索引写法的 row-wise store 首版**（Variant B 原始版）：编译错误；修正为"计算出的行指针"后可用（即 L1.3 方案 2）。
- **host 侧清理迭代**（输入布局预处理精简，迭代 #17）：无 speedup 提升，保留基线。
- **调试提示**: kernel launch 卡顿时用 `ASCEND_LAUNCH_BLOCKING=1` 同步复现定位；torch inductor compile_worker 的 traceback 属探测噪音，可忽略。
