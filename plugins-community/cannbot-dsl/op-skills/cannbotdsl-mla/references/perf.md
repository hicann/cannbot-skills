# MLA 性能

板上实测（dav-3510, 32 AIC）。方法：kernel 重复 10 次，**Task Duration 取 min**
（mean 会被 HBM L2 cache 状态污染）。

## 先确认优化目标：device time，不是 wall time

第一次测 `mla()` 得到 **1.32 s/iter**，比 roofline 估算（~10 µs）差 12 万倍。
拆解后发现那**全是 host 开销**，msprof 显示 device kernel 只有 **57 µs**：

| 项 | 时间 |
|---|---|
| device kernel（msprof Task Duration min） | **57.2 µs** |
| host 侧 `mla()` 每次调用 | ~1332 ms |

host 开销来源：`@jit` 的 frontend 在**每次调用**都重新 trace 一遍所有 `@jit` 方法
（实测 3 次调用触发 147 次 `compile_frontend`）；另外 `torch.zeros_like` 在 NPU 上要
49 ms 而 `torch.zeros(shape, device=...)` 只要 0.047 ms（前者走 `allow_internel_format`
告警路径）。

**评测口径**：`cann-bench` 的性能路径走 `perf_evaluator.run_profiled()` →
`torch_npu.profiler` → 解析 `kernel_details.csv` 的 **device 时间**，
非 profiler 路径明确不用墙钟计时（注释：「受环境影响大、与 profiler 设备时间不可比」，
且 `elapsed_us = 0.0`）。

⇒ **优化 device kernel，host 开销不计分**（但会拖慢自己的测试迭代，
彻底解法是 staged AOT 预编译）。

## Baseline

| shape | 配置 | min(µs) | mac | **mte2** | mte1 | fix | vec |
|---|---|---|---|---|---|---|---|
| prefill | B2 S128 Skv128 Nq128 d512 causal | 57.2 | 0.51 | 0.78 | 0.42 | 0.18 | **0.81** |
| long | B2 S256 Skv256 causal | 152.8 | 0.57 | **0.88** | 0.48 | 0.18 | 0.71 |
| **decode** | B16 S1 Skv2048 full | **2532** | 0.45 | **0.96** | 0.58 | 0.10 | 0.35 |
| **mtp** | B16 S2 Skv2048 causal | **2678** | 0.43 | **0.98** | 0.53 | 0.10 | 0.34 |
| d448 | B2 S128 Nq64 d448 bf16 causal | 32.0 | 0.51 | 0.80 | 0.34 | 0.12 | 0.55 |

**诊断**：
- decode/MTP 是 prefill 的 **44~47 倍**，`mte2_ratio ≈ 0.97` → **近乎纯 memory bound**
- prefill/long 是 **vec bound**（0.81/0.71），cube 只用了一半
- fixpipe 全程很闲（0.10~0.18），不是 drain 瓶颈

## 优化 #1：head folding —— decode 47.8×

**根因**：`N_kv=1` 意味着 **128 个 query head 共享同一份 K/V**，
但按 (batch, head) 切 m-tile 会让每个 head 把整份 K/V 重读一遍：

```
decode: B=16 S=1 S_kv=2048 N_q=128 N_kv=1
  m_tiles = B*N_q = 2048，每个循环 16 个 n-tile
  实际读的 K/V : 9.13 GB
  真实的 K/V   :   71 MB
  冗余倍数     :  128x
```

**手段**：非 causal 时每个 query 行彼此独立，把 query head **折进序列轴**，
当成一个「高瘦的单头问题」：

| layout | 折叠前 | 折叠后 |
|---|---|---|
| BSND | `[B, S, N_q, d]` | `[B, S*N_q, 1, d]` |
| BNSD | `[B, N_q, S, d]` | `[B, 1, N_q*S, d]` |

两种 layout 下行都内存连续，所以是**纯 host reshape，kernel 零改动、数值无损**。
m_tiles 从 2048 降到 32，流量 9.13 GB → 143 MB。

```python
folded = False
if not is_causal and N_kv == 1 and N_q > 1:
    if inputLayout == "BSND":
        q_nope = q_nope.reshape(B, S * N_q, 1, d_nope)
        q_rope = q_rope.reshape(B, S * N_q, 1, d_rope)
    else:
        q_nope = q_nope.reshape(B, 1, N_q * S, d_nope)
        q_rope = q_rope.reshape(B, 1, N_q * S, d_rope)
    folded = True
    fold_S, fold_N = S, N_q
    S, N_q = S * N_q, 1
# ... 算完后（注意先切 padding 再 unfold）
if folded:
    out = (out.reshape(B, fold_S, fold_N, d_nope) if layout == "BSND"
           else out.reshape(B, fold_N, fold_S, d_nope))
```

**结果**：

```
decode        2532.04 µs   (baseline)
decode_fold     52.97 µs   ← 47.8x
prefill         57.22 µs   (baseline)
prefill_fold    56.62 µs   ← 不变（噪声内）
```

pipe 变化印证诊断：decode 的 `mte2` **0.96 → 0.77**，内存墙被拆掉。
prefill 五项指标变动 ≤0.01，确认没有波及 causal 路径。

**精度**：优化后完整 20-case 回归，MERE/MARE 与优化前**逐位一致**——确认无损变换。

**受益范围**：所有 `is_causal=False && N_kv==1 && N_q>1` 的 case，
即 decode 与非 causal MTP——恰好是原本最慢的一档。

## 尝试过但回滚：causal 的 head folding

MTP（causal）是剩下最慢的一档（2678 µs）。理论上 BSND 下也能折：
行 `r = s*N_q + head`，当 `tile_cube_m | N_q` 时一个 tile 内 s 是常数，
单个 `base_valid` 仍精确，kernel 只需用 `s = r // fold_n` 还原真实 query 位置。

**实施后失败并回滚**：

| case | 结果 | MERE |
|---|---|---|
| MTP causal（目标） | FAIL | 3.36e-3 |
| **prefill causal（本来是对的）** | **FAIL** | **1.11** |

关键信号是 prefill 也崩了——判据写成 `N_q % TILE_CUBE_M == 0`，而 128 % 64 == 0，
所以它**也被折了**。即便修掉这个判据 bug，`fold_n` 的管线同时穿过 `_n_tile_max` 和
`_stage_softmax` 两处 causal 逻辑，把未折叠路径也扰动了。

**没有继续追**：causal 折叠要同时改对「tile→s 映射」「n_tile 剪枝上界」「mask tile 偏移」
三处，且每处都在已验证的 causal 路径上。正确做法是**先写一个只测 `base_valid` 映射的
最小探针**，而不是直接改 kernel。

## 剩余优化空间（未实施）

| # | 手段 | 预期 | 前置条件 |
|---|---|---|---|
| 1 | causal head folding | MTP 2678 → 估 ~50 µs | 先写 `base_valid` 映射探针 |
| 2 | `pv_ub` depth 1→2，PV/update 重叠 | 直接命中 prefill/long 的 vec bound | UB 需 +64 K，当前 225/256 K 装不下；先把 `o_ub` 与 `res_o` alias 省 32 K |
| 3 | 减少 vec 侧列循环开销（VF unroll、合并 update+finalize） | 降 vec active time | 动 raw-VF，风险高 |
| 4 | staged AOT 预编译 | host 1.3 s → ~0 | **不计分**，但大幅加快自己的 sweep |

## 采集方法

性能采集需要构造一个能 `import mla` 并调用 `mla(...)` 的 harness，在 msprof 下重复跑 ≥10 次（`PipeUtilization` 是采样计数器，单次短 kernel 上会全 0）。

```bash
mkdir -p /tmp/perf_runs/<variant> && cd /tmp/perf_runs/<variant>
rm -rf PROF_*                       # 不清会拼接旧数据，min 被污染
MLA_PERF_REPEAT=10 \
  msprof --aic-metrics=PipeUtilization --task-time=on --ai-core=on \
  python <your_harness>.py <variant>
```

解析 op_summary CSV 的方法见 `../../../debug-skills/cannbotdsl-msprof-compare/` skill。

### 外部依赖

| 依赖 | 用途 | 缺失后果 |
|---|---|---|
| `msprof`（CANN 工具链） | 采集 device 时间与 pipe 利用率 | 无法采集，只能看墙钟（会误导，见本页开头） |
| NPU 设备 | 全部实测 | 无法采集 |
| `cannbotdsl-msprof-compare` skill | 解析 op_summary CSV | 需自己写解析脚本 |
