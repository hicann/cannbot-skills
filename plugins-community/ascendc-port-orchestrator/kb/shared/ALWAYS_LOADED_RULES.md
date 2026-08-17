# AscendC 开发必读规则（无条件加载）

> **Worker & Orchestrator 必须在 Phase 0 开始写任何代码之前读完此文件。**
>
> **同时必读 `ANTI_PRESSURE_PROTOCOLS.md`（P1–P8 LLM-pressure drift modes）。
> 本文件管"代码层规则"，那个文件管"在压力下我会绕过哪些规则"。两份都要。**
>
> 这里收录的是**跨 op 通用的 process rule 和 universal trap** —— 不是 op-specific 的技术细节。
> Op-specific 的经验仍在 `OPERATIONAL_KNOWLEDGE.md`，通过 KB_INDEX 按 tag 选择性加载。
>
> 为什么这些 OL 要无条件加载：**它们都是"如果不在下笔前知道，后面再查也晚了"的规则**。
> tag-based 加载假设 worker 知道自己需要什么；但幻觉式错误（如 OL-80）的特征恰恰是 worker
> 不知道自己在犯错，所以不会主动去加载相关 KB。

---

## 1. Meta 规则（防止幻觉式错误）

### OL-80: API 存在性必须先查 catalog，禁止凭记忆/推测发明 workaround

**前置步骤**：写任何 VEC op 之前，grep `ASCENDC_API_CATALOG.md`。

```bash
# 想到"每元素 / 标量" → 立刻查
grep -i "Divs\|scalar div" src/skills/references/target/ascendc/API_CATALOG.md
```

反例：以为 AscendC 没有 `Divs` → 自己写 `Muls(x, 1/scalar)` workaround → 精度不匹配 → 多轮失败。
实际 catalog 第 54 行明确有 `Divs`。

如果 catalog 没找到 → `python3 src/scripts/fetch_ascendc_doc.py <ApiName>` 查官方文档。
两者都没有 → 才考虑 workaround，且必须在 `knowledge_update.md` 里标注"缺失 API"。

### OL-130: API 存在性 lookup chain — catalog → SDK header → docs（先 fall through 再放弃）

**规则（P0aau, 2026-05-07）**：当你不确定 AscendC primitive 的 signature / template params / overloads / dtype constraints 时，**lookup 优先级**严格如下：

1. **`ASCENDC_API_CATALOG.md`** — curated 一行 summary，第一站。但**它是手工维护的**，对复杂 primitive (尤其 adv_api 模板类如 `Matmul<>`、`Normalize<>`、`LayerNorm<>`) 经常 under-specifies。
2. **SDK 头文件 `$CANN_PATH/include/ascendc/...`** —— 例如 `/data/cann_b103/cann-9.0.0/include/ascendc/`。**这是权威源**，是 AscendC primitives 的实际 API 声明。**Reading SDK headers IS ALLOWED** —— 它们是 **SDK 分发的公开 header**，不是 op-impl 源代码 (后者在 `~/workspace/cann/` 是禁区)。区别清楚：
   - ✅ `/data/cann_b103/cann-9.0.0/include/ascendc/lib/matmul/matmul.h` — SDK header, 公开 API 声明，**允许读**
   - ❌ `~/workspace/cann/ops-nn/...` / `ops-transformer/...` — op-impl 源代码，**禁止读**（NPUKernelBench scope rule，CLAUDE.md）
3. **hiascend.com 文档** —— 当 SDK header 难读 (template specialization 场景) **OR 涉及自定义算子调用约定 (pybind / ACLRT_LAUNCH_KERNEL / 算子工程结构 / vendor opp install / aclnn 注册)**。**访问方式强制规定**：

   - hiascend.com 是 JS 懒加载站点。`WebFetch` 工具**返回空 placeholder**，不可用。
   - 必须用 playwright MCP 工具:
     - `mcp__plugin_playwright_playwright__browser_navigate(url=<atlas_ascendc_*.html>)`
     - `mcp__plugin_playwright_playwright__browser_evaluate(function="() => document.querySelector('article').innerText")` 拿渲染后正文
   - 起点 URL 列表: `src/skills/references/hardware/HIASCEND_DOC_URLS.md`（已在 Tier-1 manifest）。先 grep 找最相关页面，再 playwright 拉正文。
   - **当问题是"如何从 Python 调用我们自己生成的 kernel binary" / "pybind wrapper 怎么写" / "ACLRT_LAUNCH_KERNEL 宏怎么用" / "需不需要 vendor opp install"** → 必读 `atlas_ascendc_10_0057.html` ("Pybind调用") 和 `atlas_ascendc_10_0056.html` ("Kernel直调") 再写代码。
   - **P140 (2026-05-17)**: a5 agent 9h 错路径 — 试图用 aclnn-direct + vendor opp install 调用 unshipped op，未读 AscendC 文档。文档明说自定义算子用 ACLRT_LAUNCH_KERNEL 宏 + pybind 直接绑定 kernel，不走 aclnn 注册。所有 spawned agent (kw/pp/ko/fo/ar/da) 同款盲区，因为 HIASCEND_DOC_URLS.md 之前没在 manifest Tier-1。
4. **声明 "API missing"** in `knowledge_update.md` —— 只有前三步都没找到才走这步。

**为什么强调**：OL-80 说 "grep API_CATALOG before code"，但**不要止步于 catalog**。
catalog 是 cliff notes; SDK header 是 ground truth。对 6_QuantMatmul 这种依赖
adv_api `Matmul<>` 的复杂 op，catalog 的一行说明完全不够 — kw 必须读
`/data/cann_b103/cann-9.0.0/include/ascendc/lib/matmul/...` 头文件去看
template params (M_, K_, N_, A_TYPE, B_TYPE, BIAS_TYPE, etc.)。

**Cold-start tightness**: 不要直接 fall back to 前置 op 的 pattern-copy（如
`1_BatchMatmul`）来推断 API 形状 — 那个 op 可能是 CANN aclnn pybind wrapper，
不是真正的 AscendC `Matmul<>` 实现。**先看 SDK header 再决定 pattern source**。

**Evidence**：2026-05-07 6_QuantMatmul cold-start, kw 读了 `1_BatchMatmul`
作为 pattern reference 但**完全没有读** `/data/cann_b103/cann-9.0.0/include/ascendc/`，
导致对 `npu_quant_matmul` 的 AscendC adv_api 替代品判断不充分。

### OL-13: 发明 workaround 前必须搜索已有 skill / 工具

遇到问题（网络、部署、认证、编译环境）先查：
- `src/skills/` 下所有 skill 的 SKILL.md
- `docs/` 目录的设计文档
- `src/scripts/` 下的工具脚本

反例：花 30 分钟手动 scp+tar 部署，实际 `deploy_to_a5.sh` 已存在；花 1 小时写 proxy workaround，实际 `--proxy` flag 早已在 a5_op skill 文档里。

### OL-23: 不能把"平台特性坏掉"作为永久标签 — 必须定期重新验证

平台 bug 会随 CANN 版本修复（如 OL-4: TQue 在 CANN 9.0.0 已修复）。
引用 PB/OL 时检查发布日期，对超过 1 个月的结论做 sanity check。

---

## 2. 信任校准（trust calibration）

### OL-28: 经审计的 CPU PyTorch 是数值规格

设备实现与 CPU 数学规格冲突时，以经过审计的 CPU 规格为准，并记录差异。

### OL-36: 任何 PyTorch/CANN 委托都是 cheating（benchmark 模式硬性禁止）

在 benchmark 模式（NPUKernelBench）下：
- pybind11.cpp 里调用 `torch.xxx()` 做计算 → cheating
- kernel 通过 `aclnn*` / `aclop*` API 调用 CANN op → cheating
- CPU fallback → cheating

pybind 只能做: tensor metadata、memory allocation、`contiguous()`。
计算必须全部用 AscendC primitive (DataCopy, VEC ops, TQue/TBuf)。

### OL-42: CANN 性能差距是知识差距，不是硬件限制

CANN 和 AscendC 用**完全相同的硬件**。如果 kernel 比 CANN 慢，是 worker 不会用硬件，不是 CANN 有秘密 API。
永远不要写"CANN uses some HW feature we can't access"这种结论——这是给自己找借口。

---

## 3. 工作流程（process rules）

### OL-1: 专家反馈必须改代码验证，禁止在文档里标"已确认安全"

专家指出的 bug → 必须改代码 + 编译 + 跑精度测试才算修复。
反例：int64 问题被标"确认安全"三次，代码始终没改，bug 还在。

### OL-18: 每次代码更新必须同时验证精度 AND 性能

只跑精度不算完成。历史教训（OL-16）：int64 修复通过精度但带来 6% 性能退步，两个 batch 都没发现。

### OL-27: 性能声明必须基于同条件 A/B 数据

"性能无退步"/"提升 X%" 必须满足：
- 同一 NPU
- 同一 session（容器重启会改变状态）
- 背靠背运行
- 同一 benchmark 脚本

跨 NPU / 跨 session / 跨脚本的数据**不可比**。如果无法 A/B，文档中写"性能未验证"。

### OL-31: 性能评测用 benchmark 框架的标准工具

NPUKernelBench 用 `performance.py`，不要自己写测速代码（结果不可比）。

### OL-10: README 驱动的文档更新

每次工作后按 README 的文档指南表逐个检查是否需要更新：REPORT / BENCHMARK / OPTIMIZATION_PLAN / EXPERT_FEEDBACK / SKILLS_DESIGN 等。

### OL-24: 优化一个 kernel 后必须审查所有同类 kernel

如果在 kernel A 发现了一个优化/bug，立刻在所有结构类似的 kernel B/C/D 里检查是否有同样问题。

### OL-30: 性能优化不能以精度降级为代价

任何"性能 +X% 但精度 ratio 放宽"的 trade-off 必须先获得用户明确批准，不能自己决定。

### OL-127: 单线程 SIMT（`LAUNCH_BOUND(1)` / `nblk=1` + 标量 for-loop）禁止作为 op 最终状态

**规则**：单线程 SIMT 实现（`LAUNCH_BOUND(1)` 或 `nblk=1` 配合标量 for-loop）**永远不能**作为 op 的最终交付状态。仅允许作为**临时占位**——当精度问题尚未解决、需要先排除并发竞争作为 hypothesis 时可用，但精度通过后**必须**改成 multi-core / SIMD 实现，**性能测量前替换完毕**。

```cpp
// ❌ 不能 ship — 单 core 单 thread，硬件利用率 1/56 = 1.8%
__global__ __aicore__ __launch_bounds__(1) void my_kernel(...) {
    for (int i = 0; i < N; ++i) {
        out[i] = compute(in[i]);  // 标量 for-loop
    }
}

// ✅ 临时调试可以（注释清楚是临时）
// TEMPORARY: pin to single thread to rule out concurrency bug,
// MUST replace with multi-core SIMD before perf measurement
__global__ __aicore__ __launch_bounds__(1) void my_kernel(...) { ... }
```

**为什么强调**：单线程 SIMT 永远 < 1× ratio（54-56× 资源浪费）。若以此为最终态测 perf，结论是"算子结构无法优化"——这是错误归因，是**没有真正使用硬件**。

**何时短暂使用**：精度调试期间——若 multi-core 版本精度 fail，单线程版本能帮助定位是否是并发 race（atomic ordering、bank conflict、UB stride 重叠）。一旦精度根因定位，恢复 multi-core / SIMD。

**Evidence**：ops where this trap was hit and reverted — see knowledge_update.md across multiple ops where worker submitted 单线程 SIMT 当 final state、被 self-critic 或 user pushback 退回。

### OL-125: 后台命令禁止接 `| tail -N` / `head` / `wc` / `sort`（任何 EOF-only 过滤器）

**规则（P0aam, 2026-05-07）**：当用 `Bash(run_in_background=true)` 跑长任务（build / ref_preflight / msprof / orchestrator cold-start / 任何超过 30s 的命令），**禁止把输出 pipe 进 `tail` / `head` / `wc` / `sort`**。

```bash
# ❌ 错误：tail 只在 upstream EOF 时输出；如果 upstream hang，整条 pipeline 都看不到任何输出
python3 long_script.py 2>&1 | tail -40   # run_in_background=true

# ❌ 同类错误
some_command | head -20
some_command | wc -l
some_command | sort

# ✅ 正确：harness 自动捕获 stdout 到 task output file，可以 mid-run 读
python3 long_script.py 2>&1   # run_in_background=true
python3 long_script.py        # stdout 已被 harness capture，不需要 redirect
```

**为什么 pipe 会破坏一切**：`tail`/`head`/`wc`/`sort` 都只在 EOF 时 emit。配合 `run_in_background`，harness 的 output file 在整个运行期间是空的——upstream 哪怕在 line-buffered 输出，也被 pipe block 住。如果 upstream hang，整条 pipeline hang，mid-run 诊断完全失效。

**正确的 live-filter 方式**：用 `Monitor` tool + `grep --line-buffered`（不是 Bash + tail）。

**Evidence**：8_WeightQuantBatchmatmul ref_preflight 2026-05-07，5+ 分钟"无输出" 完全是 `2>&1 | tail -40` 造成的——容器内 progress.log 其实有诊断数据，只是 orchestrator 端看不见。

### OL-126: 长任务必须用 `Bash(run_in_background=true)`，禁止用 `nohup`

**规则**：`nohup <cmd> &` 会从 CC 的 task tracker 上脱离，UI 看不到进程，silent stall 检测失效。一定用 `Bash(run_in_background=true)`，让 CC 跟踪 subprocess，在 task panel 里可见、可 `TaskStop` 杀死、可 mid-run 读 output file。

```bash
# ❌ 错误：CC 完全不知道这个进程在跑
nohup python3 batch.py > log.txt 2>&1 &

# ✅ 正确：CC 跟踪、UI 可见、output 可读
Bash(command="python3 batch.py", run_in_background=true)
```

**Evidence**：早期 batch dispatcher 调试中多次因 nohup 让 CC 误以为任务已结束，结果 job 还在跑——最终通过这条规则修正。

---

## 4. Universal design traps（任何 kernel 都会踩）

### OL-63: Elementwise kernel 必须用 `TQue<VECIN, 4>`（depth=4）

```cpp
// ❌ 错误：depth=1 或 2 → pipeline 只能单 buffer，带宽利用率低
TQue<QuePosition::VECIN, 1> inQueue_;

// ✅ 正确：depth=4 → MTE2 + VEC + MTE3 三级流水能并行 4 个 tile
TQue<QuePosition::VECIN, 4> inQueue_;
```

elementwise ops（没有 reduction / 状态）的 HBM 带宽瓶颈，depth=4 是必须的，不是"可选优化"。
depth=1 会导致 perf ~0.3-0.5x。

### OL-66: `torch::zeros` on NPU 不与 custom kernel stream-ordered

```cpp
// ❌ 错误：zeros 可能在 kernel 执行后才真正清零
auto out = torch::zeros({...}, opts);
launch_kernel(out.data_ptr(), ...);  // 读到的可能不是 0

// ✅ 正确：显式用 aclrtMemset 或先 sync
auto out = torch::empty({...}, opts);
aclrtMemset(out.data_ptr(), size, 0, size);
launch_kernel(out.data_ptr(), ...);
```

pybind 里任何用 `zeros` 作为 kernel 输入 buffer 的，必须仔细考虑 stream ordering。

### OL-77: GM tiling struct 必须逐字段标量读取，禁止整体拷贝

```cpp
// ❌ 错误：整体 memcpy 可能读到垃圾
MyTiling t = *reinterpret_cast<__gm__ MyTiling*>(tilingGM);

// ✅ 正确：用 DataCopy 搬到 UB，逐字段 GetValue
// 或：用 CopyTiling 辅助函数逐字节拷贝
inline __aicore__ void CopyTiling(MyTiling *dst, GM_ADDR tilingGM) {
    auto *src = reinterpret_cast<__gm__ uint8_t *>(tilingGM);
    auto *d = reinterpret_cast<uint8_t *>(dst);
    for (int i = 0; i < sizeof(MyTiling); ++i) d[i] = src[i];
}
```

### OL-25: TBuf 没有自动同步 — 必须 TQue 或显式 sync

TBuf 是"裸"buffer，读写之间**没有** pipeline 同步。
如果跨 pipeline stage（MTE2 → VEC → MTE3）用同一 TBuf，必须 `PipeBarrier` 或 `SetFlag/WaitFlag`。
只有单一 pipeline（纯 VEC 计算）内部 TBuf 才安全。

---

## 5. 浮点 kernel 精度铁律（任何平台、任何 dtype，**最高优先级**）

> **这不是昇腾特有，这是所有浮点硬件的共同性质。IEEE 754 浮点不满足结合律。
> 任何能胜任的 source / AscendC / Intel AVX / ARM NEON 工程师都遵循此铁律。**

### 核心规则

**Reference 就是 spec。第一版 kernel 必须是 reference 的逐字翻译**：
- reference 里一个 op，kernel 里一个 VEC 调用
- 同顺序（不重排）
- 不 fusion（不融合多个 op 成一个）
- 不 strength reduction（不把 `a/b` 改成 `a * (1/b)`）
- 不 pre-compute coefficient（不把 `grad_var * 2.0 / HW` 压成一个 scalar）
- 不 reassociate（不把 `(a*b)/c` 改成 `a*(b/c)`）

**fp32 例外（kernel 内部计算全在 fp32 中时，以下变换是精度安全的，可在 Phase B 首次编码就采用）**：
- ✅ `Divs(vec_fp32, scalar_fp32)` → `Muls(vec_fp32, 1.0f/scalar_fp32)`：IEEE 754 fp32 除法与乘法均为正确舍入，
  变换前后相差 0-1 ULP。将 VEC 吞吐从 ~1 elem/cycle 提升到 ~8-16 elem/cycle（4-8×）。参见 OL-256。
- ⚠️ 上述变换在 **fp16/bf16 中不适用**（中间 `1/b` 损失 13+ 位尾数）。fp16/bf16 路径仍必须遵守上方规则。

### 工程流程

```
Step 1: 逐字翻译 → 跑 parity test → 达到 bit-exact（50/50 PASS）
        ↓
Step 2: 开始 **单步** 优化：
        - 一次引入一个"优化"（fusion、reorder、strength reduction）
        - 每引入一次立即跑 parity test
        - 如果 parity 退化 → 这个"优化"是 regression，revert
        - 如果 parity 保持 → keep，继续下一个优化
```

### 反模式（你写代码时如果出现以下念头，立刻停）

| 念头 | 实际后果 |
|------|---------|
| "这样写少一条 VEC 指令，更快" | fp16 精度破坏，50/50 退化到 27/50 |
| "把 `1/std` 预算出来，后面 Muls 就可以少一次 Div" | fp16/bf16: `a*(1/b) ≠ a/b`，bit-exact 破坏。**fp32: 精度安全，见上方 fp32 例外** |
| "把 `grad_var * 2 / HW` 预算成 `coeff_b`，后面一次 Muls" | 求值顺序变了，broadcast 被 scalar 压缩，fp16 不等价 |
| "这个中间 tensor 可以复用 buffer，省 UB" | 没问题，只要 op 顺序不变 —— **buffer 复用不是 rewrite** |
| "这两个相邻 Muls 可以 fuse 成一个 `Muls(x, a*b)`" | 除非 a、b 都是编译期常量且乘积可 fp16 精确表示，否则 rewrite 破坏精度 |

### 失败签名

如果你看到以下情况，**立即 revert 到最近的 literal 版本**：

```
"I made a math-equivalent change to simplify the kernel, but precision dropped"
"Changed (a*b)/c to a*(b/c), parity went from 50/50 to 33/50"
"Pre-computed coefficient, fp16 cases started failing"
```

### 工程基本功（不是借口）

**错误的归因**："我被 source 优化文化影响了，习惯性 fusion" ← 不是真正的借口。
任何 fusion 实现都必须完成 CPU 真值 parity 验证并声明精度模式。
有经验的 external 工程师做 fp16 kernel 时也是 literal-first → verify → optimize。

**正确的归因**："我跳过了 parity-first 流程，直接进入'看起来更优雅的版本'" ← 这是流程缺失。
无论平台，精度敏感代码都应该 literal-first + 逐步 opt-in 优化。

### 与 OL-80 的关系

- OL-80：写 code 前先查 API catalog（防止幻觉式 workaround）
- 本规则：写 code 时不做数学等价变换（防止精度破坏）

两者都是"写代码前必须做的决策点检查"，共同防止**单体 agent 的注意力失败**。

### Evidence

op #14 AdaIN2D Backward: 4 种不同的 kernel 变体（Muls(1/x), Div, Divs, hybrid 预压缩）
全部在 27/50 左右。失败模式一致：fp16 case 有 `max_abs_diff=inf` 的 mismatch，
来自**数学等价但求值顺序变了**的 rewrite。

---

## 6. 精度相关的通用规则（dtype-agnostic）

### OL-82: KB pattern 必须配 minimal repro 验证 — 否则不能当"精度必需"使用

**Meta-rule**: 从 CANN 源码或其他参考"提取"的 pattern，即使看起来语义清晰，**不能直接当作 bit-exact 必需条件应用**。必须先写 minimal repro 验证：
```python
# 示例：Muls(1/N) vs Div(N) 是否 bit-equal?
x = torch.randn(64, dtype=torch.float32).npu()
a = (x / N).cpu().numpy().view(np.uint32)
b = (x * (1.0/N)).cpu().numpy().view(np.uint32)
print((a != b).sum())  # 0 表示 bit-identical
```

**判断 pattern 类型**：
- **Requirement（精度必需）**：不匹配会产生 mismatch。必须严格应用。
- **Convention（工程偏好）**：为性能/代码风格/可读性而选，不影响精度。**不能当作必需条件**。

**反例教训（2026-04-16）**：
- 看 CANN `reduce_mean_dag.h` 用 `Muls(1/N)` 替代 `Div(N)`，**假设**是精度原因 → 写成 P-P53 "必须用 Muls"
- 应用到 op #14 fp32 路径的 `grad_mean / spatial` → 破坏原本 18/18 PASS 的 fp32
- 事后做 minimal repro：`x/N` 和 `x*(1/N)` 在 NPU 上 bit-identical → P-P53 的精度论断是错的
- 真相：CANN 用 Muls(1/N) 是 **perf 优化**（predivide once），不是 **精度必需**

**对 worker 的指令**: 遇到 "这个 pattern 必须这样写" 的疑问时：
1. 写 10 行 torch 测试 repro
2. 两种写法在 NPU 上运行
3. bit 比较
4. 有 diff → pattern 是精度必需；没 diff → pattern 是 convention，kernel 可自由选择

**和 OL-80 的关系**: OL-80 要求"先查 API"，OL-82 要求"先验证假设"。两者都是"在写代码前做调查"的具体实践。

### OL-81: bf16/fp16 Cast 默认用 **CAST_RINT**（IEEE RNE），不是 CAST_ROUND

AscendC 的 `CAST_ROUND` ≠ IEEE 754 round-to-nearest-even。它是 "round half UP"。
要对齐 PyTorch/torch_npu 行为（IEEE 754 RNE），**必须用 `CAST_RINT`**。

```cpp
// ❌ 错误：CAST_ROUND 是 round half up，不匹配 PyTorch
Cast(bf16_tensor, fp32_tensor, RoundMode::CAST_ROUND, count);

// ✅ 正确：CAST_RINT 是 IEEE 754 RNE，与 PyTorch 一致
Cast(bf16_tensor, fp32_tensor, RoundMode::CAST_RINT, count);
```

Evidence: op #14 AdaIN2D Backward，CAST_ROUND → 8 个 bf16 case FAIL；CAST_RINT → 8/9 bf16 PASS（单 flag 切换）。

两者差异只在"正好一半"的值上（如 0.5），但在长链 mul/div 里会累积到 bit-level mismatch。

### OL-79: NPU fp16/bf16 除零产生 inf，与 PyTorch CPU/NPU 完全一致

fp16 Div / Mul / Pow 在 NPU 上的行为**bit-level 匹配 PyTorch**。
如果 reference 在 native dtype 下产生 inf（如 `1/0.01` 在 fp16 下），kernel 直接在 native dtype 下算就会天然匹配。
不要为了"避免 inf"主动 clamp 或升精度——反而会导致 mismatch。

---

## 快速自检（写代码前）

- [ ] 我想用的每个 VEC op 都在 `ASCENDC_API_CATALOG.md` 里 grep 过了吗？（OL-80）
- [ ] pybind 里没有任何 torch/CANN 计算委托吗？（OL-36）
- [ ] 基础设施问题（部署/代理/编译）都先查了已有 skill 吗？（OL-13）
- [ ] TQue depth 是 4 吗？（OL-63, 仅 elementwise）
- [ ] pybind 的 zeros 是否可能和 kernel 有 stream 顺序问题？（OL-66）

## 快速自检（测试和声明前）

- [ ] 精度 AND 性能都验证了吗？（OL-18）
- [ ] 性能数字是同条件 A/B 测出来的吗？（OL-27）
- [ ] 同类 kernel 的同样问题检查过了吗？（OL-24）
- [ ] 专家反馈 / 用户纠正都改到代码里了吗？（OL-1）

## Kernel-authoring guards

Kernel-authoring agents (`aog-kernel-worker`, `aog-kernel-optimizer`) MUST also read
`shared/KERNEL_AUTHORING_GUARDS.md` before writing or editing kernel sources. Those rules
are harness-neutral: they apply identically under Claude Code and opencode.
