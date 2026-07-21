# CHANGELOG - Code Performance Advisor

## [Unreleased]

### Known Issues

#### ⚠️ ISSUE: 评测模式不一致导致 COMPARE 基准不可比 (2026-02-27)

**现象**：baseline 和优化后的 profiling CSV 可能来自不同评测模式，导致比较基准不一致：

| 阶段 | 评测脚本标志 | 评测引擎 | warmup 位置 | CSV 形态 |
|------|-------------|---------|-------------|---------|
| 初始 baseline | 默认（无 `--advanced-perf`） | `MsprofProfiler` | profiling 窗口**外**（10 次） | 干净，只有目标算子（20行）|
| 优化后评测 | `--advanced-perf --task-type vector` | `AdvancedPerformanceEngine` | profiling 窗口**内** | 每轮 3 行：MatMul + ReduceMax + 目标算子（60行）|

**影响**：
- Advanced 模式每次执行目标算子前先跑 MatMul（10240×10240）清 HBM + ReduceMax（96×1024×1024）清 L2 cache，测量条件更"干净"但与 baseline 不等价
- 若 baseline 用 basic 模式、优化后用 advanced 模式，提升幅度会被高估（测量体系不同）

**示例**（aten_native_batch_norm）：
- baseline (basic, first row): 10.144 us
- 优化后 (advanced, median of stable rows): 8.664 us → +14.6%
- 优化后 (advanced, min): 8.040 us → +20.7%（此值被误用，导致虚报达标）

**根本原因**：`ascendc_evalution` skill 的 `evaluate.py` 中，两次评测调用了不同的性能模式；`profiling_extractor.py` 原先使用 `iloc[0]`（第1行，warmup 冷启动）或 `min`，均不够稳健。

**已修复**：`profiling_extractor.py` 改为 skip 第1行后取 **median**（对 dirty CSV 已过滤出目标算子行再取中位数）。

**待修复**：`ascendc_evalution` skill 应统一 baseline 和优化后评测使用相同标志（建议均使用 `--advanced-perf` 或均不使用），以保证比较基准一致。追踪于 `ascendc_evalution/evaluate.py` 的 `compare_performance_advanced` 调用路径。



### Added (2026-02-26) - COMPARE 不达标自动 retag + 行数阈值检测

#### COMPARE 不达标 → 强制 retag + 循环优化
**动机**: 每次 APPLY 后代码已变更，若 COMPARE 不达标直接结束会丢失上下文；下一轮优化应基于更新后的代码重新分析。

**实现**:
- `_phase_compare()`:
  - 不达标时：设 `context["force_retag"] = True`，跳转回 **TAG** 而非 DONE
  - 新增 `optimization_rounds` 计数器，上限 `MAX_ROUNDS = 5`，超出后终止循环
  - 修复 `input()` EOFError（非终端环境默认 'n'）
- `_phase_tag()`:
  - 优先读取 `context["force_retag"]`（优先级高于 CLI `--force-retag`）
  - 读取后立即清零（单次触发语义）
  - `should_force = self.force_retag or ctx_force_retag` — 两种触发源统一处理

**循环流程**:
```
COMPARE (不达标, round < MAX) → TAG (force_retag=True) → code_tag → SCORE → SUGGEST → APPLY → BUILD → EVALUATE → COMPARE
```

#### TAG 行数阈值检测
**动机**: 代码变了多少决定 tag 是否仍然有效；注释改动不值得重新调用 LLM。

**实现** (`_phase_tag()`，仅在 `resolved_tag != None` 时生效):
- 统计 tag 生成后 mtime 有变动的文件总行数（粗略估算改动规模）
- 三档提示（不阻塞，仅信息输出）：
  - `< 20 行`：静默复用，打印 "trivial changes"
  - `20-200 行`：提示 "minor changes"，建议用 --force-retag
  - `≥ 200 行`：⚠️ 警告 "significant changes"，建议 --force-retag

#### 修复：EOFError 兼容
- `_phase_apply()`: `input()` 加 try/except EOFError，非终端默认继续（'y'）
- `_phase_evaluate()`: `input()` 加 try/except EOFError，静默继续

**文件修改**: `scripts/analysis_engine/workflow.py`



#### `--skip-build`：跳过 BUILD 阶段
**动机**: 每次 `./build.sh` 耗时 5-8 分钟。当代码刚刚编译好、只需重新评测时，不应重复编译。

**实现位置**: `scripts/analysis_engine/workflow.py`

**变更**:
- `WorkflowEngine.__init__()` 新增 `skip_build: bool = False` 参数
- `_phase_build()` 开头添加早返回：`if self.skip_build → 直接 transition 到 EVALUATE`
- `cmd_run()` / `cmd_resume()` 透传 `skip_build` 给引擎
- `build_parser()` `run` 和 `resume` 子命令均添加 `--skip-build` flag

**用法**:
```bash
# 代码已编译，只跑评测和分析
python3 scripts/analysis_engine/workflow.py run --op my_op --skip-build

# 恢复流程时也可跳过
python3 scripts/analysis_engine/workflow.py resume --op my_op --skip-build
```

---

#### `--force-retag` + 激进 TAG 缓存
**动机**: TAG 阶段需要调用 LLM 分析代码（耗时），但每次 workflow 都检查 tag 文件是否存在，即使代码没变也可能触发重新标注。

**实现位置**: `scripts/analysis_engine/workflow.py`，`_phase_tag()`

**变更**:
- `WorkflowEngine.__init__()` 新增 `force_retag: bool = False` 参数
- `_phase_tag()` 缓存策略：
  - 默认行为：只要找到任意 tag 文件（精确名或 glob 匹配），直接复用，不管代码是否更新
  - 增加 mtime 对比：提示"code unchanged"或"code changed but reusing"，但**不阻塞流程**
  - `--force-retag`：跳过缓存查找，强制走 code_tag 重新生成
- `build_parser()` `run` 和 `resume` 子命令均添加 `--force-retag` flag

**用法**:
```bash
# 正常用法：代码没变，自动命中缓存（不需要任何额外参数）
python3 scripts/analysis_engine/workflow.py run --op my_op

# 代码有实质性改动，需要重新分析
python3 scripts/analysis_engine/workflow.py run --op my_op --force-retag
```

**文件修改**: `scripts/analysis_engine/workflow.py`（`__init__`, `_phase_tag`, `_phase_build`, `cmd_run`, `cmd_resume`, `build_parser`）



### Issues Found (2026-02-26) - aten__fused_adamw_ 实战记录

**背景**: 对 `aten__fused_adamw_` 算子（shape 1024×4096，baseline 0.65x）运行完整优化流程时，发现以下问题。

#### 问题1：workspace/inputs 需手动初始化
**现象**: `workspace/inputs/` 目录为空，workflow INIT 阶段无法自动从 `output/{op_name}/` 复制代码和 profiling 数据。

**影响**: 用户必须手动执行 `init_workspace.py` 或参考 `Init_workspace.md` 完成初始化，降低上手体验。

**建议修复**: workflow INIT 阶段增加自动检测逻辑——若 `workspace/inputs/{op}/` 不存在，自动从 `output/{op}/` 目录拷贝代码和最新 profiling CSV。

---

#### 问题2：TAG 阶段无法自动触发 code_tag 子技能
**现象**: workflow 检测到 tag 文件缺失后，仅提示"请调用 code_tag 子技能"，但不会自动执行。TAG 阶段完全依赖 agent 手动理解并调用子技能。

**影响**: 在 auto 模式下，如果 agent 不主动调用 code_tag，流程会卡在 TAG 阶段。

**建议修复**: 在 workflow TAG 阶段增加 fallback：若 tag 文件不存在，自动调用 `scripts/analysis_engine/auto_tag.py`（或内嵌简化版 code_tag 逻辑）。

---

#### 问题3：APPLY/EVALUATE/COMPARE 阶段的 `input()` EOFError
**现象**: Interactive 模式下，workflow 在 APPLY 阶段调用 `input()` 等待确认时，在非终端环境（如后台 agent、管道调用）中抛出 `EOFError`。

**状态**: CHANGELOG 2026-02-25 中已记录移除 SUGGEST 阶段的 `input()`，但 APPLY 阶段仍存在此问题。

**建议修复**: 统一处理所有 `input()` 调用，增加 `try/except EOFError` 后自动选择默认行为（如：auto 模式默认 yes，interactive 模式输出提示后继续）。

---

#### 问题4：BUILD 阶段路径模板错误
**现象**: `assets/configs/paths.yaml` 的 `build_output` 路径模板生成 `{OpName}Custom` 子目录路径，与实际 `output/{op_name}/build_out/` 不匹配，导致 BUILD 阶段找不到 `.run` 文件。

**具体错误**: 期望路径 `output/aten__fused_adamw_/aten__fused_adamw_Custom/build_out/`，实际路径 `output/aten__fused_adamw_/build_out/`。

**建议修复**: 更新 `paths.yaml` 的 `build_output` 模板，移除 `{OpName}Custom` 层级，直接指向 `output/{op_name}/build_out/`。

---

#### 问题5：Profiling CSV 路径检测逻辑不匹配
**现象**: `ProfilingExtractor` 期望 profiling CSV 位于 `workspace/inputs/{op}/profiling/` 下，而 `init_workspace.py` 将其放在 `profiling/op_summary.csv`，两者命名约定一致但子目录检测逻辑有歧义，产生警告。

**影响**: 不影响主流程，但产生噪音警告，降低可读性。

**建议修复**: 统一文档和代码中的路径约定，消除歧义警告。

---

### Optimization Result (2026-02-26) - aten__fused_adamw_ 优化结果

**算子**: `aten__fused_adamw_`（FusedAdamW，shape 1024×4096）
**路由路径**: Moderate Path（max_score: 0.375）

**优化前后对比**:

| 指标 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| Task Duration | 309.45 us | 82.52 us | -73.3% |
| Block Dim | 8 | 20 | +2.5x 核数 |
| Speedup vs PyTorch | 0.65x | **2.48x** | +3.8x |
| aiv_vec_ratio | 0.536 | 0.848 | 向量化显著提升 |

**应用的优化规则**（按优先级）:
1. **R_TILING_CORE_LOAD_BALANCE** (score 0.364): Block Dim 8→20，Former/Tail 负载均衡
2. **R_PIPE_DOUBLE_BUFFER** (score 0.375): 所有 Queue depth 1→2，隐藏 DMA 延迟
3. **R_MEM_UB_FUSION** (score 0.320): TBuf 14个→7个，降低 UB 压力
4. **R_API_VECTOR_COUNTER_MODE** (score 0.320): 模板化消除 amsgrad/maximize 运行时标量分支

**正确性**: PASS，match_rate 100.00%（4,194,304/4,194,304）

