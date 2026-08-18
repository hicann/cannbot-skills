# Ascend C Operator Degradation Workflow

本文件是 `ops-knowledge-optimization-ingest` 的详细执行指南。执行实际劣化任务时先读 `SKILL.md`，确认需要完整流程后再读本文件。

## 输入参数
- `op_category` / `op_name`、`dataset_dir`（算子源码根，支持常见两级或三级目录布局）
- `threshold`：劣化达标阈值（默认 0.05，即至少慢 5%）
- `experience_lib`：经验库 JSON 文件路径（一个 JSON 数组，累积各算子已发现的经验，供 agent 观察去重）
- `work_root`：工作目录根（默认 `./logdir/naive_degradation`）
- `max_round`：劣化轮数（默认 3；每轮都要把三种模式全跑一遍，跑满才算完成）
- `--knowledge-root`：共享 OKF 知识库根（缺省=仓库根）；编译阶段把 `OPT-N` 卡写进 `<knowledge-root>/runbooks/operator-optimization/single-op-degradation.md`，并维护该库的 `index.md`/`graph/`/`log/`
- 评估在**本地 NPU** 上用 `msprof` 采集（见「本地评估（msprof）」）：需真实 Ascend NPU + CANN + `msprof`/`npu-smi`，先 `source` CANN `set_env.sh`

## 工作区隔离（必须先做）
```bash
SRC=$(python scripts/resolve_source.py \
  --dataset_dir <dataset_dir> --category <op_category> --op <op_name>)
WORK=<work_root>/<op_category>/<op_name>
rm -rf "$WORK" && mkdir -p "$WORK"
cp -r "$SRC" "$WORK/<op_name>"           # 工作副本：agent 只在此劣化
cp -r "$SRC" "$WORK/original_<op_name>"  # 原始基线备份，回退用
```

## 准备：基线评估 + 读经验库观察已发现机制
1. 评估工作副本，得基线耗时（本地 build + msprof 采集；`--op_dir` 指算子工作副本根）：
   ```bash
   python scripts/single_op_evaluate.py \
     --op_dir "$WORK/<op_name>" \
     --op_host "$WORK/<op_name>/op_host" --op_kernel "$WORK/<op_name>/op_kernel" \
     --op_name <op_name> --op_category <op_category> \
     --json_output "$WORK/baseline_eval.json"
   # 若算子工程的 build/run 入口非默认（build.sh / run.sh build|run / run.py），用
   #   --build-cmd "<编译命令>" --run-cmd "<运行命令>" 覆盖；--device N 指定卡；--quick 只测时间
   ```
2. 读取经验库 JSON，观察已发现的机制（供 step 1 去重）。去重只看文字性描述，剥掉 `code_diff`：
   ```bash
   jq 'map(del(.code_diff))' <experience_lib> 2>/dev/null || echo "[]"
   ```
   观察各条的 `canonical_family` / `root_mechanism` / `changed_files` / `changed_symbols` / `change_anchor` 等字段，以合适的方式判断哪些机制已被探索过。

## 主循环：共 `max_round` 轮 × 每轮三种模式

整个任务是 `max_round × 3` 个工作单元（轮 × 模式），不是做完一次 degrade 就结束。
不要把单次 degrade 达标当成任务完成，它只是下面进度表里的一格。

为防止漏跑，用一个进度表文件驱动，而不是凭记忆跑循环。

### 0. 开跑前：建进度表
在「准备」之后、进入任何模式之前，建 `$WORK/progress.json`，列出全部 `max_round × 3` 格（每格一个 `{round, mode}`，初始 `status:"todo"`）。例如 `max_round=3`：
```bash
python - <<'PY'
import json, os
work = os.environ["WORK"]; R = int(os.environ.get("MAX_ROUND", "3"))
cells = [{"round": n, "mode": m, "status": "todo"}
         for n in range(1, R+1)
         for m in ["degrade", "refactor", "structured_tiling_degrade"]]
json.dump({"cells": cells, "stopped": None}, open(f"{work}/progress.json", "w"), indent=2)
print(f"created {len(cells)} cells")
PY
```

### 驱动规则（每做完一件事都回到这里）
1. 找下一格：读 `progress.json`，取第一个 `status=="todo"` 的格 `{round N, mode}`。
2. 对该格执行 step 1→4（按其 `mode`）；完成后把该格 `status` 改为 `"done"`（达标）或 `"failed"`（未达标/回退），并写回 `progress.json`。
3. 若该格是某轮的 `degrade` 且达标，执行「推进基线」。
4. 回到第 1 步，取下一个 `todo`。如此直到没有 `todo`。
5. 提前终止只有一种合法情形：某轮 `degrade` 未达标（degrade 链到头）。此时把该轮剩余及之后所有轮的格标为 `"skipped"`，并设 `progress.stopped = "degrade_exhausted_at_round_N"`。其余任何情况都不得提前停。

### 收尾检查点（写任何总结/结论之前必须做）
在输出最终结论前，先 `cat $WORK/progress.json`：
- 只要还存在 `status=="todo"` 的格，禁止收尾，回到「驱动规则」第 1 步继续。
- 仅当没有 `todo`（全部 `done`/`failed`/`skipped`）才允许写结论；若有 `skipped`，结论里说明 `progress.stopped` 的原因。

- 三种模式（A `degrade` / B `refactor` / C `structured_tiling_degrade`，定义见「三种探索模式」）每轮都要全跑，各自在独立工作副本里并行执行；三者都走下面同一套 step 1→4，只有 step 1 的探索任务因模式而异；step 2–4（评估 / 判定 / 记录）通用。
- 只有 `degrade` 成功才推进下一轮基线；`refactor` / `structured_tiling_degrade` 只产经验、不改基线。

每个模式先各自建一份工作副本，后续 step 命令里的 `$WORK/<op_name>` 对该模式即指其副本：
```bash
mkdir -p "$WORK/_attempt_<mode>"
cp -r "$WORK/<op_name>" "$WORK/_attempt_<mode>/<op_name>"
```

推进基线（仅当本轮 `degrade` 达标）：把 degrade 副本覆盖回主工作副本，并把它的评估结果设为下一轮基线：
```bash
cp -r "$WORK/_attempt_degrade/<op_name>/." "$WORK/<op_name>/"
cp "$WORK/round_<N>_degrade_eval.json" "$WORK/baseline_eval.json"
```

### 1. 执行该模式的 step 1 探索任务（agent 本体做）
按所选模式的任务（见「三种探索模式」A/B/C）编辑算子源码。三种模式通用约束：
- 本轮只做恰好一处改动；优先改 `op_host/` 或 `op_kernel/`。
- 保持编译所需接口与分支完整；可直接编辑工作副本源码。
- 不要自己跑 `run.sh` / `python run.py` / 任何评估，评估在 step 2 统一执行。

### 2. 评估劣化后版本
评估劣化后副本；`--json_output` 文件名带上 `<mode>`，避免三种模式互相覆盖。
输出 JSON 里的 `performance_metrics`（`latency` / `pipeline_stalls` / `aiv_utilization` / `memory_bw` / `cache_l2` 等整算子 profile；`msprof` 无 `critical_path` 等价，缺省不含）就是该版本的 profile，step 4 总结要拿它和基线对比：
```bash
python scripts/single_op_evaluate.py \
  --op_dir "$WORK/<op_name>" \
  --op_host "$WORK/<op_name>/op_host" --op_kernel "$WORK/<op_name>/op_kernel" \
  --op_name <op_name> --op_category <op_category> \
  --json_output "$WORK/round_<N>_<mode>_eval.json"
# 同 step 0：build/run 入口非默认时用 --build-cmd/--run-cmd 覆盖；默认采 3 组 aic-metrics，
# --full 采全 7 组（更慢），--quick 只测时间（performance_metrics 仅 latency）。
```
必须 `success=true`。失败则该机制无效，从 `original_<op_name>` 回退重试。

### 3. 判定劣化是否达标
`score`（= `task_duration_us`，越大越慢）满足 `round_<N>_<mode>_eval.json.score / baseline_eval.json.score >= 1 + threshold` 即达标。

未达标则回退，本模式本轮无收获；换新机制留到下一轮（degrade）或本轮其余模式。

### 4. 记录经验（必须结合前后两版 profile）
达标后产出一条经验记录。`root_mechanism` 与 `bottleneck_shift` 必须基于 step 2 两份评估 JSON 各自的 `performance_metrics` 对比得出（基线 = `baseline_eval.json`，劣化后 = `round_<N>_<mode>_eval.json`），而不是泛泛推测：对照 `latency` / `pipeline_stalls` / `aiv_utilization` / `memory_bw` / `cache_l2` 等指标，说明该机制把瓶颈推向了哪里，落到具体数字上。

```json
{
  "category": "<op_category>",
  "op": "<op_name>",
  "canonical_family": "<规范化机制族标签，dotted>",
  "raw_family": "<你起的简短点分标签>",
  "optimization_title": "<简短机制标题>",
  "root_mechanism": "<一句话根因，须与下面 profile 变化一致>",
  "why_not_duplicate": "<为何不同于已探索机制>",
  "changed_files": ["relative/path"],
  "changed_symbols": ["symbol"],
  "change_anchor": "<语义改动位置>",
  "duration_before": "<baseline_us>",
  "duration_after": "<new_us>",
  "profile_delta": "<前后最相关的几个指标对比>",
  "bottleneck_shift": "<结合 profile_delta，一句话说明瓶颈如何随该机制迁移>",
  "mechanism_understood": true,
  "confidence": 0.7,
  "experience_type": "performance_recovery",
  "code_diff": "<本轮改动的 unified diff，大块文本；放在末尾，去重时被剥掉，仅供下游 RAG>"
}
```

`profile_delta` 取两份 JSON `performance_metrics` 里与本机制最相关的几个指标做前后对比即可，不要把整块 `performance_metrics` 抄进经验记录。
`code_diff` 可由 `diff -ru "$WORK/original_<op_name>" "$WORK/<op_name>"` 生成；务必放在记录最后一个字段，其余字段都是短文字描述。把该记录追加进经验库 JSON 数组 `experience_lib`（文件不存在则新建为 `[ <record> ]`）。

本格到此结束不等于任务结束。把本格在 `progress.json` 标 `done`（未达标则 `failed`），然后回到「驱动规则」第 1 步取下一个 `todo`。

## 完成标准（针对整个主循环，不是单次模式）
- 跑满 `max_round` 轮；或 `degrade` 确实再也找不到达标的新机制而提前终止，这种提前结束必须在结论里说明原因。
- 每一轮三种模式都已尝试（各自 step 1→4 走完：达标的记经验、未达标的回退）。
- 每条经验记录字段合法、已追加进 `experience_lib`（JSON 数组）；`degrade` 达标轮的记录含前后 profile 对比（`profile_delta` / `bottleneck_shift`）。
- 重新读取 `experience_lib` 能看到本次运行新增的所有经验（通常多条：多轮 × 多模式）。

## 三种探索模式

每轮三种模式全跑，各自在隔离工作副本里独立改一处、独立评估、独立记经验（可并行）。
三者都遵循「主循环」的 step 1→4，只有 step 1 的探索任务不同，step 2–4 共用。

| 模式 | step 1 任务 | 是否读经验库去重 | 是否推进劣化基线 | 作用 |
|---|---|---|---|---|
| `degrade` | 推断新劣化机制 | 读，强制不重复 | 是 | 渐进劣化链条本身 |
| `refactor` | 做等价重构 | 不读（仅看盲点经验） | 否（仅产经验） | 发现「自以为无害实则影响性能」的盲点 |
| `structured_tiling_degrade` | 按 tiling 策略做探针 | 不读常规已发现机制 | 否（仅产经验） | 验证抽取出的静态 tiling 分区策略 |

三种模式都把经验追加进 `experience_lib`，但只有 `degrade` 成功才会把劣化副本提升为下一轮起点；`refactor` 和 `structured_tiling_degrade` 是探针，只贡献经验语料。

### 模式 A：degrade（渐进劣化，核心）
step 1：
- 目标：找一个新的劣化机制，使算子比当前版本慢至少 `threshold`。
- 从代码结构、数据流、缓冲、编译期特化、分支、局部性、访存、张量级访问结构等角度推断；不要复用经验库 JSON 中已记录的机制（换位置放同一机制也算重复）。
- 若算子动态处理多张量（foreach 类），可考虑破坏张量边界感知（把 buffer 绑定移入 per-tensor 循环、让数据分布忽略张量边界等）。

step 1 的结构化自评即 step 4 追加进 `experience_lib` 的那条记录。

### 模式 B：refactor（重构盲点探针）
step 1：
- 目标：做恰好一处功能等价、且你认为不会影响性能的结构性重构。
- 不参考经验库里常规已发现的机制（刻意不看，避免被引导）；仅参考其中 `mechanism_understood=false` 的盲点经验作为提示。
- 合法重构：buffer 分配移进/移出循环；向量化换等价标量循环；调换互不依赖的拷贝顺序；内联/抽取 helper；编译期常量改运行期变量；在语义不变前提下重排数据布局字段。
- 不要自己跑评估。评估后若意外达到劣化阈值，说明是性能盲点，追加进 `experience_lib`；若性能基本不变则本探针无收获，不记录。

产出 JSON：
```json
{
  "raw_family": "<重构类型的简短点分标签>",
  "title": "<改了什么>",
  "root_mechanism": "<为何认为这是等价变换>",
  "why_not_duplicate": "n/a",
  "changed_files": ["relative/path"],
  "changed_symbols": ["symbol"],
  "change_anchor": "<结构改动位置>",
  "confidence_in_prediction": 0.9,
  "context_assumptions": ["为何认为性能不变"],
  "risk_flags": ["等价性的不确定点"]
}
```
若意外达标而追加进 `experience_lib`，同样按 step 4 补上 `profile_delta` / `bottleneck_shift`。

### 模式 C：structured_tiling_degrade（tiling 策略验证探针）
step 1（两步）：先抽策略，再按策略做受控劣化。

#### C-1. 抽取静态 tiling 策略
只读检视 `$WORK/<op_name>`，写 `static_tiling_strategy.json`；不改源码、不跑评估。
必含字段：`strategy_type`、`source_op`、`partition_axis`、`block_dim_policy`、`tiling_fields`、`host_tiling_computation`、`kernel_block_mapping`、`tile_loop_policy`、`tail_policy`、`copy_or_compute_policy`、`correctness_invariants`、`applicable_targets`、`transfer_limitations`、`source_evidence`。
`source_evidence` 必须非空，每条 `{file, evidence}` 的 `file` 为算子目录下真实相对路径、`evidence` 为该文件真实片段。校验失败则跳过本探针。

#### C-2. 按策略做一处受控劣化探针
先读 `static_tiling_strategy.json`，选恰好一个具名组件做劣化（慢 ≥`threshold`）。
探针类型五选一，写入 `probe_type`：
`block_dim_probe` / `tile_granularity_probe` / `tail_policy_probe` / `block_mapping_probe` / `copy_granularity_probe`。

- 必须保持正确性（每个输出元素仍被正确写出）。
- 禁止无关减速：不加冗余循环 / 忙等 / 无关 barrier / 标量回退 / sleep / 调试 I/O。
- 探针必须可追溯到 `static_tiling_strategy.json` 里的字段。
- 不跑评估。

产出 JSON：
```json
{
  "raw_family": "structured.tiling.partition.<short_label>",
  "title": "<机制标题>",
  "root_mechanism": "<一句话根因>",
  "changed_files": ["relative/path"],
  "changed_symbols": ["symbol"],
  "change_anchor": "<执行分区改动位置>",
  "probe_type": "<五种探针之一>",
  "probed_strategy_component": "<static_tiling_strategy.json 中确切字段>",
  "expected_strategy_effect": "<扰动该组件为何影响性能>",
  "why_this_validates_strategy": "<为何这验证了策略而非局部减速>",
  "host_tiling_changes": ["..."],
  "tiling_data_changes": ["...或 n/a"],
  "kernel_partition_changes": ["..."],
  "correctness_invariants": ["覆盖/尾块/顺序不变量"]
}
```
达标并追加进 `experience_lib` 时，同样按 step 4 补上 `profile_delta` / `bottleneck_shift`。

## 编译进知识库（ingest 阶段 · 主循环收尾后执行）

挖矿主循环产出的是**原始层** `experience_lib`（含 `code_diff` 等富字段，是轨迹不是知识）。本阶段把本次运行**新增**的经验记录编译成 **curated 层**的 OKF 优化点卡，增量合并进共享知识库，并维护治理三件套。目标文件：

```
<knowledge-root>/runbooks/operator-optimization/single-op-degradation.md
```

卡骨架严格对齐包内 [`STRUCTURE-runbook.md`](../STRUCTURE-runbook.md)（扁平 `OPT-N`、算子无关占位名、坏实践字段必填）。

### 核心视角：劣化机制 = 优化点的「坏实践（反例）」

miner 找到的每条**达标劣化机制**，本质是"这样写会慢 X%"的实测反例；它的**逆**就是一条优化点原则。所以一条经验记录天然映射成一条 `OPT-N`：劣化前后的实测 profile 提供了 `OPT` 卡最缺的**实测收益与置信度**。

### 字段映射（experience record → OPT-N 卡）

| experience record 字段 | → OPT-N 卡字段 | 说明 |
|---|---|---|
| `optimization_title` / `root_mechanism` | `## OPT-N <标题>` + **原则** | 取逆：把"劣化根因"改写成"避免它 / 反向做"的优化原则 |
| `bottleneck_shift` + `profile_delta` | **摘要** / **触发** / **优化维度** | 据瓶颈迁移到哪个指标映射优化维度（见下） |
| 劣化本身（`duration_before`→`duration_after`、`code_diff` 摘要） | **坏实践（反例）** | "如此改动 → 慢 X%〔来源：round N `<mode>`〕；✅ 正解=保持/反向" |
| `duration_before` / `duration_after`（本地 msprof 实测） | **置信度**: 已验证(独立eval) | 本库统一实测置信度 |
| `why_not_duplicate` / `canonical_family` | **迁移条件** + 增量合并去重键 | `canonical_family` 相同视为同一 OPT |
| `changed_files` / `changed_symbols` / `change_anchor` | **已知实例** | 指向 `ops/<category>/<op>.md#锚点`；知识库无该算子卡时写「待补充」并在 `log/` 记 `experience_lib` 原始出处 |
| `code_diff`（大字段） | **不进卡** | 留原始层 `experience_lib`，符合「知识≠流水账」 |

### 优化维度映射（`profile_delta` 主导指标 → 维度）

| 主导变化的 profile 指标 | 优化维度 |
|---|---|
| `memory_bw` / `cache_l2` / `pipeline_stalls`(mte2·mte3) | 搬运 |
| `aiv_utilization` / `pipeline_stalls`(非 MTE) | 计算 |
| UB 占用 / buffer 复用 / 缓冲深度 | 内存 |
| dtype / cast 升降 / 数值稳定 | 精度 |

一条 OPT 可标一或多个维度。

### 增量合并（跨算子单一共享）

`single-op-degradation.md` 是**跨算子累积**的单一库：
- 已有等价 OPT（`canonical_family` 命中）→ **只在其「已知实例」append 反链**，不新增卡、ID 不动。
- 新机制 → 新增 `## OPT-N`（N = 现有最大编号 +1，**ID 不复用、不重排**）。
- 合并到**已存在**的 runbook 时，沿用 vv-ingest 阶段 5'b 纪律：**另起干净上下文子 Agent**（Agent 工具）执行去重合并，只喂它「现有 runbook 全文 + 本次候选 OPT（已按骨架、算子无关）+ `STRUCTURE-runbook.md`」，避免被挖矿上下文污染成误泛化。

### 治理三件套（写进 `--knowledge-root`）

1. **index.md**：补/追加 `<knowledge-root>/runbooks/operator-optimization/index.md`（`kind: index` / `type: section_index`，frontmatter 见 STRUCTURE）；已存在则追加指向 `single-op-degradation.md` 的条目，不覆盖。

2. **log/**：向 `<knowledge-root>/log/<YYYY-MM-DD>.md` **顶部**（倒序，最新在前）插入一条；文件不存在则首行写 `# <YYYY-MM-DD>`：
   ```markdown
   ## [HH:MM] optimization-ingest | <category>/<op> 劣化挖矿 N 条优化点

   ### Summary
   对 <category>/<op> 跑 max_round=<R> 劣化挖矿，新增 OPT k1..k2（N 条），增量合并进 single-op-degradation.md。

   ### Changes
   - updated: runbooks/operator-optimization/single-op-degradation.md（+OPT-k1..OPT-k2 / 或 append 已知实例）
   - updated: runbooks/operator-optimization/index.md

   ### References
   - source: <dataset_dir>/<category>/<op>（挖矿工作副本）
   - experience_lib: <experience_lib 路径>（原始轨迹，含 code_diff）
   - eval: 本地 msprof（device <id>，采集档位 default/full/quick）

   ### Details
   每条 OPT 的实测收益取自 round_<N>_<mode>_eval.json 与 baseline 的 profile 对比；code_diff 只入 experience_lib、不入卡。
   ```

3. **graph/**：新增/改动卡后重跑知识图谱（**只调用**共享脚本，不修改它）。从本 skill 目录执行：
   ```bash
   G=../ops-knowledge-ingest/scripts/okf_graph.py
   python3 "$G" --knowledge-root <kb> candidates          # 确定性召回 → .build/candidates.json
   # 对新卡焦点逐对判定 related/type/reason，写 graph/edge_judgments.json
   #   （判定键 = sorted([fp(a), fp(b)]) 用 | 连接，fp 来自 okf_graph.card_fp_map(load_nodes())）
   python3 "$G" --knowledge-root <kb> inject              # 写卡片 # 相关 托管块
   # inject 改了卡片内容 → 重算 card_fp 回写 edge_judgments.json，否则 verify 报 stale
   python3 "$G" --knowledge-root <kb> verify              # 必须 OK
   ```

### 编译后校验（必跑）

本 skill 自带的分层校验（覆盖 `OPT-N` 结构与字段坏实践必填、ops 卡对 `OPT-N`/`CT-N`/`AP-N` 引用均有定义、双向锚无悬空、runbook frontmatter profile）：

```bash
python3 scripts/validate_degradation_knowledge.py \
  --knowledge-root <kb> --ops <本轮涉及的算子卡...>
# 退出码非 0 → 按报告修正后重跑至 0
```

此外，`runbooks/` 变更须跑 `CONTRIBUTING.md`「合入门槛」的通用门禁（命令在插件根，相对本 skill 目录为 `../`）：

```bash
python3 ../knowledge-query/scripts/knowledge_query.py --knowledge-root <kb> verify
python3 ../ops-knowledge-ingest/scripts/okf_graph.py --knowledge-root <kb> verify
python3 ../knowledge-lint/scripts/knowledge_lint.py --knowledge-root <kb>
```

> **说明**：`CONTRIBUTING.md` 对 `ops/`/`runbooks/` 额外要求跑
> `../ops-knowledge-vv-ingest/scripts/validate_layered_knowledge.py`，但该脚本把 runbook 名
> **硬编码**为 `vv-fusion-common.md`，不覆盖本库的 `single-op-degradation.md`。故本库以
> 本 skill 自带的 `validate_degradation_knowledge.py` 承接同类分层校验；待该脚本支持
> `--runbook` 参数后再统一。

### 编译阶段完成标准
- 本次每条**达标**经验都已映射为 `OPT-N`（新增或 append 已知实例），`code_diff` 未进卡。
- `index.md` 已含 `single-op-degradation.md` 条目；当天 `log/` 有本次 `optimization-ingest` 条目。
- `knowledge_query.py verify` / `okf_graph.py verify` / `knowledge_lint.py` 均通过（`CONTRIBUTING.md` 合入门槛）。
- `okf_graph.py verify` 后新卡 `# 相关` 块 ≥1 链接、无死链。
- `validate_degradation_knowledge.py` 退出码 0。

> **准入对齐（`CONTRIBUTING.md`）**：本库属 R3「通用优化点」——每条 `OPT` 必含**适用条件 / 失效边界 / 坏实践**（`STRUCTURE-runbook.md` 已强制）；`runbooks/` 只写跨算子可复用规律，算子私有技巧不得误写成通用优化点（增量合并交由干净上下文子 Agent，见上）。撤除/纠错既有优化点时按勘误规则标 `retracted` 或转为负知识，不静默删除。证据来源 = `experience_lib` 原始记录 + 本地 msprof 实测 profile，已在 `log/` 记录，满足「可复现、可追溯」。

## 本地评估（msprof）

评估在**本机真实 NPU** 上完成，`single_op_evaluate.py` → `_vendor/eval_client.py` → `_vendor/msprof_collect.py`。链路：**build（算子工程自带）→ msprof 包裹 run → 解析产物 CSV**。前置：`source` CANN `set_env.sh`，`msprof` / `npu-smi` 在 PATH。

### build / run 入口探测（可覆盖）
build 与 run **必须可分离**（msprof 只包裹 run，否则编译时间会污染 us）。默认探测顺序：
- **build**：`--build-cmd` 覆盖 > `build.sh` > `run.sh build` > `cmake -B build && cmake --build build -j`。
- **run**：`--run-cmd` 覆盖 > `run.sh run` > `run.py`（`python3 run.py`）> 目录内可执行 `main`/`test`/`run`。

探测不匹配你的算子工程约定时，用 `--build-cmd "<...>"` / `--run-cmd "<...>"` 显式指定；已构建好可加 `--skip-build`。

### 采集档位
| 档位 | 参数 | msprof 轮数 | performance_metrics |
|---|---|---|---|
| 默认 | （无） | 1 组 task-time 隐含 + 3 组 aic-metrics（`PipeUtilization`/`Memory`/`L2Cache`） | latency + aiv_utilization + pipeline_stalls + memory_bw + cache_l2 |
| 全量 | `--full` | 7 组 aic-metrics（最全、最慢） | 同上（指标更全） |
| 快速 | `--quick` | 1 轮 task-time | 仅 latency |

### 指标来源（CSV 字段）
- `score`(= `latency.task_duration_us`)：标准档从 `op_summary_*.csv` 的 `Task Duration(us)` 累加计算行（`AI_CORE`/`AIV`/`MIX`）；`--quick` 从 `task_time_*.csv` 的 `task_time(us)` 累加。
- `aiv_utilization` ← `aiv_vec_ratio`；`pipeline_stalls` ← `aic_mte2_ratio`/`aic_mte3_ratio`；`memory_bw` ← Memory 组读写带宽；`cache_l2` ← L2Cache `hit/(hit+miss)`。
- `critical_path`：msprof 无等价指标，**不产出**；`profile_delta`/`bottleneck_shift` 用上面可得指标即可（step 4「取最相关几个」）。

### 选卡
`--device N` 指定；缺省由 `pick_idle_npu()` 解析 `npu-smi info` 选最闲卡，失败回退 0，并写入 `ASCEND_RT_VISIBLE_DEVICES`。

### 异常处理（各自 `success:false`）
| 现象 | 处理 |
|---|---|
| `msprof not found` | 未 source CANN `set_env.sh` / 不在 NPU 机；先修环境 |
| `build exited ...` | 编译失败：查 `--build-cmd`、算子工作副本改动是否破坏编译；不进 msprof |
| `msprof exited N` | run 失败或 NPU 不可用：查 `--run-cmd`、`--device`、卡占用 |
| `duration parse failed` | 产物无计算行 / CSV 字段异常：确认 run 真跑了 kernel、msprof 版本产物路径 |

任一 `success:false` 视同该机制无效：从 `original_<op_name>` 回退，按主循环重试或标 `failed`。
