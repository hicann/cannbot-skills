# ascendc-sync-audit benchmark 最小集

随仓库提交的同步问题基准集，用于**双链回归门禁**：

- **sync_audit 链**：15 个缺陷样例（覆盖 14 条条例，SYNC-05 含拼写错误与跨函数 PIPE 两条检测路径）+ 6 个正确反例，均为精简 Ascend C 代码（`.asc`），标注见 `benchmark/labels.json`
- **analyzer 链**：`ascendc_flow_analyzer.py` 的 SYNC-14 图证据回归（4 个样例，预期 findings 数 2/1/1/0）

## 构成

### defects/ — 缺陷样例（15 个，覆盖 14 条条例）

| 条例 | 样例 | 典型问题 |
|------|------|---------|
| SYNC-01 | `bm_sync01_wait_before_set.asc` | Wait 先于 Set，同 flag 死等 |
| SYNC-02 | `bm_sync02_datacopy_then_adds.asc` | 搬入（MTE2）后未同步即计算 |
| SYNC-03 | `bm_sync03_single_core_residual_wait.asc` | 单核场景残留 CrossCoreWaitFlag |
| SYNC-04 | `bm_sync04_two_set_one_wait.asc` | Set/Wait 个数不一致（2 个 Set 1 个 Wait） |
| SYNC-05 | `bm_sync05_hardevent_typo.asc` | HardEvent 方向拼写错误 |
| SYNC-05 | `bm_sync05_conditional_pipe_wait.asc` | CrossCoreWaitFlag 跨函数 PIPE 方向不匹配（条件分支双 PIPE 场景） |
| SYNC-06 | `bm_sync06_pipebarrier_v_then_datacopy.asc` | 跨 PIPE 搬出误用 PipeBarrier\<PIPE_V\> |
| SYNC-07 | `bm_sync07_flagid_over_15.asc` | flagId 复用超过 15 次（静态展开 16 组，脚本可直接计数） |
| SYNC-08 | `bm_sync08_return_skips_setflag.asc` | 提前 return 跳过 SetFlag（真实修复样例） |
| SYNC-09 | `bm_sync09_pipe_all_narrowable.asc` | PipeBarrier\<PIPE_ALL\> 粒度过粗 |
| SYNC-10 | `bm_sync10_unsigned_underflow.asc` | (loop-1)%N 无符号下溢 |
| SYNC-11 | `bm_sync11_enque_deque_no_compute.asc` | EnQue→DeQue 后无计算（TQue 当 TBuf 用） |
| SYNC-12 | `bm_sync12_syncall_asymmetric.asc` | SyncAll 位于条件/循环块内 |
| SYNC-13 | `bm_sync13_atomic_ordering.asc` | SetAtomicAdd 后缺 PipeBarrier\<PIPE_FIX\> |
| SYNC-14 | `bm_sync14_output_bundle_index.asc` | 输出 buffer 写址索引与同步索引不同源 |

### correct/ — 正确反例（6 个，全部应产出 0 候选）

标准流水、双缓冲 ping-pong、循环内合法 Wait/Set 序、有符号下溢不报、真实算子代码（blas）——用于度量误报。

### 为什么需要 defects + correct 两组

检测是二分类问题（有缺陷/无缺陷），评测需要两类样本互补，缺一类则度量不完整：

- **仅 defects**：可度量漏检（缺陷未检出），但无法度量误报——即使把全部正确代码标记为候选，
  defects 指标也不会受影响；
- **仅 correct**：可度量误报（正确代码被标记为候选），但无法度量漏检——脚本零检出时
  correct 指标仍可通过。

两组组合覆盖两个正交维度：**漏检（defect 侧）与误报（correct 侧）**。`labels.json` 中
`label: defect/correct` 标记每文件类别——defect 贡献缺陷检出通过率与条例命中，correct 贡献文件级/红线误报率。

## 运行

```bash
cd ops/ascendc-sync-audit
python3 benchmark/eval_benchmark.py          # L0 门禁（默认，双链回归）
python3 benchmark/eval_benchmark.py --full   # L0 门禁 + L1 全量分级（消费 tests/ 语料）
```

输出分级：

- **L0 门禁（默认）**：sync_audit 链的缺陷检出通过（条例+行号命中）与误报指标 + analyzer 链 4 例 PASS/FAIL
- **L1 全量（`--full`）**：标注集（dataset/real_dataset，expected_sync + defect_lines 双重命中）+ 真实 PR 集（real_pr/real_pairs，缺陷检出通过 + 行号命中）

单文件验证（与 SKILL.md 维护自检同源）：

```bash
python3 scripts/sync_audit.py benchmark/defects/bm_sync01_wait_before_set.asc --format json
```

### 修改脚本后的验证流程

| 你修改了哪个脚本 | 如何验证 | 关注点 |
|-----------------|---------|--------|
| `scripts/sync_audit.py` | `python3 benchmark/eval_benchmark.py`，看 sync_audit 链 | 缺陷检出通过率 15/15、误报 0% 不回退；任意条例从命中变未命中即回归 |
| `scripts/sync05_resolve.py` | 同上（被 sync_audit.py import，SYNC-05 两条路径都在基准内） | 重点看 `bm_sync05_*` 两个样例 |
| `scripts/case_retriever.py` | benchmark 保证不抛异常；另跑 `python3 data/build_db.py --verify` | 数字不变（它只写 evidence 文本），抽查一条候选的 detail 看证据 |
| `scripts/ascendc_flow_analyzer.py` | `python3 benchmark/eval_benchmark.py`，看 analyzer 链 | 4 例预期 findings 2/1/1/0，全部 PASS 即通过 |
| 动了 `labels.json` 或新增样例 | 跑 benchmark 确认指标不回退 | 新增样例按「新增样例指引」执行 |

退出码：`0` = 全部通过（双链全部通过）；`1` = 存在漏检 / correct 误报 / analyzer 链 FAIL。

## 评测判定与指标公式

### 统计参数定义（公式中的字母均指下表参数）

| 参数 | 定义 | 来源 |
|---|---|---|
| `findings` | sync_audit.py 对单个文件输出的候选列表（`--format json` 的 findings 数组） | 脚本输出 |
| `n` | 单文件候选数 = `len(findings)` | 由 findings 派生 |
| `codes` | 单文件命中的条例集合（如 `{SYNC-04}`），即 `findings[].code` | 由 findings 派生 |
| `lines` | 单文件候选所在行号集合（如 `{23}`），即 `findings[].line` | 由 findings 派生 |
| `redlines` | 单文件中 severity="红线" 的候选数 | 由 findings 派生 |
| `expected_sync` | 标注（labels.json）期望该 defect 命中的条例，如 `SYNC-04` | labels.json |
| `defect_lines` | 标注的缺陷行号，如 `[23, 24]`（缺陷锚点，供行号命中比对）。benchmark 最小集为人工按实际报告行校准；tests 语料由 extract_change_lines.py 从配对/PR diff 提取 | benchmark/labels.json（人工）；tests/labels.json（extract_change_lines.py） |
| `defect_passed` | 通过三层判定的 defect 样例数（检出数） | 评测器统计 |
| `defect_missed` | 未通过三层判定的 defect 样例数（漏检，与 defect_passed 成对） | 评测器统计 |
| `correct_passed` | 零输出（未误报）的 correct 样例数 | 评测器统计 |
| `correct_missed` | 产生过任意候选的 correct 样例数（误报，与 correct_passed 成对） | 评测器统计 |
| `correct_redline_missed` | 产生过红线候选的 correct 样例数（红线误报，correct_missed 的子集） | 评测器统计 |

### defect 判定（三层口径）

一个 defect 样例要**计入检出数**（`defect_passed` 计数 +1），必须同时满足三个条件
（correct 样例除外，只要求零输出）：

```python
hit = res["n"] > 0                    # 条件1: 脚本产生了候选
      and expected_sync in codes      # 条件2: 候选条例 = 标注的 SYNC-0x（精确匹配）
      and lines ∩ defect_lines 非空   # 条件3: 候选行号落在标注的缺陷行上（交集非空）
```

以 `bm_sync04_two_set_one_wait.asc`（期望 SYNC-04，标注行 [23, 24]）为例：脚本实际检出
`code=SYNC-04, line=23` → `n=1, codes={SYNC-04}, lines={23}` → 三条件全中 → 计入检出数。
其余组合均为 +0：

| 有候选 | 条例匹配（SYNC-04） | 行号命中（23∈[23,24]） | 结果 |
|---|---|---|---|
| ✓ | ✓ | ✓ | **+1 通过** |
| ✓ | ✗（报成 SYNC-03） | ✗/✓ | +0 条例不匹配 |
| ✓ | ✓ | ✗（打在 100 行） | +0 行号未命中 |
| ✗ 无候选 | — | — | +0 漏检 |
| ✓ | ✓ | ⚠ 只命中 23 之一（如 [23, 99]） | **+1**（交集非空即可，多报不影响） |

> 条件 3 是"候选行号与标注行号交集非空"，而非全部命中：多报冗余行不扣分，
> 但完全打在错误位置不算命中。

### correct 判定（误报）

一个 correct 样例只要产生**任意级别候选**即记为误报一次（计入文件级误报率）；
其中候选含**红线级别**的再计入红线误报率：

```python
correct_missed         += 1 if n > 0 else 0        # 误报：有候选（含性能/信息级）即算
correct_redline_missed += 1 if redlines > 0 else 0 # 红线误报：候选含"红线"级别
```

**为什么要单独统计红线误报？** correct 是实际正确（修复后）的代码，但脚本的启发式规则存在
盲区（如跨函数/跨文件同步链无法解析），可能将正确代码误判为红线（死锁级）候选。例如
tests 语料 `real_pairs/correct_pr_2409_2.cpp`（修复后代码）被报 5 条红线：

```
SYNC-02 L154 疑似计算后未同步即搬出 GM: 搬出@154 (计算@153)
SYNC-02 L193 变量 repeatTensor 跨流水缺同步: MTE2(MTE2_write)@L181 → S(S_read)@L1
```

两个误报指标度量不同属性：

- **文件级误报率**（`correct_missed`）= 任意级别候选的误报占比（含性能/信息级建议，影响小）；
- **红线误报率**（`correct_redline_missed`）= 红线级别候选的误报占比（会被当作严重缺陷处理，
  排查成本高）。

真实数据对比（L1 real_pairs sync 子集 49 个 correct）：文件级误报率 79.6%（39/49），
红线误报率 36.7%（18/49）——红线级误报占文件级误报的 46%（18/39），即近半数误报为
会被当作严重缺陷处理的红线候选。L0 门禁要求红线误报率 0%（benchmark 的 6 个 correct
当前全零），红线误报仅出现在真实复杂度语料（real_pr/real_pairs）上。

### 输出指标公式

L0（benchmark 最小集）与 L1（tests 全量）共用**同一公式、同一口径**，只是分母不同；
统一叫"缺陷检出通过率"（通过三层判定的 defect 占比）：

| 指标 | 公式 | L0 达标目标 |
|---|---|---|
| 缺陷检出通过率 | `defect_passed / (defect_passed + defect_missed)` | 15/15 = 100% |
| 文件级误报率 | `correct_missed / (correct_passed + correct_missed)` | 0/6 = 0% |
| 红线误报率 | `correct_redline_missed / (correct_passed + correct_missed)` | 0/6 = 0% |
| analyzer 链通过 | 实际 findings 数 == 预期（2/1/1/0） | 4/4 PASS |

L1（`--full`）无固定门限，目标为不回退（各次评测输出即基线）；L0 门禁随仓库提交、是 CI 强约束。

## 与 tests/ 全量语料的关系

- **L0 门禁（benchmark/）**：最小集随仓库提交，reviewer 可直接复现条例级 + 行号级检出（14 条例全覆盖）
- **L1 全量（tests/）**：571 文件全量语料（dataset / real_dataset / real_pr / real_pairs）+ eval_recall.py
  + labels.json（含 defect_lines 标注）。tests/ 目录是独立仓
  [ascendc-sync-audit-tests](https://gitcode.com/XieQianyi/ascendc-sync-audit-tests) 的 git submodule，
  `git submodule update --init --recursive` 拉取后由 `--full` 直接消费；未拉取时自动跳过 L1（不影响 L0 门禁）
- 两者共用同一标注口径（label + expected_sync + defect_lines）与同一评测逻辑

## 新增样例指引

- 缺陷样例：精简 Ascend C 代码（`kernel_operator.h` 风格），每文件一个典型问题，第 2-4 行标注
  `// LABEL: defect`、`// SYNC: SYNC-xx`（与 dataset 语料同风格）
- 正确反例：合法同步写法的精简代码，确保脚本预期产出 0 候选
- 加入后更新 `labels.json` 与上方覆盖表，并跑 `eval_benchmark.py` 确认指标
- analyzer 链样例在 `references/examples/` 下的 sync14 四例，改动 analyzer 后重跑 `eval_benchmark.py` 看 analyzer 块