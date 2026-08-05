---
name: cannbotdsl-msprof-compare
description: 在 Ascend NPU 上跑 msprof 性能对比并解析 op_summary CSV。当用户要求"用 msprof 对比两个 kernel""profile 一下哪个 kernel 更快""分析 aic_mac_time / Task Duration""为什么 PA 比 dense FA 慢""比较 cube/vec 利用率"，或者只是说"上板 profile 一下"+ 两个具体 kernel/test 时，触发本 skill。也用于回答"msprof 怎么用 / 输出的 CSV 怎么看"。重点是：每个被测 kernel 必须重复执行多次（msprof 的 PipeUtilization 在单次短任务上抓不到 pipe 计数器），并按 Task Duration 取最小值——min 才是 kernel 的固有成本，mean 会被 HBM cache state 拉偏。注意：本 skill 假设两个 test 已存在、shape 已对齐；要构造 matched shape 或 Channel depth-N / vec / cube 流水设计时走对应的 op-design/channel/pipeline skill，本 skill 不掺合。
---

# msprof-compare：Ascend NPU 上的两 kernel 性能对比

## 用之前先看清楚

这个 skill 专做一件事：把两个（或多个）已有的 CANNBotDSL kernel 在 NPU 上跑 msprof，把 `op_summary_*.csv` 解析出来，给出可读的对比。

它**不**做：
- 修改 kernel 实现（去对应的 `cannbotdsl-flash-attention` / `cannbotdsl-channel` / `cube-pipeline` / `vec-pipeline` skill）
- 设计 matched shape——本 skill 假设被测 test 文件里 shape 已经对齐
- 登录 NPU + 配 CANN 环境（去 `on-board-debugging` skill；如果当前会话还没登录上 NPU，**先调用它**）

## 工作流（四步）

### 1. 让 kernel 重复执行 N 次

msprof 的 `PipeUtilization` 计数器是**采样**的：单次很短（<100us）的 kernel 任务上，大量行会回 `aic_mac_time=0.0`、其它 pipe 也是 0——你拿不到任何 pipe 利用率。

解决：在 test 函数里把 kernel 调用包成循环。建议加一个 `FA_PERF_REPEAT` 环境变量开关，default=1（不影响功能 test），profile 时设成 10-20。

```python
import os
repeat = int(os.environ.get("FA_PERF_REPEAT", "1"))
for _ in range(repeat):
    fa_kernel.run(...)
torch.npu.synchronize()
```

只 patch 一次、profile 时再开。**两边都要 patch**，否则单边的 Task Duration 会因为只跑 1 次而抓不到 pipe 计数器。

### 2. 跑 msprof（每个变体一个独立目录）

每个被测变体一个空目录 cd 进去再跑，PROF_* 会落在 cwd：

```bash
mkdir -p /tmp/perf_runs/<variant>
cd /tmp/perf_runs/<variant>
rm -rf PROF_*    # 清掉上一次的，否则 op_summary 会拼接旧数据
FA_PERF_REPEAT=10 msprof \
    --aic-metrics=PipeUtilization \
    --task-time=on \
    --ai-core=on \
    python -m pytest -s <test_file>::<test_name> -k "<test_id_filter>"
```

关键 flag：
- `--aic-metrics=PipeUtilization`：拿 mac / mte1 / mte2 / scalar / fixpipe / vec 的 active time + ratio。`ArithmeticUtilization`、`MemoryUB` 是另几个 group，需要时再换。
- `--task-time=on`：拿每个 task 的 Task Duration（必备，永远开）。
- `--ai-core=on`：默认就开，写出来防止以后被改默认。

跑完后产物：
```
/tmp/perf_runs/<variant>/PROF_<id>/mindstudio_profiler_output/op_summary_<timestamp>.csv
```

### 3. 解析 + 对比

解析 op_summary CSV，提取每个 kernel 的 Task Duration（min/mean/max）和 pipe metrics。可直接传 PROF 目录或 perf_runs 目录，找最新的 op_summary CSV。

核心逻辑：
1. **Task Duration**：n / min / mean / max（us），以及第二行开始相对第一行的倍数（`1.99x` 就是慢 2×）
2. **Pipe metrics**：取 mac_ratio 最高的那一行（采样幸运的那次），列出 aicore / mac / mte2 / mte1 / fix / vec 的 active time + ratio

### 4. 解读

按下面的顺序读：

**a. 第一行（baseline）的 Task Duration min** 是 baseline kernel 的"理论下限"。如果 mac_ratio>0.9，说明这个 kernel 是 compute-bound 的，下限基本就是 cube 算完所需的时间。

**b. 各变体的 min Task Duration 相对 baseline** 就是慢出来的"结构性"开销。

**c. Pipe ratio 告诉你瓶颈** 是哪个 pipe：
- `mac_ratio < 0.5` 且 `mte2_ratio > 0.8` → MTE2 等数（HBM bandwidth / 间接寻址）
- `mac_ratio > 0.9` → cube 已经打满，盖不住了
- `scalar_ratio > 0.3` → 标量 GM load 过多（vLLM 的 block_table 间接寻址典型症状）
- `fix_ratio` 也高且 `mac_ratio` 中等 → fixpipe drain 跟不上 mmad

**d. 不要直接读 mean**：mean 会被 HBM L2 cache state 在多次调用之间漂移污染（同一变体某几次跑得快、某几次慢）。**取 min** 是 kernel 固有成本最稳的估计。

## 为什么 min 而不是 mean

实测：dense FA 在 S=4096 shape 上重复跑 10 次，第 1-2 次都是 761us，第 3-10 次跳到 1495us 不动。mean=1349us 偏离最佳态 ~75%。同一份 kernel、同一份输入 tensor，差异完全来自 HBM L2 cache 行的留存状态。

类似的，PA identity 的 10 次跑里前 7 次 ~1515us、后 3 次掉到 ~1023us。如果用 mean 比 PA random 的 1515us，会误以为 random 反而更快。

**结论：报数据时报 min，并在备注里说明跑了 N 次取 min**。

## 常用 CSV 列速查（CANN 9.0 op_summary）

1-indexed：

| 列号 | 字段 | 用途 |
|---|---|---|
| 5 | Op Name | 匹配 kernel（`_Z31kernel_xxx`） |
| 8 | Task Type | 一般是 `MIX_AIC` |
| 10 | **Task Duration(us)** | 拿 min 比较 |
| 22 | aicore_time(us) | 整体 cube+sync 时长 |
| 24-25 | aic_mac_time(us) / ratio | cube 实际算的 active 时间 / 占比 |
| 26-27 | aic_scalar_time / ratio | scalar pipe（GM 标量 load 多则高） |
| 28-29 | aic_mte1_time / ratio | L1→L0 搬运 |
| 30-31 | **aic_mte2_time / ratio** | GM→L1/UB 搬运（最常见瓶颈） |
| 34-35 | aic_fixpipe_time / ratio | L0C→UB/GM drain |
| 37 | aiv_time(us) | vec 整体时长 |
| 39-40 | aiv_vec_time / ratio | vec active 时间 / 占比 |

如果以后 CANN 改 schema，先 `head -1 op_summary*.csv | tr , '\n' | nl` 重新对一下列号。

## 远端执行模板

NPU 通常是远端。配合 `on-board-debugging` 的 `run_remote.py`：

```bash
SSH_PWD='...' python3 .claude/skills/on-board-debugging/scripts/run_remote.py \
    --source-dir <remote_workdir> \
    -- 'mkdir -p /tmp/perf_runs/<v1> /tmp/perf_runs/<v2>
        cd /tmp/perf_runs/<v1> && rm -rf PROF_* && FA_PERF_REPEAT=10 msprof \
            --aic-metrics=PipeUtilization --task-time=on --ai-core=on \
            python -m pytest -s <test_path> -k "<v1_filter>" 2>&1 | tail -3
        cd /tmp/perf_runs/<v2> && rm -rf PROF_* && FA_PERF_REPEAT=10 msprof \
            --aic-metrics=PipeUtilization --task-time=on --ai-core=on \
            python -m pytest -s <test_path> -k "<v2_filter>" 2>&1 | tail -3'
```

`msprof` 把 PROF_* 输出到 cwd，所以两个变体务必 cd 到各自目录、并先 `rm -rf PROF_*` 清旧数据，否则 op_summary 会拼接旧的、解析会拿到错的 min。

CSV 解析可以在远端跑（直接传远端路径），也可以把 CSV scp 回本地再跑。

## 跑完别忘了清理

```bash
rm -rf /tmp/perf_runs
```

每轮 profile 100MB 级，留着没用。

## 常见坑

- **单次跑就比较**：很大概率 pipe 计数器全是 0，只能看 Task Duration / aicore_time，看不到瓶颈分布。提示用户加 `FA_PERF_REPEAT=10` 重跑。
- **PROF_\* 目录没清**：op_summary 会包含上一次的行，min 计算会污染。每次跑前 `rm -rf PROF_*`。
- **比较时没对齐 shape**：dense FA 默认跑 (4,32,32,256,1024,128)、PA 默认跑 (1,32,8,1024,128)——shape 不一样的对比毫无意义。先确认两个 test 跑的是同一个 shape；如果没有，回到上层让用户加 SHAPE_LIST 入口。
- **mean 拿来报性能差距**：见上面 "min vs mean"。
- **`-k` 写错**：pytest parametrize id 形如 `test_xxx[1-32-32-1024-128-random]`，filter 要写完整的 id 片段。先 `pytest --collect-only -k "..."` 确认。
