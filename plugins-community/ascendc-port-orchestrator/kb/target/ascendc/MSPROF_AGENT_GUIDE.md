# msprof Agent 使用指南

> 如何让 AI Agent 高效使用 msprof 进行性能分析，避免撑爆 context。
> 本文档持续更新。

---

## 0. msprof vs msopprof：选哪个工具（系统级 vs 算子级）

> Owner 2026-06-24 澄清；main 对照官方 CANN/MindStudio/Huawei 文档核实（msprof=系统级 / msopprof=算子级）。两者对象不同、功能略有重叠：

- **`msprof` = 系统级 profiling（最常用）** —— 查看整个执行的系统级时间线/瓶颈（host-device 调度、算子间隙、内存搬运、AI Core 全局利用率）。**默认先用它**找系统级瓶颈。
- **`msopprof` = 算子（operator）级 profiling** —— 对单个算子内部展开指令/pipe 级细节。`msopprof` 是 CANN 包里的可执行文件、接口用法与 **`msprof op`** 一致（同一算子级能力的两种调用名）。
- **工作流**：先 `msprof` 找系统级瓶颈 → **若瓶颈定位到某个算子**、再用 `msopprof` 对那个单算子展开看内部。
- 本指南下文的 context-安全分层读取策略主要针对 `msprof` 系统级输出；`msopprof` 单算子展开同理（只读聚合 csv、不读二进制 trace）。

---

## 1. 核心问题

msprof 原始输出可达 **数 GB**（二进制 trace 数据），但 agent context 窗口有限（~200K tokens 有效空间）。
直接读取原始数据不可行，必须有过滤策略。

## 2. msprof 输出结构

```
/tmp/msprof_out/
└── PROF_000001_xxxxxx/
    ├── device_0/data/          # 二进制 trace（数十 MB~数 GB）← NEVER READ
    │   ├── stars_soc_profile.data.0.slice_*
    │   └── ffts_profile.data.0.slice_*
    ├── device_0/sample.json    # 小 JSON，设备信息
    └── mindstudio_profiler_output/  # ← AGENT 只读这个目录
        ├── op_statistic_*.csv  # 聚合统计（2-5 行）← 首选
        ├── op_summary_*.csv    # 逐 task 详情（可达数千行）← grep 过滤
        ├── task_time_*.csv     # timeline（数千行）← 通常不需要
        └── api_statistic_*.csv # ACL API 统计（十几行）
```

## 3. Agent 分层读取策略

### Level 1: `op_statistic_*.csv`（首选，< 10 行）

**格式**: `OP Type, Core Type, Count, Total Time(us), Min/Avg/Max Time(us), Ratio(%)`

**用途**: 快速判断哪个 kernel 是瓶颈（按 Ratio% 排序）。

```bash
# Agent 命令：只读聚合统计
cat PROF_*/mindstudio_profiler_output/op_statistic_*.csv
```

**示例输出（2 行）**:
```
pooling_backward_kernel,AI_VECTOR_CORE,793,1291324us,5.8us,1628us,65302us,86.9%
pooling_forward_kernel,AI_VECTOR_CORE,793,194107us,7.5us,244us,10309us,13.1%
```
→ 一眼看出 backward 占 87% 时间。

### Level 2: `op_summary_*.csv` + grep/awk（按需，< 50 行）

**格式**: 逐 task，包含 `aiv_vec_ratio`, `aiv_scalar_ratio`, `aiv_mte2_ratio` 等关键指标。

**关键列**（按列号，从 0 开始）:
| 列 | 字段 | 含义 |
|----|------|------|
| 4 | Op Name | kernel 名 |
| 9 | Task Duration(us) | 执行时间 |
| 37 | aiv_vec_ratio | vector pipe 利用率（0~1.0） |
| 39 | aiv_scalar_ratio | scalar pipe 利用率 |
| 41 | aiv_mte2_ratio | DMA pipe 利用率 |

```bash
# Agent 命令：只提取特定 kernel 的关键指标
# 取 backward kernel 的 top-5 最慢 task
grep "backward" PROF_*/mindstudio_profiler_output/op_summary_*.csv | \
  awk -F',' '{print $10, $38, $40, $42}' | sort -rn | head -5
```

```bash
# Agent 命令：取所有 kernel 的平均 vec_ratio 和 scalar_ratio
grep -v "^Device" PROF_*/mindstudio_profiler_output/op_summary_*.csv | \
  awk -F',' '{
    name=$5; dur=$10; vec=$38; scl=$40; mte=$42;
    sum[name]+=dur; cnt[name]++; svec[name]+=vec; sscl[name]+=scl; smte[name]+=mte
  } END {
    for(n in sum) printf "%s: avg_dur=%.1fus vec=%.3f scl=%.3f mte=%.3f (n=%d)\n",
      n, sum[n]/cnt[n], svec[n]/cnt[n], sscl[n]/cnt[n], smte[n]/cnt[n], cnt[n]
  }'
```

### Level 3: `task_time_*.csv`（按需，需要 `--task-time=l2`）

Timeline 数据。文件可达数千行但单个 CSV（非二进制，可用 grep/awk 处理）。

**用途**: 分析 kernel 间 gap、stream 并发、逐 task 精确 timing。
**大小**: 通常 50K-100K（数百~千行 CSV），agent 可安全读取 header + wc -l。

```bash
# 只看行数和 header（不读全部内容）
head -1 PROF_*/mindstudio_profiler_output/task_time_*.csv
wc -l PROF_*/mindstudio_profiler_output/task_time_*.csv
```

**注意**: 需要 `--task-time=l2` 参数才会生成详细数据。默认 `--task-time=on` 只有 l0/l1 粒度。

### Level 4: `device_0/data/*`

二进制 trace 文件，数十 MB~数 GB。只能用 MindStudio GUI 或 msprof CLI 解析。**Agent 绝不直接读取。**

## 4. 不同 metrics group 的使用

msprof 的 `--aiv-metrics` 参数控制采集内容，不同 group 不能同时采集：

| Group | 关键指标 | 用途 |
|-------|---------|------|
| **PipeUtilization**（默认） | `aiv_vec_ratio`, `aiv_scalar_ratio`, `aiv_mte2_ratio`, `aiv_mte3_ratio` | **首选**：判断瓶颈在计算、标量、DMA 读还是 DMA 写 |
| Memory | HBM 带宽利用率 | 判断是否带宽瓶颈 |
| **L2Cache** | L2 读/写命中率、miss 率、eviction 计数 | **判断 L2 缓存效果**（Batch 14 验证有效） |
| **MemoryUB** | UB 读写带宽 | 判断 UB 利用率 |
| ArithmeticUtilization | MAC 利用率 | 矩阵运算密集型 kernel |
| ResourceConflictRatio | 资源冲突 | 排查 bank conflict |

**推荐流程**:
1. 先用默认 PipeUtilization 跑一次
2. 如果 vec_ratio ≈ 1.0 且性能仍差 → 跑 Memory 检查带宽
3. 如果 scalar_ratio 高 → 间接寻址/控制流瓶颈（GetValue GM 标量读）
4. 如果 mte3_ratio 高 → SetAtomicAdd 写瓶颈（Batch 14-6 确认）
5. **如果需要区分"读缓存 vs 写缓存"** → 跑 L2Cache（Batch 14-6 关键手段）

**L2Cache metrics 提取**（Batch 14-6 验证的方法）:
```bash
# 找 L2 相关列名
head -1 op_summary_*.csv | tr ',' '\n' | cat -n | grep -i 'l2\|cache'

# 提取每个 kernel 的 L2 命中率（列号需根据实际 header 调整）
grep -v '^Device' op_summary_*.csv | awk -F',' '{
  name=$5; l2_rd_hit=$XX; l2_wr_hit=$YY; l2_wr_miss=$ZZ;
  ...
}'
```

**op_summary 列号对照（PipeUtilization metrics）**:
| 列号（1-based） | 字段 | 说明 |
|----------------|------|------|
| 5 | Op Name | kernel 名 |
| 10 | Task Duration(us) | 执行时间 |
| 37 | aiv_time(us) | AIV 总时间 |
| 39 | aiv_vec_time(us) | VEC 时间 |
| **40** | **aiv_vec_ratio** | VEC 利用率 |
| 41 | aiv_scalar_time(us) | 标量时间 |
| **42** | **aiv_scalar_ratio** | 标量利用率 |
| 43 | aiv_mte2_time(us) | MTE2 读时间 |
| **44** | **aiv_mte2_ratio** | MTE2 读利用率 |
| 45 | aiv_mte3_time(us) | MTE3 写时间 |
| **46** | **aiv_mte3_ratio** | MTE3 写利用率 |

## 5. Agent 完整 workflow

```
Step 1: 采集
  msprof --output=/tmp/msprof_X -- ./benchmark_command

Step 2: 快速定位瓶颈（< 10 行输出）
  cat PROF_*/mindstudio_profiler_output/op_statistic_*.csv

Step 3: 深入分析（grep 过滤后 < 50 行）
  grep "kernel_name" op_summary_*.csv | awk 提取 vec/scalar/mte ratio

Step 4: 判断 → 实施优化 → 重新 profile
```

## 6. msprof 运行命令模板

```bash
# 基础采集（默认 PipeUtilization）
MSPROF=/usr/local/Ascend/cann-9.0.T501/tools/profiler/bin/msprof
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.T501/x86_64-linux/lib64:$LD_LIBRARY_PATH
$MSPROF --output=/tmp/msprof_out -- ./benchmark_command

# 指定 metrics group
$MSPROF --output=/tmp/msprof_mem --aiv-metrics=Memory -- ./benchmark_command

# 导出为 CSV（如果默认未生成）
$MSPROF --export=on --output=/tmp/msprof_out --type=text --summary-format=csv
```

## 7. 已知限制（A5 容器环境）

- msprof 在 `can_torch_cann_device_1` 容器内可能报 "Running profiling failed. Please check the driver package"
  - 原因：容器权限不足或 driver 版本不匹配
  - Workaround：之前的 session 成功跑过（可能需要重启容器或检查 driver 挂载）
- msprof 的 `--analyze` 和 `--parse` 模式需要 Python 3.7.5+

## 8. 使用 aog-researcher agent 做深度分析

**推荐对复杂优化问题使用 aog-researcher agent**:

当需要跑多轮 msprof（不同 metrics group）并交叉分析时，交给 researcher agent 可以：
1. 保护主 agent 的 context 不被 profiling 数据污染
2. Researcher agent 可以连续跑 3-4 轮 msprof 并综合分析
3. 返回结构化的诊断报告和假设建议

**Batch 14-6 验证的深度分析 workflow**:
```
Step 1: PipeUtilization 跑基线 kernel + 待优化 kernel
Step 2: L2Cache 跑待优化 kernel（看缓存命中）
Step 3: 对比两个 kernel 的 4 管线利用率差异
Step 4: 提取 L2 读/写 hit/miss 数据
Step 5: 综合诊断 → 输出结构化假设

关键发现模板:
  "Kernel X 的 mte3_ratio=0.88 证明瓶颈在 MTE3 原子写。
   L2 写 miss=8298 说明原子写导致 L2 thrashing。
   对比 Kernel Y 的 mte3_ratio=0.35 和 L2 写 miss=0。
   结论: X 的原子写模式在 A5 上比 Y 的直接写慢 N 倍。"
```

## 9. msprof 运行命令模板（更新 2026-04-02）

```bash
# 基础采集（PipeUtilization — 首选）
MSPROF=/usr/local/Ascend/cann-9.0.0/tools/profiler/bin/msprof
export LD_LIBRARY_PATH=/root/a5_ops/build/lib:/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH
$MSPROF --output=/tmp/msprof_out --application='./benchmark' --aic-metrics=PipeUtilization

# L2 Cache 分析
$MSPROF --output=/tmp/msprof_l2 --application='./benchmark' --aic-metrics=L2Cache

# Memory 带宽分析
$MSPROF --output=/tmp/msprof_mem --application='./benchmark' --aic-metrics=Memory

# UB 利用率
$MSPROF --output=/tmp/msprof_ub --application='./benchmark' --aic-metrics=MemoryUB

# 详细 task timing（L2 级）
$MSPROF --output=/tmp/msprof_detail --application='./benchmark' --aic-metrics=PipeUtilization --task-time=l2
```

**注意**: 使用 `--application=` 而不是 `-- command`（后者在某些 CANN 版本不工作）。LD_LIBRARY_PATH 必须包含 kernel .so 路径。
3. 真正大的是二进制 trace（永远不读），CSV 最多几千行（grep 后很小）

**如果未来需要**（比如 profiling 上千个 kernel 的大模型训练）：
- 可用 haiku sub-agent 做 CSV 过滤 + 聚合
- 输入：op_summary CSV + 查询条件（"找最慢的 5 个 kernel 的 vec_ratio"）
- 输出：< 20 行摘要
- 但对当前项目（2 个算子，< 10 个 kernel），主 agent 直接 grep 足够

## 9. 瓶颈判断速查表

| 现象 | 指标 | 含义 | 优化方向 |
|------|------|------|---------|
| vec_ratio ≈ 1.0, 性能差 | PipeUtilization | 计算已满，瓶颈在算法 | 减少计算量（如排序消除 atomicAdd） |
| scalar_ratio > 0.2 | PipeUtilization | 间接寻址/控制流开销大 | 常驻核心分发（P-P22）、减少分支 |
| mte2_ratio > 0.5 | PipeUtilization | DMA 搬运占主导 | 数据复用（UB 缓存）、减少搬运次数 |
| HBM bandwidth < 10% | Memory | 带宽未饱和 | 不是带宽瓶颈，看其他指标 |
| HBM bandwidth > 80% | Memory | 带宽饱和 | 减少数据量、提高计算密度 |

## 10. 项目中的实际使用记录

| 场景 | 采集命令 | 关键发现 | 指导的优化 |
|------|---------|---------|-----------|
| Pooling A baseline | `--aiv-metrics=PipeUtilization` | bwd vec_ratio=1.0, atomicAdd 15.9 cycles | sorted-edge 寄存器累加 (-81%) |
| Pooling D BRE=dim | 同上 | fwd vec_ratio=1.0（已到极限） | 确认 fwd 无进一步优化空间 |
| Pooling nblk sweep | nblk=56 vs 448 | bwd -28%, fwd +14% | 排序后不需要超订 |
| SG forward xlarge | 同上 | vec_ratio=0.69, scalar_ratio=0.31 | 常驻核心分发 (E8-2, 1.86x) |
| SG backward xlarge | 同上 | vec_ratio=0.989 | compute-bound, persistent 无效 |
