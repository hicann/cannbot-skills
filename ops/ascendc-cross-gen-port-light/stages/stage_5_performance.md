# 阶段 5：性能验证（含 API 差异回归）

**前置条件**：`[GATE-4] PRECISION=N/N_PASS` 已输出。

本阶段使用 `torch_npu.profiler` 采集迁移后算子在 A5 (950) 上的性能数据，生成性能评估报告。如果用户提供了 A2/A3 基线数据，则做跨平台对比。

**★ 本阶段 MUST 包含 API 差异高风险 shape 回归用例**（见 `api-diff-guide.md` §3），验证 vnchwconv/VSLDB/BilinearInterpolation 性能退化是否影响算子。

## LOADED 检查点

| 序号 | MUST READ 文件 | LOADED Token |
|------|---------------|-------------|
| 1 | `references/impl/api-diff-guide.md` | `[LOADED] api-diff-guide` |

**★ 若阶段 1 扫描发现算子命中性能退化路径，MUST 加载 `api-diff-guide` 并设计回归用例。**

## Step 5.0：全量仓性能优化参考（按需）

如需查阅性能优化技巧或参考同类算子的性能调优案例，按 `references/search-rules.md` 路由到全量仓：

- **性能优化指南**：`$DEVKIT_PATH/docs/zh/guide/operator_practice/simd_operator_optimization/` 查找访存优化、流水线优化、计算优化等
- **优秀实践案例**：`$DEVKIT_PATH/docs/zh/guide/operator_practice/best_practices/` 查找 FlashAttention、GroupedMatmul 等标杆实现
- **性能分析方法**：`$DEVKIT_PATH/docs/zh/guide/operator_practice/performance_analysis/` 查找 profiler 使用方法

读取全量仓文件后输出：`[LOADED] $DEVKIT_PATH/<相对路径>`

## Step 5.1：准备性能测试用例

**复用阶段 4 精度测试的用例设计**（shape + dtype + 边界值）。

### 用例筛选规则

从 `analysis_<op_name>.md` 和阶段 4 的测试用例中，按下表选取 **7-11 个代表性 shape**：

| 类别 | 数量 | 选取标准 |
|------|------|---------|
| 小规模 | 1-2 个 | launch 开销主导的典型 shape |
| 中规模 | 2-3 个 | 典型生产场景 shape |
| 大规模 | 1-2 个 | 带宽/算力主导的 shape |
| 边界 | 1 个 | 极端 shape 或 dtype |
| **★ API 差异高风险** | **2-3 个** | **按 `api-diff-guide.md` §3 触发 shape 设计** |

每种选取的 shape 遍历主要 dtype（FP16/BF16/FP32），**用例总数 10-18 个**（shape 数与 dtype 数的乘积）。

### ★ API 差异高风险 shape 选取（按算子类别）

根据阶段 1 扫描结果和 `api-diff-guide.md` §3 的影响排序，选取对应高风险 shape：

**命中 TransDataTo5HD 退化（P0）**：

| 算子类别 | 高风险 shape | 来源 |
|---------|------------|------|
| 纯转置（TransposeV2） | 大张量 + 大量小 tile | `api-diff-guide.md` §3.2.6 |
| Pad/Unfold 类 | 小 W、非对齐 W、边界占比大 | `api-diff-guide.md` §3.2.6 |
| GridSample | nearest/bilinear；C=8/16/24/32/64；大输出空间 | `api-diff-guide.md` §3.2.6 |
| small-R Softmax | R=8/16/24/32/64 | `api-diff-guide.md` §3.2.6 |
| ND↔NZ MatMul | 小矩阵、窄矩阵、多 group、尾块比例高 | `api-diff-guide.md` §3.2.6 |

**命中 ReduceSum VSLDB 退化（P0）**：

| 算子类别 | 高风险 shape | 来源 |
|---------|------------|------|
| 纯求和归约（ReduceSum） | `[A,16]`、`[A,24]`，A 从几十到数万 | `api-diff-guide.md` §3.3.8 |
| 对照组 | R=8、32、64、128（确认退化仅集中在 VSLDB 分支） | `api-diff-guide.md` §3.3.8 |
| LogSoftmaxGrad | 最后轴 16/24，AR recompute tiling | `api-diff-guide.md` §3.3.8 |
| SoftmaxV2 | 最后轴 16/24 | `api-diff-guide.md` §3.3.8 |
| CrossEntropyLoss | C=16/24，大 batch | `api-diff-guide.md` §3.3.8 |
| SquareSum/L2Loss/Dot/MSELoss | 局部归约轴 16/24 | `api-diff-guide.md` §3.3.8 |

**命中 BilinearInterpolation 退化（P1）**：

| 算子类别 | 高风险 shape | 来源 |
|---------|------------|------|
| GridSample | Hout×Wout 大、C 小、随机 grid、NCHW 非连续输出 | `api-diff-guide.md` §3.4.7 |
| ROIAlign | ROI 多、pooledH/W 大、samplingRatio 大、C 小 | `api-diff-guide.md` §3.4.7 |
| Deformable Sampling | query/head/level/point 多、offset 随机、headDim 小 | `api-diff-guide.md` §3.4.7 |
| 通用参数 | hRepeat=1/2/4/8；vRepeat=1/2/4/8/16/32；dstBlkStride=1/2/4/8 | `api-diff-guide.md` §3.4.7 |

### 生成性能测试脚本

基于阶段 4 的精度测试脚本改造，核心区别：

| 维度 | 精度测试（stage_4） | 性能测试（stage_5） |
|------|-------------------|-------------------|
| 用例数 | >= 30 | 10-18（7-11 代表 shape × 主要 dtype） |
| 运行次数 | 1 次 | warmup + active（多次） |
| 采集方式 | 无 profiler | `torch_npu.profiler` |
| 对比 | CPU 参考实现 | 仅 NPU 端，可选 A2/A3 基线 |

## Step 5.2：采集配置与执行

> **通用 profiler 机制**（`warmup/active` 固定约定、`*_ascend_pt` 命名、`op_statistic.csv` 定位、归一化口径）若有仓内技能 `ops/torch-ops-profiler` 可用，以其为参照；不可用时按本节配置执行——**本节已给出迁移场景所需的完整取值与目录约定，可独立执行**。（同插件的 `aog-perf-eval` 面向 op-gen 工程工作流，本阶段不适用。）

### 采集配置（本场景固定取值）

```python
import torch
import torch_npu

# 最大数据量采集配置
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=torch_npu.profiler.ExportType.Text,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    l2_cache=True,
    msprof_tx=False,
    op_attr=False,
    data_simplification=False,
    record_op_args=False,
    gc_detect_threshold=None
)
```

**参数说明**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `profiler_level` | `Level1` | 采集 CANN 层 + NPU 硬件层数据（必须） |
| `aic_metrics` | `PipeUtilization` | 流水线利用率（VEC/MTE/Scalar 占比） |
| `l2_cache` | `True` | L2 Cache 命中率 |
| `export_type` | `Text` | 输出 CSV + JSON 格式 |

**Schedule（固定）**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `wait` | 0 | 无等待 |
| `warmup` | 5 | 预热 5 步（消除 DVFS 影响） |
| `active` | 5 | 正式采集 5 步 |
| `repeat` | 1 | 不重复 |

**固定 warmup=5, active=5，不可修改。**

### 路径基准

**MUST 使用阶段 0 记录的 `${OP_DIR}`（算子工程根目录）作为绝对路径基准**，确保 Python 代码与 shell find 命令操作同一路径。

```python
import os
OP_DIR = "${OP_DIR}"  # 从阶段 0 环境记录获取，如 /home/user/code/ops-nn_reg
PROFILER_ROOT = os.path.join(OP_DIR, "perf", "op_name")  # profiler 输出根目录
```

Shell 中对应：
```bash
PROFILER_ROOT="${OP_DIR}/perf/op_name"
```

### 采集流程

```python
for case_idx, case in enumerate(perf_cases):
    inputs = build_inputs(case, device)

    # 每个 case 的输出路径（绝对路径）
    case_dir = os.path.join(PROFILER_ROOT, f"case_{case_idx:03d}")

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=5, active=5, repeat=1
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(case_dir),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        experimental_config=experimental_config
    ) as prof:
        for step in range(10):  # warmup(5) + active(5)
            output = torch.ops.npu.op_name(*inputs)
            prof.step()

    torch.npu.synchronize()
    print(f"[INFO] case {case_idx} done -> {case_dir}", flush=True)
```

**关键**：每个 case 使用独立的 `with profile` 上下文，生成独立的 `*_ascend_pt` 目录。

### 输出目录结构

```
${OP_DIR}/perf/<op_name>/
├── case_000/
│   └── *_ascend_pt/                          # 前缀是时间戳，后缀 _ascend_pt 固定
│       └── ASCEND_PROFILER_OUTPUT/
│           ├── op_statistic.csv              # 算子耗时统计
│           ├── kernel_details.csv            # Kernel 详细信息
│           ├── operator_details.csv          # 算子详细信息
│           ├── api_statistic.csv             # API 调用统计（Level1）
│           ├── l2_cache.csv                 # L2 Cache 命中率（l2_cache=True）
│           └── trace_view.json              # Chrome Trace
├── case_001/
│   └── ...
└── ...
```

### 查找 CSV 文件

profiler 输出的 `*_ascend_pt` 目录前缀是时间戳（不可预测），后缀 `_ascend_pt` 是官方固定模式。利用这一点精确查找：

```bash
# 查找某个 case 下的 CSV 文件
CASE_DIR="${PROFILER_ROOT}/case_000"

# 核心指标文件
find ${CASE_DIR}/*_ascend_pt -name "op_statistic.csv"
find ${CASE_DIR}/*_ascend_pt -name "kernel_details.csv"

# 硬件指标文件（部分可能在子目录中，递归查找）
find ${CASE_DIR}/*_ascend_pt -name "*pipe_utilization*.csv"
find ${CASE_DIR}/*_ascend_pt -name "*l2_cache*.csv"
```

**注意**：
- `*_ascend_pt` 是 glob 通配符，匹配任意前缀的 `_ascend_pt` 目录
- `pipe_utilization` 和 `l2_cache` 文件名可能带时间戳后缀，用 `*pipe_utilization*` 匹配
- 搜索范围限定在 `case_NNN/*_ascend_pt/` 内，不会误读其他 case 的数据

## Step 5.3：解析性能数据

### CSV 解析优先级

| 优先级 | CSV 文件 | 关键指标 | 用途 |
|--------|---------|---------|------|
| 1 | `op_statistic.csv` | Total Time(us)、调用次数 | 目标算子耗时（**须先按名归因，禁止全表求和**） |
| 2 | `kernel_details.csv` | Kernel 执行时间、Block Dim | Kernel 级耗时 |
| 3 | `pipe_utilization.csv` | VEC/MTE2/MTE3/Scalar ratio | 瓶颈类型判定 |
| 4 | `l2_cache.csv` | total_hit_rate | Cache 命中率 |
| 5 | `operator_details.csv` | 各算子耗时明细 | 子算子拆解 |

### 核心指标提取

从 `op_statistic.csv` 提取：

| 指标 | 来源 | 说明 |
|------|------|------|
| **耗时(us)** | `op_statistic.csv` | **目标算子行**的 Total Time(us)：先按算子名精确筛选并校验调用数 = active 次，再除以 active(5)。**禁止全表求和**（原因见下方口径说明） |
| **Block Dim** | `kernel_details.csv` | 运行核数 |

从 `pipe_utilization.csv` 提取（最大 shape 的那个 case）：

| 指标 | 字段 | 含义 |
|------|------|------|
| VEC ratio | `aiv_vec_ratio` | Vector 计算占比（>50% = VEC Bound） |
| MTE2 ratio | `ai*_mte2_ratio` | GM 搬入占比（>50% = MTE2 Bound） |
| MTE3 ratio | `ai*_mte3_ratio` | GM 搬出占比 |
| Scalar ratio | `ai*_scalar_ratio` | 标量计算占比（>30% = Scalar Bound） |

从 `l2_cache.csv` 提取：

| 指标 | 字段 | 达标标准 |
|------|------|---------|
| 总命中率 | `total_hit_rate` | >80% 良好，<50% 需关注 |

### 耗时归一化与口径说明

```
per_step_us = 目标算子行 Total Time(us) / active_steps
其中 active_steps = 5（固定）
```

> **口径说明（与整轮求和口径的区别）**：本阶段测的是**目标算子自身**的耗时，故 MUST 先按算子名归因（见上表"禁止全表求和"）；仓内 `ops/torch-ops-profiler` 服务的是**整轮对比**场景（对 `op_statistic.csv` 全部行求和，对比自定义 vs 标杆），两者口径不同，不可混用。

## Step 5.4：生成性能评估报告

### 报告内容（MANDATORY）

MUST 生成 `perf_report_<op_name>.md` 并在对话中展示：

```markdown
# 性能评估报告：<op_name> (A5 ascend950)

## 采集配置
- 平台: Ascend 950PR (A5)
- 迁移层级: L1/L2
- 采集工具: torch_npu.profiler (Level1, PipeUtilization, L2Cache)
- Schedule: warmup=5, active=5

## 性能数据

| Case | Shape | DType | 耗时(us) | Block Dim | 主导流水 |
|------|-------|-------|----------------|-----------|---------|
| 0 | [128, 4096] | fp16 | 9.75 | 28 | VEC (62%) |
| 1 | [2, 1024, 4096] | fp16 | 57.01 | 28 | VEC (58%) |
| ... | ... | ... | ... | ... | ... |

## 瓶颈分析（最大 shape 用例）

| 指标 | 值 | 说明 |
|------|-----|------|
| 算子耗时（per-step） | 75.60 us | 目标算子行的 Total Time / active |
| Block Dim | 28 核 | 用满可用核数 |
| VEC ratio | 62% | VEC Bound |
| MTE2 ratio | 18% | 正常 |
| MTE3 ratio | 12% | 正常 |
| Scalar ratio | 8% | 正常 |
| L2 Cache 命中率 | 85% | 良好 |

## ★ API 差异性能回归分析

### TransDataTo5HD 退化分析（如命中）
| 高风险 Case | Shape | 耗时(us) | 对照 Case | 对照 Shape | 对照耗时(us) | 退化比例 | 结论 |
|------------|-------|---------|----------|-----------|------------|---------|------|
| ... | ... | ... | ... | ... | ... | ... | ... |

### ReduceSum VSLDB 退化分析（如命中）
| 高风险 Case | Shape (R=16/24) | 耗时(us) | 对照 Case | Shape (R=8/32/64) | 对照耗时(us) | 退化比例 | 结论 |
|------------|---------------|---------|----------|------------------|------------|---------|------|
| ... | ... | ... | ... | ... | ... | ... | ... |

### BilinearInterpolation 退化分析（如命中）
| 高风险 Case | 参数 | 耗时(us) | 对照 Case | 参数 | 对照耗时(us) | 退化比例 | 结论 |
|------------|------|---------|----------|------|------------|---------|------|
| ... | ... | ... | ... | ... | ... | ... | ... |

## 跨平台对比（如用户提供 A2/A3 基线）

| Case | Shape | DType | 950 耗时(us) | 910b 基线(us) | 比值(910b/950) |
|------|-------|-------|-------------|--------------|---------------|
| ... | ... | ... | ... | ... | ... |

## 结论
- ...（3 条以上）
- ★ API 差异退化结论（如命中）：是否检测到性能退化，退化比例，是否在可接受范围
```

### ★ API 差异退化分析方法

对于命中性能退化路径的算子，MUST 做以下分析：

**Amdahl 模型估算**（见 `api-diff-guide.md` §3.3.3）：

```
若 A2/A3 上退化指令占原算子时间比例为 f，A5 上该部分耗时变为两倍：
T_A5 / T_A2 ≈ 1 + f
吞吐比 ≈ 1 / (1 + f)
```

| 原 VSLDB 时间占比 | A5 预计耗时增加 | 吞吐下降 |
|---:|---:|---:|
| 10% | 10% | 约 9% |
| 20% | 20% | 约 17% |
| 30% | 30% | 约 23% |
| 50% | 50% | 约 33% |
| 100% | 100% | 50% |

**退化比例计算**：
- TransDataTo5HD：高风险 Case（小 tile / 非对齐 / 双向转换）耗时 vs 对照 Case（大 tile / 对齐 / 单向）耗时
- ReduceSum VSLDB：R=16/24 Case 耗时 vs R=8/32/64 Case 耗时（归一化后对比）
- BilinearInterpolation：大 vRepeat/hRepeat Case 耗时 vs 小 vRepeat/hRepeat Case 耗时

### A2/A3 基线对比

**基线数据来源**：用户需要在 A5 测试完成后提供。

**基线可选，缺省不阻塞**：在报告中说明基线要求（下述三项），若用户已提供则追加「跨平台对比」表；未提供则跳过对比，仅展示 A5 数据，不因缺基线阻塞 GATE-5。

```
如需跨平台对比，请提供 A2/A3 上同一算子的性能数据：
1. 使用与 A5 相同的测试用例（shape + dtype 一致）
2. 提供格式：CSV 或 Markdown 表格（含 Shape、DType、Kernel 耗时(us)）
3. 采集方式不限（msprof op 或 torch_npu.profiler 均可）
```

## Step 5.5：在对话中展示结果

**MANDATORY**——MUST 在对话中展示以下内容，禁止只输出文件路径：

1. **性能数据表**：所有 case 的 Shape、DType、Kernel 耗时、Block Dim、主导流水（无法归因的 case 标注 `N/A（无法归因）` 及原因）
2. **瓶颈分析表**：最大 shape 用例的流水线占比、Cache 命中率等
3. **★ API 差异性能回归分析表**：高风险 Case vs 对照 Case 的退化比例
4. **跨平台对比表**（如用户提供基线）
5. **3 条以上结论**（含 ★ API 差异退化结论）

## Gate 输出条件

**★ Evidence-based Gate**——`[GATE-5] PERF=COLLECTED` 仅当以下 10 项证据全部存在时才允许输出。缺少任一项时 MUST 输出 `[GATE-5] PERF=NOT_COLLECTED BLOCKED`。

| 序号 | 必要证据 | 验证方法 |
|------|---------|---------|
| 1 | profiler 执行记录 | 对话中有 `torch_npu.profiler.profile(...)` 代码执行输出 |
| 2 | profiler 输出目录 | `find ${OP_DIR}/perf/<op_name>/case_*/*_ascend_pt/` 返回非空结果 |
| 3 | CSV 数据文件 | 至少存在 `op_statistic.csv` 和 `kernel_details.csv` |
| 4 | `perf_report_<op_name>.md` | 文件存在于 `${OP_DIR}/perf/<op_name>/` 下 |
| 5 | 代表 shape ≥7 个（用例总数 10-18） | 报告中列出各 case 的 Shape + DType |
| 6 | API 差异高风险 shape 回归 case | 若阶段 1 扫描命中性能退化路径：报告含高风险 shape case；若未命中：报告明确标注"未命中性能退化路径，无需回归" |
| 7 | 性能数据表 | 报告含 Shape/DType/Kernel 耗时/Block Dim/主导流水 |
| 8 | 瓶颈分析表 | 报告含最大 shape 的 VEC/MTE2/MTE3/Scalar ratio + L2 Cache 命中率 |
| 9 | API 差异回归分析 | 报告含 TransDataTo5HD/ReduceSum VSLDB/BilinearInterpolation 退化分析（如命中）或标注"未命中" |
| 10 | 3 条以上结论 | 报告含结论部分，包括 ★ API 差异退化结论 |

**禁止行为**：
- 禁止仅依据 Todo completed / Agent 自我声明 / "阶段已完成"输出 `PERF=COLLECTED`
- 禁止在没有 profiler 采集的情况下输出 `PERF=COLLECTED`
- 禁止用精度测试数据替代性能测试数据
- 禁止跳过 profiler 配置（Level1 + PipeUtilization + L2Cache + warmup=5 + active=5）

**注意**：GATE-5 的 token 是 `PERF=COLLECTED` 而非 `PERF=PASS/FAIL`。本阶段只做性能评估和数据采集，不做达标判定。`PERF=COLLECTED` 验证的是"性能数据是否真实采集并完成分析"，而不是强制要求性能提升。如需性能优化，后续可独立使用性能优化流程。
