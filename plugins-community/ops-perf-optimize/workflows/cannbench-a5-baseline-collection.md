# CANN-Bench A5 基线采集与跑分工作流

本文档说明 CANN-Bench 在 **Ascend A5（Ascend950DT_9582）** 上如何采集官方性能基线、如何跑分对比。

---

## 1. 基线是什么

| 概念 | 说明 |
|------|------|
| **Baseline（基线）** | CANN 生态在该硬件上的 **参考实现性能**，用于算加速比和 SOL 分 |
| **Golden（精度参考）** | `tasks/levelN/<op>/golden.py`，CPU/NPU 数值对比，与 baseline 无关 |
| **t_hw_us** | 硬件理论下界（带宽/算力模型），用于 SOL% 计算 |
| **baseline_perf_us** | 参考实现在 A5 上实测的 device kernel 耗时（μs） |

基线 **不是** 用户自研 kernel 的性能，而是「CANN 官方路径跑同一套 case 有多快」。

---

## 2. 采集方案总览

```
tasks/levelN/<op>/cases.yaml
        ↓
scripts/baseline/inputs.py          ← 按 case 生成输入（与评测一致）
        ↓
scripts/baseline/refs/level{N}.py  ← NPU 参考实现（ref_fn）
        ↓
PerfEvaluator + msprof             ← 复用评测框架采 kernel 耗时
        ↓
tasks/metadata/Ascend950DT_9582.json
```

**核心脚本**：`cann-bench_A5/scripts/collect_baseline.py`

**设计原则**：
- 复用评测体系的 `PerfEvaluator` + `KernelDetailsStrategy`（与正式跑分同一套计时逻辑）
- 复用 `refs/` 下的 NPU 参考算子，保证 baseline 代表 CANN 生态路径
- 每个 case **独立子进程**采集，避免 profiler session 状态污染
- 强制 `ASCEND_LAUNCH_MODE=ACL`，确保 msprof 产出完整 `kernel_details.csv`

---

## 3. 环境准备

```bash
source /path/to/cann-9.0.0/bin/setenv.bash
conda activate xxx          # 需 torch + torch_npu；Exp 等算子还需 torchair 依赖, 这块用户提供python环境

cd /path/to/cann-bench_A5
export PYTHONPATH=src
```

** 算子如果使用 torchair + torch.compile，需要额外依赖**：
```bash
pip install protobuf decorator scipy psutil absl-py
# torchair 由 torch_npu 内嵌，collect_baseline.py 会自动 alias 到 sys.modules
```

---

## 4. 采集命令

### 4.1 单算子

```bash
python scripts/collect_baseline.py \
  --op level1/exp \
  --device-id 0 \
  --warmup 5 \
  --repeat 20
```

### 4.2 按级别 / 全量

```bash
python scripts/collect_baseline.py --level 1
python scripts/collect_baseline.py --all
```

### 4.3 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--op` | — | 算子路径，如 `level1/exp`、`level2/softmax` |
| `--device-id` | 0 | NPU 卡号 |
| `--warmup` | 5 | 预热次数 |
| `--repeat` | 20 | 采集 repeat 次数 |
| `--cases` | 全部 | 逗号分隔 case id，如 `1,2,5` |
| `--skip-existing` | off | 跳过 metadata 中已有数据的 case |
| `--force-recollect` | off | 强制重采 |
| `--output` | 自动 | 输出 JSON 路径 |
| `--bench-root` | `tasks/` | 评测集根目录 |

---

## 5. 参考实现（refs）策略

refs 注册在 `scripts/baseline/ref_registry.py`，按 level 分文件：

| 算子类型 | 参考路径 | 典型 kernel 形态 |
|---------|---------|-----------------|
| **Exp** | `refs/level1.py` → `exp_ref` | **torchair + torch.compile**，GE 融合为 `AutomaticBufferFusionOp×1`（Mul/Add/Exp 合一） |
| **Softmax** | `refs/level2.py` → `softmax_ref` | `torch.nn.functional.softmax`（eager，SoftmaxV2 单 kernel） |
| **Level2+ 融合算子** | 各 level 文件 | 优先 CANN fused API / torch_npu 专用算子 |

---

## 6. 单 case 采集流程（内部）

对每个 case，`BaselineCollector.collect_one_case()` 执行：

1. 从 `cases.yaml` 读取 shape / dtype / attrs
2. `inputs.py` 生成 CPU 输入 → 搬至 NPU
3. 调用 `ref_fn(inputs, attrs)` 包装为 profiling 函数
4. `PerfEvaluator.run_profiled()` + msprof 采集
5. 从 `kernel_details.csv` 汇总 **device kernel 总耗时** → `baseline_perf_us`
6. 从已有 metadata 读取或保留 `t_hw_us`

**计时口径**：只统计 device kernel 时间（与候选算子跑分一致），不用 wall clock 替代。

---

## 7. 产出物

### 7.1 文件路径

| 文件 | 用途 |
|------|------|
| `scripts/baseline/output/Ascend950DT_9582_<op>.json` | 单算子采集中间结果 |
| `tasks/metadata/Ascend950DT_9582.json` | **正式基线库**（BaselineStore 加载） |
| `reports/` | profiler 归档（调试用） |

### 7.2 metadata JSON 结构

```json
{
  "_metadata": {
    "hardware": "Ascend950DT_9582",
    "source": "collect_baseline.py (PerfEvaluator + KernelDetailsStrategy + refs)",
    "warmup": 5,
    "repeat": 20
  },
  "level1": {
    "exp": {
      "1": { "baseline_perf_us": 11.54, "t_hw_us": 1.09 },
      "2": { "baseline_perf_us": 25.62, "t_hw_us": 8.74 }
    }
  }
}
```

硬件型号由 `DeviceManager.get_device_name()` 自动映射（如 `Ascend950DT_9582`）。

---

## 8. 跑分：候选算子 vs A5 基线

### 8.1 CANN-Bench 评测命令

```bash
cd /path/to/cann-bench_A5
source /path/to/cann-9.0.0/bin/setenv.bash
conda activate xxx  # 用户安装好依赖的环境

PYTHONPATH=src python -m kernel_eval.cli eval \
  --source-dir /path/to/candidate/exp \
  --task-dir tasks/level1/exp \
  --operator Exp \
  --device-id 0 \
  --processes-per-card 1 \
  --warmup 3 \
  --repeat 5 \
  --output reports/baseline_exp_a5_eval
```

**source-dir 要求**：
- 含 `build.sh` → 编译 `libexp_ops.so` + 打包 `cann_bench` wheel
- 含 `cann_bench/__init__.py` → 暴露 `exp(x, base, scale, shift)` 接口

### 8.2 评分公式

```
加速比 speedup     = baseline_perf_us / candidate_kernel_us
SOL% perf_score    = t_hw_us / candidate_kernel_us × 100%
综合得分           = 编译分(20) + 功能分(30) + 性能分(≤50)
```

### 8.3 报告字段

评测 JSON 报告（`reports/<op>_eval_*.json`）每个 case 含：

| 字段 | 含义 |
|------|------|
| `baseline_perf_us` | A5 基线耗时 |
| `elapsed_us` | 候选 kernel 耗时 |
| `speedup` | 加速比 |
| `t_hw_us` | 硬件理论下界 |
| `perf_score` | 归一化性能分（0~1） |

---

## 9. 常见问题

### Q1: Exp 基线采集 0/20 失败

**原因**：缺少 torchair 运行时依赖（protobuf、decorator、scipy 等）。  
**对比**：Softmax 用 eager API，无 torchair 依赖，通常一次成功。

### Q2: baseline 与 cases.csv 中 baseline_kernels 标注不一致

cases.csv 的 `baseline_kernels` 是 **设计预期**（如 `AutomaticBufferFusionOp×1`）；实际采集以 msprof 抓到的 kernel 为准。Exp 在部分 case 上可能仍是 `Muls + Add + Exp` 多 kernel（attrs 导致无法融合）。

### Q3: 直调 wall_time 与 CANN-Bench 分数对不上

直调 `run_all_cases.py` 的 wall_time 含 ACL 初始化（~5–7s/case），**不能**与 `baseline_perf_us` 比加速比。加速比必须以 CANN-Bench eval 报告的 `elapsed_us` 为准。

### Q4: 如何更新 A5 基线

```bash
python scripts/collect_baseline.py --op level1/exp --force-recollect
# 合并写入 tasks/metadata/Ascend950DT_9582.json
```

### Q5: 无 A5 metadata 时的 fallback

BaselineStore 会 fallback 到默认平台（如 `910b2`）的数据并打印 warning；正式 A5 跑分前应确保 `tasks/metadata/Ascend950DT_9582.json` 已采集。

---

## 10. 关键路径速查

| 用途 | 路径（cann-bench_A5 仓库内） |
|------|----------------------------|
| 基线采集脚本 | `scripts/collect_baseline.py` |
| 参考实现 | `scripts/baseline/refs/level{1-4}.py` |
| 输入生成 | `scripts/baseline/inputs.py` |
| 正式基线库 | `tasks/metadata/Ascend950DT_9582.json` |
| 评测用例 | `tasks/levelN/<op>/cases.yaml` |
| 性能评测核心 | `src/kernel_eval/eval/perf_eval.py` |
| 基线加载 | `src/kernel_eval/utils/baseline_store.py` |
