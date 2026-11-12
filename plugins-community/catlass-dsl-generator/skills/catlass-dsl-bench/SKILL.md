---
name: catlass-dsl-bench
description: Use when independently checking and timing a CATLASS Python DSL solution against definition and workload files; this version provides the measurement contract and requires the agent to write and run its own benchmark script.
---

# CATLASS DSL 公共 Benchmark（门禁引导）

> **入口**：本阶段开始前先加载 `catlass-dsl-workflow` 并输出阶段卡。

> **入口门禁**：作为工作流阶段执行时，`python3 <workflow>/scripts/gate.py benchmark
> --workspace <ws>` 必须通过。它要求 optimize 已按模式 A/B 完成并落账，且当前 kernel 已有
> 全量测试 passed 记录：优化采纳改动了 kernel 时必须重跑 `full test #2`，全部未采纳、
> kernel 未变化时可复用 `full test #1`；只有 optimize 自测结果的不得开始工作流内的
> benchmark。独立测量任务不经过该门禁，直接按本 Skill 执行。
> 仅当 `DESIGN.md` 声明 `性能要求：requested` 时进入工作流内的 benchmark 阶段；声明 `none`
> 时不执行 benchmark 与 profiling，当前 kernel 全量测试通过后直接 finish。

> **自动执行**：遇到问题只要有可行方案就逐步尝试，一个方案失败就换下一个，不停下来找
> 用户确认。等待远端任务、编译、benchmark、设备或长跑用例时直接等待并轮询，不结束回合。

> **超时纪律**：编译、算子运行、benchmark、profiling 与普通 shell 命令必须拆成独立命令块；
> 编译独立 300s，单例运行 `timeout -k 5 60s`，批量分片，禁止整批套一次性超时，禁止裸跑。
> 超时或卡住立即终止，不得死等或加大超时。终止后先做卡健康判定：同一张卡的小探针或
> 官方算子能过 → 判本算子死锁并转 debug；探针也失败 → 判设备问题并换卡重试一次。

**本版本不提供 `bench.py`。** 执行测量时由 Agent 在工作区写一次性脚本（建议
`tools/bench_local.py`），按本文的输入接口、测量口径和结果字段实现，并把命令、退出码和
结果路径写入 `ITERATIONS.md`。脚本可以简化，但不得改变正确性判据、计时边界或数据口径。

## 输入接口

以 `solution.json`、`workload.jsonl`、`definition.json` 作为唯一公开输入。输入会执行
Python 代码，必须来自用户信任的来源；测量脚本不提供安全沙箱。

### definition.json

```json
{
  "name": "scale",
  "op_type": "elementwise",
  "axes": {},
  "inputs": {
    "x": {"shape": null, "dtype": null},
    "alpha": {"shape": null, "dtype": "float32"}
  },
  "outputs": {"output": {"shape": null, "dtype": "float32"}},
  "reference": "import torch\n\ndef run(x, alpha):\n    return x * alpha\n"
}
```

- `axes` 固定为 `{}`；workload 的 `axes` 固定为 `null`。禁止用 `const`、`var`、`expr`
  或 axis 名称间接表达 case shape。
- definition 中 `shape` 为 `null`，或只使用十进制常量字符串表达所有 case 共有的固定
  shape。每个 tensor case 的实际 shape/dtype 必须直接写入 workload。
- `dtype` 支持 `bool`、`uint8`、`int8/16/32/64`、`float16/32/64` 和 `bfloat16`。
- `reference` 必须是包含顶层 `run` 的完整 Torch 源码。
- `custom_inputs_entrypoint` 可指定 `(axes, device) -> dict` 输入生成函数；对应 workload
  的输入统一使用 `custom`，但 shape、dtype、layout 和开关仍须在 workload 行中显式保存。

### workload.jsonl

每个非空行是一个独立用例：

```json
{"uuid":"scale-001","axes":null,"inputs":{"x":{"type":"random","shape":[1024],"dtype":"float32"},"alpha":{"type":"scalar","value":0.5}},"tolerance":{"max_atol":0.00001,"max_rtol":0.00001,"required_matched_ratio":0.999,"max_error_cap":null,"allow_negative_inf":false}}
```

- `uuid` 文件内唯一，直接作为结果 case id；不得用 uuid 承担隐藏配置查表。
- 每个 tensor input 直接声明数值 `shape` 和 `dtype`；标量用
  `{"type":"scalar","value":...}`，可选空输入写 JSON `null`。
- 支持 `random`（可带 `range`）、`zeros`、`ones`、字面量 `scalar`、`null`、
  `custom`、`safetensors` 和变长 `tensor_list`。
- tolerance 逐 workload 生效：满足
  `abs(candidate-reference) <= max_atol + max_rtol * abs(reference)` 计为匹配；匹配比例
  不得低于 `required_matched_ratio`，`max_error_cap` 是额外硬上限。
- 容差只用于吸收完全对齐标杆实现后的浮点尾差；实现与标杆在 dtype、舍入位置/模式、
  累加 dtype 或归约顺序上不一致时先修实现，不得用容差通过。
- 浮点输出默认 `required_matched_ratio: 0.999`；`1.0` 只用于位精确、整数结果或数值设计
  明确要求。NaN/Inf 默认失败；`allow_negative_inf` 只允许两侧同位置的 `-inf`。
- 同一 suite 使用同一随机 seed，保证可复现。

### solution.json

```json
{
  "name": "scale_catlass_dsl",
  "definition": "scale",
  "author": "developer",
  "spec": {
    "languages": ["python"],
    "target_hardware": ["ascend910b"],
    "entry_point": "solution.py::run",
    "destination_passing_style": false
  },
  "sources": [
    {"path": "solution.py", "content": "def run(x, alpha):\n    return x * alpha\n"}
  ]
}
```

- `definition` 必须等于 `definition.name`；`languages` 使用 `["python"]`。
- `sources[].path` 必须是安全相对路径，`content` 包含完整源码。
- `entry_point` 固定为 `<source-path>::<callable>`。
- 返回值模式使用 `destination_passing_style: false`；DPS 模式为 `true`，入口参数依次为
  全部输入和输出 buffer，不返回结果。
- 所有输入按 definition 顺序作为位置参数传入，solution 与 reference 接受相同调用。
- solution 中每个 `@tla.kernel` 必须自包含，只能读取形式参数、函数内局部值、Python
  内建符号和受信任的 `tla` API 命名空间。

## 测量脚本要求

建议脚本按以下顺序实现，命令行参数至少包含
`--solution --workload --definition --output --device --seed --warmup --trials`：

1. **加载与校验**：解析三个输入，校验 `definition.name`、`solution.definition`、
   `entry_point`、`sources[].path` 和 workload uuid 唯一性。
2. **导入候选**：把 `sources[]` 复制到输出目录后导入，按 `entry_point` 取 callable；
   `source_sha256` 取主源码文件的 SHA-256。
3. **逐例正确性**：按 workload 生成输入（固定 seed），按 definition 的输入顺序展开为
   位置参数，分别调用 reference 和 candidate；用独立 Tensor 克隆，按 tolerance 比对，
   输出结构必须与 `definition.outputs` 顺序一致。
4. **同步 warmup**：每例 candidate 执行 1 次并同步，不计时。
5. **配对计时**：同一例先 candidate 后 reference，或奇偶例交替先后（`paired`）；使用
   `torch.npu.Event` 或等价设备事件记录单次 `run` 到同步完成的耗时，采集 `trials` 次
   （默认 2；需要更稳时取 5 次中位数）。输入生成和 clone 不计入算子时延。计时必须与
   profiling 分两次运行：计入 `mean_ms` 的运行关闭 profiler，profiling 单独跑一次，其
   wall 时间只用于 kernel 级归因，不写入性能结论。
6. **聚合**：逐例 `mean_ms`、`std_ms`，再计算全部用例的均值；speedup =
   `reference_mean / candidate_mean`。
7. **profiling**：NPU 上用 `torch_npu.profiler` 文本导出。采集必须显式开到 kernel 级，
   否则不会生成 `kernel_details.csv`，`op_summary` 也不带 AIC 指标：

   ```python
   cfg = torch_npu.profiler._ExperimentalConfig(
       profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
       aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
       export_type=[torch_npu.profiler.ExportType.Text],
       data_simplification=True,
   )
   ```

   `data_simplification=True` 已能生成带 AIC/AIV 指标的 `kernel_details.csv`；只有还需要
   `op_summary` 时才改为 `False`，后者会显著放大导出体积。完整采集骨架（`activities`、
   `schedule`、`trace_handler` 与产物路径）见
   `knowledge/profiler/torch-npu-profiler-dsl-kernels.md`。

   导出后先过两道断言，再落性能结论：

   - `kernel_details.csv` 存在，且目标 kernel 行的 task duration > 0；
   - 该时长与配对计时的量级一致（相差一个数量级以上即视为采集失败）。

   任一不满足都必须修配置重采，不得用 `op_summary` 的 task 级数字或 wall 差值代替
   kernel 级耗时。重采上限 2 次（先核对 `profiler_level` 与 `aic_metrics`，再检查 `activities` 与
   `schedule`）；仍拿不到
   kernel 级产物时，在 `ITERATIONS.md` 写明「profiling 不可用」及已试配置与报错，性能
   结论只用配对计时的 `mean_ms`，并注明缺少 kernel 级归因，不得继续重试。保留每例的
   `kernel_details.csv` 与 `step_trace_time.csv`，只用于归因：可对比 `step_trace_time.csv`
   的 `Computing` 与 kernel 总时长，判断 host/device 占比与串行空洞，不得作为 `mean_ms`
   的来源。
8. **落盘**：即使某例失败也写入结果，失败用例标记 `status=failed` 并附错误摘要；正确性
   失败的用例不产生有效性能结论。

参考骨架（按实际环境补全）：

```python
import json, statistics, time
from pathlib import Path
import torch

def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def make_inputs(spec, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = {}
    for name, item in spec.items():
        if item is None:
            values[name] = None
        elif item["type"] == "scalar":
            values[name] = item["value"]
        elif item["type"] == "random":
            tensor = torch.randn(item["shape"], generator=generator,
                                 dtype=getattr(torch, item["dtype"]))
            values[name] = tensor.to(device)
        else:
            raise ValueError("unsupported input type: {}".format(item["type"]))
    return values

def timed(run, args, trials):
    times = []
    for _ in range(trials):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        run(*args)  # 按 definition 的输入顺序展开为位置参数
        end.record()
        torch.npu.synchronize()
        times.append(start.elapsed_time(end))
    return times
```

脚本必须打印每例进度与最终摘要，并把完整结果写入 `<output>/result.json`。

## 结果字段约定

为了让 optimize 和交付报告能直接消费，结果 JSON 至少包含：

```json
{
  "status": "passed",
  "solution": {"source_sha256": "<sha256>"},
  "environment": {"device": "npu:0", "arch": "...", "cann": "...", "torch": "...",
                  "python": "...", "seed": 1, "warmup": 1, "trials": 2},
  "selection": {"cases": ["scale-001"], "mode": "benchmark", "paired": true, "shard": null},
  "correctness": {"status": "passed", "passed": 1, "total": 1},
  "correctness_evidence": {"path": "evidence/full-test/result.json", "sha256": "<sha256>", "cases": 1},
  "workloads": [
    {"uuid": "scale-001", "status": "passed",
     "correctness": {"status": "passed"},
     "performance": {"status": "passed",
       "candidate": {"mean_ms": 0.12, "std_ms": 0.01, "trace_dir": "..."},
       "reference": {"mean_ms": 0.20, "std_ms": 0.02},
       "order": "candidate_first", "speedup": 1.67}}
  ],
  "performance": {"status": "passed",
    "candidate": {"mean_ms": 0.12}, "reference": {"mean_ms": 0.20}, "speedup": 1.67},
  "profiling": {"status": "passed", "artifact": "..."}
}
```

- `performance.candidate.mean_ms` 是统一性能指标路径。
- `selection.paired` 必须为 `true`（有对标实现时）。
- `profiling.status` 只有在每例都产出 CSV 时才写 `passed`。
- benchmark 跳过逐例正确性时，`correctness_evidence` 必须指向 full test 的合并结果，
  写明路径、SHA-256 和覆盖用例数；缺失时该 benchmark 结果不可作为交付证据。

## 分片与续跑（按需实现）

- 批量用例建议按 `--shard i/n` 分片，每片写独立 `--output`，最后合并；
- 每例结果单独落盘，重跑时可按 `kernel_sha256 + workload + case` 复用已通过结果；
- 分片合并时校验 uuid 集合完整、无重复；合并结果保留每片路径与 SHA-256；
- 设备不健康或单例失败时，允许换卡重试一次，记录原设备、新设备与重试结果；二次
  失败按 kernel 问题处理，不得加大超时。
- 禁止用 `rm -rf` 丢弃已完成结果。

## 单融合审计（仅 `single_kernel` 模式）

用户明确要求单融合时，测量前人工完成以下检查，并把结论写入 `ITERATIONS.md`：

- solution 只声明一个 `@tla.kernel`，每次 `run` 只 launch 它；
- 每个 decorated kernel 的自由名字只来自形式参数、函数内局部值、Python 内建符号和
  `tla` API，不存在模块级可变配置或 closure；
- 候选源码不调用 `torch` / `torch.nn` / `torch.ops` 或 Tensor 计算算子替代 DSL；
  只允许 dtype/device/shape 元数据检查和空输出分配；
- 单独一次 profiling 产出的 `kernel_details.csv` 只出现该 kernel（warmup 与主采集各
  一条），不得出现替代计算的 torch / aclnn 算子；
- 任何一项不满足即判定本轮无效，不产生性能结论。

默认 `performance` 模式不做上述审计，允许多 kernel 和 GM 中间张量。

## 与其它阶段的衔接

- full test 使用完整 `workload.jsonl`；benchmark 使用完整 workload 并逐例配对。
- optimize 使用 design 冻结的 `workload_dev.jsonl`（用户用例 ≥30 条时为 30~40 条；
  不足 30 条时全选；无用户用例时按覆盖需要自定，总数 ≤40），每轮复用同一份样本。
- benchmark 正确性失败时回到 develop 的 debug 流程；环境缺失时结果记 `blocked` /
  `not_run`，不得写成通过。
- 模板位于 [`templates/`](templates/)；脚本与结果路径必须记录到 `ITERATIONS.md`。
