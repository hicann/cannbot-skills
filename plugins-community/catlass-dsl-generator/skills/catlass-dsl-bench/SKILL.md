---
name: catlass-dsl-bench
description: Use when independently checking and timing a CATLASS Python DSL solution against definition and workload files
---

# CATLASS DSL 公共 Benchmark

以 NPU-KernelBench 风格的 `solution.json`、`workload.jsonl` 和 `definition.json`
作为唯一公开输入，在当前进程依次执行输入生成、正确性、同步 warmup 和重复计时。
输入会执行 Python 代码，必须来自用户信任的来源；本工具不提供安全沙箱。
NPU candidate 还必须通过单融合 kernel anti-hack：每次 `run` 只能下发一个由 solution
源码声明的 `@tla.kernel`，否则整套 benchmark 失败且不产生候选性能结论。

```bash
python3 skills/catlass-dsl-bench/scripts/bench.py \
  --solution <solution.json> \
  --workload <workload.jsonl> \
  --definition <definition.json> \
  --output <evidence-directory>/benchmark \
  --reference-profile-cache <project>/.catlass-dsl/profiles/reference \
  --device npu:0 --seed 1 --warmup 1 --trials 2
```

默认执行 1 次 warmup、采集 2 次；仍可通过 `--warmup` 和 `--trials` 显式覆盖。

模板位于 [`templates/`](templates/)。

## Definition 接口

`definition.json` 是算子语义的权威输入：

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

- `axes` 固定为 `{}`；workload 的 `axes` 固定为 `null`。禁止用 `const`、`var`、
  `expr` 或 axis 名称间接表达 case shape 和配置。
- definition 中 `shape` 为 `null`，或只使用十进制常量字符串表达所有 case 共有的
  固定 shape；不得引用 axis。每个 tensor case 的实际数值 shape/dtype 仍必须直接写入
  workload。标量和可选空输入的 definition shape 为 `null`。
- `dtype` 支持 `bool`、`uint8`、`int8/16/32/64`、`float16/32/64` 和
  `bfloat16`。
- `reference` 必须是包含顶层 `run` 的完整、可信 Torch Python 源码。
- `custom_inputs_entrypoint` 可指定 reference 源码中的 `(axes, device) -> dict`
  输入生成函数；此时 `axes` 仍为空对象，对应 workload 的所有输入必须统一使用
  `custom`。custom 仅用于特殊分布或约束，不得恢复 axis binding；shape、dtype、layout、
  开关、标量和可选输入状态仍须在 workload 行中显式保存。不得用 `CASE`、`case_id`、
  序号或其他不透明索引在 reference 内部查表恢复 case 配置。

## Workload 接口

`workload.jsonl` 每个非空行是一个独立用例：

```json
{"uuid":"scale-001","axes":null,"inputs":{"x":{"type":"random","shape":[1024],"dtype":"float32"},"alpha":{"type":"scalar","value":0.5}},"tolerance":{"max_atol":0.00001,"max_rtol":0.00001,"required_matched_ratio":1.0,"max_error_cap":null,"allow_negative_inf":false}}
```

- `uuid` 在文件内唯一，并直接作为结果中的 case id。
- `axes` 必须为 `null`。每个 tensor input 必须直接声明数值 `shape` 和 `dtype`；标量
  使用 `{"type":"scalar","value":...}`，可选空输入直接写 JSON `null`。
- 每个 case 的完整配置必须保存在其 JSONL 行中；`uuid` 仅用于身份标识，不能作为
  隐藏配置的 dispatch key。shape、dtype、layout selector、序列边界、布尔开关、
  标量参数和 initial state 是否存在等都必须直接保存在该行。
- `random` 必须声明 `shape`、`dtype`，并可为整数输入声明 `range: [low, high]`。
  此外支持 `zeros`、`ones`、字面量 `scalar`、JSON `null`、definition 自定义生成的
  `custom`、`safetensors` 文件输入和变长 `tensor_list`。
- 使用 `custom` 时不得借此隐藏 shape/dtype 或恢复 axis binding，也不得在 reference
  中维护 `_CASE_SPECS` 等 case table。边界、尾块和广播尺寸分别用独立 workload 表达。
- tolerance 逐 workload 生效。元素满足
  `abs(candidate-reference) <= max_atol + max_rtol * abs(reference)` 时计为匹配；
  匹配比例不得低于 `required_matched_ratio`，`max_error_cap` 是额外硬上限。
- NaN 和 Inf 默认失败；`allow_negative_inf` 只允许两侧同位置的 `-inf`。
- 随机输入由 CLI 的 `--seed` 固定。同一 suite 中每个 workload 使用相同 seed，
  以便结果可复现。

## Solution 接口

`solution.json` 保存完整方案和源码：

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

- `definition` 必须等于 `definition.name`。
- CATLASS DSL 是 Python DSL，`languages` 使用 NPU-KernelBench 兼容值
  `["python"]`，不要填写 C++ CATLASS 的 `catlass`。
- `sources[].path` 必须是安全相对路径，`content` 必须包含完整源码。所有源码会复制
  进证据目录后再导入。
- `entry_point` 固定为 `<source-path>::<callable>`。
- 返回值模式使用 `destination_passing_style: false`，入口返回 Tensor，或与
  definition 输出顺序一致的 flat tuple/list。
- DPS 模式使用 `destination_passing_style: true`，入口参数依次为全部输入和全部
  输出 buffer，不返回结果；所有输出必须在 definition 中声明 shape 和 dtype。
- 所有输入按 definition 顺序作为位置参数传入，solution 和 reference 必须接受相同
  调用。正确性调用使用独立 Tensor 克隆；计时阶段复用一组独立输入，不把输入生成和
  clone 时间计入算子时延。
- solution 中每个 `@tla.kernel` 必须自包含，只能读取形式参数、函数内局部值、Python
  内建符号和受信任的 `tla` API 命名空间。禁止模块级 tile/shape/dtype/params/开关、
  可变全局状态和 closure capture；固定值在 kernel 内定义，运行时值使用显式 ABI 或
  tensor metadata。solution 必须且只能声明一个 `@tla.kernel`；所有 shape/dtype/layout
  编译期变体复用该函数，并由形式参数类型或 metadata 生成各自编译产物。

## Anti-hack 接口

- benchmark 前必须审查每个 decorated kernel 的自由名字，并检查 host entry point 没有
  通过 `global`、模块属性或可变容器改变 kernel 编译语义。结构审查失败时不得执行或
  接受 benchmark，即使数值和单 launch profile 能通过。
- 单 kernel 同时按源码声明和运行时 launch 计数：solution 必须且只能声明一个
  `@tla.kernel`，每个 workload 的每次候选调用必须只 launch 该 kernel，且所有 profile
  中 kernel 名必须稳定并匹配该唯一声明。不得为编译期变体声明独立 dispatch kernel。
- 候选源码不得调用 `torch`、`torch.nn`、`torch.ops` 或 Tensor 计算算子，也不得用
  Python Tensor 运算替代 DSL。只允许 dtype/device/shape 元数据检查、
  `dim/numel/size/stride/is_contiguous`，以及 `torch.empty/empty_like/empty_strided`
  空输出分配；Torch reference 不受此限制。
- NPU candidate 必须生成可解析的 `kernel_details.csv`。记录数必须严格等于 trials；
  缺失、零/多 launch、vendor/Torch kernel 或名称不匹配统一记录为 `category=hack`。
  `step_trace_time.csv` 可继续作为计时来源，但不能替代 anti-hack 证据。
- candidate `run` 只在 `trials` 个 profiler 窗口内调用；其中一次输出直接用于 correctness，
  不再执行无法证明 launch 次数的 candidate correctness/prime/warmup 调用。
- 每个 candidate trace 必须同时生成 `anti_hack_manifest.json` 和 SHA-256 绑定的逐
  iteration `anti_hack/iteration-NNNN/kernel_details.csv`；每个分片严格一行，合并 CSV
  不能代替 iteration 边界证据。
- 根结果和每个 workload 都记录 `anti_hack`，字段包括 `status`、`policy`、声明/观测
  kernel 名、profile iteration 数、launch 数、每 iteration launch 数和稳定 reason。
  `policy` 固定为 `single-fused-catlass-kernel-v1`。CPU 使用
  `status=not_applicable`；NPU 只有全部 workload `status=passed` 才能通过。

## 结果与流程

根 `result.json` 汇总所有 workload，`workloads[]` 指向各用例的原始结果；兼容 optimize
的聚合指标为 `performance.candidate.mean_ms`，当前 schema 为 version 3。NPU 性能使用 `torch_npu.profiler`
采集，优先以 `step_trace_time.csv` 的 `Computing` 总时间除以 trials，缺失时回退到
`kernel_details.csv` 的 kernel 总时长。候选实现会随开发或优化轮次变化，因此每次都在
本轮输出目录的 `profiling/` 中重新采集；不变的 Torch reference 只采集一次，并按
definition、workload、设备环境和测量参数的 SHA-256 指纹保存在项目级
`.catlass-dsl/profiles/reference/<fingerprint>/`。后续 develop/optimize run 只复用该
manifest 和原始 trace，不复制或再次执行 reference profiling。若输出目录不在
`.catlass-dsl` 下，使用 `--reference-profile-cache <project>/.catlass-dsl/profiles/reference`
显式指定共享目录。definition、workload、设备或 profiler 参数变化会产生新指纹，避免
误用陈旧标杆。
任何 workload 精度失败都会使 suite 失败，且该 workload 不产生有效性能结论。
环境缺失返回退出码 2；配置或正确性失败返回 1。

独立纳入状态机时，`CONTRACT.md` 的批准命令必须把三个输入文件和输出目录作为逐项 argv。
只有契约声明性能目标时才把结果交给 `catlass-dsl-optimize`；正确性失败回到 develop
内部定位流程。
