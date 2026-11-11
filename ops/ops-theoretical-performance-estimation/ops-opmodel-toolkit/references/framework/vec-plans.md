# Vec 方案与独立 VF 预测

用于显式描述向量计算步骤、比较 VF 融合／拆分和流水计划，或预测一次 VF 调用。
输入是用户程序、实际调用、计划和有来源的成本参数；无需编译产物或外部 VFModel。
原有固定周期、affine 成本及 VFModel 各有入口，见[VF 成本](vf-costs.md)。

## 选择任务与准备输入

| 任务 | 输入与调用 | 交付 |
| --- | --- | --- |
| 预测一次 VF | `estimate_vf(program, invocation, plan, hardware_cost)` | 单次周期、身份、诊断和假设 |
| 比较 Vec 的多个计划 | 完整模型、有限计划及 `search_vec_plans` | 实际候选、图评分、搜索覆盖与验证状态 |
| 交接方案 | 选定候选的声明和实际构造状态 | VF 边界、调用域、存储和流水安排；按[设计工作流](../opmodel-design-gen.md)成文 |

先固定完整输入输出和步骤 ID，说明 dtype、转换、归约轴和依赖。无用户实现源码时，
可声明 `status: hypothetical` 并省略 `source`，保留假设身份。计划不改变原数学语义。
仅要求方案或接口时，检查表达、调用及适用的语义／调度正确性；预测误差和设备收益
属于另有数据与目标的验证任务。

## 单次预测示例

以下为可独立调用的合成示例。成本数值仅演示接口，不代表任何芯片参数或实测结果。

```python
from accel_model.model.graph.model.vf_ir import VFPlanSpec
from accel_model.model.vf import estimate_vf

program = {
    "inputs": ["x", "b"],
    "outputs": {"y": "v"},
    "steps": [
        {"id": "u", "op": "mul", "inputs": ["x", "b"], "dtype": "fp32"},
        {"id": "v", "op": "add", "inputs": ["u", "b"], "dtype": "fp32"},
    ],
}
invocation = {
    "inputs": {name: ((2, 5), "fp32") for name in ("x", "b")},
    "outputs": {"y": ((2, 5), "fp32")},
    "hardware": "example_chip",
    "clock_ghz": 1.0,
}
plan = VFPlanSpec("fused", (("u", "v"),))
hardware_cost = {
    "version": "vf_analytic_v1",
    "hardware": "example_chip",
    "clock_ghz": 1.0,
    "source": "合成接口示例，未校准",
    "dtypes": {"fp32": 4},
    "resources": {"memory_port": 1, "alu": 1},
    "call_cycles": 0,
    "loop_cycles": 0,
    "operations": {
        "load:fp32": {"lanes": 4, "latency": 1, "occupancy": {"memory_port": 1}},
        "store:fp32": {"lanes": 4, "latency": 1, "occupancy": {"memory_port": 1}},
        "mul:fp32": {"lanes": 4, "latency": 3, "occupancy": {"alu": 1}},
        "add:fp32": {"lanes": 4, "latency": 3, "occupancy": {"alu": 1}},
    },
    "max_operations": 4096,
    "max_live_bytes": 4096,
}
estimate = estimate_vf(program, invocation, plan, hardware_cost)
print(estimate.cycles, estimate.identity, estimate.assumptions)
```

`program` 是显式输入、输出和有限步骤；`invocation` 给出本次真实 shape/dtype、硬件与时钟。
单次预测的 `plan` 必须是覆盖程序的一个完整分区；不能把多分区计划直接当作一次 VF。
`hardware_cost` 提供所有实际使用的操作，包括 load/store 和转换，未知参数不补零。
硬件、时钟、dtype 字节数须匹配。参数来源及适用域由调用方保存。

`VFEstimate` 的公开结果为 `cycles`、`identity`、`operation_counts`、`resource_service`、
`peak_live_bytes`、`timeline`、`assumptions`。周期换算成微秒为
`cycles / (clock_ghz * 1000)`。资源服务量可重叠，不能相加作为总周期；活跃字节
不等于编译器的寄存器分配结果。参数、程序、调用或计划改变后重新预测。

## 模型中的字段

继续使用六份固定定义与搜索声明，无需新增固定模型文件。

| 文件与字段 | 内容 |
| --- | --- |
| `dag.yaml:task_graph.implementation_blocks` | VF 块的 `numeric_program`、`paradigms` 和默认 `plan`，保留原有必填声明 |
| `dag.yaml:task_graph.vec_regions` | 按区域绑定父块及有限 `plans` |
| `costs.yaml:analytic_profiles` | 硬件、时钟、操作参数、资源及预算 |
| `costs.yaml:vf.<块名>.<变体>` | `{model: analytic, profile: <名称>}` |
| `optimize/search.yaml:vf_plans` | 每个区域参与搜索的计划名称列表 |

`paradigms` 以步骤 ID 标注 `elementwise/reduction/broadcast/cast/copy` 及适用轴。
步骤不得遗漏或重复，范式应与程序一致；一个 VF 可以包含多个范式。
如下片段放入已有 `task_graph`，父块 `pointwise` 须具有完整程序、访问及工作量声明：

```yaml
vec_regions:
  vec:
    block: pointwise
    plans:
      - {name: fused, partitions: [[u, v]]}
      - name: split_buffered
        partitions: [[u], [v]]
        axis: 0
        tile: 16
        slots: 2
        issue_order: windowed
        unroll: 4
        interleave: 2
        order: interleaved
```

| 计划字段 | 用户需要确定的含义 |
| --- | --- |
| `partitions` | 每个 VF 包含的步骤，完整覆盖且保持依赖顺序 |
| `axis/tile` | 外轴调用分块；`tile: 0` 表示整个外轴，尾块保留实际范围 |
| `slots` | 跨 VF 中间值的 UB 槽数；无中间值时不能声明多槽 |
| `unroll/interleave/order` | VF 内工作单元安排；`order` 为 `block` 或 `interleaved`，交错深度不超过展开数 |
| `issue_order` | VF 调用间的 `region` 或 `windowed` 发射安排；与 VF 内交错分别声明 |

当前同一计划的各分区使用相同内部展开／交错设置。增加 UB 槽不表示同一 AIV 上
多个 VEC 调用并行。拆分后应预测各 VF，不能直接按元素比例分摊父块的实测周期。

YAML profile 的非零成本值通过 `coefficients` 引用，单位为 `cycles`；Python API
接收已解析的数值。`latency/call_cycles/loop_cycles` 及预约 offset 可显式为零，
向量宽度、dtype 字节数和资源数量须为有效结构参数。可选的 `reservations`、
`forwarding`、`forbidden_gaps` 只表达有来源的时序约束，不从操作名称猜硬件能力。

## 搜索与结果

`search_vec_plans` 一次只处理一个 Vec 区域，输入基线须未分片，且无预建的存储分配、
控制或发射协议。给出接口方案时先说明这些前提及是否已核对；“完整模型目录”本身
不能证明满足前提。已有模型不满足时保留原因，另行准备符合条件的基线，不删除必要约束。

在完整 `optimize/search.yaml` 中选择有限计划；当前此入口不与其他构造维度混用：

```yaml
vf_plans:
  vec: [fused, split_buffered]
rewrite_grammar: []
mapping_choices: []
```

调用片段中的 `operator_dir` 为完整模型，数组对应其数值输入，地址键来自实际别名契约：

```python
from accel_model.optimize.graph.global_search.search_runner import search_vec_plans

report = search_vec_plans(
    operator_dir,
    address_keys={"x@0": "x-buffer", "b@0": "b-buffer", "y@0": "y-buffer"},
    numeric_cases=[{"x": x_values, "b": b_values}],
)
```

也可传入已有 `mapped`；区域外的向量任务使用其已有 `vector_costs`。地址键用于
物理别名身份，不能为取得资格而随意编造。数值数组为匹配实例化语义的扁平输入。

- `rows` 保存各候选 `state/score` 和构造证据；单次预测见 `score.vf_predictions`，
  传入图周期见 `score.result.cycles`。局部图不得标为完整算子。
- `ledger` 记录实际候选及覆盖，包括首项。超容或非法构造记录为
  `capacity_invalid`／`invalid_contract` 后继续后续计划；失败候选也计入预算。
  全部非法但枚举结束时可有 `complete=True` 和空排名；预算耗尽或中断为
  `complete=False`。公共契约加载失败或其他异常仍可退出，须依据实际记录判断
  后续候选是否评估，不能从计划列表或单次失败推断搜索覆盖。
- `analytic_ranking`／`analytic_rankable` 与 VFModel 的 `prediction_rankable` 分开。
  保留条件假设和 `recommendation=false`，数值通过不授予编译或设备资格。
- 无数值输入或回放超预算时分别记录未验证／不支持；小规模通过只覆盖该用例。
- checkpoint 可保存实际构造状态；bundle 导出保留区域与计划声明，重载后仍需构造。
  原生 `search_bundle` 不接收 `vf_plans`，使用本页入口。

## 支持范围与验收

支持有限静态程序的逐元素操作、copy/cast、连续末轴 sum/max 和显式前缀广播；
实际操作与 dtype 必须有参数。纯标量、非连续布局、未知操作成本、硬件不匹配和
超预算输入明确失败。原生 `vec.cycles` 不会自动获得程序或计划。

仅切最外并行轴，各分区共享该轴；不切归约轴、不改状态输出或原地边界别名。
中间值为同一 AIV 的 UB，输出为普通 GM，不生成跨核 VF 通信。各分区继承父块
scratch 预算，不能视为精确寄存器分配、spill 或 bank 冲突预测。

验收保留实际 VF 节点、分区输入输出、调用次数、尾块、中间分配、槽版本与发射安排，
核对改变参数是否改变实际方案。融合／拆分用独立语义参考和适用的执行回放验证；
合法方案可以无收益或更慢。单次 VF、调用合计与含搬运等待的图周期分别报告。
