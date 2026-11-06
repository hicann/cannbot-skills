# 从规格到固定模型

在[建模主线](../new-operator.md)完成输入与数学规格后，按本页确定拓扑、配置及运行入口。
详细字段以 [YAML 规范](../framework/yaml-operator-schema.md)和
[DAG 语法](../framework/dag-reference.md)为准；循环并发问题见[并发与生命周期](concurrency.md)。

## 推导拓扑

先列每项计算与搬运的输入、输出、实际执行单元、消费者及状态读写，再划分 stage。
stage 是建模单元，phase 决定阶段推进，token 表示软件计算顺序；三者不等于物理核。
物理 AIC/AIV 交接须另外声明 owner、区域与协议，见[物理执行](../framework/physical-execution.md)。

| 需要确定的内容 | 推导依据 |
| --- | --- |
| phase | 真实可重叠工作、阶段交接与容量；固定一种分法不能证明其他划分无效 |
| token | M/VEC 上的软件顺序；保留实现要求的递推，不为无关工作增加门控 |
| dataflow | 同一逻辑值的生产者与消费者，包含组内/跨组 task 身份 |
| buffer | 存储层、每槽物理字节、并发版本、取得与最后使用边界 |
| 参数域 | 语义限制、dtype/对齐、硬件容量及实现可调范围 |

`dataflow.offset` 按数据归属选择：同一 task 为 0，读取上一 task 才为 1；
消费者在后面的 phase 执行并不自动意味着 offset=1。需要跨组连接时才使用
`cross_group: true`，明确首组、末组和上一组最后 task 的行为。

计算累加、原地覆盖和状态递推仍须真实数据依赖，不能依赖默认串行碰巧保证正确。
删除依赖前先证明它由有效依赖传递或资源生命周期覆盖，再对边界 case 复核；
单次 trace 相同不足以证明任意输入等价。

## 写六份固定定义

固定定义放在 `definition/`，独立于 `optimize/`。所有入口接收算子根目录。

| 文件 | 编写与核对重点 |
| --- | --- |
| `semantic.yaml` | 数学语义、输入输出与独立参考；覆盖、归约、状态及数值标准 |
| `workload.yaml` | 公开 case 输入、默认值与派生；shape、tile、逐核投影保持可追溯 |
| `split.yaml` | 任务空间、原子约束与逐核参数；异构 task 不靠平均数替代 |
| `operator.yaml` | 硬件引用、参数、buffer、phase、token、dataflow、运行策略 |
| `dag.yaml` | 真实指令、循环、条件、搬运形状、dtype、pool 与依赖 |
| `costs.yaml` | 成本来源、单位、适用域、实现绑定及未知项 |

使用已安装包的硬件资源或用户导出的配置。需要调整时复制到用户项目，保留参数来源；
不要把默认芯片、算子拟合 profile 和实际设备结果混称为理论性能。
独立评估按[硬件选择](../hardware.md#选择评估配置)确定同芯片配置，逐路由记录实际支持、
绑定和简化范围；接续已有实验则保持冻结的硬件与成本，配置错误不静默降级。
声明 `tier` 时核对该路由的 `bw_tiers`；显式 L2 档位仅表示访问假设，自动缓存复用还需
物理身份与共享服务。详见[硬件绑定](../framework/hardware-binding.md)。

## 写 DAG 时容易遗漏的契约

- `id`、`pool`、`use_pools`、`tag` 放在 `move/mmad/vec` 字典内部。
- `dtypes` 明确覆盖张量；源、目的转换分别确定字节，L0C 容量用累加 dtype。
  亚字节 dtype 使用逻辑元素数乘 dtype 字节数，不再把逻辑维手工减半。
- `move.size` 表达实际搬运块；尾块用有效大小。需要覆盖默认数量时使用
  `move.transfer` 的完整源/目的字节及 provenance，不能仅改一个成本缓存字段。
- `mmad` 的 M/K/N 与操作数相符；K 分块首块初始化、后续累加，最终 FIX 次数符合语义。
- VF 成本可来自实测、系数/公式或显式配置的预测器，均需来源与适用域。
  多 AIV 已预切分的工作量不能重复缩放，见[成本接口](../framework/vf-costs.md)。
- `loop`、`if`、`skip_if` 是按输入参数选择的模型声明；数据相关运行分支须有明确建模契约。
- `signals` 导出实际指令/张量事件，`dataflow.output/input` 对齐；条件未生成的标签
  不能当作正常完成信号，处理规则见[并发指南](concurrency.md)。

循环局部池、整阶段池、组内驻留和全程驻留使用不同 scope。slots=2 只是一个可评估
选择，不是所有 stage 的默认正确答案。pool 数量与硬件物理槽、容量分别核对，
见[存储、缓冲与版本接口](../framework/ub-storage.md)。

## 工作量与运行入口

公开输入在 `workload.yaml:parameters` 声明，经 derive 产生有效参数。不要靠覆盖
`num_tasks/num_groups` 再触发 derive 来制造测试：后续派生可能覆盖注入值。
通过实际公开 shape 构造单 task、多 task、多组与尾块，并检查最终 params 和 assignment。

纯 YAML 模型可使用 `STAGES = {}` 的 `__init__.py`。需要 Python 推导或 stage 时，
由用户模块显式导出，保持依赖自包含；不向 `sys.path` 注入开发机器路径。
Python Stage 的注册与接口见[Python Stage](python-stages.md)。

优先复用 [run_case.py](../../scripts/run_case.py) 或
`accel_model.app.case.native_case.run_native_case()`，取得标准运行产物。
一份固定配置、代表核及共享全核的 scope 分开记录，见[运行契约](../framework/runtime-and-trace.md)。

## 验收后接入搜索

先核对独立推导的 MAC、搬运量、输出次数与容量，再检查完整执行及模型支持的数值项目。
[性能清单](../framework/performance-manifest.md)的采集结果不自动生成正确预期；
[执行数值验收](../framework/numeric-execution-validation.md)不能用周期一致性代替。

在 `optimize/search.yaml`、独立实验 YAML 或 `ms_hooks.py` 声明合法空间。
`SEARCH_DIMS.tiles` 的维度与搜索参数必须同单位；参数改变后确认实际 workload 与图确实变化。
固定 DAG 搜索支持的 phase/slot 输入及使用限制见[DAG 搜索](../framework/dag-search.md)。
接入与交付复用[统一搜索要求](../agent-dag-opt/search-integration.md)，保持调用方的 case 和成本口径。

多个实现模板分别保存完整固定定义、硬件与成本身份；可复用用户项目中的公共 Python
模块，但每个模板须独立加载、验证和复现。不用一个模板的验证状态替另一个模板背书。

VF 没有适用实测时，可按[VF 成本约定](../framework/vf-costs.md)声明有依据的周期／
公式，继续初步仿真和搜索；保留单 AIV 工作量、调用次数、未校准状态与不确定性。
新模型默认普通带宽，真实工作量、依赖和容量仍需完整；详细选择见[硬件配置](../hardware.md)。
