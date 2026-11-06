# 工具选择与能力范围

本 skill 使用已安装的 `opmodel` 包。按下表选择建模、搜索、分析与验证入口，
具体操作见各参考文档，可执行例子见[运行示例](wheel-commands.md)。

## 按问题选择入口

| 当前问题 | 使用方式 | 保留的结果 |
| --- | --- | --- |
| 从公式设计实现族 | 按下文建立独立参考、任务和执行图 | 语义约束、实现选择、可运行起点 |
| 新建或扩展算子模型 | 用户目录中的六份定义、搜索声明和必要 hooks | 固定 case、任务覆盖、数量、资源与依赖 |
| 固定图分核、tile、阶段和 slots | `split_run`、`dag_run` 或 `joint_search` | 声明空间、实际展开、淘汰、排名和完整 case 复评 |
| 结构或映射变换 | `graph_search --actions`、`--apply-action` 和实验命令 | 父候选、变化机制、合法性、快照与评分 |
| 多轮候选管理 | `agent_search` 会话 | 契约、候选、评价、预算、保留及结束原因 |
| 定位流水等待 | 完整 case 的 trace、统计及参数快照 | 首次等待、依赖、容量和单变量对照 |
| 解释模型与设备偏差 | [联合流程](workflow.md#分类反馈与独立校准) | 数量、服务、结构或证据资格分类与独立验证 |
| 实现与设备验证 | 用户提供的 DSL / AscendC 工程及设备工具 | 原正确性契约、完整计时、实现与模型的对应关系 |

模型搜索、设备编译和设备测量是不同操作。会话工具不会自动生成 DSL 实现、运行设备
或回收实测结果；代理按授权范围完成交接。只有模型环境时，设备结果保持未验证。

## 从公式建立模型

1. 固定公式、输入输出、shape、dtype/layout、归约、别名与副作用，建立独立数值参考。
2. 将数学约束与实现选择分开。按硬件提出可实现的切分、融合、物化、驻留和流水方案，
   核算新增搬运、布局转换、缓冲和同步；没有生产实现时将首个正确实现记为起点。
3. 在用户目录中定义语义、任务空间、执行图和成本，先恢复一个可运行的完整 case。
4. 用适用的内置搜索器比较配置与结构，再复评完整 case。实现与设备实验按任务范围推进。

从公式建模不要求使用公式生成器。`python -m accel_model.kernel.dag.gen MODEL`
仅适用于同时提供 `definition/formula.yaml` 与 `definition/cube.yaml` 的模型；默认会写入
`definition/dag.yaml`，`--check` 只比较生成结果。它不支持 `--help`，也不会自动理解任意公式。

## 标准模型接入

```text
my_operator/
  __init__.py
  definition/
    semantic.yaml
    workload.yaml
    split.yaml
    operator.yaml
    dag.yaml
    costs.yaml
  optimize/
    search.yaml
```

- `semantic.yaml` 定义张量、运算、版本及数值语义。
- `workload.yaml` 定义输入、默认值、派生参数、约束和逐核任务投影。
- `split.yaml` 定义任务空间、任务字段、分配工作量及可拆维度。
- `operator.yaml` 定义硬件 profile、buffer、phases、执行参数和默认分核策略。
- `dag.yaml` 定义搬运、计算、循环、依赖及缓冲区使用。
- `costs.yaml` 定义成本来源、单位、校准状态及命名绑定。
- `optimize/search.yaml` 定义参数空间及模型支持的改写、映射和分核选择。

用 `opmodel-assets export` 导出可编辑资源后复制模型目录，或使用本 skill 随附的示例。
仅修改名称不改变数学语义。case、实验和输出保存在用户项目，固定执行不应依赖搜索记录。
复杂派生或约束确有需要时再添加 `workload.py`、`split_hooks.py`、`ms_hooks.py`。
存在 `ms_hooks.py` 时应提供 `SEARCH_DIMS`；不能假定还自动采用 YAML 搜索维度。

新模型先核对固定 case、尾块、适用的相邻任务与多组、输出归属和全部消费者的释放。
动态字段按当前模型 schema 使用表达式映射，不能把任意字符串或 Python 代码当作表达式。
跨 task 的 L0 驻留需显式的 allocation/version；L1/UB 驻留需模型支持的 storage 声明。
逻辑 pool 名不增加物理空间。固定执行支持不表示搜索变换也支持这些契约。

## 搜索接入

固定 DAG 配置搜索和结构方案搜索统一按
[搜索接入](../agent-dag-opt/search-integration.md)执行。传递当前模型、case、硬件、成本、
后端与预算；固定 DAG 可以直接进入配置搜索。借用入口不更换调用方的评估口径。

## 执行入口与完整 case 复评

- `opmodel MODEL --output DIR` 的普通分支运行固定配置，省略输出参数时写入模型的
  `output/`。`--case/--backend/--gates/--trace` 属于物理分支。
- 普通完整 case 使用 `run_native_case(..., derive=True)`，读 `total_cycles` 及
  `summary`，使用 SimPy；不接受物理执行专用 policies、gates 或硬件路径覆盖。
- 分核方案取其 `split` 映射传给 `strategy`；普通完整 case 要求策略核数等于选定硬件
  的 `usedCoreNum`。候选需要其他核数时不能直接套用。
- `dag_run` 支持 `--hardware` 名称或 YAML 文件。普通完整 case 和分核 CLI 不具备
  同样的覆盖接口；某次搜索覆盖硬件后，其他入口不会自动继承。
- `physical_execution` 模型需要完整物理契约，才能使用其支持的后端和检查项。
  `both` 不代表任意模型都支持双后端；后端对拍不等于设备正确性或成本验证。

DAG 局部评分与完整 case 周期范围不同，不能直接相除报告设备加速。候选的 tile、slots、
阶段及发射策略应分别同步到支持它们的 case 或模型字段，保留基线副本；不要直接用
搜索导出的片段覆盖六份完整定义，也不要把所有派生参数塞入 case 输入。

## 版本和故障处理

通过已安装包的版本、CLI 帮助、公开函数签名和随附
示例核对可用接口；遇到加载错误时记录完整报错与输入。缺少入口时说明版本和功能缺口，
再按本页支持范围选择可行路径。

分别记录固定执行、参数搜索、结构变换、设备实现和数据采集是否可用。
buffer 复用分析若出现 `tuple ... has no attribute 'name'` 错误，
保留版本与最小输入，使用 trace、指令与流水线统计继续分析。

解析或搜索资格失败时，用最小输入区分写法、模型适配、成本身份和后端限制。能力缺口
按[工作目录与执行范围](guide.md#工作目录与执行范围)处理，不通过修改二进制、运行时替换或
虚构性能参数绕过。重试必须针对新的假设并受任务预算约束。

实现交接和测量反馈沿用[记录契约](workflow.md#记录与交付)。理论、校准、代表核、
全核共享和设备实测分别呈现；没有适用峰值或硬件计数时不强行计算 MFU/MBU。
