# 框架使用参考

本参考说明 OpModel 的模型输入、调用方式、输出含义和使用限制。
先按[快速入门](../quickstart.md)运行完整 case，再按需要选择接口。

| 任务 | 参考 |
| --- | --- |
| 组织模型文件、声明参数和成本 | [模型文件](yaml-operator-schema.md) |
| 编写搬运、计算、循环和依赖 | [DAG YAML](dag-reference.md) |
| 声明缓冲区、固定存储和版本 | [存储接口](ub-storage.md) |
| 描述控制值与等待条件 | [控制接口](control-state.md) |
| 描述数据场景及有界分支／循环 | [运行时控制](runtime-control.md) |
| 编写逐核任务及 Python hooks | [分核接口](split-model.md)、[Python Stage](../new-operator/python-stages.md) |
| 描述物理 AIC/AIV 上的模型 | [物理执行接口](physical-execution.md) |
| 选择硬件及成本来源 | [硬件配置](hardware-binding.md)、[VF 成本](vf-costs.md) |
| 描述 Vec 计划或独立预测 VF | [Vec 方案与预测](vec-plans.md) |
| 声明变换搬运和解释读写量 | [搬运契约](transfer-contracts.md) |
| 接入用户实现的编译结果 | [编译反馈](compile-feedback.md) |
| 选择运行方式和理解时间线 | [运行与 trace](runtime-and-trace.md)、[支持范围](backend-capabilities.md) |
| 声明参数搜索或候选比较 | [DAG 搜索接口](dag-search.md)、[图搜索与会话](graph-search.md) |
| 定义任务目标、复用评价与多输入规则 | [外部目标](objectives.md)、[统一规则](policies.md) |
| 从公式生成模型 | [公式输入](formula-schedule-user-guide.md) |
| 检查结果和比较实测 | [性能产物](performance-manifest.md)、[数值验收](numeric-execution-validation.md)、[实测比较](sim-vs-measurement.md) |

示例 YAML 片段需要放入完整模型，不能单独作为 case 运行。安装或升级后，
先核对命令帮助和模型支持范围，再运行项目使用的示例；已有验证不能替代升级后的检查。
