# 常见问题

| 现象 | 处理方式 |
| --- | --- |
| wheel 显示不支持当前平台 | 核对发行包的 Requires-Python、wheel 平台与 ABI 标签；不能通过改文件名获得兼容性 |
| `ModuleNotFoundError` 或 `opmodel` 命令不存在 | 激活安装环境，用同一个 `python -m pip` 检查安装 |
| 动态库或 GLIBC 加载失败 | 核对实际 wheel 的 manylinux 标签与系统 glibc；例如 manylinux_2_39_x86_64 要求 glibc ≥ 2.39，不能直接用于 musl |
| 创建 venv 提示缺少 ensurepip | 安装系统对应版本的 venv 支持，或使用包含 venv 的标准 CPython |
| 安装成功但找不到 scripts/run_case.py | 该文件位于 skill 内；设置 OPMODEL_SKILL_DIR 并按快速入门从用户项目运行 |
| `case shape contains unknown or derived fields` | 仅传 workload 允许的输入；不要把 num_groups 等派生值当作输入 |
| 分块整除或容量错误 | 检查外层/内层块关系、数据类型、缓冲区字节数和 slots 数量 |
| `split ncores must equal the selected hardware` | 完整 case 使用的策略核数要与当前硬件一致 |
| `execution policies ... require physical_execution` | 当前是普通模型，使用默认 SimPy；物理模式的选项不能直接套用 |
| `native case hardware override ... requires physical_execution` | 当前普通完整 case 不支持该覆盖；参阅硬件入口表 |
| 输出目录被拒绝或旧结果混入 | 选择新的用户目录，避开系统 `/tmp`，不要把结果写入安装目录 |
| DAG 只生成单点结果 | 检查搜索空间是否存在、是否被全部固定、ms_hooks 是否覆盖了 YAML |
| 搜索候选无法采纳 | 查看 ranked/evaluation 中的 feasible、adoptable 和原因；修正模型或约束 |
| 修改 hooks 后未生效 | 检查文件所在模型目录，重新启动 Python 进程 |
| buffer 复用分析报 `tuple ... has no attribute 'name'` | 当前发行版在教程矩阵乘模型上的已知兼容问题；使用周期统计和 trace 分析 |

## 核数增加却没有加速

先看 `summary.active_cores` 和任务空间。四个不可再拆任务不会因为配置 32 个核
就变成 32 个任务。任务均衡、共享带宽和同步也会影响收益。

## 仿真完成，但某些状态是 not_run

运行仿真、执行数值检查、满足性能预期是不同的操作。未提供对应输入时，部分检查
保持 `not_run` 是正常的，不能把它标为通过，也无需据此认定仿真入口失败。

## 寻优周期与完整 case 周期不同

DAG 的候选评分范围与完整 case 的分核、共享资源和总体评价可能不同。
把选中的参数应用回完整 case，再与相同条件下的基线比较。详见[参数寻优](optimization.md)。

## 如何提供可复现的问题信息

整理包版本、Python/系统版本、调用命令、模型定义、输入 case 和错误消息。
可以附对应运行的结果文件，但先移除个人目录及不希望公开的数据。
使用本 skill 示例复现问题能帮助区分环境、模型和接口支持范围。

## 升级后的硬件与成本检查

`resolve_eval_hardware` 解析算子保存的硬件引用，显式覆盖优先。引用缺失或无法解析时，
检查配置与来源，不按芯片前缀猜另一个 profile；已有模型不因本机不同而重新选型。
新模型按[硬件准备](hardware-preparation.md)确认目标。

实现成本报 hardware／dtype／workload mismatch 或来源变化时，检查最终参数、实际
硬件内容及绑定文件。更新匹配实现的成本定义并重新评价，不能改签名、钳制输入或
覆盖绑定成本绕过检查；字段与历史校准标记的含义见[VF 成本](framework/vf-costs.md)。

## Vec、搬运和构建反馈

| 问题 | 检查与处理 |
| --- | --- |
| `vf_plans` 被原生搜索拒绝 | 使用 [Vec 入口](framework/vec-plans.md)，核对单区域及基线限制 |
| 首计划超容，后续没有结果 | 当前入口记录非法候选后继续后续计划；核对版本、异常、预算及实际 `ledger/complete`，修正真实配置后按需重跑，不从计划列表推断覆盖 |
| 预测缺少操作参数／硬件不匹配 | 补有来源的全部操作及 dtype 参数，核对调用与 profile，不填零绕过 |
| 变换写入超容 | 比较 NZ／stride 物理跨度与每实例容量，见[搬运契约](framework/transfer-contracts.md) |
| 源读完后目的仍不能消费 | 源复用和目的就绪是不同边界，保留完整依赖及消费者保护 |
| 编译状态失效或非法 | 重新绑定源码、构建、工件和候选，见[编译反馈](framework/compile-feedback.md) |
