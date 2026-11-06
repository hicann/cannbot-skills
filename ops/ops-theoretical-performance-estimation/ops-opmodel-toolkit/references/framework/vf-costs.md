# VF 成本与可选预测接口

VF 成本可以来自实测、Agent 有依据的未实测估算、系数/公式或显式配置的外部预测器。
记录实现、shape/dtype、硬件、单位、来源及适用域；预测值不当作实测样本。

## 常量和公式

```yaml
- vec: {name: measured, cycles: {expr: measured_vf_cycles}, aiv_presplit: true}
- vec: {name: formula, cycles: {expr: rows * cols * vec_factor}}
```

这是两种独立写法，按实际操作选用。第一行假设成本已属于单 AIV；
第二行的 rows/cols 与分摊规则应符合当前模型的 workload。
`costs.yaml` 使用 `source/coefficients/native/vf/sensitivity`，系数说明单位和来源。
公共 VF 成本的 `model: affine` 使用用户声明的系数；未给成本不等于零成本。

## 未实测估算与探索性排序

没有适用实测时，可依据用户 VF 的计算步骤、每次调用工作量、dtype、分块与可用硬件
吞吐／延迟资料估算 `cycles` 常量／公式或 affine 系数，继续初步仿真与搜索。
无需先补测或安装预测插件；确实无法形成可解释成本时，才保留为未知项。

- 记录值／公式、单位、时钟、推导和假设、实现或拟实现步骤、shape/dtype、分块及调用次数。
- 标记“Agent 未实测估算”和未校准状态，说明不确定性；敏感性范围不冒充实测误差。
- 单 AIV 与完整调用的分摊按模型声明处理，不重复计入已建模的外部搬运与等待。
- 保留探索性排序及入口实际返回的资格，不手填 `calibrated`、`prediction_rankable`
  或生产推荐标记。缺源码的设计仍是结构假设，估算不能替代语义、数值或资源检查。
- 不确定性足以反转关键选择时保留候选，优先安排有授权的最小判别实验；无设备时
  交付排序限制。匹配实测到位后只更新对应实现的成本，并重评受影响的基线和候选。

实现 A 的测量只支持 A，不能直接用于新实现 B。估算、预测与实测分别报告；
更新成本不修改硬件带宽或物理队列深度，整算子残差也不能唯一拆出 VF 成本。

## 实现绑定与适用域

本节的实现、源码和实现指纹均指用户算子实现；硬件摘要标识本次实际运行配置。

`costs.yaml:vf.<块名>.<变体>` 的 affine 条目可增加以下字段；这是字段示意，
摘要占位符须由实际内容替换，工作量名称使用该 VF 声明的 measure 或 `invocations`。

```yaml
implementation:
  file: source/kernel.cpp
  symbol: MyVF
  fingerprint: sha256:实际源码摘要
scope:
  hardware_signature: sha256:实际硬件内容摘要
  dtypes: [bf16, fp32]
  workload:
    elements: [1, 4096]
  includes: 单 AIV 单次 VF 计算，不含已独立计费的搬运和等待
  evidence: 对应实现的测量记录及版本
  environment: 编译器与选项、设备频率及计时协议
```

硬件摘要由公开辅助函数计算，输入是本次实际使用的硬件字典：

```python
from accel_model.kernel.core.implementation_cost import cost_hardware_signature

hardware_signature = cost_hardware_signature(hardware)
```

`scope` 要求非空 `hardware_signature/dtypes/workload/includes`；工作量范围是非负整数
闭区间，硬件不能只按 `name` 匹配。有作用域必须有实现身份；实测条目还要求
`evidence/environment`，来源文件须有真实 SHA256。证据引用不等于自动设备验收，
作用域检查也不推断未声明的布局、数据依赖或编译环境是否兼容。

原生成本 `native.<参数名>` 在原有 `refs/unit` 外可一起提供
`implementation/scope/dtype_params`。例如 `dtype_params: [dtype]` 从最终参数读取
类型；此路径的 `scope.workload` 键对应最终运行参数，图 VF 路径对应单次调用 measure。
随后仍用 `vec.cycles: {expr: 参数名}` 引用。硬件、dtype、工作量越界，源码失效或
加载后覆盖绑定成本会被拒绝；重新估算或测量并更新定义，不能钳制输入或改签名绕过。

理论估计可以绑定作用域，保持 `calibrated: false`；无源码的设计使用明确的设计指纹，
不伪造 SHA256。旧 affine 条目的值可复现，但单独的 `calibrated: true` 不再授予
实现绑定后的测量资格。原生历史条目的 `calibrated_declared` 与 `implementation_bound`
分别表示声明和绑定状态，任一字段都不能证明硬件已校准。内联 `vec.cycles` 数字
仍是估计／历史预算，不会自动获得源码绑定。

候选导出会核对 native／affine／VFmodel 实际声明的绑定源码，并保留相对路径。
来源应位于用户算子目录内；来源变更、丢失、绝对或越界路径、与生成定义冲突时拒绝
有效导出。明确的未校准设计假设无需伪造源码。导出后用新目录独立重载、复评；
VFmodel 的外部配置、预测器和工件仍须满足其交付要求，复制源码不等于依赖已齐全。

## 外部预测输入

使用 VFmodel 时提供独立发行目录、绑定与运行环境。配置示例：

```yaml
root: /absolute/path/to/predictor-version
command: [python3, "{root}/predict.py", fast]
bindings: /absolute/path/project/work/vfmodel/bindings.yaml
work_dir: /absolute/path/project/work/.tmp/vfmodel
hardware_names: [ascend_950PR]
target: dav-c310-vec
prediction_mode: strict
timeout_seconds: 120
```

`work_dir` 位于预测器目录外，包含 `.tmp` 路径段，且不在系统 `/tmp` 下。
配置内路径相对配置文件解析；汇编/ELF 路径相对绑定文件解析。
每个绑定选择 assembly 或 elf，可附加 abi_assembly；文件、符号和 implementation
身份须与实际 VF 一致。数据影响控制流时使用真实输入或经确认的输入假设。
`workload` 的 scalar_registers、arguments、parameters、memory_regions 等按预测器接口填写。

在用户模型中启用：

```yaml
plugins:
  vfmodel: /absolute/path/project/work/vfmodel/config.local.yaml
```

原生 vec 的调用声明为：

```yaml
- vec:
    name: add
    vfmodel: add_u1
    vf_workload:
      elements: {expr: N}
    aiv_presplit: true
```

`cycles` 与 `vfmodel` 互斥。`aiv_presplit` 表示单 AIV 工作量；未预切分时提供该模型要求的
`vec_split_dim`。公共图成本可在 `costs.yaml:vf` 给出 `{model: vfmodel, binding: 绑定名}`。

## 输出和错误

预测周期口径为单 AIV 单次调用的 `rv_full_cycles`；核对时钟、target 和硬件匹配。
`strict` 接受合格的 `conditional_prediction`，显式 estimate 模式才接受相应估计结果。
`needs_input/unsupported/budget_exceeded/functional_only/error` 不作为有效性能值。
显式启用后失败不会自动改用零周期或另一成本来源。

结果中的预测身份、假设、缺口和 calibrated 状态随实验保留。
`prediction_rankable` 不表示已校准或可以直接推荐设备实现。
切换版本、绑定、实现或成本来源后，重新运行基线与候选；不混用旧结果。
关闭预测时先恢复合法 cycles/affine 声明，再移除配置。
