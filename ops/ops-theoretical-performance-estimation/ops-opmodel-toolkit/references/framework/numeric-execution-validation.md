# 数值验收输入与结果

数值验收用于比较模型输出与独立数学参考。仿真产生周期不意味着已经执行数值检查。
只运行本模型及所选入口实际支持的数值模式。

## 输入

- 在 `semantic.yaml` 声明数学操作、输入输出、dtype、容差和最终状态要求。
- 准备代表输入及边界输入；预期结果来自独立参考。
- 需要执行级数值检查时，提供该版本要求的数值描述、输入绑定与实现身份。
  只有 `vec.cycles` 的节点不能表达其数学输出。
- 变化后的候选需要对应的新验证记录，不能复用另一个候选的通过状态。

数值指令接口可在 DAG 中使用 `numeric` 声明。
支持范围及数据字段取决于指令、执行入口和版本；未支持的转换、布局或运算应保留
明确的 `unsupported`/错误，不能替换为另一条数值路径后宣称同等通过。

## 调用和产物

物理完整 case 可通过命令的 `--gates` 选择支持的检查项目，见
[物理执行](physical-execution.md)。普通模型也可使用公开验证入口：

```python
from accel_model.model.validation import score_case

model, report, streams = score_case(
    "work/my_matmul", backend="both",
    gates=["runtime", "backend_agreement"],
)
print(report["validation"])
print(report["numeric_evidence"], report["qualification"])
```

`score_case(root, parameters=None, backend='both', *, gates=None, performance=False, **policies)`
返回模型、报告和指令流三元组；需要 JSON 时只保存报告，不序列化整个三元组。
普通入口使用当前模型参数，其结果为 `native_aggregate`，不验证物理分核覆盖，
也不替代 `run_native_case(..., derive=True)` 的完整 case 结果。参数覆盖不是完整
shape 的自动分核推导。物理模型按其声明返回物理执行范围。

执行数值检查需在 `operator.yaml:validation.execution_cases` 提供输入、预期结果、
`input_bindings/output_bindings`，并为相关指令给出 `numeric` 声明与语义契约。
已声明的检查不会因 `gates` 列表省略而关闭；缺失的必需检查会报错。
仅请求 runtime 与 backend_agreement 不代表数值通过：未声明执行用例时，
`numeric_evidence` 为未执行，`qualification.verified` 不成立。

读取结果中实际存在的 `validation`、数值报告、输入和候选身份。
`passed/failed/skipped/not_run/unsupported` 分开报告，缺字段不是通过。

## 布局与量化声明

数值视图可使用 `layout: NZ`，同时必须声明 `block_shape: [M0, N0]`。
padding 计入实际容量，读写绑定与真实布局一致；任意部分 NZ 视图、任意地址编码
或自动搜索组合不能据此视为受支持。

`numeric.operation` 可使用 `decode_quantized/encode_quantized`。数据以 uint8
原始编码读写，`reads/writes` 提供 allocation、memory、owner、shape、dtype 等绑定。
以下为解码字段片段，需配套真实 load/store、输入绑定和独立预期值：

```yaml
numeric:
  operation: decode_quantized
  reads:
    - {allocation: encoded, memory: UB, owner: self, shape: [1, 32], dtype: uint8}
  writes:
    - {allocation: decoded, memory: UB, owner: self, shape: [1, 32], dtype: fp32}
  attrs: {format: e4m3}
```

| format | 输入与编码约束 |
| --- | --- |
| `e4m3` | 编码使用 `rounding: rne` |
| `hif8` | 编码使用 `rounding: half_away`；外部 scale 用显式算术声明 |
| `mxfp8_e4m3` | 额外 uint8 E8M0 scale 输入，显式 `axis` 与 `block_size: 32`；不自动求 scale |

编码还须给出 `overflow: saturate` 与 `nan: preserve`；MXFP8 按对应的 E4M3
编码规则处理，尾块沿指定 axis 使用对应 scale。未知格式或舍入应保留错误。
这些显式操作不意味着量化融合 MMAD 或所有算子量化路径均已完成数值适配。

## 使用限制

语义参考通过只说明被测数学样本；执行数值通过只说明当前支持的模型与输入。
模型没有承诺覆盖设备的全部近似运算、特殊 dtype、NaN 行为或实际归约误差。
改变分块、融合、累加顺序、dtype 或实现后，重新按约定容差检查。
设备正确性、模型后端一致性、成本校准和性能收益分别验收。

## Vec 与有限变换

显式 Vec 可验证各分区独立程序及适用的执行存储回放，保存原始输入、实际计划与
验证范围，详见[Vec 接口](vec-plans.md)。无样本或超预算不表示通过，小规模结果
不推广为所有形状或设备实现正确。

有限 ND/NZ、窗口变换、跨步及完整 NoQuant 原子范围见[搬运契约](transfer-contracts.md)。
实际指令须有对应 numeric 读写视图和地址／layout 绑定；部分区域提前可见、任意
布局及逐块原子数值顺序不由普通变换通过推断。编译结果另按[构建反馈](compile-feedback.md)报告。
