# 符号计算参考

## 1. 适用范围和源码定位

本文只说明 GE 的主机侧符号计算。

核心文件：

- `compiler/graph/optimize/symbolic/symbolic_kernel_factory.h`
- `compiler/graph/optimize/symbolic/symbolic_kernel_factory.cc`
- `compiler/graph/optimize/symbolic/symbol_compute_context.h`
- `inc/graph_metadef/exe_graph/runtime/symbolic_tensor.h`
- `compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.cc`
- `tests/ge/ut/ge/graph/optimize/symbolic/symbolic_shape_compute_unittest.cc`
- `tests/depends/op_stub/op_impl/add_op_impl.cc`

实现符号计算前，优先查找同一算子的 `REGISTER_SYMBOLIC_KERNEL(OpType, Func)` 注册和方法实现：

- 如果算子已有符号计算方法，应先复用其输入输出 index、参数解析、SymbolicValue 生成方式、元素布局和输出写回约定。
- 如果只有普通 `InferShape`，只能用它确认 Shape 语义，不能据此推断 SymbolicValue 的元素顺序；此时需要结合算子数学语义和相近 symbolic kernel 单独设计并测试。
- 普通 `InferShape`、符号推导和符号计算可能处于不同调用阶段；不要将普通 Shape 结果或确定整数直接覆盖符号 Shape。

参考实现：

- Shape/Reshape：`compiler/graph/optimize/symbolic/symbolic_kernel/shape.cc`
- 逐元素和广播：`.../symbolic_kernel/binary_elementwise.cc`
- Transpose：`.../symbolic_kernel/transpose.cc`
- Reduce：`.../symbolic_kernel/reduce.cc`
- 复杂切片：`.../symbolic_kernel/stridedslice.cc`

新增计算前先按语义族检查已有 `symbolic_kernel/*.cc`。如果目标算子可以复用同类算子的参数解析、Shape 处理、SymbolicValue 映射和输出写回方式，应合并到已有文件；只有已有文件职责明显不同或合并会降低可维护性时，才新增源文件。

## 2. 职责和调用链

符号计算根据输入 `SymbolTensor` 的 Shape 和 `SymbolicValue` 构造输出 Shape 与元素表达式。它用于下游需要值级符号信息的场景，不替代只生成 Shape 的推导函数。

`gert::InferSymbolComputeContext` 继承 `InferSymbolShapeContext`，并提供：

- `GetConstInputDims(index)`：仅当输入 Shape 的所有维度可取常量时成功，适合线性索引和 stride。
- `GetOutputSymbolTensor(index)`：获取并写入输出 Shape 和 SymbolicValue。

注册使用：

```cpp
REGISTER_SYMBOLIC_KERNEL(MyOp, MyOpSymbolicKernelCompute);
```

查询使用：

```cpp
SymbolicKernelFactory::GetInstance().Create(op_type);
```

## 3. 实现步骤

### 实现前证据门禁

实现算子符号计算前，必须完成以下证据表；任一必填项缺失时停止编码，继续查找目标算子的实际注册和计算实现：

| 检查项 | 必须取得的证据 |
|---|---|
| 算子契约 | ops 仓 `REG_OP`/`OpDef` 文件路径；输入、输出、可选性、属性和 dtype/format |
| 普通算子计算注册 | `REGISTER_SYMBOLIC_KERNEL(OpType, Func)` 的文件路径和注册语句 |
| 普通算子计算实现 | 注册对应 `Func` 的文件路径、函数名和完整输入到输出计算逻辑 |
| 输出映射 | 逐项列出 `output index -> 输出名 -> Shape 来源 -> SymbolicValue 表达式/数量/顺序` |
| 特殊输出 | 常量值、空 SymbolicValue、仅 Shape 输出、可选输出及训练/推理等模式分支 |
| 测试原型 | `test_ops.h` 中是否存在原型，以及与 ops 契约的差异 |
| 测试运行桩 | GE 测试环境中的本地 `REG_OP`/FakeOp 是否与 ops 契约一致 |

门禁规则：

- 同类 symbolic kernel 只用于选择文件、复用骨架和代码风格，不能替代目标算子的计算语义证据。
- 必须阅读 `REGISTER_SYMBOLIC_KERNEL` 对应函数中每一个输出的 Shape 和 SymbolicValue 写入逻辑，尤其检查多输出、仅 Shape 输出、空值和固定常量值。
- 实现与 UT 不能共用未经验证的推测作为期望值；UT 期望必须直接来自上述输出映射表。
- ops 契约、目标计算实现、测试原型或测试桩存在冲突时，先列出差异并修正测试侧定义，再开始编码。

1. 先以 ops 仓中的 `REG_OP`/`OpDef` 注册定义确认输入输出、可选性、属性和 dtype/format，再分析同类算子和已有 `symbolic_kernel/*.cc` 文件，决定复用已有文件还是新增文件；默认优先合并。
2. 校验 context、输入输出数量和必要输入 Tensor。
3. 先定位并阅读同一算子的 `REGISTER_SYMBOLIC_KERNEL(OpType, Func)` 实现；以已有方法确认符号计算的输入输出 index、参数解析和输出写回方式。
4. 对需要线性索引、stride 或分组的算子，用 `GetConstInputDims` 验证 Shape 可确定；动态维度不能强转整数。
5. 获取输入 `GetSymbolicValue()`。空指针表示没有值，空 vector 表示值为空，按算子语义分别处理。
6. 对 axis、perm、shape、slice 等值依赖输入，逐元素 `GetConstValue`，检查 dtype、长度、范围、重复项和负值语义。
7. 输出 Shape 应尽量复用输入 OriginSymbolShape 中的 Expression，不要被确定 Shape 覆盖。
8. 按真实 layout 生成 SymbolicValue：Transpose 使用坐标和 row-major stride；广播使用尾部对齐和 stride；Reduce 使用分组规约；Slice 处理负索引和 mask。
9. 写入 `MutableOriginSymbolShape()`，再用 `SetSymbolicValue` 或 `MutableSymbolicValue()` 写值；检查元素数量和输入/输出元素总数一致。
10. 缺少值、Shape 不确定、值超过实现上限或无法安全映射时返回 `UNSUPPORTED`。

最小骨架：

```cpp
graphStatus MyOpSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  const auto input = context->GetInputSymbolTensor(0U);
  GE_UNSUPPORTED_IF_NULL(input);
  const auto values = input->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(values);
  auto output = context->GetOutputSymbolTensor(0U);
  GE_ASSERT_NOTNULL(output);
  output->MutableOriginSymbolShape() = input->GetOriginSymbolShape();
  output->SetSymbolicValue(ge::MakeUnique<std::vector<Expression>>(*values));
  return GRAPH_SUCCESS;
}
```

## 4. 数据依赖和限制

普通输入通常没有 SymbolicValue；axis、perm、shape、padding、multiples、k 等常量参数通常需要 data dependency。值来源可能是常量输入数据或输入描述中的符号属性，必须检查输入构造和依赖声明。

当前源码对可符号化常量值有数量限制，`symbolic_shape_inference.cc` 中的实现值为 200。计算逻辑还必须防止空 vector、除零、负索引、整数溢出、输出 vector 越界和重复注册。

## 5. 测试和排障

UT 使用 `tests/ge/ut/ge/graph/optimize/symbolic/symbolic_shape_compute_unittest.cc`，通过 `KernelRunContextBuilder` 构造 `InferSymbolComputeContext`，查询 factory 后执行。至少覆盖：

- 正常值表达式和元素顺序
- 广播、Transpose、Reduce、Slice 等布局变化
- 缺少 SymbolicValue、非 const 参数、Shape 不确定
- 非法轴、perm 重复/越界、值数量不匹配和超限

新增算子的 UT 或 ST 用例前，先检查 `test_ops.h` 是否已有该算子的原型定义。若原型不存在，必须先在对应的 `test_ops.h` 中补充与实际算子输入、输出和参数一致的原型，再编写用例；不能在测试中假设不存在的测试接口。

同时检查 `tests/framework/ge_running_env/include/ge_running_env/op_reg.h` 是否存在旧的本地 FakeOp 定义。若存在，必须与 ops 契约核对输入、输出、属性和数量；若不存在，不应仅为满足测试而臆造注册。

如果修改了 `test_ops.h`，在 ST 使用对应算子前必须重新生成 ES 测试接口，并以生成头文件中的真实 wrapper 签名和多输出成员名为准；不能凭算子语义猜测输出字段。

ST 使用 `tests/ge/st/testcase/autofuse/test_symbolic_shape_compute.cc`，通过 `SymbolicShapeSymbolizer::Symbolize` 和 `SymbolicShapeInference::Infer` 检查整图传播及输出描述中的符号信息。

ST 构图优先使用 `test_ops.h` 生成的 `es::` 算子接口，并参考同文件相近算子的输入创建、输出设置和图构建流程。只有生成接口无法表达目标符号计算场景时，才使用 `OP_CFG` 或手工 `OpDesc`。

使用 `SymbolicShapeSymbolizer::Symbolize` 的 ST 必须保证 Data 描述 Shape 与实际输入 Shape 的 rank 一致；符号值或符号 Shape 断言应使用 Symbolize 后生成的实际信息，不能依赖固定符号编号。

新增符号计算时必须同步设计 ST，至少验证真实图中的输出 Shape、SymbolicValue 传播和一个算子特有边界；不能只生成 `symbolic_shape_compute_unittest.cc` 中的直接调用 UT。

每个新增 UT 和 ST 用例定义前必须添加注释，明确写出：

- 测试场景：本用例验证的符号计算语义、元素映射或边界
- 测试输入：输入 Shape、SymbolicValue、参数、属性、format/dtype 等实际构造内容
- 期望输出：输出 Shape、SymbolicValue 的表达式/顺序/数量和错误状态等实际断言结果

注释必须紧邻测试用例，且不能只写泛化描述；应能让读者不查看测试函数正文也了解测试意图。

新增 UT 前必须检查 `tests/depends/op_stub/op_impl/add_op_impl.cc`。若目标算子没有 `InputsDataDependency` 配置，必须按符号计算实际读取的输入值补充；不读取输入值的算子配置为空集合 `{}`。

测试执行门禁：未获得用户明确确认前，不得执行 UT、ST、`tests/run_test.sh` 或已生成的测试二进制。只完成用例编写时，交付中将测试标记为“未运行”。

排障顺序：

1. factory 查找不到：检查 op type、`REGISTER_SYMBOLIC_KERNEL` 和编译源文件是否被包含。
2. 参数 Tensor 为空：检查 data dependency、常量输入和符号属性。
3. 输出值为空：检查上游值、实际执行路径和 `SetSymbolicValue`。
4. 表达式错误：分别打印输入 Shape、输入值、输出 Shape、输出值，核对 layout、轴归一化、广播方向、stride 和数量。

## 6. 计算复核清单

- [ ] SymbolicValue 缺失、为空和非 const 情况处理正确
- [ ] 需要值的输入已声明 data dependency
- [ ] 动态 Shape 未被当作线性索引整数
- [ ] 输出 Shape 保留 Expression，输出值顺序和数量正确
- [ ] 广播、轴、切片、除法、零维和空输入已覆盖
- [ ] factory 注册成功，UT/ST 覆盖正常、边界和失败路径
