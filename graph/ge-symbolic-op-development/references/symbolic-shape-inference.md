# 符号推导参考

## 1. 适用范围和源码定位

本文只说明 GE 的符号 Shape 推导。

核心文件：

- `compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.cc`
- `compiler/graph/optimize/symbolic/infer_symbolic_shape/op_impl_infer_symbol_shape.h`
- `compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h`
- `inc/graph_metadef/exe_graph/runtime/infer_symbol_shape_context.h`
- `tests/ge/ut/ge/graph/optimize/symbolic/symbolic_shape_infer_func_unittest.cc`
- `tests/depends/op_stub/op_impl/add_op_impl.cc`

实现符号推导前，必须先查找目标算子的普通 Shape 推导配置和实现：

- 注册形式通常为 `IMPL_OP_INFERSHAPE(OpType).InferShape(InferShape)`，也可能出现在链式 `IMPL_OP_INFERSHAPE` 配置中。
- 除现代 gert 风格外，还存在旧式 AICPU 注册风格，二者同为普通 Shape 推导的有效证据：

  ```cpp
  // 旧式 AICPU 风格（常见于 canndev/ops/built-in/op_proto/*.cc）
  IMPLEMT_INFERFUNC(OpType, FuncName) { ... }
  INFER_FUNC_REG(OpType, FuncName);
  ```

  旧式风格的数据依赖用 `op_desc->SetOpInferDepends({field_name})` 声明（`field_name` 为输入名字符串，而非输入 index），常量读取用 `op.GetInputConstData(field_name, tensor)`。排查“算子是否有 Shape 推导实现”时，必须同时检索 ops-math/ops-nn（gert 风格）与 canndev/ops/built-in（旧式 AICPU 风格），不能只搜 `*infershape*` 文件名或 `IMPL_OP_INFERSHAPE` 就下“无实现”的结论。
- 普通 `InferShape` 是符号推导的重要语义参考，用于确认输入输出关系、属性读取顺序、可选输入、动态输入、格式和边界条件。
- 使用者可直接提供目标算子的普通 `InferShape` 代码片段，或用文字描述其核心推导逻辑；其作用等价于定位到的 ops 仓 `InferShape`，可作为语义基线和证据门禁的输入。但同样需要据此补全输入输出 index、属性 index、data dependency 与边界条件，不能只凭片段/描述跳过校验。
- 参考普通 `InferShape` 的算子语义和校验逻辑，但不能直接复制其 `GeShape`/`InferShapeContext` 实现；符号推导必须使用 `InferSymbolShapeContext`、`SymbolShape` 和 `Expression`，并保留可推导的符号维度。
- 若普通 `InferShape` 与其他源码、算子 IR 或测试存在差异，先记录差异并以当前版本实际注册和测试行为为准。

参考实现：

- 简单传递：`compiler/graph/optimize/symbolic/infer_symbolic_shape/infer/identity_like_ops.cc`
- 广播：`.../infer/broadcast.cc`
- Shape 参数：`.../infer/reshape.cc`
- 轴规约：`.../infer/reduce.cc`
- 维度重排：`.../infer/transpose_ops.cc`
- 复杂属性和格式：`.../infer/conv2d.cc`
- 复杂切片：`.../infer/stridedslice.cc` 和 `.../strided_slice_common.h`

新增推导前先按语义族检查已有文件。如果目标算子可以复用同类算子的输入读取、维度处理、约束和注册组织，应合并到已有 `infer/*.cc` 文件；只有已有文件职责明显不同或合并会破坏可读性时，才新增源文件。

## 2. 职责和调用链

符号推导只负责：根据输入 `SymbolShape`、必要的常量参数和属性，生成每个输出的 `SymbolShape`。它不负责逐元素计算输出值。

典型调用链：

1. Shape 符号化阶段将动态维度转换为 `Expression`/`Symbol`。
2. `SymbolicShapeInference::Infer` 为节点构造 `gert::InferSymbolShapeContext`。
3. 通过 `OpImplInferSymbolShapeRegistry` 查找算子推导函数并执行。
4. 输出 Shape 写入输出描述的符号信息，供后续节点继续传播。

推导注册使用：

```cpp
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(MyOp).InferSymbolShape(InferShape4MyOp);
```

查询注册结果使用：

```cpp
gert::OpImplInferSymbolShapeRegistry::GetInstance().GetOpImpl(op_type.c_str());
```

## 3. 实现步骤

### 实现前证据门禁

实现算子符号推导前，必须完成以下证据表；任一必填项缺失时停止编码，继续查找目标算子的实际定义：

| 检查项 | 必须取得的证据 |
|---|---|
| 算子契约 | ops 仓 `REG_OP`/`OpDef` 文件路径；输入、输出、可选性、属性和 dtype/format |
| 普通 Shape 推导注册 | `IMPL_OP_INFERSHAPE(OpType)` 的文件路径和注册语句；或旧式 AICPU 的 `IMPLEMT_INFERFUNC(OpType, FuncName)` + `INFER_FUNC_REG(OpType, FuncName)` 文件路径和注册语句 |
| 普通 Shape 推导实现 | `.InferShape(Func)` 或旧式 `FuncName` 对应函数路径、函数名和完整输出赋值逻辑 |
| 输出映射 | 逐项列出 `output index -> 输出名 -> Shape 公式/来源` |
| 特殊输出 | 固定维度、标量、空 Shape、可选输出、训练/推理等模式分支 |
| 测试原型 | `test_ops.h` 中是否存在原型，以及与 ops 契约的差异 |
| 测试运行桩 | GE 测试环境中的本地 `REG_OP`/FakeOp 是否与 ops 契约一致 |

门禁规则：

- 同类算子只用于选择文件、复用骨架和代码风格，不能替代目标算子的语义证据。
- 必须阅读普通 InferShape 中每一个输出赋值，尤其检查最后一个输出、reserve/workspace 类输出和固定 `[1]` 等特殊 Shape。
- 实现与 UT 不能共用未经验证的推测作为期望值；UT 期望必须直接来自上述输出映射表。
- ops 契约、普通 InferShape、测试原型或测试桩存在冲突时，先列出差异并以 ops 契约与当前普通 InferShape 为准修正测试侧定义，再开始编码。

1. 确认输入输出数量、动态/optional 输入、真实 index、属性顺序、dtype 和 format；以 ops 仓中的 `REG_OP`/`OpDef` 注册定义为准，再对照 `IMPL_OP_INFERSHAPE` 的普通 host infer。GE 测试桩或历史生成代码与 ops 定义冲突时，同步修正测试定义，不反向修改生产语义迁就旧桩。
2. 分析同类算子和已有 `infer/*.cc` 文件，决定复用已有文件还是新增文件；默认优先合并。
3. 先定位同一算子的 `IMPL_OP_INFERSHAPE(OpType).InferShape(InferShape)` 配置，并阅读普通 `InferShape`；将其作为符号推导的语义基线，再把逻辑转换为符号表达式。
4. 用 `GetInputSymbolShape(index)` 读取普通输入 Shape，用 `GetOutputSymbolShape(index)` 获取输出 Shape。
5. 动态输入/输出算子使用 `GetDynamicInputSymbolShape(ir_index, relative_index)`、`GetDynamicInputSymbolTensor(ir_index, relative_index)` 和 `GetComputeNodeOutputNum()`；不能把动态实例索引误当普通输入 index。
6. 需要参数值时读取 `GetInputSymbolTensor(index)->GetSymbolicValue()`；先检查 Tensor、值指针、dtype、长度和每个 Expression 的常量性。
7. 直接复用输入维度 Expression，或用已验证的 `+`、`-`、`*`、`/`、`sym::Max`、`sym::Min`、`sym::Ceiling`、`sym::Mod` 构造输出维度。
8. 对跨输入关系建立 `ASSERT_SYMBOL_EQ/GE/GT/LE/LT/NE` 约束；不能把动态 Expression 强制转成 `int64_t`。
9. 组合多个约束生成 guard 时，优先使用 `EXPECT_SYMBOL_AND(...)`、`EXPECT_SYMBOL_OR(...)` 等封装接口（定义于 `inc/graph_metadef/graph/symbolizer/symbol_checker.h`），避免使用原生 `&&`、`||`。原生运算符会短路，短路分支之后的条件不参与 guard 生成，导致 guard 不完备；封装接口会一次性将所有子表达式纳入 `LogicalAnd`/`LogicalOr`，hint 为 true/false 时分别生成对应的正/反 guard。
10. 显式处理 rank、负轴、重复轴、广播、整除、除零、零维和空输入。
11. 缺少符号信息或参数仍动态时通常返回 `UNSUPPORTED`；明确违反契约的常量参数按同类实现返回 `PARAM_INVALID`。
12. 为所有实际支持的 op type 和别名逐一注册。
13. 常量符号 `Symbol(0)`、`Symbol(1)`、`Symbol(2)` 等改用 `kSymbolZero`、`kSymbolOne`、`kSymbolTwo` 预定义常量（定义于 `compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h`，`ge` 命名空间），避免直接构造魔法常量。
14. 读取输入、输出、属性时，仅当“多输入/多输出/多属性且数字 index 不易区分”（如 index 出现 2、3、5 等）时才把数字索引定义为具名 `constexpr` 常量（如 `constexpr size_t kXIdx = 0U;`、`constexpr size_t kAxisAttrIdx = 0U;`）；仅用 0、1 且语义清晰（如单输入单输出）的场景直接写数字即可，不必抽取。常量统一放在文件顶部匿名命名空间，用常量名避免与已有函数定义冲突，不得放在函数内。

最小骨架：

```cpp
constexpr size_t kXIdx = 0U;
constexpr size_t kOutputIdx = 0U;

graphStatus InferShape4MyOp(gert::InferSymbolShapeContext *context) {
  GE_CHECK_NOTNULL(context);
  const auto x_shape = context->GetInputSymbolShape(kXIdx);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  auto y_shape = context->GetOutputSymbolShape(kOutputIdx);
  GE_ASSERT_NOTNULL(y_shape);
  y_shape->MutableDims() = {x_shape->GetDim(0U), kSymbolOne};
  return GRAPH_SUCCESS;
}
```

## 4. 典型语义

- `Reshape`：0 复制输入维度，最多一个 -1；未知维度由输入元素总数除以已知输出元素数得到，并建立元素总数约束。
- 广播：按尾维对齐；相等或一方为 1 才兼容；输出维度尽量保留符号 Expression。
- Reduce：负轴归一化，检查轴范围和重复轴，按 `keep_dims` 保留或删除维度。
- Conv：同时核对 format、strides、pads、dilations、groups 和输出通道约束，不能默认只有 NCHW。

## 5. 数据依赖和测试

只有配置为 data dependency 的输入才保证携带 SymbolicValue。Shape、axis、perm、padding、multiples、k 等参数输入通常需要该配置；普通数据输入通常只需 Shape。实现前检查算子依赖声明、输入构造逻辑和相近 ST 的初始化。

测试桩数据依赖配置位于 `tests/depends/op_stub/op_impl/add_op_impl.cc`，通过 `IMPL_OP(OpType).InputsDataDependency({...})` 为测试上下文提供输入值依赖信息。新增 UT 前必须为目标算子补充该配置；配置索引必须与符号推导实际读取的输入一致，不读取值的算子使用空集合 `{}`。这与 ST 中针对单个图节点设置的 `OpDesc::SetOpInferDepends({...})` 不同，二者需要按测试入口分别配置。

UT 使用 `tests/ge/ut/ge/graph/optimize/symbolic/symbolic_shape_infer_func_unittest.cc`，构造 `gert::SymbolTensor`、设置 OriginSymbolShape 和参数 SymbolicValue，再通过 `KernelRunContextBuilder` 调用注册函数。至少覆盖正常符号维度、参数缺失/非 const、rank 不匹配、非法轴、零维和空输入，并比较 Expression 结构而非只比较整数结果。

新增算子的 UT 或 ST 用例前，先检查 `test_ops.h` 是否已有该算子的原型定义。若原型不存在，必须先在对应的 `test_ops.h` 中补充与实际算子输入、输出和参数一致的原型，再编写用例；不能在测试中假设不存在的测试接口。

同时检查 `tests/framework/ge_running_env/include/ge_running_env/op_reg.h` 是否存在旧的本地 FakeOp 定义。若存在，必须与 ops 契约核对输入、输出、属性和数量；若不存在，不应仅为满足测试而臆造注册。

如果修改了 `test_ops.h`，在 ST 使用对应算子前必须重新生成 ES 测试接口（通常由构建系统的 ES 生成目标完成），并阅读生成的算子头文件确认真实 wrapper 签名和多输出成员名。输出名可能与 IR 输出名不同，测试必须以生成头文件为准。

ST 使用 `tests/ge/st/testcase/autofuse/test_symbolic_shape_inference.cc`，构造真实图并检查符号 Shape 在上下游节点之间的传播。新增符号推导时必须同步设计 ST 用例，至少覆盖一个正常图场景和一个关键边界/约束场景；不能只增加直接调用推导函数的 UT。

ST 构图优先使用 `test_ops.h` 生成的 `es::` 算子接口，并参考同文件相近算子的输入创建、输出设置、图构建和符号化流程。仅当生成接口不支持目标输入形态、动态输入或特定 IR metadata 时，才使用 `OP_CFG` 或手工 `OpDesc`。

使用 `SymbolicShapeSymbolizer::Symbolize` 的 ST 必须保证每个 Data 节点的描述 Shape rank 与传入 `GeTensor` 的实际 Shape rank 一致；动态维度应在 Data 描述中标记为未知，再由实际输入 Shape 生成符号。符号断言应读取 Symbolize 后输入描述中的实际 `SymbolicDescAttr`，再比较输出是否继承对应输入 Shape，不要硬编码 `s0`、`s1` 等符号编号。

每个新增符号推导方法定义前必须添加完整注释，包含“算子功能”“算子约束”“推导逻辑”“举例”四项，且举例必须与实现一致。四项描述每项独占一行开头；某项内容超宽时在句号、逗号、分号处手动断行（下一行缩进对齐），不要让 clang-format 机械折行把下一描述项（如“【推导逻辑】”“【举例】”）挤到上一行末尾。

每个新增 UT 和 ST 用例定义前必须添加注释，明确写出：

- 测试场景：本用例验证的推导语义或边界
- 测试输入：输入 Shape、参数、属性、format/dtype 等实际构造内容
- 期望输出：每个相关输出的 Shape、Expression 和错误状态等实际断言结果

注释必须紧邻方法或测试用例，且不能只写泛化描述；应能让读者不查看函数正文也了解推导或测试意图。

测试执行门禁：未获得用户明确确认前，不得执行 UT、ST、`tests/run_test.sh` 或已生成的测试二进制。只完成用例编写时，交付中将测试标记为“未运行”。

对动态输入/动态输出算子，UT 必须在 `OpDesc` 中配置对应的 `AddDynamicInputDesc`/`AddDynamicOutputDesc`，并验证动态实例数量与输出数量的约束；不能只追加普通输入 Tensor 来模拟动态列表。

需要 `GetComputeNodeInputNum()`/`GetInputDesc()` 取输入数量或 dtype 的算子（含动态输入或读输入值的算子），UT 必须声明输入 desc：动态输入用 `AddDynamicInputDesc("x", N, true)`，普通输入用 `AddInputDesc("scalar", GeTensorDesc(GeShape(), FORMAT_ND, DT_XXX))`。漏声明会导致 `GetComputeNodeInputNum()`/`GetInputDesc()` 漏算输入、返回错误数量或 nullptr，推导失败——这是测试用例的问题，不是推导实现的问题。

## 6. 常见缺陷与避坑指南

以下为符号推导实现中实际暴露过的生产缺陷与对应正确做法，实现前务必对照规避。

### 6.1 符号维度比较必须用 `ASSERT_SYMBOL_EQ`，不能用整值比较

- 未定值的符号维度（symbolic dim）不能用 `std::equal`、`operator==` 等整值比较（`static_cast` 会抛异常，导致 Infershape 阶段报错/失败）。
- 必须逐维度使用 `ASSERT_SYMBOL_EQ` 做符号等价判断，支持符号维度与常量维度的等价校验。

### 6.2 可选输入必须用 `GetOptionalInputSymbolTensor` 读取

- 可选输入（如 StridedSliceV3 的 axes/strides）不能用 `GetInputSymbolTensor(index)` 读取——缺省时该方法行为不正确。
- 用 `GetOptionalInputSymbolTensor(index)`；返回 nullptr 时跳过读取并使用默认值（如 strides 缺省为 1）。

### 6.3 负索引先归一化再边界检查

- Gather 等算子的负索引是合法语义（从轴末尾计数）。读取后必须先归一化到 `[-dim, dim)` 再做边界检查，否则会产生 `begin() + (-1) * block` 的负偏移迭代器越界解引用（进程 SIGSEGV）。
- 注意 `INT64_MIN` 的溢出防护。

### 6.4 缺省/非法参数对齐 host InferShape 语义

- 参数为 -1 或缺省时（如 matrix_diag_v2 的 num_rows/num_cols），默认推导值必须对齐 host `InferShape` 语义，避免过早判非法（如把 num_rows=-1 提前判为小于 min_num_rows）导致符号推导失败。

### 6.5 含 -1 的 Reshape 未知维度用整除求解，不用符号除法

- 不要用符号除法 `in_shape_size / out_shape_size` 求解未知维度，会产生 Rational 分数表达式（symengine 对 div 的 subs/replace 存在已知缺陷），污染下游推导。
- 用整除性求解（`ResolveIntegralDim`），调用方检查返回码，不整除时返回 `UNSUPPORTED` 回退；多个 -1 属语义非法，统一 `GE_ASSERT_TRUE` 报错 fail-fast。

### 6.6 未知取值场景按 hint 折叠并登记 guard

- Select 条件、StridedSlice 符号索引、Reshape 维度等无法静态判定取值时，可按 hint（运行时真实值）折叠，并经 `EXPECT_SYMBOL_*` 登记运行时 guard。
- 仅靠总量约束 `input_size == known * dim` 拦不住“等元素量排列”（如编译期 `[-1,3]`→`[4,3]`，运行时变 `[2,6]` 仍同元素量），-1 维度必须补 `Eq(dim, -1)` guard。
- hint 不可得分两类处理：分支选择类（StridedSlice 索引折叠）属异常场景，`GE_ASSERT_TRUE` 断言报错；能力边界类（Select 条件、Reshape 值提取）用 `UNSUPPORTED` 优雅降级。两类都不产生无 guard 的原样传播。

### 6.7 输入符号信息缺失用 `GE_UNSUPPORTED_IF_NULL` 优雅降级

- 输入 symbol shape/tensor/value 缺失是运行时能力边界，用 `GE_UNSUPPORTED_IF_NULL` 返回 `UNSUPPORTED` 优雅降级，不要用 `GE_ASSERT_NOTNULL`（会导致进程崩溃）。
- 上下文、属性、输出容器等“框架性异常”保留 `GE_ASSERT`，因为它们不应当缺失。

### 6.8 位掩码运算使用 `1ULL` 与 `uint64_t`

- 位掩码运算字面量用 `1ULL` 而非 `1`，避免对有符号 `int` 位移运算的静态检查告警；掩码位置变量用 `uint64_t`。

### 6.9 取常量值优先用 `GetConstInt`，避免冗余类型 fallback

- `GetConstValue<int64_t>` 对 int32/int64 输入都能成功（SymEngine 内部把整数统一表示为 `Integer`），因此不要写“先试 int64 再 fallback int32”的冗余逻辑。
- 单个标量参数（axis、dimension、fft_length 等）优先用 `SymbolicInferUtil::GetConstInt(tensor, dt, value)`：先 `context->GetInputDesc(idx)->GetDataType()` 取 dtype，再取值，与 `unsortedsegment.cc`/`concatv2d.cc` 一致；多个元素的 shape 类输入直接逐个 `GetConstValue<int64_t>`。
- 需要保留符号维度时直接用 `Expression`（`at(i)`、`back()`、`emplace_back(dim)`），不要 `GetConstValue`。

### 6.10 动态输入实例数用 `GetComputeNodeInputNum`，不要 while 循环统计

- 动态输入算子的实例数用 `context->GetComputeNodeInputNum()`（返回节点实际输入总数，动态输入每个实例、普通输入每个各算一个）获取，输出实例数用 `GetComputeNodeOutputNum()`。
- 纯动态输入算子（如 AddN）直接用 `GetComputeNodeInputNum()`；动态输入 + 普通输入的算子（如 ConcatV2、ForeachNorm）用 `GetComputeNodeInputNum() - 普通输入数` 得到动态实例数。
- 不要用 `while (GetDynamicInputSymbolShape(ir_index, i) != nullptr) ++i` 自己统计实例数（动态输入实例不一定能按此方式暴露，统计结果不可靠）。
- 动态输出实例用 `GetOutputSymbolShape(i)` 按索引访问，无需 `GetDynamicOutputSymbolShape`。
- 输出 Shape 固定（不依赖输入 shape）的算子（如 ForeachNorm 每个输出恒为 `[1]`），不要用 `GetDynamicInputSymbolShape` 访问动态输入实例（测试 builder 下动态输入实例映射可能为空，拿不到实例），直接用输出实例数循环写固定 Shape 即可。

### 6.11 标量参数输入不要强制 rank == 1

- 标量/形状参数输入（如 num_segments、shape 等）可能以 scalar `[]`、`[1]`、`[1,1]` 等多种 rank 传入，不要用 `GetDims().size() == 1` 或 `GetDimNum() == 1` 强制 rank 为 1。
- 用 `IsScalar()`（参考 ConcatV2D）或 `GetSymbolicValue()->size() == 1` 校验单元素，兼容 scalar `[]`、`[1]`、`[1,1]` 输入。

## 7. 推导复核清单

- [ ] 输入/输出 index、属性 index、dtype、format 与 IR/host infer 一致
- [ ] 常量符号使用 kSymbolZero/kSymbolOne/kSymbolTwo，未直接写 Symbol(0)/Symbol(1)
- [ ] 多输入/多输出/多属性且 index 不易区分时，index 已定义为顶部匿名命名空间的具名 constexpr 常量；简单 0/1 场景未过度抽取
- [ ] data dependency 已确认，参数 Tensor 缺失时不会解引用空指针
- [ ] 输入缺失用 GE_UNSUPPORTED_IF_NULL 优雅降级，未用 GE_ASSERT_NOTNULL
- [ ] 符号维度比较使用 ASSERT_SYMBOL_EQ，未用 std::equal/== 整值比较
- [ ] 可选输入用 GetOptionalInputSymbolTensor 读取
- [ ] 动态输入实例数用 GetComputeNodeInputNum 获取，未用 while 循环统计
- [ ] 动态输入/读输入值算子的 UT 已声明输入 desc（AddDynamicInputDesc/AddInputDesc）
- [ ] 注释四要素描述项均独占行开头、未机械折行挤占下一描述项
- [ ] 负索引已归一化到 [-dim, dim) 后再做边界检查
- [ ] 含 -1 维度用整除求解（ResolveIntegralDim），未用符号除法
- [ ] 动态 Expression 未被错误转换为整数
- [ ] 标量参数取常量值用 GetConstInt（通过 dtype）或直接 GetConstValue<int64_t>，未写冗余 int64→int32 fallback
- [ ] 输出 Shape 保留符号表达式并建立必要约束
- [ ] 不支持和非法输入的状态码符合已有实现
- [ ] op type 和别名注册成功，UT 覆盖正常、边界和失败路径
- [ ] 已同步设计 UT 和 ST，ST 覆盖真实图传播而非仅直接调用函数
- [ ] 执行测试前已获得用户明确确认，否则未运行并如实说明
