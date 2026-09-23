---
name: ge-symbolic-op-development
description: >-
  GE 图编译中的符号算子开发与解读。触发：用户需要新增或修改算子的符号推导
  （infer_symbolic_shape/infer）、符号计算（symbolic_kernel），或排查 SymbolShape、
  SymbolicValue、数据依赖、guard、算子注册和符号推导链路问题时使用。覆盖需求分析、
  实现、注册、边界检查、UT/ST 验证和常见问题定位。
---
# GE 符号算子开发

本 skill 用于指导 GE 图编译中的符号推导和符号计算新增、修改与问题定位。详细内容按任务读取：符号推导见 `references/symbolic-shape-inference.md`，符号计算见 `references/symbolic-kernel-compute.md`。

## 处理边界

- 符号推导负责根据输入 `SymbolShape`、必要的常量参数和属性生成输出 `SymbolShape`。
- 符号计算负责根据输入 `SymbolTensor` 的 `SymbolicValue` 构造输出值表达式，同时设置输出 Shape。
- 两者是独立注册、前后衔接的阶段；不要为了“完整”给只需要 Shape 的算子增加无意义的值计算。
- 无法安全推导时返回 `UNSUPPORTED`，不要用虚假的维度或值掩盖信息不足。

## 工作流程

### 1. 需求分析

先阅读目标算子 IR/host infer 和最接近的已有实现。算子输入、输出、可选性、属性和 dtype/format 以 ops 仓中的 `REG_OP`/`OpDef` 注册定义为准；GE 测试桩、历史生成代码或本地旧定义与其冲突时，应按 ops 定义同步修正。

使用者可直接提供目标算子的普通 `InferShape` 代码片段，或用文字描述其核心 Shape 推导逻辑（如“输出 Shape = 输入 Shape，最后一维替换为 fft_length/2 + 1”）；这些片段或描述与 ops 仓 `REG_OP`/`InferShape` 等价，可直接作为需求分析和实现前证据门的输入，不必每次都重新定位 ops 源码。但仍应据此补全输入输出数量、属性 index、data dependency 与边界条件。明确：

- 输入输出数量、动态输入、optional 输入和实际 index
- Shape、axis、perm、shape 参数等输入是否为 data dependency
- 输出 Shape 的数学定义和符号约束
- 是否需要输出 `SymbolicValue`，以及元素布局/顺序映射
- 属性 index、dtype、format、layout 和非法输入语义
- 应返回 `UNSUPPORTED` 还是 `PARAM_INVALID` 的场景

实现算子符号推导前，必须完成 `references/symbolic-shape-inference.md` 中的实现前证据门禁；证据必须来自目标算子的实际定义、测试原型、测试运行桩和测试数据依赖配置，不能用同类算子替代。

完成证据门禁后才进入源码实现。

实现算子符号计算前，必须完成 `references/symbolic-kernel-compute.md` 中的实现前证据门禁；证据必须来自目标算子的实际注册和计算实现，不能用同类算子替代。

完成证据门禁后才进入符号计算实现。

### 2. 源码实现

- 符号推导代码放在 `compiler/graph/optimize/symbolic/infer_symbolic_shape/infer/`。
- 符号计算代码放在 `compiler/graph/optimize/symbolic/symbolic_kernel/`。
- 新增实现前，先分析已有实现中是否存在同类算子；若语义、上下文和测试形态相近，优先合并到已有文件，不要为每个算子机械新增源文件。
- 实现符号推导时，优先查找同一算子通过 `IMPL_OP_INFERSHAPE(OpType).InferShape(InferShape)` 配置的普通 `InferShape` 方法，借此确认算子语义、输入输出关系、属性读取和边界条件。
- 普通 `InferShape` 存在两种注册风格：现代 gert 风格 `IMPL_OP_INFERSHAPE(OpType).InferShape(Func).InputsDataDependency({index})`（ops-math/ops-nn），以及旧式 AICPU 风格 `IMPLEMT_INFERFUNC(OpType, FuncName){...}` + `INFER_FUNC_REG(OpType, FuncName);`（canndev/ops/built-in/op_proto/*.cc）；后者用 `op_desc->SetOpInferDepends({field_name})` 声明数据依赖、用 `op.GetInputConstData(field_name, tensor)` 读常量。两种风格同为普通 Shape 推导的有效证据，不可因只搜到旧式风格就判为“无 Shape 推导实现”。
- 实现符号计算时，优先查找同一算子通过 `REGISTER_SYMBOLIC_KERNEL(OpType, Func)` 注册的已有符号计算方法，借此确认 `SymbolicValue` 的构造、元素顺序和输出写回方式。
- 推导使用 `IMPL_OP_INFER_SYMBOL_SHAPE_INNER(OpType).InferSymbolShape(Func)` 注册。
- 计算使用 `REGISTER_SYMBOLIC_KERNEL(OpType, Func)` 注册。
- 新增符号推导方法必须在方法定义上方添加完整注释，且注释必须包含“算子功能”“算子约束”“推导逻辑”“举例”四项内容；示例必须与实际输入输出和 Shape 推导结果一致。四项描述每项独占一行开头，某项内容超宽时在句号、逗号、分号处手动断行（下一行缩进对齐），不要让 clang-format 机械折行把下一描述项挤到行尾。
- 保留输入维度中的 `Expression`，只在确实需要线性索引时将 Shape 验证为常量。
- 对轴、广播、整除、除零、负索引、零维、空输入和 SymbolicValue 长度做显式校验。
- 使用常量符号 `Symbol(0)`、`Symbol(1)`、`Symbol(2)` 时，改用 `kSymbolZero`、`kSymbolOne`、`kSymbolTwo` 等预定义常量（定义于 `compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h` 的 `ge` 命名空间）。
- 读取输入、输出、属性时，仅当“多输入/多输出/多属性且数字 index 不易区分”（如 index 出现 2、3、5 等）时才把数字索引定义为具名 `constexpr` 常量（如 `constexpr size_t kXIdx = 0U;`、`constexpr size_t kAxisAttrIdx = 0U;`）；仅用 0、1 且语义清晰（如单输入单输出）的场景直接写数字即可，不必抽取。常量统一放在文件顶部匿名命名空间，用常量名避免与已有函数定义冲突，不得放在函数内。
- 生成 guard/符号约束时，优先使用 `EXPECT_SYMBOL_AND`、`EXPECT_SYMBOL_OR`（以及 `ASSERT_SYMBOL_*`、`EXPECT_SYMBOL_*` 等封装接口）组合多个表达式，避免直接使用原生 `&&`、`||` 运算符。原生运算符会短路，导致短路分支之后的条件不参与 guard 生成，无法得到完备的宏；封装接口会一次性将所有子表达式纳入 `LogicalAnd`/`LogicalOr`，hint 为 true/false 时分别生成对应的正/反 guard。
- 符号维度比较用 `ASSERT_SYMBOL_EQ`/`EXPECT_SYMBOL_*`，不要用 `std::equal`、`==` 等整值比较——未定值的符号维度会让整值比较失败或崩溃。
- 输入 symbol shape/tensor/value 缺失时用 `GE_UNSUPPORTED_IF_NULL` 返回 `UNSUPPORTED` 优雅降级，不要用 `GE_ASSERT_NOTNULL`（会导致进程崩溃）；仅上下文、属性、输出容器保留 `GE_ASSERT`。
- 可选输入用 `GetOptionalInputSymbolTensor` 读取，缺省时用默认值，不要用 `GetInputSymbolTensor` 取可选输入。
- 含 -1 的 Reshape 未知维度用整除求解（`ResolveIntegralDim`），不要用符号除法（会产生 Rational 分数表达式污染下游）；更多避坑见 `references/symbolic-shape-inference.md` 第 6 章。

### 3. 测试与复核

- 实现符号推导时，同时设计并补充 UT 和 ST：UT 验证注册函数、Expression 和边界，ST 验证真实图中的符号信息传播。
- 实现符号计算时，同时设计并补充符号计算 UT 和 ST；不能只生成 UT 而遗漏整图 ST。
- 新增算子的 UT 或 ST 用例前，必须先检查 `test_ops.h` 是否已有该算子的原型定义；如果没有，先在对应的 `test_ops.h` 中补充原型，再编写用例，避免测试因算子接口缺失而无法编译。
- 新增 UT 用例时，必须检查 `tests/depends/op_stub/op_impl/add_op_impl.cc` 是否存在目标算子的 `InputsDataDependency` 配置；如果不存在，必须根据符号推导实际读取的输入值补充配置，并确保未参与 Shape 推导的输入配置为空集合。
- ST 构图优先参考同文件已有用例，使用由 `test_ops.h` 生成的 `es::` 算子接口；只有 `es::` 接口无法表达目标场景时，才使用 `OP_CFG` 或手工 `OpDesc`，并说明原因。
- 修改 `test_ops.h` 后，编写或编译依赖该原型的 ST 前必须重新生成 ES 测试接口，并以生成头文件中的真实函数签名和输出成员名为准；不能凭算子语义猜测 wrapper 字段名称。
- ST 使用 `SymbolicShapeSymbolizer::Symbolize` 时，Data 输入的描述 Shape rank 必须与传入 `GeTensor` 的实际 Shape 一致；需要动态符号时将 Data 描述维度设为对应的未知维度，再传入实际 Shape。
- ST 断言符号结果时，应读取 `Symbolize` 后输入 Tensor 的实际 `SymbolicDescAttr` 作为期望值，避免硬编码 `s0` 等可能随图输入顺序变化的符号编号。
- 新增任何 UT 或 ST 用例时，必须在测试用例定义上方添加注释，说明测试场景、测试输入和期望输出；三项内容都要与用例实际构造和断言一致。
- ST 通过 `SymbolicShapeSymbolizer::Symbolize` 和 `SymbolicShapeInference::Infer` 检查整图传播。
- 测试正常、符号维度、边界、非法参数、缺少值、非 const 参数和注册函数查找失败路径。
- 执行任何 UT/ST 或测试二进制前，必须先获得用户明确确认；仅“实现”“补充测试”“验证一下”不等同于确认执行测试。未确认时只分析、修改测试文件或做静态检查，不得运行 `tests/run_test.sh`。
- 获得确认后，读取仓库中的 `docs/zh/build.md`，遵循当前版本的构建和 `tests/run_test.sh` 流程；优先使用 `--ut=ge` 和 `--st=ge` 等文档列出的目标，并记录完整命令和结果。

## 交付要求

回复或交付中说明：

- 修改的源码和测试路径
- 推导与计算分别覆盖的语义和不支持范围
- 注册项、数据依赖和关键约束
- 实际执行的验证命令及结果；未执行项标为“未运行”并说明原因
- 若只完成静态分析，明确证据来自哪些源码文件

## 参考资料

- `references/symbolic-shape-inference.md`：符号推导调用链、上下文 API、Expression 规则和测试要点
- `references/symbolic-kernel-compute.md`：符号计算调用链、SymbolicValue 处理、元素映射和测试要点
