---
name: ge-symbolic-op-development
description: >-
  为 GE 自动融合场景开发自定义算子符号化推导。用户需要为自定义算子补
  InferSymbol、注册符号 Shape、修复动态 Shape 下的符号传播中断、检查
  IMPL_OP_INFERSYMBOL/SetInferSymbol/SymbolicInferOp、弱符号部署或
  符号化推导兼容性时使用。触发：自定义算子符号化推导、InferSymbol、
  动态 Shape 符号传播、Guard 或符号值推导。
---

# 自定义算子符号化推导

本 skill 面向自定义算子开发者。目标是在算子包的 host 侧实现符号化推导，让动态 Shape 在编译期从输入沿自定义算子继续传播，使自动融合能够继续分析后续节点并扩大可融合范围。

## 回答输出协议

回答用户问题时，不要只引用某个参考章节或给出笼统原则；根据问题类型逐项说出以下可核对结论：

- **自定义算子场景**：说明符号 Shape 用于编译期传播并影响后续自动融合；在 `op_host/` 新增 `*_symbolic.cpp`；将它加入现有 host 构建，与已有 host 代码一起产出同一个 host so；再说明注册方式。
- **输入材料场景**：明确说文字 Shape 逻辑和 `InferShape` 代码都可以输入，`InferShape` 优先；无论哪种输入，都逐项核对 IR 原型、每个输入索引、每个输出索引、optional/dynamic 输入、数据依赖、目标 CANN 版本发布头、目标版本注册库、注册库入口和符号化消费链路。
- **兼容性场景**：明确说不能只凭本机编译成功判断兼容，必须检查目标版本发布头、注册库入口、弱符号行为和老环境兼容策略。
- **值推导场景**：明确检查 `HasValue()`、`GetValueCount()`、输出 Shape 与期望的一致性、输出元素顺序；不能只说“构造索引表达式”。
- **Guard 场景**：至少给出一个 `EXPECT_SYMBOL_AND(ge::sym::Eq(...), ge::sym::Ge(...))` 或 `EXPECT_SYMBOL_OR(...)` 的具体组合示例，并明确禁止用原生 `&&`/`||` 包裹或串联 `EXPECT_SYMBOL_*`。
- **内部开发场景**：明确说明 GE、metadef、graph-autofusion 内部改动和内部测试资产不属于本 skill，应转对应组件的内部开发流程。

最终回复前必须做一次显式终检。只要问题涉及对应主题，回复中必须逐项出现以下原文要点，不能只用“按目标版本检查”“注意兼容”“检查值映射”等概括语句替代：

- 输入契约：`IR 原型`、`每个输入索引`、`每个输出索引`、`optional/dynamic`、`数据依赖`、`目标 CANN 版本的发布头`、`目标版本注册库`、`注册库入口`、`符号化消费链路`。
- 兼容性：`不能只凭本机编译成功判断兼容`，还要检查目标发布头、注册库入口、弱符号行为和老环境兼容策略。
- 值推导：`HasValue()`、`GetValueCount()`、`输出 Shape 与期望一致`、`输出元素顺序`。
- Guard：给出 `EXPECT_SYMBOL_AND(ge::sym::Eq(...), ge::sym::Ge(...))` 或 `EXPECT_SYMBOL_OR(...)` 的具体形式，并写明禁止原生 `&&`/`||` 包裹或串联 `EXPECT_SYMBOL_*`。

## 处理边界

- 修改范围是自定义算子工程，通常在 `op_host/` 中新增 `*_symbolic.cpp` 并加入现有 host 构建。
- 以算子 IR 原型、已有 `InferShape` 实现、目标 CANN 版本的发布头和注册库为事实来源。自然语言描述只能辅助理解，不能替代代码证据。
- 输入可以是已有 `InferShape` 代码，也可以是对 Shape 数学逻辑的文字描述；优先使用代码，并要求用 IR 原型核对输入输出索引、属性、可选输入和数据依赖。
- 不修改 GE、metadef 或 graph-autofusion 源码，不新增独立 symbolizer 库，不把符号化逻辑放进 kernel 侧。
- GE 内置算子注册、metadef 或 graph-autofusion 内部改动，以及 GE 内部测试资产维护，不属于本 skill；遇到这类需求应转到对应组件的内部开发流程。
- 一个 `InferSymbol` 回调可写输出符号 Shape，也可在数据依赖场景写输出符号值；开发者不需要为同一自定义算子另建独立的符号计算注册。
- 必须区分两类推导：符号化 Shape 推导只根据输入 Shape、属性和可证明的约束生成输出维度；符号化值推导还需要输入值、元素布局和逐元素映射。能推导 Shape 不代表能推导值。
- 符号表达式只在编译期使用。算子包对 symbolizer 的公共表达式 API 使用弱引用，运行机不能调用这些 API。
- `GRAPH_SUCCESS`、`GRAPH_NOT_CHANGED` 和其他错误码的语义必须按发布接口契约处理。若当前安装版本的消费链路未完成 `GRAPH_NOT_CHANGED` 降级，应先核对版本，不能把版本缺陷改写成算子语义。

## 工作流程

### 1. 确认算子契约

用户可以提供以下任一种初始材料：

- **已有 `InferShape` 代码**：作为主要 Shape 语义依据，重点提取输入输出关系、属性读取、边界检查和输出映射。
- **文字逻辑描述**：例如“输出 Shape 等于输入 Shape，最后一维替换为 `axis/2 + 1`”；只能作为推导草案，必须继续核对 IR 原型和已有实现。

两种材料都不能单独证明完整契约。缺少 IR 原型、输入顺序、可选/动态输入、数据依赖或目标 CANN 版本信息时，应先标记待核实，不得猜测接口和索引。

无论输入是文字逻辑还是 `InferShape` 代码，都必须逐项核对：IR 原型、每个输入索引、每个输出索引、optional/dynamic 输入、数据依赖、目标 CANN 版本的发布头、目标版本注册库及注册库入口、符号化消费链路。文字逻辑和 `InferShape` 代码都必须执行这套完整核对，不能只核对目标版本名称。

先检查以下信息：

- 输入、输出、可选输入、动态输入和属性的 IR 顺序。
- 每个输出 Shape 相对输入 Shape 的数学关系。
- Shape 参数、axis、perm、indices 等输入是否声明为数据依赖。
- 输出是否需要符号值，以及符号值的元素顺序。
- Shape 推导和符号值推导是否分别具备足够条件：前者看维度关系，后者还要看数据依赖、值数量、元素顺序和索引是否可确定。
- 输入符号缺失、值无法确定、非法 Shape 和不满足约束时的处理。
- 目标 CANN 版本是否已经提供所选注册入口和消费链路。
- 目标 CANN 版本的发布头、注册库入口和符号化消费链路是否彼此匹配。
- 不能只凭本机编译成功判断兼容；必须按目标 CANN 版本检查发布头、注册库入口、弱符号行为和符号化消费链路。

已有 `InferShape` 是 Shape 语义的重要依据。符号化实现必须保留动态维表达式，不得把未知维度未经证明转换成普通整数。

证据优先级为：算子 IR 原型和注册定义、已有 `InferShape` 实现、host/kernel 中可核验的 Shape 语义、用户文字描述。文字描述与代码冲突时，以代码和当前版本发布头为准。

### 2. 选择注册方式

支持的接口形态如下：

| 方式 | 入口 | 适用工程 |
| --- | --- | --- |
| 宏式 | `IMPL_OP_INFERSYMBOL(OpType).InferSymbol(Func)` | 使用算子实现注册表的自定义算子包 |
| OpDef 式 | `OpDef::SetInferSymbol(Func)` / `GetInferSymbol()` | 通过 OpDef 封装注册的算子工程 |
| 语言无关式 | 继承 `ge::SymbolicInferOp` 并实现 `InferSymbol` | 使用 `graph/custom_op.h` 能力接口的工程 |

三种方式最终表达同一个 `InferSymbol` 能力。使用前应以目标环境的发布头和注册库为准；如果某种入口在目标版本不存在，切换到已发布入口或明确标记该版本不支持。

### 3. 实现回调

在自定义算子工程的 `op_host/` 新增 `*_symbolic.cpp` 后，必须把该文件加入现有 host 构建，与算子已有 host 代码一起产出并安装同一个 host so；不能只新增文件和注册入口而遗漏构建接线。

回调签名为：

```cpp
graphStatus InferSymbolMyOp(gert::InferSymbolContext *ctx);
```

实现规则：

- 用 `GetInputSymbolShape`、可选/动态/必选输入对应的定位接口读取输入符号 Shape。
- 需要读取输入符号值时，先声明数据依赖，再使用对应的 SymbolTensor 定位接口。
- 读取输入和输出指针后立即判空；输入符号缺失时按接口契约返回 `GRAPH_NOT_CHANGED`。
- 用 `GetDimNum`、`GetDim`、`SetDimNum`、`SetDim`、`AppendDim` 和 `Clear` 逐维读写 Shape。
- 标量使用零维 Shape 表示：读取时检查 `GetDimNum() == 0`，写出时使用 `SetDimNum(0)`。
- 使用 `ge::Expression`、`ge::Symbol` 和 `ge::sym::*` 构造表达式；需要常量值时使用表达式提供的常量读取接口并检查返回值。
- 用 `EXPECT_SYMBOL_*` 表达可通过运行时 Shape 约束选择的分支，用 `EXPECT_SYMBOL_AND/OR` 组合多个条件，避免 C++ 短路导致约束缺失。
- 用 `ASSERT_SYMBOL_*` 表达不满足即应终止编译的约束。
- 复合条件必须显式写成 `EXPECT_SYMBOL_AND(ge::sym::Eq(a, b), ge::sym::Ge(c, ge::Symbol(1)))` 或对应的 `EXPECT_SYMBOL_OR`；禁止用原生 `&&`/`||` 包裹或串联 `EXPECT_SYMBOL_*`，因为短路会遗漏后续 Guard。
- 例如多个数据依赖条件应写成 `EXPECT_SYMBOL_AND(ge::sym::Eq(axis, ge::Symbol(1)), ge::sym::Ge(length, ge::Symbol(1)))`，不能只笼统描述“使用复合 Guard”。
- Guard 不是普通的编译期判断：`EXPECT` 选择当前 hint 对应的分支并登记运行时假设，`ASSERT` 检查必须成立的算子契约；不得用 hint 折叠后继续无 Guard 地传播结果。
- `GRAPH_SUCCESS` 返回前，为每一个输出写出完整符号 Shape；透传算子也必须显式写出输出。
- 不要把“没有写输出”当作透传，也不要把 `GRAPH_NOT_CHANGED` 当作输出等于输入。

### 4. 自检

逐项检查：

- 注册名称与 IR 算子类型完全一致。
- 所有输出都被写入，输出维度顺序和 `InferShape` 一致。
- 可选输入使用对应的可选定位接口，动态输入使用 IR index 加相对 index。
- Shape 数据依赖和值数据依赖没有混淆。
- 无值、显式空值和有值三种 SymbolTensor 状态按 `HasValue()` 与 `GetValueCount()` 区分。
- 不能确定输入值时返回未提供状态或保留合法符号表达式，不强转成整数。
- 值推导用例必须同时验证输入值存在、元素数量、输出 Shape 和输出元素顺序；索引表达式构造正确不等于输出顺序已经验证。
- 常量表达式必须检查类型和值域；整数参数只接受整数常量，浮点或有理数不能静默转换为整数。
- 负轴按算子语义规定的 rank 归一化，插入维度通常相对输出 rank；负索引按被索引轴的实际长度归一化，二者不能混用。
- 标量 Shape 是零维，但标量 Tensor 的元素数量是 1；Shape 的 rank 和值符号化的元素数量不能混用。
- 可选输入必须按 IR index 定位；未连接的可选输入不应被当成普通输入参与校验或阻断其他输入的推导。
- 相同算子的 Shape 路径、值路径和降级路径必须使用同一套轴、索引、广播和空输入语义。
- `EXPECT_SYMBOL_AND/OR` 覆盖了需要一起进入 Guard 的全部条件。
- 每个依赖运行时 Shape 或值的分支都能列出对应 Guard；已覆盖 hint 为真、hint 为假、无法取得 hint 和常量输入场景。
- 算子包没有新增对运行机专有库的强依赖。
- 弱符号检查、构建检查和端到端测试使用了目标版本对应的发布包。

### 5. 验证和交付

按 `references/infer-symbol-verification.md` 执行静态检查、动态库检查和动态 Shape 端到端验证。执行任何编译、模型运行、执行机加载或端到端命令前，必须先向用户确认完整命令；未确认时只做静态检查并标记动态验证为“未运行”。

交付内容应包含：

- 修改的自定义算子源文件和构建入口。
- 选择的注册方式及目标 CANN 版本。
- 每个输出的 Shape 公式、值依赖和 Guard 约束。
- 未提供、非法输入和错误返回路径。
- 实际执行的检查命令及结果；未执行的项目标记为“未运行”。
- 发布头和运行环境的版本要求。

## 参考资料

- `references/infer-symbol-interface.md`：发布接口、三种注册方式、回调和符号值语义。
- `references/infer-symbol-verification.md`：算子包结构、弱符号检查、版本兼容和端到端验证。
