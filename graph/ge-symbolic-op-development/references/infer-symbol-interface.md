# InferSymbol 接口参考

## 1. 能力和调用位置

自动融合需要在编译期沿图传播动态 Shape。自定义算子通过 `InferSymbol` 根据输入符号 Shape、属性和必要的符号值写出输出符号信息；符号 Shape 的编译期传播决定后续节点能否继续参与自动融合。

回调在符号化阶段按拓扑顺序调用。符号信息只服务于编译期的融合、循环轴分析、内存规划和 Guard 生成，不写入执行期需要解析的新 OM 字段。

## 1.1 Shape 推导与值推导

两类推导的目标和前置条件不同：

| 类型 | 原理 | 主要输入 | 可以独立成功的条件 |
| --- | --- | --- | --- |
| 符号化 Shape 推导 | 用输入 Shape、属性和约束构造输出维度表达式 | `SymbolShape`、属性、必要的 Shape 参数 | 维度关系和必要参数可确定，或可以用符号表达 |
| 符号化值推导 | 根据输入符号值构造输出逐元素表达式，并保持元素顺序 | `SymbolTensor`、符号值、Origin Shape、属性 | 已声明数据依赖，输入值存在，元素数量和线性映射可确定 |

Shape 推导可以在没有符号值时成功。例如逐元素算子通常只需透传输入 Shape。值推导不能因为 Shape 已知就自动成功；它还必须确认值是否存在、值数量是否与 Shape 对应，以及广播、拼接、规约或索引后的元素顺序。

如果只能完成 Shape 推导，写输出 Shape 并保持输出 Tensor 无值即可。只有能够证明值语义时，才调用输出 Tensor 的值写入接口。

## 1.1 输入材料和证据优先级

开发请求可以直接提供已有 `InferShape` 代码，也可以只描述 Shape 逻辑。两者的使用方式不同：

| 输入材料 | 用途 | 约束 |
| --- | --- | --- |
| `InferShape` 代码 | 提取输入输出关系、属性读取、边界和输出映射 | 仍需用 IR 原型和目标 CANN 版本核对索引、optional/dynamic 输入、数据依赖及发布接口 |
| 文字逻辑 | 在没有现成代码时形成推导草案 | 不能替代 IR 原型、注册定义、数据依赖和版本发布头 |

证据优先级是：IR 原型和注册定义、目标 CANN 版本的发布头与注册库、已有 `InferShape`、可核验的 host/kernel 语义、文字描述。无论输入是文字逻辑还是 `InferShape` 代码，都必须逐项核对发布头、目标版本注册库及注册库入口、输入索引、输出索引和符号化消费链路。若材料之间冲突，应暂停生成具体索引或接口调用，先标记冲突并核对当前版本代码。

## 2. 发布头

自定义算子只依赖目标 CANN 安装包提供的发布头：

| 头文件 | 内容 |
| --- | --- |
| `exe_graph/runtime/infer_symbol_context.h` | `gert::InferSymbolContext`，输入/输出符号 Shape 和符号 Tensor 定位接口 |
| `exe_graph/runtime/symbolic_types.h` | `gert::SymbolShape` 和 `gert::SymbolTensor` |
| `graph/symbolizer/symbolic.h` | `ge::Expression`、`ge::Symbol`、`ge::sym::*`、`EXPECT_SYMBOL_*`、`ASSERT_SYMBOL_*` |

发布头的公开 Shape/Tensor 接口是逐元素接口，不依赖容器内部表示。

## 3. 三种注册方式

### 3.1 宏式

```cpp
#include "exe_graph/runtime/infer_symbol_context.h"
#include "register/op_impl_registry.h"

static graphStatus InferSymbolMyAdd(gert::InferSymbolContext *ctx) {
  // 推导逻辑见后文
}

IMPL_OP_INFERSYMBOL(MyAdd).InferSymbol(InferSymbolMyAdd);
```

注册对象需要按目标 CANN 版本的注册库要求完成链式构造和落表。实现前必须确认目标发布头声明了该入口、目标注册库导出了对应落表能力，并用最小编译样例验证；不能只凭本机编译成功判断目标环境兼容；不要把注册代码放到只在运行期加载的 kernel 二进制中。

### 3.2 OpDef 式

```cpp
class MyOpDef : public OpDef {
 public:
  MyOpDef(const char *name) : OpDef(name) {
    SetInferSymbol(InferSymbolMyOp);
  }
};
```

`SetInferSymbol` 写入回调，`GetInferSymbol` 读取已配置的回调；两者都属于 OpDef 式接口。OpDef 封装的实际宏和构造方式以目标版本的 `register/op_def.h`、`register/op_def_factory.h` 和 `register/op_def_registry.h` 为准。不要凭名称猜测注册宏。

### 3.3 语言无关式能力

```cpp
class MyCustomOp : public ge::BaseCustomOp,
                   public ge::SymbolicInferOp {
 public:
  graphStatus InferSymbol(gert::InferSymbolContext *ctx) override {
    return InferSymbolMyOp(ctx);
  }
};
```

这种方式由框架检测对象是否实现 `SymbolicInferOp` 能力。它适用于使用 `graph/custom_op.h` 的工程，具体对象工厂和算子原型仍按目标版本接口实现。

## 4. 上下文定位接口

### 输入 Shape

```cpp
const gert::SymbolShape *GetInputSymbolShape(size_t index) const;
const gert::SymbolShape *GetOptionalInputSymbolShape(size_t ir_index) const;
const gert::SymbolShape *GetDynamicInputSymbolShape(size_t ir_index,
                                                    size_t relative_index) const;
const gert::SymbolShape *GetRequiredInputSymbolShape(size_t ir_index) const;
```

### 输入和输出 Tensor

输入 Tensor 的符号值只有在算子声明对应数据依赖后才可用：

```cpp
const gert::SymbolTensor *GetInputSymbolTensor(size_t index) const;
const gert::SymbolTensor *GetOptionalInputSymbolTensor(size_t ir_index) const;
const gert::SymbolTensor *GetDynamicInputSymbolTensor(size_t ir_index,
                                                      size_t relative_index) const;
const gert::SymbolTensor *GetRequiredInputSymbolTensor(size_t ir_index) const;
```

输出接口由框架持有，回调负责填写：

```cpp
gert::SymbolShape *GetOutputSymbolShape(size_t index);
gert::SymbolTensor *GetOutputSymbolTensor(size_t index);
```

非法 index、未实例化的可选输入和缺少上游符号信息都可能返回空指针。

## 5. SymbolShape

```cpp
size_t GetDimNum() const;
const ge::Expression &GetDim(size_t index) const;
SymbolShape &SetDimNum(size_t dim_num);
SymbolShape &SetDim(size_t index, const ge::Expression &value);
SymbolShape &AppendDim(const ge::Expression &value);
void Clear();
```

维度是 `ge::Expression`，可以是常量、输入符号或由多个符号构成的表达式。标量是零维 Shape，不要为标量人为添加一个值为 1 的维度。

### Shape 推导检查点

- 透传、广播、拼接、规约、重排和插入维度分别使用对应的 Shape 数学定义。
- 插入维度的负轴通常相对输出 rank 归一化，即输入 rank 加上新增维度后的 rank；不能直接按输入 rank 归一化。
- 负轴合法范围、零维输入和输出 rank 都要单独覆盖。
- Shape 参数为 `-1` 时，区分真实整数常量和带 hint 的符号表达式。对带 hint 的表达式，应通过 Guard 固化该假设；不能只用常量比较判断。
- 文字描述、Shape 代码和 IR 原型不一致时，暂停生成并核对输入输出索引。

## 6. SymbolTensor 值语义

```cpp
const gert::SymbolShape &GetOriginSymbolShape() const;
void SetOriginSymbolShape(const gert::SymbolShape &shape);
bool HasValue() const;
size_t GetValueCount() const;
const ge::Expression &GetValue(size_t index) const;
SymbolTensor &SetValueCount(size_t count);
SymbolTensor &SetValue(size_t index, const ge::Expression &value);
```

必须区分以下状态：

| 状态 | `HasValue()` | `GetValueCount()` | 含义 |
| --- | --- | --- | --- |
| 无值 | `false` | `0` | 没有提供符号值 |
| 显式空值 | `true` | `0` | 已明确提供空列表，例如空 axes |
| 有值 | `true` | `n` | 提供了 n 个符号值 |

读取值前先检查 `HasValue()`，再检查数量。不要用数量为 0 推断“无值”。

### 值推导检查点

- 只有声明数据依赖的输入才应读取符号值；值缺失时返回未提供状态或保留合法的 Shape 推导。
- `HasValue() == false` 表示没有值；`HasValue() == true && GetValueCount() == 0` 表示显式空值。空 axes 等语义可能依赖这一区分。
- 零维 Shape 代表标量，标量 Tensor 仍然有 1 个元素。计算值数量时不能把空维数组当成 0 个元素。
- 需要线性索引时，先确认每个维度和元素数量可确定；不能把符号维度强转成整数。
- 拼接、Gather、规约和广播必须分别验证值的元素顺序。Shape 正确但值顺序错误，仍然属于错误推导。
- 负索引先按被索引轴的实际长度归一化，再检查范围；N-D 索引要按索引最后一维的坐标宽度解释，不能按一维索引处理。

## 7. 返回值契约

### `GRAPH_SUCCESS`

表示本次成功提供符号信息。返回前必须为每个输出写入完整 Shape；只做 Shape 推导时，输出值可以保持无值。

### `GRAPH_NOT_CHANGED`

表示本次没有提供符号信息，与未注册等价。框架不采纳本次部分写入，并继续降级；它不表示输出 Shape 透传。

当前消费链路的实现版本可能尚未完成这一归并逻辑。若目标环境把该返回值记录为错误，应核对 GE 消费侧版本并使用匹配版本的发布包；不要修改算子语义来掩盖版本不一致。

### 其他错误码

表示推导失败，应中断编译并记录节点名和算子类型。强校验宏失败属于此类路径。

## 8. Expression、运算和 Guard

表达式可以通过 `ge::Symbol` 构造常量或变量，并通过 `ge::sym::*` 组合：

```cpp
const ge::Expression out_dim = ge::sym::Max(x_dim, ge::Symbol(1));
```

分支约束使用 `EXPECT_SYMBOL_*`：它根据当前 hint 选择分支，同时记录运行期约束。多个条件使用组合宏：

```cpp
if (EXPECT_SYMBOL_AND(ge::sym::Ge(a, ge::Symbol(1)),
                     ge::sym::Eq(b, c))) {
  // 两个条件都进入约束
}
```

必须成立的条件使用 `ASSERT_SYMBOL_*`。它失败时返回参数错误并终止当前推导。除法、整除、负轴、广播、空输入和维度数量都应明确检查。

### 8.1 Guard 的两种用途

Guard 有两种不同职责，不能混用：

| 类型 | 用途 | 失败时的处理 |
| --- | --- | --- |
| `EXPECT_SYMBOL_*` | 当前 hint 下选择一个合法分支，并记录运行时必须满足的条件 | 选择其他分支或进入未提供路径 |
| `ASSERT_SYMBOL_*` | 检查算子契约、索引范围、除数非零等必须成立的条件 | 返回错误码，中断当前推导 |

例如广播不是一个可以直接用 C++ 布尔值决定的静态条件：

```cpp
if (EXPECT_SYMBOL_EQ(a, b)) {
  out_dim = a;
} else if (EXPECT_SYMBOL_EQ(a, ge::Symbol(1))) {
  out_dim = b;
} else if (EXPECT_SYMBOL_EQ(b, ge::Symbol(1))) {
  out_dim = a;
} else {
  return GRAPH_NOT_CHANGED;
}
```

这里的每个 `EXPECT` 都把当前选择对应的运行时条件记录下来。不能把它改成先读取 hint、再直接写出某个分支结果，否则其他运行时 Shape 可能复用错误的输出表达式。

### 8.2 复合 Guard 必须完整收集条件

不要使用原生 `&&` 或 `||` 组合多个符号条件：

```cpp
// 错误示例：C++ 短路可能让后面的条件不参与 Guard
if (EXPECT_SYMBOL_EQ(a, b) && EXPECT_SYMBOL_GE(c, ge::Symbol(1))) {
  // ...
}
```

使用发布接口提供的复合宏，让所有子表达式一次性进入 Guard：

```cpp
const auto valid = ge::sym::LogicalAnd({
    ge::sym::Eq(a, b),
    ge::sym::Ge(c, ge::Symbol(1)),
});
if (EXPECT_SYMBOL_CHECK(valid, __FILE__, __LINE__)) {
  // 两个条件都被记录
}
```

或者在多个分支中使用 `EXPECT_SYMBOL_AND(...)` 与 `EXPECT_SYMBOL_OR(...)`。例如 `EXPECT_SYMBOL_AND(ge::sym::Eq(axis, ge::Symbol(1)), ge::sym::Ge(length, ge::Symbol(1)))` 会把两个运行时条件一起加入 Guard。复合条件要覆盖完整的合法域，不能只记录 hint 当前恰好满足的一个子条件；禁止用原生 `&&`/`||` 包裹或串联 `EXPECT_SYMBOL_*`。

### 8.3 何时使用 Guard，何时降级

- Shape 等价、广播选择、`-1` 维含义、符号索引方向等依赖运行时输入的条件，应使用 `EXPECT` 并保留对应表达式。
- 算子规定的 axis 范围、除数非零、索引最终合法等不可违反的契约，应使用 `ASSERT`。
- hint 可用但表达式不是整数常量时，不能强转；按能力边界返回 `GRAPH_NOT_CHANGED`。
- hint 不可用时，不得写出未经约束的分支结果；如果仍可用符号表达式描述，就保留表达式，否则返回 `GRAPH_NOT_CHANGED`。
- `EXPECT` 不能把非法输入变成合法分支；非法输入必须走错误码或由算子既有契约处理。

### 8.4 `-1` 和 hint Guard

当 `-1` 可能来自值符号化输入或 hint 时，不能只判断表达式是否为整数常量：

```cpp
const ge::Expression dim = target->GetValue(i);
int64_t hint = 0;
if (dim.IsConstExpr()) {
  // 仅接受整数常量；其他类型返回 GRAPH_NOT_CHANGED
} else if (dim.GetHint(hint) && hint == -1) {
  if (!EXPECT_SYMBOL_EQ(dim, ge::Symbol(-1))) {
    return GRAPH_NOT_CHANGED;
  }
  // 在 Guard 约束下使用“沿用输入维度”的结果
}
```

Guard 必须与输出表达式的选择保持一致。明确结论：**只保证元素总量相等，不能约束 `dim == -1` 的真实维度语义**；如果不登记 `dim == -1` Guard，运行时可能出现元素总量相同但维度排列不同的输入。

### 8.5 Guard 测试矩阵

至少验证以下组合：

- 约束为常量真值时不生成错误分支。
- hint 为真和 hint 为假时分别得到正确输出 Shape 或值表达式。
- 多个条件同时存在时，所有条件都进入 Guard。
- `EXPECT_SYMBOL_AND` 和 `EXPECT_SYMBOL_OR` 的正向、反向条件都能覆盖。
- hint 不可用时不会无 Guard 写出结果。
- `ASSERT` 失败会返回错误，`EXPECT` 失败只改变分支或进入未提供路径。

## 9. 完整示例：广播 Add

```cpp
#include <algorithm>
#include <vector>

#include "exe_graph/runtime/infer_symbol_context.h"
#include "graph/symbolizer/symbolic.h"
#include "register/op_impl_registry.h"

using ge::Expression;
using ge::Symbol;

static graphStatus InferSymbolMyAdd(gert::InferSymbolContext *ctx) {
  const auto *x = ctx->GetInputSymbolShape(0);
  const auto *y = ctx->GetInputSymbolShape(1);
  auto *out = ctx->GetOutputSymbolShape(0);
  if (x == nullptr || y == nullptr || out == nullptr) {
    return GRAPH_NOT_CHANGED;
  }
  const size_t x_rank = x->GetDimNum();
  const size_t y_rank = y->GetDimNum();
  const size_t rank = std::max(x_rank, y_rank);
  std::vector<Expression> dims;
  dims.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    const size_t x_index = i + x_rank >= rank ? i + x_rank - rank : i;
    const size_t y_index = i + y_rank >= rank ? i + y_rank - rank : i;
    const Expression a = x_rank < rank && i < rank - x_rank ? Symbol(1) : x->GetDim(x_index);
    const Expression b = y_rank < rank && i < rank - y_rank ? Symbol(1) : y->GetDim(y_index);
    if (EXPECT_SYMBOL_EQ(a, b)) {
      dims.push_back(a);
    } else if (EXPECT_SYMBOL_EQ(a, Symbol(1))) {
      dims.push_back(b);
    } else if (EXPECT_SYMBOL_EQ(b, Symbol(1))) {
      dims.push_back(a);
    } else {
      return GRAPH_NOT_CHANGED;
    }
  }

  out->SetDimNum(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    out->SetDim(i, dims[i]);
  }
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSYMBOL(MyAdd).InferSymbol(InferSymbolMyAdd);
```

示例中的 `GRAPH_NOT_CHANGED` 表示当前输入组合无法安全提供符号结果；如果该条件属于算子非法输入而不是“不支持符号化”，应使用明确错误码。

## 10. 通用降级原则

符号化推导是编译期优化能力。无法证明时应安全降级，不能为了保持符号传播而制造错误表达式：

- 需要整数的 axis、stride、shape、index、num_segments 等参数，如果表达式不是整数常量，应返回未提供状态，不要执行浮点到整数的隐式转换。
- 输入值来自外部构造或 hint 时，不能只依据数据类型标签判断其表达式类型；应检查表达式本身。
- 可选输入没有连接时，不能因为它的描述缺失或格式标签不同而拒绝整个推导；只有实际参与语义的输入才进入对应校验。
- 一维或标量输入的格式标签通常没有维序变换意义；不要仅凭标签差异阻断其 Shape 推导。真正的多维布局变换仍需按算子契约处理。
- 同一算子的 Shape 推导、值推导和降级路径必须保持相同的负轴、负索引、标量、空值和广播语义。
