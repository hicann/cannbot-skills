# _infershape UT 开发规范

## 一、何时需要 infershape UT

### 判断条件

存在以下两个条件时，必须有对应的 infershape UT：
1. 存在 `op_host/*_infershape.cpp` 文件
2. **存在 IR 原型文件 `op_graph/*_proto.h`**

> **重要说明**：infershape 用于 graph 模式的 shape 推导，只有在算子有 IR 原型定义（`op_graph/*_proto.h`）时才有实际用途。没有 IR 原型文件的算子，即使有 infershape 实现也暂时不需要 UT。

### 文件位置

- IR 原型文件：`{算子目录}/op_graph/{op}_proto.h`
- 源文件：`{算子目录}/op_host/{op}_infershape.cpp`
- UT 文件：`{算子目录}/tests/ut/op_host/test_{op}_infershape.cpp`

---

## 二、测试内容

infershape UT 主要测试形状推导逻辑的正确性：

| 测试场景 | 测试目的 | 优先级 |
|---------|---------|--------|
| 静态 shape | 验证基本 shape 推导 | 高 |
| 动态 shape (-1) | 验证动态维度处理 | 高 |
| 多输入 broadcast | 验证 broadcast 后 shape | 中 |
| 空 tensor (shape 含 0) | 验证空 tensor 边界处理 | 中 |
| 多输出 | 验证多个输出 shape | 中 |

---

## 三、测试用例结构

### 基本框架

```cpp
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class OpInfershape : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "OpInfershape SetUp" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "OpInfershape TearDown" << std::endl;
    }
};

TEST_F(OpInfershape, infershape_test_static_shape) {
    gert::InfershapeContextPara para(
        "OpName",
        {
            // 输入 tensor 描述
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // 输出 tensor 描述
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
```

### 参数说明

**InfershapeContextPara 构造参数**：
1. 算子名称（字符串）
2. 输入 tensor 列表：每个包含 shape range、dtype、format
3. 输出 tensor 列表：shape range 为空（待推导）

**Shape Range 格式**：`{{最小shape, 最大shape}, ...}`
- 静态 shape：最小和最大相同，如 `{{3, 4}, {3, 4}}`
- 动态 shape：使用 -1 表示动态维度，如 `{{5, -1}, {5, -1}}`

---

## 四、典型测试场景示例

### 4.1 静态 shape 测试

```cpp
TEST_F(OpInfershape, infershape_test_static) {
    gert::InfershapeContextPara para(
        "TanhGrad",
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND});
    
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
```

### 4.2 动态 shape 测试

```cpp
TEST_F(OpInfershape, infershape_test_dynamic) {
    gert::InfershapeContextPara para(
        "TanhGrad",
        {
            {{{5, -1}, {5, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, -1}, {5, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND});
    
    std::vector<std::vector<int64_t>> expectOutputShape = {{5, -1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
```

### 4.3 空 tensor 测试

```cpp
TEST_F(OpInfershape, infershape_test_empty) {
    gert::InfershapeContextPara para(
        "OpName",
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND});
    
    std::vector<std::vector<int64_t>> expectOutputShape = {{0}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
```

---

## 五、测试命名规范

### 测试类命名

```cpp
class {Op}Infershape : public testing::Test
```

示例：`TanhGradInfershape`

### 测试用例命名

格式：`{op}_infershape_test{序号}_{场景描述}`

示例：
- `tanh_grad_infershape_test1`
- `tanh_grad_infershape_test2_dynamic_shape`

---

## 六、覆盖率提升策略

### 识别未覆盖的 infershape 代码

通过覆盖率报告定位：
- shape 分支（if (shape.GetDim(0) > 0)）
- dtype 相关分支
- 异常处理分支

### 设计针对性测试

| 未覆盖代码类型 | 测试设计 |
|--------------|---------|
| 特定 shape 条件 | 构造满足条件的 shape |
| dtype 相关分支 | 使用对应 dtype |
| 异常返回 | 构造非法参数 |

---

## 七、常见问题

### Q1: infershape 源文件没有对应 UT 怎么办？

**解决方案**：
1. 分析 infershape 源代码的 shape 推导逻辑
2. 设计覆盖主要路径的测试用例
3. 创建 `tests/ut/op_host/test_{op}_infershape.cpp`

### Q2: 动态 shape 如何测试？

使用 -1 表示动态维度：
```cpp
{{{dim1, -1}, {dim1, -1}}, dtype, format}
```

### Q3: 多输出算子如何测试？

提供多个输出描述和期望 shape：
```cpp
{
    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
    {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
}
expectOutputShape = {{shape1}, {shape2}}
```