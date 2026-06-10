# op_api UT 开发规范

## 一、何时需要 op_api UT

### 判断条件

存在 `op_api/*.cpp` 文件时，必须有对应的 op_api UT。

### 文件位置

- 源文件：`{算子目录}/op_api/aclnn_{op}.cpp`
- UT 文件：`{算子目录}/tests/ut/op_api/test_aclnn_{op}.cpp`

### 多 API 变体

一个算子可能有多个 API 变体，每个变体都需要对应的 UT：

```
op_api/
├── aclnn_ne_scalar.cpp       → test_aclnn_ne_scalar.cpp
├── aclnn_ne_tensor.cpp       → test_aclnn_ne_tensor.cpp
├── aclnn_inplace_ne_scalar.cpp → test_aclnn_inplace_ne_scalar.cpp
└── aclnn_inplace_ne_tensor.cpp → test_aclnn_inplace_ne_tensor.cpp
```

---

## 二、测试内容

op_api UT 主要测试 API 接口的参数校验和兼容性：

| 测试场景 | 测试目的 | 优先级 |
|---------|---------|--------|
| 各 dtype 正常场景 | 验证 dtype 支持范围 | 高 |
| BF16 (仅部分 SOC) | 验证 SOC 版本 dtype 限制 | 高 |
| 空 tensor | 验证空 tensor 边界处理 | 高 |
| 非连续 tensor | 验证 Contiguous 处理 | 中 |
| 参数异常（空指针等） | 验证错误返回码 | 中 |
| dtype cast | 验证不同输入输出 dtype | 中 |
| shape 不一致 | 验证 broadcast/错误处理 | 中 |

---

## 三、测试用例结构

### 基本框架

```cpp
#include <gtest/gtest.h>
#include "opdev/platform.h"
#include "../../../op_api/aclnn_{op}.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;

class l2_{op}_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        cout << "l2_{op}_test SetUp" << endl;
    }
    static void TearDownTestCase() {
        cout << "l2_{op}_test TearDown" << endl;
    }
};

TEST_F(l2_op_test, case_float) {
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    
    auto input = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(...);
    auto output = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnOp, INPUT(input), OUTPUT(output));
    
    uint64_t workspace_size = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}
```

### 参数说明

**TensorDesc 构造参数**：
- shape：tensor 形状
- dtype：数据类型（ACL_FLOAT, ACL_FLOAT16, ACL_BF16, ...）
- format：格式（ACL_FORMAT_ND, ACL_FORMAT_NCHW, ...）

**常用方法**：
- `.Value(vector)`：设置具体值
- `.ValueRange(min, max)`：设置值范围
- 非连续构造：`TensorDesc({shape}, dtype, format, {stride}, offset, {storage_shape})`

**OP_API_UT 宏**：
- 第一个参数：API 函数名
- INPUT：输入 tensor/scalar 列表
- OUTPUT：输出 tensor 列表

---

## 四、典型测试场景示例

### 4.1 不同 dtype 测试

```cpp
// INT8
TEST_F(l2_op_test, test_op_int8) {
    auto input = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnOp, INPUT(input), OUTPUT(output));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

// FLOAT16
TEST_F(l2_op_test, test_op_float16) {
    auto input = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    // ...
}

// FLOAT
TEST_F(l2_op_test, test_op_float) {
    auto input = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    // ...
}

// BF16（仅特定 SOC）
TEST_F(l2_op_test, ascend910B_support_bf16) {
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    auto input = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    // ...
}
```

### 4.2 SOC 版本限制测试

```cpp
// BF16 在 ascend310p 上不支持
TEST_F(l2_op_test, ascend310p_not_support_bf16) {
    op::SetPlatformSocVersion(op::SocVersion::ASCEND310P);
    auto input = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnOp, INPUT(input), OUTPUT(output));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// BF16 在 ascend910b 上支持
TEST_F(l2_op_test, ascend910b_support_bf16) {
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    auto input = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}
```

### 4.3 空 tensor 测试

```cpp
TEST_F(l2_op_test, test_op_empty_tensor) {
    auto input = TensorDesc({2, 3, 0}, ACL_FLOAT, ACL_FORMAT_ND);  // shape 含 0
    auto ut = OP_API_UT(aclnnOp, INPUT(input), OUTPUT(output));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}
```

### 4.4 非连续 tensor 测试

```cpp
TEST_F(l2_op_test, case_non_contiguous) {
    // 非连续 tensor：stride 不等于 shape 的自然顺序
    auto input = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
    auto output = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
    auto ut = OP_API_UT(aclnnOp, INPUT(input), OUTPUT(output));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
```

### 4.5 参数异常测试

```cpp
// 空指针
TEST_F(l2_op_test, test_op_nullptr) {
    auto input = nullptr;
    auto ut = OP_API_UT(aclnnOp, INPUT(input), OUTPUT(output));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
}

// shape 不一致
TEST_F(l2_op_test, test_op_diffshape) {
    auto input = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);  // shape 不匹配
    auto ut = OP_API_UT(aclnnOp, INPUT(input), OUTPUT(output));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// dtype 不支持
TEST_F(l2_op_test, test_op_unsupported_dtype) {
    op::SetPlatformSocVersion(op::SocVersion::ASCEND310P);
    auto input = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnOp, INPUT(input), OUTPUT(output));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}
```

### 4.6 维度限制测试

```cpp
TEST_F(l2_op_test, test_op_error_shape) {
    // 超过最大维度限制
    auto input = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnOp, INPUT(input), OUTPUT(output));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}
```

---

## 五、测试命名规范

### 测试类命名

```cpp
class l2_{op}_test : public testing::Test
```

示例：`l2_tanh_grad_test`、`l2_ne_scalar_test`

### 测试用例命名

| 场景类型 | 命名格式 | 示例 |
|---------|---------|------|
| 正常场景 | `case_{dtype}` 或 `test_op_{dtype}` | `case_float` |
| SOC dtype | `{soc}_support_{dtype}_{soc}` | `ascend910B_support_bf16_910B` |
| 异常场景 | `test_op_error_{场景}` | `test_op_error_shape` |
| 空 tensor | `test_op_empty_tensor` | `test_op_empty_tensor` |
| 非连续 | `case_non_contiguous` | `case_non_contiguous` |
| PR 回归 | `{PR号}_case_{场景}` | `Ascend950PR_89_case_norm_float32` |

---

## 六、SOC 版本设置

### 必须设置的场景

- BF16 dtype 测试（必须设置支持的 SOC）
- dtype 不支持测试（必须设置不支持的 SOC）
- SOC 特定功能测试

### SOC 版本枚举

```cpp
op::SocVersion::ASCEND310P    // 不支持 BF16
op::SocVersion::ASCEND910B    // 支持 BF16
op::SocVersion::ASCEND910_93  // 支持 BF16
op::SocVersion::ASCEND950     // 支持 BF16
```

### 默认 SOC

如果不设置，使用默认 SOC 版本，通常为 ASCEND910B。

---

## 七、覆盖率提升策略

### 识别未覆盖的 op_api 代码

通过覆盖率报告定位：
- dtype 支持列表检查
- SOC 版本分支
- shape/dim 校验
- 空指针检查
- broadcast 逻辑

### 设计针对性测试

| 未覆盖代码类型 | 测试设计 |
|--------------|---------|
| dtype 分支 | 使用对应 dtype |
| SOC 分支 | 设置对应 SOC |
| shape 校验 | 构造对应 shape |
| 空指针 | 使用 nullptr 输入 |
| broadcast | 使用不同 shape 输入 |

---

## 八、常见问题

### Q1: 如何确定 API 支持的 dtype 列表？

查看 `op_api/aclnn_{op}.cpp` 中的 dtype 支持列表定义：
```cpp
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {...};
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {...};
```

### Q2: Scalar 参数如何测试？

使用 `ScalarDesc`：
```cpp
auto scalar = ScalarDesc(1.0f);  // float scalar
auto scalar = ScalarDesc((int64_t)10);  // int scalar
```

### Q3: 多输入多输出如何测试？

```cpp
auto ut = OP_API_UT(aclnnOp, 
    INPUT(input1, input2, scalar), 
    OUTPUT(output1, output2));
```

### Q4: inplace 算子如何测试？

inplace 算子输入和输出是同一个 tensor：
```cpp
auto input = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
auto ut = OP_API_UT(aclnnInplaceOp, INPUT(input), OUTPUT(input));
```

---

## 九、完整测试用例清单模板

一个完整的 op_api UT 应包含以下场景：

| 序号 | 场景 | 测试目的 |
|-----|------|---------|
| 1 | 各 dtype 正常 | dtype 支持覆盖 |
| 2 | BF16 特定 SOC | SOC 版本兼容 |
| 3 | 空 tensor | 边界处理 |
| 4 | 非连续 tensor | 内存布局兼容 |
| 5 | 空指针 | 参数校验 |
| 6 | shape 不一致 | 错误处理 |
| 7 | dtype 不支持 | 错误处理 |
| 8 | 维度超限 | 错误处理 |
| 9 | dtype cast | 类型转换 |
| 10 | 不同 shape | broadcast |