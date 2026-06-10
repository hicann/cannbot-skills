# _tiling UT 开发规范

## 一、何时需要 tiling UT

### 判断条件

以下任一条件满足时，必须有对应的 tiling UT：

| 条件 | 源文件位置 | UT 文件位置 |
|-----|-----------|-------------|
| 存在通用 tiling | `op_host/{op}_tiling.cpp` | `tests/ut/op_host/test_{op}_tiling.cpp` |
| 存在 arch20 特定 tiling | `op_host/arch20/{op}_tiling*.cpp` | `tests/ut/op_host/arch20/test_{op}_tiling*.cpp` |
| 存在 arch32 特定 tiling | `op_host/arch32/{op}_tiling*.cpp` | `tests/ut/op_host/arch32/test_{op}_tiling*.cpp` |
| 存在 arch35 特定 tiling | `op_host/arch35/{op}_tiling*.cpp` | `tests/ut/op_host/arch35/test_{op}_tiling*.cpp` |

### 文件位置对照表

| 源文件路径示例 | 对应 UT 文件路径 |
|--------------|-----------------|
| `op_host/abs_tiling.cpp` | `tests/ut/op_host/test_abs_tiling.cpp` |
| `op_host/arch35/abs_tiling_arch35.cpp` | `tests/ut/op_host/arch35/test_abs_tiling_arch35.cpp` |
| `op_host/arch35/abs_tiling.cpp` | `tests/ut/op_host/arch35/test_abs_tiling.cpp` |
| `op_host/arch35/transpose_tiling_arch35.cpp` | `tests/ut/op_host/arch35/test_transpose_tiling.cpp` |

### 源文件命名模式

`arch*` 目录下的源文件命名可能有以下模式：
- `{op}_tiling.cpp` - 无架构后缀
- `{op}_tiling_arch{N}.cpp` - 带架构后缀（如 `_arch35`）
- `{op}_tiling_{variant}.cpp` - 带变体后缀（如 `_base`、`_simt`）

**UT 文件命名应与源文件保持一致**。

---

## 二、测试内容

tiling UT 主要测试数据切分策略和参数计算：

| 测试场景 | 测试目的 | 优先级 |
|---------|---------|--------|
| 不同 dtype (FP16/FP32/BF16) | 验证 dtype 影响 UB 分块策略 | 高 |
| 不同 shape 规模 | 验证数据切分和多核分配 | 高 |
| 不同 SOC 版本 | 验证平台参数差异 | 高 |
| 边界 shape（小于 tile） | 验证单核处理逻辑 | 中 |
| 大 shape（多核） | 验证多核均衡分配 | 中 |

---

## 三、测试用例结构

### 基本框架

```cpp
#include <gtest/gtest.h>
#include "{op}_tiling.h"
#include "../../../op_kernel/{op}_tiling_data.h"
#include "../../../op_kernel/{op}_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace optiling;

class OpTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        cout << "OpTiling SetUp" << endl;
    }
    static void TearDownTestCase() {
        cout << "OpTiling TearDown" << endl;
    }
};

TEST_F(OpTiling, ascend910b_test_tiling_fp16_001) {
    OpCompileInfo compileInfo = {64, 262144, true};
    gert::TilingContextPara para(
        "OpName",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    
    uint64_t expectTilingKey = 0;
    string expectTilingData = "...";
    std::vector<size_t> expectWorkspaces = {...};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, 
                    expectTilingData, expectWorkspaces);
}
```

### 参数说明

**CompileInfo 结构**：
- UB 相关参数（ubSize、blockSize 等）
- coreNum 相关参数
- 其他编译时常量

**TilingContextPara 构造参数**：
1. 算子名称
2. 输入 tensor 列表
3. 输出 tensor 列表
4. CompileInfo 指针

**期望值**：
- `expectTilingKey`：预期的 tilingKey 值
- `expectTilingData`：预期的 tilingData 字符串
- `expectWorkspaces`：预期的 workspace 大小列表

---

## 四、典型测试场景示例

### 4.1 不同 dtype 测试

```cpp
// FP16
TEST_F(OpTiling, ascend910b_test_tiling_fp16) {
    gert::TilingContextPara para(
        "OpName",
        {{{{1, 64, 2, 64}, ...}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {...}, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0, "...");
}

// FP32
TEST_F(OpTiling, ascend910b_test_tiling_fp32) {
    gert::TilingContextPara para(
        "OpName",
        {{{{1, 64, 2, 64}, ...}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {...}, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0, "...");
}

// BF16（仅 910B/910_93/950）
TEST_F(OpTiling, ascend910b_test_tiling_bf16) {
    gert::TilingContextPara para(
        "OpName",
        {{{{1, 64, 2, 64}, ...}, ge::DT_BF16, ge::FORMAT_ND}},
        {...}, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0, "...");
}
```

### 4.2 不同 SOC 版本测试

```cpp
// ascend310p (arch20)
TEST_F(OpTiling, ascend310p_test_tiling_fp32) {
    // 设置对应 SOC 的 compileInfo
    OpCompileInfo compileInfo = {32, 131072, true};
    // ...
}

// ascend910b (arch32)
TEST_F(OpTiling, ascend910b_test_tiling_fp32) {
    OpCompileInfo compileInfo = {64, 262144, true};
    // ...
}

// ascend950 (arch35)
TEST_F(OpTiling, ascend950_test_tiling_fp32) {
    OpCompileInfo compileInfo = {...};
    // ...
}
```

### 4.3 边界 shape 测试

```cpp
// 小 shape（单核处理）
TEST_F(OpTiling, ascend910b_test_tiling_small) {
    gert::TilingContextPara para(
        "OpName",
        {{{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {...}, &compileInfo);
    // 验证 coreNum = 1
}

// 大 shape（多核处理）
TEST_F(OpTiling, ascend910b_test_tiling_large) {
    gert::TilingContextPara para(
        "OpName",
        {{{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {...}, &compileInfo);
    // 验证多核分配策略
}
```

---

## 五、测试命名规范

### 测试类命名

```cpp
class {Op}Tiling : public testing::Test
```

示例：`TanhGradTiling`

### 测试用例命名

格式：`{soc}_test_tiling_{dtype}_{序号}`

示例：
- `ascend910b_test_tiling_fp16_001`
- `ascend910b_test_tiling_fp32_002`
- `ascend910b_test_tiling_bf16_003`

---

## 六、SOC 版本与架构对应

| SOC 版本 | 架构 | compileInfo 特征 |
|---------|------|-----------------|
| ascend310p | arch20 | UB 较小，coreNum 较少 |
| ascend910b | arch32 | UB 较大，BF16 支持 |
| ascend910_93 | arch32 | 同 ascend910b |
| ascend950 | arch35 | 特殊架构参数 |

---

## 七、覆盖率提升策略

### 识别未覆盖的 tiling 代码

通过覆盖率报告定位：
- dtype 分支（if (dtype == DT_FLOAT)）
- shape 条件（if (inputNum > tileDataNum)）
- SOC 版本分支
- tilingKey 选择分支

### 设计针对性测试

| 未覆盖代码类型 | 测试设计 |
|--------------|---------|
| dtype 分支 | 使用对应 dtype 的 shape |
| shape 条件 | 构造满足条件的 shape |
| SOC 分支 | 设置对应 SOC 版本参数 |
| tilingKey 分支 | 构造触发特定 key 的参数 |

---

## 八、常见问题

### Q1: tilingData 字符串如何获取？

运行 tiling 源代码获取实际输出，或查看源码中的 tiling 数据结构。

### Q2: arch35 架构 UT 如何放置？

如果算子在 arch35 上有特殊实现：
```cpp
tests/ut/op_host/arch35/test_{op}_tiling.cpp
```

### Q3: 多个 tilingKey 如何测试？

设计多个测试用例，每个覆盖不同的 tilingKey：
```cpp
TEST_F(OpTiling, test_tiling_key_001) { expectTilingKey = 1; }
TEST_F(OpTiling, test_tiling_key_002) { expectTilingKey = 2; }
```