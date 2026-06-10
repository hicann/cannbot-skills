# op_kernel UT 开发规范

## 一、何时需要 op_kernel UT

### 判断条件

存在 `op_kernel/*.cpp` 内核实现文件时，必须有对应的 op_kernel UT。

### 文件位置

- 源文件：`{算子目录}/op_kernel/{op}.cpp`
- UT 文件：`{算子目录}/tests/ut/op_kernel/test_{op}.cpp`
- AICPU 算子：`{算子目录}/tests/ut/op_kernel_aicpu/test_{op}.cpp`

---

## 二、测试内容

op_kernel UT 主要测试实际计算逻辑的正确性：

| 测试场景 | 测试目的 | 优先级 |
|---------|---------|--------|
| 不同 dtype 计算 | 验证各 dtype 计算精度 | 高 |
| 不同 shape 规模 | 验证各规模数据处理 | 高 |
| 边界 shape | 验证边界条件处理 | 中 |
| 特殊值处理 | 验证 NaN、Inf 等处理 | 中 |

---

## 三、测试流程

op_kernel UT 采用端到端测试模式：

```
1. 执行 Tiling 获取参数
2. 生成测试数据（Python 脚本）
3. 分配内存、加载输入
4. 执行 kernel 核函数
5. 保存输出并验证结果
```

---

## 四、测试用例结构

### 基本框架

```cpp
#include <gtest/gtest.h>
#include "tikicpulib.h"
#include "data_utils.h"
#include "../../../op_kernel/{op}.cpp"

extern "C" __global__ __aicore__ void {op}(GM_ADDR ..., GM_ADDR workspace, GM_ADDR tiling);

class OpTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        // 复制测试数据目录
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
    }
    static void TearDownTestCase() {}
private:
    const static std::string rootPath = "../../../../";
    const static std::string dataPath = rootPath + ".../tests/ut/op_kernel/{op}_data";
};

TEST_F(OpTest, test_case_float32) {
    // 1. Tiling
    OpCompileInfo compileInfo = {...};
    gert::TilingContextPara para(...);
    TilingInfo tilingInfo;
    ExecuteTiling(para, tilingInfo);
    
    // 2. 生成数据
    system("cd ./{op}_data/ && python3 gen_data.py '(256, 33)' 'float32'");
    
    // 3. 分配内存
    uint8_t* input = AscendC::GmAlloc(size);
    ReadFile("./{op}_data/input.bin", size, input);
    
    uint8_t* output = AscendC::GmAlloc(size);
    uint8_t* workspace = AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = AscendC::GmAlloc(tilingInfo.tilingDataSize);
    
    // 4. 执行 kernel
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF({op}, tilingInfo.blockNum, input..., output, workspace, tiling);
    
    // 5. 验证结果
    WriteFile("./{op}_data/output.bin", output, size);
    system("cd ./{op}_data/ && python3 compare_data.py 'float32'");
    
    // 释放内存
    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
```

---

## 五、典型测试场景示例

### 5.1 不同 dtype 测试

```cpp
TEST_F(OpTest, test_case_float32) {
    system("python3 gen_data.py '(256, 33)' 'float32'");
    // ...
}

TEST_F(OpTest, test_case_float16) {
    system("python3 gen_data.py '(256, 33)' 'float16'");
    // ...
}

TEST_F(OpTest, test_case_bf16) {
    system("python3 gen_data.py '(256, 33)' 'bfloat16'");
    // ...
}
```

### 5.2 不同 shape 测试

```cpp
TEST_F(OpTest, test_case_small_shape) {
    system("python3 gen_data.py '(16, 16)' 'float32'");
    // ...
}

TEST_F(OpTest, test_case_large_shape) {
    system("python3 gen_data.py '(1024, 1024)' 'float32'");
    // ...
}

TEST_F(OpTest, test_case_non_aligned) {
    system("python3 gen_data.py '(256, 33)' 'float32'");  // 33 不对齐
    // ...
}
```

---

## 六、测试数据目录结构

op_kernel UT 通常需要配套的数据生成脚本：

```
tests/ut/op_kernel/{op}_data/
├── gen_data.py           # 生成输入数据
├── compare_data.py       # 验证输出数据
├── {dtype}_input.bin     # 输入数据文件
└── {dtype}_output.bin    # 输出数据文件
```

### gen_data.py 示例

```python
import numpy as np

shape = eval(sys.argv[1])  # 如 (256, 33)
dtype = sys.argv[2]        # 如 'float32'

# 生成输入数据
input_data = np.random.randn(*shape).astype(dtype)
input_data.tofile(f'{dtype}_input.bin')

# 计算期望输出（用于 compare_data.py）
expected = np_function(input_data)  # 算子对应的 numpy 函数
expected.tofile(f'{dtype}_expected.bin')
```

### compare_data.py 示例

```python
import numpy as np

dtype = sys.argv[1]

# 读取实际输出和期望输出
output = np.fromfile(f'{dtype}_output.bin', dtype=dtype)
expected = np.fromfile(f'{dtype}_expected.bin', dtype=dtype)

# 比较精度
diff = np.abs(output - expected)
max_diff = np.max(diff)
mean_diff = np.mean(diff)

print(f"Max diff: {max_diff}")
print(f"Mean diff: {mean_diff}")

# 根据 dtype 设置允许误差
tolerance = 1e-5 if dtype == 'float32' else 1e-3
assert max_diff < tolerance, f"Precision check failed"
```

---

## 七、测试命名规范

### 测试类命名

```cpp
class {Op}Test : public testing::Test
```

示例：`TanhGradTest`

### 测试用例命名

格式：`test_case_{dtype}` 或 `test_case_{场景描述}`

示例：
- `test_case_float32`
- `test_case_float16`
- `test_case_small_shape`
- `test_case_non_aligned`

---

## 八、AICPU 算子特殊处理

AICPU 算子使用不同的目录和测试框架：

```cpp
// 文件位置
tests/ut/op_kernel_aicpu/test_{op}.cpp

// 使用 AICPU 测试框架
#include "aicpu_test_common.h"

TEST_F(OpAicpuTest, test_case) {
    // AICPU 特定测试流程
    AicpuOpTest opTest;
    opTest.SetOpName("{op}");
    opTest.AddInputTensor(shape, dtype);
    opTest.SetOutputTensor(shape, dtype);
    opTest.Run();
    opTest.CompareResult();
}
```

---

## 九、覆盖率提升策略

### 识别未覆盖的 kernel 代码

通过覆盖率报告定位：
- dtype 处理分支
- shape 相关循环和条件
- 特殊值处理（NaN、Inf）
- 对齐/不对齐处理

### 设计针对性测试

| 未覆盖代码类型 | 测试设计 |
|--------------|---------|
| dtype 分支 | 使用对应 dtype 测试 |
| shape 条件 | 构造满足条件的 shape |
| 特殊值 | 生成包含特殊值的输入 |
| 对齐分支 | 使用不对齐的 shape |

---

## 十、常见问题

### Q1: 如何生成测试数据？

使用 Python 脚本，配合 numpy 生成输入数据和期望输出。

### Q2: 精度验证标准是什么？

根据 dtype 设置不同容忍度：
- float32: 1e-5
- float16: 1e-3
- bf16: 1e-3

### Q3: AICPU 和 AICore 的 UT 有何区别？

| 类型 | 目录 | 测试框架 |
|-----|------|---------|
| AICore | op_kernel | tikicpulib, ICPU_RUN_KF |
| AICPU | op_kernel_aicpu | aicpu_test_common |

### Q4: 如何测试多输入多输出算子？

```cpp
ICPU_RUN_KF({op}, blockDim, input1, input2, output1, output2, workspace, tiling);
```