# ACLNN 代码目录结构参考

## 标准算子项目结构

当进行检查时，代码目录通常包含以下结构：

```
example-abs/
├── op_api/                      # API实现目录
│   ├── aclnn_xxx.h              # API头文件，包含函数声明和注释
│   ├── aclnn_xxx.cpp            # API实现，包含参数检查逻辑
│   ├── xxx.h                    # 内部头文件
│   └── xxx.cpp                  # 内部实现
├── op_host/                     # Host侧实现目录
│   ├── xxx_def.cpp              # 算子定义，包含数据类型和格式支持列表
│   ├── xxx_infershape.cpp       # Shape推导实现
│   ├── arch35/                  # 架构相关实现
│   │   └── xxx_tiling_arch35.cpp  # 架构tiling实现
│   └── config/                  # 配置文件目录
│       └── ascend950/
│           └── xxx_*.json       # 配置文件
├── op_kernel/                   # Kernel实现目录
│   ├── xxx_apt.cpp              # Kernel实现
│   ├── xxx_struct.h             # 数据结构定义
│   └── arch35/                  # 架构相关Kernel
│       └── xxx_dag.h            # DAG实现
├── docs/                        # 文档目录
│   └── aclnnXxx.md              # 待检查的API文档
└── tests/                       # 测试目录
    ├── st/                      # 系统测试
    └── ut/                      # 单元测试
```

## 关键代码文件说明

### op_api/aclnn_xxx.h

API头文件，包含：

- 函数声明（两段式接口）
- 函数功能注释
- 参数说明注释
- 数据类型支持说明

### op_api/aclnn_xxx.cpp

API实现文件，包含：

- 数据类型支持列表（不同平台）
- 数据类型转换映射（复数类型）
- 参数检查逻辑（空指针、dtype、format、shape等）
- 错误码定义

**关键代码位置**：

```cpp
// 数据类型支持列表
static const std::initializer_list<DataType> ASCEND910_DTYPE_SUPPORT_LIST = {...};
static const std::initializer_list<DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {...};
static const std::initializer_list<DataType> ARCH_REGBASE_DTYPE_SUPPORT_LIST = {...};

// 复数类型输入输出映射
static const std::unordered_map<DataType, DataType> COMPLEX_IN_AND_OUT_DTYPE_MAP = {...};

// 参数检查函数
static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out);
static bool CheckFormat(const aclTensor* self, const aclTensor* out);
static bool CheckShape(const aclTensor* self, const aclTensor* out);
```

### op_host/xxx_def.cpp

算子定义文件，包含：

- 输入输出参数定义
- 数据类型支持列表
- 数据格式支持列表
- 芯片配置（AddConfig）

**关键代码位置**：

```cpp
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_BF16, ge::DT_FLOAT, ...})
    .Format({ge::FORMAT_ND, ...});

this->AICore().AddConfig("ascend950", aicoreConfig);
this->AICore().AddConfig("mc62cm12a", aicoreConfig);
```

### op_host/arch35/xxx_tiling_arch35.cpp

架构相关的tiling实现，包含：

- 数据类型判断逻辑
- 不同数据类型的处理分支
- SoC平台判断

**关键代码位置**：

```cpp
// 平台判断
auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
if (npuArch == NpuArch::DAV_2201) { ... }  // Ascend 910B
else if (IsRegBase(npuArch)) { ... }  // Ascend 950等REG_BASE架构
```

## 代码分析检查点

### 1. 函数原型一致性

检查 `op_api/aclnn_xxx.h` 中的函数声明与文档中的函数原型是否一致。

### 2. 数据类型支持

检查 `op_api/aclnn_xxx.cpp` 中的数据类型支持列表：

- `ASCEND910_DTYPE_SUPPORT_LIST` - 昇腾910支持的数据类型
- `ASCEND910B_DTYPE_SUPPORT_LIST` - 昇腾910B支持的数据类型
- `ARCH_REGBASE_DTYPE_SUPPORT_LIST` - 昇腾950等REG_BASE架构支持的数据类型

### 3. 复数类型转换

检查 `op_api/aclnn_xxx.cpp` 中的 `COMPLEX_IN_AND_OUT_DTYPE_MAP`，确认复数输入时的输出类型。

### 4. 芯片支持

检查 `op_host/xxx_def.cpp` 中的 `AICore().AddConfig()` 调用，确认支持的芯片。

### 5. 维度限制

检查 `op_api/aclnn_xxx.cpp` 中的 `ACLNN_MAX_SHAPE_RANK` 常量，确认最大维度限制。

### 6. 非连续Tensor支持

检查 `op_api/aclnn_xxx.cpp` 中的非连续Tensor处理逻辑。
