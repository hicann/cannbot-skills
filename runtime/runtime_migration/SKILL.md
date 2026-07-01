---
name: runtime_migration
description: CUDA 应用迁移到 CANN 平台指南，仅用于用户自身合法拥有或已获授权的代码资产。当用户请求将其有权处理的应用程序迁移到华为昇腾 NPU、询问迁移可行性分析、或需要帮助改写相关代码为 CANN Runtime 时触发此技能。
---
# CUDA 应用迁移到 CANN 平台指南

## 授权与使用边界

本技能仅用于帮助用户将其自身合法拥有或已获授权的代码资产延伸与适配至 CANN 计算架构。用户应确保对被适配代码享有合法权利，或已取得相应授权。

本技能不用于、也不鼓励在未获授权的情况下复制、翻译或转换第三方代码。若用户明确表示代码属于第三方且未取得授权，或请求规避许可证、版权声明、访问控制、商业限制等，应拒绝执行迁移，并建议用户先取得权利人的许可或授权。

## 概述

在满足上述授权与使用边界的前提下，本技能用于指导将 CUDA 应用程序迁移到华为昇腾 NPU (CANN Runtime) 平台。基于 `assets/src` 目录下的兼容层源码，提供两种迁移方式：

1. **兼容层方式**：替换头文件并链接兼容库，最小代码修改
   说明：默认使用兼容层方式,除非用户明确要求“直接迁移方式、直接改成 CANN API”， 并遵守以下约束：
    - **尽量保持用户源码不变**：优先只替换 include、构建配置和 CUDA 编译器专用语法；不要无故重写控制流、变量名、输出文本、错误处理风格。
    - **不要主动引入新风格**：原代码没有 `CUDA_CHECK`、RAII、异常、封装类、`std::vector` 改写时，不要为了“更规范”而加入。
    - **CANN 细节不泄漏到业务源码**：兼容层方式下，业务源码不应直接 include `acl/acl.h`、`aclnnop/*.h`，也不应自己创建/销毁 `aclTensor`、`aclScalar`、workspace。
    - **缺失能力补兼容层，不堆到样例里**：遇到 `cann_runtime_compat.h` 没覆盖的能力，优先在 `assets/src` 下补兼容 API/适配函数，再让业务代码调用该 compat 接口。
    - **明确区分 Runtime 与算子**：`cudaMalloc/cudaMemcpy/cudaFree` 是 Runtime API，可由兼容层映射；`kernel<<<...>>>` 是 CUDA 编译器语法，代表设备侧算子/Kernel 执行，必须按用户目标选择处理方式。
    - **算子迁移选择权交给用户**：当源代码包含 `kernel<<<...>>>` 时，先确认用户是否希望把具体算子迁移为 Ascend C/SIMT 设备侧实现。用户选择“要转 / 需要真实 NPU 计算 / 需要性能验证”时，调用并遵循 `ops-lab/cuda2ascend-simt/SKILL.md`；用户选择“不转 / 只验证 Runtime API / 快速跑通”时，使用 `cudaCompatLaunchHostKernel()` 做 Host fallback，并明确记录这不是 NPU kernel 执行。
    - **每次迁移都要编译并运行最小验证**：不能只改代码不验证；如机器有坏卡或需指定设备，用环境变量如 `ASCEND_RT_VISIBLE_DEVICES` 限制可见设备。

2. **直接迁移方式**：将 CUDA API 调用改写为 CANN Runtime API, 
  说明： 直接迁移方式适用于需要深度适配或优化的场景，或应用使用部分不支持 API 需自定义实现的情况。直接迁移方式下，程序需要将 CUDA API 调用改写为 CANN Runtime API 调用，并处理参数差异、错误码映射等细节。直接迁移方式的改写步骤包括：
   - API 映射对照：参考 `references ` 中的 API 支持状态表、接口文档等文件，了解哪些 CUDA API 有对应的 CANN API 支持，不支持的接口如何处理，并查看 `assets/src` 中的实现示例。
   - 类型映射：了解 CUDA 类型与 CANN 类型的映射关系，如 `cudaStream_t` 映射为 `aclrtStream`。
   - 错误码转换：参考 `references/error_mapping.md` 进行错误码映射，将 CANN 错误码转换为 CUDA 错误码。  
   - 初始化/清理：CANN 需要显式初始化和清理，改写代码时需要添加 `aclInit` 和 `aclFinalize` 调用。
   - 必要时联网查询 CANN API 文档，了解最新的接口变更和使用建议。


## 迁移流程

按照以下步骤执行迁移：

### Step 1: 使用边界检查

用户应确保对所转换的代码享有合法权利，或已取得相应授权。一般情况下，不因授权状态重复询问用户，按正常迁移流程继续。

当用户表示代码可能属于第三方、请求处理闭源/受限材料，或上下文出现明显疑似侵权、许可不明、访问受限等风险信号时，应先简短询问权利来源或授权情况；在用户确认具备合法权利或授权后，方可继续迁移。

当用户明确表示代码属于第三方且未取得授权，或请求规避许可证、版权声明、访问控制、商业限制，或要求移除、弱化、隐藏版权/许可证/NOTICE/作者署名等归属信息时，应拒绝执行相关迁移或改写，并建议用户先取得权利人的许可或授权。

迁移过程中不得移除、弱化、隐藏或规避原代码中的版权、许可证、NOTICE、作者署名或其他归属信息。

### Step 2: 迁移可行性分析

分析应用程序使用的 CUDA API 是否有对应的 CANN 接口支持。

#### 2.1 识别 CUDA API 使用情况

使用以下方法扫描用户代码：

```bash
# 扫描 CUDA Runtime API 调用
grep -E "cuda[A-Z][a-zA-Z]+\(" --include="*.c" --include="*.cpp" --include="*.cu"

# 扫描 CUDA Driver API 调用
grep -E "cu[A-Z][a-zA-Z]+\(" --include="*.c" --include="*.cpp" --include="*.cu"

# 扫描头文件依赖
grep -E "#include.*cuda(_runtime|\.h)" --include="*.c" --include="*.cpp" --include="*.cu"
```

#### 2.2 API 支持状态检查

对照以下支持状态表判断可行性：


| API 类别            |  支持状态  | 说明                                                      |
| ------------------- | :---------: | --------------------------------------------------------- |
| **设备管理**        | ✅ 完全支持 | cudaGetDeviceCount, cudaSetDevice 等 16 个 API            |
| **内存管理**        | ✅ 完全支持 | cudaMalloc, cudaMemcpy 等 23 个 API                       |
| **流管理**          | ✅ 完全支持 | cudaStreamCreate, cudaStreamSynchronize 等 14 个 API      |
| **事件管理**        | ✅ 完全支持 | cudaEventCreate, cudaEventRecord 等 7 个 API              |
| **IPC**             | ✅ 完全支持 | cudaIpcGetMemHandle 等 5 个 API (注意 opaque handle 约束) |
| **库管理**          | ✅ 完全支持 | cudaLibraryLoadFromFile 等 5 个 API                       |
| **内存池**        | ⚠️ 不支持 | cudaMemPoolCreate 等返回 cudaErrorNotSupported            |
| **Texture/Surface** |  ❌ 不支持  | 需额外适配层                                              |
| **Driver VMM**      | ✅ 完全支持 | cuMemCreate, cuMemMap 等 14 个 API                        |

#### 2.3 输出可行性分析报告

报告包含：

- 使用到的 CUDA API 清单
- API 支持状态（支持/不支持/Mock）
- 不支持 API 的替代方案建议
- 迁移可行性结论

---

### Step 3: 迁移方式选择

根据分析结果，推荐或让用户选择迁移方式。

如果代码中存在 CUDA Kernel 启动语法 `kernel<<<...>>>(...)`，迁移方式选择必须额外记录一个“算子执行路径”决策：

| 用户目标 | 算子执行路径 | 执行要求 |
| -------- | ------------ | -------- |
| 真实 NPU 设备侧计算、性能评估、尽量一比一迁移 CUDA kernel | Ascend C/SIMT 算子迁移 | 先读取并遵循 `ops-lab/cuda2ascend-simt/SKILL.md`，按其模式、计划、验证和降级门要求执行 |
| 快速验证 CUDA Runtime API 兼容层、数据搬运、stream/event 流程和结果正确性 | Host fallback | 将 CUDA kernel body 的等价 CPU 实现隔离成函数，通过 `cudaCompatLaunchHostKernel()` 调用 |

选择规则：

- 用户明确说“转 Ascend C / 转 NPU 算子 / 不要 CPU fallback / 要真实性能”时，选择 Ascend C/SIMT 算子迁移。
- 用户明确说“先跑通 / 只验证 Runtime / 不转算子 / 可以 CPU fallback”时，选择 Host fallback。
- 用户没有明确选择且任务目标是批量兼容性验证时，默认 Host fallback，但报告中必须醒目标注“算子未迁移到 NPU”。
- 用户没有明确选择且任务目标涉及性能、设备计算正确性或最终交付质量时，先简短询问用户选择，不要擅自把设备计算降级为 Host fallback。

#### 3.1 兼容层方式（推荐）

**适用场景**：

- 应用主要使用已支持的 CUDA Runtime API
- 希望最小化代码修改
- 快速验证迁移可行性

**优点**：

- Drop-in replacement，仅替换头文件
- 无需修改业务逻辑代码
- 自动初始化/清理（constructor/destructor）

**修改点**：

```c
// 原代码
#include <cuda_runtime.h>

// 迁移后
#include "cann_runtime_compat.h"
```

**编译链接**：

```bash
# 编译兼容层库（在 assets/src 目录）
cd assets/src
mkdir -p build && cd build
cmake .. && make

# 编译并链接用户应用
cd your_app_dir
gcc -I/path/to/runtime_migration/assets/src \
    your_code.c \
    -L/path/to/runtime_migration/assets/src/build -lcudacompat \
    -L~/Ascend/cann/lib64 -lacl_rt \
    -o your_app
```

#### 3.2 直接迁移方式

**适用场景**：

- 需要深度适配或优化
- 应用使用部分不支持 API，需自定义实现
- 希望直接使用 CANN Runtime 特性

**修改点**：

```c
// 原代码
cudaMalloc(&ptr, size);
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);

// 迁移后
aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
```

#### 3.3 混合方式

部分代码使用兼容层，部分代码直接调用 CANN API，不推荐此方式。

---

### Step 4: 应用代码改写迁移

根据选择的迁移方式执行代码改写。

#### 4.1 兼容层方式改写步骤

##### 4.1.1 头文件替换

```c
// 替换 CUDA 头文件
// 原：#include <cuda_runtime.h>
// 原：#include <cuda.h>
// 新：#include "cann_runtime_compat.h"
```

##### 4.1.2 初始化适配

兼容层自动初始化，但静态链接需添加：

```c
#include "cann_runtime_compat.h"

int main() {
    CUDA_COMPAT_FORCE_LINK();  // 仅静态链接时需要
    // ... 业务代码 ...
}
```

##### 4.1.3 错误处理适配

```c
// 使用 CUDA_CHECK 宏
CUDA_CHECK(cudaMalloc(&ptr, size));

// 或使用原有 CUDA 错误检查逻辑（兼容层返回 cudaError_t）
cudaError_t err = cudaMalloc(&ptr, size);
if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
}
```

##### 4.1.4 Kernel 启动适配

CUDA Kernel 启动语法需要单独处理：

```c
// 原 CUDA Kernel 启动
kernel<<<grid, block, shared_mem, stream>>>(args);

// 不能由普通 C++ 兼容头直接保留；必须根据用户选择改写为：
// A. Ascend C/SIMT 设备侧算子
// B. Host fallback 兼容性验证函数
```

**路径 A：迁移为 Ascend C/SIMT 设备侧算子**

当用户选择迁移具体算子，或请求真实 NPU 执行/性能验证时：

- 立即读取并遵循 `ops-lab/cuda2ascend-simt/SKILL.md`
- 由 `cuda2ascend-simt` 负责 CUDA kernel body、device helper、launch policy、dtype/shape 分支、验证计划和降级门
- 本技能只负责 host 侧 CUDA Runtime 到 CANN/兼容层的迁移边界，以及与算子工程的调用衔接
- 不要用 `cudaCompatLaunchHostKernel()` 替代核心设备计算，除非用户在 `cuda2ascend-simt` 的降级门后明确接受该降级
- README/报告必须说明算子是否真正运行在 Ascend 设备侧，以及验证硬件、构建命令、运行命令和结果

常见 CANN 算子编写和启动方式包括：

```c
// 1. 编写 Ascend C Kernel 或使用算子库
// 2. 通过 aclrtBinaryLoadFromFile 加载编译后的二进制
// 3. 通过 aclrtBinaryGetFunction 获取函数句柄
// 4. 通过 aclrtLaunchKernel 启动
```

如果项目使用 Ascend C SIMT 单源样例形态，也可以按 `cuda2ascend-simt` 的规则使用 SIMT kernel launch 语法；两种方式都必须以真实设备侧执行和验证证据为准。

**路径 B：使用 Host fallback 验证 Runtime 兼容性**

当用户选择不迁移具体算子，或当前目标只是最小化验证 Runtime API，且暂时没有 Ascend C Kernel/算子二进制时：

- 不要把 fallback 计算直接散落在主流程里
- 将 CUDA kernel body 的等价 Host fallback 隔离成独立函数
- 通过 `cudaCompatLaunchHostKernel()` 调用该 fallback，保留主流程中的“launch”语义
- README 必须明确说明该路径不是 NPU kernel 执行，后续如需真实性能/设备计算，需要改写为 Ascend C Kernel 或 CANN 算子

##### 4.1.5 IPC 适配注意事项

CANN IPC 使用 opaque handle 而非 POSIX fd：

```c
// 跨进程传递 IPC handle 必须使用共享内存
// 不能使用 UNIX socket 传递文件描述符

// Producer
cudaIpcMemHandle_t handle;
cudaIpcGetMemHandle(&handle, d_ptr);

// 通过 shm_open/mmap 传递 handle
int shm_fd = shm_open("/ipc_shm", O_CREAT|O_RDWR, 0666);
ftruncate(shm_fd, sizeof(cudaIpcMemHandle_t));
void* shm_ptr = mmap(NULL, sizeof(cudaIpcMemHandle_t), 
                     PROT_READ|PROT_WRITE, MAP_SHARED, shm_fd, 0);
memcpy(shm_ptr, &handle, sizeof(cudaIpcMemHandle_t));

// Consumer
cudaIpcMemHandle_t handle;
memcpy(&handle, shm_ptr, sizeof(cudaIpcMemHandle_t));
cudaIpcOpenMemHandle(&d_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
```

#### 4.2 直接迁移方式改写步骤

##### 4.2.1 API 映射对照


| CUDA API                | CANN API                      | 参数映射                             |
| ----------------------- | ----------------------------- | ------------------------------------ |
| `cudaMalloc`            | `aclrtMalloc`                 | size, 添加 ACL_MEM_MALLOC_HUGE_FIRST |
| `cudaFree`              | `aclrtFree`                   | 直接映射                             |
| `cudaMemcpy`            | `aclrtMemcpy`                 | 增加 count 参数（dstSize, srcSize）  |
| `cudaMemcpyAsync`       | `aclrtMemcpyAsync`            | 同上 + stream                        |
| `cudaStreamCreate`      | `aclrtCreateStreamWithConfig` | 需配置参数                           |
| `cudaEventCreate`       | `aclrtCreateEventExWithFlag`  | 需标志参数                           |
| `cudaSetDevice`         | `aclrtSetDevice`              | 直接映射                             |
| `cudaDeviceSynchronize` | `aclrtSynchronizeDevice`      | 直接映射                             |

##### 4.2.2 类型映射


| CUDA 类型        | CANN 类型         |
| ---------------- | ----------------- |
| `cudaStream_t`   | `aclrtStream`     |
| `cudaEvent_t`    | `aclrtEvent`      |
| `cudaError_t`    | `aclError`        |
| `cudaMemcpyKind` | `aclrtMemcpyKind` |

##### 4.2.3 错误码转换

参考 `references/error_mapping.md` 进行错误码映射。

##### 4.2.4 初始化/清理

```c
// CANN 需显式初始化
aclInit(NULL);  // 或指定配置文件路径

// 业务代码...

// 清理
aclFinalize();
```

#### 4.3 编译配置改写

##### 4.3.1 CMake 改写示例

```cmake
# 原 CUDA 项目
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(your_target ${CUDA_LIBRARIES})

# 迁移后（兼容层方式）
# 设置兼容层源码路径（指向 runtime_migration/assets/src）
set(RUNTIME_MIGRATION_DIR /path/to/runtime_migration/assets/src)
include_directories(${RUNTIME_MIGRATION_DIR})

# 编译兼容层库（在 assets/src 目录）
# cd assets/src && mkdir -p build && cd build && cmake .. && make
# 编译产物：libcudacompat.so, libcudacompat.a

find_library(CUDACOMPAT libcudacompat.so 
    PATHS ${RUNTIME_MIGRATION_DIR}/build)
target_link_libraries(your_target ${CUDACOMPAT} acl_rt)
```

##### 4.3.2 Makefile 改写示例

```makefile
# 原 CUDA 项目
CFLAGS += -I/usr/local/cuda/include
LDFLAGS += -L/usr/local/cuda/lib64 -lcudart

# 迁移后（兼容层方式）
# 设置兼容层源码路径（指向 runtime_migration/assets/src）
RUNTIME_MIGRATION_DIR = /path/to/runtime_migration/assets/src
CFLAGS += -I$(RUNTIME_MIGRATION_DIR)
LDFLAGS += -L$(RUNTIME_MIGRATION_DIR)/build -lcudacompat
LDFLAGS += -L~/Ascend/cann/lib64 -lacl_rt
```

##### 4.3.3 独立编译兼容层库

兼容层支持在 `assets/src` 目录独立编译：

```bash
# 在 assets/src 目录独立编译
cd assets/src
mkdir -p build && cd build
cmake .. && make

# 编译产物
# libcudacompat.so - 共享库
# libcudacompat.a  - 静态库

# Debug 模式编译
cmake .. -DCUDA_COMPAT_DEBUG=ON && make
```

---

### Step 5: 编译测试验证

#### 5.1 编译步骤

```bash
# 编译兼容层库
cd assets/src
mkdir -p build && cd build
cmake .. && make

# 编译产物位置
# assets/src/build/libcudacompat.so
# assets/src/build/libcudacompat.a

# 编译迁移后的应用
cd your_app_dir
gcc -I/path/to/runtime_migration/assets/src \
    your_app.c \
    -L/path/to/runtime_migration/assets/src/build -lcudacompat \
    -L~/Ascend/cann/lib64 -lacl_rt \
    -o your_app
```

#### 5.2 运行测试

```bash
# 设置环境变量（可选）
export CUDA_COMPAT_INIT_FILE_PATH=/path/to/acl.ini

# 运行应用
./your_app
```

#### 5.3 Debug 模式排查

如果遇到问题，启用 Debug 模式：

```bash
# 编译兼容层 Debug 版本
cd assets/src/build
cmake .. -DCUDA_COMPAT_DEBUG=ON && make

# 运行应用查看详细错误信息
./your_app
# 输出：[CUDA_COMPAT_DEBUG] cann_compat.c:34: aclInit failed with ACL error: 107000
```

#### 5.4 常见问题处理


| 问题                           | 原因               | 解决方案                        |
| ------------------------------ | ------------------ | ------------------------------- |
| 初始化失败                     | aclInit 配置错误   | 检查 CUDA_COMPAT_INIT_FILE_PATH |
| 内存分配失败                   | 设备内存不足       | 减小分配大小或检查设备状态      |
| API 返回 cudaErrorNotSupported | 使用了不支持 API   | 查阅支持表，使用替代方案        |
| IPC 传递失败                   | 使用 UNIX socket   | 改用共享内存传递 opaque handle  |
| Kernel 无法运行                | CUDA Kernel 未迁移 | 编写 Ascend C Kernel            |

---

## 参考资源

本技能依赖以下参考文件，按需加载：

- `references/cuda_api_common.md` - 常用 CUDA API 参考
- `references/cann_api_common.md` - 常用 CANN API 参考
- `references/error_mapping.md` - 错误码映射表
- `references/api_support_table.md` - API 支持状态表

### 项目代码参考

`assets/src` 目录提供了 CUDA API 到 CANN Runtime API 的完整实现，可作为迁移代码参考：

#### 头文件（API 声明）


| 文件路径                            | 功能说明               | 参考用途                                       |
| ----------------------------------- | ---------------------- | ---------------------------------------------- |
| `assets/src/cann_runtime_compat.h`  | 主头文件，整合所有 API | 查看支持的完整 API 列表                        |
| `assets/src/cann_compat_types.h`    | CUDA 类型定义          | 了解类型映射（cudaStream_t → aclrtStream 等） |
| `assets/src/cann_compat_cu_types.h` | CUDA Driver 类型定义   | Driver API 类型映射                            |
| `assets/src/cann_compat_device.h`   | 设备管理 API           | cudaGetDeviceCount 等实现参考                  |
| `assets/src/cann_compat_memory.h`   | 内存管理 API           | cudaMalloc, cudaMemcpy 等实现参考              |
| `assets/src/cann_compat_stream.h`   | 流管理 API             | cudaStreamCreate 等实现参考                    |
| `assets/src/cann_compat_event.h`    | 事件管理 API           | cudaEventCreate 等实现参考                     |
| `assets/src/cann_compat_ipc.h`      | IPC API                | cudaIpcGetMemHandle 等实现参考                 |
| `assets/src/cann_compat_library.h`  | 库管理 API             | cudaLibraryLoad 等实现参考                     |
| `assets/src/cann_compat_exec.h`     | 执行配置 API           | 执行配置相关宏定义                             |
| `assets/src/cann_compat_mempool.h`  | 内存池 API             | cudaMemPool 等实现（Mock）                     |
| `assets/src/cann_compat_cu_vmm.h`   | VMM API                | cuMemCreate 等 Driver VMM 实现                 |

#### 源文件（API 实现）


| 文件路径                           | 功能说明   | 参考用途                            |
| ---------------------------------- | ---------- | ----------------------------------- |
| `assets/src/cann_compat.c`         | 主实现文件 | 查看所有 Runtime API 的完整实现逻辑 |
| `assets/src/cann_compat_ipc.c`     | IPC 实现   | 了解 IPC 跨进程通信的实现细节       |
| `assets/src/cann_compat_library.c` | 库管理实现 | 了解库加载和 Kernel 执行的实现      |

#### 编译配置


| 文件路径                    | 功能说明                                      |
| --------------------------- | --------------------------------------------- |
| `assets/src/CMakeLists.txt` | 独立的 CMake 配置，支持独立编译 cudacompat 库 |

#### 使用方式

迁移时，可按以下方式参考项目代码：

1. **查看 API 映射实现**：阅读 `assets/src/cann_compat.c` 中对应函数，了解 CUDA API 如何转换为 CANN API 调用
2. **理解参数转换**：查看函数内部如何处理参数差异（如 cudaMemcpyKind → aclrtMemcpyKind）
3. **错误处理模式**：学习如何将 CANN 错误码映射为 CUDA 错误码
4. **特殊场景处理**：如 IPC 的 opaque handle 处理、Stream 配置等

示例 - 查看 cudaMalloc 实现：

```bash
# 在 cann_compat.c 中搜索 cudaMalloc 实现
grep -A 20 "cudaError_t cudaMalloc" assets/src/cann_compat.c
```

独立编译（可选）：

```bash
# 在 assets/src 目录独立编译 cudacompat 库
cd assets/src
mkdir -p build && cd build
cmake .. && make
# 生成 libcudacompat.so 和 libcudacompat.a
```

本地头文件路径：

- CUDA 类型定义: `assets/src/cann_compat_types.h`
- CANN 头文件: `~/Ascend/cann/include/acl/acl_rt.h`

---

## 示例

### 示例 1: 简单内存拷贝程序迁移

**原 CUDA 代码**：

```c
#include <cuda_runtime.h>

int main() {
    float *d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));
  
    float h_data[1024];
    cudaMemcpy(d_data, h_data, 1024 * sizeof(float), cudaMemcpyHostToDevice);
  
    cudaFree(d_data);
    return 0;
}
```

**迁移后（兼容层方式）**：

```c
#include "cann_runtime_compat.h"

int main() {
    CUDA_COMPAT_FORCE_LINK();
  
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, 1024 * sizeof(float)));
  
    float h_data[1024];
    CUDA_CHECK(cudaMemcpy(d_data, h_data, 1024 * sizeof(float), cudaMemcpyHostToDevice));
  
    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
```

**迁移后（直接迁移方式）**：

```c
#include <acl/acl.h>

int main() {
    aclInit(NULL);
  
    float *d_data;
    aclrtMalloc(&d_data, 1024 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  
    float h_data[1024];
    aclrtMemcpy(d_data, 1024 * sizeof(float), h_data, 1024 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
  
    aclrtFree(d_data);
    aclFinalize();
    return 0;
}
```

### 示例 2: 多流并发程序迁移

**原 CUDA 代码**：

```c
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

cudaMemcpyAsync(d1, h1, size, cudaMemcpyHostToDevice, stream1);
cudaMemcpyAsync(d2, h2, size, cudaMemcpyHostToDevice, stream2);

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

**迁移后（兼容层方式）**：

```c
// API 调用完全相同，仅替换头文件
#include "cann_runtime_compat.h"
// ... 同原代码 ...
```

---

## 输出格式

完成迁移后，输出以下内容：

1. **迁移可行性分析报告**
2. **迁移方式推荐**
3. **改写后的代码**
4. **编译命令**
5. **测试验证步骤**
