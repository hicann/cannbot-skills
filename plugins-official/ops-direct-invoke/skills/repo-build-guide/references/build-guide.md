# direct launch 工程结构与构建指南

> 本仓算子工程采用 direct launch 结构，权威样例见 direct launch 工程样例（如 cann-bench 仓 `examples/direct_launch_example/`）。本文给通用指南，具体模板文件见 `repo-op-templates`。

## 工程结构

单个提交为一个工程目录，可含一个或多个算子。结构以 direct launch 工程样例（如 cann-bench `examples/direct_launch_example/`）为准：

```
generated_project/
├── build.sh            # 编译入口（支持 --soc= / --install）
├── setup.py            # wheel 打包（ABI3 + cmake_build）
├── CMakeLists.txt      # 顶层编译配置
├── cmake/              # 公共 CMake 模块（不感知算子）
│   ├── func.cmake      # register_direct_launch_op 注册宏
│   ├── ascend.cmake    # CANN/ASCEND 路径发现
│   ├── python.cmake    # Python3 发现
│   ├── torch.cmake     # Torch 发现
│   └── torch_npu.cmake # torch_npu 发现
├── cann_bench/
│   └── __init__.py     # Python 包：from . import _C；导出 torch.ops.cann_bench.<op>
├── csrc/
│   ├── extension.cpp   # Python 扩展入口（PyInit__C，触发 TORCH_LIBRARY 静态初始化）
│   └── ops/
│       ├── CMakeLists.txt  # 自动发现算子子目录（file GLOB → add_subdirectory）
│       └── <op>/
│           ├── CMakeLists.txt      # 调用 register_direct_launch_op(<kernel_srcs> <plugin_srcs>)
│           ├── op_kernel/
│           │   ├── <op>_kernel.cpp # bisheng 编译：Kernel 类 + Tiling + extern "C" Launch
│           │   └── <op>_launch.h   # Launch 声明（g++ 可见，供 plugin include）
│           └── op_plugin/
│               └── <op>_plugin.cpp # g++ 编译：TORCH_LIBRARY_FRAGMENT + Meta + NPU impl
└── tests/              # 提交方自测脚本（评测不依赖，可选）
```

### 算子自注册机制

新增算子**无需修改任何公共 CMakeLists.txt**——只需在 `csrc/ops/` 下建算子子目录，写自己的 `CMakeLists.txt` 调用 `register_direct_launch_op()` 注册。顶层 `csrc/ops/CMakeLists.txt` 自动 `file(GLOB)` 发现所有算子子目录。

## 双编译器分工

direct launch 的核心是 **bisheng 与 g++ 双编译器**：

| 编译对象 | 编译器 | 编译标志 | 产物 |
|----------|--------|----------|------|
| `op_kernel/*.cpp` | bisheng | `--npu-arch=<arch> -xasc` | `all_kernels_obj`（OBJECT 库） |
| `op_plugin/*.cpp` | g++ | `-O3 -std=c++17` | `all_plugins_obj`（OBJECT 库） |
| `extension.cpp` | g++ | 同上 | 合并入 `_C.abi3.so` |

顶层 `CMakeLists.txt` 通过临时切换 `CMAKE_CXX_COMPILER` 为 bisheng 编译 kernel，再切回 g++ 编译 plugin，最后合并为 `_C.abi3.so` 并复制到 `cann_bench/` 目录。

> **Kernel 文件用 `.cpp`**（非 `.asc`）。direct launch 用 bisheng `-xasc` 模式编译 `.cpp`，`<<<>>>` launch 语法在此模式下有效。

### NPU 架构映射

`build.sh` 自动检测 SoC 版本（优先 `torch.npu.get_device_name()`，兜底 `npu-smi info`），映射到 bisheng `--npu-arch` 值：

| NPU_ARCH | bisheng `--npu-arch` | AICore 微架构 | 芯片 |
|----------|----------------------|---------------|------|
| `ascend910b` | `dav-2201` | c220 | 910B1/B2/B3/B4 |
| `ascend910_93` | `dav-2201` | c220（共享 910B 微架构） | 910_93 系列 |
| `ascend950` | `dav-3510` | c310 | 950 系列 |

> 评测镜像 per-SoC（`cann_bench_utils` 为评测集强制依赖，含 SoC 相关 kernel），故提交工程须与评测镜像的 NPU_ARCH 一致。

## 构建流程

```bash
# 自动检测 SoC，编译 wheel
bash build.sh

# 指定 SoC
bash build.sh --soc=ascend910b

# 编译 + 安装（评测前安装到 Python 环境）
bash build.sh --soc=ascend910b --install
```

`build.sh` 内部调用 `scripts/build_wheel.sh` → `setup.py` 的 `cmake_build` → `cmake` + `make`，产出 `dist/cann_bench-1.0.0-cp38-abi3-*.whl`。

## 评测器的编译契约

评测集评测器（如 cann-bench 的 `run_evaluation.sh --source-dir <dir>`）对提交工程的编译流程：

1. 卸载已安装的 `cann_bench` 包（避免算子重复注册冲突，保留 `cann_bench_utils` 强制依赖）
2. 在提交目录执行 `build.sh --install`（或等价 cmake 流程），编译 + 安装 wheel
3. 编译失败 → `_compile.log` 收至 `reports/build/`，相关算子**整批计 0 分**

> **整批编译失败处理**：一份提交多算子一起编译时，任一算子编译失败，本次提交涉及算子**全部按编译失败计 0**（不隔离、不补救、不改用户源码）。提交前务必 `bash build.sh` 验证全量编译通过。

## 稳定运行路径

编译产物与运行时导入的包**不是同一份**，两者不一致会让测试结果失真——现象是代码明明改了、测试却跑的是旧算子，容易被误判为代码缺陷。固定按下列路径运行：

1. **先做算子符号探针，再跑测试**：导入包后检查目标算子符号是否存在（`hasattr(<pkg>, "<op>")` 或 `torch.ops.<pkg>.<op>` 可取），不通过则先 `bash build.sh --install` 恢复后重跑探针。探针不通过就跑出来的失败结果无效，不得作为功能/精度结论。
2. **固定导入上下文**：优先从**工程本地上下文**运行（工作目录切到工程目录，导入本地产物），使结果与全局 site-packages 的状态解耦；确需全局包生效时（如评测器按已安装包评测），用 `bash build.sh --install` 恢复，并在复测前再跑一次探针。
3. **同一轮内保持一致**：一轮验证中不混用两种导入来源；报告中记录本轮采用的运行上下文与探针结论。

> 环境侧可能把已安装包回退为旧构建。发现符号消失时按上述路径恢复即可——这属环境现象，不是代码回归；判定依据是**产物与源码一致性**（见下方构建抖动条），不是"改了代码没生效"的直觉。

### 构建抖动

高并行度下 cmake 配置阶段可能瞬时失败，无确定触发条件。处理方式：

- 重试 1~2 次；重试即成功的属构建抖动，不按代码错误定位。
- 构建结论以「重试成功 + 产物与源码的一致性核对（时间戳 / 校验和）通过」为准，不以单次失败裁定。
- 重试仍稳定失败的，才按编译错误处理（诊断信息在编译日志）。

## 验证程度

仅编译通过**不等于**验证通过，必须实际运行测试：

| 项 | 要求 | 方法 |
|----|------|------|
| 独立编译 | `bash build.sh` 成功，无编译错误 | 提交前本地验证（瞬时失败按「构建抖动」重试） |
| 算子符号探针 | 运行时导入的包含本次构建的算子 | 见「稳定运行路径」，每轮测试前先跑 |
| schema 注册 | `torch.ops.cann_bench.<op>` 可调用，schema 与 `proto.yaml` 一致 | `python -c "import cann_bench; print(torch.ops.cann_bench.<op>)"` |
| 精度验证 | 评测集 `run_evaluation.sh --no-perf`（如 cann-bench）用例全通过 | 提交前本地或 docker 验证 |
| 性能验证 | 评测集 `run_evaluation.sh`（含 perf）HAP 有效 | 可选，正式提交前验证 |

存在失败用例时验证结论判为失败，禁止标为通过。
