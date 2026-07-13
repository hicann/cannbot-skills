# 代码结构与构建指南

> 本仓算子代码的默认目录/文件结构与编译验证方法。默认结构以 `ascendc-direct-invoke-template` 的工程模板为准；具体命令与结构由该模板给出，本文给通用指南。

## 算子代码结构（默认）

单个算子为独立工程目录。默认结构以 `ascendc-direct-invoke-template` 复制出的工程模板为准，通常包含：

```
operators/{operator}/
├── CMakeLists.txt        # 编译配置
├── run.sh                # 编译 + 运行 + 比对入口
├── op_kernel/            # Kernel 侧：Tiling 数据结构 + Kernel 实现（.asc）
├── op_host/              # Host 侧：Tiling 计算、内存管理、<<<>>> 调用（.asc）
├── op_extension/         # PyTorch 直调层（TORCH_LIBRARY 注册），可选
├── scripts/              # 测试/验证脚本
└── docs/                 # 设计/计划等文档
```

Kernel 代码文件为 `.asc`（ASC 编译器只识别 `.asc`）；Tiling 参数计算在 Host 侧完成（如 `op_host` 中的 `ComputeXxxTiling()`），不在 Kernel 的 `Init()`/`Process()` 中计算。

## 编译配置要点

CMakeLists.txt 需满足 Ascend C 构建要求：

- `find_package(ASC REQUIRED)`
- `project(... LANGUAGES ASC CXX)`
- 用 `add_executable`
- 链接 `tiling_api` / `register` / `platform` / `m` / `dl`
- 设置 `--npu-arch`（取值按目标芯片，须与目标架构一致）

关键环境变量为 `ASCEND_HOME_PATH`（不是 `ASCEND_HOME`）；编译器 `bisheng` 位于 `$ASCEND_HOME_PATH/<arch>/ccec_compiler/bin/bisheng`，CMake 经 `find_package(ASC)` 自动发现。

## 构建流程

```bash
mkdir build && cd build
cmake ..
make
```

或经模板提供的 `run.sh` 完成「编译 + 运行 + 比对」。

## 验证程度

仅编译通过**不等于**验证通过，必须实际运行测试：

| 项 | 要求 |
|----|------|
| 独立编译 | `cmake .. && make` 成功，无代码级警告 |
| 分级功能 | Level 0（8-16 元素）/ Level 1（1K 元素）/ Level 2（极值、零值）逐级通过 |
| 非对齐场景 | 数据长度非 32 字节倍数时正确（DataCopyPad 路径） |
| PyTorch 通路 | 编出 `lib{operator}_ops.so`，`torch.ops.npu.{operator}()` 跑通（端侧开发不含此项） |

存在失败用例时验证结论判为失败，禁止标为通过。
