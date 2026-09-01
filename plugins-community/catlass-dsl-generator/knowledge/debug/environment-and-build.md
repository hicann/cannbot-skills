---
type: CATLASS DSL Debugging Guide
title: 环境与构建问题
description: CANN、AscendNPU-IR、Python binding 和本地构建故障的分层定位方法。
tags: [catlass-dsl, debug, environment, build, cann]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
  - {by: process:catlass-dsl-source-audit, at: '2026-08-29T00:00:00Z'}
sources:
  - id: readme
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/README.md
    title: TLA DSL environment and build guide
  - id: build
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/build.sh
    title: TLA DSL build script
arch: [c310]
---

# 接口与概念

构建链包含 Python 包、MLIR/C++ 扩展、AscendNPU-IR 与 CANN 工具链。README
将环境准备、editable install、pytest、lit 和端到端构建列为独立步骤。[^readme]

# 用法

## 必需版本与环境

| 项目 | 固定提交声明 |
| --- | --- |
| Python | `>=3.10,<3.14` |
| CANN | `>=9.1.0` |
| CMake | `>=3.28,<4.0` |
| Ninja | `>=1.12` |
| Clang/Clang++ | `>=10`，推荐 19，需与 LLVM 工具配套 |
| AscendNPU-IR | `feature/regbase@a07821269…` |

必须记录并检查：

```bash
printf 'ASCEND_HOME_PATH=%s\n' "$ASCEND_HOME_PATH"
printf 'TLA_DSL_PREBUILT_ASCENDNPU_IR=%s\n' "$TLA_DSL_PREBUILT_ASCENDNPU_IR"
python3 -c 'import catlass, mlir; print(catlass.__file__); print(mlir.__file__)'
cmake --version
ninja --version
llvm-lit --version
```

# 代码模式

## AscendNPU-IR 产物检查

```bash
test -f "$TLA_DSL_PREBUILT_ASCENDNPU_IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVM.h"
test -f "$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/tools/bishengir/include/bishengir/Interfaces/BiShengIREnums.h.inc"
test -f "$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/install/lib/cmake/mlir/MLIRConfig.cmake"
test -x "$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/bin/mlir-tblgen"
```

MLIR Python binding 和 native library 必须来自同一 AscendNPU-IR build：

```bash
export MLIR_TBLGEN_INCLUDE_DIR="$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/install/include"
export PYTHONPATH="$TLA_DSL_PREBUILT_ASCENDNPU_IR/build/install/python_packages/mlir_core:${PYTHONPATH:-}"
```

## 分层最小命令

```bash
cd "$CATLASS_ROOT/python/tla_dsl"
python3 -m pip install -e .
python3 -m pytest -q tests/test_core_api_preconditions.py
llvm-lit -sv csrc/mlir/build/tests/lit/tla-compile
test -x csrc/mlir/build/tools/tla-compile/TlaCompile
```

`build.sh` 会先把工作目录切到脚本所在的 `python/tla_dsl`，因此可从任意目录调用。
debug 模式原地构建扩展，再从项目外目录执行隔离的 editable install，避免 in-tree
`*.egg-info` 遮蔽安装；`--release` 构建 wheel 到 `dist/`。[^build]

# 约束

- CATLASS 源码、已安装 Python 包和 native build 产物必须来自同一 revision。
- CANN/AscendNPU-IR 路径及工具版本必须记录，不能只记录“已 source 环境”。
- pytest 成功不等于 lit、build-only 或设备运行成功。

# 失败表现

| 表现 | 首查 |
| --- | --- |
| `No module named catlass` | editable install、解释器路径 |
| `No module named mlir` | `PYTHONPATH` 是否指向 AscendNPU-IR build |
| generated binding 缺少 op | `mlir-tblgen` 与 `.td` 是否同 revision |
| undefined symbol | MLIR Python extension 与动态库 ABI 是否同 build |
| 找不到 `MLIRConfig.cmake` | `TLA_DSL_PREBUILT_ASCENDNPU_IR/build/install` |
| 找不到 dialect/pass | `TlaCompile` 是否重新构建、运行时库路径 |
| `ASCEND_HOME_PATH` 为空 | CANN `set_env.sh` 未加载 |
| 从其他目录调用时相对路径错误 | 确认使用当前 `build.sh`；旧脚本曾依赖调用目录 |
| editable install 被源码 metadata 遮蔽 | 检查是否走脚本的项目外 `python -I -m pip install -e` |

设备编译错误应在 Python import、TLAIR 和 lit 都通过后再归因到 CANN/HIVMC
后端，并保留首个后端 stderr。

# 验证方法

保存 `python -c` import 路径、包版本、CANN 环境、build 命令和首个失败层的原始
stderr。本文只核对源码中的构建说明，未执行环境命令。

[^readme]: 固定提交 README 的环境、安装和测试分层。
[^build]: 固定提交 `build.sh` 的 native 构建入口。
