# 工程骨架模板

> direct launch 提交工程的共享骨架。所有算子共享同一套骨架，一次搭建；新增算子只需在 `csrc/ops/` 下建子目录（见 [operator-template.md](operator-template.md)）。权威样例见 direct launch 工程样例（如 cann-bench `examples/direct_launch_example/`）。

## 目录结构

```
generated_project/
├── build.sh            # 见下方模板
├── setup.py            # 见下方模板
├── CMakeLists.txt      # 见下方模板
├── cmake/
│   ├── func.cmake      # register_direct_launch_op 宏（见下方）
│   ├── ascend.cmake    # CANN 路径发现（从样例复制）
│   ├── python.cmake    # Python3 发现（从样例复制）
│   ├── torch.cmake     # Torch 发现（从样例复制）
│   └── torch_npu.cmake # torch_npu 发现（从样例复制）
├── cann_bench/
│   └── __init__.py     # 见下方模板
├── csrc/
│   ├── extension.cpp   # 见下方模板
│   └── ops/
│       └── CMakeLists.txt  # 见下方模板（自动发现算子）
└── scripts/
    └── build_wheel.sh  # build.sh 调用（从样例复制）
```

> `cmake/ascend.cmake` / `python.cmake` / `torch.cmake` / `torch_npu.cmake` / `scripts/build_wheel.sh` 直接从 direct launch 工程样例（如 cann-bench `examples/direct_launch_example/`）复制，无需修改。

## build.sh

SoC 自动检测 + wheel 构建 + 可选安装。`<OP>` 无需修改——build.sh 不感知算子。

```bash
#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

detect_soc_version() {
    local torch_soc=$(python3 -c "import torch, torch_npu; print(torch.npu.get_device_name(0))" 2>/dev/null)
    if [ -n "${torch_soc}" ]; then
        case "${torch_soc}" in
            Ascend910B*)     echo "ascend910b" ; return ;;
            Ascend910_93*)   echo "ascend910_93" ; return ;;
            Ascend950*)      echo "ascend950" ; return ;;
        esac
    fi
    local npu_name=$(npu-smi info 2>/dev/null | grep -oP 'Ascend\S+' | head -1)
    case "${npu_name}" in
        Ascend910B1|Ascend910B2|Ascend910B3|Ascend910B4) echo "ascend910b" ;;
        Ascend910_93*)  echo "ascend910_93" ;;
        Ascend950*)     echo "ascend950" ;;
        *)              echo "" ;;
    esac
}

SOC_VERSION=""
INSTALL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --soc=*) SOC_VERSION="${1#*=}"; shift ;;
        --install) INSTALL=true; shift ;;
        *) shift ;;
    esac
done

if [ -z "${SOC_VERSION}" ]; then
    SOC_VERSION=$(detect_soc_version)
    if [ -z "${SOC_VERSION}" ]; then
        echo "[ERROR] Cannot detect SoC version. Use --soc=<soc_version>."
        echo "Supported: ascend910b, ascend910_93, ascend950"
        exit 1
    fi
    echo "[INFO] Auto-detected SoC: ${SOC_VERSION}"
fi
export NPU_ARCH="${SOC_VERSION}"

echo "=== Building cann_bench wheel package ==="
echo "NPU_ARCH: ${NPU_ARCH}"
DIST_DIR="${SCRIPT_DIR}/dist"
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"
bash "${SCRIPT_DIR}/scripts/build_wheel.sh"
if [[ "${INSTALL}" == "true" ]]; then
    echo "=== Installing wheel package ==="
    pip install ${DIST_DIR}/cann_bench*.whl --force-reinstall --no-deps
fi
echo "=== Build complete ==="
ls -la "${DIST_DIR}"
```

## setup.py

ABI3 wheel + cmake_build。直接从 direct launch 工程样例（如 cann-bench `examples/direct_launch_example/setup.py`）复制，无需修改（`PACKAGE_NAME="cann_bench"`、`VERSION="1.0.0"` 已固定，`cann_bench` 为评测集约定的包名）。

## 顶层 CMakeLists.txt

关键配置：NPU_ARCH → bisheng `--npu-arch` 映射、双编译器（bisheng 编 kernel / g++ 编 plugin）、合并 `_C.abi3.so`。直接从 direct launch 工程样例（如 cann-bench `examples/direct_launch_example/CMakeLists.txt`）复制，无需修改。

核心段落（理解用，勿手改）：

```cmake
# NPU_ARCH → Bisheng --npu-arch 映射
if(NPU_ARCH STREQUAL "ascend910b" OR NPU_ARCH STREQUAL "ascend910_93")
    set(BISHENG_NPU_ARCH "dav-2201")
elseif(NPU_ARCH STREQUAL "ascend950")
    set(BISHENG_NPU_ARCH "dav-3510")
endif()

# bisheng 编译 kernel（临时切换编译器）
set(CMAKE_CXX_COMPILER ${BISHENG})
set_source_files_properties(${ALL_KERNEL_SRCS} PROPERTIES
    LANGUAGE CXX COMPILE_FLAGS "--npu-arch=${BISHENG_NPU_ARCH} -xasc")
add_library(all_kernels_obj OBJECT ${ALL_KERNEL_SRCS})

# 切回 g++ 编译 plugin
set(CMAKE_CXX_COMPILER ${_SAVED_CMAKE_CXX_COMPILER})
add_library(all_plugins_obj OBJECT ${ALL_PLUGIN_SRCS})

# 合并为 _C.abi3.so，复制到 cann_bench/
add_library(_C SHARED ${EXTENSION_CPP}
    $<TARGET_OBJECTS:all_kernels_obj> $<TARGET_OBJECTS:all_plugins_obj>)
```

## cmake/func.cmake

算子自注册宏。直接从样例复制，无需修改：

```cmake
macro(register_direct_launch_op KERNEL_SRCS KERNEL_INCLUDE_DIR PLUGIN_SRCS PLUGIN_INCLUDE_DIR)
    get_filename_component(OP_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    # 将 kernel/plugin 源文件与 include 目录追加到全局列表
    # （详见 direct launch 工程样例 cann-bench examples/direct_launch_example/cmake/func.cmake）
endmacro()
```

## csrc/ops/CMakeLists.txt

自动发现算子子目录。直接从样例复制，无需修改：

```cmake
file(GLOB SUB_DIRS LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${SUB_DIRS})
    if(IS_DIRECTORY ${SUB_DIR})
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
```

## csrc/extension.cpp

Python 扩展入口，触发 `TORCH_LIBRARY` 静态初始化。直接从样例复制，无需修改。

## cann_bench/__init__.py

Python 包入口。每新增一个算子，在此追加导出函数：

```python
import torch
try:
    from . import _C
except ImportError as e:
    raise ImportError("Cannot import _C. Please install the cann_bench package.") from e

# 每新增算子追加一行
def <op>(...) -> torch.Tensor:
    return torch.ops.cann_bench.<op>(...)
```
