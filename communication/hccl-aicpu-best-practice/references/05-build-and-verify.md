# 05 · CMake 接线与跨层证据接口

> 本文里的 `$HCCL_REPO` 一律指调用方显式传入的 HCCL 仓根目录（见 `SKILL.md` §1.0），本 Skill 不预设路径。

## 1. 两处 CMake 登记（漏一处必踩坑）

AICPU template 同一份源码要编进**两个**产物：

| 产物 | 目标名 | 登记位置 |
|---|---|---|
| host 侧 `libhccl.so` | `hccl` | `src/ops/<op>/algorithm/template/aicpu/CMakeLists.txt` |
| device 侧 `libscatter_aicpu_kernel.so` | `scatter_aicpu_kernel` | `src/scatter_aicpu_kernel.cmake` |

**只加第 1 处的症状**：host 编译通过、链接通过，运行到 AICPU 展开时找不到符号 / 行为回退到旧算法。
这是本仓最高频的接线 bug，`scripts/check_template.py` 专门查它。

以下示例使用新布局。若探测到旧布局，两处登记与 include 路径须一致去掉 `algorithm/`，不能混用。

### 1.1 `src/ops/<op>/algorithm/template/aicpu/CMakeLists.txt`

结构固定：

```cmake
set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/ins_temp_all_reduce_mesh_1D_one_shot.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ins_temp_all_reduce_mesh_1D_two_shot.cc
    ...
    ${CMAKE_CURRENT_SOURCE_DIR}/ins_temp_你的新文件.cc          # ← 加这里
)
if(NOT HCCL_CANN_COMPAT_850)
    list(APPEND src_list
        ${CMAKE_CURRENT_SOURCE_DIR}/ins_temp_xxx_dpu.cc         # ← 需要 CANN 9.x 才加这里
    )
endif()

if(TARGET hccl)
    target_sources(hccl PRIVATE ${src_list})
endif()
```

### 1.2 `src/scatter_aicpu_kernel.cmake`

源文件列表在文件开头的 `add_library(scatter_aicpu_kernel SHARED ...)` 块里，按算子分段。找到你的算子那一段追加：

```cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_two_shot.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/algorithm/template/aicpu/ins_temp_你的新文件.cc   # ← 加这里
```

需要 CANN 9.x 的放文件末尾的守卫块：

```cmake
if(NOT HCCL_CANN_COMPAT_850)
    target_sources(scatter_aicpu_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/xxx/algorithm/template/aicpu/ins_temp_xxx_dpu.cc
    )
endif()
```

### 1.3 头文件搜索路径

`src/CMakeLists.txt` 已经把所有 `ops/<op>/algorithm/template/{,aicpu,aiv,ccu}` 加进 include 目录了，
所以 `#include "ins_temp_xxx.h"` 直接写文件名即可，**不要写相对路径**。
新增算子目录才需要动 `src/CMakeLists.txt`；新增 template 文件不用。

## 2. 版本与编译开关

| 宏 / 变量 | 含义 | 用法 |
|---|---|---|
| `HCCL_CANN_COMPAT_850` | CMake 变量，目标 CANN 为 8.5.0 | `if(NOT HCCL_CANN_COMPAT_850) ... endif()` |
| `CANN_VERSION_NUM` / `CANN_VERSION(9,0,0)` | C++ 宏，源码里做版本分支 | `#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)` |
| `AICPU_COMPILE` | C++ 宏，正在编 device 侧 kernel | `#ifndef AICPU_COMPILE` 包住只有 host 才有的东西（AIV/CCU 头文件等） |

`cmake/config.cmake` 探测 `include/version/cann_version.h` 得到版本号并设置上述变量。

**AICPU template 里的常见用法**：如果你的 template 引用了只在 host 侧存在的类型，
用 `#ifndef AICPU_COMPILE` 包住——但正常的 AICPU template 不应该有这种依赖，出现了先怀疑设计。

## 3. Template 层静态验证

```bash
python3 "$SKILL_DIR/scripts/check_template.py" --repo "$HCCL_REPO" <template.cc>
python3 "$SKILL_DIR/scripts/check_template.py" --repo "$HCCL_REPO" --strict-new <new-template.cc>
```

`check_template.py` 输出两处 CMake 登记、接口、命名、安全和可解析的数据流问题。模式 C 还需同时保留：

- `check_spec.py` 和 `check_layout.py` 的输出与退出码；
- spec 到 C++ 的 offset/length/repeat/stride/scratch/thread/notify/同步映射核对结果；
- 目标仓路径、分支、HEAD 和未提交变更身份；
- 上游输入缺口和未验证范围。

`scripts/selftest.sh` 可利用目标仓的 `compile_commands.json` 在隔离 worktree 中验证骨架的 host 侧 C++17/`-Werror` 兼容性。该 probe 只是 Skill 自身回归，不等于 HCCL 候选源码的完整构建。

## 4. 跨层证据交接

AICPU template 同时进入 host 侧 `libhccl.so` 和 device 侧 `libscatter_aicpu_kernel.so`，因此静态检查无法单独证明实现可用。交给调用 Agent 的证据至少包含：

| 证据 | 内容 |
|---|---|
| 源码身份 | 仓路径、分支、HEAD、未提交差异或可恢复快照 |
| Template 文件 | `.h` / `.cc`、局部 `CMakeLists.txt`、`scatter_aicpu_kernel.cmake` |
| 设计证据 | Dataflow Spec 标识、布局量化样例、绑定 executor 和选路约束 |
| 静态结果 | 命令、退出码、报告路径及未解析 WARN |
| 外部前提 | 构建所需 CANN/第三方环境，以及未验证范围 |

完整构建、基线与产物归档、安装及 hccl-vm A/B 双轨的技术契约见[共享契约](../aicpu-shared-contract.md)和[构建验证](build-verification.md)；ST/UT 执行范围、角色调度和完成判定由调用 Agent/Plugin 负责。本文不复制命令、环境处置或状态门禁，避免产生第二套流程真源。
