# ops-transformer 目标契约

结构取自 2026-10-21 的 master `d0a83f16`。复杂模拟交付件的实跑记录见 [验证矩阵](validation-matrix.md)。
官方来源：[ops-transformer](https://gitcode.com/cann/ops-transformer)。

## 分类与布局

顶层 `CMakeLists.txt::ASCEND_ALL_MODULE_NAME`：`mc2 attention moe ffn mhc posembedding gmm mamba`。
各类有特殊收集分支，核对顶层 CMake、分类 CMake 与 `cmake/func.cmake::op_add_subdirectory`，不修改它们。
`torch_extension`、common、examples 不是迁移目标。

当前仓**没有** `add_all_modules_sources`。产品接入在 `op_host/CMakeLists.txt`；
算子顶层 CMake 用于遍历 op_host / tests 等子目录，UT 构建依赖 `tests/CMakeLists.txt`。
工作流已经采用平级 op_api，保留原路径并显式告诉宏，避免为了照搬旧算子而移动实现文件。

## 产品 CMake

`op_host/CMakeLists.txt` 对照 `attention/flash_attention_score/op_host/CMakeLists.txt` 与
`cmake/obj_func.cmake::add_modules_sources_with_soc`，已有手写 ACLNN 的最小形态：
`<OP>` 取当前算子目录名；未手写 ACLNN 的分支按下文切换桶和模式。

```cmake
# attention / mhc / mamba 分类需要此登记；其他分类由 op_add_subdirectory 收集。
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE <OP>_def.cpp)
endif()
if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources_with_soc(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OPTYPE <OP> ACLNNTYPE aclnn_exclude)
endif()
```

`add_op_to_compiled_list()` 放在 `op_host/CMakeLists.txt` 中，登记当前算子及其目录。attention / mhc / mamba 由分类目录直接遍历，
不自动进入 `op_add_subdirectory` 的返回列表；遗漏登记会导致 kernel 源与配置缺失，UT 也可能退出 0 却未执行用例。
这三个分类保留此调用；gmm / posembedding / moe / ffn / mc2 由 `op_add_subdirectory` 收集，不额外要求登记。
核对当前算子进入 `OP_DIR_LIST`、实际 kernel 产物及 UT 数量；不能把此调用放在算子顶层，否则目录推导会错层。

该宏从 `ARCH_DIRECTORY` 收 `op_host/arch*/` 的 tiling 和推导源码，并收 `op_graph/*_graph_infer*.cpp`。
普通 `add_modules_sources` 不收相同的架构子目录，不能随意替换；这里也没有 math 风格的 `SUPPORT_TILING_DIR` 参数。
没有手写 ACLNN 时用 `aclnn`，相应 `target_sources` 的桶为 `op_host_aclnn`；始终与源包接口一致。
算子顶层 CMake 接入 op_host；`if(ENABLE_TEST)` 下接入 tests。如真实功能需要依赖、通信库或专门编译选项，从同类算子核对后只在本算子 CMake 中表达。

`ARCH_DIRECTORY` 来自顶层的 SOC_VERSION_LIST / ARCH_DIRECTORY_LIST；ascend950 → arch35，ascend910b / ascend910_93 → arch22，mc62 → arch38。
ascend960dt 还可能补 arch92；遇到这种额外架构必须检查实际收源，不能仅凭通用脚本主架构检查认定完整。
保留 OpDef 的真实 soc 注册；不得因参考例有更多芯片而扩大支持面。

## UT

`cmake/func.cmake::op_add_subdirectory` 在 ENABLE_TEST 下会跳过没有 `tests/CMakeLists.txt` 的算子。
需完整接入：算子顶层 → tests → ut → op_host / op_api。
新接入使用 `add_modules_ut_sources(UT_NAME ... MODE PRIVATE DIR ...)`，参考
`moe/moe_init_routing/tests/ut/op_host/CMakeLists.txt`、`posembedding/rotary_position_embedding/tests/`。

- 根 `tests/ut/op_host/` 放 `test_<OP>_infershape.cpp` 和可选通用 tiling UT。
- 架构 tiling 放 `tests/ut/op_host/<ARCH_DIRECTORY>/test_<OP>_tiling.cpp`；在该目录的 CMake 调用 `add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})`，父目录按实际架构显式 add_subdirectory。函数内部按 ARCH_DIRECTORY 过滤。
- `cmake/ut.cmake` 仅匹配 `test_*_tiling.cpp`，**不匹配** `test_*_tiling_arch35.cpp`。
- ACLNN UT 放 `tests/ut/op_api/test_aclnn_*.cpp`，该目录的 CMake 用 `UT_NAME ${OP_API_MODULE_NAME}` 注册。
- 使用本仓 tests 工具，不移植 math helper；不能只看退出码，必须检查非零用例和对应算子。

## 出包、安装与 example

- 参数在顶层 `build.sh`；`--pkg --ops=<OP> --vendor_name=<v> --soc=<soc>`，UT 为独立的 `-u --ophost --ops=<OP> --soc=<soc>`。
- 包名：`cann-ops-transformer-<v>_linux-<arch>.run`；vendor 为 `<v>_transformer`，来源实际 custom 构建分支 `cmake/custom_build.cmake`；`cmake/package.cmake::pack_custom` 的另一种命名不是此命令的输出。
- eager cust 从 `ASCEND_CUSTOM_OPP_PATH` **首项的父目录**拼出 `<v>_transformer`。因此该变量应指向隔离安装下的实际 vendor，且 `--vendor_name` 必须一致。
- example 使用 `test_aclnn_*.cpp` / `test_geir_*.cpp`；GEIR 和 torch 命中新包的证明仍按共同验收规则执行。
- kernel 配置生成走 `cmake/custom_build.cmake` 与 `cmake/scripts/util/`，不要照搬 math 对 binary.json 的内部推导解释。以构建日志和实际 opt_info / kernel / tilingKey 覆盖为准。

## 已验证的补充事项

- 平级手写 `op_api` 可以由上述宏收集；当前 custom 包的 public headers 安装在
  `op_api/include/aclnnop/`。按实际头文件层级设置 include，不要因只扫描上一层就认定漏包。
- 同一算子多个 ACLNN 入口分别核对声明、导出与调用，包括可选 Tensor 的空指针通路。
- 本基线 `build.sh` 的部分 UT 分支使用按 CPU 数计算的 `CORE_NUMS`，可能覆盖外层 `-j`。
  执行前核对嵌套命令并将实际并行度限制到任务上限；记录验证环境对构建驱动的适配，
  不把这一适配混入算子迁移提交。
