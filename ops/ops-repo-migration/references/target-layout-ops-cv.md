# ops-cv 目标契约

结构取自 2026-10-21 的 master `47d74481f41824a56dacda49368c03b915bdcae1`。复杂模拟交付件的实跑记录见 [验证矩阵](validation-matrix.md)。
官方来源：[ops-cv](https://gitcode.com/cann/ops-cv)。

## 分类与构建

标准分类是 `image`、**`objdetect`**，不是 `objetect`。
依据为 `cmake/variables.cmake::OPS_CATEGORY_LIST` 与顶层 `CMakeLists.txt`。
`image/CMakeLists.txt`、`objdetect/CMakeLists.txt` 遍历算子，不改分类清单。

算子顶层使用 `add_all_modules_sources` + 等长 `COMPUTE_UNIT` / `TILING_DIR`，
保持工作流的平级 `op_api` / `op_host` / `op_graph` / `op_kernel`。
**该宏只收产品源码，不自动接入 UT**（`cmake/func.cmake` 宏末尾没有 add_all_ut_sources）。
参考 `image/resize_bilinear_v2`、`objdetect/roi_align`：必须显式接入测试树。

在算子顶层 CMake 中保留以下接入，再调用产品聚合宏；不要重复加入其他含产品注册的子目录：
占位符展开为当前算子的实际目录名、ACLNN 模式、支持芯片及 tiling 目录。

```cmake
if(ENABLE_TEST)
    add_subdirectory(tests)
endif()
set(SUPPORT_COMPUTE_UNIT <source_soc_list>)
set(SUPPORT_TILING_DIR <source_tiling_dirs>)
add_all_modules_sources(OPTYPE <OP> ACLNNTYPE <source_aclnn_mode>
    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR}
    DISABLE_IN_OPP TRUE)
```

## UT 接入链

`tests/CMakeLists.txt` → `add_subdirectory(ut)`；
`tests/ut/CMakeLists.txt` 按 `UT_TEST_ALL OR OP_HOST_UT` 加 `op_host`，按 `UT_TEST_ALL OR OP_API_UT` 加 `op_api`。
采用 `tests/ut/op_host/<TILING_DIR>/test_<OP>_tiling.cpp` 和平级 `test_<OP>_infershape.cpp`。
`tests/ut/op_host/CMakeLists.txt`（以下分支按源包的每个 soc 与对应 TILING_DIR 分别展开）：

```cmake
add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME}
    MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
if(ASCEND_COMPUTE_UNIT STREQUAL "<soc>")
    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME}
        MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR}/<TILING_DIR>)
endif()
```

`cmake/ut.cmake` 只 glob 传入 DIR 的 `test_*_tiling.cpp` / `test_*_infershape.cpp`，不递归。
`tests/ut/op_api/CMakeLists.txt` 同样显式调用 `add_modules_ut_sources(UT_NAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})`。
部分历史算子把 op_api UT 放在 `op_host/op_api`，新迁移采用平级位置即可。
不得删除 tests/CMakeLists 后凭 `-u` 退出 0 判断通过；要核对收集的源码和实际用例数。

## 包与验证

- `scripts/build_options.sh` 解析 `--pkg`、`--ops`、`--vendor_name`、`--soc` 和 UT 参数；构建/UT/example 分开运行。
- 包是 **`cann-ops-cv-<v>-linux.<arch>.run`**（注意 `-linux.`）；vendor 为 `<v>_cv`。依据：`cmake/package.cmake` / `variables.cmake`。
- `scripts/build_example.sh` 收 `examples/test_aclnn_*.cpp` / `test_geir_*.cpp`，ascend950 额外收 arch35；eager cust 支持 `ASCEND_CUSTOM_OPP_PATH`。
- 许可证按本仓 `OAT.xml`；不搬 math 的 tests 工具或命名空间。
- 本次复杂样例暴露了 NCHW 二进制 key 与默认格式匹配元数据不一致的问题。按源契约适配打包索引后重新验证；处理原则见共同配方，不能无条件把所有算子改成相同的 formatMode。
- GEIR 还须验证完整 dtype 推导注册。已有源示例失败时先记录源基线问题，不能把迁移后新增推导逻辑冒充原样迁移。
