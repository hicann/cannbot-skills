# ops-nn 目标契约

结构取自 2026-10-21 的 master `82889267c9eee08660fc28a5e9bb98196ea0248f`；该版本已用于真实安装包迁移验证。
官方来源：[ops-nn](https://gitcode.com/cann/ops-nn)。实际 checkout 更新时重新核对以下文件。

## 分类与目录

`cmake/variables.cmake::OP_CATEGORY_LIST` 的 **14** 个标准分类：
`matmul conv activation foreach hash vfusion index loss norm optim pooling quant rnn control`。
`torch_extension`、`common`、`tests`、`examples` 都不是迁移分类。
`cmake/func.cmake::add_category_subdirectory` / `process_subdirectories` 收集算子目录；不用改仓级清单。

## 构建接入

工作流的 `op_api/` 与 `op_host/` 平级，保留这个形态。新迁移使用算子顶层 `CMakeLists.txt`：
占位符按当前算子的目录名、源支持芯片及 tiling 目录展开，不能复制参考算子的取值。

```cmake
set(SUPPORT_COMPUTE_UNIT <source_soc_list>)
set(SUPPORT_TILING_DIR <source_tiling_dirs>)
add_all_modules_sources(OPTYPE <OP> ACLNNTYPE <source_aclnn_mode>
    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR}
    DISABLE_IN_OPP TRUE)
```

`cmake/func.cmake::add_all_modules_sources` 自动选择 op_api 路径并调用 `add_all_ut_sources`。
`loss/sigmoid_focal_loss/CMakeLists.txt` 展示该入口；`matmul/mat_mul_v3` 也使用该宏。
旧算子的 `add_modules_sources`、嵌套 op_api 和多层 tests/CMakeLists 不要求复制到新迁移。
源码支持列表可直接写在宏参数，也可用上述等长列表；不能把算子未支持的 soc 从参考例复制过来。

## UT 契约

`cmake/ut.cmake::add_modules_ut_sources` 根据 **soc 映射的 target_dir** 收测试，
映射在 `cmake/func.cmake::map_compute_unit` / `get_target_dir`，不直接采用 `SUPPORT_TILING_DIR`。

| 类型 | 收集路径（相对 tests/ut） |
|---|---|
| Tiling | `op_host/test_*_tiling.cpp` 和 `op_host/<target_dir>/test_*_tiling.cpp` |
| InferShape | `op_host/test_*_infershape.cpp` 和 `op_host/<target_dir>/test_*_infershape.cpp` |
| ACLNN | `op_api/test_aclnn_*.cpp`（也兼容历史 op_host 位置） |

特别注意：`test_x_tiling_arch35.cpp` **不会**被 nn 收集；改用 `arch35/test_x_tiling.cpp`。
ascend950 → arch35；ascend910b / ascend910_93 → arch22。其他 soc 查当前映射。

`soc_arch` 在本仓表示 **UT 收集目录**，不是芯片物理架构表。`ascend310p` 经 `map_compute_unit` 转为 `ascend310p1` 后，`get_target_dir` 实际返回 `arch22`；源码 tiling 目录仍取算子的 `COMPUTE_UNIT/TILING_DIR`，example 的 310P 子目录则是 `arch20`，三者不能混用。依据：[固定验证版本](https://gitcode.com/cann/ops-nn/blob/82889267c9eee08660fc28a5e9bb98196ea0248f/cmake/func.cmake#L1208)与 [2026-10-29 复核版本](https://gitcode.com/cann/ops-nn/blob/d14537dcbf2e0b8ac1439e8bc94d214b7c203a7e/cmake/func.cmake#L1208)。目标 checkout 改变映射时，应按实际收源结果同步配置，不能照其他仓的物理架构名称替换。
参考 `index/scatter_add` 的用例内容时，只取参数构造与断言，接入方式仍用新宏；使用 nn 自己的 `tests/ut/common`，不照抄 `Ops::Math` 命名空间。

## 出包与 example

- 命令参数由仓顶层 `build.sh` 解析：`--pkg --ops=<OP> --vendor_name=<v> --soc=<soc>`，UT 单独 `-u --ophost --ops=<OP> --soc=<soc>`。
- 包：`cann-ops-nn-<v>_linux-<arch>.run`；安装 vendor 为 `<v>_nn`。来源：`cmake/variables.cmake`、`cmake/package.cmake`。
- example：`examples/test_aclnn_<OP>.cpp` / `test_geir_<OP>.cpp`；运行 `--run_example <OP> eager cust --vendor_name=<v> --soc=<soc>` / `graph --soc=<soc>`。
- `build.sh` 的 example 收集还支持 ascend910b 的 `examples/arch22/`、ascend310p 的 `examples/arch20/` 及 ascend950/ascend350/ascend960dt/ascend960pr 的 `examples/arch35/`；其他 soc 不自动套用这些子目录。文件名匹配 `test_aclnn_*.cpp` / `test_geir_*.cpp`，可对应实际公开接口名。
- eager cust 读取 `ASCEND_CUSTOM_OPP_PATH`；依赖 `opapi_math` 等内置库，保留同一 toolkit 的系统库路径。
- `OAT.xml` / `.pre-commit-config.yaml` 为本仓门禁依据；实现语义保持不变，必要格式化按共同配方核验。

## 共享库链接依赖

上述基线的 `gen_cust_proto_symbol` 把公共 `cube_utils` 对象合入 `cust_proto`。
若其 `aclsysGetVersionNum` 未通过依赖库解析，安装器会拒绝加载 `libcust_opsproto_rt2.0.so`。
先确认当前 toolkit 的 `libacl_rt.so` 实际导出该符号，并用 `ldd -r` 验证缺失来自这个目标；
确认后可在**迁移算子目录**的 CMake 增加以下链接适配，不改公共仓文件：

```cmake
# CMake >= 3.19；cust_proto 在算子目录收集结束后才创建。
if(ENABLE_CUSTOM)
    function(migration_link_graph_runtime)
        if(TARGET cust_proto)
            target_link_libraries(cust_proto PRIVATE acl_rt)
        endif()
    endfunction()
    cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL migration_link_graph_runtime)
endif()
```

这是已核实的仓级依赖关系，不是待迁移算子的固定依赖列表。
其他缺失符号应重新定位提供库；仓库已修复、目标名变化或 CMake 版本不支持 `DEFER` 时，
按实际构建接入点处理，不能无条件复制。重新出包后保留安装器校验，并复核原生程序正常退出。
