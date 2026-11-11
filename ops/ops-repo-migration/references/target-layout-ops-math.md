# ops-math 目标布局与命令参考

结构基线：2026-10-21，master `16700ac8d`。本页只用于 ops-math；共同环境、隔离安装和回归规则见 [共同配方](migration-recipe.md)。

## 分类目录 `{op_class}`

`cmake/variables.cmake` 的 `OPS_CATEGORY_LIST` 是唯一真值：

| `{op_class}` | 说明 |
|---|---|
| `math` | 数值计算类 |
| `conversion` | 张量形态变换类 |
| `random` | 随机数生成类 |

- 目标路径形态：`<repo>/{op_class}/{OP}/`。
- 算子目录**无需在仓库任何清单里注册**：`{op_class}/CMakeLists.txt` 用 `file(GLOB SUBDIRECTORIES ...)` + `add_subdirectory` 收子目录。
- **不用 `experimental/`**：仓库根 `CMakeLists.txt:132-137` 是
  `if(ENABLE_EXPERIMENTAL) add_subdirectory(experimental) else() foreach(ops_category ${OPS_CATEGORY_LIST}) ... endif()`——标准分类与 experimental 是**互斥的两种构建模式**，进 experimental 后所有构建/example 命令都要带 `--experimental`。
- `ge/` 也在 `OPS_CATEGORY_LIST` 里，但放的是图算子（`Data`/`Const`/`Variable` 等），不是算子迁移的目标。

## 迁移后的标准布局

以 `math/abs/` 为参照（`M` = 必备，`O` = 可选，按算子实际情况保留）：

```
{OP}/
├── CMakeLists.txt                              M
├── README.md                                   M  产品支持情况 / 功能说明 / 参数说明 / 约束说明 / 调用说明
├── docs/<actual_aclnn_interface>.md             O  每个实际 aclnn 接口一个文件
├── examples/test_aclnn_{OP}.cpp                M  eager 示例（arch35 专属时可放 examples/arch35/）
├── examples/test_geir_{OP}.cpp                 O  仅当算子支持 GEIR 通路
├── framework/{OP}_tf_plugin.cpp                O  TensorFlow 插件注册（纯 aclnn+GEIR 场景不需要）
├── op_api/aclnn_<interface_file>.{h,cpp}        O  ACLNN L2，保留源包的全部接口文件名
├── op_api/{OP}.{h,cpp}                         O  ACLNN L0（namespace l0op）
├── op_graph/{OP}_proto.h                       M  REG_OP 内联
├── op_graph/{OP}_graph_infer.cpp               O  图级 **dtype** 推导（`InferDataType`）；InferShape 不放这里
├── op_host/{OP}_def.cpp                        M
├── op_host/{OP}_infershape.cpp                 M
├── op_host/{TILING_DIR}/{OP}_tiling*.{h,cpp}   M  每个 soc 对应的 tiling 实现目录
├── op_host/config/{soc}/{OP}_binary.json        O  缺省时构建自动生成
├── op_host/config/{soc}/{OP}_simplified_key.ini O
├── op_kernel/<entry>.cpp                       M  仓库支持的 kernel 入口（opFile.value 指向它）
├── op_kernel/{OP}_tiling_data.h                O
├── op_kernel/{OP}_tiling_key.h                 O
├── op_kernel/<source_subdir>/                  O  保留源包实际使用的 kernel 子目录
├── tests/ut/op_host/test_{OP}_infershape.cpp   M
├── tests/ut/op_host/{TILING_DIR}/test_{OP}_tiling*.cpp  M
├── tests/ut/op_api/test_aclnn_{OP}*.cpp        O  有 op_api 时
├── tests/assets/golden.py                      O  源 golden 副本；许可证及必要格式化按共同配方核验
└── tests/st/<source_arch>/ttk_kernel_{OP}_st.csv O  ST 用例（来自源工程 tests/blackbox/**）
```

单独的图级推导可放 `op_graph/{OP}_graph_infer.cpp`。源包已在同一个
`op_host/*_infershape.cpp` 中注册 InferShape / InferDataType 时，可以保留；核对当前聚合宏实际收集该文件，
不为套用目录模板拆改已验证的推导实现或重复注册。


## 算子 `CMakeLists.txt` 聚合宏

算子目录下**只有顶层这一份** `CMakeLists.txt`（新版形态：一份文件覆盖 `op_host` / `op_graph` / `op_api` / `op_kernel` 与 UT）。历史算子存在把 CMakeLists 拆到 `op_host/`、`op_graph/`、`tests/` 各一份的旧布局（用 `add_modules_sources`），**属历史遗留，不要照抄**。

以下占位列表必须展开为源包的实际值；不是默认芯片配置：

```cmake
set(SUPPORT_COMPUTE_UNIT <source_soc_list>)   # 支持的芯片版本，与下一行逐项对应、等长
set(SUPPORT_TILING_DIR   <source_tiling_dirs>) # 每个芯片版本用 op_host/ 下哪个子目录的 tiling
add_all_modules_sources(OPTYPE {OP} ACLNNTYPE <source_aclnn_mode> COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR} DISABLE_IN_OPP TRUE)
```
- `arch35` = Ascend950（dav_3510）；`arch22` = Atlas A2/A3（ascend910b / ascend910_93）。
- `ACLNNTYPE` ∈ {`aclnn`, `aclnn_inner`, `aclnn_exclude`}，与 `OPTYPE` 一一对应（两者长度不等会 `FATAL_ERROR`）。语义是**是否让框架自动生成 aclnn 接口**：`aclnn` = 没有手写 `op_api/`、由框架 autogen；`aclnn_exclude` = 不 autogen（算子不支持 aclnn，或**已有手写 aclnn**）。`op_api/*.cpp` 的收集与它无关，所以**有手写 `op_api/aclnn_*` 时必须 `aclnn_exclude`**。
- `DISABLE_IN_OPP TRUE`：tiling 源码不编进 CANN 内置包。
- 宏内部自动 glob：`op_api/*.cpp`（除 `aclnn_*`）、`op_api/aclnn_*.{h,cpp}`、`op_host/*_infershape*.cpp`、`op_host/{TILING_DIR}/*_tiling*.cpp`、`op_kernel_aicpu/*`，并在末尾调 `add_all_ut_sources(...)` 把 `tests/ut/` 的用例挂进来。
- 需要按 soc 传 kernel 编译选项时，另加 `add_kernel_sources(KERNEL_SRC <source_kernel_path> COMPUTE_UNITS <source_soc_list> OPTIONS ...)`（见 `math/add/CMakeLists.txt` 的机制，不复制其配置值）。

### 自定义 op_api 库的符号隔离

本页基线的 `--ops` 并不裁掉其他算子的全部 L0 源码；这些源码共用 `${OPHOST_NAME}_opapi_obj`。
其默认可见性会把内部 C++ 全局对象导出到 `libcust_opapi.so`，与 toolkit 的内置库重名时可能重复初始化、重复析构。
只给最终 shared target 加编译选项不会改变已经编好的 object target。

先核对源包全部公开接口、动态查找入口及回调有显式 default visibility（如 `ACLNN_API`），再将
[符号隔离片段](../assets/custom-opapi-visibility.cmake) 放在**迁移算子顶层 CMakeLists 的聚合宏之后**。
该片段只对 `ENABLE_CUSTOM` 下已存在的 opapi object target 生效，不写死算子名，不改仓级文件。
若当前仓已有等价隔离则保留；目标名或公共 ABI 不符合此前提时按真实构建链适配，不能直接隐藏所需接口。

重建后比对安装库的完整公共接口集合、检查内部符号不再抢占内置库，并完成共同配方中的正常退出验证。
不能通过换链接顺序、删除内置库依赖或绕过析构来代替符号隔离。故障证据见 [已验证问题](pitfalls-ops-math.md#26-数值通过但进程退出时堆损坏)。

## UT glob 契约

`cmake/ut.cmake::add_modules_ut_sources`（`MODULE_DIR` 即 `<op>/tests/ut/<层>/`）：

| 类型 | glob |
|---|---|
| Tiling | `${MODULE_DIR}/test_*_tiling.cpp` ＋ `${MODULE_DIR}/${MODULE_TILING_DIR}/test_*_tiling*.cpp` |
| InferShape | `${MODULE_DIR}/test_*_infershape.cpp` |
| op_api | `${MODULE_DIR}/test_aclnn_*.cpp` |
| op_graph | `${MODULE_DIR}/test_*_pass.cpp` |
| framework | `${MODULE_DIR}/test_*_onnx_plugin.cpp` |
| op_kernel | 需 `tests/ut/op_kernel/CMakeLists.txt` 存在，再 `add_subdirectory` |

参数构建与用例执行**都有仓库工具函数**，不必自己造：`tests/ut/common/` 下的 `tiling_context_faker.h` / `tiling_case_executor.h`（`gert::TilingContextPara` + `ExecuteTestCase`）与 `infershape_context_faker.h` / `infershape_case_executor.h`（`gert::InfershapeContextPara` + `ExecuteTestCase`）。**写法参考 `conversion/mirror_pad` 的 UT**（`tests/ut/op_host/arch35/test_mirror_pad_tiling_arch35.cpp`、`tests/ut/op_host/test_mirror_pad_infershape.cpp`）。

## 构建 / 运行命令（`scripts/build_options.sh` 为准）

| 目的 | 命令 |
|---|---|
| 编译算子包 | `bash build.sh --pkg --ops={OP} --vendor_name=<v> --soc=<soc> -j8` |
| 产物 | `build_out/cann-ops-math-<v>_linux-<arch>.run`（arch = x86_64 / aarch64） |
| 安装 | 按 .run 帮助安装到隔离目录；vendor 为 `<v>_math`（详见共同配方） |
| 只编译 op_host | `bash build.sh --ophost --soc=<soc> -j8` |
| op_host UT | `bash build.sh -u --ophost --ops={OP} --soc=<soc> -j8` |
| 其他 UT | `-u --opapi` / `-u --opgraph` / `-u --opkernel`；`--gtest_filter=<pattern>` 只跑匹配用例 |
| aclnn example | `bash build.sh --run_example {OP} eager cust --vendor_name=<v> --soc=<soc>` |
| graph example | `bash build.sh --run_example {OP} graph --soc=<soc>` |
| 仿真跑 example | `bash build.sh --run_example {OP} eager cust --simulator --soc=<soc>`（先确认该 soc 有仿真环境；graph 模式与 `--simulator` 互斥） |
| 脚手架新算子 | `bash build.sh --genop={op_class}/{op_name}` |

关键选项：
- `--soc`：默认 **ascend910b**（`CMakeLists.txt` 的 `ASCEND_COMPUTE_UNIT` 缓存默认值）；可选值含 `ascend950` / `ascend910b` / `ascend910_93` / `ascend350` / `ascend310p` / `mc62` 等。
- `--vendor_name`：**下划线**；默认 `custom`；决定包名与安装目录 `${ASCEND_HOME_PATH}/opp/vendors/{vendor}_math`。
- `--experimental`：切到 `experimental/` 分类构建模式；本次迁移不用。
- `--pkg` 与 `-u` / `--ophost` 互斥；`--run_example` 会强制 `ENABLE_CUSTOM=FALSE`。
- `--noexec`：只编不跑；`--ccache=off` 可关缓存；`--make_clean` 清 `build/`、`build_out/`、`third_party/`。

`--run_example` 的 example 发现使用 `find -path "*/{OP}/examples/*"`，文件名匹配 `test_aclnn_*.cpp`（eager）/ `test_geir_*.cpp`（graph）。注意 `find -path` 的 `*` 可以匹配 `/`，因此非 ascend950 分支也会发现 `examples/arch35/` 等子目录；ascend950 分支另加的 arch35 模式不是其他 soc 的排除规则。profile 的递归匹配据此保留，不能按 shell 顶层 glob 理解。**发现文件不等于该示例支持当前芯片**，仍须实际编译并运行全部适用示例；不可运行应明确阻断或提供源通路 N/A 依据。

规则来源：[固定验证版本的 build_example.sh](https://gitcode.com/cann/ops-math/blob/16700ac8d829f8e66180c612f45ed4384b51e2e0/scripts/build_example.sh#L234)；2026-10-29 复核[主线版本](https://gitcode.com/cann/ops-math/blob/f16246af9672fa03810ee5beb583d1aabfefa7d5/scripts/build_example.sh#L228)相同。发现后命中全部编译并逐个执行，可执行文件名固定为 `test_aclnn_{OP}` / `test_geir_{OP}`；后续仓库调整规则时同步复核 profile。
eager `cust` 模式的 include/lib/rpath 优先取 `ASCEND_CUSTOM_OPP_PATH`，未设置时取 `${ASCEND_HOME_PATH}/opp/vendors/{vendor}_math/op_api/{include,lib}`。

## 合规门禁（pre-commit / CI）

`.pre-commit-config.yaml` 对新增文件的要求：
- `clang-format`（仓库根 `.clang-format`，`--style=file -i`）——C/C++/`.asc` 必须格式化干净；
- `ruff-check` / `ruff-format`——Python；
- `trailing-whitespace`、`end-of-file-fixer`、`check-json`、`check-yaml`、`check-merge-conflict`、`detect-private-key`；
- `codespell`（跳过 `*.py,*.cpp,*.hpp,*.c,*.h,*.md`）；
- `oat-check`（`scripts/oat_check.sh`，按 `OAT.xml`）——**每个新文件的 CANN-2.0 许可证头与 Copyright 行必须合规**，否则阻断提交。

CI（`.gitcode/workflows/`，PR 评论 `compile` 触发）依次跑：编译 → 静态检查 → UT → 冒烟；`scripts/ci/check_example.sh` 用的就是 `bash build.sh --run_example $name eager cust --vendor_name=$name --soc=${SOC_VERSION}`。

## 参考算子
- `math/abs/`：一元 + op_api + docs + framework，完整布局。
- `math/add/`：二元 + `add_kernel_sources`。
- `math/square/`：eager example 只放 `examples/arch35/`。
- `math/cholesky/`：带 `tests/ut/op_kernel/CMakeLists.txt` 的 Kernel UT 写法。
- `examples/add_example/` 与 `scripts/opgen/template/add_example/`：官方最小脚手架。
