> 本页是 ops-math 的既有实跑经验，不代表其他三仓。通用环境规则以 SKILL.md / migration-recipe.md 为准。

# 踩过的坑（ops-math 版）

## A. 构建参数

### 1. `--soc` 必须显式取自当前源包
ops-math 的 `--soc` 默认是 **ascend910b**（`CMakeLists.txt` 的 `ASCEND_COMPUTE_UNIT` 缓存默认值），不能把它当当前算子的支持芯片。历史复现样例仅支持 Ascend950，tiling 只在 `op_host/arch35/`，因此该样例需要 `--soc=ascend950`。其他算子按源 OpDef、构建配置和已验证产物逐 soc 执行，不能据此扩大、缩小或替换支持集。
`--soc` 可选值见 `scripts/build.conf.sh` 的 `SUPPORT_COMPUTE_UNIT_SHORT`（含 `ascend950` / `ascend910b` / `ascend910_93` / `ascend350` / `ascend310p` / `mc62` 等）。

### 2. 构建开关是 `--vendor_name`（下划线）
写成 `--vendor-name` 会被 usage 拒绝。名称同时决定包名 `cann-ops-math-{vendor}_linux-{arch}.run` 与**安装目录 `${ASCEND_HOME_PATH}/opp/vendors/{vendor}_math/`**（注意多出的 `_math` 后缀）。默认 `custom`。

### 3. `--pkg` 与 `-u` / `--ophost` 互斥，必须分两次调用
`scripts/build_options.sh::check_param`：`--pkg` 不能和 `-u`、`--ophost`、`--opapi`、`--opgraph` 同批。包构建、UT、example 是三次独立调用。
反过来，`-u --ophost` **可以**带 `--ops=`（仅在 `ENABLE_TEST=FALSE` 时才互斥）。

### 4. `.run` 的 `--quiet --force` 在 makeself 层消费
`.run` 由 makeself 打包，仓库自己的 CI 就用 `./cann-ops-math-<v>_linux-*.run --quiet --force`（`scripts/ci/check_example.sh`）。内层安装脚本（`scripts/package/ops_math/scripts/install.sh`）另有 `--full` / `--upgrade` / `--uninstall` / `--check` / `--install-path=<path>` / `--install-for-all` / `--quiet`，**没有** `--force`——不要把它当内层参数用。不确定时先跑 `./<pkg>.run --help`。
重复安装同一 vendor 用 `--upgrade`，或先 `--uninstall`。

### 5. graph example 与 `cust` / `--simulator` 的关系
`compile_graph_example` 不分包模式，直接链 `-lgraph -lge_runner -lgraph_base -lge_compiler`——graph 模式下给 `cust` 没有意义。且 `graph` 与 `--simulator` 互斥（`check_param` 直接报错）。
eager `cust` 模式的 include / lib / rpath 优先取 `ASCEND_CUSTOM_OPP_PATH`，未设置时取 `${ASCEND_HOME_PATH}/opp/vendors/{vendor}_math/op_api/{include,lib}`。

## B. 算子代码与 CMake

### 6. `opFile.value` 必须等于 kernel 入口文件名
`ExtendCfgInfo("opFile.value", "<name>")` 与实际 `op_kernel/` 下入口 `.cpp` 文件名不一致时，build 报 `kernel entry '...' not implement`。工作流包入口名通常带 `_apt`（如 `{OP}_apt`），对应 `op_kernel/{OP}_apt.cpp`。
- 默认入口还须映射到仓库暂存的 `{OP}/` 目录：使用 `{OP}`、`{OP}_apt` 或 `{OP}_910b`，对应文件直接放在 `op_kernel/` 下。独立包的 `{OP}_arch35.cpp` 不能仅因同名文件存在就通过；须保持实现不变，适配入口文件名与 opFile，见共用迁移配方。
- 例外：`math/add` 使用 `add_kernel_sources(KERNEL_SRC arch35/add.cpp ...)` 显式指定源码，`add_apt` 是 autogen 产物名。这类自定义映射须单独核对源码列表及实际构建结果，不在默认静态模板支持范围内。

### 7. 缺 `.AddConfig("<soc>")` 会被静默跳过
`cmake/gen_ops_info.cmake::check_op_supported` 用 `grep '.AddConfig("<soc>"'` 判定算子是否支持某 soc；不匹配时该 soc 下**不报错、直接不编**。只编 ascend950 的算子必须在 `_def.cpp` 里显式 `this->AICore().AddConfig("ascend950", aicoreConfig);`。

### 8. `_def.cpp` 里多一行含 `OP_ADD(` 的注释 → binary 配置生成失败
`cmake/func.cmake::get_op_type_from_op_name()` 用
```cmake
find ${OP_DIR} -name ${OP_NAME}_def.cpp -exec grep OP_ADD {} \;
```
反解 op_type，再靠正则截取。**当 `op_host/config/{soc}/{OP}_binary.json` 未随包提供、需构建自动生成时**，`_def.cpp` 中除唯一一行 `OP_ADD(<Op>);` 外若还有含 `OP_ADD(` 的行（典型是注释），会污染反解（实测 op_type 变成注释残句），导致 binary json 无法生成、`--pkg` 失败。
- 若该 `_binary.json` 已存在，此路径被跳过，注释无影响（`gen_ops_info.cmake` 的 `get_op_type_and_validate`：有 json 走 self config，无 json 走 def config）。
- 迁移自工作流包（通常无 binary.json）时务必保证 `_def.cpp` 只有一行 `OP_ADD(...)`，或显式提供 binary.json。

### 9. `SUPPORT_COMPUTE_UNIT` 与 `SUPPORT_TILING_DIR` 必须等长
两者是**按位置对应**的列表，由 `find_value_by_key` 按当前 soc 取同下标项。长度不等或下标错位会把 tiling 目录指错。

### 10. 算子目录下不要建 `tests/CMakeLists.txt`
ops-math 的 UT 由仓库根 `tests/ut/CMakeLists.txt` 统一构建，并按 glob 契约从各算子的 `tests/ut/` 收用例。算子目录下多一个 `tests/CMakeLists.txt` 不会被任何 `add_subdirectory` 引用，是死文件。注意这与"ops-nn 写法"无关：ops-math 里确实有约 144 个历史算子带 `tests/CMakeLists.txt`，那是把 CMakeLists 拆到 `op_host/`、`op_graph/`、`tests/` 多份的**旧布局**（`add_modules_sources`）；当前形态是**算子目录下只有顶层一份** `CMakeLists.txt`（`add_all_modules_sources`，见 `target-layout-ops-math.md`），迁移产物一律按新形态，不要照抄旧布局。

### 11. tiling UT 的目录名必须等于 `SUPPORT_TILING_DIR`
glob 是 `${MODULE_DIR}/${MODULE_TILING_DIR}/test_*_tiling*.cpp`。tiling UT 放在 `tests/ut/op_host/arch35/` 而 `SUPPORT_TILING_DIR` 又不是 `arch35` 时，用例**不会被收进 UT 目标**，`-u` 跑出"全绿"其实是零用例。

### 12. `examples/` 命中即全编全跑
example 发现是 `find */{OP}/examples/*`（ascend950 额外含 `examples/arch35/*`），文件名匹配 `test_aclnn_*.cpp` / `test_geir_*.cpp` 就全部编译并逐个执行。目录里不能留草稿、注释掉的半成品或工作流的 ST 套件——任何一个编不过都会让 `--run_example` 失败。

### 13. `.AutoContiguous()` 在 ops-math 不强制
`math` / `conversion` / `random` 共 311 个 `*_def.cpp`，其中只有 65 个使用 `.AutoContiguous()`；`math/abs`、`math/add`、`math/sqrt`、`math/square` 都不用。迁移时**保留源工程原样**，不新增也不删除。（这条与面向 ops-nn 的迁移口径不同，不要照搬。）

### 14. `experimental/` 是互斥构建模式
仓库根 `CMakeLists.txt`：`if(ENABLE_EXPERIMENTAL) add_subdirectory(experimental) else() <标准分类> endif()`。把算子放进 `experimental/{op_class}` 后，`--pkg`、`--run_example` 都必须带 `--experimental`，否则既编不到也跑不到。本次迁移目标目录是 `math` / `conversion` / `random`，不涉及。

### 15. `ACLNNTYPE` 写错 → 重复定义，或根本没有 aclnn 接口
`ACLNNTYPE` 的含义是**"要不要让框架帮我生成 aclnn"**，不是"我要不要导出 aclnn"：
- `aclnn`：没有手写 `op_api/`，让框架 autogen 出 aclnn 接口；
- `aclnn_exclude`：不要 autogen——算子不支持 aclnn，或**已有手写 aclnn 接口**。

`op_api/*.cpp` 的收集与 `ACLNNTYPE` **无关**（`cmake/func.cmake` 里 op_api 的 glob 在 OPTYPE 循环之前、无条件执行）。所以**有手写 `op_api/aclnn_*.{h,cpp}` 却填 `aclnn`**，会把手写实现与 autogen stub 一起编进去 → 重复定义；反过来没有手写 `op_api/` 却填 `aclnn_exclude`，则完全不会有 aclnn 接口。

迁移判定：源工程有 `op_api/aclnn_*` → `aclnn_exclude`（工作流产物通常自带手写 aclnn）。实测：全仓用 `aclnn` 的算子（`hans_decode` / `amp_update_scale` / `hans_encode` / `strided_slice_grad` / `sim_thread_exponential` / `add_lora`）**都没有 `op_api/` 目录**。

## C. 环境与测试

### 16. 单一 CANN 版本（否则 graph example 链接 / 运行失败）
- 现象：`--pkg` 通过，但 `run_example graph` 报 `undefined reference` 或运行期加载内置 graph so 失败。
- 根因：`LD_LIBRARY_PATH` 同时混入两个 CANN 版本。来源叠加——`/etc/profile` 先 source 了一个版本的 `set_env.sh`；`set_env.sh` 只 `remove_env` **自身前缀**、从不重置，再 source 另一版本不会清掉前一版本；手动 `export ASCEND_HOME_PATH=...` / 反复 source 会继续累积。
- 修法（构建前归一）：
  ```bash
  unset LD_LIBRARY_PATH
  source <目标 CANN>/set_env.sh
  ```
  `ASCEND_HOME_PATH` 决定"选哪个版本"，`LD_LIBRARY_PATH` 决定运行时解析顺序，两者要一致。

### 17. 工作流 UT 不可直接复用
源 UT 的独立驱动可能自带 `int main`，与仓级测试入口冲突。适配为仓库支持的测试驱动、移除独立 main，并按第 11 条的目录/命名契约落位；保留实际测试类、参数组合、断言、oracle 和期望值，不强制缩成单 class 或用少量硬编码期望替代原用例。

### 18. example 无 golden，proto include 是相对路径
ops-math example 只验证调用链打通，不做精度对比；不要把工作流的 ST suite 搬进来。
GEIR example 的 proto include：`examples/` 下用 `"../op_graph/{OP}_proto.h"`（`math/abs/examples/test_geir_abs.cpp:30`）；`examples/arch35/` 下用 `"../../op_graph/{OP}_proto.h"`。

### 19. 注册算子类型名未必是朴素 PascalCase
以 `OP_ADD(...)` / `class ... : public OpDef` / `REG_OP(...)` 的实际登记名为准（如 `GNTrainingReduce`，非 `GnTrainingReduce`）。

### 20. 目标注册放在 proto 头文件，不能盲删源 `.cc`
核对的仓库布局将 `REG_OP` 内联在 `op_graph/{OP}_proto.h`。源 `*_proto.cc` 只有确认为模板空壳时才可直接删除；若承载真实注册或推导实现，先确认等价接入与源码映射。无法在允许的注册胶水改动内完整保留时，报告阻碍，不以删除文件满足静态模板。

### 21. 优先参考库上同类算子
遇到迁移问题，对照 ops-math 已合并算子的工程写法。**只在"不好用语言描述、必要的时候"才给参考算子**——不要每一项都塞一句"参考某某"，能用文字说清的就用文字说清。

### 22. graph example 失败先做对照，并复用已验证环境
`--run_example {OP} graph` 报 `fatal error: array_ops.h: No such file or directory` 时，先检查锁定 toolkit 的实际头文件路径。需要对照时选择在同一环境和 soc 已验证、确实支持 graph 的参考算子，并记录其名称与依据；不能默认某个参考算子或芯片总是可用。

工作流先复用本条链 verify.log / env-info.txt 中已验证的 toolkit 与 TTK 路径，不重复询问、不自行换版本。独立调用仅在任务未给定环境时才询问。校验 `${ASCEND_OPP_PATH}/built-in/op_proto/inc/array_ops.h` 可提前暴露缺失依赖。

### 23. UT 的三个细节（`math/abs` / `mirror_pad` 的用例里看不出来的）
1. **`ExecuteTestCase` 会把整个 tiling buffer 按 `int64` 全量比对**（`tiling_case_executor.cpp`：`to_string<int64_t>(rawTilingData->GetData(), rawTilingData->GetDataSize())`）。自定义 TilingData 结构要按 int64 视角拼那个期望串；**不要**因为不好拼就绕开它自己去 `ExecuteTiling()`（那样不规范）。
2. **InferShape 的输出 shape 不来自 para 里的 output desc**（`infershape_case_executor.cpp` 只把 dtype/format 传给 `NodeOutputTd`）。所以**数据依赖型 InferShape**（读输入张量取值来定输出）在 UT 里拿不到张量数据，输出会退化成动态维（`-1`）。
3. **UT 的 soc 来自 `cmake/ut.cmake` 的 `BUILD_SOC_VERSION=${ASCEND_COMPUTE_UNIT}`**：UT 必须使用当前源包支持、与该用例匹配的 soc。历史样例的 `ASCEND950` 守卫只是其中一种，不应复制到其他算子的实现或测试。

### 24. 去注释比较必须保留字符串与 token 边界
`sed 's://.*::'` 会误删字符串里的 `//`；`tr -d '[:space:]'` 又会把 `"a b"` 与 `"ab"`、`x + ++y` 与 `x++ + y` 错判为相同。
使用 `scripts/verify_logic_unchanged.sh`，保留预处理后的空白并检查预处理退出码。脚本仅输出差异清单；移位文件仍须按路径映射核对，不能凭清单退出 0 宣称零算法改动。

## D. 通用口径

### 25. 任何"要不要顺手改成某某形态"的犹豫，默认答案都是**不改**
迁移的红线是**不改代码**：源码一律原样搬过来，**最多只改注释**（例如 OAT 要求的 CANN 许可证头）。算法逻辑（kernel 计算链 / TilingFunc 公式 / InferShape·InferDataType 语义）只是其中一个子项。
碰到"这里要不要顺手统一一下风格""要不要挪个位置"这类念头时——**不改**。只做 skill 允许且经当前源契约确认的结构适配；构建缓存、独立驱动和模板空壳须先辨认，不能按固定文件清单盲删实现。

### 26. 数值通过但进程退出时堆损坏

- 现象：冻结回放全部数值通过，随后报 `corrupted size vs. prev_size in fastbins`；检测栈可能落在 `ArgsPool` 或 `KernelMgr` 析构处。
- 已确认根因：本页基线把其他算子的 L0 实现一并编入自定义 op_api 库，未隐藏的内部全局对象与内置 `libopapi.so` 同名，动态链接将两库引用绑定到同一对象，两次析构造成 use-after-free / double-free。检测栈最后一帧不是首次破坏内存的位置。
- 区分证据：不初始化 ACL、不执行算子，仅以原依赖顺序加载两库后正常返回也会崩溃；Valgrind 记录同一地址先由自定义库释放、再由内置库释放，`LD_DEBUG=bindings` 确认跨库绑定。
- 修复：按 [目标布局](target-layout-ops-math.md#自定义-op_api-库的符号隔离) 隔离 **object target** 的内部符号，保留源 ABI 的显式公开导出。无需改 kernel、tiling、推导或 ACLNN 算法，也不能只屏蔽某个历史样例的变量名。
- 验证：实际安装库与打包 payload 哈希一致；公开接口仍可调用；最小装载程序正常退出且无非法内存访问；原冻结回放全部通过并正常退出。此根因已在 x86_64/CANN 9.3.0 的本机重现，不能据此声称旧 ARM64/CANN 9.0.0 环境已重新验证。
