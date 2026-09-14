# 阶段 3 编译与安装：错误速查表

> 定位：build_and_install.sh 执行失败时的排查速查——按「错误信号 → 原因 → 解决」组织。
> 素材来源：真实迁移踩坑记录（编译 → 安装 → 验证三步的实际故障），非泛泛理论。

## 1. 安装步报 `No such file or directory`（路径拼接错误）

| 项 | 内容 |
|----|------|
| **信号** | `build_and_install.sh: line N: .//home/.../build_out/cann-ops-transformer-*.run: No such file or directory`（路径带 `.//` 前缀） |
| **原因** | `RUN_PKG` 已是绝对路径（`${OP_DIR}/build_out/...`），执行时误写 `./${RUN_PKG}`——`./` + 绝对路径拼成 `.//home/...`。bash 把 `.//` 当相对路径解析，找不到文件 |
| **解决** | 绝对路径变量**禁止加 `./` 前缀**，直接执行 `${RUN_PKG}`。模板 `build_and_install.sh.template` 已按正确写法提供 |
| **判据** | 编译日志（CPack 段）实际已成功——`.run` 包已生成，问题只在 install 步的路径写法，不是构建失败 |

## 2. 功能行为异常：runtime 回退内置 kernel（tiling 未注册）

| 项 | 内容 |
|----|------|
| **信号** | 编译安装全 PASS，但精度/行为与预期不符，疑似跑的不是自定义实现 |
| **原因** | build.sh 缺 `--enable_host_tiling`：regbase 算子的 tiling 模板（`REGISTER_TILING_TEMPLATE_WITH_SOCVERSION`）需编译进 vendor 的 op_tiling 库；缺此选项时 run 包不含 tiling 注册，runtime 回退内置 tiling → 加载内置 kernel |
| **解决** | 编译命令必须带 `--enable_host_tiling`（模板已含） |
| **验证** | `strings libcust_opmaster_rt2.0.so \| grep -c "<tiling 类名关键词>"` 应 > 0 |

## 3. `libcust_opapi.so` 运行时加载失败（stub 库缺失）

| 项 | 内容 |
|----|------|
| **信号** | 安装校验/调用时 `libcust_opapi.so` 加载报错，找不到符号 `libopapi_math` |
| **原因** | `libopapi_math.so` 是 open 构建框架自产 stub（`common/stub/op_api`），run 包打包遗漏；缺失导致 `libcust_opapi.so` 加载失败 |
| **解决** | 安装后检查 `VENDOR_DIR/op_api/lib/libopapi_math.so`，缺失则从 `${OP_DIR}/build/libopapi_math.so` 手动补齐（模板 Step 4 已含） |

## 4. `nm` 验证符号看不到（符号被 strip）

| 项 | 内容 |
|----|------|
| **信号** | 按常规 `nm -D libcust_opapi.so` 验证 aclnn 符号，输出为空 |
| **原因** | vendor 库编译为 `-fvisibility=hidden` 且链接时 `-s` strip——nm 不可见属正常 |
| **解决** | 用 `strings` 检查字符串（tiling 注册类名等）替代 nm；aclnn 接口符号用 `grep -i "aclnn"` 从 nm 输出过滤仍可见的部分 |

## 5. 首次编译失败（编译期错误）

| 项 | 内容 |
|----|------|
| **事实** | cube 类算子迁移编译通常一次通过（真实记录：编译 0 error）——编译期错误优先怀疑代码改动，其次才是环境 |
| **常见点** | ① tiling 相关链接错误 → 检查 `REGISTER_TILING_TEMPLATE_WITH_SOCVERSION` 宏与 host 侧 CMakeLists 的 arch35 分支；② 第三方依赖（json/protobuf/abseil）下载失败 → 见第 6 条 |
| **解决** | 按编译器报错定位；C++ 模板实例化错误（kernel 头）先核对模板参数与平台宏（`__CCE_AICORE__ == 310`）分支 |

## 6. 依赖下载失败（网络/离线）

| 项 | 内容 |
|----|------|
| **信号** | cmake FetchContent 阶段卡住或报下载失败（`cmake/third_party/*.cmake` 自动拉取 json/protobuf/abseil/makeself 等） |
| **原因** | 构建依赖不在仓库内，构建时自动下载；离线/受限网络下失败 |
| **解决** | 首次构建前预置依赖或保证网络可达；`third_party/` 为下载产物，删除后构建会重新下载（无需手动归档） |
