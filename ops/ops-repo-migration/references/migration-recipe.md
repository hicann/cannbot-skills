# 共用迁移配方

仓差异见 `target-layout-ops-*.md`，参数见 `repo-profiles.json`。只读取本次目标仓的布局。
以下命令在目标 checkout 执行，`OP`、`SOC`、`VENDOR`、`JOBS`、`INSTALL_ROOT` 取当前任务的真实值，不能照抄示例值扩大支持面。

## 从源工程确定迁移参数

先在迁移报告中列出参数及事实来源，再生成目标文件和执行命令：

| 参数 | 来源与约束 |
|---|---|
| `SOURCE_PACKAGE` / `OP` | 工作流为 `$WORK_DIR/op` 与 spec 的 `op.name`；独立调用使用给定源包和规格，核对实际文件名，不默认任何样例算子 |
| `OP_TYPE` | 源 `OP_ADD` / `REG_OP` 的实际注册类型，保留缩写、大小写和数字，不从 snake_case 猜 PascalCase |
| ACLNN 接口集合 | 原 public headers、导出符号及已验证调用；保留全部接口和签名，不假定接口后缀等于 OP_TYPE，也不限定输入输出数量 |
| `SOURCE_VENDOR` | 已验证的安装记录和实际 vendor 目录；核对头文件与库属于当前源包，不假设 vendor 名等于 OP_TYPE |
| soc / tiling 目录 / tilingKey | 源 OpDef、构建配置、设计与已生成产物交叉核对；逐 soc 取值，不默认为单芯片或固定 key 集 |
| 验证通路 / 用例 | 原 kernel、ACLNN、GEIR、torch 交付件和冻结套件；保留实际 dtype、shape、属性、golden 与完整数量，不套用参考算子的算式或用例 |

这些值互相矛盾或无法从现有产物确定时，报告具体缺失，不能通过样例值补齐。
自动生成与手写 ACLNN 不是可互换的包装方式；迁移须保留已验证的接口实现。
逐个核对 public header 中的 GetWorkspaceSize 与执行函数，不能假定每个带后缀的 workspace 入口都有同名执行函数。
头文件可能位于 `op_api/include/` 或其 `aclnnop/` 子目录；递归核对声明、实际导出符号和示例 include 路径。
仅在磁盘找到头文件不代表接口可链接。
源包含多个注册算子或自定义构建布局时先核对依赖闭包及目标仓接入能力；不得删掉依赖或强行改写成单算子模板来通过检查。

## 环境与隔离

工作流复用已验证的 CANN 与 `$WORK_DIR/ops-test-kit`；证据在 `ttk-verify/env-info.txt` / `op/examples/verify.log`。
独立调用优先复用任务已提供的路径，缺失时才询问。source 前清理其他 CANN 版本的动态库路径，不临时更换 toolkit。
有 GEIR 通路时用 `preflight.sh --require-geir` 检查 `${ASCEND_OPP_PATH}/built-in/op_proto/inc/array_ops.h`。如果缺失，报告环境阻碍，不另选环境；无此通路时记录源包依据，不增加 GEIR 要求。

`preflight.sh` 只检查指定环境和目标；不能用它自动发现“能用的版本”。在目标仓装好 pre-commit。
`git status`、remote、HEAD、目标目录占用都确认后再建 `feat/{OP}`。已有迁移 checkout 只在远端和迁移基线一致、无无关改动时续用，不能仅凭存在 `.git` 就复用。
源包、原安装产物、golden、spec、设计及冻结套件保持只读；工作流回归日志与结果统一写 `$WORK_DIR/migration/evidence/`，独立调用使用任务的迁移证据目录。

## 拷贝与注册

只搬源码、文档及后续需要适配的测试输入，排除 build、build_out、dev-package、缓存、二进制和日志。
移除目标副本的独立构建脚本和工作流测试驱动；源套件留在原位置做迁移后回归。
`op_graph/*_proto.cc` 若确为模板空壳才删除；若含真实注册，先保持等价注册定义再移除，不能丢签名。

允许的产品适配为目录、构建、注册胶水、许可证注释及经核验的目标仓必需格式化。以下实现必须保持：kernel 计算、tiling 公式、InferShape/InferDataType、L0/L2 ACLNN 语义。格式化在构建前按 [格式化核验](formatting-validation.md) 收敛，不修改冻结源包。
不改 dtype/format/属性默认值、soc 支持集或 `.AutoContiguous()`。
`OP_ADD` 与 `REG_OP` 类型一致，每个受支持 soc 有显式 AddConfig。
`opFile.value` 不仅要对应文件，还必须映射到仓库暂存的 `{OP}` 目录：math/nn/cv 支持 `{OP}`、`{OP}_apt`、`{OP}_910b`；transformer 支持 `{OP}`、`{OP}_apt`、`{OP}_a2`、`{OP}_a3`。入口 `.cpp` 放在 `op_kernel/` 根目录。
独立包若由 CMake 指定任意入口名（如 `{OP}_arch35.cpp`），迁移时按路径映射重命名为仓库支持的入口，保持内容与入口符号不变，再设置对应 opFile；不能只因原文件存在就认为构建能找到它。其他自定义 kernelSrc 布局须单独核对实际构建规则，不在此静态模板支持范围内。
四仓真实构建均须核对当前算子的二进制；transformer 无法匹配入口时可能跳过 kernel 而仍出包成功。
math/nn/cv 的 grep 推导路径要求 `_def.cpp` 仅一处 `OP_ADD(`，包括注释也不能额外出现。
手写 `op_api/aclnn_*` 时禁止再让框架 autogen 同名接口；选择 `aclnn_exclude`。

## 许可证与 CMake

从目标仓 `OAT.xml`、同类文件核对 CANN 许可证，年份取当前年，保留原有合法作者信息。
常用文本如下，必须最终以目标仓门禁为准；CMake/Python 改为 `#` 行注释。

```cpp
/**
 * Copyright (c) <year> Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
```

math/nn 用顶层聚合宏并自动收 UT；cv 用顶层聚合宏但显式接 tests；transformer 用 op_host 的架构感知宏和显式 tests。
详细模板仅放在各仓布局页。宏不会收集的源码必须通过本算子 CMake 接入，不能改仓级框架或改写算法绕过。
核对头文件和辅助 `.cpp` 的完整依赖，包括跨算子公共头、仓内公共实现及 toolkit 提供的外部接口；
源包已经内置的公共实现与目标仓已有实现不能重复编译。
手写 `IMPL_OP_OPTILING(...).TilingParse<...>()` 需要保留真实注册。
某些 SDK 的 `OP_TILING_LIB` 分支会由 OpDef 注册占位 TilingParse；检查实际收源与注册顺序，
避免覆盖手写解析函数。不能通过删除 CompileInfo 检查或改 tiling 公式来让错误注册继续运行。

## example 与测试

- `examples/test_aclnn_<OP>.cpp`：按当前签名申请输入/输出、GetWorkspaceSize、执行、同步、读取结果、释放资源；不搬整套 ST。
- `examples/test_geir_<OP>.cpp`：只在源包支持 GEIR 时提供，使用 `../op_graph/<OP>_proto.h`；arch35 子目录则多一级 `../`。
- 头文件 include 对照目标仓 example 的实际 include flags，不能假定所有仓都带同一前缀。
- example 是否被收集按目标仓及当前 soc 的规则判断（profile 的 `example_dirs`），不能统一只查根目录和 arch35。文件后缀可以是实际接口名，不必等于算子目录名。
- UT 按目标仓工具改写驱动，保留源包已验证的断言；不新增自带 main。math 与其他三仓的 tiling 文件名规则不同，nn 的 UT 选目录还取决于 soc 映射。
- golden 整文件拷到 `tests/assets/golden.py`，只补注释许可证及经 AST 比较和差异审阅确认等价的必要格式化；源 golden 不变。ST 可随仓约定归档，但不能靠随机抽样取代冻结套件回归。

## 构建与安装

先对每个源包支持的 SOC 跑 `post_migration_check.sh <operator_dir> <soc> <target_repo> <source_optype>`；有 FAIL 先修结构。soc 必填；source_optype 取源注册定义，不从目标副本反推。
脚本只做静态检查，显式测试接入是否在当前条件下生效还要看实际 CMake 收源和 UT 用例数。

```bash
bash build.sh --pkg --ops="$OP" --vendor_name="$VENDOR" --soc="$SOC" -j"$JOBS"
# 在 build_out 中按 profile.package_glob 找到本次生成的唯一 .run，记录文件路径和时间。
# 先查看该 .run --help / -- --help 及仓内安装脚本，再用其支持的 --install-path 安装到 INSTALL_ROOT。
# 不沿用默认 toolkit 安装位置，不用通配符同时执行多个历史包。
bash build.sh -u --ophost --ops="$OP" --soc="$SOC" -j"$JOBS"
```

工作流构建并行度遵守机器限制；未给限制时示例默认 `JOBS=8`。`--pkg` 和 UT 分两次调用。
四仓包名分隔符不同，不用统一拼字符串。确认隔离安装后实际 vendor 路径为 `<VENDOR><vendor_suffix>`。
编译成功不代表共享库可独立加载。保留安装器的共享库校验；遇到 `undefined symbol` 时，
在锁定的 toolkit 环境中用 `readelf -d`、`ldd -r` 和 `nm -D` 核对依赖、实际解析路径及符号提供方。
优先在迁移算子目录的 CMake 中给实际缺依赖的目标补链接，再重新出包、正常安装；
不要用安装器 `--force`、`LD_PRELOAD` 或删除功能源码把失败变成验收通过。
目标仓共用对象也可能引入依赖，不能只检查当前算子的 `.cpp`；已验证的 nn 例子见其布局页。
优先按新包环境脚本设置运行环境，并确保 `ASCEND_CUSTOM_OPP_PATH` 指向该 vendor，
`LD_LIBRARY_PATH` 中该 vendor 的 op_api / op_tiling / op_proto 库优先于旧 dev-package，同时保留锁定 toolkit 的内置库。
记录展开后的真实路径；不能让旧包同名库覆盖新包。

```bash
bash build.sh --run_example "$OP" eager cust --vendor_name="$VENDOR" --soc="$SOC" -j"$JOBS"
# 源包有 GEIR 通路时：
bash build.sh --run_example "$OP" graph --soc="$SOC" -j"$JOBS"
```

transformer 的 eager cust 会用 ASCEND_CUSTOM_OPP_PATH 首项的父目录重新拼 vendor 名，参见其布局页。
多 soc 逐个核对构建、UT 与产物；机器不能运行的 soc 如实标为未运行，不写“全部通过”。

## 产物与工作流回归门禁

以已确认的 `SOURCE_VENDOR` 的结构和接口，加 spec/OpDef/soc/`design/TilingKey.md` 为冻结基准。
比较的是当前算子的结构集合，不要求其他依赖算子或辅助文件与独立包完全相同，更不要求二进制逐字节一致。

特别核对二进制索引中的 `formatMode` 与实际 `simplifiedKey`。
带 NCHW/NHWC 等布局约束的源包不能因为生成器默认值而被改成忽略布局的匹配方式。
索引、已编译 key 和冻结输入必须一致；仅出包成功或 ND 样例通过不能证明这一点。
确属生成器元数据不一致时，按源契约在本算子的构建/打包环节修正**当前算子**的索引，
重新出包、正常安装、重跑冻结用例；不得修改已安装包、改变输入格式或放宽用例来规避失败。
修正不得覆盖其他算子的条目，也不得把某个样例的格式策略当作所有算子的默认值。

1. 新 vendor 的 `op_api/lib/libcust_opapi.so` 非空，头文件/导出接口与原算子签名一致。
2. `op_impl` 中当前算子的 kernel、opt_info 和相关配置覆盖全部支持 soc 与 tilingKey；实际生成路径按仓/版本记录，不能只看某个目录非空。
3. `op_proto` 和 `op_tiling` 注册产物存在且对应当前算子；常见 tiling 路径在 `op_impl/ai_core/tbe/op_tiling`，不是 vendor 根目录。
4. 仓内 host UT 实际执行到预期 tiling / 推导用例，非零且全部通过；手写 ACLNN 的 UT 也须接入相应构建目标。适用 example 成功。
5. 原工作流 blackbox L0/L1/L2、whitebox 各分支、GEIR 冻结套件用同一 TTK 与 golden 原样重跑。沿用原门禁的两阶段比较、负例判定及各路线验证器；结果和日志输出到 migration/evidence，不覆盖原证据。
6. blackbox / whitebox 的结果用原 `parse_result.py --expect-total <冻结数量>`，负例加 `--negative`；GEIR 沿用其专用验证器和 check-frozen，不能误套 ACLNN 解析规则。
7. `check_op_binpath.py --optype <实际注册类型> --expect-dir <新vendor目录> --require` 证明执行新包。原 torch 冒烟存在则换包重跑；无产物记录不涉及。
8. 保留实际调用进程的退出码，完整执行资源清理和 C++ 析构；数值 PASS 后异常退出仍为失败。若测试框架用 `_exit` / `os._exit` 结束 worker，另用源工程原生驱动或执行相同调用与清理的最小程序验证正常返回，不能以父进程退出 0 代替。程序级装卸载/退出调试无需强行套用算子数值测试框架。

存在自定义库与内置库共存时，核对实际加载路径和导出符号；内部全局状态不能因同名符号绑定到另一份库。
出现退出堆损坏时，先用同一依赖组合的“仅加载并从 main 返回”程序隔离，再用动态链接日志或内存检查定位。
按当前目标的 object target 可见性或已验证的 ABI 导出规则修复，保留所有公开接口及注册入口；不可只改崩溃栈最后一帧或屏蔽析构。

重跑驱动如果硬编码旧 dev-package 路径，把**驱动副本**放 migration/evidence 并只调整路径；不得修改原套件、用例或 golden。
只跑构建、源码检查或一两个 example，不能宣称工作流级回归已通过。

## 源码核验与提交

`verify_logic_unchanged.sh <src> <dst>` 输出同路径 C/C++/Ascend 文件的去注释差异清单，覆盖 .c/.cc/.cpp/.cxx/.h/.hpp/.hxx/.asc。保留字符串及 token 间空白，格式变化可能保守报告为差异。正常清单返回 0（包含有差异），预处理或工具失败返回非零；仍必须审阅每个差异和新增/删除。
构建/注册/example/UT 的必要改动列入报告；kernel、tiling、推导、ACLNN 实现和目标 golden 的注释及必要格式化单独核验，不能改变计算逻辑或行为。
若移动了实现文件，按源/目标路径映射追加逐文件核验，不能让 ONLY-IN-SRC/ONLY-IN-DST 掩盖改写。

所有门禁完成且最后一次 pre-commit 未改写受测文件后，仅提交 `<op_class>/<OP>/`，形成一个 commit，不 push、不开 PR。
`MIGRATION-REPORT.md` 包含源契约与参数来源、决策及证据、目标基线/分支/commit、许可证与源码核验、三项产物清单、每套用例计数、命令退出码及日志路径。
失败、N/A、未运行分别说明具体依据；完整迁移未验证时不得记录成功。
