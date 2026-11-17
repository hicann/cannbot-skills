# 阶段 3：构建、安装与加载清单

前置：`[GATE-2] FILES_MODIFIED=[...]`。产物为 `${EVIDENCE_DIR}/build_install.md`、构建/安装日志及真实产物清单。

## LOADED 检查点

读取 [build-and-install.sh.template](../references/build-and-install.sh.template) 和 [platform-and-api-diff.md](../references/platform-and-api-diff.md)，记录 `[LOADED] 3510-to-920x/build-template` 与对应 API 资料版本。

本路线使用本包模板；根公共模板的 transformer 包名、vendor 后缀、stub 库名和 x86_64 路径不能直接套到 ops-nn。此处默认 ops-repo 的注册算子包形态；其他工程形态需在 Stage 1 明确构建/安装适配及相同的加载证明，不使用未验证的命令。

## Step 3.1：生成可复核的构建脚本

将模板复制到 `${EVIDENCE_DIR}/build_and_install.sh`。从 `env_info.md` 取得并导出 `CANN_SET_ENV`、`PYTHON_PATH`、`OP_DIR`、`OP_NAME`、`SOC`、`TARGET_ARCH`、`VENDOR_NAME`、`EVIDENCE_DIR` 和 `JOB_NUM`。路径使用绝对路径，变量值及额外构建选项在 `build_install.md` 展示，不把字符串交给 `eval`。

核对该版本的 `build.sh --help`、源列表和打包配置：

- DAV_9201 使用 `SOC=ascend960dt TARGET_ARCH=9201`，DAV_9202 使用 `SOC=ascend960pr TARGET_ARCH=9202`；平台注册与源选择覆盖实际目标，配对不符时停止。
- 使用实际支持的 `--pkg --soc --ops --vendor_name`；需要额外 host tiling 选项时先确认该版本支持，不从旧模板无条件搬入。
- 构建包含目标 kernel 与 tiling；本阶段不使用跳过 kernel 的 JIT 包代替迁移二进制证据。

执行 `bash "${EVIDENCE_DIR}/build_and_install.sh" build`，附加选项按实际帮助确认。日志需保留命令、工具版本、退出状态和产物位置；失败定位根因后重跑，不用仅 host 构建成功冒充整包通过。

## Step 3.2：明确安装包与目标路径后安装

1. 从本轮构建输出选择确切 `.run` 包，核对生成时间、SHA256 以及 `build_target.txt` 中的 SOC/架构；多个包时按构建记录消歧，不取目录中的第一个旧包，不将 9201 包记作 9202 产物。
2. 读该包的安装帮助，在 Stage 0 已授权范围内选本轮尚不存在的 vendor 目录，确认安装参数、vendor 实际目录及对应的 `OPAPI_LIB`、`TILING_LIB` 绝对路径。包内 vendor 名由构建时确定的，应在构建前选择本轮名称；不删除旧目录来绕过检查。
3. 设置 `VENDOR_DIR`、`OPAPI_LIB`、`TILING_LIB`、`ACLNN_SYMBOL`（执行入口完整名，如 `aclnnFoo`），执行 `bash "${EVIDENCE_DIR}/build_and_install.sh" install "${RUN_PKG}"`，其后传已核实的安装参数。
4. 核对本次安装在新目录内生成的库、完整执行符号及配套的 `GetWorkspaceSize` 符号和动态依赖；解析符号链接后仍须位于该目录。安装到别处、缺库或校验失败均不能发布成功清单；修改时间或 hash 变化不能单独证明包与库的对应关系。缺库时查实际依赖与打包规则，不能固定复制 `libopapi_math.so`。

## Step 3.3：记录加载约定

记录自定义环境的激活方式、库路径、kernel/tiling 标识及 hash。文件存在、`nm` 或 `strings` 命中只能证明安装/符号事实；实际调度到自定义实现还需 Stage 4 的真实调用及 Stage 5 的 trace/profiler 关联证据。若存在回退内置实现的迹象，先排查注册、环境和搜索顺序，不能继续宣称迁移已验证。

## Gate

- 构建退出成功，目标 kernel/host/tiling 产物齐全；安装成功，库、符号和依赖校验通过，日志与清单落盘：`[GATE-3] BUILD=PASS INSTALL=PASS`。
- 缺任何证据：分别记录 BUILD/INSTALL 的实际状态并加 `BLOCKED`；不得仅以脚本执行过作为通过依据。
