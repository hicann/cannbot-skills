---
name: ascendc-kirin-env-check
description: 麒麟平台（Kirin NPU）环境检查：开发依赖、CANN 包能力（atc→omc 与 camodel sim 两条独立能力线）、设备连接。触发关键词：麒麟环境检查、Kirin 环境、端侧环境检查、鸿蒙开发环境、kirin、harmony。
---

# 麒麟平台环境检查（Kirin NPU）

面向 **麒麟平台（Kirin NPU）** 的端侧算子开发环境检查。麒麟芯片是鸿蒙设备 AI 处理器，开发验证以 **麒麟设备部署推理（hdc/hap）+ CANN Simulator（camodel 仿真）** 两条链路进行。

**前置假设**：默认用户已安装的 CANN 包具备 Kirin 开发的完整功能（atc 转 omc 与 camodel sim 直调均可用），本 skill 通过检查验证该假设，发现缺项时如实报告。

## 工作流程

```
麒麟环境检查
    │
    ├─ [1/3] 开发环境检查（官方前提条件）
    │   └─ scripts/check_kirin_env.sh（段 1）
    │
    ├─ [2/3] CANN 包能力检查（atc→omc / camodel sim 直调 两条能力线独立判定）
    │   └─ scripts/check_kirin_env.sh（段 2）
    │
    └─ [3/3] 设备检查（hdc 连接 + 芯片型号识别）
        └─ scripts/check_kirin_env.sh（段 3）+ 复用 tool-hdc skill
```

```bash
# 一键检查（含设备段）
bash scripts/check_kirin_env.sh
# 仅宿主机检查（--skip-device）
bash scripts/check_kirin_env.sh --skip-device
```

## [1/3] 开发环境检查

依据官方前提条件（[kirin_platform_guide.md](references/kirin_platform_guide.md) 前提条件节，源出 cann-recipes-harmony-infer quick_install/install_deps；`command -v` 存在性 + `version_ge` 版本比较；缺失项只输出补齐指引）：

| 检查项 | 要求 | 缺失时影响 |
|---|---|---|
| python3 | >= 3.7.0 | 用例数据生成/onnx 构造不可用 |
| gcc | >= 7.3.0 | 算子工程 host 侧编译不可用 |
| cmake | >= 3.16.0 | 工程/npu_op_package 构建不可用 |
| pigz（可选） | >= 2.4 | 仅影响打包速度 |
| dos2unix、gawk | 存在 | 构建脚本兼容性 |
| glibc | >= 2.34（推荐 Ubuntu 22.04+） | CANN 运行时依赖基线 |
| python 包 numpy/onnx | 可导入 | golden 数据生成、onnx 模型构造、精度比对 |
| python 包 paramiko（可选） | 可导入 | 跨机 hdc 场景的自动化脚本依赖（是否需要视用户环境而定） |

缺失时指向：[kirin_platform_guide.md](references/kirin_platform_guide.md) 前提条件与安装节。

## [2/3] CANN 环境与能力检查

### CANN 环境就绪检查（先于能力线）

| 检查项 | 判定 | 缺失后果 |
|---|---|---|
| `source set_env.sh`（ASCEND_HOME_PATH） | 环境变量生效 | 全部组件定位失败 |
| 安装信息文件（包内 `${HOST_ARCH}-linux/ascend_toolkit_install.info` 或包根 `ascend_all_cann_install.info`，形态因包而异） | 包已正常安装 | 环境不完整或未安装 |
| 核心目录 `bin/`、`lib64/`、`ccec_compiler/`（内含 ccec→bisheng 编译器） | 组件在位 | 编译/转换入口缺失 |

**默认判定规则**：一个 CANN 包通过上述检查即视为环境就绪，继续逐条能力线检查并报告结果；某个检查未通过时才展开具体问题（缺什么→哪个功能不可用→如何补齐）。

### 两条能力线独立判定

不同发布形态的 CANN 包可能只满足其中一条链路（完整包与裁剪包的能力各有取舍），因此两条能力线**分别报告**，勿以单包结论外推：

#### 能力线 1：atc→omc 转换

| 检查项 | 判定 | 缺失后果 |
|---|---|---|
| `bin/atc` 可执行 | atc 在 PATH/包内 | 转换入口缺失 |
| `opp/built-in/` 下 `op_graph`（旧版 `op_proto`）、`op_impl`、`framework` 齐备 | **完整 opp 标志**（与包名无关；判据见 tool-atc 报错排查 E20002） | opp 子目录不全 → atc FE 初始化失败 **E20002** |
| 自定义算子 vendor 包（`opp/vendors/<vendor_name>`） | 可选，仅自定义算子转换需要 | 转换自定义算子报找不到算子实现 |

> atc 的具体用法（转换路线、参数、精度模式）见 **tool-atc skill**，本 skill 只负责环境能力检查。

#### 能力线 2：camodel sim 直调（ST/UT 仿真执行载体）

| 检查项 | 判定 | 缺失后果 |
|---|---|---|
| `${HOST_ARCH}-linux/simulator/<SIM_PLATFORM>/lib/libruntime_camodel.so` | simulator 在包 | 无仿真载体 |
| 同目录存在 **camodel 版 `libascendcl.so` 壳**（其 CANN 侧依赖仅为 libruntime_camodel.so） | camodel ACL 入口 | **缺壳 → aclInit 500000，全部用例 Segfault/挂死/Abort**；`ldd <可执行> \| grep ascendcl` 看解析落点可确认（缺壳时 libascendcl 会解析到包 lib64 的 stub 版而非 sim 目录壳） |
| GLIBCXX 运行时依赖（按 `readelf -V` 判定，系统 libstdc++ 或 LD_PRELOAD 提供） | camodel 运行时依赖 | 加载报 `version 'GLIBCXX_x.y.z' not found`；用满足该符号的 libstdc++ 经 `LD_PRELOAD` 解决 |

> GLIBCXX 的判别方法与解决方式见 [kirin_platform_guide.md](references/kirin_platform_guide.md) 的「GLIBCXX 运行时」节。

```bash
HOST_ARCH=$(uname -m)                          # x86_64 / aarch64
SIM_PLATFORM=<simulator 目录名>                # 由 ls $ASCEND_HOME_PATH/${HOST_ARCH}-linux/simulator/ 获取

# camodel 直调可执行程序链接要求（sim 目录 lib 必须排在包 lib64 之前，否则报 100039）
# 以下为顺序示意（前两段为关键顺序）；完整配方（含 lib64/device/lib64 与 devlib 段）以执行 harness 为准
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/${HOST_ARCH}-linux/simulator/${SIM_PLATFORM}/lib:$ASCEND_HOME_PATH/${HOST_ARCH}-linux/lib64:$LD_LIBRARY_PATH
```

### 能力矩阵输出

```
包: <CANN安装路径>
  [atc→omc]        ✓/✗（缺什么 → omc 转换不可用 → 如何补齐）
  [camodel sim直调] ✓/✗（缺什么 → ST/UT 仿真不可用 → 如何补齐）
```

缺失时必须向用户明确三件事：**缺了什么组件 → 导致哪个功能不可用 → 如何补齐**（指向 [kirin_platform_guide.md](references/kirin_platform_guide.md) 对应章节）。

## [3/3] 设备检查（复用 tool-hdc skill）

设备连接、命令用法与错误码统一见 **[tool-hdc skill](../../tools/tool-hdc/SKILL.md)**（`hdc list targets`、`hidumper`、`param get`、定位与自检脚本 `find_hdc.sh`），本 skill 不重复维护命令细节。

**识别链**：先用 tool-hdc 确认设备在位，再经 `hidumper -c base` 读取 `MarketName` / `ProductModel` / `HardwareModel`；字段缺失时用 `param get` 补充。解析过滤在宿主机侧完成（`hdc shell` 会把管道整体传入设备侧 shell）。

**芯片型号判定**：优先以 hdc 查询结果为准；若查询结果中无芯片型号信息，**由用户指定**设备芯片型号。

## 常见问题

依据设备与包形态对照：

- **aclnn 两段式 ST 在 sim 上报 361001**：camodel 版 libascendcl 为裁剪符号面，与真实 runtime 隔离 → 精度验证用 simulator 直调承接
- **hdc 找不到设备**：检查设备连接与 hdc 环境（`hdc list targets`，定位见 tool-hdc skill）
- **hap 下载 403**：gitcode raw 对二进制 403，须 `git clone --depth 1` 获取

安装方法与前提条件见 [kirin_platform_guide.md](references/kirin_platform_guide.md)

## 参考文档

| 文件 | 内容 |
|---|---|
| [kirin_platform_guide.md](references/kirin_platform_guide.md) | Kirin 概述 + 环境要点：前提条件、CANN 包安装 |
| [scripts/check_kirin_env.sh](scripts/check_kirin_env.sh) | 三段式检查脚本（开发环境 / CANN 环境与能力 / 设备），只检查不安装 |
