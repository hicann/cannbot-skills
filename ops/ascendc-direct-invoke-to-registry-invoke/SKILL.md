---
name: ascendc-direct-invoke-to-registry-invoke
description: 当用户想把`<<<>>>` kernel 直调形式改造成自定义算子工程时使用。触发：用户提到"kernel直调转自定义算子"、"为kernel直调工程接入ACLNN/GEIR接口"、"`<<<>>>` 改自定义算子工程"等。不适用于从零开发新算子
---

# Ascend C Kernel 直调转 msOpGen 自定义算子

本 skill 用于把已有 Ascend C kernel 直调工程迁移为 **CANN 标准 msOpGen** 自定义算子工程。典型输入是单个或少量 `.asc` 文件、host/kernel 混写代码、`kernel<<<blockNum, nullptr, stream>>>` 直调入口、原始 input/output bin 和生成数据脚本。目标产物是可通过 `bash build.sh` 生成 `custom_opp_*.run` 的工程，并完成安装与 ACLNN 输出对原直调输出的 byte-level 一致性验证。

## 使用边界

- **只做迁移，不重新设计算法**：kernel 计算逻辑只搬不改。除 include、入口签名、`GET_TILING_DATA`、`TILING_KEY_IS`、`AscendC::` 前缀、`bfloat16_t`、STL 脱敏这 7 类形式适配外，不改写计算顺序、API 选择或数学表达。见 [R-054](references/rules.md#r-054kernel-计算逻辑原样迁移禁止改写最高优先级凌驾于任何风格偏好)。
- **唯一目标框架是 msOpGen**：交付结构为 `op_host/` + `op_kernel/`，必要时加 `op_graph/`；禁止混用非 msOpGen tiling 宏体系。见 [R-001](references/rules.md#r-001唯一目标框架--cann-标准-msopgen)。
- **不用于从零开发新算子**：如果用户没有可参考的直调 `.asc`、原 host 调用或原输入输出，应先要求补齐材料，而不是凭经验生成一个新算子。
- **不生成泛化精度框架**：只生成阶段 7 所需的 ACLNN 二进制一致性验证 harness。见 [R-051](references/rules.md#r-051skill-只生成阶段-7-二进制一致性验证-harness)。

## 开工前读取

先按任务需要加载参考资料，不要把所有大文件一次性塞进上下文：

- `references/rules.md`：规则权威源。遇到命名、TilingData、OpDef、JSON、SOC、构建或交付争议时读取对应 `R-xxx`。
- `references/reference.md`：各目标文件模板。生成 `<op>.json`、host 四件套、kernel 两件套、config 或 README 时读取。
- `references/build-verify.md`：msopgen 生成、构建、安装、ACLNN 一致性验证和环境不可用降级流程。进入阶段 4 前必须读取。
- `references/verification-checklist.md`：交付前逐项自检。阶段 7 完成或降级交付前必须读取。
- `references/cases.md`：遇到编译报错、double free、找不到算子、TBE DSL 旧结构等历史问题时读取。
- `references/examples.md`：需要端到端范例时读取，尤其是 `rms_norm.asc` 拆解路径。

## 总体原则

- **七阶段不得跳步**：读 `.asc` → 拆 host/kernel → 生成 `<op>.json` → `msopgen gen` → `bash build.sh` → 安装 `.run` → ACLNN 二进制一致性验证。任一阶段产物缺失即交付失败。见 [R-057](references/rules.md#r-057七阶段严格流程不得跳步或合并)。
- **host 必须拆成四件套**：`<op>_def.cpp`、`<op>_tiling.cpp`、`<op>_infershape.cpp`、`<op>_tiling.h`。禁止保留或交付合并版 `<op>.cpp`。见 [R-056](references/rules.md#r-056host-必须严格拆成-4-个文件禁止合并单-cpp)。
- **双目录并列交付**：`<UserOutputDir>` 是纯净成果，`<VerifyProjectDir>` 是编译证据链。两者绝对路径不同、不互为父子，且都是最终交付件。验证目录任何阶段都不得 `rm -rf`；需要重跑时用 `mv` 备份。见 [R-058](references/rules.md#r-058useroutputdir-与-verifyprojectdir-必须独立共存禁止清理验证目录)。
- **msopgen 必须显式 `-lan cpp`**：默认 `py` 会生成旧版 TBE DSL 结构。执行后必须校验生成了 `op_host/` + `op_kernel/`，且没有 `tbe/`、`impl/`、`op_info_cfg/`。见 [R-053](references/rules.md#r-053msopgen-gen-必须显式带--lan-cpp-并校验生成结构)。
- **环境不可用要诚实降级**：Windows 或未安装 CANN/msopgen 时，按 `references/build-verify.md § D` 输出人工验证手册，并明确说明本次未完成编译、安装和二进制一致性验证。

## 七阶段工作流

### 阶段 1：读取原 `.asc`

产物：OpDef 契约表 + kernel 清单。

扫描原工程并先填完整 OpDef 契约表：Input/Output 名称、DataType 数组、Format 数组、Attr 默认值、SOC 系列名、schMode 数 N。同步抽取 tiling 结构字段、kernel 类名、`__aicore__ inline` 成员函数、`schMode` 分发逻辑，以及 `ReduceSum` / `Mul` / `Div` / `Sqrt` / `Rsqrt` / `Cast` 等计算 API 调用清单。契约表不完整时不得进入阶段 2。见 [R-027](references/rules.md#r-027开工前必填-opdef-契约表)。

### 阶段 2：拆分 host / kernel

产物：`<UserOutputDir>/op_host/` 四件套、`op_kernel/` 两件套、`op_host/config/<soc>/` 双文件。

host 侧按四件套生成：

- `<op>_tiling.h`：定义 TilingData 并 `REGISTER_TILING_DATA_CLASS`。SDK 官方 tiling 结构用 `TILING_DATA_FIELD_DEF_STRUCT` 直接嵌入。见 [R-009](references/rules.md#r-009sdk-官方-tiling-结构用-tiling_data_field_def_struct-直接嵌入)。
- `<op>_tiling.cpp`：实现 `TilingFunc` 并 `IMPL_OP_OPTILING`。host API 只从 `tiling/tiling_api.h` 获取；填充 SDK tiling 结构时，直接把 `tiling.fieldName` 作为 OUT 引用传给 `AscendC::GetXxxTilingInfo`。见 [R-015](references/rules.md#r-015host-侧-ascendc-tiling-api-只能来自-tilingtiling_apih)。
- `<op>_def.cpp`：实现 `class <Op> : public OpDef`、`AICore().AddConfig("<soc>", cfg)`、`OP_ADD(<Op>)`。SOC 使用全小写系列名，不带尾部型号数字。见 [R-034](references/rules.md#r-034soc-用系列名三处一致)。
- `<op>_infershape.cpp`：实现 `InferShape` / `InferDataType` 并 `IMPL_OP_INFERSHAPE`。注意 `gert::InferShapeContext::GetInputShape()` 直接返回 `gert::Shape*`，不要套用 tiling context 的 `GetStorageShape` 习惯。

kernel 侧生成两件套：

- `<op>.h`：保留原 kernel class 声明和 inline 成员函数，遵守“只搬不改”。
- `<op>.cpp`：提供 `extern "C" __global__ __aicore__` 入口，使用 `GET_TILING_DATA(...)` 和 `TILING_KEY_IS(...)` 分发；`TILING_KEY_IS` 使用数字字面量，`bfloat16` 写作 `bfloat16_t`，`GetBlockIdx` 加 `AscendC::` 前缀。见 [R-020](references/rules.md#r-020tiling_key_is-用数字字面量) / [R-023](references/rules.md#r-023device-端-c-数据类型用实现名)。

config 文件放在 `op_host/config/<soc>/`：`<op>_binary.json` 条目数至少等于 schMode 数，另有 `<op>_simplified_key.ini`。原 `.asc` 保留但不被新文件 `#include`；原 `CMakeLists.txt` 可另存为 `CMakeLists.txt.kernel-direct-call`；不要迁移或生成 `main` / `ReadConfig` / `KernelCall`。见 [R-006](references/rules.md#r-006保留-asc-原文件但不-include) / [R-052](references/rules.md#r-052本-skill-不生成不修改-main--readconfig--kernelcall)。

### 阶段 3：生成 `<op>.json`

产物：`<UserOutputDir>/<op>.json`。

根据阶段 1 的 OpDef 契约表手写 msopgen 原型 JSON。字段必须与 `<op>_def.cpp` 逐列一致；顶层必须是数组；`op` 使用大驼峰；dtype 使用 `float` / `float16` / `bfloat16` 等语义名，不带 `DT_` 前缀。漏生成 JSON 直接视为交付失败，因为阶段 4 的 `msopgen gen` 只能从该 JSON 入口生成工程。见 [R-043](references/rules.md#r-043opjson-顶层数组--op-大驼峰) / [R-046](references/rules.md#r-046json-与-opdef-逐列严格一致)。

### 阶段 4：生成验证工程

产物：`<VerifyProjectDir>/`，包含 `CMakeLists.txt`、`CMakePresets.json`、`build.sh`、`op_host/`、`op_kernel/`。

选择与 `<UserOutputDir>` 同级并列的 `<VerifyProjectDir>`，推荐 `<parent>/<op>_verify_project_<timestamp>`。实际执行：

```bash
msopgen gen -i <UserOutputDir>/<op>.json -c ai_core-<soc> -lan cpp -out <VerifyProjectDir>
```

执行后按 `references/build-verify.md § B.2` 做结构校验。删除 msopgen 生成的合并 host 文件 `<VerifyProjectDir>/op_host/<op>.cpp`，再把 `<UserOutputDir>/op_host/` 和 `op_kernel/` 同步覆盖到验证工程。随后运行 host 文件拆分校验、kernel 语义完整性预检和 OpDef 静态预检。见 [R-049](references/rules.md#r-049编译循环前先跑-opdef-静态预检) / [R-055](references/rules.md#r-055kernel-语义完整性静态校验编译前执行) / [R-056](references/rules.md#r-056host-必须严格拆成-4-个文件禁止合并单-cpp)。

### 阶段 5：编译构建

产物：`<VerifyProjectDir>/build_out/custom_opp_*.run`。

必须实际执行 `bash build.sh`。失败时不要交付半成品，按 `references/build-verify.md § B.5` 的错误关键字、修复动作和规则映射循环修复，最多 5 轮。若 msopgen/CANN 环境不可用，走人工验证手册降级，并在最终回复中明确未完成哪些验证。

### 阶段 6：安装算子包

产物：安装日志 + 当前环境可发现该自定义算子。

编译成功后直接执行：

```bash
./build_out/custom_opp_*.run
```

不要追加 `--quiet` / `--install`。安装失败不得进入阶段 7；保留安装日志，修复环境或包结构后重试。见 [R-059](references/rules.md#r-059阶段-6-必须直接执行-custom_opp-run-安装算子包)。

### 阶段 7：ACLNN 二进制一致性验证

产物：`<VerifyProjectDir>/example/` 可执行工程、ACLNN 输出 bin、byte-level compare 结果。

在 `<VerifyProjectDir>/example/` 下生成 `main.cpp` 和 `CMakeLists.txt`，把原直调工程的 `input/`、`output/` 复制到 `example/input/`、`example/output/`。输入/输出维度、dtype、epsilon、对齐规则等只能从原脚本或配置抽取，不能猜测。

`main.cpp` 构造与原直调完全一致的 ACLNN 输入参数，接口按 `<Op>` 拼接为 `aclnn<Op>GetWorkspaceSize` 和 `aclnn<Op>`。固定编译运行流程：

```bash
cd example && mkdir -p build && cd build
cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
make
cd bin && ./opapi_test
```

ACLNN 输出写入 `example/aclnn_output/`，逐文件与 `example/output/` 下原直调输出做 `cmp -s` 或 SHA256 比较；任一字节不同即视为验证失败。见 [R-060](references/rules.md#r-060阶段-7-必须生成-example-并用原直调输入做-aclnn-二进制一致性验证)。

## 交付汇报

最终回复必须给出：

- `<UserOutputDir>` 绝对路径。
- `<VerifyProjectDir>` 绝对路径，且说明该目录是编译证据链，需要原样保留。
- `.run` 路径、安装日志位置、`example/` 构建日志位置。
- `example/aclnn_output/` 路径和二进制一致性结果。
- 若环境不可用或某阶段失败，明确列出未完成的阶段、失败原因、已保留的产物路径和下一步人工验证命令。

交付前读取并执行 `references/verification-checklist.md`。违反零号原则、七阶段流程、host 四件套、双目录并列、`-lan cpp` 或二进制一致性验证中的任一项，都不要宣称任务完成。
