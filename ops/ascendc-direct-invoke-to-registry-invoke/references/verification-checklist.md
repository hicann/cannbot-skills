# 自检清单(Verification Checklist)

> 本文件是生成完所有文件后的逐项**自检清单**。每条只给"检查要点",详细规则展开见 [rules.md](rules.md) 对应 `[R-xxx]`;失败现象 / 修复方法见 [cases.md](cases.md) / [build-verify.md § B.5](build-verify.md#b5-错误关键字--修复动作--规则引用)。
>
> 交付前必须逐项打勾,任一红灯 → 回到对应规则或案例修复。

## 0. 前置:目标框架

- [ ] 确认唯一目标框架为 **CANN 标准 msOpGen**(TilingData 走 `BEGIN_TILING_DATA_DEF` / kernel 走 `GET_TILING_DATA` / 注册走 `IMPL_OP_OPTILING` + `IMPL_OP_INFERSHAPE` + `OP_ADD`) — [R-001](rules.md#r-001唯一目标框架--cann-标准-msopgen)
- [ ] 全工程**无**非 msOpGen 宏:`GET_TILING_DATA_WITH_STRUCT` / `REGISTER_TILING_DEFAULT` / `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL` / `GET_TPL_TILING_KEY` — [R-001](rules.md#r-001唯一目标框架--cann-标准-msopgen) / [R-005](rules.md#r-005tilingdata-唯一定义于-hostkernel-侧不得重复声明)
  - grep:`rg "GET_TILING_DATA_WITH_STRUCT|REGISTER_TILING_DEFAULT|ASCENDC_TPL_ARGS_DECL|ASCENDC_TPL_SEL|GET_TPL_TILING_KEY" .` 必须 == 0
- [ ] 顶层 `CMakeLists.txt` **无** `add_all_modules_sources`(由 msopgen 生成,不手写) — [R-001](rules.md#r-001唯一目标框架--cann-标准-msopgen)
- [ ] 所有 `msopgen gen` 命令都显式带 `-lan cpp`,并在命令后有"生成结构校验"(校验 `op_host/` + `op_kernel/` 同时存在,且无 `tbe/impl/op_info_cfg/`) — [R-053](rules.md#r-053msopgen-gen-必须显式带--lan-cpp-并校验生成结构)
- [ ] **七阶段产物齐全**(缺一阶段 = 交付失败) — [R-057](rules.md#r-057七阶段严格流程不得跳步或合并):
  - [ ] **阶段 1**:OpDef 契约表 + kernel 类名 / 函数名 / 计算 API 清单完整
  - [ ] **阶段 2**:`<UserOutputDir>/op_host/` 下有且仅有 `_tiling.h` / `_tiling.cpp` / `_def.cpp` / `_infershape.cpp`(无合并 `<op>.cpp`)+ `op_kernel/<op>.{h,cpp}` + `op_host/config/<soc>/<op>_binary.json` + `<op>_simplified_key.ini` — [R-056](rules.md#r-056host-必须严格拆成-4-个文件禁止合并单-cpp)
  - [ ] **阶段 3**:`<UserOutputDir>/<op>.json` 存在,顶层非空数组,与 OpDef 契约表逐列一致 — [R-043](rules.md#r-043opjson-顶层数组--op-大驼峰) / [R-046](rules.md#r-046json-与-opdef-逐列严格一致)
  - [ ] **阶段 4**:`<VerifyProjectDir>/{CMakeLists.txt,CMakePresets.json,build.sh,op_host/,op_kernel/}` 齐全;msopgen 生成的合并 `<VerifyProjectDir>/op_host/<op>.cpp` 已被删除
  - [ ] **阶段 5**:`<VerifyProjectDir>/build_out/custom_opp_*.run` 产出(或 msopgen 不可用 → 明确告知用户已出具人工验证手册 + 本次未做编译验证 / 未安装 / 未做二进制一致性验证)
  - [ ] **阶段 6**:已直接执行 `<VerifyProjectDir>/build_out/custom_opp_*.run`,保留 `install_custom_opp.log` — [R-059](rules.md#r-059阶段-6-必须直接执行-custom_opp-run-安装算子包)
  - [ ] **阶段 7**:`<VerifyProjectDir>/example/{main.cpp,CMakeLists.txt,input/,output/}` 齐全;已编译运行 `opapi_test`,输出写入 `example/aclnn_output/`,并与 `example/output/` 逐 `.bin` 二进制一致 — [R-060](rules.md#r-060阶段-7-必须生成-example-并用原直调输入做-aclnn-二进制一致性验证)
- [ ] 交付完成前无任一"跳步"模式:
  - grep:`[ -f "${USER_DIR}/${OP_NAME,,}.json" ]` 必须为真(否则没过阶段 3)
  - grep:`[ -f "${USER_DIR}/op_host/${OP_NAME,,}.cpp" ]` 必须为假(否则阶段 2 未完成)
- [ ] **双目录独立共存**([R-058](rules.md#r-058useroutputdir-与-verifyprojectdir-必须独立共存禁止清理验证目录)):
  - [ ] `realpath <UserOutputDir>` ≠ `realpath <VerifyProjectDir>`
  - [ ] 两者不互为父子(一条不能以另一条为前缀)
  - [ ] `<VerifyProjectDir>` 不在 `/tmp` / `/var/tmp`
  - [ ] 最终交付件清单**同时包含** `<UserOutputDir>` 和 `<VerifyProjectDir>` 两个绝对路径
  - [ ] 整个过程**未执行过** `rm -rf "${VERIFY_DIR}"`(需要换工程一律用 `mv` 重命名备份)
  - [ ] agent 的最终汇报**未建议**用户清理验证目录

---

## 1. 命名与目录

- [ ] 算子名大驼峰在全部注册点(OpDef / REG_OP / REGISTER_TILING_DATA_CLASS / IMPL_OP_OPTILING / IMPL_OP_INFERSHAPE / OP_ADD / JSON)完全一致 — [R-003](rules.md#r-003算子名一处大驼峰全局对齐)
- [ ] kernel 函数名 snake_case,等于 `ExtendCfgInfo("opFile.value", "<op>")` 的值 — [R-004](rules.md#r-004kernel-函数名-snake_case与-opfilevalue-一致)
- [ ] 目录结构符合 `op_host/ + op_kernel/ + op_graph/`,无 `op_kernel/*_tiling_data.h` / `op_kernel/*_tiling_key.h` — [R-005](rules.md#r-005tilingdata-唯一定义于-hostkernel-侧不得重复声明)
- [ ] 原 `.asc` 文件保留,未被新生成文件 `#include` — [R-006](rules.md#r-006保留-asc-原文件但不-include)

---

## 2. TilingData / TilingKey(`op_host/<op>_tiling.h` + `op_host/<op>_tiling.cpp`)

- [ ] `BEGIN_TILING_DATA_DEF(<Op>TilingData)` + `END_TILING_DATA_DEF;`(末尾分号) — [R-007](rules.md#r-007begin_tiling_data_def-范式)
- [ ] 字段 POD;非 SDK 用户自定义 struct 已扁平化 — [R-008](rules.md#r-008字段类型约束)
- [ ] SDK 官方 tiling 结构用 `TILING_DATA_FIELD_DEF_STRUCT`,**未**扁平化 — [R-009](rules.md#r-009sdk-官方-tiling-结构用-tiling_data_field_def_struct-直接嵌入)
- [ ] **嵌套 SDK tiling 字段填充**:直接把 `tiling.fieldName` 作 OUT 引用传给 `AscendC::GetXxxTilingInfo`,**无** `XxxTiling local = {}` + 拷贝 / setter — [R-009](rules.md#r-009sdk-官方-tiling-结构用-tiling_data_field_def_struct-直接嵌入) / [Case 2](cases.md#case-2tiling_data_field_def_struct-嵌套-sdk-tilinglocal--拷贝-触发-double-free)
  - grep:`rg "(RmsNormTiling|LayerNormTiling|SoftmaxTiling|DeepNormTiling)\s+\w+\s*=\s*\{\}" op_host/` 必须 == 0
  - grep:`rg "tiling\.\w+Tiling\s*=|tiling\.set_\w+Tiling\s*\(" op_host/` 必须 == 0
- [ ] `REGISTER_TILING_DATA_CLASS(<Op>, <Op>TilingData)` 已注册 — [R-010](rules.md#r-010register_tiling_data_class-注册)
- [ ] host 写入走 `set_xxx()` setter(标量/数组);**不用**成员直赋 — [R-011](rules.md#r-011host-侧字段写入走-setter)
- [ ] `TilingFunc` 五板斧齐全:`SaveToBuffer` + `SetDataSize` + `SetBlockDim` + `SetTilingKey` + `GetWorkspaceSizes` — [R-012](rules.md#r-012tilingfunc-五板斧)
- [ ] `IMPL_OP_OPTILING(<Op>).Tiling(TilingFunc);` — [R-013](rules.md#r-013tilingfunc-注册)
- [ ] schMode / TilingKey 映射在 host / kernel / JSON 一致 — [R-014](rules.md#r-014schmode--tilingkey-映射表保持在一处)

---

## 3. Host include 与 API

- [ ] host 侧 AscendC tiling API 只通过 `#include "tiling/tiling_api.h"`;**无** `adv_api/...` / `lib/...` / `impl/...` — [R-015](rules.md#r-015host-侧-ascendc-tiling-api-只能来自-tilingtiling_apih) / [Case 1](cases.md#case-1host-侧盲目照搬-asc-的-adv_api-等-sdk-私有-include)
  - grep:`rg '"(adv_api|lib|impl)/' op_host/` 必须 == 0
- [ ] host 必需头三件套齐全:`register/op_impl_registry.h` + `tiling/platform/platform_ascendc.h` + `<op>_tiling.h` — [R-016](rules.md#r-016host-必需头三件套)
- [ ] **无**非公开私有头(例:`util/math_util.h` / `op_host/tiling_util.h` / `tiling/tiling_templates_registry.h` / `ascendc/host_api/tiling/template_argument.h`) — [R-015](rules.md#r-015host-侧-ascendc-tiling-api-只能来自-tilingtiling_apih)
  - grep:`rg '"(util/math_util|op_host/tiling_util|tiling/tiling_templates_registry|ascendc/host_api/tiling/template_argument)' .` 必须 == 0
- [ ] `TilingContext::GetInputShape(i)` 后有 `->GetStorageShape()` — [R-039](rules.md#r-039gerttilingcontext-专用-api与-r-038-区别)

---

## 4. Kernel(`op_kernel/<op>.cpp` + `op_kernel/<op>.h`)

### 4.0 零号原则:**原样迁移,只搬不改**(重中之重) — [R-054](rules.md#r-054kernel-计算逻辑原样迁移禁止改写最高优先级凌驾于任何风格偏好) / [R-055](rules.md#r-055kernel-语义完整性静态校验编译前执行)

- [ ] 对照 `.asc` 原 kernel,`op_kernel/` 下**类名**(含模板参数)逐一存在
  - grep:`for C in $(rg -oN "class\s+(\w+)" <asc> -r '$1' | sort -u); do rg -q "\b${C}\b" op_kernel/ || echo MISSING:${C}; done` 无 MISSING
- [ ] 原 `.asc` 每个 `__aicore__ inline` 成员函数,在 `op_kernel/` 中都能 grep 到同名
- [ ] `ReduceSum` / `Mul` / `Div` / `Add` / `Sub` / `Sqrt` / `Rsqrt` / `Cast` / `Duplicate` / `Brcb` / `DataCopyPad` / `DataCopy` 在新 kernel 的出现次数 **≥ 原 `.asc`**(允许因 schMode 分支复制而增加,不允许减少)
- [ ] **无**凭空新增的辅助函数 / lambda / helper(不在原 `.asc` 里的)
- [ ] **无** `main()` / `ReadConfig()` / `KernelCall()` 残留在 `op_kernel/`
  - grep:`rg "\b(main|ReadConfig|KernelCall)\s*\(" op_kernel/` 必须 == 0
- [ ] 算术表达式**逐字对齐**原 `.asc`:未把 `1.0f / sqrt(x)` 换成 `rsqrt(x)`;未合并 `Mul + Div`;未改动 eps 相加位置、rstd 公式、累加顺序
- [ ] 模板实例化路径对齐:原 `.asc` 里 `template <uint32_t schMode>` / `if constexpr` 的编译期分发保留为模板特化,**未**退化为运行时 `if`
- [ ] Init 中 tiling 参数的唯一合法改动:从 `__gm__ XxxTiling*` + `memcpy` 改为接收 `GET_TILING_DATA` 产出的 `XxxTilingData&`,其它初始化顺序、局部变量名全部保留

### 4.1 形式适配(允许的 7 类)

- [ ] kernel 入口签名 `extern "C" __global__ __aicore__ void <op>(...)` — [R-018](rules.md#r-018kernel-入口签名固定)
- [ ] 函数体第一行 `GET_TILING_DATA(tilingData, tiling);` — [R-019](rules.md#r-019get_tiling_data-第一行)
- [ ] `TILING_KEY_IS(...)` 用**数字字面量**;**无** `constexpr` 命名常量传入 — [R-020](rules.md#r-020tiling_key_is-用数字字面量) / [Case 3](cases.md#case-3tiling_key_is-用数字字面量不用-constexpr)
  - grep:`rg "TILING_KEY_IS\s*\(\s*[A-Za-z_]" op_kernel/` 建议改为字面量
- [ ] device 端**不含** STL(`std::min` / `std::max` / `std::swap` / `<algorithm>` / `<vector>` / `<iostream>` / `<cstring>`) — [R-021](rules.md#r-021device-端禁用-stl)
  - grep:`rg "\bstd::" op_kernel/` 必须 == 0
  - grep:`rg '#include\s*<(algorithm|vector|iostream|string|cstring)>' op_kernel/` 必须 == 0
- [ ] 内建函数统一 `AscendC::` 前缀(`GetBlockIdx` / `GetBlockNum` / `PipeBarrier` / ...) — [R-022](rules.md#r-022device-端内建函数加-ascendc-前缀)
- [ ] C++ 类型用**实现名**:BF16 用 `bfloat16_t`(**不是** `bfloat16`)、FP16 用 `half`、整型用 `<cstdint>` — [R-023](rules.md#r-023device-端-c-数据类型用实现名) / [Case 5](cases.md#case-5kernel-侧-bf16-类型名统一用-bfloat16_t)
  - grep:`rg "\bbfloat16\b" op_kernel/ | rg -v "bfloat16_t"` 必须 == 0
- [ ] Init 签名模板化接收 TilingData:`template <class TD> Init(..., const TD& td, ...)` — [R-024](rules.md#r-024init-模板化接收-tilingdata)
- [ ] LocalTensor 字节数 32 对齐 — [R-025](rules.md#r-025localtensor-32-字节对齐)
- [ ] SDK 高阶 API 调用保真(`AscendC::RmsNorm<...>` / `AscendC::LayerNorm<...>` / `AscendC::Softmax<...>` 等原样保留,未被手写展开替换) — [R-026](rules.md#r-026不强行删改-kernel-内-sdk-高阶-api-调用)

---

## 5. OpDef(`op_host/<op>_def.cpp`)**重点:禁止空壳**

> ⚠️ 本节**高危**。见 [Case 6](cases.md#case-6op_hostop_defcpp-空壳最隐蔽最常见的偷懒型失败)。档三(语法合法但长度不对齐)编译不报,运行时才炸。

**开工前**:

- [ ] 已完整落纸 [R-027](rules.md#r-027开工前必填-opdef-契约表) 契约表(Input/Output name + param_type + 完整 DataType[]/Format[] 数组、Attr 具体 default_value、SOC 系列名、schMode 数 N)

**完工后九点自检**([R-028](rules.md#r-028opdef-完工九点自检)):

- [ ] 1. `class <Op> : public OpDef` 或 `: public ops::OpDef`(1 次) — [R-030](rules.md#r-030opdef-在-ops-命名空间作用域内可解析)
- [ ] 2. 构造函数 `<Op>::<Op>(...) : OpDef(name)`,大驼峰
- [ ] 3. `.Input(` 计数 ≥ 1(`rg -c "\.Input\(" ...`)
- [ ] 4. `.Output(` 计数 ≥ 1
- [ ] 5. 每个 Input/Output 后跟 `.DataType({ge::DT_..., ...})`,**大括号内至少一个具体 `ge::DT_xxx`**;`rg "\.DataType\(\s*\{\s*\}\s*\)" ...` == 0
- [ ] 6. 每个 Input/Output 后跟 `.Format({ge::FORMAT_..., ...})`,同上非空;`Format` 用 `ge::FORMAT_ND`(**不是** `ge::Format::ND`) — [R-031](rules.md#r-031format-用-geformat_nd)
- [ ] 7. 同一 Input/Output 内 `DataType({})` 长度 == `Format({})` 长度 == schMode 数 N — [R-032](rules.md#r-032datatype--format-数组长度--schmode-数)
- [ ] 8. AICore 配置字面量:`rg 'AICore\(\)\.AddConfig\("ascend[0-9a-z_]+"' ...` ≥ 1;前置 `OpAICoreConfig` 六 flag + `ExtendCfgInfo("opFile.value","<op>")` 齐全 — [R-034](rules.md#r-034soc-用系列名三处一致), [R-035](rules.md#r-035opaicoreconfig-六-flag--extendcfginfo)
- [ ] 9. 末尾 `OP_ADD(<Op>);`(`rg "OP_ADD\(" ...` == 1) — [R-036](rules.md#r-036op_addop-末尾一行)

**占位符清零**:

- [ ] `rg "<DType[0-9]+>|<Format[0-9]+>|<soc>|<Op>|<op>|TODO:" op_host/<op>_def.cpp` 必须 == 0 — [R-029](rules.md#r-029禁止任何占位符残留)
- [ ] Attr default_value 是具体字面量(不得 `TODO`) — [R-033](rules.md#r-033attr-必须有具体-default_value)

---

## 6. InferShape(`op_host/<op>_infershape.cpp`)

- [ ] `IMPL_OP_INFERSHAPE(<Op>).InferShape(...).InferDataType(...);` 已注册 — [R-037](rules.md#r-037infershape--inferdatatype-注册)
- [ ] **无** `GetStorageShape()` / `SetOutputShape()`(这些是 TilingContext API,用错会报 `'const struct gert::Shape' has no member named 'GetStorageShape'`) — [R-038](rules.md#r-038gertinfershapecontext-专用-api) / [Case 4](cases.md#case-4infershape-与-tiling-混用-shape-apigetstorageshape--setoutputshape-不存在)
  - grep:`rg "GetStorageShape|SetOutputShape" op_host/<op>_infershape.cpp` 必须 == 0
- [ ] 输出 Shape 走 `GetOutputShape(i)->SetDimNum(n)` + `SetDim(i, v)` / `AppendDim(v)`

---

## 7. 图模式(`op_graph/`,可选)

- [ ] 若生成,`IMPL_OP(<Op>).InferDataType(...);` 函数名与 `IMPL_OP_INFERSHAPE` 里的不冲突(加 `Graph` 后缀或放匿名命名空间) — [R-040](rules.md#r-040图模式-inferdatatype-走-impl_op)

---

## 8. Config / JSON

- [ ] `op_host/config/<soc>/<op>_binary.json` bin 条目数 ≥ schMode 数 N — [R-041](rules.md#r-041op_binaryjson-条目数--schmode-数)
- [ ] `op_host/config/<soc>/<op>_simplified_key.ini` 与 bin 对齐 — [R-042](rules.md#r-042op_simplified_keyini-与-bin-对齐)
- [ ] `<op>.json` 顶层数组,`"op"` 大驼峰 — [R-043](rules.md#r-043opjson-顶层数组--op-大驼峰)
- [ ] `<op>.json` 中 `type` / `format` 数组长度统一 = N,每列与 OpDef 同下标元素严格一致 — [R-044](rules.md#r-044json-中-typeformat-数组长度统一), [R-046](rules.md#r-046json-与-opdef-逐列严格一致)
- [ ] dtype 用语义字符串(`float` / `float16` / `bfloat16` / `int8` / ...,**不带** `DT_`) — [R-045](rules.md#r-045json-用-msopgen-语义名)

---

## 9. SOC 系列名四处一致

- [ ] `op_host/<op>_def.cpp` 的 `AICore().AddConfig("<soc>", cfg)` 是系列名(如 `"ascend910b"`,不是 `"ascend910b4"` / `"Ascend910B3"`)— [R-034](rules.md#r-034soc-用系列名三处一致)
- [ ] `msopgen gen -c ai_core-<soc>` 的 `<soc>` 相同
- [ ] `CMakePresets.json` 的 `ASCEND_COMPUTE_UNIT.value` 相同
- [ ] `op_host/config/<soc>/` 目录名相同
- grep:`rg 'AICore\(\)\.AddConfig\("(ascend[0-9]+[a-z]+[0-9]+|Ascend[0-9])' op_host/` 必须 == 0(不能有具体型号或大驼峰)

---

## 10. 构建验证闭环

详见 [build-verify.md](build-verify.md)。

- [ ] 环境探测已执行(`command -v msopgen` 等六条路径) — [R-047](rules.md#r-047优先探测-msopgen不可用则降级人工手册)
- [ ] 若 msopgen 可用:`<VerifyProjectDir>` 与 `<UserOutputDir>` 同级,未放进 UserOutputDir 内部,未用 `/tmp`,带时间戳 — [R-048](rules.md#r-048验证工程目录与用户输出同级)
- [ ] 进编译循环前已跑 OpDef 静态预检([build-verify.md § B.3](build-verify.md#b3-编译前-opdef-静态预检-r-049)),全部通过 — [R-049](rules.md#r-049编译循环前先跑-opdef-静态预检)
- [ ] 编译循环最多 5 轮,按错误关键字 → 修复动作 → 规则引用表修复,优先改 `<UserOutputDir>` 再同步 — [R-050](rules.md#r-050编译循环最多-5-轮--错误映射表)
- [ ] 编译成功后直接执行 `.run` 安装包,命令未追加 `--quiet` / `--install`,安装日志保留 — [R-059](rules.md#r-059阶段-6-必须直接执行-custom_opp-run-安装算子包)
- [ ] 阶段 7 harness 位于 `<VerifyProjectDir>/example/`,复用原直调 `input/` / `output/`,未随机生成输入,未覆盖原直调 `output/` — [R-060](rules.md#r-060阶段-7-必须生成-example-并用原直调输入做-aclnn-二进制一致性验证)
- [ ] `example/main.cpp` 中 aclnn 接口名按算子名拼接(`<Op>` → `aclnn<Op>GetWorkspaceSize` / `aclnn<Op>`),输入输出 tensor 与 attr 顺序来自 OpDef 契约表和原直调代码 — [R-060](rules.md#r-060阶段-7-必须生成-example-并用原直调输入做-aclnn-二进制一致性验证)
- [ ] `example/CMakeLists.txt` 生成 `opapi_test`,并能按 `cd example && mkdir -p build && cd build && cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE && make && cd bin && ./opapi_test` 跑通 — [R-060](rules.md#r-060阶段-7-必须生成-example-并用原直调输入做-aclnn-二进制一致性验证)
- [ ] `example/binary_compare.log` 显示所有 aclnn 输出 `.bin` 与 `example/output/` 原直调输出 `.bin` byte-level identical;未用 rtol/atol 容差放行 — [R-060](rules.md#r-060阶段-7-必须生成-example-并用原直调输入做-aclnn-二进制一致性验证)
- [ ] 若 msopgen 不可用:已输出完整人工验证手册(见 [build-verify.md § D](build-verify.md#d-降级环境不可用时的人工验证手册)),**禁止**伪装执行

---

## 11. 交付约束

- [ ] **未生成**泛化端到端精度测试框架;仅保留阶段 7 所需的二进制一致性验证 harness — [R-051](rules.md#r-051skill-只生成阶段-7-二进制一致性验证-harness)
- [ ] **未保留/重生成**原 `.asc` 的 `main` / `ReadConfig` / `KernelCall` 入口 — [R-052](rules.md#r-052本-skill-不生成不修改-main--readconfig--kernelcall)

---

## 常见红灯与对照表

| 现象 | 查规则 / 案例 | 修复入口 |
| ---- | ---- | ---- |
| `adv_api/... No such file or directory` | [R-015](rules.md#r-015host-侧-ascendc-tiling-api-只能来自-tilingtiling_apih) / [Case 1](cases.md#case-1host-侧盲目照搬-asc-的-adv_api-等-sdk-私有-include) | `op_host/<op>_tiling.cpp\|.h` 改 `#include "tiling/tiling_api.h"` |
| `free(): double free detected in tcache 2` / `malloc_consolidate` | [R-009](rules.md#r-009sdk-官方-tiling-结构用-tiling_data_field_def_struct-直接嵌入) / [Case 2](cases.md#case-2tiling_data_field_def_struct-嵌套-sdk-tilinglocal--拷贝-触发-double-free) | `op_host/<op>_tiling.cpp` 删 local,改直接 OUT 引用 |
| `Var: SCH_MODE_X in TILING_KEY_IS(...) can not be processed as numeric variables` | [R-020](rules.md#r-020tiling_key_is-用数字字面量) / [Case 3](cases.md#case-3tiling_key_is-用数字字面量不用-constexpr) | `op_kernel/<op>.cpp` 改数字字面量 |
| `'const struct gert::Shape' has no member named 'GetStorageShape'` / `'gert::InferShapeContext' has no member named 'SetOutputShape'` | [R-038](rules.md#r-038gertinfershapecontext-专用-api) / [Case 4](cases.md#case-4infershape-与-tiling-混用-shape-apigetstorageshape--setoutputshape-不存在) | `op_host/<op>_infershape.cpp` 按模板改写 |
| `'ND' is not a member of 'ge::Format'` | [R-031](rules.md#r-031format-用-geformat_nd) | `op_host/<op>_def.cpp` 用 `ge::FORMAT_ND` |
| `The soc version of op <Op> is not configured` / `cannot find chip config for ai_core-...` | [R-034](rules.md#r-034soc-用系列名三处一致) | 四处统一改系列名(全小写+无尾部数字) |
| `no member named 'min' in namespace 'std'` | [R-021](rules.md#r-021device-端禁用-stl) | `op_kernel/<op>.*` 改三元 |
| `use of undeclared identifier 'GetBlockIdx'` | [R-022](rules.md#r-022device-端内建函数加-ascendc-前缀) | `op_kernel/<op>.*` 加 `AscendC::` |
| `unknown type name 'bfloat16'; did you mean 'bfloat16_t'?` | [R-023](rules.md#r-023device-端-c-数据类型用实现名) / [Case 5](cases.md#case-5kernel-侧-bf16-类型名统一用-bfloat16_t) | `op_kernel/<op>.*` 改 `bfloat16_t` |
| `'OpDef' has not been declared` | [R-030](rules.md#r-030opdef-在-ops-命名空间作用域内可解析) | `op_host/<op>_def.cpp` 包进 `namespace ops {}` 或用 `public ops::OpDef` |
| `undefined macro GET_TILING_DATA_WITH_STRUCT` | [R-001](rules.md#r-001唯一目标框架--cann-标准-msopgen) / [R-005](rules.md#r-005tilingdata-唯一定义于-hostkernel-侧不得重复声明) | 改回 msOpGen 标准:`GET_TILING_DATA(tilingData, tiling)` + host 侧 `REGISTER_TILING_DATA_CLASS` |
| `TILING_DATA_FIELD_DEF ... is not POD` | [R-008](rules.md#r-008字段类型约束) / [R-009](rules.md#r-009sdk-官方-tiling-结构用-tiling_data_field_def_struct-直接嵌入) | 扁平化或改用 `_STRUCT` 嵌入 |
| `undefined reference to TilingFunc` / `<Op>::<Op>` | [R-003](rules.md#r-003算子名一处大驼峰全局对齐), [R-013](rules.md#r-013tilingfunc-注册), [R-036](rules.md#r-036op_addop-末尾一行) | 三处算子名对齐 |
| `The input/output/attr of op <Op> is not configured` / `Failed to get op definition of <Op>` / `op proto of <Op> is empty` | [R-027](rules.md#r-027开工前必填-opdef-契约表) ~ [R-036](rules.md#r-036op_addop-末尾一行) / [Case 6](cases.md#case-6op_hostop_defcpp-空壳最隐蔽最常见的偷懒型失败) | OpDef 按契约表**整文件重写** |
| msopgen 输出目录无 `op_host/op_kernel/`,出现 `tbe/impl/op_info_cfg/*.py` | [R-053](rules.md#r-053msopgen-gen-必须显式带--lan-cpp-并校验生成结构) | `msopgen gen` 补 `-lan cpp` 重跑 |
| 编译能过,aclnn 结果与原 `.asc` 直调**精度对不上 / NaN / 量级偏差** | [R-054](rules.md#r-054kernel-计算逻辑原样迁移禁止改写最高优先级凌驾于任何风格偏好) / [R-055](rules.md#r-055kernel-语义完整性静态校验编译前执行) | 跑 B.3.0 kernel 语义完整性比对,发现差异段**从 `.asc` 整段重贴**,仅做 7 类形式适配 |
| aclnn 运行时 `<Op>NotRegistered` / `Op<Op>:GetOpInfoFailed`(编译却成功) | [R-032](rules.md#r-032datatype--format-数组长度--schmode-数), [R-044](rules.md#r-044json-中-typeformat-数组长度统一) / [Case 6 档三](cases.md#case-6op_hostop_defcpp-空壳最隐蔽最常见的偷懒型失败) | OpDef 数组长度 / 列顺序校对 |
| `ASCEND_COMPUTE_UNIT` mismatch / `device target is empty` | [R-034](rules.md#r-034soc-用系列名三处一致) | 同 SOC 条 |
| 交付里出现 `op_host/<op>.cpp`(把 OpDef + InferShape + Tiling 揉一起) | [R-056](rules.md#r-056host-必须严格拆成-4-个文件禁止合并单-cpp) | 按 § B.2.1 拆成 `_def.cpp` / `_tiling.cpp` / `_infershape.cpp`;删原合并文件 |
| 没生成 `<op>.json`,跳到只交 op_host/op_kernel | [R-057](rules.md#r-057七阶段严格流程不得跳步或合并) | 回阶段 3:按 OpDef 契约表手写 JSON,再跑阶段 4 |
| 生成了代码但**没跑** `msopgen gen` / `bash build.sh` / `.run` 安装 / 二进制一致性验证 | [R-057](rules.md#r-057七阶段严格流程不得跳步或合并) | 回阶段 4 / 5 / 6 / 7 实际执行;msopgen 不可用才能走 § D 人工手册 |
| `.run` 已生成但未安装或安装失败 | [R-059](rules.md#r-059阶段-6-必须直接执行-custom_opp-run-安装算子包) | 直接执行 `./build_out/custom_opp_*.run`,保留 `install_custom_opp.log`;失败先修环境/权限/旧包冲突 |
| aclnn 输出文件缺失或与原直调输出二进制不一致 | [R-060](rules.md#r-060阶段-7-必须生成-example-并用原直调输入做-aclnn-二进制一致性验证) / [R-054](rules.md#r-054kernel-计算逻辑原样迁移禁止改写最高优先级凌驾于任何风格偏好) | 先核对 `example/` 下 input/output 是否来自原直调,再核对 harness dtype/shape/attr 和 aclnn 接口参数,最后按 B.3.0 回查 kernel 保真;禁止容差放行 |
| `<VerifyProjectDir>` == `<UserOutputDir>` / 互为父子 / 落在 `/tmp` | [R-058](rules.md#r-058useroutputdir-与-verifyprojectdir-必须独立共存禁止清理验证目录) | 停手,重新挑一个与 UserOutputDir 并列的绝对路径(加时间戳),从 B.1 重跑 |
| 验证目录被 `rm -rf` / 被建议清理 | [R-058](rules.md#r-058useroutputdir-与-verifyprojectdir-必须独立共存禁止清理验证目录) | 改为 `mv "${VERIFY_DIR}" "${VERIFY_DIR}.<reason>_$(date +%s)"` 备份;最终汇报同时列出 `<UserOutputDir>` 和 `<VerifyProjectDir>` 两个路径 |
| 最终交付清单只列了 `<UserOutputDir>`,没列 `<VerifyProjectDir>` | [R-058](rules.md#r-058useroutputdir-与-verifyprojectdir-必须独立共存禁止清理验证目录) | 汇报补齐双目录路径及 `build_out/custom_opp_*.run` 位置,并说明验证目录原封保留 |
