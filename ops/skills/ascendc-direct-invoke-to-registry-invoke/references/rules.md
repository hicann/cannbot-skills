# 规则权威册(Rules Registry)

> 本文件是整个 skill 唯一的**规则权威源**(single source of truth)。所有流程文档、模板、校验清单、案例集都通过 `[R-xxx]` 引用这里的条目,不得在其它文件里重复展开。修订规则请**只修改此文件**。
>
> 编号规则:扁平整数 `R-001` ~ `R-NNN`,按归类顺序排列,**永不复用已退役的编号**(退役条目保留墓碑并标记 `DEPRECATED`,以免外部引用失效)。

## 读法

每条规则固定四段:

- **要点**:一句话概括,违反即失败
- **展开**:必要的背景、边界和反例
- **适用位置**:文件 / 代码段 / 构建阶段
- **关联**:引用其它 rule 或 [cases.md](cases.md) 中的案例号

---

## 一、框架选择与目录结构

### R-001:唯一目标框架 — CANN 标准 msOpGen

- **要点**:本 skill **唯一支持**的自定义算子框架是 **CANN 标准 msOpGen**(`register/tilingdata_base.h` + `register/op_def_registry.h` + `register/op_impl_registry.h` + `tiling/platform/platform_ascendc.h`),产物用 msOpGen 工程 `build.sh` 编译为 `custom_opp_*.run`。
- **展开**:
  - TilingData 用 `BEGIN_TILING_DATA_DEF` + `REGISTER_TILING_DATA_CLASS` 定义
  - kernel 侧用 `GET_TILING_DATA(tilingData, tiling)` 取数据
  - host 侧用 `IMPL_OP_OPTILING` + `IMPL_OP_INFERSHAPE` + `OP_ADD` 注册
  - 任何其他 tiling 宏体系(如 `GET_TILING_DATA_WITH_STRUCT` / `REGISTER_TILING_DEFAULT` / `ASCENDC_TPL_ARGS_DECL` 等)**一律禁止**使用。见 R-005 / R-015
- **适用位置**:开工即确认,贯穿全流程

### R-002:(DEPRECATED,已被 R-001 吸收)

- 原规则"禁止两框架混用"已退役:本 skill 不再支持任何非 msOpGen 框架,所以不存在混用可能。禁止使用非标准 tiling 宏的约束在 R-001 展开条与 R-005 中已覆盖。外部文档若仍引用 `[R-002]`,等价跳转至 R-001。

### R-003:算子名一处大驼峰,全局对齐

- **要点**:大驼峰(PascalCase)算子名在全部注册点**完全一致**。
- **展开**:同名出现在 `OpDef` 子类名 / `OpDef(<name>)` 构造实参 / `REG_OP(<Op>)` / `REGISTER_TILING_DATA_CLASS(<Op>, <Op>TilingData)` / `IMPL_OP_OPTILING(<Op>)` / `IMPL_OP_INFERSHAPE(<Op>)` / `OP_ADD(<Op>)` / `<op>.json` 里的 `"op"` 字段。
- **适用位置**:全局

### R-004:kernel 函数名 snake_case,与 `opFile.value` 一致

- **要点**:`op_kernel/<op>.cpp` 中 `extern "C" ... void <op>(...)` 的函数名必须是**小写下划线**形式,且**一字不差**等于 `op_host/<op>_def.cpp` 中 `ExtendCfgInfo("opFile.value", "<op>")` 的值。
- **展开**:`RmsNorm` → `rms_norm`、`LayerNorm` → `layer_norm`。CANN build 通过这个字符串绑定 kernel 文件,不一致时链接阶段丢算子。
- **适用位置**:`op_kernel/<op>.cpp` + `op_host/<op>_def.cpp`

### R-005:TilingData 唯一定义于 host,kernel 侧不得重复声明

- **要点**:TilingData 结构**只在** `op_host/<op>_tiling.h` 用 `BEGIN_TILING_DATA_DEF` 宏定义;kernel 侧通过 `GET_TILING_DATA(tilingData, tiling)` 自动生成等价视图。
- **展开**:**禁止**在 `op_kernel/` 下放 `<op>_tiling_data.h` / `<op>_tiling_key.h` 等 TilingData 副本或变体声明;**禁止**使用 `GET_TILING_DATA_WITH_STRUCT` 之类非 msOpGen 标准宏。
- **适用位置**:`op_host/<op>_tiling.h` + `op_kernel/`

### R-006:保留 `.asc` 原文件但不 include

- **要点**:原 `.asc` 作参考保留在原路径,**不得**被新生成的 `op_kernel/<op>.*` 或 `op_host/<op>_*.cpp` `#include`。
- **适用位置**:所有新生成文件

---

## 二、TilingData / TilingKey

### R-007:`BEGIN_TILING_DATA_DEF` 范式

- **要点**:使用 `BEGIN_TILING_DATA_DEF(<Op>TilingData)` / `END_TILING_DATA_DEF;`(末尾**必须**带分号)。
- **适用位置**:`op_host/<op>_tiling.h`

### R-008:字段类型约束

- **要点**:`TILING_DATA_FIELD_DEF` 的 type 参数必须是 POD 标量(`uint32_t` / `uint64_t` / `int32_t` / `float` / `bool`);**非** SDK 官方注册的用户自定义 struct 必须扁平化。
- **展开**:违反时报 `TILING_DATA_FIELD_DEF ... is not POD`。
- **适用位置**:`op_host/<op>_tiling.h`
- **关联**:R-009

### R-009:SDK 官方 tiling 结构用 `TILING_DATA_FIELD_DEF_STRUCT` 直接嵌入

- **要点**:`RmsNormTiling` / `LayerNormTiling` / `SoftmaxTiling` / `DeepNormTiling` / `MatmulTiling` 等 **SDK 官方** tiling 结构体,使用 `TILING_DATA_FIELD_DEF_STRUCT(XxxTiling, fieldName)` 宏嵌入,**不得**扁平化。
- **展开**:host 侧调用 SDK 的 `AscendC::GetXxxMaxMinTmpSize` + `AscendC::GetXxxTilingInfo` 填充;**必须**把 `tiling.fieldName` 作为 OUT 引用**直接**传给 fill API,**禁止**"local 变量 + 拷贝 / setter"这种绕行。
- **正反例**:
  - ✅ `AscendC::GetRmsNormTilingInfo(..., tiling.rmsNormTiling, ...)`
  - ❌ `RmsNormTiling local = {}; AscendC::GetRmsNormTilingInfo(..., local, ...); tiling.rmsNormTiling = local;`
  - ❌ `RmsNormTiling local = {}; ...; tiling.set_rmsNormTiling(local);`
- **适用位置**:`op_host/<op>_tiling.h` + `op_host/<op>_tiling.cpp`
- **关联**:Case 2

### R-010:`REGISTER_TILING_DATA_CLASS` 注册

- **要点**:文件末尾 `REGISTER_TILING_DATA_CLASS(<Op>, <Op>TilingData);`,算子名遵循 R-003。
- **适用位置**:`op_host/<op>_tiling.h`

### R-011:host 侧字段写入走 setter

- **要点**:标量 / 数组字段用宏生成的 `tiling.set_<fieldName>(v)`;**禁止** `tiling.<fieldName> = v;` 直接成员赋值(R-009 指定的嵌套 SDK struct 例外,那种**反而不能**走 setter)。
- **适用位置**:`op_host/<op>_tiling.cpp`

### R-012:TilingFunc 五板斧

- **要点**:`TilingFunc(gert::TilingContext* context)` 结尾必须调齐以下五个:
  - `tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());`
  - `context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());`
  - `context->SetBlockDim(<实际核数>);`
  - `context->SetTilingKey(<schMode>);`
  - `context->GetWorkspaceSizes(1)[0] = <字节数 或 0>;`
- **适用位置**:`op_host/<op>_tiling.cpp`

### R-013:TilingFunc 注册

- **要点**:文件末尾 `IMPL_OP_OPTILING(<Op>).Tiling(TilingFunc);`
- **适用位置**:`op_host/<op>_tiling.cpp`

### R-014:schMode / TilingKey 映射表保持在一处

- **要点**:host 侧 `MapDtypeToSchMode` 返回的整数 = `context->SetTilingKey(n)` 的 `n` = kernel 侧 `TILING_KEY_IS(n)` 的 `n` = `<op>.json` 中 dtype 列的下标。四处必须完全一致。
- **适用位置**:`op_host/<op>_tiling.cpp` + `op_kernel/<op>.cpp` + `<op>.json`

---

## 三、Host include 与 API

### R-015:host 侧 AscendC tiling API 只能来自 `tiling/tiling_api.h`

- **要点**:`GetRmsNormTilingInfo` / `GetLayerNormTilingInfo` / `GetSoftmaxTilingInfo` 等 host-side 公开 API,**只能且必须**通过 `#include "tiling/tiling_api.h"` 获得。
- **展开**:`.asc` 中常见的 `#include "adv_api/normalization/rmsnorm_tiling.h"` / `#include "lib/..."` / `#include "impl/..."` 都是 kernel 直调构建独有的私有 include,msOpGen 工程里**不存在**这些路径,迁移时必须改写。
- **适用位置**:`op_host/<op>_tiling.h` + `op_host/<op>_tiling.cpp`
- **关联**:Case 1

### R-016:host 必需头三件套

- **要点**:TilingFunc 所在 `.cpp` 必须 include:
  - `"register/op_impl_registry.h"`
  - `"tiling/platform/platform_ascendc.h"`
  - 本算子的 `"<op>_tiling.h"`
  - 若需要 dtype utils:`"graph/utils/type_utils.h"`
- **适用位置**:`op_host/<op>_tiling.cpp`

### R-017:(DEPRECATED,已被 R-015 吸收)

- 原规则"禁止 ops-math 私有头渗入"已退役:本 skill 只支持 CANN 标准 msOpGen 框架(R-001),不存在其它框架私有头的引入需求。host 侧 include 白名单由 R-015 + R-016 统一约束:只允许 `tiling/tiling_api.h` / `tiling/platform/platform_ascendc.h` / `register/*` / `graph/*` 这类 CANN 对外公开聚合入口。任何其它"看起来像 CANN 内部"的头(如 `util/math_util.h` / `op_host/tiling_util.h` / `tiling/tiling_templates_registry.h` / `ascendc/host_api/tiling/template_argument.h` 等)均视为违规。外部引用 `[R-017]` 等价跳转至 R-015。

---

## 四、Kernel 编码约束

### R-018:kernel 入口签名固定

- **要点**:`extern "C" __global__ __aicore__ void <op>(GM_ADDR <inputs...>, GM_ADDR <outputs...>, GM_ADDR workspace, GM_ADDR tiling)`
- **展开**:参数顺序 = OpDef.Input 顺序 + OpDef.Output 顺序 + workspace + tiling;函数名遵循 R-004。
- **适用位置**:`op_kernel/<op>.cpp`

### R-019:`GET_TILING_DATA` 第一行

- **要点**:kernel 函数体第一行必须是 `GET_TILING_DATA(tilingData, tiling);`。
- **展开**:变量名约定 `tilingData`。
- **适用位置**:`op_kernel/<op>.cpp`

### R-020:`TILING_KEY_IS` 用数字字面量

- **要点**:`if (TILING_KEY_IS(0)) { ... } else if (TILING_KEY_IS(1)) { ... } ...`,使用**整数字面量**,不传 `constexpr` 命名常量、不传变量。
- **展开**:这是 CANN 官方公开算子库的惯例写法,跨版本最稳。部分 CANN 版本 / 预编译配置下,`TILING_KEY_IS(SCH_MODE_X)` 里的 `__builtin_constant_p` 对 `constexpr` 识别失败,会报 `Var: SCH_MODE_X ... can not be processed as numeric variables in the precompilation phase`。数值语义用注释表维护(`// 0 : FP32 / 1 : FP16 / ...`)。
- **展开**:**禁用** `if constexpr (...)` 做 schMode 分发。
- **适用位置**:`op_kernel/<op>.cpp`
- **关联**:R-014, Case 3

### R-021:device 端禁用 STL

- **要点**:`op_kernel/` 下**不得**出现 `<algorithm>` / `<vector>` / `<iostream>` / `<cstring>` / `<string>` 等 host 标准库头,也不得出现 `std::min` / `std::max` / `std::swap` / `std::memcpy` / `std::abs` 等 `std::*` 符号。
- **展开**:`std::min(a,b)` 统一替换为三元表达式 `(a < b) ? a : b`。host 段(`op_host/`)仍可自由使用 STL。
- **grep 自查**:`rg "\bstd::" op_kernel/` 命中即错
- **适用位置**:`op_kernel/`

### R-022:device 端内建函数加 `AscendC::` 前缀

- **要点**:`GetBlockIdx()` / `GetBlockNum()` / `PipeBarrier<PIPE_V>()` 等全部显式写 `AscendC::` 命名空间前缀,裸写会报 `use of undeclared identifier`。
- **适用位置**:`op_kernel/`

### R-023:device 端 C++ 数据类型用"实现名"

- **要点**:BF16 用 **`bfloat16_t`**(**不是** `bfloat16`);FP16 用 `half`;FP32 用 `float`;整型用 `<cstdint>` 的 `int8_t` / ... / `int64_t` / `uint8_t` / ... / `uint64_t`。
- **展开**:`bfloat16` 裸名在部分 CANN 版本是 `using bfloat16 = bfloat16_t;` 的兼容别名,但另一些版本直接没有,会报 `unknown type name 'bfloat16'`。**所有** C++ 类型位置(`GlobalTensor<T>` / `LocalTensor<T>` / `TQue<T>` / `sizeof(T)` / `(__gm__ T*)ptr` / `AllocTensor<T>()` / `DeQue<T>()`)一律用实现名。
- **相对的**:host 端 OpDef 用 `ge::DT_BF16` / `ge::DT_FLOAT16`,JSON / 注释用字符串 `"bfloat16"` / `"float16"`;两套命名互不借用。
- **grep 自查**:`rg "\bbfloat16\b" op_kernel/ | rg -v "bfloat16_t"` 命中即错
- **适用位置**:`op_kernel/`
- **关联**:Case 5

### R-024:Init 模板化接收 TilingData

- **要点**:kernel 类 `Init` 方法签名用 `template <class TD> Init(..., const TD& td, ...)`,不要硬编码 `RmsNormTilingData` 等具体类型。
- **展开**:`GET_TILING_DATA` 宏展开出来的结构体类型名在 CANN 不同版本可能微调,模板化可以免疫。
- **适用位置**:`op_kernel/<op>.h`

### R-025:LocalTensor 32 字节对齐

- **要点**:UB 上分配 buffer 的字节数必须 32 对齐:`(H * sizeof(T) + 31) / 32 * 32`。
- **适用位置**:`op_kernel/<op>.h`

### R-026:不强行删改 kernel 内 SDK 高阶 API 调用

- **要点**:原 `.asc` 里如 `AscendC::RmsNorm<T, ...>(...)` / `AscendC::LayerNorm<T, ...>(...)` 的 SDK 高阶 API 调用,**保真不改写**;需要的 tiling 从 `tilingData.fieldName` 透传。
- **展开**:为了"规避嵌套"而扁平化 SDK 字段、改写 kernel 调用——一律**禁止**。R-009 和本条是配套的。
- **适用位置**:`op_kernel/<op>.h` + `op_kernel/<op>.cpp`

---

## 五、OpDef(`op_host/<op>_def.cpp`)

> **OpDef 是最容易被 agent 写成"空壳"的文件**。下列 R-027 ~ R-036 需整组启用,配合 [cases.md](cases.md) 中 Case 6 阅读。

### R-027:开工前必填 OpDef 契约表

- **要点**:动手写 `op_host/<op>_def.cpp` 或 `<op>.json` 之前,必须**先在上下文中完整落纸**下列契约:
  - 每个 Input 的 `name` / `param_type` / **完整** `DataType[]` / **完整** `Format[]`
  - 每个 Output 的 `name` / `param_type` / **完整** `DataType[]` / **完整** `Format[]`
  - 每个 Attr 的 `name` / C++ 类型 / `param_type` / 具体 `default_value`(不能留 `<default>` / `TODO`)
  - SOC 系列名(R-034)
  - schMode 数 `N`(= `DataType[]` / `Format[]` 数组长度)
- **展开**:`.asc` 若无原型信息(kernel 直调模式),按以下规则推导:name 取 kernel 形参名;param_type 默认 REQUIRED;DataType 从 `.asc` 的 dtype 分支逻辑 / kernel 类模板实参 / main 里传入的数据类型倒推;Format 默认 `FORMAT_ND`;Attr 从 tiling struct 中的浮点/整型标量(`epsilon` / `dim` 等)与 main 的默认值抽取;SOC 从 `.asc` 顶部注释 / `run.sh` 推断,不确定**必须向用户确认,禁止猜**。
- **适用位置**:流程控制(开工准入)
- **关联**:R-028, R-029, Case 6

### R-028:OpDef 完工九点自检

- **要点**:写完 `op_host/<op>_def.cpp` 必须立即对照下表 grep + 肉眼核对,九点全过才能进入下一个文件。
- **展开**:

  | # | 检查点 | 通过标准 |
  | - | ---- | ---- |
  | 1 | 类名 | `class <Op> : public OpDef` 或 `: public ops::OpDef` 出现一次 |
  | 2 | 构造函数 | `<Op>::<Op>(...) : OpDef(name)`,算子名大驼峰 |
  | 3 | `.Input(` 计数 | ≥ 1 且与契约一致(`rg -c "\.Input\(" ...` ≥ 1) |
  | 4 | `.Output(` 计数 | ≥ 1 且与契约一致 |
  | 5 | DataType 实质非空 | 每个 Input/Output 后跟 `.DataType({ge::DT_..., ...})`,大括号内**至少一个**具体 `ge::DT_xxx`;`rg "\.DataType\(\s*\{\s*\}\s*\)" ...` 必须返回 0 |
  | 6 | Format 实质非空 | 每个 Input/Output 后跟 `.Format({ge::FORMAT_..., ...})`,同上 |
  | 7 | 长度对齐 | 同一 Input/Output 内 `Format({})` 长度 == `DataType({})` 长度 == schMode 数 N |
  | 8 | AICore 字面量 | `rg 'AICore\(\)\.AddConfig\("ascend[0-9a-z_]+"' ...` ≥ 1;`aicConfig` 前置完整设 6 个 flag + `ExtendCfgInfo("opFile.value","<op>")` |
  | 9 | 注册宏 | 末尾 `OP_ADD(<Op>);`(`rg "OP_ADD\(" ...` == 1) |
- **适用位置**:`op_host/<op>_def.cpp` 生成后立即执行
- **关联**:R-027, R-029, Case 6

### R-029:禁止任何占位符残留

- **要点**:最终交付的 `op_host/<op>_def.cpp` / `<op>.json` 中**不得**出现 `<Op>` / `<op>` / `<DType1>` / `<DType2>` / `<DTypeN>` / `<Format1>` / `<soc>` / `// TODO` / `/* ... */` 等占位符或待办标记。
- **grep 自查**:`rg "<DType[0-9]+>|<Format[0-9]+>|<soc>|<Op>|<op>|TODO:" op_host/<op>_def.cpp` 必须返回 0
- **适用位置**:`op_host/<op>_def.cpp`

### R-030:`OpDef` 在 ops 命名空间作用域内可解析

- **要点**:CANN 9.x 起 `OpDef` 声明在 `namespace ops {}` 中,编译点必须能解析它。下列两种等价做法任选:
  - ✅ 把 `class <Op> : public OpDef {...};` 整个包进 `namespace ops { ... }`(本 skill 推荐)
  - ✅ 放在全局域,写 `class <Op> : public ops::OpDef {...};`
  - ❌ 全局域 + 裸写 `public OpDef` → 报 `'OpDef' has not been declared`
- **适用位置**:`op_host/<op>_def.cpp`

### R-031:`Format` 用 `ge::FORMAT_ND`

- **要点**:`Format` / `UnknownShapeFormat` 的枚举用 `ge::FORMAT_ND`(带 `FORMAT_` 前缀),**不得**写 `ge::Format::ND` 或字符串 `"ND"`,否则报 `'ND' is not a member of 'ge::Format'`。
- **适用位置**:`op_host/<op>_def.cpp`

### R-032:DataType / Format 数组长度 = schMode 数

- **要点**:同一 Input/Output 内 `DataType[]` 长度 = `Format[]` 长度 = `UnknownShapeFormat[]` 长度 = schMode 数 N = `<op>.json` 里对应 `type` / `format` 数组长度。**列序**严格对齐(列 `i` 对应 schMode `i`)。
- **展开**:此条违反后果最阴险——编译可能通过,aclnn 运行时才报 `<Op>NotRegistered` / `Op<Op>:GetOpInfoFailed`。
- **适用位置**:`op_host/<op>_def.cpp` + `<op>.json`

### R-033:Attr 必须有具体 default_value

- **要点**:`.Attr("<name>").AttrType(OPTIONAL).Float(1e-6f)` 这类调用,default 必须是**具体字面量**,不得 `TODO` / 留空。
- **适用位置**:`op_host/<op>_def.cpp`

### R-034:SOC 用系列名,三处一致

- **要点**:`AICore().AddConfig("<soc>", cfg)` 的 `<soc>` 是 CANN 芯片**系列名**(family,全小写+无尾部型号数字):`"ascend910b"` / `"ascend310p"` / `"ascend910"` / `"ascend910_93"`。
- **展开**:同一 `<soc>` 在**四处**保持一字不差:
  1. `op_host/<op>_def.cpp` 的 `AICore().AddConfig("<soc>", cfg)`
  2. `msopgen gen -c ai_core-<soc>` 命令参数
  3. `CMakePresets.json` 的 `ASCEND_COMPUTE_UNIT.value`
  4. `op_host/config/<soc>/` 目录名
- **反例**:
  - ❌ `"ascend910b4"` / `"ascend910b3"`(具体型号)
  - ❌ `"Ascend910B"` / `"Ascend910B3"`(大驼峰)
  - 任何反例会报 `The soc version of op <Op> is not configured` 或 `cannot find chip config for ai_core-...`
- **映射**:`npu-smi info` 里的 `Ascend910B3` → 全小写 + 去尾部数字 → `ascend910b`。
- **适用位置**:四处
- **关联**:R-035

### R-035:`OpAICoreConfig` 六 flag + `ExtendCfgInfo`

- **要点**:`OpAICoreConfig` 实例必须完整设置:
  - `.DynamicCompileStaticFlag(bool)`
  - `.DynamicFormatFlag(bool)`
  - `.DynamicRankSupportFlag(bool)`
  - `.DynamicShapeSupportFlag(bool)`
  - `.NeedCheckSupportFlag(bool)`
  - `.PrecisionReduceFlag(bool)`
  - `.ExtendCfgInfo("opFile.value", "<op>")`(值遵循 R-004 snake_case)
- **适用位置**:`op_host/<op>_def.cpp`

### R-036:`OP_ADD(<Op>)` 末尾一行

- **要点**:文件末尾(算子类定义之后)有且仅有一行 `OP_ADD(<Op>);`,算子名大驼峰。
- **适用位置**:`op_host/<op>_def.cpp`

---

## 六、InferShape / InferDataType

### R-037:InferShape / InferDataType 注册

- **要点**:`IMPL_OP_INFERSHAPE(<Op>).InferShape(<Fn>).InferDataType(<Fn>);`
- **适用位置**:`op_host/<op>_infershape.cpp`

### R-038:`gert::InferShapeContext` 专用 API

- **要点**:`InferShapeContext::GetInputShape(i)` 返 **`const gert::Shape*`**(**直接就是 Shape**,没有 `GetStorageShape()`);`GetOutputShape(i)` 返可写 `gert::Shape*`,用 `SetDimNum(n)` + `SetDim(i, v)` 或 `SetDimNum(0)` + 循环 `AppendDim(v)` 填写输出 shape。
- **反例**:
  - ❌ `context->GetInputShape(i)->GetStorageShape()` → 报 `'const struct gert::Shape' has no member named 'GetStorageShape'`
  - ❌ `context->SetOutputShape(i, shape)` → 报 `'class gert::InferShapeContext' has no member named 'SetOutputShape'`
- **适用位置**:`op_host/<op>_infershape.cpp`
- **关联**:R-039, Case 4

### R-039:`gert::TilingContext` 专用 API(与 R-038 区别)

- **要点**:`TilingContext::GetInputShape(i)` 返 **`const gert::StorageShape*`**,需要 `->GetStorageShape()` 才拿到 `gert::Shape`。
- **展开**:这是 TilingFunc 里的姿势,与 InferShape **API 不同**,不要互相照搬。
- **适用位置**:`op_host/<op>_tiling.cpp`
- **关联**:R-038

---

## 七、图模式(`op_graph/`)

### R-040:图模式 InferDataType 走 `IMPL_OP`

- **要点**:若保留 `op_graph/<op>_graph_infer.cpp`,用 `IMPL_OP(<Op>).InferDataType(<Fn>);`,函数名避免与 `IMPL_OP_INFERSHAPE` 的冲突(加 `Graph` 后缀或放匿名命名空间)。
- **适用位置**:`op_graph/<op>_graph_infer.cpp`

---

## 八、Config / JSON

### R-041:`<op>_binary.json` 条目数 = schMode 数

- **要点**:`op_host/config/<soc>/<op>_binary.json` 中 bin 编译 matrix 的条目数 ≥ schMode 数,每条配置对应一个 dtype 组合列。
- **适用位置**:`op_host/config/<soc>/`

### R-042:`<op>_simplified_key.ini` 与 bin 对齐

- **要点**:若 TilingKey 由运行时 schMode 决定,`op_host/config/<soc>/<op>_simplified_key.ini` 为每个 dtype 组合添加一条 `[<binary-filename>]` 段。
- **适用位置**:`op_host/config/<soc>/`

### R-043:`<op>.json` 顶层数组 + op 大驼峰

- **要点**:`<op>.json` 顶层是数组 `[{...}]`(不是单对象);内层对象 `"op"` 字段用大驼峰,遵循 R-003。
- **适用位置**:`<op>.json`

### R-044:JSON 中 type/format 数组长度统一

- **要点**:`input_desc[*].type` / `output_desc[*].type` / `format` 所有数组**长度全部相等** = schMode 数。
- **展开**:每列(按下标 i 对齐)代表一个合法 dtype 组合,对应 `TILING_KEY_IS(i)`;JSON 中仅列合法组合。
- **适用位置**:`<op>.json`
- **关联**:R-032

### R-045:JSON 用 msopgen 语义名

- **要点**:JSON 的 dtype 值用 msopgen 支持的语义字符串:`float` / `float16` / `bfloat16` / `int8` / `uint8` / `int16` / `int32` / `int64` / `uint32` / `uint64` / `bool` / `double`。**不带** `DT_` 前缀,**不用** `ge::DT_xxx`。
- **适用位置**:`<op>.json`
- **关联**:R-023(这里讲的是"语义层"字符串,device 端 C++ 类型走 R-023)

### R-046:JSON 与 OpDef 逐列严格一致

- **要点**:`<op>.json` 里每一列 dtype / format,必须与 `op_host/<op>_def.cpp` 中 `.DataType({...})` / `.Format({...})` 的同下标元素**严格对齐**(名字不同但指同一类型:JSON 里 `"bfloat16"` 对 OpDef 里 `ge::DT_BF16`,映射唯一)。
- **适用位置**:`<op>.json` + `op_host/<op>_def.cpp`

---

## 九、验证闭环(由 [build-verify.md](build-verify.md) 执行)

### R-047:优先探测 msopgen,不可用则降级人工手册

- **要点**:skill 最终交付前必须执行 msopgen 环境探测;可用则走自动闭环,不可用则给出完整人工验证手册(**不**伪装执行)。
- **展开**:探测路径:`command -v msopgen` → `${ASCEND_HOME_PATH}/python/site-packages/bin/msopgen` → `${ASCEND_TOOLKIT_HOME}/...` → `${INSTALL_DIR}/...` → `/usr/local/Ascend/ascend-toolkit/latest/...` → `${HOME}/Ascend/ascend-toolkit/latest/...`;Windows 平台直接判定为不可用。
- **适用位置**:步骤 10(验证闭环)

### R-048:验证工程目录与用户输出同级

- **要点**:`<VerifyProjectDir>` 创建在 `<UserOutputDir>` 的**同级**(`${PARENT_DIR}/<op>_verify_project_<timestamp>`)。
- **展开**:**禁止**放在 `<UserOutputDir>` 内部(会污染最终交付),**禁止**使用 `/tmp`(失联风险),目录名加时间戳防重复。
- **适用位置**:步骤 10

### R-049:编译循环前先跑 OpDef 静态预检

- **要点**:进入 `bash build.sh` 循环**之前**,对 `<UserOutputDir>/op_host/<op>_def.cpp` 跑 R-028 的 grep 硬校验;任一点不通过→**整文件重写**,**不**进入编译循环。
- **适用位置**:步骤 10 编译前置

### R-050:编译循环最多 5 轮 + 错误映射表

- **要点**:`bash build.sh` 失败时读最后 200 行日志,按 [build-verify.md](build-verify.md) 的"错误关键字→修复动作→规则引用"表定位,**优先修 `<UserOutputDir>` 下的原文件**再同步到 `<VerifyProjectDir>`。连续 5 轮仍失败或同错连续 2 轮未解决→终止,返完整错误日志给用户。
- **适用位置**:步骤 10 编译循环

---

## 十、交付约束

### R-051:skill 只生成阶段 7 二进制一致性验证 harness

- **要点**:本 skill **不**生成泛化端到端精度测试框架;但阶段 7 **必须**生成或复用最小 aclnn harness,用于把原 kernel 直调 `input/` 数据喂给转换后的 aclnn 接口,并将输出 `.bin` 与原直调 `output/` 做二进制一致性比较。
- **展开**:
  - 允许产物仅限验证闭环需要的 `example/main.cpp` / `run_binary_compare.sh` / `CMakeLists.txt` 这类最小文件,调用方式参考 msOpGen 工程 `example/` 下 ACL 初始化、读 bin、创建 `aclTensor`、两段式 `aclnnXxxGetWorkspaceSize` + `aclnnXxx`、同步、写 bin。
  - 禁止扩展成随机数据生成、容差比对、torch/numpy 参考实现、性能测试或覆盖率测试。阶段 7 的判据只有 byte-level 相等。
- **适用位置**:[build-verify.md § B.7](build-verify.md#b7-阶段-7二进制一致性验证)

### R-052:本 skill 不生成/不修改 `main` / `ReadConfig` / `KernelCall`

- **要点**:原 `.asc` 的 `main()` / `ReadConfig()` / `KernelCall()` 在新工程中**全部删除**(框架自动拉起 kernel)。

### R-053:`msopgen gen` 必须显式带 `-lan cpp` 并校验生成结构

- **要点**:所有 `msopgen gen` 调用**必须**显式传 `-lan cpp`(等价 `--language cpp`),生成结束后**必须**校验输出目录下同时存在 `op_host/` 与 `op_kernel/`。任一缺失视为"落成了 TBE DSL 旧结构",判定失败。
- **展开**:
  - `msopgen gen` 的 `-lan` 参数**默认值是 `py`**(TBE Python DSL 旧模式,产物为 `tbe/impl/*.py` + `op_info_cfg/`),与本 skill 交付的 `op_host/*.cpp` + `op_kernel/*.cpp` 结构**完全不兼容**。只要命令里漏了 `-lan cpp`,后续所有"覆盖源文件 → 编译"的步骤都会失败,且错误形态杂乱,很难逆向定位根因。所以**必须**在命令里显式写 `-lan cpp`,不依赖任何默认值或环境推断。
  - **结构校验命令**(B.2 结束后立刻跑):
    ```bash
    test -d "${VERIFY_DIR}/op_host" && test -d "${VERIFY_DIR}/op_kernel" \
        || { echo "[FATAL] msopgen 未生成 Ascend C 标准结构,检查 -lan cpp"; exit 1; }
    # 负向检查:出现以下任一即判定失败
    [ -d "${VERIFY_DIR}/tbe" ] || [ -d "${VERIFY_DIR}/impl" ] || [ -d "${VERIFY_DIR}/op_info_cfg" ] \
        && { echo "[FATAL] 出现 TBE DSL 残留目录(tbe/impl/op_info_cfg),重跑 msopgen gen 并确保 -lan cpp"; exit 1; }
    ```
- **校验失败处理**:按 [R-058](rules.md#r-058useroutputdir-与-verifyprojectdir-必须独立共存禁止清理验证目录) 将坏结构 `mv` 重命名备份,用新时间戳目录重跑带 `-lan cpp` 的 `msopgen gen`。**不要**尝试手工改目录结构,也不得 `rm -rf` 验证目录。
- **适用位置**:[build-verify.md § B.2](build-verify.md#b2-首次生成骨架--替换源文件) 及降级手册第 1 步;[examples.md 使用 msopgen 创建工程](examples.md#使用-msopgen-创建工程并替换文件)
- **关联**:R-047 / R-048 / R-050

### R-054:kernel 计算逻辑**原样迁移,禁止改写**(最高优先级,凌驾于任何"风格偏好")

- **要点**:`.asc` 中的 **kernel 计算逻辑是黑盒**,本 skill 只做**位置搬运 + 形式适配**,**绝不**做"优化 / 简化 / 重构 / 合并 / 重命名 / 调整顺序 / 改变精度"。如果搬完之后 agent 觉得"这里写得不优雅"、"这段可以用 `Ascend C` 更新 API 简化"、"这个临时变量没必要"——**按住手,不要改**。
- **必须原样保留**:
  1. **类名 / 模板参数名 / 类成员函数名**:原 `.asc` 中 `template <typename T> class RmsNormCustom { __aicore__ inline void CopyIn(...); __aicore__ inline void Compute(...); __aicore__ inline void CopyOut(...); ... };`,新 `op_kernel/<op>.h` / `.cpp` 里**类名、模板参数、公有/私有成员函数全集**必须一一对应,连函数顺序都保持。
  2. **Init / Process 两段式结构**:`Init(GM_ADDR ..., TilingData& tiling)` + `Process()` 两个入口,参数列表、局部变量名、初始化顺序**全部照抄**;仅因 TilingData 由 `GET_TILING_DATA` 提供而**允许**调整 `Init` 的 tiling 参数类型(由原 `const XxxTiling&` 改为 `XxxTilingData&`)。
  3. **算术表达式**:ReduceSum / Mul / Div / Sqrt / Add / Cast 的**调用序列、括号、eps 相加位置、中间张量复用模式、rstd 计算公式**逐字不动。`sum / n + eps` 不得改成 `sum / (n + eps)`;`1.0f / sqrt(x)` 不得改成 `rsqrt(x)`——**哪怕数学上等价**,累加舍入顺序都可能不同。
  4. **循环边界、UB 块切分、双缓冲与 Pipe 配置**:`InitBuffer` / `TPipe` / `TQue` 的深度、大小、`DataCopyPad` 参数、`Duplicate` / `Brcb` 调用**不动**。
  5. **Tiling 字段读取方式**:原 `.asc` 是 `tiling.totalN`,迁移后在 `Init` 里仍读 `tiling.totalN`(`GET_TILING_DATA` 宏生成的 `tilingData.totalN` 就是同名字段)。**禁止**在 kernel 里把 `tilingData` 字段改名、扁平化或拆散传入。
  6. **schMode 分发逻辑**:原 `.asc` 中 `if (...) { op.template Process<0>(); }` 这类分支,迁移到 `extern "C"` 入口后**仅**把外层 `if (mode == 0)` 换成 `TILING_KEY_IS(0)`([R-020]),内层模板实例化路径、模板特化参数**不动**。
- **允许(必须)改的只有以下 7 类"形式适配"**(其它都不许改):
  1. `#include`:加 `kernel_operator.h` / 删 `.asc` 独有的私有路径头 — 见 [R-015](rules.md#r-015host-侧-ascendc-tiling-api-只能来自-tilingtiling_apih)
  2. 入口签名:`__global__` 直调 → `extern "C" __global__ __aicore__ void <op>(GM_ADDR ..., GM_ADDR workspace, GM_ADDR tiling)` — [R-018]
  3. TilingData 获取:`<<<>>>` 侧手拷结构体 → `GET_TILING_DATA(tilingData, tiling);` — [R-019]
  4. schMode 分发:`if/switch` → `TILING_KEY_IS(N)` 数字字面量 — [R-020]
  5. 内建函数前缀:`GetBlockIdx()` → `AscendC::GetBlockIdx()` — [R-022]
  6. 裸类型名:`bfloat16` → `bfloat16_t`,保留 `half` / `float` 不动 — [R-023]
  7. STL 脱敏:`std::min(a,b)` → `(a<b)?a:b`,删 `#include <algorithm>` — [R-021]
- **删/改** `main` / `ReadConfig` / `KernelCall` / `<<<>>>` 启动包装——这些是**直调专属**入口,迁移时整段删除(不属于 kernel 计算逻辑)— [R-052]
- **禁止**:新增任何**原 `.asc` 没有**的成员函数、辅助 lambda、内联 helper、算子间融合。需要这些说明你不是在迁移、是在重写。
- **校验方法**:见 [R-055](rules.md#r-055kernel-语义完整性静态校验编译前执行)
- **适用位置**:`op_kernel/<op>.cpp` + `op_kernel/<op>.h`
- **关联**:[R-018] ~ [R-023], [R-052], [R-055]

### R-055:kernel 语义完整性静态校验(编译前执行)

- **要点**:进入编译循环**之前**,对 `<UserOutputDir>/op_kernel/` 与原 `.asc` 做一次**函数名集合比对**和**关键算子调用计数比对**,任一失败即回到 [R-053b] 重写,**不进**编译循环。
- **展开**:
  1. **kernel 类名必须存在**:抓 `.asc` 里 `class <ClassName>` / `struct <ClassName>`(仅 `__aicore__` 上下文),每个名字在 `op_kernel/<op>.{cpp,h}` 里必须能 grep 到。
  2. **`__aicore__ inline` 成员函数名集合比对**:`.asc` 里的每一个 `__aicore__ inline <ret> <Func>(` 函数名,在 `op_kernel/` 里必须**出现且仅出现一次**声明(+ 可选一次定义)。
  3. **AscendC 计算 API 调用计数单调不减**:`ReduceSum` / `Mul` / `Div` / `Add` / `Sub` / `Sqrt` / `Rsqrt` / `Cast` / `Duplicate` / `Brcb` / `DataCopyPad` / `DataCopy` 这些 API,**新 kernel 的出现次数 ≥ 原 `.asc`**(允许因为新增 schMode 分支而复制粘贴整段,所以是"不减"而非"相等")。出现**严格小于**即判定有计算被删除/合并。
  4. **禁止新增 `main` / `ReadConfig` / `KernelCall`**:`rg "\b(main|ReadConfig|KernelCall)\s*\(" op_kernel/` 必须 == 0([R-052])。
  5. **参考 grep 片段**(shell):
     ```bash
     # (1) 类名存在性
     for C in $(rg -oN "class\s+(\w+)" "${ASC_FILE}" -r '$1' | sort -u); do
         rg -q "\b${C}\b" "${KERNEL_DIR}" || { echo "[FATAL] kernel class ${C} 丢失"; exit 1; }
     done
     # (2) __aicore__ inline 成员函数名
     rg -oN "__aicore__\s+inline\s+\w[\w:<>\s,\*&]*?\s+(\w+)\s*\(" "${ASC_FILE}" -r '$1' | sort -u | \
     while read F; do
         rg -q "\b${F}\s*\(" "${KERNEL_DIR}" || { echo "[FATAL] kernel 成员函数 ${F} 丢失"; exit 1; }
     done
     # (3) AscendC 计算 API 计数
     for API in ReduceSum Mul Div Add Sub Sqrt Rsqrt Cast Duplicate Brcb DataCopyPad DataCopy; do
         N_ASC=$(rg -c "\b${API}\s*\(" "${ASC_FILE}" || echo 0)
         N_NEW=$(rg -c "\b${API}\s*\(" -g '!*.asc' "${KERNEL_DIR}" | awk -F: '{s+=$NF} END{print s+0}')
         [ "${N_NEW}" -ge "${N_ASC}" ] || { echo "[FATAL] ${API} 调用次数 ${N_NEW} < 原 ${N_ASC}"; exit 1; }
     done
     ```
- **校验失败处理**:**不要**改这个脚本去"适配",**也不要**小修小补迁移结果。回到 `.asc`,从原类开始**整段重搬**。连续两次仍失败 → 告诉用户 `.asc` 结构需要先分段,手工协助。
- **适用位置**:[build-verify.md § B.3](build-verify.md#b3-编译前-opdef-静态预检-r-049) 之前;独立成门 § B.3.1
- **关联**:[R-049], [R-054]

### R-056:host 必须严格拆成 4 个文件,**禁止**合并单 cpp

- **要点**:`op_host/` 下**必须且仅有**以下五个交付文件(加上 `config/` 目录),多一个少一个都判失败:
  1. `<op>_tiling.h`      — TilingData 结构 + `REGISTER_TILING_DATA_CLASS`
  2. `<op>_tiling.cpp`    — `TilingFunc` 实现 + `IMPL_OP_OPTILING`
  3. `<op>_def.cpp`       — `class <Op> : public OpDef` + `AICore().AddConfig` + `OP_ADD`
  4. `<op>_infershape.cpp` — `InferShape` / `InferDataType` 函数 + `IMPL_OP_INFERSHAPE`
  5. `config/<soc>/<op>_binary.json` + `config/<soc>/<op>_simplified_key.ini`
- **展开**:msopgen 默认把 OpDef / InferShape / Tiling **合并**生成一个 `<op>.cpp`;本 skill 交付产物**必须是拆分版**。具体做法:
  - 在 `<UserOutputDir>/op_host/` 直接生成拆分的 4 个文件(**不**生成合并的 `<op>.cpp`)
  - 在 B.2 覆盖源文件之前,**删除** msopgen 刚生成的 `<VerifyProjectDir>/op_host/<op>.cpp`
  - 再 `cp -rf <UserOutputDir>/op_host/. <VerifyProjectDir>/op_host/`
- **硬校验命令**(B.2 源文件替换后立刻跑):
  ```bash
  HOST_DIR="${USER_DIR}/op_host"
  # 正向:拆分 4 件套全部存在
  for F in "${OP_NAME,,}_tiling.h" "${OP_NAME,,}_tiling.cpp" "${OP_NAME,,}_def.cpp" "${OP_NAME,,}_infershape.cpp"; do
      [ -f "${HOST_DIR}/${F}" ] || { echo "[FATAL] 缺少拆分文件 ${F}"; exit 1; }
  done
  # 负向:不得存在未拆分的 <op>.cpp
  [ -f "${HOST_DIR}/${OP_NAME,,}.cpp" ] && { echo "[FATAL] 出现合并文件 ${OP_NAME,,}.cpp,拆分未完成"; exit 1; }
  # 负向:不得在错误位置合并(例如 OpDef 被写进 tiling.cpp)
  rg -q "class\s+${OP_NAME}\s*:\s*public\s+\w*OpDef\b" "${HOST_DIR}/${OP_NAME,,}_tiling.cpp" \
      && { echo "[FATAL] OpDef 定义误写在 tiling.cpp"; exit 1; }
  rg -q "IMPL_OP_OPTILING" "${HOST_DIR}/${OP_NAME,,}_def.cpp" \
      && { echo "[FATAL] IMPL_OP_OPTILING 误写在 def.cpp"; exit 1; }
  rg -q "IMPL_OP_INFERSHAPE" "${HOST_DIR}/${OP_NAME,,}_tiling.cpp" \
      && { echo "[FATAL] IMPL_OP_INFERSHAPE 误写在 tiling.cpp"; exit 1; }
  ```
- **违反处理**:把错放位置的代码段**整段搬到正确文件**,不要在合并文件上做小补丁。
- **适用位置**:`<UserOutputDir>/op_host/` + [build-verify.md § B.2.1](build-verify.md#b21-host-文件拆分校验门-r-056)
- **关联**:R-013, R-030, R-036, R-038, R-039

### R-057:七阶段严格流程,**不得跳步或合并**

- **要点**:拆解交付必须严格走七阶段顺序,每阶段**产物齐全**才能进下一阶段。缺任何一阶段的输出 → 本次拆解**判失败**,不允许"差一点就算完成"。
  ```
  阶段 1: 读取 .asc        → 产物:OpDef 契约表(R-027)、kernel 类名/函数名/计算 API 清单
  阶段 2: 拆分 host/kernel  → 产物:<UserOutputDir>/op_host/ 4 件套 + op_kernel/ 2 件套 + config/<soc>/*
  阶段 3: 生成 <op>.json    → 产物:<UserOutputDir>/<op>.json (msopgen 原型定义,由 agent 手写,作为 msopgen -i 输入)
  阶段 4: msopgen gen 生成工程 → 产物:<VerifyProjectDir>/(op_host/op_kernel/CMakePresets.json/build.sh)
                              + 覆盖替换 <UserOutputDir> 的拆分文件
  阶段 5: 编译构建          → 产物:<VerifyProjectDir>/build_out/custom_opp_*.run
  阶段 6: 安装算子包        → 产物:<VerifyProjectDir>/install_custom_opp.log + 当前环境可发现该自定义算子
  阶段 7: 二进制一致性验证  → 产物:aclnn 输出 bin + 与原直调 output 的 byte-level compare 报告
  ```
- **展开**:
  - **阶段 2 不能先于阶段 1**:没有契约表和 kernel 清单就直接写代码 → OpDef 会空壳、kernel 会漏迁移。
  - **阶段 3 不能省略**:`<op>.json` **不是**可选步骤。msopgen 的**唯一输入入口**就是 JSON;漏生成 JSON 直接让阶段 4 无法执行。JSON 必须与 OpDef 契约表**逐列对齐**([R-046])。
  - **阶段 4 必须实际跑 `msopgen gen`**:只生成 op_host/op_kernel 源文件**不算交付完成**。msopgen 会生成工程所需的 `CMakeLists.txt` / `CMakePresets.json` / `build.sh` / `op_host/CMakeLists.txt` / `op_kernel/CMakeLists.txt` 等编译必需的脚手架,**agent 不得手写**这些文件替代。
  - **阶段 5 必须实际跑 `bash build.sh`**:跳过编译就交付 = 任务失败。msopgen 可用但 agent 拒绝跑编译 = 偷懒,按失败论处。msopgen 不可用则按 [build-verify.md § D](build-verify.md#d-降级环境不可用时的人工验证手册) 出人工手册,**并明确告知用户本次未做编译验证**。
  - **阶段 6 必须实际安装 `.run`**:编译成功但未执行 `./build_out/custom_opp_*.run` = 未完成 aclnn 运行前置条件。安装失败不能进入阶段 7。
  - **阶段 7 必须实际二进制比较**:只说"可参考 example"或只跑 aclnn 不比较原直调输出,都不算完成。比较必须以原直调输入/输出目录为基准,逐 `.bin` byte-level 相等。
- **阶段门校验**(每阶段结束自检;任一失败回本阶段重做,不跳):
  | 阶段 | 完成判据 | 校验命令(示意) |
  | ---- | ---- | ---- |
  | 1 | 契约表完整 + kernel 类/函数清单完整 | 打印契约表并核对 `.asc` 源文件 |
  | 2 | host 拆分 4 件套 + kernel 2 件套 + config 双文件齐全 | [R-056] 硬校验 + `ls <UserOutputDir>/op_{host,kernel}/` |
  | 3 | `<UserOutputDir>/<op>.json` 存在,顶层数组非空,字段与 OpDef 契约表逐列一致 | `jq . <op>.json` + [R-046] 逐列对齐检查 |
  | 4 | `<VerifyProjectDir>/` 下有 `CMakeLists.txt` + `CMakePresets.json` + `build.sh` + `op_host/` + `op_kernel/`;**无** `tbe/impl/op_info_cfg/`([R-053]) | `ls` + [R-053] 结构校验门 |
  | 5 | `<VerifyProjectDir>/build_out/` 下有 `custom_opp_*.run` | `ls build_out/*.run` |
  | 6 | `.run` 已直接执行且安装日志保留 | `./build_out/custom_opp_*.run 2>&1 \| tee install_custom_opp.log` |
  | 7 | aclnn 输出 `.bin` 与原直调输出逐文件二进制一致 | `cmp -s <origin_output>.bin <aclnn_output>.bin` / `sha256sum` |
- **禁止模式**(一旦踩中就是"跳步交付",本次拆解判失败):
  - ❌ 只交付 `<UserOutputDir>/op_host/` + `op_kernel/`,没生成 `<op>.json`
  - ❌ 生成了 `<op>.json` 但没跑 `msopgen gen`
  - ❌ 跑了 `msopgen gen` 但没执行 `bash build.sh`
  - ❌ `bash build.sh` 失败就直接把编译失败的结果交给用户(没走 [R-050] 5 轮错误修复循环)
  - ❌ 生成了 `custom_opp_*.run` 但没直接执行安装
  - ❌ 安装成功但没用原直调 `input/` 通过 aclnn 产出结果
  - ❌ 只做容差比较 / 打印数值摘要,没做 byte-level binary compare
  - ❌ host 生成单个合并 `<op>.cpp` 而非拆分 4 件套([R-056])
- **适用位置**:贯穿全流程,[SKILL.md § 工作流程](../SKILL.md#工作流程) 直接对应此七阶段
- **关联**:[R-027], [R-046], [R-049], [R-050], [R-053], [R-055], [R-056], [R-059], [R-060]

### R-058:`<UserOutputDir>` 与 `<VerifyProjectDir>` 必须独立共存,禁止清理验证目录

- **要点**:两套目录是**并列的双交付件**,**绝不**允许重叠、嵌套、覆盖或在交付前清理:
  1. `<UserOutputDir>` ≠ `<VerifyProjectDir>`(绝对路径不相同,且互不为父子关系)
  2. 最终交付清单**同时包含**两者:
     - `<UserOutputDir>`:纯净的拆分成果(host 4 件套 + kernel 2 件套 + config + `<op>.json`)——用户真正要归档/提交的源头
     - `<VerifyProjectDir>`:msopgen 落地的完整单算子工程(加 `CMakeLists.txt` / `CMakePresets.json` / `build.sh` / `build_out/custom_opp_*.run`)——编译证据链,必须留存
  3. **任何阶段**(包括 msopgen 结构校验失败 / 编译失败 / 五轮修复耗尽)都**禁止** `rm -rf "${VERIFY_DIR}"`;需要重跑 msopgen 时,先**重命名备份**再新建
- **展开**:
  - `<VerifyProjectDir>` 不能:① 等于 `<UserOutputDir>`;② 位于 `<UserOutputDir>` 内部(如 `<UserOutputDir>/verify/`);③ 把 `<UserOutputDir>` 包在自己里面;④ 放在 `/tmp` 或其它会被系统回收的路径
  - 推荐目录名范式:`${PARENT_DIR}/<op_snake>_verify_project_<timestamp>`,与 `<UserOutputDir>` **同级并列**
  - 失败场景处置:
    - msopgen 结构落回 TBE DSL → **重命名** `${VERIFY_DIR}` 为 `${VERIFY_DIR}.badstructure_<ts>`,再用新时间戳目录重跑
    - 五轮编译循环耗尽仍失败 → `${VERIFY_DIR}` **原封保留**,上报给用户(里面的编译日志是下一步定位的核心证据)
    - 用户要求清理 → 只在用户**显式确认**后才删,且必须分别确认两个目录
  - 最终向用户汇报时,两个路径**都**列在产物清单,且说明"验证目录已保留,可直接 `cd <VerifyProjectDir> && bash build.sh` 复现编译"
- **硬校验命令**(B.1 结束立刻跑,不通过不得进 B.2):
  ```bash
  # 绝对路径规范化
  USER_ABS="$(readlink -f "${USER_DIR}" 2>/dev/null || realpath "${USER_DIR}")"
  VERIFY_ABS="$(readlink -f "${VERIFY_DIR}" 2>/dev/null || realpath "${VERIFY_DIR}" 2>/dev/null || echo "${VERIFY_DIR}")"

  # 1. 不得相同
  [ "${USER_ABS}" = "${VERIFY_ABS}" ] && { echo "[FATAL] VerifyProjectDir 与 UserOutputDir 相同 (${USER_ABS}),违反 R-058"; exit 1; }

  # 2. 不得互为父子
  case "${VERIFY_ABS}/" in
      "${USER_ABS}/"*) echo "[FATAL] VerifyProjectDir 位于 UserOutputDir 内部 (${VERIFY_ABS} in ${USER_ABS}),违反 R-058"; exit 1 ;;
  esac
  case "${USER_ABS}/" in
      "${VERIFY_ABS}/"*) echo "[FATAL] UserOutputDir 位于 VerifyProjectDir 内部 (${USER_ABS} in ${VERIFY_ABS}),违反 R-058"; exit 1 ;;
  esac

  # 3. 不得在 /tmp
  case "${VERIFY_ABS}" in
      /tmp/*|/var/tmp/*) echo "[FATAL] VerifyProjectDir 落在临时目录 (${VERIFY_ABS}),违反 R-058"; exit 1 ;;
  esac

  echo "[OK] <UserOutputDir> 与 <VerifyProjectDir> 独立性校验通过"
  echo "     USER   = ${USER_ABS}"
  echo "     VERIFY = ${VERIFY_ABS}"
  ```
- **违反处理**:
  - 若已经手滑把 `VERIFY_DIR` 指到 `USER_DIR` 或其子目录 → **停手**,重命名备份可能已污染的产物,重新挑一个并列路径再从 B.1 开始
  - 若脚本里出现过任何 `rm -rf "${VERIFY_DIR}"` → 删掉这条命令,改为重命名备份(`mv "${VERIFY_DIR}" "${VERIFY_DIR}.<reason>_$(date +%s)"`)
- **适用位置**:[build-verify.md § B.1](build-verify.md#b1-准备变量) 之后,B.2 之前;汇报阶段 [§ C](build-verify.md#c-汇报)
- **关联**:[R-048](rules.md#r-048验证工程目录与用户输出同级), [R-050](rules.md#r-050最多-5-轮错误修复循环), [R-053], [R-057]

---

### R-059:阶段 6 必须直接执行 `custom_opp_*.run` 安装算子包

- **要点**:阶段 5 产出 `<VerifyProjectDir>/build_out/custom_opp_*.run` 后,必须进入 `<VerifyProjectDir>` 并直接执行 `./build_out/custom_opp_*.run` 完成安装,安装日志必须保留为交付证据。
- **展开**:
  - 命令形态固定:`cd <VerifyProjectDir> && ./build_out/custom_opp_*.run 2>&1 | tee install_custom_opp.log`
  - 不追加 `--quiet` / `--install` / 自造参数;CANN 自定义算子包的 `.run` 默认交互/安装逻辑由包自身处理。
  - 若安装失败,不得进入阶段 7。先检查 CANN 环境变量、权限、旧包冲突、`ASCEND_CUSTOM_OPP_PATH` / `ASCEND_OPP_PATH` 等环境问题,修复后重试并保留失败日志。
- **适用位置**:[build-verify.md § B.6](build-verify.md#b6-阶段-6安装算子包)
- **关联**:[R-057], [R-060]

### R-060:阶段 7 必须生成 `example/` 并用原直调输入做 aclnn 二进制一致性验证

- **要点**:阶段 7 必须在最终 msopgen 工程 `<VerifyProjectDir>/example/` 下生成可编译运行的 aclnn 示例工程,使用原 kernel 直调工程的 `input/` `.bin` 作为 aclnn 输入,转换后算子的 aclnn 输出 `.bin` 与原 kernel 直调 `output/` 对应 `.bin` **逐字节一致**。
- **展开**:
  - `example/` 目录必须包含 `main.cpp`、`CMakeLists.txt`、从原直调工程复制来的 `input/` 和 `output/`。`output/` 是原直调基准,不得覆盖。
  - 输入源固定为原直调工程输入目录(如 `rms_norm/input`),不得重新随机生成输入。若需维度/dtype/attr,从原直调的 `scripts/gen_data.py`、`run.sh`、`main` 或配置文件抽取。例如 RmsNorm 从 `gen_data.py` 读取 `rows` / `cols` / `epsilon` / `dtype`、`input_x.bin` / `input_gamma.bin`、`golden_y.bin` / `golden_rstd.bin` 以及 32B 对齐规则。
  - `main.cpp` 必须参考 `rms_norm_verify_project_20260425_181328/example/main.cpp` 的关键结构:ACL 初始化 → 读 `.bin` → 创建 `aclTensor` → `aclnn<Op>GetWorkspaceSize` → `aclnn<Op>` → `aclrtSynchronizeStream` → device 输出拷回 host → 写 `.bin`。
  - aclnn 接口名按算子名拼接:`<Op>` 大驼峰 → `aclnn<Op>GetWorkspaceSize` + `aclnn<Op>`。示例:`RmsNorm` → `aclnnRmsNormGetWorkspaceSize` / `aclnnRmsNorm`。头文件通常为 `aclnn_<op>.h`,但以实际安装后的 custom op api 头为准。
  - `CMakeLists.txt` 参考 [reference.md](reference.md) 的 `example/CMakeLists.txt`,必须生成 `opapi_test`,通过 `ASCEND_CUSTOM_PATH` 或 `/usr/local/Ascend/cann` 定位 CANN,include `${ASCEND_PATH}/include`、`${ASCEND_PATH}/include/aclnn` 和 custom op api 头路径,并显式链接 `libascendcl.so` / `libnnopbase.so` / `libopapi_math.so` / `libopapi_nn.so` / `libcust_opapi.so`。
  - 固定执行方式:
    ```bash
    cd <VerifyProjectDir>/example
    mkdir -p build
    cd build
    cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
    make
    cd bin
    ./opapi_test
    ```
  - aclnn 输出目录必须与原直调输出目录分离,推荐 `<VerifyProjectDir>/example/aclnn_output/`;不得覆盖 `example/output/`。
  - 比较必须逐文件 `cmp -s` 或 SHA256 完全相同。任一字节不同 = 精度验证失败;禁止用 rtol/atol 容差放行,也禁止只比较 shape、文件大小或打印前几个值。
  - 不一致时优先回查 [R-054] / [R-055] kernel 保真,其次核对 OpDef dtype/format/attr、shape 推导、aclnn harness 的输入输出 dtype 与文件尺寸。
- **适用位置**:[build-verify.md § B.7](build-verify.md#b7-阶段-7二进制一致性验证)
- **关联**:[R-051], [R-054], [R-055], [R-057], [R-059]

---

## 引用指引

- 其它文件(`../SKILL.md` / `cases.md` / `verification-checklist.md` / `reference.md` / `examples.md` / `build-verify.md`)引用规则时**只**写 `[R-xxx]`,不复述内容。
- 如果发现规则**真的需要**在其它地方做本地化阐释(如某个案例想强调某条规则的特殊触发条件),可在案例正文中先写 `[R-xxx]` 再展开,**且展开部分不要改动规则语义**,只做情境化解读。
- 新增规则:继续往末尾追加 `R-NNN`,永不复用空缺编号。
- 退役规则:保留墓碑,格式:

  ```
  ### R-0NN:(DEPRECATED,YYYY-MM-DD 退役,被 R-xxx 吸收)
  ```
