# 典型案例集(Cases)

> 本文件是真实出错记录的**案例库**。每个案例聚焦一个**错误现象→来源→修正→反思**的完整剧本,不重复 [rules.md](rules.md) 里的规则展开——正文末尾的"涉及规则"段给出反向引用。
>
> 新增案例:追加 Case-NN,永不复用已退役的编号;案例号与规则号是两个独立序列。

---

## Case 1:host 侧盲目照搬 `.asc` 的 `adv_api/...` 等 SDK 私有 include

**错误现象**:host 侧编译器报

```
fatal error: adv_api/normalization/rmsnorm_tiling.h: No such file or directory
```

(或 `lib/...` / `impl/...` 等其他 SDK 内部路径同类错误)

**错误来源**:原 `.asc` 中 host 端写 `#include "adv_api/normalization/rmsnorm_tiling.h"` / `#include "lib/matmul/matmul_server.h"` 之类**私有内部路径**。这些路径在 `.asc` 的 kernel 直调构建脚本里通过自定义 CMake `target_include_directories` 补进了 include search path;**但 msOpGen 生成的标准算子工程不会暴露这些私有路径**——它只暴露 CANN 对外**统一的聚合入口**。

**正确做法对照**:

| 能力 | `.asc` 常见写法(私有路径) | CANN 标准 msOpGen host 侧统一入口 |
| ---- | ---- | ---- |
| RmsNorm / LayerNorm / Softmax / DeepNorm 等 Normalization host tiling API(`GetXxxMaxMinTmpSize` / `GetXxxTilingInfo` / `XxxTiling` 结构) | `adv_api/normalization/rmsnorm_tiling.h` | **`tiling/tiling_api.h`** |
| AscendC 所有公开高阶 host tiling API 聚合 | `adv_api/...` / `lib/...` 分散头 | **`tiling/tiling_api.h`** |
| 平台查询(核数 / UB 容量) | `tiling/platform/platform_ascendc.h` | 同(两边都存在) |
| OpDef / IMPL_OP_OPTILING / IMPL_OP_INFERSHAPE 注册宏 | —— | `register/op_def_registry.h` + `register/op_impl_registry.h` |
| TilingData 基类宏 | —— | `register/tilingdata_base.h` |

**自查步骤**(每次生成完 host 侧 tiling 文件):

- 是否把 `.asc` 的 host include 原样复制到新 tiling 文件?若是 → 立刻把 `adv_api/` / `lib/` / `impl/` 开头的 SDK 私有路径统一改为 `tiling/tiling_api.h`。
- 是否用到 `AscendC::GetRmsNormTilingInfo` / `GetLayerNormTilingInfo` / `GetSoftmaxTilingInfo` 等公开 host tiling API?若是 → 头文件**必须且只需**是 `tiling/tiling_api.h`。
- 若 `op_host/<op>_tiling.h` 使用了 `TILING_DATA_FIELD_DEF_STRUCT(XxxTiling, fieldName)`,该 `.h` **必须**也 include `tiling/tiling_api.h`,否则宏展开时类型未定义。

**根因反思**:"照抄 `.asc` 的 include"是 host 迁移最高频的翻车点。`.asc` 的 include 面向**直调构建脚本**,其 include search path 经常包含 CANN 安装目录下的**私有**子树;msOpGen 工程只暴露**对外发布 API**。迁移时**按"API 能力"选 include,而不是按"源文件字面"**。

**涉及规则**:[R-015](rules.md#r-015host-侧-ascendc-tiling-api-只能来自-tilingtiling_apih), [R-016](rules.md#r-016host-必需头三件套)

---

## Case 2:`TILING_DATA_FIELD_DEF_STRUCT` 嵌套 SDK tiling,`local + 拷贝` 触发 `double free`

**错误现象**:算子编译通过、aclnn 调用也能进入 kernel,但在 **host tiling 函数返回后**(或 `aclnnXxxGetWorkspaceSize` 结束那一刻)进程崩溃:

```
free(): double free detected in tcache 2
Aborted (core dumped)
```

有时也表现为 `malloc_consolidate(): invalid chunk size` / `corrupted size vs. prev_size`。**全部**来自 glibc 堆校验,**与 kernel 侧无关**。

**错误来源**(两种等价错法,都会崩):

```cpp
// ❌ 错法 A:直接成员赋值
RmsNormTiling rmsNormTiling = {};
AscendC::GetRmsNormTilingInfo(..., rmsNormTiling, ...);
tiling.rmsNormTiling = rmsNormTiling;

// ❌ 错法 B:setter 拷贝(看似"更合规",一样崩)
RmsNormTiling rmsNormTiling = {};
AscendC::GetRmsNormTilingInfo(..., rmsNormTiling, ...);
tiling.set_rmsNormTiling(rmsNormTiling);
```

**根因**:`TILING_DATA_FIELD_DEF_STRUCT(XxxTiling, fieldName)` 展开出的 `tiling.fieldName` 是 **TilingData 内部序列化 buffer 上的一块真实槽位**(类型等同 `XxxTiling` 的引用/视图),**不是**独立值对象。SDK 的 `AscendC::GetXxxTilingInfo` 签名就是 `XxxTiling&` 作为 OUT 参数,设计意图是"调用方指定最终落地位置、SDK 原地填充"。

插一个栈上的 `XxxTiling local = {}` 做中转,就等于把 SDK 填好的同一份内部状态**浅拷贝成两份**:栈上 `local` 析构一次、TilingData 析构时再析构一次,glibc tcache 立即报 `double free`。即便字段看起来纯 POD,多一次拷贝也可能破坏 SDK 未明示的内部不变量,行为未定义。

**正确做法**:把 `tiling.fieldName` **直接作为 OUT 引用参数**交给 SDK fill API,全链路只有**一份**:

```cpp
// ✅ 正确:直接把嵌套槽位作为 OUT 传给 SDK fill API
RmsNormTilingData tiling;
tiling.set_totalRows(BS);
// ... 其余标量字段走 set_xxx ...

if (!AscendC::GetRmsNormTilingInfo(srcShape, originSrcShape, stackBufferSize,
                                    rmsNormTypeSize,
                                    tiling.rmsNormTiling,   // ← 直接把嵌套槽位作为 OUT 引用
                                    isBasicBlock)) {
    return ge::GRAPH_FAILED;
}

tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
```

kernel 侧读法不变:

```cpp
GET_TILING_DATA(tilingData, tiling);
AscendC::RmsNorm<T, false>(..., tilingData.rmsNormTiling, ...);  // 直接成员访问,原样传给 kernel API
```

**自查步骤**:

- grep `op_host/<op>_tiling.cpp` 中是否存在 `(RmsNormTiling|LayerNormTiling|SoftmaxTiling|DeepNormTiling|MatmulTiling)\s+\w+\s*=\s*\{\}` 这类**裸 SDK tiling 局部变量**?若有 → **立即删除 local,把 SDK fill API 实参改为 `tiling.<fieldName>`**。
- grep 同文件 `tiling\.\w+Tiling\s*=` 或 `tiling\.set_\w+Tiling\s*\(` ?两者都错。
- 运行时一旦出现 `double free` / `malloc_consolidate`,**第一嫌疑**就是此处多了 SDK tiling 拷贝。

**根因反思**:这一题曾走了两道弯路——先以"直接赋值"写入,报错;被指出后改"local + `set_`",**仍然崩**。真正的正解是 `TILING_DATA_FIELD_DEF_STRUCT` 预留的是 TilingData 内部 buffer 上的**最终存储位**;SDK fill API 本就要求 `XxxTiling&` 作 OUT 参数,**把槽位本身交给 SDK,中间不落任何 local**,才是设计意图。这种模式也通用于所有"SDK fill API(OUT 引用)+ 框架可寻址槽位"组合:同样适用于 `GetLayerNormTilingInfo` / `GetSoftmaxTilingInfo` / `GetDeepNormTilingInfo` / `MatmulApiTiling::GetTiling` 等。

**涉及规则**:[R-009](rules.md#r-009sdk-官方-tiling-结构用-tiling_data_field_def_struct-直接嵌入), [R-011](rules.md#r-011host-侧字段写入走-setter)

---

## Case 3:`TILING_KEY_IS(...)` 用数字字面量,不用 `constexpr`

**性质声明**:这**不是**"C++ 语义必然失败"的规则,而是**经验性最佳实践**,综合了用户在 CANN 9.0.0-beta.1 下真实踩过的报错 + CANN 官方公开样例一致惯例。

**错误现象**(CANN 9.0.0-beta.1 + 某预编译配置下):

```
Var: SCH_MODE_0 in TILING_KEY_IS(SCH_MODE_0) can not be processed as numeric
variables in the precompilation phase. please use numeric constants or macros.
```

**触发条件**:kernel 侧按"常规 C++ 命名习惯"写:

```cpp
// ⚠️ 某些 CANN 版本/配置下触发上述报错
constexpr uint32_t SCH_MODE_0 = 0;
constexpr uint32_t SCH_MODE_1 = 1;

if (TILING_KEY_IS(SCH_MODE_0)) { ... }
else if (TILING_KEY_IS(SCH_MODE_1)) { ... }
```

**成因(推测)**:`TILING_KEY_IS` 在 AscendC 编译器中参与 TilingKey 的预编译期**分支裁剪**(类似 `#if` 但基于运行期注入的 key),内部实现用 `__builtin_constant_p(key)` 判断实参是否字面常量。`__builtin_constant_p` 对 `constexpr` 变量的识别**在不同编译器版本 / 优化级别下不一致**——`-O0` 或某些场景下会返回 `false`,触发上述报错。

**推荐做法**:kernel 侧 `TILING_KEY_IS(...)` 写数字字面量,语义用**注释表**维护:

```cpp
// ✅ 推荐写法
// TilingKey 语义映射(与 op_host/<op>_tiling.cpp::SetTilingKey 完全一致)
//   0 : x=FP32, gamma=FP32
//   1 : x=FP16, gamma=FP16
//   2 : x=FP16, gamma=FP32
//   3 : x=BF16, gamma=BF16
//   4 : x=BF16, gamma=FP32
if (TILING_KEY_IS(0))      { /* FP32/FP32 */ }
else if (TILING_KEY_IS(1)) { /* FP16/FP16 */ }
else if (TILING_KEY_IS(2)) { /* FP16/FP32 */ }
else if (TILING_KEY_IS(3)) { /* BF16/BF16 */ }
else if (TILING_KEY_IS(4)) { /* BF16/FP32 */ }
```

**host 侧**(`op_host/<op>_tiling.cpp`)**可以**用 `constexpr uint64_t XXX_SCH_MODE_X = N;` 作为 `context->SetTilingKey(...)` 的实参 —— host 端是普通 CPU 编译,没有这个限制。

**环境判断**:若你的环境下 `constexpr` 能编过(部分 CANN 版本 + 编译选项组合没问题),保留也 OK 不影响功能;迁到报错的环境就降级成字面量,不是"必须强改"。

**自查 / 降级**:编译报 `... can not be processed as numeric variables in the precompilation phase` → 把该 `TILING_KEY_IS(SYMBOL)` 改为 `TILING_KEY_IS(N)`。

**涉及规则**:[R-020](rules.md#r-020tiling_key_is-用数字字面量), [R-014](rules.md#r-014schmode--tilingkey-映射表保持在一处)

---

## Case 4:`InferShape` 与 `Tiling` 混用 Shape API——`GetStorageShape` / `SetOutputShape` 不存在

**错误现象**:`op_host/<op>_infershape.cpp` 编译失败:

```
error: 'const struct gert::Shape' has no member named 'GetStorageShape'
error: 'class gert::InferShapeContext' has no member named 'SetOutputShape'
```

**错误来源**:把 TilingFunc(`gert::TilingContext`)里能用的 Shape API 原样搬到 InferShape(`gert::InferShapeContext`)里。两个上下文**不是同一类 API**。

**两类上下文对照**:

| 上下文 | `GetInputShape(i)` 返回 | 取 `Shape` 的方式 | 写入输出 Shape |
| ---- | ---- | ---- | ---- |
| `gert::InferShapeContext` | **`const gert::Shape*`**(直接就是 Shape) | 直接解引用 | `GetOutputShape(i)->SetDimNum(n); SetDim(i,v);` 或 `AppendDim(v);`(**无** `SetOutputShape(i, shape)`) |
| `gert::TilingContext` | **`const gert::StorageShape*`** | 再调 `->GetStorageShape()` 得到 `gert::Shape` | 不在 tiling 阶段设置输出 shape |

**InferShape 模板**:

```cpp
#include "register/op_impl_registry.h"
namespace ops {
using namespace ge;

static ge::graphStatus InferShape<Op>(gert::InferShapeContext* context) {
    const gert::Shape* xShape = context->GetInputShape(0);   // const Shape*,非 StorageShape
    gert::Shape* yShape       = context->GetOutputShape(0);  // Shape* (可写)
    if (xShape == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }

    const size_t xDim = xShape->GetDimNum();
    yShape->SetDimNum(xDim);
    for (size_t i = 0; i < xDim; ++i) {
        yShape->SetDim(i, xShape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}
```

**TilingFunc 对应写法**:

```cpp
auto xSh = context->GetInputShape(0);              // const StorageShape*
if (xSh == nullptr) return ge::GRAPH_FAILED;
const gert::Shape xShape = xSh->GetStorageShape(); // 取出 Shape,用于逐维读取
```

**自查**:

- grep `op_host/<op>_infershape.cpp` 出现 `GetStorageShape` / `SetOutputShape` → 按 InferShape 模板改写。
- grep `op_host/<op>_tiling.cpp`:`GetInputShape(...)` 后是否立即 `->GetStorageShape()`?缺失补上。

**根因反思**:CANN 把"图 Infer 阶段"(只有逻辑 Shape)和"Tiling 阶段"(需访问带存储布局的 StorageShape)设计成**两套独立 Context 接口**。这是设计约定,不是遗漏。迁移时按**上下文所属阶段**选 API,而不是按"名字看起来对"就用。

**涉及规则**:[R-038](rules.md#r-038gertinfershapecontext-专用-api), [R-039](rules.md#r-039gerttilingcontext-专用-api与-r-038-区别)

---

## Case 5:kernel 侧 BF16 类型名——统一用 `bfloat16_t`

**错误现象**(在某些 CANN 版本 / include 组合下):

```
error: unknown type name 'bfloat16'; did you mean 'bfloat16_t'?
```

同一份用 `bfloat16` 写的代码,CANN A 版本能编过,B 版本就炸。

**错误来源**:agent 按"常识"把 host 端的**数据类型语义名**直接当 device 端 C++ 类型用:

```cpp
// ⚠️ 跨版本不稳定
AscendC::GlobalTensor<bfloat16> xBf16Gm;
pipe_.InitBuffer(inQueueBf16, 1, len * sizeof(bfloat16));
AscendC::LocalTensor<bfloat16> xBf16 = inQueueBf16.AllocTensor<bfloat16>();
```

**根因**:CANN 里 BF16 有**两套命名**:

| 场景 | 正确名字 | 说明 |
| ---- | ---- | ---- |
| host 端 OpDef.DataType | `ge::DT_BF16` | 枚举,所有版本稳定 |
| host 端 JSON / msopgen 原型 | `"bfloat16"` | 语义字符串,所有版本稳定 |
| host 端 Printf / 日志 | `"bfloat16"` | 同上 |
| **device 端** C++ 类型 | **`bfloat16_t`** | **所有** CANN 版本稳定 |
| device 端 ~~`bfloat16` 裸名~~ | ⚠️ **不稳定** | 仅在部分版本作为 `using bfloat16 = bfloat16_t;` 别名存在,取决于版本+include+宏;新版本有直接移除这个顶层别名的趋势 |

**其它 dtype 命名对照**:

| 语义名(host / JSON) | device C++ 类型(kernel) |
| ---- | ---- |
| `"float"` | `float` |
| `"float16"` | `half` |
| `"bfloat16"` | **`bfloat16_t`** |
| `"int8"` / `"int16"` / `"int32"` / `"int64"` | `int8_t` / `int16_t` / `int32_t` / `int64_t`(标准 `<cstdint>`) |
| `"uint8"` ~ `"uint64"` | `uint8_t` ~ `uint64_t` |
| `"bool"` | `bool` |
| `"double"` | `double`(NPU 少用) |

**正确做法**:

```cpp
// ✅ 跨版本稳定
AscendC::GlobalTensor<bfloat16_t> xBf16Gm;
pipe_.InitBuffer(inQueueBf16, 1, len * sizeof(bfloat16_t));
AscendC::LocalTensor<bfloat16_t> xBf16 = inQueueBf16.AllocTensor<bfloat16_t>();
xBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t*)x);
```

**自查**:`rg "\bbfloat16\b" op_kernel/ | rg -v "bfloat16_t"` 命中即为隐患。**注意**:**不要**误改 `op_host/` 下的字符串字面量 `"bfloat16"` 与注释。

**根因反思**:CANN 数据类型系统有两层——"**语义层**"(给 host / 调度框架看)和"**实现层**"(给 device C++ 编译器看)。两层命名刻意不一样,是为了让 host 不必 include device-only 的 C++ 类型定义。agent 迁移 `.asc` 时如果看到 `bfloat16` 就照抄,**恰好** `.asc` 作者那个 CANN 版本有这个顶层别名——就会掩盖问题;到另一个版本就炸。迁移时**主动**把 kernel 端的 `bfloat16` 裸名全部升级为 `bfloat16_t`,而不是"先照抄再等报错"。

**涉及规则**:[R-023](rules.md#r-023device-端-c-数据类型用实现名)

---

## Case 6:`op_host/<op>_def.cpp` 空壳——最隐蔽、最常见的"偷懒型"失败

**错误现象**(表现分三档,危险度递增):

- **档一**(最明显):`class <Op> : public OpDef { ... }` 的类体里**只有**构造函数,`Input` / `Output` / `Attr` / `AICore().AddConfig` 全部缺失
- **档二**(半空):有 `.Input("x")` / `.Output("y")`,但 `.DataType({})` 是空大括号或只有占位符 `<DType1>`;`AICore().AddConfig("<soc>", cfg)` 的 `<soc>` 没被替换
- **档三**(最隐蔽):语法上完全合法,但 DataType 数组长度 ≠ schMode 数 / 列顺序和 `<op>.json` 不一致 / 某个 Input 的 DataType 数组为空

**下游症状**(档三最坑——编译能过、运行才炸):

- 档一 / 档二:编译期 `opbuild` 报
  - `The input/output/attr of op <Op> is not configured`
  - `Failed to get op definition of <Op>`
  - `op proto of <Op> is empty`
- 档三:编译通过,`custom_opp_*.run` 安装成功,运行时 aclnn 调用才报
  - `<Op>NotRegistered`
  - `Op<Op>:GetOpInfoFailed` / `op select failed`
  - 或 silently 走错 schMode 导致数值不对

**错误来源**:

1. `.asc` 是 kernel 直调模式,**本身不含 OpDef 信息**,agent 不知道去哪抽参数,把 reference 的占位符骨架原封不动交付
2. 上下文窗口紧张时,agent 把 OpDef 写一半就去写下一个文件
3. 契约表没提前落纸,agent 边想边写、想不全就省略

**正确三步法**:

### Step 1:填完 OpDef 契约表(见 [R-027](rules.md#r-027开工前必填-opdef-契约表))

RmsNorm 示例(5 种合法 dtype 组合):

```
Inputs:
  [0] name=x, param_type=REQUIRED
      DataType = {DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_BF16,  DT_BF16}
      Format   = {FORMAT_ND, FORMAT_ND, FORMAT_ND, FORMAT_ND, FORMAT_ND}
  [1] name=gamma, param_type=REQUIRED
      DataType = {DT_FLOAT, DT_FLOAT16, DT_FLOAT,   DT_BF16,  DT_FLOAT}
      Format   = 同上
Outputs:
  [0] name=y,    DataType = {DT_FLOAT, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_BF16}
  [1] name=rstd, DataType = {DT_FLOAT, DT_FLOAT,   DT_FLOAT,   DT_FLOAT, DT_FLOAT}
Attrs:
  [0] name=epsilon, type=Float, param_type=OPTIONAL, default=1e-6f
SOC: ascend910b(系列名,非具体型号)
schMode 数: 5
```

### Step 2:按契约表写 `op_host/<op>_def.cpp`,逐项对照九点自检([R-028](rules.md#r-028opdef-完工九点自检))

### Step 3:grep 确认

```bash
rg "\.Input\(" op_host/<op>_def.cpp            # 每个 Input 一行
rg "\.Output\(" op_host/<op>_def.cpp           # 每个 Output 一行
rg "\.DataType\(\s*\{\s*\}\s*\)" op_host/<op>_def.cpp   # 必须 0 (禁止空数组)
rg "AICore\(\)\.AddConfig\(\"ascend" op_host/<op>_def.cpp   # 必须 ≥ 1
rg "<DType[0-9]+>|<Format[0-9]+>|<soc>|TODO:" op_host/<op>_def.cpp  # 必须 0 (禁止占位符残留)
rg "OP_ADD\(" op_host/<op>_def.cpp             # 必须 == 1
```

任一条不通过 → 回 Step 1 重做。

**错误对照**(同一个 RmsNorm 算子):

```cpp
// ❌ 档一:空壳
class RmsNorm : public OpDef {
public:
    explicit RmsNorm(const char* name) : OpDef(name) {
        // TODO: 填写输入输出
    }
};
OP_ADD(RmsNorm);

// ❌ 档二:占位符没替换
class RmsNorm : public OpDef {
public:
    explicit RmsNorm(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED)
            .DataType({/*<DType1>*/}).Format({/*<Format1>*/});
        this->AICore().AddConfig("<soc>", aicConfig);
    }
};

// ✅ 正确(完整正确示例见 rms_norm_single_ops/rms_norm/op_host/rms_norm_def.cpp)
class RmsNorm : public OpDef {
public:
    explicit RmsNorm(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // ... gamma / y / rstd 同构,不省略 ...
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6f);

        OpAICoreConfig aicConfig;
        aicConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "rms_norm");
        this->AICore().AddConfig("ascend910b", aicConfig);  // 系列名,四处一致
    }
};
OP_ADD(RmsNorm);
```

**根因反思**:OpDef 空壳问题**只能靠前置约束**消除,不能靠后置编译报错兜底——档三(长度不对齐/列顺序错)编译期根本不报,只在 aclnn 调用时炸。agent 在写 `<op>_def.cpp` 之前如果没完成契约表,就永远有这个风险。契约表是**一次性交付的前提**,不是"之后再补"的可选项。

**涉及规则**:[R-003](rules.md#r-003算子名一处大驼峰全局对齐), [R-027](rules.md#r-027开工前必填-opdef-契约表), [R-028](rules.md#r-028opdef-完工九点自检), [R-029](rules.md#r-029禁止任何占位符残留), [R-030](rules.md#r-030opdef-在-ops-命名空间作用域内可解析), [R-031](rules.md#r-031format-用-geformat_nd), [R-032](rules.md#r-032datatype--format-数组长度--schmode-数), [R-033](rules.md#r-033attr-必须有具体-default_value), [R-034](rules.md#r-034soc-用系列名三处一致), [R-035](rules.md#r-035opaicoreconfig-六-flag--extendcfginfo), [R-036](rules.md#r-036op_addop-末尾一行)

---

## 追加案例指引

新发现的失败模式加在本文件末尾,按 Case N 顺序递增。每个案例建议按本文件模板的四段结构:

1. **错误现象**:具体报错日志(越原汁原味越好)
2. **错误来源 / 根因**:为什么 agent 会写出这段错的代码
3. **正确做法**:修正后的完整代码或检查方法
4. **涉及规则**:反向引用 [rules.md](rules.md) 中对应编号

**不要**在案例里重写规则本身——规则永远只在 rules.md 维护。
