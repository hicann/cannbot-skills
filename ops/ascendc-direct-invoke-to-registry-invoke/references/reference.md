# 文件模板参考(Reference Templates)

本文件为每个目标文件提供**带占位符的骨架模板**。所有硬约束 / 规则只走 `[R-xxx]` 引用到 [rules.md](rules.md),本文不再复述,以免多处陈述漂移。

**占位符约定**:

- `<Op>`:大驼峰(如 `RmsNorm`)
- `<op>`:snake_case(如 `rms_norm`)
- `<soc>`:SoC 系列名(如 `ascend910b`),全小写 + 无尾部型号数字 — 见 [R-034](rules.md#r-034soc-用系列名三处一致)
- `<DType1>, <DType2>, ...`:具体 `ge::DT_*` 值,**禁止**原样保留占位符交付 — 见 [R-029](rules.md#r-029禁止任何占位符残留)

所有 C/C++ 文件保留 10 行 Huawei 版权头。

---

## `<op>.json`(msOpGen 原型定义)

**作用**:喂给 `msopgen gen -i <op>.json` 工具,生成标准算子工程骨架。

**规则**:[R-003](rules.md#r-003算子名一处大驼峰全局对齐), [R-043](rules.md#r-043opjson-顶层数组--op-大驼峰), [R-044](rules.md#r-044json-中-typeformat-数组长度统一), [R-045](rules.md#r-045json-用-msopgen-语义名), [R-046](rules.md#r-046json-与-opdef-逐列严格一致)

**合法 `type` 值**:`float` / `float16` / `bfloat16` / `int8` / `uint8` / `int16` / `int32` / `int64` / `uint32` / `uint64` / `bool` / `double`(不带 `DT_` 前缀)

**合法 `format` 值**:`ND` / `NCHW` / `NHWC` / `NC1HWC0` / `FRACTAL_NZ` 等

**合法 `attr.type`**:`int` / `float` / `bool` / `str` / `list_int` / `list_float` / `list_bool` / `list_str`

**骨架**(按 schMode 数 N 展开 `type` / `format` 数组,展开到**具体字面量**,不要留占位):

```json
[
    {
        "op": "<Op>",
        "input_desc": [
            {
                "name": "<input1>",
                "param_type": "required",
                "format": ["ND", "ND", "ND"],
                "type": ["<dtype_1a>", "<dtype_2a>", "<dtype_3a>"]
            }
        ],
        "output_desc": [
            {
                "name": "<output1>",
                "param_type": "required",
                "format": ["ND", "ND", "ND"],
                "type": ["<dtype_1a>", "<dtype_2a>", "<dtype_3a>"]
            }
        ],
        "attr": [
            {
                "name": "<attr_name>",
                "param_type": "optional",
                "type": "float",
                "default_value": 1e-6
            }
        ]
    }
]
```

---

## `op_host/<op>_tiling.h`

**作用**:用 `BEGIN_TILING_DATA_DEF` 定义 TilingData,通过 `REGISTER_TILING_DATA_CLASS` 与算子关联。**TilingData 唯一定义处**([R-005](rules.md#r-005tilingdata-唯一定义于-hostkernel-侧不得重复声明))。

**规则**:[R-007](rules.md#r-007begin_tiling_data_def-范式), [R-008](rules.md#r-008字段类型约束), [R-009](rules.md#r-009sdk-官方-tiling-结构用-tiling_data_field_def_struct-直接嵌入), [R-010](rules.md#r-010register_tiling_data_class-注册)

```cpp
#ifndef <OP>_TILING_H
#define <OP>_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"   // 当使用 TILING_DATA_FIELD_DEF_STRUCT(SDKTiling,...) 时必需 [R-015]

namespace optiling {
BEGIN_TILING_DATA_DEF(<Op>TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalRows);
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
    TILING_DATA_FIELD_DEF(float,    epsilon);
    // SDK 官方 tiling 结构用 _STRUCT,不扁平化 [R-009]
    TILING_DATA_FIELD_DEF_STRUCT(<SDKTiling>, <sdkFieldName>);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(<Op>, <Op>TilingData)
}
#endif
```

---

## `op_kernel/<op>.h`

**作用**:kernel 类(`Init` / `Process` / `CopyIn` / `Compute` / `CopyOut`)声明与实现。

**规则**:[R-021](rules.md#r-021device-端禁用-stl)(禁 STL), [R-022](rules.md#r-022device-端内建函数加-ascendc-前缀)(AscendC 前缀), [R-023](rules.md#r-023device-端-c-数据类型用实现名)(`bfloat16_t` / `half`), [R-024](rules.md#r-024init-模板化接收-tilingdata)(Init 模板化), [R-025](rules.md#r-025localtensor-32-字节对齐)(LocalTensor 32 字节对齐), [R-026](rules.md#r-026不强行删改-kernel-内-sdk-高阶-api-调用)(SDK 高阶 API 保真)

**device 端 C++ dtype 对照**(kernel 必须用下表右列;左列是 host 字符串):

| CANN 语义(host 字符串 / JSON) | device C++ 类型(kernel) |
| ---- | ---- |
| `"float"` | `float` |
| `"float16"` | `half` |
| `"bfloat16"` | **`bfloat16_t`**(**不是** `bfloat16`) |
| `"int8"` ~ `"int64"` / `"uint8"` ~ `"uint64"` | `int8_t` ~ `int64_t` / `uint8_t` ~ `uint64_t`(`<cstdint>`) |
| `"bool"` | `bool`(1 字节) |

```cpp
#ifndef <OP>_KERNEL_H
#define <OP>_KERNEL_H
#include "kernel_operator.h"
// 可选:#include "lib/..." / "adv_api/..."(device 端 kernel 头可保留;host 端禁用,见 [R-015])

namespace Ns<Op> {

template <typename Tx, typename Tgamma>
class Kernel<Op> {
public:
    template <class TD>
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd,
        const TD& td, uint32_t startRow, uint32_t endRow)
    {
        // 直接按字段名访问 GET_TILING_DATA 生成的结构体 [R-019]
        totalRows_ = td.totalRows;
        epsilon_   = td.epsilon;
        // 保存嵌入的 SDK tiling 字段,透传给 SDK 高阶 API [R-009] / [R-026]
        sdkTiling_ = td.<sdkFieldName>;
        // ... UB buffer 布局严格照抄 .asc [Kernel 保真原则]
    }

    __aicore__ inline void Process() {
        // ... 逐行保留 .asc 原实现,SDK 高阶 API 原样调用 [R-026]
        // AscendC::RmsNorm<Tx, false>(yLocal, xLocal, gammaLocal, rstdLocal,
        //                              workLocal, sdkTiling_);
    }

private:
    <SDKTiling> sdkTiling_;
    uint32_t totalRows_;
    float    epsilon_;
    // TPipe / TQue / TBuf / LocalTensor / GlobalTensor 声明严格照抄 .asc
};

} // namespace Ns<Op>
#endif
```

---

## `op_kernel/<op>.cpp`

**作用**:kernel 入口 + TilingKey 分发。

**规则**:[R-004](rules.md#r-004kernel-函数名-snake_case与-opfilevalue-一致), [R-014](rules.md#r-014schmode--tilingkey-映射表保持在一处), [R-018](rules.md#r-018kernel-入口签名固定), [R-019](rules.md#r-019get_tiling_data-第一行), [R-020](rules.md#r-020tiling_key_is-用数字字面量), [R-021](rules.md#r-021device-端禁用-stl), [R-022](rules.md#r-022device-端内建函数加-ascendc-前缀), [R-023](rules.md#r-023device-端-c-数据类型用实现名)

```cpp
#include "<op>.h"

// TilingKey 语义映射(与 op_host/<op>_tiling.cpp::SetTilingKey 完全一致) [R-014]
//   0 : x=FP32, gamma=FP32
//   1 : x=FP16, gamma=FP16
//   2 : x=FP16, gamma=FP32
//   3 : x=BF16, gamma=BF16   ← kernel 用 bfloat16_t [R-023]
//   4 : x=BF16, gamma=FP32

extern "C" __global__ __aicore__ void <op>(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);   // [R-019]

    // [R-020] 字面量,不用 constexpr 命名常量
    if (TILING_KEY_IS(0)) {
        Ns<Op>::Kernel<Op><float, float> op;
        op.Init(x, gamma, y, rstd, tilingData, 0, tilingData.totalRows);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        Ns<Op>::Kernel<Op><half, half> op;
        op.Init(x, gamma, y, rstd, tilingData, 0, tilingData.totalRows);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        Ns<Op>::Kernel<Op><bfloat16_t, bfloat16_t> op;    // ← 实现名 [R-023]
        op.Init(x, gamma, y, rstd, tilingData, 0, tilingData.totalRows);
        op.Process();
    }
    // ... 其余 schMode
}
```

---

## `op_host/<op>_def.cpp`

**作用**:算子元信息注册(输入/输出/属性/dtype/format/硬件支持)。

> ⚠️ **本文件是本 skill 下最容易被 agent 偷懒写成"空壳"的文件**(三档症状见 [Case 6](cases.md#case-6op_hostop_defcpp-空壳最隐蔽最常见的偷懒型失败))。交付前**必须**跑 [R-028](rules.md#r-028opdef-完工九点自检) 的九点自检。

**规则**(整组必读):[R-003](rules.md#r-003算子名一处大驼峰全局对齐), [R-004](rules.md#r-004kernel-函数名-snake_case与-opfilevalue-一致), [R-027](rules.md#r-027开工前必填-opdef-契约表) ~ [R-036](rules.md#r-036op_addop-末尾一行)

**开工前**:先落纸 [R-027](rules.md#r-027开工前必填-opdef-契约表) 的契约表(每个 Input/Output 的**完整** DataType/Format 数组、每个 Attr 的具体 default_value、SOC 系列名、schMode 数 N)。

**下面模板带占位符,禁止原样交付**:把 `<Op>` / `<op>` / `<DTypeN>` / `<soc>` 全部替换成契约表里的具体值;数组元素**根据 schMode 数 N 展开**(例如 5 种组合就写 5 个元素)。

```cpp
#include "register/op_def_registry.h"

namespace ops {   // ← 包在 ops 命名空间里,避免 'OpDef' has not been declared [R-030]
class <Op> : public OpDef {
public:
    explicit <Op>(const char* name) : OpDef(name) {
        this->Input("<input1>")
            .ParamType(REQUIRED)
            .DataType({ge::<DType1>, ge::<DType2>, /* ... 展开到 N 个 */})     // [R-032]
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, /* ... 展开到 N 个 */})     // [R-031] FORMAT_ND 不是 Format::ND
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, /* ... N 个 */})
            .AutoContiguous();
        // ... 其余 Input 同构展开,不得省略

        this->Output("<output1>")
            .ParamType(REQUIRED)
            .DataType({ge::<DType1>, ge::<DType2>, /* ... N 个 */})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, /* ... N 个 */})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, /* ... N 个 */});
        // ... 其余 Output 同构展开

        this->Attr("<attr_name>").AttrType(OPTIONAL).Float(1e-6f);   // [R-033] default 必须具体字面量

        OpAICoreConfig cfg;      // [R-035] 六 flag + ExtendCfgInfo 全部必填
        cfg.DynamicCompileStaticFlag(true)
           .DynamicFormatFlag(false)
           .DynamicRankSupportFlag(true)
           .DynamicShapeSupportFlag(true)
           .NeedCheckSupportFlag(false)
           .PrecisionReduceFlag(true)
           .ExtendCfgInfo("opFile.value", "<op>");   // [R-004] snake_case,等于 kernel 函数名
        this->AICore().AddConfig("<soc>", cfg);      // [R-034] 系列名(ascend910b 不是 ascend910b4),四处一致
    }
};
OP_ADD(<Op>);        // [R-036]
} // namespace ops
```

**完工自检 grep**(见 [R-028](rules.md#r-028opdef-完工九点自检)):

```bash
rg "\.Input\(" op_host/<op>_def.cpp              # ≥ 1
rg "\.Output\(" op_host/<op>_def.cpp             # ≥ 1
rg "\.DataType\(\s*\{\s*\}\s*\)" op_host/<op>_def.cpp                  # == 0 (禁空数组)
rg "<DType[0-9]+>|<Format[0-9]+>|<soc>|TODO:" op_host/<op>_def.cpp     # == 0 (禁占位符)
rg 'AICore\(\)\.AddConfig\("ascend' op_host/<op>_def.cpp               # ≥ 1 (字符串字面量)
rg "OP_ADD\(" op_host/<op>_def.cpp               # == 1
```

任一不通过 → 按契约表**整文件重写**,不做局部打补丁。

---

## `op_host/<op>_infershape.cpp`

**作用**:InferShape + InferDataType(运行时路径)。

**规则**:[R-003](rules.md#r-003算子名一处大驼峰全局对齐), [R-037](rules.md#r-037infershape--inferdatatype-注册), [R-038](rules.md#r-038gertinfershapecontext-专用-api)

> ⚠️ **InferShape API 与 TilingFunc 不一样**(见 [Case 4](cases.md#case-4infershape-与-tiling-混用-shape-apigetstorageshape--setoutputshape-不存在)):
> - InferShape 的 `GetInputShape(i)` 返 `const gert::Shape*`(直接就是 Shape,**没有** `GetStorageShape()`)
> - 输出用 `GetOutputShape(i)->SetDimNum(n)` + `SetDim(i, v)` / `AppendDim(v)`,**没有** `SetOutputShape(i, ...)`

```cpp
#include "register/op_impl_registry.h"

namespace ops {
using namespace ge;

static ge::graphStatus InferShape<Op>(gert::InferShapeContext* context) {
    const gert::Shape* xShape = context->GetInputShape(0);   // const Shape* [R-038]
    gert::Shape* yShape       = context->GetOutputShape(0);  // Shape* 可写
    if (xShape == nullptr || yShape == nullptr) return GRAPH_FAILED;

    const size_t xDim = xShape->GetDimNum();
    yShape->SetDimNum(xDim);
    for (size_t i = 0; i < xDim; ++i) {
        yShape->SetDim(i, xShape->GetDim(i));
    }
    // ... 其余输出同理
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType<Op>(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    // ... 其余输出
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(<Op>).InferShape(InferShape<Op>).InferDataType(InferDataType<Op>);
} // namespace ops
```

---

## `op_host/<op>_tiling.cpp`

**作用**:`TilingFunc` 实现 + `IMPL_OP_OPTILING` 注册。

**规则**:[R-009](rules.md#r-009sdk-官方-tiling-结构用-tiling_data_field_def_struct-直接嵌入)(SDK 嵌入字段直接传引用), [R-011](rules.md#r-011host-侧字段写入走-setter), [R-012](rules.md#r-012tilingfunc-五板斧), [R-013](rules.md#r-013tilingfunc-注册), [R-014](rules.md#r-014schmode--tilingkey-映射表保持在一处), [R-015](rules.md#r-015host-侧-ascendc-tiling-api-只能来自-tilingtiling_apih)(`tiling/tiling_api.h` 唯一入口), [R-016](rules.md#r-016host-必需头三件套), [R-039](rules.md#r-039gerttilingcontext-专用-api与-r-038-区别)

```cpp
#include "<op>_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"           // [R-015] host 侧 AscendC tiling API 唯一入口
#include "graph/utils/type_utils.h"      // 如需 dtype utils

namespace optiling {

// schMode 映射(host 侧可用 constexpr,仅 kernel 端 TILING_KEY_IS 禁用 [R-020])
constexpr uint64_t SCH_FP32_FP32 = 0;
constexpr uint64_t SCH_FP16_FP16 = 1;
// ...

static inline uint64_t MapDtypeToSchMode(ge::DataType xDt, ge::DataType gDt) {
    if (xDt == ge::DT_FLOAT   && gDt == ge::DT_FLOAT)   return SCH_FP32_FP32;
    if (xDt == ge::DT_FLOAT16 && gDt == ge::DT_FLOAT16) return SCH_FP16_FP16;
    // ... 非法组合 return UINT64_MAX
    return UINT64_MAX;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t coreNum = platform.GetCoreNumAiv();

    // [R-039] TilingContext 这边 GetInputShape 返 const StorageShape*,要 GetStorageShape()
    const auto xShape = context->GetInputShape(0)->GetStorageShape();
    auto xDtype = context->GetInputDesc(0)->GetDataType();
    auto gDtype = context->GetInputDesc(1)->GetDataType();
    const float* eps = context->GetAttrs()->GetAttrPointer<float>(0);

    uint64_t schMode = MapDtypeToSchMode(xDtype, gDtype);
    if (schMode == UINT64_MAX) return ge::GRAPH_FAILED;

    <Op>TilingData tiling;
    tiling.set_totalRows(xShape.GetDim(0));         // [R-011] 走 setter
    tiling.set_coreNum(coreNum);
    tiling.set_epsilon(*eps);

    // [R-009] SDK 嵌入字段:直接把 tiling.fieldName 作 OUT 引用传给 SDK fill API
    // ❌ 禁止 "XxxTiling local = {}; GetXxx(local); tiling.set_field(local)"
    AscendC::GetRmsNormTilingInfo(srcShape, originSrcShape, stackBufferSize,
                                   rmsNormTypeSize,
                                   tiling.<sdkFieldName>,   // ← OUT 引用
                                   isBasicBlock);

    // [R-012] 五板斧
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(coreNum);
    context->SetTilingKey(schMode);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(<Op>).Tiling(TilingFunc);   // [R-013]
} // namespace optiling
```

---

## `op_host/config/<soc>/<op>_binary.json`

**作用**:每个 dtype 组合一条 bin 编译配置。

**规则**:[R-041](rules.md#r-041op_binaryjson-条目数--schmode-数)

模板省略,msOpGen 会生成初始版本,按 schMode 数扩展即可。

---

## `op_host/config/<soc>/<op>_simplified_key.ini`

**规则**:[R-042](rules.md#r-042op_simplified_keyini-与-bin-对齐)

```ini
[<op>_binary_0]
tiling_key=0
[<op>_binary_1]
tiling_key=1
; ...
```

---

## `op_graph/<op>_proto.h`

**作用**:图模式(可选)使用的 REG_OP 原型。

```cpp
#ifndef <OP>_PROTO_H
#define <OP>_PROTO_H
#include "graph/operator_reg.h"

namespace ge {
REG_OP(<Op>)
    .INPUT(<input1>, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(<input2>, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(<output1>, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(<attr>, Float, 1e-6)
    .OP_END_FACTORY_REG(<Op>)
} // namespace ge
#endif
```

---

## `op_graph/<op>_graph_infer.cpp`

**规则**:[R-040](rules.md#r-040图模式-inferdatatype-走-impl_op)(函数名避免与 `IMPL_OP_INFERSHAPE` 中的冲突,加 `Graph` 后缀或放匿名命名空间)

```cpp
#include "register/op_impl_registry.h"

namespace {
ge::graphStatus InferDataType<Op>Graph(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}

IMPL_OP(<Op>).InferDataType(InferDataType<Op>Graph);
```

---

## `example/` 阶段 7 二进制一致性验证 harness

只生成/复用阶段 7 所需的最小 aclnn harness,不扩展为泛化端到端精度测试框架 — 见 [R-051](rules.md#r-051skill-只生成阶段-7-二进制一致性验证-harness), [R-060](rules.md#r-060阶段-7-必须生成-example-并用原直调输入做-aclnn-二进制一致性验证)。

**目录固定**:

```text
<VerifyProjectDir>/example/
├── CMakeLists.txt
├── main.cpp
├── input/          # 从原 kernel 直调工程复制
├── output/         # 从原 kernel 直调工程复制,作为二进制比较基准
└── aclnn_output/   # opapi_test 生成,不得覆盖 output/
```

**数据流固定**:`example/input/*.bin` → `aclnn<Op>` → `example/aclnn_output/*.bin` → 与 `example/output/*.bin` 做 `cmp -s` / SHA256 byte-level compare。

### `example/main.cpp` 关键内容

从参考工程 `rms_norm_verify_project_20260425_181328/example/main.cpp` 提取这些通用片段:

- `ReadFile` / `WriteFile`:二进制读写 input/output
- `GetShapeSize`:按 shape 计算元素个数
- `Init`:固定 `aclInit` / `aclrtSetDevice` / `aclrtCreateStream`
- `CreateAclTensor`:host vector → device malloc/memcpy → `aclCreateTensor`
- aclnn 两段式调用:`aclnn<Op>GetWorkspaceSize(...)` → workspace malloc → `aclnn<Op>(...)`
- `aclrtSynchronizeStream` 后把每个 output 从 device 拷回 host,写入 `example/aclnn_output/`
- 释放 `aclTensor`、device memory、workspace、stream、device、ACL

接口名按算子大驼峰拼接:

```cpp
// <Op> = RmsNorm
#include "aclnn_rms_norm.h"

uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
auto ret = aclnnRmsNormGetWorkspaceSize(/* inputs */, /* attrs */, /* outputs */, &workspaceSize, &executor);
// malloc workspace if workspaceSize > 0
ret = aclnnRmsNorm(workspaceAddr, workspaceSize, executor, stream);
```

输入/输出 tensor、attr 顺序必须来自阶段 1 的 OpDef 契约表和原直调 `.asc` / `main`。输入输出 shape、dtype、epsilon 等参数优先从原直调 `scripts/gen_data.py` / `params.txt` / `run.sh` 抽取。例如 RmsNorm 的 `gen_data.py` 给出 `rows`、`cols`、`epsilon`、`dtype`、`input_x.bin`、`input_gamma.bin`、`golden_y.bin`、`golden_rstd.bin` 和对齐规则。

### `example/CMakeLists.txt`

参考 `rms_norm_verify_project_20260425_181328/example/CMakeLists.txt`:

```cmake
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

cmake_minimum_required(VERSION 3.14)

project(ACLNN_EXAMPLE)

add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./bin")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

add_executable(opapi_test
    main.cpp
)

if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/cann")
endif()

set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")

include_directories(
    ${INCLUDE_BASE_DIR}
    ${INCLUDE_BASE_DIR}/aclnn
    /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/include/
)

target_link_libraries(opapi_test PRIVATE
    ${ASCEND_PATH}/lib64/libascendcl.so
    ${ASCEND_PATH}/lib64/libnnopbase.so
    ${ASCEND_PATH}/lib64/libopapi_math.so
    ${ASCEND_PATH}/lib64/libopapi_nn.so
    /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/libcust_opapi.so
)

install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
```

### 编译运行命令

```bash
cd <VerifyProjectDir>/example
mkdir -p build
cd build
cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
make
cd bin
./opapi_test
```

---

## `CMakeLists.txt`(顶层)

由 msopgen 生成,不手写。

---

## `README.md`

必含:

- 算子名大驼峰 + 简述
- **步骤 1.2 OpDef 契约表**(打印出来供用户复核)
- SOC 系列名(遵循 [R-034](rules.md#r-034soc-用系列名三处一致))
- 构建命令:`bash build.sh` → `./build_out/custom_opp_*.run`
- 阶段 7 验证说明:原直调 input/output 路径、aclnn 输出目录、`binary_compare.log` 路径
- 原 `.asc` 的存留位置说明([R-006](rules.md#r-006保留-asc-原文件但不-include))
