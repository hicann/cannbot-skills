# 端到端示例:rms_norm.asc → CANN 标准自定义算子

本文以真实 `rms_norm.asc` 的拆解为例,演示一步到位生成可被 **CANN 标准 msOpGen 框架编译** 的自定义算子目录。

> 本文件是**示范场景**,关注"怎么做决策 / 产物长什么样"。所有硬约束不在这里重复,通过 `[R-xxx]` 引用到 [rules.md](rules.md)。失败案例走 [cases.md](cases.md)。

**本示例涉及的关键规则**([R-001](rules.md#r-001唯一目标框架--cann-标准-msopgen)(msOpGen 唯一框架) / [R-003](rules.md#r-003算子名一处大驼峰全局对齐)(算子名对齐) / [R-009](rules.md#r-009sdk-官方-tiling-结构用-tiling_data_field_def_struct-直接嵌入)(SDK tiling 嵌入) / [R-014](rules.md#r-014schmode--tilingkey-映射表保持在一处)(schMode 四处一致) / [R-020](rules.md#r-020tiling_key_is-用数字字面量)(TILING_KEY_IS 字面量) / [R-023](rules.md#r-023device-端-c-数据类型用实现名)(bfloat16_t) / [R-027](rules.md#r-027开工前必填-opdef-契约表)(OpDef 契约表) / [R-034](rules.md#r-034soc-用系列名三处一致)(ascend910b 系列名))

## 源文件关键信息

- 文件：`x_skills_report/ops/rms_norm/rms_norm.asc`
- Kernel 入口：`rms_norm_custom(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR tiling)`
- 合法 `(x, gamma)` dtype 组合（5 种）：
  - `(FP32, FP32)` / `(FP16, FP16)` / `(FP16, FP32)` / `(BF16, BF16)` / `(BF16, FP32)`
- 原 tiling 由 `RmsNormCustomTiling`（12 字段 POD）+ 末尾追加 `AscendC::RmsNormTiling`（由 `GetRmsNormTilingInfo` 填充）两段组成
- 3 个 kernel 类：`KernelRmsNormFloat` / `KernelRmsNormHalf` / `KernelRmsNormBf16`，差异大（BF16 内部 Cast 到 FP32 再 Cast 回 BF16）

## 关键决策

| 源特征 | 目标决策 |
| ---- | ---- |
| `RmsNormCustomTiling` 12 字段 | 去掉 `dtypeBranch` / `gammaIsFloat` / `xDtypeSize` / `gammaDtypeSize` 四个分发字段（迁至 TilingKey），其余 8 字段迁入 `BEGIN_TILING_DATA_DEF` |
| `RmsNormTiling` 外部 struct 嵌入 | **扁平化** —— 将 kernel 实际使用的 6 个字段（`mainBsLength / mainBshLength / tailBsLength / tailBshLength / loopRound / reciprocalOfHLength`）逐一加入 TilingData |
| 3 个 kernel 类主体差异大 | **保留 3 个类**，不合并为模板；在 `<op>.cpp` 用 5 个 `TILING_KEY_IS` 分支实例化 |
| `gammaIsFloat` 子分支 | 作为 schMode 的细分值（FP16+FP32 gamma = SCH_FP16_GAMMA_FP32；BF16+FP32 gamma = SCH_BF16_GAMMA_FP32） |
| 原 host 端 `AscendC::GetRmsNormTilingInfo` 依赖 | 为避免 CANN 版本间 host 端 tiling 库头路径差异，自行计算这 6 个字段（UB 预算 / 按行分批） |

## schMode 映射表

| schMode | x dtype | gamma dtype | Kernel 类 | gammaIsFloat 参数 |
| --- | --- | --- | --- | --- |
| 0 | FLOAT | FLOAT | KernelRmsNormFloat | - |
| 1 | FLOAT16 | FLOAT16 | KernelRmsNormHalf | 0 |
| 2 | FLOAT16 | FLOAT | KernelRmsNormHalf | 1 |
| 3 | BF16 | BF16 | KernelRmsNormBf16 | 0 |
| 4 | BF16 | FLOAT | KernelRmsNormBf16 | 1 |

## 目录骨架

```
rms_norm/
├── rms_norm.json                         # msOpGen 原型定义（步骤 3 生成，供 msopgen -i 消费）
├── CMakeLists.txt
├── CMakeLists.txt.kernel-direct-call     # 原 kernel 直调构建脚本归档
├── README.md
├── rms_norm.asc                          # 原文件保留作为参考
├── op_graph/
│   ├── rms_norm_proto.h
│   ├── rms_norm_graph_infer.cpp
│   └── fusion_pass/.gitkeep
├── op_host/
│   ├── rms_norm_tiling.h                 # BEGIN_TILING_DATA_DEF + REGISTER_TILING_DATA_CLASS
│   ├── rms_norm_tiling.cpp               # TilingFunc + IMPL_OP_OPTILING
│   ├── rms_norm_def.cpp                  # OpDef
│   ├── rms_norm_infershape.cpp           # InferShape + InferDataType
│   └── config/ascend910b/{rms_norm_binary.json, rms_norm_simplified_key.ini}
└── op_kernel/
    ├── rms_norm.cpp                      # extern "C" 入口，5 个 TILING_KEY_IS 分支
    └── rms_norm.h                        # 3 个 kernel 类，Init 模板化 TilingData 类型
```

## 原型 JSON（rms_norm.json）

```json
[
    {
        "op": "RmsNorm",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": ["ND", "ND", "ND", "ND", "ND"],
                "type":   ["float", "float16", "float16", "bfloat16", "bfloat16"]
            },
            {
                "name": "gamma",
                "param_type": "required",
                "format": ["ND", "ND", "ND", "ND", "ND"],
                "type":   ["float", "float16", "float",   "bfloat16", "float"]
            }
        ],
        "output_desc": [
            {
                "name": "y",
                "param_type": "required",
                "format": ["ND", "ND", "ND", "ND", "ND"],
                "type":   ["float", "float16", "float16", "bfloat16", "bfloat16"]
            },
            {
                "name": "rstd",
                "param_type": "required",
                "format": ["ND", "ND", "ND", "ND", "ND"],
                "type":   ["float", "float",   "float",   "float",    "float"]
            }
        ],
        "attr": [
            {
                "name": "epsilon",
                "param_type": "optional",
                "type": "float",
                "default_value": 1e-6
            }
        ]
    }
]
```

**列↔schMode 对应关系**（严格 0-based 列下标）：

| 列下标 | x | gamma | schMode | kernel 类 |
| --- | --- | --- | --- | --- |
| 0 | float | float | 0 | KernelRmsNormFloat |
| 1 | float16 | float16 | 1 | KernelRmsNormHalf(gammaIsFloat=0) |
| 2 | float16 | float | 2 | KernelRmsNormHalf(gammaIsFloat=1) |
| 3 | bfloat16 | bfloat16 | 3 | KernelRmsNormBf16(gammaIsFloat=0) |
| 4 | bfloat16 | float | 4 | KernelRmsNormBf16(gammaIsFloat=1) |

## 使用 msopgen 创建工程并替换文件

说明：下文用 `<OutputProjectDir>` 指代**用户自选的持久化目录**（例如 `~/projects/RmsNormProj`、`$HOME/work/RmsNormProj` 等），**不要**使用 `/tmp/xxx`，否则重启或清理后工程会丢失。

```bash
# 变量（按实际替换）
OP_DIR=$(pwd)/rms_norm                     # 本 skill 生成的源文件目录
OUT=<OutputProjectDir>                     # 用户自定义的工程目录
SOC=ascend910b                             # CANN 芯片"系列名"(全小写+无尾部数字),不是 npu-smi 的具体型号
                                           # 映射: npu-smi 的 Ascend910B3/B4 → ascend910b
                                           #       Ascend310P3 → ascend310p; Ascend910A → ascend910
                                           # 合法名查 ${ASCEND_HOME_PATH}/compiler/data/platform_config/*.ini 文件名
                                           # 三处必须一致: 这里 / op_host/rms_norm_def.cpp AddConfig / CMakePresets.json ASCEND_COMPUTE_UNIT

# 1. 用 JSON 生成标准算子工程骨架
# 注意: -lan cpp 不能漏!默认是 py (旧版 TBE DSL),漏掉就会生成 tbe/impl/*.py 而非 op_host/op_kernel
${INSTALL_DIR}/python/site-packages/bin/msopgen gen \
    -i ${OP_DIR}/rms_norm.json \
    -c ai_core-${SOC} \
    -lan cpp \
    -out ${OUT}

# 1.5 结构校验 ([R-053]): 必须有 op_host/ + op_kernel/, 不得出现 tbe/impl/op_info_cfg/
test -d ${OUT}/op_host && test -d ${OUT}/op_kernel && \
    ! test -d ${OUT}/tbe && ! test -d ${OUT}/impl && ! test -d ${OUT}/op_info_cfg || \
    { echo "[FATAL] msopgen 生成了 TBE DSL 旧结构,补 -lan cpp 重跑"; exit 1; }

# 2. 用本 skill 生成的文件覆盖 msopgen 默认模板
rm -f ${OUT}/op_host/rms_norm.cpp          # 删除 msopgen 合并文件
cp ${OP_DIR}/op_host/rms_norm_tiling.h       ${OUT}/op_host/
cp ${OP_DIR}/op_host/rms_norm_tiling.cpp     ${OUT}/op_host/
cp ${OP_DIR}/op_host/rms_norm_def.cpp        ${OUT}/op_host/
cp ${OP_DIR}/op_host/rms_norm_infershape.cpp ${OUT}/op_host/
cp -r ${OP_DIR}/op_host/config               ${OUT}/op_host/

cp ${OP_DIR}/op_kernel/rms_norm.cpp          ${OUT}/op_kernel/
cp ${OP_DIR}/op_kernel/rms_norm.h            ${OUT}/op_kernel/

# 3. 修改 CANN 包路径
# 编辑 ${OUT}/CMakePresets.json，把 ASCEND_CANN_PACKAGE_PATH 改为本地 CANN 安装路径

# 4. 编译打包 / 安装定制算子包
cd ${OUT}
bash build.sh
./build_out/custom_opp_*.run 2>&1 | tee install_custom_opp.log
```

## 阶段 7 二进制一致性验证

本 skill 只生成/复用最小 aclnn harness,用于把原直调 `input/` 喂给 `aclnnRmsNorm`,并将输出与原直调 `output/` 做 byte-level compare。调用方式参考 msOpGen 工程 `example/main.cpp`:

```bash
DIRECT_DIR=<rms_norm_direct_call>
mkdir -p ${OUT}/example/aclnn_output
cp -rf ${DIRECT_DIR}/input ${OUT}/example/input
cp -rf ${DIRECT_DIR}/output ${OUT}/example/output

# 在 ${OUT}/example/ 下生成 main.cpp + CMakeLists.txt
# main.cpp 读取 example/input,调用 aclnnRmsNormGetWorkspaceSize/aclnnRmsNorm,
# 输出写到 example/aclnn_output。
cd ${OUT}/example
mkdir -p build
cd build
cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
make
cd bin
./opapi_test

cd ${OUT}/example

# 运行 harness 后,逐文件比较
for ref in output/*.bin; do
    name="$(basename "$ref")"
    cmp -s "$ref" "aclnn_output/${name}" || {
        echo "[FATAL] ${name} binary differs"
        exit 1
    }
done
echo "[OK] all outputs are binary identical"
```

二进制一致表示迁移后的 aclnn 算子与原 kernel 直调输出完全一致;任一字节不同都回查 kernel 保真与 harness dtype/shape/attr。

## 核心文件要点

### op_host/rms_norm_tiling.h

```cpp
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(RmsNormTilingData)
    // 原 RmsNormCustomTiling 中保留的 8 字段
    TILING_DATA_FIELD_DEF(uint32_t, totalRows);
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, hLength);
    TILING_DATA_FIELD_DEF(uint32_t, hLengthAligned);
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
    TILING_DATA_FIELD_DEF(uint32_t, tailRows);
    TILING_DATA_FIELD_DEF(float,    epsilon);
    TILING_DATA_FIELD_DEF(uint32_t, rstdElements);
    // 扁平化的 AscendC::RmsNormTiling 字段
    TILING_DATA_FIELD_DEF(uint32_t, mainBsLength);
    TILING_DATA_FIELD_DEF(uint32_t, mainBshLength);
    TILING_DATA_FIELD_DEF(uint32_t, tailBsLength);
    TILING_DATA_FIELD_DEF(uint32_t, tailBshLength);
    TILING_DATA_FIELD_DEF(uint32_t, loopRound);
    TILING_DATA_FIELD_DEF(float,    reciprocalOfHLength);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RmsNorm, RmsNormTilingData)
}
```

### op_kernel/rms_norm.cpp

```cpp
#include "rms_norm.h"

constexpr uint32_t RMS_NORM_SCH_FP32            = 0;
constexpr uint32_t RMS_NORM_SCH_FP16_GAMMA_FP16 = 1;
constexpr uint32_t RMS_NORM_SCH_FP16_GAMMA_FP32 = 2;
constexpr uint32_t RMS_NORM_SCH_BF16_GAMMA_BF16 = 3;
constexpr uint32_t RMS_NORM_SCH_BF16_GAMMA_FP32 = 4;

extern "C" __global__ __aicore__ void rms_norm(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    uint32_t coreId = AscendC::GetBlockIdx();
    if (coreId >= tilingData.coreNum) return;

    uint32_t startRow = 0, endRow = 0;
    NsRmsNorm::ComputeRowRange(coreId, tilingData.coreNum, tilingData.totalRows,
                               tilingData.rowsPerCore, tilingData.tailRows, startRow, endRow);
    if (startRow >= endRow) return;

    if (TILING_KEY_IS(RMS_NORM_SCH_FP32)) {
        NsRmsNorm::KernelRmsNormFloat op;
        op.Init(x, gamma, y, rstd, tilingData, startRow, endRow); op.Process();
    } else if (TILING_KEY_IS(RMS_NORM_SCH_FP16_GAMMA_FP16)) {
        NsRmsNorm::KernelRmsNormHalf op;
        op.Init(x, gamma, y, rstd, tilingData, startRow, endRow, /*gammaIsFloat=*/0); op.Process();
    } else if (TILING_KEY_IS(RMS_NORM_SCH_FP16_GAMMA_FP32)) {
        NsRmsNorm::KernelRmsNormHalf op;
        op.Init(x, gamma, y, rstd, tilingData, startRow, endRow, /*gammaIsFloat=*/1); op.Process();
    } else if (TILING_KEY_IS(RMS_NORM_SCH_BF16_GAMMA_BF16)) {
        NsRmsNorm::KernelRmsNormBf16 op;
        op.Init(x, gamma, y, rstd, tilingData, startRow, endRow, /*gammaIsFloat=*/0); op.Process();
    } else if (TILING_KEY_IS(RMS_NORM_SCH_BF16_GAMMA_FP32)) {
        NsRmsNorm::KernelRmsNormBf16 op;
        op.Init(x, gamma, y, rstd, tilingData, startRow, endRow, /*gammaIsFloat=*/1); op.Process();
    }
}
```

### op_kernel/rms_norm.h 要点

三个 kernel 类（`KernelRmsNormFloat` / `KernelRmsNormHalf` / `KernelRmsNormBf16`）的 `Init` 均为：

```cpp
template <class TD>
__aicore__ inline void Init(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd,
    const TD& td, uint32_t startRow, uint32_t endRow /*, uint32_t gammaIsFloat*/)
{
    hLengthAligned_ = td.hLengthAligned;
    epsilon_        = td.epsilon;
    reciprocalH_    = td.reciprocalOfHLength;
    mainBsLength_   = td.mainBsLength;
    startRow_       = startRow;
    endRow_         = endRow;
    // ... GlobalTensor / TPipe InitBuffer
}
```

三个类都**不再**持有 `AscendC::RmsNormTiling` 成员；RmsNorm 的计算直接由基础向量指令（`Mul` / `Muls` / `ReduceSum` / `Sqrt`）手工完成。

### op_host/rms_norm_tiling.cpp 要点

```cpp
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    // 平台
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t coreNumPlat = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // shape / dtype / attr
    auto xDtype     = context->GetInputDesc(0)->GetDataType();
    auto gammaDtype = context->GetInputDesc(1)->GetDataType();
    auto xShape     = context->GetInputShape(0)->GetStorageShape();
    auto gammaShape = context->GetInputShape(1)->GetStorageShape();
    const float* epsilonPtr = context->GetAttrs()->GetAttrPointer<float>(0);

    // 计算 BS / H / H_aligned / coreNum / rowsPerCore / tailRows
    // 计算 mainBsLength / mainBshLength / tailBsLength / tailBshLength / loopRound（UB 预算）
    // reciprocalOfHLength = 1.0f / H

    RmsNormTilingData tiling;
    tiling.set_totalRows(BS);
    // ... 14 个 set_ 调用
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    context->SetBlockDim(coreNum);
    uint64_t schMode = 0;
    MapDtypeToSchMode(xDtype, gammaDtype, schMode);   // 5 种 dtype 对 -> 5 个 schMode
    context->SetTilingKey(schMode);

    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(RmsNorm).Tiling(TilingFunc);
```

## 处理遗留 .asc 与 run.sh

- **保留** `rms_norm.asc`、`run.sh`、`scripts/gen_data.py`、`data_utils.h` 作为数据生成参考，**不参与**新算子构建
- **CMakeLists.txt 处理**：
  - 把原 `CMakeLists.txt`（`find_package(ASC)` + `add_executable`）另存为 `CMakeLists.txt.kernel-direct-call`
  - 新 `CMakeLists.txt` 由 msOpGen 生成，不手写

## 完整文件清单

所有生成文件可在 `x_skills_report/ops/rms_norm/` 对应路径找到，作为本 skill 的端到端参考样例。

## 参考文档

- CANN Host 侧 Tiling 基本流程：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_00021.html
- CANN Kernel 侧算子实现：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_0063.html
