---
name: model-recommend-custom-op-dev
description: 昇腾 AscendC 自定义算子开发与 GE 接入（内核生成 + 功能补齐）。用于 GE 无对应内核、当前 dtype/format 不支持、或现有算子加插件拼不出/不划算时，按标准工程开发算子包（def/infershape/tiling/kernel/op_api/op_graph/UT）并完成安装与验证。触发：无匹配内核、assign engine failed、OpSupported 未找到、int64/bool 不支持、自定义算子、AscendC kernel、msopgen、算子包部署。不用于解析期翻译（见 model-recommend-plugin-dev）与纯图优化（见 model-recommend-fusion-pass-dev）。
---

# 自定义算子开发（AscendC + GE 接入）

## 何时用 / 边界

- 触发：`assign engine failed` / `OpSupported: op type ... is not found in this op store` / `data type ... is not supported`；或现有算子加插件拼不出语义、拼出来算子过多不划算。
- 开发前必须先试 Pass（前后加 Cast/Transpose 规避）与插件（现有算子拼接）；都不成立才开发算子，并通过 ROI 人工确认（开发周期 vs 收益上界）。
- 覆盖两类：① 全新功能（用标准工程模板）；② 已有算子有 bug/版本不支持需重编（只给思路，不入标准模板）。
- 交付为独立算子包（`custom_opp_<os>_<arch>.run`），不修改框架源码；配套插件放 `framework/`，只服务本算子。

## 触发与决策（写之前必须过一遍）

1. 确认官方算子库确实没有该算子/内核；或该算子存在但当前 dtype/format 无匹配内核（日志 `assign engine failed` / `OpSupported ... is not found in this op store` / `data type ... is not supported`）。
2. 确认无法用**现有算子 + 插件**拼出来（语义达不成），或拼出来**算子过多、性能太差**（如 GroupedBroadcast 用十几个小算子能实现但不可接受）。
3. dtype/format 不支持的场景，**优先用 Pass 在前后加 Cast/Transpose 规避**，不行再写自定义算子。
4. 二者都不成立才写自定义算子。生产级算子要保证多核切分正确性，成本高，按投入产出比决定。

自定义算子两类（本 skill 覆盖第一类，第二类只给思路）：

- **全新功能**：NPU 完全没有对应算子 → 用本 skill 的标准工程（模板 `assets/`）开发。
- **已有算子重编**：内置算子有 bug 或当前版本不支持、后续版本已修复 → 思路参考样例仓的 `custom_op/fa_headnum_align/`（拉官方 ops-transformer 仓库打补丁重编单算子；样例仓非默认参考，由用户提供），**不进本模板**。

## 交付规范（强制）

- 一个工程只包含一个算子（一个 `OP_ADD` + 一套 def/infershape/tiling/kernel）；禁止一个包塞多个算子。
- `package_name = <op>_custom`（即 vendor 名），产物 `custom_opp_<os>_<arch>.run`。
- **不得直接切分/删改原图**：全部改动落在算子/插件代码，原模型保持不变。
- 配套插件（如有）放 `framework/`，只服务于本算子。
- 必带 README（功能说明书，简洁）：功能、分类、实现原理、工程结构、安装使用说明、适配版本/芯片矩阵。
- 完整交付总则见本文末「交付总则（统一）」。

## 标准工程结构

```
<op_name>/
├── CMakeLists.txt                  # find_package(ASC) + npu_op_package，声明芯片
├── build.sh                        # --soc=ascend910b|910_93 一键编译打包
├── README.md                       # 交付用功能说明书模板（6 字段）
├── op_host/                        # Host 侧：规划
│   ├── <op>_def.cpp                # OP_ADD 算子定义（输入输出/属性/AICore 配置）
│   ├── <op>_infershape.cpp         # IMPL_OP_INFERSHAPE 形状推导
│   └── arch22/<op>_tiling.cpp      # IMPL_OP_OPTILING tiling 切分（SetBlockDim + TPL 派发）
├── op_kernel/                      # Device 侧：内核
│   ├── <op>_arch22.cpp             # kernel 入口（模板化 __global__ __aicore__）
│   └── arch22/
│       ├── <op>.h                  # AscendC kernel 类（Init/Process）
│       ├── <op>_tiling_data.h      # TilingData 结构体
│       └── <op>_tiling_key.h       # ASCENDC_TPL 模板参数与组合
├── op_api/                         # ACLNN 封装（可选，默认开启）
│   ├── aclnn_<op>.h/.cpp           # GetWorkspaceSize + Execute 两入口
│   └── <op>.cpp                    # l0op IR 层
├── op_graph/
│   └── <op>_proto.h                # REG_OP IR 描述
├── framework/                      # 可选：配套 onnx 插件（注册规范见 `model-recommend-plugin-dev`）
└── tests/ut/                       # gtest UT（仅算子需要，tiling/infershape）
```

## 开发步骤（对应模板文件）

1. `op_graph/<op>_proto.h`：`REG_OP` 声明输入输出与属性。
2. `op_host/<op>_def.cpp`：`OpDef` 链式声明每个输入输出的 dtype/format，`SetInferDataType`，`AICore().AddConfig("ascend910b"/"ascend910_93", cfg)`；动态形状开 `DynamicShapeSupportFlag(true)/DynamicRankSupportFlag(true)`；`ExtendCfgInfo("opFile.value", "<op>")` 与 kernel 文件名对应；`OP_ADD(<OpType>)` 注册。
3. `op_host/<op>_infershape.cpp`：输入→输出形状，`IMPL_OP_INFERSHAPE(<OpType>).InferShape(...)`；数据相关形状用 `InputsDataDependency({index})` 声明。
4. `op_host/arch22/<op>_tiling.cpp`：算分核/UB 规划，`context->SetBlockDim(blockDim)`，`ASCENDC_TPL_SEL_PARAM(context, dtype, schMode)` 按 dtype+模式派发内核模板，`IMPL_OP_OPTILING(<OpType>).Tiling(...)` 注册。
5. `op_kernel/arch22/<op>_tiling_data.h`/`<op>_tiling_key.h`：TilingData 结构体与 `ASCENDC_TPL_ARGS_DECL/SEL` 模板组合（host 与 device 共用，两边 include 同一份头）。
6. `op_kernel/arch22/<op>.h`：AscendC kernel 类，`Init` 读 GM 地址与 tiling，`Process` 内 `DataCopy/CopyIn/CopyOut` 实现计算。
7. `op_kernel/<op>_arch22.cpp`：`__global__ __aicore__` 入口，`REGISTER_TILING_DEFAULT` + `GET_TILING_DATA_WITH_STRUCT`。
8. `op_api/`（可选）：`aclnn<Op>GetWorkspaceSize` + `aclnn<Op>` 两入口，l0op 层注册 `OP_TYPE_REGISTER(<OpType>)` 并派发 AICORE。
9. 配套 onnx 插件（可选）：放 `framework/onnx_plugin/`，随 .run 打包，把 onnx 算子对接到本算子。

## 构建 / 安装

- `bash build.sh --soc=ascend910b`（或 `ascend910_93`；950 为 arch35，另行适配）→ `build/custom_opp_<os>_<arch>.run`。
- 重编先 `bash build.sh --make_clean`。
- 安装：`./custom_opp_<os>_<arch>.run --quiet`，然后 `source $ASCEND_OPP_PATH/vendors/<package_name>/bin/set_env.bash`。
- `<package_name>` 即 vendor 名（命名规范见本文末「交付规范速查」）。
- 升级：保留旧 `.run` → 安装新包 → 复测；注意 vendor 版本共存。
- 卸载/回退：按安装器实际支持的卸载方式；若无，删除 `$ASCEND_OPP_PATH/vendors/<package_name>` 并重新 `source`；恢复上一版后清 `kernel_meta`/`fusion_result` 并 `--verify`。

## 验证（先单算子、后整网）

> 以下集成命令在 `demo/` 下执行（测试验收只使用 GE_SESSION_DEMO）。

1. **先单独验证算子精度**：用 CPU/参考实现（或 Python golden）对同一输入比对内核输出，通过后再进整网。
2. UT（**仅自定义算子需要**；具体用例不提供样例，按官方 gtest 规范补，见 `assets/tests/ut/README.md`）：`cd tests/ut && ./run.sh`（gtest，输出 `[PASS]`、exit 0）。
3. 集成：`bash run.sh <id>` 跑通；`bash run.sh <id> --verify`。
4. 精度口径：自定义算子存在实现差异时，默认 1e-3 失败不要直接判死，可与用户确认放宽到 ≈2e-3 或 cosine 相似度后重验，并记录决策与置信度。
5. 统计口径：3 次取中位数，可剔除首样本/最大偏差但保留原始记录；首迭代不计入平均；性能改善未超过噪声带（典型 ±2%～±5%）不判收益。
6. ROI：开发前确认收益上界与开发周期匹配；周期显著大于收益 → 不开发，记录暂缓原因。

## 真实报错样本（无匹配内核 / dtype 不支持）

```
[ERROR] ... [dnnengine_manager.cc] GetDNNEngineName: ErrorNo: 1343242282(assign engine failed)
  [Check][OpSupported] Op type <OpType> of ops kernel AIcoreEngine is unsupported, reason :
  [tbe-custom]: op type <OpType> is not found in this op store.
  [Dynamic shape check]: data type <dtype> of input [<input>] is not supported.
  All supported data type and format of tensor <input> is: Data Type: {...} Format: {...}
```

先判断属于"算子完全缺失"还是"仅当前 dtype/format 不支持"：后者优先 Pass + Cast 规避。

## 开发参考资料（外部）

- 参考代码仓：https://gitcode.com/CharmanderFang/custom_development_code
- 自定义算子：msOpGen 快速入门：https://www.hiascend.com/document/detail/zh/mindstudio/latest/msOT/Operatordevelopmenttools/docs/zh/quick_start/msopgen_quick_start.md
- 标准工程写法可参考 grouped_broadcast 类样例。

## 全流程硬性限制（不可违背）

1. **不直接动原图**：禁止手工切分、删除、改写原始模型/图；全部改动落在算子/插件代码，原模型保持完整。
2. **先单件精度、后整网精度**：先在 CPU/参考实现上验证算子单独计算结果，通过后再核对整网精度。
3. **一个工程一个算子**：`package_name = <op>_custom`，禁止一个包塞多个算子。
4. **UT 仅自定义算子需要**：按官方 gtest 规范补，模板见 `assets/tests/ut/README.md`。

## 统一验收口径（跨交付件）

- 测试验收**只使用GE_SESSION_DEMO**，不接入其它测试框架。
- 流程：`prepare.sh <id>` → `compile.sh` → `run.sh <id>` → `run.sh <id> --verify`；模型放 `models/M<id>/model/model.onnx`。
- 验收三件套：
  1. **跑通**：日志 `[GE RESULT] inference finished: success`，进程 exit 0。
  2. **精度**：`run.sh <id> --verify` 输出 `[verify] PASS`（浮点默认 rtol=atol=1e-3，可被 demo 精度配置覆盖；int/bool 精确匹配；自定义算子可放宽口径，见 `model-recommend-custom-op-dev`）。检查输出数量/顺序/形状/有限性（无 NaN/Inf）。
  3. **性能**：以**首次 NPU 开箱跑通后的稳定性能**为基线（记录 QPS/TP99/TP999）；用户提供 GPU 基线则增加 GPU 对比；否则只报告相对 NPU 基线的变化。
  4. **统计**：3 次取中位数，可剔除首样本/最大偏差但保留原始记录；首迭代不计入平均；改善必须超过噪声带（典型 ±2%～±5%）才算收益。
  5. **顺序**：先单件精度 → 再整网精度 → 最后性能；单件不通过不进入整网。
- 插件/Pass 高度依赖具体模型，不提供通用功能测试：安装后直接跑真实模型，结合 `--verify`、`--dump_graph` 与 `--prof ge`/`msprof` 确认功能与性能收益。**UT 仅自定义算子需要**，且不随 skill 提供样例。

## 交付总则（统一）

- **一个交付件 = 一个功能单元**，禁止多功能合并；每个交付件可独立构建、部署、验证。
- 必带 README（功能说明书），固定 6 个字段：功能、分类、实现原理、工程结构、安装使用、版本/芯片矩阵。
- 交付件的本地源码、安装包、开关与默认值、设置参数、精度模式、验收结论与回退方式，须写入模型交付说明书（唯一写者为 `model-recommend-regression-baseline-mgmt`，其 `references/delivery.md` 定义 10 章结构）。
- 覆盖/卸载交付件后必须清理 `kernel_meta` 与 `fusion_result`，并复跑 `--verify`。

## 与工作流的关系

- 选型（Pass 优先 > Plugin 其次 > 自定义算子最后）、ROI 门禁、开发优先级排序与动作流转由优化工作流统一路由；本 skill 只负责本类交付件的开发、验证与交付。
- 上游输入：`model-recommend-adaptation` 的算子覆盖缺口、`model-recommend-analysis` 的优化建议。
