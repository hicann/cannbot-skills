---
name: model-recommend-plugin-dev
description: 昇腾图引擎 GE 的 ONNX 自定义插件（解析期算子翻译）开发。用于 GE 解析器报"算子不支持"（GPU 定制/大融合算子、第三方域算子）时，把原始算子翻译成 NPU 现有算子组合，只做翻译/拼接、不携带新内核。触发：算子不支持、未知 domain、CheckTypeSupported、parse model failed、onnx_plugin、REGISTER_CUSTOM_OP、解析期算子翻译。不用于图优化/性能融合（见 model-recommend-fusion-pass-dev）与缺内核场景（见 model-recommend-custom-op-dev）。
---

# 自定义 ONNX 插件开发（GE 解析期算子翻译）

## 何时用 / 边界

- 触发：模型在 `parse` 阶段报 `it is not supported` / `CheckTypeSupported` / `parse model failed`；图内出现非 `ai.onnx` 域或 GPU 定制融合算子。
- 解决"能否跑通"。翻译目标只能是 **NPU 现有算子**；所需原语 NPU 也没有 → 转 `model-recommend-custom-op-dev`。
- 已跑通、要提性能 → `model-recommend-fusion-pass-dev`；无匹配内核 → `model-recommend-custom-op-dev`。
- 不用于：首次模型接入/适配（见 `model-recommend-adaptation`）、GE_SESSION_DEMO 框架代码修改、训练场景。

## 职责边界

- 插件负责"**翻译/拼接**"：把 GE 解析器不认识的原始算子（GPU 定制/大融合算子）翻译成 NPU **现有算子**的组合，**只能使用 NPU 现有算子**。
- 所需中间原语 NPU 也没有 → 该部分下沉为自定义算子（`model-recommend-custom-op-dev`），插件只做翻译与拼接，两者通常配套交付。
- 插件解决"能否跑通"；跑通后精度不达标 = 翻译逻辑错，回查本插件两个回调的计算逻辑。
- **禁止手工切分/删改原始 onnx 图绕过问题**；适配必须在本插件内完成，原始模型保持完整。

## 交付规范（强制）

- 一个插件工程只对应**一个原始算子**（同算子多 opset 版本属同一交付件）；禁止一个插件翻译多个不同算子。
- 只做翻译/拼接，使用 NPU 现有算子，不携带新内核；缺原语转 `model-recommend-custom-op-dev`。
- 插件依赖自定义算子时允许合包，但能解耦时优先独立交付。
- **不得直接切分/删改原图**：全部改动落在插件代码，原模型保持不变。
- 形式：独立插件工程（方式 A），`framework/onnx_plugin/<domain>_<optype>.cc`，产物 `libcust_onnx_parsers.so` → `opp/vendors/<vendor>/framework/onnx/`。
- 必带 README（功能说明书，简洁）：功能、分类、实现原理、工程结构、安装使用说明、适配版本/芯片矩阵；并注明**翻译依据来源**（GPU 源码 / 说明文档 / 用户确认）。
- 完整交付总则见本文末「交付总则（统一）」。

## 前置硬性门槛（语义来源）

**插件是"翻译"，不是"猜译"。** 必须拿到原算子的**完整 IR 语义**才能开工；语义不全时停下与用户多轮确认，**禁止凭空捏造**。臆测语义会导致精度偏差，推理结果不可信。

开工前必须逐项确认（不清则列为问题清单，多轮问清）：

1. **输入输出**：个数、名字、shape 依赖关系、dtype、format/布局。
2. **属性**：全部属性名、类型、默认值、取值域。
3. **算子语义**：逐输入到输出的计算公式/伪代码，含激活、归一化 eps、缩放、mask、轴顺序等细节。
4. **版本**：domain、opset/version 覆盖范围。
5. **边界情况**：空张量、动态 shape、广播/轴语义、多输出端口。
6. **精度要求**：累积/中间计算类型、是否允许精度放宽。

材料来源优先级：

1. GPU 自定义算子的**实现源码**（最可靠）；
2. 算子的**详细说明文档/规范**；
3. 都没有 → 要求用户按上述 6 项**逐项给出语义**，并多轮确认到无歧义。

确认结论写入交付 README 的"实现原理/来源"，便于追溯。

## 前置步骤

1. 定位未知算子：解析 onnx 图的 domain/op_type，标记非 `ai.onnx` 域与可疑算子名；也可在**不安装插件**的情况下跑模型，从 GE 日志的 `CheckTypeSupported` 行直接拿到 `<domain>::<version>::<OpType>`（见下「真实报错样本」）。
2. 按上面的**硬性门槛**获取并确认语义；未确认完整前不写翻译代码。

## 工程模板（assets/，独立插件工程）

```
assets/
├── CMakeLists.txt          # 顶层：g++、CANN include 解析、输出 vendors/<vendor>
├── build.sh                # 一键编译；clean 清理；自动探测 ASCEND include
├── README.md               # 交付用功能说明书模板（6 字段）
├── compare_single_op.py    # 插件级精度比对（CPU 参考 vs 子图输出）
└── onnx_plugin/
    ├── CMakeLists.txt      # 编 libcust_onnx_parsers.so（依赖系统安装的 nlohmann/json，find_package）
    ├── sample_common.h     # CreateScalar/Const、UpdateMeta、GetOriName 公共工具
    └── sample_plugin.cc    # 完整范例：属性解析 + Mul/Add 子图 + BUILDIN 注册
```

`sample_plugin.cc` 是**可对照的完整范例**（`<OpType>(x,y)` + `scale` 属性 ⇒ `Mul→Add`），按真实算子替换构图即可。

拷贝模板后：改 `<vendor>`、改源文件名（规范 `<domain>_<optype>.cc`，点转下划线，如 `com_example_OpType.cc`）。

## 两个回调

### ParseParamsByOperatorFn（解析属性）

- onnx 原算子属性在 `"attribute"` 字段的 **JSON 字符串**里，**不能直接 GetAttr**，先 `json::parse`。
- 属性类型 ↔ JSON 字段：

| onnx 属性 | JSON 字段 | 取值 |
| --- | --- | --- |
| int | `"i"` | `attr["i"]` |
| float | `"f"` | `attr["f"].is_string() ? std::stof(...) : attr["f"].get<float>()` |
| string | `"s"` | `attr["s"]` |
| ints | `"ints"` | `attr["ints"].get<std::vector<int>>()` |
| floats | `"floats"` | `attr["floats"].get<std::vector<float>>()` |

- 逐属性 `op_dest.SetAttr(...)`；属性名不一致在此映射，可算派生属性。
- 结尾用 `DynamicInputRegister/DynamicOutputRegister` 传原算子输入输出个数，并 `SetAttr("original_type", "domain::version::OpType")`。

### ParseOpToGraphFn（构建子图）

- 属性已在 ParseParams 设好，此处直接 `op.GetAttr(...)`。
- 每条输入边建一个 `ge::op::Data(name).set_attr_index(i)`；固定值/固定形状用 `ge::op::Const`；多输出算子取端口用 `set_input_x(prev_op, port)`。
- `graph.SetInputs(inputs)`；输出用显式类型 `std::vector<std::pair<ge::Operator, std::vector<size_t>>>`（避免 `SetOutputs` 重载歧义），**输出个数必须与原算子输出个数一致**。

## 注册（通用范式）

```cpp
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("domain::1::OpType"),
                   ge::AscendString("domain::8::OpType"),
                   // ... 覆盖该算子全部 opset 版本 ...
                   ge::AscendString("domain::18::OpType")})
    .ParseParamsByOperatorFn(ParseParamsXxx)
    .ParseOpToGraphFn(ParseOpToGraphXxx)
    .ImplyType(ImplyType::BUILDIN);
```

- 子图展开统一 **`ImplyType::BUILDIN`**；一对一直映射现有 GE 算子用 **`ImplyType::GELOCAL`**；**TVM 为旧写法，不用**。
- 一个 `.so` 内多个插件文件时：每个文件函数名加唯一前缀；公共工具函数放共享头，声明为 `static inline` 防符号冲突。
- vendor 插件加载优先级高于内置解析，同名 op 会覆盖内置解析。

## 构建、安装与卸载

- `bash build.sh`（重编先 `bash build.sh clean`）。
- 产物：`build_out/makepkg/packages/vendors/<vendor>/framework/onnx/libcust_onnx_parsers.so`。
- 生效二选一：
  1. 临时验证：`source build_out/makepkg/set_env.bash`（写入 `ASCEND_CUSTOM_OPP_PATH`，靠前优先）；
  2. 正式：拷到 `${ASCEND_HOME_PATH}/opp/vendors/<vendor>/framework/onnx/`。
- 前提：已 `source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh`。
- 升级：备份旧 `.so` → 覆盖 → 清 `kernel_meta`/`fusion_result` → `--verify`。
- 卸载/回退：删除该 vendor 下插件文件，或从 `ASCEND_CUSTOM_OPP_PATH` 移除；恢复上一版后清缓存并复验。

## 验证（先单件、后整网）

> 以下命令在 `demo/` 下执行（测试验收只使用 GE_SESSION_DEMO）。

1. **先单独验证插件精度**：在 CPU 上按已确认的 IR 语义写参考实现（或调用 GPU 原算子），用同一份输入与插件翻译后子图的输出逐元素比对（可用 `assets/compare_single_op.py <expected.npy> <actual.npy>`，支持 `--rtol/--atol`，`.bin` 用 `--dtype/--shape`）；不一致先修翻译，**不要**直接进入整网验证。
2. 跑通：`bash run.sh <id>`，解析不再报"算子不支持"，`[GE RESULT] inference finished: success`。
3. `bash run.sh <id> --dump_graph` 检查解析后的图已展开成目标子图。
4. 整网精度：`bash run.sh <id> --verify`，`[verify] PASS`（浮点默认 rtol=atol=1e-3，可被 demo 精度配置覆盖；int/bool 精确；检查输出数量/顺序/形状/有限性）。
5. **模型级功能/性能**：验收口径见本文末「统一验收口径（跨交付件）」；安装后直接跑真实模型，用 `--prof ge`（或 `msprof`）采集性能，与基线比 QPS/TP99，并结合 `--verify` 确认功能与精度。
6. **统计与收益**：3 次取中位数，可剔除首样本/最大偏差但保留原始记录；首迭代不计入平均；改善未超过噪声带（典型 ±2%～±5%）不判收益。
7. **ROI**：插件只做翻译/拼接；若需新增内核或改动范围超出"翻译"，先回到选型与 ROI 评估，不直接开工。

## 真实报错样本（解析期算子不支持）

插件未安装/未生效时，GE 解析期 prechecker 报错；demo stdout 通常只显示笼统的 `parse model failed, status=4294967295`，详细原因在日志。通用模式：

```
[ERROR] ... [pre_checker.cc] CheckTypeSupported: ErrorNo: 4294967295(failed) Check op[<node>]'s type[<domain>::<version>::<OpType>] failed, it is not supported.
[ERROR] ... [onnx_parser.cc] ModelParseToGraphImpl: ... [Invoke][Prechecker] failed.
[ERROR] ... [onnx_parser.cc] aclgrphParseONNX: ... failed.
[ERROR] parse model failed, status=4294967295
```

定位步骤：

1. 确认插件未安装，或 `ASCEND_CUSTOM_OPP_PATH` 不含该插件 vendor。
2. 复现：`bash run.sh <id> --debug`（日志落到 `models/M<id>/report/run_debug_*.log`）；或直接 `bash run.sh <id>` 后查 GE plog。
3. 在日志中找 `CheckTypeSupported` 行，取 `type[<domain>::<version>::<OpType>]`，即插件 `OriginOpType` 的写法。
4. 按「前置硬性门槛」向用户索取该算子的完整 IR 语义，再开始翻译。

## 开发参考资料（外部）

- 参考代码仓：https://gitcode.com/CharmanderFang/custom_development_code
- 插件：ONNX 自定义算子接入图：https://gitcode.com/cann/cann-samples/blob/master/Samples/0_Introduction/99_system/custom_op_in_graph/README_CN.md
- 插件：ParseOpToGraph 开发指导：https://gitcode.com/guopeian/cann_develop_guide/blob/main/onnx_plugin_guide/ParseOpToGraph_Development_Guide.md
- 外部参考样例（不作为标准）：具体算子以图内实际 domain/opset 与已确认语义为准；上述链接仅用于工程形态参考。

## 全流程硬性限制（不可违背）

1. **不直接动原图**：禁止手工切分、删除、改写原始模型/图来绕过问题；全部适配落在插件代码，原始模型作为输入保持完整。
2. **先单件精度、后整网精度**：插件完成后先在 CPU/参考实现上验证其单独翻译结果的精度，通过后再核对整网精度。
3. **语义未确认不得开工**：必须拿到原算子完整 IR 语义（见上文「前置硬性门槛」），禁止凭空捏造。
4. **不携带新内核**：插件只做翻译/拼接；需要新内核时与 `model-recommend-custom-op-dev` 配套交付。

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
