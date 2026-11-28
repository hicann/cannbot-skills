---
name: model-recommend-fusion-pass-dev
description: 昇腾图引擎 GE 的自定义图融合 Pass 开发。用于模型已跑通、需在不新增算子的前提下提性能：把可一次下发的算子链改写为融合结构或等价算子，以独立 .so 交付到 custom_fusion_passes/。含 FlashAttention 族与 GELU 链的实战结论。触发：图融合、自定义 Pass、REGISTER_CUSTOM_PASS、减少下发、算子融合、attention 融合、FlashAttention、PromptFlashAttention、类型下压、冗余 Cast 消除、常量折叠。不用于解析期算子不支持（见 model-recommend-plugin-dev）与缺内核（见 model-recommend-custom-op-dev）。
---

# 自定义图融合 Pass 开发

## 何时用 / 边界

- 触发：模型已跑通，但 profiling 显示小算子多 / 下发次数高 / 存在可融合子结构，且官方融合未覆盖（`fusion_result.json` 中 match>0 effect=0）。
- 解决"跑得快"，**不改数学语义**；只做设备侧可一次下发的重组，依赖 host 调度的组合不做。
- **不得修改 GE_SESSION_DEMO 框架源码**：Pass 以独立 `.so` 交付到 `opp/vendors/<vendor>/custom_fusion_passes/`；确需框架侧改动按"框架需求移交"，由用户确认后由维护者处理。
- 解析期算子不支持 → `model-recommend-plugin-dev`；缺内核或 Pass 表达不了 → `model-recommend-custom-op-dev`。

## 职责边界

- Pass 只做"**纯拼接**"：把图内可融合的、不依赖 host CPU 调度的若干算子，改写成**一次下发即可在 device 侧算完**的结构，减少下发次数与算子间调度等待。
- **不改数学语义**；能用已有算子重组达成的就不写自定义算子。
- 在图构建/编译阶段生效；对已固化的 om 无法注入，改图需重新走带图优化的编译。

## 适用性判断（开发前必答）

1. 模型已跑通？未跑通先回到工作流选型（先确认 `model-recommend-adaptation` 跑通）。
2. 目标组合的算子是否**不依赖 host 调度、可一次下发**？依赖 host 的（如需要 host 中间结果再回灌）不值得融合。
3. 官方融合是否已覆盖？覆盖了就别写。
4. 常量判定（如"Add 的输入是固定常量"）在 **Pass 内部做校验**（读输入是否为 Const/Constant 节点），不需要人工确认。
5. 报 `data type ... is not supported`（算子存在但该 dtype/format 无内核）时，优先本 Pass 方案：在前后加 Cast/Transpose 规避。

## 交付规范（强制）

- 一个 Pass 工程只做一个融合/改写模式，且是该结构的**最大同结构融合**；禁止把多个不同结构合成一个巨型 Pass，否则功能定界不清。
- 同一模式、仅算子版本变体（如 MatMulV2/MatMulV3）可放同一 Pass，但融合效果未必好，作为尝试方案；结构不同必须拆 Pass。
- 命名由同一 `<pattern>` 派生：目录 `<pattern>_pass/`、注册名 `<Pattern>Pass`、产物 `lib<pattern>_pass.so`。
- 部署 `opp/vendors/<vendor>/custom_fusion_passes/`。
- **不得直接切分/删改原图**：全部改动落在 Pass 代码，原图保持不变、可回溯。
- 必带 README（功能说明书，简洁）：功能、分类、实现原理、工程结构、安装使用说明、适配版本/芯片矩阵。
- 完整交付总则见本文末「交付总则（统一）」。

## 执行顺序与命名

- 多个自定义 Pass 按**名称字母序执行**；命名需考虑先后依赖（如拆分/预处理类用靠前名字，融合类用靠后名字）。
- 同一 Pass 只匹配一个结构模式；不同结构必须拆不同 Pass，识别范围不得跨模式复用。

## 开发流程

```
识别图中目标模式 → 复制 assets 模板 → 实现回调（遍历→匹配→构造新算子→替换→删旧节点）
→ REGISTER_CUSTOM_PASS 注册（kAfterInferShape）→ 编译出 .so → 部署 → 模型验证
```

### 回调实现要点（参考 assets/src/sample_pass.cpp）

- 遍历：`graph->GetAllNodes()`，`node.GetType(type)` 按算子类型收集目标节点。
- 匹配：`node.GetInDataNodesAndPortIndexs(input_index)` 查上下游，校验常量输入、dtype（`GetOutputDesc(i, desc).GetDataType()`）。
- 改写：`graph->AddDataEdge(src, src_port, dst, dst_port)` 连新边；`graph->RemoveEdge(...)` 删旧边；`graph->RemoveNode(node)` 删孤立旧节点；`node.UpdateInputDesc(index, desc)` 同步输入描述。
- 日志约定：pass 进入/退出打 `---- <PassName> begin/end ----` 与命中数量（如 `affected count: N`），便于验证时在日志中确认执行。

### 注册

```cpp
REGISTER_CUSTOM_PASS("<PassName>").CustomPassFn(<PassFn>).Stage(CustomPassStage::kAfterInferShape);
```

- 回调签名：`graphStatus <PassFn>(GraphPtr& graph, CustomPassContext& context)`。
- 阶段默认 `kAfterInferShape`；特殊需求（如常量权重 NZ 化）才用其他阶段，需自查官方文档。
- `REG_FUSION_PASS` 为旧写法，不用。

## 工程与构建

- 模板 `assets/`：`CMakeLists.txt` + `include/pass_utils.h`（节点类型/改名/断边工具）+ `src/sample_pass.cpp`（**完整范例**：MatMulV2/V3、BatchMatMulV2/V3 + 一维常量 Add ⇒ BiasAdd）。
- CMake 标准见 `assets/CMakeLists.txt`：C++17、`ASCEND_HOME_PATH` 定位 CANN、include 集含 `include/graph`、`include/ge`、`include/register`、`opp/built-in/op_graph/inc`、`opp/built-in/op_proto/inc`，链接 `graph register`。
- 构建：`mkdir build && cd build && cmake .. && make` → `build/lib<pass_name>.so`。
- 部署：`cp lib<pass_name>.so ${ASCEND_HOME_PATH}/opp/vendors/<vendor>/custom_fusion_passes/`，GE 启动自动加载。
- 升级：备份旧 `.so` → 覆盖 → 清 `kernel_meta`/`fusion_result` → `--verify`。
- 卸载/回退：移除 `.so` 或把 `GE_FUSION_SWITCH_FILE` 中该 Pass 设为 `off`；清缓存后复验。
- assets 内含交付用 `README.md` 模板（6 字段）。

## 验证

> 以下命令在 `demo/` 下执行（测试验收只使用 GE_SESSION_DEMO）。

1. 确认加载：`bash run.sh <id>`，日志出现 pass 自打的 `---- begin ----`/`affected count`。
2. 确认改写正确：`bash run.sh <id> --dump_graph`（或 `DUMP_GE_GRAPH=1`）对比 dump 图，目标结构已被融合（如 MatMul+Add → BiasAdd）。
3. 精度（含开关对比）：`bash run.sh <id> --verify` PASS。融合/算子替换可能引入精度问题，须保证数学等价；若精度异常，做 A/B 对比定位：
   - 最稳妥：把 `lib<pattern>_pass.so` 移出/放回 `custom_fusion_passes/`，分别跑 `--verify`；
   - 若融合开关能识别该 Pass 名：设置 `GE_FUSION_SWITCH_FILE=<file>` 后重跑，格式见 `assets/fusion_switch.example.json`（`Switch.GraphFusion.<PassName> = "off"/"on"`）。
4. 性能：`bash run.sh <id> --prof ge`（或 `--prof msprof`、`tools/sweep.py`）采集算子耗时/QPS/TP99，与基线对比报告收益。
5. **模型级功能确认**：验收口径见本文末「统一验收口径（跨交付件）」；安装 `.so` 后直接跑真实模型，以图 dump + `--verify` + profiling 确认功能与性能收益。
6. **effect/match 证据**：以 Pass 日志的 `affected count` 与 `fusion_result`/dump 图确认真实生效；无 effect 不得给出收益结论。
7. **统计与 ROI**：3 次取中位数，可剔除首样本/最大偏差但保留记录；首迭代不计入平均；改善未超过噪声带不判收益；单 Pass 收益 <1% 且在噪声内不保留。

## 融合模式速查（参考样例，不作为标准命名）

开发参考资料（外部）：

- 参考代码仓：https://gitcode.com/CharmanderFang/custom_development_code
- 自定义 Pass 开发：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/graphdevg/docs/zh/user_guides/graph_dev/custom_pass_development/introduction.md

每个模式对应一个独立 Pass 交付件（禁止合并）。

| 场景 | 案例目录 | 模式 |
| --- | --- | --- |
| MatMul+Add 融合 | `mmadd_pass`（参考原文档示例） | MatMulV2/V3、BatchMatMulV2/V3 + Add(一维常量) ⇒ BiasAdd |
| LayerNorm 融合 | `FuseLayerNormBroadcastToPass` | (Add)+BroadcastTo+LayerNorm+Adds+Rsqrt ⇒ LayerNormV3 |
| Attention 融合 | `AttentionFusionPass` / `FaPass` | BMM+...+Softmax+BMM ⇒ FlashAttention(Score) |
| RmsNorm 融合 | `RmsNormPass` | Cast+Pow+ReduceMean+...+Mul ⇒ RmsNorm |
| int64 类型下压 | `CastInt64ReducePass` / `CastInt64EqualPass` | 归约轴积 < max_int32 时 int64⇒int32 |
| 冗余 Cast 消除 | `cast_remove_pass` | 删目标算子输入上的 Cast 链并重连 |
| 常量折叠 | `RangeCastPass` | Const+Range+Cast ⇒ Const |
| 批量矩阵乘重组 | `BatchMatmulFlattenPass` / `MoePass` / `bmm_tile_pass` | BatchMatMul ⇒ Reshape+MatMul+Reshape 等 |

## 全流程硬性限制（不可违背）

1. **不直接动原图**：禁止手工切分、删除、改写原始模型/图；全部改写落在 Pass 代码，原图保持不变、可回溯。
2. **Pass 命名即执行顺序**：多个自定义 Pass 按名称字母序执行，命名需体现先后顺序；同一 Pass 只匹配一个结构模式，不同结构必须拆 Pass，识别范围不得跨模式复用。
3. **Pass 可能引入精度问题**：融合/算子替换必须数学等价；出现精度异常时用启用/停用开关 A/B 定位。
4. **独立交付**：Pass 一律以 `.so` 形式交付，不得向框架 `src/` 新增或修改文件。

## 参考：FlashAttention 族与 GELU 链实战

attention 链替换为 `PromptFlashAttention`、GELU 链替换为 `GeluV2` 的功能确认参数矩阵、实测收益、已知坑与模式匹配要点，见 `references/flash-attention.md`。

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
