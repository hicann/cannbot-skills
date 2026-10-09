# AllGather AICPU：Sole / Parallel 接入卡

适用于新增用户算法并复用现有 AllGather Sole/Parallel executor，默认选路保持不变。
来源为 HCCL `978d01b68c477572962b5be67c4ea9929ca9e2d7`，2026-09-28 核对；
路径均相对已确认的 HCCL 仓。本文描述基准源码，不声称其中已有 Ring 注册或支持全部模式。

```bash
python3 <template子skill路径>/scripts/check_sources.py --repo "$HCCL_REPO" \
  --hcomm-repo "$HCOMM_REPO" --op all_gather --contract all-gather-integration
```

看 `groups["hccl_contract:all-gather-integration"].status` 判断本卡来源，其他组按各自范围处理。
总体 CHANGED 不否定未变分组，但使用到变化组的结论仍须定向核对；工作区新增注册自然会改变相应文件指纹。
本卡不覆盖完整传递依赖、插件选择器快路径或安装库身份。布局/分块预算仍用
[Sole 算子卡](../../hccl-aicpu-best-practice/references/11-operator-contracts.md)和
[Parallel 四阶段卡](../../hccl-aicpu-best-practice/references/12-allgather-parallel.md)。

## 1. 先固定绑定组合

| 绑定 | 注册名示例 | HCCL_ALGO 示例 | 含义 |
|---|---|---|---|
| Sole Ring | AicpuAllGatherSoleRing | `allgather:sole{ring}` | 单 template |
| Parallel Ring/Ring | AicpuAllGatherParallelRingRing | `allgather:parallel{ring,ring}` | intra/inter 均为 Ring |
| Parallel Mesh/Ring | AicpuAllGatherParallelMeshRing | `allgather:parallel{mesh,ring}` | intra 为 Mesh，inter 为 Ring |

这些是新增登记后才可用的示例。Mesh/Ring 与 Ring/Ring 是不同算法组合，不能因参考分支已有其中一个就替换用户需求。
尚未确定组合时先根据目标调度设计，不把“Parallel”自动翻译为 Mesh/Ring；不默认读取未选用的 Sequence executor。

## 2. 注册与修改文件

| 位置 | 要做的事 |
|---|---|
| `src/ops/all_gather/algorithm/template/aicpu/ins_temp_<name>.{h,cc}` | 新 template；生成器提供接口骨架，资源与数据流按用户算法实现 |
| 同目录 `CMakeLists.txt` | host `src_list` 加新 .cc |
| `src/scatter_aicpu_kernel.cmake` | device `ops/all_gather/algorithm/template/aicpu/...cc` 加同一源码 |
| `src/ops/all_gather/algorithm/executor/ins_v2_all_gather_sole_executor.cc` | include 新头、单 template 注册、同名 REGISTER_ALG_ATTRS；核对成本参数和初始化契约 |
| 同目录 `ins_v2_all_gather_parallel_executor.cc` | include 新头、按已选组合注册两个 template、同名属性；核对四个成本调用 |
| `src/common/alg_parse.h` / `.cc` | 新类型缺失时补 AlgoType、ALGO_TYPES、GetAlgoTypeToNameMap、GetAlgoNameToTypeMap；保留既有枚举数值 |

注册接口来自 `src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.h`：

```cpp
// 接入示例：需先实现 template，并核验 Ring 类型及映射。
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLGATHER, AicpuAllGatherSoleRing,
    InsV2AllGatherSoleExecutor, TopoMatchOneLevel, InsTempAllGatherRing);
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER, AicpuAllGatherParallelRingRing,
    InsV2AllGatherParallelExecutor, TopoMatchTwoLevel, InsTempAllGatherRing, InsTempAllGatherRing);
```

Sole 使用 `TopoMatchOneLevel`，Parallel 使用 `TopoMatchTwoLevel`；模板顺序是 intra、inter。
宏既登记 creator，也向候选算法目录登记名字与模板关系。全名还须被 `AlgAttrsRegistry::ParseAlgName` 正确解析。
`REGISTER_ALG_ATTRS` 的拓扑、dtype、模式限制从本算法适用范围推导，不照抄 NHR/Mesh 的专用过滤条件。

编译守卫：本版本 Sole 的 AICPU 注册位于后续 CCU 的 `#ifndef AICPU_COMPILE` 之外；
Parallel 的 AICPU 注册在 `CANN_VERSION_NUM >= CANN_VERSION(9,0,0)` 内，也在 CCU 的 host-only 守卫之外。
`REGISTER_ALG_ATTRS` 自身在 AICPU_COMPILE 下为空宏，不据此把整个 executor 注册包进 host-only 守卫。
仅给已有 executor 增加注册时，不新增 executor 文件，也无需修改 executor CMake 或上层目录 CMake。
只走本卡的新选择链时无需给旧 `all_gather_auto_selector.cc` 添加默认分支；涉及其他路径则重新界定范围。
UT stub/文档按实际交付范围核对，本卡不要求复用历史任务的 UT、测试矩阵或性能系数。

## 3. 执行期对象初始化

资源申请期的对象与执行期重新构造的对象分开核对，不能依赖申请期对象上的副作用。

| 调用方 / 真源 | 执行期顺序（本版本 AICPU_TS 分支） |
|---|---|
| Sole `OrchestrateLoop` | 新建 template → 对新 Ring 等满足条件的模板调用 SetchannelsPerRank → KernelRun；具体条件为 `isPod || algName != AicpuAllGatherSoleNHR` |
| Parallel `Orchestrate` | 新建 intra/inter → **仅 inter** 调用 SetchannelsPerRank → PrepareResForTemplate 读取两者 GetRes → OrchestrateLoop/KernelRun |

基类 `SetchannelsPerRank` 对空 map 返回错误。Sole 即使 KernelRun 内能处理 N=1，仍可能在调用它之前失败。
只有已声明支持该路径时才实现相应空 channel 行为，并核对 executor 的其他拒绝条件。
Parallel 的 intra 未经过该初始化，GetRes 若依赖 channelsPerRank 等状态，须证明其执行期值与申请预算一致。
该版本 Parallel 对上述 SetchannelsPerRank/GetRes 调用没有传播返回码，不能靠基类报错保证立即停止。

接口的纯虚/默认实现以直接基类 `InsAlgTemplateBase` 为准：Describe 和两种 GetNotifyIdx 为纯虚；
CalcRes/KernelRun/GetRes/FastLaunch 有返回错误的默认实现，GetThreadNum 默认 0，CalcScratchMultiple 默认 0。
“能编译”不代表可运行；按实际消费者覆盖，细节复用[接口契约](../../hccl-aicpu-best-practice/SKILL.md#3-接口契约)。
Sole 的 host FastLaunch 实现通过 `std::make_unique<InsAlgTemplate>()` 构造对象，Parallel 也有默认构造路径；
保留骨架默认构造，不把它解释为本算法已经支持 FastLaunch。

## 4. 成本参数传递矩阵

字段定义：`src/ops/op_common/selector/cost_model.h::CalcCostCoeffParam`。

| 基准调用方 | 直接聚合初始化数 | algName | comm | topoInfo |
|---|---:|---|---|---|
| AllGather Sole `CalcCostCoeff` | 1 | 已传 | 显式 nullptr | 已传 |
| AllGather Parallel `CalcCostCoeff` | 4 | 省略 | 省略 | 省略 |

后三字段省略时都默认 nullptr。按模板消费者选取所需字段：algName 用于名称守卫，comm 用于通信域配置读取，
topoInfo 仅在模板消费拓扑时要求；不为凑齐字段机械改动所有调用方。
确认需要通信域配置和拓扑后，可运行：

```bash
python3 <父skill路径>/scripts/check_cost_forwarding.py \
  --executor <所选executor.cc> --cost-header "$HCCL_REPO/src/ops/op_common/selector/cost_model.h" \
  --expected-calls <Sole为1或Parallel为4> \
  --require-field algName=algName --require-field comm=comm --require-field topoInfo=topoInfo
```

删除本次不依赖字段的 `--require-field`；不指定时兼容旧命令，只检查 algName。
脚本 PASS 不证明字段实际有效或非空；别名、捕获列表、配置优先级和最终命中仍需核对。
补传字段可能影响共享 executor 的其他模板消费者，需检查行为变化并保留默认路径回归。

## 5. 选择器版本与配置来源

当前卡片的普通分发路径：`src/ops/op_common/op_common.cc::Selector` 在
`level0Topo == MESH_1D || AutoSelectorBase::IsDevType960()` 时进入 SelectorEngine，否则走 ExecuteSelector。
该条件不消费 `HCCL_USE_NEW_SELECTOR`；不能把旧版本开关加进命令当作启用证据。
新链建立成本候选，过滤引擎和 HCCL_ALGO，再选择最小成本；空成本结果不会被显式配置凭空加入。

`src/common/alg_parse.cc::FilterCmByHcclAlgo` 的配置来源：

1. 读取通信域 `HcclGetHcclAlgo`；失败时清空结果，继续回退环境变量。
2. 通信域结果为空时读取 `HCCL_ALGO`；环境 `EmptyString` 表示未配置。
3. 经 HcclAlgoParser 解析后按注册名匹配；解析失败时保留原候选，不代表已命中新算法。

模板显式启用守卫需使用一致的配置来源和优先级。只实现 env 入口时注明范围，不能声称支持通信域配置。
未标定新算法在无有效正向匹配时不进入默认竞争；禁止从日志中的 `hasRing` 示例直接复制而跳过
目标算子、executor、禁用项和混合列表语义检查。命中用 hccl-vm 的 `--expect-algo` 验证，不修改 HCCL API 来增加该参数。
旧的 `a9b0839d` 开关路径及跨版本通用约束见[显式选路](executor-selector.md#4-hccl_algo-显式选路保留默认选择)。
