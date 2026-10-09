# Executor 与 Selector 开发

> **真源归属**：规划拆分——§1 迁移至 hccl-aicpu-executor skill，§2 与 §4 迁移至 hccl-aicpu-selector skill；§3 为跨层构建登记。迁移前以本文为准。事实来源：`src/ops/op_common/algorithm/executor/executor_v2_base.h`、`src/ops/op_common/selector/auto_selector_base.h`、`src/ops/all_reduce/algorithm/executor/ins_v2_all_reduce_sole_executor.{h,cc}`、`src/ops/all_reduce/selector/all_reduce_auto_selector.cc`；显式选路真源见 §4。

## 1. Executor

### 1.1 基类契约（InsCollAlgBase）

纯虚函数（必须实现）：

| 方法 | 职责 |
|------|------|
| `CalcAlgHierarchyInfo(comm, topoInfo, algHierarchyInfo)` | 调用 TopoMatch 计算分层信息 |
| `CalcRes(comm, param, topoInfo, algHierarchyInfo, resourceRequest)` | 构造 template 并调 `template->CalcRes()` |
| `Orchestrate(param, resCtx)` | 准备 TemplateResource + TemplateDataParams，分段 loop 调 `template->KernelRun()` |

主要虚函数（按需覆盖）：

| 方法 | 职责 |
|------|------|
| `CalcCostCoeff(comm, topoInfo, algName, param)` | 转发到 template 的静态 CalcCostCoeff |
| `GetAlgNetMeta(topoInfo)` | 算法网络元信息 |
| `FastLaunch(param, fastLaunchCtx)` | 快速下发路径（`#ifndef AICPU_COMPILE` 守卫） |
| `CalcAlgHierarchyInfoV2` / `GetAlgoMeta` / `RestoreChannelMap` | 特殊场景 |

### 1.2 最简路径：复用已有 executor，只换 template

大多数新算法**不需要新建 executor 类**——在已有 executor .cc 末尾追加注册行即可。sole executor（单 template）是最常用形态：
AllGather Sole/Parallel 的完整落点、宏参数、编译守卫和初始化时序直接查
[AllGather 接入卡](allgather-integration.md)，不用从 AllReduce 示例或历史分支重新推导。

```cpp
// executor/ins_v2_all_reduce_sole_executor.cc 末尾追加（真实示例结构）
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE, AicpuAllReduceSole<NewAlgo>, InsV2AllReduceSoleExecutor, TopoMatch1D,
    InsTempAllReduce<NewAlgo>);
REGISTER_ALG_ATTRS(
    AicpuAllReduceSole<NewAlgo>, topo.maxTopoLevelNum = 1;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D;
    /* 按新算法的拓扑约束填写 */);
```

注意：`InsV2AllReduceSoleExecutor` 是模板类 `InsV2AllReduceSoleExecutor<AlgTopoMatch, InsAlgTemplate>` 的别名形式，REGISTER 宏通过 `DefaultExecCreatorV2<insCollAlgBase<AlgTopoMatch, InsAlgTemplate>>` 完成绑定。

### 1.3 需要新 Executor 类的形态

算法编排逻辑不同于已有形态时，创建 `ins_v2_<op>_<algo>_executor.h/.cc` 继承 `InsCollAlgBase`。仓内已有形态参考（all_reduce executor 目录）：

| 形态 | 参考文件 | 特点 |
|------|---------|------|
| Sole | `ins_v2_all_reduce_sole_executor.cc` | 单 template，OrchestrateLoop 分段执行 |
| TwoShot Sole | `ins_v2_all_reduce_two_shot_sole_executor.cc` | RS+AG 双 template |
| Sequence（3 级） | `ins_v2_all_reduce_sequence_executor_aicpu.cc` | 4 template：RS_intra→RS_inter→AG_inter→AG_intra |
| Parallel | `ins_v2_all_reduce_parallel_executor.cc` | mesh+clos 并行切分 |
| Omnipipe | `ins_v2_all_reduce_omnipipe_executor.cc` | 流水线编排（V2_MULTI 注册） |
| Concurrent | `ins_v2_all_reduce_concurrent_executor.cc` | 并发执行 |

## 2. Selector

### 2.1 结构

旧 selector 基类 `AutoSelectorBase` 按引擎拆分虚函数：`SelectCcuMsAlgo` / `SelectCcuScheduleAlgo` / **`SelectAicpuAlgo`** / `SelectAivAlgo` / `SelectDPUAlgo`。本节介绍修改旧路径分支；仅注册并通过 `HCCL_ALGO` 显式选中、保留默认选路时，走 §4，不添加本节分支。此类任务只定位目标算子的候选注册、显式配置过滤和无配置成本守卫，以及 executor 的资源消费契约；不默认通读整个旧 selector 或重复追溯已核验解析链。三处注册名一致只是必要检查，默认路径仍由双轨回归验证。

可用辅助方法（AutoSelectorBase 提供）：

| 方法 | 用途 |
|------|------|
| `IsSmallData(dataSize)` / `IsLargeData(dataSize)` | 数据量分档 |
| `Is64BitDataType(dataType)` / `Is8BitDataType(dataType)` | INT64/UINT64/FP64、INT8 判定 |
| `CheckMeshNumEqualToClosNum` / `CheckClosNumMultipleOfMeshNum` | mesh-clos 数量关系检查 |
| `IsDefaultAlg(algoType)` / `IsInputOutputOverlap(opParam)` | 其它常用判定 |

### 2.2 添加新算法分支

在 `<op>_auto_selector.cc` 的 `SelectAicpuAlgo()` 中，对照已有分支结构添加：

```cpp
SelectorStatus AllReduceAutoSelector::SelectAicpuAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap,
    std::string& selectAlgName) const
{
    u64 dataSize = opParam.DataDes.count * DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];

    if (topoInfo->topoLevelNums == 1) {              // 单级拓扑
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            // ★ 新增分支：按条件选择新算法
            if (/* 触发条件，如 dataSize 档位 */) {
                selectAlgName = "AicpuAllReduceSole<NewAlgo>";
                // 必须与 REGISTER_EXEC_V2 的 name 标识符字符串化结果逐字符一致
                return SelectorStatus::MATCH;
            }
            // 已有分支……
        }
    }
    return SelectorStatus::NOT_MATCH;
}
```

**一致性检查**：selector 字符串字面量 ↔ `REGISTER_EXEC_V2` 的 name ↔ `REGISTER_ALG_ATTRS` 的 name，三处必须一致（建议 grep 校验：`grep -rn "NewAlgo" src/ops/<op>/`）。

### 2.3 选择顺序与回退

selector 内多个分支按**书写顺序**短路返回（MATCH 即止）。新算法分支插入位置决定优先级：更特化的条件放前面，兜底分支（如 one_shot 小数据默认）放后面。已有分支的阈值常量（如 `AR_AICPU_1D_64DATATYPE_DATA_SIZE`）定义在 selector .cc 文件头，新分支阈值沿用同一常量风格。

## 3. CMakeLists 修改

| 改动 | 需要修改 |
|------|---------|
| 新增 template .cc | `algorithm/template/aicpu/CMakeLists.txt` 的 `src_list` 追加 |
| 新建 executor .cc | `executor/CMakeLists.txt` 的 `src_list` 追加 |
| 仅在已有 executor .cc 末尾追加注册行 | 无需改动 |
| selector 一般复用 | 无需改动 |

## 4. HCCL_ALGO 显式选路（保留默认选择）

本节是显式选路的统一入口。预检按源码/环境指纹保存本节所列入口的核对结论、目标算子 DSL 与注册名；相关文件未变时复用，不再从历史聊天重新调查整条调用链。跨算子只核对算子 token、枚举、正反映射、拼名和资源消费的差异，不机械替换 AllGather 示例中的名字。

### 4.1 适用范围与源码定位

适用于“新增 AICPU 算法，注册 executor，通过标准配置测试，不改变默认选择”的任务。
先选择与来源匹配的版本卡，不把历史开关当成通用前置：

| 已核验版本 | 普通分发入口 | 使用方式 |
|---|---|---|
| `978d01b68c477572962b5be67c4ea9929ca9e2d7` | `op_common.cc::Selector` 按 MESH_1D / IsDevType960 条件进入 SelectorEngine | [AllGather 接入卡](allgather-integration.md)；`--contract all-gather-integration` 校验组内来源，不要求 HCCL_USE_NEW_SELECTOR |
| `a9b0839d9afc908a97cca5d2636316b642050755` | IsNewSelectorEnabled 与支持算子集合控制入口 | 下方旧版本定位表及命令示例，要求开关启用 |

文件内容及适用分支决定卡片是否可复用，不能只看 HEAD 前缀。其他版本或分支变化只核对对应入口。
下表及带 `HCCL_USE_NEW_SELECTOR=1` 的命令来自 2026-09-10 核对的第二行基线，
本地 CANN 9.2.0 / CheckerL2 的 A5 路径；Ring 命令在该基线补登记后验证，**基线本身不含新增 Ring**。

以下路径均相对 `${HCCL_ROOT}`：

| 定位点 | 已核验行为 |
|---|---|
| `src/common/alg_env_config.cc`：`IsNewSelectorEnabled`、搜索 `HCCL_USE_NEW_SELECTOR` | 新 selector 默认关闭，环境变量设为 `1` 开启 |
| `src/ops/op_common/op_common.cc`：`Selector`；`src/ops/op_common/selector/selector_engine.cc`：`IsOpSupported` | 开关开启且算子受支持才走新 selector；AllGather 在支持集合内 |
| `src/ops/op_common/selector/cost_model.cc`：`CostModelManager::InitCostModel` | 检查拓扑与 executor 注册，调用 `CalcCostCoeff`；返回空则 `uncalibrated, skip` |
| `src/ops/op_common/selector/selector_engine.cc`：`SelectorEngine::InitCostModel` | 建立成本候选后，依次按引擎、HCCL_ALGO 配置过滤 |
| `src/common/alg_parse.cc`：`FilterCmByHcclAlgo` | 优先读取通信域 `HcclGetHcclAlgo` 配置；为空时才读取环境变量 |
| `src/common/alg_parse.cc`：`HcclAlgoParser`、`ComposeAlgoName`、`ALGO_TYPES`、`GetAlgoTypeToNameMap`、`GetAlgoNameToTypeMap` | 解析 DSL、拼接注册名与映射算法类型；枚举在同目录 `alg_parse.h` |
| `src/ops/op_common/selector/alg_attrs_registry.cc`：`AlgAttrsRegistry::ParseAlgName` | 从注册名解析引擎、算子、executor 和算法属性 |

### 4.2 配置值与注册名分开使用

| executor | `HCCL_ALGO` 的值 | `--expect-algo` 的注册全名 |
|---|---|---|
| Sole，单 Ring template | `allgather:sole{ring}` | `AicpuAllGatherSoleRing` |
| Parallel，两个 Ring template | `allgather:parallel{ring,ring}` | `AicpuAllGatherParallelRingRing` |

该 DSL 按引擎、算子、executor、template 算法列表拼名。例如 `Aicpu` + `AllGather` + `Parallel` + `Ring` + `Ring`。
其他算子/组合应核对解析器及实际注册，不机械替换名称。注册全名用于命中断言，不能直接作为本表的环境变量值。

旧开关版本的 CheckerL2 示例（仅 §4.1 第二行；路径、通信域按环境确认，两条命令顺序执行）：

```bash
export HWLOC_COMPONENTS=-gl,-opencl
export HCCL_DFS_CONFIG=cluster_heartbeat:off  # 本次 CheckerL2 环境要求
bash <hccl-test-tool-hvm路径>/run.sh --role directed -m AI_CPU -o allgather -d int32 -s 64 -t 112 \
  --env HCCL_USE_NEW_SELECTOR=1 --env 'HCCL_ALGO=allgather:sole{ring}' \
  --expect-algo AicpuAllGatherSoleRing --timeout 180
bash <hccl-test-tool-hvm路径>/run.sh --role directed -m AI_CPU -o allgather -d int32 -s 64 -t 122 \
  --env HCCL_USE_NEW_SELECTOR=1 --env 'HCCL_ALGO=allgather:parallel{ring,ring}' \
  --expect-algo AicpuAllGatherParallelRingRing --timeout 180
```

112 为单级 2 卡冒烟，122 为两级 4 卡冒烟；不替代目标规模/布局验证。指定配置前确认通信域没有覆盖它的算法配置；各配置对照使用新测试进程/通信域，避免复用已缓存的候选状态。

### 4.3 候选接入与默认行为

`HCCL_ALGO` 过滤已有候选，不能绕过注册、拓扑属性或空成本候选。因此当前新路径中，**仅显式选中也需要提供非空 `CalcCostCoeff` 结果**；不要把“没有性能标定需求”理解为“不需要候选”。

- 新增算法类型时核对枚举、DSL 拼名表、正反映射、属性解析与 executor 注册。只补缺项，保留已有枚举数值，不添加私有环境变量。
- 用户要求默认选路不变时，新增模板无显式匹配配置不得进入默认竞争；默认 selector 的分支/阈值保持不变。不能仅以“没有修改 selector 文件”证明这一点。
- 将**显式启用候选**与**自动选择的性能模型**分开。未标定系数不得作为默认竞争依据；如采用仅供定向测试的占位系数，必须受已解析的正向匹配配置约束，并明确不是性能估计。空配置、其他算法配置、禁用配置不能误启用。
- 无条件返回非空系数仍然会让算法进入默认候选集，系数来自 `1e9`、命名常量或接口推导均不改变这一点。必须审查显式正向匹配目标算子/executor/算法配置的控制流；无配置及不匹配配置不能启用候选。不能用“大概率选不中”或少量默认回归通过代替守卫证据。
- 启用条件与最终过滤应使用一致的配置来源及优先级。只验证环境变量入口时，写明范围；不能声称覆盖通信域配置入口。不要用子串搜索替代 DSL 解析。
- 成本计算中的解析等 host 专用依赖须核对 device 可用性，必要时加 `#ifndef AICPU_COMPILE`；双端编译通过后仍需加载冒烟。
- 安装前后均以**不设 HCCL_ALGO、无通信域算法覆盖**的配置对比默认选路。旧开关版本若定向测试开启新 selector，除通常默认路径外，也对比 `HCCL_USE_NEW_SELECTOR=1` 的无配置路径；其他版本按实际分发条件选取回归路径。同一配置、拓扑和档位前后算法应一致。

涉及“不支持”能力、模式或协议过滤时，同时执行 [支持性契约](support-contract.md)：
必须核对条件字段在筛选时已经有效，及后续执行是否可能看到不同状态。显式配置应覆盖匹配、
不匹配、空配置和禁用配置；混合列表/超集配置依据真实 DSL 语义记录边界，不凭子串判断。
新增 DSL 登记时补对应解析/正反映射回归，测试桩漂移先分清本次与存量。

### 4.4 未命中时按证据定位

| 现象 | 下一步 |
|---|---|
| 仍走旧 selector | 先查该版本的分发条件；旧开关版本再查开关/算子支持集合，新版本查实际拓扑/设备及进程加载库 |
| `parse algo failed` / 实际配置不同 | 查 `use algo config` 日志、通信域覆盖、DSL 映射和版本；旧语法回退不代表目标已命中 |
| `executor not registered` / `skipped by topo filter` | 查注册、属性、实际拓扑与安装产物 |
| `CalcCostCoeff uncalibrated, skip` | 查成本结果和显式启用条件；此时 HCCL_ALGO 过滤尚未执行 |
| 有候选但最终名称不符 | 查引擎过滤、拼名、配置过滤和 `the selected algo type is`；以命中断言判定 |
| 定向命中但默认算法变化 | 查无配置时新增候选是否仍参与；恢复默认行为后重验，不能用 Checker PASS 豁免 |
