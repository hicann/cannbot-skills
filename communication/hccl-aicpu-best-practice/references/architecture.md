# 架构与注册机制

> **真源归属**：hccl-aicpu-best-practice（跨层总览）。事实来源：`${HCCL_ROOT}/AGENTS.md`、`docs/zh/architecture/architecture-brief.md`、`src/ops/` 真实代码。本文所有代码引用路径均相对 `${HCCL_ROOT}`。
>
> **路径基准**：2026-09 目录重构后，template/executor 统一在 `<op>/algorithm/` 下（入口文件去 `_op` 后缀，TopoMatch 迁至 `op_common/algorithm/topo_match/`）。旧布局（`<op>/template/`、`<op>/executor/`）的仓仍存在，本文以新布局为准。

## 1. 三层结构与调用链

```text
src/ops/<op>/
├── <op>.h / <op>.cc               # 算子入口（如 HcclAllReduce；重构前为 <op>_op.cc）
├── selector/
│   ├── <op>_auto_selector.h       # 继承 AutoSelectorBase
│   └── <op>_auto_selector.cc      # SelectAicpuAlgo() 等分支方法，末尾 REGISTER_SELECTOR_BY_OPTYPE
├── algorithm/
│   ├── executor/
│   │   ├── ins_v2_<op>_*_executor.h  # 继承 InsCollAlgBase（模板类，TopoMatch + Template 为模板参数）
│   │   └── ins_v2_<op>_*_executor.cc # Orchestrate 编排 + 末尾 REGISTER_EXEC_V2 等注册
│   └── template/
│       ├── aicpu/                 # AICPU 引擎模板（ins_temp_<algo>.h/.cc）
│       ├── aiv/                   # AIV 引擎模板
│       └── ccu/                   # CCU 引擎模板
├── op_graph/                      # 图模式 proto
└── CMakeLists.txt                 # add_subdirectory(algorithm selector op_graph)
```

调用链：

```
HcclAllReduce()                                # <op>.cc 入口
  → AllReduceOutPlaceCommon()
    → Selector(topoInfo, opParam, algName)     # selector 按 topo + dataSize 选算法名
    → HcclExecOp(comm, param, topoInfo, algName)
      → executor->CalcAlgHierarchyInfo()       # TopoMatch 计算分层信息
      → executor->CalcRes()                    # 调 template->CalcRes() 计算线程/channel/notify
      → executor->Orchestrate()                # 分段 loop 调 template->KernelRun()
```

三层职责边界：selector 只产出算法名；executor 负责分层与资源编排；template 只做数据搬运，不引入控制面逻辑。

## 2. 注册机制

### 2.1 Selector 注册

```cpp
// selector/<op>_auto_selector.cc 末尾
REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLREDUCE, 18, AllReduceAutoSelector);
// 参数：optype、priority（数字，同 opType 多 selector 时的优先级）、selector 类
```

selector 基类 `AutoSelectorBase`（`src/ops/op_common/selector/auto_selector_base.h`）按引擎拆分多个虚函数，AICPU 算法写在 `SelectAicpuAlgo()`：

```cpp
virtual SelectorStatus SelectAicpuAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap,
    std::string& selectAlgName) const;
```

### 2.2 Executor 注册（★ name 是标识符，不是字符串）

```cpp
// algorithm/executor/ins_v2_all_reduce_sole_executor.cc 末尾（真实示例）
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLREDUCE, AicpuAllReduceSoleMeshOneShot, InsV2AllReduceSoleExecutor, TopoMatch1D,
    InsTempAllReduceMesh1DOneShot);
```

宏内部将 `name` 标识符字符串化（`#name`）写入注册表。**selector 中必须赋值同样的字符串字面量**：

```cpp
selectAlgName = "AicpuAllReduceSoleMeshOneShot";   // 与注册名逐字符一致
```

按 template 数量选择宏（定义见 `src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.h`）：

| 宏 | 参数 | 适用场景 |
|----|------|---------|
| `REGISTER_EXEC_V2` | (type, name, executor, topoMatch, template) | 单 template（sole executor） |
| `REGISTER_EXECUTOR_BY_TWO_TEMPS` | (type, name, executor, topoMatch, temp0, temp1) | 双 template（RS+AG 两阶段） |
| `REGISTER_EXECUTOR_BY_FOUR_TEMPS` | (type, name, executor, topoMatch, temp0..3) | 四 template（3 级 sequence：RS_intra→RS_inter→AG_inter→AG_intra） |
| `REGISTER_EXEC_V2_MULTI` | (type, name, executor, topoMatch, ...) | 任意数量 template（变参） |
| `REGISTER_EXECUTOR_BY_TOPO` | (type, name, executor, topoMatch) | 无 template 绑定 |

### 2.3 算法拓扑属性：REGISTER_ALG_ATTRS

注册算法时可附带拓扑约束/优先级检查（定义见 `src/ops/op_common/selector/alg_attrs_registry.h`）：

```cpp
REGISTER_ALG_ATTRS(
    AicpuAllReduceSoleMeshOneShot, topo.maxTopoLevelNum = 1;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS; topo.isSupportLevel0PcieMix = true;
    topo.requireAllMeshConnected = true; topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        /* 自定义检查，返回 true 才允许选中 */ });
```

常用属性：`topo.maxTopoLevelNum`（最大层级数）、`topo.supportLevel0Topos`（支持的 level0 拓扑位图）、`topo.requireAllMeshConnected`（要求全互联）、`topo.topoPriorityCheck`（自定义谓词）。

## 3. TopoMatch

executor 模板参数，负责拓扑匹配与分层计算（2026-09 重构后位于 `src/ops/op_common/algorithm/topo_match/`，拓扑信息提取在 `src/ops/op_common/topo_info/`）：

| 类 | 适用 |
|----|------|
| `TopoMatch1D` | 单级 1D 拓扑 |
| `TopoMatchMultilevel` | 多级拓扑 |
| `TopoMatchConcurrent` | 并发 mesh+clos |
| `TopoMatchUbx` | UBX 机型 |

## 4. 架构红线（硬性，不可违反）

来源：hccl 仓 `AGENTS.md` 第 3 节，改动 `src/`、`include/` 前必须对照：

| 约束 | 要求 |
|------|------|
| 分层依赖方向 | HCCL 不得被 HCOMM 反向依赖；HCCL 通过 dlsym 调 HCOMM |
| HCCL 与 HCOMM 解耦 | 不得 `#include` HCOMM 私有头；不得引入对 cann/hcomm 的编译期硬依赖；跨仓调用走 `src/common/hcomm_dlsym/` 符号表 |
| 控制面/数据面分离 | template 属数据面，不得耦合 HCOMM 控制面内部实现 |
| 新算子落标准结构 | 官方算子落 `src/ops/<op>/`；社区试验算子落 `experimental/ops/<op>/`（结构一致，不编入商用版本） |
| legacy 不承接新特性 | `src/legacy/` 仅 bug 修复 |
| 对外 API 向后兼容 | `include/` 变更需向后兼容（L1 `hccl.h`、MC2 `hccl_mc2.h`） |
