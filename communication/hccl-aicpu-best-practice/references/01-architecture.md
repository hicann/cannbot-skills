# 01 · Template 在 HCCL 架构中的位置

## 1. 完整调用链

以 AllReduce 单算子模式为例：

```
include/hccl.h              HcclAllReduce
  └─ src/ops/all_reduce/all_reduce.cc
       · 参数校验、生成 tag、FillAllReduceOpParam 填 OpParam
       └─ src/ops/op_common/op_common.cc :: Selector
            · 算拓扑 TopoInfoWithNetLayerDetails
            └─ selector/execute_selector.cc :: ExecuteSelector
                 · 按 opType 从 SelectorRegistry 取 selector
                 · src/ops/all_reduce/selector/all_reduce_auto_selector.cc
                 · 综合 拓扑 / rankSize / 数据量 / dataType / reduceOp / 环境变量
                 └─▶ 产出一个 **算法名字符串**，如 "AicpuAllReduceSoleMeshOneShot"
            └─ HcclExecOp
                 └─ CollAlgExecRegistryV2 按算法名取 executor 实例
                      · src/ops/all_reduce/algorithm/executor/ins_v2_all_reduce_sole_executor.cc
                      · Orchestrate → OrchestrateLoop：算 loop 次数、填 TemplateDataParams
                      └─▶ ★ Template ★  algTemplate->KernelRun(param, tempAlgParams, templateAlgRes)
```

一句话分工：

- **Selector** 决定"用哪个算法"
- **Executor** 决定"分几次搬、每次搬多少字节、buffer 怎么摆"
- **Template** 决定"这一次搬，谁往谁发、走哪条 channel、跑在哪个 thread、什么时候同步"

## 2. Template 是怎么被实例化的

**不是**通过注册表按名字查，而是**通过 executor 的 C++ 模板参数静态绑定**：

```cpp
// src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.h
#define REGISTER_EXEC_V2(type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate)

// 用法（在 executor 的 .cc 末尾）
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLREDUCE,   // 算子类型
                 AicpuAllReduceSoleMeshOneShot,      // ← selector 吐出的算法名
                 InsV2AllReduceSoleExecutor,         // executor 类模板
                 TopoMatch1D,                        // 拓扑匹配策略
                 InsTempAllReduceMesh1DOneShot);     // ← 你的 template 类
```

executor 内部：

```cpp
template <typename AlgTopoMatch, typename InsAlgTemplate>
class InsV2AllReduceSoleExecutor { ... };

// CalcRes / OrchestrateLoop 里都是直接构造具体类型：
std::shared_ptr<InsAlgTemplate> algTemplate =
    std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
```

**推论（写 template 时务必记住）：**

1. AICPU template **不写** `REGISTER_TEMPLATE_V2` 宏。写了也不错，但没人用它。
2. 因为是静态分发，`CalcCostCoeff` 可以是 `static` 而不是 virtual —— executor 用 `InsAlgTemplate::CalcCostCoeff(...)` 直接调。
3. 构造函数签名被 executor 硬编码，**必须**是
   `(const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)`，
   另外建议保留默认构造 `XXX() = default;`：executor 的 FastLaunch 路径用 `make_unique<InsAlgTemplate>()` 无参构造，
   绑定到走 FastLaunch 的 executor（各 `*SoleExecutor`）时**必须**有，否则编译失败。
   仓内 52 个 AICPU template 里 32 个有它，缺的那些（DPU / Intra / Inter / OmniPipe 变体）不走 FastLaunch。
4. 多层算法用 `REGISTER_EXEC_V2_MULTI(type, name, exec, topoMatch, Tmpl0, Tmpl1, ...)`，
   executor 模板参数包里 level0/level1 各一个 template（如 sequence executor：intra 用 Mesh，inter 用 NHR）。

**唯一例外 —— DPU 展开路径**：`src/ops/op_common/algorithm/template/dpu/kernel_launch.cc` 会
`InsAlgTemplateRegistry::Instance().GetAlgTemplate(dpuRunInfo.templateName)` 按字符串反查，
所以名字里带 `Dpu` 的 template 必须在 .cc 末尾写：

```cpp
REGISTER_TEMPLATE_V2("InsTempReduceScatterMesh1dDpu", InsTempReduceScatterMesh1dDpu);
```

## 3. 两代 template API（别写错代）

| | V1（遗留） | V2（**新代码一律用这个**） |
|---|---|---|
| 基类 | `AlgTemplateBase` | `CommonAlgTemplateBase` → `InsAlgTemplateBase` |
| 头文件 | `alg_template_base.h` | `alg_v2_template_base.h` |
| 主入口 | `RunAsync()` | `KernelRun()` |
| 注册 | `REGISTER_TEMPLATE(TemplateType::TEMPLATE_XXX, Cls)`，枚举 key | `REGISTER_TEMPLATE_V2("Cls", Cls)`，字符串 key（且 AICPU 侧一般不用） |
| 取用 | `AlgTemplateRegistry::Instance().GetAlgTemplate(type)` | executor 模板参数静态绑定 |
| 存量 | 仅 `experimental/.../birs/` 等历史角落 | `src/ops/*/algorithm/template/aicpu/ins_temp_*` 全部 |

类继承树：

```
CommonAlgTemplateBase          (common_alg_template_base.h) — Describe/CalcRes/GetRes/GetThreadNum/
  ├── InsAlgTemplateBase       (alg_v2_template_base.h)      CalcScratchMultiple/KernelRun/FastLaunch
  │     └── ins_temp_*   ← 本 skill 范围（AICPU / DPU / host_nic）
  ├── CcuAlgTemplateBase       (ccu_alg_template_base.h)
  │     └── ccu_temp_*
  └── AivAlgTemplateBase       (aiv_alg_template_base.h)
        └── aiv_temp_*
```

## 4. Executor 给 Template 的输入契约

executor 在调用 `KernelRun` 前填写本阶段的 `TemplateDataParams`。定位具体函数、分支、
buffer 来源与 rank 映射后再写 template 地址计算；不能仅由 executor 名称推断布局。

五参数定义、普通单层与分层实例、量化检查统一见 [09-template-data-params.md](09-template-data-params.md)。
`repeatNum` 是本次调用内的逻辑组数；四个 stride 是输入/输出各自的两维字节步长。
还需结合 `sliceSize`、尾块、base offset 与 scratch 布局使用。
输入可能来自上一阶段的 OUTPUT，此时 AllGather 的 `inputSliceStride` 可以非零；
PR 2901 的 Ring 缺项案例见该文 §8。
