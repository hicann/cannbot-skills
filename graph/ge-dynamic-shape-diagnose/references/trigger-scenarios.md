# 动态调度触发场景速查表

本文档是 `ge-dynamic-shape-diagnose` skill 的场景事实源。按 **GE判定 → GE设置 → FE设置 → 图级别** 顺序排查，命中即停。

## 一键定位命令

```bash
grep -E "SetGraphUnknownFlag|Mark.*unknown|Collect.*as unknown|force unknown|cannot support no tiling|DT_STRING|HCCL|ACLNN|DVPP" ge_compiler.log | head -50
```

## GE 判定场景（算子级别根因）

GE 在判定阶段根据条件直接标记 unknown，不设置属性。

| 场景名称 | 日志关键字 | 触发条件 |
|---|---|---|
| Shape unknown 不支持静态 | `cannot support no tiling` | 算子 Shape 含 -1/-2，且不支持 no tiling |
| HostCpu 引擎算子 | `unknown as host engine` | 算子引擎为 DNN_VM_HOST_CPU |
| 子图 unknown 传播 | `unknown as subgraph unknown` | 子图包含 unknown 算子 |

### Shape unknown 不支持静态

```text
日志示例：
Mark node xxx unknown as shape unknown, and cannot support no tiling
```

- 根因：算子 Shape 含 -1/-2，且不支持 no tiling 机制。
- 解决方案：使用 `--input_shape` 指定具体 shape；或 `--dynamic_batch_size` / `--dynamic_dims` 配置多档位。

### HostCpu 引擎算子

```text
日志示例：
Mark host cpu node xxx unknown as host engine as it relies on the runtime scheduler for execution
```

- 根因：算子引擎为 DNN_VM_HOST_CPU，依赖运行时调度器执行。

### 子图 unknown 传播

```text
日志示例：
Mark node xxx unknown as subgraph unknown
```

- 根因：子图包含 unknown 算子，导致父节点被标记为 unknown。需继续向子图内部定位真正的首个 unknown 算子。

## GE 设置场景（算子级别根因）

GE 在动态拆分阶段设置 `ATTR_NAME_FORCE_UNKNOWN_SHAPE=true`，总开关日志关键字 `marked force unknown node forcibly`。

| 场景名称 | 日志关键字 | 触发条件 |
|---|---|---|
| TilingDepend 不支持 sink | `force unknown as dynamic tilingDependent` | Dynamic TilingDepend 不支持 tiling sink |
| 不支持地址刷新 | `force unknown as it does not support address refresh` | 算子不支持地址刷新机制 |
| Stage 分割节点 | `Stage`, `ATTR_STAGE_LEVEL` | 节点被标记为 Stage 分割节点 |
| DataFlow 算子 | `DataFlow`, `MarkDataFlowOpAttr` | 动态数据流算子 |

### TilingDepend 不支持 sink

```text
日志示例：
Judge node xxx is dynamic tilingDependent, index=0, input_index=0, is_const_input=false
force unknown as dynamic tilingDependent
```

- 根因：算子为 TilingDepend 类型且输入非 const，Tiling 计算依赖输入数据，编译期无法完成推导。
- 补充判定：TilingDepend 类型且输出为 Unknown Shape → 必须动态调度；TilingDepend 类型且输入非 const → 必须动态调度。

### 不支持地址刷新

```text
日志示例：
force unknown as it does not support address refresh
```

- 根因：算子不支持地址刷新机制，必须走动态调度。

### Stage 分割节点

```text
日志示例：
marked force unknown node forcibly
Stage partition: node xxx marked with ATTR_STAGE_LEVEL
```

- 根因：节点被标记为 Stage 分割节点，强制 unknown。

### DataFlow 算子

```text
日志示例：
marked force unknown node forcibly
MarkDataFlowOpAttr: node xxx is DataFlow op
```

- 根因：动态数据流算子，必须走动态调度。

## FE 设置场景（算子级别根因）

FE 在优化阶段设置属性，GE 日志统一显示 `marked force unknown node forcibly`（如 `Collect node xxx as unknown as it was marked force unknown node forcibly`），**必须追溯 FE 日志关键词才能区分具体场景**。

| 场景名称 | FE 日志关键字 | 触发条件 | 解决方案 |
|---|---|---|---|
| DT_STRING 类型 | `DT_STRING`, `data_type is DT_STRING` | 算子输入/输出为 DT_STRING | 转 INT32 等可静态编译类型 |
| HCCL 算子限制 | `AllToAll`, `unknown shape value is set` | AllToAllV/AllToAll/AllGatherV/ReduceScatterV | 考虑 AllReduce 等替代算子 |
| HCCL 内存超限 | `memory size exceeds CCL buffer` | 通信内存超过 CCL buffer | 减少单次通信数据量 |
| HCCL 任务数超限 | `taskNum >= taskMaxNum` | 任务数超过最大限制 | 优化通信策略，减少任务数 |
| DVPP 算子 | `DVPP`, `强制设置为动态算子` | DVPP 预处理算子 | — |
| ACLNN 算子 | `ACLNN`, `aclnn_only` | ACLNN_ONLY 类型或 fallback | — |
| TF 融合 DT_STRING | `TfOptimizer`, `spec_string_flag` | TensorFlow 融合图包含 DT_STRING | — |

FE 场景追溯命令：

```bash
# DT_STRING类型
grep -E "DT_STRING|data_type is DT_STRING" ge_compiler.log

# HCCL算子
grep -E "AllToAll|AllGather|ReduceScatter|unknown shape value is set|memory size exceeds|taskNum" ge_compiler.log

# DVPP算子
grep -E "DVPP|强制设置为动态算子" ge_compiler.log

# ACLNN算子
grep -E "ACLNN|aclnn_only|is an aclnn_only operation" ge_compiler.log

# TF融合DT_STRING
grep -E "TfOptimizer|spec_string_flag" ge_compiler.log
```

DT_STRING 典型 FE 日志：`Op [xxx], data_type is DT_STRING, set attr[_force_unknown_shape] success`
HCCL 算子限制典型 FE 日志：`node [xxx] op type [AllToAllV] unknown shape value is set`

## 图级别根因

| 场景名称 | 日志关键字 | 触发条件 |
|---|---|---|
| 纯 Data+NetOutput 图 | `Graph.*do not need dynamic shape partition` | 图中只有 Data(-1) 和 NetOutput，无中间算子 |
| 纯常量无输入子图 | （待补充） | 子图仅含 NetOutput/Const/Variable，无外部数据输入 |
| 原始模型动态 Shape | 无日志关键字 | 模型输入/输出本身含 -1/-2 |

- 纯 Data+NetOutput 图典型日志：`Graph xxx do not need dynamic shape partition, only has Data and NetOutput`
- 原始模型动态 Shape 排查方法：用 Netron 查看模型原始输入/输出 Shape 是否含 -1/-2，属于原生动态模型。

## 属性与常量定义

| 属性/常量 | 设置者/值 | 含义 |
|---|---|---|
| `ATTR_NAME_FORCE_UNKNOWN_SHAPE` | FE/GE | 强制标记为 unknown shape 算子 |
| `ATTR_NAME_IS_UNKNOWN_SHAPE` | GE | 标记为 unknown shape 算子 |
| `ATTR_NAME_DYNAMIC_TILING_DEPEND_OP` | GE | 标记为 Dynamic TilingDepend 算子 |
| `UNKNOWN_DIM` | -1 | 单个维度未知 |
| `UNKNOWN_DIM_NUM` | -2 | 维度数量未知 |
