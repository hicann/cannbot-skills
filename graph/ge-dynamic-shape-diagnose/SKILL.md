---
name: ge-dynamic-shape-diagnose
description: >-
  昇腾 GE 动态调度/Unknown Shape 触发场景诊断。触发条件（Use when）：用户遇到图被标记为
  unknown graph、触发动态 shape 调度、动态拆分、unknown 子图下沉、算子被标记
  force unknown、shape 含 -1/-2、性能劣化怀疑动态调度导致，或用户提供 ge_compiler
  日志要求定位动态调度根因场景/根因算子时使用。覆盖 GE判定/GE设置/FE设置/图级别四类
  根因场景的日志关键字定位方法。不依赖外部仓库配置，基于日志关键字排查。
---

# GE 动态调度触发场景及根因定位

诊断目标：对触发动态 shape 调度（unknown graph）的模型，定位其**根因场景**和（可选的）**根因算子**，并给出消除建议。

## 1. 核心概念（先对齐再排查）

| 概念 | 含义 |
|---|---|
| Unknown Graph | 图被标记 `ATTR_NAME_GRAPH_UNKNOWN_FLAG=true`，走动态调度；图拆分为 known 子图（静态下沉）和 unknown 子图（动态调度执行） |
| Unknown Shape Node | 算子 Shape 含 -1（单维度未知，`UNKNOWN_DIM`）或 -2（维度数量未知，`UNKNOWN_DIM_NUM`） |
| 根因算子 | 编译阶段第一个输出 Shape 变为 unknown 的算子；其下游算子级联标记为动态 shape |
| No Tiling 机制 | shape unknown 但编译期可确定 tiling 参数，无需运行时计算。**shape unknown + 支持 no tiling → 可静态编译；不支持 → 必须动态调度** |

## 2. 主流程（三步定位）

严格按顺序执行，前一步结论决定是否进入下一步。

### Step 1. 确认是否触发动态调度

```bash
grep -E "SetGraphUnknownFlag|mark graph.*unknown|Graph.*do not need dynamic shape partition" ge_compiler.log
```

- 命中 unknown 标记 → 图走了动态调度，进入 Step 2。
- 命中 `do not need dynamic shape partition` → 图级别场景，直接跳到 Step 2 的图级别排查。
- 都没命中且用户怀疑动态调度 → 让用户确认日志级别（`atc --log=info` 起步）和日志文件是否为本次编译产物，再重新采集。

### Step 2. 按层级定位根因场景

按 **GE判定 → GE设置 → FE设置 → 图级别** 顺序排查，命中即停：

```bash
# 第一步：GE 判定场景
grep -E "cannot support no tiling|unknown as host engine|unknown as subgraph unknown" ge_compiler.log

# 第二步：GE 设置场景
grep -E "force unknown as dynamic tilingDependent|force unknown as it does not support address refresh|Stage|DataFlow" ge_compiler.log

# 第三步：FE 设置场景（先找总开关，再按场景追溯）
grep "marked force unknown node forcibly" ge_compiler.log
```

命中第三步的 `marked force unknown node forcibly` 时，必须继续追溯 FE 日志关键词（`DT_STRING` / `AllToAll` / `memory size exceeds CCL buffer` / `taskNum` / `DVPP` / `ACLNN` / `TfOptimizer`）才能确定具体场景，详见 `references/trigger-scenarios.md`。

算子级无果后做图级别排查（纯 Data+NetOutput 图、纯常量无输入子图、原始模型动态 Shape）。原始模型动态 Shape 无日志关键字，用 Netron 查看模型原始输入/输出 Shape 是否含 -1/-2。

每个场景的日志示例、根因解释和解决方案对照 `references/trigger-scenarios.md` 输出，不要凭记忆编造日志内容。

### Step 3. 定位根因算子（可选）

用户需要具体算子名时，开启完整日志后搜 InferShape 记录：

```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1
atc --log=info
grep -E "before_infer|after_infer" ge_compiler.log
```

定位第一个输出 Shape 由固定值变为 -1/-2 的算子。

## 3. 报告要求

- 结论必须包含：根因场景名称、命中的日志关键行（原样粘贴）、根因算子（如已定位）、解决方案。
- 多个场景同时命中时，GE 判定 > GE 设置 > FE 设置优先级呈现，并列出全部命中行；级联标记的下游算子不算根因。
- `marked force unknown node forcibly` 命中但 FE 关键词全部无匹配时，结论写"FE 设置场景，具体子场景未能定位"，列出建议补采的日志，不得编造场景。
- 解决方案按场景给：Shape unknown 不支持静态 → `--input_shape` 指定具体 shape 或 `--dynamic_batch_size`/`--dynamic_dims` 配置多档位；DT_STRING → 转 INT32 等可静态编译类型；HCCL 算子限制 → 考虑 AllReduce 等替代算子；HCCL 内存超限 → 减少单次通信数据量；HCCL 任务数超限 → 优化通信策略减少任务数；其余场景按 `references/trigger-scenarios.md` 的解决方案列。
- 用户只要根因场景结论时，不强制做 Step 3。

## 4. 执行纪律

- 只用 `grep -RIn` 在用户提供的日志文件中检索，命令可直接复制执行。
- 禁止在没有日志证据时断言根因场景；没有命中任何关键字就如实说"未定位到"。
- 日志是唯一判据；模型结构推测只能作为候选原因写，不得写进结论。
- 涉及源码级追溯（如需查 `dynamic_shape_partition.cc` 的判定逻辑）时，先向用户确认 GE 源码仓路径，源码引用格式参照 `references/source-code-map.md`。
