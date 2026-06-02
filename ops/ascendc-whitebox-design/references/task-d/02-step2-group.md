# Step 2：路径分组

> **前置条件**：Step 1 已完成；仅对 reachable 路径分组

对所有 reachable 路径，根据 tiling 模式选择逻辑分组：

- 共享同一 tiling 入口条件的路径归入同一 group
- 为每个 group 命名：读 tiling 源码中的模式常量或分支标识命名（如 `MODE_SPLIT` → "split_d"）。降级路径在上游的路径 id 对应 group 名后加 `_degrade` 后缀即可
- 回填每条路径的 `group` 字段，构建顶层 `groups` 列表
- 为每个 group 撰写 `mode` 描述：一句话概括该 group 的触发条件（如 `"MODE_SPLIT_D — numCol > ubFactor"`）。`mode` 将在 Step 4 写入 S2P2_param_def.json

## 分组规则

**规则 1（主维度）**：按 tiling 源码分支判断逻辑划分 group。dtype 不同但 tiling 条件相同 → 同一 group（dtype 是 group 内部参数差异）。

**规则 2（正交维度不拆组）**：tiling 存在多个独立决策维度、各自独立判断后组合为 dispatch key 时，不拆为独立 group。适用：变体共用同一套 mode 选择判断链 + 仅作为 key 编码偏移量 + 差异可通过结构化约束表达。

**规则 3（聚焦参数范围）**：每个 group 的参数范围严格约束为仅能触发该模式。互补 group 的维度取值区间互斥。

**规则 3 补充（平台可达性前置过滤）**：确定 group 列表前，将触发条件中的平台常量（`core_count`、`ub_size` 等）替换为目标平台实际值。若目标平台无法触发该路径，不创建对应 group。不可达路径单独标注平台原因。

**规则 4（特殊子分支独立）**：具有独立触发条件且导致不同 kernel 行为的子分支，独立为单独 group。若某 tiling 模式仅在某 dtype 下触发，该 group 的 dtype 集合仅包含触发 dtype，不强制补充其他 dtype。

**规则 4 补充（降级路径独立为组）**：存在降级路径（路径 id 含 `{id}d{num}` 后缀格式，或 conditions 含 `boundary_check`）时，其参数触发区间与降级目标 group 不连续，必须独立为单独 group。

**规则 5（覆盖完整性）**：每个 group 覆盖该模式内所有 kernel 子分支（`TILING_KEY_IS`），验证 tiling key 范围完整。

## 分组完整性校验

1. 遍历 paths，确认每条 reachable 路径已分配 `group`
2. 汇总每个 group 覆盖的 tiling key 范围
3. 对比源码读取范围文本块中 `【kernel】` 节末尾的 total_key_count，验证所有 key 均被覆盖

发现遗漏 → 标记并补上对应路径后再输出。
