# orphan-dispatch：孤儿 Dispatch 回收

对应 Task A 执行步骤 6：孤儿 Dispatch 回收，补充 dead 路径。

---

## 目的

kernel dispatch 中可能存在 tiling 无法触发的条目（如 tiling 某分支硬编码了特定 dtype，导致其他 dtype 的 kernel dispatch 永远不可达）。回收这些"孤儿 dispatch"，补充为 dead 路径，确保 dispatch_coverage 闭环。

---

## 前置知识：Kernel 覆盖策略

kernel 侧分析范围：读取 manifest 列出的 kernel P0 文件（dispatch 入口），提取每个 dispatch 条目的 key 值、模板参数、行号。将这些条目映射为对应 tiling 路径的 `key_instructions`。`source` 字段格式为 `tiling:line → kernel_dispatch:line`。tiling 无法映射的 kernel dispatch 条目由本步骤「孤儿 Dispatch 回收」处理。

---

## 输入

- `S2P0_file_manifest.json` 的 `kernel.total_key_count`
- `S2P0_scout_k.md` P0 的逐条 dispatch 映射
- 已提取的 paths 数组
- completeness_checklist.dispatch_coverage.evidence 中记录的孤儿 kernel key

---

## 执行步骤

1. **收集已覆盖 key**：从 paths 的 `source` 字段关联 `S2P0_scout_k.md` 逐条映射，提取所有已被覆盖的 kernel key

2. **收集全部有效 key**：从 `S2P0_scout_k.md` P0 逐条映射获取全部有效 key 集合

3. **计算差集**：`orphan_keys = 全部有效 key - 已覆盖 key`。为空则 dispatch_coverage.status 记为 `"covered"`，跳过后续步骤

4. **创建 dead 路径**：对每个 orphan key，从 `S2P0_scout_k.md` 提取 kernel 类名和行号，分析 tiling 源码中无法触达的原因，追加 dead 路径。

---

## Dead 路径 JSON Schema

```json
{
  "id": "P_dead_{N}",
  "name": "{dtype}_{mode}_unreachable_in_tiling",
  "group": null,
  "reachability": "dead",
  "dead_reason": "kernel_has_impl_but_no_tiling_path",
  "dead_detail": "说明 tiling 为什么无法产出此 key（需溯源到具体源码行号）",
  "conditions": [],
  "input_variables": [],
  "caller_options": [],
  "internal_variables": [],
  "key_instructions": ["从 S2P0_scout_k.md 提取的 kernel 类名"],
  "source": "kernel_dispatch:{行号} (无 tiling 源码对应)"
}
```

## 字段填写规则

- `id`：从 1 递增，不与已有路径 ID 冲突
- `name`：格式 `{dtype}_{mode}_unreachable_in_tiling`，dtype 和 mode 从 kernel dispatch 模板参数推断
- `conditions`：尽力从 `#if` 守卫和 tiling 排除条件推导；无法精确推导时留空，禁止猜测
- `dead_detail`：必须溯源到 tiling 源码具体行号，格式如 `"mode 某模式要求 dtype==某值 (tiling.cpp:行号)，某 dtype 无法进入"`

5. **更新 completeness_checklist**：dispatch_coverage.status → `"covered"`，evidence 追加补充的路径 ID

6. **tiling 侧分支存在性校验**（补充检查）：遍历 paths 每条路径，在 tiling 源码的 mode 选择函数中检查是否存在能产出该 key 的分支。不存在 → 该路径 reachability 设为 `"dead"`

---

## 约束

- 只使用 `S2P0_scout_k.md`、`S2P0_file_manifest.json` 和已读取的 tiling 源码中的已有信息，不新增读取
- `dead_detail` 必须包含具体源码行号
- 禁止将孤儿 dispatch 标记为 reachable 或 disputed
- `conditions` 字段不确定时留空
