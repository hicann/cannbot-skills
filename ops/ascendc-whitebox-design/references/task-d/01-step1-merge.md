# Step 1：合并 + 冲突标注

> **前置条件**：已 Read 完整的 S2P1_path_list.json 和 S2P1_operator_model.json

将路径清单与接口合法输入空间交叉，为每条路径标注可达性：

**注意**：torch_npu API 可达性检查（dead 规则 1）是可达性判定的第一道闸门。先读取 `S2P1_operator_model.json` 的 `torch_npu_api_exposure`；无此节或为 null 时跳过。

- **前置规则**：路径已被上游标记为 `reachability: "dead"` 且 `dead_reason: "kernel_has_impl_but_no_tiling_path"` 时，直接保留 dead 分类，跳过重新判定。

判定顺序：先判 dead → 再判 disputed → 最后判 reachable。

**reachable**：存在至少一组合法输入能同时满足该路径的所有 conditions。
**disputed**：接口层声明不支持但 kernel 有完整实现。
**dead**（按序号顺次判定，满足任一即丢弃）：

1. **torch_npu API 不可达**（标记为 `api_dead` 或 `api_warn`）：读取 `torch_npu_api_exposure.param_gaps`，对每个 gap 匹配 `aclnn_param` 到 path conditions 变量名。按 `torch_npu_status` 判定：`absent` 且取值∈blocked_values → `api_dead`；`fixed` 且取值≠fixed_value → `api_dead`；`derived` → `api_warn`；映射失败 → `api_warn`。`dead_reason` 格式：`"torch_npu_api_unsupported: {param}={value} - {desc}"`。`api_dead` 和 `api_warn` 不纳入 Step 2 分组。

2. **kernel 中无实现**（`key_instructions` 标记为 `NO_KERNEL_DISPATCH`）
3. **条件组合被源码约束完全排除**
4. **tiling 无法产出对应 key**（如 dtype 与 tiling 分支硬编码矛盾）

disputed 路径列入输出的 disputed 列表，交由主流程用户确认。disputed 路径的 `reachability` 标记为 `disputed`，一并写入 S2P1_path_list.json。
