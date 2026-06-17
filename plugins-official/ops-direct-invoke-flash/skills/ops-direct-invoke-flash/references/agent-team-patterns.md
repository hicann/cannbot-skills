# Agent 团队协作模式

当需要同时移植多个算子，或单个算子能从并行评审中获益时，使用 Agent 团队进行协调。

## 何时使用 Agent 团队

- 同时移植 2 个及以上算子（每个算子位于各自独立的 worktree 中）
- 对于复杂算子，阶段 3-4 的评审与阶段 5-6 的实现可从并行工作中获益

## 单算子团队（以评审为主）

对于单个算子的并行评审，使用标准的 Agent 工具并设置 `run_in_background=true`：

```
# 主控 Agent 负责阶段 0-3（编写定义文档）
# 随后启动评审 Agent 进行阶段 3 评审：
Agent(name="math-reviewer", subagent_type="general-purpose",
      run_in_background=true, prompt="...")
Agent(name="semantics-reviewer", subagent_type="general-purpose",
      run_in_background=true, prompt="...")
```

评审 Agent 写入各自独立的文件，无需 worktree 隔离。

## 多算子团队（并行移植）

对于同时移植多个算子的场景，启动相互隔离的 Agent——每个 Agent 位于各自独立的 worktree 中：

```
# 创建用于协调的任务
TaskCreate(subject="Port {OP1}", description="完整实现 {OP1}")
TaskCreate(subject="Port {OP2}", description="完整实现 {OP2}")

# 启动相互隔离的 Agent
Agent(name="{OP1}-porter", subagent_type="general-purpose",
      isolation="worktree",
      prompt="使用 ops-direct-invoke-flash skill 构建 {SOURCE_1}。遵循所有阶段。")

Agent(name="{OP2}-porter", subagent_type="general-purpose",
      isolation="worktree",
      prompt="使用 ops-direct-invoke-flash skill 构建 {SOURCE_2}。遵循所有阶段。")
```

## Worktree 隔离规则

当多个 Agent 并行工作时，共享文件会引发冲突：

| 文件 | 冲突风险 | 缓解措施 |
|------|---------------|------------|
| `docs/index.md` / `AGENTS.md` | 高——多个 Agent 都要登记各自的算子 | 使用 worktree 隔离；最后合并 |
| `operators/{OP}/{OP}.asc` | 无——每个算子独立目录 | 可安全并行工作 |
| `operators/{OP}/CMakeLists.txt` | 无——每个算子独立目录 | 可安全并行工作 |
| `operators/{OP}/test_{OP}.py` | 无——每个算子独立目录 | 可安全并行工作 |

当 Agent 需要修改 `docs/index.md`、`AGENTS.md` 等共享的仓库级文件时，务必使用 `isolation="worktree"`。

## 合并策略

当所有算子都在各自独立的 worktree 中移植完成后：

1. 指定一个 Agent（或主控 Agent）作为合并协调者。
2. 将各 worktree 分支依次合并到主分支。
3. 通过合并各算子的登记条目来解决 `docs/index.md` 冲突。
4. 通过合并各算子的条目来解决 `AGENTS.md` 冲突。
5. 在最终合并后执行完整的构建与测试，以验证集成结果。

## 通信模式

团队成员通过共享任务列表进行协调：

1. **主控 Agent** 创建任务并进行分配
2. **工作 Agent** 通过 `TaskUpdate(owner="my-name")` 认领任务
3. **评审 Agent** 将评审发现写入 `plans/review_*.md` 并将评审任务标记为完成
4. **主控 Agent** 监控 `TaskList` 并吸纳评审发现

使用 `SendMessage` 进行直接协调：
- `SendMessage(to="math-reviewer", message="Please also check the edge case for x=0")`
- `SendMessage(to="*", message="CMakeLists.txt has been updated, pull before building")`

只有常驻的团队 Agent 才需要关闭，对于会自然结束的后台评审 Agent 则无需关闭。
