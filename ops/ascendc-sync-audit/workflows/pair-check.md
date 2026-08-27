# pair-check — Set/Wait 配对快查工作流

> 用户只关心 set/wait 是否配对、个数是否一致、是否先 wait 后 set。轻量工作流。

## 阶段

### Stage 1 — 配对扫描
- 运行：`python3 scripts/sync_audit.py <file-or-dir> --check pair`
- 脚本按 EVENT_ID / flag 变量分组，统计 Set/Wait 数量、顺序、EVENT_ID 匹配
- 禁止否决脚本候选（见 [SKILL.md 执行规则 6](../SKILL.md)）

### Stage 2 — 输出
脚本输出的候选即为结果，编号展示修改方案（文件名+行号+diff），等用户选择后执行。
