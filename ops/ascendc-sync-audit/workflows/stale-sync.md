# stale-sync — 残留/多余同步清理工作流

> 迁移/重构后（如多核→单核、手写同步→Matmul 高阶 API、Cube+Vec 合并）怀疑残留了不该有的同步。

## 触发场景

| 场景 | 该清理的同步 | 风险 |
|------|------------|------|
| 多核 → 单核 | 所有 `CrossCoreSetFlag`/`CrossCoreWaitFlag` | 单核无人 SetFlag → WaitFlag 永久阻塞（SYNC-03） |
| 手写同步 → Matmul 高阶 API | 手写的 `CrossCoreSetFlag`/`CrossCoreWaitFlag` | 与高阶 API 内部 flagId 冲突（SYNC-07） |
| Cube+Vec 分核 → 合并单核 | 跨核 Set/WaitFlag | 同上 |
| EnQue/DeQue 已同步 → 又加 Flag | 冗余 `SetFlag`/`WaitFlag`/`PipeBarrier` | 性能损失（SYNC-11） |

## 阶段

### Stage 1 — 全量检查，聚焦残留类条例
- 运行：`python3 scripts/sync_audit.py <file-or-dir> --check all --format json`
- 保留与残留同步相关的候选：SYNC-03（单核残留 CrossCore）、SYNC-07（与高阶 API flagId 冲突）、SYNC-11（冗余同步）、SYNC-12（残留 SyncAll）
- 其余红线候选（如 SYNC-01/04）也一并呈报，不筛选（见 [SKILL.md 执行规则 6](../SKILL.md)）

### Stage 2 — 列出全部同步点清单
- 运行：`python3 scripts/sync_audit.py <file-or-dir> --list-only`
- 产出所有同步 API 调用清单（行号、类型、参数），供用户结合迁移上下文逐个判断"该同步保护的数据依赖是否还存在"——这一语义判断脚本不做，由用户完成

### Stage 3 — 输出
Stage 1 的候选编号展示修改方案（文件名+行号+diff），等用户选择后执行；Stage 2 的清单附在报告末尾供人工排查。
