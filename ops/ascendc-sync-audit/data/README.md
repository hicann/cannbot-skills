# data/ — 历史同步修复 case 数据

> `case_retriever.py` 的数据源。来自 gitcode.com 上 CANN 官方算子仓的**已合入**同步修复 PR。

## 文件

| 文件 | 角色 | 说明 |
|------|------|------|
| `sync_cases.jsonl` | **source of truth** | 明文 JSONL，每行一个 case，git diff 可审计 |
| `sync_cases.db` | 运行时产物 | SQLite，由 `build_db.py` 从 jsonl 重建，供 `case_retriever.py` 查询 |
| `build_db.py` | 构建/校验工具 | `jsonl → db` 重建、`--export` 反向导出、`--verify` 一致性校验 |

## 数据来源与规模

333 条 case，全部来自已合入（is_merged=TRUE）的真实同步修复 PR：

| 仓库 | case 数 |
|------|--------|
| cann/ops-transformer | 181 |
| cann/ops-nn | 132 |
| cann/ops-math | 20 |

采集方式：按同步关键词（SetFlag/WaitFlag/CrossCore/PipeBarrier/sync/死锁等）检索 PR 标题/描述，
用 `git show <base_sha>:<file>` / `<head_sha>:<file>` 提取修复前后 diff，人工标注 `fix_type`。
每条 case 保留 `repo + pr_id + base_sha + head_sha`，可回溯原始 PR 验证。

## 字段（schema）

| 字段 | 说明 |
|------|------|
| `id` | 主键 |
| `repo` / `issue_number` / `pr_id` / `pr_title` / `pr_desc` / `is_merged` | PR 元信息（可回溯来源） |
| `file_path` | 修复涉及的文件路径 |
| `base_sha` / `head_sha` | 修复前/后 commit |
| `diff_patch` | 修复 diff（case_retriever 的证据来源） |
| `keywords` / `sync_apis` / `index_vars` | 检索特征（逗号分隔） |
| `fix_type` | 修复类型：flag_pair(56) / buffer_index_mismatch(39) / atomic_ordering(32) / sync_fix(27) / pipe_barrier_missing(22) / cross_core(14) / missing_sync(4) / unknown(139) |
| `fix_summary` | 修复摘要 |

## 修改 case 数据的流程

```bash
# 1. 编辑明文 jsonl（增删改 case）
vim data/sync_cases.jsonl

# 2. 重建 db
python3 data/build_db.py

# 3. 校验一致
python3 data/build_db.py --verify
```

> 禁止直接改 db 不同步 jsonl；若确实先改了 db，用 `python3 data/build_db.py --export` 回写明文再提交。
