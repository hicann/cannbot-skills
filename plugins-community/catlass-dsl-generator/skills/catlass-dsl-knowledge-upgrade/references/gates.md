# 升级门禁（K0–K5）

阶段记录：`<workspace>/ITERATIONS.md` 与 `.catlass-dsl/upgrade/<version>/` 下的产物。
每阶段完成前逐项核对下表产物，缺一项不得进入下一阶段。

## 产物键

| 键 | 产生者 | 内容要求 |
|---|---|---|
| `api_matrix_from` | 矩阵生成脚本（自写） | 基线版本矩阵，必须由 tag 生成 |
| `api_matrix_to` | 矩阵生成脚本（自写） | 目标版本矩阵，必须由目标 tag 生成 |
| `api_diff` | diff 脚本（自写） | 新增/移除/签名变化的机器可读结果 |
| `classification` | Agent | 每条差异的类别、动作、是否需要 review |
| `knowledge_changes` | Agent | 改动文件列表、每处理由、对应差异条目 |
| `check_report` | 人工检查报告 | `status=passed` 的 Markdown 或 JSON 报告 |
| `test_report` | `pytest` | 全绿输出或 JUnit/文本日志 |
| `upgrade_report` | 模板 | 面向评审的升级报告 |
| `pr_checklist` | 模板 | 提交前逐项勾选 |

## K0 预检

- 目标版本与基线版本都必须是 `X.Y.Z`。
- catlass checkout 路径下必须能找到 `python/tla_dsl/catlass/core_api.py` 或
  `catlass/core_api.py`。
- 目标版本必须能从 tag 或已安装 wheel 得到；只有 master 而没有 tag 时**不允许**开始，
  因为无法固定语义基线。
- 分支必须是新分支，工作区干净；不得在 `master` 上直接改知识库。

## K1 矩阵

- `from` 矩阵必须来自**已发布 tag**，`to` 矩阵必须来自**新 tag**。
- 不允许两份矩阵都来自 worktree HEAD：master 每天变化，无法复现。
- 只允许把 master 用于路径解析（样例解析脚本），不允许用于 API 差异判定。
- `api_digest` 必须写进 manifest；两份矩阵的 digest 相同而版本不同时，仍需人工确认是否
  只是版本号变化。

## K2 分类

每条差异必须落入以下之一，不允许"其它"：

| 类别 | 判定 | 动作 |
|---|---|---|
| API 新增 | `api_added` | 查 tag 源码与样例，决定新增知识条目或补进已有条目 |
| API 移除 | `api_removed` | 查是否仍被算子使用；删除或标注弃用 |
| 签名变化 | `api_signature_changed` | 判断是否影响已记录用法；受影响必须改用法 |
| 样例新增/移除 | `examples_*` | 更新样例引用，路径由 resolver 给出 |
| 文档变化 | `docs_count` 变化 | 只作为线索，不作为语义证据 |

`classification` 最小结构：

```json
{
  "version": "2.3.0",
  "items": [
    {"kind": "api_added", "name": "histograms", "action": "add|update|none", "target": "knowledge/dsl/core-api.md", "review": false}
  ],
  "review": [
    {"name": "mmad", "reason": "签名变化，需确认默认参数语义"}
  ]
}
```

- 每一项必须有 `action`；`review=true` 的条目禁止在 K3 直接落盘。
- 无法从 tag 源码与样例得出结论的，一律进 `review`。

## K3 迁移

- 每改一个文件，必须在 `knowledge_changes` 中记录：文件、章节、对应差异、改动理由。
- 禁止"顺手"重写无关章节；禁止删除带来源脚注的条目来消除校验错误。
- `dsl_version` 必须与 manifest 完全一致（`ascend-catlass-dsl X.Y.Z`）。
- 新增知识条目必须带 `sources`，且来源 commit 在 manifest 的 `allowed_commits` 内。
- 样例路径必须能通过样例解析脚本核对，禁止写死只在 master 存在的目录。

## K4 验证

两项都必须通过：

```bash
python3 -m pytest tests/ -q
```

- 人工检查 `check-report.md`：error 必须为 0；warning 允许存在但要在报告中列出。
- 人工核对来源白名单与 Core API 集合，必须随版本更新；**放宽标准**不算通过，
  必须同步更新为真实新基线。
- 任一失败即回到 K3，不得在 K4 直接改测试绕过。

## K5 交付

- 升级报告必须包含：版本对、矩阵摘要、改动清单、验证结果、review 结论、未覆盖项。
- PR checklist 必须逐项勾选，包括 manifest 与知识库同 PR、无 master-only 内容混入。
- K5 完成后在 `ITERATIONS.md` 记为 done；未完成的版本不得在 manifest 中标注为已支持。

## 禁止清单

- 用 master 生成 API 矩阵并据此判定 API 增减。
- 把 `review` 条目直接写进知识库。
- 只更新 manifest 不更新知识库，或只更新知识库不更新 manifest。
- 删除/放宽测试断言来让 K4 通过。
- 把"待排期""以后再说"写成已支持。
