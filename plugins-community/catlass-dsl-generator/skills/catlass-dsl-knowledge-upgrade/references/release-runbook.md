# 发布升级 Runbook

## 1. 双基线模型

生产环境有两个事实来源，不能混用：

| 基线 | 用途 | 取值方式 |
|---|---|---|
| 已安装 wheel / 发布 tag | 判断"客户能用什么"：API 是否存在、签名、语义、样例 | `importlib.metadata.version("ascend-catlass-dsl")` 或 `git show <tag>:...` |
| 客户 master | 判断"文件在哪"：目录结构、样例路径、文档位置 | `git ls-tree -r <ref>` + 样例解析脚本 |

冲突处理顺序：**wheel > tag > master**。master 只提供路径，不提供能力结论。

典型冲突：master 把样例从 `python/tla_dsl/examples/end_to_end/simt/` 移到
`examples/python/tla_dsl/simt/`，而 v2.2.0 wheel 只认前者。此时：

- 知识库正文的能力描述按 v2.2.0 写；
- 样例引用只写**逻辑名**（如 `simt/basic_vadd_simt`），由样例解析脚本解析；
- 若必须写路径，标注"以已安装版本为准"。

## 2. master 每天更新怎么办

- **不做**每日知识库同步。master 变化不等于客户能用，追 master 会把未发布能力写进知识库。
- **做**每日漂移检测：用漂移检查脚本对比已安装版本与 manifest；版本或 digest 变化即报警。
- 允许每日跑矩阵生成脚本对 master 生成"预览矩阵"，但只能存到临时目录，
  **禁止**用预览矩阵判定 API 增减，也禁止提交。
- master 的路径变化通过 resolver 吸收，不写进知识库正文。

## 3. 版本发布时的 8 步

```bash
WS=.catlass-dsl/upgrade-2.3.0
```

1. **取数**：`git fetch --tags`，确认 `v2.3.0` 存在且 wheel 版本为 `2.3.0`。
2. **建工作区**：创建 `$WS/ITERATIONS.md`，记录目标版本、基线版本、catlass checkout 与 wheel 版本。
3. **基线矩阵**：用自写脚本从 v2.2.0 tag 生成 `$WS/api_matrix_from.json`，记录生成命令与 commit。
4. **目标矩阵**：用同一脚本从 v2.3.0 tag 生成 `$WS/api_matrix_to.json`，并产出 manifest 草案。
5. **差异**：用自写 diff 脚本生成 `$WS/api_diff.json` 与 `$WS/api_diff.md`。
6. **分类**：按 `gates.md` 的 K2 规则写 `classification.md`；不确定项进 `review`。
7. **迁移**：改知识库、写 manifest、按 K4 做人工检查并跑 pytest。
8. **交付**：填 `templates/UPGRADE_REPORT.md` 与 `templates/PR_CHECKLIST.md`，同 PR 提交。

## 4. 知识条目怎么改

- **新增 API**：先读 tag 源码中的定义与至少一个样例，确认参数与约束后再写。只写"能做什么、
  怎么调用、什么情况下失败"，不复制整段源码。
- **签名变化**：只改受影响的用法示例；语义未变时不要重写整篇。
- **移除 API**：先查知识库中引用处与样例；仍被使用的，标注"目标版本不可用，替代方案 X"。
- **样例变化**：只在知识条目中引用逻辑名；需要具体文件时用样例解析脚本输出。
- **版本标注**：统一 `dsl_version: ascend-catlass-dsl 2.3.0`，与 manifest 完全一致。
- **来源**：新增条目的 `sources` 必须指向 manifest `allowed_commits` 中的 commit。

## 5. manifest 字段

```yaml
schema: catlass-dsl-knowledge-manifest.v1
dsl_version: ascend-catlass-dsl 2.3.0
package: ascend-catlass-dsl
tag: v2.3.0
commit: <40 位 commit>
api_digest: <由矩阵生成脚本写入>
api_count: <整数>
examples_root: <v2.3.0 的样例根>
examples_count: <整数>
allowed_commits:
  - <v2.3.0 commit>
  - <允许保留的旧来源 commit>
```

- `api_digest` 只能由矩阵生成脚本写入，禁止手填。
- `allowed_commits` 只放能长期引用的 commit（发布 tag），不放个人 fork 或开发分支。
- 仓库不预置 manifest：由 K1 的矩阵脚本生成到升级工作区，完成迁移后再随知识库一起提交。

## 6. 用 v2.2.0 做 dry-run

首次落地时用已发布版本自证链路，不改任何知识条目：

用 v2.2.0 自证链路：自写矩阵脚本生成 `api_matrix_from.json` 与 manifest 草案，解析一个
已知样例路径，再按 K4 人工核对当前知识库的版本与路径问题。

预期：矩阵可生成、样例可解析、人工检查能如实报出当前知识库的版本与路径问题（这些就是
首批要修的条目，不要为了让 dry-run 通过而放宽标准）。

## 7. 回滚

- 知识库与 manifest 同 PR，回滚即 revert 该 PR。
- 升级状态目录 `.catlass-dsl/upgrade/<version>/` 不提交，回滚不影响运行态。
- manifest 回滚后必须重跑漂移检查（自写脚本），确认与已安装版本重新对齐。
