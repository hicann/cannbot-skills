# 知识库升级 PR Checklist

- [ ] 分支从最新 `upstream/master` 拉出，未在 master 上直接修改
- [ ] `from` / `to` 两份矩阵均由**发布 tag** 生成，未使用 master 判定 API 增减
- [ ] `api_diff` 已归档，新增/移除/签名变化均有对应处理
- [ ] 所有 `review` 条目已给出结论，未直接写入知识库
- [ ] `dsl_version` 全部为 `ascend-catlass-dsl <to>`
- [ ] 新条目 `sources` 的 commit 均在 manifest `allowed_commits` 内
- [ ] 样例引用可被样例解析脚本核对，未写死 master-only 路径
- [ ] manifest 已更新（`tag`、`commit`、`api_digest`、`api_count`、`examples_root`）
- [ ] 人工检查报告错误数为 0
- [ ] `pytest tests/ -q` 全绿，且未放宽既有断言
- [ ] 漂移检查与已安装版本对齐
- [ ] 升级报告已填写，未覆盖项已列出
- [ ] 未改动算子工作流产物（`workspace/`、`develop-runs/`、`optimize-runs/`）

## 提交信息

```text
docs(knowledge): 升级知识库基线至 <to>

- 矩阵：<api_added>/<api_removed>/<signature_changed>
- 知识条目：<新增/更新/删除> 数量
- 验证：人工检查 passed，pytest <N> passed
```
