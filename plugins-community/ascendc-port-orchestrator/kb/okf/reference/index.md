# reference

参考层仅保留尚未迁移的 porter 本地资产；这里不再存放上游文档搬运副本。

## [porter](porter/index.md)

本插件自产，无上游。playbook/patterns/precision/toolchain 与 handbook 的 4 张共 33 张已迁移；
仅 handbook 的 api_catalog、language_reference 保留。

## 非卡片资产

`.txt` / `.json` 等非 Markdown 文件不进 OKF 语料（索引器只收 `.md`），但作为**证据**保留在原目录：
`porter/patterns/fa_a3_multicore_mc_{verify,perf}.json`（pb-56 的 20/20 精度实测数据，卡已迁移至
cannbot-knowledge `examples/fa_a3_multicore_result.md`）。
另有 `runbooks/operator-optimization/cube_required_ops.txt`（cube 必需算子清单，被 brief 按路径下发）。
