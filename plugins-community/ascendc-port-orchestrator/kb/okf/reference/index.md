# reference

参考层：API 参考卡与长文指南。按 **溯源制度** 分 bundle —— bundle 名决定 `knowledge_lint.py`
适用哪套溯源规则（见 `kb/README.md` §2）。

## [asc-devkit-vendored](asc-devkit-vendored/index.md) — 511 张

从 `gitcode.com/cann/asc-devkit` 搬运的上游 API 文档（经 `Ascend/agent-skills` PR #103 的
`ascendc-operator-A5-migration` skill 二次搬运后导入本插件）。**`-vendored` 后缀表示这是搬运副本，
不是官方 bundle**：确切的上游 commit 已不可考（PR 分支已删除，上游文档树已改版），
因此 `resource` 记录的是上游文件路径而非固定 commit 的永久链接。

- `api/` 490 —— 逐 API 参考，23 个功能族：
  `reg_vector` 124、`vector_compute` 101、`cube_datamove` 48、`tensor_layout` 37、
  `class_api` 26、`sys_var` 25、`vector_datamove` 19、`reg_load` 15、`sync` 15、
  `cube_compute` 13、`reg_store` 10、`scalar_compute` 10、`tensor_pointer` 10、
  `simd_atomic` 7、`tensor_datamove` 7、`cache_ctrl` 5、`struct` 5、`tensor_atom` 4、
  `tensor_struct` 3、`tensor_coord` 2、`tensor_tile` 2、`misc` 1、`tensor_cube_compute` 1
- `guide/` 21 —— 长文指南：`programming_model` 9、`api_overview` 7、`compat_migration` 5

## [porter](porter/index.md) — 35 张

本插件自产，无上游。

- `patterns/` 11 —— 代码模板与实测记录（FA 类、GMM SwiGLU、a3 多核）
- `playbook/` 9 —— arch22→arch35 迁移方法论 L1–L5 + ops-nn A5 产物布局
- `precision/` 7 —— 精度标准与测试流程
- `handbook/` 6 —— API 目录 / 语言参考 / Roofline / SIMD/SIMT 决策
- `toolchain/` 2 —— msprof / NPU UT

## 非卡片资产

`.txt` / `.json` 等非 Markdown 文件不进 OKF 语料（索引器只收 `.md`），但作为**证据**与引用它们的卡
同目录存放：`porter/patterns/fa_a3_multicore_mc_{verify,perf}.json`（pb-56 的 20/20 精度实测数据）。
另有 `runbooks/operator-optimization/cube_required_ops.txt`（cube 必需算子清单，被 brief 按路径下发）。
