---
name: catlass-cpp-knowledge
description: 查询和校验 CATLASS C++ OKF v0.2 知识 bundle。适用于查询 CATLASS 模板 API、五阶段工作流知识和按 family 组织的算子算法知识；仅参考 catlass-dsl-knowledge 的形式，不处理 Python DSL 或自动 learned 入库。
---

# CATLASS C++ Knowledge

内置 bundle 位于插件 `knowledge/`，运行时位于项目
`.catlass-cpp/knowledge/`。业务分区严格为 `catlass/`、`workflow/` 与 `operator/<family>/`。

先初始化，再 compact 查询，最后只读取选中的安全相对路径：

```bash
python scripts/record_knowledge.py initialize --project-root <workspace>
python scripts/record_knowledge.py query --project-root <workspace> --text BlockMmad --compact
python scripts/record_knowledge.py get --project-root <workspace> --path catlass/block-mmad.md
```

新增或修改 concept 前，先完整读取
[公共入库标准](references/common-entry-standards.md)，再根据目标分区读取
[CATLASS 入库标准](references/catlass-entry-standards.md)、
[Workflow 入库标准](references/workflow-entry-standards.md) 或
[Operator 入库标准](references/operator-entry-standards.md)。

`validate` 校验 OKF 版本、三个固定分区、frontmatter、来源脚注、索引、query vocabulary 和 PR1069
抽取 manifest；`reindex` 只重建三个分区索引。`record` 固定返回 unsupported，不创建
`learned/`。知识是设计输入，不能替代用户 contract、固定源码、CPU golden 或实际验收证据。
