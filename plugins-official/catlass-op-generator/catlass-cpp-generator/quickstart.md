# CATLASS C++ Linear Attention Generator

独立执行 PR1069 五阶段 CATLASS C++ Linear Attention 工作流。

## 安装

```bash
bash init.sh project codex <project-root>
```

支持 `opencode`、`claude`、`trae`、`cursor`、`codex`、`copilot` 和 `codearts`。
使用 `bash init.sh --check ...` 检查安装，使用 `bash init.sh --uninstall ...` 保守卸载白名单内容。

## 直接分类

```bash
python scripts/select_operator_workflow.py \
  --workspace <workspace> \
  --operator-name catlass_gdn \
  --algorithm-family linear_attention
```

只有 JSON 结果中的 `route` 为 `linear_attention` 才进入本插件。新工程从
`catlass-cpp-interface` 开始；既有专用工程按 `docs/workflow.json` 恢复。

新建工程的 `target_architecture` 初始为 `pending`。离开 interface 阶段前必须按目标环境改为：

- A2/A3：`atlas_a2_a3`，对应 `CATLASS_ARCH=2201`、`Arch::AtlasA2`；
- A5/Ascend950：`ascend950`，对应 `CATLASS_ARCH=3510`、`Arch::Ascend950`。

## 知识查询

```bash
python skills/catlass-cpp-knowledge/scripts/record_knowledge.py initialize --project-root <workspace>
python skills/catlass-cpp-knowledge/scripts/record_knowledge.py query \
  --project-root <workspace> --family linear-attention --compact
```

运行时 bundle 位于 `<workspace>/.catlass-cpp/knowledge/`，初始化只复制缺失文件。
