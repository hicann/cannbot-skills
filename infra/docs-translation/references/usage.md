# 翻译skill使用方法

本文档说明翻译skill的部署与启动方法，以及`exclude_docs.json`排除配置的修改方法（排除配置仅非算子仓使用）。完整的执行流程与规则见SKILL.md。

## 部署skill

将skill整个目录下载到本地任意位置存放（如`D:\skills\docs-translation\`），保持目录内相对结构不变：

```tree
<skill_dir>/
├── SKILL.md                       # skill主文件（执行流程与规则）
├── exclude_docs.json              # 排除配置（仅非算子仓读取）
└── references/
    ├── translation-style-guide.md # 翻译标准（怎么译）
    ├── quality_check.md           # 质量检查清单（查什么）
    └── usage.md                   # 本文档
```

`<skill_dir>`指SKILL.md所在目录，与待翻译仓库的位置无关。

## 排除配置（仅非算子仓）

skill按仓库类型采用不同的排除方式：

- **算子仓**：仓库名一般以`ops-`开头（如ops-nn、ops-cv、ops-transformer），按固定排除规则扫描（一级`docs/`、`experimental/`目录整体排除；文件名匹配`aclnn*.md`、`torchapi_*.md`、`*Pass.md`；`.git`目录；已有译稿），**不读取exclude_docs.json**，无需配置
- **非算子仓**：如metadef、ge、pypto、runtime、driver等仓库，从`<skill_dir>/exclude_docs.json`读取排除配置；配置文件不存在时视为无排除项，整个仓库进行翻译

exclude_docs.json修改说明（仅非算子仓需要关注）：

- `repoName`：仓库名。skill会将用户输入的本地仓库路径与`repoName`进行模糊匹配（路径中包含repoName关键字即匹配成功，匹配到多个时取关键字最长的配置项）；匹配成功则按`excludeDocsPaths`排除翻译，未匹配则无文件排除，整个仓库进行翻译
- `excludeDocsPaths`：需要跳过翻译的docs子路径列表，支持自定义配置，可根据仓库实际情况调整，用于过滤无需翻译的内容。文件内配置仅为示例，请结合各仓库自身情况修改。各仓库需要排除的目录参考[白名单](https://gitcode.com/cann/docs/blob/master/specification/repo_zh_en_ci_check_whitelist.md)

## 启动翻译任务

在opencode中输入类似如下信息，启动skill翻译任务：

```text
根据<skill_dir>\SKILL.md skill，处理xxx仓或yyy.md文件。
```

其中，`<skill_dir>`为skill所在目录，xxx为用户clone到本地的组件仓库，yyy为仓中的md文档。

skill启动后会通过交互式提问工具（question）索取仓库路径，输入已clone到本地的仓库绝对路径（如`D:\WorkSpace\metadef_trunk`）即可。skill不会通过远程地址推断本地仓库路径，也不会自行克隆仓库，本地仓库路径一律由用户提供。

## 执行结果与提交

**执行结果**：

- 译稿：按SKILL.md「译稿存放规则」生成（`zh`文件夹下的文档译至同级`en/`目录下的同名文件，其余文档译至同目录`<原文件名>_en.md`）
- 翻译报告：生成在仓库同级目录，文件名形如`<仓库名>_translate_report_时间戳.md`，仅备查，不参与提交

**提交**：skill**默认不提交**到远程仓库。如需将翻译结果提交到远程仓库并创建PR，需在任务描述中明确要求（如补充"翻译完成后，提交到远程仓库"）。提交时新建分支并将全部译稿合并为单个提交，无origin推送权限时通过fork创建PR（见SKILL.md执行流程第12步）。
