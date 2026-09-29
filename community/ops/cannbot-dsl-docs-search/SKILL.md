---
name: cannbot-dsl-docs-search
description: "从指定 CANNBotDSL wheel 提取 cannbotdsl 顶层公开 Python API，按职责分类，并在 cannbot-dsl-docs-search 目录生成 YAML 索引和分类 Markdown 文档。当需要建立或刷新指定 wheel 的顶层 API 参考时使用；不安装或执行 wheel。"
---

# CANNBotDSL API 文档检索与参考生成

本 Skill 从调用方指定的 wheel 生成分类 API 文档和索引，供查找公开接口、签名及约束时使用。

产出范围严格限定为可通过 `cannbotdsl.<名称>` 直接访问的顶层公开函数、类、类型别名和常量，以及这些顶层公开类的公开成员。仅能从 `cannbotdsl.ops.*`、`cannbotdsl.types.*` 等子模块访问的符号不进入产物。模块命名空间本身不作为一条 API；脚本不安装或导入目标包。

## 输入

- 调用方指定的 CANNBotDSL wheel。
- 调用方指定的产物根目录。

## 运行

在 skill 目录中运行脚本。wheel 和产物根目录均使用相对于当前工作目录的路径：

```bash
python3 scripts/inspect_wheel_api.py \
  --wheel wheels/cannbotdsl.whl \
  --package cannbotdsl \
  --output-dir build/api-docs
```

可选开关：

- `--include-source`：在各分类 Markdown 文档中附带有限源码片段。

不要传入 symbol、keyword、limit 或请求文件。该脚本每次都遍历选定范围内的全部 API。

## 提取规则

脚本不执行 wheel 中的代码，只进行确定性静态解析：

1. 读取 wheel ZIP 和唯一 `.dist-info/METADATA`，确认可解析的输入范围；
2. 读取目标包的 `.py`，没有对应源码时使用 `.pyi`；内部模块也参与导出链解析，默认只输出从公开模块重导出的内部定义；
3. 提取函数、异步函数、类、方法、模块级类型别名和常量的声明，以及 decorator、docstring、`assert` 和 `raise` 约束；
4. 解析绝对/相对 import、别名、星号导入、静态 `__all__`，以及可由字面量分支证明的 `__getattr__` 延迟导出；
5. 沿重导出链恢复每个 API 的公开 import 路径；
6. 依据 API 的定义模块分类，而不是依据可能被扁平化的最短公开路径。

内部模块仍参与导出链解析，使定义在 `core/`、`_mlir/` 或 `aicpu/`、但由 `cannbotdsl` 顶层明确重导出的 API 不会丢失；没有顶层公开路径的定义一律排除。运行摘要记录解析定义总数、非顶层排除数、私有成员数、二进制成员数和 AST 错误数，使覆盖范围可以审计。

## 分类

分类依据定义模块前缀，保持跨重导出层次的稳定性：

- 语言、JIT 与 Kernel；
- Tensor、类型与布局；
- Buffer、Channel 与数据流；
- 数据搬运与格式转换；
- Vector/寄存器操作；
- Scalar 与 SIMT；
- Cube、Matmul 与 Convolution；
- 同步与缓存控制；
- 分布式与通信；
- 架构与平台信息；
- Native 打包与运行时；
- 其他算子、内部实现和无法归类的 API。

分类的完整前缀表、匹配优先级、可见性边界和维护方法见 [references/api-categories.md](references/api-categories.md)。评审兜底类别或修改 `CATEGORY_RULES` 前必须先读取该参考。

分类规则是 CANNBotDSL 领域规则。新增顶层 API 落入兜底类别时，应先保留完整顶层 API，再根据其定义模块职责更新分类映射，不能为了消除兜底而按名称猜分类。

## 输出

`--output-dir` 是调用方指定的产物根目录。生成器固定在该目录下创建并管理 `cannbot-dsl-docs-search/`，每次运行生成：

```text
cannbot-dsl-docs-search/
├── api_index.yaml
└── api-info/
    ├── language-and-kernel.md
    ├── tensor-types-and-layout.md
    ├── memory-and-dataflow.md
    └── ...
```

- `api-info/`：主要文档产物；每个实际包含 API 的类别对应一份 Markdown，包含该类 API 的签名、说明、公开导入路径、decorator 和参数校验/异常约束；
- `api_index.yaml`：辅助结构化索引；一项一个 API，包含分类、符号、类型、定义路径、签名、导入路径、decorator、约束数量和文档摘要。

`references/api-categories.md` 是 skill 自身维护的分类规则，不属于运行产物，因此不复制到生成的 `cannbot-dsl-docs-search/`。分类文档的文件名使用 `CATEGORY_RULES` 中稳定的类别 ID；顶层 API 数量为零的类别不生成空文档。

API 文档不展示 wheel 内 `.py` 文件路径和行号。源码位置属于实现审计信息，容易随重新打包变化，不是公开 API 文档的稳定内容。生成器仍在解析期间使用 AST 位置信息，但不会把它写入 Markdown 或 YAML。

生成文档的标题、栏目、统计、分类说明和边界说明使用中文。API 符号和 Python 签名保持源码原文。文档只提取 wheel docstring 的首段作为功能摘要，并把其中常见 reStructuredText 标记规范化为 Markdown；完整原始 docstring、issue 记录、硬件实验过程和内部实现长说明不进入产物。文档保留输入 wheel 的来源信息和摘要，不记录输入文件或产物目录的绝对路径。

重跑时会刷新 `cannbot-dsl-docs-search/api_index.yaml` 和 `cannbot-dsl-docs-search/api-info/*.md`，并删除 `api-info/` 中已经不再产生的旧分类文档。不要把人工维护文档放进生成器管理的 `cannbot-dsl-docs-search/`。

## 分类规则参考

分类原则、前缀映射、匹配顺序和维护方法见 [references/api-categories.md](references/api-categories.md)。`references/` 只存放 skill 自身维护的规则，不存放某次运行生成的 API 产物。API 产物始终写入调用方通过 `--output-dir` 指定的相对目录。

## 完成检查

运行成功后读取脚本 stdout，并检查 Markdown 与 YAML：

- `status` 为 `ok`；
- 指定 wheel 可解析，且产物明确标识输入来源；
- `api_count`、`category_count` 非零；
- `parse_error_count` 为零，或已明确记录无法解析的成员；
- `static_export_unresolved_count` 为零；模块命名空间导出只计入 `module_namespace_export_count`，不伪装成 API；
- 同一公开符号同时由真实定义和模块级别名暴露时，只生成一条 API；函数、类和方法优先于别名，并在 `symbol_collision_count` 记录合并数；
- 非字面量动态 `__getattr__` 和没有 Python stub 的二进制扩展成员已经作为静态分析边界核查；
- `cannbot-dsl-docs-search/api-info/*.md` 中的 API 总数与 `cannbot-dsl-docs-search/api_index.yaml` 一致；
- `other`、`other-operations` 或 `internal-implementation` 中的条目符合本次范围，没有因分类失败而丢失。

动态生成的导出、运行时 monkey patch 和没有 Python stub 的二进制扩展符号无法由 AST 完整恢复。必须在 inventory 和产物说明中保留这一边界，不得把静态目录描述为运行时反射结果。
