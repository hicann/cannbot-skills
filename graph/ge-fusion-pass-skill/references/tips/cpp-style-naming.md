# tip: 生成 C++/Python pass 的命名与日志风格，以 GE 仓样例为准

> 📎 导航落点：`references/pass-development-paradigm.md` §7（实现）。本文件仍是命名 / 日志风格规范的**唯一权威定义源**，正文未改动。迁移映射见 `tips/MIGRATION.md`。

**读者**：② 开发。

## 适用

生成 `src/*.cpp|*.py` 时的命名法与日志方式。目的：让新增代码读起来和 GE 仓 `examples/fusion_pass/` 的真实样例一致，而不是套用别的算子仓（ops-nn/ops-math）的规范。

## 根因

不同仓的 C++ 风格并不相同：**GE 融合 pass 样例用 snake_case 局部变量**（`graph_builder`、`replace_graph_builder`、`match_result`、`alpha_const`），而 ops-nn/ops-math 仓强制 camelCase。照搬别处的 camelCase 规范到 GE pass 里，会与目标仓、与你要参考的样例源码割裂。**命名规范以你实际要落地的仓的样例为准，不跨仓照抄。**

## 硬性做法（GE 仓约定，写码前对着 `$GE_REPO_PATH/examples/fusion_pass/**/src/*.cpp` 复核）

> glob 用 `**`：样例源码多在 `<组>/<样例>/cpp/src/` 层级，`*/src/*.cpp` 不能可靠覆盖嵌套和扁平两种布局。
> `find "$GE_REPO_PATH/examples/fusion_pass" -path '*/src/*.cpp'` 最稳。

| 类别 | GE 样例约定 | 示例 |
|---|---|---|
| 局部变量 / 参数 | **snake_case** | `graph_builder`、`match_result`、`matched_node`、`data_format`、`alpha_const` |
| 类名 / 结构体 | PascalCase | `FuseMatMulAndAddPass`、`DecomposeGroupedConvToSplitedPass` |
| 重写的框架方法 | PascalCase（框架定义，照抄） | `Patterns`、`Replacement`、`MeetRequirements` |
| 自定义辅助方法 | PascalCase | `IsTensorValueEqualToZero`、`InferShapeAndCheckSupport` |
| 文件名 | snake_case | `fuse_matmul_add_pass.cpp` |
| 框架枚举/宏 | 原样 | `CustomPassStage::kBeforeInferShape`、`REG_FUSION_PASS`、`REG_DECOMPOSE_PASS` |
| `#include` 路径 | **原样，不改** | `#include "es_all_ops.h"`、`#include "ge/fusion/pass/pattern_fusion_pass.h"` |

- **日志用 `std::cout << ... << std::endl;`**（C++）/ `print`（Python），关键分支直接落 stdout——官方样例即如此，验证时无条件可见，不依赖 slog 配置。日志点清单与理由见统一 skill 阶段二“实现规则”。
- 不重命名既有变量、不为"统一风格"重排既有代码；只让**新增**代码贴合样例风格。
- **迁移/参考别的仓的样例时**：命名法切到目标仓（要落 GE 就用 GE 的 snake_case；要落 ops-nn/ops-math 才用它们的 camelCase），不要把源仓风格带过去。

## 自查

- 新增 C++ 的局部变量是不是 snake_case（对齐 GE 样例），没有从别处带来的 camelCase/`kXxx` 局部量？
- 类名/重写方法是不是 PascalCase，`#include` 路径原样未改？
- 关键分支是不是用 `std::cout`/`print` 打了、且带 pass 名便于 grep？
- 写码前是否真的打开过 `$GE_REPO_PATH/examples/fusion_pass/**/src/*.cpp` 对过风格（而不是凭印象）？
