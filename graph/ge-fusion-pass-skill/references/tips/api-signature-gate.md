# tip: API 签名核对表（写码前硬门禁 G2）

> 📎 导航落点：`references/pass-development-paradigm.md` §6（API 签名核实）。本文件仍是 G2 核对表字段的**唯一权威定义源**，正文未改动。迁移映射见 `tips/MIGRATION.md`。

**读者**：② 开发。

## 症状

凭经验猜 API 签名就开写，导致参数顺序/返回值/所有权语义错，编译或运行期才暴露，反复返工。

## 硬门禁

**凡源码里要调用的 GE/ES API，写代码前必须列出「API 签名核对表」；没有签名证据不得开始写实现。** 核对表缺任一将调用的 API → 先补文档/头文件证据，不写码。

## 核对表字段（逐项列出）

| 字段 | 说明 |
|---|---|
| API 名 | 函数/方法名 |
| 语言 | C++ / Python |
| 完整签名 | 完整函数/方法签名 |
| 返回值/所有权语义 | 返回类型、谁持有、是否转移 |
| 关键参数 | 顺序、含义、单位 |
| 证据来源 | 文件路径 + 行号/章节 |
| 当前调用方式 | 本 case 代码将如何调用 |

C++ graph/FusionBasePass 常见最低集合：`Graph::GetAllNodes`、`GNode::GetName/GetType/GetAttr/GetInputsSize/GetOutputsSize/GetOutDataNodesAndPortIndexs`、`EsGraphBuilder::CreateInput/BuildAndReset`、涉及的 ES 算子函数、`SubgraphInput/SubgraphOutput/SubgraphBoundary`、`SubgraphRewriter::Replace`、注册宏。Python pass 同理核对 `ge.passes`、`ge.graph`、`ge.es` 的实际可用签名或 probe 结果。

## 证据优先级

1. `$GE_REPO_PATH/docs/zh/api/graph_engine_api/README.md` 后按 C++ `cpp/ge/**` 或 Python `python/ge/**` 分层目录定位（不要假设 `<API名>.md` 位于根目录）；再查 `examples/fusion_pass/` 开发指南与样例、现场 ES API 清单。具体路由见 `references/ge-repo-map.md` §1 / §3。
2. 只有文档缺失、冲突或需确认重载时，才回退 CANN/GE 头文件，并写明回退原因。

## 降级条款（文档与头文件都不可达）

- **不得靠经验猜签名当已证事实。**
- 把无法核实的 API 单列，标注 `【签名未核实，假设 X，待确认】`，写清确认它需要读哪个文档/头文件。
- 门禁允许在**显式标注假设**下产出诊断型/占位实现（如 Python 能力缺失时的诊断型 pass：注册成功、扫描目标结构、打印缺失 API 与 skip reason、**返回 `True`**——Python `run()` 返回 `0` 是**失败**，见 `interface-catalog.md` §二）。
- 禁止把假设当已证事实，禁止"明知签名不确定还当成功写死"。

## 自查

- 每个将调用的 API 都在核对表里有一行、且有证据来源（路径+行号/章节）吗？
- 有没有未核实却当成已证的 API？有→标注假设并单列，或补证据。
