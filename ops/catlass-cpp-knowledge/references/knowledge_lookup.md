# 查询与读取规则

## 知识路径

`--knowledge-root` 优先于环境变量 `CANNBOT_KNOWLEDGE_ROOT`，两者均指向 cannbot-knowledge 仓库根目录。
未指定时，从 `--project-root` 下的 `.cannbot/dependencies/catlass-op-generator/cannbot-knowledge/` 读取。
脚本在仓库根目录后追加 `knowledge/ops/catlass/`，因此不要将参数直接设为该知识子目录。

`initialize` 和 `validate` 只确认知识路径可读且存在卡片，不下载、复制或重建知识。
出现 `CATLASS knowledge route is unavailable` 时，先核对插件依赖安装和上述根目录配置。

## 查询范围

- `--type apis` 查询接口卡，`--type guides` 查询五阶段指南，`--type operators` 查询算子卡。
- `--family` 按 Operator 文件名前缀过滤。别名维护在 [query-vocabulary.yaml](query-vocabulary.yaml) 中；
  例如 `block_sparse_attention` 对应 `sparse-attention`，`sparse_flash_mla` 对应 `sparse-flash-mla`。
- 前缀重叠时采用最长匹配：`flash_attention` 查询前向卡，`flash_attention_grad` 查询反向卡。
  先根据数学语义确认家族，不能把相邻家族卡当成该家族的设计依据。
- 未指定家族时默认只列 `stable` 卡；`--include-draft` 显式纳入 `draft` 卡。
  显式家族查询同时列出该家族的 `stable` 与 `draft` 卡，并保留 `status`；其他状态不返回。
- `--arch` 过滤卡片的 `platforms`：包含 `950` 的架构值映射到 `950`，
  包含 `a2`、`2201` 或 `910b` 的架构值映射到 `a2`，其他值按小写原值匹配。
- `--tag` 可重复指定，要求全部标签匹配；`--text` 按空格拆词，要求所有词出现在文件名、标题、描述或正文中。
  `--compact` 省略结果中的标签和平台详情，保留路径、标题、描述、类型和状态。

## 芯片 Stage 规则

先确认 `workflow.json` 已冻结的 `target_architecture`；值为 `pending` 或未知架构时，
先确认目标产品与对应架构，不得猜测或使用其它架构兜底。

| target_architecture | 对应规则路径 |
|---|---|
| `atlas_a2_a3` | `guides/stage_design_atlas_a2_a3.md` |
| `ascend950` | `guides/stage_design_ascend950.md` |

使用 `scripts/record_knowledge.py query --project-root <workspace> --type guides --arch <target_architecture> --text "Stage 设计规则" --compact`
查询，再将返回的对应规则路径原样传给 `get --project-root <workspace> --path`，完整读取后开始设计。
`--compact` 不改变 `--arch` 过滤；通过 `get` 返回的完整正文核对 `platforms` 和 `status`。
平台列表须包含目标产品，不要求只有一个值，例如 A2/A3 规则同时支持 `a2` 和 `a3`。
不能混用另一架构的规则；目标芯片规则缺失或无法确认时，先修复知识路径或补齐规则，不得跳过规则开始设计。

## 完整读取

`query` 返回的 `path` 相对于 `knowledge/ops/catlass/`。将该路径原样传给 `get --path`，
读取当前阶段所需卡片的完整正文；查询摘要不能替代正文。
`get` 只接受安全的相对 Markdown 卡片路径，拒绝目录越界、符号链接和 `index.md`。

通用算子按实际组件查询 API，读取 draft API 时使用 `--include-draft`；
注意力家族按已确认的 family 和 `workflow.json` 中的 `target_architecture` 查询 Operator 卡，
并核对返回卡片的计算语义。家族查询结果为空表示当前架构没有匹配的家族知识，不得移除 `--arch`
或读取其它架构卡片兜底。记录知识缺口，继续依据冻结的接口和标杆、已完整读取的目标芯片
Stage 规则、目标版本 CATLASS/CANN API 与源码完成设计；家族卡缺失本身不阻塞设计，
不需要等待补卡。draft 内容仍需结合冻结 contract、固定源码和运行证据核对。
知识新增和状态治理在 cannbot-knowledge 仓库完成。
