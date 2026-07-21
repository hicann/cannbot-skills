# externel_refs —— 外部参考语料（按需获取，不随仓库分发）

> ⚠️ 本目录原先整包内置了 Ascend 官方算子文档仓与硬件资料（体积大、含图片二进制、且部分文件携带与本仓库不一致的许可证头）。
> 为保持社区插件轻量与许可证合规，**这些外部语料已不再随仓库提交**，改为「按需在本地放置」。
> 本 README 仅作为索引与获取指引保留；`code-performance-advisor` 在缺少这些语料时会优雅降级（跳过 API 参考查阅环节）。

---

## 1. 这里原本放什么

| 子目录 | 内容 | 来源 |
|---|---|---|
| `official_operator_api_introduction/` | Ascend 官方算子 API / 调用 / 开发文档（ops-math / ops-nn / ops-transformer 等域） | Ascend 官方算子文档仓（ops-math-docs、ops-nn-docs、ops-transformer-docs） |
| `hardware/` | 各型号硬件能力与差异（Ascend310P3 / 910B* / CUDA_A100） | Ascend 硬件规格资料 |
| `IdeaPool/` | 历史算子优化经验（diff_analysis、expert_ideas） | 内部经验沉淀 |

## 2. 如何按需启用

1. 从上述上游获取对应文档仓，按下面的目录形态放到本目录：
   - `official_operator_api_introduction/<domain>-docs/docs/zh/op_api_list.md`
   - `hardware/Ascend910B*.md`、`hardware/CUDA_A100.md`
   - `IdeaPool/IdeaPool/<op_name>/{diff_analysis,expert_ideas}.md`
2. 放置时请**去除图片等二进制资产**，并确保引入的文本文件许可证头符合本仓库 `CANN-2.0` 要求（详见仓库根 `OAT.xml`）。
3. 放置后无需额外配置，skill 会自动发现并在需要查 API 时检索本目录。

## 3. 检索优先级（启用后）

1. `**/docs/zh/op_api_list.md`（API 索引）
2. `**/docs/zh/op_list.md`（算子列表）
3. `**/docs/zh/invocation/quick_op_invocation.md`（快速调用）
4. `**/docs/zh/invocation/op_invocation.md`（完整调用说明）
5. 参数/格式/类型约束：`**/docs/zh/context/`（`数据格式.md`、`数据类型.md`、`aclnn返回码.md` 等）
