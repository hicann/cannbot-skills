# 基准回归对齐报告

- 基线（golden）：infra\docs-common-review\docs-skills-benchmark\common_bench.md — 87 条
- 新结果：infra\docs-common-review\docs-skills-benchmark\runs\common_bench.md — 87 条（比对范围内 87 条）
- 判定：✅ 对齐（阻断性差异 0 处）

## 按引擎统计

| 检查引擎 | 基线 | 新结果 | 对齐 | 漏检 | 多检 | 字段变化 |
| --- | --- | --- | --- | --- | --- | --- |
| AICheck | 1 | 1 | 1 | 0 | 0 | 0 |
| codespell-check | 1 | 1 | 1 | 0 | 0 | 0 |
| compliance-check | 12 | 12 | 12 | 0 | 0 | 0 |
| filename-check | 1 | 1 | 1 | 0 | 0 | 0 |
| punctuation-check | 69 | 69 | 69 | 0 | 0 | 0 |
| resource-existence-check | 1 | 1 | 1 | 0 | 0 | 0 |
| tag-closed-check | 2 | 2 | 2 | 0 | 0 | 0 |

## ℹ️ 说明

- link_validity-check 差异受网络环境影响（外链可访问性随环境变化），已标注【网络相关】，请人工确认后再定性
- 对齐口径：阻断=漏检/多检/严重程度/问题类型/行号/文件变化；上下文/描述/建议措辞差异默认仅提示（LLM 表述波动）
- 退出码：0=对齐；1=不对齐（存在阻断性差异）；2=输入错误

对齐判定: 对齐（exit 0）
