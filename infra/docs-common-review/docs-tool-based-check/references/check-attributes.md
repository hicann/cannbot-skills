# 结果转换属性参考（check-attributes.json / markdownlint-severity.json）

本文档说明 `references/` 下两个配置文件的用途、字段语义与维护方法。二者均由 `scripts/run_check.js` 在结果转换阶段（`convertItems`）读取，用于把 docs-ci-v2.js 输出的原始问题（checkType / message / content / position）映射为本技能的统一 12 字段结果 JSON。

## check-attributes.json（检查项 → 展示属性）

按 checkType（5 项合法检查项名）定义转换属性：

| 字段 | 语义 | 说明 |
| --- | --- | --- |
| rule_name | 规则名称 | 展示用中文名（如"Markdown 语法检查"） |
| rule_type | 规则类型 | 规范性 / 正确性，与统一 JSON 的 rule_type 一致 |
| rule_level | 级别 | markdownlint 固定为 null（按规则号动态查表，见下）；其余 4 项为静态级别 |
| suggestion | 修改建议模板 | 占位符 `{content}` 替换为问题内容片段、`{message}` 替换为脚本原始错误信息 |

- 未知 checkType（理论上不出现）由 run_check.js 按默认属性兜底转换并告警。
- suggestion 中的白名单引导文案与 `.doctools/` 项目级配置的文件名保持一致，修改文件名时须同步更新。

## markdownlint-severity.json（MD 规则号 → 级别）

markdownlint 的严重程度按规则号逐条定级，映射未覆盖的规则号由 run_check.js 回退为"一般"。

**定级口径**（与 SKILL.md"类型与严重程度映射"一致）：

- 严重：直接影响读者获取内容（如 MD011 链接语法错、MD042 空链接、MD051/MD052 锚点相关）
- 提示：可辨识的轻微问题（如 MD012 多余空行、MD047 文件末尾空行）
- 一般：纯规范性问题，按规则影响面归档

## 维护说明

- 两文件仅影响结果展示口径（rule_name/rule_type/rule_level/suggestion），不影响检查项范围与判定逻辑；修改后须重跑 `evals/dataset/static_check_sample.md` 并按实测结果同步 `evals/evals.json` 的 golden 描述。
- 新增检查项时（需先扩展 run_check.js 的 LEGAL_CHECKS 与社区脚本能力），同步在 check-attributes.json 增加对应条目。
- markdownlint-severity.json 未覆盖的新规则号按"一般"回退，无需为每个规则号补齐映射。
