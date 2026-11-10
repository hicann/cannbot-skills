# Markdown 合规性质检规则 - 规范性检查详细说明

**重要提示：**

- 仅使用本文件中明确定义的规则进行检查
- 规则名称、规则级别必须与本文件完全一致

## 1. LLM 规则检查（SKILL.md 步骤 2.2 由 LLM 逐行执行）

检查文档中术语使用、隐私信息和敏感词的合规性。此类检查仅基于文档内容，不需要读取代码。

| 规则项名称 | 规则名称 | 规则类型 | 描述 | 级别 |
| --- | --- | --- | --- | --- |
| compliance-check | 术语检查 | 规范性 | 检查文档中术语和缩略语的使用规范性 | 提示 |
| compliance-check | 隐私检查 | 规范性 | 检查文档中是否存在敏感信息和不合规内容 | 严重 |
| compliance-check | 敏感词检查 | 规范性 | 检查文档中是否存在敏感词（LLM 与脚本双数据源互补，见下方详细定义） | 提示 |

以上 3 条规则缺一不可：隐私检查与术语检查均无脚本可替代，敏感词检查的 LLM 部分与脚本部分互为补充，**严禁跳过其中任何一条**。

### 术语检查

- 规则编号：1
- 规则项名称：compliance-check
- 规则类型：规范性
- 规则级别：提示
- 参考资料路径：[terms-and-abbreviations.md](./terms-and-abbreviations.md)
- 位置：文档中所有文本内容，但排除代码块和代码片段内的内容。
- 规则：
   1. 术语存在性：术语和缩略语必须在参考资料[术语/缩略语]列和[产品名术语]章节中存在
      - 正例：UnalignRegForLoad（存在参考资料）
      - 反例：nalignRegForLoad（不存在参考资料）
   2. 大小写一致性：术语和缩略语的字母大小写必须与参考资料完全一致
      - 正例：UEFI、AllGather
      - 反例：uefi、allgather

### 隐私检查

- 规则编号：2
- 规则项名称：compliance-check
- 规则类型：规范性
- 规则级别：严重
- 参考资料路径：[sensitive-words-internal.md](./sensitive-words-internal.md)（仅规则 2 使用其「错别字」章节）
- 位置：文档中所有文本内容。
- 规则：
   1. 个人隐私数据检测：严禁出现的隐私信息类型：
      - 工号：华为员工工号
        - 识别格式特征：首字母（可选）+ 8位数字
        - 具体模式：00+6位数字 或 01+6位数字；300+5位数字 或 600+5位数字；wx+5位数字；无首字母+8位纯数字
        - 示例：30012345、wx67890、00876543
      - IP地址：IPv4或IPv6格式
      - 个人姓名：中文姓名或外文姓名
      - 邮箱地址：符合邮箱格式的文本
      - 手机号码：11位手机号或国际电话格式
   2. 拼写错误：
      - 检查范围限定：仅检查参考资料 [sensitive-words-internal.md](./sensitive-words-internal.md) 中「错别字」章节明确列出的错误词
      - 检查逻辑：参考资料「错别字」章节包含"错误词 → 正确词"的映射列表（如"帐户 → 账户"）；检查文档中是否出现参考资料中列出的错误词；如果文档中出现参考资料中列出的错误词，则报错；文档中出现的其他单词，即使有拼写错误但不在参考资料列表中，不进行检查
      - 示例：参考资料中包含"帐户 → 账户"，则检查文档中是否有"帐户"，如有则报错；文档中有"mistake"但参考资料中未列出，则不检查"mistake"是否拼写错误

### 敏感词检查

- 规则编号：3
- 规则项名称：compliance-check
- 规则类型：规范性
- 规则级别：提示
- 参考资料路径：[sensitive-words-internal.md](./sensitive-words-internal.md)
- 位置：文档中所有文本内容。
- 数据来源（双源互补，任一来源检出的问题都必须保留在最终结果中，禁止互相覆盖或删除）：
   - **LLM 逐行检查**：依据本规则参考资料 sensitive-words-internal.md（含错别字、通用敏感词、领域敏感词、非专业用语全部章节），在 SKILL.md 步骤 2.2 与其他 LLM 规则一并逐行执行
   - **脚本机器检查**：依据 references/prohibited-words.json 机器词库，由 scripts/prohibited_words_checker.js 在 SKILL.md 步骤 2.5 执行
   - 两个词库内容不完全相同（如"点击 → 单击"仅存在于 sensitive-words-internal.md），**严禁以脚本（机器词库）未检出为由，删除或否决 LLM 依据参考资料检出的真实问题**，反之亦然
- 规则：
   1. 敏感词检测：禁止使用的敏感词类型（详见参考资料）

## 2. 工具检查规则（SKILL.md 步骤 2.3 ~ 2.7 由脚本执行）

以下规则由 scripts/ 目录下对应脚本执行，脚本输出的 rule_name、rule_type、rule_level 必须与本节定义完全一致，脚本检出的问题直接追加写入最终结果文件：

| 规则项名称 | 规则名称 | 规则类型 | 描述 | 级别 |
| --- | --- | --- | --- | --- |
| compliance-check | 芯片标签的合法性检查 | 规范性 | 检查文档中 NPU 芯片标签是否合法（scripts/tag_checker.js） | 严重 |
| compliance-check | 位置标签的合法性检查 | 规范性 | 检查文档中位置标签（@ref）格式是否合法（scripts/tag_checker.js） | 严重 |
| compliance-check | MD编码格式检查 | 规范性 | 检查文档 MD 编码格式问题（scripts/encoding_format_checker.js） | 严重 |
| compliance-check | table规范 | 规范性 | 检查 Markdown/HTML 表格单元格是否为空（scripts/null_cell_checker.js） | 一般 |
| compliance-check | 单md层级合法 | 规范性 | 检查单个 md 文件标题目录层级是否合法（scripts/catalog_num_checker.js） | 提示 |

> 敏感词检查的脚本部分（scripts/prohibited_words_checker.js，依据 references/prohibited-words.json 机器词库）沿用第 1 节"敏感词检查"定义：规则类型为规范性、级别为提示，与脚本实际输出一致。
