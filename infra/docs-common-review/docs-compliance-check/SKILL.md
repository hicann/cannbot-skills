---
name: docs-compliance-check
description: |
  对Ascend生态开发文档、API文档等Markdown（MD）文档进行合规性质检，基于预设规则检测文档中的规范性问题。适用于对"一段Markdown内容、一个Markdown文档、多个Markdown文档（包输入一个文档目录）"三种场景下Markdown内容的语法规范性检查和内容正确性检查，并输出结构化问题列表及修改建议。当用户提到以下场景时触发：
  - 检查/质检/审核 Markdown 文档是否符合合规性规范
  - Markdown 内容规范性检查
  - Markdown 文档术语、表格、隐私合规性检查
  - 审查/检查 Markdown 文档合规性
  触发词：合规检查、术语检查、敏感词检查、标签检查、编码格式检查、table规范、目录层级检查
  仅承接用户点名的合规类专项检查；通用质检（多检查项组合、全量模式、门禁模式）请使用统一入口 docs-common-review。
version: 1.0.0
metadata:
  author: Huawei Technologies Co., Ltd.
  category: utility-doc
---

# 生态资料合规性质检技能

本 skill 用于对 Markdown 文档进行合规性质检，基于预设规则检测文档中的规范性问题。

## 重要原则（必须严格遵守，防止幻觉）

### 1. 仅使用给定规则

- **只能**使用 `references/rules.md` 中明确定义的规则
- **严禁**创造新规则、修改规则名称或规则级别
- 规则名称必须与规则文件中的完全一致（如"术语检查"、"隐私检查"）

### 2. 每项检查后追加写入 JSON 文件

- **每项检查完成后**：立即将结果追加写入 JSON 文件
- **质检期间**：禁止输出任何解释性文字、总结、表格
- **最终输出时**：保证输出到文件，无需输出到控制台
- 输出格式**只能**是标准 JSON

### 3. 输出前验证最终质检结果

- **只报告**真实存在问题的检查项
- 无问题的检查项**严禁报告**

### 4. 多子项规则独立报告原则

- 当质检规则包含多个子项（如"隐私检查"包含3个检查维度）时，**每个子项的问题必须独立报告**
- **禁止**将同一规则的多个问题合并为一个问题项
- 每个问题位置需要独立的 line_number 和 snippet
- 示例：如果"术语检查"在文档中有3处术语错误，应报告独立的问题项，每个指向不同的 line_number

### 5. 双源互补、禁止删除已检出的真实问题

- LLM 规则检查（步骤 2.2，依据 `references/rules.md` 第 1 节及其引用的参考资料）与脚本检查（步骤 2.3 ~ 2.7，依据 `references/rules.md` 第 2 节）相互独立、互为补充
- "敏感词检查"存在两个数据源：LLM 依据 sensitive-words-internal.md（全部章节），脚本 prohibited_words_checker.js 依据 references/prohibited-words.json 机器词库；两个词库内容不完全相同，**任一来源检出的真实问题都必须写入最终结果**
- **已写入结果文件的真实问题禁止删除**；仅当确认误报（如命中参考资料中的例外说明、词在代码块内等）才可剔除，且必须在上下文中保留剔除依据
- **严禁**以"脚本未检出"为由删除 LLM 依据规则文件检出的真实问题，**严禁**以"LLM 未检出"为由删除脚本写出的结果，反之亦然

## 输入

1. **待检查的内容**：用户输入需要检查质量的 Markdown 内容，支持以下三种形式：
   - 一段 Markdown 内容（直接输入文本）
   - 一个 Markdown 文档的绝对路径
   - 多个 Markdown 文档路径（英文逗号分隔）或一个文档目录路径

2. **`--skip-tag` 跳过标签检查**（可选）
   - 格式：`--skip-tag`（无值参数，存在即生效）
   - 作用：跳过标签检查步骤（步骤2.3），仅执行规则检查
   - 示例：`D:\markdown\doc.md --skip-tag`

3. **`--skip-encoding` 跳过MD编码格式检查**（可选）
   - 格式：`--skip-encoding`（无值参数，存在即生效）
   - 作用：跳过MD编码格式检查步骤（步骤2.4），仅执行规则检查
   - 示例：`D:\markdown\doc.md --skip-encoding`

### 示例

```
# 全部检查
/docs-compliance-check "文档路径"

# 跳过标签检查
/docs-compliance-check "文档路径" -skip-tag
```

## 输出

输出一个质检结果JSON文件

- 默认输出目录：当前项目根目录下创建 `/output/` （如果目录不存在则先创建目录）
- 文件命名规则：`compliance-markdown-check-result.json`

### 质检结果统一 JSON 格式

```json
[
  {
    "rule_group": "xxx",
    "rule_item": "xxx",
    "rule_type": "xxx",
    "rule_name": "xxx",
    "snippet": "xxx",
    "description": "xxx",
    "rule_level": "xxx",
    "file_name": "xxx",
    "line_number": "xxx",
    "location": "xxx",
    "suggestion": "xxx"
  }
]
```

#### 输出字段说明

| 字段名 | 类型 | 描述 | 要求 |
| --- | --- | --- | --- |
| rule_group | string | 问题大类 | 固定为"通用规则" |
| rule_item | string | 规则项名称 | 必须与规则文件中的规则项名称一致 |
| rule_type | string | 规则类型 | 必须与规则文件中的规则类型一致 |
| rule_name | string | 规则名称 | 必须与规则文件中的规则名称一致 |
| snippet | string | 问题内容片段 | 直接引用文档内容 |
| description | string | 问题的具体描述 | 清晰说明问题，不包含行号 |
| rule_level | string | 级别 | 必须与规则文件中的级别一致 |
| file_name | string | 文档名称 | 必须是用户输入的Markdown文档名称 |
| line_number | string | 问题行号 | 必须从带行号的文档内容中直接获取 |
| location | string | 问题所在的行链接 | vscode://file/绝对路径:行号 |
| suggestion | string | 问题的具体修改建议 | 直接给出正确形式，不包含行号 |

## 执行流程

### 核心原则：按需读取，逐任务执行

- 待检查文档按"逐文件 / 分批"处理，禁止一次性把整个大目录全读入上下文；
- 每个文件可在不同步骤、不同规则下多次按行号回读；
- 仅"同行已不好走内对同一文件同一区域无意义的重复读取"才需避免。

在执行任何步骤前，在对话中先输出当前检查点：[CHECKPOINT: 开始]

**规则**：如果前一个CHECKPOINT未输出，禁止执行后一步骤。

### 第 1 步：数据准备

#### 1.1 解析参数与读取配置

- 获取用户输入中的 `--skip-tag` 参数（如果存在），表示跳过标签检查
- 检测用户输入内容是否为文件路径：
    - 如果是 `.md` 文件或者目录路径，则为文件路径模式
    - 如果不是，作为 Markdown 内容模式处理
- **文件路径模式**：直接使用文件路径作为处理文档路径
- **Markdown 内容模式**：
    - 创建 `output/compliance/temp/auto_temp.md` 文件
    - 将用户输入的Markdown内容写入该文件
    - 将该文件的绝对路径作为后续处理的文档路径
- **仓名获取方式**：仓名即当前项目根目录的文件夹名称。从项目根目录的绝对路径中提取最后一级目录名作为仓名。例如项目根目录为 `D:\workspace\branch\cloud`，则仓名为 `cloud`。将仓名保留在上下文中供步骤2.3使用

**检测是否是Markdown内容的标准**：

- 没有明显文档路径特征（如 `D:/...、/...、./...等`）
- 不包含文件扩展名或路径分隔符
- 极长的内容（超过单行命令行输入长度）

#### 1.2 读取规则并筛选

- **此时读取**规则文件 `references/rules.md`，以及规则中引用的参考文件（如 `terms-and-abbreviations.md`、`sensitive-words-internal.md` 等，**仅在规则需要时读取**）
    - `terms-and-abbreviations.md`术语表需要从文档内容中获取**昇腾术语表**、**产品名术语**、**其它中文术语**三个章节的内容，并从内容中读取三个官方页面或者文件的内容。
    - **术语数据源有效性验证（强制）**：逐一核验三个术语数据源的实际返回内容，必须确认取到的是术语表格/词条数据本体；若返回内容是 JS 提示页（如"Please Enable JavaScript and Cookie"）、错误页、登录页或空内容，判定该数据源**获取失败**。
    - **数据源获取失败时的处理**：(1) 记录获取失败的数据源名称；(2) 仅基于成功获取的数据源执行术语检查（全部失败则术语检查无法执行）；(3) 在第 3 步汇总输出时向用户如实说明数据源缺失情况与影响范围。**严禁**在数据源获取失败、术语检查未实际执行或未完整执行时，输出"术语检查：未发现问题"或其他无问题的结论。
- 将读取到的规则保留在上下文中直接传递给后续步骤

#### 1.3 生成文档列表

- 读取当前技能目录下 `assets/exclude-files.json` 中的排除文件列表
- 根据文档路径生成待检查文件列表：
    - 单文件：直接加入列表
    - 目录：递归遍历该目录下所有 `.md` 文件，排除白名单中的文件名
- 将文档列表保留在上下文中

#### 1.4 批量读取待检查文档

- 生成文档列表后步预读全文。步骤 2 在处理每个文件时，按需用带行号方式读取该文件，处理完一个文件再处理下一个。
- 将带行号的文档内容保留在上下文中

#### 1.5 清理历史结果文件

- 检查 `output/compliance-markdown-check-result.json` 文件是否存在
    - 如果存在，则清空文件内容
    - 如果不存在，则创建该文件

完成后必须在对话中输出 [CHECKPOINT: 数据准备完成]

### 第 2 步：执行分析

**脚本执行顺序（强制，禁止调整）**：步骤 2.3 ~ 2.7 的检查脚本必须严格按下表顺序**依次**执行，禁止调整顺序、禁止并行执行、禁止因用户强调"重点检查某项"而改变顺序；某步骤被用户参数跳过（`--skip-tag`/`--skip-encoding`）时，按编号顺序继续执行下一个未跳过的步骤。每个脚本执行完毕必须先输出对应 CHECKPOINT，再执行下一个脚本。脚本检出的结果（`rule_name`/`rule_type`/`rule_level`）以 `references/rules.md` 第 2 节定义为准。

| 执行顺序 | 步骤 | 脚本 | 对应规则（rules.md 第 2 节） |
| --- | --- | --- | --- |
| 1 | 2.3 | scripts/tag_checker.js | 芯片标签的合法性检查、位置标签的合法性检查 |
| 2 | 2.4 | scripts/encoding_format_checker.js | MD编码格式检查 |
| 3 | 2.5 | scripts/prohibited_words_checker.js | 敏感词检查（机器词库部分） |
| 4 | 2.6 | scripts/null_cell_checker.js | table规范 |
| 5 | 2.7 | scripts/catalog_num_checker.js | 单md层级合法 |

#### 2.1 基础准备

- 确保 `output` 目录存在

完成后必须在对话中输出 [CHECKPOINT: 基础准备完成]

#### 2.2 批量检查文档（LLM 规则检查）

本步骤由 LLM 逐条执行 `references/rules.md` 第 1 节定义的**全部 3 条规则**，缺一不可：

| 规则名称 | 数据来源 | 强制要求 |
| --- | --- | --- |
| 术语检查 | terms-and-abbreviations.md 引用的三个官方术语源 | 数据源获取失败时按步骤 1.2 的失败处理执行，禁止假报"未发现问题" |
| 隐私检查 | rules.md 隐私检查定义（工号/IP/姓名/邮箱/手机号 + 错别字映射） | **无对应脚本，只能由 LLM 在本步骤逐行执行**，漏检即为执行不完整 |
| 敏感词检查 | sensitive-words-internal.md 全部章节 | 与步骤 2.5 脚本（prohibited-words.json）双源互补，本步骤不得因脚本存在而跳过或缩小范围 |

对于每个待检查文档：先用带行号方式读取本文件内容，逐条应用规则；行号须从本次读取的带行号内容中获取。

1. LLM 规则集合固定为上表 3 条规则，**严禁跳过其中任何一条**（仅当用户明确要求跳过某条规则时才可跳过该条）
2. 对于每个规则：
    - **读取规则描述**：直接从上下文中的规则数据获取
    - **读取参考资料**：直接从上下文中的参考资料获取（如果该规则引用了参考文件）
    - **执行检查逻辑**：根据规则描述严格按照 **逐字逐行** 检查文档内容，不允许根据常识或经验进行扩展性检查
    - **检查结果格式**：检查出来的问题按照 **质检结果统一 JSON 格式** 的要求输出
    - **检查结果验证**：检查每个问题的描述和建议是否与发现问题的规则相互匹配
        - 如果不匹配，则剔除该问题结果
    - **保留结果**：将检查结果保存到 `output/compliance-markdown-check-result.json` 文件中
        - 如果是非问题或者未发现问题，禁止将结果保存到 `output/compliance-markdown-check-result.json` 文件中
3. 每处理完一个文件的一条规则，必须按"逐行扫描执行规范"第 4 节追加 `[COVERED] <file_name> | <rule_name>` 记录；**全部 [COVERED] 记录齐全（文档列表 × 3 条 LLM 规则）后才允许进入步骤 2.3**

完成后必须在对话中输出 [CHECKPOINT: 批量检查文档完成]

#### 2.3 执行标签检查脚本

- 如果步骤1中解析到 `--skip-tag` 参数，则跳过本步骤，直接输出 [CHECKPOINT: 标签检查完成]
- 调用当前技能目录下 `scripts/tag_checker.js` 脚本
- 命令格式：`node <技能目录绝对路径>/scripts/tag_checker.js <md文件绝对路径> <项目根目录绝对路径>/output/compliance-markdown-check-result.json <仓名>`
- `npus.json` 配置文件的路径脚本内部已通过 `__dirname` 自动定位，无需额外传参
- 仓名从步骤1上下文中获取

完成后必须在对话中输出 [CHECKPOINT: 标签检查完成]

#### 2.4 执行MD编码格式检查脚本

- 如果步骤1中解析到 `--skip-encoding` 参数，则跳过本步骤，直接输出 [CHECKPOINT: MD编码格式检查完成]
- 调用当前技能目录下 `scripts/encoding_format_checker.js` 脚本
- 命令格式：`node <技能目录绝对路径>/scripts/encoding_format_checker.js <md文件绝对路径> <项目根目录绝对路径>/output/compliance-markdown-check-result.json`

完成后必须在对话中输出 [CHECKPOINT: MD编码格式检查完成]

#### 2.5 执行敏感词检查脚本

- 调用当前技能目录下 `scripts/prohibited_words_checker.js` 脚本
- 命令格式：`node <技能目录绝对路径>/scripts/prohibited_words_checker.js <md文件绝对路径> <项目根目录绝对路径>/output/compliance-markdown-check-result.json`

完成后必须在对话中输出 [CHECKPOINT: 敏感词检查完成]

#### 2.6 执行table规范检查脚本

- 调用当前技能目录下 `scripts/null_cell_checker.js` 脚本
- 命令格式：`node <技能目录绝对路径>/scripts/null_cell_checker.js <md文件绝对路径> <项目根目录绝对路径>/output/compliance-markdown-check-result.json`

完成后必须在对话中输出 [CHECKPOINT: table规范检查完成]

#### 2.7 执行Markdown目录层级检查脚本

- 调用当前技能目录下 `scripts/catalog_num_checker.js` 脚本
- 命令格式：`node <技能目录绝对路径>/scripts/catalog_num_checker.js <md文件绝对路径> <项目根目录绝对路径>/output/compliance-markdown-check-result.json`

完成后必须在对话中输出 [CHECKPOINT: Markdown目录层级检查完成]

### 第 3 步：汇总输出

- 汇总 `output/compliance-markdown-check-result.json`：LLM 规则检查（2.2）与脚本检查（2.3 ~ 2.7）的结果均须完整保留，禁止互相删除（见重要原则第 5 条）
- 若步骤 1.2 存在获取失败的术语数据源，必须在本步如实向用户说明失败的数据源名称与影响范围（如"昇腾术语表获取失败，本次术语检查仅基于其余数据源执行"），禁止隐瞒

完成后必须在对话中输出 [CHECKPOINT: 汇总输出完成]

### 第 4 步：清理临时文件

- 删除 `output` 目录下的 `compliance` 文件夹
- 删除 `output/tag_check_result.json` 文件
- 删除 `output/encoding_format_check_result.json` 文件
- 删除 `output/prohibited_words_check_result.json` 文件
- 删除 `output/null_cell_check_result.json` 文件
- 删除 `output/catalog_num_check_result.json` 文件

完成后必须在对话中输出 [CHECKPOINT: 清理临时文件完成]

## 逐行扫描执行规范（强制遵守）

### 1. 逐行扫描顺序

- 对每个文档按行号从小到大逐行检查
- 每行从左到右逐字符/逐词扫描
- 发现问题时记录后必须继续向右扫描当前行剩余内容，不得中断当前行的扫描
- 只有当当前行从左到右完整扫描完毕后，才能进入下一行

### 2. 同行穷举规则

- 当在某行发现规则违规时，必须继续检查该行剩余内容是否还存在同类或其他违规
- 禁止因已发现一个问题而跳过同行后续内容
- 每行必须从头到尾完整扫描后才能进入下一行
- 同一行同一规则可以命中多个问题，每个问题必须单独记录

### 3. 结论验证规则

- 对每条检查结论（合规/不合规），必须在原文中定位到具体字符后再判定，**行号必须从步骤1带行号的文档内容中直接获取，禁止凭记忆心算或估算行号**
- 禁止凭对原文的印象或记忆下结论
- 特别是标注"合规"时，必须在上下文中明确记录判定依据（如：原文此处为XXX，符合规则YYY的ZZZ要求）
- 如果无法在原文中精确定位对应内容，必须回溯原文重新确认，禁止跳过
- **记录问题行号时，必须从步骤1已读取的带行号文档内容中直接获取行号，禁止通过心算或人工计数推算行号。如果上下文中带行号的文档内容已被截断，必须重新使用带行号方式读取原文档对应区域来确认行号**

### 4. 文件x规则覆盖记录（完成判据）

- 处理每个文件每条规则后，再对话中追加一行：
    [COVERED] <file_name> | <rule_name>
- 全部完成条件：文档列表中每个文件 x 筛选规则集合中的每条规则，都有对应的 [COVERED] 记录
- 该记录是"检查已完整覆盖"的唯一判据

## 常见幻觉陷阱（避免！）

| 陷阱 | 错误做法 | 正确做法 |
| --- | --- | --- |
| 创造规则 | 添加规则文件中不存在的检查项 | 仅使用规则文件中明确定义的规则 |
| 规则名称错误 | 使用"术语规范"代替"术语检查" | 使用规则文件中的"术语检查" |
| 扩展检查范围 | 根据常识添加规则未描述的检查项 | 严格按照规则描述的检查对象和范围执行 |
| 合并多子项问题 | 将同一规则的多处问题合并为一个问题项 | 每个问题位置独立报告，使用独立的行号 |
| 遗漏其他位置的错误 | 只报告第一处错误，忽略同一规则的后续错误 | 全面检查文档，同一规则的所有问题都要独立报告 |
| 输出多余内容 | 输出总结、表格、解释 | TL;DR, 仅输出 JSON |
| 心算行号 | 凭记忆或计数推算问题行号 | 从带行号的文档内容中直接获取行号 |

## 文件结构

```
docs-compliance-check/
├── SKILL.md (本文件)
├── references/
│   ├── rules.md (检查规则定义：第1节 LLM 规则、第2节工具检查规则)
│   ├── prohibited-words.json (机器敏感词词库，prohibited_words_checker.js 使用)
│   ├── sensitive-words-internal.md (敏感词和不规范用语（对内），LLM 敏感词检查依据)
│   └── terms-and-abbreviations.md (术语和缩略语)
├── scripts/
│   ├── tag_checker.js (标签检查脚本)
│   ├── catalog_num_checker.js (Markdown目录层级检查脚本)
│   ├── encoding_format_checker.js (MD编码格式检查脚本)
│   ├── null_cell_checker.js (table规范检查脚本)
│   ├── prohibited_words_checker.js (敏感词检查脚本)
│   └── npus.json (NPU配置)
└── assets/
    └── exclude-files.json (Markdown文件白名单)
```

## 约束与边界

- **场景约束**：一段内容场景不校验MD专属格式（标题层级、代码块等），仅聚焦基础规范和核心正确性

## 注意事项

1. 所有使用的文件路径都应是相对路径，在执行时转换为绝对路径
2. output 目录是在当前项目根目录下
3. 步骤2的分析必须逐个检查用户输入的所有文档
4. 检查完成后需要清理临时文件，仅保留最终结果文件
5. 支持 `.md` 文件扩展名的质量检查
6. **仅使用给定规则文件的范围进行检查，严禁增删改规则条目**
