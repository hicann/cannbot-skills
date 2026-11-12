# 翻译标准（docs-translation）

> 本文件为 docs-translation skill 的翻译标准全文，基于 Google 开发者文档风格指南（<https://developers.google.com/style>）整理，并融合算子库翻译规范中适用于文档翻译的条目（术语、标点、frontmatter、数值符号占位符、语义保真、行内标签等）。翻译前必须完整阅读并逐条遵守；SKILL.md 仅保留核心红线摘要。

## 语态规范

原则：主动语态优先，操作类用祈使句；被动语态仅限例外场景（动作执行者未知/无关、状态描述、错误提示中避免责备用户）。

| 场景 | 规范 | 示例 |
| ------ | ------ | ------ |
| 主动语态优先 | 面向用户的资料以主动语态为主，直接陈述能力 | ❌ "Designed for..." → ✅ "This guide provides..."；<br>❌"WebIDE can provide..."  vs ✅ "WebIDE provides..."    |
| 用户操作指引 | 祈使句：省去you，直接使用动词开头 | ❌ "You can enter the password" → ✅ "Enter the password"；<br>❌"Software installation method selection..."  vs ✅ "Select a software installation method..."    |
| 条件说明 | 祈使句或you为主语，条件句必须完整 | ❌ "If need to run samples..."  vs ✅ "If you need to run samples..."   |
| 被动语态例外 | 动作执行者未知/无关、状态描述、避免责备用户时可用被动 | ✅"The dialog box is displayed"；✅"The necessary software packages are already installed"  |

**地道表达对比：**

| 中文原文 | ❌ 错误翻译 | ✅ 正确翻译 | 分析 |
| --------- | ----------- | ----------- | ------ |
| WebIDE可提供... | "WebIDE can provide..." | "WebIDE provides..." | 主动语态更直接 |
| 该平台为您提供... | "provides...for you" | "provides you with..." | "provide you with"更地道 |

## 时态规范

| 场景 | 使用时态 | 示例 |
| ------ | --------- | ------ |
| 陈述规律/原理/机制 | 一般现在时 | "A ping command sends packets to test connectivity." |
| 操作后的瞬时结果 | 一般现在时 | ❌ "The dialog box will appear" → ✅ "The dialog box appears" |
| 需间隔较长时间的结果 | 将来时 | "The system will restart after installation." |
| 已完成的动作 | 现在完成时 | "You have successfully logged in." |

## 词汇与文风规范

**禁用词与替代表达：**

| 类别 | 禁止使用 | 正确用法 |
| ---- | --------- | --------- |
| 外来语 | etc. / e.g. / i.e. / via | and so on（需限定范围）/ for example / that is / through, by, using |
| 禁用术语 | interface | API（中文"接口"一律译为API，详见"术语与产品名称规范"） |
| 缩略词 | can't, it's, don't / you're, they're / won't, shouldn't | cannot, it is, do not / you are, they are / will not, should not |
| 过时词汇 | thus, hereinafter, hence | 使用现代通用表达 |
| 口语词汇 | figure out | 使用具体、正式的动词 |
| 词汇简缩 | char, config | character, configuration |

**文风要求：**

- 重要信息置前：将关键信息放在句首或段落开头
- 避免超长句子：每句不超过25个单词；并列长句用分号连接或拆分为独立句子
- 使用并行结构：相似描述使用统一句式
- 避免双重否定：使用直接陈述
- 使用前后一致的术语（见[术语与产品名称规范](#术语与产品名称规范)）、使用简单词汇；缩略语首次出现时定义
- 清晰、正确地组织信息；避免虚悬前置词和无谓重复或累赘

> 主动语态、一般现在时、第二人称/祈使句等要求见[语态规范](#语态规范)和[时态规范](#时态规范)，此处不重复。

## 句子结构与反直译规范

> ⚠️ 警告：中文省略主语常见，但英文必须有明确主语和动词！忽略此规则会导致严重语法错误！

译文应能脱离中文原文自然成立，避免逐字对应中文语序；明显直译属质量问题，需调整为地道英文。

**语法完整性（必须检查）：**

- 每个句子必须有明确主语：中文省略主语常见，无主句须补主语（优先 you / The API / The function，也可用 the system、this guide 等）
- 每个句子必须有明确动词：避免"名词 + please refer to"的错误结构

**反直译技巧：**

- 中文长前置定语后置（"支持循环展开的迭代器函数" → "an iterator function that supports loop unrolling"）
- 消解冗余虚词：进行/通过…来/以便/从而、多余的please等，直接省略或改写
- 被动转主动（见[语态规范](#语态规范)）；"将/将会"用一般现在时（见[时态规范](#时态规范)）
- 中文长句拆为英文短句或从句；技术含义优先于字面意思；固定句式直接套用，不临时造词

**固定句型重构：**

| 中文句型 | ❌ 直译 | ✅ 正确 |
| ----- | -------- | ------ |
| XX操作请参考《指南》中的"准备软件包"章节 | "XX operations please refer to 'Prepare Software Package' chapter in Guide" | "For XX operations, refer to the 'Prepare Software Package' chapter in the Guide" |
| 若仅编译算子，可以不安装（条件句省略主语） | "if only compiling operators, can not install" | "if you are only compiling operators, they do not need to be installed" |
| 并列长句 | 逐词串联成一个长句 | 使用"For..."引导，或用分号连接相关句子，或拆分为独立句子 |

**反直译示例：**

| 中文源 | ❌ 直译 | ✅ 优化 |
| ------ | ------- | ------ |
| 多层循环使用了loop_unroll并且配置unroll_list，会大大增加编译出的图数量，影响编译性能 | Multi-layer loops use loop_unroll and configure unroll_list, will greatly increase the number of compiled graphs, affecting compilation performance. | Using loop_unroll with unroll_list configured across multiple nested loops significantly increases the number of compiled graphs, which may impact compilation performance. |

## 冠词使用规范

必须检查冠词：

- **单数可数名词前必须有冠词**

   | 错误 | 正确 |
   | ----- | ------ |
   | "WebIDE development platform" | "the WebIDE development platform" |
   | "Ascend environment" | "an Ascend environment" |
   | "Docker engine" | "the Docker engine" |

- **特指名词前用the**
  - "the host machine"（特指宿主机）
  - "the root user"（特指root用户）
  - "the CANN software package"（特指某个包）

- **泛指名词前用a/an**
  - "an Ascend environment"（泛指一个环境）
  - "a compilation environment"（泛指编译环境）

## 表格与固定句式规范

**表头与字段固定译法：**

| 中文 | ✓ 标准翻译 | 说明 |
| ------ | ----------- | ------ |
| 注意事项 | "Precautions" | 比"Note"更专业 |
| 说明 | "Description" | 标准用法 |
| 必选 | "Required" | 标准用法 |
| 可选 | "Optional" | 标准用法 |
| 建议 | "Recommended" | 比"Suggestion"更常见 |
| 参数名 | "Parameter" | API文档参数表头 |
| 输入/输出 | "Input/Output" | API文档参数表头；单元格内"输入"→Input、"输出"→Output |

**固定句式：**

| 场景 | 中文模式 | 固定英文句式 |
| ---- | -------- | ------------ |
| 参数描述"是否XXX" | 是否XXX | Whether XXX（直接以 Whether 开头，禁止 Indicates whether / Specifies whether） |
| 返回值一般描述 | 返回XXX | Returns XXX. |
| 返回值成功/失败 | 接口成功返回XXX，其他失败。 | Returns XXX on success; otherwise, returns a failure value. |
| 无内容 | 无 | None（不加句号） |
| 列举清单 | A、B、C（无论末项前有无"和"） | A, B, and C（最后一项前统一加 and） |

**表格单元格内容规则：**

- 每个单元格必须是完整句子或名词词组，不可出现语法错误的片段
- 名词词组前不加冠词（Loop identifier name，而非 The Loop identifier name）；此为表格单元格内的例外，正文句子中单数可数名词仍须加冠词（见[冠词使用规范](#冠词使用规范)）
- 单元格内容句末加句号

## 术语与产品名称规范

**术语一致性表（必须保持一致）：**

| 中文术语 | 标准英文翻译 | 备注 |
| --------- | ------------ | ------ |
| 样例 | sample | 不使用example（不规范） |
| 环境 | environment | - |
| 部署 | deployment | - |
| 安装 | installation | install为动词，installation为名词 |
| 编译 | compilation/compile | compilation为名词，compile为动词 |
| 运行 | run/running | - |
| 宿主机 | host machine | 不使用host（不完整） |
| 容器 | container | - |
| 镜像 | image | - |
| 算子 | operator | - |
| 固件 | firmware | - |
| 驱动 | driver | - |
| 磁盘 | disk | 不使用drive（drive为"驱动器"） |
| 通信域 | communicator | - |
| 单算子比对 | single-operator comparison | 精度比对工具中Tensor比对的一种方式 |
| 整网比对 | network-wide comparison | 精度比对工具中Tensor比对的一种方式 |
| 邻接表 | adjacency list | 表示图的数据结构，每个顶点存储其相邻顶点的列表 |
| 片上内存 | on-chip memory | 集成在微处理器芯片上的存储器 |
| 管理面 | management plane | 健康状态信息接口所处的平面 |
| 业务面 | service plane | 推理等业务接口所处的平面 |
| 接口 | API | **绝对禁用 interface**，见下“禁用词” |
| 系列产品 | products | 不使用 series products 或 product series，译文中不体现 series |
| 尾轴 | last axis | 不使用 tail axis 或 tailing axis |
| 计算 | compute/computation | 不使用 calculate/calculation |
| 开发者 | you | 不使用 developers（文档目标读者是开发者，直接用第二人称） |
| 参数传入 | pass | 不使用 pass in |
| 张量 | tensor | 不使用 Tensor（例外：原文为 Tensor 或 Tensor 标识数据类型时保留，如 "Supported data types: Tensor, int..."） |

**禁用词（译文绝对禁止出现）：**

- **interface**：中文接口（含"该接口/此接口/接口函数/接口名/API接口"等所有形态）一律译为 API。
  - ❌ "This interface returns..." → ✓ "This API returns..."
  - ❌ "the interface name" → ✓ "the API name"
  - ❌ "API interface" → ✓ "API"
  - 两处例外：代码块原样内容；源文本身就是英文单词 "interface" 且指编程语言接口/接口类型概念时保留原词（如 C 语言的 interface 关键字）。

**术语一致性机制：**

- 术语首次出现即确定译法并记录，全文后续一律沿用；发现更优译法时，回改前文后统一。
- 若上述术语表未收录某术语，按"行业标准 → 上下文最贴切译法"确定，并在全文保持一致。
- 同一中文术语全文英文译法唯一，禁止同义混用。
- 避免形近词与臆造词（如复杂度为 complexity，非 complicity）；对不确定的术语必须查证，不得凭印象拼写。

**产品名称对照表：**

| 中文名 | 英文名 |
| --- | --- |
| Ascend 950PR&950DT系列产品 | Ascend 950PR&950DT products |
| Atlas A3系列产品 | Atlas A3 products |
| Atlas A3训练系列产品 | Atlas A3 training products |
| Atlas A2系列产品 | Atlas A2 products |
| Atlas A2训练系列产品 | Atlas A2 training products |
| Atlas 200I/500 A2推理产品 | Atlas 200I/500 A2 inference products |
| Atlas训练系列产品 | Atlas training products |
| Atlas推理系列产品 | Atlas inference products |

**产品名称翻译规则：**

- 产品名称（如 Ascend 950PR、Ascend 950DT）必须与中文完全保持一致，不得改写。
- "训练/推理系列产品"译为 training/inference products，切勿体现 series。
- 斜杠（/）左右不加空格（如 Ascend 950PR/Ascend 950DT）。
- 描述性文本（如 training products）不使用 Title Caps。
- 中文"支持"译为 Supported；中文冒号"："译为英文冒号":"后接一个空格。

示例：

```text
# 中文源：
- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持

# 英文输出：
- Ascend 950PR&950DT products: Supported
- Atlas A3 products: Supported
```

## Markdown 文件格式规范

**连字符处理：** 英文翻译中，将非断行连字符（U+2011）统一替换为 ASCII 连字符（U+002D，即普通减号 `-`）。

**文档末尾空行：** 按照 Markdown 规范（对应 markdownlint MD047），翻译输出的英文文档末尾必须保留一行空行，即文件以单个换行符结尾、最后一行内容之后留有一行空行：

- 文件必须以单个换行符结尾（最后一行内容后紧跟换行符，不得缺失，也不得堆积空行）。
- 增量更新已有翻译文件时，编辑后同样保证文件末尾保留一行空行，不因编辑丢失。
- 源中文文档末尾缺少空行时，译文仍须补齐末尾空行。本条为 md 文件级规范，优先于[格式镜像规则](#格式镜像规则)（格式镜像仅约束文内格式标记，不约束文件末尾空行）。

**标点符号规范：**

- 译稿中不得残留中文全角标点（，。；：？！（）《》【】等），必须使用对应的英文半角标点（, . ; : ? ! ( ) " "）。
- 中文破折号（——）改写为英文 em dash（—），中文省略号（……）改写为英文省略号（...）。
- 例外：引用中文书籍、文档或栏目名称时，优先使用确定的英文名称（如官方英文版标题）并去除书名号；仅当查无确定英文名称时保留中文原名，此时书名号等全角标点一并保留。须原样保留的中文内容（如中文路径、中文文件名），其中的标点保持原样。

## 格式镜像规则

> ⚠️ 警告：中文源文是什么格式，英文就是什么格式，逐处一致、一字不差。译文中每一个加粗（`**`）、code（`` ` ``）、转义符（`\_`）都必须能在中文源文对应位置找到依据，找不到即为格式错误，须立即删除。

**格式跟随判定表：**

| 中文源文格式 | 英文译文格式（必须一致） |
| ------------ | ------------------------ |
| 纯文本 | 纯文本（禁止自行加粗/加code） |
| `**name**`（加粗） | `**name**`（加粗） |
| `` `name` ``（code） | `` `name` ``（code） |
| `\_`（转义符） | `\_`（转义符原样保留） |
| 组合样式（如加粗+code） | 相同组合样式原样保留 |

**执行规则：**

1. **只译文本，不碰格式**：先把文本内容与格式标记分离，格式标记是"壳"，原样保留；只翻译"壳"内的文本内容，再把译文放回原"壳"中。
2. **中文纯文本 > 英文必须纯文本**：不得因为是 API 名、函数名、参数名、常量、返回值、路径、文件名，就自行加粗或加 code。技术标识符是否加格式，完全由中文源文决定，不由其"身份"决定。
3. **逐处对应，不强制统一**：同一名称在中文源文不同位置格式不同，英文译文也逐处对应，格式以"该处中文源文"为准。
4. **禁止格式升级/降级/迁移**：禁止把纯文本升级为加粗/code，禁止把加粗/code降级为纯文本，禁止把A处源文的格式搬到B处译文。

**格式溯源自检（写完每段后执行）：** 检查译文中的每个 `**`/`` ` ``/`\_`，回到中文源文同一位置确认是否存在相同标记：中文有→保留；中文没有→立即删除该标记。

| 中文源文（纯文本）       | ❌ 错误输出                          | ✅ 正确输出                      |
| ------------------------ | ----------------------------------- | ------------------------------- |
| aclmdlExecute函数        | **aclmdlExecute** function          | aclmdlExecute function          |
| unroll_list参数          | **unroll_list** parameter           | unroll_list parameter           |
| 接口成功返回HCCL_SUCCESS | Returns **HCCL_SUCCESS** on success | Returns HCCL_SUCCESS on success |

**行内标签与结构保真：**

- `<term>` 等行内标签必须与源文件一一对应（数量与位置不变）；源文件中未加标签的名称不得补加标签，按纯文本翻译。
- `<!-- npu="..." idN -->`、`<!-- end idN -->` 等标记注释原样保留，不得增删或改写；含中文说明文字的普通 HTML 注释须翻译内部文字、保留注释形式。
- 表格行、列表项、标题层级的数量不得增减；代码块数量与顺序保持不变。

## 标题固定译法

API Reference 类文档的章节标题使用固定映射，命中即替换，不得使用同义词、不得改写、不得临时造词；标题层级（#数量）保持不变：

| 中文标题（含变体，先归一化） | 唯一英文译法 |
| ---------------------------- | ------------ |
| 产品支持情况/产品支持状态/产品支持/支持产品/应用产品 | Applicable Products |
| 功能说明 | Description |
| 实现原理 | Implementation Principles |
| 函数原型/原型定义/定义原型 | Prototype |
| 参数说明 | Parameters（复数） |
| 相关常量 | Related Constants（复数） |
| 返回值说明/返回值 | Return Value |
| 约束说明 | Constraints（复数） |
| 调用示例/使用示例/示例 | Examples（必须复数，禁止 Example） |

**执行规则：**

1. **命中即替换**：标题文本匹配上表左侧任一中文变体时，必须替换为右侧唯一英文译法。
2. **归一化优先**：同义变体（如"产品支持状态""产品支持"）先归一化再映射，绝不允许翻译成不同英文。
3. **复数形态不可更改**：Examples、Parameters、Constraints、Related Constants、Implementation Principles 恒为复数，即使语义上指单个章节也一律用复数。
4. **标题用 Title Case，正文用 Sentence Case**。
5. 标题内命中固定短语的部分查表替换，其余文字按正文规则正常翻译。

## 代码块与注释翻译规则

- 代码块整体保持原样，仅翻译其中的**注释**部分；无注释的代码块原样保留。
- 代码样式（缩进、语言标识、变量名、常量名）与中文原文完全一致，字符不可改动。
- 译文核心用词与代码本身（函数名、参数名）保持一致。
- 注释符号保持源代码注释语法，不得擅改（Python/Shell 用 `#`，C/C++ 用 `//`；禁止把 `# 注释` 改为 `// 注释`）；注释翻译后仍以原注释符号开头，与代码保持原有分行与缩进，不得混在同一行。
- 注释风格：简洁英文，操作类以动词原形开头，句末统一加句号。
- 源文为中英双语注释时，译文仅保留英文部分。
- YAML frontmatter（文件头部 `---` 之间）： `title`、`description` 等描述性字段的内容需翻译为英文；键名、布尔值、路径/文件名类字段值保持原样。

| 注释样式 | 规则 |
| -------- | ---- |
| `#` | `#` 后接一个空格再接英文注释 |
| `//` | 保留 `//` 不变，不在两个 `/` 中间添加空格 |
| `/* */` | 译文与 `/*` 和 `*/` 之间各留一个空格 |

## 数值、符号与占位符规范（原样保留，禁止改动）

- 数学符号必须完整保留，向上/向下取整符号、上下标等不得省略：❌ `[k/16]` 译为 `k/16` ✅ 保留 `[k/16]`
- 图片尺寸、数字、版本号必须与原文完全一致：❌ `[201*101]` 译为 `[201 * 100]` ✅ 保留 `[201*101]`
- 参数、变量、宏、环境变量的大小写严格保持原样：❌ `SIP` 译为 `sip` ✅ 保持 `SIP`
- 占位符保持原有语法格式：❌ `{arch}` 译为 `<arch>` ✅ 保持 `{arch}`
- 计量单位大小写严格保留（B=字节，b=比特，含义不同）：❌ `GB/s` 译为 `Gb/s` ✅ 保持 `GB/s`
- 数学记法与语义词必须与原文精确一致：
  - 元组尾逗号不得省略：❌ `(C,)` 译为 `(C)` ✅ 保留 `(C,)`
  - 整除/可整除译为 integer-divide/divisible：❌ "整除64" 译为 "divided by 64" ✅ "integer-divided by 64"
  - 正规数 = normal number，非正规数/次正规数 = subnormal number，不得混用

## 语义保真规范（否定、范围与限定词）

- 否定词（非、不、无、未、除…之外）必须完整保留，译后回读核对语义方向与原文一致：

  ❌ "非root用户" 译为 "the root user" ✅ "a non-root user"

- 包含/排除范围必须逐项与原文一致，不得替换、增删所列产品或对象：

  ❌ "不包含A5" 译为 "except A3" ✅ "excluding A5"

- 限定词必须逐一保留，不得省略或弱化，固定译法：

  - 默认→by default、必须→must、仅→only、当前→currently、暂→not yet；

  - "建议"用 It is recommended（❌ "建议提交" 译为祈使句 "Submit" ✅ "It is recommended that you submit"）
