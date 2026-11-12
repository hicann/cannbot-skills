# 翻译质量检查清单（必须执行）

> **⚠️ 重要：翻译完成后必须执行以下检查，确保翻译质量！**
>
> **使用定位**：本清单是**逐篇检查表**，用于逐篇自查与语义审查的判定依据。
> **批量机械验收**（脚本化扫描：译稿齐全、残留中文、全角标点、切换行、行数比对、链接有效、结构对齐、数值一致）清单在SKILL.md中，两者配合使用：机械验收管"面"（全量扫描拦截硬伤），本清单管"点"（逐篇判定语言质量与语义保真）。

1. **语法完整性检查**

   - [ ] 每个句子是否有明确主语（you、the system、this guide等）
   - [ ] 每个句子是否有明确动词
   - [ ] 主谓是否一致、单复数是否正确（"This type of APIs is..."，而非 are）
   - [ ] 条件句是否完整（If you...而非If need...）
   - [ ] 并列句是否正确连接（使用分号或拆分为独立句子）
   - [ ] "请参考"句型是否使用"For..."引导

2. **词汇与文风检查**

   - [ ] 是否无违禁外来语（via, e.g., i.e., etc.）
   - [ ] 是否无缩写（can't, it's）
   - [ ] 是否无口语词汇（figure out）
   - [ ] 是否无冗余表达（去掉多余的please）
   - [ ] 是否无拼写错误与形近词误用（for example 而非 or example；complexity 而非 complicity）
   - [ ] 单词间空格是否完整（isavailable → is available）
   - [ ] 是否无重复单词（namespace namespace → namespace）
   - [ ] 介词搭配是否正确（remove...from 而非 remove...to）

3. **语态检查**

   - [ ] 操作指引是否使用祈使句（Select...而非You can select...）
   - [ ] 功能描述是否使用主动语态（WebIDE provides...而非WebIDE can provide...）
   - [ ] 避免过度被动语态
   - [ ] 条件说明是否使用you为主语

4. **冠词检查**

   - [ ] 单数可数名词前是否有冠词（the/a/an）
   - [ ] 特指名词前是否用the（the host machine、the root user）
   - [ ] 泛指名词前是否用a/an（an Ascend environment）

5. **术语一致性检查**

   - [ ] 关键术语是否统一翻译（sample而非example）
   - [ ] 是否使用标准术语表（见[术语与产品名称规范](./translation-style-guide.md#术语与产品名称规范)）
   - [ ] 名词/动词形式是否正确（installation vs install）
   - [ ] 产品名称是否符合产品名称对照表（training/inference products，不体现 series）

6. **句子结构检查**

   - [ ] 长句是否合理拆分或使用"For..."引导
   - [ ] 是否无"名词+please refer to"错误结构
   - [ ] 是否无中文式句子结构（如"Host machine...operations please refer to..."）

7. **表格检查**

   - [ ] 表格字段是否使用标准翻译（Precautions而非Note）
   - [ ] 每个单元格是否为完整句子或短语
   - [ ] 是否无语法错误的片段

8. **格式检查**

   - [ ] Markdown格式是否正确
   - [ ] 表格结构是否完整
   - [ ] 相对路径链接指向的文件在仓库中存在（不检查外网可达性）
   - [ ] 指向本仓库且已存在对应译稿的链接（译稿即英文版文件，含文档与图片等资源），是否已改写为指向译稿（如README.md → README_en.md、zh路径 → en路径、docs/zh/figures/x.png → docs/en/figures/x.png）；链接形式需同时覆盖相对路径与指向本仓库的URL（如`https://gitcode.com/cann/<仓库名>/blob/<分支>/X.md`中`blob/<分支>/`后的仓库内路径）；无对应译稿或指向其他仓库的外部链接保持原样
   - [ ] 文档内锚点链接已同步为译后英文标题的锚点
   - [ ] `<term>`等行内标签数量与源文件一致：不得为源文件中无标签的内容补加标签（无`<term>`包裹的产品名按纯文本翻译），不得删除源文件已有标签
   - [ ] `<!-- npu="..." idN -->`/`<!-- end idN -->`等标记注释与源文件一一对应（数量、位置不变）
   - [ ] 语言切换行仅存在于源文件名为 README.md 且源文含切换行的译稿：源README含切换行时译稿原位改写（当前语言 English 不可点击，链接指向对应的中文原README）；源README不含切换行时译稿同样不含切换行；非README译稿必无切换行
   - [ ] 代码块格式是否正确
   - [ ] 非断行连字符（U+2011）已替换为 ASCII 连字符 -
   - [ ] 段落边界与空行位置同源文基本对应（行数允许浮动，按第 15 项 ±30% 判定）
   - [ ] 正文、标题无残留中文字符
   - [ ] 文件必须以单个换行符结尾（最后一行内容后紧跟换行符，不得缺失，也不得堆积空行）

9. **格式镜像检查（必须执行）**

   - [ ] 译文中每个 `**`/`` ` ``/`\_` 都能在中文源文对应位置找到依据
   - [ ] 无任何自行加粗或自行加 code，纯文本未被升级为加粗/code
   - [ ] 中文加粗/code/转义符在译文中逐处一致保留，无增、删、改、移

10. **标题与固定句式检查**

    - [ ] API文档标题命中映射表即替换（Applicable Products/Description/Prototype/Parameters/Examples 等）
    - [ ] Examples/Parameters/Constraints/Related Constants/Implementation Principles 均为复数，无单数形态
    - [ ] "是否XXX"译为 Whether XXX，无 Indicates/Specifies 冗余引导词
    - [ ] 返回值套用固定句式（Returns XXX. / Returns XXX on success; otherwise, returns a failure value.）
    - [ ] 源文"无"统一译为 None 且不加句号；列举清单最后一项前已加 and

11. **禁用词扫描**

    - [ ] 除源文自身为 interface（编程语言接口概念）和代码块原样内容外，所有中文"接口"（含各种形态）均已译为 API
    - [ ] 术语禁用译法未出现：series products、tail axis、calculate/calculation、pass in 等

12. **语义保真检查（否定、范围与限定词）**

    - [ ] 否定词（非、不、无、未、除…之外）完整保留，译后回读核对语义方向与原文一致（"非root用户" → a non-root user，而非 the root user）
    - [ ] 包含/排除范围与原文逐项一致，不得替换、增删所列产品或对象（"不包含A5" → excluding A5，而非 except A3）
    - [ ] 限定词完整保留，不得省略或弱化：默认→by default、必须→must、仅→only、当前→currently、暂→not yet；"建议"用 It is recommended（"建议提交"不得译为祈使句 "Submit"）

13. **标点、代码注释与 frontmatter 检查**

    - [ ] 无残留中文全角标点（，。；：？！（）《》【】等），统一使用英文半角标点；中文破折号（——）→ em dash（—），省略号（……）→ ...；引用中文书名、中文路径等须原样保留的内容除外
    - [ ] 代码块内中文注释已翻译为英文，代码、命令、输出本身保持原样；代码本体、缩进、语言标识、变量名未被改动
    - [ ] 注释符号保持源语言注释语法（Python/Shell 的 `#` 不得改为 `//`），注释翻译后仍以原注释符号开头，与代码保持原有分行与缩进，不得混在同一行
    - [ ] YAML frontmatter：title、description 等描述性字段已翻译为英文；键名、布尔值、路径/文件名类字段值保持原样

14. **数值与符号一致性检查（原样保留，禁止改动）**

    - [ ] 数学符号完整保留，取整符号、上下标等不得省略（`[k/16]` 不得写成 `k/16`）
    - [ ] 图片尺寸、数字、版本号与原文完全一致（`[201*101]` 不得写成 `[201 * 100]`）
    - [ ] 参数、变量、宏、环境变量大小写与原文一致（`SIP` 不得写成 `sip`）
    - [ ] 占位符保持原有语法格式（`{arch}` 不得写成 `<arch>`）
    - [ ] 计量单位大小写严格保留（B=字节、b=比特，含义不同；`GB/s` 不得写成 `Gb/s`）
    - [ ] 数学记法与源文件一致：元组 `(C,)` 尾逗号不得省略为 `(C)`；区间记法原样保留
    - [ ] 数学语义词准确：整除/可整除 → integer-divide/divisible（不得弱化为 divided by）；正规数 → normal number；次正规数（非正规数）→ subnormal number，不得混用

15. **行数比对检查（翻译完成度校验）**

    - [ ] 翻译完成后，比对中文原文与英文翻译的实质性文本行数
    - [ ] 仅当英文实质性文本行比中文少超过 30% 时判定漏翻译，需逐段对照补充；英文偏多或差距在 30% 以内均属正常

     **检查方法：**

     ```bash
     # 过滤代码块、空行、纯标记行（如---、|---|---|）后，统计实质性文本行数
     zh_lines=$(sed '/^```/,/^```/d' zh_file.md | grep -cvE '^[[:space:]]*$|^[[:space:]]*[#>|*-]+[[:space:]]*$')
     en_lines=$(sed '/^```/,/^```/d' en_file.md | grep -cvE '^[[:space:]]*$|^[[:space:]]*[#>|*-]+[[:space:]]*$')
     # 计算英文相对中文的缺失比例（仅英文偏少时为正值）
     shortfall=$(( (zh_lines - en_lines) * 100 / zh_lines ))
     # shortfall > 30 判定为漏翻译，需逐段对照补充
     ```

    **处理方式：**

    - 英文行数明显少于中文：逐段对照检查，找出未翻译段落并补充
    - 行数基本相等（±30%以内）：检查通过
    - 对于表格密集型文档，英文因换行展开导致行数偏多属于正常现象

**检查示例对比：**

| 问题类型     | ❌ 错误示例                                     | ✅ 正确示例                                       | 检查项            |
| ------------ | ---------------------------------------------- | ------------------------------------------------ | ----------------- |
| 缺少主语     | "can directly use..."                          | "you can directly use..."                        | 语法完整性-主语   |
| 缺少动词     | "Software installation method selection..."    | "Select a software installation method..."       | 语法完整性-动词   |
| 条件句不完整 | "If need to run samples..."                    | "If you need to run samples..."                  | 语法完整性-条件句 |
| 主谓不一致   | "This type of APIs are..."                     | "This type of APIs is..."                        | 语法完整性-主谓一致 |
| 冗余表达     | "please refer to..."                           | "refer to..."                                    | 词汇与文风-冗余   |
| 拼写错误     | "or example"                                   | "for example"                                    | 词汇与文风-拼写   |
| 词语重复     | "namespace namespace"                          | "namespace"                                      | 词汇与文风-重复词 |
| 缺少冠词     | "WebIDE development platform"                  | "the WebIDE development platform"                | 冠词检查          |
| 术语不一致   | "example"                                      | "sample"                                         | 术语一致性        |
| 句子结构混乱 | "Host machine...operations please refer to..." | "For downloading and installing..., refer to..." | 句子结构          |
| 表格字段     | "Note"                                         | "Precautions"                                    | 表格检查          |
| 自行加粗     | "Returns **HCCL_SUCCESS** on success"          | "Returns HCCL_SUCCESS on success"                | 格式镜像          |
| 标题单数     | "## Example"                                   | "## Examples"                                    | 标题固定译法      |
| 冗余引导词   | "Indicates whether to enable..."               | "Whether to enable..."                           | 固定句式          |
| 禁用词       | "This interface returns..."                    | "This API returns..."                            | 禁用词扫描        |
| 否定丢失     | "the root user"（原文"非root用户"）            | "a non-root user"                                | 语义保真          |
| 注释符号改变 | "// install CANN"                              | "# install CANN"                                 | 标点与代码        |
| 全角标点残留 | "Enter the password, then press Enter。"       | "Enter the password, then press Enter."          | 标点检查          |
| 数值改动     | "[201 * 100]"                                  | "[201*101]"                                      | 数值与符号        |
| 数学符号丢失 | "k/16"                                         | "[k/16]"                                         | 数值与符号        |
| 单位大小写错误 | "Gb/s"                                       | "GB/s"                                           | 数值与符号        |
| 漏翻译       | 原文100行，翻译仅60行                          | 逐段对照补充缺失章节                             | 行数比对检查      |
