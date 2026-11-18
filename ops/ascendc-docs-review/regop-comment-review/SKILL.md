---
name: regop-comment-review
description: 算子原型注释审查Skill - 用于 Committer/Reviewer 审查 PR 中算子原型头文件（.h）的注释变更，覆盖原型与注释一致性、通用注释格式、Brief/Inputs/Attributes/Outputs/Constraints/Data Types/可选字段写作规范、芯片约束产品名规范等；也支持本地文件/目录/片段评审。触发条件：审查PR、检查PR、评审PR、原型注释评审、注释检视、注释规范检查、头文件注释审查、op_graph注释、算子原型注释、规格导出注释规范。
---

# 算子原型注释审查 Skill

## 概述

算子原型注释审查工具，帮助 Committer/Reviewer 高效审查 GitCode 上 Pull Request 中算子原型头文件（通常位于 `op_graph/inc` 的 `.h` 文件）的 `/** ... */` 注释变更。仅关注原型注释的规格导出合规性，不审查代码逻辑（`.cpp`、`.py` 等）。

原型注释是算子规格导出工具的唯一数据源：工具按 doxygen 标识字段（`@brief`、`@par Inputs`、`@attention Constraints` 等）解析注释，生成对外发布的算子规格资料。注释问题会直接传导到发布件：

- 字段名拼写/大小写错误 → 工具识别不到，字段**无法导出**
- 注释参数名/个数与 `REG_OP` 定义不一致 → 参数说明**不会被导出**
- 多余冒号、错位 `\n`、芯片代号 → **原样进入发布件**

## 功能特性

- 支持手动触发模式。
- **PR 原型注释检视**：当 PR 包含 `op_graph/inc/*.h` 原型头文件变更时，按内置规则检视，覆盖原型与注释一致性、通用注释格式、Brief、Inputs、Attributes、Outputs、Constraints、Data Types、可选字段九大维度（45 条规则）。
- **本地评审模式**：不接 PR 也可直接评审头文件、目录或注释片段。
- **仅输出报告**：审查产物为结构化审查报告，不向 GitCode 提交任何评论。

## 使用方法

### 触发方式

在 OpenCode 中加载此 skill 后，支持两种评审场景，通过参数区分：

| 场景 | 调用方式 | 分支要求 |
| ------ | --------- | --------- |
| 用户本地评审 | `/regop-comment-review "文档路径"` | 无需切换分支，直接在用户本地仓库当前状态评审 |
| 评审其他用户 PR | `/regop-comment-review PR：<pr_number>` | 必须先切换到 PR 对应分支（见下文） |

也支持以下自然语言触发词（仅输入触发词时交互补充评审目标）：

| 触发词 | 说明 |
| ------ | ------ |
| `请审查PR <pr_number>` | 评审指定 PR 中的 `.h` 原型注释变更 |
| `请审查PR：<pr_number>` | 触发词变体（中文冒号） |
| `请检查PR <pr_number>` | 等效触发词 |
| `请检查PR：<pr_number>` | 等效触发词变体 |
| `请评审PR <pr_number>` | 等效触发词 |
| `请评审PR：<pr_number>` | 等效触发词变体 |
| `原型注释评审` | 本地模式，需后续补充文件/目录/片段 |
| `注释检视` / `注释规范检查` | 本地模式触发词 |
| `头文件注释审查` / `op_graph注释审查` | 本地模式触发词 |

**触发流程**：

1. 用户传入评审目标：本地文件/目录/片段路径（本地评审），或 `PR：<pr_number>`（PR 评审）；仅输入触发词时交互补充目标
2. Skill 自动执行完整审查流程（见「工作流程」章节）
3. 生成审查报告并展示给用户，流程结束（不向 GitCode 提交评论）

**示例**：

```
# 用户本地评审
/regop-comment-review "文档路径"

# 评审其他用户PR
/regop-comment-review PR：1411
```

### 切换到 PR 分支（仅评审其他用户 PR）

**用户本地评审无需任何分支操作**，直接在用户本地仓库的当前状态评审指定目标即可。

**评审其他用户的 PR 前，必须切换到 PR 对应的本地分支**，否则无法看到 PR 新增的文件！PR 评审触发后自动执行：

```
# 1. 确保远程 master 已同步
git fetch origin master

# 2. 获取 PR 的远程引用并创建本地分支
git fetch origin refs/merge-requests/<pr_number>/head:pr_<pr_number>

# 3. 切换到该分支（关键步骤！）
git checkout pr_<pr_number>

# 4. 开始原型注释审查
# 此时可以查看 PR 变更的 .h 原型头文件、解析 REG_OP 定义、检视注释
```

**常见问题**：

- 如果不执行 `git checkout`，后续分析仍停留在原分支（如 master）
- PR 新增的文件（如 `op_graph/inc/xx.h`）只有在切换到 PR 分支后才能看到
- `git show pr_<pr_number>:<path>` 只查看特定文件，不会切换工作区
- **使用三个点 (...) 而非两个点 (..)**：`git diff master..pr` 会包含 master 上的其他提交，正确做法是 `git diff master...pr`
- **本地 master 与远程不同步**：如果本地 master 落后于远程，`git diff master...pr` 会包含不属于 PR 的变更。**必须先执行 `git fetch origin master` 并使用 `remotes/origin/master` 进行比较**

### 本地评审模式

用户本地评审场景：无需切换分支，直接在用户本地仓库的当前状态评审。三种输入：

| 输入 | 评审能力 |
| ------ | --------- |
| 单个头文件（.h） | 完整评审：一致性 + 完备性 + 格式 |
| 头文件目录 | 批量评审：汇总表 + 逐文件问题清单 |
| 粘贴的注释片段 | 仅格式与写法检查；若同时给出 `REG_OP` 定义可做一致性核对。报告中必须说明此局限 |

## 审查维度

### 审查维度总表

当 PR 包含 `.h` 原型头文件变更时，**必须**按以下维度逐项审查。

| 维度 | 最高严重级 | 规则编号 | 说明 |
| ------ | ----------- | --------- | ------ |
| 原型与注释一致性 | CRITICAL | PC-01~06 | 算子名、参数名/个数/类型与 REG_OP 定义完全一致；PR 中原型与注释同步修改 |
| 通用注释格式 | HIGH | FM-01~09 | 注释块形式、\n 换行、@li、表格、公式、多余符号、英文写作 |
| 功能简介 Brief | HIGH | BF-01~04 | @brief 无冒号、功能描述详尽、代码段标记 |
| 输入 Inputs | CRITICAL | IN-01~05 | 字段名精确、逐参数解释完整 |
| 属性 Attributes | CRITICAL | AT-01~05 | 字段名精确、默认值、Reserved 标注 |
| 输出 Outputs | CRITICAL | OT-01~03 | 字段名精确、逐参数解释完整 |
| 约束 Constraints | CRITICAL | CT-01~07 | 芯片约束用产品名、严禁芯片代号、废弃话术 |
| 数据类型 Data Types | MEDIUM | DT-01 | 由脚本自动生成，不得手写 |
| 可选字段 | HIGH | OF-01~05 | Graphs/see/Quantization/Third-party/Restrictions |

**严重级别含义**：

- **CRITICAL**：工具无法导出，或违规内容进入发布件——合入前必须修改
- **HIGH**：发布件内容错误或关键信息缺失——强烈建议修改
- **MEDIUM**：格式规范偏差——建议修改
- **LOW**：质量提升建议

---

#### 原型与注释一致性

审查目标与基准在同一头文件内：`/** ... */` 注释块与 `REG_OP` 链式定义逐项对照。PR 场景下还需对照 diff 两端——只改 `REG_OP` 不改注释（或反之）即违反 PC-06。

| 编号 | 规则 | 严重级别 |
| ------ | ------ | --------- |
| PC-01 | `REG_OP(OpType)` 中的算子名必须与头文件名完全一致，否则工具不识别、算子无法导出 | CRITICAL |
| PC-02 | 注释中输入的参数名、个数必须与 `.INPUT()` 声明完全一致（多写、少写、改名均违规） | CRITICAL |
| PC-03 | 注释中输出的参数名、个数必须与 `.OUTPUT()` 声明完全一致 | CRITICAL |
| PC-04 | 注释中属性的参数名、个数必须与 `.ATTR()`/`.REQUIRED_ATTR()` 声明完全一致 | CRITICAL |
| PC-05 | 注释中描述的数据类型必须与 `REG_OP` 定义类型一致 | CRITICAL |
| PC-06 | PR 修改 `REG_OP` 定义（加/删/改参数）时，对应注释必须同步更新，不得遗留旧版描述 | CRITICAL |

**审查方法**：

1. 识别 PR 中变更的 `op_graph/inc/*.h` 原型头文件
2. 解析 `REG_OP` 链式定义，提取算子名及 `.INPUT()`、`.OUTPUT()`、`.ATTR()`、`.REQUIRED_ATTR()`、`.GRAPH()` 声明的参数名、类型、默认值，建立"原型基准"
3. 与注释块中各字段的参数描述逐项对比（参数名、个数、类型）
4. PR 模式下用 `git diff remotes/origin/master...pr_<n>` 核对原型与注释是否同步修改
5. 记录不一致项，标注文件绝对行号

---

#### 通用注释格式

| 编号 | 规则 | 严重级别 |
| ------ | ------ | --------- |
| FM-01 | 注释块以 `/**` 开头、`*/` 结尾；原型定义缺少注释块即为缺失 | CRITICAL |
| FM-02 | 一句话未说完只能回车换行，不得加 `\n`（导出的资料会强制换行，把一句话切成两行）；完整句子/段落结束后才可加 `\n` | HIGH |
| FM-03 | 详细描述多段落时，每段以 `\n` 结束，使用空行划分段落 | MEDIUM |
| FM-04 | ≥2 个条目（输入/输出/属性/约束）时每条用 `@li` 区分；仅 1 个条目时不用 `@li` | MEDIUM |
| FM-05 | `@li` 后完整的一句话可不加 `\n`；多句话需要强制换行时用 `\n` | LOW |
| FM-06 | 表格前后必须加 `\n`（Attributes 段中：表格前两个 `\n`、表格后一个 `\n`），否则表格无法正确导出 | MEDIUM |
| FM-07 | 公式中乘号 `*` 前后都要加空格，否则无法导出正确的公式 | MEDIUM |
| FM-08 | 无多余符号：多余短横线、多余 `@`（删 `@li` 时只删了 `li` 的残留）、多余标识符、多余 `\n`（doxygen 标识符处会自动换行，无需手写） | HIGH |
| FM-09 | 描述内容用英文写作（发布资料为英文），并且要符合英语语法规范 | MEDIUM |
| FM-10 | 一句话描述完成后，需要使用英文句点结尾 | MEDIUM |

**快速预扫**：逐行核对前，可先用文本搜索定位高风险模式，再结合上下文确认：

```
rg -n "@brief\s*[:：]" <file>     # @brief 后多余冒号
rg -n "910B|910C|V350" <file>     # 严禁出现的芯片代号
rg -n "@par\s+\S+" <file>         # 列出全部 @par 字段名，逐一核对拼写
rg -n "^\s*\*?\s*\|" <file>       # 定位表格行，检查表格前后是否有 \n
```

---

#### 功能简介 Brief

| 编号 | 规则 | 严重级别 |
| ------ | ------ | --------- |
| BF-01 | `@brief` 后不得加冒号（`:` 或 `：`）——加了冒号，发布资料中算子描述开头会带多余冒号 | HIGH |
| BF-02 | 功能描述在 `@brief` 的下一行呈现 | MEDIUM |
| BF-03 | 不得只写"某某算子"式的简单描述，必须详细写出算子的功能特性 | MEDIUM |
| BF-04 | 功能描述中的公式或代码段用 `@code{.c} ... @endcode` 标记（`{}` 中为开发语言） | LOW |

---

#### 输入 Inputs

| 编号 | 规则 | 严重级别 |
| ------ | ------ | --------- |
| IN-01 | 字段名必须为 `@par Inputs`，大小写和末尾 `s` 完全一致，否则无法导出 | CRITICAL |
| IN-02 | 每个输入必须解释：含义、可选/必选、数据类型、format、shape 要求、取值范围 | HIGH |
| IN-03 | 单个输入自身的特殊要求就近写在该输入描述后 | MEDIUM |
| IN-04 | 输入**之间**的约束不写在 Inputs，统一放 Constraints | MEDIUM |
| IN-05 | 输入的芯片差异约束放 Constraints（不写在 Inputs） | MEDIUM |

---

#### 属性 Attributes

| 编号 | 规则 | 严重级别 |
| ------ | ------ | --------- |
| AT-01 | 字段名必须为 `@par Attributes`，拼写完全一致，否则无法导出 | CRITICAL |
| AT-02 | 每个属性必须说明：可选/必选、数据类型、取值范围、默认值、使用约束（仅支持的属性值、属性值来源限制等） | HIGH |
| AT-03 | 可选属性必须给出默认值 | HIGH |
| AT-04 | 预留属性必须写明 `Reserved.`（预留参数，暂不使用） | MEDIUM |
| AT-05 | 属性之间的配合约束放 Constraints | MEDIUM |

---

#### 输出 Outputs

| 编号 | 规则 | 严重级别 |
| ------ | ------ | --------- |
| OT-01 | 字段名必须为 `@par Outputs`，大小写和末尾 `s` 完全一致，否则无法导出 | CRITICAL |
| OT-02 | 每个输出必须解释：含义、数据类型、format、shape | HIGH |
| OT-03 | 输出之间的约束、芯片差异约束放 Constraints | MEDIUM |

---

#### 约束 Constraints

| 编号 | 规则 | 严重级别 |
| ------ | ------ | --------- |
| CT-01 | 有约束时必须使用 `@attention Constraints` 标识 | HIGH |
| CT-02 | 只放"跨参数/跨对象"约束（使用场景约束、多输入配合、多属性配合、对上下层算子的要求）；单参数约束就近写在参数描述后 | MEDIUM |
| CT-03 | 芯片约束按产品英文名分条目，格式：`@li <产品名>: <英文描述>` | HIGH |
| CT-04 | **严禁出现 910B、910C、V350 等芯片代号字眼**，发布资料只允许产品名 | CRITICAL |
| CT-05 | 多条约束用 `@li` 区分；或不用 `@li`，在每条完整约束后加 `\n` | MEDIUM |
| CT-06 | 废弃算子说明话术：`This operator will be deprecated in the future. Replace it with the xx operator.`（xx 为替代算子名） | MEDIUM |
| CT-07 | 类似二级标题的分组用 `- <产品名>:` 一行，下一行起缩进 `@li:xx` | LOW |

**芯片约束标准写法**：

```
*@attention Constraints
*@li Atlas A2 products: <英文描述>  \n
*    If the xxx .... \n
*    If the xxx ....
*@li Atlas A3 products: <英文描述>  \n
*    If the xxx ....
```

**产品与芯片映射**（注释中只允许出现产品英文名，出现"软件芯片名"即为 CT-04 违规）：

| 产品英文名 | 软件芯片名（禁止出现在注释中） |
| --- | --- |
| Ascend 960PR&960DT products | 960 |
| Ascend 950PR&950DT products | 950 |
| Atlas A3 products | 910_93 |
| Atlas A2 products | 910b/910Bxx |
| Atlas 200/500 A2 inference products | 310B/310b/310b1、310b2\310bxxx |
| Atlas inference products | 310P/310p/310p1/310P3 |
| Atlas training products | 910 |

---

#### 数据类型 Data Types

| 编号 | 规则 | 严重级别 |
|------|------|---------|
| DT-01 | `@par Data Types` 由导出脚本读取信息库自动生成并与原型内容合并，头文件中不得手写（发现即建议删除，交给脚本生成） | MEDIUM |

背景说明：该字段反映当前芯片支持的类型，允许是 `REG_OP` 定义类型的子集；部分融合算子和 GE Local 算子没有 Data Types 信息，属正常情况，不作为问题。

---

#### 可选字段

| 编号 | 规则 | 严重级别 |
| ------ | ------ | --------- |
| OF-01 | `@par Graphs`（子图信息，一般 GE 算子才有）：参数名、个数与 `.GRAPH()` 声明一致；≥2 个参数用 `@li`，单个不用；表格前后 `\n`、公式乘号前后空格 | HIGH |
| OF-02 | `@par see`：填关联算子（如反向算子的关联算子是正向算子）；无则删除该字段 | LOW |
| OF-03 | `@par Quantization supported or not`：是否支持量化；不支持可不加；内容如 `Yes` | LOW |
| OF-04 | `@par Third-party framework compatibility`：列举对标的第三方友商算子，如 `Compatible with the TensorFlow operator UnsortedSegmentSum` | LOW |
| OF-05 | `@par Restrictions`：内部算子标识，**该字段不会导出**，出现即表示算子仅内部使用；话术二选一：有外部替代 `Warning: THIS FUNCTION IS DEPRECATED. Please use Xxx instead.`（Xxx 为实际替代算子名）；无替代 `Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.` | MEDIUM |

含 `@par Restrictions` 的算子照常审查其余字段，但在报告概要中注明"内部算子，不在对外导出列表"。

---

#### 常见错误速查

| 错误类型 | 典型表现 | 规则 |
| ------ | ------ | ------ |
| 多余的冒号 | `@brief: xxx` | BF-01 |
| 多余的 @ | 删 `@li` 时只删了 `li`，残留孤立 `@` | FM-08 |
| 多余的 \n | doxygen 标识符旁手写 `\n`（会自动换行） | FM-08 |
| 多余的短横线/标识符 | 残留符号进入发布件 | FM-08 |
| 算子名不一致 | REG_OP 名与头文件名不同 | PC-01 |
| 类型不一致 | 注释类型与 REG_OP 定义类型不同 | PC-05 |
| 参数名不一致 | 注释参数名与 REG_OP 定义不同 | PC-02/03/04 |
| 参数个数不一致 | 注释多写/漏写参数 | PC-02/03/04 |
| 漏解释 | 参数只有名字没有含义说明 | IN-02/AT-02/OT-02 |
| 功能未解释 | @brief 只写算子名或"某某算子功能" | BF-03 |
| 芯片代号 | 出现 910B、910C、V350 等字眼 | CT-04 |

---

### 检视输出格式

每条检视意见使用以下 JSON 格式：

```json
{
  "severity": "CRITICAL|HIGH|MEDIUM|LOW",
  "file": "op_graph/inc/nn_calculation_ops.h",
  "line": 45,
  "rule_id": "PC-02",
  "dimension": "原型与注释一致性",
  "title": "注释缺少输入参数说明",
  "body": "注释 @par Inputs 中缺少 offset_w 的说明，与 .INPUT() 声明不一致，该参数说明不会导出。请补充：@li offset_w: Reserved.",
  "code_snippet": ".INPUT(offset_w, TensorType(...))"
}
```

`dimension` 取值：原型与注释一致性、通用注释格式、功能简介、输入、属性、输出、约束、数据类型、可选字段。

### 原型注释检视流程

1. 识别 PR 中变更的 `op_graph/inc/*.h` 原型头文件（本地模式为用户指定目标）
2. **原型一致性检查**：解析 `REG_OP` 建立原型基准，逐项核对算子名、参数名、个数、类型（PC-01~06）
3. **字段完备性检查**：Brief、Inputs、Attributes、Outputs、可选字段逐项检查（BF/IN/AT/OT/OF）
4. **格式与符号检查**：`\n`、`@li`、表格、公式、多余符号（FM）
5. **约束与芯片检查**：约束归属是否正确、产品名用法对照产品芯片映射表（CT）
6. **Data Types 检查**（DT-01）
7. 记录每条违规的规则编号、严重级别、文件绝对行号
8. 输出结构化检视意见
9. 汇总报告：按严重级别统计违规数量，列出高频违规类型

对任何写法拿不准时，先查参考文件（见「参考文件」章节）再下结论。

## 工作流程

用户触发后，按评审场景分支执行：

- **本地评审模式**：跳过步骤 1~4，直接解析用户指定的头文件/目录/片段，按「审查维度」逐项检视，执行步骤 5 生成报告
- **PR 评审模式**：按步骤 1~5 完整执行

用户触发后，Skill 自动按以下步骤执行：

1. **同步远程 master**（仅 PR 评审模式）

   ```bash
   git fetch origin master
   ```

   **重要**：这会更新 `remotes/origin/master` 指针。如果本地 master 落后于远程，对比 PR 时会包含不属于 PR 的变更，导致审查结果不准确。
2. **获取并切换到 PR 分支**（仅 PR 评审模式）

   ```bash
   git fetch origin refs/merge-requests/<pr_number>/head:pr_<pr_number>
   git checkout pr_<pr_number>
   ```

3. **获取变更文件列表**
   - 使用三个点查看 PR 的真正 diff：

     ```bash
     git diff remotes/origin/master...pr_<pr_number> --stat   # 变更文件列表
     git diff remotes/origin/master...pr_<pr_number> -p       # 详细变更
     ```

   - **必须使用 `remotes/origin/master`**，不能使用本地 `master`，因为本地 master 可能与远程不同步
4. **检测原型头文件变更**
   - 检查 diff 中是否包含 `.h` 文件的变更：

     ```bash
     git diff remotes/origin/master...pr_<pr_number> --stat | grep '\.h'
     ```

   - 本 skill 仅审查算子原型头文件（含 `REG_OP` 定义的 `.h`，通常位于 `op_graph/inc`）；其他代码文件不在审查范围，报告中说明跳过。
   - **评审范围以注释块为单位**：PR 修改了注释的任何部分，就以 PR 分支上的完整 `/** ... */` 注释块 + 同文件 `REG_OP` 定义为评审对象；未变更的原型头文件不主动评审（除非用户明确要求全量）。
   - 如果检测到 `.h` 原型头文件变更，**必须按本文档「审查维度」中的规则逐项检视**：
     - **原型与注释一致性**：核对算子名、参数名、个数、类型是否与 `REG_OP` 完全一致；diff 两端原型与注释是否同步修改（规则 PC-01~06）
     - **通用注释格式**：注释块形式、`\n`、`@li`、表格、公式、多余符号、英文写作（规则 FM-01~09）
     - **功能简介/输入/属性/输出**：`@brief` 冒号、功能描述、逐参数解释、默认值、Reserved（规则 BF/IN/AT/OT）
     - **约束**：约束归属、芯片约束产品名、废弃话术，对照产品芯片映射表（规则 CT-01~07）
     - **Data Types 与可选字段**：Data Types 不手写；Graphs/see/Quantization/Third-party/Restrictions 规范（规则 DT-01、OF-01~05）
   - 将检视结果合并到审查报告中，维度标记为 `[原型注释]`
5. **生成审查报告**
   - 输出结构化的问题列表，按严重程度分类并展示给用户
   - **报告即最终交付物**：本 skill 不向 GitCode 提交任何评论；如需在 PR 中反馈，由用户自行复制报告内容

## 审查报告格式

```markdown
## 原型注释审查报告: PR #1411 - [PR标题]

**作者:** @author
**分支:** feature/xxx → main

### 概览

| 维度 | 问题数 | 最高严重级 |
|------|--------|-----------|
| 原型与注释一致性 | 2 | CRITICAL |
| 通用注释格式 | 1 | HIGH |
| 输入 | 1 | HIGH |

### 严重问题 (2)

**[1]** [原型一致性] op_graph/inc/nn_calculation_ops.h:45
- **问题:** 注释 @par Inputs 中缺少 offset_w 的说明，与 .INPUT() 声明不一致
- **建议:** 补充 `@li offset_w: Reserved.`（写法参考 references/writing-example.md）

### 高危问题 (N)

...

### 一般问题 (N)

...

### 建议问题 (N)

...
```

本地批量评审（目录）时，报告开头增加汇总表：

```markdown
| 文件 | 算子名 | CRITICAL | HIGH | MEDIUM | LOW | 结论 |
|------|--------|----------|------|--------|-----|------|
```

本地单文件/片段评审时，报告末尾增加"通过项确认"清单：

- 一致性：算子名、输入、输出、属性与 REG_OP 原型定义一致
- 字段名：@brief、@par Inputs、@par Attributes、@par Outputs 拼写正确
- 格式：\n、@li、表格、公式、无多余符号
- 芯片约束使用产品名

## 依赖

- OpenCode LLM 集成
- Git（PR 模式下用于同步远程分支与获取 diff；本地评审模式无需网络）

## 参考文件

- `references/field-spec.md` — 12 个字段的完整规范（来源、获取路径、功能说明、注释要求、写作样例）。对某字段的写法拿不准时读它。
- `references/writing-example.md` — Conv2D 完整注释样例与芯片约束、表格、公式、内部算子等标准写法。给开发者写"修改建议"时参考它。

两份参考文件均未覆盖的写法，如实标注"规范未明确"，不要凭空发明规则。

## 文件结构

```tree
~/.config/opencode/skills/regop-comment-review/
├── SKILL.md                    # 本文件（含原型注释审查规则）
├── README.md                   # 介绍文档
├── references/
│   ├── field-spec.md           # 算子原型注释字段规范全文（评审权威依据）
│   └── writing-example.md      # 头文件注释写作样例（标准写法参考）
└── evals/
    └── evals.json              # 测试用例
```
