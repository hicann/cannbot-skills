# aclnn 领域质检规则 - 规范性检查详细说明

**重要提示：**

- 仅使用本文件中明确定义的规则进行检查
- 规则名称、规则级别必须与本文件完全一致

## 1. 规范性检查

检查文档格式和命名是否符合统一规范。此类检查仅基于文档内容，不需要读取代码。

| 规则名称                               | 描述                                                                                       | 级别 |
| -------------------------------------- | ------------------------------------------------------------------------------------------ | ---- |
| API 文件命名规范                       | 检查 API 文件命名是否遵循 aclnnXxx 前缀、文件名与一级标题一致                              | 严重 |
| API 章节标题命名规范                   | 检查##二级标题是否符合必选章节要求（产品支持情况、功能说明、函数原型、约束说明、调用示例） | 提示 |
| 产品型号命名规范                       | 检查产品名称是否在允许范围内、是否使用 term 标记                                           | 严重 |
| API 函数原型格式规范                   | 检查原型格式（返回类型、参数格式、缩进、末尾符号、语言类型标识、两段式接口参数形式）       | 一般 |
| API 参数表里字段规范                   | 检查参数表格字段名称是否在允许范围内                                                       | 提示 |
| API 参数表里参数名列不规范             | 检查参数名是否与函数原型一致、参数顺序是否一致                                             | 一般 |
| API 参数表里输入/输出列不规范          | 检查输入/输出列的书写方式是否在允许范围内（输入、输出、输入/输出等）                       | 一般 |
| API 参数表里数据类型不规范             | 检查 dtype 是否大写、是否在支持范围内                                                      | 严重 |
| API 参数表里数据格式不规范             | 检查 format 是否大写、FRACTAL_NZ 与 NZ 是否混用、基础类型是否有 format                     | 严重 |
| API 参数表里 shape 不规范              | 检查基础数据类型是否错误描述了 shape                                                       | 严重 |
| API 参数表里非连续 Tensor 支持度不规范 | 检查是否对非 aclTensor 类型错误描述了非连续 Tensor 支持度                                  | 严重 |
| API 参数描述不规范                   | 检查参数表不允许出现"None"和"默认值"                                                       | 一般 |
| API 参数表里的固定参数说明不规范       | 检查一阶段参数表中workspaceSize和executor行、二阶段参数表中四个固定参数行的描述是否符合规范 | 提示 |
| 返回值/错误码不规范                    | 检查返回值和错误码是否在定义范围内                                                         | 严重 |
| 表格使用规范性                         | 检查两段式接口的参数说明和拦截校验是否使用表格形式                                         | 提示 |
| 芯片差异化描述必须k：v形式 | 检查芯片差异化描述是否使用key：value形式，中文用顿号分隔多芯片，英文用逗号分隔；差异化描述不允许出现在表格内 | 一般 |
| 废弃/试验接口说明 | 检查废弃或试验接口是否在文档大标题下方第一行使用固定话术说明 | 提示 |
| HTML表格语法规范 | 检查HTML表格内是否混用Markdown语法（如代码段、link等） | 提示 |

### API 文件命名规范

- 级别：严重
- 位置：文章的文件名和文章开头的 `# （一级标题）`。
- 规则：
   1. API 命名前缀必须是 aclnnXxx，注意 API 名 (Xxx) 必须为大驼峰格式，（仅 2 个接口例外：aclStft 和 aclRfft1D）
   2. 允许多 API 合并命名，采用&符号拼接（示例：aclnnAdd&aclnnInplaceAdd）
   3. 要求 API 文件命名、API 文件的 `#一级标题`保持一致。
      - 示例 1：`aclnnAbs.md`与 `# aclnnAbs`
      - 示例 2：`aclnnAcos&aclnnInplaceAcos.md`与 `# aclnnAcos&aclnnInplaceAcos`

### API 章节标题命名规范

- 级别：提示
- 位置：全文的所有 `## （二级章节标题）` 。
- 规则：
   1. 二级标题**必选章节**必须存在，且名称必须与下述列表完全一致（一字不差）：[`## 产品支持情况`、`## 功能说明`、`## 函数原型`、`## 约束说明`、`## 调用示例`]
   2. 除上述第2条必选章节必须存在外，还可增加自定义二级章节标题，但自定义标题不允许与必选标题语义重复（例如：已有必选章节"## 调用示例"就无法自定义"## 调用说明"）

### 产品型号命名规范

- 级别：严重
- 位置：`## 产品支持情况` 章节中的产品无序列表。
- 规则：
   1. 只允许使用指定范围的产品名，按[文档结构参考](./aclnn-doc-structure.md#21-产品支持情况)中的无序列表模板组织，保留各产品项的 HTML 注释边界，支持状态按接口实际能力填写。
   2. 需要加 term 标记对，例如 `<term>` 产品名 `</term>`。对标签内末尾存在多余空格无要求。
   3. 允许的产品名称列表：[产品名称列表](./aclProductList.md)

### API 函数原型格式规范

- 级别：一般
- 位置：`## 函数原型` 章节。
- 规则：**必须逐一检查以下8项，每项错误均需独立报告。每段接口必须分别独立检查每项规则，禁止仅检查一段接口而跳过另一段：**
   1. 返回类型必须是 `aclnnStatus`（每段接口均须检查）
   2. 参数名必须使用小驼峰格式，不带英文或中文符号，如下划线、中横线等（如 `cubeMathType`、`bias`）
   3. 单参数应单行呈现
   4. 多参数时缩进对齐，`*` 号位置随意但所有参数要保持一致
      - **对齐标准**：所有参数的「参数名」左边界（即参数名首字母的列位置）应在同一列，形成视觉上的垂直对齐。
      - **检查范围**：必须对每段接口分别独立检查缩进对齐。
      - **常见错误模式**：
        a. 第2段接口 `void *workspace` 的 workspace 紧贴 `*` 后开始，而其余参数名（workspaceSize、executor、stream）在更右列位置开始，参数名列位置不平齐。
        b. 第1段接口因多参数同行（违反第3项）导致同行内参数名列位置无法与其他行对齐。
        c. `*` 号位置不一致：部分参数 `*` 紧跟类型名（如 `aclTensor*`），部分参数 `*` 紧贴参数名前（如 `*workspace`），两种风格混用导致参数名对齐被破坏。
      - **错误示例1（第1段接口缩进未对齐）**：
        ```Cpp
        aclnnStatus aclnnAbsGetWorkspaceSize(
           const aclTensor *self,
           aclTensor       *out,
                 uint64_t        *workspaceSize, （该行参数名workspaceSize起始列与上下行不平齐）
           aclOpExecutor  **executor)
        ```
      - **错误示例2（第2段接口缩进未对齐）**：
        ```Cpp
        aclnnStatus aclnnAbs(
        void *workspace,              ← workspace紧贴*号后，列位置与下方参数名不平齐
        uint64_t             workspaceSize,
        aclOpExecutor        *executor,
        aclrtStream           stream)
        ```
      - **正确示例（第2段接口参数名对齐）**：
        ```Cpp
        aclnnStatus aclnnAbs(
          void              *workspace,
          uint64_t           workspaceSize,
          aclOpExecutor     *executor,
          aclrtStream        stream)
        ```
        所有参数名（workspace、workspaceSize、executor、stream）左边界在同一列对齐。
   5. 原型末尾不加分号或其他符号
   6. API命名规范：每个算子接口可以对应n个接口，但只允许存在如下2种命名样式：{`aclnnXxxGetWorkspaceSize`, `aclnnXxx`}
      错误样例：aclnnXxxGetWorkSpaceSize、aclnnXxxGetWorkspacesize
   7. 第 1 段接口（特指aclnnXxxGetWorkspaceSize）最后 2 个参数形式必须为：`aclnnStatus aclnnXxxGetWorkspaceSize(xxx, uint64_t* workspaceSize, aclOpExecutor** executor)`
   8. 第 2 段接口（特指aclnnXxx）只允许 4 个参数，参数形式必须为下述两种：
      - `aclnnStatus aclnnXxx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`
      - `aclnnStatus aclnnXxx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)` （注意：executor 参数不允许添加 `const` 修饰符，仅 aclrtStream stream 的 `const` 可有可无。）

### API 参数表里字段规范

- 级别：提示
- 位置：每个 `## （函数接口）`中的 `参数说明` 表格的表头字段。
- 规则：
   1. md 所有表格里的字段名称固定，只允许在以下范围中：{`参数名`、`输入/输出`、`描述`、`使用说明`、`数据类型`、`数据格式`、`维度(shape)`、`非连续Tensor`}
   2. 大小写完全一致。

### API 参数表里参数名列不规范

- 级别：一般
- 位置：每个 `## （函数接口）`中的 `参数说明` 表格的 `参数名` 列内容。
- 规则：
   1. 支持不带参数类型的参数名，要求与函数原型中的参数命名一致。
   2. 支持带参数类型的参数名，要求与函数原型中的参数命名一致。（例如a (aclTensor*)带参数类型，注意括号中英文均可）
   3. 参数名顺序是否与函数原型中的参数顺序一致。
   4. 带参数类型的参数名，其参数类型要与函数原型中参数类型一致（例外：const带不带皆可，*号位置无所谓）。

### API 参数表里输入/输出列不规范

- 级别：一般
- 位置：每个 `## （函数接口）`中的 `参数说明` 表格的 `输入/输出` 列内容。
- 规则：只允许七种书写方式：{`输入`，`输出`，`输入/输出`，`输入 | 输出`，`输入和输出`，`可选输出`，`可选输入`}，其余情况不支持。

### API 参数表里数据类型不规范

- 级别：严重
- 位置：每个 `## （函数接口）`中的 `参数说明` 表格的 `数据类型` 列内容。
- 规则：
   1. 支持文字类描述，保证无语病（例如与xx类型一致、同上等）
   2. 支持文字描述里包含 dtype，要求大写，可以是 ACL_XX 或 XX 形式。
   3. 支持单纯罗列 dtype 范围，要求中文顿号分隔（>=2个）
   4. dtype 范围参见 ops-math 官方[数据类型](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/data_type.md)，按需读取该源文件后判断，超出范围不支持。无法获取源文件时，不得凭记忆判定 dtype 范围。

### API 参数表里数据格式不规范

- 级别：严重
- 位置：每个 `## （函数接口）`中的 `参数说明` 表格的 `数据格式` 列内容。
- 规则：
   1. 支持文字类描述，保证无语病（例如与xx格式一致、同上等）
   2. 支持文字描述里包含 format，要求大写，可以是 ACL_FORMAT_XX 或 XX 形式。
   3. 支持单纯罗列format范围，要求中文顿号分隔（>=2个）
   4. format 范围参见 ops-math 官方[数据格式](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/data_format.md)，按需读取该源文件后判断，超出范围不支持。无法获取源文件时，不得凭记忆判定 format 范围。
   5. FRACTAL_NZ 和 NZ 格式含义一样，均可
   6. 针对基础数据类型（int64_t/float/double/bool等），不存在Tensor，就不会有数据格式的描述，单元格写 `-`（表示不涉及）

### API 参数表里 shape 不规范

- 级别：严重
- 位置：每个 `## （函数接口）`中的 `参数说明` 表格的 `维度(shape)` 列内容。
- 规则：
   1. 支持文字类描述，保证无语病（例如与xx维度一致、与xx shape一致、同上、1~8、[1, 8]、1-8等）
   2. 针对基础数据类型（int64_t/float/double/bool等），不存在Tensor，就不存在shape的描述，单元格写 `-`（表示不涉及）

### API 参数表里非连续 Tensor 支持度不规范

- 级别：严重
- 位置：每个 `## （函数接口）`中的 `参数说明` 表格的 `非连续Tensor` 列内容。
- 规则：针对 `aclTensor*`、`aclTensorList*` 参数类型，才需要判断是否支持非连续 Tensor。否则其他参数类型，不存在非连续判断，单元格写 `-`（表示不涉及）

### API 参数描述不规范

- 级别：一般
- 位置：每个 `## （函数接口）` 与 `## 约束说明` 的内容。
- 规则：内容中不允许出现 `None`和 `默认值`的字眼，如有则提出建议：删除默认值的说法或改为“建议值为xxx”符合 C 语言规范的说法；将 None 改为空指针或 Nullptr 等符合 C 语言规范的说法。

### API 参数表里的固定参数说明不规范

- 级别：提示
- 位置：每个 `## （函数接口）`中的 `参数说明` 表格。
- 规则：
   1. 一阶段参数说明表（即 aclnnXxxGetWorkspaceSize 接口的参数表）中，`workspaceSize` 和 `executor` 两行必须严格满足：
      - `输入/输出` 列均为 `输出`
      - `workspaceSize` 的 `描述` 列必须为：`返回需要在Device侧申请的workspace大小。`
      - `executor` 的 `描述` 列必须为：`返回op执行器，包含了算子计算流程。`
      - 其余列（使用说明、数据类型、数据格式、维度(shape)、非连续Tensor）均为 `-`
   2. 二阶段参数说明表（即 aclnnXxx 接口的参数表）中，`workspace`、`workspaceSize`、`executor`、`stream` 四个参数必须严格满足：
      - `输入/输出` 列均为 `输入`（四个参数须一致）
      - `描述` 列内容需与下述示例保持 80% 以上相似（接口名差异除外）：
         - `workspace`：`在Device侧申请的workspace内存地址。`
         - `workspaceSize`：`在Device侧申请的workspace大小，由第一段接口aclnnXxxGetWorkspaceSize获取。`（其中 aclnnXxxGetWorkspaceSize 为实际接口名）
         - `executor`：`op执行器，包含了算子计算流程。`
         - `stream`：`指定执行任务的Stream。`

### 返回值/错误码不规范

- 级别：严重
- 位置：`## （函数接口）`中的 `返回值` 表格。
- 规则：如有返回值和错误码，则按需读取 opbase 官方头文件 [op_errno.h](https://gitcode.com/cann/opbase/blob/master/include/nnopbase/opdev/op_errno.h)，以其中定义的错误码名称及数值为依据核对；无法获取源文件时，不得凭记忆判定错误码不合法。

### 表格使用规范性

- 级别：提示
- 位置：每个 `## （函数接口）`中的 `参数说明`和 `参数拦截校验`部分。
- 规则：必须是表格形式，需支持 md 和 html 语法格式。

### 芯片差异化描述必须k：v形式

- 级别：一般
- 位置：`## （函数接口）`中的`参数说明`表格后。
- 规则：如违反下述规则，在描述中提醒："芯片差异化描述不符合规范，请按键值对写作，产品名为key，差异内容为value，具体如下：1.中文规范：`key：value`（key+中文冒号+value） 2.英文规范：`key: value`（key+英文冒号+空格+value）"。
   1. 中文规范：key+中文冒号+value，多芯片时使用"顿号"分割多个key。
      - 样例：
         a. 昇腾910B AI处理器：XXX。
         b. 昇腾910B AI处理器、昇腾310P AI处理器：不支持BFLOAT16。
   2. 英文规范：key+英文冒号+空格+value，多芯片时使用"英文逗号+空格"分割多个key。
      - 样例：
         a. Ascend 910B AI Processors: XXX.
         b. Ascend 910 AI Processors, Ascend 910C AI Processors, Ascend 310P AI Processors: XXX.
   3. 差异化描述不允许出现在任何表格内（包括 HTML 表格和 Markdown 表格）。如发现在表格中包含芯片差异化描述，需报错并在描述中提醒："表格仅阐述所有产品的并集描述，差异化内容单独呈现，可在表格下方或约束说明章节说明"。

### 废弃/试验接口说明

- 级别：提示
- 位置：文章大标题下面第一行。
- 规则：如有废弃或试验说明，则进行检查：
   1. 检查位置是否符合规范：位于文章大标题下面第一行。
   2. 检查除提示接口废弃之外，还需明确给出替换接口信息。
- 样例：要求使用下述类似话术（无需完全一致）：
   1. 废弃："须知：该接口后续版本会废弃，请使用最新aclnnQuantMatmulV5接口。"
   2. 试验："须知：该接口为试验特性，后续版本可能存在变更，暂不支持应用于商用产品中。"

### HTML表格语法规范

- 级别：提示
- 位置：全文中各表格
- 规则：检查 HTML 表格内不许混用 Markdown 语法，例如代码段、link等。
- 样例：
   ``` HTML
   <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
   <col style="width: 173px">
   <col style="width: 112px">
   <col style="width: 668px">
   </colgroup>
   <thead>
      <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>`描述`</th> (HTML 表格内混用了 Markdown 语法)
      </tr></thead>
   <tbody>
      <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAbsGetWorkspaceSize获取。</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
      </tr>
   </tbody>
   </table>
   ```

---
