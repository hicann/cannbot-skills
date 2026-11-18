# 算子原型注释规范：字段规范全文（评审权威依据）

> 本文件内容来自《算子原型注释规范_md格式》之"注释规范及导出字段"章节，是 SKILL.md 中评审规则的权威依据。字段与规则编号的对应：算子名→PC-01、原型一致性→PC-01~06、Brief→BF-01~04、Inputs→IN-01~05、Attributes→AT-01~05、Outputs→OT-01~03、Constraints→CT-01~07、Data Types→DT-01、可选字段→OF-01~05、通用格式要求→FM-01~09。

## 产品与芯片映射关系

**表 1** 版本所有产品（发布+在研）

| zh产品名（按新到旧顺序） | en产品名 | 软件芯片名 |
| --- | --- | --- |
| Ascend 960PR&960DT系列产品 | Ascend 960PR&960DT products | 960 |
| Ascend 950PR&950DT系列产品 | Ascend 950PR&950DT products | 950 |
| Atlas A3系列产品 | Atlas A3 products | 910_93 |
| Atlas A2系列产品 | Atlas A2 products | 910b/910Bxx |
| Atlas 200I/500 A2推理产品 | Atlas 200/500 A2 inference products | 310B/310b/310b1、310b2\310bxxx |
| Atlas推理系列产品 | Atlas inference products | 310P/310p/310p1/310P3 |
| Atlas训练系列产品 | Atlas training products | 910 |

## 字段总览

参数说明一共 11 个字段：Brief、Inputs、Attributes、Outputs、Constraints、Data Types、Graph（可选）、see（可选）、Quantization supported or not（可选）、Third-party framework compatibility（可选）、Restrictions（对内）；另有"算子名"来自 REG_OP 定义本身。

除 Data Types 来源于信息库外，其余字段来源均为头文件，获取路径：`op_graph/inc`。

## 算子名（→PC-01）

- 来源：头文件，算子原型定义 `REG_OP(OpType)` 中的 OpType，必须和 OpType 中的内容完全一致
- 获取路径：op_graph/inc

## Brief（→BF-01~04）

- 功能说明：简述算子功能
- 注释字段：`@brief`
- 注释要求：
  - 不能简单写"某某算子功能"，需要详细写出该算子的功能特性
  - `@brief` 后面请不要加冒号，后面功能描述换行呈现
  - 如果功能描述中有公式或者代码段想展示，可以使用 `@code{.c} ... @endcode` 标识符进行标记，`{}` 中为使用的开发语言
  - 功能描述结尾处，不强制加 `\n`；功能描述中一句话未说完，不要使用 `\n` 强制换行，使用回车即可，否则发布后的资料也会强制换行

## Inputs（→IN-01~05、PC-02）

- 功能说明：
  - 描述该算子有几个输入、每个输入的含义、每个输入需要说明可选/必选、每个输入的数据类型的要求、format 的要求、shape 的要求以及取值范围
  - 算子输入如果有特殊要求，需要就近明确说明。可以增加示例代码说明，要求根据描述，能清晰完成接口调用
  - 算子输入之间的约束，统一放在 Constraints 中
  - 如果算子输入有芯片上的差异，也放在 Constraints 中
- 注释字段：`@par Inputs`
- 注释要求：
  - `@par Inputs` 中的 Inputs 大小写以及后面的 s 必须完全一致，否则无法导出
  - 多个参数使用 `@li` 进行区分，单个参数请不要使用该标记
  - 参数名以及个数要和 REG_OP 原型定义保持一致
  - 如果注释中有表格形式的，表格前后增加 `\n`
  - 如果注释中有公式的，请在 `*` 乘号前后都加空格，否则无法导出正确的公式
  - 一句话未说完，请不要使用 `\n`，使用回车换行即可，完整的一句话后面才可以加 `\n`

## Attributes（→AT-01~05、PC-04）

- 功能说明：
  - 描述该算子有几个属性、每个属性需要说明可选/必选、数据类型、取值范围、默认值、属性的使用约束（包括当前仅支持的属性值、属性值的来源限制等）
  - 如果某个属性为预留的，需要说明"预留参数，暂不使用"；可以增加示例代码说明，要求根据描述，能清晰完成接口调用
- 注释字段：`@par Attributes`
- 注释要求：
  - 多个参数使用 `@li` 进行区分，单个参数请不要使用该标记
  - 参数名以及个数要和 REG_OP 原型定义保持一致
  - 如果注释中有表格形式的，请在表格前后增加 `\n`
  - 如果注释中有公式的，请在 `*` 乘号前后都加空格，否则无法导出正确的公式
  - 一句话未说完，请不要使用 `\n`，使用回车换行即可，完整的一句话后面才可以加 `\n`
  - 可选属性必须要有默认值

## Outputs（→OT-01~03、PC-03）

- 功能说明：
  - 描述该算子有几个输出、每个输出的含义、每个输出的数据类型、每个输出的 format、每个输出的 shape
  - 算子输出如果有特殊要求，需要就近明确说明。可以增加示例代码说明，要求根据描述，能清晰完成接口调用
  - 算子输出之间的约束，统一放在 Constraints 中
  - 如果算子输出有芯片上的差异，也放在 Constraints 中
- 注释字段：`@par Outputs`
- 注释要求：
  - `@par Outputs` 中的 Outputs 大小写以及后面的 s 必须完全一致，否则无法导出
  - 多个参数使用 `@li` 进行区分，单个参数请不要使用该标记
  - 参数名以及个数要和原型定义保持一致
  - 如果注释中有表格形式的，请在表格前后增加 `\n`
  - 如果注释中有公式的，请在 `*` 乘号前后都加空格，否则无法导出正确的公式
  - 一句话未说完，请不要使用 `\n`，使用回车换行即可，完整的一句话后面才可以加 `\n`

## Constraints（→CT-01~07）

- 功能说明：
  - 描述使用该算子的约束，例如某算子只支持在某些特定网络下使用、多个输入配合起来的约束、多个属性配合起来的约束、对上下层算子的要求等；单个参数的约束就近增加到参数后面
  - 算子在某个芯片下有约束，需要特别说明的，需要在此说明；具体的芯片名称可以找资料获取，严禁出现 910B、910C、V350 等字眼
  - 算子废弃的说明：如果算子已废弃或者将要废弃，在该字段中增加说明，如果有替代的算子，也要同步给出，话术为：This operator will be deprecated in the future. Replace it with the xx operator. xx 为替代的算子
- 注释字段：`@attention Constraints`
- 注释要求：
  - 若有多条注释，请使用 `@li` 进行区分；或者不使用 `@li`，在每一条完整约束后使用 `\n` 换行符
  - 如果注释中有表格形式的，请在表格前后增加 `\n`
  - 如果注释中有公式的，请在 `*` 乘号前后都加空格，否则无法导出正确的公式（原文此条写"乘号前面"，"算子头文件注释写作样例"章节要求"前后"，以样例章节为准）
  - 一句话未说完，请不要使用 `\n`，使用回车换行即可，完整的一句话后面才可以加 `\n`
- 写作样例：

  ```
  *@attention Constraints
  *//约束，有约束的话，需要使用该标识符；如果有针对芯片的约束，请使用如下的描述方式
  *@li Atlas A2 products: 910B芯片描述  \n
  *    If the xxx .... \n
  *    If the xxx .... \n（此处的换行符可加可不加，识别到下面的@li会自动分段）
  *@li Atlas A3 products: 910C芯片英文描述  \n
  *    If the xxx .... \n
  *    If the xxx ....
  ```

  注：样例中的"910B芯片描述""910C芯片英文描述"是教学标注，实际注释中只写产品名加英文描述，严禁出现芯片代号。

## Data Types（→DT-01）

- 来源：信息库
- 获取路径：
  - ai_core 算子：opp/built-in/op_impl/ai_core/tbe/config
  - aicpu 算子：opp/built-in/op_impl/aicpu/aicpu_kernel/config
  - dis_core：opp/built-in/op_impl/dsa_core/config
  - dvpp：opp/built-in/op_impl/dvpp/dvpp_kernel/config
  - vector_core：opp/built-in/op_impl/vector_core/tbe/config
- 功能说明：当前芯片支持的数据类型，不要求和 REG_OP 原型定义中的类型保持一致，但是要求是其子集；原型针对所有芯片，而该字段只适用于当前芯片
- 注释字段：`@par Data Types`
  - 该字段不需要手动处理，由脚本读取信息库后进行增加，和原型内容进行合并
  - 算子信息库 input 或者 output 存在 dtype(type) 时，增加 `@par Data Types` 字段
  - 注：部分融合算子（需要人为梳理）和 GE Local 算子没有 Data Types 信息
- 注释要求：无

## Graph（可选）（→OF-01）

- 功能说明：注册算子中包含的子图信息，一般 GE 算子才会有该字段
- 注释字段：`@par Graphs`
- 注释要求：
  - 多个参数使用 `@li` 进行区分，单个参数请不要使用该标记
  - 参数名以及个数要和 REG_OP 原型定义保持一致
  - 如果注释中有表格形式的，请在表格前后增加 `\n`
  - 如果注释中有公式的，请在 `*` 乘号前后都加空格，否则无法导出正确的公式

## see（可选）（→OF-02）

- 功能说明：需要填写相关的算子，比如，反向算子的关联算子是正向的，如无，可删除
- 注释字段：`@par see`
- 注释要求：无

## Quantization supported or not（可选）（→OF-03）

- 功能说明：算子是否支持量化，不支持量化，该字段可不加
- 注释字段：`@par Quantization supported or not`
- 注释要求：无
- 写作样例：

  ```
  * @par Quantization supported or not
  * Yes
  ```

## Third-party framework compatibility（可选）（→OF-04）

- 功能说明：列举本算子对标的第三方友商算子
- 注释字段：`@par Third-party framework compatibility`
- 注释要求：无
- 写作样例：

  ```
  * @par Third-party framework compatibility
  * Compatible with the TensorFlow operator UnsortedSegmentSum
  ```

## Restrictions（对内）（→OF-05）

- 功能说明：仅内部使用算子所用的标识，该字段不会导出；如果有该字段，表示算子是内部算子，不在对外导出的算子列表中
- 注释字段：`@par Restrictions`
- 注释要求：该字段是否填写有两个原则：
  1. SE 从设计技术角度明确该原型是否对用户发布，还是仅内部使用；如果定义为内部则必须添加该字段，如果有对应的外部发布替代 IR，需要说明清楚，使用写作样例中的第一句注释，没有的话选择下面一句注释
  2. 对于 SE 明确了可以对外发布，但是未调试完成，或者测试验收未给出可对外发布结论的原型定义，都按内部使用添加字段。后续如测试验收通过，提单删除
- 写作样例：

  ```
  *Warning: THIS FUNCTION IS DEPRECATED. Please use AvgPool1D(算子名称为举例，请根据实际替换) instead.
  *Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
  //如果有对应的可替换IR，需要说明清楚使用第一句注释。没有的话选择下面一句注释
  ```
