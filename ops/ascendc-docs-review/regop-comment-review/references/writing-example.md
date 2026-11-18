# 算子头文件注释写作样例（标准写法参考）

> 本文件内容来自《算子原型注释规范_md格式》之"算子头文件注释写作样例"章节。给用户输出"修改写法"时，按此样例的风格和格式编写。样例代码中 `*//` 之后的内容是规范说明（教学标注），不是注释正文；芯片约束条目中的"910B芯片描述"等同样是标注，实际注释中**严禁出现芯片代号**。

## 总体原则

- 详细描述中，如果有多个段落，每段必须以 `\n` 结束。使用空行划分段落。
- 算子头文件注释以 `/** ... */` 形式开头和结尾。
- 注释中的换行要求：必须换行的地方使用 `\n` 换行；如果一句话没有说完，可以回车换行，但是不要加 `\n` 换行符，否则会导致导出的资料也换行。

## 完整样例（Conv2D）

```
/**
* @brief
*//功能简介，格式严格按@brief形式，brief后面不要加冒号，后面的功能解释，换行展示
* Computes a 2D convolution given 4D "x" and "filter" tensors.
* @par Inputs
*//输入参数，格式按@par Inputs形式，如果有>=2个以上的输入，则每个输入必须使用@li标识进行区分，一个输入场景下请不要使用@li标识。@li后面完整的一句话，可以不用加\n换行符；如果多句话，需要强制换行的，则使用\n。
* @li x: A 4D tensor of input image. With the format "NHWC", the data is stored
* in the order of: [batch, in_height, in_width, in_channels].
* @li filter: A 4D tensor of learnable filters. Must have the same type as "x".
* With the format "HWCN" , the data is stored in the order of: [filter_height,
* filter_width, in_channels / groups, out_channels].
* @li bias: An optional 1D tensor of additive biases to the filter outputs.
* The data is stored in the order of: [out_channels].
* @li offset_w: Reserved.
*\n
* The following are the supported data types and data formats:
*//如果注释中有表格形式的，表格前面使用一个\n，表格后面使用一个\n。
*\n
| Tensor    | x       | filter  | bias    | y       |
| :-------: | :-----: | :-----: | :-----: | :-----: |
| Data Type | float16 | float16 | float16 | float16 |
|           | float16 | float16 | float16 | float32 |
|           | bfloat16| bfloat16| bfloat16 | bfloat16|
|           | bfloat16| bfloat16| bfloat16 | float32 |
|           | float32 | float32 | float32 | float32 |
|           | int8    | int8    | int32   | int32   |
|           | int4    | int4    | int32   | int32   |
| Format    | NCHW    | NCHW    | ND      | NCHW    |
|           | NHWC    | HWCN    | ND      | NHWC    |
|           | NCHW    | HWCN    | ND      | NCHW    |
*\n
* For float32 type, the actual calculation on the chip is based on
* float16.
*\n
* Data types for Ascend 610Lite AI Processor:
| Tensor    | x       | filter  | bias    | y       |
| :-------: | :-----: | :-----: | :-----: | :-----: |
| Data Type | int16   | int8    | int32   | int32   |
|           | int8    | int8    | int32   | int32   |
| Format    | NCHW    | NCHW    | ND      | NCHW    |
|           | NHWC    | HWCN    | ND      | NCHW    |
*\n
*
* @par Attributes
*//属性中的参数名必须和原型定义中的保持一致，>=2个参数使用@li标识进行区分，单个参数请不要使用该标识；如果注释中有表格，请在每一个行后面增加\n换行符，表格前面必须使用两个\n，表格后面使用一个\n。@li后面完整的一句话，可以不用加\n换行符；如果多句话，需要强制换行的，则使用\n。
* @li strides: Required. A list of 4 integers. The stride of the sliding window
* for each dimension of input. The dimension order is determined by the data
* format of "x". The N and C dimensions must be set to 1.
* @li pads: Required. A list of 4 integers. The number of pixels to add to each
* (top, bottom, left, right) side of the input.
* @li dilations: Optional. A list of 4 integers. The dilation factor for each
* dimension of input. The dimension order is determined by the data format of
* "x". The N and C dimensions must be set to 1. Defaults to [1, 1, 1, 1].
* @li groups: Optional. An integer of type int32. The number of blocked
* connections from input channels to output channels. In_channels and
* out_channels must both be divisible by "groups". Defaults to 1.
* @li offset_x: Optional. An integer of type int32. The negative offset added
* to the input image for int8 type. Ensure that the output is within the
* effective range. Defaults to 0.
* @li data_format: Reserved.
*\n
* The following value range restrictions must be met:
*\n
| Name             | Field    | Scope       |
| :--------------: | :------: | :---------: |
| Input Image Size | H        | [1, 100000] |
|                  | W        | [1, 4096]   |
| Filter Size      | H        | [1, 255]    |
|                  | W        | [1, 255]    |
| Stride           | H        | [1, 63]     |
|                  | W        | [1, 63]     |
| Padding          | Top      | [0, 255]    |
|                  | Bottom   | [0, 255]    |
|                  | Left     | [0, 255]    |
|                  | Right    | [0, 255]    |
| Dilation         | H        | [1, 255]    |
|                  | W        | [1, 255]    |
| Offset_x         | -        | [-128, 127] |
*\n
* The W dimension of the input image supports cases exceeding 4096, but it may
* cause compilation errors.
*\n
* If any dimension of Input/Filter/Bias/Output shape exceeds max int32(2147483647),
* the product of each dimension of Input/Filter/Bias/Output shape exceeds max int32(2147483647) or
* the value of stride/padding/dilation/offset_x exceeds the range in the above table,
* the correctness of the function cannot be guaranteed.
*\n
* If the conv2d enters the DMA process, a timeout AiCore error may be reported.
* You are advised to reduce the conv2d specifications and try again.
* You can view the warning log to check whether the DMA process is entered.
* For example: 'The conv2d has entered the DMA processing process. A timeout AiCore error may be reported.
* If a timeout AiCore error is reported, reduce the conv2d specifications and try again'
*\n
*@par Outputs
*//输出参数，格式完全按照@par Outputs形式，如果有>=2个以上的输出，则每个输出必须使用@li标识进行区分，一个输出场景下请不要使用@li标识；如果注释中有公式，请在乘号（*）前后都加空格。@li后面完整的一句话，可以不用加\n换行符；如果多句话，需要强制换行的，则使用\n。
* y: A 4D Tensor of output feature map. Has the same type as "x". With the
* format "NHWC", the data is stored in the order of: [batch, out_height,
* out_width, out_channels].
*\n
*     out_height = (in_height + pad_top + pad_bottom -
*                   (dilation_h * (filter_height - 1) + 1))
*                  / stride_h + 1
*\n
*     out_width = (in_width + pad_left + pad_right -
*                  (dilation_w * (filter_width - 1) + 1))
*                 / stride_w + 1
*\n
*
*@attention Constraints
*//约束，有约束的话，需要使用该标识符；如果有针对芯片的约束，请使用如下的描述方式（存量注释沿用不改，新增按此规范）
*@li Atlas 200/500 A2 inference products: 310B芯片描述  \n
*    If the xxx .... \n
*    If the xxx .... \n
*@li Atlas inference products: 310P芯片英文描述  \n
*    If the xxx .... \n
*    If the xxx ....
*@li Atlas training products: 910芯片描述  \n
*    If the xxx .... \n
*    If the xxx ....
*@li Atlas A2 products: 910B芯片描述  \n
*    If the xxx .... \n
*    If the xxx ....
*@li Atlas A3 products: 910C芯片描述  \n
*    If the xxx .... \n
*    If the xxx ....
*@li Ascend 950PR&950DT products: 910D芯片描述，目前使用这个，后面确定后资料统一替换头文件  \n
*    If the xxx .... \n
*    If the xxx ....
*@li Ascend 610 AI Processor: 610芯片描述  \n
*    If the xxx .... \n
*    If the xxx ....
*@li BS9SX1A AI Processor: 610B芯片描述  \n
*    If the xxx .... \n
*    If the xxx ....
*@li BS9SX2A AI Processor: 610Lite芯片描述  \n
*    If the xxx .... \n
*    If the xxx ....
*@li MC62CM12AA AI Processor: 620芯片描述  \n
*    If the xxx .... \n
*    If the xxx ....
*@li AS31XM1X AI Processor: 310M芯片描述  \n
*    If the xxx .... \n
*    If the xxx ....
*//如果有类似二级标题的，可以用如下方式呈现
* - Atlas A2 products:
* @li:xx
* @li:x
* - Atlas A3 products:
* @li:xx
* @li:xx
*
* @par Quantization supported or not
* Yes

*@par Third-party framework compatibility
*@li Compatible with the TensorFlow operator "conv2d".
*@li Compatible with the Caffe operator 2D "Convolution".
*/
Conv2D();
```

**再次强调**：上例芯片约束条目中的"310B芯片描述""910B芯片描述"等为教学标注。实际注释中每个条目只写产品名 + 英文描述，例如：

```
*@li Atlas A2 products: The scenario where xxx is used has constraints.  \n
*    If the xxx ....
```

## 样例要点提炼

| 位置 | 要点 |
| --- | --- |
| @brief | 不加冒号；功能解释换行展示 |
| @par Inputs | ≥2 个输入每个用 `@li`；单个输入不用 `@li`；`@li` 后完整一句可不加 `\n` |
| Inputs 中的表格 | 表格前一个 `\n`，表格后一个 `\n` |
| @par Attributes | 参数名与原型定义一致；≥2 个参数用 `@li`；表格每行后加 `\n`，表格前两个 `\n`、表格后一个 `\n` |
| @par Outputs | ≥2 个输出每个用 `@li`；单个输出不用 `@li`；公式乘号前后加空格 |
| @attention Constraints | 芯片约束按产品分 `@li` 条目；多条约束用 `@li` 或每条完整约束后 `\n` |
| 公式 | 乘号 `*` 前后加空格；公式行用空格缩进呈现 |

## 参数描述推荐句式

- 输入：`@li <参数名>: <An optional/A/Required ...> tensor of <含义>` + 数据类型/format/shape/取值范围
- 属性：`@li <参数名>: Required/Optional. <数据类型与含义>. Defaults to <默认值>.`
- 预留参数：`@li <参数名>: Reserved.`

## 废弃与内部算子话术

- 废弃算子（写在 Constraints 中）：`This operator will be deprecated in the future. Replace it with the xx operator.`（xx 为替代算子名）
- 内部算子（`@par Restrictions`，二选一）：
  - 有外部替代：`Warning: THIS FUNCTION IS DEPRECATED. Please use Xxx instead.`（Xxx 为实际替代算子名）
  - 无替代：`Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.`
