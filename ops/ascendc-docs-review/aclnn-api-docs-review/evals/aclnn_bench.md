# aclnn领域验证集

> 说明：本文提供aclnn API规则的测试样例集，后续支持不断扩充，欢迎大家共同构建！

## 问题1

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：1
- 问题上下文：# aclnndropoutV3
- 问题描述：一级标题不规范，不是大驼峰格式，且与aclnnDropoutV3.md不一致
- 修改建议：请修改文档一级标题，要求驼峰且与文件命名一致
- 严重程度：严重

## 问题2

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：236
- 问题上下文：## 支持的产品
- 问题描述：“支持的产品”和“产品支持情况”章节定位冗余或矛盾
- 修改建议：请根据实际情况修改章节标题，确保层级清晰、内容明了
- 严重程度：提示

## 问题3

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：25
- 问题上下文：- <term>610Lite</term>：支持
- 问题描述：产品型号不规范，产品型号不在支持的范围内
- 修改建议：请将“610Lite”改为正确的产品型号名
- 严重程度：严重

## 问题4

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：20
- 问题上下文：- Atlas 推理系列产品：不支持
- 问题描述：产品型号不规范，需成对加上<term>标记对
- 修改建议：请“Atlas 推理系列产品”修改为“<term>Atlas 推理系列产品</term>”
- 严重程度：严重

## 问题5

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：43
- 问题上下文：aclnnDropoutV3GetWorkspaceSize(
- 问题描述：函数原型不规范，缺少返回值aclnnStatus
- 修改建议：请在API原型中补齐返回值aclnnStatus
- 严重程度：一般

## 问题6

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：45
- 问题上下文：aclTensor*         optionalNoise_Shape,
- 问题描述：函数原型不规范，aclnn API参数名要求小驼峰而非下划线形式
- 修改建议：请检查头文件“optionalNoise_Shape”参数是否为小驼峰，资料与软件保持一致
- 严重程度：一般

## 问题7

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：48
- 问题上下文：int64_t            offset, aclTensor*       out,
- 问题描述：函数原型不规范，要求单行单参数呈现，offset与out参数同行呈现
- 修改建议：请将参数“out”单独换行呈现，并与其他参数缩进对齐
- 严重程度：一般

## 问题8

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：56
- 问题上下文：void *workspace,
- 问题描述：函数原型不规范，workspace与其他参数上下缩进位置不平齐
- 修改建议：参数“workspace”的缩进空格需要与其他参数缩进位置平齐
- 严重程度：一般

## 问题9

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：59
- 问题上下文：aclrtStream          stream)：
- 问题描述：函数原型不规范，末尾不加分号或其他符号
- 修改建议：请删除原型定义末尾的冒号
- 严重程度：一般

## 问题10

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：58
- 问题上下文：const aclOpExecutor *executor,
- 问题描述：函数原型不规范，固定参数aclOpExecutor *executor不应该带“const”
- 修改建议：请删除固定参数中冗余的const
- 严重程度：一般

## 问题11

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：79
- 问题上下文：<th>输入/输出、属性</th>
- 问题描述：表格字段不规范，参数定位没有“属性”
- 修改建议：请将表格字段“输入/输出、属性”改为“输入/输出”
- 严重程度：提示

## 问题12

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：89
- 问题上下文：<td>inputs</td>
- 问题描述：参数表中参数名不规范，参数inputs与函数原型input不一致
- 修改建议：请根据头文件正确定义，修改参数表或函数原型的参数名
- 严重程度：一般

## 问题13

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：159
- 问题上下文：<td>executor</td>
- 问题描述：参数表中参数顺序不规范，参数executor和workspaceSize顺序与函数原型不一致
- 修改建议：请根据头文件定义，调整参数顺序使得参数表与函数原型顺序一致
- 严重程度：一般

## 问题14

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：90
- 问题上下文：<td>变量</td>
- 问题描述：参数表中输入/输出列不规范，不存在“变量”写法
- 修改建议：请按规范填写参数对应的输入/输出列信息
- 严重程度：一般

## 问题15

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：93
- 问题上下文：<td>float、FLOAT16、BFLOAT16、INT64</td>
- 问题描述：参数表中数据类型不规范，dtype要求大写，一般为ACL_FLOAT或简写FLOAT
- 修改建议：请将float改为大写，一般为ACL_FLOAT或简写FLOAT
- 严重程度：严重

## 问题16

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：103
- 问题上下文：<td>QINT32</td>
- 问题描述：参数表中数据类型不规范，QINT32不在支持的aclDataType范围内
- 修改建议：请修改QINT32数据类型，使用aclDataType支持的数据类型
- 严重程度：严重

## 问题17

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：144
- 问题上下文：<td>nd</td>
- 问题描述：参数表中数据格式不规范，format要求大写，一般为ACL_FORMAT_ND或简写ND
- 修改建议：请将nd改为大写，一般为ACL_FORMAT_ND或简写ND
- 严重程度：严重

## 问题18

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：94
- 问题上下文：<td>ND、NC1HW</td>
- 问题描述：参数表中数据格式不规范，NC1HW不在支持的aclFormat范围内
- 修改建议：请修改NC1HW数据格式，使用aclDataType支持的数据类型
- 严重程度：严重

## 问题19

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：114
- 问题上下文：<td>ND</td>
- 问题描述：参数表中数据格式不规范，基础类型（如int64_t/double/bool等）的参数不存在数据格式的说法
- 修改建议：参数p不存在数据格式描述，请填写英文中横线-，表示不涉及
- 严重程度：严重

## 问题20

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：125
- 问题上下文：<td>1-4</td>
- 问题描述：参数表中shape不规范，基础类型参数（如int/double/bool等）不存在shape描述
- 修改建议：参数seedOptional不存在shape描述，请填写英文中横线-，表示不涉及
- 严重程度：严重

## 问题21

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：116
- 问题上下文：<td>×</td>
- 问题描述：参数表中非连续Tensor不规范，基础类型（如int64_t/double/bool等）的参数不存在张量连续性说法
- 修改建议：参数p不存在非连续描述，请填写英文中横线-，表示不涉及
- 严重程度：严重

## 问题22

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：173
- 问题上下文：<td>INT64</td>
- 问题描述：workspaceSize参数类型为uint64_t*，表格中填写INT64不匹配。
- 修改建议：参数workspaceSize的数据类型无需填写，使用英文中横线-，表示不涉及
- 严重程度：严重

## 问题23

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：112
- 问题上下文：<td>默认值为0.5</td>
- 问题描述：参数表中不应该出现“默认值”的说法，请遵循C语言规范
- 修改建议：请删除默认值的说法或改为“建议值为0.5”
- 严重程度：一般

## 问题24

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：200
- 问题上下文：<td>传入的input、out、maskOut为None。</td>
- 问题描述：参数表中不应该出现“None”，Python语言支持，C语言不支持
- 修改建议：请将None改为空指针或Nullptr等符合C语言规范的说法
- 严重程度：一般

## 问题25

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：203
- 问题上下文：<td rowspan="5">ACLNN_ERR_PARAMs_INVALID</td>
- 问题描述：入参校验表中返回码ACLNN_ERR_PARAMs_INVALID拼写错误
- 修改建议：请根据op_errno.h定义修改“ACLNN_ERR_PARAMs_INVALID”，疑似为“ACLNN_ERR_PARAM_INVALID”
- 严重程度：严重

## 问题26

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：226
- 问题上下文：- workspace（入参）：在Device侧申请的workspace内存地址。
- 问题描述：参数说明请优先采用表格样式（md或html均可）
- 修改建议：请将二段接口的参数说明从无序列表改为表格样式
- 严重程度：提示

## 问题27

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：233
- 问题上下文：- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>不支持BFLOAT16。
- 问题描述：芯片差异化描述格式不规范，请用键值对描述（产品：差异），不推荐其他形式。
- 修改建议：请修改为“<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持BFLOAT16。”
- 严重程度：提示

## 问题28

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：36
- 问题上下文：- 说明：该接口将从CANN 9.0.0版本开始废弃！
- 问题描述：接口废弃声明位置和话术不符合规范
- 修改建议：请修改废弃声明，位置要求在文档一级标题下方第一行，明确废弃接口和替换的新接口。
- 严重程度：提示

## 问题29

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：规范性
- 文件路径：aclnnDropoutV3.md
- 位置行号：141
- 问题上下文：<td>公式中的`out`</td>
- 问题描述：Html表格里公式混用md语法（反引号），导致预览效果不佳。
- 修改建议：请将`out`改为Html语法<code>out</code>
- 严重程度：提示

## 问题30

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：完备性
- 文件路径：aclnnDropoutV3.md
- 位置行号：40
- 问题上下文：每个算子分为两段式接口：
- 问题描述：函数原型缺少两段式接口调用过程说明
- 修改建议：请在函数代码段上方，补齐类似话术“每个算子分为两段式接口，先调用aclnnXxxGetWorkspaceSize接口获取xxx，再调用aclnnXxx接口执行计算。”
- 严重程度：一般

## 问题31

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：完备性
- 文件路径：aclnnDropoutV3.md
- 位置行号：92
- 问题上下文：<td>shape支持0-8维，形如[M, N/2]</td>
- 问题描述：参数inputs的使用说明列信息不完整，缺少空Tensor支持度说明
- 修改建议：请在参数inputs“使用说明”列补齐空Tensor支持度说明
- 严重程度：提示

## 问题32

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：完备性
- 文件路径：aclnnDropoutV3.md
- 位置行号：122
- 问题上下文：<td>-</td>
- 问题描述：参数seedOptional使用说明列信息不完整，基础类型（int）缺少取值范围说明
- 修改建议：请在参数seedOptional“使用说明”列补齐取值说明
- 严重程度：提示

## 问题33

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：完备性
- 文件路径：aclnnDropoutV3.md
- 位置行号：183
- 问题上下文： aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
- 问题描述：一段接口入参校验表上方缺少必要的使用说明
- 修改建议：请在入参校验表上方补齐使用说明，类似“第一段接口完成入参校验，出现以下场景时报错”
- 严重程度：严重

## 问题34

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：完备性
- 文件路径：aclnnDropoutV3.md
- 位置行号：229
- 问题上下文：  - stream（入参）：指定执行任务的Stream。
- 问题描述：二段接口缺少返回值说明
- 修改建议：请补齐接口对应的返回值说明
- 严重程度：严重

## 问题35

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：完备性
- 文件路径：aclnnDropoutV3.md
- 位置行号：231
- 问题上下文：## 约束说明
- 问题描述：约束说明缺少“确定性计算说明”
- 修改建议：请在约束说明补齐接口确定性计算说明，支持按不同芯片描述
- 严重程度：严重

## 问题36

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：完备性
- 文件路径：aclnnDropoutV3.md
- 位置行号：249
- 问题上下文：仅供参考，具体编译和执行过程请参考编译与运行样例
- 问题描述：“编译与运行样例”缺少必要的跳转链接
- 修改建议：请补齐“编译与运行样例”跳转链接（使用相对路径），以指导用户运行调用示例
- 严重程度：提示

## 问题37

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：完备性
- 文件路径：aclnnDropoutV3.md
- 位置行号：29
- 问题上下文：- 接口功能：训练过程中，按照概率p随机将输入中的元素置零，并将输出按照1/(1-p)的比例缩放。
- 问题描述：针对V版本接口，缺少必要的版本演进说明
- 修改建议：请在“功能说明”章节阐述多个V版本接口的功能差异
- 严重程度：提示

## 问题38

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：正确性
- 文件路径：aclnnDropoutV3.md
- 位置行号：43
- 问题上下文：aclnnDropoutV3GetWorkspaceSize(
- 问题描述：md函数原型与aclnn头文件里的参数定义不一致（例如const）
- 修改建议：请检查md函数原型参数类型、参数名、参数顺序是否与头文件一致
- 严重程度：严重

## 问题39

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：正确性
- 文件路径：aclnnDropoutV3.md
- 位置行号：47
- 问题上下文：int64_t            seedOptional,
- 问题描述：基础类型（int/bool等）参数不允许可选，参数名不支持Optional后缀
- 修改建议：请检查文档与aclnn头文件定义是否一致，参数seedOptional需删除后缀Optional
- 严重程度：提示

## 问题40

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：正确性
- 文件路径：aclnnDropoutV3.md
- 位置行号：121
- 问题上下文：<td>随机。</td>
- 问题描述：参数seedOptional含义不明，“随机”歧义
- 修改建议：请修改参数seedOptional含义描述
- 严重程度：提示

## 问题41

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：正确性
- 文件路径：aclnnDropoutV3.md
- 位置行号：255
- 问题上下文：#include "aclnnop/aclnn_dropout_v4.h"
- 问题描述：文档调用示例与examples目录下的cpp代码不一致
- 修改建议：请参考examples目录下cpp代码，文档需要保持一致
- 严重程度：严重

## 问题42

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：可读性
- 文件路径：aclnnDropoutV3.md
- 位置行号：92
- 问题上下文：<td>shape支持0-8维，形如[M, N/2]</td>
- 问题描述：参数inputs“使用说明”列信息冗余，shape描述与“维度(shape)”列信息重复
- 修改建议：请删除参数inputs“使用说明”列维度冗余描述
- 严重程度：提示

## 问题43

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：可读性
- 文件路径：aclnnDropoutV3.md
- 位置行号：208
- 问题上下文：<td>p、seed的取值不在支持范围内。</td>
- 问题描述：参数说明表里seedOptional与入参校验表里seed写法不一致
- 修改建议：请修改入参校验表中seed参数名，需与参数说明表保持一致
- 严重程度：严重

## 问题44

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：可读性
- 文件路径：aclnnDropoutV3.md
- 位置行号：227
- 问题上下文：由第一段接口aclnnDropoutV3GetWorkSpaceSize获取
- 问题描述：一阶段接口命名前后文写法不一致
- 修改建议：请将aclnnDropoutV3GetWorkSpaceSize修改为aclnnDropoutV3GetWorkspaceSize
- 严重程度：严重

## 问题45

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：可读性
- 文件路径：aclnnDropoutV3.md
- 位置行号：349
- 问题上下文：// 调用aclnnDropoutV2第一段接口
- 问题描述：调用示例的代码注释使用的接口名与前文不一致
- 修改建议：请将“aclnnDropoutV2”修改为“aclnnDropoutV3”
- 严重程度：严重

## 问题46

- 检查引擎：DomainCheck_aclnn_API
- 问题类型：可读性
- 文件路径：aclnnDropoutV3.md
- 位置行号：234
- 问题上下文：- <term>Atlas 训练系列产品</term>：输入参数不支持INT64。
- 问题描述：产品支持度前后矛盾，前文不支持Atlas 训练系列产品，但后文有该产品相关描述
- 修改建议：请删除API不支持的产品描述
- 严重程度：严重
