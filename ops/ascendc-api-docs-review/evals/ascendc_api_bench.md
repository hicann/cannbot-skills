# DomainCheck_AscendC_API结果集

## 问题1

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：完备性
- 文件路径：asc_ldca.md
- 位置行号：3
- 问题上下文：## 功能说明
- 问题描述：API章节结构顺序错误
- 修改建议：“功能说明”应在“产品支持情况”后，请调整顺序
- 严重程度：严重

## 问题2

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：完备性
- 文件路径：__float2half_rn.md
- 位置行号：3
- 问题上下文：## 功能说明
- 问题描述：API章节结构缺少“产品支持情况”
- 修改建议：请增加“产品支持情况”
- 严重程度：严重

## 问题3

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：完备性
- 文件路径：__bfloat162bfloat16_rd.md
- 位置行号：13
- 问题上下文：## 函数原型
- 问题描述：API章节结构缺少“功能说明”
- 修改建议：请增加“功能说明”
- 严重程度：严重

## 问题4
- 检查引擎：DomainCheck_AscendC_API
- 问题类型：完备性
- 文件路径：__float2half_rn.md
- 位置行号：7
- 问题上下文：## 参数说明
- 问题描述：API章节结构缺少“函数原型”
- 修改建议：请增加“函数原型”
- 严重程度：严重

## 问题5

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：完备性
- 文件路径：__bfloat162bfloat16_rd.md
- 位置行号：19
- 问题上下文：## 返回值说明
- 问题描述：API章节结构缺少“参数说明”
- 修改建议：请增加“参数说明”
- 严重程度：严重

## 问题6

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：完备性
- 文件路径：__bfloat162bfloat16_rd.md
- 位置行号：23
- 问题上下文：## 需要包含的头文件
- 问题描述：API章节结构缺少“约束说明”
- 修改建议：请增加“约束说明”
- 严重程度：严重

## 问题7

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：完备性
- 文件路径：Trunc.md
- 位置行号：159
- 问题上下文：## 约束说明<a name="section633mcpsimp"></a>
- 问题描述：API章节结构缺少“返回值说明”
- 修改建议：请增加“返回值说明”
- 严重程度：严重

## 问题8

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：完备性
- 文件路径：Trunc.md
- 位置行号：156
- 问题上下文：-   操作数地址对齐要求请参见[通用地址对齐约束](../../../通用说明和约束.md#section796754519912)。
- 问题描述：API章节结构缺少“调用示例”
- 修改建议：请增加“调用示例”
- 严重程度：严重

## 问题9

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：完备性
- 文件路径：Trunc.md
- 位置行号：52
- 问题上下文：按元素做浮点数截断，即向零取整操作。计算公式如下：
- 问题描述：缺少公式或数学表达式
- 修改建议：请补充公式
- 严重程度：一般

## 问题10

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：完备性
- 文件路径：asc_ldcg.md
- 位置行号：173
- 问题上下文：| 参数名 | 描述 |
- 问题描述：参数说明表格缺少“输入/输出”列
- 修改建议：请修改表格，包含“参数名”、“输入/输出”、“描述”三列
- 严重程度：严重

## 问题11

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：完备性
- 文件路径：Tanh.md
- 位置行号：120
- 问题上下文：</tbody>
- 问题描述：模版参数说明表格缺少对config参数的说明
- 修改建议：请在模版参数说明表格中增加config参数的说明，以保证与函数原型中的参数保持一致
- 严重程度：一般

## 问题12

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：完备性
- 文件路径：Tanh.md
- 位置行号：193
- 问题上下文：结果示例如下：
- 问题描述：未提供关键代码片段作为调用示例
- 修改建议：请补充关键代码片段
- 严重程度：严重

## 问题13

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：完备性
- 文件路径：__float2half_rz.md
- 位置行号：33
- 问题上下文：## 返回值说明
- 问题描述：未说明返回值的含义和取值范围
- 修改建议：请说明返回值的含义和取值范围
- 严重程度：严重

## 问题15

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：可读性
- 文件路径：AscendQuant.md
- 位置行号：804
- 问题上下文：源操作数与操作数可以复用。
- 问题描述：约束描述不清晰，有歧义
- 修改建议：需要明确源操作数与什么操作数可以进行复用。
- 严重程度：一般

## 问题16

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：可读性
- 文件路径：AscendQuant.md
- 位置行号：781
- 问题上下文：n：n方向元素个数。n值对应的数据大小需满足32字节对齐的要求，即shape最后一维为n的输入输出均需要满足该维度上32B对齐的要求。
- 问题描述：字节和B前后文使用的单位不一致
- 修改建议：请将B修改为字节，保持上下文单位一致。
- 严重程度：一般

## 问题17

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：完备性
- 文件路径：ReduceSum-90.md
- 位置行号：170
- 问题上下文：Atlas A3 推理系列产品，当前只支持true。
- 问题描述：未说明srcInnerPad参数的完整约束。
- 修改建议：补充“Atlas A2 训练系列产品只支持true”。
- 严重程度：一般

## 问题18

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：完备性
- 文件路径：__hmul-154.md
- 位置行号：36
- 问题上下文：无
- 问题描述：未完整说明不同返回值的场景和意义。
- 修改建议：接口有返回值，需要说明返回值含义。
- 严重程度：一般

## 问题19

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：正确性
- 文件路径：asc_ldcg.md
- 位置行号：22
- 问题上下文：inline long int asc_ldcg(long int * address)
- 问题描述：函数原型与代码不一致
- 修改建议：修改为与代码一致：inline long int asc_ldcg(long int* address)
- 严重程度：严重

## 问题20

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：正确性
- 文件路径：asc_ldca.md
- 位置行号：78
- 问题上下文：inline char2 asc_ldca(char2* address)
- 问题描述：未列出所有函数原型
- 修改建议：增加遗漏的函数原型：inline uchar2 asc_ldca(uchar2* address)
- 严重程度：严重

## 问题21

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：正确性
- 文件路径：ReduceMean.md
- 位置行号：67
- 问题上下文：__aicore__ inline void ReduceMean(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t srcShape[], bool srcInnerPad)
- 问题描述：未列出函数原型中的模版参数
- 修改建议：增加遗漏的模版参数：template <class T, class pattern, bool isReuseSource = false>
- 严重程度：提示

## 问题22

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：正确性
- 文件路径：AscendQuant.md
- 位置行号：867
- 问题上下文：输出数据（dstLocal）: [-9 -9 -9 ... 11 510. 11.]
- 问题描述：PER\_CHANNEL量化场景示例输出最后一个数据11不准确
- 修改建议：PER\_CHANNEL量化场景示例输出最后一个数据为511
- 严重程度：一般

## 问题23

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：正确性
- 文件路径：AscendQuant.md
- 位置行号：810
- 问题上下文：PER\_TOKEN/PER\_GROUP场景，连续计算方向（即n方向）的数据量要求16B对齐。
- 问题描述：约束说明上下文不一致
- 修改建议：上下文对于n方向的数据量的对齐要求保持一致
- 严重程度：一般

## 问题24

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：规范性
- 文件路径：AscendQuant.md
- 位置行号：17
- 问题上下文：Atlas A3 训练品
- 问题描述：产品型号名称不规范
- 修改建议：修正产品型号为：Atlas A3 训练系列产品/Atlas A3 推理系列产品
- 严重程度：严重

## 问题25

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：规范性
- 文件路径：Cosh.md
- 位置行号：110
- 问题上下文：**表 1**  参数说明
- 问题描述：模版参数与函数入参未分两个表格进行说明
- 修改建议：模版参数与函数入参调整为分两个表格进行说明
- 严重程度：一般

## 问题26

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：规范性
- 文件路径：Acos.md
- 位置行号：118
- 问题上下文：isReuseSource
- 问题描述：模版参数的顺序与函数原型不一致
- 修改建议：调换T与isReuseSource在表格中的顺序
- 严重程度：一般

## 问题27

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：规范性
- 文件路径：Acos.md
- 位置行号：142
- 问题上下文：srcTensor
- 问题描述：函数入参的顺序与函数原型不一致
- 修改建议：调换srcTensor与dstTensor在表格中的顺序
- 严重程度：一般

## 问题28

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：规范性
- 文件路径：Acos.md
- 位置行号：120
- 问题上下文：操作数的数据类型。支持的数据类型为：float、half。
- 问题描述：函数入参的顺序与函数原型不一致
- 修改建议：数据类型应按照标准顺序描述：支持的数据类型为：half、float。
- 严重程度：一般

## 问题29

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：规范性
- 文件路径：Cosh.md
- 位置行号：183
- 问题上下文：完整样例请参考[算子样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/11_matmul_kernellaunch)。
- 问题描述：链接未指向Gitcode仓库
- 修改建议：修改为Gitcode仓库的样例链接
- 严重程度：一般

## 问题30

- 检查引擎：DomainCheck_AscendC_API
- 问题类型：规范性
- 文件路径：Cosh.md
- 位置行号：132
- 问题上下文：是否允许修改源操作数。该参数预留，传入默认值false即可。
- 问题描述：代码中已实现该功能，请补充说明参数功能。
- 修改建议：说明isReuseSource参数功能。
- 严重程度：一般
