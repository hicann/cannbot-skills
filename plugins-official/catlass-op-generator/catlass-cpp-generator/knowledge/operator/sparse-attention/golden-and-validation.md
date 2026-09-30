---
type: operator
title: block_sparse_attention Arch22 Golden 与验证
description: 冻结实际输入、参考实现和比较器，覆盖布局、压缩索引、数值模式与 O/LSE。
tags:
- catlass-cpp
- sparse-attention
- block_sparse_attention
- BlockSparseAttention
- 块稀疏注意力
- arch22
operator_families:
- sparse-attention
architectures:
- DAV_2201
status: stable
generated: {"by": "process:sparse-attention-knowledge-port"}
verified: [{"by": "process:cross-source-check"}]
sources: [{"id": "sparse-attention-golden-and-validation-source", "resource": "audit:custody/sparse-attention#golden-and-validation", "title": "golden-and-validation 固定来源（提取侧独立保管）", "kind": "audit-record"}]
---

# 接口与概念

## 算子算法

Golden 不要求来自 ATK，也不要求同时提供两个参考实现。先根据用户提供的参考和任务约定，
确认验收针对数学结果，还是包含指定计算顺序、类型转换和舍入行为的数值模式。
模式相关的在线参考与 FP32 全量数学参考可能具有不同的归约粒度和舍入路径；只有测试规范
要求双标杆时，才分别维护两者。不能为套用既有数值模式而擅自改变用户参考。

比较器及阈值由测试规范确定。相对双标杆的误差比与单标杆的 atol/rtol 属于不同判据，
不能直接互换；只有一个 golden 时，不能凭空构造第二标杆或沿用双标杆的误差比阈值。

## 分核策略与基本块切分

覆盖[十二分派](semantics-and-dispatch.md#十二分派覆盖)、B>1、MHA/GQA/MQA、D64/128、
X 小/非对齐/大于128、Y 多个128、Q/KV tail、不同稀疏密度和非连续块。
冻结后不因实现失败而删除合法样本。

## 数据路径与存储层级

冻结最终实际 mask/Q/K/V/长度、参考输出及哈希，而不只冻结造数配置或随机种子。
检查参考初始化是否会重新生成 mask；两个实现必须使用同一份最终输入。
记录实际入口、适配层和库版本，BNSD 有效区与 padding 的输出要求分别明确。

## 流水排布、同步关系与数值精度

在线分组、P cast、scale 舍入和低精 LSE 均可影响误差。
保持参考数学和比较器不变，分别验证 O 与开启的 LSE；不能修改参考来迎合候选，
判据变更应单独说明，避免将不同标准下的结果混合比较。

# 用法

## 接入用户提供的 Golden

| 提供形式 | 接入与验证要求 |
| --- | --- |
| PyTorch、NumPy 或其他可运行参考 | 确认调用入口、依赖和执行设备，适配参数及返回结构；用相同输入核对适配前后的结果 |
| 固定输入与预期输出文件 | 配对加载实际输入与输出，保留 shape、dtype 和布局；这些文件只能证明所覆盖样本，不能替代新增输入的参考结果 |
| 仅公式或伪代码 | 先明确边界语义并建立独立参考，取得实际运行证据后才能用于验收 |

接入时核对 mask 的含义、长度表示、GQA 映射、scale、布局、有效区、输出 dtype 与 O/LSE
返回约定。参考只有 O 时，可以验证仅要求 O 的任务；任务要求 LSE 时必须补齐其参考，
不能静默跳过。本知识中的十二分派和低精模式只适用于匹配该接口的任务，不能强加给其他参考。

上述形式需要按当前执行工具适配，并不表示任意 golden 文件都能直接运行。若工作流规定
使用统一的 PyTorch CPU 参考，先核对用户参考与该参考的语义及实际结果；存在差异时明确
裁决后再验收。数值参考与性能基线分别指定，不能直接用 CPU golden 的耗时作为 NPU 加速比基线。

## 测试标识与配置绑定

使用稳定的用例标识绑定实际输入、参考输出与比较配置；使用 fixture 的框架还需保留其映射。
重复、重排或排队执行后，仍按该绑定获取参考与阈值，不能用队列位置或日志编号当配置下标。
若在数值比较前出现索引异常，应报告测试框架错误和未执行数量，不记作数值通过或失败。
修复接线后重新执行相应套件。

## 物理列与压缩索引边界

物理 mask 列数和实际 retained count 分别覆盖。窗口容量 W 对应至少以下诊断：

| 输入构造 | 目的 |
| --- | --- |
| 实际保留 W−1、W、W+1 块 | 窗口前、满窗口、进入下一窗口 |
| 物理列多于 W、实际只保留少量块 | 区分扫描边界与压缩列表长度 |
| W+1 个非连续保留块，含物理首末块 | 验证压缩位置不等于物理块编号 |
| KV=Y+1，末块分别选中/不选中 | 验证真实 token tail 与归约范围 |

从实际 mask 计算 count、索引连续性与窗口数，不用固定窗口常数或样本名代替覆盖证明。

## 参数精度核对

混合 dtype、标量或零维数组的运算，可能随参考库版本、类型提升模式、参数 shape 和值域
改变中间 dtype。对真实参考记录操作数、乘前/乘后 dtype、cast 位置和值；
参数声明 FP32 不足以推导乘法结果必为 FP32。

将修复作为独立数值变更验证，覆盖正负值域边界、端点、零输入控制、非零输入和多行 tail。
不要把某版本的经验阈值硬编码为算子定义；升级参考环境后重新确认。
`torch.equal` 表示逐元素数值相等，不等同于 IEEE 字节或位模式一致。

# 代码模式

```text
case identity -> actual inputs / reference outputs / comparison_config
actual inputs + reference outputs + comparator version -> frozen validation inputs
candidate O / LSE -> matching reference and criterion -> per-output verdict
```

排查时沿“稀疏索引→QK→m/l/P→PV→O/LSE”逐段定位。
诊断 dump 不混入正式性能样本。

# 约束

| 语义缺口 | 冻结前必须明确 |
| --- | --- |
| 空 sparse row / 全零稀疏行 | 参考是否定义输出；不能默认零或负无穷 |
| BNSD padding | 只比较有效区，还是要求完整 padding 清零；必要清零纳入计时 |
| 非正长度、非二进制 mask、无效块置1、非有限输入 | 是否属于合法输入及对应参考行为 |
| dense attenMask / paged 参数 | 不从保留参数推导本输入域支持 |
| 在线粒度不同 | 按约定误差判据判断，不预设逐位一致 |

参考实现、误差定义与阈值应在比较前固定。适配输入输出时保持参考数学不变，
测试工具需要能表达所选判据；尚未支持的比较方式应明确记录。

# 失败表现

mask 被参考重生成、展示编号错配阈值、只检查 O、忽略 padding 约定，都会使结果不可比较。
导入或运行环境失败应与数值失败分开；比较器未执行不能报告精度通过。

# 验证方法

报告分别列构建、运行、O、LSE、比较器及失败样本，绑定实际输入和加载库。
通过正确性后，在相同设备、输入与输出要求下做配对性能测量。
定义 `speedup=T_baseline/T_candidate`、逐例或汇总规则与目标；其中 baseline 是约定的性能
基线实现，不必与数值 golden 相同。不使用隐含的固定达标线。
完整调用范围见[执行与测量](execution-and-measurement.md)。

[^sparse-attention-golden-and-validation-source]: 审计编号 sparse-attention-golden-and-validation-source；固定来源与版本记录由提取侧独立保管，不随生成侧资料分发。
