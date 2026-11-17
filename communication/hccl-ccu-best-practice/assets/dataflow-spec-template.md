# CCU Dataflow Spec

每个 template 一份。仅记录算法技术证明；需求与支持范围引用 TASK_SPEC，文件改动引用 PLAN，测试矩阵引用 TEST_PLAN，通用接入依据引用 DESIGN。独立使用时引用调用方已有记录，不要求这些固定文件名。

## 1. 身份与技术输入

| spec-id / template | 源码身份与引用的设计版本 | op / CCU模式 / 适用rank与拓扑 |
|---|---|---|
| | | |

## 2. 数据所有权与地址

| 阶段 | 本rank/peer局部索引 | src归属、基址和字节偏移 | dst归属、基址和字节偏移 | slice/tail/repeat | 结束时的数据不变式 |
|---|---|---|---|---|---|
| | | | | | |

说明元素到字节换算、executor已加偏移、整次stride与本轮slice的区别、访问范围；归约算法另证明贡献次数与初始化。

## 3. 资源、同步与活性

| kernel / die / resGroup | channel顺序与peer映射 | scratch / buffer / thread | 申请与实际消费 |
|---|---|---|---|
| | | | |

| 阶段 | producer / consumer | Event或Notify、接收表 | 条件、完成与复用顺序 | 等待依赖与无环证明 |
|---|---|---|---|---|
| | | | | |

XN/slot/mask的完整值引用本次 `sync-contract.json` 的pool/epoch/signal，正文只解释生产消费和复用证明；不能用JSON一致代替活性证明。

## 4. 参数、算术与重放

引用本次 `launch-contract.json` 的kernel/字段和 `sync-contract.json.compute_ops` 的表达式/rep；不重复完整槽位表。解释缓存key/失效、地址与token派生更新、立即数/溢出边界及声明对应的源码位置。

## 5. 边界推演与未确认项

| 条件 | 实际路径与协议影响 | 静态证明或关联case-id | 未验证范围 |
|---|---|---|---|
| 单rank / 零片 / N=2去重 / 最大rank / 尾片 / 多loop / 重放（按实际适用） | | | |

关键语义尚未确定时列出问题与所需证据；不能仅因模板填完或检查器成功就声明证明完成。证明、反例和独立审查所需细节不因精简而省略。
