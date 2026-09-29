# Operator 分区入库标准

1. `type` 固定为 `operator`，必须声明 `operator_families` 和 `architectures`。
2. 除公共六章外，正文必须包含“算子算法、分核策略与基本块切分、数据路径与存储层级、
   流水排布、同步关系与数值精度”四个二级主题。
3. 算法公式、shape、layout、state、有效区和边界语义必须来自固定来源。
4. 实现候选需写明任务映射、GM/UB/L1/L0/workspace 生命周期、Stage/slot/flag/event 和验证矩阵。
5. concept 是设计输入，不得声称当前工程已经通过精度、性能或 sanitizer 验收。
6. 新 family 必须先注册 canonical 名称和无冲突别名，再建立 family 索引。
