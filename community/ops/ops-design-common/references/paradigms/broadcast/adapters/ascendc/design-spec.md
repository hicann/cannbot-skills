# Broadcast 范式 Ascend C 设计约束

## 输入与输出

输入为算子公式、公开接口、形状与类型范围、目标平台事实，以及当前设计问题。输出为适用场景、模板参数、TilingData、资源与同步约束和验证要点；文档文件名与章节组织由调用方指定。

## 按主题读取

- 模板与数据结构：[overview](overview.md)。
- 输入与轴处理：[tiling-preprocess](tiling-preprocess.md)。
- 入口与分发：[kernel-entrance](kernel-entrance.md)。
- 逻辑与物理存活节点：[dag-buffers](dag-buffers.md)。
- Host 切分：[tiling](tiling.md)。
- Kernel 搬运与计算：[kernel-template](kernel-template.md)。
- 坐标与广播辅助函数：[kernel-helpers](kernel-helpers.md)。

## 一致性检查

- 分支覆盖与规格范围一致；TilingKey 的取值与 Kernel 模板实例对应。
- TilingData 每个字段的 Host 来源与 Kernel 消费含义一致。
- 逻辑存活节点从数学依赖推导；物理 Buffer 数量结合融合、临时空间与生命周期计算，资源预算与切分参数一致。
- 搬入、计算、搬出各阶段的尾块、布局和同步规则有明确依据。
- API 验证状态查阅 [API 白名单](../../../../knowledge/api/regbase_api_whitelist.md)，未知项输出为待验证。
- 精度验证使用独立参考结果，并覆盖实际分支与边界。
