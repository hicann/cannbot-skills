---
name: hccl-aicpu-debug
description: 调试 HCCL AICPU 算法的构建、注册选路、Checker 数据错误或线程同步故障。触发：出现 AICPU 构建失败、目标算法未命中、Checker 失败或超时时使用。依据目标源码与原始日志诊断；源码修复和状态裁决由调用角色负责。
---

# AICPU 算法故障定位

适用于 AICPU template、executor、selector 和 TopoMatch 的故障。先取得已确认的 HCCL 仓路径、具名分支与源码身份、目标算法、实际运行配置、构建或 Checker 原始日志；缺少证据时标记 `UNVERIFIED`，不从一条报错推定根因。

## 按故障发生位置检查

| 现象 | 优先核对 | 技术真源 |
|---|---|---|
| host/device 构建或加载失败 | 完整命令、首次失败行、真实退出码、两侧 CMake 接线、`AICPU_COMPILE` 守卫、实际装载包身份 | [构建故障与接入审查](../hccl-aicpu-best-practice/references/build-verification.md) |
| 指定算法未命中 | 通信域配置优先级、目标版本分发入口、DSL 与注册全名、成本候选、拓扑属性和当前安装包 | [显式选路与未命中定位](../hccl-aicpu-best-practice/references/executor-selector.md) |
| Checker 数据或语义失败 | 用例完整配置和 Checker 阶段；Spec 与源码的源/目标方向、rank 映射、offset、length、repeat、stride、scratch、特殊 dtype | [数据流规则](../hccl-aicpu-best-practice/references/07-rules.md) 和 [五参数说明](../hccl-aicpu-best-practice/references/09-template-data-params.md) |
| 卡住、超时或资源错误 | 首个失败用例的线程数、Notify 申请与索引、Pre/Post 同步配对、channel 实际 peer、日志中最后完成的动作 | [线程同步模式](../hccl-aicpu-best-practice/references/08-thread-sync-patterns.md) |

先核对 [HCCL-VM 原始证据与比较](../hccl-test-tool-hvm/references/evidence.md)：`--dry-run`、静态检查、单条成功文本或未命中目标算法的 Checker PASS 均不能证明目标故障已修复。基线与候选的包、源码、环境或矩阵不同，先记录差异，再决定是否可比较。

每次诊断只提出能区分当前候选原因的最小检查，保留原始失败。输出包含：症状与阶段、已核验事实及路径、仍有竞争的原因、建议的单变量检查、`SOURCE` / `ENVIRONMENT` 分类和未验证范围。证据不足以分类时明确写“未判定”，不能伪装成策略校验器接受的失败类别；由调用方按其契约处理阻塞。分类是技术意见；是否修源码、重测或推进状态由调用方的 Agent/Plugin 决定。
