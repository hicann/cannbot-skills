# 历史 Ring 样例：可选定位索引

仅在用户指定借鉴或需要对照某个实现细节时读取；新算法从算子契约和接口开始，不以找到蓝本为前提。
2026-09-28 在用户提供的 HCCL Git 仓核对以下对象和文件内容。记录不可变 commit，避免每次遍历分支；
仅证明文件存在及其内容身份，不代表已独立检视、通过完整构建/Checker 或完成性能标定。

| 用途 | 固定 commit | 文件（相对 HCCL 仓） |
|---|---|---|
| AllGather Ring 历史模板 | `326bdf01a030f14cb3d173f4232fce613a8c67e1` | `src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_ring.h` / `.cc` |
| Broadcast Ring 历史模板 | `a296b7264a5eb9d1d8bf9da4de9e6b43dd99ac85` | `src/ops/broadcast/algorithm/template/aicpu/ins_temp_broadcast_ring.cc` / `ins_temp_broadcast_ring_parallel.cc` |
| Scatter Ring 历史模板 | `88a2a65cdd46a6b16091282fdeac81a391dcfd22` | `src/ops/scatter/algorithm/template/aicpu/ins_temp_scatter_ring.cc` |

读取前用 `git -C "$HCCL_REPO" cat-file -e '<commit>^{commit}'` 确认对象存在，
再用 `git -C "$HCCL_REPO" show '<commit>:<path>'` 查看所需函数。对象不存在时注明不可用并继续自定义设计，
不默认 fetch 分支或寻找另一份未经核对的副本。以下为对应文件 SHA-256：

```text
4f2cb206e7f332a487b8e5a2b2fad796c29f6cd27afcbb8c27abe94fd4cd7bbb  AllGather .h
129cbcc06ec56ce0d344ef5707215592139474f17ac271a5b16de579858f41ce  AllGather .cc
04dda6a3de26298409b980e37b5e83c4ecea7509431227911b22bf5c111730fb  Broadcast .cc
b53662f9f395e11de45e40039b0093ac702d1dc817893734878b21b077b311bf  Broadcast parallel .cc
07ee535d5fbc5cea80aa10e89301e87e7b6fa9995a8e862aa7732930f4248373  Scatter .cc
```

## 哪些内容不能固化为通用算法规则

| 日志/样例中的说法 | 复用边界 |
|---|---|
| AllGather Ring 可以直接用于独立 AllGather | 上述历史 .cc 的显式守卫检查 AllReduce Parallel；不能直接照搬到 AllGather，布局、配置来源和注册须按目标绑定重做 |
| Ring 固定两个线程、OFFLOAD scratch 为 0 | 是上述历史模板的实现选择；新调度从并发、同步与最大活跃数据推导，不作为骨架默认值 |
| NHR 和 Ring 都是 N−1 步 | 基准 `978d01b68c477572962b5be67c4ea9929ca9e2d7` 的 `template_utils.cc::GetNHRStepNum` 计算 rankSize−1 的位数；8 ranks 为 3 步，不可混用 Ring step 公式 |
| 所有必须实现的方法都是纯虚 | 纯虚与返回错误的默认方法分开，见[接口契约](../SKILL.md#3-接口契约)；“能实例化”不代表功能完整 |
| 历史成功/失败报告可替代本次验证 | 报告绑定其源码、拓扑、模式与包；历史 Checker 失败不能豁免候选失败，历史成功也不能证明新组合正确 |

迁移前核对算法阶段、rank/peer、尾片和 repeat/stride、scratch、线程/notify、HCOMM 支持及显式启用守卫。
AllGather 的当前接入位置和选择器版本见[父层接入卡](../../hccl-aicpu-best-practice/references/allgather-integration.md)。
