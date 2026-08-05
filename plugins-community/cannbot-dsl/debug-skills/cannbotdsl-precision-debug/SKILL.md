---
name: cannbotdsl-precision-debug
description: "调试 CANNBotDSL runtime/NPU 精度、随机输出、hang、DataCopy/mem_copy、local_slice offset、AIC/AIV sync、cube matmul 或端到端示例失败时使用。适用于把问题从期望输出、DSL frontend、CANNIR lowering、AscendC translate 到 NPU runtime 分层定位。Triggers: CANNBotDSL precision, NPU 精度失败, local_slice offset, sync id, mem_copy, DataCopy, cube matmul, mixed AIV/AIC, hang, random output."
---

# CANNBotDSL 精度调试技能

用于 CANNBotDSL 算子 runtime/NPU 精度调试。目标是先分层定位，再做最小修复，避免在大 kernel 中盲调。

## 快速路由

| 症状 | 先读 |
|------|------|
| 混合 AIV/AIC 或 CANNBotDSL 工具链问题 | `references/mixed-kernel-debug-lessons.md` |
| `local_slice`、UB→UB copy、sync、32×32 matmul 风险 | `references/runtime-precision-checklist.md` |
| `run()` 已返回但 `torch.npu.synchronize()` hang | `references/runtime-precision-checklist.md` 的“长跑与 hang”和“混合 AIC/AIV 同步审计” |
| 单 kernel 精度基线、layout 连续性风险或 split launch 验收口径 | `../../core-skills/cannbotdsl-op-design/references/single-kernel-fusion-lessons.md` |
| 编译+launch 均成功、无 device fault，却部分 row/全 NaN | `@kernel` launch 按位置绑定，先核对 `run()` 传参顺序 vs signature 槽位（`../../core-skills/cannbotdsl-api-reference/SKILL.md` §5 #18） |

## 调试原则

1. **先固定最小复现**：shape、dtype、seed、期望输出和后端阶段。
2. **先做问题归因**：只有能定位到前端、IR、lowering、translator、runtime/debug tooling 的问题，才归类为 CANNBotDSL 缺陷或能力缺口。
3. **再分层定位**：Python DSL → CANNIR/lowering → AscendC translate → NPU runtime。
4. **一次只改一个变量**：不要同时改 dtype、sync、layout 和 lowering。
5. **先区分编译慢和 kernel hang**：阶段打印 `alloc/run start/run returned/sync returned`，`run()` 返回后卡在 `synchronize()` 才归入 kernel 执行或同步问题。
6. **hang 先审计同步关系**：结构性改代码前，先看源码里的 `setwait`、`notify`、sync id、pipe 和 producer/consumer 是否配对。
7. **精度通过后再更新 TODO**：记录真实命令和误差，不用“应该通过”。
8. **连续两次修复不改善 = 假设错了，停止打补丁，改做对照实验**。每次修复前先写下"如果这个假设成立，误差应该降到 X 量级"；实测没降就是假设被证伪，此时再改第三处只是在赌。**正确动作是做一个能二分的对照**：把可疑段与一个**已知正确的等价实现**放进同一个 kernel、喂同一份输入、各自把结果写回 GM，直接对比。真机实证：本仓 GQA 通用化改造中，修 A（存储源）让 MERE 从"全 0"降到 53、修 B（标量 lane0 掩码存）降到 10.4 —— 两次都有效；随后修 C（lane0 广播读）、修 D（补 `vmem_bar`）各自都有独立证据支持其必要性，但 MERE 纹丝不动（10.4→10.6→10.4）。此时继续找"第五个 bug"是错的。
   > **副产品价值**：C/D 虽未改善当前指标，但其正确性由独立 probe 证实（C：running max 真值 0.26 被邻行读成 4.89），属于"修对了但不是主因"。**不要因为指标没动就回滚它们** —— 也不要因为改了很多处就认为一定在接近真相。
9. **区分"误差量级"而非只看是否达标**：`全 0` / `O(10)` / `O(1)` / `1 ULP` 是四类不同的病。全 0 = 没写入（查存储绑定与 producer）；O(10) 且**两个 split-M 半区误差相同** = 两核在一致地算错（逻辑 bug，与分工无关）；半区误差**差异悬殊** = per-core 绑定/索引问题；1 ULP 量级 = 已达量化下限（见 `../../core-skills/cannbotdsl-op-test/SKILL.md` §L1.1，别再"优化"）。**按半区拆开看误差**是分辨第二、三类的最便宜手段，值得在测试脚本里常备。

## 必查项

- CPU/translate 是否能生成目标路径。
- `local_slice` source/destination offset 是否进入 lowering。
- UB、L1、L0 buffer 物理地址和 `buf_id` 是否存在非法 alias。
- sync id 是否单 producer/consumer，pipe 是否匹配。
- cube matmul 的 shape、L0B transpose、ND/NZ layout 是否有独立最小测试。
- AIV/AIC 混合 kernel 是否存在长时间无输出或等待错配。
- AIC 分支开头是否等待 AIV 后面才会生产的 L1/UB token。
- C2V/V2C handoff 的初始 token、ready/free token、`id+16` 语义是否明确。
- 源码中的 `setwait`、`notify`、sync id 和 pipe 是否与 handoff 生命周期一致。
- 执行边界处的 allocator rewind 是否只是回收游标，复用地址前是否确认没有跨 pipe 消费者仍在读。
- 手动 UB/L1/L0 地址别名和 L0 double buffer 是否按真实字节容量检查过。
- `transpose=True` 是否用于 CANNBotDSL 已支持的 copy 方向。
- `@jit/@kernel` 动态控制流是否位于 AST 预处理区域内。

## 推荐命令模板

```bash
python -m py_compile <test_file.py>
pytest --collect-only -q <test_file.py>
pytest <cpu_compile_selector> -q -s
timeout 720 pytest <npu_selector> -q -s
```

NPU 长跑时另开检查：

```bash
ps -ef | rg "pytest .*<test_file>"
```

## 输出要求

交付调试结论时必须给出：

- 失败层级：expected output / DSL frontend / lowering / translate / runtime / environment。
- 最小复现路径和命令。
- 修改文件。
- 通过或失败的真实误差。
- 未解决风险和下一步。
