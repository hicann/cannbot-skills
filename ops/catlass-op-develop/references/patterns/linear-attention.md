# Linear Attention — GDN/KDA/Retention 类实现注意事项

> **导航**：设计路由见 design skill [kernels/attention/linear-attention.md](../../../catlass-op-design/references/kernels/attention/linear-attention.md)。GDN/KDA 专项设计见 [gdn-kda.md](../../../catlass-op-design/references/kernels/attention/gdn-kda.md)。开源参考路径和 User Contract Priority 规则见 [open-source-linear-attention-map.md](../../../catlass-op-design/references/kernels/open-source-linear-attention-map.md)。本文聚焦实现期规则：跨 stage workspace、同步、tile 复用、shape/精度落地和性能归档。
>
> 命中 GDN/KDA/retention/RWKV 时，先按开源参考地图读取 Triton/NPU 同族实现和已生成 Catlass 经验，再实现当前算子；不要依赖用户每次在 prompt 中手动给路径。用户没有显式给本地实现参考路径时，`OPEN_SOURCE_ALIGNMENT.md` 的 `reference_source` 必须是 `OPEN_SOURCE`。clone 可用时实现骨架来自 clone 的开源参考；clone 不可用时基于仓内开源规范摘要、远程搜索路径和 curated reference 继续。只有用户显式给本地实现参考路径时才允许 `USER_LOCAL`。

---

## 1. 先实现可验证 stage，再合并 full-flow

线性 Attention 融合算子通常包含 matmul、gate/beta、scan/state update、layout transpose 和输出写回。实现顺序建议：

1. 冻结 full-flow baseline 和输入布局。
2. 为每个 stage 明确输入、输出、workspace、flag。
3. 先让单 stage 在独立 golden 下通过。
4. 再串联 stage，对 full-flow baseline 做端到端比对。
5. 最后做 launch 合并、workspace 复用和流水调优。

不要先写一个大 kernel 再从最终输出猜错因。跨 stage 中间量必须能按开关 dump，并与 golden 中间量比对。

### 1b. 用户数学 contract 冻结

实现前先把用户 prompt 或 DESIGN.md 中的公式、scale 作用位置、mask、gate、state、layout 和输出 dtype 写成唯一 contract。open-source、仓内旧算子和历史调试代码只作为参考；如果它们与本次 contract 不一致，以本次 contract 为准，并把差异写进 README/报告。golden.py、verify_result.py、性能 baseline case 命名必须使用同一 contract。

### 1c. Reference source 执行

实现前读取 `docs/OPEN_SOURCE_ALIGNMENT.md`：

- `reference_source=OPEN_SOURCE` 且 `clone_status=CLONED`：读取文档列出的开源 clone 路径和 commit/tag，以该 primary reference 的 pipeline/接口/baseline 口径为骨架。
- `reference_source=OPEN_SOURCE` 且 `clone_status=UNAVAILABLE`：读取文档列出的仓内开源规范摘要章节、远程搜索路径和 curated reference，以这些可追溯材料形成实现骨架继续开发。
- `reference_source=USER_LOCAL`：仅当用户 prompt 明确给出本地实现参考路径时成立，以该路径为骨架，并记录本地版本状态。
- `evaluation_baseline`：用户 prompt 中给出的 baseline / 评测 / 性能对比路径只能用于评测脚本、shape、报告字段和 baseline_status，禁止作为实现骨架或 source-of-truth。
- 仓内旧算子、历史实现和已生成 Catlass 经验只可作为 curated engineering reference；不得在 `OPEN_SOURCE` 模式下替代 primary reference。
- 若 `reference_source` 缺失、与用户输入不符，或文档把未指定的本地实现/evaluation_baseline 写成 primary reference，停止开发并返回 design_issue。

---

## 2. GM workspace 与 CrossCoreFlag 约定

Cube / Vector 跨 stage 中间量默认经 GM workspace 中转。每条跨 stage 边必须满足：

| 规则 | 说明 |
|------|------|
| 稳定 slot | 同一个 `(chunk, hv)` 或设计指定逻辑块在所有 stage 使用同一 workspace slot 公式 |
| stage 末尾 set | producer 完整写完该 slot 后再 set flag |
| stage 入口 wait | consumer 在进入内层 tile 循环前 wait，避免循环中频繁握手 |
| slot 不混用 | GM slot 不能和 UB ping/pong 变量共用概念；二者生命周期不同 |
| 清零明确 | partial reduce、max、state 初值等依赖中性元的区域由 host 或 kernel 明确清零 |

A2/A3 上不要依赖未证实的 L0C -> UB 直通。若要使用非 GM 中转，必须在设计和实现中引用当前 catlass header/example 或芯片文档依据。

---

## 3. 2-head/window 分组调度

当算法可以按 2 个 head 或 2 个 window 分组流水时，推荐按 stage-by-stage 调度：

```
for window in windows:
  run stage A for heads in window
  run stage B for same heads
  run stage C for same heads
```

不要对每个 head 先跑完整 full-flow 再切到下一个 head；这会破坏 workspace 复用和跨 stage cache 命中。常见 slot 公式：`windowStartSlot = (windowIdx & 1) * 2`，窗口内两个 head 分别占两个稳定 slot。

---

## 4. L1/L0/UB 复用策略

详细 A2/A3 stage/window、workspace slot、flag、resident、double buffer 和 split accumulation 规则见 [a2-a3-linear-attention-stage-design.md](a2-a3-linear-attention-stage-design.md)。本节只保留实现期摘要；实现多 stage 算子时必须按该文档 checklist 逐项核对。

| 资源 | 建议 |
|------|------|
| L1 resident | Q/K/V/WY 等跨多个 stage 复用的块状输入优先常驻；resident 区和 scratch 区分开 |
| L0A/L0B | matmul 内使用 double buffer，LoadData/Matmul 事件成对管理 |
| L0C | 只作为 Cube 累加区；需要给 Vector 消费时按 GM workspace 路径设计 |
| UB matrix input | 矩阵输入搬入和 Vector 后处理使用 double buffer |
| UB beta/g input | beta、g、exp(g) 可在 Vector stage 内短生命周期常驻，避免重复 GM 读 |
| UB output | 输出写回使用独立 ping/pong，避免覆盖仍被 MTE3 使用的数据 |

scratch 的生命周期按 stage 收敛，不能因为某块数据在下个 stage 可能复用就无界扩大 scratch 常驻范围；跨 stage 复用应优先表达为 resident 区或 GM workspace。

### 4b. 非 GEMM 逻辑的 Catlass-style 自定义组件

当公开 Catlass epilogue 不能表达线性 Attention 的 gate、decay、causal mask、validRows、clamp、cast、双输入 finalize、layout 整理时，不要把工程停在设计阶段，也不要把真实计算移到 host。实现期应新增可审查的自定义 Block/Tile：

| 组件类别 | 实现要求 |
|---|---|
| gate/decay/mask/finalize Tile | 使用 Ascend C Vector API、固定 tile 内循环、mask/writeback，写清 dtype、shape、UB 预算 |
| workspace/layout stage | 明确 GM layout、slot、flag、清零和 validRows 写回 |
| scan/state recurrence stage | 标为 dependency-based non-L0 exemption，给出同步边界 |
| GEMM stage | 仍必须用 Catlass `Kernel`/`BlockMmad`，禁止手写矩阵乘 |

合规边界：

- 自定义组件内部允许固定 tile 内逐元素循环；op_kernel 顶层不要散写大段标量计算。
- 自定义组件必须 device 执行，host 只做 tiling、数据准备、runner 和验证。
- 空 device kernel + host 真实计算不是可交付 MVP。
- 如果性能还未达到目标，可标记 `Catlass-stage MVP`，但必须完成编译、精度、性能脚本、baseline_status 和已知限制。

---

## 5. K/V/HK/HV 特化经验

### `V=256` 的 K 维 split accumulation

当 V 维较大导致一次 tile 超出 L0/UB 预算时，按 K 维 split accumulation：

- 每个 K tile 做局部累加。
- 只有最后一个 K tile 执行 fixpipe / cast / 输出写回。
- 中间 tile 不写最终输出，避免重复 GM 写和错误覆盖。

### GQA: KKT 按 HK 缓存

`HV > HK` 且多个 value head 共享同一个 key head 时，KKT 或同类 K-side 中间量按 HK 缓存和复用，不要按 HV 重算。

```
hk = hv / (HV / HK)
workspace_kkt_slot = f(batch, chunk, hk)
```

设计和实现都要写清 `HV/HK` 的整除约束和 slot 映射。

---

## 5b. KDA dAv backward 实现经验

适用于 `chunk_kda_bwd_dAv` / `catlass_chunk_kda_bwd_dav` / KDA backward 中的 `dA,dV` stage。

### 接口对标

- 先读取 FLA Triton orchestration 与 backend，逐项对齐输入输出 tensor、layout、`scale`、`BT`、`cu_seqlens`/`chunk_indices`、GVA `HV/HK` 映射和 case 命名。
- 再读取 NPU AscendC KDA 已有实现，复用 host tiling、runner、ACL 初始化、case 数据布局等工程模式。
- Python/Triton baseline 有些 shape 不能跑是允许状态；测试脚本要能记录 `baseline_status`，custom 结果不能依赖 baseline 每次在线重跑。

### 计算顺序

KDA dAv stage 的常见依赖顺序：

```text
RunDa: dA_raw = dO @ V^T
Vector/AIV: causal/gate/scale/mask 后处理 dA，并按需要生成/整理 A
RunDv: dV = A^T @ dO
```

如果 `RunDv` 消费 AIV 处理后的 `A/dA` workspace，则保持 `RunDa -> signal -> wait -> RunDv`。未设计完整 flag/credit/resource 证明前，不要为了 overlap 把 `RunDv` 提前；KDA dAv 经验中这种重排容易造成长跑或同步风险。

### Varlen/partial chunk

- Tiling 用 `chunk_indices`/`cu_seqlens` 推导 `nt` 和每个任务的 base offset，`taskNum` 覆盖真实 `(chunk, hv)`，不要只用 `ceil(T/BT)`。
- 物理 matmul 尺寸和有效行数分开：`BT` 是 Catlass matmul 主路径物理块，`validRows` 用于 causal/mask/writeback。
- 对当前 Catlass TLA 路径，已验证的保守写法是 `chunkLen = validRows == 0 ? 0 : BT`；把 `chunkLen` 改成 `validRows` 曾让长 varlen case 分钟级长跑。只有在独立分支、全量精度和性能报告都通过后，才能保留 actual shape 收敛优化。
- 尾块优化失败时，先回滚到 full-BT 物理块 + validRows 掩码，再排查其他性能问题。

### Tile 与分支调优

- `V=256` 先确保 split accumulation 只在最后 subtile 后 fixpipe/writeback，再比较 TileShape。
- GVA/GQA 先检查 `HV/HK` 整除、`hk = hv / (HV/HK)` 映射和 K-side cache scope。
- 任何 TileShape、window、flag 修改都必须单变量实验，并保留 before/after report。已知 KDA dAv 中 `Dv256 64x256x64` 一类尝试可能编译通过但运行长跑，不能只凭编译成功合入。

### 构建与 runner

- kernel 修改后必须确认 OPP 重建和 runner 重新链接；只看源码 mtime 不足以说明当前 runner 已使用新 kernel。
- CANN 版本切换后重新 source `set_env.sh`，确认 `ASCEND_CUSTOM_OPP_PATH`、`OP_HOST_LIB_DIR`、`LD_LIBRARY_PATH` 完整，否则可能出现 `libruntime.so` 或旧 OPP 被加载。

---

## 6. 物理 transpose 优先于 Vector scatter

如果后续 stage 需要转置布局，优先设计一次物理 transpose 或 layout-converted workspace，使后续读写连续。不要在高频 Vector stage 中用 scatter/gather 访问模拟转置；这通常会把瓶颈转到 MTE/Vector 地址计算。

---

## 7. CrossCore raw flag backpressure

使用跨核 raw flag 时，producer/consumer 的窗口深度必须受控。2-slot 或 4-slot 轮转可以用固定 flag 协议；如果窗口扩大，必须新增 credit/free 协议，防止 producer 覆盖 consumer 尚未读取的 slot。

固定窗口规则：

| 项 | 要求 |
|----|------|
| 2-head window | 2 个稳定数据 slot + 对应 ready flag |
| 4-slot workspace | `windowStartSlot=(windowIdx&1)*2`，两窗口 ping/pong |
| stage 边界 | producer set ready，consumer wait ready；复用前确认上一 consumer 已完成 |

---

## 8. TilingKey 与 shape 参数化

线性 Attention 实现中，TilingKey 至少覆盖改变模板实例化或调度模式的条件：

| 条件 | 常见取值 |
|------|----------|
| dtype | fp16 / bf16 / fp32 accumulator |
| V_DIM | 64 / 128 / 256 |
| CHUNK_SIZE | 32 / 64 / 128 等设计值 |
| schedule mode | normal / small-BT / split-K / split-V |
| head mapping | `HK == HV` / GQA |

脚本中的 case 应由这些条件和边界类别生成，不要只把历史调试 shape 硬编码成无法解释的一串 tuple。若为了复现问题保留固定 shape，必须在注释或报告里标明它对应的覆盖类别。

---

## 9. 精度脚本落地

- 标准优先复用 `ops-precision-standard`，浮点输出用 mixed tolerance：`atol + rtol * abs(golden)`、`matched_ratio`、`max_abs`。
- golden 必须独立于当前 kernel。stage operator 可用 stage-aware golden，但上游中间量必须已由 independent baseline 验证。
- gen_data 必须能按 CLI 参数生成任意覆盖矩阵 case，不能只服务单个固定 shape。
- 报告至少输出 case id、shape、dtype、rtol/atol、matched_ratio、max_abs、结论。

线性 Attention 精度报告还需满足：

- 阈值来源必须对齐 `ops-precision-standard`，浮点输出使用 mixed tolerance，不自创线性 Attention 专属阈值。
- 近零输出通过 `atol` 兜底，报告必须记录 matched_ratio 和 max_abs，避免只看相对误差。
- `--reuse-existing` 代表复用已有 custom 输出重新汇总；若修改 kernel，必须删除对应 case 输出或关闭复用，避免用旧结果验证新代码。

建议覆盖类别：TilingKey、BT/chunk 边界、V/K、HK/HV/GQA、多 batch/head、zero gate、high beta、近零输出和状态初值边界。

---

## 10. 性能归档

性能验收不只看单 kernel duration。必须归档：

| 字段 | 说明 |
|------|------|
| baseline | 算子名、版本/commit、shape、layout、dtype |
| Task Duration | custom 与 baseline 同 shape 对比 |
| launch count | full-flow baseline 和 custom 都要列出 |
| speedup | 用同一统计口径计算 |
| dominant pipeline | Cube/MTE/Vector/同步等待中的主瓶颈 |
| workspace peak | host tiling 计算值和 profiler 观察值 |
| profiler path | `docs/perf/round_NNN/` 下原始数据路径 |

如果性能差距只出现在 `B*chunk` 小于核数、小 V、`V=256` 或 GQA 场景，先归因到调度/tiling 分支，不要直接调单个 TileShape。

性能脚本实践：

- 支持复用已有 inputs 和 baseline。需要“只测当前算子”时，case-root 可指向已生成数据目录，custom 重新跑，baseline 从历史报告读取。
- `--reuse-baseline` 只读取当前 report dir 中已有报告；若要复用旧 baseline，先把旧 report 复制进新 report dir。
- `/usr/bin/time python3 scripts/benchmark_perf.py ...` 同时出现 `time`、Python driver、C++ runner 三个进程是正常现象。
- varlen case 长时间无输出时，先检查是否误把 `chunkLen=BT` 改成 `chunkLen=validRows`、runner 是否重链、设备 runtime 是否健康，再考虑 TileShape。

---

## 强制检查表

| # | 检查项 |
|---|--------|
| LA1 | 已区分 full-flow / stage operator，并冻结同语义 baseline |
| LA2 | 已画 dependency graph，stage 切分来自依赖而非公式顺序 |
| LA3 | 跨 stage 中间量有 GM workspace layout、稳定 slot 和 flag 协议 |
| LA4 | A2/A3 未声称未证实的 L0C -> UB 直通 |
| LA5 | L1 resident 与 scratch 生命周期分离 |
| LA6 | L0A/L0B、UB input/output 的 double buffer 事件配对完整 |
| LA7 | `V=256` 等大 V 场景的 split accumulation 只在末 tile fixpipe/writeback |
| LA8 | GQA 场景 KKT 等 K-side 中间量按 HK 缓存，不按 HV 重算 |
| LA9 | shape 覆盖矩阵包含 BT/chunk、V/K、HK/HV、多 batch/head 和数值边界 |
| LA10 | 精度报告使用 mixed tolerance，并记录 matched_ratio/max_abs |
| LA11 | 性能报告包含 baseline、launch count、workspace peak 和 profiler path |
| LA12 | 多 stage / A2/A3 场景已读取并执行 `a2-a3-linear-attention-stage-design.md` checklist |
| LA13 | GDN/KDA 需求已读取 open-source-linear-attention-map.md，并在报告中记录参考路径 |
| LA14 | KDA dAv varlen/partial 已区分物理 `BT` 与 `validRows` 掩码/writeback |
| LA15 | mixed tolerance 阈值来源对齐 `ops-precision-standard`，近零输出由 `atol` 兜底并记录 matched_ratio/max_abs |
| LA16 | 用户 prompt/DESIGN 的数学 contract 已冻结到 golden、verify、README 和报告，未静默继承参考实现差异 |
| LA17 | 非 GEMM 逻辑已封装为 Catlass-style 自定义 Block/Tile 或明确 stage 化，device 主路径不为空且 host 不做真实计算 |
| LA18 | `reference_source` 合法：未显式给本地实现参考路径时使用 `OPEN_SOURCE`，显式本地实现参考路径时才使用 `USER_LOCAL`；evaluation_baseline 不作为实现参考；OPEN_SOURCE clone 失败时有 `clone_status=UNAVAILABLE` 和降级依据 |
