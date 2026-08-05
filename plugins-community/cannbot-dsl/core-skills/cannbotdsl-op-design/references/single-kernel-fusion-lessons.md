# 融合算子单 kernel 化经验

## 适用场景

- 把多 kernel、host loop 或多阶段调度收敛成单 kernel。
- 参考已有融合算子时，需要复用它的职责边界、buffer 生命周期和同步方式。
- 已有精度基线，但实现仍依赖 GM 中间量、host 侧状态更新、golden 回填或临时调试路径。

## 先排除环境问题

- 先运行同机器上已知可用的参考 example，确认 Python、torch_npu、CANN、`PYTHONPATH`、`LD_LIBRARY_PATH` 正确。
- 不要默认使用系统 `python3`；同一机器常有多个 torch_npu/CANN 组合。
- 记录真实命令和结果。环境未确认前，不把 compile crash、unsupported SoC 或 runtime 异常直接归因到算子逻辑。

## 公开契约先冻结

- 写代码前明确公开输入、输出、dtype、layout、shape 约束和错误条件。
- 不要为了 kernel 内实现方便，把公开输入在 host 侧整体转成另一套大 layout 后再测试。
- 如果公开 layout 让目标 tile 在 GM 上非连续，要先做设计选择：
  - 改公开契约为连续 layout；
  - 或增加明确的 wrapper / layout 转换层；
  - 或在 kernel 内用正确 stride/tile 映射处理非连续访问。
- 文件名、测试名、TODO 和实际 layout 必须一致；过渡状态要明确标注。

## 单 kernel 验收口径

- 单 kernel 通过必须同时满足：一次 kernel 下发、目标 NPU case 跑通、NPU 输出和 CPU reference 对齐。
- split launch、多次下发、host loop、host 侧状态递推、把中间量落 GM 后二次 kernel 合成，都只能算精度基线或过渡实现。
- CPU compile 通过、编译通过、小 case 通过、大 case 通过是不同阶段，结论里必须分开说。
- 不要把“某个中间结果用 golden 写回后输出正确”当作 kernel 正确。

## Golden 和输入构造

- Golden 只作为 CPU reference 参与对比，不能回填 workspace，不能替代 kernel 输出。
- 单 stage 单测要直接构造本 stage 所需输入；不要运行上游 stage 来掩盖当前 stage 的独立性。
- 输入范围要模拟算法稳定区间，避免用过大随机值制造非目标溢出、病态矩阵或无意义误差。
- 记录 dtype 策略：公开输入 dtype、关键累加 dtype、workspace dtype、输出 dtype 和 CPU reference dtype。
- 记录 GM add 策略：L0C 计算 dtype、FIXPIPE cast dtype、GM 目标 dtype、atomic/add dtype。BF16 输出应验证 BF16 GM add，不要因为中间 L0C 为 FP32 就把最终输出路径升级成 FP32。

## 写代码前先画生命周期表

> **前提**：CANNBotDSL 默认走 channel-first（Channel 直接当操作数，buf_id/sync_id 由框架自动合成，4 相协议零手写，见 `../SKILL.md` §5.0 与 `cannbotdsl-channel`）。channel-first 下**不需要**人肉画 token/wait/publish/release。下面这张生命周期表只用于排查 channel-first 下 hang / 数值错时理清预期的数据流——把它对照源码里每条 Channel 的 Write/Read 操作数，确认生产/消费关系是否符合预期。

手写四原语时，每个跨角色或跨阶段 buffer 都要写清楚：

| 数据 | producer | consumer | 位置 | token/id | wait | publish | release | 复用周期 |
|------|----------|----------|------|----------|------|---------|---------|----------|
| AIC->AIV tile | AIC FIXPIPE | AIV V/MTE2 | UB/GM | 独立 id | 写前等 free | 写后发 ready | 读后 release | 每 tile |
| AIV->AIC tile | AIV MTE3 | AIC MTE1 | L1 | 独立 id | 写前等 free | 写后发 ready | 读后 release | 每 tile |

要求：

- 每个 sync id 只服务一个 producer/consumer 生命周期，不跨无关数据复用。
- `id+16` 只用于明确的第二 subblock 或同一 handoff 的成对 token。
- 初始化 token 必须覆盖后续所有会 wait 的 buffer id。
- `tile_view` / `local_slice` view 只作为操作数；同步锁 owning buffer。
- 遇到 hang 时先审计 wait/publish/release 是否成对，再改算法。
- channel-first 场景遇 hang / 静默数值错，先确认 Channel 操作数 + 数据依赖是否正确传给框架（累加器被 per-op 化最常见），而不是先去人肉配对——arena 合成的配对本身不会漏。

## 复用参考实现的正确方式

- 先复用参考实现的结构，而不是只复制 API 调用：
  - 顶层 kernel 负责 tile 调度和 stage 顺序；
  - Cube/AIC 类负责 GM/L1/L0/matmul/FIXPIPE；
  - Vector/AIV 类负责 UB、elementwise、mask、规约和写回；
  - BlockInfo/Tiling helper 负责 shape、tile 坐标和边界。
- copy engine 放在实际使用者里。GM->L1、L1->L0、L0C->UB/GM 通常归 Cube；UB 内变换、UB->L1/GM 通常归 Vector。
- shared handoff 用窄对象表达，只暴露 buffer 和 token，不把整个 Cube/Vector 对象当“杂物箱”互相传。
- 如果参考实现有稳定的 preload、ping-pong、triple buffer 或 drain 结构，先保持生命周期一致，再替换内部公式。

## GM scratch 使用原则

- 默认短生命周期中间量走 UB/L1/L0/FIXPIPE，不落 GM。
- GM scratch 只用于：
  - 公开输入/输出；
  - 跨 kernel 或跨 stage 必须保留的数据；
  - 容量不足；
  - 明确的 layout/stride 限制；
  - 已定位的工具链能力边界。
- 临时用 GM 绕过工具链限制时，要在 TODO 和代码注释里写清后续 on-chip 化路径。

## 复杂子算法迁移

- 对分块求逆、递推、online 更新、packed layout 等复杂子算法，先迁移生命周期和数据槽位，再迁移具体公式。
- 注释要说明每个槽位保存什么、哪一步生产、哪一步消费、何时可覆盖。
- 不要把 CPU reference、workspace slot、临时近似值和最终输出混用。

## 验证顺序

推荐按阶段推进：

```bash
python -m py_compile <test_file.py>
pytest --collect-only -q <test_file.py>
pytest <cpu_compile_case> -q -s
pytest <small_npu_case> -q -s
pytest <target_npu_case> -q -s
```

长跑时单独检查进程，必要时及时终止旧 pytest，避免占用设备。

如果目标 case 曾经出现间歇精度失败：

- 清理编译缓存，避免复用旧 codegen。
- 连续重复跑目标 shape，记录每轮 `max_abs/max_rel`。
- 先审 producer/consumer 生命周期、GM 初始化、atomic dtype、sync ready/free token，再改数学公式。
- 未完成重复验证前，不把“一次 pass”写成稳定通过。

## 结论写法

- 明确当前阶段：环境通过、CPU compile 通过、编译通过、split 精度通过、single-kernel 小 case 通过、single-kernel 大 case 通过。
- 精度结论必须说明比较对象是 NPU 输出和 CPU reference。
- 未跑的验证要说明原因。
- 不在单 kernel 目标 case 真实通过前说“任务完成”。
