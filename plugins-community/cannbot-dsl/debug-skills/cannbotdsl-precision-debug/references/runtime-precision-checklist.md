# CANNBotDSL Runtime 精度调试 Checklist

## 1. 期望输出与问题归因

- 固定 seed、shape、dtype、数据范围。
- 先用期望输出确认可观察差异，再定位到具体失败层级。
- 只有能定位到 CANNBotDSL 前端/lowering/translator/runtime/tooling 的问题，才写入 CANNBotDSL 改进建议。

## 2. Tensor 初始化

- 不读取未初始化 UB/GM。
- 清零 UB 时优先从有限 seed copy 后 `muls(..., 0.0)`，不要对未初始化 UB 自减。
- NPU 测试中避免依赖缺失 packaged operator；CPU 创建 tensor 后 `.npu()` 更稳。

## 3. local_slice/tile_view

检查 lowering 是否保留：

- static offset。
- source offset。
- destination offset。
- element byte size。
- dynamic index 是否走 `tile_view(coord=SSA)` 等已支持路径。
- `local_slice(offset=SSA)` 当前不支持，属于 API 能力边界。

高风险 API：

- `muls`、`mul`、`sub`、`add`。
- `reduce_sum` / reduction。
- UB→UB `mem_copy`。
- `storealign_1st` 单点写。

## 4. mem_copy / DataCopy

- GM↔UB、UB↔L1、L1↔L0、L0C↔GM/UB 要分别确认 layout。
- L0B transpose 语义必须用最小 matmul 证明。
- `mem_copy(..., transpose=True)` 只在已支持方向使用；unsupported 方向应要求 verifier 报错。
- 非对齐或子矩阵 copy 要特别查 stride、burst、offset。
- 多输出或 scratch 较多时，先核对 store 的 GM 目标、base offset 和 element offset，再判断数值公式是否错。

## 5. buffer 与 allocator 复用

- 手动地址、别名 buffer、`reserve_*` 后的 view 必须核对物理字节区间，确认 shape 不会越过被复用 buffer 的真实容量。
- `rewind_*` 只回收 allocator 游标，不自动表达跨 pipe handoff；复用地址前确认上一段执行已经完成，且后续不再读取同一物理 buffer。
- 如果某个 L0A/L0B/L0C tile 单槽已占满硬件空间，不要再按 double buffer 方式 advance 到第二槽；设备错误可先按 L0 越界方向排查。

## 6. sync id

为每个 handoff 建表：

| 数据 | producer | consumer | pipe arrive | pipe wait | id |
|------|----------|----------|-------------|-----------|----|
| 例：`gm_M` | AIV MTE3 | AIC MTE2 | MTE3 | MTE2 | `_M_GM_SYNC_ID` |

风险模式：

- 同 id 多个不相关 producer。
- wait pipe 和 arrive pipe 不匹配。
- id 与 buffer `buf_id` 复用但生命周期不同。
- `id+16` 与另一半 subblock 或 cube lock 混淆。
- ready token 被初始化 token 或 free/release token 顶替，导致 consumer 没有真正等待 producer。
- 使用 per-subblock token offset 的 handoff 只发布或只等待 base id，导致非 base subblock 抢错 token；发布和等待都要按 `id + subblock*16` 对称审计。
- 同一阶段自 `SetFlag` 后 `WaitFlag` 只能说明本阶段初始化/释放链存在，不能证明上游数据已经可见。
- barrier、notify/wait 和 release 链的增删都要回到真实消费链验证，不能只看某个调试 shape 是否返回。

## 7. cube matmul

- 对每种 cube copy/layout 路径先做最小 DSL/codegen/NPU 测试。
- 对比 IR 中的 `copy_l12l0a/b`、transpose 属性、m/k step 和 stride。
- 如果最小复现无法定位到 CANNBotDSL 工具链路径，先不要归入 CANNBotDSL 改进建议。

## 8. 动态控制流

- 动态 `if/for` 必须在 `@kernel` / `@jit` 可预处理区域内。
- 普通 helper 中不能对动态 i64 SSA / dynamic boolean 做 Python truthiness。
- 可选替代：把分支放到 caller，或拆成两个静态 helper。

## 9. 长跑、设备错误与 NaN

出现长时间无输出时：

1. 查进程是否仍消耗 CPU。
2. 查 `.asc` 是否已生成。
3. 查生成代码中的 wait/set flag 是否成对。
4. 缩小到 CPU compile、translate、inverse-only、matmul-only。
5. 必要时终止旧 pytest，避免占用设备。

运行能返回但报设备错误或出现 NaN/Inf 时：

- 先设置可观测检查点或早退开关，把失败压到首个产生异常的计算/搬运边界，不同时修改 sync、layout、dtype 和 tile。
- 设备错误优先审计最近进入的 DataCopy、L1→L0、L0C/FIXPIPE 和手动地址复用；先核对地址、slot、stride、copy 方向是否越界或不支持。
- NaN/Inf 先二分关键中间量的生产链，记录“进入某个检查点前正常、检查点之后异常”的最小边界。
- 如果完整路径才异常，而局部计算早退正常，把最终 store、barrier、wait-only 和写前/写后路径拆开验证，先区分等待链问题还是 copy 出问题。
- `run()` 返回但 `torch.npu.synchronize()` 后失败，说明 kernel 执行期或同步期仍有问题；不要把它归类为 Python host 或 expected output 问题。

## 10. 结果记录

每次调试记录：

- 命令。
- shape/dtype/value range。
- `max_abs/max_rel/rmse` 或关键中间量误差。
- 是否 NPU-only。
- 问题归因：expected output / CANNBotDSL API 边界 / verifier 缺口 / lowering bug / tooling 缺口。
