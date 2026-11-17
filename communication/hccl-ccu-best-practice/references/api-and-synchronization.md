# CCU API 与同步

索引中的 `HCOMM/include/ccu/`、`HCCL/src/common/hcomm_dlsym/ccu/` 是两侧接口入口。使用一个 API 时同时核对 HCOMM 声明/实现、HCCL 包装和目标安装版本。公共文档与代码存在差异时记录差异，按目标链路查证；例如当前宏实现包含 Variable 比较路径，而部分 `CCU_IF` 文档仍描述旧限制，不应据其中一处声称所有版本支持。

## 按动作查接口

| 动作 | 入口/关键符号 | 要核对的契约 |
|---|---|---|
| 加载启动参数 | `ccu_primitives.hpp`: `LoadArg` | index 对应 host 的 64 位槽 |
| 本地搬运 | `LocalCopy` 重载 | LocalAddr/CcuBuffer 方向、长度单位、完成 Event |
| 远端读写 | `Read` / `Write` | channel、地址类型、token、长度及本地完成信号 |
| 归约 | `LocalReduce` / `ReadReduce` / `WriteReduce` | 目标位置、dtype/op 支持、初值与贡献次数 |
| 交换地址/token | `GetResByChannel<Variable>`、`WriteVariableWithNotify` | peer 索引、XN 槽、Notify 槽/mask 一致 |
| 本地完成 | `EventRecord(Event, mask)` / `EventWait(Event, mask)` | 同一 Event 与待完成 bit 集合 |
| 跨端同步 | `NotifyRecord` / `NotifyWait` | 对端/本端 Notify 索引、mask、channel 所属 die |
| 同 die 跨 kernel | `EventRecord/Wait(const char *tag, mask)` | 当前注册上下文及同 die 限制；不代替跨 die/rank 通知 |
| 动态控制流 | `ccu_control_flow_macro.h` | 运行期条件、配对、目标版本支持 |
| 硬件循环 | `ccu_loop.hpp`、`Loop/LoopGroup` | loop body 允许的原语、资源和版本差异 |
| 注册与启动 | `ccu_launch.h` | 注册期参数与 taskArgs 不同，错误必须传播 |
| 资源/token | `ccu_res.h` | host 侧生命周期、注册范围、die；不要在 kernel 中调用 host 资源查询 |

详细 API 文档入口为 `HCOMM/docs/zh/api_ref/comm_opdev/data_plane_api/ccu/`。只读取本次使用的原语，不将全部手册复制到任务上下文。

## 指令图阶段和运行期

kernel 的 C++ 代码在 host 上为 CCU 生成指令图。普通 `for` 会展开指令；CCU 动态循环由 DSL 构造。生成期拿不到 `LoadArg` 的运行期数值，不能用 host 控制流对它做分支。

当前 `ccu_variable.hpp` 中，`Variable` 拷贝构造复制句柄，拷贝赋值发出值赋值指令，而移动赋值又复制句柄。这影响别名、资源占用和后续算术：新建独立寄存器并赋值与直接拷贝构造不是同一动作。以实际重载为准，不用一般 C++ 数值直觉推测。

`CCU_IF` 的关闭可能延迟至后续 API/注册结束；`CCU_ELSE` 必须紧跟对应块，不插入其他 CCU API。控制流宏内部失败可能跳过 body，不能仅以注册返回成功判断分支已正确生成。`ccu::Loop` body 与普通 CCU 动态控制流不是同一机制，组合前核对约束。

## 完成事件与远端握手

- 搬运接口的 `(Event, mask)` 表达该操作的本地完成，不能自动等同于所有 peer 已读完 buffer。
- 远端地址/token 交换中，发送方写远端 XN 并发 Notify；接收方在匹配 Notify 到达后使用对应 XN。
- Notify 的 remote/local 索引是双方约定，框架不自动匹配。channel 绑定 die 与 kernel die 一致。槽数量按实际资源和版本核对。
- 事件 mask 是 16 位。超过 16 个独立信号须分组，不能把 `1 << rank` 截断到 `uint16_t`。HCCL 的 `CcuEventGroup` 是可参考的分组辅助，不必强制所有算法使用它。
- 聚合等待 mask 可以对应多个操作完成位；须证明所有位在同一轮可达。条件跳过的 producer、单 rank、零片和尾片都要验证。
- 重复使用 Event/Notify 前查清当前实现的等待/清位语义及复用时序。不要凭事件变量名称假设信号已清零，不增造当前 API 没有的 reset 方法。

## 地址与 token

`LocalAddr`、`RemoteAddr` 不是裸指针：地址及访问 token 都属于协议。token 的来源/注册范围必须跟随地址，不能将一个 buffer 的 token 任意用于另一注册域。HCOMM `ccu_res.h` 明确 `HcommCcuGetMemToken` 在 host 调用，token 不应写入日志。

Host 计算 `base + offset` 时核对整数溢出、长度与已注册范围；kernel 的地址增量核对 slice/repeat/step 的边界。保持 dtype count 和字节长度区分，尤其是归约及跨 executor 分块。


## 设计期同步契约检查

`knowledge-baseline.json` 的 `sync-limits` 卡覆盖 `ccu_transport_.{h,cc}`、`ccu_primitives.hpp`、V1/V2 translator 与 microcode。在已核对的 HCOMM d1c2b18c2 实现中：

- `CcuTransport::Init` 以 INIT_CKE_NUM=4、INIT_XN_NUM=4 初始化；GetLoc/Rmt*ByIndex 检查实际 vector 边界。此路径每 channel 合法 Notify/XN index 为 0..3，不能照旧 Notify 文档的 0..7 使用。4 是此实现的已分配数，不是硬件理论上限。
- mask 在公共 C++ 原语为 uint16_t。**slot index 和 mask bit 是两个维度**；样本 POST_SYNC_ID=3 若用于 `1<<3`，不表示用了第3号 slot。
- V1/V2 的 Event/Notify Wait 译码传 clearType=1，microcode 对 wait mask 对应位清零。Wait 是消费操作，不是可任意重复读取的锁存状态；清位不自动证明下一轮不会过早发信号。

把设计中的各接收 channel 表（含 N=2 去重后的表）分别声明到 [sync-contract.example.json](../assets/sync-contract.example.json)，涵盖允许 rank 上界而非只写冒烟规模：

```bash
python3 "$SKILL_DIR/scripts/check_sync_contract.py" /path/to/task/sync-contract.json \
  --hcomm-root "$HCOMM_ROOT"
```

`pools` 是互相独立的接收表/边界场景；每个 pool 声明 XN indices 和 epochs，每个 epoch 的 signals 列 name、slot、mask，表示可能同时存活的位。未知是否能同时到达时放在同一 epoch。跨 epoch 复用位须有 `reuse_proof`，证明所有旧消费者完成在新 producer 之前；仅“本 rank 已 Wait”不一定足够。每个 pool 的 `source_evidence` 关联本任务设计和实际源码。

检查器对照目标 HCOMM 的已核对字节与实际常量，检查配额、16位 mask、同表同 epoch 碰撞，并拒绝无说明复用。来源漂移先定向核验/维护卡，不使用文档上限或随意抬高参数绕过。它只校验声明，不自动解析 C++、验证所有 rank/epoch 已列全、证明清位/复用或死锁自由；Reviewer 仍需核对这些语义。


## 生成器算术能力

`generator-arith` 卡把公共 Variable API、C适配、kernel lowering、rep枚举、V1/V2生成器与运行期选择一起锚定。已核对的 HCOMM d1c2b18c2：`ccu_ins_generator_v1.h` 的 repTypeInstrCount（160–197行）算术仅 ASSIGN/ADD；V2 表（v2.h:217–220及位运算项）增加其他算术。V1 `GetInstrCount`（v1.cc:265–271）查不到rep即抛 Unsupported repType；不是编译错误。

| 运行期 Variable 操作 | 需核对的 rep | V1 | V2 |
|---|---|---|---|
| 赋值 | ASSIGN | 支持 | 支持 |
| 加法、`+=` | ADD | 支持 | 支持 |
| 乘法、`*=` | MUL | 不支持 | 支持 |
| 减法、`-=` | SUB | 不支持 | 支持 |
| 与/或/异或/取反 | AND/OR/XOR/NOT | 不支持 | 支持 |
| 位移 | SHL/SHR | 不支持 | 支持 |

这里的表只讨论算术；V1表也有 JUMP_EQ/JUMP_NE，不能把所有比较一概列为仅V2。其他关系比较和控制流需按具体translator与DSL核对，本算术检查器不覆盖控制流rep。

`Variable * uint16_t` 的路径是 `ccu_variable.hpp` → `CcuVariableMulImmToVar` → `CcuKernel::VariableMulImmToVar` → `ccu_rep_mul.cc::SetCommonInfo`（20–24行，设置MUL并查询GetInstrCount）。V2 translator的 `VAR_MUL_IMMED_TO_VAR` 分支（v2.cc:601–644）生成 `CcuV2::MulI`；这两处也纳入卡的字节范围。其他表达式仍需核对自己的subtype，rep表有条目不替代具体lowering审查。公共重载存在、语法检查通过或编译成功都不能证明目标生成器可翻译。`ccu_kernel_mgr.cc:108–123` 从运行期 CcuVersion 选择 V1/V2；不能凭“CCU_SCHED”“VM”或芯片名字猜代际。

设计在 `sync-contract.json.compute_ops` 声明 `target_generator`、每个运行期算术表达式、实际rep和源码位置。目标未确认用 UNKNOWN，检查V1/V2交集（当前算术为ASSIGN/ADD）。明确使用V2时，`target_evidence` 必须给出初始化日志/摘录的 path 和 sha256，其中能识别 `Init CcuInsGeneratorV2` 且无矛盾版本；相对路径基于契约文件目录。证据属于哪个设备/部署身份仍由Verifier/Reviewer确认，检查器不会自行探测设备或启动VM。

```bash
python3 "$SKILL_DIR/scripts/check_arith_ops.py" /path/to/task/sync-contract.json \
  --hcomm-root "$HCOMM_ROOT"
```

工具在来源字节MATCH后，从目标头文件实际提取rep计数表，检查声明、目标证据内容与版本支持。它不解析任意C++、证明映射完整、验证日志属于当前部署或证明替代算法正确。Reviewer须逐项核对新kernel的实际类型和lowering；grep用于定位残留算术，不能机械禁止合法host乘法，也不能以无乘号证明兼容。

在V1上计算 `runtime_stride * host_index`：可在host预乘后传启动槽（同步更新launch ABI/cache），或用host常量index的二进制分解生成ASSIGN/ADD链。位移/判断只作用于host整数，不能偷偷改成Variable SHL。使用独立的Variable句柄再赋值，不用拷贝构造别名修改原stride；证明立即数范围（相关API为uint16_t）与地址溢出边界。将原MUL改为ADD后，契约登记实际ADD lowering，不能仍声称直接MUL可用。
