# Broadcast 范式 AscendC 能力映射

> **层级**：AscendC 语言适配层。对 [../../generic/capability-contract.md](../../generic/capability-contract.md) 中全部能力谓词给出 AscendC 真值与 API 实体。

## 1 谓词真值表

| 谓词 | AscendC 真值 | API 实体 / 规则文档 |
|------|------|------|
| `CAP_BROADCAST_IN_TRANSFER` | **true** | NDDMA `DataCopy<T, ND, cfg>`，B 轴 `srcStride=0` 原地重读；规则见 [common/nddma-rules.md](../../../common/nddma-rules.md) |
| `CAP_REG_CHAIN_FUSION` | **true** | `asc_vf_call` + RegBase（`Reg::Mul→Reg::Sub→Reg::Add` 链式中间值留 VF 寄存器）；规则见 [common/vf-programming-rules.md](../../../common/vf-programming-rules.md) |
| `CAP_FUSED_ARITH` | `MulAddDst, FusedMulAdd, AddRelu, SubRelu, FusedMulAddRelu, MulCast` | 一条指令完成两步，dst 就地覆写省 1 个中间 buffer；存在性以 [API 白名单](../../../../knowledge/api/regbase_api_whitelist.md)为准 |
| `ALIGN_GRANULARITY` | **32 字节** | `ONE_BLK_SIZE = 32`；`perBufBytes = (UB / P) & ~31` |
| `TRANSFER_MAX_DIMS` | **5** | NDDMA `dim ∈ [1, 5]`；超出的维度走 outer iters 逐段搬运（见 [kernel-helpers.md](kernel-helpers.md) §3） |
| `FUSION_CHAIN_MAX_LEN` | **7** | 超过 7 条 Vector 操作必须拆为多次 VF 调用 |
| `CONVERT_BREAKS_CHAIN` | **true** | Cast 不走 Vector 寄存器链路，前后必须断链 |
| `CONVERT_INPLACE` | **false** | FP16 2B vs FP32 4B，硬件不支持重叠读写；Cast 需独立中转 buffer |
| `COMPUTE_PRECISION` | **FP32（4 字节）** | VF 函数始终以 `<float>` 实例化；`perBufElems = perBufBytes / 4` |
| `BUFFER_MODEL` | **TBuf 显式单缓冲**（禁 TQue） | Broadcast 算子一律 TBuf 管理 UB，禁止 TQue 和裸指针 |
| `SYNC_PRIMITIVE` | `SetFlag<HardEvent::XXX>` / `WaitFlag<HardEvent::XXX>`，成对使用 | eventID 经 `GetTPipePtr()->FetchEventID()` 获取，不可硬编码；Atlas 训练系列 0–3，其他 0–7，6/7 不可用 |
| `EXEC_UNITS` | MTE2（搬入）/ V（计算）/ MTE3（搬出） | 事件命名 `源流水_目标流水`，如 `MTE2_V` = V 等 MTE2 |

## 2 通用术语 → AscendC 实体对照

| 通用层术语 | AscendC 实体 |
|------|------|
| 片上 buffer / 片上容量 | UB / `GetCoreMemSize(CoreMemType::UB)` |
| 搬入（随路广播） | NDDMA `DataCopy<T, ND, cfg>`（B 轴 srcStride=0） |
| 搬出 | `DataCopyPad`（`blockLen = count × sizeof(T)`） |
| 寄存器链融合 | VF 寄存器链（`asc_vf_call` + `LoadAlign`/`StoreAlign`） |
| 硬件融合指令 | `MulAddDst` 等（`CAP_FUSED_ARITH` 表） |
| 精度转换 | `Cast(dst, src, roundMode, count)`，第 4 参数不可省略 |
| 单缓冲管理 | `TBuf<TPosition::VECCALC>` + `pipe_.InitBuffer` |
| 正向/反向同步 | `SetFlag`/`WaitFlag`（MTE2_V / V_MTE3 / V_MTE2 / MTE3_MTE2） |
| 编译期特化（RANK 分档） | `ASCENDC_TPL_*` 模板参数（禁 `TILING_KEY_IS`） |
| dtype 注入 | `DTYPE_IN0` 编译宏（不进 TilingKey） |

## 3 关键语言级附加约束

1. 所有 API 可用性判定的唯一入口是 [API 白名单](../../../../knowledge/api/regbase_api_whitelist.md)；不得把"待验证"写成"已验证"
2. 交叉编译器限制：宏参数里不能有 `<>`（用 `using` 别名）；`reinterpret_cast` 跨地址空间禁用——详见 [pitfalls.md](pitfalls.md)
3. 平台参数（coreNum/ubSize/blockSize 等）严禁写死，必须通过接口获取并做零值守卫
