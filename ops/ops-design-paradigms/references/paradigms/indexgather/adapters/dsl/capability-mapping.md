# IndexGather 范式 CANNBotDSL 能力映射

> **层级**：DSL 语言适配层。对 [../../generic/capability-contract.md](../../generic/capability-contract.md) 中全部 17 项通用能力谓词给出 DSL 真值与实体；DSL 专属谓词仍见 [language-rules.md](language-rules.md) §0。

## 1 通用谓词真值表

| 谓词 | DSL 真值 | DSL 实体/来源 |
|------|------|------|
| `SIMT_SUPPORT` | **true** | `vf(mode="simt")` |
| `VF_SERIAL_EXEC` | **true** | 多 `vf` 区域串行；跨区域访存天然有序 |
| `VF_START_COST` | 固定启动成本 | VF 启动固定成本；纯计算循环合入单 vf |
| `GM_DIRECT_ACCESS` | **true** | tracked `gm[k]` 读 / `gm[k] = v` 写 |
| `DCACHE_SIZE` | **32KB** | SIMT 侧固定；无查询接口 |
| `DCACHE_RESERVE_RANGE` | 固定 **32KB** | `get_mem_size("ub")` 已扣 8KB；DCache 另留 32KB；DSL 无 host 可配范围 |
| `UB_CAPACITY` | **253952**（248KB） | `cannbotdsl.get_mem_size("ub")`；不含 SIMT DCache 32KB |
| `SIMT_REGISTER_MODEL` | 寄存器总量固定；线程翻倍→每线程减半；溢出到 GM 栈 | `vf(mode="simt", thread=N)` 的线程数选型 |
| `SIMT_THREAD_CHOICES` | **1024 / 2048**（示例档位；其他档位待平台验证） | `vf(mode="simt", thread=N)` |
| `CORE_NUM` | 运行期派生 | `get_platform_info().vector_core_num` |
| `MAX_RANK` | **8** | 8 维结构化 TensorSpec / metadata |
| `INDEX_DTYPE_SET` | **int32 / int64** | index 张量 dtype；与 use64 分离 |
| `UINTDIV_MAGIC_SHIFT` | 不适用（无 magic/shift 接口） | 固定除数统一 `//` |
| `UINTDIV_VF` | 语言级 `//` | 设备循环内固定除数统一 `//` |
| `UINT32_MULHI_NATIVE` | `simt.mulhi` 可用 | 当前解码路径统一 `//`，不依赖 mulhi |
| `UINT64_MUL_HIGH_COST` | i64 算术链；无专用展开特化 | use64 路径用 i64 域；当前不依赖高乘指令 |
| `BYTE_LEVEL_MOVE` | **true** | 原生 dtype 纯搬移；受限 dtype 在 torch 层 `view` 别名 |

## 2 DSL 专属谓词

语言特有规则不进入通用契约，保留在 [language-rules.md](language-rules.md) §0：

- `SIMT_LOOP_FORM`
- `SCALAR_CANONICAL_GUARD`
- `RANGE_ITER_BARE`
- `SAME_TYPE_CAST_REJECT`
- `DIM_SPEC`
- `META_DIRECT_READ`
- `HOST_DERIVE`
- `COMPILED_PROGRAM_DISPATCH`
- `UNSIGNED_LITERAL_BITCAST_BUG`
- `ABI_DTYPE_GUARD`
- `EMPTY_TENSOR_EARLY`

## 3 关键约束

1. 通用层只引用谓词名；DSL 侧具体值以本表和 [language-rules.md](language-rules.md) 为准。
2. `UINTDIV_*` 在 DSL 路径统一使用 `//`，不引入 magic/shift 分支。
3. `SIMT_THREAD_CHOICES` 的 1024/2048 为示例档位；其他档位需按平台与寄存器预算验证。
4. `BYTE_LEVEL_MOVE` 通过原生 dtype 纯搬移实现；fp64/complex/fp8/bool 等受限 dtype 在 torch 层做位级 view 别名。
