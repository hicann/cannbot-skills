# 选到最优的 BlockMmad 与 BlockEpilogue（影响精度 + 性能）

> **导航**：本文件是 `catlass-op-design` SKILL.md「Component Selection Methodology」的性能/精度深化版，专讲**怎么把 mmad 组件和 epilogue 组件选到最优**。规则源自实战：把 matmul+激活类算子从「整块 [M,N] HBM 往返」改为「L2 驻留轮转 workspace + 异步回调流水」后，5 个实网 shape 全部反超 torch（1.15×–3.3×）。
>
> 仍以 `catlass/docs/` 与 `examples/` 为准；本文件给出**决策依据与判定规则**，不替代源码阅读。

---

## 0. 一句话结论

> **融合 matmul+epilogue 的性能上限，由 Kernel 的「中间结果 C 怎么落盘」和 BlockMmad 的「流水是否异步重叠」共同决定。** 优先选「小型轮转 workspace（L2 驻留）+ `MmadAtlasA2PreloadAsyncWithCallback`」，避免「整块 [M,N] fp32 C 走 HBM 往返」。

参考基准：本仓 `operators/catlass_quant_matmul_gelu` 用 `QuantMatmulMultiStageWorkspace` + `MmadAtlasA2PreloadAsyncWithCallback`，是「又快又准」的范式，所有融合算子都应向它对齐。

---

## 1. Kernel 选型：中间结果 C 的落盘方式是性能分水岭

catlass 的 matmul+epilogue 类 Kernel 分两大流派：

| 流派 | 代表 Kernel | C 中间结果 | 后果 |
|------|------------|-----------|------|
| **整块 workspace** | `MatmulActivation`、`MatmulEpilogue` | 整块 `[M,N]` fp32 写回 HBM，AIV 再整块读回 | HBM 流量 = `M*N*4 (写) + M*N*4 (读)`，**脱离 L2**；大 N 时成为瓶颈 |
| **多级轮转 workspace** | `QuantMatmulMultiStageWorkspace`（及其 fp16 类比） | 每 (core,stage) 一个 `[L1.M, L1.N]` tile，循环复用，**总量小、L2 驻留** | AIV 读 C **命中 L2**（实测命中率 96–99%），AIC/AIV 通过回调细粒度重叠 |

**判定规则**：
- 算子是 **matmul + 逐元素/反量化 epilogue**（GELU/SILU/RELU/dequant）且 **N 较大**（实网常见 N≥8192）→ **优先多级轮转 workspace**。
- `MatmulActivation`/`MatmulEpilogue` 是「示例级」实现（27/28 号示例用它），**正确但非最优**；小 shape 或一次性验证可用，性能交付**不要**停留在它上面。
- catlass 未提供 fp16 版多级轮转 Kernel 时，**可在算子自己的 op_kernel/ 下新建**一个 fp16 类比（照搬 `quant_matmul_multistage_workspace.hpp` 的轮转 + 回调骨架，去掉 scale/perTokenScale，epilogue 换成逐元素激活）。**不得改动 `catlass/` 源码**。

**workspace 大小公式（多级轮转）**：
```
sizeWorkspace = L1TileShape::M * L1TileShape::N * aicCoreNum * WORKSPACE_STAGES * sizeof(C元素)
```
`WORKSPACE_STAGES` 常取 2（双缓冲）。host 侧必须用 `Kernel::GetWorkspaceSize` 同款公式，**不要**再按 `M*N` 分配。

---

## 2. BlockMmad DispatchPolicy 选型

| DispatchPolicy | 适用 | 性能定位 |
|----------------|------|---------|
| `MmadAtlasA2PreloadAsyncWithCallback` | 融合 matmul+epilogue 流水（配多级轮转 Kernel，量化/非量化均可） | **最优**：预取 + 异步 + 回调，AIC/AIV 细粒度重叠 |
| `MmadAtlasA2Preload` / `MmadAtlasA2PreloadAsync` | GM→L1 带宽瓶颈、需预取/ShuffleK | 较好 |
| `MmadAtlasA2Pingpong` | 基线 | **次优**：双缓冲但无预取，纯 matmul 或一次性验证可用 |

**要点**：`...AsyncWithCallback` 不再是「量化专属」。任何想让 epilogue 隐藏在 matmul 之后的融合算子都应优先用它（搭配多级轮转 Kernel 的 callback 接口）。模板参数（preloadStages/l1Stages/l0A/l0B/l0C/enableUnitFlag/enableShuffleK）以 example 12 与本仓 `catlass_quant_matmul_gelu` 为准；注意 **int8 输入要求 `enableUnitFlag=false`**。

---

## 3. TileShape 选型：受 L1/L0 容量硬约束

大 N-tile（256）能提升 MTE2 复用、减少 A/B 重读；但 K-tile 受 L1 容量限制。

**容量估算（AtlasA2，L1≈512KB，L0C≈128KB）**：
```
L1 占用 ≈ (L1.M*L1.K + L1.K*L1.N) * sizeof(输入) * l1Stages   ≤ ~512KB
L0C 占用 ≈ L1.M * L1.N * sizeof(fp32=4)                        ≤ 128KB
```
- **fp16 输入**：K-tile 通常**上限 256**。`L1<128,256,512>` 在 fp16 下约需 `(128*512+512*256)*2*2 ≈ 768KB > 512KB` → 不行；故 fp16 取 **`L1<128,256,256>` / `L0<128,256,64>`**。
- **int8 输入**：每元素 1B，可放到 **`L1<128,256,512>` / `L0<128,256,128>`**（即 `catlass_quant_matmul_gelu` 的配置）。
- L0C 约束：`128*256*4 = 128KB`，正好贴边；再大 M/N 需切。

**经验起点**（再按 `catlass-op-perf-tune` 单变量微调）：
- int8：`L1<128,256,512>`、`L0<128,256,128>`
- fp16：`L1<128,256,256>`、`L0<128,256,64>`

> `MmadAtlasA2PreloadAsyncWithCallback` 通常要求 `L0.M == L1.M`。

---

## 4. BlockEpilogue 选型：先判「操作数拓扑」，再判槽位

### 4.1 决策树（按 epilogue 需要几路输入、是否跨 N）

```
epilogue 计算需要的算子操作数？
├── 单操作数逐元素（GELU / RELU / 单输入 SILU）
│     → EpilogueAtlasA2ElemWiseNoSource + TileElemWise*
│     → 每个输出块只依赖本块 C，可直接用多级轮转 workspace（每 slot 一个 tile）
├── 双输入（Bias + 激活，第二路来自同列 GM）
│     → EpilogueAtlasA2ElemWiseOneSource（2 个计算 Tile 槽）
├── 量化反量化（+可选激活）
│     → EpilogueAtlasA2PerTokenDequant 家族；AIV 侧执行；
│       per-channel scale[n] 行广播、per-token scale[m] 列广播，fp32 计算
└── ★ 双操作数 / 跨 N-half 门控（SwiGLU = silu(C[:, :H]) · C[:, H:]，H=N/2）
      → 见 §4.2（最容易设计错）
```

### 4.2 ★ 跨 N-half 门控（SwiGLU）：默认的多级轮转 epilogue 会算错

SwiGLU 输出 `D[:, j] = silu(C[:, j]) · C[:, j+H]`，两个操作数**相距 H=N/2 列**。

- **陷阱**：stock `QuantMatmulMultiStageWorkspace` 每个轮转 slot 只存一个 `[L1.M, L1.N]` 的 C tile，于是「+H 的配对列」落在**另一个 N-block**，本块 epilogue 读不到 → 只有当 **`N ≤ L1TileShape::N`**（如 512）才碰巧正确，**N 更大时整片输出错误**（本仓曾出现：512³ 过、N=18432 时 ~44% 元素错、最大绝对误差 171.5）。
- **正确设计（二选一）**：
  1. **按输出形状 `[M, H]` 调度**；每个输出块让 AIC 发起**两次 matmul**（左 N-tile 在列 c、右 N-tile 在列 c+H）写入**同一 stage 的两个共存 workspace region**，右块写完才置「stage 就绪」标志；epilogue 取双 C 反量化后 `silu(x1)*x2`。（推荐，保留异步回调多级流水。）
  2. **整块/全宽 C** + epilogue 用第二个 GM 源指针读 `+H` 列（全局列偏移）。正确但回到整块 HBM 往返，性能次之。
- **校验**：设计阶段必须问「epilogue 的操作数是否跨 N-block？」凡是 gate/门控类（SwiGLU、GeGLU、ReGLU）都跨，必须按输出形状调度并产出两路 tile。

### 4.3 槽位与 UB 预算（自定义 epilogue）

- 打开目标 `EpilogueDispatchPolicy` 特化头，逐槽列模板形参（见 SKILL.md Step 5），标 ✅/🔧/❌。
- **UB 预算**：`UB_STAGES * (各 staged buffer) + 常驻 fp32 工作 buffer ≤ ArchTag::UB_SIZE`，双 C 的 SwiGLU epilogue 常驻 buffer 多（C1f/C2f/x1/x2/silu 等），需据此定 `UB_STAGES`。
- **硬件事件 ID 上限（易致死锁）**：同一 `HardEvent` 类型（如 `V_MTE2`）的事件 ID **最多 8 个**。`UB_STAGES` 过大会让所需事件 ID 超限（如双 C swiglu epilogue `UB_STAGES=2` 需 10 个 `V_MTE2` ID > 8）→ **运行期挂死**。**先数事件 ID 再定 `UB_STAGES`**，双输入/多 buffer epilogue 常取 `UB_STAGES=1`。

---

## 5. 精度相关的选型约束

- **累加精度**：`CType = GemmType<float, ...>`（fp32 累加）；量化为 `int32`。
- **epilogue 在 fp32 域计算激活/反量化，最后一步才 `Cast` 到输出 dtype**；golden 必须镜像这一顺序（见 [precision-verification.md](../../catlass-op-develop/references/precision-verification.md) §3）。
- 选错累加/计算精度（如过早 cast 到 fp16 再算激活）会引入系统性误差——属选型问题，非验证问题。

---

## 6. 选型自检清单（mmad + epilogue）

- [ ] Kernel 是否避免了整块 `[M,N]` HBM 往返？大 N 是否用了多级轮转 workspace？
- [ ] DispatchPolicy 是否为融合流水选了 `MmadAtlasA2PreloadAsyncWithCallback`？
- [ ] TileShape 是否满足 L1/L0 容量约束（fp16 K≤256；int8 可 K=512）？大 N-tile？
- [ ] epilogue 操作数是否跨 N-block（SwiGLU 类）？若是，是否按输出形状 `[M,H]` 调度并产出两路 tile？
- [ ] 自定义 epilogue 的 `UB_STAGES` 是否通过了 UB 预算 + 事件 ID(≤8) 双重核算？
- [ ] CType=fp32 累加、epilogue fp32 计算末尾 cast，golden 已镜像？
- [ ] workspace 大小用 `Kernel::GetWorkspaceSize` 同款公式，host 侧已同步？
