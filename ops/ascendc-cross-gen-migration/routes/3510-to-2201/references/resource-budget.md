# 预算差异与重算

> 迁移后最容易被忽略的失败不是"编不过"，而是**UB 越界 / L0C 溢出 / tiling 假设失效**。A5 的 tile 常量直接搬到 A2 几乎一定不匹配。

## 1. 架构预算对照（典型值，运行时以接口为准）

| 资源 | DAV_2201（A2/A3） | DAV_3510（A5） | 迁移影响 |
|---|---|---|---|
| UB（每 AIV 用户可用） | **192 KB** | 248 KB | 可用空间少约 56KB；A5 若跑 SIMT/RegBase 还另扣 40KB（见下） |
> 口径说明：248KB 是 `GetCoreMemSize(UB)` 返回的**每 AIV 用户可用值**；`l1-guide.md` 的 216KB 是从**物理 256KB** 起算（256−40）的另一个口径，两者不可混用。本路线统一按 `248−40=208KB` 估算 A5 可用 UB。

| A5 实际可用 UB（SIMT/RegBase 路径） | — | 248 − 40 = 208 KB | 真正常见的对比是 **192 vs 208**，比例约 **0.923** |
| L0C | **128 KB** | 256 KB | Cube 输出驻留能力减半，baseM×baseN 必须重算 |
| L0A / L0B | 64 KB | 64 KB | 不变 |
| L1 | 512 KB | 512 KB | 不变（通路能力不同，见 `a5-only-inventory.md`） |
| BT | **1 KB** | 4 KB | 依赖大 BT 的广播/fixpipe 参数需调整 |
| 4:2 稀疏 | 支持 | 不支持 | A2 可恢复（性能机会） |
| L2 | 典型 192 MB（910B2） | 950PR 112/128 MB | A2 缓存更大，tiling 可少拆 |
| Cube / Vector 核 | 典型 24 / 48（910B2，1:2） | 典型 28 / 56 或 32 / 64 | 多核切分与负载均衡重算 |
| 频率 | 典型 1.8 GHz（910B2） | 典型 1.65 GHz | 跨架构比时延没有意义 |

> 来源：`npu-arch` 的 `references/npu-hardware-params.md` §2.3/§2.5 与「子型号变化参数」。**禁止硬编码这些数值**：kernel/tiling 用 `PlatformAscendC::GetCoreMemSize(...)`、`GetCoreNumAiv()/GetCoreNumAic()`。

## 2. UB 预算公式

```
UB_total   = GetCoreMemSize(CoreMemType::UB)           // A2 典型 196608 B
UB_avail   = UB_total - reserve                        // reserve 由 kernel 自身决定（对齐/临时区/框架预留）
UB_used    = Σ(TQue_depth × tileElems × sizeof(T)) + Σ(TBuf_len)
约束：UB_used ≤ UB_avail
```

- A5 代码里的 `ubSizePlatForm - SIMT_UB_SIZE_BYTE`（40KB）在 A2 上**必须删除**：A2 无 SIMT DCache 复用。
- 删除后不要直接把 248→192 一改了事：A5 的 `tileElems` 是按 208KB 算的，A2 应按 192KB 重新求 `tileElems`，并保证 32B（或 dtype 对应 DataBlock）对齐。

### 2.1 A5 → A2 tile 缩放

```
scale = UB_avail_A2 / UB_avail_A5            // 典型 192/208 ≈ 0.923
tileElems_A2 = floor(tileElems_A5 × scale / align) × align
```

- `align` 取 32B / sizeof(T) 与向量 VL 的最小公约（按算子实际访问粒度）。
- 只缩 tile 不改循环结构时，注意多 tile 带来的循环开销与尾块比例上升；尾块占比高时优先调整切分维度而不是硬缩。

## 3. L0C 预算

```
L0C_used ≈ baseM × baseN × sizeof(C) (+ 对齐/padding)
约束：L0C_used ≤ GetCoreMemSize(CoreMemType::L0C)     // A2 典型 131072 B
```

- A5 的 `256×256 fp32 = 256KB` 在 A2 直接超限（L0C 128KB）：`baseN` 或 `baseM` 至少减半，或在 L0C 上分段 `Fixpipe`。
- L0C→UB 直通被移除后，`workspace` 大小要计入 tiling 的 `GetWorkspaceSizes`；workspace 是 GM，不受 UB 限制，但影响带宽与延迟。

## 4. BT / 分形相关

- BT 从 4KB → 1KB：依赖 BT 的广播表、fixpipe 参数、`LoadData` 边界表要重新核对容量。
- L0A 分形 NZ → ZZ：`baseK` 的对齐粒度与 `LoadData` 参数随之变化，不能用 A5 的 `srcStride/dstStride`。

## 5. 多核切分

- 核数从 A5 的 28/32（Cube）降到 A2 典型 24：原来"每核一块"的切分会出现**核数不整除**，尾核负载不均。
- 切分策略优先级：① 按 batch/head 等天然维度切；② 按 M/N 切并让尾核拿小片；③ 单核内再按 UB 分块。
- 用 `GetBlockNum()` 而不是硬编码；tiling 里用 `GetCoreNumAiv()` 计算 `usedCoreNum`。

## 6. 重算步骤（阶段 2 必做）

1. 列出 kernel 内所有 `TQue`/`TBuf`（名字、dtype、元素数、depth）。
2. 用 `scripts/a2_ub_budget.py` 跑一遍 A5 原始预算，确认"A5 能过、A2 超限"的具体项。
3. 按 §2.1 缩放候选 tile，再检查：
   - 尾块占比是否可接受；
   - 循环次数增加是否让 scalar 开销变主瓶颈；
   - `DataCopyPad` 次数是否暴增。
4. L0C/Cube 相关按 §3 重算 base 块。
5. 把最终预算表写进分析报告（阶段 1 模板 §5）与交付报告。

## 7. `scripts/a2_ub_budget.py` 用法

```bash
# 按 buffer 明细核算 A2 预算（元素数 × dtype 字节 × 份数）
python3 scripts/a2_ub_budget.py \
  --buffer in_x:float16:32768:2 \
  --buffer in_y:float16:32768:2 \
  --buffer out:float32:32768:2 \
  --buffer tmp:float32:16384:1 \
  --ub-kb 192 --reserve-kb 0

# 从 A5 tiling 推导 A2 建议 tile（默认 A5 可用 208KB = 248-40）
python3 scripts/a2_ub_budget.py --scale-from-a5 --tile-elements 32768 --dtype float16 --align-bytes 16

# 机器可读
python3 scripts/a2_ub_budget.py ... --json
```

输出包含：逐 buffer 字节、总量、余量、判定（PASS/OVER）、超限时的最大可缩放比例与建议值。

> 脚本只做算术，不替代对访问模式（对齐、padding、双缓冲策略）的判断；判定 PASS 后仍要在阶段 3/4 用真机验证无 `UB out-of-range`。
