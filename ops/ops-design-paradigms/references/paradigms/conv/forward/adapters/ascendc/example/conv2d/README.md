# conv2d — Conv2D 推理算子 AscendC 参考实现（tiling + kernel）

> 定位：Convolution 范式的 AscendC 参考实现，覆盖 **tiling（分核 + 单核切分）与 kernel（Cube 主通路迭代）** 两条主线，每个代码段注释标注对应范式章节。
>
> **拉起方式不在本示例范围**：`<<<>>>` 内核直调与 aclnn 注册两段式是拉起本算子的两种接入形态，由上层工作流技能保证——本仓 `ops/ascendc-direct-invoke-template`（内核直调）、`ops/ascendc-registry-invoke-template`（aclnn 注册，含 Cube kernel 核类型声明 `ASCENDC_TPL_KERNEL_TYPE_DECL(KT, ASCENDC_TPL_AIC_ONLY)`，漏声明会按 VectorCore 编译空跑）。

## 支持的规格范围（范式 verification.md §2.1 验收标准）

| 维度 | 范围 | 实现位置 |
|------|------|---------|
| 空间维 | **1D / 2D / 3D** | `conv2d_kernel.cpp` 含 D 轴外层循环与 kd 切片归约；1D 由调用方升维 di=hi=1 |
| dtype | fp16 | 全代码使用 `half` |
| stride | 1~63（指令上限），≥1 防御 | `CalcConv2DTiling` 指令预检 |
| dilation | 1~255 | 同上 |
| pad | 非对称、< kernel | 逐维头尾独立传递 |
| groups | =1 | 不含分组逻辑 |
| L1 容量 | BL1 驻留 / NONE 流式自动判定 | `CalcConv2DTiling` 步骤④⑤ |
| 多核 | batch × dout × M 三维绑核 | `CalcConv2DTiling` 步骤② |
| K 分块 | 大通道/大核自动切 kChunk | `CalcConv2DTiling` 收缩循环 |
| 空输出 | 合法规格跳过 | `CalcConv2DTiling` 返回 1 |

> **fullLoad 状态说明**：参考实现按 tiling.md §3.3 四态**裁剪**为两态——**BL1 驻留**（weight 全量驻留 L1，含"fmap 行带 + weight 同时容纳"的 ALL 退化态）与 **NONE**（双侧流式）。A 侧驻留（N-first 迭代序、fmap 跨 N 块复用）为范式扩展点（compute-flow.md §6.7/§6.8），生成算子按范式四态实现时以 tiling.md 为准。

## 文件结构

```
conv2d/
└── op_kernel/
    ├── conv2d_tiling_data.h   TilingData POD + 平台常量 + 指令限制常量
    ├── conv2d_tiling.h        泛化 Tiling 计算（预检 → 多核 → L0/L1 切分 → fullLoad）
    └── conv2d_kernel.cpp      泛化 Kernel（batch×dout×mBlk×nBlk×kd×kChunk×kL0 七层迭代）
```

## Tiling 泛化决策流程（`CalcConv2DTiling`）

```
输入规格 (Conv2DSpec)
    │
    ▼
⓪输入防御/指令预检 ─ stride≥1 / stride≤63 / dil≤255 / H·W pad≤255 / dk≤511 ── 超限 → return -1
    │
    ▼
①输出/MNK ──── dout = f(di, padD, kd, sd) × ho × wo
    │            m = ho × wo（Do 独立外层）
    │            k = ciAlign × kd × kh × kw
    │
    ▼
②多核切分 ──── batchDim × doDim × mDim ≤ 32
    │            逐维贪心 + CeilDiv 兜底
    │
    ▼
③L0/L1切分 ── mBlk: L0C 上限向下搜（≤2048），并按 Hi_L1×Wi ≤ 32767 收缩
    │            nBlk / kChunk: L1 预算内迭代收缩（级0a缩N→级0b缩K）
    │            nBlk 上界受 BT 段容量约束（≤ BT容量/4B = 1024）
    │            kL0: L0A/L0B 容量取最大 K0 倍数
    │
    ▼
④fullLoad ─── BL1 驻留?（kd=1 且 coAlign≤BT上限 且 weight全量+A侧行带 ≤ L1 预算）
    │            → bResident=1, nBlk=coAlign, kChunk=全K
    │            否则 NONE（流式）；按最终参数重算 al1Size/bl1Size
    ▼
输出 Conv2DTilingData（blockDim + 块大小 + 驻留标志）
```

与范式完整算法的差异（简化项，生成算子按范式实现）：

- 绑核为 do → batch → M 逐维贪心绑定剩余核数（CeilDiv 兜底），未实现 tiling.md §2.3 的因子搜索与代价模型，也未绑 N（cout）轴——Co 切分由单核内 nBlk 循环承接；
- fullLoad 状态收敛为 BL1/NONE 两态（见上）。

## Kernel 迭代骨架（七层循环）

```cpp
for batch → for dOut → for mBlk → for nBlk → for kd → for kChunk → for kL0 {
    A侧L1装载（或全零填充）→ LOAD3D → LOAD2D → MMAD → FIXPIPE
}
```

## 关键正确性约束（实现/移植时勿回退）

| 约束 | 位置 | 说明 |
|------|------|------|
| 空尾核判界 | kernel 多核反解 | CeilDiv 切分存在 `mStart ≥ M` 的空尾核，必须先判界再算余量——无符号减法下溢会绕过 `==0` 检查导致越界迭代（compute-flow.md §5.3） |
| 行推理下边界钳制 | kernel 行推理 | `hiLoad ≤ Hi − hiIdx`；越界行属底部 pad，由 FMatrix `padBottom=255` 虚拟填零，不得从 GM 读入（compute-flow.md §6.3） |
| 最坏情形行推理 | tiling A 侧预算 | 中间列起块比行对齐多跨最多 1 行，预算按 `CeilDiv(mBlk+Wo−1, Wo)` 口径（tiling.md §3.3） |
| 16bit 指令限制 | tiling | `Hi_L1×Wi ≤ 32767`（LOAD3D L1 图积，Hi_L1≥1 时覆盖 DN2NZ nValue/dstNzC0Stride 16bit）；超限收缩 mBlk，仍超限 return -1 |
| BT 段容量 | tiling | `nBlk ≤ BT容量/sizeof(L0cT) = 1024`；BL1 驻留要求 `coAlign ≤ 1024`（驻留态 nBlk=coAlign），否则保持流式 |
| 3D 通道步进 | kernel B 侧装载 | weight `[Co][Ci][Kd][Kh][Kw]` 中 kd 片跨通道步进 = `Kd·Kh·Kw` 而非 `Kh·Kw`（compute-flow.md §2.2） |
| 末步 unitFlag | kernel MMAD | K 归约末步 `unitFlag=3`（置位+翻转），判定须与逐层取余的循环界一致（compute-flow.md §6.6） |

## 拉起与验证

本示例只交付 tiling + kernel 主体。拉起接入、用例数据生成、golden 与精度比对由上层负责：

- 内核直调（`<<<>>>`）：`ops/ascendc-direct-invoke-template`
- aclnn 注册（OpDef/TilingFunc/两段式 API）：`ops/ascendc-registry-invoke-template`
- 验证方法论（用例设计/输入构造/golden/容差判定/失败定位）：范式 [verification.md](../../../../generic/verification.md)
