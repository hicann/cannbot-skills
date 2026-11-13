> **层级**：AscendC 语言适配层（adapters/ascendc/）。bias 的数据流与语义见 [../../generic/compute-flow.md](../../generic/compute-flow.md) §2.3（语言中立层）；本文件只收录 BT 装载序列的落地写法。参考实现见 [example/conv2d/](example/conv2d/)。

# bias 通路落地（BT）

## 1 装载流程

```
GM ──────(全量一次)──► L1 ──(逐 N 段)──► BT ──(首步 Mmad 读 BT)──► L0C 累加初值
  Co^align 个元素        coAlign·2B           nAlign·2B               cmatrixSource=true
  （host 侧已零填充）    （bias dtype）       （bias dtype）           （nAlign = AlignB(curN,16)）
                        驻留全程
```

**字节口径统一按 bias dtype（fp16 → 2B/元素）计算**：L1 段与 BT 段都装 bias 原始数据；`DataCopyParams` 的 burst 按 32B 块显式折算（`nAlign·2/32`），与目的张量的模板类型无关。

## 2 装载步骤表

| # | 步骤 | 代码 | 说明 |
|---|------|------|------|
| 1 | 声明队列 | `TQue<TPosition::A1, 1> queBiasL1;`<br>`TQue<TPosition::C2, 1> queBt;` | L1 侧与 BT 侧各一条 |
| 2 | 初始化缓冲 | `pipe.InitBuffer(queBiasL1, 1, coAlign * sizeof(bias));`<br>`pipe.InitBuffer(queBt, 1, nBlk * sizeof(L0cT));` | BT 缓冲按 L0C 类型大小分配（保守覆盖两种读取口径），须 ≤ BT 总容量（§5） |
| 3 | GM→L1（全量一次） | `DataCopy(bl, biasGm, DataCopyParams(1, coAlign * 2 / 32, 0, 0));` | host 侧已把 bias 零填充到 Co^align（32B 对齐） |
| 4 | 每个 N 块：L1→BT | `DataCopy(bt, biasL1v[coStart], DataCopyParams(1, AlignB(curN,16) * 2 / 32, 0, 0));` | 拷贝 `nAlign·2` 字节（bias dtype 口径）；bt 张量按 `float`（L0C 类型）声明仅为与 `Mmad<L0cT,…>` 的模板参数一致，不影响 DataCopy 的字节量 |
| 5 | Mmad 首步读 BT | Mmad 参数：`cmatrixInitVal=false, cmatrixSource=true` | L0C 初值 = BT 内容（即 bias） |
| 6 | 释放 | 本输出块**全部 Mmad 发出后**再 `FreeTensor(btv)` | 过早释放会与后续 Mmad 竞争 |

> ⚠️ **BT 读取口径须核实**：上表按"BT 以 bias dtype（16bit）存储、Mmad 读 BT 时转换到累加类型"的口径书写（与参考实现一致）。若所用环境要求 BT 内容与 L0C 类型（fp32）逐元素一致，则步骤 4 须先做 half→fp32 扩张（拷 `nAlign·4` 字节）。两种口径不可混用；落地前按所用头文件定义或汇编验证确定，并在产出中声明假设。

## 3 Mmad 首步参数对照

| 场景 | cmatrixInitVal | cmatrixSource | L0C 初值来源 |
|------|---------------|---------------|-------------|
| 带 bias | `false` | `true` | BT（装载的 bias 段） |
| 无 bias | `true` | `false` | 零初始化 |

## 4 与真实算子输入的适配（越界读风险）

上述序列假设 bias 在 GM 中已被补齐到 `Co^align`（host 侧预零填充为既定契约）。真实算子的 bias 输入只有 Co 个元素时，按 `coAlign * 2 / 32` 个 32B 块整块搬运会**越界读 GM**。

| 适配写法 | 做法 | 优缺点 |
|---------|------|--------|
| **推荐：集成层预填充** | host / 调度侧保证 bias 缓冲区按 Co^align 预零填充 | 最简单、无额外 kernel 开销；写入算子集成说明即可 |
| kernel 内自适应 | 主体段（32B 对齐部分）普通搬运，尾段改 DataCopyPad 类（先零填目的、再按真值长度搬、余量由补齐参数置零） | 适应任意输入，但增加 kernel 复杂度 |

## 5 BT 容量约束

| 项 | 公式 | 说明 |
|----|------|------|
| 缓冲分配上限 | `nBlk × sizeof(L0cT) ≤ BT总容量(4KB)` | 即 `nBlk ≤ 4096/4 = 1024`；参考 tiling 已落地该上限（`BT_MAX_NBLK`，见 example README"关键正确性约束"） |
| 单段装载量 | `nAlign × 2B`（bias dtype 口径） | 恒 ≤ 上述分配 |
| 超限处理 | tiling 侧把 nBlk 封顶在 1024，超出部分由 kernel 的 nBlk 循环**分段装载**（每段一次 BT 装载） | 装载次数增加但功能不变 |
