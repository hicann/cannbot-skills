---
type: CATLASS DSL Optimization Guide
title: 整调度 makespan 空转判定与二维再切分
description: 总块数略超核数时 ceil(块数/核数) 使整程耗时成倍；先用零成本算术与边界形状实验判定，再把工作单元切二维并在整除因子里选组数。
tags: [catlass-dsl, optimization, multicore, load-balance, makespan, scheduling]
status: stable
generated: {by: process:catlass-dsl-pattern-capture, at: '2026-10-27T20:40:00+08:00'}
verified:
  - {by: process:catlass-dsl-kernel-run-audit, at: '2026-10-27T20:40:00+08:00'}
sources:
  - id: exp
    resource: project-evidence#kernel-sha256=6fb120e821de2df20bca6b7b193848d6e61bb77bec69bf4d52fc0b542bcf931b
    title: A5 单融合 GEMM + epilogue kernel makespan 消除实测（24.85→15.35 ms）
operator_families: [matmul, mixed]
arch: [c310, ascend-950pr]
---

# 接口与概念

按核数步进的工作单元分配（`block_range` 类循环）使每核承担 `ceil` 序列的块，总时长由
最忙核决定：`makespan = ceil(总块数 / 核数)` 个块，而不是平均值。[^exp]

与「多核尾轮再平衡」是空转规模不同、候选相通的两种形态：总块数远大于核数时只有末轮
空闲；**总块数只是略超核数**（如 32 块 / 28 核）时整个调度只有 2 轮，43%–48% 算力
空转（32 块：24/56 ≈ 43%；29 块：27/56 ≈ 48%），空转集中在少数多拿一块的核上。
两种形态按序评估同一组候选：**尾轮再切**（只对溢出轮的块沿第二维再切，本案例 4 块 × 7
组 = 28 子块铺满 28 核，总代价 ≈ 558 tile ≈ 14.3 ms，跨组归并只涉及 4 个第一维块的行）
与**全量二维再切**（统一 (第一维块 × 第二维段) 工作单元，本案例 g=5 → 594 tile ≈ 15.2 ms，
全部行需归并但工作单元同构）。尾轮再切 tile 数与归并面更优，但要求异构工作单元
（子块重算 GM 起点与 L0C layout）；全量再切 tile 略多但实现简单——实测案例采用
后者（15.35 ms）。

判定信号（满足任一即高度可疑，三者齐备即坐实）：

1. **算术吻合**：实测耗时 ≈ makespan × 单块理想耗时。实测案例：每块 485 tile ×
   25.6 µs ≈ 12.4 ms，×2 ≈ 24.8 ms ≈ 实测 24.85 ms。[^exp]
2. **台阶跳变**：形状跨过"块数 = 核数"边界时耗时成倍跳变。实测 28 块（每核
   恰 1 块）12.42 ms、29 块 24.80 ms、32 块 24.85 ms——多做 14%
   的活慢一倍。[^exp]
3. **单块已达上限**：最忙核耗时 ≈ 同规模理论值（案例中 cube 12.42 ms ≈ torch.mm
   理论 12.14 ms），说明数据通路没有剩余空间。

profiling 计数器不可用时（msprof `aic_*_ratio` 全 0）本条仍可执行：判定只需要算术和
一次运行，是**无 profiler 环境的首选取证路径**；数据通路排查改用单变量消融计时实验。

# 用法

- **第 0 步（先于任何实验）**：列出四元组（总块数、核数、`ceil`、单块理想耗时），
  与实测相除。单块理想耗时必须有依据（tile 数 × 实测每 tile 时间，或同形状 GEMM
  理论值），不得拍脑袋。
- **判定实验**：构造块数恰等于核数的形状，**先写预测再跑**（预测 = 单块理想耗时）；
  耗时减半即坐实铺放空转。成本一次运行，通常两分钟内。
- **修法（按序评估候选）**：候选 A 尾轮再切——只对 `ceil` 溢出轮的块沿第二维再切成约
  `cores / 溢出块数` 份，子块铺满空转核（案例 ≈ 558 tile ≈ 14.3 ms，归并面最小，
  但要求异构工作单元）；候选 B 全量二维再切——工作单元从一维（第一维块）扩成二维
  （第一维块 × 第二维段），host 在第二维总块数的**因子**里枚举组数按代价模型选最优
  （案例 g=5 → 594 tile ≈ 15.2 ms，工作单元同构、kernel 免运行期除法，实现更简单，
  实测案例采用）。

# 代码模式

```python
def _choose_groups(num_first_blocks, num_second_blocks, cores, setup_tiles=2):
    best_g, best_tiles = 1, None
    for g in range(1, min(num_second_blocks, 128) + 1):
        if num_second_blocks % g:      # 因子保证整除，kernel 免运行期除法
            continue
        rounds = -(-num_first_blocks * g // cores)                    # ceil
        tiles = rounds * (num_second_blocks // g + setup_tiles)       # setup 必须计入
        if best_tiles is None or tiles < best_tiles:
            best_g, best_tiles = g, tiles
    return best_g
```

跨组的部分统计量落 GM，host 一次归并。下面是**带 max/sum 归一化统计的 epilogue 示例**
（纯 GEMM epilogue 只需按组累加，无重归一化）：

```python
m = pmax.amax(1)                                        # [M_pad]
s = (psum * torch.exp(pmax - m[:, None])).sum(1)        # [M_pad]
```

# 约束

- **适用范围**：GEMM + epilogue 融合类（matmul/mixed）；实测 Ascend950PR（CANN
  9.2.0-beta.2）；触发前提：总块数 ≥ 核数、`ceil ≥ 2` 且单块理想耗时有可靠依据；
  第二维存在可用因子切分；每块固定开销占比不高（setup 占比高时重估候选 A/B 的
  tile 数与归并面再选）。
- **保持条件**：每个输出元素被工作单元恰好覆盖一次；跨组归并改变归一化统计的
  **累加顺序与中间 dtype**——必须用原算子的 dtype/容差合同判断，不能默认 bitwise
  相同（`G == 1` 才与旧路径逐位等价，`g > 1` 按容差比对）；部分统计量须待 kernel
  完成后由 host 归并，不得提前读。
- **资源代价**：组数必须整除第二维块数，否则 kernel 引入运行期除法/边界判断；每块
  固定开销（如索引预处理、建链）随组数线性重复，**必须折算进选择器目标函数**——案例中
  建链约 47 µs/块 ≈ 2 tile，不建模会选最细切分 g=97 得 19.1 ms，计入后选 g=5 得
  14.9 ms；[^exp] 部分统计量落 GM 增加 `M × G` 写入与一次 host 归并（案例
  ~10 µs，量级随形状重估）；`G == 1` 保持零开销退化路径（直接拷贝，不引入额外
  kernel）。
- **可证伪预期**：候选的 makespan tile 数可先验计算（候选 B：
  `ceil(总块数 × g / 核数) × (nb/g + setup)`；候选 A：整轮 tile + 溢出轮子块
  tile），实测耗时按 tile 比例下降；再切后实测不降即为数据通路存在第二瓶颈，
  回到单变量消融取证。

# 失败表现

- 切分后数值错：跨组归并的重归一化写错，或尾块 pad 列防护写错组。
- 切分后更慢：固定开销未计入选择器，或组数选成非因子引入运行期除法。
- 耗时未按预期下降：铺放之外数据通路仍有第二瓶颈，回到单变量消融取证。

# 验证方法

- 选择器对 `(num_first_blocks, num_second_blocks, cores)` 三元组做 host 单测，覆盖整除、
  余 1、余 cores-1、块数 < 核数等形态。
- 相邻形状扫描（块数 = 核数、核数 + 1、2×核数）确认台阶消失、耗时随工作量平滑变化。
- 归并正确性：`G == 1` 与旧路径逐位等价；`g > 1` 按原算子 dtype/容差合同比对
  （跨组重归一化改变累加顺序，不得要求 bitwise 相同）；极端组（某组无有效输入、
  全 pad 列）单独 case 验证。

[^exp]: 项目实测（Ascend950PR，CANN 9.2.0-beta.2，2026-10-27）：
  单融合 GEMM + epilogue kernel，makespan 判定后二维再切分，
  24.85 ms → 15.35 ms（torch 参考 40.51 ms），6 组精度场景全 PASS；
  知识条目绑定最终 kernel SHA-256（见 sources）。
