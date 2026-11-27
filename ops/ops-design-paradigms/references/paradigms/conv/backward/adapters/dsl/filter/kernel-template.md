# dW：DSL Kernel 组织

本页将 [通用计算流](../../../generic/filter/compute-flow.md) 映射为设备函数。接口细节见 [instruction-mapping.md](instruction-mapping.md)，参数分类见 [tiling.md](tiling.md)。主线仍是 MN 输出任务、核内完整归约、分段原子写回。

## 1 资源与职责

| 对象 | 保存什么 | DSL 组织 |
|---|---|---|
| L1A | dedy 父段 | `RingBuffer`；多个 L0 K 块共用一个读出的父视图 |
| L1B | x 父行带 | 独立 `RingBuffer`；同时保留该槽的 H/W、局部 pad 和通道起点 |
| L0A / L0B | 当前计算的两路操作数 | 按最终计划声明 `RingBuffer` 或单槽 `Buffer` |
| L0C | 一个输出子块在当前归约段的部分和 | 单槽 `Buffer` 或按计划声明的多槽；整段沿用同一视图 |
| dedw GM | 已初始化为零的 FP32 输出 | 当前任务的有效 co/ci/kd/kh/kw 区域 |

RingBuffer 创建不代表预取已经发生。阶段函数既要选择正确槽，也要把下一父段搬入放在当前父段计算之前可并行的位置；父段只在它的最后一次 Load2D/3D 消费结束后才能被覆盖。

## 2 循环骨架

下面是**组织示意，不是可执行 Python**。`decode_task`、`reduce_segments`、`load_parent` 等是生成工程需要实现的函数，均不是 DSL API。顺序及公式从 generic 承接；示意省略双缓冲首尾轮。

```text
AIV 分片清零 dedw：MTE3 mode 0 汇合，再发 mode 2 通知
AIC 公共入口：FIXPIPE mode 2 等待一次，约束后续首次写回

for task = block_id; task < total_tasks; task += launched_aic:
    解码 group、kd、任务 M/N 范围
    for 每个 M/N 输出子块:
        for 每个既定归约段:
            取得本段的 L0C 槽
            普通窗口：first = true
            拆核窗口：按既定协议初始化整个输出块及各 tap 的初值状态
            for 本段 batch × dout:
                计算 id = od*strideD - padFront + kd*dilationD
                深度无交集时跳过，不读取非法 GM 地址
                for 本段 H 行带和宽片:
                    按当前子块和边界建立 A/B 父段
                    需要换父段时发射 GM→L1；复用时沿用原父视图
                    for 本父段内的有效 K 块:
                        由同一批深度/空间范围配对 A 与 B
                        A：L1→L0A 的转置加载
                        B：普通 Load3D，或按原 kh/kw 的单位窗口加载
                        MMAD(init=本输出子区间首次实际贡献)
            已进入计算但没有有效 MMAD：按零操作数协议建立零 L0C
            按输出格式和 N 核区间构造写回视图
            Fixpipe FP32 atomic add → dedw
            等本槽最后一次 Fixpipe 消费完成后才能复用
```

不存在归约贡献的输出仍为零。只有纯粹跳过整个任务、且输出清零已完成的路径可以不写；不能把一个未初始化 L0C 槽交给 Fixpipe。若拆核按 tap 写不同 L0C 子区间，首段标志也按子区间维护。

## 3 普通窗口与拆核的分界

| 阶段 | 普通窗口 | `isSplitKernelHW=true` |
|---|---|---|
| x GM 采样 | 连续有效行带，Load3D 解释原窗口 | kh/kw 外置；GM 地址使用原 stride/dilation/pad 采样 |
| Load3D | 原 H/W kernel、stride、dilation | 单位窗口、单位 stride/dilation、N=16 |
| MMAD 目标 | 当前完整 N 子块 | 原 kh/kw 对应的 L0C 列分形 |
| 输出恢复 | 整窗或非整窗状态 | 同一权重布局，保留原 kernel 坐标与实际 L0C M pitch |

特别检查拆核 M 尾块：如果计算用 `baseM=144` 作为相邻 N0 分形间距，当前有效 M=113，输出源 pitch 仍是 144，不能改成 `alignUp(113,16)=128`。

## 4 生成结果检查

1. 看 Host 与 kernel 的实际参数：两次不同空间/通道 shape 是否真的共用同一模板；`usedCoreNum`、任务步进与同步参与者是否一致。
2. 看每个父段的 GM→L1：H 批轴是否落到矩阵数、源步幅是否仍含完整通道/深度，是否出现不必要的逐行 DMA。
3. 看 L1→L0：Load2D 转置和 Load3D 窗口是否对应同一 K，尾长是否只缩小有效范围而保留父 pitch。
4. 看每个归约段：首个 MMAD 初始化一次，后续累加；无效深度跳过后不遗留旧 L0C。
5. 看 FIX：三种输出方向、批矩阵数、源物理跨度、尾通道和原子范围；非整窗状态在 M 切换后是否重放。
6. 看同步：局部依赖由编译器生成的事件覆盖；初始化涉及全部写者，末轮空核不跳过必要汇合。

通过编译只证明接口可接受。完整精度和执行覆盖按 [verification](../../../generic/filter/verification.md)；比较性能时统计完整融合 kernel，包含 AIV 清零与初始化同步。

## 5 Fixpipe 异常的最小定位顺序

将初始化、计算、写回分开核对，避免把测试数据或地址错误归因于原子指令：

1. **先验证清零。** 使用非零预填，只执行清零，检查整个有效区间为零及区间外哨兵；核对 UB 已初始化长度与 DMA 读取长度。向量写入范围见 [initialization.md](../initialization.md)。
2. **再验证写回语义。** 用能区分 co/ci/tap 的小整数构造已知 L0C 结果，分别测试普通覆盖写、非零预填上的原子加、连续多次原子加。覆盖写的 golden 为本次结果，原子加才叠加初值。直接写 NZ 目标时按实际 GM 物理布局解码，不能按 dense 展平值比较。
3. **核对探针自身。** 每种模式的输入分配都覆盖 kernel 实际读取范围；例如读取 `tap_count*16` 个元素的分支，不能只分配 `ci_use` 个元素。新增模式时同时更新输入生成、golden 和布局解释。使用 `ci_use>1`、多个 tap 和尾块，暴露单通道测试掩盖的步幅错误。
4. **最后比较生成指令。** 逐项核对源/目的地址、物理 pitch、矩阵数及步进、转换方向、原子开关和同步。重复运行先保存独立的输出快照，从第二次开始比较；第一次没有前次结果，不能计为不稳定。`.so` 哈希不同本身不证明指令调度有误，须检查实际代码差异。

只有在初始化、地址映射和探针均成立后，剩余可复现差异才作为框架问题记录；保留版本、最小代码和对应生成指令，不将尚未排除用法错误的现象固化为 shape 或搬运长度限制。
