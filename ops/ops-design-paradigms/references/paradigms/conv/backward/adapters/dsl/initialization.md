# 融合 AIV 清零与 AIC 写回

在同一个 MIX kernel 中完成输出清零和卷积计算。dW 在首次原子写回前清零 dedw；dX 按既有 `initOutputFlag` 决定是否清零。输出类型和触发条件沿用 scene/generic，本页只说明 DSL 的实现与同步。

## 同步顺序

```text
全部 AIV：分片清零 GM → MTE3 mode 0 arrive/wait → MTE3 mode 2 arrive
全部 AIC：                                      FIXPIPE mode 2 wait → 首次写回
```

mode 0 先汇合全部 AIV，mode 2 再通知各自配对的 AIC。清零分片与卷积输出任务的归属未必相同，只通知本核对应的 AIC 不能证明其他区域已清零。AIC 在 FIXPIPE 流水等待；不为此额外阻塞 MTE2 或 MMAD，也不叠加 `global_sync_all()`。实际能重叠多少计算，仍须看生成代码和 profiling。

## DSL 接口片段

下面 helper 放入已有卷积程序类，使用连续输出的一维视图。dtype 和元素字节数是编译配置，长度取自本次 Tensor；非连续输出须按真实布局划分，不能直接展平。

```python
import cannbotdsl as cbd
from cannbotdsl import MemLoc, Tensor, make_layout

class FusedOutputInit:
    def __init__(self, dtype, element_bytes):
        self.dtype = dtype
        self.element_bytes = element_bytes  # FP32 为 4，FP16/BF16 为 2
        self.tile = 256 // element_bytes

    @cbd.jit
    def clear_and_notify(self, output: Tensor):
        zeros = cbd.Buffer(MemLoc.UB, self.dtype, make_layout((self.tile,)))
        with cbd.vf():
            mask = cbd.reg.create_mask("all", elem_bits=self.element_bytes * 8)
            cbd.reg.vstore(zeros, 0, cbd.reg.vdups(0., self.dtype), mask)

        worker = cbd.get_block_idx() * cbd.get_subblock_dim() + cbd.get_subblock_id()
        workers = cbd.get_block_num() * cbd.get_subblock_dim()
        size = output.shape[0]
        for start in range(worker * self.tile, size, workers * self.tile):
            count = min(self.tile, size - start)
            dst = cbd.from_tensor(output, layout=make_layout((count,)),
                                  offset=start * self.element_bytes)
            cbd.mem_copy(dst, zeros[:count])

        # 没有清零数据的 AIV 也参与；事件必须放在清零循环外。
        cbd.vec_sync_block_arrive(cbd.PIPE.MTE3, 8, mode=0)
        cbd.vec_sync_block_wait(cbd.PIPE.MTE3, 8, mode=0)
        cbd.vec_sync_block_arrive(cbd.PIPE.MTE3, 9, mode=2)
```

256B 是便于展示的一次向量写入，不是推荐的 GM 清零分片大小。上例已经复用同一片 UB 零数据，但每片 GM 仍由循环单独发起 copy；性能实现可进一步评估用 DMA 硬件循环批量重复读取该 UB 区域。保留默认自动同步来保证 VF 写零与 MTE3 读 UB 的依赖；GM 跨核依赖使用上面的显式事件。

### 初始化范围按 DMA 的实际 UB 读地址计算

950 上 `vdups` 产生一个向量，`vstore` 写入该向量的有效 lane；`create_mask("all", elem_bits=32)` 表示全部 64 个 FP32 lane，不是整个 UB Buffer。分配 8192 个 FP32 后只在偏移 0 执行一次 `vstore`，仅初始化前 64 个元素；随后从 UB 连续读取 8192 个元素会把未初始化内容写到 GM。自动同步只保证先写后读，不会补齐未写入的数据。

GM 清零长度不要求与 UB 零缓冲长度相等。优先评估“初始化固定大小 UB 零块，再重复搬出”的组织。例如用 64 个 FP32 的零块覆盖单个 worker 独占的连续 8192 个 FP32，地址语义如下（伪代码，不是已验证的 DSL 调用）：

```text
UB 只初始化 zeros[0:64]
重复 128 次：GM[base + i*64 : base + (i+1)*64] ← UB[0:64]
期望 DMA 硬件重复循环（loop1）：源起点步进 0，目的起点步进 256B
```

这里的连续区间须按 worker 重新划分，不能直接替换前面交错分片 helper 的单次 copy。若保留原交错分片，DMA 目的循环步进应为 `workers*零块字节数`，循环次数按该 worker 的完整有效片数计算，尾片单独处理；无数据的 worker 仍参与原同步协议。

源重复读取可减少 UB 占用和 VF 填零量；若合成硬件循环，还能减少逐块发射开销，但 GM 总写入字节数不变，零块大小仍需通过 profiling 选择。[ASC 3510 UB→GM 文档](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMD-API/c_api/vector_datamove/asc_copy_ub2gm/asc_copy_ub2gm_arch_3510.md) 的连续 Broadcast 方案为：`burst_count=1` 且 burst `src_stride` 非零；`loop1_size=重复次数`、`loop1_src_stride=0`、`loop1_dst_stride=零块字节数`；`loop2_size=1`，loop2 源/目的步进均为 0。结束后将两层循环次数复位为 `(1,1)`。该接口的 burst stride 表示相邻块起点间距，不能与其他接口的块间空隙混淆；文档不建议直接用 burst `src_stride=0` 实现 Broadcast。

DSL 能否将源零步幅视图直接降为上述硬件循环，须核对当前版本的生成代码及数值结果；本页末尾的实测结论只覆盖逐片 copy 示例。未验证该映射时可沿用示例复用 UB 零块；若选择从更大的 UB 区域连续搬出，才需要逐向量初始化那段完整源区域。两种方式都处理 GM 尾长，并保证 DMA 实际读取的 UB 地址全部已初始化。256B 不是 UB→GM 或 Fixpipe 原子写回的长度上限。

排查清零时先单独检查整个目标区间是否等于零，并检查区间外哨兵；只统计“还等于预填值”的元素会漏掉被未初始化 UB 内容覆盖的位置。原子写回异常随分配历史变化时，先核对这些写入范围，再检查同步和搬运指令。

主 kernel 按下面的位置接入。`compute_convolution` 代表已有卷积方法，不是 DSL API；该方法应包含真实 Cube 计算和 Fixpipe 写回。

```python
# 接口片段：置于已有卷积程序类内。
@cbd.kernel
def main(self, x, dy, output, output_flat, init_output):
    if init_output:
        self.clear_and_notify(output_flat)
        cbd.cube_sync_block_wait(cbd.PIPE.FIXPIPE, 9, mode=2)
    self.compute_convolution(x, dy, output)

@cbd.host
def run(self, x, dy, output, output_flat, init_output, blocks):
    self.main[blocks](x, dy, output, output_flat, init_output)
```

`output_flat` 与 `output` 必须指向同一连续存储；可在普通 Python 入口用 `output.view(-1)` 建立。`init_output` 是所有参与核一致的条件，dW 当前模板每次调用均清零，dX 沿用计划中的标志。关闭初始化时，发事件和等事件必须一起跳过。编译器依据 AIV/Cube 指令拆分 MIX 代码，无须让 AIC 执行 UB 操作。

## 集成约束与验证

默认调度下，本例按无其他竞争核资源任务的场景验证；多流并发的 mode 0 全核汇合仍受 [ASC 核间同步约束](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.html) 限制。

- FIXPIPE 等待放在全部 AIC 都经过的公共路径，每次调用执行一次，位于首次写回之前；不能在每个输出块内反复消费同一个通知。
- 没有计算任务的 AIC、没有清零分片的 AIV 仍完成上述协议，不提前返回。950 的 MIX 1:2 启动块数不超过有效 AIC 数及有效 AIV 数除以 2，不写死 32/36 核。
- 示例事件 8/9 需在完整 kernel 中预留，与其他手动或自动跨核事件不冲突。DSL 的 mode 0 等待也显式传 `mode=0`，不能依赖默认值。
- 性能统计整个融合 kernel，包含清零。检查生成代码中的 AIV MTE3 汇合/通知、AIC FIX 等待以及实际 MIX 核配比。

上述接口组合已在 950 编译并运行：用小矩阵计算验证 FP32 两次原子写回，以及 FP16/BF16 普通写回；覆盖初始化开关、任务少于启动核数、非对齐尾块和输出后哨兵。15 组配置各连续执行 3 次，输出预置非零值，结果与哨兵检查通过。示例采用默认调度；生成代码保留 `asc_sync_block_wait(PIPE_FIX, 9)`。此验证针对初始化和写回协议，不代表完整卷积的精度或性能验证。
