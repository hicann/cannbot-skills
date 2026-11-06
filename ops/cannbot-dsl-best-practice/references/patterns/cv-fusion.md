# CV 融合：四阶段函数与集中调度

以 `out = 0.5 * (fp16(2 * (A @ B.T)) @ W.T)` 展示一个 launch 内的 C1→V1→C2→V2。各阶段用 `@cb.jit` 函数封装，主入口集中分配 tile、错位调度并排空流水。`c/v` 表示 Cube/Vector，序号表示算法阶段，不是物理核编号；实际执行域由 DSL 操作决定。

示例基于 DSL 0.8.0：连续 NPU FP16 `A[rows,32]`、`B[64,32]`、`W[64,64]`，FP32 `out[rows,64]`，`rows` 为 32 的正整数倍。仅处理完整 tile；调用方负责检查形状、类型、设备、连续性和合法 launch 核数。它是组织示例，不是所有 CV 算子的固定流水或最优分块。

```python
# 计算 out = 0.5 * (fp16(2 * (A @ B.T)) @ W.T)，两次矩阵乘均以 FP32 累加。
# C1: A @ B.T；V1: 乘 2 并转 FP16；C2: 与 W.T 相乘；V2: 乘 0.5 并写回。
# 四个阶段分别封装，主入口集中调度不同 tile，并排空流水。
import cannbotdsl as cb
from cannbotdsl import reg
from cannbotdsl.ops.reg import PackMode, RegLayout, RoundingMode
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id

M, N, K = 32, 64, 32

@cb.kernel
class Pipeline:
    def __init__(self, delay):
        self.delay = delay
        self.a1 = cb.RingBuffer(cb.MemLoc.L1, cb.dtypes.float16, cb.make_layout((M,K)), depth=1)
        self.b1 = cb.RingBuffer(cb.MemLoc.L1, cb.dtypes.float16, cb.make_layout((N,K)), depth=1)
        self.a0 = cb.RingBuffer(cb.MemLoc.L0A, cb.dtypes.float16, cb.make_layout((M,K)), depth=1)
        self.b0 = cb.RingBuffer(cb.MemLoc.L0B, cb.dtypes.float16, cb.make_layout((N,K)), depth=1)
        self.acc1 = cb.RingBuffer(cb.MemLoc.L0C, cb.dtypes.float32, cb.make_layout((M,N)), depth=1)
        self.cv1 = cb.RingBuffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((M//2,N)), depth=delay+1, data_format="nz", physical_layout=cb.make_layout((4,1,16,16)))
        self.half = cb.Buffer(cb.MemLoc.UB, cb.dtypes.float16, cb.make_layout((M//2,N)), data_format="nz")
        self.vc = cb.RingBuffer(cb.MemLoc.L1, cb.dtypes.float16, cb.make_layout((M,N)), depth=delay+1, data_format="nz")
        self.w1 = cb.RingBuffer(cb.MemLoc.L1, cb.dtypes.float16, cb.make_layout((N,N)), depth=1)
        self.x0 = cb.RingBuffer(cb.MemLoc.L0A, cb.dtypes.float16, cb.make_layout((M,N)), depth=1)
        self.w0 = cb.RingBuffer(cb.MemLoc.L0B, cb.dtypes.float16, cb.make_layout((N,N)), depth=1)
        self.acc2 = cb.RingBuffer(cb.MemLoc.L0C, cb.dtypes.float32, cb.make_layout((M,N)), depth=1)
        self.cv2 = cb.RingBuffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((M//2,N)), depth=delay+1)

    @cb.jit
    def c1_matmul_ab(self, a, b, tile):
        av=self.a1.next_write(); bv=self.b1.next_write()
        cb.mem_copy(av, cb.tile_slice(a,(M,K),(tile,0))); cb.mem_copy(bv,b)
        al=self.a0.next_write(); bl=self.b0.next_write()
        cb.mem_copy(al,av); cb.mem_copy(bl,bv)
        c=cb.matmul(self.acc1.next_write(),al,bl,init=True)
        cb.mem_copy(self.cv1.next_write(),c.partition(dim=0), options=cb.copy_options(channel_split=False))

    @cb.jit
    def v1_scale_cast(self):
        x=self.cv1.next_read()
        with cb.vf(mode='simd'):
            mask=reg.create_mask('all',32)
            for row in cb.range(M//2):
                v=reg.vmuls(reg.vload(x,row*N),2.0,mask=mask)
                h=reg.vcast(v,cb.dtypes.float16,mask=mask,rounding=RoundingMode.RN,reg_layout=RegLayout.ZERO)
                reg.vstore_pack(self.half,row*N,h,mask,pack_mode=PackMode.B32_TO_B16)
        target=self.vc.next_write()
        cb.mem_copy(target.partition(dim=0,part_id=get_subblock_id()),self.half)

    @cb.jit
    def c2_matmul_weight(self, w):
        x=self.vc.next_read(); weight=self.w1.next_write()
        cb.mem_copy(weight,w)
        xl=self.x0.next_write(); wl=self.w0.next_write()
        cb.mem_copy(xl,x); cb.mem_copy(wl,weight)
        c=cb.matmul(self.acc2.next_write(),xl,wl,init=True)
        cb.mem_copy(self.cv2.next_write(),c.partition(dim=0))

    @cb.jit
    def v2_scale_store(self,out,tile):
        x=self.cv2.next_read()
        with cb.vf(mode='simd'):
            mask=reg.create_mask('all',32)
            for row in cb.range(M//2):
                v=reg.vmuls(reg.vload(x,row*N),0.5,mask=mask)
                reg.vstore(x,row*N,v,mask)
        cb.mem_copy(cb.tile_slice(out,(M//2,N),(tile*2+get_subblock_id(),0)),x)

    def __call__(self,a,b,w,out):
        block=get_block_idx(); blocks=get_block_num()
        count=max(0,(a.shape[0]//M-block+blocks-1)//blocks)
        for tick in cb.range(count+3*self.delay):
            if tick<count:
                self.c1_matmul_ab(a,b,block+tick*blocks)
            if tick>=self.delay and tick-self.delay<count:
                self.v1_scale_cast()
            if tick>=2*self.delay and tick-2*self.delay<count:
                self.c2_matmul_weight(w)
            if tick>=3*self.delay and tick-3*self.delay<count:
                self.v2_scale_store(out,block+(tick-3*self.delay)*blocks)

@cb.host
def run0(a,b,w,out,blocks:int):
    op=Pipeline(0)
    op[blocks](a,b,w,out)

@cb.host
def run1(a,b,w,out,blocks:int):
    op=Pipeline(1)
    op[blocks](a,b,w,out)

@cb.host
def run2(a,b,w,out,blocks:int):
    op=Pipeline(2)
    op[blocks](a,b,w,out)
```

## 交接与调度

- C1→V1 保留 FP32 NZ 的 C0=16 L0C mirror；V1 按物理连续顺序缩放、转 FP16，保持 NZ 顺序，两个 AIV 各写 L1 的一半 M，再由 C2 消费。中间结果不落 GM。这里 V1 是逐元素运算；行归约等操作需按实际 NZ 布局重新设计索引。
- 主入口在 tick t 调度 C1(t)、V1(t-d)、C2(t-2d)、V2(t-3d)，追加 3d 拍排空。阶段交接 RingBuffer 深度为 d+1，其他暂存深度为 1；调整阶段和调度时按数据存活区间重新确定容量。示例使用 DSL 自动同步，选槽本身不等待。
- `run0/run1/run2` 分别展示延迟 0/1/2 拍。调用方用 `cb.compile` 编译对应 host，传入 A、B、W、out 和 blocks；在输入所属设备上下文中编译和运行，等待执行完成后再释放程序句柄。工作流的公共入口及导出仍遵循其既有契约。

0.8.0 NPU 实测覆盖 rows=32/192、blocks=1/3/8、延迟 0/1/2 拍及两组随机输入，36/36 通过；这验证了交接、复用和数值正确性，不代表性能收益。同步与布局见 [同步](synchronization.md)、[布局与尾块](layout-tail.md)。
