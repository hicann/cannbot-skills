# raw-VF 操作 Channel 与 NZ 格式 — 代码示例

§7 和 §8 规则对应的代码示例。所有示例自包含，不引用外部文件。

## 1. raw-VF softmax 完整模式

展示 raw-VF 直接读写 Channel 缓冲区的完整 softmax 链路：`vload_deinterleave` 读 QK fp32 → scale + rowmax + exp → `vcast` + `vor` + `vstore_strided` 写 P fp16 NZ。

> **操作数序陷阱**：下面的代码里 `vmuls(ve, SCALE)` / `vsub(ve, rmb)` / `vexp(...)` 等的参数顺序看起来"自然"，但 raw-VF 的操作数序与签名不可观测、源码无语义注释。`vmadd(acc, lhs, rhs)` 实测是 `acc*lhs + rhs`（不是 `acc + lhs*rhs`）；`vexp_sub(a, b)` 是 `exp(a-b)`（不是 `exp(b-a)`）。在复用本节代码前，先用三个互不对称的值写探针钉死每个 op 的语义 —— 详见 `../SKILL.md` 陷阱 10。

```python
from cannbotdsl.jit_runner import jit
from cannbotdsl import dtypes
from cannbotdsl.vf import vf
from cannbotdsl.raw_reg import (full_mask, vload_deinterleave, vmuls, vsub, vexp,
                              vreduce_max, vreduce_sum, vmax, vadd, vdup_lane0,
                              vcast, vor, vstore_strided, vstore, vstore_first,
                              vmem_bar, RegLayout)

SCALE = 1.0 / (128 ** 0.5)
VL = 64   # fp32 向量长度

@jit
def softmax_init(qk_ch, p_nz_ub, m_ub, l_ub):
    """首迭代: m=rowmax, l=rowsum, P=exp(scale*S - m).

    qk_ch:   Channel (UB, fp32, ND) — Cube 侧 fixpipe 写入
    p_nz_ub: Channel (UB, fp16, NZ) — raw-VF vstore_strided 写入
    m_ub:    Buffer(UB, fp32, (64, 8)) — running row max
    l_ub:    Buffer(UB, fp32, (64, 8)) — running row sum
    """
    with vf(mode='raw'):
        m32 = full_mask(elem_bits=32)
        m16 = full_mask(elem_bits=16)
        src_stride = qk_ch.physical_stride[0]   # ND 行 stride

        # NZ stride 动态推导
        s = p_nz_ub.physical_stride               # (s_n1, s_m1, s_m0, s_n0)
        s_n1, s_m1, s_m0, s_n0 = s[0], s[1], s[2], s[3]
        m0 = s_m1 // s_m0        # 每组行数 (通常 16)
        n0 = s_m0 // s_n0        # 每元素宽度 (fp16: 16)
        block_stride = s_n1 // n0

        for row in range(64):
            base = row * src_stride
            # 一次加载 128 fp32，拆成 even/odd 两个 64-element 向量
            ve, vo = vload_deinterleave(qk_ch, base, width="b32")
            # scale
            ve = vmuls(ve, SCALE, mask=m32)
            vo = vmuls(vo, SCALE, mask=m32)
            # rowmax
            rmax = vmax(vreduce_max(ve, mask=m32),
                        vreduce_max(vo, mask=m32), mask=m32)
            rmb = vdup_lane0(rmax, mask=m32)       # 替代 expand 广播
            # P = exp(S - rmax)
            pe = vexp(vsub(ve, rmb, mask=m32), mask=m32)
            po = vexp(vsub(vo, rmb, mask=m32), mask=m32)
            # rowsum
            rsum = vadd(vreduce_sum(pe, mask=m32),
                        vreduce_sum(po, mask=m32), mask=m32)
            # 存储 m, l (标量, 用 vstore_first 只写 lane0)
            vstore(m_ub, row * 8, rmb, m32)
            vstore(l_ub, row * 8, vdup_lane0(rsum, mask=m32), m32)
            # fp32 → fp16: even/odd cast + merge
            he = vcast(pe, dtypes.float16, mask=m32, reg_layout=RegLayout.ZERO)
            ho = vcast(po, dtypes.float16, mask=m32, reg_layout=RegLayout.ONE)
            merged = vor(he, ho, mask=m16)
            # 写入 NZ 格式 UB Channel
            nz_off = (row // m0) * s_m1 + (row % m0) * s_m0
            vstore_strided(p_nz_ub, nz_off, merged, m16,
                           block_stride=block_stride, repeat_stride=0)
        vmem_bar('vst_vld')
```

## 2. NZ stride 动态推导

硬编码 NZ stride 在 `n1_pad≠0` 时错误。必须从 `channel.physical_stride` 动态读取。

```python
# NZ 格式 (128, 128) fp16, n1_pad=16
# physical_stride 返回 (s_n1, s_m1, s_m0, s_n0) — 元素单位

p_nz_ub = Channel(MemLoc.UB, (128, 128), dtypes.float16, depth=1,
                  data_format="nz", n1_pad=16)

s = p_nz_ub.physical_stride
s_n1, s_m1, s_m0, s_n0 = s[0], s[1], s[2], s[3]

# 推导 NZ layout 参数
m0 = s_m1 // s_m0        # 每组行数: 256 // 16 = 16
n0 = s_m0 // s_n0        # 每元素宽度: 16 // 1 = 16
block_stride = s_n1 // n0  # vstore_strided 的 block_stride 参数

# vstore_strided 的 offset 计算
# 对于第 row 行:
nz_off = (row // m0) * s_m1 + (row % m0) * s_m0

# 错误写法（硬编码，n1_pad=16 时 stride 不对）:
# block_stride = 128   # ← 如果 n1_pad=16, 实际应为 129
```

## 3. UB(NZ) → L1(NZ) 直接拷贝

raw-VF 写入 UB NZ Channel 后，`mem_copy` 将数据从 UB(NZ) 拷贝到 L1(NZ)，无需 engine。

```python
# Vec 侧: raw-VF 写入 UB NZ Channel
p_nz_ub = Channel(MemLoc.UB, (128, 128), dtypes.float16, depth=1,
                  data_format="nz", n1_pad=16)

# ... softmax_init/softmax_step 中 vstore_strided 写入 p_nz_ub ...

# P handoff: UB(NZ) → L1(NZ), 无 engine
p_l1 = Channel(MemLoc.L1, (128, 128), dtypes.float16, depth=1,
               data_format="nz", n1_pad=16)       # ← 需与 p_nz_ub 相同的 nz/n1_pad

mem_copy(p_l1, p_nz_ub)                           # ← NZ→NZ 直接拷贝, 无需 nd2nz engine

# Cube 侧: 从 L1 NZ Channel 读取 P 做 matmul
mem_copy(l0a, p_l1)
mem_copy(l0b, v_l1, transpose=True)
matmul(l0c, l0a, l0b)
```

## 4. pre-loop 建立 channel producer

首迭代在循环外执行，建立 channel 的首个 producer，后续循环内可安全做 in-place 或 rescale。

```python
# Pre-loop: 首迭代（n_idx=0）在循环外
n_idx = 0
kt_tile = tile_view(kt_gm, (128, 128), (ki, n_idx))
mem_copy(b_l1, kt_tile, engine=nd2nz)
mem_copy(l0a, a_l1)
mem_copy(l0b, b_l1, transpose=True)
matmul(l0c, l0a, l0b)
mem_copy(qk_ub, l0c, engine=fixpipe, partition=split_m)   # ← qk_ub 首次 producer

# raw-VF softmax init (读 qk_ub, 写 p_nz_ub/m_ub/l_ub)
softmax_init(qk_ub, p_nz_ub, m_ub, l_ub)

# P handoff
mem_copy(partition_view(p_l1, split_m, subblock_idx), p_nz_ub)

# PV matmul
mem_copy(b_l1, v_tile, engine=nd2nz)
mem_copy(l0a, p_l1)
mem_copy(l0b, b_l1, transpose=True)
matmul(l0c, l0a, l0b)
mem_copy(pv_ub, l0c, engine=fixpipe, partition=split_m)   # ← pv_ub 首次 producer

# O += PV (通过 muls 中转 Channel → Channel, 再 raw-VF add)
muls(pv_tmp, pv_ub, 1.0)                                   # ← pv_tmp 首次 producer
add_pv(pv_tmp, o_ub)                                        # ← o_ub 首次写入

# Main loop: 剩余迭代 — 所有 channel 已有 producer
for n_iter in range(1, num_n):
    # ... QK matmul → qk_ub (已有 producer, 可安全 rescale) ...
    softmax_step(qk_ub, p_nz_ub, m_ub, l_ub, alpha_ub)
    rescale_o(o_ub, alpha_ub)                               # ← o_ub 已有 producer
    # ... PV matmul → pv_ub ...
    muls(pv_tmp, pv_ub, 1.0)                                # ← pv_tmp 已有 producer
    add_pv(pv_tmp, o_ub)
```
