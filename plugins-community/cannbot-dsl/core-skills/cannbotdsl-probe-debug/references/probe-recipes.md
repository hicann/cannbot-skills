# 探针配方

可复制的最小探针骨架。每个都在 dav-3510 上实测跑通过。

## 目录

- [1. 单核 cube 探针（验证 matmul 形状/语义）](#1-单核-cube-探针)
- [2. 跨核 cube→UB→GM 探针](#2-跨核-cubeubgm-探针)
- [2.1. 探针假 PASS 的两种方式](#21-探针假-pass-的两种方式)
- [3. 参数扫描骨架](#3-参数扫描骨架)
- [4. 误差形态定位脚本](#4-误差形态定位)
- [5. 数值路径模拟器（精度归因）](#5-数值路径模拟器)
- [6. 单变量 A/B 探针](#6-单变量-ab-探针)

---

## 1. 单核 cube 探针

验证一个 matmul 的形状、操作数方向、累加语义。`op[1]` 单核，输入直接构造成 `[1,1,M,K]`
避免 layout 干扰。

```python
"""PROBE Pn — <一句话说明验证什么，以及为什么怀疑它>"""
import torch
import torch_npu  # noqa: F401

from cannbotdsl.arch import get_subblock_id
from cannbotdsl.channel import Channel
from cannbotdsl import dtypes
from cannbotdsl.jit_runner import jit
from cannbotdsl.kernel_launcher import kernel
from cannbotdsl.math import matmul
from cannbotdsl.runtime import from_torch_npu
from cannbotdsl.tensor import (local_slice, make_copy_engine,
                               make_partition_tiler, mem_copy,
                               partition_view, tile_view)
from cannbotdsl.typing.types import MemLoc, Tensor

M, N, K = 64, 128, 128


@kernel
class probe_kernel:
    def __init__(self):
        self.nd2nz = make_copy_engine(format_transform="nd2nz",
                                      dtype=dtypes.float16, pad_value=0.0)
        self.fixpipe = make_copy_engine(dtype=dtypes.float32, dual_dst_ctl=1)
        # L0A/L0B 上限各 64KB：(128,128)f16 = 32K，depth=2 就打满
        self.a_l1 = Channel(MemLoc.L1, shape=(M, K), dtype=dtypes.float16, depth=2)
        self.b_l1 = Channel(MemLoc.L1, shape=(N, K), dtype=dtypes.float16, depth=2)
        self.l0a = Channel(MemLoc.L0A, shape=(M, K), dtype=dtypes.float16, depth=2)
        self.l0b = Channel(MemLoc.L0B, shape=(K, N), dtype=dtypes.float16, depth=1)
        self.l0c = Channel(MemLoc.L0C, shape=(M, N), dtype=dtypes.float32, depth=1)
        self.ub = Channel(MemLoc.UB, shape=(M // 2, N), dtype=dtypes.float32,
                          depth=1)
        self.subblock_idx = get_subblock_id()

    def __call__(self, out: Tensor, a: Tensor, b: Tensor):
        a_s, b_s = a[0, 0, None, None], b[0, 0, None, None]
        # coord 是 TILE 索引，不是元素偏移
        mem_copy(self.a_l1, tile_view(a_s, (M, K), (0, 0)), engine=self.nd2nz)
        mem_copy(self.b_l1, tile_view(b_s, (N, K), (0, 0)), engine=self.nd2nz)
        mem_copy(self.l0a, self.a_l1)
        mem_copy(self.l0b, self.b_l1)          # 需要 B^T 打包时加 transpose=True
        matmul(self.l0c, self.l0a, self.l0b, init=True)

        split = make_partition_tiler((M, N), (M, N))
        mem_copy(self.ub, self.l0c, engine=self.fixpipe, partition=split)
        ot = tile_view(out[0, 0, None, None], (M, N), (0, 0))
        half = partition_view(ot, split, self.subblock_idx)
        mem_copy(half, self.ub)


@jit
def launch(out: Tensor, a: Tensor, b: Tensor):
    probe_kernel()[1](out, a, b)


def main():
    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K, dtype=torch.float16)
    b = torch.randn(1, 1, N, K, dtype=torch.float16)
    out = torch.zeros(1, 1, M, N, dtype=torch.float32).npu()
    launch(from_torch_npu(out), from_torch_npu(a.npu()), from_torch_npu(b.npu()))
    torch.npu.synchronize()

    ref = a.float() @ b.float().transpose(-2, -1)
    got = out.cpu()
    rel = ((got - ref).abs() / (ref.abs() + 1e-7)).mean().item()
    print(f"PROBE: MERE={rel:.4e} {'PASS' if rel < 1e-2 else 'FAIL'}")
    # 打印几个元素——PASS/FAIL 之外，数值本身能看出是不是「差一个转置」这类错误
    print(f"  got={got[0,0,0,:4].tolist()}")
    print(f"  ref={ref[0,0,0,:4].tolist()}")


if __name__ == "__main__":
    main()
```

## 2. 跨核 cube→UB→GM 探针

在 §1 基础上把 vec 侧也拉进来，验证 fixpipe split-M、UB 布局、
`partition_view` 的行分配。要点：每个缓冲区的消费区间内必须有可分类的
consumer 数据 op，否则报
`cannot classify consumer pipe in the wait/release interval`。

---

## 2.1 探针假 PASS 的两种方式

探针挂掉很好办 —— 你会看到报错。**危险的是探针"跑通了"但结论无效**：它给你绿灯，而你据此改了 kernel。

**坑 A：一个 region 里连写同一 buffer 的多个偏移。** 想省事把 N 个问题写进一个 buffer 的 N 个偏移一次跑完，结果前几个对、后几个错 —— **包括一个逻辑上平凡正确的对照组也错了**。根因是 `../../cannbotdsl-vf-fusion/SKILL.md` §6 陷阱 4（融合时桥接到陈旧 store）反噬探针本身，与被测 op 无关。

> 症状识别：**对照组都错了，先怀疑探针结构，别怀疑硬件。**
> 规避：**每个问题一个独立 `@jit` region、一个独立 buffer。** 多跑几次的成本远低于一个假结论。

**坑 B：用了让两种假设无法区分的输入。** 验证 `vmadd(acc,lhs,rhs)` 是 `acc + lhs*rhs` 还是 `acc*lhs + rhs` 时，第一版探针写的是 `vmadd(a, b, a)` —— 而 `a + b*a == a*b + a`（交换律），**两种读法数值完全相同**。探针返回 PASS，"证实"你原本相信的那个，你无从察觉。

> 症状识别：这类探针**永远 PASS**，无论真相如何 —— 所以它的 PASS 不携带任何信息。
> 规避：设计输入时先自问「**如果我的假设是反的，这个输入会给出不同的数吗？**」答案是"不会"就换输入。测操作数序要用**三个互不相同、不满足任何对称性**的值（如 2.0 / 3.0 / 5.0）。

**通用判据**：一个探针的 PASS 只有在**它的 FAIL 是可能的**时才有意义。写完先问一句：**什么情况下这个探针会失败？** 答不出来，它测的就不是你以为的东西。详见 `../SKILL.md` §2.1。

### 2.2 raw-VF 操作数序探针的具体写法

验 raw-VF 操作数序（如 `vmadd`/`vexp_sub`）时，用**三个互不相同、不满足任何对称性**的值。下面是 `vmadd(acc, lhs, rhs)` 的探针骨架：

```python
@jit
def probe_vmadd(out_buf):
    """验证 vmadd(acc, lhs, rhs) = acc*lhs + rhs 还是 acc + lhs*rhs"""
    with vf(mode='raw'):
        m32 = full_mask(elem_bits=32)
        # 三个不对称值：2.0, 3.0, 5.0
        # 若 vmadd = acc*lhs + rhs: 2*3+5 = 11
        # 若 vmadd = acc + lhs*rhs: 2+3*5 = 17
        acc = vdup_scalar(2.0, mask=m32)
        lhs = vdup_scalar(3.0, mask=m32)
        rhs = vdup_scalar(5.0, mask=m32)
        res = vmadd(acc, lhs, rhs)
        vstore(out_buf, 0, res, m32)
```

判定：读回 `out_buf[0]`，11 → `acc*lhs + rhs`，17 → `acc + lhs*rhs`。**不要用 `vmadd(a, b, a)`** —— 交换律下两种读法数值完全相同，探针永远 PASS。详见 `../../cannbotdsl-vf-fusion/SKILL.md` 陷阱 10。

---

## 3. 参数扫描骨架

**最高效的单一定位手段**。关键是两端都要取到「应该对」的点。

```python
def run(param):
    """构造 -> 跑 -> 对 fp64 打分，返回 (passed, mere)"""
    ...

print("sweep <param>:")
for p in (256, 384, 448, 512):     # 覆盖边界两侧，含应该对的点
    ok, mere = run(p)
    print(f"  {p:>5}: {'PASS' if ok else 'FAIL'}  MERE={mere:.3e}")
```

用环境变量参数化 shape，一份探针跑多个配置，避免复制粘贴出不一致：

```python
D = int(os.environ.get("PROBE_D", "448"))
```

```bash
for d in 256 384 448 512; do PROBE_D=$d python probe.py; done
```

---

## 4. 误差形态定位

拿到错误结果**先看形态再改代码**。形态→机制的对应表见 SKILL.md §5。

以下片段供内联调试用。

```python
d = (got.float() - ref.float()).abs()      # [B, S, N, D]
print(f"overall max={d.max():.3e} mean={d.mean():.3e}")

# 按 query 行——错行落在 tile 边界上就是 tile 划分问题
per_s = d.amax(dim=(0, 2, 3))
bad = [i for i, e in enumerate(per_s.tolist()) if e > 1e-2]
print(f"bad rows: {bad[:12]}{'...' if len(bad) > 12 else ''} ({len(bad)}/{len(per_s)})")

# 按列带——定位输出轴分块
for c in range(0, D, 128):
    hi = min(c + 128, D)
    print(f"  cols[{c}:{hi}] max={d[..., c:hi].max():.3e}")

# 按 split-M 的行段——定位分区
for lo in range(0, S, 32):
    hi = min(lo + 32, S)
    print(f"  rows[{lo}:{hi}] max={d[:, lo:hi].max():.3e}")

# 输出是否被整体平移（stride 错位的典型特征）
for r0 in (1, 2):
    best = min(range(S), key=lambda r1: (got[0, r0] - ref[0, r1]).abs().mean())
    print(f"  out row {r0} best-matches ref row {best}")
```

均匀错（所有行列都错、量级相近）指向**共享状态污染**，不要去查寻址。

---

## 5. 数值路径模拟器

回答「这个误差是实现 bug 还是设计固有？」——用 torch 复刻 kernel 的
**精确数值路径**（含每一次 cast 和累加精度），对 fp64 打分。

```python
def online_tiled(p_cast=torch.float16):
    """复刻 kernel：fp32 累加、tile 化 online softmax、P cast 后进 cube"""
    acc = torch.zeros(...)                      # fp32
    running_max = torch.full(..., float("-inf"))
    running_sum = torch.zeros(...)
    for n0 in range(0, S_kv, TILE_N):
        s = (q @ k[n0:n1].T) * scale            # fp32 累加
        tile_max = s.amax(-1, keepdim=True)
        new_max = torch.maximum(running_max, tile_max)
        rescale = torch.exp(running_max - new_max)
        p = torch.exp(s - new_max)
        running_sum = running_sum * rescale + p.sum(-1, keepdim=True)
        p_mm = p.to(p_cast).to(torch.float32)   # ← kernel 里的 cast
        acc = acc * rescale + p_mm @ v[n0:n1]
        running_max = new_max
    return (acc / running_sum).to(out_dtype)

score(online_tiled(torch.float16), ref_fp64, "B  模拟 kernel 路径")
score(online_tiled(None),          ref_fp64, "C  P 保持 fp32")
score(ref_fp32.to(dt),             ref_fp64, "D  精度地板")
```

模拟值 ≈ 实测值 ⇒ **实现无 bug**，误差来自设计选择；C/D 给出「换设计能换回多少」。

---

## 6. 单变量 A/B 探针

验证「是不是 X 导致的」时，同一份代码跑两版，只差 X。注意 **`@jit` 的参数不能是
Python `bool`**（`TypeError: unsupported runtime value`），编译期开关要走模块级常量
或 `@kernel` 类的 `__init__` 属性：

```python
USE_TRANSPOSE = True          # 模块级，由 main() 改写

@kernel
class probe_kernel:
    def __init__(self, use_transpose):
        self.use_transpose = use_transpose
    def __call__(self, ...):
        if self.use_transpose:
            mem_copy(self.l0b, self.b_l1, transpose=True)
        else:
            mem_copy(self.l0b, self.b_l1)

@jit
def launch(...):
    probe_kernel(USE_TRANSPOSE)[1](...)      # 读模块级常量

def run(use_transpose):
    global USE_TRANSPOSE
    USE_TRANSPOSE = use_transpose
    ...
```

```bash
python probe.py t    # transpose=True
python probe.py      # transpose=False
```

---

## 常用样板

**远端同步 + 执行**（探针必须上板，本地只做编辑）：

```bash
#!/bin/bash
set -e
rsync -az --delete-after --exclude '__pycache__' --exclude '.git' \
      ./ <host>:<remote_dir>/
if [ $# -gt 0 ]; then
  ssh <host> "source <venv>/bin/activate; source <env.sh>; cd <remote_dir>; $*"
fi
```

**编译慢**：每个新 shape 都会重新 JIT（首次约 10s+，之后命中缓存）。扫描多个配置时
把它放后台，用 Monitor/轮询取结果，不要同步等。
