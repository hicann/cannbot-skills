# GEMM 流水线骨架模板（含 MMAD 算子的 matmul 骨架起点）

> 本文件是结构模板，不是可直接运行的算子。它与目标 DSL 版本的官方 auto_sync 双缓冲
> 样例同构；`unit_flag`、`allocate` 对齐、`make_tensor_like` 的 layout 推导等 API 细节，
> 以该版本官方样例为准。目标版本不支持 `auto_sync` 时，改用官方 flag-sync 样例
> （`examples/python/tla_dsl/` 下的 `basic_matmul.py`）的等价双缓冲，结构不变，并在
> `ITERATIONS.md` 记录样例路径。
>
> 输入假设：`a [M,K]`、`b [K,N]`、`c [M,N]` 均为 RowMajor，`c` 为 fp32；host 负责把
> `M/N/K` 补齐到 `BM/BN/BK` 的整数倍。若真实接口的 weight 是 `[N,K]`，先在 DESIGN 里
> 写明视图转换方式，不要在 kernel 内隐式改变 layout 语义。主 GEMM 含 batch 维时，本
> 模板只覆盖单 batch；batch 循环在 implement 阶段加在外层，不得改动缓冲与同步结构。
> 复制到主实现后只允许修改 `BM/BN/BK/SK`、dtype 与块映射（波次量化 / L2 swizzle）；
> 尾块、padding、多 dtype 分派与跨核交棒按 `DESIGN.md` 补充，补充时不得破坏缓冲与
> 同步结构。
>
> 改完 tile 与 dtype 后先按 `knowledge/dsl/pattern-capacity-driven-tiling` 校核容量：
> A5/3510 上 L1 = 512KB、L0A = 64KB、L0B = 64KB、L0C = 256KB。默认参数
> `BM/BN/BK/SK = 256/256/128/32` 在 f16 下 L0A、L0B 各占 32KB（双缓冲 = 2 份 × 16KB），
> f32 输入翻倍；超限编译期不报错，上板才触发 aicore OOB。
>
> 默认块映射是 `sm × sn = 8 × 8` 的 super-tile swizzle，用来提升 B tile 的 L2 复用；
> 形状不适合时（`nm`/`nn` 小于 8，或实测行优先更快）再退回 `task // nn`。两种映射都要
> 实测对比，并把 `nm`、`nn`、重读倍数与带宽下界写入 `ITERATIONS.md`，避免把块映射重读
> 误判成流水缺失。
>
> 每个 GEMM 至少扫 3 组配置（主 GEMM 含一组 `BM=256` 量级的大 tile），按下面这张表记录，
> 取最好的一组写回 `DESIGN.md` 并带进实现：
>
> | `BM/BN/BK/SK` | 块映射 | `gemm_ms` | `torch.mm` | 比值 | `mac` | `mte2` | 结论 |
> |---|---|---|---|---|---|---|---|
> | | | | | | | | |

```python
"""参数化 GEMM 流水线骨架（含 MMAD 算子的 matmul 骨架起点，单 batch）。

契约（首版单 GEMM 骨架阶段，见 `catlass-dsl-develop/SKILL.md`）：

- 只允许修改 bm/bn/bk/sk、dtype 与块映射（波次量化 / L2 swizzle）；
- 固定参数必须在 kernel 函数内定义，不得读取模块级配置（kernel 自包含约束）；
- 必须保留 L1A/L1B 各 2 份缓冲，按 l1_idx 交替；
- 必须保留 L0A/L0B 各 2 份缓冲，按 l0_idx 交替；
- 必须满足 bk > sk 且 bk % sk == 0；
- 禁止单缓冲 + 每个 copy 紧跟 set_flag/wait_flag 的串行写法；
- 禁止多个 ready flag 共享同一个 MTE1→CUBE 对。

依据：`knowledge/dsl/pattern-gemm-optimization-ladder` 第 2、3 级。
"""

import catlass.tla as tla


@tla.kernel(auto_sync="v0")
def gemm_pipelined(a: tla.Tensor, b: tla.Tensor, c: tla.Tensor) -> None:
    """C[M, N] = A[M, K] @ B[K, N] 的分块骨架（单 batch）。"""
    bm, bn, bk, sk = 256, 256, 128, 32
    # 契约：bk > sk 且 bk % sk == 0；不满足时先改参数再运行。
    dtype = a.ptr.dtype
    m = a.origin_shape[0]
    ktot = a.origin_shape[1]
    n = b.origin_shape[1]
    nm = (m + bm - 1) // bm
    nn = (n + bn - 1) // bn
    nk = (ktot + bk - 1) // bk

    # 第 2 级：L1 双侧双缓冲（具名指针；2.2.0 lowering 不支持列表推导式分配）
    l1a0_ptr = tla.allocate(bm * bk, dtype, tla.AddressSpace.l1, 512)
    l1a1_ptr = tla.allocate(bm * bk, dtype, tla.AddressSpace.l1, 512)
    l1b0_ptr = tla.allocate(bk * bn, dtype, tla.AddressSpace.l1, 512)
    l1b1_ptr = tla.allocate(bk * bn, dtype, tla.AddressSpace.l1, 512)

    # 第 3 级：L0 双侧双缓冲
    l0a0_ptr = tla.allocate(bm * sk, dtype, tla.AddressSpace.l0a, 512)
    l0a1_ptr = tla.allocate(bm * sk, dtype, tla.AddressSpace.l0a, 512)
    l0b0_ptr = tla.allocate(sk * bn, dtype, tla.AddressSpace.l0b, 512)
    l0b1_ptr = tla.allocate(sk * bn, dtype, tla.AddressSpace.l0b, 512)
    l0c = tla.allocate(bm * bn, tla.Float32, tla.AddressSpace.l0c, 512)

    # 块映射：8×8 super-tile swizzle（默认值；形状不适合时退回行优先并实测对比）
    sm, sn = 8, 8
    gsm = (nm + sm - 1) // sm
    gsn = (nn + sn - 1) // sn
    total = gsm * gsn * sm * sn

    with tla.cube():
        l1_idx = 0
        l0_idx = 0
        for task in tla.range(tla.arch.block_idx(), total, tla.arch.block_num()):
            per_super = sm * sn
            sb = task // per_super
            inner = task % per_super
            mt = (sb // gsn) * sm + inner // sn
            nt = (sb % gsn) * sn + inner % sn
            if mt < nm and nt < nn:
                og = tla.tile_view(c, tla.make_shape(bm, bn), tla.make_coord(mt, nt))
                l0_c = tla.make_tensor_like(l0c, og)
                for kb in tla.range(0, nk, 1):
                    ag = tla.tile_view(a, tla.make_shape(bm, bk), tla.make_coord(mt, kb))
                    bg = tla.tile_view(b, tla.make_shape(bk, bn), tla.make_coord(kb, nt))
                    al = tla.make_tensor_like(
                        l1a0_ptr if (l1_idx == 0) else l1a1_ptr, ag
                    )
                    bl = tla.make_tensor_like(
                        l1b0_ptr if (l1_idx == 0) else l1b1_ptr, bg
                    )
                    tla.copy(al, ag)
                    tla.copy(bl, bg)
                    for kk in tla.range(0, bk // sk, 1):
                        aa = tla.tile_view(al, tla.make_shape(bm, sk), tla.make_coord(0, kk))
                        bb = tla.tile_view(bl, tla.make_shape(sk, bn), tla.make_coord(kk, 0))
                        a0 = tla.make_tensor_like(
                            l0a0_ptr if (l0_idx == 0) else l0a1_ptr, aa
                        )
                        b0 = tla.make_tensor_like(
                            l0b0_ptr if (l0_idx == 0) else l0b1_ptr, bb
                        )
                        tla.copy(a0, aa)
                        tla.copy(b0, bb)
                        is_last = (kb == nk - 1) and (kk == bk // sk - 1)
                        tla.mmad(
                            l0_c,
                            a0,
                            b0,
                            init_c=(kb == 0 and kk == 0),
                            unit_flag=0b11 if is_last else 0b10,
                        )
                        l0_idx = 1 - l0_idx
                    l1_idx = 1 - l1_idx

                tla.copy(
                    og,
                    l0_c,
                    tla.params.CopyL0C2DstParams(unit_flag=0b11),
                )
```
