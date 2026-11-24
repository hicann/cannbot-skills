# GEMM 流水线骨架模板（含 MMAD 算子的 matmul 骨架起点）

> 本文件是结构模板，不是可直接运行的算子。骨架的核内同步只允许 `tla.mutex`：禁止
> `auto_sync`、`tla.flag` / `set_flag` / `wait_flag`、`tla.pipe_barrier` 与 `cross_flag`
> （骨架不含跨核交棒，融合阶段的跨核同步另见 develop 规则）。唯一豁免：主参考样例本身
> 不使用 mutex 且结构高度匹配时，按样例原语实现并在 `DESIGN.md` / `ITERATIONS.md` 记录。
> 结构与 `examples/python/tla_dsl/experimental/kernel/matmul/matmul_exp_scale_ub/` 的
> GEMM 段一致（每份片上缓冲一块独立 mutex、`L0C` 单列，缓冲份数变化时按实际份数增减）；
> `unit_flag`、`allocate` 对齐、`make_tensor_like` 的 layout 推导等 API 细节，以该样例为准。
>
> 输入假设：逻辑形状 `a [M,K]`、`b [K,N]`、`c [M,N]`，`c` 为 fp32；`a` / `b` 的物理布局由
> host 的 `layout_tag` 决定，可为 RowMajor 或 ColumnMajor（`col` 即转置视图，A 物理 K×M、
> B 物理 N×K），kernel 内不做 layout 分支、`origin_shape` 始终传逻辑形状；host 负责把
> `M/N/K` 补齐到 `BM/BN/BK` 的整数倍。主 GEMM 含 batch 维时，本模板只覆盖单 batch；batch
> 循环在 implement 阶段加在外层，不得改动缓冲与同步结构。
> 复制到主实现后只允许修改 `BM/BN/BK/SK`、dtype 与块映射（波次量化 / L2 swizzle）；
> 尾块、padding、多 dtype 分派与跨核交棒按 `DESIGN.md` 补充，补充时不得破坏缓冲与
> 同步结构。
>
> 改完 tile 与 dtype 后先按 `knowledge/dsl/pattern-capacity-driven-tiling` 校核容量：
> 以 A5/3510 为例：L1 = 512KB、L0A = 64KB、L0B = 64KB、L0C = 256KB（其它型号以设备读取
> 为准，容量与核数一律现读，不要照抄这里的数值）。该容量下默认参数
> `BM/BN/BK/SK = 256/256/128/32` 在 f16 下 L0A、L0B 各占 32KB（双缓冲 = 2 份 × 16KB），
> f32 输入翻倍；超限编译期不报错，上板才触发 aicore OOB。
>
> 默认块映射是 `sm × sn = 8 × 8` 的 super-tile swizzle，用来提升 A/B tile 的 L2 复用。
> 映射按整除关系算、不占实跑名额：`ceil(nn/8)·8/nn − 1 > 25%` 时 `sn` 取能整除 `nn` 的
> 因子，上限放宽到 `sn ≤ nn`（`nn ≤ 16` 时 `sn = nn` 也纳入候选；行方向对 `nm` 对称处理
> `sm`），否则保持 8。行优先只作为预测失败时的对照，
> 不默认参与扫描。`nm`、`nn`、重读倍数与名义搬运量写入 `ITERATIONS.md`，避免把块映射
> 重读误判成流水缺失。
>
> 上板前先按 `catlass-dsl-develop/SKILL.md` 的解析预筛淘汰超容量、低流水强度与映射浪费
> 的组合，每个 GEMM 实跑不超过 3 组（含预测最优组与一组方向对照），按下面这张表记录，
> 取最好的一组写回 `DESIGN.md` 并带进实现：
>
> | `BM/BN/BK/SK` | 块映射 | 预测 ms | 预测比值 | `gemm_ms` | `torch.mm` | 比值 | `mac` | `mte2` | 结论 |
> |---|---|---|---|---|---|---|---|---|---|
> | | | | | | | | | | |

```python
"""参数化 GEMM 流水线骨架（含 MMAD 算子的 matmul 骨架起点，单 batch）。

契约（首版单 GEMM 骨架阶段，见 `catlass-dsl-develop/SKILL.md`）：

- 只允许修改 bm/bn/bk/sk、dtype 与块映射（波次量化 / L2 swizzle）；
- 固定参数必须在 kernel 函数内定义，不得读取模块级配置（kernel 自包含约束）；
- 必须保留 L1A/L1B 各 2 份缓冲，按 l1_phase 交替；
- 必须保留 L0A/L0B 各 2 份缓冲，按 l0_phase 交替；
- 必须满足 bk > sk 且 bk % sk == 0；
- 每个片上缓冲使用独立 `tla.mutex`（id 全 kernel 唯一），lock/unlock 的 pipe 必须成对；
- 禁止单缓冲 + 每个 copy 紧跟 lock/unlock 的串行写法。

依据：`knowledge/dsl/pattern-gemm-optimization-ladder` 第 2、3 级。
"""

import catlass.tla as tla


@tla.kernel
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

    # 同步：每个片上缓冲一块 mutex（id 全 kernel 唯一），lock/unlock 的 pipe 成对
    m_l1a0 = tla.mutex("l1_a0", id=0)
    m_l1b0 = tla.mutex("l1_b0", id=1)
    m_l1a1 = tla.mutex("l1_a1", id=2)
    m_l1b1 = tla.mutex("l1_b1", id=3)
    m_l0a0 = tla.mutex("l0_a0", id=4)
    m_l0b0 = tla.mutex("l0_b0", id=5)
    m_l0a1 = tla.mutex("l0_a1", id=6)
    m_l0b1 = tla.mutex("l0_b1", id=7)
    m_l0c = tla.mutex("l0_c", id=8)

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
    l0c = tla.allocate(bm * bn, tla.Float32, tla.AddressSpace.l0c, 1024)

    # 块映射：8×8 super-tile swizzle（默认值；按整除规则调整 sm/sn，行优先只作对照）
    sm, sn = 8, 8
    gsm = (nm + sm - 1) // sm
    gsn = (nn + sn - 1) // sn
    total = gsm * gsn * sm * sn

    with tla.cube():
        for task in tla.range(tla.arch.block_idx(), total, tla.arch.block_num()):
            per_super = sm * sn
            sb = task // per_super
            inner = task % per_super
            mt = (sb // gsn) * sm + inner // sn
            nt = (sb % gsn) * sn + inner % sn
            if mt < nm and nt < nn:
                og = tla.tile_view(c, tla.make_shape(bm, bn), tla.make_coord(mt, nt))
                l0_c = tla.make_tensor_like(l0c, og)

                ag_head = tla.tile_view(a, tla.make_shape(bm, bk), tla.make_coord(mt, 0))
                bg_head = tla.tile_view(b, tla.make_shape(bk, bn), tla.make_coord(0, nt))
                al = tla.make_tensor_like(l1a0_ptr, ag_head)
                bl = tla.make_tensor_like(l1b0_ptr, bg_head)
                m_l1a0.lock(pipe=tla.arch.MTE2)
                tla.copy(al, ag_head)
                m_l1a0.unlock(pipe=tla.arch.MTE2)
                m_l1b0.lock(pipe=tla.arch.MTE2)
                tla.copy(bl, bg_head)
                m_l1b0.unlock(pipe=tla.arch.MTE2)
                m_l0c.lock(pipe=tla.arch.CUBE)

                l1_phase = tla.as_numeric(zero)
                for kb in tla.range(0, nk, 1):
                    if l1_phase == zero:
                        m_l1a0.lock(pipe=tla.arch.MTE1)
                        m_l1b0.lock(pipe=tla.arch.MTE1)
                    else:
                        m_l1a1.lock(pipe=tla.arch.MTE1)
                        m_l1b1.lock(pipe=tla.arch.MTE1)

                    if kb + 1 < nk:
                        ag_n = tla.tile_view(
                            a, tla.make_shape(bm, bk), tla.make_coord(mt, kb + 1)
                        )
                        bg_n = tla.tile_view(
                            b, tla.make_shape(bk, bn), tla.make_coord(kb + 1, nt)
                        )
                        na_ptr = l1a1_ptr if l1_phase == zero else l1a0_ptr
                        nb_ptr = l1b1_ptr if l1_phase == zero else l1b0_ptr
                        al_n = tla.make_tensor_like(na_ptr, ag_n)
                        bl_n = tla.make_tensor_like(nb_ptr, bg_n)
                        if l1_phase == zero:
                            m_l1a1.lock(pipe=tla.arch.MTE2)
                            m_l1b1.lock(pipe=tla.arch.MTE2)
                        else:
                            m_l1a0.lock(pipe=tla.arch.MTE2)
                            m_l1b0.lock(pipe=tla.arch.MTE2)
                        tla.copy(al_n, ag_n)
                        tla.copy(bl_n, bg_n)
                        if l1_phase == zero:
                            m_l1a1.unlock(pipe=tla.arch.MTE2)
                            m_l1b1.unlock(pipe=tla.arch.MTE2)
                        else:
                            m_l1a0.unlock(pipe=tla.arch.MTE2)
                            m_l1b0.unlock(pipe=tla.arch.MTE2)

                    ca_ptr = l1a0_ptr if l1_phase == zero else l1a1_ptr
                    cb_ptr = l1b0_ptr if l1_phase == zero else l1b1_ptr
                    ag_c = tla.tile_view(a, tla.make_shape(bm, bk), tla.make_coord(mt, kb))
                    bg_c = tla.tile_view(b, tla.make_shape(bk, bn), tla.make_coord(kb, nt))
                    al_c = tla.make_tensor_like(ca_ptr, ag_c)
                    bl_c = tla.make_tensor_like(cb_ptr, bg_c)

                    aa = tla.tile_view(al_c, tla.make_shape(bm, sk), tla.make_coord(0, 0))
                    bb = tla.tile_view(bl_c, tla.make_shape(sk, bn), tla.make_coord(0, 0))
                    a0 = tla.make_tensor_like(l0a0_ptr, aa)
                    b0 = tla.make_tensor_like(l0b0_ptr, bb)
                    m_l0a0.lock(pipe=tla.arch.MTE1)
                    m_l0b0.lock(pipe=tla.arch.MTE1)
                    tla.copy(a0, aa)
                    tla.copy(b0, bb)
                    m_l0a0.unlock(pipe=tla.arch.MTE1)
                    m_l0b0.unlock(pipe=tla.arch.MTE1)

                    l0_phase = tla.as_numeric(zero)
                    for kk in tla.range(0, bk // sk, 1):
                        if l0_phase == zero:
                            m_l0a0.lock(pipe=tla.arch.CUBE)
                            m_l0b0.lock(pipe=tla.arch.CUBE)
                        else:
                            m_l0a1.lock(pipe=tla.arch.CUBE)
                            m_l0b1.lock(pipe=tla.arch.CUBE)

                        if kk + 1 < bk // sk:
                            aa_n = tla.tile_view(
                                al_c, tla.make_shape(bm, sk), tla.make_coord(0, kk + 1)
                            )
                            bb_n = tla.tile_view(
                                bl_c, tla.make_shape(sk, bn), tla.make_coord(kk + 1, 0)
                            )
                            na0_ptr = l0a1_ptr if l0_phase == zero else l0a0_ptr
                            nb0_ptr = l0b1_ptr if l0_phase == zero else l0b0_ptr
                            a0_n = tla.make_tensor_like(na0_ptr, aa_n)
                            b0_n = tla.make_tensor_like(nb0_ptr, bb_n)
                            if l0_phase == zero:
                                m_l0a1.lock(pipe=tla.arch.MTE1)
                                m_l0b1.lock(pipe=tla.arch.MTE1)
                            else:
                                m_l0a0.lock(pipe=tla.arch.MTE1)
                                m_l0b0.lock(pipe=tla.arch.MTE1)
                            tla.copy(a0_n, aa_n)
                            tla.copy(b0_n, bb_n)
                            if l0_phase == zero:
                                m_l0a1.unlock(pipe=tla.arch.MTE1)
                                m_l0b1.unlock(pipe=tla.arch.MTE1)
                            else:
                                m_l0a0.unlock(pipe=tla.arch.MTE1)
                                m_l0b0.unlock(pipe=tla.arch.MTE1)

                        ca0_ptr = l0a0_ptr if l0_phase == zero else l0a1_ptr
                        cb0_ptr = l0b0_ptr if l0_phase == zero else l0b1_ptr
                        aa_c = tla.tile_view(al_c, tla.make_shape(bm, sk), tla.make_coord(0, kk))
                        bb_c = tla.tile_view(bl_c, tla.make_shape(sk, bn), tla.make_coord(kk, 0))
                        a0_c = tla.make_tensor_like(ca0_ptr, aa_c)
                        b0_c = tla.make_tensor_like(cb0_ptr, bb_c)
                        is_last = (kb == nk - 1) and (kk == bk // sk - 1)
                        tla.mmad(
                            l0_c,
                            a0_c,
                            b0_c,
                            init_c=(kb == 0 and kk == 0),
                            unit_flag=0b11 if is_last else 0b10,
                        )
                        if l0_phase == zero:
                            m_l0a0.unlock(pipe=tla.arch.CUBE)
                            m_l0b0.unlock(pipe=tla.arch.CUBE)
                        else:
                            m_l0a1.unlock(pipe=tla.arch.CUBE)
                            m_l0b1.unlock(pipe=tla.arch.CUBE)
                        l0_phase = one - l0_phase

                    if l1_phase == zero:
                        m_l1a0.unlock(pipe=tla.arch.MTE1)
                        m_l1b0.unlock(pipe=tla.arch.MTE1)
                    else:
                        m_l1a1.unlock(pipe=tla.arch.MTE1)
                        m_l1b1.unlock(pipe=tla.arch.MTE1)
                    l1_phase = one - l1_phase

                m_l0c.unlock(pipe=tla.arch.CUBE)
                m_l0c.lock(pipe=tla.arch.FIX)
                tla.copy(
                    og,
                    l0_c,
                    tla.params.CopyL0C2DstParams(unit_flag=0b11),
                )
                m_l0c.unlock(pipe=tla.arch.FIX)
```
