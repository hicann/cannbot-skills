# FA 四阶段时序与状态示例

适用于采用 C1→V1→C2→V2 的在线 softmax 计算图，Cube、Vector 可独立推进，同一 task 的有效 KV 工作项连续排列。该例选择 Cube 先 C1 后 C2、Vector 先 V1 后 V2；阶段偏移与份数对应本排布。输入装载、额外阶段、交错 task 或更长异步存续区间另行推导。

参考依据为 `cannbot-arena/samples/flash_attn/flash_attn.py` 中的任务推进、四阶段交接和分开的状态版本。本文展开算法与完成协议；实际物理布局、分配深度及类型在所选路径的资源预算中代入。

## 1. 公式与状态所有权

对 task g 的 KV 工作项 n，逐输出行定义：

```text
C1: S_n = Q_g * K_n^T
V1: Z_n = scale * S_n，按语义应用掩码
    m_n = max(m_prev, row_max(Z_n))
    alpha_n = exp(m_prev - m_n)
    P_n = exp(Z_n - m_n)
    l_n = alpha_n * l_prev + row_sum(P_n)
C2: B_n = P_n * V_n
V2: U_n = alpha_n * U_prev + B_n
末项: O_g = U_last / l_last
```

首次有效项取 `m_prev=-inf、l_prev=0`，首次 V2 直接令 U=B；本例每个工作项的各行有有效分数。全掩码、零工作项及特殊数值按接口语义展开相应转移。P 的转换与累积精度由数值计算链实例化。

| 对象 | 更新者 | 后续读取 | 保留范围 |
|---|---|---|---|
| task 的 max/sum，即 m、l | V1，按该 task 的 KV 顺序 | 后续 V1；末次 V2 读取最终 l | 首次 V1 到最终读取完成 |
| 工作项的 alpha | 对应 V1 | 对应 V2 | 当前 V1 到延迟 V2 的最后读取 |
| task 的输出累计 U | V2，按该 task 的 KV 顺序 | 后续 V2 与最终输出 | 首次 V2 到输出读取完成 |

V1 可继续更新同一 task 的 m/l，V2 使用所属工作项已保存的 alpha；非末次 V2 只更新 U，末次 V2 读取最终 l。该 task 的末次 V1 已完成，且其状态尚未被新 task 覆盖。这些条件支持本例的 V1 先于 V2。

## 2. 工作项与完整时序

g 是当前任务流中连续推进的有效 task 序号，w 是有效工作项的连续发射序号；输出坐标另行记录。全局工作项携带 `D[w]=(task 序号 g, 实际 KV 序号 n, n_begin, n_end, 有效行列, 输出范围)`。首次判断为 n=n_begin，末次判断为 n=n_end-1；task 的起点可以非零。状态槽按有效 task 序号轮转，alpha 槽按工作项发射序号轮转。

| w | task g | 有效区间 | n | 首次 / 末次 |
|---|---|---|---|---|
| 0 | 0 | [3,5) | 3 | 是 / 否 |
| 1 | 0 | [3,5) | 4 | 否 / 是 |
| 2 | 1 | [0,1) | 0 | 是 / 是 |
| 3 | 2 | [7,8) | 7 | 是 / 是 |
| 4 | 3 | [1,3) | 1 | 是 / 否 |
| 5 | 3 | [1,3) | 2 | 否 / 是 |
| 6 | 4 | [2,3) | 2 | 是 / 是 |
| 7 | 5 | [5,6) | 5 | 是 / 是 |

调度步 t 中，Cube 先 C1(t) 后 C2(t-2)，Vector 先 V1(t-1) 后 V2(t-3)。索引在 `[0,N)` 内才发射；同一资源栏中的两个阶段顺序执行。调度步表示依赖与发射顺序，实际重叠由完成协议与目标资源支持。

| t | 推进阶段 | Cube 顺序 | Vector 顺序 | task 状态的关键动作 |
|---|---|---|---|---|
| 0 | warmup | C1(0) | — | — |
| 1 | warmup | C1(1) | V1(0) | 初始化 task0 的 m/l |
| 2 | warmup | C1(2) → C2(0) | V1(1) | 更新 task0 的 m/l |
| 3 | steady | C1(3) → C2(1) | V1(2) → V2(0) | 初始化 task1 的 m/l；首次更新 task0 的 U |
| 4 | steady | C1(4) → C2(2) | V1(3) → V2(1) | 初始化 task2 的 m/l；输出 task0 并释放其状态 |
| 5 | steady | C1(5) → C2(3) | V1(4) → V2(2) | 初始化 task3 的 m/l；输出 task1 |
| 6 | steady | C1(6) → C2(4) | V1(5) → V2(3) | 更新 task3 的 m/l；输出 task2 |
| 7 | steady | C1(7) → C2(5) | V1(6) → V2(4) | 初始化 task4 的 m/l；首次更新 task3 的 U |
| 8 | drain | C2(6) | V1(7) → V2(5) | 初始化 task5 的 m/l；输出 task3 |
| 9 | drain | C2(7) | V2(6) | 输出 task4 |
| 10 | drain | — | V2(7) | 输出 task5；全部释放 |

## 3. 各对象的槽与代次

| 对象 | 本例份数与索引 | 生产 | 最后读取与释放 | 推导依据 |
|---|---|---|---|---|
| S 队列 | 2，槽 w mod 2 / 代 floor(w/2) | C1(w)，步 w | V1(w)，步 w+1 | 两资源可先生产新项再释放旧项 |
| P 队列 | 2，槽 w mod 2 / 代 floor(w/2) | V1(w)，步 w+1 | C2(w)，步 w+2 | 相邻生产、消费间最多持有两项 |
| B 队列 | 2，槽 w mod 2 / 代 floor(w/2) | C2(w)，步 w+2 | V2(w)，步 w+3 | 两资源可先生产新项再释放旧项 |
| alpha 版本 | 3，槽 w mod 3 / 代 floor(w/3) | V1(w)，步 w+1 | V2(w)，步 w+3 | 同一步先 V1(w+2) 再 V2(w)，可同时持有三项 |
| task 的 m/l | 3，槽 g mod 3 / 代 floor(g/3) | 首次 V1 | 末次 V2 的最终读取 | 步4初始化 task2 后仍保留 task0、task1 的状态 |
| U | 1，所属 task 在首次 V2 切换 | 首次 V2 初始化，后续 V2 更新 | 末次输出的最后读取 | V2 按全局工作项顺序执行，前一 task 输出完成后再初始化下一 task |

连续 task 模型下，末次 V1 到末次 V2 跨两步；新 task 每步到来时，m/l 最多三份未释放。U 的初始化发生在首次 V2，因此该排布可独立使用一份 U。alpha 每工作项一版，其索引与 m/l 的 task 索引分别推进。

本例交接队列的份数为所示完成协议的推导值；额外预取、异步读者或实际采用的更深分配各自记账。物理量按池分别汇总：`2*bS + 2*bP + 2*bB + 3*bAlpha + 3*bML + bytes_U`，另加 Q/K/V、副本、转换临时区、同步及描述存储。各 b 包含实际类型、步长、布局、对齐与填充。

每槽记录所属 task/工作项和代次。写完成且对读者可见后发布 READY，最后读取完成后释放。m/l 保留到最终 l 的读取完成；U 保留到输出读取完成；alpha 保留到所属 V2 使用完成。

## 4. warmup / steady / drain 伪代码

以下完成操作表达抽象协议。两角色各自按顺序执行阶段，并通过依赖与可用槽等待推进。

```text
run(N, D):
    if N == 0:
        complete_boundary_outputs()
        return
    initialize_ownership_and_generations()
    start_roles_concurrently(Cube, Vector):
        for t in [0, min(N,3)): advance(role, t, N)   # warmup
        for t in [3, N):       advance(role, t, N)   # steady
        for t in [N, N+3):     advance(role, t, N)   # drain
    join_roles_and_finish_outputs()

advance(Cube, t, N):
    if 0 <= t   < N: C1(t,   D[t])
    if 0 <= t-2 < N: C2(t-2, D[t-2])

advance(Vector, t, N):
    if 0 <= t-1 < N: V1(t-1, D[t-1])
    if 0 <= t-3 < N: V2(t-3, D[t-3])

C1(w, d):
    s = acquire_free(S, w mod 2, floor(w/2))
    write(s, d, Q[d.g] * K[d.n]^T)
    publish_after_write_complete(s, w)

V1(w, d):
    s = wait_ready(S, w, floor(w/2))
    if d.first:
        ml = acquire_free(ML, d.g mod 3, floor(d.g/3))
        initialize(ml, owner=d.g, max=-inf, sum=0, next_n=d.n_begin)
    else:
        ml = wait_owned(ML, d.g, floor(d.g/3))
    require(ml.next_n == d.n)
    z = scale_and_apply_mask(s, d.valid_extent)
    m_new = max(ml.max, row_max(z))
    alpha = exp(ml.max - m_new)
    p_value = exp(z - m_new)
    ml.sum = alpha * ml.sum + row_sum(p_value)
    ml.max = m_new
    ml.next_n = d.n + 1
    complete_online_state_update(ml)
    a = acquire_free(Alpha, w mod 3, floor(w/3))
    write(a, d, alpha)
    publish_after_write_complete(a, w)
    p = acquire_free(P, w mod 2, floor(w/2))
    write(p, d, convert_as_designed(p_value))
    publish_after_write_complete(p, w)
    release_after_last_read(s)

C2(w, d):
    p = wait_ready(P, w, floor(w/2))
    b = acquire_free(B, w mod 2, floor(w/2))
    write(b, d, p.value * V[d.n])
    publish_after_write_complete(b, w)
    release_after_last_read(p)

V2(w, d):
    b = wait_ready(B, w, floor(w/2))
    a = wait_ready(Alpha, w, floor(w/3))
    if d.first:
        u = acquire_free(U)
        initialize(u, owner=d.g, value=b.value, next_n=d.n+1)
    else:
        u = wait_owned(U, d.g)
        require(u.next_n == d.n)
        u.value = a.value * u.value + b.value
        u.next_n = d.n + 1
    complete_accumulator_update(u)
    if d.last:
        ml = wait_owned(ML, d.g, floor(d.g/3))
        require(ml.next_n == d.n_end)
        finish_output(d.output_range, u.value / ml.sum)
        release_after_last_read(ml)
        release_after_last_read(u)
    release_after_last_read(a)
    release_after_last_read(b)
```

实际机制核对任务描述、代次和读写范围。首次 alpha=0 在本例按统一协议传递；采用首次项直接计算的路径时，对省略的访问及其索引规则另行展开。

## 5. 边界、选择与验证

- N=1：warmup 为步0，drain 为步1、2、3；每阶段恰处理工作项0一次。
- N=2：warmup 为步0、1，drain 为步2、3、4；索引检查按有效 N 执行。
- N≥3：warmup 三步、steady 共 `max(0,N-3)` 步、drain 三步。三步来自本例的最大阶段偏移。
- task 的实际有效起点随描述传递；首末判断、最终 l 与 U 的归属在非零起点和短 task 下保持一致。
- 尾部、被跳过工作项、空 task 及错误后的完成协议按实际语义补齐。

用不同 KV 起点、不同 task 长度与不同值核对独立的全量 attention 参考结果。协议检查覆盖描述、阶段次数、alpha 代次、m/l 归属、U 更新顺序与最终释放。V2 先于 V1 的候选重新推导生命周期；比较资源、等待与 [阶段成本](stage-cost.md)，实际重叠和收益在后续实现验证。
