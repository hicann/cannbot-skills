# 理论性能模型

实现思路参考 `ops-theoretical-performance-estimation` Skill 的 golden flow + vector/MTE 计时模型，
并按本 skill 的通用路由需求扩展到 MTE 引擎、Cube（可选）、自动 L2/HBM 带宽选择和逐 shape 排名。

本 skill 按用户指定口径：

```text
T_vector = vector 指令时间
T_copy   = (read_bytes + write_bytes) / effective_bandwidth
T_hw     = max(T_vector, T_copy)
headroom = T_candidate / T_hw
```

可选 Cube：

```text
T_hw = max(T_vector, T_copy, T_cube)
```

## Vector 时间

```text
elements_per_cycle = hardware.vector_elements_per_cycle[dtype]
                    或 vector_width_bits / (8 * dtype_bytes)
cycles = ceil(elements / elements_per_cycle) * instruction_cost * count
T_vector(unit) = Σ cycles / (active_cores * frequency_hz)
T_vector = max(T_vector(unit))
```

`instruction_cost` 来自 `references/hardware/*.json` 的 `vector_instruction_costs`，可在 flow 中逐指令覆盖。

## Copy 时间

```text
T_memory = (read_bytes + write_bytes) / effective_bandwidth
T_engine = max(read_bytes / MTE2_capacity, write_bytes / MTE3_capacity)
T_copy   = max(T_memory, T_engine)
```

`effective_bandwidth` 由 `bandwidth_model` 决定：

- `hbm`：HBM 峰值
- `l2`：L2 峰值
- `auto`：工作集 ≤ L2 容量用 L2，否则 HBM

## 排名

按 `headroom` 降序排序，越靠前越差。`headroom > 20` 通常表示架构级差距；`5~20` 为 mixed；`<5` 接近微调区间。

## 模型门控

- 每个 shape 至少一个非零耗时项。
- 实测不得低于理论下限（允许测量误差）。
- 不支持的 dtype / 指令 / Cube 必须报错。
- 每个 case 的 vector/copy/cube 明细必须落盘。
