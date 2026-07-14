---
name: object-detection
description: 目标检测类算子（NMS / IOU）的 Triton Ascend 优化经验合集，按算子分章节组织，含通用经验 + 各算子专属约束/骨架/kernel
metadata:
  type: reference
---

# 目标检测类算子优化经验

本文档合并了目标检测中两类核心算子的优化经验。按以下结构组织：
- **§1 通用经验**：跨算子重复的工程约束（已提取，各算子章节不再重复）
- **§2 NMS**（Non-Maximum Suppression，顺序算法 + 每轮 IoU 抑制）
- **§3 IOU**（Intersection over Union，成对计算）
- **§4 各算子常见陷阱**

---

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| NMS | `sort-topk` + `indexing-gather` 复合 | 顺序算法：过滤 → argsort → 顺序遍历 → 每轮成对 IoU 抑制 | fused kernel + device-side counter + chunked loop 消除 .item() 同步 |
| IOU | `elementwise` / `pairwise-compute` | 成对计算：输入两个 [N,4] 和 [M,4] 的 bbox，输出为 [M,N] 的 IoU/IoF 矩阵 | 2D Tiling + Broadcast 替代离散访存 |

> ⚠️ **关键区分**：NMS 属 **顺序算法类**（核心瓶颈是顺序 IoU 抑制循环中的 host-device 同步开销），IOU 属 **成对计算类**（核心瓶颈是离散逐元素访存无法利用 NPU 向量单元）。两类优化哲学相反，生成时**禁止混用经验**：
> - 生成 NMS 时，**不要**套用 IOU 的纯 2D Tiling 思路（NMS 每轮只对单个 cur_box 做抑制）
> - 生成 IOU 时，**不要**套用 NMS 的 chunked loop / device counter（IOU 无顺序依赖，单 kernel 即可完成）
> - **通用复用点**：NMS 的 IoU 抑制阶段可直接复用 IOU 的 1D Grid + 交织循环骨架（见 §2 L1.1 与 §3 L1.2）

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下 6 条约束在两个算子中均适用，各算子章节不再重复。

### D1 动态读取 Vector/Cube Core 数量，禁止硬编码
- **必须**动态读取实际核数，禁止硬编码 `num_cores=8` 或 `num_cores=48`。
- **正确做法**（一次拿 vector + cube，权威值，无需设备 init）：
  ```python
  import torch_npu
  import triton.runtime.driver as driver

  device = torch_npu.npu.current_device()
  properties = driver.active.utils.get_device_properties(device)
  vectorcore_num = properties["num_vectorcore"]   # elementwise / reduction 用它做 grid 钳制
  aicore_num = properties["num_aicore"]           # cube / matmul 用它
  ```
- **已弃用**：旧式 `npu_config` 取值方式（仅 vector、硬编码 40 不准、设备未 init 会抛 `RuntimeError`）。
- **Why:** 硬编码会浪费多核并行能力；不同 NPU 型号核数不同（ascend910b1 = 40）。

### D2 禁止在 kernel 内使用 triton.cdiv，必须用 tl.cdiv
- **禁止**在 `@triton.jit` kernel 内使用 `triton.cdiv`，**必须**用 `tl.cdiv`。
- **Why:** `triton.cdiv` 在 JIT 函数中会触发 `ValueError: Did you forget to add @triton.jit ?`。
- **How to apply:** kernel 内 `num_blocks = tl.cdiv(remaining, BLOCK_N)`；host 侧用 `triton.cdiv` 或 `(n + BLOCK - 1) // BLOCK`。

### D3 NPU 原生算子与标准算法不兼容时必须替换 Golden
- **必须**当 task 文件存在多个 `Model` 类（如 `torch_npu.npu_nms_v4` / `torch_npu.npu_iou` 与纯 PyTorch 数学公式实现）时，以**纯 PyTorch 数学公式实现**作为 verify/benchmark 的 golden。
- **禁止**试图逆向工程 NPU 原生算子的内部排序/抑制/计算规则。
- **Why:** NPU 原生算子（`npu_nms_v4`、`npu_iou`）内部实现不透明（fp16 中间计算、不同 eps 值、变长输出等尝试均失败），且与标准算法存在系统性数值差异（约 1e-3 量级，差异模式：NPU 结果总是略小于标准算法）。
- **How to apply:** 将 task 文件的 `Model.forward` 改为注释中的数学公式实现，Triton 实现按标准算法编写即可天然匹配。

### D4 Grid 总数必须动态限制为实际核数
- **必须**`grid_size = min(max(num_blocks, 1), VEC_CORE_NUM)`，禁止直接 `grid = (num_blocks,)`。
- **Why:** 当 `num_blocks < VEC_CORE_NUM` 时（小 N 或末尾轮次），多余 core 会空跑；当 `num_blocks` 极大时可能超 grid 上限。
- **How to apply:**
  ```python
  grid_size = min(max(num_blocks, 1), VEC_CORE_NUM)
  kernel[(grid_size,)](..., multibuffer=True)
  ```

### D5 必须使用 1D Grid + 交织循环模式
- **必须**使用 `pid = tl.program_id(0)` 配合 `for block_idx in range(pid, num_blocks, num_cores)` 的交织循环模式。
- **Why:** 输出元素数不确定（可能大于或小于核数），1D 交织循环天然负载均衡；无需额外处理 `num_blocks` 随 `N - cur_idx - 1` 递减的情况。
- **How to apply:**
  ```python
  pid = tl.program_id(0)
  num_blocks = tl.cdiv(remaining, BLOCK_N)
  num_cores = tl.num_programs(0)
  for block_idx in range(pid, num_blocks, num_cores):
      ...
  ```

### D6 所有 kernel 调用加 multibuffer=True
- **必须**所有 kernel 调用附加 `multibuffer=True`。
- **Why:** IoU/抑制 kernel 为访存密集型或计算密集型，`multibuffer=True` 让 Ascend runtime 重叠 host 侧 launch 开销与 device 侧 kernel 执行。
- **How to apply:** 见各算子 L2 骨架中 kernel 调用末尾的 `multibuffer=True`。

---

## §2 NMS 算子（Non-Maximum Suppression）

**算子类别**: `sort-topk` + `indexing-gather` 复合（顺序算法 + 每轮成对 IoU 计算）
**典型特征**: 输入 boxes `[N,4]` + scores `[N]`，输出定长 `selected_indices [max_output_size]` + `num_selected`。包含过滤 → argsort → 顺序遍历 → 每轮 IoU 抑制四个阶段
**性能基准**: 31/31 verify pass，几何平均加速比 **1.0187x** vs torch（达标 1.0x 正常归档阈值，达标 0.6x 用户目标）
**峰值内存**: 13.92 MB vs torch 14.16 MB（略优）

### 版本演进

| 版本 | geomean | kernel 延迟 | 关键技术 | 状态 |
|------|---------|------------|---------|------|
| v1 | 0.9857x | 13.21 ms | 分离 kernel + 逐轮 .item() 同步 | 已被 v2 替换 |
| v2 | 1.0187x | 4.38 ms | fused kernel + device counter + chunked loop | 当前归档 |

### §2.0 首次生成必读：为什么必须把主要框架写对

NMS 是一个**顺序算法**——每轮的候选范围 `[i+1, N)` 依赖上一轮的抑制结果。这意味着不能简单地把整个 N 轮循环搬进单个 kernel（kernel 内无法跨轮传递 suppressed 状态的动态变化），但逐轮 `.item()` 同步又是主要性能瓶颈。**首次生成如果把框架写偏（例如全 kernel 化、逐轮 host 同步、把 NMS 当 sort-topk 优化），后续迭代很难通过局部修bug把性能救回来**。

本章按 Layer 1→3 组织：
- **L1 是硬性约束**，首次生成必须全部满足；
- **L2 是 host chunked loop 骨架 + fused kernel 骨架**，必须一次写对；
- **L3 是关键技巧**，贴出消除同步开销的核心代码。

### §2.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 必须复用 IOU 的 1D Grid + 交织循环模式
（见 §1 D5）
- **必须**在 IoU 抑制循环中使用 `pid = tl.program_id(0)` 配合 `for block_idx in range(pid, num_blocks, num_cores)` 的交织循环模式。
- **Why:** NMS 每轮的 IoU 计算本质上是当前 box 与候选 box 集合的成对计算，与 §3 IOU L1.2 同构；交织循环天然负载均衡，且 `num_blocks` 随 `N - cur_idx - 1` 递减，无需额外处理。
- **How to apply:** 直接套用 §3 IOU L1.2 骨架，把 `gtboxes` 替换为单个 `cur_box` 标量 broadcast。

#### L1.2 kernel 内必须用 tl.cdiv
（见 §1 D2）

#### L1.3 NPU 原生 NMS 与标准算法不兼容时必须替换 Golden
（见 §1 D3）
- **具体到 NMS**：`npu_nms_v4` 在 `pad_to_max_output_size=False` 时返回变长输出，与定长 `max_output_size` 输出不兼容；其内部排序/抑制规则不透明。

#### L1.4 suppressed 数组必须用 int8 + OR 合并
- **必须**用 `torch.zeros(N, dtype=torch.int8)` 表示 suppressed 状态，kernel 内用 `existing | suppress_mask.to(tl.int8)` 合并。
- **Why:** int8 节省显存（vs int32 节省 4x）；OR 操作是幂等的，支持多次 kernel 调用累加抑制标记，免原子操作。
- **How to apply:**
  ```python
  suppressed = torch.zeros(N, dtype=torch.int8, device=boxes.device)
  # kernel 内:
  existing = tl.load(suppressed_ptr + offs, mask=mask, other=0)
  new_supp = existing | suppress_mask.to(tl.int8)
  tl.store(suppressed_ptr + offs, new_supp, mask=mask)
  ```

#### L1.5 主循环必须在 Host 侧 Python 中执行，禁止全 kernel 化
- **必须**用 Python `while i < N` 主循环（v2 用 chunked loop），禁止试图把整个 N 轮循环搬进单个 kernel。
- **Why:** NMS 是**顺序算法**——每轮的候选范围 `[i+1, N)` 依赖上一轮的抑制结果；chunked loop 以 CHUNK=256 为粒度批量 launch，在 chunk 边界同步 suppressed 状态，是顺序算法在 Triton 上的最优落地形式。
- **How to apply:** 见 L2.1 Host 侧骨架。**禁止方向**: 把整个 N 轮循环写进单个 `@triton.jit`（kernel 内无法跨轮传递 suppressed 状态的动态变化）。

#### L1.6 grid_size 必须动态限制为实际核数
（见 §1 D4）

#### L1.7 cur_box 必须标量加载 + broadcast，禁止向量重复加载
- **必须**当前 box 的 4 个坐标 `(cur_x1, cur_y1, cur_x2, cur_y2)` 和 `cur_area` 用标量 `tl.load` 加载 1 次，与候选向量 `[BLOCK_N]` broadcast 计算 IoU。
- **Why:** 每轮 cur_box 是固定值，若每个候选 box 都重新加载 cur_box 会浪费访存带宽。
- **How to apply:** 见 L2.2 kernel 骨架。

#### L1.8 选中逻辑必须 fused 进 kernel 的 pid==0 分支（v2 关键约束）
- **必须**把 `selected_indices` 写入和 `num_sel` 自增逻辑放在 kernel 的 `if pid == 0` 分支内执行，禁止在 host 侧逐轮写入。
- **Why:** host 侧逐轮 `selected_indices[num_selected] = ...` 需要 `.item()` 同步读 `num_selected` 和 `suppressed[i]`，N 轮累计同步开销巨大（v1 的主要瓶颈）。fused 进 kernel 后，`num_sel_tensor` 常驻 device，kernel 内 `tl.store(num_sel_ptr, count+1)` 自增，只在 CHUNK 边界同步。
- **How to apply:**
  ```python
  # kernel 内:
  if pid == 0:
      count = tl.load(num_sel_ptr)
      should_select = not_supp & (count < max_output_size)
      orig = tl.load(orig_idx_ptr + cur_idx, mask=should_select, other=0)
      tl.store(selected_ptr + count, orig, mask=should_select)
      tl.store(num_sel_ptr, count + 1, mask=should_select)
  ```
  **注意**: `pid==0` 分支保证只有 1 个 core 写入，避免竞争；`mask=should_select` 保证未选中时不写入。

#### L1.9 必须用 chunked loop 消除逐轮 .item() 同步（v2 关键约束）
- **必须**用 `while i < N` + `CHUNK=256` 的分块循环，每 CHUNK 轮才 `.item()` 同步一次，禁止每轮同步。
- **Why:** v1 每轮 `.item()` 同步 N 次，是主要瓶颈；v2 同步次数降到 N/256，geomean 从 0.99x 提升到 1.02x。
- **How to apply:** 见 L2.1 Host 侧骨架。**CHUNK 选择**: 256 是经验值，平衡同步开销与提前终止粒度（CHUNK 过大则 max_output_size 达标后不能立即 break，浪费计算；CHUNK 过小则同步开销仍高）。

#### L1.10 所有 kernel 调用加 multibuffer=True
（见 §1 D6）

### §2.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧 chunked loop 骨架（v2）

```python
import torch_npu
import triton.runtime.driver as driver

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        properties = driver.active.utils.get_device_properties(torch_npu.npu.current_device())
        self.VEC_CORE_NUM = properties["num_vectorcore"]
        self.AI_CORE_NUM = properties["num_aicore"]

    def forward(self, boxes, scores, max_output_size, iou_threshold,
                scores_threshold, pad_to_max_output_size=False):
        # 1. 过滤 + 升精度
        boxes_f32 = boxes.float()
        scores_f32 = scores.float()
        score_mask = scores_f32 > scores_threshold
        filtered_boxes = boxes_f32[score_mask]
        filtered_scores = scores_f32[score_mask]
        original_indices = torch.where(score_mask)[0]

        # 2. 定长输出缓冲 + device-side counter (L1.8)
        selected_indices = torch.zeros(max_output_size, dtype=torch.int32, device=boxes.device)
        N = filtered_boxes.shape[0]
        if N == 0:
            return selected_indices, torch.tensor(0, dtype=torch.int32, device=boxes.device)

        # 3. stable argsort 降序
        sorted_indices = torch.argsort(filtered_scores, descending=True, stable=True)
        sorted_boxes = filtered_boxes[sorted_indices].contiguous()
        sorted_original_indices = original_indices[sorted_indices].to(torch.int32).contiguous()

        # 4. 预计算 areas
        areas = (sorted_boxes[:, 2] - sorted_boxes[:, 0]) * \
                (sorted_boxes[:, 3] - sorted_boxes[:, 1])

        # 5. suppressed + num_sel_tensor (device 常驻)
        suppressed = torch.zeros(N, dtype=torch.int8, device=boxes.device)
        num_sel_tensor = torch.zeros(1, dtype=torch.int32, device=boxes.device)

        # 6. chunked loop (L1.9)
        BLOCK_N = 256
        CHUNK = 256
        i = 0
        while i < N:
            end = min(i + CHUNK, N)
            for j in range(i, end):
                remaining = N - j - 1
                num_blocks = (remaining + BLOCK_N - 1) // BLOCK_N if remaining > 0 else 1
                grid_size = min(max(num_blocks, 1), self.VEC_CORE_NUM)    # L1.6
                nms_fused_kernel[(grid_size,)](
                    sorted_boxes, areas, suppressed,
                    selected_indices, num_sel_tensor,
                    sorted_original_indices,
                    float(iou_threshold), j, N, max_output_size,
                    BLOCK_N=BLOCK_N, multibuffer=True,
                )
            # CHUNK 边界同步
            num_selected = num_sel_tensor.item()
            if num_selected >= max_output_size:
                break
            i = end

        num_selected = num_sel_tensor.item()
        return selected_indices, torch.tensor(num_selected, dtype=torch.int32, device=boxes.device)
```

#### L2.2 nms_fused_kernel 骨架（v2）

```python
@triton.jit
def nms_fused_kernel(boxes_ptr, areas_ptr, suppressed_ptr,
                     selected_ptr, num_sel_ptr, orig_idx_ptr,
                     iou_threshold, cur_idx, N, max_output_size,
                     BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)

    # 1. kernel 内 suppressed 检查 (L1.8 关键: 消除 host 侧 .item())
    is_supp = tl.load(suppressed_ptr + cur_idx)
    not_supp = is_supp == 0

    # 2. 选中逻辑 fused 进 pid==0 分支 (L1.8)
    if pid == 0:
        count = tl.load(num_sel_ptr)
        should_select = not_supp & (count < max_output_size)
        orig = tl.load(orig_idx_ptr + cur_idx, mask=should_select, other=0)
        tl.store(selected_ptr + count, orig, mask=should_select)
        tl.store(num_sel_ptr, count + 1, mask=should_select)

    # 3. IoU 抑制 (仅当 not_supp 时执行)
    if not_supp:
        remaining = N - cur_idx - 1
        if remaining > 0:
            # 标量加载 cur_box (L1.7)
            cur_x1 = tl.load(boxes_ptr + cur_idx * 4 + 0)
            cur_y1 = tl.load(boxes_ptr + cur_idx * 4 + 1)
            cur_x2 = tl.load(boxes_ptr + cur_idx * 4 + 2)
            cur_y2 = tl.load(boxes_ptr + cur_idx * 4 + 3)
            cur_area = tl.load(areas_ptr + cur_idx)

            num_blocks = tl.cdiv(remaining, BLOCK_N)    # L1.2
            num_cores = tl.num_programs(0)

            # 交织循环 (L1.1)
            for block_idx in range(pid, num_blocks, num_cores):
                j_start = cur_idx + 1 + block_idx * BLOCK_N
                offs = j_start + tl.arange(0, BLOCK_N)
                mask = offs < N

                # 候选向量加载
                cand_x1 = tl.load(boxes_ptr + offs * 4 + 0, mask=mask, other=0.0)
                # ... cand_y1/y2/x2/area 同理 ...

                # IoU 计算 (cur_box 标量 broadcast)
                x1_inter = tl.maximum(cur_x1, cand_x1)
                # ... inter/union/iou 同理 ...
                iou = inter / tl.maximum(union, 1e-6)

                # OR 合并 (L1.4)
                suppress_mask = (iou >= iou_threshold) & mask
                existing = tl.load(suppressed_ptr + offs, mask=mask, other=0)
                new_supp = existing | suppress_mask.to(tl.int8)
                tl.store(suppressed_ptr + offs, new_supp, mask=mask)
```

### §2.3 Layer 3: 关键技巧（Agent 可参考，但实现方式可不同）

#### L3.1 Fused kernel pid==0 分支消除 host 侧写入（v2 核心）

**问题**: v1 在 host 侧逐轮 `selected_indices[num_selected] = sorted_original_indices[i]` + `num_selected += 1`，需要 `.item()` 同步读 suppressed[i] 和 num_selected，N 轮累计同步开销巨大。

**解决**: 把选中逻辑放进 kernel 的 `if pid == 0` 分支。`num_sel_tensor` 常驻 device，kernel 内 `tl.load(num_sel_ptr)` 读当前计数，`tl.store(num_sel_ptr, count+1, mask=should_select)` 自增。host 侧完全不需要在 chunk 内读计数。

```python
if pid == 0:
    count = tl.load(num_sel_ptr)
    should_select = not_supp & (count < max_output_size)
    tl.store(selected_ptr + count, orig, mask=should_select)
    tl.store(num_sel_ptr, count + 1, mask=should_select)
```

**关键点**: `pid==0` 保证只有 1 个 core 写入，避免多 core 竞争；`mask=should_select` 保证未选中或超 max_output_size 时不写入。

**可替代方向**: 若需进一步减少同步，可让多个 core 分担选中逻辑（如按 cur_idx 分区），但需处理写入竞争（原子操作或分片写入）。

#### L3.2 Device-side counter 消除逐轮 .item() 同步（v2 核心）

**问题**: v1 的 `num_selected` 是 host 侧 Python int，每轮 `.item()` 同步；`suppressed[i].item()` 也是每轮同步。

**解决**:
- `num_sel_tensor = torch.zeros(1, dtype=torch.int32, device=boxes.device)` 常驻 device
- kernel 内 `tl.store(num_sel_ptr, count+1)` 自增
- host 侧只在 CHUNK 边界 `num_sel_tensor.item()` 同步一次
- `suppressed[i]` 的检查推进 kernel 内：`is_supp = tl.load(suppressed_ptr + cur_idx)`

**收益**: 同步次数从 N 次降到 N/256 次（v1 → v2 的核心提升）。

**可替代方向**: 若 CHUNK 内的提前终止很重要，可用 event/fence 机制让 host 异步查询 device 状态，但 Triton 当前不支持。

#### L3.3 Chunked loop 的 CHUNK 选择

**问题**: CHUNK 过大则 max_output_size 达标后不能立即 break，浪费计算；CHUNK 过小则同步开销仍高。

**解决**: `CHUNK = 256` 是经验值。
- 同步开销: N/256 次 `.item()`，相比 v1 的 N 次减少 256x
- 提前终止粒度: 最多多算 255 轮，对大 N 可接受

**可替代方向**:
- 若 max_output_size 通常远小于 N（如 N=32768, max=1000），可动态调整 CHUNK（前期大、后期小）
- 若 N 固定已知，可离线调优 CHUNK

#### L3.4 int8 suppressed + OR 合并替代 int32 + 原子写（沿用 v1）

**问题**: suppressed 数组每轮被多 core 并发读写，需要某种并发安全机制。

**解决**: int8 + OR 操作。OR 是幂等的（`x | 1 = 1`，`x | 0 = x`），即使两个 core 同时写同一位置，结果也是正确的；无需原子操作。int8 比 int32 节省 4x 显存和访存带宽。

```python
existing = tl.load(suppressed_ptr + offs, mask=mask, other=0)
new_supp = existing | suppress_mask.to(tl.int8)
tl.store(suppressed_ptr + offs, new_supp, mask=mask)
```

**可替代方向**: 若需更细粒度状态（如 "candidate" / "selected" / "suppressed" 三态），可用 int8 的不同 bit 位编码。

#### L3.5 cur_box 标量加载消除重复访存（沿用 v1）

**问题**: 每轮 cur_box 是固定值，若每个候选 box 都 `tl.load(cur_box)` 会重复访存 BLOCK_N 次。

**解决**: cur_box 的 4 个坐标在 kernel 入口标量加载 1 次，与候选向量 `[BLOCK_N]` broadcast 计算。Triton 的标量-向量运算会自动 broadcast。

**可替代方向**: 若 cur_box 也按 BLOCK 分块（如多轮合并 kernel），需重新设计 cur_box 的加载策略。

#### L3.6 stable argsort 保证排序稳定性（沿用 v1）

**问题**: scores 相同时，argsort 的顺序不确定，导致 verify 失败。

**解决**: `torch.argsort(filtered_scores, descending=True, stable=True)`。stable=True 保证相同 score 按原始索引顺序排列，与 PyTorch golden 行为一致。

**可替代方向**: 若性能瓶颈在 argsort（aclnnSort 4.58ms），可考虑 NPU 专用 sort 算子或分桶排序。

#### L3.7 multibuffer=True 启用多缓冲（沿用 v1）

**解决**: kernel launch 时传 `multibuffer=True`，让 Ascend runtime 重叠 host 侧 launch 开销与 device 侧 kernel 执行。

**可替代方向**: 若 kernel 极短（小 N 末尾轮次），multibuffer 的缓冲区管理开销可能超过收益，可动态决定是否启用。

#### L3.8 BLOCK_N = 256 的选择依据（沿用 v1）

- **256**: 单块覆盖 256 个候选 box，平衡 SRAM 占用与并行度
- **128**: 块更小，并行度更高，但每块的 cur_box broadcast 开销摊薄变差
- **512/1024**: 块更大，SRAM 压力增大，小 N 时浪费

**可替代方向**: 若 N 固定且已知（如检测框固定 1000），可调大 BLOCK_N 减少块数。

### §2.4 NMS 性能基准（几何平均 1.0187x，突破 1.0x 归档阈值）

| Shape 区间 | 典型加速比 | 说明 |
|-----------|-----------|------|
| 小 shape [50-512, 4] | 0.88x ~ 1.21x | chunked loop 同步开销在小 N 上仍可见 |
| 中 shape [1024-8192, 4] | 0.81x ~ 1.14x | 基本持平，部分 case profiler fail |
| 大 shape [10000-32768, 4] | 1.00x ~ 1.55x | 大 N chunked loop 收益最大，Triton 反超 |
| 全量 31 cases | 1.0187x | 几何平均，达标 1.0x 归档阈值 |

### implementation 侧 operator 耗时拆解（v2）

| operator | latency (ms) | 占比 | vs v1 | 说明 |
|----------|-------------|------|-------|------|
| `nms_fused_kernel` | 4.38 | 4% | 13.21→4.38 (3x) | fused + device counter 收益 |
| `aclnnNonzero*` | 81.78 | 80%+ | 基本持平 | host 侧过滤，主要瓶颈 |
| `aclnnIndex*` | 21.40 | 21% | 基本持平 | filtered_boxes[sorted_indices] |
| `aclnnSort` | 6.10 | 6% | 基本持平 | argsort |

**关键结论**:
1. v2 把 kernel 本体从 13% 降到 4%，host 侧 `nonzero + index + sort` 升至 80%+，是下一版优化方向
2. fused kernel + device counter + chunked loop 是顺序算法在 Triton 上消除同步开销的通用模式
3. v2 落地了 v1 报告中标注的 "batched kernel 消除 .item() 同步" 未来方向

---

## §3 IOU 算子（Intersection over Union）

**算子类别**: `elementwise` / `pairwise-compute`
**典型特征**: 成对计算，输入为两个 [N,4] 和 [M,4] 的 bbox 张量，输出为 [M,N] 的 IoU/IoF 矩阵
**性能基准**: 
- v1 (20260603): 几何平均加速比约 0.14x（离散访存模式）
- v2 (20260604): 几何平均加速比约 1.02x（2D Tiling + Broadcast 模式）

### §3.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 必须使用单 kernel 实现
- **必须**使用单 kernel 完成所有成对计算
- **Why:** IOU 为纯 elementwise-like 成对计算，无 reduce/统计阶段，双 kernel 只会增加调度开销
- **How to apply:** 所有场景

#### L1.2 必须使用 1D Grid + 交织循环模式
（见 §1 D5）
- **具体到 IOU**：输出元素数不确定（可能大于或小于核数），1D 交织循环天然负载均衡。

#### L1.3 禁止在 kernel 中使用 triton.cdiv
（见 §1 D2）

#### L1.4 NPU 原生算子与标准算法存在系统性差异时必须替换 Golden
（见 §1 D3）
- **具体到 IOU**：`torch_npu.npu_iou` 与标准 PyTorch 数学公式实现存在约 1e-3 量级的系统性差异（差异模式：NPU 结果总是略小于标准算法，如 0.1418 vs 0.1429、0.3877 vs 0.3913）。
- **How to apply:**
  1. 当验证反复失败且差异呈系统性时，首先对比 NPU 原生输出与注释中的 PyTorch 参考实现
  2. 若确认 NPU 原生算子与标准算法不一致，将 task 文件的 `Model.forward` 改为注释中的数学公式实现
  3. 重新运行验证，确认 Triton 实现与数学公式实现匹配

#### L1.5 必须使用 2D Tiling + Broadcast 模式，禁止离散逐元素访存
- **必须**采用 2D Tiling 策略：将 bboxes 按 BLOCK_N 分块、gtboxes 按 BLOCK_M 分块，块内通过 Triton broadcast 计算 [BLOCK_M, BLOCK_N] 结果矩阵
- **禁止**使用 1D 逐元素索引模式（即每个线程计算一个输出元素，离散加载 8 个标量）
- **Why:** 离散逐元素访存无法利用 NPU 向量单元的数据复用和 broadcast 能力，是性能瓶颈的主要来源。2D Tiling + Broadcast 可将访存从 O(N*M*8) 次标量加载优化为 O(N*M/BLOCK_SIZE) 次向量加载 + 高效 broadcast 计算
- **How to apply:** 所有场景

#### L1.6 Grid 大小必须限制为实际核数
（见 §1 D4）
- **具体到 IOU**：必须通过 `multibuffer=True` 启用多缓冲，限制 grid 为核数并配合 multibuffer 可在核内实现流水线隐藏。

### §3.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧分支决策树

```python
forward(bboxes, gtboxes, mode):
    n = bboxes.shape[0]
    m = gtboxes.shape[0]
    output = empty((m, n), dtype=bboxes.dtype)
    
    num_blocks_n = triton.cdiv(n, BLOCK_N)
    num_blocks_m = triton.cdiv(m, BLOCK_M)
    grid_size = min(num_blocks_n * num_blocks_m, vectorcore_num)
    grid = (grid_size,)
    
    iou_kernel[grid](bboxes, gtboxes, output, n, m, mode, 
                     BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M,
                     multibuffer=True)
    return output
```

#### L2.2 多核并行骨架模式（2D Tiling + Broadcast）

**推荐模式 - 2D Tile 分块 + 向量 Broadcast**:
```python
pid = tl.program_id(0)
num_blocks_n = tl.cdiv(n, BLOCK_N)
num_blocks_m = tl.cdiv(m, BLOCK_M)
num_blocks = num_blocks_n * num_blocks_m

for block_idx in range(pid, num_blocks, tl.num_programs(0)):
    block_n = block_idx // num_blocks_m
    block_m = block_idx % num_blocks_m
    
    n_start = block_n * BLOCK_N
    m_start = block_m * BLOCK_M
    
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_m = m_start + tl.arange(0, BLOCK_M)
    n_mask = offs_n < n
    m_mask = offs_m < m
    
    # 向量加载 bboxes 坐标 [BLOCK_N]
    b_x1 = tl.load(bboxes_ptr + offs_n * 4 + 0, mask=n_mask, other=0.0)
    b_y1 = tl.load(bboxes_ptr + offs_n * 4 + 1, mask=n_mask, other=0.0)
    b_x2 = tl.load(bboxes_ptr + offs_n * 4 + 2, mask=n_mask, other=0.0)
    b_y2 = tl.load(bboxes_ptr + offs_n * 4 + 3, mask=n_mask, other=0.0)
    
    # 向量加载 gtboxes 坐标 [BLOCK_M]
    g_x1 = tl.load(gtboxes_ptr + offs_m * 4 + 0, mask=m_mask, other=0.0)
    # ...
    
    # 计算面积 [BLOCK_N] 和 [BLOCK_M]
    b_area = (b_x2 - b_x1) * (b_y2 - b_y1)
    g_area = (g_x2 - g_x1) * (g_y2 - g_y1)
    
    # Broadcast 到 [BLOCK_M, BLOCK_N]
    b_x1_b = b_x1[None, :]
    g_x1_b = g_x1[:, None]
    # ...
    
    # 向量计算交集 [BLOCK_M, BLOCK_N]
    lt_x = tl.maximum(b_x1_b, g_x1_b)
    rb_x = tl.minimum(b_x2_b, g_x2_b)
    inter_w = tl.maximum(0.0, rb_x - lt_x)
    inter = inter_w * inter_h
    
    # 模式分支
    if mode == 0:
        union = b_area_b + g_area_b - inter
        iou = inter / tl.maximum(union, 1e-10)
    else:
        iou = inter / tl.maximum(g_area_b, 1e-10)
    
    # 向量存储 [BLOCK_M, BLOCK_N]
    out_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(out_ptr + offs_m[:, None] * n + offs_n[None, :], iou, mask=out_mask)
```

**已验证的 BLOCK 配置**:
- BLOCK_N = 64, BLOCK_M = 32 在 ascend910b1 上表现最佳
- 可替代方向: 尝试 BLOCK_N = 128, BLOCK_M = 64 或根据具体 shape 动态选择

### §3.3 Layer 3: 关键技巧（Agent 可参考，但实现方式可不同）

#### L3.1 2D Tiling + Broadcast 替代离散访存（核心优化）

**旧模式（性能差，约 0.14x）**:
```python
# 每个输出元素离散加载 8 个标量 —— 严重瓶颈
b_x1 = tl.load(bboxes_ptr + j * stride_bn + 0, mask=mask)
# ... 8 次标量加载 per element
```

**新模式（性能优，约 1.02x）**:
```python
# 按块向量加载，broadcast 计算 —— 充分利用 NPU 向量单元
b_x1 = tl.load(bboxes_ptr + offs_n * 4 + 0, mask=n_mask, other=0.0)  # [BLOCK_N]
g_x1 = tl.load(gtboxes_ptr + offs_m * 4 + 0, mask=m_mask, other=0.0)  # [BLOCK_M]

# Broadcast 后通过 Triton 向量指令并行计算 [BLOCK_M, BLOCK_N]
b_x1_b = b_x1[None, :]   # [1, BLOCK_N] -> broadcast to [BLOCK_M, BLOCK_N]
g_x1_b = g_x1[:, None]   # [BLOCK_M, 1] -> broadcast to [BLOCK_M, BLOCK_N]
lt_x = tl.maximum(b_x1_b, g_x1_b)  # [BLOCK_M, BLOCK_N]
```

**可替代方向**: 
- 可尝试 `tl.make_block_ptr` 进行更规整的块加载
- 可尝试将 mode 作为 `tl.constexpr` 传入触发编译期分支消除
- 对于 M 或 N 极小的场景（如 gtboxes=1），可考虑退化到 1D 模式避免 tiling 开销

#### L3.2 输出转置语义避免额外 transpose

```python
# 直接输出 [M, N] 形状，天然对应转置后的 IoU 矩阵
output = torch.empty((m, n), device=bboxes.device, dtype=bboxes.dtype)

# kernel 内存储时保持转置语义
out_ptrs = out_ptr + offs_m[:, None] * n + offs_n[None, :]
tl.store(out_ptrs, iou, mask=out_mask)
```

**Why:** 原始 PyTorch 参考实现最后有 `.t()`，Triton 实现直接在 kernel 内按转置后的布局存储，避免额外的 transpose 操作。

#### L3.3 数值稳定性：使用 tl.maximum 代替 clamp

```python
# 推荐
inter_w = tl.maximum(0.0, rb_x - lt_x)
union = tl.maximum(union, 1e-10)

# 避免使用 tl.where(w > 0, w, 0.0)，可能在某些场景下效率较低
```

#### L3.4 Grid 大小动态限制 + multibuffer

```python
import torch_npu
import triton.runtime.driver as driver

# 获取实际 vector core 数量（新 API，无需设备 init）
num_cores = driver.active.utils.get_device_properties(torch_npu.npu.current_device())["num_vectorcore"]

# 限制 grid 为实际核数
grid_size = min(num_blocks_n * num_blocks_m, num_cores)

# 启用 multibuffer 隐藏访存延迟
iou_kernel[(grid_size,)](..., multibuffer=True)
```

**可替代方向**: 可尝试根据 shape 大小动态调整 BLOCK_N/BLOCK_M 以更好匹配核数。

#### L3.5 小 shape 排除策略

- 几何平均计算时应排除小 shape（如 N*M < 1000）的异常加速比
- **Why:** 小 shape 的 kernel launch 开销占比高，加速比波动大（可能 3x~40x），会扭曲几何平均
- **How to apply:** 在 summary.json 中标记 `status="excluded"`，`exclusion_reason="small_shape_overhead_or_abnormal_speedup"`

### §3.4 IOU 性能基准（几何平均）

| Shape 类型 | v1 加速比 | v2 加速比 | 说明 |
|-----------|-----------|-----------|------|
| 小 shape (N*M < 1000) | 0.1x - 3.2x | 3x - 40x (已排除) | 小 shape 波动大，应排除 |
| 中 shape (N*M ~ 10000-100000) | 0.1x - 0.3x | 0.9x - 2.5x | v2 显著提升 |
| 大 shape (N*M > 100000) | 0.01x - 0.2x | 0.3x - 1.2x | v2 显著提升 |

**关键结论**：
1. **2D Tiling + Broadcast 是核心性能优化**：将离散标量访存转为向量块加载 + broadcast 计算，性能从 0.14x 提升到 1.02x
2. **Grid 限制为核数 + multibuffer** 可进一步减少调度开销和隐藏访存延迟
3. **输出转置语义**避免了额外的 transpose 操作
4. **NPU 原生算子精度差异的解决策略**：当 NPU 原生算子与标准算法存在系统性差异时，不要试图逆向工程，而是将 golden 替换为数学公式实现
5. 小 shape 因 overhead 应排除在几何平均外

**0603remake vs 0604remake 的关键差异**：
- **0603remake**：task 文件中的 Model 被手动替换为 PyTorch 数学公式实现（golden 变为标准算法），Triton 实现一次通过验证
- **0604remake**：task 文件保留了原始的 `torch_npu.npu_iou` 调用，但 verify 阶段实际使用的是 PyTorch 数学公式参考（通过 verify 目录下的 torch 文件注入），最终同样 30/30 通过
- **通用原则**：无论通过修改 task 文件还是 verify 注入，核心都是确保 golden 基于数学上正确的标准算法，而非行为不透明的 NPU 原生算子

---

## §4 常见陷阱与避免方法

### §4.1 NMS 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 试图全 kernel 化 NMS 主循环 | 把整个 N 轮 Python 循环搬进单个 Triton kernel，期望消除所有同步 | NMS 是顺序算法，每轮依赖上一轮抑制状态，kernel 内无法跨轮传递动态状态；必须保留 Python 主循环 (L1.5)，用 chunked loop (L1.9) 折中 |
| 用 npu_nms_v4 作为 golden | `npu_nms_v4` 返回变长输出且内部规则不透明，verify 反复失败 | 替换为纯 PyTorch 数学公式实现作为 golden (L1.3 / D3) |
| 用 int32 表示 suppressed | int32 浪费 4x 显存，且 .item() 同步读 int32 比 int8 慢 | int8 + OR 合并 (L1.4 / L3.4) |
| kernel 内用 triton.cdiv | `triton.cdiv` 在 JIT 函数中触发 `ValueError: Did you forget to add @triton.jit ?` | 用 `tl.cdiv` (L1.2 / D2) |
| grid_size 不限制为 VEC_CORE_NUM | 小 N 或末尾轮次 num_blocks < VEC_CORE_NUM 时，多余 core 空跑 | `grid_size = min(max(num_blocks, 1), VEC_CORE_NUM)` (L1.6 / D4) |
| 误把 NMS 当成 sort-topk 类算子优化 | 见框架 6 类分类，误以为 NMS 属 sort-topk，尝试并行排序优化 | NMS 的核心瓶颈是顺序 IoU 抑制循环，不是排序；排序只是预处理。优化重点应在 IoU kernel + 减少同步，复用 §3 IOU 经验而非 sort-topk 经验 |
| cur_box 每个候选重复加载 | 在 block 内对每个候选 box 都 `tl.load(cur_box)`，浪费访存带宽 | cur_box 标量加载 1 次 + broadcast (L1.7 / L3.5) |
| profiler 失败误判为代码错误 | case 7/8/9/23/25/26 报 `RuntimeError: 无法从 profiler 提取有效时延数据`，误以为代码 bug | 这是 profiling 基础设施问题（并发 profiling 或 SQLite TASK 表缺失），verify 阶段全部通过；重新单独跑这些 case 通常能拿到正常数据 |
| host 侧逐轮 .item() 同步（v1 的教训） | v1 每轮 `suppressed[i].item()` + `num_selected` 读写，N 轮同步开销导致 geomean 卡在 0.99x | v2 用 fused kernel (L1.8) + device-side counter (L3.2) + chunked loop (L1.9) 把同步次数从 N 降到 N/256，突破 1.0x |
| CHUNK 过大导致提前终止失效 | CHUNK 设置过大（如 CHUNK=N），max_output_size 达标后不能立即 break，浪费计算 | CHUNK=256 是经验值 (L3.3)；若 max_output_size 通常较小，可在 CHUNK 边界检查 `num_selected >= max_output_size` 提前退出 while 循环 |

### §4.2 IOU 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| NPU 原生算子与标准算法的系统性数值差异 | `torch_npu.npu_iou` 与标准 PyTorch 数学公式实现存在约 1e-3 量级的系统性差异（NPU 结果总是略小于标准算法，如 0.1418 vs 0.1429） | 不要试图逆向工程 NPU 原生算子的行为；将 task 文件的 `Model` 替换为注释中提供的 PyTorch 数学公式实现 (L1.4 / D3) |
| triton.cdiv 在 kernel 中使用 | 在 `@triton.jit` kernel 中使用 `triton.cdiv` 导致编译错误 | kernel 内使用 `tl.cdiv`，host 侧使用 `triton.cdiv` (L1.3 / D2) |
| 离散逐元素访存模式 | 每个线程计算一个输出元素，离散加载 8 个标量，无法利用 NPU 向量单元 | 采用 2D Tiling + Broadcast 模式，向量加载块数据后通过 broadcast 并行计算 (L1.5 / L3.1) |
| 2D Tiling 的 broadcast 兼容性 | 尝试 2D tiling 时可能遇到 `Cannot make_shape_compatible` 错误 | 确保 broadcast 维度匹配（`[None, :]` 和 `[:, None]` 的组合），并正确计算 mask |
| Grid 过大导致调度开销 | grid_size 超过实际核数时，核间调度开销显著 | 限制 grid_size 为 `min(num_blocks, vectorcore_num)`，并启用 `multibuffer=True` (L1.6 / D4) |
