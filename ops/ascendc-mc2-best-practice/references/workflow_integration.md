# MC2 通算融合算子开发工作流

本文档是 SHMEM/UDMA 路径（§1）的技术参考，不感知调用方流程。涵盖环境校验、工程起手、约束确认、性能采集等 MC2 场景的差异化要点。

> 通信范式选择和路由见 SKILL.md 决策树。本文档仅适用于 SHMEM/UDMA 路径。

---

## 1. 环境校验

| 校验项 | 命令 / 方法 | 失败处理 |
|--------|------------|---------|
| **NPU 架构必须为 dav-3510** | `npu-smi info` 读 `Chip Name` 应为 `Ascend 950` 系列；CMake 阶段校验 `NPU_ARCH=dav-3510` | 非 3510 直接终止，告知用户"SHMEM 路径仅支持 Ascend 950 (dav-3510)" |
| **SHMEM 第三方库可解析** | 检查 `references/all_to_all_matmul/third_party/shmem/CMakeLists.txt` 是否存在；若不存在，`cmake/shmem.cmake` 会自动 `git clone --branch v1.5.0`（gitcode.com/cann/shmem） | 提示用户检查网络访问 gitcode.com |
| **tensor_api（`AscendC::Te::*`）可解析** | 检查 `third_party/tensor_api/include/tensor_api/tensor.h` 是否存在；若不存在，`cmake/tensor_api.cmake` 会自动 `git clone` asc-devkit | 提示用户检查网络访问 gitcode.com |
| **Blaze 头文件可解析** | blaze 头（`blaze/gemm/block/block_mmad*.h`）位于 CANN toolkit 的 `opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common`，由 CMakeLists 的 `_BLAZE_COMMON_DIR` 指向；无需 clone | 提示用户检查 CANN 安装路径是否正确 |
| **多卡环境可用** | `npu-smi info` 至少能看到 `rankNum` 张卡（默认 4 卡） | 提示用户准备多卡环境；单卡只能跑精度模式，性能模式需多卡 |

---

## 2. 约束确认

设计文档中应显式确认以下两大约束（SKILL.md §1 红线）：

```markdown
## 两大约束确认

### 约束 ①：通信走 SHMEM，禁止 HCCL 高阶 API
- 选用的 SHMEM 原语：<aclshmemx_udma_put_nbi / aclshmem_barrier_all / ...>
- 确认无 Hccl:: 高阶 API：✅

### 约束 ②：Matmul 走 Blaze 模板，禁止 asc-devkit Matmul 高阶 API
- 选用的 Blaze 模板：<Blaze::Gemm::Block::BlockMmad + Kernel::QuantMatmulMxKernelSwat 等>
- DispatchPolicy：<MatmulWithScaleMx / ...>
- 确认无 asc-devkit Matmul 高阶 API：✅
```

---

## 3. 切分策略

- **两阶段 `tileCnt` 策略**（详见 [`pipeline_tuning.md`](pipeline_tuning.md)）：
  - 设计/审查阶段：`tileCnt=1`（`headMSize=m`）做串行基线，简化精度/审查调试；
  - 性能验收阶段：扫描 `tileCnt ∈ {1, 2, 4, 8, 16, 32}` 找最优，`headMSize=512` 只是起点不是最优值。
- M 轴切分：`headMSize = m / tileCnt`，`bufferSize=4`
- SHMEM 空间预算（默认 1 GB，需够装下 `rankSize * bufferSize * bufferBlockSize + scale`）

### AIV/AIC 分工

明确哪些 work 在 AIV（通信），哪些在 AIC（计算），同步点在哪里（`CrossCoreSetFlag<0x2, PIPE_*>`，其中 `0x2` 是 modeId 即 AIV↔AIC 跨核同步模式）。

---

## 4. 工程起手流程

```bash
# 1. 从基底工程复制
cp -r references/all_to_all_matmul operators/{op_name}

# 2. 按 [MODIFY] 标记定点改造（详见 codebase_map.md）
#    - src/{op_name}.cpp：函数名、kernel 调用、参数解析
#    - include/kernel/all_to_all_comm_udma.h：通信原语（若非 AllToAll）
#    - include/kernel/all_to_all_matmul_impl.h：通算融合主类
#    - include/kernel/qbmm_mx_kernel.h：Blaze Kernel 包装（改 Scale/dtype/bias 时动）
#    - include/tiling/*：TilingData 字段
#    - scripts/gen_data.py + verify_result.py：dtype/容差

# 3. 编译
cd operators/{op_name}
cmake -S . -B build -DNPU_ARCH=dav-3510
cmake --build build -j

# 4. 跑精度
bash run.sh  # 默认 m=2048 k=8192 n=3584 rank=4 precision
```

### 开发阶段红线

- **禁止**新增 `Hccl::` 调用；
- **禁止**包含 asc-devkit 的 Matmul 高阶 API 头（`matmul.h`）；
- **禁止**为了赶进度跳过 `run.sh` 的精度模式直接跑 perf；
- 改完每个 `[MODIFY]` 文件后立即跑一次 `bash run.sh` 做冒烟。

---

## 5. 代码审查速查

按 SKILL.md 的 R1~R6 逐项检查：

```bash
# R1 架构=3510
grep -r "npu-arch" operators/{op}/CMakeLists.txt

# R2 无 HCCL 高阶 API（应为空）
grep -rn "Hccl::" operators/{op}/

# R3 无 asc-devkit matmul API（应为空）
grep -rn "AscendC::Matmul\b" operators/{op}/

# R4 通信走 SHMEM
grep -rn "shmem.h\|aclshmem" operators/{op}/include/kernel/

# R5 Matmul 走 Blaze
grep -rn "blaze/gemm/block/" operators/{op}/include/

# R6 L2 flush 证据（性能验收时检查 heavy_kernels.h 是否被正确 include）
grep -rn "heavy_add_kernel\|cache_flush" operators/{op}/src/
```

### 常见问题

| 现象 | 根因 | 修复方向 |
|------|------|---------|
| grep 到 `Hccl::AllReduce` | 开发者把 HCCL 当成"熟悉的 API"用了 | 改写为 `aclshmemx_udma_put_nbi` + 自实现 Reduce 逻辑 |
| grep 到 `AscendC::Matmul` | 开发者误用 asc-devkit 接口 | 替换为 `Blaze::Gemm::Block::BlockMmad` |
| SHMEM 空间不足崩溃 | 空间预算没算对 | 重算 `SHMEM_SPACE_SIZE` |
| 精度对不上但无报错 | ProcessSingleBatch 中 rank==rankId 分支错（未切换到本卡 GM）/ `remoteRankCnt` 没从 0 起算 | 对照 `qbmm_mx_kernel.h` 注释核对 |
| 跨核同步 flag 错位 | `CrossCoreSetFlag<0x2, PIPE_MTE3>(mLoopIdx)` 与 `CrossCoreWaitFlag<0x2, PIPE_MTE2>(mLoopIdx)` 的 flagId 必须一致 | 检查 mLoopIdx 两侧是否对应 |
| UDMA Put 数据未到达 | `aclshmemx_udma_quiet(remoteRank)` 必须在每次 Put 后调用 | 确保 quiet 调用未被遗漏 |
| remoteRankCnt 错位 | `splitKNum` 必须等于 `rankSize`，否则 fixpipe 时机不对 | 检查 SetupParams 中 `splitKNum = rankSize_` |

---

## 6. 性能验收

> 详细步骤见 `profiling_mc2.md`。这里只给摘要。

```bash
PROJ="$(pwd)"

# 1. 算子二进制已具备 perf 模式（run.sh 第 5 参传 perf）
bash run.sh 2048 8192 3584 4 perf

# 2. msprof task-based 采集（无 warm-up；L2 flush 由 perf 主循环内部保证）
msprof --ai-core=on --aic-mode=task-based \
    --output="${PROJ}/docs/perf/round_001" \
    --application="${PROJ}/build/{op_name} 2048 8192 3584 4 perf"

# 3. 多卡后处理：使用 ops-profiling skill 的 msprof_perf_summary.py
python3 ${SKILL_PATH}/scripts/msprof_perf_summary.py "$ROUND_DIR" ops/{op_name}
```

### 关键点

- **两阶段 `tileCnt` 扫描**：扫描 `tileCnt ∈ {1, 2, 4, 8, 16, 32}`，每个值跑一次 msprof 采集 + 后处理，对比整体 Task Duration 找最优。完整流程见 [`pipeline_tuning.md`](pipeline_tuning.md) §3 阶段 B + §6 决策树；
- **L2 cache flush 必须在主循环每轮触发**：参考工程在 perf 主循环中每轮先调用 `heavy_add_kernel`（256 MB bf16 全核扫一遍）刷 L2，再跑主 kernel——所以 msprof 不需要 `--warm-up`；
- **perf 模式默认跑 10 轮**：msprof task-based 为每轮 main kernel 生成一条 `op_summary_*.csv` 记录；
- **多卡数据提取规则**：每卡取最后 5 次 main kernel 的 Task Duration 求平均 → 4 卡平均值取最大值作为整体性能（多卡并行由最慢卡决定）。

### 验收标准

- `docs/perf/round_NNN/` 存在且包含 4 个 `PROF_*` 子目录（每卡一份）；
- `msprof_perf_summary.py`（ops-profiling skill）输出每卡 `rank_avg` 与整体 `max` Task Duration；
- 最优 `tileCnt` 与对应整体 Task Duration 已记录；
- 若整体 Task Duration 与理论耗时差距 >50% → 检查通算流水是否真的并行（AIV time vs AIC time）。

---

## 7. 后续阅读

| 想了解 | 读 |
|--------|---|
| MC2 整体架构 | `mc2_architecture.md` |
| SHMEM API 细节 | `comm_shmem.md` |
| Blaze 模板选型 | `matmul_blaze.md` |
| 性能采集完整流程 | `profiling_mc2.md` |
| tileCnt 调优 | `pipeline_tuning.md` |
| 参考工程改造食谱 | `codebase_map.md` |
