# {operator_name} 算子开发计划

> ⚠️ `{operator_name}` → 实际算子名称。本文档在开发流程中持续更新。

---

## 1. 需求概述

| 项目 | 内容 |
|-----|------|
| 算子名称 | {operator_name} |
| 数学公式 | y = f(x) |
| 输入 | A: shape=[...], dtype=...; B: shape=[...], dtype=... |
| 输出 | D: shape=[...], dtype=... |
| 目标 SoC | Atlas A2 / Ascend950 |
| 算子类型 | Matmul / Matmul+Epilogue / QuantMatmul |

---

## 2. 文件清单

| 文件 | 状态 |
|------|------|
| `{operator_name}_tiling.h` — Tiling 结构体（kernel/host 共用） | ⬜ |
| `{operator_name}_kernel.asc` — Kernel 计算逻辑（catlass 拼装） | ⬜ |
| `{operator_name}.asc` — Host + main 入口 | ⬜ |
| `CMakeLists.txt` — 构建脚本 | ⬜ |
| `run.sh` + `gen_data.py` + `golden.py` + `verify_result.py` | ⬜ |

---

## 3. Catlass 编译选项

CMakeLists.txt 必须注入：

| 选项 | 说明 | 值 |
|------|------|-----|
| `-I<catlass>/include` | catlass 头文件路径 | `-I${CMAKE_SOURCE_DIR}/../../catlass/include` |
| `-DCATLASS_ARCH` | 芯片架构号 | 2201（910b）/ 3510（950） |

---

## 4. 测试计划

精度标准：atol=0.001, rtol=0.005（量化/非量化通用，无 catlass 专属放宽）

**Catlass 测试 shape 约束**：
- 不宜用过小 M/N（个位数易触发 AIV UB 越界）
- 宜选 L1 分块 M/N 整数倍（如 M=128、N=256）

| 编号 | 用例 | shape | 覆盖分支 |
|-----|------|------|---------|
| T1 | L1 分块整数倍 | M=128, N=256, K=512 | 默认 branch |
| T2 | M>=N 分支 | M=512, N=256, K=512 | BlockScheduler<3,0> |
| T3 | M<N 分支 | M=128, N=512, K=512 | BlockScheduler<3,1> |

---

## 5. 开发进度

| 阶段 | 检查项 | 状态 |
|------|--------|------|
| 框架搭建 | 工程创建 + CMake + 空 Kernel 编译通过 | ⬜ |
| Kernel 实现 | catlass 拼装 + Device 调用 + 编译通过 | ⬜ |
| 可执行文件验证 | T1–T3 全部通过 | ⬜ |
| 性能验收 | msprof 采集 + 数据归档 + 达标判定 | ⬜ |

---

## 6. 已知问题和决策记录

| 日期 | 问题/决策 | 说明 |
|------|----------|------|

---

## 7. 测试结果

| 编号 | 结果 | Max Diff | 备注 |
|-----|------|----------|------|
| T1 | ⬜ | | |
| T2 | ⬜ | | |
| T3 | ⬜ | | |

---

## 8. 性能验收

**状态**: ⬜ | **数据**: docs/perf/round_NNN/

| 指标 | 值 | 判定 |
|------|------|------|
| Task Duration | | |
| Block Dim | | |
| 主导流水 | | |

**达标判定**: ⬜ | **理由**:
