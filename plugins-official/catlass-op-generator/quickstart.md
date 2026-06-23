# CANNBot · Catlass 算子直调开发快速入门

## 概述

`catlass-op-generator` 是 CANNBot 旗下的 catlass 算子直调开发 plugin。算子工程结构与通用 Ascend C 直调完全一致，catlass **仅决定 op_kernel 内部如何用模板拼装计算 pipeline**。

**与其他 plugin 的关系**：

| Plugin | 适用场景 |
|--------|---------|
| `ops-direct-invoke` | 通用 Ascend C 直调（自写 Tiling + 矢量 API） |
| **`catlass-op-generator`（本 plugin）** | catlass 模板拼装 GEMM / Matmul + Epilogue / Quant Matmul |

---

## 一、环境搭建

### 前置条件

- 已安装 CANN Toolkit（建议 ≥ 9.0.0），具体版本配套关系请查阅 [CANN Release Notes](https://www.hiascend.com/cann/document)
- 已配置 NPU 设备（支持 Ascend 910/950 PR 等芯片）
- 已安装 OpenCode、Claude Code、TRAE、Cursor 等受支持的 AI 编程工具

### 安装

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/catlass-op-generator
bash init.sh project opencode   # 项目级
bash init.sh global opencode    # 全局级
```

支持工具：`opencode`（默认）、`claude`、`trae`、`cursor`、`copilot`。

### catlass 源码就绪（自动处理）

工作区根需存在 `./catlass/`（与 `operators/` 平级）。Step 1 会自动检测并在缺失时执行 `git clone https://gitcode.com/cann/catlass.git`。

---

## 二、快速上手

在交互界面输入 catlass 算子开发需求：

```
帮我开发一个 catlass_matmul_gelu 算子：
- A/B 为 FP16 行主序，输出 FP16
- GELU 激活
- 目标 SoC Ascend 910
- 主要 shape M=N=K=512
```

CANNBot 自动按 Step 1–7 调度：

```
Step 1: 环境检查 + catlass 命名校验 + catlass 源码就绪
Step 2: 设计（Architect 加载 /catlass-op-design）
Step 2.5: 设计串讲
Step 3: 开发（Developer 加载 /catlass-op-develop）
Step 4: 审查（Reviewer 含 C1–C11 检视）
Step 5: 修复循环（最多 3 轮）
Step 6: 性能验收（Developer 按需 /catlass-op-perf-tune）
Step 7: 完成汇报
```

### 产出物示例

```
operators/catlass_matmul_gelu/
├── docs/
│   ├── DESIGN.md              # 含 catlass 选型表
│   ├── PLAN.md                # 含 catlass 编译选项
│   ├── REVIEW.md              # 含 catlass C1–C11 状态
│   ├── WALKTHROUGH.md
│   └── environment.json
├── catlass_matmul_gelu_tiling.h   # Tiling 结构体
├── catlass_matmul_gelu_kernel.asc # catlass 拼装
├── catlass_matmul_gelu.asc        # Host + main
├── CMakeLists.txt                 # 含 -I<catlass>/include + -DCATLASS_ARCH
├── run.sh
├── gen_data.py
├── golden.py
└── verify_result.py
```

---

## 三、可用技能

| Skill | 触发时机 |
|-------|---------|
| `catlass-op-design` | **强制：Architect 设计阶段** |
| `catlass-op-develop` | **强制：Developer 开发阶段** |
| `catlass-op-perf-tune` | 性能调优阶段 |
| `ascendc-tiling-design` | host 侧 Tiling 计算 |
| `npu-arch` | NPU 架构查询 |
| `ascendc-env-check` | 环境检查 |

---

## 四、Catlass 编译选项

算子 CMakeLists.txt 中唯一需要的 catlass 编译选项：

| 选项 | 值 |
|------|-----|
| `-I` | `<catlass>/include` |
| `-DCATLASS_ARCH` | 2201（910）/ 3510（950 PR） |

**不使用 catlass 仓库自身的 CMake 函数**。算子工程使用标准 Ascend C CMake 构建方式。

---

## 五、常见问题

### Q: catlass 源码为什么不能放在 operators/ 内？

C2 检视项要求。多个 catlass 算子共用一份源码（位于工作区根），避免重复克隆。

### Q: 为什么 op_kernel 不能用 DeviceGemm 适配器？

C4 检视项。DeviceGemm 是 catlass example 的 host 侧便捷入口，会带来不必要的运行时开销。算子直调场景直接 `Kernel{}(params)`。

### Q: 算子名为什么必须含 catlass？

C1 检视项要求，便于识别 catlass 算子并保持命名一致性。

### Q: catlass 仓库的 CMake 函数能用吗？

**不能**。catlass 仓库的 CMake 函数是其 example 的构建辅助，不适用于算子直调工程。算子工程使用标准 Ascend C CMake，仅通过 `-I` 和 `-D` 引用 catlass 头文件。
