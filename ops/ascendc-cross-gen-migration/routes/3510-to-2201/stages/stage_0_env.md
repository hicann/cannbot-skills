# 阶段 0：环境与输入确认

**目标**：确认"能不能开工、在什么平台上开工"。本阶段不写业务代码。

## Step 0.1 输入形态

按 `references/input-contract.md` §1 判定形态 A/B/C：

- A（有 A5 源码）→ 记录 `OP_REPO`、`OP_NAME`、构建入口。
- B（只有设计文档）→ 记录语义基线缺口清单，确认后才继续；基线确认属于阶段 1 的输入。
- C（信息不足）→ 按 §4 模板提问，**停止**。

## Step 0.2 目标平台

1. 确认目标芯片：`910B` 系列 → `SOC_BUILD=ascend910b`；`910_93` → `SOC_BUILD=ascend910_93`。
2. 在目标环境实测架构：

```bash
source "${CANN_SET_ENV}"
npu-smi info 2>/dev/null | head -20            # 设备型号
python3 -c "import torch, torch_npu; print(torch.npu.get_device_name(0))" 2>/dev/null
```

3. 记录 `NpuArch`（应为 **2201**）。若目标机报 3510/2002/其他 → 停止：目标平台与技能范围不符。
4. 记录芯片子型号（影响核数/L2），但**不要**把核数写死到代码。

## Step 0.3 工具链

```bash
echo "${ASCEND_HOME_PATH}"                      # CANN 安装路径
which bisheng ccec 2>/dev/null
python3 --version
```

- 记录 CANN 版本（决定 API 是否存在、`--soc` 拼写、`Fixpipe` 结构体命名）。
- 确认目标 CANN 支持 `ascend910b`：`bash build.sh --help | grep -i soc`（在目标仓内执行）。

## Step 0.4 工程入口

```bash
cd "${OP_REPO}"
ls build.sh CMakeLists.txt 2>/dev/null
grep -rn "AddConfig(" --include=*_def.cpp . | head
grep -rn "SUPPORT_TILING_DIR" --include=CMakeLists.txt . | head
```

记录：算子名、`build.sh` 路径、安装方式（`install.sh` 或 run 包）、现有 A2 先例目录（用于阶段 2 照抄接线）。

## Step 0.5 执行环境

| 项 | 要求 |
|---|---|
| A2 真机 | 可执行 kernel 与 `msprof`；无设备则 GATE-3/4/5 不能 PASS。**A5 机器不能替代 A2 验证**，A5 上的编译/安装/运行结果不作为 A2 证据 |
| 替代环境 | CANN simulator 等仅可用于编译/冒烟；其数值与性能不能作为验收证据 |
| 磁盘/内存 | 够放 build、算子包、profiling 归档（`PROF_GROUP_*` 可能数 GB） |

## Step 0.6 变量落盘

在报告中记录（不写死个人路径，用环境变量或占位）：

```bash
export CANN_SET_ENV=<CANN>/set_env.sh
export OP_REPO=<工程根目录>
export OP_NAME=<算子名>
export SOC_BUILD=ascend910b|ascend910_93
export PYTHON_PATH=$(which python3)
```

## Step 0.7 测试依赖自检

在依赖仓库自带 pytest/UT 之前，先确认目标环境具备运行条件；不满足时提前规划替代验证（见 `references/verification-harness.md`）：

```bash
python3 -m pytest --version 2>&1 | head -1
python3 -c "import numpy, pandas; print('pandas ok')" 2>&1 | tail -1
python3 -c "import tensorflow" 2>&1 | tail -1   # 部分算子 pytest 仅用它做 bf16 dtype 映射
python3 -c "import ml_dtypes" 2>&1 | tail -1
```

- 缺包且离线：从同版本环境拷贝 site-packages，或用 numpy stub 替代仅做 dtype 映射的依赖；
- 无法补齐：不要卡在 pytest 上，改用 ACLNN 直调 + CPU 参考建立最小验收。

## 完成判据

- 输入形态明确，必需事实（`input-contract.md` §2）无缺失。
- 目标平台确认为 `DAV_2201`，`SOC_BUILD` 已定。
- CANN/Python/构建入口可用，已记录版本。
- 执行环境明确（真机 / 模拟器 / 无）。

## Gate

```
[GATE-0] ENV_CONFIRMED target=<SOC_BUILD> arch=2201 cann=<version> npu=<device|none>
```

任一缺失：输出 `BLOCKED` 并列出缺项，不进入阶段 1。
