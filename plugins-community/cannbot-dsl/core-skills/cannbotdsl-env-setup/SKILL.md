---
name: cannbotdsl-env-setup
description: "搭建或诊断 CANNBotDSL 开发环境时使用。用户首次上手 CANNBotDSL、编译莫名失败疑似环境问题、或需要确认 NPU/bisheng 工具链是否就绪时触发。检查 cannbotdsl wheel 安装、import 可达性、ASCEND_HOME_PATH/bisheng 编译器、pytest markers（cannir_install/ascendc_toolchain/npu），输出 env_check.json 环境诊断报告，并据此决定是否降级为编译验证模式。Triggers: cannbotdsl 环境, 环境预检, env_check, bisheng, PYTHONPATH, pytest markers, NPU 可用性。工作流 Stage 0 由 Primary 直接调用。"
---

# cannbotdsl-env-setup

CANNBotDSL 环境搭建与验证。工作流 Stage 0 由 Primary 直接调用；产出一份分层能力诊断，决定后续 Stage 能跑到哪一级（import / translate / compile / NPU）。

**API 参考以源码为准**。构建/安装指南在仓库根 `README.md`。

## 触发条件

- 首次使用 CANNBotDSL 开发
- 编译失败疑似环境问题（`import cannbotdsl` 失败、bisheng not found）
- 需要确认 NPU 设备可用性 / 决定是否降级为编译验证模式

## 能力分级（决定后续 Stage 上限）

环境不是全有或全无。按依赖链从低到高分四级，任一级失败即为该级上限：

| 级别 | 能力 | 依赖 | 探测方法 |
|------|------|------|----------|
| **L0 Python 前端** | `@jit`/`@kernel` trace、AST 预处理、IR build | `import cannbotdsl` 成功（wheel 已安装） | `python -c "import cannbotdsl"` |
| **L1 translate** | 生成 AscendC 源、代码生成断言 | cannbotdsl wheel 内置 translate 工具链 | `pytest -m ascendc_toolchain --collect-only` |
| **L2 compile** | bisheng 编 `.asc→.o→.so` | `$ASCEND_HOME_PATH/x86_64-linux/ccec_compiler/bin/bisheng` | `pytest -m ascendc_toolchain --collect-only` |
| **L3 NPU execute** | 真机跑 kernel + 精度对比 | Ascend NPU 设备 + torch_npu runtime | `pytest -m npu --collect-only` |

> **关键事实**：本仓通过 wheel 安装 cannbotdsl（`pip install cannbotdsl-*-cp312-cp312-manylinux_*_x86_64.whl`），编译器后端随 wheel 分发，不需要单独构建或设 PYTHONPATH 指向源码树。`import cannbotdsl` 失败最常见原因是 wheel 未装或 Python 版本不匹配（需 CPython 3.12）。

## 环境变量校验清单

安装（`README.md`）：

```bash
python -m pip install /absolute/path/to/cannbotdsl-*-cp312-cp312-manylinux_*_x86_64.whl
python -c 'import cannbotdsl; print(cannbotdsl.__file__)'
```

运行期（NPU 编译+执行需要）：

```bash
export ASCEND_HOME_PATH=<cann-home>                # bisheng 与 toolchain 版本探测的锚点
source <cann-install>/set_env.sh
```

- `ASCEND_HOME_PATH` 未设 → bisheng 编译不可用，L2 降级。
- bisheng 实际路径：`$ASCEND_HOME_PATH/x86_64-linux/ccec_compiler/bin/bisheng`；不存在 → L2 不可用。
- bisheng 硬编码 `--npu-arch=dav-3510`，跨架构目前无法配置 —— 记为已知限制。
- `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 适用于避免 torch 自动加载 NPU 后端的场景（如 bench 脚本）。

## pytest markers 能力探测

markers 如下：

| marker | 门控 | 缺失时 |
|--------|------|--------|
| `cannir_install` | 需要 CANNIR 命令行工具（wheel 不分发） | skip |
| `ascendc_toolchain` | 需要 translate + AscendC 工具 | skip |
| `npu` | 需要 Ascend NPU 设备 / runtime | skip |
| `slow` | 较重的全量精度扫描 | deselect with `-m 'not slow'` |

## 诊断流程

1. **L0**：`python -c "import cannbotdsl"`。失败 → 查 wheel 是否安装、Python 版本是否 3.12（`python --version`）。
2. **L1**：`pytest -m ascendc_toolchain --collect-only -q`。失败 → 查 cannbotdsl wheel 完整性。
3. **L2**：`echo $ASCEND_HOME_PATH` + `ls $ASCEND_HOME_PATH/x86_64-linux/ccec_compiler/bin/bisheng`。
4. **L3**：`python -c "import torch, torch_npu; print(torch.npu.is_available())"`。
5. 汇总输出 `env_check.json`，记录每级 PASS/FAIL 与 max_stage，供后续 Stage 决定降级。

## 输出：env_check.json

```json
{
  "l0_python_frontend": "PASS",
  "l1_translate":       "PASS",
  "l2_compile":         "FAIL: ASCEND_HOME_PATH not set",
  "l3_npu":             "SKIP: no NPU device",
  "max_stage": "translate",
  "downgrade": "只验证代码生成，不上真机"
}
```

`max_stage` 决定后续 Stage 的验证上限：无 NPU 时后续 Stage 只做代码生成断言，精度验证标记为 blocked。

## 门禁

- 报告必须区分"环境未就绪"和"代码有 bug"：L0/L1 失败是环境问题，不要改 kernel 代码。
- 明确 max_stage，并把它作为后续所有 Stage 的验证上限；不能跑的级别一律标 blocked 并说明缺什么。

## 参考

- 仓库根 `README.md`（wheel 安装、pytest 运行指南）
- `../cannbotdsl-op-develop/SKILL.md`（可复用的 shell 环境模板）
