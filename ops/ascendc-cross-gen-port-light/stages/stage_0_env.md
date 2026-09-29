# 阶段 0：环境信息获取与验证

**本阶段 MUST 在一切操作之前完成。未输出 GATE-0 前禁止执行编译、安装、测试等任何操作。**

## 核心规则

1. **CANN 路径 MUST 由用户提供**——禁止自行探测（`find`、`locate`、扫描 `/home` 等）
2. **检测不到时 MUST 向用户询问**——不得猜测或使用默认路径
3. **所有后续 Shell 命令 MUST 使用本阶段记录的变量**——禁止在命令中写死路径

## Step 0.1：获取 CANN 环境

| 信息项 | 自动检测方式 | 检测失败时向用户询问 |
|--------|------------|-------------------|
| CANN 安装路径 | `echo $ASCEND_HOME_PATH` | "请提供 CANN 安装路径" |
| CANN 版本 | `cat $ASCEND_HOME_PATH/ascend_toolkit_install.info 2>/dev/null` | "请提供 CANN 版本号" |
| set_env.sh | `$ASCEND_HOME_PATH/../set_env.sh` | "请提供 CANN 环境激活命令（如 `source /path/set_env.sh`）" |

**关键**：如果 `$ASCEND_HOME_PATH` 为空或不存在，MUST 暂停并向用户请求，不得自行搜索。

## Step 0.2：获取 Python 环境

| 信息项 | 自动检测方式 |
|--------|------------|
| Conda 环境名 | `echo $CONDA_DEFAULT_ENV`（非 base 且非空时有效） |
| Python 路径 | `which python3` |
| Python 版本 | `python3 --version` |
| torch 版本 | `python3 -c "import torch; print(torch.__version__)"` |
| torch_npu 版本 | `python3 -c "import torch_npu; print(torch_npu.__version__)"` |
| pytest 版本 | `python3 -c "import pytest; print(pytest.__version__)"` |

## Step 0.3：获取 NPU 硬件

| 信息项 | 检测方式 |
|--------|---------|
| NPU 可用性 | `python3 -c "import torch_npu; print(torch.npu.is_available())"` |
| 设备数量 | `python3 -c "import torch_npu; print(torch.npu.device_count())"` |
| 可用设备 | `npu-smi info 2>/dev/null` 选择空闲设备 |

## Step 0.4：执行验证脚本

```bash
source ${CANN_SET_ENV} && ${PYTHON_PATH} -c "
import torch, torch_npu
print(f'torch: {torch.__version__}')
print(f'torch_npu: {torch_npu.__version__}')
print(f'NPU available: {torch.npu.is_available()}')
if torch.npu.is_available():
    x = torch.randn(4, 4, device='npu:0')
    print(f'NPU tensor OK: {x.device}')
"
```

## Step 0.5：记录环境信息

将以下信息同时保存到**会话上下文**和**文件**，供所有后续阶段引用：

1. **会话上下文**：直接在对话中展示环境信息表格
2. **文件落盘**：写入 `${OP_DIR}/.env_info.md`（算子工程根目录下），格式如下：

```
## 环境信息记录

### CANN 环境
- CANN_PATH: <path>
- CANN_VERSION: <version>
- CANN_SET_ENV: <path>/set_env.sh    # 不含 source 前缀

### Python 环境
- CONDA_ENV: <name>
- PYTHON_PATH: <abs_path>
- PYTHON_VERSION: <version>
- TORCH_VERSION: <version>
- TORCH_NPU_VERSION: <version>

### NPU 硬件
- NPU_AVAILABLE: True/False
- NPU_DEVICE_COUNT: <count>
- NPU_VISIBLE_DEVICE: <device_id>

### 关键路径
- CANN_INCLUDE: <CANN_PATH>/aarch64-linux/include
- CANN_ACLNN_INCLUDE: <CANN_PATH>/aarch64-linux/include/aclnn
- TORCH_NPU_PATH: <from python>
- TORCH_CMAKE_PREFIX: <from python>
```

**文件写入要求**：
- 文件路径固定为 `${OP_DIR}/.env_info.md`
- 如果文件已存在，覆盖写入
- 写入后在对话中确认文件路径和写入成功

## Step 0.6：记录算子工程路径

| 信息项 | 说明 |
|--------|------|
| 算子工程根目录 | 如 `/home/user/code/ops-nn_reg` |
| 目标算子名 | 如 `top_k_top_p_sample` |
| 目标 SoC | 如 `ascend950` |
| 构建脚本 | 如 `build.sh`（在工程根目录下） |

## Step 0.7：全量仓初始化（asc-devkit）

**目标**：确认 asc-devkit 全量仓已可用，记录路径到 session context。本 skill 的 API 文档、算子样例、头文件等均依赖此仓库。

### 0.7.1 获取全量仓路径

向用户询问是否已 clone asc-devkit 仓库（https://gitcode.com/cann/asc-devkit）：

- **如已 clone**：用户提供路径 → 记录为 `$DEVKIT_PATH`
- **如未 clone**：执行 git clone
  ```bash
  git clone https://gitcode.com/cann/asc-devkit.git <用户指定路径>
  ```
  clone 完成后记录路径为 `$DEVKIT_PATH`

### 0.7.2 验证仓库完整性

**★ 布局识别（MUST）**：asc-devkit 布局随版本演进（2026-09 起 `docs/zh|en/` + 英文目录名；更早 `docs/api|guide/` + 中文目录名）。先识别再校验，结果写入 `.env_info.md`：

```bash
# 1) 识别布局与新鲜度
git -C $DEVKIT_PATH log -1 --date=short --format='DEVKIT_COMMIT=%h DEVKIT_DATE=%ad'
if [ -d "$DEVKIT_PATH/docs/zh/api" ]; then echo "DEVKIT_LAYOUT=new-zh";
elif [ -d "$DEVKIT_PATH/docs/api" ]; then echo "DEVKIT_LAYOUT=legacy";
else echo "DEVKIT_LAYOUT=unknown"; fi

# 2) 校验（以 new-zh 为例；legacy/unknown 按下述规则换算）
ls $DEVKIT_PATH/docs/zh/api/ && \
ls $DEVKIT_PATH/impl/ && \
ls $DEVKIT_PATH/include/ && \
ls $DEVKIT_PATH/examples/
```

- `new-zh` → 按 `docs/zh/…` 路径使用（实测映射表见 `references/devkit-path-map.md` 顶部「布局演进与定位协议」）
- `legacy` → 旧版仓：提示本 skill 按最新布局编写，建议重新 clone
- `unknown` → MUST 用 `ls $DEVKIT_PATH/docs/` + `find … -maxdepth 3 -type d` 现场推导映射，禁止按旧表硬拼
- `DEVKIT_DATE` 距今超过 6 个月 → 提示布局可能再次演进，按 unknown 处理

四个目标目录全部存在才算验证通过；任一缺失 → 提示仓库不完整，重新 clone 或指定其他路径。**禁止**在布局未识别时继续后续阶段。

### 0.7.3 记录全量仓信息

将 `$DEVKIT_PATH` 同时写入**会话上下文**和 `${OP_DIR}/.env_info.md`：

在 `.env_info.md` 的末尾追加：

```
### asc-devkit 全量仓
- DEVKIT_PATH: <全量仓绝对路径>
- 验证状态: PASS
```

## Gate 输出条件

MUST 同时满足：
- [ ] CANN 路径、版本、set_env.sh 已确认
- [ ] Python 环境全部确认
- [ ] NPU 可用且设备已选择
- [ ] 验证脚本执行成功
- [ ] 环境信息已记录到会话
- [ ] 环境信息已写入 `${OP_DIR}/.env_info.md`
- [ ] 算子工程路径已记录
- [ ] asc-devkit 全量仓已验证并记录 `$DEVKIT_PATH`

**全部通过 → 输出 `[GATE-0] ENV_CONFIRMED` → 进入阶段 1**

## 环境复用规则

| 场景 | 处理 |
|------|------|
| 同一会话内再次需要 | 直接复用，无需重复获取 |
| 检测到路径变化 | 提示用户确认 |
| 新会话 | 执行完整获取流程 |

## 常见环境问题

| 报错 | 原因 | 修复 |
|------|------|------|
| `Cast ADD_TO_LAUNCHER_LIST_AICORE failed` | 未 source CANN | 执行 `source ${CANN_SET_ENV}` |
| `ModuleNotFoundError: torch_npu` | torch_npu 未安装 | 安装对应 CANN 版本 |
| `NPU available: False` | 设备未就绪 | 检查 `npu-smi info` |
| NPU tensor 创建卡住 | 设备被占用 | 设置 `ASCEND_RT_VISIBLE_DEVICES` |
