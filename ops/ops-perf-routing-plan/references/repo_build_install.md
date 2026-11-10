# 四仓编装包 / 独立目录装包 / 环境隔离知识

> 目标算子来源为四种仓之一。Step 1 在需要现场采集 baseline profiling 时，按本页对应方式
> **编译 kernel 包 → 装到独立工作目录 → source 自己的环境**，绝不污染共享 CANN 或他人任务。
> 本 skill 不构建优化后的模板，也不生成 final code。

## 0. 仓类型判定

`--repo-type` 取值：`nn` | `transformer` | `omni-inference` | `omni-training`。

缺省按算子路径自动探测：

| 特征 | 判定 |
|---|---|
| 路径含 `ops-nn` | `nn` |
| 路径含 `ops-transformer` | `transformer` |
| 路径含 `omni-ops/inference` | `omni-inference` |
| 路径含 `omni-ops/training` | `omni-training` |

## 一、四仓编装包对照表

| 仓 | 构建根 | build 命令 | 包产物 | vendor 目录 | 装包 |
|---|---|---|---|---|---|
| **ops-nn** | 仓库根 | `bash build.sh --pkg --soc=ascend910b --ops=<snake> --vendor_name=<uniq> -j16` | `build_out/cann-ops-nn-<vendor>_linux-aarch64.run` | `vendors/<vendor>`（首用时 dry-run 核对是否拼 `_nn`） | `*.run --install-path="$ASCEND_OPP_PATH"` |
| **ops-transformer** | 仓库根 | `bash build.sh --pkg --soc=ascend910b --ops=<snake> --vendor_name=<uniq> -j16` | `build_out/cann-ops-transformer-<vendor>_linux-aarch64.run` | `vendors/<vendor>_transformer` | 同上 |
| **omni-ops inference** | `inference/ascendc` | `bash build.sh -n <snake> -c ascend910b -p "$ASCEND_HOME_PATH"` | `output/CANN-omni_custom_ops-*.run` | `vendors/omni_custom_transformer`（cmake 写死） | 同上 |
| **omni-ops training** | `training/ascendc` | `bash build.sh -n <snake> -c ascend910b -p "$ASCEND_HOME_PATH"` | `output/CANN-omni_training_custom_ops-*.run` | `vendors/omni_training_custom_transformer`（cmake 写死） | 同上 |

要点：

- `--vendor_name` 每个任务唯一（如 `mgtk_l00943230`），禁用默认 `custom` 或他人名字。
- omni 仓 vendor 名由 cmake 写死，无需传。
- ops 仓 `--ops` 与 omni 仓 `-n` 都传 **snake_case**（CamelCase 会报 `No SOURCES given to target: opsproto` / 找不到算子）。

## 二、独立目录装包（防止污染共享环境）

Step 1 只有一个 baseline 工作区：

```text
$WORK/
├── cann_env/                  # 可选：独立 CANN
├── cann_opp/                  # 本任务独立 OPP，装包只写这里
├── code/ori_code/<op>/        # 只读分析基底
├── env.sh                     # prepare.py 生成
└── logs/{build.log,install_check.log}
```

### 2.1 env.sh 基本动作

```bash
export WORK_ROOT=$WORK
export CANN_ROOT=$WORK/cann_env
export TORCH_EXTENSIONS_DIR=$WORK/torch_extensions
export CCACHE_DIR=$WORK/.ccache
export PIP_CACHE_DIR=$WORK/.pip_cache
export TMPDIR=$WORK/.tmp
mkdir -p "$TORCH_EXTENSIONS_DIR" "$CCACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

# ① unset 全局 CANN，避免继承变量顶到共享 CANN
unset ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_TOOLKIT_HOME ASCEND_AICPU_PATH
# ② 需要时激活本会话专用 Python 环境
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate <本会话环境>
# ③ 钉死本工作区 CANN
source $CANN_ROOT/ascend-toolkit/set_env.sh
# ④ 自检：ASCEND 路径必须在本 WORK 内
for v in ASCEND_HOME_PATH ASCEND_OPP_PATH; do
  p="${!v}"; case "$p" in $WORK/*) ;; *) echo "FATAL: $v=$p 不在 $WORK 内" >&2; exit 1 ;; esac
done
```

### 2.2 装包到独立 OPP

```bash
export ASCEND_OPP_PATH="$WORK/cann_opp"
./<包>.run --quiet --install-path="$ASCEND_OPP_PATH"
source "$ASCEND_OPP_PATH/vendors/<vendor>/bin/set_env.bash"
```

`prepare.py build` 已封装上述流程，并把结果写入 `$WORK/env.sh`。

### 2.3 装包自证清单（写入 install_check.log）

1. `$ASCEND_OPP_PATH` 在 `$WORK` 内；
2. `vendors/<vendor>/` 存在；
3. `.run` 内 vendor 路径一致；
4. `op_impl/ai_core/tbe/kernel/**/` 有本次编译的 `.json` / `.o`；
5. 归因验证：msprof `task_time.csv` 的 `kernel_name` 命中本 vendor hash（确认走自己的 kernel，非内建 fallback）。

## 三、隔离纪律

1. **只装进自己的目录**：`--install-path` 必须显式指向 `$WORK/cann_opp`；禁止写 `/usr/local/Ascend`、他人的 `cann_env` 或共享仓。
2. **vendor 名唯一**：只用本任务的 vendor，禁止 `custom` 或他人名字。
3. **只 source 自己的 env**：禁止 source `/usr/local/Ascend/.../set_env.sh` 或他人 env；`pip install` 禁止 `--user`。
4. **缓存与产物全部本地**：`TORCH_EXTENSIONS_DIR` / `CCACHE_DIR` / `PIP_CACHE_DIR` / `TMPDIR` / msprof 输出 / build 产物全在 `$WORK` 内。
5. **禁止跨 agent 全局锁**（`flock /tmp/*.lock`）。
6. **装错即停**：只回滚自己刚建的目录，如实上报，不得继续评测。
