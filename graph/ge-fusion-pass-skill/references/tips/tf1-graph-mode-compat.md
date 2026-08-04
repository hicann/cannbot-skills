# tip: TF1 graph-mode 脚本兼容

> 📎 导航落点：`references/pass-development-paradigm.md` §1（输入识别）。本文件仍是该兼容纪律的**唯一权威定义源**，正文未改动。迁移映射见 `tips/MIGRATION.md`。

**读者**：② 开发 / ③ 分析。

## 症状

`data/` 下 TensorFlow 生成脚本用 `tf.compat.v1.placeholder`/`Session`（TF1 graph mode），但当前环境只有 TF2，直接跑报 eager 相关错误；或 `tf_forward.py` 导入 `npu_bridge.npu_init` 但环境缺 `npu_bridge`。

## 硬性做法

- **不改 `data/` 原始脚本、不在 workspace 内重装 TensorFlow。**
- 若用户显式提供 TF1 Python 解释器，先确认路径可执行并探测 TensorFlow 版本，再用它运行；skill 不猜宿主私有环境变量，也不自行寻找未声明的解释器。
- 若只有 TF2、需生成 PB：在**新写的辅助脚本**里临时关闭 eager，不动原脚本：
  ```python
  import tensorflow as tf
  tf.compat.v1.disable_eager_execution()
  # 或：import tensorflow.compat.v1 as tf; tf.disable_v2_behavior()
  ```
- 若用户已经提供 `.pb` 文件：不必重新运行 TensorFlow 生成脚本，直接用 `adapt_input.py compile <model.pb>` 走 `atc/pyatc --framework=3`；只有 PB 生成和 TF 在线执行仍需 TF1/`npu_bridge`。
- `tf_forward.py` 依赖 `npu_bridge` 但环境缺失时：**如实记录并跳过在线 TF 路径**，改用 PB + ATC 离线编译验证 pass。

## 自查

- 有没有改到 `data/` 原始脚本？（不允许）
- 有没有在 workspace 里 `pip install tensorflow`？（不允许，优先用已有环境或临时 wrapper）
- 缺 `npu_bridge` 时，是如实跳过在线路径走离线，还是伪造了在线结果？必须是前者。
