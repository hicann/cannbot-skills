# Protocol: Layer 2 — 日志与错误码分析

---

## 错误码速查（aclnn 返回值）

| 错误码范围 | 含义 | 对应 hypothesis |
|---|---|---|
| `0` | 成功 | — |
| `161000-161999` | 参数错误 / Runtime 通用错误 | H14 → msaicerr |
| `507034` | Vector Core 超时 | H12（event_id 复用）|
| `561002` | Executor 内部错误 | H14 → msaicerr |
| `561003` | 属性配置错误 | 检查 op 参数 |
| `561107` | JSON 描述错误 | 检查 op proto |
| `561112` | 二进制包错误 | 重新编译 |
| DDR out-of-range | 内存越界 | H02（workspace）/ H08（越界访问）|

---

## plog 解析

```bash
# plog 默认路径
PLOG_DIR=~/ascend/log/debug/plog

# 关键关键词 grep
grep -E "ERROR|WARNING|AICore|timeout|DDR|out.of.range|exception" \
     "${PLOG_DIR}"/*.log | tail -50
```

**plog 中的关键信息**：
- `[ERROR]` 行：错误码 + 模块 + 描述
- `AICore exception`：转到 H14（msaicerr）
- `timeout`：转到 H12（event_id 复用）
- `DDR out of range`：转到 H02（workspace）

---

## aclGetRecentErrMsg 使用

```cpp
// 在 Python 端调用算子后获取最近错误信息
import acl
err_msg = acl.get_recent_err_msg()
print(err_msg)
```

---

## 环境快速检查

```bash
# 确认 CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# 确认环境变量已设置
source install_out/vendors/customize/bin/set_env.bash

# 确认设备可用
python -c "import torch_npu; print(torch_npu.npu.is_available())"
```
