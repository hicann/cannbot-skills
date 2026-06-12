---
skill_name: ascendc-env-check
---
# Case 1: NPU设备信息查询

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

如何查看NPU设备的详细信息，应该使用什么命令？

## Expected Output

回复应说明使用 npu-smi info 或者相关命令查看NPU设备列表和状态

## Expectations
- [contains] npu-smi


---

# Case 2: NPU架构检测

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

如何查看当前昇腾设备的NPU架构信息？

## Expected Output

回复应推荐使用 get_npu_arch.py 脚本检测NPU架构，并说明输出格式为 dav-xxx（如 dav-3510）

## Expectations
- [contains] get_npu_arch.py
- [contains] dav-


---

# Case 3: 环境变量配置注意点

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

配置CANN环境变量的时候有什么需要注意的地方？ASCEND_HOME_PATH和ASCEND_HOME有什么区别？

## Expected Output

回复应指出官方环境变量为 ASCEND_HOME_PATH，并说明部分旧文档中使用的 ASCEND_HOME 是错误用法，不应混淆

## Expectations
- [contains] ASCEND_HOME_PATH
- [contains] ASCEND_HOME


---

# Case 4: NPU不可见故障排查

## Config
- Max Tokens: 100000
- Disabled: true
- Ascend Platform: A2

## Prompt

我的昇腾服务器上看不到NPU设备，npu-smi info没有输出设备信息，怎么排查？

## Expected Output

回复应提供NPU设备不可见的排查思路；可能会先输出当前环境的NPU设备状态再给出排查建议

## Expectations
- [contains] npu-smi info -m


---

# Case 5: 算子运行失败排查

## Config
- Max Tokens: 100000
- Disabled: true
- Ascend Platform: A2

## Prompt

我开发的算子在昇腾NPU上运行失败了，怎么排查是不是环境配置的问题？

## Expected Output

回复应建议优先运行 check_env.sh 检查环境配置是否完整，并检查关键环境变量是否已正确设置

## Expectations
- [contains] check_env.sh


---

# Case 6: 正向看护-多skill环境下正确触发目标skill

## Config
- Max Tokens: 120000
- Distractor skills: cann-env-setup;npu-arch;ascendc-runtime-debug;ascendc-docs-search
- Ascend Platform: A2

## Prompt

我有一台昇腾服务器，想对NPU设备进行环境检查和设备查询，查看设备的详细信息和资源使用情况，应该用什么命令？

## Expected Output

回复应说明使用 npu-smi info 命令查看NPU设备信息，推荐使用 npu_info.sh 脚本进行综合查询。即使在多个 skill 同时可用的环境下，也应正确识别并激活 ascendc-env-check skill 而不是干扰 skill。

## Expectations
- [contains] npu-smi info

- [skill_activated] ascendc-env-check

---

# Case 7: npu-smi 结构化子命令查询

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

我想在脚本里自动获取 NPU 的温度和功耗，但直接解析 `npu-smi info` 的输出经常因为格式变化而失效。有没有更稳定的方法？

## Expected Output

回复应推荐使用 `npu-smi info -t <type> -i <device_id>` 结构化子命令，说明其返回 key:value 格式，比主表格更稳定。应列出常用 type（temp、power、memory、usages、common、health 等），并说明第一步先用 `npu-smi info -m` 确认设备 ID。

## Expectations
- [contains] npu-smi info


---

# Case 8: npu-smi 已变更/废弃命令

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

我以前的脚本里用了 `npu-smi top` 和 `npu-smi health` 命令，现在好像不能用了？还有 `npu-smi release` 和 `npu-smi lock` 呢？

## Expected Output

回复应说明 `npu-smi top` 已被 `npu-smi info -t usages -i <device_id>` 替代，`npu-smi health` 已被 `npu-smi info -t health -i <device_id>` 替代。`npu-smi release`、`npu-smi lock/unlock`、`npu-smi perf` 等命令在新版本中可能已不存在。应建议通过 `npu-smi --help` 和 `npu-smi info --help` 查询当前版本支持的命令。

## Expectations
- [contains] usages
- [contains] npu-smi --help


---

# Case 9: 脚本中可靠获取 NPU 信息

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

我需要写一个 Python 脚本来自动收集 NPU 设备信息（温度、功耗、使用率等），有什么可靠的方法可以避免硬编码 npu-smi 命令？

## Expected Output

回复应推荐编写 Python 脚本来调用 npu-smi 结构化子命令，说明可通过 `npu-smi info --help` 动态发现可用子命令，优先使用 `npu-smi info -t common` 批量获取数据，避免硬编码命令列表。应给出使用示例。

## Expectations
- [contains] Python
- [contains] 脚本
- [contains] --json