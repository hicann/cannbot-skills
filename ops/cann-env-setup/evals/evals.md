---
skill_name: cann-env-setup
---
# Case 1: 检查NPU驱动安装命令

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

我有一台昇腾服务器，想检查NPU驱动是否已安装，应该用什么命令？

## Expected Output

回复应说明使用 npu-smi info 命令检查驱动，并解释如何根据命令输出判断驱动是否已安装

## Expectations
- [contains] npu-smi info


---

# Case 2: 配置环境变量永久生效

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

我已经用离线安装包安装完CANN Toolkit和Ops，现在需要配置环境变量使其永久生效，应该怎么做？

## Expected Output

回复应说明如何配置环境变量实现永久生效：通过 source set_env.sh 命令并将其写入 ~/.bashrc 文件

## Expectations
- [contains] source
- [contains] set_env.sh


---

# Case 3: 验证安装是否成功

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

CANN安装完成后，如何验证安装是否成功？

## Expected Output

回复应提供至少一种验证 CANN 安装是否成功的方法

## Expectations

---

# Case 4: 离线安装顺序

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

我需要安装CANN，但服务器没有网络，只有离线安装包Ascend-cann-toolkit_8.0.RC1_linux-aarch64.run和Ascend-cann-910b-ops_8.0.RC1_linux-aarch64.run。请问安装顺序是什么？

## Expected Output

回复应明确说明安装顺序：先安装 Toolkit 包，再安装 Ops 包

## Expectations
- [contains] toolkit


---

# Case 5: Conda在线安装CANN

## Config
- Max Tokens: 120000
- Max Tokens (deepseek-v4-flash): 140000
- Max Tokens (glm-5): 130000
- Ascend Platform: A2

## Prompt

我想用conda方式在线安装CANN，需要Python 3.10环境，具体步骤是什么？

## Expected Output

回复应说明 conda 安装 CANN 的完整步骤：创建 conda 环境、添加昇腾 conda 源、使用 conda install 安装 cann-toolkit 和 ops 包

## Expectations
- [contains] conda


---

# Case 6: 安装前依赖检查

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

安装CANN之前需要检查哪些依赖？

## Expected Output

回复应说明安装 CANN 前需要检查的依赖项，至少包括 Python 和 pip 是否已安装

## Expectations
- [contains] Python
- [contains] pip


---

# Case 7: 正向看护-多skill环境下正确触发目标skill

## Config
- Max Tokens: 120000
- Max Tokens (deepseek-v4-flash): 140000
- Max Tokens (glm-5): 130000
- Distractor skills: ascendc-task-focus;npu-arch;ascendc-docs-search;ascendc-runtime-debug
- Ascend Platform: A2

## Prompt

我有一台昇腾服务器，需要做一遍CANN环境配置检查，确认驱动是否已安装、环境是否就绪，标准的排查流程是怎样的？

## Expected Output

回复应说明使用 npu-smi info 命令检查驱动，并解释如何根据命令输出判断驱动是否已安装，给出完整的排查流程。即使在多个 skill 同时可用的环境下，也应正确识别并激活 cann-env-setup skill 而不是干扰 skill。

## Expectations
- [contains] npu-smi info

- [skill_activated] cann-env-setup
