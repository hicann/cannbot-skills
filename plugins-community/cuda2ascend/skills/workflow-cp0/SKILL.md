---
name: workflow-cp0
description: 环境确认的验收标准。触发：仅在明确调用时触发，不主动触发。
disable-model-invocation: true
---

# 环境确认验收标准

获取到 `.cannbot/环境信息.md` 时，立即用 question 工具按下列格式向用户发送问卷。每个标题对应问卷的一页（`questions` 数组的一项），`{...}` 为占位符，取值自环境信息文档的实测结果：

```json
{
  "questions": [
    {
      "question": "硬件设备检测结果：\n\n| 项目 | 内容 |\n|------|------|\n| NPU 设备版本 | {当前环境上的NPU设备版本，如Ascend910B} |\n| NPU 设备数量 | {N} |\n| NPU 设备状态 | {OK / 异常说明} |\n\n结论：具备 {archxx} 算子的上板验证条件，其余架构需要使用 cannsim 进行仿真验证。\n\n对此结果是否有疑问？",
      "header": "硬件设备",
      "options": [
        { "label": "已确认，无异议", "description": "检测结果无误，继续当前算子开发" },
        { "label": "有异议，停止开发", "description": "我来自行检查环境，立即停止当前算子开发" }
      ]
    },
    {
      "question": "软件环境检测结果：\n\n| 项目 | 内容 |\n|------|------|\n| CANN 版本 | {如 cann-9.0.1} |\n| torch_npu | {版本，import torch_npu 可用} |\n| Python 版本 | {如 3.10+} |\n| CMake 版本 | {如 3.16+} |\n\n结论：CANN 版本与 torch_npu 版本匹配，Python 与 CMake 安装正常，可以支撑算子开发任务。\n\n对此结果是否有疑问？",
      "header": "软件环境",
      "options": [
        { "label": "已确认，无异议", "description": "检测结果无误，继续当前算子开发" },
        { "label": "有异议，停止开发", "description": "我来自行检查环境，立即停止当前算子开发" }
      ]
    },
    {
      "question": "Git 凭据检测结果：\n\n| 项目 | 内容 |\n|------|------|\n| Git 凭据位置 | {~/.git-credentials / GITCODE_TOKEN 环境变量 / git credential.helper 配置 / 用户手动提供} |\n\n结论：{可找到 / 未提供} Git 凭据，{可支撑 / 无法支撑} Git 自动代码提交操作。\n\n是否同意将检测到的凭据位置记录到环境信息文档？（仅记录位置，不呈现、不索要凭据内容）",
      "header": "Git 凭据",
      "options": [
        { "label": "同意记录位置", "description": "保留环境信息文档中已记录的凭据位置" },
        { "label": "我另行提供", "description": "删除已记录的检测位置，凭据由我提供" },
        { "label": "不同意记录", "description": "删除已记录的检测位置，提交 PR 前再处理" }
      ]
    }
  ]
}
```

## 判定规则

- **通过**：硬件设备、软件环境两页均选「已确认，无异议」；Git 凭据页三选一给出明确结论即可，不作为环境不通过的判定项。
- **停止**：任一页选「有异议，停止开发」，判定不通过，立即停止当前算子开发，交用户检查环境。
- 用户通过自定义输入补充的说明随结论一并记录。

## 结果处理

1. 发出的问卷 json 落盘 `.cannbot/<算子名>/questionnaires/CP0-环境确认.json`；question 工具返回的用户回复（按 questions 顺序的选中 label 数组，含自定义输入）落盘同名 `.reply.json`。
2. Git 凭据页选「同意记录位置」时，保留环境信息文档「Git 凭据」节已记录的凭据位置；选「我另行提供」或「不同意记录」时，删除该节已记录的凭据位置，不再持久化。
3. 回传结构化摘要（状态 / 结论 / 交付件路径）。
