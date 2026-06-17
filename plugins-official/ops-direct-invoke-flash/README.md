# ops-direct-invoke-flash

Ascend C 算子开发工具 **CANNBot（Flash 版）**。从 CPU 函数、数学公式、代码片段或文本描述出发，端到端构建并验证一个高性能 Ascend NPU 核函数。

## 能力概览

完整工作流覆盖：环境探测 → 定义/设计文档 → 增量实现 → NPU 验证 → 收尾文档。

## 安装

使用仓库内的 `init.sh` 将 skill、agent 与配置以软链接方式安装到目标工具。

```bash
# 在当前项目中为 opencode 安装
./init.sh project opencode

# 全局安装到 opencode（~/.config/opencode）
./init.sh global opencode
```

## 使用

安装后在对应工具中触发 skill，并提供算子来源：

```
/ops-direct-invoke-flash 开发一个 Abs 算子
```

默认在 `operators/` 目录下开发。仓库已内置 `operators/add`、`operators/sqrt` 作为结构参考。
