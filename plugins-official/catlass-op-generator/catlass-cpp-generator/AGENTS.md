---
description: CATLASS C++ Linear Attention 五阶段开发工作流。仅处理已确认属于 linear_attention 的 CATLASS C++ 新工程或带专用 workflow marker 的既有工程。
mode: primary
skills:
  - catlass-cpp-interface
  - catlass-cpp-reference
  - catlass-cpp-design
  - catlass-cpp-develop
  - catlass-cpp-test
  - catlass-cpp-knowledge
# generated-by-cannbot-catlass-cpp-generator-init
---

# CANNBot CATLASS C++ Linear Attention

本插件独立执行 PR1069 的五阶段流程。它不接管普通 CATLASS 算子。直接入口收到经数学分类确认的
非 Linear Attention 请求时，提示改用 `catlass-op-generator`。

## 数学分类

新工程由需求接收阶段根据需求中的数学信息确认 `algorithm_family`。数学信息充分时，Agent 直接
分类为 `linear_attention` 或 `legacy`，不要求用户显式说出 Linear Attention、GDN、KDA 等
专用名词。分类只能依据已确认的数学算法，禁止仅按算子名称或自由文本关键词猜测；信息不足时
保持 `pending`，补充数学需求后重新分类。selector 只消费分类结果和已有工程 marker，不负责解析
名称或请求文本。

## 启动与恢复

1. 执行 `scripts/select_operator_workflow.py`，只使用需求接收阶段确认的 `algorithm_family` 和已有工程 marker。
2. 返回 `linear_attention` 后，读取当前阶段 Skill 的完整 `SKILL.md`。
3. 新工程由 `catlass-cpp-interface/scripts/init_operator_project.sh` 初始化。
4. 每次继续前校验 `operators/<name>/docs/workflow.json`；状态文件是跨会话唯一机器权威。
5. interface 阶段把 `target_architecture` 从 `pending` 冻结为 `atlas_a2_a3` 或 `ascend950`；后续设计、
   开发和测试必须与该字段一致，不能混用 `2201/Arch::AtlasA2` 与 `3510/Arch::Ascend950` 路径。
6. 严格按 interface -> reference -> design -> implementation -> validation 推进，只有 full 验收可进入 complete。

## 五阶段

| 状态 | Skill | 核心产物 |
|---|---|---|
| interface | `catlass-cpp-interface` | `docs/api.md` 与冻结的 operator contract |
| reference | `catlass-cpp-reference` | 唯一 `reference.py`、definition、precision policy |
| design | `catlass-cpp-design` | 按完整参考推演（可保存 `docs/design.full.md`），交付两章 `docs/design.md` |
| implementation | `catlass-cpp-develop` | 按 Stage 实现、构建和定向精度证据 |
| validation | `catlass-cpp-test` | 全量精度、性能与最终验收 |

`catlass-cpp-knowledge` 是横向能力，不是第六阶段。先 compact query，再 get 选定 concept；
知识不能替代用户 contract、固定源码、CPU 标杆或实际运行证据。

## 固定边界

- `reference/reference.py` 是唯一可编辑标杆源码。
- 工作流 ID 固定为 `catlass-linear-attention-v1`。
- `target_architecture` 是架构分支的机器权威；A2/A3 使用 `atlas_a2_a3`，A5 使用 `ascend950`。
- 技术知识只位于 `knowledge/catlass/`、`knowledge/workflow/` 与 `knowledge/operator/<family>/`。
- 性能不达标仍属于 validation，不能创建 optimize 阶段。
- 状态恢复必须使用 validator 允许的 issue_type/resume_from 组合。
- 非 Linear Attention 和没有专用 marker 的既有工程继续使用原插件 Step 1-7。
- 单用例连续运行 60 秒无返回视为 kernel 超时；清理进程和设备资源后记录现象、Stage、
  TilingKey 和 `blockDim`。

始终使用中文交流。
