---
description: CATLASS C++ 领域算子五阶段开发工作流（Linear Attention / BSA Arch22 / SparseFlashMLA）。仅处理已确认属于 linear_attention、block_sparse_attention 或 sparse_flash_mla 的 CATLASS C++ 新工程或带专用 workflow marker 的既有工程。
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

# CANNBot CATLASS C++ 领域算子（Linear Attention / BSA Arch22 / SparseFlashMLA）

本插件独立执行五阶段流程。它不接管普通 CATLASS 算子。直接入口收到经数学分类确认的
非本插件支持 family（非 linear_attention / block_sparse_attention / sparse_flash_mla）请求时，提示改用 `catlass-op-generator`。

## 数学分类

新工程由需求接收阶段根据需求中的数学信息确认 `algorithm_family`。数学信息充分时，Agent 直接
分类为 `linear_attention`、`block_sparse_attention`、`sparse_flash_mla` 或 `legacy`，不要求用户显式说出 Linear
Attention、GDN、KDA、BSA 等专用名词。数学上以 Q 块 × KV 块粒度的 0/1 block mask 表达稀疏的
注意力（块稀疏注意力）分类为 `block_sparse_attention`。符合共享 KV、窗口/压缩/索引注意力和配套 metadata
契约的 MLA 计算分类为 `sparse_flash_mla`，从对应家族索引核对数学与接口。分类只能依据已确认的数学算法，禁止仅按
算子名称或自由文本关键词猜测；信息不足时保持 `pending`，补充数学需求后重新分类。selector 只
消费分类结果和已有工程 marker，不负责解析名称或请求文本。

## 启动与恢复

1. 执行 `scripts/select_operator_workflow.py`，只使用需求接收阶段确认的 `algorithm_family` 和已有工程 marker。
2. 返回已注册算法族后，读取当前阶段 Skill 的完整 `SKILL.md` 和对应家族 [知识索引](knowledge/operator/index.md)。
3. 新工程由 `skills/catlass-cpp-interface/scripts/init_operator_project.sh <name> --algorithm-family <family>` 初始化。
4. 每次继续前校验 `operators/<name>/docs/workflow.json`；状态文件是跨会话唯一机器权威。
5. interface 阶段把 `target_architecture` 从 `pending` 冻结为目标家族支持的架构（当前 SparseFlashMLA 和 BSA Arch22 仅 `atlas_a2_a3`，Linear Attention 可选 `atlas_a2_a3` 或 `ascend950`）；后续设计、
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

## BSA Arch22 知识入口与条款

`block_sparse_attention`（BSA / 块稀疏注意力，目标 DAV_2201）工程启动后，经
`catlass-cpp-knowledge` 先 compact query family `sparse-attention`，先读 `scope-and-evidence.md`
与 `blockxy-generalization.md`，再按各阶段需要读取其余条目（含任务边界同步、高吞吐结构规则、
设备行为笔记等）。

- 用户提供的 golden 与冻结评测 contract 决定语义、精度与合法 shape；不得删减冻结用例、放宽容差。
- QK/PV 等矩阵计算必须使用 CATLASS 组件；稀疏索引、online softmax、rescale、LSE 与同步可封装为
  device 模块。
- 资料授权分流：默认及约定的盲生成/盲验证遵守源码隔离；用户明确授权源码可见学习时按授权范围执行。
- 性能按完整调用计时口径验收（必要预处理/后处理不得漏计），达标线以冻结 contract 为准。

## SparseFlashMLA 知识入口

`sparse_flash_mla` 读取 [家族索引](knowledge/operator/sparse-flash-mla/index.md)，
按当前阶段加载主算子、独立 metadata 及对应模式的知识；通用五阶段与恢复流程保持一致。

## 固定边界

- `reference/reference.py` 是唯一可编辑标杆源码。
- 工作流 ID 按 family 固定：`linear_attention` → `catlass-linear-attention-v1`；
  `block_sparse_attention` → `catlass-block-sparse-attention-arch22-v1`；
  `sparse_flash_mla` → `catlass-sparse-flash-mla-v1`。
- `target_architecture` 是架构分支的机器权威；A2/A3 使用 `atlas_a2_a3`，A5 使用 `ascend950`。
- 技术知识只位于 `knowledge/catlass/`、`knowledge/workflow/` 与 `knowledge/operator/<family>/`。
- 性能不达标仍属于 validation，不能创建 optimize 阶段。
- 状态恢复必须使用 validator 允许的 issue_type/resume_from 组合。
- 非本插件支持 family 和没有专用 marker 的既有工程继续使用原插件 Step 1-7。
- 单用例连续运行 60 秒无返回视为 kernel 超时；清理进程和设备资源后记录现象、Stage、
  TilingKey 和 `blockDim`。

始终使用中文交流。
