# Sparse Flash MLA

本家族根据调用者提供的计算标杆识别共享 KV、窗口/压缩/索引选择、共同 Softmax 与 sink，
生成 SparseFlashMla 主算子及独立 SparseFlashMlaMetadata。当前硬件工程知识面向
Ascend 910B / AtlasA2（CATLASS_ARCH=2201）；仅要求普通 SWA 时只实现该模式，
但主算子与 metadata 仍需成对交付，不把回归用例子集当作合法输入的全部范围。

知识以主题文档完整保存，专用目录没有可执行脚本或完整算子源码资产。
用例构造、诊断、性能评价、参考模型和构建准备均提供规则与步骤，需要自动化时在本轮生成工程中实现。
目标 CANN/CATLASS 的实际声明、库与编译器仍在构建时核对，不以知识文档代替设备证据。

| 文档 | 内容 |
| --- | --- |
| [workflow.md](workflow.md) | 数学分类、五阶段工作流、知识检索、调用材料和交付边界 |
| [computation.md](computation.md) | SWA/CFA/SCFA 数学、窗口/压缩/索引、共同归一化和舍入 |
| [interface.md](interface.md) | 两个完整 Python/ACLNN 接口及一般 SWA 的 dtype/layout/stride/长度/sink/LSE 契约 |
| [metadata.md](metadata.md) | 独立任务表、游标、FA/FD 分核、生产消费与初始化 |
| [development.md](development.md) | 分阶段开发、工作量检查、动态工程、正式部署和 SDK 构建工具准备 |
| [pipeline.md](pipeline.md) | 双槽跨任务协议、片上结构、PV 尾块、三 actor 状态模型 |
| [performance.md](performance.md) | decode/prefill、分核与复用、5+5采样、基线解析和逐例门槛 |
| [validation.md](validation.md) | 用例构造、负例、完整输出、动态诊断 ABI、ABAB、原 ATK 与 sanitizer 验收 |

## 各阶段读取顺序

1. **interface**：workflow、computation、interface、metadata。按数学识别模式，冻结两个入口和目标架构。
2. **reference**：computation、validation 的数学参考和精度规则，读取调用者本轮 golden、原 ATK 和用例。
3. **design**：前述接口与数学知识，以及 development、pipeline、performance、validation。普通 SWA
   必须读完整握手、结构与验收内容后设计；长 prefill 同时读 performance 中的连续窗口推导。
4. **implementation**：按已冻结设计使用 metadata、development、pipeline，逐 Stage 实现并以 validation 定向验证。
5. **validation**：validation 和 performance，真实完成两个入口、原 ATK、完整 O/LSE、输入复用及逐例性能验收。

五阶段的工作流状态与 kernel 计算阶段不同，不把 metadata 当作普通 SWA 中缺省的稀疏 V0。
普通 SWA 从所需的 C1/V1/C2/V2 和实际寻址/可选 pack 推导流水；具体生命周期按 pipeline 完整正文核对。

## 直接定位

- [两个公开 ACLNN 声明](interface.md#api--5-aclnn-两段式-c-接口)、[一般 SWA 合法域](interface.md#swa-contract)。
- [双槽协议](pipeline.md#protocol)、[片上结构与 PV 尾块](pipeline.md#structure--pv-tail)。
- [SDK 构建工具准备](development.md#sdk-build)、[三 actor 参考模型](pipeline.md#model)。
- [用例生成和静态覆盖审计](validation.md#case-recipes)、[动态诊断运行器](validation.md#diagnostics)。
- [流水设备验收](validation.md#pipeline-validation)、[逐例性能证据校验](performance.md#acceptance)。

调用输入由用户本轮提供，路径、完整 spec、SHA256 与适用范围见 workflow，不使用固定服务器路径或历史样本补齐。
用例必须显式选择真实 case_id 或全部用例；缺例、重复或空选择不得显示通过。新候选采用5次预热、
5次正式采样且全部参与均值；默认结束门槛为相同 case 的基线时间/候选时间≥0.8，研究任务按其指定目标执行。
直调诊断、参考模型和文档检查不能替代正式 ATK、真实 metadata/main 或实际设备性能证据。

运行时 reindex 可重建本索引，完整阶段要求仍保存在 workflow 等 concept 中。知识工具 get 只接收文件路径，
章节锚点用于阅读定位；不要把 `#...` 附到 get 的 path。恢复已有工程时核对运行时知识副本，initialize 不覆盖旧文件。
