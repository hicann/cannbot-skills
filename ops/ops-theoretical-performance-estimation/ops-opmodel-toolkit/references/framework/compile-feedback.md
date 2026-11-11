# 编译反馈与交付资格

用于已有真实实现的联合优化与设计交接。编译结果与模型可行性、模型数值、设备正确性
和设备耗时分别保存；只做模型或方案时记为未执行，不强制启动编译。

说明证据更新顺序时，同时说明失败分支：编译失败、资源超限、设备正确性或计时
验证失败，均保留本次实际退出码、原始日志及已有资源／验证／计时报告，记录失败
范围与原因。未生成的工件保持缺失，不补造成功记录，也不回退使用已失效的旧设备数据。

## 准备反馈

项目编译适配器负责调用确定的工具链、保存实际退出码并提取资源报告。以下为结构
示意，占位摘要和资源值须替换为本次真实证据，不可直接作为通过记录：

```yaml
identity:
  source_sha256: <源码工件摘要>
  compiler: <工具链版本和构建身份>
  flags: [<有序编译参数>]
  soc: <目标SoC>
exit_code: 0
resources:
  vf_stack: {used: 4096, limit: 6144, unit: byte}
artifacts:
  source: {path: kernel.cpp, sha256: <摘要>}
  log: {path: compile.log, sha256: <摘要>}
  resource_report: {path: resources.json, sha256: <摘要>}
  binary: {path: kernel.bin, sha256: <摘要>}
```

证据根目录和导出目录放在用户工作区，不能位于系统 `/tmp` 下；导出须使用新目录。
工件路径相对于证据根目录，须在目录内并与实际摘要一致。源码摘要等于身份字段，纳入
所有生成代码、相关头文件及编译输入；可用完整源码包。成功需要二进制，失败不伪造
二进制；声明资源数值须带原始报告。资源含义与提取正确性由适配器核实，接口不解析
任意编译器日志，也不证明填写的工具链字符串与实际执行一致。

## 调用及状态

公开接口来自 `accel_model.model.validation.compile_evidence`：

```python
from accel_model.model.validation.compile_evidence import validate_compile_evidence

status = validate_compile_evidence(record, evidence_root, current_identity)
print(status)
```

调用片段的 `record` 为上述真实记录，`evidence_root` 为工件根目录，
`current_identity` 为当前待验证构建的身份。独立导出使用同模块的
`export_compile_evidence(record, root, destination, identity)`，在导出目录重新验证。

直接校验遇到非法结构或工件可抛出异常，保留错误原因；需要统一状态时使用同模块的
`compile_qualification(record=None, root=..., identity=...)`，无记录为 `NOT_RUN`，
非法／不可读取证据为 `INVALID`。

| 状态 | 对交付的含义 |
| --- | --- |
| `NOT_RUN` | 无编译反馈，不等于通过 |
| `PASSED` | 身份、退出码、资源及必要工件检查通过 |
| `FAILED` | 编译失败或资源超限，保留原始原因 |
| `STALE` | 源码、选项、工具链、SoC 或候选身份改变 |
| `INVALID` | 工件缺失、改写、结构非法或不可读取 |

会话接入使用 `accel_model.optimize.graph.global_search.device_evidence` 的
`import_compile_feedback(session, candidate_id, bundle_path, identity=...)`。
bundle 包含匹配当前候选的 `candidate_snapshot` 与 `evidence`；变更构建目标时用
`set_compile_target(session, candidate_id, identity)` 更新身份，再提交匹配反馈。

模型探索结果与 `compile_qualified_ranking` 分别呈现。编译通过不生成设备耗时，
失败记录可以作为探索证据保留和导出，但不能作为可部署候选。

## 导出与设备结果绑定

已有编译反馈时，设备证据中的源码、二进制摘要、compiler、有序 flags、soc 必须
同时匹配当前成功编译、冻结导出和实际运行版本。`versions.candidate.soc` 非空，
由设备验证器与原始运行记录核对；字符串相同不等于硬件已鉴定。

源码、选项、工具链、SoC 或二进制变化后重新导出并采集匹配运行证据。先前导出
未带反馈或带失败反馈，也须在编译成功后更新。不能用当前编译通过恢复旧二进制
的实测资格。设计文档分别列模型、编译、设备正确性和计时状态与剩余缺口。
