# 平台和编程模式矩阵

下表是基于 `cann/asc-devkit` 文档的参考快照。它用于初步选择接口，不替代目标 CANN 版本的 API 文档、公开头文件和实际编译验证。

| 接口 | Ascend 950 | A3 | A2 | 310P/310B | 910/Kirin |
| --- | --- | --- | --- | --- | --- |
| `printf` | SIMD、SIMT、SIMT VF、SIMD VF | 主要为 SIMD | 主要为 SIMD | 当前 `printf` 文档不支持 | 不支持 |
| `assert`/`ascendc_assert` | SIMD、SIMT/VF 变体 | SIMD | SIMD | 310P SIMD 文档支持；310B 不支持 | 不支持 |
| `__trap` | SIMT/SIMT VF | 当前文档不支持 | 当前文档不支持 | 当前文档不支持 | 不支持 |
| `asc_dump` | SIMD，另有受限 VF/Buffer 变体 | SIMD | SIMD | 当前文档不支持 | 不支持 |
| `clock` | SIMD、SIMT | SIMD | SIMD | 当前文档不支持 | 不支持 |
| `asc_time_stamp` | SIMD、NPU 实机运行 | SIMD、NPU 实机运行 | SIMD、NPU 实机运行 | 当前文档不支持 | 不支持 |

## 来源基线

- 文档仓库：`cann/asc-devkit`；本次核对提交：`6597014b0655a56ad8600f4825e9f616cdff3fbd`。
- 相对目录：`docs/zh/api/Utils-API/tuning_interface/`。
- 对应文档：`printf.md`、`assert.md`（同时说明 `ascendc_assert`）、`__trap.md`、`asc_dump.md`、`clock.md`、`asc_time_stamp.md`。
- 该目录在核对时无工作树改动。提交基线不等同于用户安装的 CANN 版本，也不代表本技能已完成各平台 NPU 实机验证。
- 使用前记录目标 SDK 版本及实际文档来源；与快照不一致时以目标版本资料重新核对，不直接沿用本表。

## 使用规则

1. “主要为 SIMD”表示当前公开 tuning 文档没有为 A2/A3 承诺 SIMT 或 VF 变体，不代表所有其他开发路径都没有相关能力。
2. 目标架构未知时，先获取 `__NPU_ARCH__`/SoC 信息，再选择 API；不要用产品名称推断每个接口都支持。
3. `__trap` 的上述基线文档只列出 Ascend 950PR/950DT。即使头文件存在其他条件编译分支，也必须以目标版本公开文档为准。
4. SIMD VF 的 预留 UB 空间、FIFO 和编译选项限制需要单独检查，不能套用普通 SIMD 的容量结论。
5. API 支持不等于当前调用场景可运行；还要检查 CPU/NPU、直接调用/入图、编译模式和版本。
