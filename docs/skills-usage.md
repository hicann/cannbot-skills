# CANNBot Skills 使用样例

本文档汇总各 Skill 的典型使用样例。每个 Skill 给一段可直接复制、按需替换占位符的 prompt。

---

## Ascend C 算子开发

### ascendc-registry-invoke-to-direct-invoke

注册算子转 `<<<>>>` kernel 直调（算子迁移，不是从零开发）。

```
请使用 ascendc-registry-invoke-to-direct-invoke 技能，完成如下算子迁移：

【任务】将 rms_norm 算子从注册算子工程迁移到当前代码仓的 `<<<>>>` kernel 直调形式。

【源码路径】
- 算子原型与 tiling（host 侧）：<源工程 op_host 绝对路径>
- kernel 入口函数（device 侧）：<源工程 op_kernel 绝对路径>
- torch 接口定义（可选）：<torch_adapter 绝对路径>
- 原始测试脚本（用于精度对齐）：<test 脚本绝对路径>

【目标】
- 目标代码仓：当前工作目录
- 目标平台版本：dav-2201
- 交付标准：kernel + tiling + host 独立可编译运行，精度与原始测试脚本对齐
```

**使用建议**：

- 路径写**绝对路径**，让 skill 不必猜测源码位置。
- 明确**平台版本**（如 `dav-2201` / `dav-3510`），影响 cmake 配置与目标仓约定对齐。
- 没有 torch / 测试脚本时对应行可删，但"精度对齐"需至少保留一份可跑的原始用例作为参考系。
- 三原则（kernel 零修改 / tiling 数学零修改 / 只改框架胶水）、全量迁移、先确认交付边界等行为已内置在 SKILL.md，prompt 里不必重复。


### ascendc-direct-invoke-to-registry-invoke

 `<<<>>>` kernel 直调转注册算子（算子迁移，不是从零开发）。

```
请使用 ascendc-direct-invoke-to-registry-invoke 技能，完成如下算子迁移：

【任务】将/path/rms_norm.asc <源kernel直掉文件绝对路径> kernel直调工程接入ACLNN/GEIR接口，生成在/path/rms_norm_single_op <迁移后的绝对路径>，目标芯片ascend910b <目标芯片版本，建议和当前运行环境一致，否则无法进行结果验证>

```

**使用建议**：

- 路径写**绝对路径**，让 skill 不必猜测源码位置。