# 安装与升级验证

使用本工具集需要完整的 `ops-opmodel-toolkit` 目录、OpModel wheel 和用户项目。
wheel 可通过华为云 PyPI 镜像或离线 wheelhouse 安装；设备实现与计时另需项目的设备工具链。

## 安装版本与验证记录

- 默认安装：使用 `pip install --upgrade --pre --index-url https://repo.huaweicloud.com/repository/pypi/simple opmodel` 获取当前环境兼容的最新版本。
- 环境要求：以包声明的 `Requires-Python`、依赖及发布 wheel 的平台和 ABI 标签为准。
- 实际版本：通过 `importlib.metadata.version("opmodel")` 读取，随本次验证结果保存。
- 环境复现：验证后在用户项目中保存实际依赖记录和 wheelhouse，见[安装说明](install.md)。

[框架参考](framework/index.md)及七项工作流说明公开接口的使用方式。
验证记录只覆盖当次安装的包、输入和执行环境。升级 OpModel 后，核对实际接口、
模型配置及[使用限制](capabilities.md)，重新验证项目使用的工作流。

## 验证步骤

1. 读取并记录实际包版本，核对平台、Python 范围和运行依赖。
2. 配置 `OPMODEL_SKILL_DIR` 和 `OPMODEL_WORKSPACE`；skill 和安装包保持只读，
   模型、会话和结果写入用户工作区。
3. 运行快速入门的完整 case、分核与复评，再运行 DAG、联合搜索和适用示例；
   检查实际搜索覆盖、输出 scope、数值与后端检查状态。
4. 运行随附公开接口冒烟检查，核对本地链接与命令中的资源路径，实际加载 Python
   Stage，验证普通/物理入口、硬件选择及适用的数值声明。
5. 项目使用外部目标时，核对重评分后的目标修订、原评价与实际调用预算；使用统一
   规则或动态控制时，另验证适配器、场景、路径覆盖和不支持组合。随附冒烟示例
   不代表这些自定义输入均已验证，接口见[目标](framework/objectives.md)、
   [规则](framework/policies.md)和[控制](framework/runtime-control.md)。
6. 使用实现绑定成本时，核对源码与硬件身份、dtype／工作量范围，验证域外输入和
   失效来源被拒绝；导出候选后独立重载、复评。布局发现同时核对选项和诊断，
   不把部分程序失败或空选项直接解释为全能力不支持。
7. 记录已验证的工作流和限制；出现失败时保留输入、版本及日志，便于复现。

安装和离线命令见[安装说明](install.md)。目录内 `evals/` 提供行为评测用例，`tests/` 提供文档调用的
软件冒烟验证；这些检查不替代模型与真实设备的验收。

激活安装 OpModel 的环境并设置 `OPMODEL_SKILL_DIR` 后，可在用户项目复跑：

```bash
python "$OPMODEL_SKILL_DIR/tests/smoke.py" --work "$PWD/opmodel-smoke-run"
```

`opmodel-smoke-run` 须为新目录且不在系统 `/tmp` 下，测试报告与命令日志均写入该目录。
冒烟检查记录实际包版本，不因版本不同而拒绝运行；通过只说明随附示例和检查项通过。
`tests/interfaces.py` 由冒烟入口在新项目内调用，检查 Python Stage 的参数变化与
物理入口、普通聚合结果范围、量化声明和错误数值拒绝。测试只使用已安装 wheel 的
公开接口与合成输入，不证明任意模型、全部 Python 平台或设备性能已验证。
`evals/` 中的行为评测用例需通过 cannbot-skills 的评测框架执行；脚本冒烟验证与
代理行为评测分别记录结果。
