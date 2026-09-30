# Ascend950 源码与产物路径

`PLUGIN_DIR` 为已安装的 `tilelang-op-orchestrator` 插件真实根目录。源码准备阶段输出下列两个固定目录，不接受路径参数或环境变量覆盖，也不按旧仓库名称搜索其他工作区。

| 含义 | 固定目录 | 传给角色与技能的值 |
|---|---|---|
| TK：算子参考源码 | `PLUGIN_DIR/repositories/Ascend950/ops-tilelang/` | `OPS_TILELANG_DIR` / `{ops_tilelang_repo}` |
| TL：TileLang 框架源码 | `PLUGIN_DIR/repositories/Ascend950/tilelang/` | `TILELANG_DIR` / `{tilelang_repo}` |

TK 中的实现位于 `src/cann_ops_tilelang/`，测试和独立 reference 位于 `tests/`。TL 中的框架实现位于 `tilelang/`、`src/ascend/`、`src/backend/`，示例和框架测试位于 `examples/ascend/`、`testing/ascend/`。这些路径均相对于对应源码根目录，不相对于用户工作目录或 Skill 目录。

所有角色沿用这两个位置。准备阶段仅浅克隆主仓库；查证需要子模块内部实现时，按 [Ascend950 工作流](../workflows/Ascend950/workflow.md#源码准备) 核对 `.gitmodules` 并单独获取所需子模块，不默认递归下载依赖。源码或符号仍缺失时报告具体缺口，不到其他仓库寻找同名文件。实际 Python 导入位置单独记录；`SOURCE_READY` 仅表示两个主仓库的参考源码就绪，不代表编译依赖完整、运行环境已安装或与参考版本一致。

`{code_dir}` 是当前待优化算子工程或本轮 baseline 副本，与 TK、TL 分别记录；`{operator_relpath}`、`{test_file}` 相对于 `{code_dir}`。优化方案、profiling、报告和归档仍写入工作流规定的用户项目 `operators/` 目录。切换优化副本只更新 `{code_dir}`，运行前核对目标算子及其本地依赖从该副本导入。Skill 的 `references/`、`scripts/` 和 `assets/` 从该 Skill 的 Ascend950 正文目录定位。

精度门禁使用基线确认的完整目标测试清单和测试配置。TK 的 `tests/testing/generator.py` 读取 `OPS_TILELANG_TEST_LEVEL`，默认值为 1；仅运行使用该生成器的测试时记录并沿用此变量，扩展测试时按该生成器实际定义选择级别。生成算子的独立 pytest 和 TL 的测试不假定支持此变量。最终验收不能因为迁移路径而减少用例或放宽精度。
