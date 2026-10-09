# 维护与来源

本 skill 从用户提供的 `template-review-skill-is-all-you-need` 迁入；迁入时源仓 HEAD 为
`82e59a982d2178a994932f9112dcebfce54017db`。保留检查器、规则文档、报告模板和夹具，
原仓 `CLAUDE.md` 的维护要点收录于本页。HCCLBot 的路径、报告和验证职责适配见 SKILL.md。

## 知识边界

- `00-domain-primer.md` 是随本 skill 分发的评审背景快照，不是 HCCL API 的权威定义。
  实际接口以被审 HCCL 版本的源码与官方文档为准；与开发 skill 文档冲突时回到同一源码版本核对。
- 评审规则与严重度在本 skill 内维护；ring/HD 的实验映射不得升级为未经验证的硬性判据。
- 不写死 HCCL 仓路径。检查器只读被审仓；`--base` 在临时目录解快照，不创建 worktree。

## 修改时同步检查

- `review_template.py` 的 `RULES`、references 章节索引、SKILL.md 参考表。
- 脚本严重度与 `01-review-method.md`、各族规则的口径。
- `check_common()` 的期望值与各族 scratch 表；API 字段、原语名单及两处 CMake 路径。
- `AlgoType` 仍从目标仓解析，不维护硬编码 API 镜像。
- `selftest.sh` 依赖的真实蓝本路径与夹具期望；不为使测试归零放宽规则。

## 回归

修改脚本或规则后运行：

```bash
bash "$SKILL_DIR/scripts/selftest.sh" "$HCCL_REPO"
```

离线行为回归覆盖 Ring 规则适用范围与 Markdown 围栏；控制流语义评测见
[09-regression-cases.md](09-regression-cases.md)。开发侧 `check_spec.py` 独立分发同口径的围栏解析器，
修改解析时同步运行两侧回归，避免新增运行时跨 skill 依赖。

不传仓路径且未设置 `HCCL_REPO` 时仅运行不依赖真仓的检查，其余会明确跳过，不能据此声称完整回归通过。
全仓基线须记录 HCCL commit 和工作树状态；动态 finding 数字不复制到入口文档。
