# 成文法门禁规则

本文件记录 CANNBot Skill 入库审查中的硬性规则。成文法规则必须与仓库自动化测试保持一致，不能由人工评分抵消。

## 规则来源

| 规则来源 | 执法入口 | 说明 |
|---------|----------|------|
| [CANNBot Skills 治理规范](https://gitcode.com/cann/cannbot-skills/blob/master/docs/GOVERNANCE.md) | 治理规范 | 定义成文法与判例法的职责边界 |
| `docs/STANDARDS.md` | 开发规范 | 定义目录、命名、渐进披露等基础约束 |
| `tests/lib/rules.yaml` | 规则配置 | 关键词、阈值、保留前缀等可配置规则 |
| `tests/lib/skill_validator.py` | 规则实现 | `S-STR-*`、`S-CON-*` 的结构和内容校验 |
| `tests/unit/skills/*.sh` | UT 门禁 | 批量执行并决定是否阻塞合入 |

## 阻塞项

出现以下任一情况，结论应为 `REJECT`：

- `tests/lib/skill_validator.py` 输出 `error`
- `SKILL.md` 缺少 frontmatter 或正文为空
- `name` 与目录名不一致
- `name` 不符合 kebab-case 或使用保留前缀
- `description` 缺少触发条件，且未显式设置 `disable-model-invocation: true`
- `references/` 目录存在但没有可用 Markdown 资料
- frontmatter 出现 XML tag 注入模式
- 同名 skill 已存在

## 建议命令

单个 skill 门禁：

```bash
python tests/lib/skill_validator.py validate-skill <path-to-SKILL.md>
```

全仓 fast 门禁：

```bash
bash tests/run-tests.sh --fast
```

## 维护要求

- 新增或调整成文法规则时，必须同步更新 `tests/lib/rules.yaml`、`tests/lib/skill_validator.py` 或对应 `test-*.sh`。
- 审查报告中不能把成文法 `error` 降级成 warning。
- 成文法负责“是否格式合规”；设计价值、边界合理性和演进方向交由判例法审查。
