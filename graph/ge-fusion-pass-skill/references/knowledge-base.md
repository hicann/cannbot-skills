# 知识库映射（knowledge-base）

把散落的知识库（KB）资料整理成有序的"资料 → 讲什么 → 线上仓库/链接 → 哪个 skill 读"，外加文档检索规则。**所有本地路径靠 env 探测 / 脚本解析，缺失即降级，不硬编码默认安装路径。**

> **GE 资料已全量开源到 GE 仓**（`docs/` + `examples/`），**不再依赖单独的 ge-document 仓**。文档索引统一走 `ge-repo-map.md`。

> 资料与代码**都可以读**（包括 `examples/` 下的 pass 实现源码）——没有"只读 README、禁读 src"的限制。需要参考实现就去看。唯一保留的是**正确性纪律**（不是读的限制）：调用的 API 签名以真实文档/头文件为准（`tips/api-signature-gate.md`），匹配的 op type 以 dump 真值为准（`tips/dump-first-op-type.md`），不照抄别的样例的硬编码节点名。

## 一、KB 入口映射（按查阅优先级排序）

优先级从上到下：先 GE 仓（文档 + 样例），再教程/同类参考，最后本地生成物与头文件。

| # | 资料（本地定位） | 内容 | 线上仓库 / 链接 | 读者 |
|---|---|---|---|---|
| 0a | **开发主流程导航**：`references/pass-development-paradigm.md` | 从输入识别到验证的顺序、每步权威知识在哪份 reference/tip（导航层，不复制正文） | skill 内 | ②（主线）/①③ |
| 0b | **融合诊断树导航**：`references/fusion-troubleshooting.md` | pass 加载→执行→匹配→守卫→replacement→InferShape/format/engine→图与输出，逐节点指向权威 tip | skill 内 | ③（主线）/② |
| 1 | **GE 仓**：`$GE_REPO_PATH` —— **文档索引见 `ge-repo-map.md`**（任务路由表直达文件+章节） | 融合 pass 机制 `docs/zh/design/features/fusion_pattern_pass.md` + 实操指南与样例 `examples/fusion_pass/` + API `docs/zh/api/graph_engine_api/`（853 篇）+ 用户指南 `docs/zh/user_guides/` + 设计文档 `docs/zh/design/` + 源码 | 仓库：<https://gitcode.com/cann/ge>（根路径解析 + master 缓存见 §二） | ① / ② / ③ |
| 2 | **GE options / 开关参数说明**：`$GE_REPO_PATH/docs/zh/api/graph_engine_api/cpp/ge/options_params/options_parameters_description.md` | `GEInitialize` / `Session` 构造 / `AddGraph` 传入的配置参数，按全局 / session / graph 级检索；另见 `docs/zh/design/features/atc_raw_ge_options.md`（ATC 透传） | 同上（GE 仓内）；亦可参昇腾社区在线文档 <https://www.hiascend.com/document/> | ①（开关策略、注册阶段） |
| 3 | **Python pass 路径发现**：`$GE_REPO_PATH/docs/zh/user_guides/ge_python/env/ASCEND_GE_PY_PASS_PATH.md` | Python pass 插件的发现机制与取值格式 | 同上（GE 仓内） | ① / ② |
| 4 | **CANN 学习中心教程**：本地缓存（如有） | GE 图开发系统教程、示例与优化方案（背景） | 仓库：<https://gitcode.com/cann/cann-learning-hub> | ① / ② |
| 5 | **现场生成的 ES API 清单**：`es_all.generated_api.md` 或 `gen_es_api/` 产物 | 当前环境真实可用的 ES wrapper | 无线上——从**当前环境** op proto 现场生成，随环境而变 | ② / ③ |
| 6 | **CANN 头文件**：`ge::fusion/*.h`、`compliant_node_builder.h`… | 真实符号/签名回退核对 | 无线上——本地 CANN 安装（`$ASCEND_HOME_PATH`/`$ASCEND_OPP_PATH` 下） | ② |

> `$GE_REPO_PATH/docs/zh/api/graph_engine_api/` 是 **API 文档路径**，不能只当设计文档。第 5、6 项是本地/环境相关产物，**无线上仓库**，必须以本机实际为准。

## 二、GE 仓根路径解析与 master 缓存

**动态确认仓根，禁硬编码。** 统一走脚本，stdout 只输出一行绝对路径：

```bash
if [[ -n "${SKILL_ROOT:-}" && -f "$SKILL_ROOT/scripts/sync_ge_repo.sh" ]]; then
  GE_REPO_PATH="$(bash "$SKILL_ROOT/scripts/sync_ge_repo.sh")" \
    || echo "GE 仓不可达 → 按门禁 G3 如实标注证据缺失，不猜路径"
else
  echo "skill 根未显式提供 → GE 仓同步未运行，按门禁降级"
fi
```

解析顺序与语义（核心：**解析与联网解耦，默认纯只读、绝不联网**）：

| 情况 | 行为 |
|---|---|
| `$GE_REPO_PATH` 已设且是 GE 仓根（有 `examples/fusion_pass` + `docs/zh`） | **永远只读复用，绝不 fetch/reset**——与模式无关；可能是 fork、可能有本地提交或未提交改动 |
| 默认 / `--offline` + 既有缓存 | **直接用缓存，不刷新**（内容可能陈旧）；绝不联网 |
| 默认 / `--offline` + 无缓存 | 脚本 exit 1 → 调用方按门禁 G3 如实标注「GE 仓证据缺失」，**不猜路径、不静默联网** |
| `--allow-network` + 既有缓存 | `fetch --depth 1` + `reset --hard FETCH_HEAD` 刷新到 master 最新 |
| `--allow-network` + 无缓存 | `git clone --depth 1 -b master` 浅克隆到 `$GE_REPO_CACHE_DIR` |
| `--allow-network` 刷新失败但有旧缓存 | 回退用旧缓存，**明确告警"内容可能陈旧"** |

> - **默认不联网**：无参 / `--offline` 绝不执行 `git fetch` / `git clone`。需要最新 master 证据时，skill **必须先向用户取得授权**（说明对象 / URL / 用途 / 落盘目录 / 可复用本地 checkout 的配置方式 / 拒绝后的降级范围），再用 `--allow-network` 重跑。
> - 脚本**从不删除任何目录**：缓存目录若已存在且非 git 仓、非空，拒绝操作并提示手动清理。
> - **access log**：每次调用向 stderr 输出一行 `GE_REPO_ACCESS: mode=... network=yes|no ...`（含 URL / 目录 / 动作 / 结果 / revision / 未联网）。设置了 `$GE_REPO_ACCESS_LOG` 时另追加写入该文件，可作证据引用。stderr 记录不污染 stdout。

## 三、KB 不可达时的降级（配合需求分析文档 G1 / 门禁 G3）

- `$GE_REPO_PATH` 解析失败（脚本 exit 1）：如实标注证据缺失。需求分析里“注册阶段 / 开关策略”按统一 skill 阶段一的降级规则填**假设值 + 待确认标注**，不编结论。
- 现场 ES API 清单缺失：引导本地生成 `es_all`，或标注"ES wrapper 未确认"。
- CANN 头文件不可达：签名按 `tips/api-signature-gate.md` 标注 `【签名未核实，假设 X，待确认】`。

## 四、文档检索规则

- **单根**：GE 文档根就是 `$GE_REPO_PATH/docs`。中文在 `docs/zh/`；`docs/en/` 只镜像 `design/`、`user_guides/`、`contributions/`，**没有 `docs/en/api/`——API 文档只有中文**。**目录名是 `user_guides`（下划线），不是 `user-guides`。**
- **先查索引，再检索**：`ge-repo-map.md` §1 的任务路由表已把高频意图直连到「文件 + 章节」。**能查表就别 grep 全仓**——这是减少跳转的关键。
- **包含式 glob（不用前缀式 / 精确名）**：API 条目常有配套页——如 `PatternFusionPass_constructor.md`、`CustomPassContext_constructor_and_destructor.md`、`MatchResult_constructor_and_destructor.md`。只按单个类主页匹配会漏掉它们；先从 `graph_engine_api/README.md` 进入对应分层目录。
  ```bash
  rg --files "$GE_REPO_PATH/docs" -g '**/*REGISTER_CUSTOM_PASS*' -g '**/*SubgraphRewriter*' -g '**/*CompliantNodeBuilder*'
  find "$GE_REPO_PATH/docs" -name '*REGISTER_CUSTOM_PASS*' -o -name '*SubgraphRewriter*' -o -name '*CompliantNodeBuilder*'
  ```
- **先文件名后内容**：按 API 名找不到文件再做内容搜索：
  ```bash
  rg -n 'REGISTER_CUSTOM_PASS|CustomPassFn|GraphPtr|AddNodeByOp|SubgraphRewriter|CompliantNodeBuilder|DynamicInputRegister' "$GE_REPO_PATH/docs"
  ```
- **examples 可全量读**：`$GE_REPO_PATH/examples` 下 README、开发指南、以及 `fusion_pass/**/src/*` 样例实现源码都可搜可读、可作参考。**按目标场景选哪个样例见 `references/example-map.md`**（先 `find $GE_REPO_PATH/examples/fusion_pass -maxdepth 2` 对齐真实目录，布局随 GE 版本而变）。参考实现时仍按上面的正确性纪律：签名以真实文档/头文件为准、op type 以 dump 为准，不把别的样例的节点名写死照抄。
- **CANN 路径从 env 展开，禁硬编码**：涉及 CANN 安装目录（`opp/`、`op_proto/`、`all_ops.h`、`include/`、`lib64/`、`site-packages`、`atc`、`pyatc`）的检索/构建/编译，一律从 `$ASCEND_OPP_PATH`（OPP/op_proto 优先）、`$ASCEND_HOME_PATH`（CANN 根优先）、其次 `$ASCEND_TOOLKIT_HOME` 展开后的路径开始。**不硬编码、不猜默认安装目录**；即使 `CMakeLists.txt` 有 fallback 也不当真实安装路径。变量缺失、标志文件缺失或能力探测失败时，如实说明环境缺失并跳过相关验证。
- **优先级**：优先读 `$GE_REPO_PATH/docs` 的 API 条目、`examples/` 的开发指南与样例；仅当这些缺失、签名冲突或需核对真实符号时，才回退 CANN/GE 头文件。
- **委托子 agent 检索时**：必须把上述"从 env 展开、禁默认路径、缺失即返回"约束**原样写进子任务 prompt**，否则子 agent 会误探默认安装目录、甚至卡在外部目录权限交互挂死。详见 `tips/subagent-search-path-scope.md`。能自己在已知根下 `rg --files`/`find` 定位就不派子 agent。
