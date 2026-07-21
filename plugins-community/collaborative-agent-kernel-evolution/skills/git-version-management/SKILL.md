---
name: git-version-management
disable-model-invocation: true
description: Git 版本管理 - 算子工作区初始化、逐阶段提交、worktree 并行隔离与审计追踪
---

## 概述

在算子生成和进化工作流中集成 git 版本管理。git 管理是流水线内建行为，始终开启，不可选。

---

## 模块 1：Init（仓库初始化）

agent 启动、确认 op_name 和输出目录后，立即执行本模块。

### 1.1 确定仓库根目录

- **cake 模式**：`REPO_ROOT=output/{op_name}`
- **cake-evo 模式**：`REPO_ROOT=output/{op_name}_evo_{timestamp}`

### 1.2 初始化步骤

```bash
# 若目录不存在则创建
mkdir -p {REPO_ROOT}

# 检查是否已是 git 仓库，不是则初始化
if [ ! -d {REPO_ROOT}/.git ]; then
  git -C {REPO_ROOT} init -b main
fi
```

### 1.3 生成 .gitignore

若 `{REPO_ROOT}/.gitignore` 不存在，则创建。

**cake 模式**：

```
kernel_meta/
__pycache__/
*.pyc
build_out/
*.o
*.so
_msprof_work/
vendors/
ascend_op_pybind/
pybind_lib/
```

**cake-evo 模式**（额外忽略 worktree 目录和运行时产物）：

```
kernel_meta/
__pycache__/
*.pyc
build_out/
*.o
*.so
round_*/
_msprof_work/
vendors/
ascend_op_pybind/
pybind_lib/
```

### 1.4 初始 commit

```bash
# 用 git -C 指定仓库根，不依赖当前 shell 的 cwd —— 避免 Bash 调用之间
# cd 丢失导致 commit 落到上层 cake 代码仓（见下方「安全原则」）。
git -C {REPO_ROOT} add .gitignore
git -C {REPO_ROOT} commit -m "init: initialize {op_name} workspace"
```

若仓库已存在（已有 commit 历史）则跳过初始 commit。

### 1.5 安全原则（MUST）

以下规则适用于本 skill 每一处 `git add` / `git commit` 的调用：

1. **必须用 `git -C {REPO_ROOT} ...`** 指定仓库根，**禁止**使用 `cd {REPO_ROOT}` + 独立 `git` 行的写法。Bash tool 每次调用都是新 shell，`cd` 不跨调用保留；agent 若在不同 Bash 调用中先 `cd` 再 `git commit`，commit 会落到原 cwd（通常是上层 cake 代码仓），污染仓库历史。
2. **提交前必须验证仓库根**（guard）：

   ```bash
   # 把 {REPO_ROOT} 解成绝对路径，和 git 实际识别的 toplevel 做比对
   EXPECTED_ROOT=$(realpath {REPO_ROOT})
   ACTUAL_ROOT=$(git -C "$EXPECTED_ROOT" rev-parse --show-toplevel 2>/dev/null)
   if [ "$ACTUAL_ROOT" != "$EXPECTED_ROOT" ]; then
     echo "[git-vm] ABORT: expected repo root $EXPECTED_ROOT, got $ACTUAL_ROOT" >&2
     exit 1
   fi
   ```

   guard 触发即中止 commit，并回报 agent；不得忽略该错误继续提交。
3. **绝对不要在外层 cake 代码仓的 cwd 跑本 skill 的 `git commit`**。若发现 main cake repo 的 `git log` 里出现 `stage X: ...` 这种消息，即属于本规则被违反，需要 `git reset --mixed` 回退并重走流程。

---

## 模块 2：Commit（逐阶段提交）

每个阶段成功完成后立即执行。**阶段失败不 commit，保持上次成功状态。**

### 2.1 Commit 操作

每个阶段使用该阶段**明确对应的文件模式**执行 add，不使用 `git add -A`（避免意外提交 profiling 中间文件或远程同步残留）。**必须使用 `git -C` 指定仓库根，见模块 1.5 安全原则**：

```bash
# 先做 guard：确保 {REPO_ROOT} 真的是我们要提交的 repo
EXPECTED_ROOT=$(realpath {REPO_ROOT})
ACTUAL_ROOT=$(git -C "$EXPECTED_ROOT" rev-parse --show-toplevel 2>/dev/null)
[ "$ACTUAL_ROOT" = "$EXPECTED_ROOT" ] || { echo "[git-vm] ABORT: wrong repo root"; exit 1; }

# 再做 add + commit（都带 -C，不依赖 cwd）
git -C "$EXPECTED_ROOT" add <该阶段对应文件模式>   # 见 2.2 表格
git -C "$EXPECTED_ROOT" commit -m "{commit_message}"
```

### 2.2 阶段与 Commit 消息映射（cake 模式）

| 阶段 | git add 目标 | Commit 消息 |
|------|-------------|------------|
| 阶段 0（环境检测） | — | **无 commit**（纯检测，无文件产出） |
| 阶段 1（op-desc-generation 完成） | `"*.json"` | `stage 1: generate op description for {op_name}` |
| 阶段 2（reference-generation 完成） | `"*_reference.py"` | `stage 2: generate PyTorch reference for {op_name}` |
| 阶段 3（functional-conversion 完成） | `"*_functional.py"` | `stage 3: convert to functional API for {op_name}` |
| 阶段 3.5（算子类型检测） | — | **无 commit**（纯检测，无文件变更） |
| 阶段 4（ascend-call-generation 完成） | `"{op_name}Custom/" "{op_name}.cpp" "*_custom.py"` | `stage 4: generate AscendC project for {op_name}` |
| 阶段 5（dsl-baseline-generation 完成） | `"*_dsl.py"` | `stage 5: generate DSL baseline for {op_name}` |
| 阶段 6（dsl-lowering 完成） | `"{op_name}Custom/"` | `stage 6: lower DSL to AscendC for {op_name}` |
| 阶段 7（cake-code-review 完成） | `"{op_name}Custom/"` | `stage 7: pass coding red-line for {op_name}` |
| 阶段 8（ascendc-evaluation 完成，精度未通过） | `"evaluation_results.json"` | `stage 8: code review complete for {op_name}` |
| 阶段 9（评估精度通过，含重试） | `"evaluation_results.json"` | `stage 9: evaluation complete for {op_name} (speedup: {speedup}x)` |
| 阶段 10（Advisor 精炼完成） | `"{op_name}Custom/" "evaluation_results.json"` | `stage 10: advisor refinement for {op_name} (speedup: {new_speedup}x)` |
| 阶段 11（总结） | — | **无 commit**（纯输出，无文件变更） |
| 阶段 12（看板生成，精度通过时执行） | `"dashboard.html"` | `stage 12: generate dashboard for {op_name}` |

> **阶段 8 说明**：若评估精度未通过（仍在重试中），commit 消息使用 `stage 8`；精度最终通过后使用 `stage 9`。每次重试通过后均执行一次 commit。

### 2.3 Commit Hash 嵌入 evaluation_results.json

不引入独立日志文件。在 stage 9 commit 完成后，将 commit hash 写入 `evaluation_results.json` 的 `commit_hash` 字段：

```bash
EXPECTED_ROOT=$(realpath {REPO_ROOT})

# 先执行 stage 9 commit（见 2.2 和 1.5 guard）
git -C "$EXPECTED_ROOT" add evaluation_results.json
git -C "$EXPECTED_ROOT" commit -m "stage 9: evaluation complete for {op_name} (speedup: {speedup}x)"

# 获取 hash 并绑定到 evaluation_results.json（用 -C 读 HEAD，路径相对 EXPECTED_ROOT）
HASH=$(git -C "$EXPECTED_ROOT" rev-parse HEAD)
python3 -c "
import json, pathlib
path = pathlib.Path('$EXPECTED_ROOT') / 'evaluation_results.json'
with open(path) as f: d = json.load(f)
d['commit_hash'] = '${HASH}'
with open(path, 'w') as f: json.dump(d, f, indent=2)
"
git -C "$EXPECTED_ROOT" add evaluation_results.json
git -C "$EXPECTED_ROOT" commit -m "stage 9.hash: bind commit hash for {op_name}"
```

`evaluation_results.json` 最终包含 `"commit_hash"` 字段，hash 与评估结果直接绑定，无需额外日志文件。

### 2.4 cake-partial 模式（cake-evo 子 Agent）

cake-partial 在自己所属的 worktree branch 上按同样规则 commit，从阶段 5 开始：

| 阶段 | git add 目标 | Commit 消息 |
|------|-------------|------------|
| 阶段 5（dsl-baseline-generation 完成） | `"*_dsl.py"` | `stage 5: generate DSL baseline for {op_name}` |
| 阶段 6（dsl-lowering 完成） | `"{op_name}Custom/"` | `stage 6: lower DSL to AscendC for {op_name}` |
| 阶段 7（cake-code-review 完成） | `"{op_name}Custom/"` | `stage 7: pass coding red-line for {op_name}` |
| 阶段 9（评估精度通过） | `"evaluation_results.json"` | `stage 9: evaluation complete for {op_name} (speedup: {speedup}x)` |

---

## 模块 3：Worktree（cake-evo 专用）

### 3.1 Shared 阶段（main branch）

完成步骤 3（共享前置文件生成）后，在 main branch 上 commit（走模块 1.5 的 `git -C` + guard 规则）：

```bash
EXPECTED_ROOT=$(realpath {REPO_ROOT})
ACTUAL_ROOT=$(git -C "$EXPECTED_ROOT" rev-parse --show-toplevel 2>/dev/null)
[ "$ACTUAL_ROOT" = "$EXPECTED_ROOT" ] || { echo "[git-vm] ABORT: wrong repo root"; exit 1; }

git -C "$EXPECTED_ROOT" add "{op_name}Custom/" "{op_name}.cpp" "*_custom.py" "*_functional.py" "*.json"
git -C "$EXPECTED_ROOT" commit -m "stage 4: generate shared files for {op_name}"
```

### 3.2 创建并行变体 Worktree

在步骤 4.1 创建轮次目录时（取代直接 mkdir + cp），对每个并行索引 p：

```bash
# ⚠️ 必须使用绝对路径防止 cd 后相对路径嵌套（WF-001/WF-002）
REPO_ROOT=$(realpath output/{op_name}_evo_{timestamp})

# 创建 worktree，基于 main 创建新 branch
git -C ${REPO_ROOT} worktree add ${REPO_ROOT}/round_{r}/parallel_{p} \
    -b evo/{op_name}/r{r}-p{p} main

# 复制 shared 文件到 worktree（使用绝对路径，避免 cd 后路径嵌套）
cp -r ${REPO_ROOT}/shared/* ${REPO_ROOT}/round_{r}/parallel_{p}/
```

> **注意**：`round_*/` 在 cake-evo 的 `.gitignore` 中被排除（防止主仓库追踪），每个 worktree 拥有独立的 `.git` 文件指向自身 branch，因此 `git -C ${REPO_ROOT}/round_{r}/parallel_{p}` 操作的是该 worktree 的 branch。

### 3.3 子 Agent 在 Worktree 中 commit

cake-partial 子 Agent 的工作目录为 `{REPO_ROOT}/round_{r}/parallel_{p}/`，所有 commit 操作走 `git -C` + guard：

```bash
# {output_dir} 为该变体的绝对路径，由 cake-evo 在 prompt 中传入
EXPECTED_ROOT=$(realpath {output_dir})
ACTUAL_ROOT=$(git -C "$EXPECTED_ROOT" rev-parse --show-toplevel 2>/dev/null)
[ "$ACTUAL_ROOT" = "$EXPECTED_ROOT" ] || { echo "[git-vm] ABORT: wrong worktree root"; exit 1; }

git -C "$EXPECTED_ROOT" add <该阶段对应文件模式>   # 见 2.4 表格
git -C "$EXPECTED_ROOT" commit -m "{commit_message}"
```

> **WF-004**：禁止使用 `git -C {EVO_DIR} add -A` 提交 worktree 文件——`{EVO_DIR}` 的 `.gitignore` 中排除了 `round_*/`，这样提交会落到 main branch 而非变体 branch。必须用 `git -C {output_dir}` 直接在 worktree 内提交（worktree 的 `.git` 文件指向自身 branch）。

### 3.4 轮次结束：Merge 最优变体

每轮结束，选出最优变体（精度通过 + 最高加速比）后，merge 到 main：

```bash
git -C {REPO_ROOT} merge --no-ff evo/{op_name}/r{r}-p{p} \
    -m "merge best variant r{r}-p{p} (speedup: {speedup}x)"
```

> **--no-ff 原因**：所有变体 branch 共享 stage4-7 的相同 commit 对象，直接 merge 会 fast-forward，导致 main 上看不到 merge 节点，无法区分哪些 stage9 来自哪个变体。`--no-ff` 强制生成 merge commit，保留拓扑可读性。

### 3.5 保留所有 Branch 和 Worktree

merge 完成后，**不清理任何 branch 或 worktree**，包括已 merge 的变体。所有历史保留供审计和灵感采样。

### 3.6 分支拓扑

```
main ── shared commit ── merge best r1 ── merge best r2 ── ...
            |                 ^                 ^
            +-- evo/{op}/r1-p0 +               |
            +-- evo/{op}/r1-p1 (archived)      |
            +-- evo/{op}/r2-p0 ----------------+
            +-- evo/{op}/r2-p1 (archived)
```

### 3.7 Remote-CANN-Development 模式兼容性

在 `remote-cann-development` 场景下，worktree 在**本地**创建，编译/评估通过远程执行：

- **Worktree 创建**：与本地模式相同，`git worktree add` 在本地机器执行
- **编译/评估命令**：通过 `uv run ${CLAUDE_SKILL_DIR}/scripts/npu.py exec <target> "<cmd>"` 在远程 NPU 机器执行，结果写回本地
- **代码同步**：使用 `npu.py -w <worktree_path>` 显式指定 worktree 源路径，远程侧会落到 `{repo}.worktrees/{worktree_name}/` 做隔离，避免多 worktree 间互相覆盖：

  ```bash
  # ✅ 正确：显式传 -w，远程路径自动变为 {repo}.worktrees/parallel_{p}/
  uv run ${CLAUDE_SKILL_DIR}/scripts/npu.py \
    -w {REPO_ROOT}/round_{r}/parallel_{p} sync push <target>

  # 或以 worktree 根目录为 cwd，脚本会自动识别为 worktree
  cd {REPO_ROOT}/round_{r}/parallel_{p}
  uv run ${CLAUDE_SKILL_DIR}/scripts/npu.py sync push <target>
  ```

- **Commit 时机**：仍在本地执行（所有 `git commit` 操作不涉及远程），与本地模式规则一致

---

## 执行规则

1. git 管理是**内建行为**，agent 不得询问用户是否需要 git 管理
2. **只在阶段成功后 commit**，失败时不 commit
3. 所有编译产物（`.o`、`.so`、`build_out/`）通过 `.gitignore` 排除，不进入版本控制
4. **每阶段使用对应的显式文件模式执行 git add**（见 2.2/2.4 表格），不使用 `git add -A`
5. 初始化时若已是 git 仓库则跳过 init，不覆盖已有历史
6. **所有 `git add` / `git commit` 必须通过 `git -C {REPO_ROOT} ...` 显式指定仓库根**（见模块 1.5）。**禁止**依赖 `cd` + shell cwd；Bash tool 每次调用都是新 shell，cwd 不跨调用保留，写成 `cd ... ; git ...` 多行有可能让 commit 落到上层 cake 代码仓。违反此规则的典型征兆：`main` cake repo 的 `git log` 里出现 `stage X: ...` 消息。
