# Kernel 模式 TTK 执行验收

> **前置条件**：任务 5 初步验证通过。

**目的**：在 TTK kernel 模式下验证 golden 函数与算子输出的精度比对结果。

## 6a：单用例探测

```bash
cd {ops_test_kit_path} && python3 -m ttk kernel \
  -i {whitebox_dir}/ttk_{op_name}_cases_low.csv \
  --plugin {plugin_path} \
  -t {random_testcase_name} \
  --pc 1 \
  --seed 42
```

- 命令固定，仅 `-i`/`--plugin`/`-t` 根据算子变化
- `-t` 指定**随机挑选**的一个用例（避开 case00000，优先选中间位置如 case00042），防止首个用例恰巧正常而掩盖 CSV 格式问题
- 不传 `-b`/`-d`/`-c`，TTK 根据用例自行选择编译模式

### 验收检查项

| # | 检查项 | 通过标准 | 日志关键字 |
|---|--------|---------|-----------|
| 1 | Custom golden 加载 | 日志出现加载成功 | `Loaded custom golden: kernel.{op_name}` |
| 2 | 编译成功 | 编译通过 | `Compilation Result: SUCC` |
| 3 | Golden/Output 一致 | Shape 和 Dtype 匹配 | `Golden Shape` == `Output Shape` |
| 4 | 精度比对 | DYN_GOLD 结果输出 | `DYN_GOLD:` 后有数值百分比 |

**失败处理**：见「失败诊断与纠错流程」。

## 6b：采样门禁执行（必须执行）

> **前置条件**：6a 全部通过。

使用 nohup 后台执行，防止会话超时：

```bash
cd {ops_test_kit_path} && setsid bash -c 'exec python3 -m ttk kernel \
  -i {whitebox_dir}/ttk_{op_name}_cases_low.csv \
  -o {whitebox_dir}/ttk_{op_name}_cases_low_sample_result.csv \
  --plugin {plugin_path} \
  --tc 10 \
  --pc 1 \
  --seed 42 > {whitebox_dir}/ttk_batch_6b.log 2>&1' &
disown
```

- `--tc 10`：从 low CSV 中随机取 10 条作为门禁验证集，默认 N=10
- `-o`：输出 `ttk_{op_name}_cases_low_sample_result.csv`

执行后提供进度查询指令：

```bash
pgrep -af "ttk kernel" | grep -v pgrep
grep -c "Performance result" {whitebox_dir}/ttk_batch_6b.log
tail -5 {whitebox_dir}/ttk_batch_6b.log
```

执行完成后验收：

```bash
python3 -c "
import csv
with open('{whitebox_dir}/ttk_{op_name}_cases_low_sample_result.csv') as f:
    rows = list(csv.DictReader(f))
passed = sum(1 for r in rows if r.get('perf_status','') == 'PASS')
print(f'{passed}/{len(rows)} = {passed/len(rows)*100:.1f}%')
"
```

### 验收检查项

| # | 检查项 | 通过标准 | 检查方法 |
|---|--------|---------|---------|
| 1 | perf_status 通过率 | ≥ 90% 用例 PASS | 统计 `ttk_{op_name}_cases_low_sample_result.csv` 中 `perf_status` 列 |
| 2 | 编译成功率 | 全部 SUCC | 日志无 `Compilation Result: FAIL` |
| 3 | 无内存越界 | 全部 `memory_oob_status` 为空或 PASS | `ttk_{op_name}_cases_low_sample_result.csv` 检查 |

**失败处理**：见「失败诊断与纠错流程」。若批量通过率 < 90%，先执行步骤一区分批量干扰。

## 失败诊断与纠错流程

> 适用于 6a/6b/6c 中任何用例执行失败的场景。6a/6b 的「失败处理」均指向本节。

### 步骤一：区分批量干扰 vs 真实用例问题

**现象**：TTK 批量执行（`--tc N`，N > 1）时，部分用例 PASS、部分 FAIL 交替出现，但单独执行同一 FAIL 用例却 PASS。这是 TTK 多进程执行环境下的已知干扰现象（进程间共享设备状态导致 tiling 上下文污染）。

**判定方法**：将所有批量执行中 FAIL 的用例逐个单独执行：

```bash
cd {ops_test_kit_path} && python3 -m ttk kernel \
  -i {whitebox_dir}/ttk_{op_name}_cases_low.csv \
  --plugin {plugin_path} \
  -t {fail_case_id} \
  --pc 1 --seed 42
```

| 单独执行结果 | 判定 | 处理 |
|-------------|------|------|
| PASS | 批量干扰，非用例问题 | 记录为 TTK 工具限制，不修改用例。全部 FAIL 用例单独执行均 PASS → 门禁视为通过 |
| FAIL | 真实用例问题 | 进入步骤二 |

### 步骤二：收集失败用例的输入输出空间信息

对每个真实失败的用例，从 `S5_mapped_cases_low.json` 提取：

- `case["params"]`：路由参数（x_dtype、scalar_dtype、path、key 等）
- `case["tensors"]["inputs"]`：各输入的 shape、dtype
- `case["tensors"]["outputs"]`：各输出的 shape、dtype

从 TTK 日志中提取错误类型：

| 日志关键字 | 错误类型 | 常见原因 |
|-----------|---------|---------|
| `OPTILING_FAILURE` + `CheckScalar` / `Check` | tiling 参数校验失败 | dtype 组合非法、shape 超限 |
| `Compilation Result: FAIL` | 编译失败 | kernel 不支持该 dtype/shape |
| `DYN_GOLD:` 非 100% | 精度不匹配 | golden 函数实现有误 |

### 步骤三：与 S5_mapping_spec.md 交叉核查

将失败用例的输入输出空间信息与 `S5_mapping_spec.md` 中声明的约束逐条比对：

| 核查维度 | 检查内容 | 对照来源 |
|---------|---------|---------|
| dtype 组合 | x_dtype 与 scalar_dtype 的组合是否合法 | S5_mapping_spec §dtype + 源码 SUPPORT_DTYPE_COMB |
| shape 约束 | 各 tensor 的 shape/dtype 是否满足算子约束 | S5_mapping_spec §shape 构造参数 |
| 输出推导 | 输出 shape/dtype 是否与输入一致 | S5_mapping_spec §输出 tensor |
| 参数语义 | case 字段名是否与 tiling 源码变量语义一致 | S5_mapping_spec + tiling 源码 |

### 步骤四：溯源错误传播链

从失败点向上游逐层追溯，定位根因：

```
TTK 执行失败（tiling/编译/精度）
  ↑
CSV 字段值错误（ttk_{op}_cases_low.csv）
  ↑
S5_case_mapper.py 映射逻辑错误（如从错误字段读取 dtype）
  ↑
S2P2_cases.json 枚举数据错误（如跨参数约束未过滤）
  ↑
S2P2_param_def.json 参数定义/约束遗漏（如 per_dtype 未结构性过滤）
```

**关键检查点**：
- 跨参数约束（如 dtype 组合限制）是否在 `per_dtype` 中结构性过滤，而非仅在文本中描述
- case 字段名是否与 tiling 源码变量语义一致
- mapper 是否从正确的 case 字段读取值

### 步骤五：修正并重新生成

按依赖顺序从根因文件开始向下游修正：

| 顺序 | 文件 | 操作 |
|------|------|------|
| 1 | `S2P2_param_def.json` | 修正 per_dtype / 约束定义 |
| 2 | `S2P2_gen_cases.py` → `S2P2_cases.json` | 重新运行枚举 |
| 3 | `S5_mapping_spec.md` | 修正参数语义描述 |
| 4 | `S5_case_mapper.py` / `S5_network_mapper.py` | 修正映射逻辑 |
| 5 | `S5_merge_expand.py` → low/high JSON | 重新合并展开 |
| 6 | TTK CSV（任务 3-4） | 重新生成 |

### 步骤六：重新验证

修正后必须重新执行完整验收链：6a → 6b（→ 6c）。

**回归检查**：对比修正前后的 PASS/FAIL 用例集合，确认修正未引入新的失败。

## 6c：全量执行（可选，默认不执行）

> **前置条件**：6b 门禁通过。

6c 为可选步骤，**默认不执行**。主 Agent 向用户确认：

| 选项 | 说明 |
|------|------|
| 执行全量 | 用 nohup 后台执行全量 low CSV |
| 跳过 | 结束 TTK 模块 |

**超时处理**：若发出确认后 1 分钟内用户未回复，默认选择「跳过」。

用户确认「执行全量」后，使用 nohup 后台执行：

```bash
cd {ops_test_kit_path} && setsid bash -c 'exec python3 -m ttk kernel \
  -i {whitebox_dir}/ttk_{op_name}_cases_low.csv \
  -o {whitebox_dir}/ttk_{op_name}_cases_low_result.csv \
  --plugin {plugin_path} \
  --pc 1 \
  --seed 42 > {whitebox_dir}/ttk_batch.log 2>&1' &
disown
```

执行后提供进度查询指令：

```bash
pgrep -af "ttk kernel" | grep -v pgrep
grep -c "Performance result" {whitebox_dir}/ttk_batch.log
tail -5 {whitebox_dir}/ttk_batch.log
```

执行完成后验收：

```bash
python3 -c "
import csv
with open('{whitebox_dir}/ttk_{op_name}_cases_low_result.csv') as f:
    rows = list(csv.DictReader(f))
passed = sum(1 for r in rows if r.get('perf_status','') == 'PASS')
print(f'{passed}/{len(rows)} = {passed/len(rows)*100:.1f}%')
"
```
