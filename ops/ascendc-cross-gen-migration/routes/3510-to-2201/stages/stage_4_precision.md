# 阶段 4：精度验证

**目标**：证明 A2 实现在业务 dtype/shape 上达到 `ops-precision-standard`，并把迁移引入的差异（FTZ、dtype 转换）说清楚。

## 必读

- `[LOADED] precision-verification`
- `ops-precision-standard`（阈值与标杆构造，唯一真源）

## Step 4.1 标杆

1. 用 CPU 高精度实现（numpy/torch float64）实现语义，**含边界分支**。
2. 自检：float32 vs float64 互验，确认标杆自身误差远小于阈值。
3. 记录标杆版本、代码位置、容差来源。

> A5 输出不能当标杆（FTZ 差异）；只能作为"行为参考"记录。

## Step 4.2 用例设计

按 `precision-verification.md` §3 的矩阵，至少 30 例、单例 ≤200K 元素：

- 每个 dtype ≥2 例；每个迁移 dtype ≥3 例。
- 非对齐/尾块/多核尾核 shape 必含。
- 特殊值：0、±max、subnormal、Inf/NaN（按算子约定）。
- 重复执行 ≥3 次验证确定性。
- 迁移特有：FTZ 回退、dtype 等价、恢复项（int4/4:2/ZZ）、workspace 中转。

## Step 4.3 执行

```bash
source "${CANN_SET_ENV}"
python3 -m pytest <测试目录> -v 2>&1 | tee precision.log
```

- 被测实现必须走真实接口（aclnn / `torch.ops.npu.<op>`），禁止 PyTorch 同名算子替代。
- 每例输出 matched_ratio、max_abs_error、rtol/atol、dtype、shape。
- 汇总表落盘。

## Step 4.4 失败分诊

按 `precision-verification.md` §5：

1. 先分桶（normal / subnormal / zero / 极值）。
2. 再定位（最小 shape 二分、打印中间张量、对照 Cast 前后）。
3. NaN/Inf 必须根因定位。
4. 归类：实现错误 / FTZ 差异 / dtype 表示粒度 / 标杆问题。
5. 实现错误 → 回阶段 2；标杆问题 → 修标杆并重跑全量。

## Step 4.5 条件接受（如需要）

仅限非缺陷差异，且用户明确接受；输出 `CONDITIONAL` + `USER_ACCEPTED` + 豁免明细。禁止用调阈值/删用例达成 PASS。

## 完成判据

- 全部计划用例执行完毕（无 SKIP 未说明）。
- `N_PASS == N_TESTS`，或对失败项完成分诊并给出结论（BLOCKED 或条件接受）。
- 精度报告含用例清单、统计、分诊、标杆信息。

## Gate

```
[GATE-4] PRECISION=N/N_PASS
[GATE-4] PRECISION=N/M_PASS BLOCKED      # 有失败且未解决
```

`N_FAIL>0` 且未走条件接受时，禁止输出 PASS。

## BF16 输出的位级校验

- BF16 输出除了 `max_abs`，必须同时统计 `bad`（超出 atol+rtol）与 `1ulp_bad`（相差 1 个 BF16 ulp 的个数）；
- 把 NPU 输出按 `uint16` 位模式与 golden 的 BF16 位模式逐元素对比：`h_1ulp_bad=0` 才说明舍入模式（CAST_RINT）与累加顺序都对；
- golden 用 **FP64** 计算再转 BF16；用 FP32 golden 会把 golden 自身的舍入误差算到实现头上；
- 只报 `max_abs` 容易掩盖“整体偏移 1 ulp”这类系统性问题。

- SFAG/sinks 反向移植按 `references/sfag-sinks-port.md` 执行：复用 `sparse_flash_mla_grad` arch22 的 `CalDsinks`/`CopyOutDsinks`/post，不要自写公式。
**反向算子分项口径**：`dq`、`dk`、`dv`、`dq_rope`、`dk_rope`、`d_sinks` 必须**逐项**统计 matched_ratio；任一项失败即失败。实测 SFAG 的 `dq` matched_ratio 仅 0.009、`d_sinks` 0.25，接口 `getws/run` 却全部正常。
