---
name: cannbotdsl-op-test
description: "为 CANNBotDSL kernel 设计和编写系统化测试时使用。测试需覆盖编译期、功能期（NPU 精度）和性能期（msprof），且要处理 pytest markers 门控。当需要设计分层测试（L0 编译测试 AOT compile + 产物断言 / L1 功能测试 NPU 精度 vs CPU golden / L2 边界测试 tail block+Dim+极端 dtype / L3 性能测试 msprof）、写独立于 NPU 实现的 CPU golden（torch 参考）、用 @pytest.mark.parametrize 参数化 shape、写 TensorSpec/Dim staged AOT 测试，或按 markers（cannir_install/ascendc_toolchain/npu）门控时触发。Triggers: cannbotdsl 测试, CPU golden, torch.allclose, pytest markers, parametrize, L0 L1 L2 测试, AOT 测试。Tester sub-agent 在 Stage 4 调用。"
---

# cannbotdsl-op-test

CANNBotDSL 算子测试设计与执行。Tester sub-agent 在 Stage 4 调用。核心原则：**按环境能力分层**，无 NPU 时也能跑 L0（编译/codegen）测试，把"不能测精度"限制在 L1+。

**真实来源以源码为准**：测试模式取自真实 test，均已核对行号。

## 触发条件

- 需要为 CANNBotDSL kernel 设计测试用例
- 需要编写独立于 NPU 实现的 CPU golden
- Tester sub-agent 在 Stage 4 调用

## 测试分层（对应 `cannbotdsl-env-setup` 的能力分级）

| 层 | 验证什么 | 需要环境 | marker | 无 NPU 可跑? |
|----|----------|----------|--------|:---:|
| **L0 编译/codegen** | 编译链走通、产出 `.so` | bisheng | `ascendc_toolchain` | ✅ |
| **L1 功能** | NPU 精度 vs CPU golden | NPU | `npu` | ❌ |
| **L2 边界** | tail block、动态 shape（Dim）、极端 dtype | NPU（AOT 编译部分可无 NPU） | `npu` / — | 部分 |
| **L3 性能** | msprof 基准 + pipe utilization | NPU | `npu` | ❌ |

## L0：编译/codegen 测试（无 NPU 主力）

真实模式（AOT 编译 + 产物断言）：

```python
@pytest.mark.ascendc_toolchain
def test_xxx_codegen():
    x_spec = cannbotdsl.TensorSpec((128, 32), dtypes.float32)
    y_spec = cannbotdsl.TensorSpec((128, 32), dtypes.float32)
    fn = driver.compile(x_spec, y_spec)          # 只到编译，不上真机
    assert fn.so_path and os.path.exists(fn.so_path)
    fn.close()
```

> L0 断言检查什么：kernel 能走完整条编译链并产出 `.so`——即 trace、IR 构建、lowering、AscendC 生成、bisheng 编译全部通过。这不需要 NPU，是无 NPU 环境下验证 kernel 结构正确性的主要手段。编译期报错（类型不匹配、Buffer 超限、sync 不配对）都会在这一层暴露；错误分类与定位见 `../../debug-skills/cannbotdsl-runtime-debug/SKILL.md`。

## L1：功能测试（NPU 精度 vs CPU golden）

真实模式（channel-first CV-mix 的 L1 NPU 精度测试范式；完整可跑文件见文末 References）：

```python
@pytest.mark.npu
def test_xxx_npu():
    pytest.importorskip("torch_npu")
    import torch
    torch.manual_seed(0)                         # 固定 seed
    a = torch.randn(M, K, dtype=torch.float16)   # CPU 创建再 .npu()
    out_npu = torch.zeros(M, N, dtype=torch.float32).npu()

    op.run(from_torch_npu(out_npu), from_torch_npu(a.npu()), ...)
    torch.npu.synchronize()                      # 必须同步

    ref = a.float() @ b.float()                  # CPU golden：fp32 参考
    assert torch.allclose(out_npu.cpu(), ref, atol=1e-4, rtol=1e-4)
```

CPU golden 要点：

- **dtype 契约对齐**：device 是 fp16 输入 fp32 计算，golden 就用 `.float()` 算，比较时对齐 device 输出 dtype。exp/reduce 类精度提升点在 golden 里同样用 fp32。
- **NPU 算子缺失**：优先 CPU 创建 tensor 后 `.npu()`，避免 torch_npu 打包算子缺失报错（见 `cannbotdsl-op-develop` 常见风险）。
- **精度容差**：matmul/线性 `atol=rtol=1e-3~1e-4`；含 exp/softmax 的 sum 放宽到 `1e-3`。

### L1.1 相对误差指标（MERE/MARE）的陷阱：低 dtype 输出下阈值可能不可达

生态算子精度标准用**相对**误差：`MERE = mean(|a-g|/(|g|+1e-7)) < T`、`MARE = max(...) < 10T`，T = 2⁻¹⁰(fp16) / 2⁻⁷(bf16)。

**问题**：分母只加 `1e-7`，而 fp16 在 0 附近的量化步长可达 `1e-4` 量级。golden 元素本身是 `1e-5` 量级时，**1 个 ULP 的绝对误差被放大成 O(1) 的相对误差**。少量近零点就能把 MARE 顶到 10²、把 MERE 拉过阈值。

**实测**：某 attention 算子，**把精确 fp32 结果直接舍入到 fp16**（任何 fp16 输出算子的理论最优）再按上式比，MERE = 1.89e-3 > T = 9.77e-4 —— **阈值对任何 fp16 输出实现都不可达**。只看原始指标会把已达物理下限的正确实现判为失败，并浪费时间"优化"根本不是算子引入的误差。

**做法：永远同时报告「量化下限」并逐项对比**。下限 = 精确 fp32 结果舍入到输出 dtype，再与 golden 走**完全相同**的指标计算：

```python
ideal   = ref_fp32.to(out_dtype)          # 任何实现都无法超越的输出
m_impl  = metrics(actual, ref_golden)
m_floor = metrics(ideal,  ref_golden)     # 量化下限
big     = ref_golden.abs().flatten() >= 1e-3   # 剔除近零点（相对误差在那里无意义）
at_limit = (m_big["MERE"] <= 1.5 * m_big_floor["MERE"] and ...)   # 逐项比 ideal，不比阈值
```

**两个走过的弯路**：① "MERE ≤ 下限"作二值判据**太脆**——两侧同阶噪声，大小关系随 seed 翻转；② "剔除近零点后 < 阈值"**也不对**——受限 MARE 的**下限本身**就可能超 10T，会把完美实现判失败。**只能逐项与同等处理下的 ideal 比较**（容 1.5× 噪声）。

**报告口径**：同时给出原始 `THRESHOLD_CHECK`（benchmark 官方口径，如实报）与 `AT_QUANTISATION_LIMIT`（是否已达 dtype 物理下限）。**不要单方面把"达下限"改称"通过"**——官方 harness 比对方式可能不同，结论以其为准。

## L2：边界 + AOT 动态 shape

### L2.0 先问一句：benchmark 全过，到底覆盖了什么？

**一份官方用例集全绿，不等于算子对它自己声称支持的输入都正确。** benchmark 的 shape 按典型场景选，不按实现的分支选 —— 它天然覆盖不到"host 规划器接受、但没人测过"的几何。

**做法：把规划器接受的域枚举出来，减去用例集覆盖的域，差集就是要补的 L2。** 用脚本枚举，别目测：

```python
covered = {plan(c.shape) for c in cases}          # 用例集实际覆盖的几何
for shape in enumerate_accepted_shapes():         # 规划器接受的全部组合
    if plan(shape) not in covered:
        print("untested geometry:", shape)        # ← 这些才是 L2 该测的
```

同时对每条**被拒绝**的 shape 确认它是**干净拒绝**（host 抛异常、不下发设备），而不是算出错数。

> **实测**（GQA）：20 个官方用例的 `G ∈ {4,12,16}` 全过，而 Mode D 在 **`G > 32` 时静默算错** —— 缓冲高度 `VH` 被硬编码成 16、行循环却按 `BMV = G/2` 迭代，越界读写；`NPAD = VH-BMV` 变负还让 `const_expr(NPAD > 0)` 守卫自己失效（见 `../cannbotdsl-vf-fusion/SKILL.md` 陷阱 11）。`G=64` 经官方比较器是 MERE **1.048** / MARE **98.8** —— 无 crash、无 fault、无告警。而 `desc.md` 声明 `N_q ≤ 256`、`N_kv ≥ 1`，一次普通 MQA decode 就落在这个洞里。**用例集永远发现不了它，只有"按实现枚举"能。**

**其余 L2 常规项**：

- **tail block**：非整除 shape（如 M=130，tile=64 → 尾块 2 行），验 `tile_view` 尾块处理。
- **AOT 动态 shape**（`test/cannbotdsl/test_aot_p1a.py:156-164 cite-skip (cannbot-dsl 源码仓测试)`，**编译部分无需 NPU**）：

```python
M = cannbotdsl.Dim("M", multiple_of=16)
x_spec = cannbotdsl.TensorSpec((M, 32), dtypes.float32)
y_spec = cannbotdsl.TensorSpec((M, 32), dtypes.float32)
fn = identity_copy.compile(x_spec, y_spec)                 # 产 .so
assert fn.so_path and os.path.exists(fn.so_path)
fn.close()
```

  `TensorSpec` 校验：拒未知 dtype、拒 stride/rank 不匹配。`Dim` 覆盖 min/max/
  multiple_of、同名约束冲突和“至少一个裸槽位直接绑定”。缓存测试除命中跳过
  bisheng 外，还要验证相同动态 IR 下不同约束、共享关系或派生表达式产生不同 key。

- **极端 dtype 组合**：fp8（E4M3/E5M2）、bf16 输出等，配 §3 dtype 表。

## L3：性能测试

msprof 采集 + op_summary CSV 解析，Task Duration 取 **min** 不取 mean。详见 `../../debug-skills/cannbotdsl-msprof-compare/` 和 `../cannbotdsl-perf-optimize/`。

## 参数化模板

```python
@pytest.mark.parametrize("shape", [(64, 128), (128, 256), (130, 96)])  # 含尾块
@pytest.mark.npu
def test_xxx(shape): ...
```

## 门禁

- 无 NPU 时至少交付 L0 codegen 测试（AOT 编译 + 产物断言），并明确标注 L1+ 为 blocked（缺 NPU），不能跳过测试。
- 每个 NPU 测试必须有对应 CPU golden，且说明 dtype 契约与容差来源。
- 测试用公共接口维度做参数，不依赖个人绝对路径或临时中间产物。
- 固定 seed；CPU 创建 tensor 再 `.npu()`。

## 参考

- `../cannbotdsl-code-review/SKILL.md`（Stage 4 配套审查）、`../cannbotdsl-perf-optimize/SKILL.md`（L3）
