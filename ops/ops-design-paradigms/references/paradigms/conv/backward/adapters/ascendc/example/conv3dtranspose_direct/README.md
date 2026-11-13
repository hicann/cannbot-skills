# conv3dtranspose_direct — Conv3DTranspose 直调算子参考实现（`<<<>>>` 直调，转置卷积变体）

> 教学目标：本示例是卷积反向范式**转置卷积变体（dx-as-forward）的算子代码落地参考**——与 [conv3ddx_direct](../conv3ddx_direct/README.md) 共享同一核主体（三 tiling 模板求解、三 weight 变换、双格式、完整写回协议），按范式差异点适配转置几何（output_padding/rho/显式 input_size）与 **bias BT 通路**（转置卷积专属）。只含算子实现四件，编译与测试方法见 [conv3ddx_direct README](../conv3ddx_direct/README.md)（编译配方与测试口径完全一致）；本 README 聚焦与 conv3ddx_direct 的差异要点。
>
> 实现经全工作流验收（4 节点，2026-11-02）：黑盒 20/20 PASS（torch conv_transpose3d golden，含 16 例带 FP32 bias、2 例 output_padding 非零）、白盒 35/35（D-1 修复后全量回归）、L2 负向 16/16 拒绝、tiling tag 35/35 命中（实测平台 Ascend950DT_9592 / AIC=36 / L1=512KB / L0C=256KB / BT=4096B）。

## 与 conv3ddx_direct 的差异（范式迁移要点）

| 维度 | conv3ddx_direct（输入梯度） | conv3dtranspose_direct（转置卷积） | 范式依据 |
|------|---------------------------|-----------------------------------|---------|
| 进入绑定 | input_size=dedx 目标形状，正向形状公式逐轴校验 | x→dedy、filter→weight、y→dedx 绑定；**显式 input_size 模式**（rho=Ly−(base+q)∈[0,stride) 校验、q<stride or q<δ、全零哨兵=属性推导模式） | preprocessing §2.1/§4.2 |
| 反向补边 b⁺ | Keff−1−p⁺+rho | **Keff−1−p⁺+q+rho**（q 并入后侧补边） | preprocessing §4.3 |
| initOutputFlag | DX 条件表（rhoH>padDown 等） | 转置条件表（增补 qD>0 或 qH>0；rho 按转置形状关系计算） | preprocessing §5.2 |
| bias | 无 | **可选 FP32 一维 (Cin,)**：CalcBiasFullLoadFlag 重载判定、BT 通路（FullLoadBias GM→L1→LoadBiasToBT、带 BT 首次 MMAD 初值、ComputeForBias 无贡献块写 bias、奇 32B 块尾零初始化） | preprocessing §5.3 + bias-bt-path 全文 |
| 无贡献块写回 | ComputeZeroBlock（标量 GM 写零——仅 aclnn 版改为 MMAD 变体） | **ComputeZero**：LoadL0Zero 全补零窗口 + 零初始化 MMAD（cmatrixInitVal=true）+ Fixpipe 正常写回，与主计算通道一致（预填重复执行协议下位级正确——白盒 D-1 缺陷修复后形态） | bias-bt-path §4 同构 + sync-implementation §1.2（End() FreeAllEvent） |
| launch ABI | 六入口（2 dtype × 3 loadB2） | 同构六入口 + **bias 指针参数**（空指针=无 bias） | kernel-template §2.1 |

其余（三模板求解与容量回调、loadB2 判定序、双格式、GM→L1A/L1B 装载、MTE1/MTE3 事件对、Host 校验层）与 conv3ddx_direct 逐条一致，见其 [README](../conv3ddx_direct/README.md)。

## 支持的规格范围（范式 [verification](../../../../generic/input/verification.md) §2.1）

| 维度 | 范围 | 实现位置 |
|------|------|---------|
| 场景 | 转置卷积 Conv3DTranspose（可选 bias、可选 output_padding；output_padding 满足 q<stride or q<δ） | `conv3dtranspose_tiling.cpp` |
| dtype | 数据 fp16/bf16 三路一致；**bias FP32**（16/20 例带 bias 验证） | `conv3dtranspose_launch.h` |
| format | dedy/dedx NCDHW 或 NDHWC（配对）；weight NCDHW 或 NDHWC | `conv3dtranspose_tiling.cpp` 归一层 |
| tiling 模板 / weight 变换 | 三模板 + 容量回调；loadB2=1/2/3 | 同 conv3ddx_direct |
| 异常输入 | Host 校验层拒绝（转置形状关系 rho/q 越界、bias 非一维 FP32、几何/dtype/属性/分组/格式） | `conv3dtranspose_tiling.cpp` |

## 代码结构

```text
conv3dtranspose_direct/
`-- op_kernel/
    +-- conv3dtranspose_launch.h        (146) 直调 ABI：Spec/Tiling/SolveTiling/Launch* 六入口（+bias 指针）
    +-- conv3dtranspose_tiling_data.h   (135) Host/Device 共享 TilingData POD（+bias/initOutputFlag 转置字段）
    +-- conv3dtranspose_tiling.cpp     (1242) Host 求解器：转置几何归一（q/rho/显式 input_size）+ 校验
    |                                          + 三模板求解 + loadB2 判定 + CalcBiasFullLoadFlag + 发布
    `-- conv3dtranspose_kernel.cpp      (1390) 设备 kernel（AIC-only）：三 loadB2 变换 + 双格式 + BT 通路
                                               + ComputeZero/ComputeForBias + End() FreeAllEvent
```

## 测试说明（口径同 [conv3ddx_direct README 测试说明](../conv3ddx_direct/README.md)，差异点如下）

1. **golden**：CPU torch `F.conv_transpose3d`（dedy 为输入、含 bias/output_padding 语义），固定种子 U(-0.5,0.5) 立即量化落盘
2. **tag 核对**：本实现 35/35 命中（黑盒 20 + 白盒 15，模板分布 small_shape×18 / inner_product×15 / full_load×2）
3. **L2 负向**：16/16 按预期拒绝（含 rho/q 越界、bias 非法维度）
4. **优先复跑点**：① 输出预填非零的重复执行协议（V1——D-1 缺陷在此协议下暴露，ComputeZero 修复后位级通过）；② 无贡献块写回（ComputeZero/ComputeForBias）；③ bias 奇 32B 块（validBytes 非 64B 对齐）；④ q+rho≥stride 边界（golden 表达域受限，torch 固有限制）

## 已知边界（登记，不构成本示例缺陷）

1. **性能未优化**（同 conv3ddx_direct 首版性质）：无预取流水/装载整槽 Fill 开销；优化属后续轮次
2. **检视登记项（未纳入，非缺陷）**：bias 指针与 hasBiasFlag 契约防御（无用例触发，不引入无实证分支）；LoadL0Zero 名义源窗 1024B 对极小槽的 L1 尾部过读（低/信息级，wb01/02/04 位级通过为实证）
3. **范式外分支排除**：kernelSplitMode/C04/vec 等同 conv3ddx_direct；golden 在 q+rho≥stride∧dilation 的表达域受限（torch 固有限制，白盒登记）
