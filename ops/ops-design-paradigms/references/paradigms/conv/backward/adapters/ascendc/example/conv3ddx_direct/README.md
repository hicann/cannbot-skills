# conv3ddx_direct — Conv3DBackpropInput 直调算子参考实现（`<<<>>>` 直调）

> 教学目标：本示例是卷积反向范式的**算子代码落地参考**——只含算子实现四件（Host 求解器 + 设备 kernel + 共享数据 + 直调 ABI），编译与测试方法在本文档说明。实现经 20 例（fp16/bf16×10，覆盖三 tiling 模板与三 weight 变换）功能验证全绿与 tiling tag 逐例核对（实测平台 Ascend950DT_9592 / AIC=36 / L1=512KB / L0A/B=64KB / L0C=256KB）。
>
> 注册/aclnn 交付形态另见同目录 [conv3ddx_aclnn](../conv3ddx_aclnn/README.md)——共享本示例的求解器与 kernel 主体，仅适配为 GEIR 注册接入（OpDef/值依赖推导/TilingFunc/registry 入口/算子包）与 aclnn 两段式接口，对应 forward 范式 `conv2d_cube_aclnn` 的放置；仅参考工程组织与 launch 形态时另见 forward 范式 `conv2d_cube_direct`；本示例的反向语义（前放大/rot180/反向边界/三模板求解/三 weight 变换）一律以 backward 范式文档为准。

## 支持的规格范围（范式 [verification](../../../../generic/input/verification.md) §2.1）

| 维度 | 范围 | 实现位置 |
|------|------|---------|
| 场景 | 输入梯度 Conv3DBackpropInput（无 bias、无 output_padding） | `conv3ddx_kernel.cpp` |
| dtype | fp16 / bf16（三路一致，6 个编译实例 = 2 dtype × 3 loadB2） | `conv3ddx_launch.h` Launch* 表 |
| format | dedy/dedx NCDHW 或 NDHWC（配对）；weight NCDHW 或 NDHWC（独立） | `conv3ddx_tiling.cpp` 归一层 |
| 空间维 | 1D/2D/3D（升维后统一 3D） | `conv3ddx_tiling.cpp` 规格归一 |
| tiling 模板 | small_shape / inner_product / full_load 三模板 + 容量回调 | `conv3ddx_tiling.cpp` |
| weight 变换 | loadB2=1 转置加逆序 / 2 仅逆序 / 3 仅转置 | `conv3ddx_kernel.cpp` LoadToB1/B2 |
| 初始化协议 | 主计算完整写回变体（无独立清零） | `conv3ddx_kernel.cpp` |
| 异常输入 | Host 校验层拒绝（几何/dtype/属性/分组/格式五类） | `conv3ddx_tiling.cpp` |

## 代码结构

```text
conv3ddx_direct/
`-- op_kernel/
    +-- conv3ddx_launch.h         (126) 直调 ABI 契约：Conv3DdxSpec/TilingData/Conv3DdxSolveTiling/
    |                             Launch* 六入口（fp16/bf16 × loadB2 1/2/3）
    +-- conv3ddx_tiling_data.h    (121) Host/Device 共享 TilingData POD（payload）
    +-- conv3ddx_tiling.cpp       (1133) Host 求解器：校验/归一 + 三模板求解 + loadB2 判定 + 发布
    `-- conv3ddx_kernel.cpp       (988) 设备 kernel（AIC-only）+ 指令参数链 + Launch 封装
```

## 编译说明

**两条编译路线 + 一条链接配方**（源文件职责不同，编译器选项不同）：

| 源 | 编译方式 | 关键项 |
|----|---------|--------|
| `conv3ddx_kernel.cpp`（设备源） | bisheng **NPU 编译路径** | 逐文件 `-xasc --npu-arch=dav-3510`；头文件树用 `compiler/ascendc`（勿用运行时 `asc/include` 树） |
| `conv3ddx_tiling.cpp`（Host 求解器） | 普通 C++17 | `-D_GLIBCXX_USE_CXX11_ABI=0`（fe::PlatformInfoManager/libplatform 为旧 C++ ABI，直调下经其查询核数/容量） |
| 链接 | bisheng 或 g++ 驱动 | `-lascendcl -lruntime -lplatform -lregister -ltiling_api -lstdc++ -lm -ldl` |

命令行形态：

```bash
source <CANN>/bin/setenv.bash
BISHENG=<CANN>/x86_64-linux/ccec_compiler/bin/bisheng
INC="-I<CANN>/compiler/ascendc/include -I<CANN>/include -Iop_kernel"

# 设备源（AIC-only kernel + Launch 封装；AIC 指令段在 if ASCEND_IS_AIC 守卫内）
$BISHENG -std=c++17 --npu-arch=dav-3510 -xasc $INC -fPIC -c op_kernel/conv3ddx_kernel.cpp -o conv3ddx_kernel.o
# Host 求解器（Conv3DdxSolveTiling 强符号；平台资源运行时查询）
g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 $INC -fPIC -c op_kernel/conv3ddx_tiling.cpp -o conv3ddx_tiling.o
# 与宿主测试程序链接（验证过的配置以 bisheng 作项目 CXX 驱动最终链接）
$BISHENG <宿主main>.cpp conv3ddx_kernel.o conv3ddx_tiling.o \
    -L<CANN>/lib64 -lascendcl -lruntime -lplatform -lregister -ltiling_api -lstdc++ -lm -ldl -o test_conv3ddx
```

CMake 形态要点（生成工程实测配置）：`cmake -DCMAKE_CXX_COMPILER=<BISHENG>` 作项目 CXX；设备源 `set_source_files_properties(... COMPILE_OPTIONS "--npu-arch=dav-3510;-xasc")`；Host 源加 `_GLIBCXX_USE_CXX11_ABI=0` 定义；算子目标用 **OBJECT 库**并入宿主可执行（强符号覆盖宿主侧弱符号接入位——静态库按需取成员会在弱符号已定义时静默丢弃算子实现，勿用）。

## 测试说明（方法论在范式 [verification](../../../../generic/input/verification.md)，此处为执行口径）

1. **用例与数据**：外部 case CSV → 数据生成（固定种子、U(-0.5,0.5) 立即量化落盘、格式按坐标 permutation 禁 reshape）——verification §3
2. **golden**：CPU torch `F.conv_transpose3d` 基准（CANN filter 直接作 weight、非对称 pad 对称+切片、fp16 溢出截断后 cast、fp32 落盘）——verification §4
3. **判定**：逐元素混合容差 + 两步通过判定 + 非有限值硬门禁（意外 NaN/Inf 一律 FAIL 并记录坐标）——verification §5
4. **tag 核对**：Host 求解打印 templateId/loadB2 与用例 expected 比对（本实现 20/20 命中：small_shape×8 / inner_product×8 / full_load×4）
5. **L2 负向**：非法规格（几何/dtype/strides=0/负 pad/groups 不整除/batch 不一致/input_size 畸形/不支持格式）必须被校验层拒绝且错误信息明确——本实现 10/10 按预期拒绝
6. **优先复跑点**：负右边界裁剪（`woEnd` 口径曾出缺陷并按 compute-flow §5.2.1 修复）、无贡献位置写回、尾块双口径——改动裁剪/边界/尾块逻辑后先跑这些

## Tiling 求解决策流程（范式 [tiling](../../../../generic/input/tiling/tiling.md) §2）

```text
Input spec: normalize rank; validate forward shape formula
    |
    |
    v
    small_shape?      -- yes --> Balanced search -----------------------------------+
    |                                                                               |
    | no                                                                            |
    v                                                                               |
    full_load?        -- yes --> Full K residency ----------------------------------+
    |                                                                               |
    | no                                                                            |
    v                                                                               |
    standard          ---------> AdjustBaseMNK -> SetSingleCoreInfo -> CalStepK ----+
                                                                                    |
                                                                                    |
                                                                                    |
    +-------------------------------------------------------------------------------+
    |
    |
    v
loadB2 selection on final baseN
    |
    |
    v
LegalProtection (same template) -> payload

small_shape: kValue <= 32768 && estTasks < coreNum*5/2 && estTasks%coreNum != 0
full_load:  Dk == 1 && mValue > nValue && Wi <= 512 && benefit && capacity
standard:   initial (512,128,32)

loadB2 (in order):
  kernelDHW == 1                  -> 3
  NDHWC && baseN >= hk*wk          -> 2
  NCDHW && vol*cin*2B <= 64        -> 2
  otherwise                       -> 1

LegalProtection: reduce K -> reduce M/N -> restore larger K -> change buffer counts
```

## 代码段 ↔ 范式章节映射

| 代码段 | 范式章节 |
|--------|---------|
| `conv3ddx_tiling.cpp` 校验层 | preprocessing §4.1/§4.2（正向形状公式、逐轴校验） |
| `conv3ddx_tiling.cpp` 三模板求解 | generic/input/tiling/tiling.md §2/§4/§5/§6/§7/§8 + generic/input/tiling/templates/ 三篇 |
| `conv3ddx_tiling.cpp` loadB2 判定 | tilingkey-templates §2（判定顺序与 64B 等价除法一致） |
| `conv3ddx_tiling.cpp` 平台资源查询 | buffer-and-types §1（直调路径 platformInfo 获取） |
| `conv3ddx_kernel.cpp` GM→L1A | compute-flow §5.2（行带交集/负边界裁剪/插零） |
| `conv3ddx_kernel.cpp` GM→L1B | compute-flow §5.3（三父布局）+ instruction-mapping §3 |
| `conv3ddx_kernel.cpp` Load3D/Load2D | instruction-mapping §4/§5 |
| `conv3ddx_kernel.cpp` Mmad/Fixpipe | instruction-mapping §6/§7（末 M 块对齐、NZ2DN/NZ2ND） |
| `conv3ddx_kernel.cpp` 任务循环 | compute-flow §6/§7（解码轴序、K 父段、尾块双口径） |
| `conv3ddx_launch.h` 六入口 | kernel-template §2.1（五参数实例 ×2 dtype） |

## 实测记录（本示例的验证证据基线）

- 功能：20/20 黑盒 PASS（混合容差 + NaN 硬门禁 + 无残留旧值复跑协议）；L2 异常 10/10 按预期拒绝
- tiling tag：20/20 命中（small_shape×8 / inner_product×8 / full_load×4；loadB2 1/2/3 全覆盖，key=2^24+lb2）
- 白盒：11 例分支触发验证（含负右边界缺陷暴露-修复回归、full_load 驻留失败回退 standard 分支的插桩实证）

## 已知边界（登记，不构成本示例缺陷）

1. **性能未优化**：无 MTE2→MTE1 预取流水、装载整槽 Fill 清零开销（长轴/驻留场景差距大）——优化属后续轮次
2. **范式外分支排除**：kernelSplitMode≠0（如 1D 超长轴 HW 拆分）、C04、vec 等扩展执行配置不在范围（tilingkey-templates §1）
3. 负右边界 `woEnd` 曾出实现缺陷并已按 compute-flow §5.2.1 修复——该分支是高风险点，移植改动裁剪逻辑时优先复跑负边界用例
