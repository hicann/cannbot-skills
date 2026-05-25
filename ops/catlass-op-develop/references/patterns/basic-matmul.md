# 纯 Matmul — 完整代码骨架

> **导航**：[architecture/01-kernel-assembly.md](../architecture/01-kernel-assembly.md) 的场景 A 展开
> 参考 example：`catlass/examples/00_basic_matmul/`

```cpp
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/layout/layout.hpp"

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace Catlass {

template <class InputType, class OutputType>
CATLASS_DEVICE void CatlassBasicMatmulTemplate(GemmCoord problemShape,
                                               GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC,
                                               GM_ADDR userWs)
{
    using ArchTag        = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
    using L1TileShape    = GemmShape<128, 256, 256>;
    using L0TileShape    = GemmShape<128, 256, 64>;

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;

    using AType = Gemm::GemmType<InputType,  LayoutA>;
    using BType = Gemm::GemmType<InputType,  LayoutB>;
    using CType = Gemm::GemmType<OutputType, LayoutC>;

    using BlockMmad      = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape,
                                                  AType, BType, CType>;
    using BlockEpilogue  = void;
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    using MatmulKernel   = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

    LayoutA layoutA{problemShape.m(), problemShape.k()};
    LayoutB layoutB{problemShape.k(), problemShape.n()};
    LayoutC layoutC{problemShape.m(), problemShape.n()};
    typename MatmulKernel::Params params{problemShape,
                                         gmA, layoutA,
                                         gmB, layoutB,
                                         gmC, layoutC,
                                         userWs};
    MatmulKernel{}(params);
}

} // namespace Catlass
```

## 调用方使用方式

调用方（工程模板）在 op_kernel 入口对应分支内取 user workspace、构造 problemShape，转发到模板函数：

```cpp
GM_ADDR userWs = const_cast<GM_ADDR>(AscendC::GetUserWorkspace(workspace));
Catlass::GemmCoord problemShape{m, n, k};
Catlass::CatlassBasicMatmulTemplate<half, float>(problemShape, gmA, gmB, gmC, userWs);
```

## 从骨架到实例化的注意事项

1. **CType = float（推荐）**：MMAD 累加用 fp32，保证精度
2. **BlockEpilogue = void**：纯 matmul 无后处理，必须为 void
3. **Kernel = BasicMatmul**：不能用 MatmulActivation
4. **Layout 对象需要在函数内构造**：`RowMajor{m, k}` 这样的写法是合法的（int 参数构造函数）
5. **本文件中的函数封装（template void 函数）是可选的**：也可以直接在 op_kernel 入口分支内展开 using 链 + Device 调用。选择哪种方式由工程模板决定
