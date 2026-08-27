// cube_kernel.cpp — AIC 侧，配 vec_kernel.cpp 验证 CrossCore 跨文件配对
#include "kernel_operator.h"
using namespace AscendC;

class KernelCube {
public:
    __aicore__ inline void ComputeMm1(int32_t syncC1V1) {
        // ... MMAD ...
        CrossCoreSetFlag<HardEvent::MTE2_V>(syncC1V1);  // AIC→AIV 通知就绪
    }
    __aicore__ inline void WaitVecDone(int32_t syncV1C2) {
        CrossCoreWaitFlag<HardEvent::MTE2_V>(syncV1C2); // AIC 等 AIV
    }
};
