// vec_kernel.cpp — AIV 侧，配 cube_kernel.cpp 验证 CrossCore 跨文件配对
// 故意让 syncC1V1 配对、syncV1C2 不配对（Vec 侧缺 SetFlag）
#include "kernel_operator.h"
using namespace AscendC;

class KernelVec {
public:
    __aicore__ inline void ComputeVec1(int32_t syncC1V1) {
        CrossCoreWaitFlag<HardEvent::MTE2_V>(syncC1V1); // 等 AIC 就绪（有配对）
        // ... Vector 计算 ...
        // syncV1C2 应在此 SetFlag，但故意漏写 → SYNC-04 不配对
    }
};
