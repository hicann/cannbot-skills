// sync05_conditional_pipe_wait.cpp — SYNC-05 最小坏例：
// AIV 侧 CrossCoreWaitFlag<..., PIPE_MTE3> 后调用 epilogueOp()，
// 而 epilogue 内部首个操作是 if constexpr 分支内的 Relu(PIPE_V)。
// wait 只拦 MTE3 流水，enableRelu 成立时 V 流水的 Relu 可提前读到
// fixpipe 尚未写完的 UB 数据（真实来源：kernel_matmul_mix_fixpipe_opti.h:242）。
// 期望：sync_audit.py 报 SYNC-05（含条件双 PIPE wait 修法建议），
// 解析链 = 变量类型 BlockEpilogue → 模板形参 BlockEpilogue_ → 前缀匹配
// BlockEpilogueFixpipeMini。
#include "kernel_operator.h"

constexpr uint16_t AIC_SYNC_AIV_MODE_4 = 4;
constexpr int16_t AIC_SYNC_AIV_FLAG = 6;
constexpr int16_t AIV_SYNC_AIC_FLAG = 4;

template <typename DataTypeOut, bool EnableRelu>
class BlockEpilogueFixpipeMini {
public:
    AscendC::LocalTensor<DataTypeOut> ubLocalTmp_;
    AscendC::GlobalTensor<DataTypeOut> outputGlobal_;

    __aicore__ inline void Run(int64_t blockShapeM, int64_t blockShapeN, int64_t offset)
    {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(blockShapeM),
            static_cast<uint32_t>(blockShapeN * sizeof(DataTypeOut)), 0, 0, 0};
        if constexpr (EnableRelu) {
            AscendC::Relu(ubLocalTmp_, ubLocalTmp_, blockShapeM * blockShapeN);
        }
        AscendC::DataCopyPad(outputGlobal_[offset], ubLocalTmp_, copyParams);
    }

    __aicore__ inline void operator()(int64_t m, int64_t n, int64_t offset)
    {
        Run(m, n, offset);
    }
};

template <class BlockEpilogue_>
class KernelSync05Mini {
public:
    using BlockEpilogue = BlockEpilogue_;

    __aicore__ inline void Run(int64_t m, int64_t n, int64_t offsetC, uint16_t pingPongIdx)
    {
        BlockEpilogue epilogueOp;
        if ASCEND_IS_AIV {
            // ❌ SYNC-05：epilogue 首个操作是 enableRelu 分支内的 Relu(PIPE_V)，
            // wait 却固定用 PIPE_MTE3；应按同条件 if constexpr 选择 PIPE_V/PIPE_MTE3
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG + pingPongIdx);
            epilogueOp(m, n, offsetC);
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + pingPongIdx);
        }
    }
};
