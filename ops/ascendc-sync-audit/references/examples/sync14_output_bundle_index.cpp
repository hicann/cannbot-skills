// DESC: SYNC-14 producer output fields use different buffer-id roots.

struct UB_BUFFER_INFO {
    static constexpr int ubWeightOutputHighBitBufferNum = 4;
    static constexpr int biasReducedSingleBufferSize = 128;
};

struct VecConfig {
    int ubMte2BufferNum;
} vecConfig;

struct Buffer {
    void *GetPhyAddr();
};

struct MxA8W4NzParams {
    void *weightHighBitPhyAddr;
    void *biasOutUbAddr;
};

Buffer ubHighBitTotalBuffer_[4096];
Buffer ubBiasOutTotalBuffer_[4096];
int vecEventIdMte3ToV[4];
constexpr int VECTOR_REG_WIDTH = 256;

void Kernel()
{
    uint64_t ubMte2LoopIdx_ = 1;
    uint64_t ubComputeLoopIdx_ = 1;
    uint64_t ubMte2BufferIdx = (ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1);
    MxA8W4NzParams mxA8W4NzParams;

    mxA8W4NzParams.weightHighBitPhyAddr =
        ubHighBitTotalBuffer_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO::ubWeightOutputHighBitBufferNum - 1)) *
                              VECTOR_REG_WIDTH]
            .GetPhyAddr();
    mxA8W4NzParams.biasOutUbAddr =
        ubBiasOutTotalBuffer_[ubMte2BufferIdx * UB_BUFFER_INFO::biasReducedSingleBufferSize].GetPhyAddr();
    SetFlag<HardEvent::MTE3_V>(
        vecEventIdMte3ToV[ubComputeLoopIdx_ & (UB_BUFFER_INFO::ubWeightOutputHighBitBufferNum - 1)]);
}
