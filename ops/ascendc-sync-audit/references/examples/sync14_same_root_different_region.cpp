// DESC: SYNC-14 same index root but sync protects a different buffer region.

struct CopyParams {};

struct Buffer {
    void *GetPhyAddr();
};

Buffer ubBiasOutTotalBuffer_[4096];
Buffer inputGm_[4096];
Buffer outputGm_[4096];
int eventId[4];
CopyParams copyParams;
constexpr int TILE_SIZE = 128;

void Kernel()
{
    uint64_t idx = 1;

    DataCopyPad(ubBiasOutTotalBuffer_[idx * TILE_SIZE], inputGm_[0], copyParams);
    DataCopyPad(outputGm_[0], ubBiasOutTotalBuffer_[(idx - 1) * TILE_SIZE], copyParams);
    SetFlag<HardEvent::MTE3_V>(eventId[idx]);
}
