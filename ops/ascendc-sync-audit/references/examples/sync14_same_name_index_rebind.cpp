// DESC: SYNC-14 same index variable name is rebound to a different loop root.

#include "sync14_common.h"

void Kernel()
{
    uint64_t ubMte2LoopIdx_ = 1;
    uint64_t ubComputeLoopIdx_ = 1;

    uint64_t idx = ubMte2LoopIdx_;
    DataCopyPad(outputGm_[0], ubBiasOutTotalBuffer_[idx * TILE_SIZE], copyParams);

    idx = ubComputeLoopIdx_;
    SetFlag<HardEvent::MTE3_V>(eventId[idx & 3]);
}
