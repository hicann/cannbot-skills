// DESC: SYNC-14 equivalent modulo forms should map to the same buffer region.

#include "sync14_common.h"

void Kernel()
{
    uint64_t idx = 1;

    DataCopyPad(outputGm_[0], ubBiasOutTotalBuffer_[(idx & 3) * TILE_SIZE], copyParams);
    SetFlag<HardEvent::MTE3_V>(eventId[idx % 4]);
}
