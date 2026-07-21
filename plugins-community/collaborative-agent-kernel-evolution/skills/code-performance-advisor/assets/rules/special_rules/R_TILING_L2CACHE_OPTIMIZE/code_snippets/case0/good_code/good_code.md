# Good Code

```cpp
// Z型遍历：换行反向循环
bool reverse = false;
for (uint64_t mTileIndex = 0; mTileIndex < block_.params_.mTileCntL2; mTileIndex++) {
    reverse = !reverse; // 换行时反转方向
    
    for (uint64_t nTileIndexTemp = 0; nTileIndexTemp < block_.params_.nTileCntL2; nTileIndexTemp++) {
        // Z型遍历：换行后反向，实现单边数据复用
        uint64_t nTileIndex = reverse 
            ? (block_.params_.nTileCntL2 - nTileIndexTemp - 1) 
            : nTileIndexTemp;
        
        block_.UpdateBlockCnt(mTileIndex, nTileIndex);
        block_.InitBlockIndex(index);
        
        for (uint64_t j = 0; j < block_.params_.realRound; j++) {
            mm_.SetSingleShape(block_.params_.singleCoreM, 
                             block_.params_.singleCoreN,
                             matmulTiling.singleCoreK);
            mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], isTransposeA);
            mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], isTransposeB);
            mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
            mm_.Iterate();
            mm_.GetTensorC(cGlobal_[block_.offset_.offsetC], enAtomic);
        }
    }
}
```