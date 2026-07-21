### Example input part1
```cpp
AscendC::DataCopy(outputGm, outputLocal, 1);
```
### Example output part1
```cpp
AscendC::DataCopyPad(outputGm, outputLocal, {1, static_cast<uint16_t>(1*sizeof(float)), 0, 0});
```

### Example input part2
```cpp
AscendC::DataCopy(xTileLocal[i], inputGm[offset], 1);
```
### Example output part2
```cpp
AscendC::DataCopyPad(xTileLocal[i], inputGm[offset], {1, static_cast<uint16_t>(1*sizeof(float)), 0, 0}, {false, 0, 0, 0});
```