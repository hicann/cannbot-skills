# Add 算子

## 编译

```bash
mkdir -p build && cd build
cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..
make -j
```

## 测试

```bash
pytest test_add.py -v
```
