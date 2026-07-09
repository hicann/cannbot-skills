# Syntax Rules

Use this file when a CUDA source file contains syntax or language constructs that need to be rewritten for Ascend C SIMT.

## `assert`

Required header:

```cpp
#include "utils/debug/asc_assert.h"
```

## `printf`

Required header:

```cpp
#include "utils/debug/asc_printf.h"
```

## Device function qualifier

Replace CUDA device-function qualifiers like:

```cpp
__device__ int *getPointer()
```

with:

```cpp
__aicore__ int *getPointer()
```

## Shared memory

### Fixed-size shared memory

Replace CUDA shared-memory declarations like:

```cpp
__shared__ float staticBuf[1024];
```

with:

```cpp
__ubuf__ float staticBuf[1024];
```

### Dynamic shared memory

Replace CUDA declarations like:

```cpp
extern __shared__ float dynamicBuf[];
```

with:

```cpp
extern __ubuf__ float dynamicBuf[];
```
