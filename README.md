# Milvus Common

## Overview

Milvus Common is a core component library that serves as a bridge between Milvus and Knowhere, designed to decouple these two major components and provide common functionality. This library contains essential utilities and interfaces that are shared between Milvus and Knowhere, making the codebase more modular and maintainable.

## Components

### Caching Layer

### Common Utilities
- `Monitor`: Metrics collection and monitoring

### Log Module


## Build from source

```
pip install conan==1.64.0
mkdir build && cd build
conan install .. --install-folder conan --build=missing  -s compiler.libcxx=libstdc++11
# ut is off by default
cmake .. -DENABLE_UNIT_TESTS=ON
make

# run ut
export LD_LIBRARY_PATH=$PWD/lib/:$LD_LIBRARY_PATH
./bin/cachinglayer_test 
```
