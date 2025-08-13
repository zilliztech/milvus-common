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
conan install .. --build=missing -o with_ut=True -s compiler.libcxx=libstdc++11 -s compiler.version=12 -s build_type=Release
conan build ..

# run ut
./test/test_cachinglayer/cachinglayer_test
```
