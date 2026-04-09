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
pip install conan==2.25.1
conan profile detect
mkdir build && cd build
conan install .. --build=missing -o "&:with_ut=True" -s compiler.cppstd=17 -s build_type=Release -of .
conan build .. -of .

# run ut
./test/test_cachinglayer/cachinglayer_test
```
