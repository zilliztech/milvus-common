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

## Build using the provided Docker builder image (alternative)

If you don't want to install the toolchain and conan locally, you can use the included
`Dockerfile.builder` image which mirrors the CI environment (Ubuntu 22.04, gcc-12,
conan 1.61). 

```bash
# Build the builder image (run from repository root)
docker build -f Dockerfile.builder -t milvus-common-builder:latest .

# Start an interactive shell with the repository mounted at /workspace
docker run --rm -it   -v "$(pwd)":/workspace   -v "${HOME}/.conan":/root/.conan --add-host=host.docker.internal:host-gateway   -w /workspace   milvus-common-builder:latest

# Inside the container run the same build commands as CI:
mkdir -p build && cd build
conan install .. --build=missing -o with_ut=True -o with_asan=True -s compiler.libcxx=libstdc++11 -s compiler.version=12 -s build_type=Release
conan build ..

# Run tests inside the container (example)
./test/test_cachinglayer/cachinglayer_test
```


