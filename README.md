# Milvus Common

## Overview

Milvus Common is a core component library that serves as a bridge between Milvus and Knowhere, designed to decouple these two major components and provide common functionality. This library contains essential utilities and interfaces that are shared between Milvus and Knowhere, making the codebase more modular and maintainable.

## Components

### Caching Layer

### Common Utilities
- `Monitor`: Metrics collection and monitoring

### Log Module


## Build from source

### Prepare

```
pip install conan==2.25.1
conan profile detect
conan remote add default-conan-local2 https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local2
```

### Build
```
make
```

Note: If your Conan profile defaults to a lower C++ standard, pass `CONAN_EXTRA="-s compiler.cppstd=gnu20"` to satisfy dependencies that require C++20.

```
make CONAN_EXTRA="-s compiler.cppstd=gnu20"
```

Useful Makefile variables:

- `BUILD_TYPE`: CMake/Conan build type, default `Release`.
- `CONAN_EXTRA`: extra arguments passed to `conan install`, for example `-s compiler.cppstd=gnu20`.
- `CMAKE_EXTRA`: extra arguments passed to `cmake -S . -B build`, for example `-DWITH_COMMON_UT=ON`.
- `NPROC`: parallel job count for Conan dependency builds and CMake builds.

Example:

```
make BUILD_TYPE=Release CONAN_EXTRA="-s compiler.cppstd=gnu20" CMAKE_EXTRA="-DWITH_COMMON_UT=ON" NPROC=4
```

### Run test

```
make test
```
