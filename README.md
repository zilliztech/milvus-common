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

Useful Makefile variables:

- `BUILD_TYPE`: CMake/Conan build type, default `Release`.
- `CONAN_CPPSTD`: C++ standard passed to Conan, default `20`.
- `CONAN_EXTRA`: extra arguments passed to `conan install`.
- `CMAKE_EXTRA`: extra arguments passed to `cmake -S . -B build`, for example `-DCMAKE_VERBOSE_MAKEFILE=ON`.
- `NPROC`: parallel job count for Conan dependency builds and CMake builds.

Example:

```
make BUILD_TYPE=Release CONAN_CPPSTD=gnu20 NPROC=4
```

### Run test

Use `make test` to enable the Conan unit-test option that installs GTest.

```
make test
```
