# Builder image for milvus-common unit tests (based on ubuntu-22.04)
# Usage examples:
#  docker build -f Dockerfile.builder -t milvus-common-builder:latest .
#  docker run --rm -it -v$(pwd):/workspace -w /workspace milvus-common-builder:latest  # get an interactive shell

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install toolchain + dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
       software-properties-common \
       ca-certificates \
       curl \
       gnupg2 \
       cmake \
       libopenblas-dev \
       libaio-dev \
       python3 \
       python3-pip \
       build-essential \
       libpci3 \
       redis-server \
 && add-apt-repository ppa:ubuntu-toolchain-r/test -y \
 && apt-get update \
 && apt-get install -y --no-install-recommends gcc-12 g++-12 \
 && rm -rf /var/lib/apt/lists/*

# Make gcc-12 / g++-12 available via CC/CXX env vars
ENV CC=gcc-12
ENV CXX=g++-12

# Install a specific conan
RUN pip3 install --no-cache-dir conan==1.61.0

# Add default conan remote if reachable (best-effort)
RUN conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local || true

WORKDIR /workspace

# Create entrypoint script that starts Redis and then runs bash
RUN echo '#!/bin/bash\nredis-server --daemonize yes\nexec "$@"' > /entrypoint.sh \
 && chmod +x /entrypoint.sh

# Default entrypoint starts Redis, then runs bash for interactive use
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
