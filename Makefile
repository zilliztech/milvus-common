BUILD_DIR     := build
BUILD_TYPE    := Release
CMAKE_EXTRA   ?=
NPROC         := $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
UNAME_S       := $(shell uname -s)

.PHONY: all configure build clean test install conan deps

all: build

# Auto-install platform dependencies
deps:
ifeq ($(UNAME_S),Darwin)
	@command -v brew >/dev/null || (echo "Error: Homebrew not found. Install from https://brew.sh" && exit 1)
	@brew list libomp >/dev/null 2>&1 || (echo "Installing libomp via Homebrew..." && brew install libomp)
endif

# Resolve SDKROOT only on macOS; empty on Linux (harmless)
ifeq ($(UNAME_S),Darwin)
    SDKROOT_ENV := SDKROOT=$(shell xcrun --show-sdk-path)
else
    SDKROOT_ENV :=
endif

conan: deps
	@mkdir -p $(BUILD_DIR)
	$(SDKROOT_ENV) \
	conan install . --build=missing \
		-s build_type=$(BUILD_TYPE) \
		-if $(BUILD_DIR)

configure: conan
	cmake -S . -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_TOOLCHAIN_FILE=$(CURDIR)/$(BUILD_DIR)/conan_toolchain.cmake \
		-Wno-dev \
		$(CMAKE_EXTRA)

build: configure
	cmake --build $(BUILD_DIR) -j$(NPROC)

test: CMAKE_EXTRA += -DWITH_COMMON_UT=ON
test: configure
	cmake --build $(BUILD_DIR) -j$(NPROC)
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	rm -rf $(BUILD_DIR)

install: build
	cmake --install $(BUILD_DIR)
