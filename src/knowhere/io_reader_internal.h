#pragma once

#include <cstddef>

namespace knowhere_internal {
inline bool
ShouldRetryInterruptedSyscall(size_t& retry, size_t max_retries) {
    return ++retry <= max_retries;
}
}  // namespace knowhere_internal
