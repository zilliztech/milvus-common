from conan.tools.microsoft import is_msvc, msvc_runtime_flag
from conan.tools.build import check_min_cppstd
from conan.tools.scm import Version
from conan.tools import files
from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.gnu import PkgConfigDeps
from conan.errors import ConanInvalidConfiguration
from conans import tools
import os

required_conan_version = ">=1.55.0"


class MilvusCommonConan(ConanFile):
    keep_imports = True
    settings = "os", "compiler", "build_type", "arch"
    requires = (
        "gtest/1.13.0#f9548be18a41ccc6367efcb8146e92be",
        "glog/0.6.0#d22ebf9111fed68de86b0fa6bf6f9c3f",
        "fmt/9.1.0#95259249fb7ef8c6b5674a40b00abba3",
        "prometheus-cpp/1.1.0#ea9b101cb785943adb40ad82eda7856c",
        "libcurl/7.86.0#bbc887fae3341b3cb776c601f814df05",
        "opentelemetry-cpp/1.8.1.1@milvus/2.4#7345034855d593047826b0c74d9a0ced",
        "xz_utils/5.4.0#a6d90890193dc851fa0d470163271c7a",
        "zlib/1.2.13#df233e6bed99052f285331b9f54d9070",
        "libevent/2.1.12#4fd19d10d3bed63b3a8952c923454bc0",
        "openssl/3.1.2#02594c4c0a6e2b4feb3cd15119993597",
        "folly/2023.10.30.08@milvus/dev#81d7729cd4013a1b708af3340a3b04d9",
    )
    generators = ("cmake", "cmake_find_package")
    default_options = {
        "folly:shared": True,
        "gtest:build_gmock": True,
        "glog:with_gflags": True,
        "glog:shared": True,
        "prometheus-cpp:with_pull": False,
        "fmt:header_only": True,
    }

    def configure(self):
        if self.settings.arch not in ("x86_64", "x86"):
            del self.options["folly"].use_sse4_2
        if self.settings.os == "Macos":
            # By default abseil use static link but can not be compatible with macos X86
            self.options["abseil"].shared = True
            self.options["arrow"].with_jemalloc = False

    def requirements(self):
        if self.settings.os != "Macos":
            self.requires("libunwind/1.7.2")

    def imports(self):
        self.copy("*.dylib", "../lib", "lib")
        self.copy("*.dll", "../lib", "lib")
        self.copy("*.so*", "../lib", "lib")
        self.copy("*", "../bin", "bin")
        self.copy("*.proto", "../include", "include")
