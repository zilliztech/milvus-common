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
    settings = "os", "compiler", "build_type", "arch"
    requires = (
        "gtest/1.13.0#f9548be18a41ccc6367efcb8146e92be",
        "glog/0.6.0#d22ebf9111fed68de86b0fa6bf6f9c3f",
        "fmt/9.1.0#95259249fb7ef8c6b5674a40b00abba3",
        "prometheus-cpp/1.1.0#ea9b101cb785943adb40ad82eda7856c",
        "libcurl/7.86.0#bbc887fae3341b3cb776c601f814df05",
        "gflags/2.2.2#b15c28c567c7ade7449cf994168a559f",
        "opentelemetry-cpp/1.8.1.1@milvus/2.4#7345034855d593047826b0c74d9a0ced",
        "xz_utils/5.4.0#a6d90890193dc851fa0d470163271c7a",
        "zlib/1.2.13#df233e6bed99052f285331b9f54d9070",
        "libevent/2.1.12#4fd19d10d3bed63b3a8952c923454bc0",
        "openssl/3.1.2#02594c4c0a6e2b4feb3cd15119993597",
        "folly/2023.10.30.10@milvus/dev",
        "boost/1.82.0"
    )

    options = {
        "with_ut": [True, False],
    }

    default_options = {
        "folly:shared": True,
        "gtest:build_gmock": True,
        "glog:with_gflags": True,
        "glog:shared": True,
        "prometheus-cpp:with_pull": False,
        "fmt:header_only": True,
        "with_ut": False,
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

    @property
    def _minimum_cpp_standard(self):
        return 17

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["CMAKE_POSITION_INDEPENDENT_CODE"] = self.options.get_safe(
            "fPIC", True
        )
        # Relocatable shared lib on Macos
        tc.cache_variables["CMAKE_POLICY_DEFAULT_CMP0042"] = "NEW"
        # Honor BUILD_SHARED_LIBS from conan_toolchain (see https://github.com/conan-io/conan/issues/11840)
        tc.cache_variables["CMAKE_POLICY_DEFAULT_CMP0077"] = "NEW"

        cxx_std_flag = tools.cppstd_flag(self.settings)
        cxx_std_value = (
            cxx_std_flag.split("=")[1]
            if cxx_std_flag
            else "c++{}".format(self._minimum_cpp_standard)
        )
        tc.variables["CXX_STD"] = cxx_std_value
        if is_msvc(self):
            tc.variables["MSVC_LANGUAGE_VERSION"] = cxx_std_value
            tc.variables["MSVC_ENABLE_ALL_WARNINGS"] = False
            tc.variables["MSVC_USE_STATIC_RUNTIME"] = "MT" in msvc_runtime_flag(self)

        tc.variables["WITH_UT"] = self.options.with_ut
        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()

        pc = PkgConfigDeps(self)
        pc.generate()


    def build(self):
        # files.apply_conandata_patches(self)
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
