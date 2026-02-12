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
        "gtest/1.15.0",
        "glog/0.7.1",
        "fmt/11.0.2",
        "prometheus-cpp/1.2.4",
        "libcurl/8.10.1",
        "gflags/2.2.2",
        "opentelemetry-cpp/1.23.0@milvus/dev",
        "grpc/1.67.1@milvus/dev",
        "abseil/20250127.0",
        "xz_utils/5.4.5",
        "zlib/1.3.1",
        "libevent/2.1.12",
        "openssl/3.3.2",
        "folly/2024.08.12.00",
        "boost/1.83.0"
    )

    options = {
        "with_ut": [True, False],
        "with_asan": [True, False],
    }

    default_options = {
        "folly:shared": True,
        "gtest:build_gmock": True,
        "glog:with_gflags": True,
        "glog:shared": True,
        "prometheus-cpp:with_pull": False,
        "fmt:header_only": False,
        "opentelemetry-cpp:with_stl": True,
        "with_ut": False,
        "with_asan": False,
    }

    def requirements(self):
        # Force all dependencies to use protobuf from milvus/dev channel
        # This is needed to resolve conflicts between opentelemetry-cpp and grpc
        self.requires("protobuf/5.27.0@milvus/dev", force=True, override=True)
        self.requires("lz4/1.9.4", force=True, override=True)
        if self.settings.os != "Macos":
            self.requires("libunwind/1.8.1")

    @property
    def _minimum_cpp_standard(self):
        return 17

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generator = "Unix Makefiles"
        tc.variables["CMAKE_POSITION_INDEPENDENT_CODE"] = self.options.get_safe(
            "fPIC", True
        )
        # Relocatable shared lib on Macos
        tc.cache_variables["CMAKE_POLICY_DEFAULT_CMP0042"] = "NEW"
        # CMake 4.x removed compatibility with cmake_minimum_required < 3.5
        tc.cache_variables["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"
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

        tc.variables["WITH_COMMON_UT"] = self.options.with_ut
        tc.variables["WITH_ASAN"] = self.options.with_asan
        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()

        pc = PkgConfigDeps(self)
        pc.generate()


    def build(self):
        # files.apply_conandata_patches(self)
        cmake = CMake(self)
        cmake.generator = "Unix Makefiles"
        cmake.configure()
        cmake.build()
