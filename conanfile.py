required_conan_version = ">=2.0"

from conan.tools.microsoft import is_msvc, msvc_runtime_flag
from conan.tools.build import check_min_cppstd, cppstd_flag
from conan.tools.scm import Version
from conan.tools import files
from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.gnu import PkgConfigDeps
from conan.errors import ConanInvalidConfiguration
import os

class MilvusCommonConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    requires = (
        "gtest/1.15.0",
        "glog/0.7.1#a306e61d7b8311db8cb148ad62c48030",
        "prometheus-cpp/1.2.4#0918d66c13f97acb7809759f9de49b3f",
        "gflags/2.2.2#7671803f1dc19354cc90bd32874dcfda",
        "opentelemetry-cpp/1.23.0@milvus/dev#11bc565ec6e82910ae8f7471da756720",
        "grpc/1.67.1@milvus/dev#efeaa484b59bffaa579004d5e82ec4fd",
        "abseil/20250127.0#481edcc75deb0efb16500f511f0f0a1c",
        "xz_utils/5.4.5#fc4e36861e0a47ecd4a40a00e6d29ac8",
        "zlib/1.3.1#8045430172a5f8d56ba001b14561b4ea",
        "libevent/2.1.12#95065aaefcd58d3956d6dfbfc5631d97",
        "folly/2026.04.20.00@milvus/dev#f72c1b4271ff64215e9b1797a32bf8ad",
        "boost/1.83.0#4e8a94ac1b88312af95eded83cd81ca8",
    )

    options = {
        "with_ut": [True, False],
        "with_asan": [True, False],
    }

    default_options = {
        "folly/*:shared": True,
        "gtest/*:build_gmock": True,
        "openssl/*:shared": True,
        "gflags/*:shared": True,
        "glog/*:with_gflags": True,
        "glog/*:shared": True,
        "prometheus-cpp/*:with_pull": False,
        "fmt/*:header_only": False,
        "opentelemetry-cpp/*:with_stl": True,
        # Use OpenMP threading to match knowhere's libopenblas-openmp-dev
        "openblas/*:use_openmp": True,
        "with_ut": False,
        "with_asan": False,
    }

    def requirements(self):
        # Force all dependencies to use protobuf from milvus/dev channel
        # This is needed to resolve conflicts between opentelemetry-cpp and grpc
        self.requires("protobuf/5.27.0@milvus/dev#42f031a96d21c230a6e05bcac4bdd633", force=True, override=True)
        self.requires("lz4/1.9.4#7f0b5851453198536c14354ee30ca9ae", force=True, override=True)
        # Force overrides openssl to resolve opentelemetry-cpp's transitive deps
        self.requires("openssl/3.3.2#9f9f130d58e7c13e76bb8a559f0a6a8b", force=True, override=True)
        self.requires("libcurl/8.10.1#a3113369c86086b0e84231844e7ed0a9", force=True, override=True)
        # folly/2026.x recipe still pins fmt/10.2.1; align on fmt/11 (matches knowhere)
        self.requires("fmt/11.2.0#eb98daa559c7c59d591f4720dde4cd5c", force=True, override=True)
        # nlohmann_json is a direct dependency (used in Tracer.cpp) and also forces
        # the transitive version from opentelemetry-cpp to align
        self.requires("nlohmann_json/3.11.3#ffb9e9236619f1c883e36662f944345d", force=True)
        if self.settings.os != "Macos":
            self.requires("libunwind/1.8.1#748a981ace010b80163a08867b732e71")
            # openblas is only used on Linux (thread_pool.cc is guarded by __linux__)
            self.requires("openblas/0.3.30")

    @property
    def _minimum_cpp_standard(self):
        return 20

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

        cxx_std_flag = cppstd_flag(self)
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
