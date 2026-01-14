#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "filemanager/impl/LocalInputStream.h"
#include "filemanager/impl/LocalOutputStream.h"
#include "filemanager/impl/MemoryInputStream.h"
#include "filemanager/impl/MemoryOutputStream.h"

namespace milvus {

class StreamTest : public ::testing::Test {
 protected:
    std::string temp_file_;

    void
    SetUp() override {
        // Create a unique temp file path
        temp_file_ = "/tmp/stream_test_" + std::to_string(std::rand()) + ".bin";
    }

    void
    TearDown() override {
        // Clean up temp file if it exists
        std::remove(temp_file_.c_str());
    }

    // Helper to create test data
    static std::vector<uint8_t>
    GenerateTestData(size_t size) {
        std::vector<uint8_t> data(size);
        std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::uniform_int_distribution<> dist(0, 255);
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<uint8_t>(dist(gen));
        }
        return data;
    }

    // Helper to write test data to file
    void
    WriteTestFile(const std::vector<uint8_t>& data) {
        std::ofstream file(temp_file_, std::ios::binary);
        file.write(reinterpret_cast<const char*>(data.data()), data.size());
        file.close();
    }
};

// ============================================================================
// LocalOutputStream Tests
// ============================================================================

TEST_F(StreamTest, LocalOutputStream_BasicWrite) {
    LocalOutputStream out(temp_file_);

    std::vector<uint8_t> data = {1, 2, 3, 4, 5};
    size_t written = out.Write(data.data(), data.size());

    EXPECT_EQ(written, data.size());
    EXPECT_EQ(out.Tell(), data.size());

    out.Close();

    // Verify file contents
    std::ifstream verify(temp_file_, std::ios::binary);
    std::vector<uint8_t> read_data(data.size());
    verify.read(reinterpret_cast<char*>(read_data.data()), data.size());
    EXPECT_EQ(data, read_data);
}

TEST_F(StreamTest, LocalOutputStream_MultipleWrites) {
    LocalOutputStream out(temp_file_);

    std::vector<uint8_t> data1 = {1, 2, 3};
    std::vector<uint8_t> data2 = {4, 5, 6, 7};

    out.Write(data1.data(), data1.size());
    EXPECT_EQ(out.Tell(), data1.size());

    out.Write(data2.data(), data2.size());
    EXPECT_EQ(out.Tell(), data1.size() + data2.size());

    out.Close();

    // Verify combined contents
    std::ifstream verify(temp_file_, std::ios::binary);
    std::vector<uint8_t> read_data(data1.size() + data2.size());
    verify.read(reinterpret_cast<char*>(read_data.data()), read_data.size());

    std::vector<uint8_t> expected;
    expected.insert(expected.end(), data1.begin(), data1.end());
    expected.insert(expected.end(), data2.begin(), data2.end());
    EXPECT_EQ(expected, read_data);
}

TEST_F(StreamTest, LocalOutputStream_LargeData) {
    LocalOutputStream out(temp_file_);

    auto data = GenerateTestData(100000);  // 100KB
    size_t written = out.Write(data.data(), data.size());

    EXPECT_EQ(written, data.size());
    EXPECT_EQ(out.Tell(), data.size());

    out.Close();

    // Verify file size
    std::ifstream verify(temp_file_, std::ios::binary | std::ios::ate);
    EXPECT_EQ(static_cast<size_t>(verify.tellg()), data.size());
}

TEST_F(StreamTest, LocalOutputStream_TemplateWrite) {
    LocalOutputStream out(temp_file_);

    int32_t value = 12345;
    out.Write(value);

    out.Close();

    std::ifstream verify(temp_file_, std::ios::binary);
    int32_t read_value = 0;
    verify.read(reinterpret_cast<char*>(&read_value), sizeof(read_value));
    EXPECT_EQ(value, read_value);
}

TEST_F(StreamTest, LocalOutputStream_InvalidPath) {
    EXPECT_THROW(LocalOutputStream("/nonexistent/path/file.bin"), std::runtime_error);
}

// ============================================================================
// LocalInputStream Tests
// ============================================================================

TEST_F(StreamTest, LocalInputStream_BasicRead) {
    auto data = GenerateTestData(100);
    WriteTestFile(data);

    LocalInputStream in(temp_file_);

    EXPECT_EQ(in.Size(), data.size());
    EXPECT_EQ(in.Tell(), 0u);
    EXPECT_FALSE(in.Eof());

    std::vector<uint8_t> read_data(data.size());
    size_t bytes_read = in.Read(read_data.data(), data.size());

    EXPECT_EQ(bytes_read, data.size());
    EXPECT_EQ(data, read_data);
}

TEST_F(StreamTest, LocalInputStream_PartialRead) {
    auto data = GenerateTestData(100);
    WriteTestFile(data);

    LocalInputStream in(temp_file_);

    // Read first 50 bytes
    std::vector<uint8_t> first_half(50);
    in.Read(first_half.data(), 50);
    EXPECT_EQ(in.Tell(), 50u);

    // Read remaining 50 bytes
    std::vector<uint8_t> second_half(50);
    in.Read(second_half.data(), 50);
    EXPECT_EQ(in.Tell(), 100u);

    // Verify both halves
    EXPECT_TRUE(std::equal(first_half.begin(), first_half.end(), data.begin()));
    EXPECT_TRUE(std::equal(second_half.begin(), second_half.end(), data.begin() + 50));
}

TEST_F(StreamTest, LocalInputStream_Seek) {
    auto data = GenerateTestData(100);
    WriteTestFile(data);

    LocalInputStream in(temp_file_);

    EXPECT_TRUE(in.Seek(50));
    EXPECT_EQ(in.Tell(), 50u);

    uint8_t byte;
    in.Read(&byte, 1);
    EXPECT_EQ(byte, data[50]);

    EXPECT_TRUE(in.Seek(0));
    EXPECT_EQ(in.Tell(), 0u);

    in.Read(&byte, 1);
    EXPECT_EQ(byte, data[0]);
}

TEST_F(StreamTest, LocalInputStream_ReadAt) {
    auto data = GenerateTestData(100);
    WriteTestFile(data);

    LocalInputStream in(temp_file_);

    std::vector<uint8_t> read_data(10);
    size_t bytes_read = in.ReadAt(read_data.data(), 50, 10);

    EXPECT_EQ(bytes_read, 10u);
    EXPECT_TRUE(std::equal(read_data.begin(), read_data.end(), data.begin() + 50));
}

TEST_F(StreamTest, LocalInputStream_ReadAtConcurrent) {
    auto data = GenerateTestData(10000);
    WriteTestFile(data);

    LocalInputStream in(temp_file_);

    std::vector<std::thread> threads;
    std::vector<std::vector<uint8_t>> results(10);

    for (int i = 0; i < 10; ++i) {
        results[i].resize(100);
        threads.emplace_back([&in, &results, &data, i]() {
            size_t offset = i * 100;
            in.ReadAt(results[i].data(), offset, 100);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Verify all reads
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(std::equal(results[i].begin(), results[i].end(), data.begin() + i * 100));
    }
}

TEST_F(StreamTest, LocalInputStream_ReadOutOfRange) {
    auto data = GenerateTestData(100);
    WriteTestFile(data);

    LocalInputStream in(temp_file_);

    std::vector<uint8_t> buffer(200);
    EXPECT_THROW(in.Read(buffer.data(), 200), std::runtime_error);
}

TEST_F(StreamTest, LocalInputStream_ReadAtOutOfRange) {
    auto data = GenerateTestData(100);
    WriteTestFile(data);

    LocalInputStream in(temp_file_);

    std::vector<uint8_t> buffer(50);
    EXPECT_THROW(in.ReadAt(buffer.data(), 80, 50), std::runtime_error);
}

TEST_F(StreamTest, LocalInputStream_TemplateRead) {
    int32_t original_value = 12345;
    {
        std::ofstream file(temp_file_, std::ios::binary);
        file.write(reinterpret_cast<const char*>(&original_value), sizeof(original_value));
    }

    LocalInputStream in(temp_file_);
    int32_t read_value = 0;
    in.Read(read_value);
    EXPECT_EQ(original_value, read_value);
}

TEST_F(StreamTest, LocalInputStream_InvalidPath) {
    EXPECT_THROW(LocalInputStream("/nonexistent/file.bin"), std::runtime_error);
}

// ============================================================================
// MemoryOutputStream Tests
// ============================================================================

TEST_F(StreamTest, MemoryOutputStream_BasicWrite) {
    MemoryOutputStream out;

    std::vector<uint8_t> data = {1, 2, 3, 4, 5};
    size_t written = out.Write(data.data(), data.size());

    EXPECT_EQ(written, data.size());
    EXPECT_EQ(out.Tell(), data.size());

    auto [ptr, size] = out.Release();
    EXPECT_EQ(size, static_cast<int64_t>(data.size()));
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(ptr[i], data[i]);
    }
    delete[] ptr;
}

TEST_F(StreamTest, MemoryOutputStream_MultipleWrites) {
    MemoryOutputStream out;

    std::vector<uint8_t> data1 = {1, 2, 3};
    std::vector<uint8_t> data2 = {4, 5, 6, 7};

    out.Write(data1.data(), data1.size());
    EXPECT_EQ(out.Tell(), data1.size());

    out.Write(data2.data(), data2.size());
    EXPECT_EQ(out.Tell(), data1.size() + data2.size());

    std::vector<uint8_t> expected;
    expected.insert(expected.end(), data1.begin(), data1.end());
    expected.insert(expected.end(), data2.begin(), data2.end());

    auto [ptr, size] = out.Release();
    EXPECT_EQ(size, static_cast<int64_t>(expected.size()));
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(ptr[i], expected[i]);
    }
    delete[] ptr;
}

TEST_F(StreamTest, MemoryOutputStream_LargeData) {
    MemoryOutputStream out;

    auto data = GenerateTestData(100000);  // 100KB
    size_t written = out.Write(data.data(), data.size());

    EXPECT_EQ(written, data.size());
    EXPECT_EQ(out.Tell(), data.size());

    auto [ptr, size] = out.Release();
    EXPECT_EQ(size, static_cast<int64_t>(data.size()));
    EXPECT_TRUE(std::equal(data.begin(), data.end(), ptr));
    delete[] ptr;
}

TEST_F(StreamTest, MemoryOutputStream_TemplateWrite) {
    MemoryOutputStream out;

    int32_t value = 12345;
    out.Write(&value, sizeof(value));

    EXPECT_EQ(out.Tell(), sizeof(value));

    auto [ptr, size] = out.Release();
    int32_t read_value;
    std::memcpy(&read_value, ptr, sizeof(read_value));
    EXPECT_EQ(value, read_value);
    delete[] ptr;
}

TEST_F(StreamTest, MemoryOutputStream_Release) {
    MemoryOutputStream out;

    std::vector<uint8_t> data = {1, 2, 3, 4, 5};
    out.Write(data.data(), data.size());

    auto [ptr, size] = out.Release();
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(size, static_cast<int64_t>(data.size()));
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(ptr[i], data[i]);
    }
    delete[] ptr;
}

TEST_F(StreamTest, MemoryOutputStream_AutoExpand) {
    MemoryOutputStream out;

    // Write data larger than initial capacity to test auto-expansion
    auto data = GenerateTestData(3 * 1024 * 1024);  // 3MB, larger than 2MB initial
    size_t written = out.Write(data.data(), data.size());

    EXPECT_EQ(written, data.size());
    EXPECT_EQ(out.Tell(), data.size());

    auto [ptr, size] = out.Release();
    EXPECT_EQ(size, static_cast<int64_t>(data.size()));
    EXPECT_TRUE(std::equal(data.begin(), data.end(), ptr));
    delete[] ptr;
}

TEST_F(StreamTest, MemoryOutputStream_GetData) {
    MemoryOutputStream out;

    std::vector<uint8_t> data = {1, 2, 3, 4, 5};
    out.Write(data.data(), data.size());

    auto span = out.GetData();
    EXPECT_EQ(span.size(), data.size());
    EXPECT_TRUE(std::equal(span.begin(), span.end(), data.begin()));

    // Write more data and verify GetData reflects the update
    std::vector<uint8_t> more_data = {6, 7, 8};
    out.Write(more_data.data(), more_data.size());

    span = out.GetData();
    EXPECT_EQ(span.size(), data.size() + more_data.size());

    std::vector<uint8_t> expected;
    expected.insert(expected.end(), data.begin(), data.end());
    expected.insert(expected.end(), more_data.begin(), more_data.end());
    EXPECT_TRUE(std::equal(span.begin(), span.end(), expected.begin()));
}

TEST_F(StreamTest, MemoryOutputStream_GetDataAt) {
    MemoryOutputStream out;

    std::vector<uint8_t> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    out.Write(data.data(), data.size());

    // Get slice from middle
    auto span = out.GetDataAt(3, 4);
    EXPECT_EQ(span.size(), 4u);
    EXPECT_EQ(span[0], 3);
    EXPECT_EQ(span[1], 4);
    EXPECT_EQ(span[2], 5);
    EXPECT_EQ(span[3], 6);

    // Get slice from beginning
    span = out.GetDataAt(0, 3);
    EXPECT_EQ(span.size(), 3u);
    EXPECT_EQ(span[0], 0);
    EXPECT_EQ(span[1], 1);
    EXPECT_EQ(span[2], 2);

    // Get slice at end
    span = out.GetDataAt(7, 3);
    EXPECT_EQ(span.size(), 3u);
    EXPECT_EQ(span[0], 7);
    EXPECT_EQ(span[1], 8);
    EXPECT_EQ(span[2], 9);
}

TEST_F(StreamTest, MemoryOutputStream_GetDataAt_OutOfRange) {
    MemoryOutputStream out;

    std::vector<uint8_t> data = {1, 2, 3, 4, 5};
    out.Write(data.data(), data.size());

    // Offset + size exceeds written data
    EXPECT_THROW((void)out.GetDataAt(3, 5), std::invalid_argument);

    // Offset beyond written data
    EXPECT_THROW((void)out.GetDataAt(10, 1), std::invalid_argument);

    // Valid boundary case
    EXPECT_NO_THROW((void)out.GetDataAt(0, 5));
    EXPECT_NO_THROW((void)out.GetDataAt(4, 1));
}

// ============================================================================
// MemoryInputStream Tests
// ============================================================================

TEST_F(StreamTest, MemoryInputStream_BasicRead) {
    std::vector<uint8_t> data = GenerateTestData(100);

    MemoryInputStream in(data.data(), data.size());

    EXPECT_EQ(in.Size(), data.size());
    EXPECT_EQ(in.Tell(), 0u);
    EXPECT_FALSE(in.Eof());

    std::vector<uint8_t> read_data(data.size());
    size_t bytes_read = in.Read(read_data.data(), data.size());

    EXPECT_EQ(bytes_read, data.size());
    EXPECT_EQ(data, read_data);
}

TEST_F(StreamTest, MemoryInputStream_PartialRead) {
    auto data = GenerateTestData(100);

    MemoryInputStream in(data.data(), data.size());

    // Read first 50 bytes
    std::vector<uint8_t> first_half(50);
    in.Read(first_half.data(), 50);
    EXPECT_EQ(in.Tell(), 50u);

    // Read remaining 50 bytes
    std::vector<uint8_t> second_half(50);
    in.Read(second_half.data(), 50);
    EXPECT_EQ(in.Tell(), 100u);
    EXPECT_TRUE(in.Eof());

    // Verify both halves
    EXPECT_TRUE(std::equal(first_half.begin(), first_half.end(), data.begin()));
    EXPECT_TRUE(std::equal(second_half.begin(), second_half.end(), data.begin() + 50));
}

TEST_F(StreamTest, MemoryInputStream_Seek) {
    auto data = GenerateTestData(100);

    MemoryInputStream in(data.data(), data.size());

    EXPECT_TRUE(in.Seek(50));
    EXPECT_EQ(in.Tell(), 50u);

    uint8_t byte;
    in.Read(&byte, 1);
    EXPECT_EQ(byte, data[50]);

    EXPECT_TRUE(in.Seek(0));
    EXPECT_EQ(in.Tell(), 0u);

    in.Read(&byte, 1);
    EXPECT_EQ(byte, data[0]);
}

TEST_F(StreamTest, MemoryInputStream_SeekInvalid) {
    auto data = GenerateTestData(100);
    MemoryInputStream in(data.data(), data.size());

    EXPECT_FALSE(in.Seek(-1));
    // New implementation: Seek to offset >= size returns false
    EXPECT_FALSE(in.Seek(100));
    EXPECT_FALSE(in.Seek(101));
    // Valid seek within bounds
    EXPECT_TRUE(in.Seek(99));
}

TEST_F(StreamTest, MemoryInputStream_ReadAt) {
    auto data = GenerateTestData(100);

    MemoryInputStream in(data.data(), data.size());

    std::vector<uint8_t> read_data(10);
    size_t bytes_read = in.ReadAt(read_data.data(), 50, 10);

    EXPECT_EQ(bytes_read, 10u);
    EXPECT_TRUE(std::equal(read_data.begin(), read_data.end(), data.begin() + 50));

    // Position should not change after ReadAt
    EXPECT_EQ(in.Tell(), 0u);
}

TEST_F(StreamTest, MemoryInputStream_TemplateRead) {
    int32_t original_value = 12345;
    std::vector<uint8_t> data(sizeof(original_value));
    std::memcpy(data.data(), &original_value, sizeof(original_value));

    MemoryInputStream in(data.data(), data.size());
    int32_t read_value = 0;
    in.Read(&read_value, sizeof(read_value));
    EXPECT_EQ(original_value, read_value);
}

TEST_F(StreamTest, MemoryInputStream_EmptyData) {
    MemoryInputStream in(nullptr, 0);

    EXPECT_EQ(in.Size(), 0u);
    EXPECT_TRUE(in.Eof());

    // Read on empty stream should return 0
    uint8_t buffer[10];
    EXPECT_EQ(in.Read(buffer, 10), 0u);
}

TEST_F(StreamTest, MemoryInputStream_GetData) {
    auto data = GenerateTestData(100);
    MemoryInputStream in(data.data(), data.size());

    // GetData returns full buffer regardless of read position
    auto span = in.GetData();
    EXPECT_EQ(span.size(), data.size());
    EXPECT_TRUE(std::equal(span.begin(), span.end(), data.begin()));

    // Advance read position
    std::vector<uint8_t> tmp(50);
    in.Read(tmp.data(), 50);
    EXPECT_EQ(in.Tell(), 50u);

    // GetData still returns full buffer
    span = in.GetData();
    EXPECT_EQ(span.size(), data.size());
    EXPECT_TRUE(std::equal(span.begin(), span.end(), data.begin()));
}

TEST_F(StreamTest, MemoryInputStream_GetDataAt) {
    std::vector<uint8_t> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    MemoryInputStream in(data.data(), data.size());

    // Get slice from middle
    auto span = in.GetDataAt(3, 4);
    EXPECT_EQ(span.size(), 4u);
    EXPECT_EQ(span[0], 3);
    EXPECT_EQ(span[1], 4);
    EXPECT_EQ(span[2], 5);
    EXPECT_EQ(span[3], 6);

    // Get slice from beginning
    span = in.GetDataAt(0, 3);
    EXPECT_EQ(span.size(), 3u);
    EXPECT_EQ(span[0], 0);
    EXPECT_EQ(span[1], 1);
    EXPECT_EQ(span[2], 2);

    // Get slice at end
    span = in.GetDataAt(7, 3);
    EXPECT_EQ(span.size(), 3u);
    EXPECT_EQ(span[0], 7);
    EXPECT_EQ(span[1], 8);
    EXPECT_EQ(span[2], 9);

    // GetDataAt does not affect read position
    EXPECT_EQ(in.Tell(), 0u);
}

TEST_F(StreamTest, MemoryInputStream_GetDataAt_OutOfRange) {
    std::vector<uint8_t> data = {1, 2, 3, 4, 5};
    MemoryInputStream in(data.data(), data.size());

    // Offset + size exceeds data size
    EXPECT_THROW((void)in.GetDataAt(3, 5), std::invalid_argument);

    // Offset beyond data
    EXPECT_THROW((void)in.GetDataAt(10, 1), std::invalid_argument);

    // Valid boundary case
    EXPECT_NO_THROW((void)in.GetDataAt(0, 5));
    EXPECT_NO_THROW((void)in.GetDataAt(4, 1));
}

// ============================================================================
// Integration Tests - Memory to Local and vice versa
// ============================================================================

TEST_F(StreamTest, Integration_MemoryToLocal) {
    auto data = GenerateTestData(1000);

    // Write to memory
    MemoryOutputStream mem_out;
    mem_out.Write(data.data(), data.size());

    // Get data and write to file
    auto [ptr, size] = mem_out.Release();

    LocalOutputStream local_out(temp_file_);
    local_out.Write(ptr, size);
    local_out.Close();
    delete[] ptr;

    // Read back from file
    LocalInputStream local_in(temp_file_);
    std::vector<uint8_t> read_data(local_in.Size());
    local_in.Read(read_data.data(), read_data.size());

    EXPECT_EQ(data, read_data);
}

TEST_F(StreamTest, Integration_LocalToMemory) {
    auto data = GenerateTestData(1000);
    WriteTestFile(data);

    // Read from file
    LocalInputStream local_in(temp_file_);
    std::vector<uint8_t> file_data(local_in.Size());
    local_in.Read(file_data.data(), file_data.size());

    // Create memory stream from file data
    MemoryInputStream mem_in(file_data.data(), file_data.size());

    std::vector<uint8_t> read_data(mem_in.Size());
    mem_in.Read(read_data.data(), read_data.size());

    EXPECT_EQ(data, read_data);
}

TEST_F(StreamTest, Integration_RoundTrip) {
    auto original_data = GenerateTestData(5000);

    // Write to memory output stream
    MemoryOutputStream mem_out;
    mem_out.Write(original_data.data(), original_data.size());

    // Transfer to local file
    auto [ptr, size] = mem_out.Release();
    LocalOutputStream local_out(temp_file_);
    local_out.Write(ptr, size);
    local_out.Close();
    delete[] ptr;

    // Read from local file
    LocalInputStream local_in(temp_file_);
    std::vector<uint8_t> file_data(local_in.Size());
    local_in.Read(file_data.data(), file_data.size());

    // Read from memory input stream
    MemoryInputStream mem_in(file_data.data(), file_data.size());
    std::vector<uint8_t> final_data(mem_in.Size());
    mem_in.Read(final_data.data(), final_data.size());

    EXPECT_EQ(original_data, final_data);
}

}  // namespace milvus
