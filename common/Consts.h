// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <stdint.h>
#include <string>

const int64_t INVALID_FIELD_ID = -1;
const int64_t INVALID_SEG_OFFSET = -1;

const int64_t DEFAULT_FIELD_MAX_MEMORY_LIMIT = 128 << 20;  // bytes
const int64_t DEFAULT_HIGH_PRIORITY_THREAD_CORE_COEFFICIENT = 10;
const int64_t DEFAULT_MIDDLE_PRIORITY_THREAD_CORE_COEFFICIENT = 5;
const int64_t DEFAULT_LOW_PRIORITY_THREAD_CORE_COEFFICIENT = 1;

const int64_t DEFAULT_INDEX_FILE_SLICE_SIZE = 16 << 20;  // bytes

const int DEFAULT_CPU_NUM = 1;

const int64_t DEFAULT_EXEC_EVAL_EXPR_BATCH_SIZE = 8192;

const int64_t DEFAULT_MAX_OUTPUT_SIZE = 67108864;  // bytes, 64MB

const int64_t DEFAULT_CHUNK_MANAGER_REQUEST_TIMEOUT_MS = 10000;

const int64_t DEFAULT_BITMAP_INDEX_BUILD_MODE_BOUND = 500;

const int64_t DEFAULT_HYBRID_INDEX_BITMAP_CARDINALITY_LIMIT = 100;

const size_t MARISA_NULL_KEY_ID = -1;

const std::string JSON_CAST_TYPE = "json_cast_type";
const std::string JSON_PATH = "json_path";
const bool DEFAULT_OPTIMIZE_EXPR_ENABLED = true;
const int64_t DEFAULT_CONVERT_OR_TO_IN_NUMERIC_LIMIT = 150;
const int64_t DEFAULT_JSON_INDEX_MEMORY_BUDGET = 16777216;  // bytes, 16MB
const bool DEFAULT_GROWING_JSON_KEY_STATS_ENABLED = false;
const int64_t DEFAULT_JSON_KEY_STATS_COMMIT_INTERVAL = 200;
const bool DEFAULT_CONFIG_PARAM_TYPE_CHECK_ENABLED = true;

// index config related
const std::string SEGMENT_INSERT_FILES_KEY = "segment_insert_files";
const std::string INSERT_FILES_KEY = "insert_files";
const std::string PARTITION_KEY_ISOLATION_KEY = "partition_key_isolation";
const std::string STORAGE_VERSION_KEY = "storage_version";
const std::string DIM_KEY = "dim";
const std::string DATA_TYPE_KEY = "data_type";

// storage version
const int64_t STORAGE_V1 = 1;
const int64_t STORAGE_V2 = 2;