// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/math/random.h"

#include <cstdlib>

#include <glog/logging.h>
#include <gmock/gmock.h>

namespace {

constexpr unsigned kDefaultTestPRNGSeed = 0;

// ONNX Runtime's POSIX telemetry uploader thread lives in a static that is
// torn down at process exit, where it can race static destruction and segfault
// after all tests passed. ORT reads this variable once, when the first
// Ort::Env is created, so it must be set before any test runs. An explicit
// user setting is kept. Windows telemetry uses ETW and ignores this variable.
void DisableOnnxRuntimeTelemetry() {
#ifndef _WIN32
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  setenv("ORT_DISABLE_TELEMETRY", "1", /*overwrite=*/0);
#endif
}

class PRNGTestEventListener : public testing::EmptyTestEventListener {
 public:
  void OnTestStart(const testing::TestInfo&) override {
    colmap::SetPRNGSeed(kDefaultTestPRNGSeed);
  }
};

}  // namespace

int main(int argc, char** argv) {
  DisableOnnxRuntimeTelemetry();
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleMock(&argc, argv);
  testing::UnitTest::GetInstance()->listeners().Append(
      new PRNGTestEventListener);
  return RUN_ALL_TESTS();
}
