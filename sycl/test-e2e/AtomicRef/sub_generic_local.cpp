// UNSUPPORTED: target-nvidia, target-amd, spirv-backend
// UNSUPPORTED-INTENDED: only supported by backends with atomic16 support
// RUN: %{build} %{atomic16_flags} -o %t.16.out
// RUN: %{run} %t.16.out
// RUN: %{build} %{atomic32_flags} -o %t.32.out
// RUN: %{run} %t.32.out
// RUN: %{build} %{atomic64_flags} -o %t.64.out
// RUN: %{run} %t.64.out

// UNSUPPORTED: windows && arch-intel_gpu_bmg_g21
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/23416

#define TEST_GENERIC_IN_LOCAL 1
#include "sub.h"

int main() { sub_test_all<access::address_space::generic_space>(); }
