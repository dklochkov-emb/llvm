// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %{build} %level_zero_options -o %t.ze.out
// RUN: %{run} %t.ze.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/get_kernel_info.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>


namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;
static constexpr auto FFTestMark = "Free function Kernel Test:";

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void func_range(float start, float *ptr) {}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void func_single(float start, float *ptr) {}


SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void kernel_func(sycl::item<1> idx, float value, sycl::accessor<int, 1> acc) {

}

template <typename T>
static void call_kernel_code(sycl::queue &q, sycl::kernel &kernel) {
  T *ptr = sycl::malloc_shared<T>(NUM, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(3.14f, ptr);
     sycl::nd_range ndr{{NUM}, {WGSIZE}};
     cgh.parallel_for(ndr, kernel);
   }).wait();
  sycl::free(ptr, q);
}

template <auto *Func>
int test_num_args(sycl::context &ctxt, const int expected_num_args) {
  const int actual =
      syclexp::get_kernel_info<Func, sycl::info::kernel::num_args>(ctxt);
  const bool res = actual == expected_num_args;
  if (!res)
    std::cout << FFTestMark << "test_num_args failed: expected_num_args "
              << expected_num_args << "actual " << actual << std::endl;
  return res;
}

int main() {
    sycl::queue q;
    sycl::context ctx = q.get_context();
    sycl::device dev = q.get_device();

    auto bundle_range =
        syclexp::get_kernel_bundle<func_single, sycl::bundle_state::executable>(
            ctx);

    auto actual =
        syclexp::get_kernel_info<func_single, sycl::info::kernel::num_args>(
            ctx, dev);

  std::cout << "Actual number of arguments: " << actual << std::endl;
    assert(actual == 2 && "kernel should take 2 args");
    return 0;
}
