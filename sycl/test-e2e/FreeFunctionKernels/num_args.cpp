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
int test_num_args_free_function_api(sycl::context &ctxt, sycl::device &dev,
                                    const int expected_num_args) {
  const int actual =
      syclexp::get_kernel_info<Func, sycl::info::kernel::num_args>(ctxt, dev);
  const bool res = actual == expected_num_args;
  if (!res)
    std::cout << FFTestMark << "test_num_args failed: expected_num_args "
              << expected_num_args << "actual " << actual << std::endl;
  return res;
}

template <auto *Func>
int test_num_args_kernel_api(sycl::context &ctxt, sycl::device &dev,
                             const int expected_num_args) {
  auto bundle =
      syclexp::get_kernel_bundle<Func, sycl::bundle_state::executable>(ctxt);
  const int actual = bundle.template ext_oneapi_get_kernel<Func>()
                         .template get_info<sycl::info::kernel::num_args>();
  std::cout << FFTestMark << "actual number of args: " << actual
            << " expected: " << expected_num_args << std::endl;
  const bool res = actual == expected_num_args;
  if (!res)
    std::cout << FFTestMark << "test_num_args_kernel_api failed: expected_num_args "
              << expected_num_args << "actual " << actual << std::endl;
  return res;
}

static bool call_kernel_code(sycl::queue &q, sycl::kernel &kernel) {
  int *ptr = sycl::malloc_shared<int>(NUM, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(3, ptr);
     sycl::nd_range ndr{{NUM}, {WGSIZE}};
     cgh.parallel_for(ndr, kernel);
   }).wait();
  sycl::free(ptr, q);
  return true;
}

int main() {
  sycl::queue q;
  sycl::context ctx = q.get_context();
  sycl::device dev = q.get_device();

  /*auto bndl_r =
      syclexp::get_kernel_bundle<func_range, sycl::bundle_state::executable>(
          ctx);
  auto bndl_s =
      syclexp::get_kernel_bundle<func_single, sycl::bundle_state::executable>(
          ctx);*/
  auto bndl_k =
      syclexp::get_kernel_bundle<func_range, sycl::bundle_state::executable>(
          ctx);
  //sycl::kernel k_func_range = bndl_r.ext_oneapi_get_kernel<func_range>();
  //sycl::kernel k_func_single = bndl_s.ext_oneapi_get_kernel<func_single>();
  sycl::kernel k_kernel_func = bndl_k.ext_oneapi_get_kernel<kernel_func>();
  //call_kernel_code(q, k_func_range);
 // call_kernel_code(q, k_func_single);
  call_kernel_code(q, k_kernel_func);

  int ret = test_num_args_free_function_api<func_range>(ctx, dev, 2);
  ret |= test_num_args_free_function_api<func_single>(ctx, dev, 2);
  ret |= test_num_args_free_function_api<kernel_func>(ctx, dev, 3);
  ret |= test_num_args_kernel_api<func_range>(ctx, dev, 2);
  ret |= test_num_args_kernel_api<func_single>(ctx, dev, 2);
  ret |= test_num_args_kernel_api<kernel_func>(ctx, dev, 3);
  return ret;
}
