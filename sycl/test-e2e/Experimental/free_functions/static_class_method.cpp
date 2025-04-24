
// RUN: %clangxx -fsycl --save-temps -v -std=c++17 %s -o %t.out
// RUN: %{run} %t.out

// The name mangling for free function kernels currently does not work with PTX.
// UNSUPPORTED: cuda
// UNSUPPORTED-INTENDED: Not implemented yet for Nvidia/AMD backends.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>
#include <type_traits>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;
/*
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void func(float start, float *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id] = start + static_cast<float>(id);
}
*/
template <typename T>
void check_result(T *ptr) {
  for (size_t i = 0; i < NUM; ++i) {
    const T expected = 3.14f + static_cast<T>(i);
    assert(ptr[i] == expected &&
           "Kernel execution did not produce the expected result");
  }
}

template <typename T>
static void call_kernel_code(sycl::queue &q, sycl::kernel &kernel) {
  T *ptr = sycl::malloc_shared<T>(NUM, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(3.14f, ptr);
     sycl::nd_range ndr{{NUM}, {WGSIZE}};
     cgh.parallel_for(ndr, kernel);
   }).wait();
  check_result<T>(ptr);
}

class TestClass {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
  static void static_method(float start, float *ptr) {
    size_t id =
        syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
    ptr[id] = start + static_cast<float>(id);
  }
};

void test_function_without_ns(sycl::queue &q, sycl::context &ctxt) {
  auto exe_bndl =
      syclexp::get_kernel_bundle<TestClass::static_method,
                                 sycl::bundle_state::executable>(ctxt);
  sycl::kernel k_func =
      exe_bndl.template ext_oneapi_get_kernel<TestClass::static_method>();
  call_kernel_code<float>(q, k_func);
}

int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();

  test_function_without_ns(q, ctxt);
  return 0;
}
