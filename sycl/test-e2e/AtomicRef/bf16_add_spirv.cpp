#include <sycl/sycl.hpp>
#include <cstdio>


using bf16 = sycl::ext::oneapi::bfloat16;
int main() {
  sycl::queue q;
  bf16 *p = sycl::malloc_shared<bf16>(1, q);
  *p = 0.0f;
  q.parallel_for(256, [=](sycl::id<1>) {
     sycl::atomic_ref<bf16, sycl::memory_order::relaxed, sycl::memory_scope::device,
                      sycl::access::address_space::global_space>(*p)
         .fetch_add(bf16(1.0f));
   }).wait();
  printf("sum = %g (expect 256)\n", float(*p));
  sycl::free(p, q);
}
