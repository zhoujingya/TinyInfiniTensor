#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"
#include <cassert>
#include <iostream>

#include "test.h"
using namespace infini;

void test_allocator_basic() {
  Shape shape = Shape{1, 2, 2, 3};
  Runtime runtime = NativeCpuRuntimeObj::getInstance();
  Tensor a = make_ref<TensorObj>(shape, DataType::Float32, runtime);
  Tensor b = make_ref<TensorObj>(shape, DataType::Float32, runtime);
  Tensor c = make_ref<TensorObj>(shape, DataType::Float32, runtime);
  Tensor d = make_ref<TensorObj>(shape, DataType::Float32, runtime);
  Allocator allocator = Allocator(runtime);
  // allocate a->b->c
  size_t offsetA = allocator.alloc(a->getBytes());
  size_t offsetB = allocator.alloc(b->getBytes());
  size_t offsetC = allocator.alloc(c->getBytes());
  // free b, then allocate d
  allocator.free(offsetB, b->getBytes());
  size_t offsetD = allocator.alloc(d->getBytes());
  // expected to be a->d->c
  assert(offsetB == offsetD);
  assert(offsetA != 0 || offsetB != 0 || offsetC != 0 || offsetD != 0);
  std::cout << "All allocator tests passed!" << std::endl;
}

void testAllocWithEndFreeBlock() {
  Shape shape = Shape{1, 2, 2, 3};
  Runtime runtime = NativeCpuRuntimeObj::getInstance();
  Tensor a = make_ref<TensorObj>(shape, DataType::Float32, runtime);
  Tensor b = make_ref<TensorObj>(shape, DataType::Float32, runtime);
  Tensor c = make_ref<TensorObj>(shape, DataType::Float32, runtime);
  Tensor d = make_ref<TensorObj>(Shape{2, 2, 2, 3}, DataType::Float32, runtime);
  Allocator allocator = Allocator(runtime);
  // allocate a->b->c
  allocator.alloc(a->getBytes());
  allocator.alloc(b->getBytes());
  size_t offsetC = allocator.alloc(c->getBytes());
  allocator.info();
  // free c, then allocate d
  allocator.free(offsetC, c->getBytes());
  size_t offsetD = allocator.alloc(d->getBytes());
  allocator.info();
  // expected to be a->b->d, with no free block between b and c
  assert(offsetC == offsetD);
}

void testGetPtr() {
  Shape shape = Shape{1, 2, 2, 3};
  Runtime runtime = NativeCpuRuntimeObj::getInstance();
  Tensor a = make_ref<TensorObj>(shape, DataType::Float32, runtime);
  Tensor b = make_ref<TensorObj>(shape, DataType::Float32, runtime);
  Tensor c = make_ref<TensorObj>(shape, DataType::Float32, runtime);
  Tensor d = make_ref<TensorObj>(shape, DataType::Float32, runtime);
  Allocator allocator = Allocator(runtime);
  // allocate a->b->c->d
  allocator.alloc(a->getBytes());
  allocator.alloc(b->getBytes());
  allocator.alloc(c->getBytes());
  allocator.alloc(d->getBytes());
  // multiple calls to the getPtr() function should return the same pointer
  void *ptr1 = allocator.getPtr();
  void *ptr2 = allocator.getPtr();
  assert(ptr1 == ptr2);
}

int main() {
  try {
    test_allocator_basic();
    testAllocWithEndFreeBlock();
    testGetPtr();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Test failed: " << e.what() << std::endl;
    return 1;
  }
}
