#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <cassert>

#include "test.h"

using namespace infini;
void test_graph() {
  Runtime runtime = NativeCpuRuntimeObj::getInstance();
  Graph g = make_ref<GraphObj>(runtime);
  Tensor i1 = g->addTensor({2, 3, 4, 5}, DataType::UInt32);
  Tensor i2 = g->addTensor({2, 3, 4, 5}, DataType::UInt32);
  Tensor t1 = g->addTensor({2, 3, 5, 4}, DataType::UInt32);
  Tensor t2 = g->addTensor({2, 3, 4, 5}, DataType::UInt32);
  Tensor t3 = g->addTensor({2, 3, 5, 4}, DataType::UInt32);
  Tensor o = g->addTensor({2, 3, 4, 4}, DataType::UInt32);
  g->addOpWithOutputs<TransposeObj>(i1, t1, Shape{0, 1, 3, 2});
  g->addOpWithOutputs<TransposeObj>(t1, t2, Shape{0, 1, 3, 2});
  g->addOpWithOutputs<TransposeObj>(i2, t3, Shape{0, 1, 3, 2});
  g->addOpWithOutputs<MatmulObj>(t2, t3, o);
  // 优化前
  g->print();
  g->optimize();
  // 优化后
  g->print();
  assert(g->getOperators().size() == 1);
  assert(g->getTensors().size() == 3);
  assert(g->getOperators()[0]->getOpType().underlying() == 7);
  auto op = as<MatmulObj>(g->getOperators()[0]);
  assert(op->getInputs(0)->getGuid() == 2);
  assert(op->getInputs(1)->getGuid() == 3);
  assert(op->getOutputs()[0] == o);
  assert(op->getTransA() == false);
  assert(op->getTransB() == true);
}

int main() {
  test_graph();
  return 0;
}
