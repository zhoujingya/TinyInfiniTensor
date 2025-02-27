#include "core/graph.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <numeric>
#include <optional>
#include <queue>

namespace infini {

void GraphObj::addOperatorAndConnect(const Operator &op) {
  sorted = false;
  ops.push_back(op);
  for (auto &input : op->getInputs()) {
    if (input) {
      input->addTarget(op);
      if (auto pred = input->getSource()) {
        pred->addSuccessors(op);
        op->addPredecessors(pred);
      }
    }
  }
  for (auto &output : op->getOutputs()) {
    if (output) {
      output->setSource(op);
      for (auto &succ : output->getTargets()) {
        succ->addPredecessors(op);
        op->addSuccessors(succ);
      }
    }
  }
}

string GraphObj::toString() const {
  std::ostringstream oss;
  oss << "Graph Tensors:\n";
  for (const auto &tensor : tensors)
    oss << tensor << "\n";

  oss << "Graph operators:\n";
  for (const auto &op : ops) {
    vector<UidBaseType> preds, succs;
    for (auto &o : op->getPredecessors())
      preds.emplace_back(o->getGuid());
    for (auto &o : op->getSuccessors())
      succs.emplace_back(o->getGuid());
    oss << "OP " << op->getGuid();
    oss << ", pred " << vecToString(preds);
    oss << ", succ " << vecToString(succs);
    oss << ", " << op << "\n";
  }
  return oss.str();
}

bool GraphObj::topo_sort() {
  if (this->sorted) {
    return true;
  }
  std::vector<Operator> sorted;
  std::unordered_set<OperatorObj *> flags;
  sorted.reserve(ops.size());
  flags.reserve(ops.size());
  while (sorted.size() < ops.size()) {
    auto modified = false;
    auto const &op = ops[0];
    auto const &inputs = op->getInputs();
    for (auto const &op : ops) {
      if (flags.find(op.get()) == flags.end() &&
          std::all_of(inputs.begin(), inputs.end(),
                      [&flags](auto const &input) {
                        auto ptr = input->getSource().get();
                        return !ptr || flags.find(ptr) != flags.end();
                      })) {
        modified = true;
        sorted.emplace_back(op);
        flags.insert(op.get());
      }
    }
    if (!modified) {
      return false;
    }
  }
  this->ops = std::move(sorted);
  return this->sorted = true;
}

void GraphObj::optimize() {
  // =================================== 作业
  // ===================================
  // TODO: 设计一个算法来实现指定的图优化规则
  // 图优化规则如下：
  // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose
  // 算子，且做的是相反的操作，可以将其全部删除）
  // 2.
  // 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
  // =================================== 作业
  // ===================================
  // 如果拓扑排序失败,直接返回
  if (!this->topo_sort())
    return;
  bool optimized;
  do {
    optimized = false;

    // 遍历所有算子
    for (size_t i = 0; i < ops.size(); ++i) {
      auto op = ops[i];
      // 处理Transpose算子
      if (op->getOpType() == OpType::Transpose) {
        auto opd = as<TransposeObj>(op);
        auto input = op->getInputs(0);
        auto prevOp = input->getSource();

        // 检查是否有两个相邻的Transpose算子
        if (prevOp && prevOp->getOpType() == OpType::Transpose &&
            input->getTargets().size() == 1) {
          auto prevOpd = as<TransposeObj>(prevOp);
          auto prevInput = prevOp->getInputs(0);

          // 合并两个Transpose的置换操作
          auto perm = opd->getPermute();
          // 检查两个Transpose操作组合后是否为恒等变换
          // 通过将两个置换操作组合，并检查结果是否为[0,1,2,...]来判断
          bool isIdentity = true;
          for (size_t j = 0; j < perm.size(); ++j) {
            // 组合两个置换操作:先做prevOpd的置换,再做当前op的置换
            perm[j] = prevOpd->getPermute()[perm[j]];
            // 如果最终结果中任何位置不等于其下标,说明不是恒等变换
            if (perm[j] != int(j)) {
              isIdentity = false;
            }
          }

          // 更新连接关系
          prevInput->removeTarget(prevOp);
          if (isIdentity) {
            // 如果合并后是恒等变换,直接删除两个Transpose
            for (auto succ : op->getSuccessors()) {
              succ->replaceInput(op->getOutput(), prevInput);
              prevInput->addTarget(succ);
            }
            this->removeTensor(op->getOutput());
          }

          // 清理旧的连接关系
          for (auto pred : prevOp->getPredecessors())
            pred->removeSuccessors(prevOp);
          for (auto succ : op->getSuccessors())
            succ->removePredecessors(op);

          // 删除旧的Tensor和Operator
          removeTensor(input);
          removeOperator(op);
          removeOperator(prevOp);
          optimized = true;
          i -= 2;
          break;
        }
      }

      // 处理MatMul算子
      if (op->getOpType() == OpType::MatMul) {
        auto matmulOp = as<MatmulObj>(op);

        // 检查MatMul的输入是否有Transpose
        for (int inputIdx = 0; inputIdx < 2; ++inputIdx) {
          auto input = op->getInputs(inputIdx);
          auto transposeOp = input->getSource();

          // 如果输入来自Transpose且只有这一个目标
          if (transposeOp && transposeOp->getOpType() == OpType::Transpose &&
              input->getTargets().size() == 1) {
            auto transposeObj =
                as<TransposeObj>(transposeOp);
            auto perm = transposeObj->getPermute();

            // 检查是否只交换了最后两个维度
            bool isLastTwoSwapped =
                (perm.size() > 1 &&
                 perm[perm.size() - 2] == int(perm.size() - 1) &&
                 perm[perm.size() - 1] == int(perm.size() - 2));
            bool isIdentityElsewhere = true;
            for (size_t j = 0; j < perm.size() - 2; ++j) {
              if (perm[j] != int(j)) {
                isIdentityElsewhere = false;
                break;
              }
            }
            if (!isLastTwoSwapped || !isIdentityElsewhere) {
              continue;
            }

            // 将Transpose融入到MatMul的属性中
            if (inputIdx == 0) {
              matmulOp->setTransA(!matmulOp->getTransA());
            } else {
              matmulOp->setTransB(!matmulOp->getTransB());
            }

            // 更新连接关系
            auto prevInput = transposeOp->getInputs(0);
            prevInput->removeTarget(transposeOp);
            prevInput->addTarget(matmulOp);
            matmulOp->replaceInput(input, prevInput);
            matmulOp->removePredecessors(transposeOp);

            // 删除Transpose
            this->removeTensor(input);
            this->removeOperator(transposeOp);
            optimized = true;
            break;
          }
        }
      }
    }
  } while (optimized); // 如果有优化发生就继续循环
}

Tensor GraphObj::getTensor(int fuid) const {
  for (auto tensor : tensors) {
    if (tensor->getFuid() == fuid) {
      return tensor;
    }
  }
  return nullptr;
}

void GraphObj::shape_infer() {
  for (auto &op : ops) {
    auto ans = op->inferShape();
    IT_ASSERT(ans.has_value());
    auto oldOutputs = op->getOutputs();
    IT_ASSERT(ans.value().size() == oldOutputs.size());
    // replace the old outputshape and size with new one
    for (int i = 0; i < (int)ans.value().size(); ++i) {
      auto newShape = ans.value()[i];
      auto oldShape = oldOutputs[i]->getDims();
      auto fuid = oldOutputs[i]->getFuid();
      if (newShape != oldShape) {
        auto tensor = this->getTensor(fuid);
        tensor->setShape(newShape);
      }
    }
  }
}

// 运行时的图内存分配
// 1. 为所有输入tensor分配内存
// 2. 按拓扑顺序为算子的输出tensor分配内存
// 3. 获取实际分配的内存指针并绑定到tensor
// 4. 打印内存分配信息
// 图的抽象：输入，张量，算子
void GraphObj::dataMalloc() {
  // topological sorting first
  IT_ASSERT(topo_sort() == true);
  // =================================== 作业
  // ===================================
  // TODO：利用 allocator 给计算图分配内存
  // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor
  // 绑定内存
  // =================================== 作业
  // =================================== 为每个tensor分配内存
  std::unordered_map<int, size_t> tensorOffsets; // 记录每个tensor的内存偏移量

  // 1. 为所有输入tensor分配内存
  for (auto &tensor : tensors) {
    size_t bytes = tensor->getBytes();
    size_t offset = allocator.alloc(bytes);
    tensorOffsets[tensor->getFuid()] = offset;
  }
  // 2. 按拓扑顺序为算子的输出tensor分配内存
  for (auto &op : ops) {
    for (auto &output : op->getOutputs()) {
      size_t bytes = output->getBytes();
      size_t offset = allocator.alloc(bytes);
      tensorOffsets[output->getFuid()] = offset;
    }
  }
  // 3. 获取实际分配的内存指针并绑定到tensor
  void *basePtr = allocator.getPtr();
  for (auto &tensor : tensors) {
    auto fuid = tensor->getFuid();
    if (tensorOffsets.find(fuid) != tensorOffsets.end()) {
      size_t offset = tensorOffsets[fuid];
      char *tensorPtr = static_cast<char *>(basePtr) + offset;
      auto blob = make_ref<BlobObj>(runtime, tensorPtr);
      tensor->setDataBlob(blob);
    }
  }

  allocator.info();
}

Tensor GraphObj::addTensor(Shape dim, DataType dtype) {
  auto tensor = make_ref<TensorObj>(dim, dtype, runtime);
  tensors.push_back(tensor);
  return tensor;
}

Tensor GraphObj::addTensor(const Tensor &tensor) {
  IT_ASSERT(tensor->getRuntime() == runtime,
            std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                tensor->getRuntime()->toString() + " to " +
                runtime->toString());
  tensors.emplace_back(tensor);
  return tensor;
}

TensorVec GraphObj::addTensor(const TensorVec &tensors) {
  for (auto &t : tensors)
    addTensor(t);
  return tensors;
}

// tensor's "source" and "target" must be in "ops".
// tensor has no "source" and no "target" must not exist.
// "inputs" or "outputs" of operators must be in "tensors"
// "predecessors" and "successors" of an operator of "ops" must be in "ops".
bool GraphObj::checkValid() const {
  for (auto tensor : tensors) {
    IT_ASSERT(
        !(tensor->getTargets().size() == 0 && nullptr == tensor->getSource()));
    for (auto op : tensor->getTargets()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
    }
    auto op = tensor->getSource();
    IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
  }
  for (auto op : ops) {
    for (auto tensor : op->getInputs()) {
      IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                tensors.end());
    }
    for (auto tensor : op->getOutputs()) {
      IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                tensors.end());
    }
    for (auto pre : op->getPredecessors()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
    }
    for (auto suc : op->getSuccessors()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
    }
  }
  std::set<UidBaseType> s;
  // check whether two tensors with the same FUID exist
  for (auto tensor : tensors) {
    int cnt = s.count(tensor->getFuid());
    IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
    s.insert(tensor->getFuid());
  }
  return true;
}

} // namespace infini
