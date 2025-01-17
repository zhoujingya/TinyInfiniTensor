#include "operators/concat.h"
#include "utils/operator_utils.h"
#include <cstddef>

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
  int rank = inputs[0]->getRank();
  dim = get_real_axis(_dim, rank);
  IT_ASSERT(checkValid(graph));
}

// 除了指定的维度，其他维度必须相同，指定的维度需要相加
optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
  Shape dims = inputs[0]->getDims();
  auto rank = inputs[0]->getRank();
  size_t expected_dim = getDim();

  // Check if all input tensors have the same shape except for the concatenated
  // dimension
  for (size_t i = 1; i < inputs.size(); i++) {
    auto cur_dims = inputs[i]->getDims();
    IT_ASSERT(cur_dims.size() == rank);
    for (size_t j = 0; j < rank; j++) {
      if (j != expected_dim) {
        IT_ASSERT(dims[j] == cur_dims[j]);
      }
    }
  }

  // Calculate the total length of the concatenated dimension
  size_t concat_dim_size = dims[expected_dim];
  for (size_t i = 1; i < inputs.size(); i++) {
    concat_dim_size += inputs[i]->getDims()[expected_dim];
  }
  dims[expected_dim] = concat_dim_size;
  return {{dims}};
}

std::string ConcatObj::toString() const {
  std::ostringstream os;
  os << "Concat[" << getGuid() << "]";
  os << "(";
  for (auto input : inputs)
    os << vecToString(input->getDims()) << ",";
  os << "dim=" << dim << ",";
  os << "input=";
  for (auto input : inputs)
    os << input->getGuid() << ",";
  os << "output=" << outputs[0]->getGuid() << ")";
  return os.str();
}

} // namespace infini
