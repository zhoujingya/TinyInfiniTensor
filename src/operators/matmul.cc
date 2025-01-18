#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    /*
     * 矩阵乘法形状推导算法:
     * 1. 处理转置:
     *    - 如果设置了transA，交换A矩阵最后两个维度
     *    - 如果设置了transB，交换B矩阵最后两个维度
     *
     * 2. 维度检查:
     *    - 确保两个输入至少是2维的
     *    - 检查内部维度匹配(A的最后一维 == B的倒数第二维)
     *
     * 3. 批量维度广播:
     *    - 对除最后两维外的所有维度进行广播
     *    - 广播规则:维度必须相等,或其中一个为1
     *    - 输出取较大的维度值
     *
     * 4. 输出形状:
     *    - 前面的维度为广播后的批量维度
     *    - 最后两维是: [A的倒数第二维(M), B的最后一维(N)]
     */
    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ==================================='
        if(inputs.size() != 2)
            return std::nullopt;
        auto A = inputs[0]->getDims();
        auto B = inputs[1]->getDims();

        // Get ranks
        int rankA = A.size();
        int rankB = B.size();

        // Handle transpositions
        if (transA && rankA >= 2) {
            std::swap(A[rankA-1], A[rankA-2]);
        }
        if (transB && rankB >= 2) {
            std::swap(B[rankB-1], B[rankB-2]);
        }

        // Check if matrix multiplication is valid
        IT_ASSERT(rankA >= 2 && rankB >= 2);
        IT_ASSERT(A[rankA-1] == B[rankB-2]); // Inner dimensions must match

        // Calculate output shape
        Shape outShape;

        // Handle broadcasting of batch dimensions
        int maxRank = std::max(rankA, rankB);
        for (int i = 0; i < maxRank - 2; i++) {
            int dimA = (i < rankA-2) ? A[i] : 1;
            int dimB = (i < rankB-2) ? B[i] : 1;
            IT_ASSERT(dimA == dimB || dimA == 1 || dimB == 1);
            outShape.push_back(std::max(dimA, dimB));
        }

        // Add matrix multiplication dimensions
        outShape.push_back(A[rankA-2]); // M dimension
        outShape.push_back(B[rankB-1]); // N dimension

        return {{outShape}};
    }

} // namespace infini
