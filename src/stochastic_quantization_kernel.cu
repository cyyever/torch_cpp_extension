#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include "stochastic_quantization.hpp"

namespace cyy{
namespace pytorch {

void stochastic_quantization_gpu(at::Tensor &slot_ret, const at::Tensor &src,
                                 uint64_t quantization_level) {
  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(at::get_tensor_base(slot_ret))
                  .add_input(at::get_tensor_base(src))
                  .build();

  at::native::gpu_kernel(
      iter, [quantization_level] GPU_LAMBDA(const float src_val) -> float {
        uint64_t slot = static_cast<uint64_t>(src_val * quantization_level);
        return slot;
      });
}

}
}
